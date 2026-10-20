#!/usr/bin/env python
"""The external sandbox watcher of bcb-bfcl-200: from the
grading host (``~/venvs/modal``), watch every cell container of ONE launcher
run and make sure no OWNED Modal sandbox outlives its attempt — after a hard
death (``FunctionCall.cancel(terminate_containers=True)``, a container OOM, a
Modal timeout) the container's own ``finally`` sweep never ran, and the
provider's sandboxes would otherwise live to their 3 h timeout.

    ~/venvs/modal/bin/python scripts/tinycap_sandbox_watch.py --run-id <id> \\
        --suffix <out-suffix> --model gemma4e4b-sdfbase --cells none,<derived> \\
        --poll 120 --max-hours 6

Per cell it reads the NEWEST ``watch/attempt_<fcid>.json`` under the cell's
out dir on the volume (the launcher writes one per attempt; a resume is a new
FunctionCall), classifies that FunctionCall (:func:`classify_functioncall`
— PENDING / OBSERVATION_UNAVAILABLE with exponential backoff / TERMINAL), and
on TERMINAL: accepts the container's ``watch/sweep_<fcid>.json`` ONLY when its
``owner.function_call_id`` is THAT attempt's and it reads ``verified_empty:
true`` (a dead attempt can never inherit an older attempt's certificate);
otherwise it runs the SAME ``sandbox_scope.sweep_owned`` the container's
``finally`` runs, over ``Sandbox.list(app_id=<inspect_modal_sandbox>,
tags=owner_filter(owner))`` — the FULL owner tuple in the filter, so a sibling
cell's or another attempt's sandbox is never listed — and writes its own
certificate under ``~/.local/state/tinycap-sandboxes/<run_id>/sweep_<fcid>.json``.
It exits when every watched attempt is terminal and certified (0), on a sweep it
could not verify (2), or at ``--max-hours`` (3; re-arm). ``--once`` runs one tick
and exits 0 ONLY when that tick certified every attempt — anything still
pending, unobservable, unswept or sharing an attempt id exits 2.

The watcher runs OUTSIDE any container: it reads the volume with the remote
``listdir`` / ``read_file`` APIs (committed state) and never calls
``Volume.reload()``, a container-only operation. Every cell's newest attempt
must carry its OWN FunctionCall id — the launcher spawns one call per
configuration (``fan_out``) — an id two cells share is never classified or
swept (``.map()``'s one id read input 0's state for every cell).

``--row-dir <volume path>`` (tinycap-shared-standup §1.10) watches a ROW ENGINE's
run instead of per-cell out dirs: every ``watch/attempt_<fcid>.json`` under the row
dir. A DECLARED row pointer (``cells`` listed, a ``row_attempt`` block — the GPU
phase's one pointer for the whole row, ``phase coupled|generate``) shares its
FunctionCall id across its cells BY DESIGN: it is classified ONCE per call and, on
TERMINAL, every listed cell's owner ``(run_id, cell, fcid)`` is swept separately
(the container's own certificate for a cell is ``watch/sweep_<fcid>__<cell>.json``,
:func:`row_sweep_certificate_name`, admitted by the same exact-owner rule); a
verifier pointer (``phase verify``, one ``cell``) is classified and swept per its
one cell. Two PER-CELL pointers sharing an id in a row dir are refused exactly as
in the per-cell mode. Exit 0 only when every pointer's call is terminal and every
owner is certified — AND (landing 2, code review L2 H1) every verifier the run's
``watch/verifiers_<run_id>.json`` record expects has a terminal, certified ``phase
verify`` pointer or a terminal no-verifier state (``spawn_failed`` / ``verified`` /
``torn`` / ``skipped``). The row container writes that record before it returns under
``generate`` and the launcher updates it after dispatch, so a tick in the gap between
the GPU call's end and the CPU verifiers' pointers is never DONE. No record (a coupled
row, a landing-1 run) = the rule above alone.

``--sweep-now`` sweeps every listed attempt once and exits. ``--canary
create|check|sweep|terminate --owner-from <attempt pointer>`` is the S7b / S7d
sibling-survival instrument (plan rev 5): a HELD-OPEN sandbox in the production
app carrying that attempt's EXACT owner tags (``owner_tags(owner, "canary",
{"__sample_id__": "canary"})``, the pinned BCB image, ``tail -f /dev/null``,
cpu 0.125, timeout 1,800 s); ``check`` lists it by its owner filter (and asserts
a SIBLING owner's filter does not list it); ``sweep`` runs the very same
``sweep_owned`` by that owner; ``terminate`` cleans up. Run from the grading host
AFTER the attempts have completed and been certified — never during a run.
"""

from __future__ import annotations

import argparse
import builtins
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import sandbox_scope as scope  # noqa: E402

STATE_DIR_DEFAULT = "~/.local/state/tinycap-sandboxes"
DEFAULT_POLL_S = 120.0
DEFAULT_MAX_HOURS = 6.0
EXIT_DONE, EXIT_UNVERIFIED, EXIT_TIMEOUT = 0, 2, 3
CANARY_TASK = "canary"
CANARY_CPU = 0.125
CANARY_MEMORY_MIB = 256
CANARY_TIMEOUT_S = 1800
CANARY_EXIT_WAIT_S = 120.0     # bound on waiting for a terminated canary's exit to be observable


def flush_print(s: str) -> None:
    print(s, flush=True)


# --------------------------------------------------------------------------- #
# pure helpers (tested without Modal)                                          #
# --------------------------------------------------------------------------- #


def cell_out_dir(model: str, cell: str, mode: str, suffix: str, *,
                 serving_engine: str = C.SERVING_ENGINE_LENS) -> str:
    """The launcher's out dir for one configuration (``contracts.cell_out_dir``
    + ``isolate_out_dirs``' suffix) under the volume root ``/vol``.
    native-serving-default: a NATIVE launch plans its out dirs under
    the ``native/`` namespace — ``contracts.gen_root_for_engine`` is THE ONE
    derivation the launcher (``native_gen_root``) and this watcher share, so the
    attempt pointers are looked for where they were written; the lens default is
    byte-identical to before."""
    base = C.cell_out_dir(C.gen_root_for_engine(C.VOL_GEN_ROOT, serving_engine), cell, mode, model_slug=model)
    return f"{base}/{suffix.strip('/')}" if suffix.strip("/") else base


def newest_pointer(pointers: Sequence[Mapping[str, Any]], *, run_id: str) -> Optional[Dict[str, Any]]:
    """The NEWEST attempt pointer of THIS run among ``pointers`` (by
    ``started_at``); pointers of another run id are ignored — a stale pointer
    from an earlier launch in the same out dir is not this run's attempt."""
    mine = [p for p in pointers if str(p.get("run_id")) == str(run_id)]
    if not mine:
        return None
    return dict(max(mine, key=lambda p: float(p.get("started_at") or 0)))


def owner_of_pointer(pointer: Mapping[str, Any]) -> scope.Owner:
    return scope.Owner(str(pointer["run_id"]), str(pointer["cell"]), str(pointer["function_call_id"]))


def certificate_admissible(cert: Optional[Mapping[str, Any]], owner: scope.Owner) -> Optional[str]:
    """Why a ``sweep_<fcid>.json`` does NOT certify ``owner``'s attempt clean —
    None when it does. The owner tuple must match EXACTLY (a resumed attempt
    must never inherit a dead one's verified sweep) and the sweep must have
    verified empty."""
    if cert is None:
        return "no certificate"
    if not isinstance(cert, Mapping):
        return "certificate is not an object"
    got = cert.get("owner")
    if not isinstance(got, Mapping) or {k: str(v) for k, v in got.items()} != owner.as_dict():
        return f"certificate owner {got!r} != this attempt's {owner.as_dict()}"
    if cert.get("verified_empty") is not True:
        return (f"certificate did not verify empty (still_live {cert.get('still_live')}, "
                f"could_not_terminate {len(cert.get('could_not_terminate') or [])}, "
                f"error {cert.get('error')!r})")
    return None


def local_certificate_path(run_id: str, function_call_id: str, state_dir: str = STATE_DIR_DEFAULT) -> Path:
    return Path(state_dir).expanduser() / run_id / f"sweep_{function_call_id}.json"


class Backoff:
    """60 s -> 10 min cap, doubling per consecutive observation failure."""

    def __init__(self, initial: float = 60.0, cap: float = 600.0):
        self.initial, self.cap, self.failures = float(initial), float(cap), 0

    def fail(self) -> float:
        self.failures += 1
        return min(self.cap, self.initial * (2 ** (self.failures - 1)))

    def reset(self) -> None:
        self.failures = 0


# --------------------------------------------------------------------------- #
# Modal-side helpers                                                           #
# --------------------------------------------------------------------------- #


def _volume():
    """The volume handle for REMOTE reads (``listdir`` / ``read_file`` read
    committed state from anywhere). Never ``reload()`` here: that is a
    container-only operation (the ``VolumeReload`` RPC refreshes a volume
    MOUNTED in the running container) and raises from the grading host before the
    first ownership / liveness check."""
    import modal
    return modal.Volume.from_name(C.MODAL_VOLUME)


def shared_attempt_ids(pointers: Mapping[str, Optional[Mapping[str, Any]]]) -> Dict[str, set]:
    """``{function_call_id: {cells}}`` for every attempt id the newest pointers
    of TWO OR MORE cells carry. The launcher's ``.map()`` gave every cell one
    FunctionCall id; its ``.spawn()`` fan-out gives one per cell — a shared id
    here means the liveness of one cell's attempt cannot be read from its
    FunctionCall, and the watcher must not act on it."""
    by_id: Dict[str, set] = {}
    for cell, p in pointers.items():
        if p is not None:
            by_id.setdefault(str(p.get("function_call_id")), set()).add(cell)
    return {fcid: cells for fcid, cells in by_id.items() if len(cells) > 1}


def _volume_relative(out_dir: str) -> str:
    """An out dir as the remote Volume APIs address it: relative to the mount
    root ``/vol``."""
    return out_dir[len("/vol/"):] if out_dir.startswith("/vol/") else out_dir.lstrip("/")


def read_pointers(vol, out_dir: str) -> List[Dict[str, Any]]:
    """Every ``watch/attempt_*.json`` under ``out_dir`` on the volume."""
    watch = f"{_volume_relative(out_dir)}/watch"
    out: List[Dict[str, Any]] = []
    try:
        entries = list(vol.listdir(watch))
    except Exception:  # noqa: BLE001 — no watch dir yet
        return out
    for e in entries:
        name = Path(e.path).name
        if name.startswith("attempt_") and name.endswith(".json"):
            try:
                data = b"".join(vol.read_file(f"{watch}/{name}"))
                out.append(json.loads(data.decode("utf-8")))
            except Exception as ex:  # noqa: BLE001
                flush_print(f"[sandbox-watch] unreadable pointer {name}: {type(ex).__name__}: {ex}")
    return out


def read_certificate(vol, out_dir: str, function_call_id: str, *, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The sweep certificate under ``<out_dir>/watch/`` — today's ``sweep_<fcid>.json`` by default, or ``name`` (the
    row dir's per-cell ``sweep_<fcid>__<cell>.json``, :func:`row_sweep_certificate_name`); absent -> None."""
    try:
        data = b"".join(vol.read_file(f"{_volume_relative(out_dir)}/watch/{name or f'sweep_{function_call_id}.json'}"))
        return json.loads(data.decode("utf-8"))
    except Exception:  # noqa: BLE001 — absent
        return None


def sandbox_app_id(app_name: str = C.BCB_SANDBOX_APP_NAME) -> str:
    import modal
    return str(modal.App.lookup(app_name, create_if_missing=True).app_id)


def owned_lister(owner: scope.Owner, app_id: str):
    import modal
    tags = scope.owner_filter(owner)

    def _list():
        return list(modal.Sandbox.list(app_id=app_id, tags=tags))
    return _list


def sweep_attempt(owner: scope.Owner, *, app_id: str, state_dir: str, by: str) -> Dict[str, Any]:
    """The SAME ``sandbox_scope.sweep_owned`` the container's ``finally`` runs,
    by this attempt's owner filter; the certificate is written locally. A
    listing / termination RPC that raises (a transient Modal failure) is
    RECORDED — an owner-bound local certificate with ``verified_empty: false``
    and the error — and re-raised as :class:`SweepFailed` for the caller's
    backoff, never propagated bare (it used to kill the external
    backstop)."""
    try:
        cert = scope.sweep_owned(list_sandboxes=owned_lister(owner, app_id), owner=owner, by=by,
                                 log=flush_print)
    except Exception as e:  # noqa: BLE001 — recorded, then retried by the caller
        cert = scope.failed_sweep_record(owner, by=by, error=e)
        _write_local_certificate(cert, owner, app_id=app_id, state_dir=state_dir)
        raise SweepFailed(cert["error"]) from e
    return _write_local_certificate(cert, owner, app_id=app_id, state_dir=state_dir)


class SweepFailed(RuntimeError):
    """A sweep whose listing / termination RPC raised; its failed certificate is
    on disk (``verified_empty: false`` + ``error``)."""


def _write_local_certificate(cert: Dict[str, Any], owner: scope.Owner, *, app_id: str, state_dir: str) -> Dict[str, Any]:
    cert["sandbox_app_id"] = app_id
    cert["certificate"] = "sweep"
    p = local_certificate_path(owner.run_id, owner.function_call_id, state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cert, indent=1, sort_keys=True, default=str))
    flush_print(f"[sandbox-watch] certificate -> {p}" + (" (sweep FAILED: " + cert["error"] + ")" if cert.get("error") else ""))
    return cert


# --------------------------------------------------------------------------- #
# FunctionCall observation: exactly one of three states per poll               #
# --------------------------------------------------------------------------- #
#
# Lifted verbatim from the retired Terminal-Bench watcher (its only other user;
# user ruling 2026-09-11). The observation failures are tested FIRST because
# they subclass ``modal.exception.Error`` too: a generic ``except Error`` before
# them would sweep a healthy run on a connectivity or auth blip. The ONE branch
# ahead of them is the not-ready builtin ``TimeoutError`` — an ``OSError``
# subclass, so it must be caught before that tuple; classing it as an
# observation failure made the watcher back off and alert on every tick of a
# run that was simply still running.

PENDING = "PENDING"
CONFIRMED_TERMINAL = "CONFIRMED_TERMINAL"
OBSERVATION_UNAVAILABLE = "OBSERVATION_UNAVAILABLE"
OBSERVATION_STATES = (PENDING, CONFIRMED_TERMINAL, OBSERVATION_UNAVAILABLE)

#: the terminal kinds :func:`classify_functioncall` files
TERMINAL_KINDS = ("returned", "raised", "failed", "function_timeout", "output_expired",
                  "infra_failure")


@dataclasses.dataclass
class Observation:
    state: str
    kind: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[BaseException] = None
    rec: object = None

    @property
    def terminal(self) -> bool:
        return self.state == CONFIRMED_TERMINAL

    @property
    def pending(self) -> bool:
        return self.state == PENDING

    @property
    def unavailable(self) -> bool:
        return self.state == OBSERVATION_UNAVAILABLE

    def describe(self) -> str:
        tail = self.kind or self.reason or ""
        err = f": {type(self.error).__name__}: {str(self.error)[:200]}" if self.error is not None else ""
        return f"{self.state}({tail}){err}"


def classify_functioncall(fc) -> Observation:
    """One ``fc.get(timeout=0)`` → exactly one of the three states, in a fixed
    order (see the module docstring for why the order matters), with the
    not-ready builtin ``TimeoutError`` — SDK 1.5.4's "no output yet", and an
    ``OSError`` subclass — taken ahead of the observation tuple as PENDING.

    The ``except`` ladder wraps ONLY the ``fc.get`` call, so a ``BaseException``
    caught here came OUT of that call. Modal serializes every ``BaseException``
    the function raises except ``KeyboardInterrupt`` / ``GeneratorExit``
    (``container_io_manager.handle_input_exception`` re-raises those unreported)
    and ``_process_result`` re-raises the decoded object as is — so a
    ``SystemExit`` the function let escape reaches the caller as a raw
    ``SystemExit``, which ``except Exception`` does not catch: the watcher used to
    die on it before reconciling, and successor admission (this same classifier)
    died the same way, so nothing could ever resume the run. A remotely decoded
    ``BaseException`` is therefore TERMINAL (``kind="raised"``); a
    ``KeyboardInterrupt`` is by construction LOCAL (this process's own Ctrl-C) and
    propagates untouched; an ``asyncio.CancelledError`` is the local loop
    machinery failing (synchronicity's loop shutting down) — nothing is known,
    so it is an observation failure, never a sweep."""
    import asyncio
    import grpclib.exceptions as GE
    import modal.exception as ME
    try:
        rec = fc.get(timeout=0)
    # 0. NOT READY — the common case. SDK 1.5.4 signals "no output yet" with the
    #    BUILTIN TimeoutError: modal/_functions.py `_Invocation.poll_function`
    #    (what FunctionCall.get delegates to) does `raise TimeoutError()` when the
    #    poll returns no outputs, and that module imports only ExecutionError /
    #    InvalidError / NotFoundError / OutputExpiredError from modal.exception,
    #    so the bare name is builtins.TimeoutError (== asyncio.TimeoutError on
    #    3.11+); modal's synchronicity blocking bridge re-raises it unchanged.
    #    It subclasses OSError, so this branch MUST precede the OSError
    #    observation branch. Nothing else reaches this caller as a builtin
    #    TimeoutError: a transport deadline is grpclib's client-side
    #    asyncio.TimeoutError('Deadline exceeded') or a server
    #    GRPCError(Status.DEADLINE_EXCEEDED), and modal's RPC retry layer
    #    (_utils/grpc_utils.process_exception_before_retry) re-raises the former
    #    as modal.exception.ConnectionError and the latter as the GRPCError —
    #    both branch 1. The tests pin each of these facts from the installed source.
    except builtins.TimeoutError:
        return Observation(PENDING)
    # 1. OBSERVATION FAILURES — the RPC failed; nothing is known about the
    #    function. These subclass modal.exception.Error too, so they MUST precede
    #    every terminal branch.
    except (ME.ConnectionError, ME.AuthError, ME.ServiceError, ME.InternalError,
            GE.GRPCError, GE.StreamTerminatedError, GE.ProtocolError,
            builtins.ConnectionError, OSError) as e:
        return Observation(OBSERVATION_UNAVAILABLE, reason=type(e).__name__, error=e)
    # 2. TERMINAL CLASSES — FunctionTimeoutError / OutputExpiredError SUBCLASS
    #    modal.exception.TimeoutError, so they come before the generic pending branch.
    except ME.FunctionTimeoutError as e:
        return Observation(CONFIRMED_TERMINAL, kind="function_timeout", error=e)
    except ME.OutputExpiredError as e:
        # the result is gone; the attempt's status is read from the volume (CLEAN / records)
        return Observation(CONFIRMED_TERMINAL, kind="output_expired", error=e)
    except ME.TimeoutError:
        # forward compatibility: an SDK that signals "no result yet" with its OWN
        # TimeoutError (1.5.4 raises the builtin — branch 0) is PENDING all the same
        return Observation(PENDING)
    except (ME.RemoteError, ME.ExecutionError) as e:
        return Observation(CONFIRMED_TERMINAL, kind="failed", error=e)
    except ME.InternalFailure as e:
        # a RETURNED GENERIC_STATUS_INTERNAL_FAILURE result, decoded by
        # function_utils._process_result — the container is gone, so this IS
        # termination; distinct from InternalError, an RPC failure (branch 1)
        return Observation(CONFIRMED_TERMINAL, kind="infra_failure", error=e)
    # 3. Any OTHER control-plane error is AMBIGUOUS, never proof of termination.
    except ME.Error as e:
        return Observation(OBSERVATION_UNAVAILABLE, reason=type(e).__name__, error=e)
    # 4. The function's own exception, re-raised by .get().
    except Exception as e:                                   # noqa: BLE001 — the function's own raise
        return Observation(CONFIRMED_TERMINAL, kind="raised", error=e)
    # 5. BaseExceptions out of fc.get: a LOCAL interrupt propagates (the container
    #    runtime never serializes one); a local loop cancellation is an observation
    #    failure; anything else is the function's own decoded raise (SystemExit).
    except KeyboardInterrupt:
        raise
    except asyncio.CancelledError as e:
        return Observation(OBSERVATION_UNAVAILABLE, reason=type(e).__name__, error=e)
    except BaseException as e:                               # noqa: BLE001 — a remotely decoded SystemExit et al.
        return Observation(CONFIRMED_TERMINAL, kind="raised", error=e)
    else:
        return Observation(CONFIRMED_TERMINAL, kind="returned", rec=rec)


def classify(function_call_id: str):
    """:func:`classify_functioncall` on this attempt's FunctionCall."""
    import modal

    return classify_functioncall(modal.FunctionCall.from_id(function_call_id))


# --------------------------------------------------------------------------- #
# the watch loop                                                               #
# --------------------------------------------------------------------------- #


def watch(*, run_id: str, suffix: str, model: str, cells: Sequence[str], mode: str, poll_s: float,
          max_hours: float, state_dir: str, once: bool = False,
          serving_engine: str = C.SERVING_ENGINE_LENS) -> int:
    vol = _volume()
    deadline = time.time() + float(max_hours) * 3600
    backoff = Backoff()
    certified: Dict[str, str] = {}          # cell -> fcid certified clean
    unverified: Dict[str, str] = {}
    app_id: Optional[str] = None
    flush_print(f"[sandbox-watch] run {run_id}: cells {list(cells)} under suffix {suffix!r}, model {model}, "
                f"poll {poll_s:.0f}s, max {max_hours}h"
                + (f", serving engine {serving_engine} (out root {C.gen_root_for_engine(C.VOL_GEN_ROOT, serving_engine)})"
                   if serving_engine != C.SERVING_ENGINE_LENS else ""))
    while True:
        if time.time() > deadline:
            flush_print(f"[sandbox-watch] WATCH-TIMEOUT after {max_hours}h — re-arm with the same arguments")
            return EXIT_TIMEOUT
        if app_id is None:
            # the sandbox app lookup is an RPC too: a failure backs off and retries, never exits
            try:
                app_id = sandbox_app_id()
                flush_print(f"[sandbox-watch] sandbox app {app_id}")
            except Exception as e:  # noqa: BLE001
                delay = backoff.fail()
                flush_print(f"[sandbox-watch] sandbox app lookup UNAVAILABLE ({type(e).__name__}: {e}); retry in {delay:.0f}s")
                time.sleep(delay)
                if once:
                    return EXIT_UNVERIFIED
                continue
        pending = []
        # the newest pointer of EVERY cell first: an attempt id shared by two
        # cells cannot be observed per cell (one FunctionCall for a whole .map
        # read input 0's state for both), so such cells are never classified,
        # never swept, and named as pending until the launcher's
        # per-configuration fan-out gives each its own id
        pointers = {cell: newest_pointer(read_pointers(vol, cell_out_dir(model, cell, mode, suffix,
                                                                          serving_engine=serving_engine)), run_id=run_id)
                    for cell in cells}
        shared = shared_attempt_ids(pointers)
        for cell in cells:
            out_dir = cell_out_dir(model, cell, mode, suffix, serving_engine=serving_engine)
            pointer = pointers[cell]
            if pointer is None:
                pending.append(f"{cell}: no attempt pointer yet under {out_dir}/watch")
                continue
            owner = owner_of_pointer(pointer)
            if certified.get(cell) == owner.function_call_id:
                continue
            if owner.function_call_id in shared:
                pending.append(f"{cell}: attempt {owner.function_call_id} is shared with "
                               f"{sorted(shared[owner.function_call_id] - {cell})} — one FunctionCall cannot be "
                               "observed per cell; NOT swept (the launcher must spawn one call per configuration)")
                continue
            obs = classify(owner.function_call_id)
            if obs.unavailable:
                delay = backoff.fail()
                flush_print(f"[sandbox-watch] {cell}: UNAVAILABLE ({obs.describe()}); retry in {delay:.0f}s")
                time.sleep(delay)
                pending.append(f"{cell}: observation unavailable")
                continue
            backoff.reset()
            if obs.pending:
                pending.append(f"{cell}: attempt {owner.function_call_id} PENDING")
                continue
            flush_print(f"[sandbox-watch] {cell}: attempt {owner.function_call_id} TERMINAL ({obs.describe()})")
            cert = read_certificate(vol, out_dir, owner.function_call_id)
            why = certificate_admissible(cert, owner)
            if why is None:
                flush_print(f"[sandbox-watch] {cell}: the container's own sweep certified clean "
                            f"({len(cert.get('ids_terminated') or [])} terminated)")
                certified[cell] = owner.function_call_id
                continue
            flush_print(f"[sandbox-watch] {cell}: no admissible container certificate ({why}) — sweeping "
                        f"by owner {owner.as_dict()}")
            try:
                mine = sweep_attempt(owner, app_id=app_id, state_dir=state_dir, by="watcher:tinycap_sandbox_watch")
            except SweepFailed as e:
                # the failed certificate is on disk; the cell stays pending and the
                # tick backs off like an unavailable observation
                delay = backoff.fail()
                flush_print(f"[sandbox-watch] {cell}: sweep FAILED ({e}); retry in {delay:.0f}s")
                time.sleep(delay)
                unverified[cell] = owner.function_call_id
                pending.append(f"{cell}: sweep failed ({e})")
                continue
            if mine["verified_empty"]:
                certified[cell] = owner.function_call_id
                unverified.pop(cell, None)
            else:
                unverified[cell] = owner.function_call_id
                pending.append(f"{cell}: sweep NOT verified ({mine['still_live']})")
        if not pending:
            flush_print(f"[sandbox-watch] DONE: every attempt terminal and certified {certified}")
            return EXIT_DONE
        flush_print("[sandbox-watch] " + "; ".join(pending))
        if once:
            # reached only with attempts still pending / unobservable / unswept:
            # a one-shot tick that did not certify everything is not DONE
            return EXIT_UNVERIFIED
        time.sleep(float(poll_s))


def sweep_now(*, run_id: str, suffix: str, model: str, cells: Sequence[str], mode: str, state_dir: str,
              serving_engine: str = C.SERVING_ENGINE_LENS) -> int:
    vol = _volume()
    app_id = sandbox_app_id()
    rc = EXIT_DONE
    for cell in cells:
        out_dir = cell_out_dir(model, cell, mode, suffix, serving_engine=serving_engine)
        pointer = newest_pointer(read_pointers(vol, out_dir), run_id=run_id)
        if pointer is None:
            flush_print(f"[sandbox-watch] {cell}: no attempt pointer under {out_dir}/watch")
            continue
        try:
            cert = sweep_attempt(owner_of_pointer(pointer), app_id=app_id, state_dir=state_dir, by="watcher:sweep-now")
        except SweepFailed as e:
            flush_print(f"[sandbox-watch] {cell}: sweep FAILED ({e}) — its failed certificate is recorded")
            rc = EXIT_UNVERIFIED
            continue
        if not cert["verified_empty"]:
            rc = EXIT_UNVERIFIED
    return rc


# --------------------------------------------------------------------------- #
# tinycap-shared-standup §1.10: the row dir mode (--row-dir)                     #
# --------------------------------------------------------------------------- #
#
# One row container serves every cell of a model row on one engine, so one FunctionCall id stands
# for N cells — the very thing the per-cell mode refuses (`.map()`'s one id read input 0's state
# for every cell). The difference is DECLARATION: a row pointer says which cells ride the
# call (`cells`) and carries the row attempt's facts (`row_attempt`), so the liveness of the one
# container IS the liveness of every listed cell, and each cell's sandboxes are still swept by its
# OWN exact owner tuple. An undeclared shared id (two per-cell pointers, one fcid) stays refused.

ROW_POINTER_CELLS_KEY = "cells"
ROW_ATTEMPT_KEY = "row_attempt"
POINTER_PHASE_KEY = "phase"
PHASE_COUPLED, PHASE_GENERATE, PHASE_VERIFY = "coupled", "generate", "verify"


def is_row_pointer(pointer: Mapping[str, Any]) -> bool:
    """A pointer that lists the cells riding its FunctionCall (the GPU phase's row pointer)."""
    return isinstance(pointer, Mapping) and isinstance(pointer.get(ROW_POINTER_CELLS_KEY), list)


def row_pointer_undeclared(pointer: Mapping[str, Any]) -> Optional[str]:
    """Why a row pointer is NOT declared (None when it is): no cells listed, a non-string cell, a
    duplicate cell, or no ``row_attempt`` block. An undeclared row pointer is never classified or
    swept — the watcher names it as pending until the runner writes it whole."""
    cells = pointer.get(ROW_POINTER_CELLS_KEY)
    if not cells:
        return "lists no cells"
    if any(not isinstance(c, str) or not c for c in cells):
        return f"lists a non-string cell ({cells!r})"
    if len(set(cells)) != len(cells):
        return f"lists a cell twice ({cells!r})"
    if not isinstance(pointer.get(ROW_ATTEMPT_KEY), Mapping):
        return f"carries no {ROW_ATTEMPT_KEY} block"
    return None


def pointer_owners(pointer: Mapping[str, Any]) -> List[scope.Owner]:
    """The owner tuples a pointer stands for: one per listed cell on a row pointer, the one cell
    otherwise (a verifier pointer, or a per-cell pointer written into a row dir)."""
    if is_row_pointer(pointer):
        return [scope.Owner(str(pointer["run_id"]), str(c), str(pointer["function_call_id"]))
                for c in pointer[ROW_POINTER_CELLS_KEY]]
    return [owner_of_pointer(pointer)]


def row_sweep_certificate_name(function_call_id: str, cell: str) -> str:
    """``sweep_<fcid>__<cell>.json`` — the per-(call, cell) certificate the row container writes
    under ``<row dir>/watch/`` (the runner's ``sweep_certificate_path`` gains the cell suffix in row
    mode); the cell is verbatim (P35: ``__`` inside a cell id is fine, the separator is the one after
    the call id, which never carries one)."""
    return f"sweep_{function_call_id}__{cell}.json"


def read_row_certificate(vol, row_dir: str, function_call_id: str, cell: str) -> Optional[Dict[str, Any]]:
    """The row container's per-cell certificate ``sweep_<fcid>__<cell>.json`` under ``<row_dir>/watch/``."""
    return read_certificate(vol, row_dir, function_call_id, name=row_sweep_certificate_name(function_call_id, cell))


def adopted_call_ids(record: Optional[Mapping[str, Any]]) -> set:
    """The call ids this run's record JOINED (state ``joined``): pointers of those calls are this run's to watch even
    though they carry the run id that spawned them."""
    if not isinstance(record, Mapping) or not isinstance(record.get("cells"), Mapping):
        return set()
    return {str(ent["function_call_id"]) for ent in record["cells"].values()
            if isinstance(ent, Mapping) and ent.get("state") == VERIFIER_STATE_JOINED and ent.get("function_call_id")}


def pointers_by_call(pointers: Sequence[Mapping[str, Any]], *, run_id: str, adopt: Sequence[str] = ()) -> Dict[str, List[Dict[str, Any]]]:
    """This run's pointers grouped by FunctionCall id (every pointer kept, so a shared id can be seen); ``adopt`` —
    call ids the run's record joined — are grouped too, whatever run id their pointers carry."""
    adopted = {str(a) for a in adopt}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in pointers:
        if not p.get("function_call_id"):
            continue
        if str(p.get("run_id")) != str(run_id) and str(p["function_call_id"]) not in adopted:
            continue
        out.setdefault(str(p["function_call_id"]), []).append(dict(p))
    return out


def undeclared_shared_cells(group: Sequence[Mapping[str, Any]]) -> Optional[List[str]]:
    """The cells an fcid is shared by WITHOUT a declaration — two or more per-cell pointers with
    distinct cells, or a per-cell pointer beside a row pointer — else None. The per-cell mode's
    refusal, kept byte-for-byte in spirit: such a call is never classified or swept."""
    singles = [p for p in group if not is_row_pointer(p)]
    rows = [p for p in group if is_row_pointer(p)]
    cells = sorted({str(p.get("cell")) for p in singles})
    if len(cells) > 1 or (rows and singles):
        return sorted(set(cells) | {c for r in rows for c in (r.get(ROW_POINTER_CELLS_KEY) or [])})
    return None


def pointer_label(pointer: Mapping[str, Any]) -> str:
    if is_row_pointer(pointer):
        return f"row {pointer.get('row') or pointer.get('model')} [{pointer.get(POINTER_PHASE_KEY) or PHASE_COUPLED}] cells {list(pointer[ROW_POINTER_CELLS_KEY])}"
    phase = pointer.get(POINTER_PHASE_KEY)
    return f"{'verify ' if phase == PHASE_VERIFY else ''}{pointer.get('cell')}"


#: landing 2, code review L2 H1: the run's expected-verifiers record under ``<row dir>/watch/`` — the harness's
#: ``tinycap_served.VERIFIERS_RECORD_TEMPLATE`` (pinned equal by test); its ``cells`` = {label: {cell, out_dir, state, …}}
VERIFIERS_RECORD_TEMPLATE = "verifiers_{run_id}.json"
VERIFIER_STATE_EXPECTED, VERIFIER_STATE_SPAWNING, VERIFIER_STATE_SPAWNED = "expected", "spawning", "spawned"
#: the run JOINED an earlier run's still-running verifier (the entry's call id) — non-terminal; the watcher
#: ADOPTS that call's pointer whatever run id it carries (its owner tuple is the pointer's own), so DONE waits for it
VERIFIER_STATE_JOINED = "joined"
#: the run refused to dispatch because an earlier owner is unresolved — pending, never DONE
VERIFIER_STATE_BLOCKED = "blocked"
#: a listed cell in one of these states gets NO verifier under this run id (the launcher's spawn failed; a resume found
#: the store already verified, or torn and parked; nothing to dispatch) — the watcher does not wait for a pointer
VERIFIER_TERMINAL_STATES = ("spawn_failed", "verified", "torn", "skipped")
#: the exceptions a Volume read raises for a path that is CONFIRMED absent (the builtin, and modal's NotFoundError by
#: name — never imported here, so the pure helpers stay Modal-free); anything else is an UNAVAILABLE read
_NOT_FOUND_NAMES = ("NotFoundError", "FileNotFoundError")


def verifiers_record_name(run_id: str) -> str:
    return VERIFIERS_RECORD_TEMPLATE.format(run_id=str(run_id))


def _is_not_found(e: BaseException) -> bool:
    return isinstance(e, FileNotFoundError) or type(e).__name__ in _NOT_FOUND_NAMES


def read_verifiers_record(vol, row_dir: str, run_id: str) -> Optional[Dict[str, Any]]:
    """``<row dir>/watch/verifiers_<run_id>.json`` — None ONLY when the read CONFIRMS the file absent (a not-found
    error: no expectation was recorded); any other read failure (a transient Volume / RPC error) is
    ``{"_unavailable": …}`` — the tick withholds DONE and retries, never reads "nothing owed" off a failed read
    (a blip in the GPU→CPU gap used to print DONE); an unreadable or non-object file is
    ``{"_problem": …}`` — withheld by name."""
    path = f"{_volume_relative(row_dir)}/watch/{verifiers_record_name(run_id)}"
    try:
        data = b"".join(vol.read_file(path))
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            return None
        return {"_unavailable": f"{type(e).__name__}: {str(e)[:160]}"}
    try:
        doc = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return {"_problem": f"unreadable ({type(e).__name__}: {str(e)[:120]})"}
    if not isinstance(doc, Mapping):
        return {"_problem": "not a JSON object"}
    return dict(doc)


def generate_row_pointer_present(pointers: Sequence[Mapping[str, Any]], *, run_id: str) -> bool:
    """Does this run carry a GENERATE-phase row pointer (a GPU row that owes CPU verifiers by construction)?"""
    return any(is_row_pointer(p) and str(p.get("run_id")) == str(run_id) and p.get(POINTER_PHASE_KEY) == PHASE_GENERATE
               for p in pointers)


def _same_dir(a: Any, b: Any) -> bool:
    return bool(a) and bool(b) and str(a).rstrip("/") == str(b).rstrip("/")


def outstanding_verifiers(record: Optional[Mapping[str, Any]], *, run_id: str, pointers: Sequence[Mapping[str, Any]],
                          certified: Mapping[Tuple[str, str], bool], generate_row: Optional[bool] = None) -> List[str]:
    """The pending lines the expected-verifiers record adds to a tick (empty = nothing outstanding): for every
    listed cell whose state is not terminal, a ``phase verify`` pointer of this run must match it — by the recorded
    ``function_call_id`` once the launcher has spawned it, else by ``(cell, out_dir)`` (the verifier's pointer names
    the cell and the store root it verifies) — and that pointer's owner must be certified. A record of another run
    id is not this run's expectation; a torn record withholds DONE by name; an UNAVAILABLE read withholds and retries.
    ``generate_row`` (derived from ``pointers`` when None): a GENERATE-phase row pointer of this run OWES a record —
    the row container writes it before returning, for every generate row — so a confirmed-missing record fails CLOSED
    (withheld, named: the container never got to write it) instead of open; a coupled row without one owes nothing."""
    if generate_row is None:
        generate_row = generate_row_pointer_present(pointers, run_id=run_id)
    if record is None:
        if generate_row:
            return [f"no expected-verifiers record {verifiers_record_name(run_id)} for this generate row — the row container "
                    "writes it before returning, so its absence means the row never got that far; DONE withheld"]
        return []
    if record.get("_unavailable"):
        return [f"verifiers record {verifiers_record_name(run_id)}: read UNAVAILABLE ({record['_unavailable']}) — retrying; DONE withheld"]
    if record.get("_problem"):
        return [f"verifiers record {verifiers_record_name(run_id)}: {record['_problem']} — DONE withheld"]
    if str(record.get("run_id")) != str(run_id):
        return []
    cells = record.get("cells")
    if not isinstance(cells, Mapping):
        return [f"verifiers record {verifiers_record_name(run_id)} lists no cells — DONE withheld"]
    adopted = adopted_call_ids(record)
    mine = [p for p in pointers if p.get(POINTER_PHASE_KEY) == PHASE_VERIFY and p.get("function_call_id")
            and (str(p.get("run_id")) == str(run_id) or str(p["function_call_id"]) in adopted)]
    out: List[str] = []
    for label, ent in sorted(cells.items()):
        ent = ent if isinstance(ent, Mapping) else {}
        state = str(ent.get("state") or VERIFIER_STATE_EXPECTED)
        if state in VERIFIER_TERMINAL_STATES:
            continue
        if state == VERIFIER_STATE_BLOCKED:
            prior = ent.get("prior") if isinstance(ent.get("prior"), Mapping) else {}
            out.append(f"verifier for {label}: BLOCKED — an earlier owner is unresolved (run {prior.get('run_id')}, {prior.get('state')!r}"
                       f"{', call ' + str(prior['function_call_id']) if prior.get('function_call_id') else ', no call id'}); "
                       "nothing dispatched under this run; DONE withheld")
            continue
        fcid = ent.get("function_call_id")
        if fcid:
            matches = [p for p in mine if str(p.get("function_call_id")) == str(fcid)]
        else:
            matches = [p for p in mine if str(p.get("cell")) == str(ent.get("cell")) and _same_dir(p.get("out_dir"), ent.get("out_dir"))]
        if not matches:
            out.append(f"verifier for {label} ({state}{f', call {fcid}' if fcid else ''}): no phase-verify pointer under the row "
                       "dir yet — DONE withheld")
            continue
        if not any(certified.get((str(p["function_call_id"]), str(p.get("cell")))) for p in matches):
            out.append(f"verifier for {label}: attempt(s) {sorted(str(p['function_call_id']) for p in matches)} not yet terminal "
                       "and certified — DONE withheld")
    return out


def watch_row_dir(*, run_id: str, row_dir: str, poll_s: float, max_hours: float, state_dir: str,
                  once: bool = False) -> int:
    """The row-dir watch loop: every pointer under ``<row_dir>/watch`` of this run; a call classified
    ONCE per tick however many cells it carries; on TERMINAL every owner swept separately (the
    container's own per-cell certificate accepted first); exit codes as :func:`watch`."""
    vol = _volume()
    deadline = time.time() + float(max_hours) * 3600
    backoff = Backoff()
    certified: Dict[Tuple[str, str], bool] = {}      # (fcid, cell) -> certified clean
    unverified: Dict[Tuple[str, str], bool] = {}
    app_id: Optional[str] = None
    flush_print(f"[sandbox-watch] run {run_id}: row dir {row_dir}, poll {poll_s:.0f}s, max {max_hours}h")
    while True:
        if time.time() > deadline:
            flush_print(f"[sandbox-watch] WATCH-TIMEOUT after {max_hours}h — re-arm with the same arguments")
            return EXIT_TIMEOUT
        if app_id is None:
            try:
                app_id = sandbox_app_id()
                flush_print(f"[sandbox-watch] sandbox app {app_id}")
            except Exception as e:  # noqa: BLE001
                delay = backoff.fail()
                flush_print(f"[sandbox-watch] sandbox app lookup UNAVAILABLE ({type(e).__name__}: {e}); retry in {delay:.0f}s")
                time.sleep(delay)
                if once:
                    return EXIT_UNVERIFIED
                continue
        pending: List[str] = []
        raw_pointers = read_pointers(vol, row_dir)
        # the run's record first: a JOINED verifier's pointer (another run's id) is this run's to classify and sweep
        record = read_verifiers_record(vol, row_dir, run_id)
        groups = pointers_by_call(raw_pointers, run_id=run_id, adopt=sorted(adopted_call_ids(record)))
        if not groups:
            pending.append(f"no attempt pointer yet under {row_dir}/watch")
        for fcid, group in sorted(groups.items()):
            shared = undeclared_shared_cells(group)
            if shared is not None:
                pending.append(f"attempt {fcid} is shared with {shared} without a row declaration — one FunctionCall cannot "
                               "be observed per cell; NOT swept (a row pointer must list its cells)")
                continue
            pointer = newest_pointer(group, run_id=run_id) or newest_pointer(group, run_id=str(group[0].get("run_id")))   # an adopted call: its own run
            label = pointer_label(pointer) + (f" [joined from run {pointer.get('run_id')}]" if str(pointer.get("run_id")) != str(run_id) else "")
            if is_row_pointer(pointer):
                why = row_pointer_undeclared(pointer)
                if why:
                    pending.append(f"{label}: row attempt {fcid} {why} — not classified")
                    continue
            owners = pointer_owners(pointer)
            todo = [o for o in owners if not certified.get((fcid, o.cell))]
            if not todo:
                continue
            obs = classify(fcid)
            if obs.unavailable:
                delay = backoff.fail()
                flush_print(f"[sandbox-watch] {label}: UNAVAILABLE ({obs.describe()}); retry in {delay:.0f}s")
                time.sleep(delay)
                pending.append(f"{label}: observation unavailable")
                continue
            backoff.reset()
            if obs.pending:
                pending.append(f"{label}: attempt {fcid} PENDING")
                continue
            standup = (pointer.get(ROW_ATTEMPT_KEY) or {}).get("standup_id") if is_row_pointer(pointer) else None
            flush_print(f"[sandbox-watch] {label}: attempt {fcid} TERMINAL ({obs.describe()})"
                        + (f"; standup {standup}" if standup else
                           ("; the standup never declared readiness — nothing was served, the owners are swept all the same"
                            if is_row_pointer(pointer) else "")))
            for owner in todo:
                cert = read_row_certificate(vol, row_dir, fcid, owner.cell)
                why = certificate_admissible(cert, owner)
                if why is None:
                    flush_print(f"[sandbox-watch] {owner.cell}: the container's own sweep certified clean "
                                f"({len(cert.get('ids_terminated') or [])} terminated)")
                    certified[(fcid, owner.cell)] = True
                    continue
                flush_print(f"[sandbox-watch] {owner.cell}: no admissible container certificate ({why}) — sweeping "
                            f"by owner {owner.as_dict()}")
                try:
                    mine = sweep_attempt(owner, app_id=app_id, state_dir=state_dir, by="watcher:tinycap_sandbox_watch")
                except SweepFailed as e:
                    delay = backoff.fail()
                    flush_print(f"[sandbox-watch] {owner.cell}: sweep FAILED ({e}); retry in {delay:.0f}s")
                    time.sleep(delay)
                    unverified[(fcid, owner.cell)] = True
                    pending.append(f"{owner.cell}: sweep failed ({e})")
                    continue
                if mine["verified_empty"]:
                    certified[(fcid, owner.cell)] = True
                    unverified.pop((fcid, owner.cell), None)
                else:
                    unverified[(fcid, owner.cell)] = True
                    pending.append(f"{owner.cell}: sweep NOT verified ({mine['still_live']})")
        # landing 2 (code review L2 H1): the run's expected verifiers — DONE is withheld while one is still owed a
        # terminal, certified pointer (the gap between the GPU call's end and the CPU containers' pointers)
        pending += outstanding_verifiers(record, run_id=run_id, pointers=raw_pointers,
                                         certified=certified, generate_row=generate_row_pointer_present(raw_pointers, run_id=run_id))
        if not pending:
            done = sorted(f"{c}@{f}" for (f, c) in certified)
            flush_print(f"[sandbox-watch] DONE: every attempt terminal and every owner certified {done}")
            return EXIT_DONE
        flush_print("[sandbox-watch] " + "; ".join(pending))
        if once:
            return EXIT_UNVERIFIED
        time.sleep(float(poll_s))


def sweep_now_row_dir(*, run_id: str, row_dir: str, state_dir: str) -> int:
    """``--sweep-now`` for a row dir: every owner of every declared pointer swept once."""
    vol = _volume()
    app_id = sandbox_app_id()
    rc = EXIT_DONE
    groups = pointers_by_call(read_pointers(vol, row_dir), run_id=run_id)
    if not groups:
        flush_print(f"[sandbox-watch] no attempt pointer under {row_dir}/watch")
    for fcid, group in sorted(groups.items()):
        if undeclared_shared_cells(group) is not None:
            flush_print(f"[sandbox-watch] attempt {fcid}: shared without a row declaration — NOT swept")
            rc = EXIT_UNVERIFIED
            continue
        pointer = newest_pointer(group, run_id=run_id)
        if is_row_pointer(pointer) and row_pointer_undeclared(pointer):
            flush_print(f"[sandbox-watch] attempt {fcid}: row pointer {row_pointer_undeclared(pointer)} — NOT swept")
            rc = EXIT_UNVERIFIED
            continue
        for owner in pointer_owners(pointer):
            try:
                cert = sweep_attempt(owner, app_id=app_id, state_dir=state_dir, by="watcher:sweep-now")
            except SweepFailed as e:
                flush_print(f"[sandbox-watch] {owner.cell}: sweep FAILED ({e}) — its failed certificate is recorded")
                rc = EXIT_UNVERIFIED
                continue
            if not cert["verified_empty"]:
                rc = EXIT_UNVERIFIED
    return rc


# --------------------------------------------------------------------------- #
# canaries (S7b / S7d): held-open sandboxes with an attempt's exact owner tags  #
# --------------------------------------------------------------------------- #


def canary_state_path(run_id: str, state_dir: str) -> Path:
    return Path(state_dir).expanduser() / run_id / "canaries.json"


def _wait_gone(sb: Any, timeout_s: float, *, poll_s: float = 2.0) -> Tuple[bool, float]:
    """``(exited, seconds waited)``: ``poll()`` non-None within ``timeout_s``.
    A sandbox's termination is acknowledged before its exit is observable on a
    handle, so any check that a sweep or a terminate took effect waits here."""
    t0 = time.time()
    while sb.poll() is None and time.time() - t0 < timeout_s:
        time.sleep(poll_s)
    return sb.poll() is not None, time.time() - t0


def canary(action: str, *, owner_from: str, state_dir: str, sibling_from: Optional[str] = None) -> int:
    import modal

    pointer = json.loads(Path(owner_from).read_text())
    owner = owner_of_pointer(pointer)
    run_id = owner.run_id
    app = modal.App.lookup(C.BCB_SANDBOX_APP_NAME, create_if_missing=True)
    app_id = str(app.app_id)
    state = canary_state_path(run_id, state_dir)
    doc = json.loads(state.read_text()) if state.is_file() else {}
    key = owner.function_call_id
    if action == "create":
        tags = scope.owner_tags(owner, CANARY_TASK, {"__sample_id__": CANARY_TASK})
        sb = modal.Sandbox.create("tail", "-f", "/dev/null", app=app, tags=tags,
                                  image=modal.Image.from_registry(C.BCB_SANDBOX_IMAGE),
                                  cpu=CANARY_CPU, memory=CANARY_MEMORY_MIB, timeout=CANARY_TIMEOUT_S,
                                  block_network=True)
        doc[key] = {"sandbox_id": sb.object_id, "owner": owner.as_dict(), "tags": tags,
                    "created_at": time.time(), "app_id": app_id}
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps(doc, indent=1, sort_keys=True))
        flush_print(f"[canary] created {sb.object_id} for owner {owner.as_dict()} -> {state}")
        return EXIT_DONE
    entry = doc.get(key)
    if entry is None:
        flush_print(f"[canary] no canary recorded for attempt {key} in {state}")
        return EXIT_UNVERIFIED
    sid = entry["sandbox_id"]
    sb = modal.Sandbox.from_id(sid)
    if action == "check":
        live = sb.poll() is None
        listed = [s.object_id for s in modal.Sandbox.list(app_id=app_id, tags=scope.owner_filter(owner))]
        by_sibling: Optional[bool] = None
        if sibling_from:
            sib = owner_of_pointer(json.loads(Path(sibling_from).read_text()))
            by_sibling = sid in [s.object_id for s in modal.Sandbox.list(app_id=app_id, tags=scope.owner_filter(sib))]
        rep = {"sandbox_id": sid, "live": live, "listed_by_own_owner_filter": sid in listed,
               "listed_by_sibling_owner_filter": by_sibling, "tags": scope.read_tags(sb)}
        flush_print(f"[canary] check {json.dumps(rep, sort_keys=True)}")
        ok = live and sid in listed and (by_sibling is not True)
        return EXIT_DONE if ok else EXIT_UNVERIFIED
    if action == "sweep":
        cert = sweep_attempt(owner, app_id=app_id, state_dir=state_dir, by="canary:sweep")
        # ``terminate()`` is acknowledged before the sandbox has EXITED: the sweep
        # verifies by a fresh owner-filtered listing (which drops a terminating
        # sandbox at once), while ``poll()`` on this handle reports None until the
        # exit lands — observed live 2026-09-10 (S7b: ids_terminated held the
        # canary, poll() still None in the same second, gone moments later). So
        # wait for the exit the way ``terminate`` does, bounded, and report the wait.
        gone, waited_s = _wait_gone(sb, CANARY_EXIT_WAIT_S)
        flush_print(f"[canary] after sweep_owned by {owner.as_dict()}: canary {sid} "
                    f"{'terminated' if gone else 'STILL LIVE'} (exit observed after {waited_s:.0f}s); "
                    f"ids_terminated {cert['ids_terminated']}")
        return EXIT_DONE if (gone and cert["verified_empty"] and sid in cert["ids_terminated"]) else EXIT_UNVERIFIED
    if action == "terminate":
        sb.terminate()
        gone, _ = _wait_gone(sb, CANARY_EXIT_WAIT_S)
        entry["terminated_at"] = time.time() if gone else None
        state.write_text(json.dumps(doc, indent=1, sort_keys=True))
        flush_print(f"[canary] terminate {sid}: {'gone' if gone else 'STILL LIVE'}")
        return EXIT_DONE if gone else EXIT_UNVERIFIED
    raise SystemExit(f"unknown canary action {action!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", help="the launcher's run id ([tinycap] run id: …)")
    ap.add_argument("--suffix", default="", help="the launch's --out-suffix")
    ap.add_argument("--model", default=C.MODEL_SLUG)
    ap.add_argument("--cells", default="", help="csv of the launch's cells")
    ap.add_argument("--mode", default=C.THINKING_OFF, help="the launch's thinking mode (off)")
    ap.add_argument("--serving-path", dest="serving_path", default=C.SERVING_ENGINE_LENS, choices=list(C.SERVING_ENGINES),
                    help="the launch's --serving-path: a native launch's attempt pointers live under the native/ out "
                         "namespace (the launcher prints this flag on a native launch's watcher command)")
    ap.add_argument("--row-dir", dest="row_dir", default=None,
                    help="tinycap-shared-standup §1.10: watch a ROW ENGINE's run — every attempt pointer under "
                         "<row-dir>/watch (the volume path the launcher prints, /vol/tinycap/gen/rows/<row>/<suffix>); "
                         "--cells / --suffix / --model are not read in this mode")
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL_S)
    ap.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    ap.add_argument("--state-dir", default=STATE_DIR_DEFAULT)
    ap.add_argument("--once", action="store_true", help="one tick, then exit")
    ap.add_argument("--sweep-now", action="store_true", help="sweep every listed attempt once and exit")
    ap.add_argument("--canary", choices=("create", "check", "sweep", "terminate"), default=None)
    ap.add_argument("--owner-from", default=None, help="an attempt pointer (watch/attempt_<fcid>.json), for --canary")
    ap.add_argument("--sibling-from", default=None, help="another attempt's pointer: --canary check asserts "
                                                        "ITS owner filter does not list this canary")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.canary:
        if not a.owner_from:
            raise SystemExit("--canary needs --owner-from <attempt pointer>")
        return canary(a.canary, owner_from=a.owner_from, state_dir=a.state_dir, sibling_from=a.sibling_from)
    if a.row_dir:
        if not a.run_id:
            raise SystemExit("--run-id is required with --row-dir")
        if a.sweep_now:
            return sweep_now_row_dir(run_id=a.run_id, row_dir=a.row_dir, state_dir=a.state_dir)
        return watch_row_dir(run_id=a.run_id, row_dir=a.row_dir, poll_s=a.poll, max_hours=a.max_hours,
                             state_dir=a.state_dir, once=a.once)
    if not a.run_id or not a.cells:
        raise SystemExit("--run-id and --cells are required")
    cells = [c for c in a.cells.split(",") if c]
    if a.sweep_now:
        return sweep_now(run_id=a.run_id, suffix=a.suffix, model=a.model, cells=cells, mode=a.mode,
                         state_dir=a.state_dir, serving_engine=a.serving_path)
    return watch(run_id=a.run_id, suffix=a.suffix, model=a.model, cells=cells, mode=a.mode, poll_s=a.poll,
                 max_hours=a.max_hours, state_dir=a.state_dir, once=a.once, serving_engine=a.serving_path)


if __name__ == "__main__":
    raise SystemExit(main())
