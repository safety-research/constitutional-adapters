"""Sandbox OWNERSHIP for the bcb-bfcl-200 BigCodeBench leg — pure (no Modal
import, no I/O beyond an optional JSONL append), so the ``modal_owned``
provider inside the serving container, the container's ``finally`` sweep, the
grading host's watcher and the tests all read ONE definition.

The OWNER of a sandbox is the tuple ``(run_id, cell, function_call_id)``:
the launcher invocation (``run_id``, minted once per ``modal run``), the cell
whose container created it, and that container's Modal FunctionCall id (the
ATTEMPT — a resumed cell is a new FunctionCall). Every create-time tag, every
list filter, every pointer / record / certificate carries the FULL tuple, so:

* a cell's sweep can never terminate a sibling cell's live sandboxes (both
  share ``run_id``; they differ in ``cell``), and
* a resumed attempt can never inherit a dead attempt's ``verified_empty``
  certificate (they differ in ``function_call_id``).

Modal tag values are protobuf STRINGS (``api_pb2.SandboxTag(tag_value=v)``):
an int — BigCodeBench's ``__sample_id__`` IS an int — is a ``TypeError``
before the RPC, so :func:`owner_tags` stringifies every name and value.

The sweep (:func:`sweep_owned`) is a terminate-then-verify loop over the
owner's own sandboxes and nothing else: terminate ONLY the live
sandboxes whose tags are IN SCOPE for the owner; a sandbox whose tags cannot
be read, or that is not the owner's, is always LEFT and reported; after every
round re-list and verify BY OBJECT ID; ``verified_empty`` only when no
terminate failed, nothing selected still lists, nothing in scope still lists
and every listed sandbox's tags were readable. A ``Sandbox.terminate()`` is
asynchronous (wait=False REQUESTS termination), so an accepted one may list
for another round.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional

__all__ = [
    "Owner", "OWNER_ENV_RUN_ID", "OWNER_ENV_CELL", "OWNER_ENV_FUNCTION_CALL_ID",
    "SANDBOX_RECORD_ENV", "OWNER_ENV_VARS", "CREATED_BY_TAG", "TAG_RUN_ID", "TAG_CELL",
    "TAG_FUNCTION_CALL_ID", "TAG_TASK", "TAG_SAMPLE_ID", "OWNER_TAG_NAMES",
    "EVENT_CREATED", "EVENT_TERMINATED", "EVENT_TERMINATE_UNVERIFIED",
    "EVENT_CLEANUP_SKIPPED_INTERRUPTED", "EVENT_CREATE_RATE_LIMITED", "EVENTS", "VERIFIED_BY",
    "owner_from_env", "owner_env", "owner_tags", "owner_filter", "in_scope",
    "append_event", "read_events", "read_tags", "sweep_owned", "failed_sweep_record", "SWEEP_RECORD_FIELDS",
]

#: The environment the container hands the driver (and the driver's in-process
#: Inspect provider): the owner tuple + the durable record path.
OWNER_ENV_RUN_ID = "TINYCAP_OWNER_RUN_ID"
OWNER_ENV_CELL = "TINYCAP_OWNER_CELL"
OWNER_ENV_FUNCTION_CALL_ID = "TINYCAP_OWNER_FUNCTION_CALL_ID"
SANDBOX_RECORD_ENV = "TINYCAP_SANDBOX_RECORD"
OWNER_ENV_VARS = (OWNER_ENV_RUN_ID, OWNER_ENV_CELL, OWNER_ENV_FUNCTION_CALL_ID)

#: inspect-sandboxes' own marker tag (``INSPECT_SANDBOX_TAG``), kept so its
#: ``inspect sandbox cleanup`` CLI still lists our sandboxes.
CREATED_BY_TAG = ("created_by", "inspect-ai")
TAG_RUN_ID = "tinycap.run_id"
TAG_CELL = "tinycap.cell"
TAG_FUNCTION_CALL_ID = "tinycap.function_call_id"
TAG_TASK = "tinycap.task"
TAG_SAMPLE_ID = "tinycap.sample_id"
OWNER_TAG_NAMES = (TAG_RUN_ID, TAG_CELL, TAG_FUNCTION_CALL_ID)

#: ``sandboxes.jsonl`` events (one JSON object per line, appended, fsync'd).
EVENT_CREATED = "created"
#: ONLY after termination was OBSERVED — ``poll()`` returned an exit code, or
#: a re-list no longer showed the id (``verified_by``: ``poll`` | ``relist``).
EVENT_TERMINATED = "terminated"
#: ``terminate()`` was requested but the observation FAILED or did not confirm.
EVENT_TERMINATE_UNVERIFIED = "terminate_unverified"
#: The provider's ``sample_cleanup`` returned early on ``interrupted`` (a
#: cancelled sample): nothing was terminated here — the task-level cleanup or
#: the sweep owns it.
EVENT_CLEANUP_SKIPPED_INTERRUPTED = "cleanup_skipped_interrupted"
#: tinycap-verify-rate-limit (2026-09-30): Modal REFUSED a create with its sandbox-creation
#: rate limit (``ResourceExhaustedError``, 5/s per workspace) and the owned provider is
#: backing off before it retries — ``attempt`` (0-based), ``wait_s``, ``elapsed_s``,
#: ``error``; ``sandbox_id`` is empty (nothing was created). One line per retry, so the
#: record shows how often the limit bit and how long each verifier waited on it.
EVENT_CREATE_RATE_LIMITED = "create_rate_limited"
EVENTS = (EVENT_CREATED, EVENT_TERMINATED, EVENT_TERMINATE_UNVERIFIED,
          EVENT_CLEANUP_SKIPPED_INTERRUPTED, EVENT_CREATE_RATE_LIMITED)
VERIFIED_BY = ("poll", "relist")

SWEEP_RECORD_FIELDS = (
    "swept_at", "by", "owner", "ids_terminated", "left", "could_not_terminate",
    "still_live", "tags_unreadable", "verified_empty", "rounds", "n_listed_first_round")


class Owner(NamedTuple):
    run_id: str
    cell: str
    function_call_id: str

    def as_dict(self) -> Dict[str, str]:
        return {"run_id": self.run_id, "cell": self.cell, "function_call_id": self.function_call_id}


def owner_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[Owner]:
    """The owner the container declared, or ``None`` when NONE of the three
    variables is set (a hand run outside the launcher). A PARTIAL tuple refuses:
    two of three names is a wiring error, never an owner."""
    env = os.environ if environ is None else environ
    vals = [env.get(k) for k in OWNER_ENV_VARS]
    present = [v for v in vals if v]
    if not present:
        return None
    if len(present) != len(OWNER_ENV_VARS):
        raise ValueError(f"partial sandbox owner in the environment: "
                         f"{dict(zip(OWNER_ENV_VARS, vals))} — all three or none")
    return Owner(*[str(v) for v in vals])


def owner_env(owner: Owner, record_path: Optional[str | Path] = None) -> Dict[str, str]:
    """The environment block the container adds for the driver process."""
    out = dict(zip(OWNER_ENV_VARS, owner))
    if record_path is not None:
        out[SANDBOX_RECORD_ENV] = str(record_path)
    return out


def owner_tags(owner: Owner, task_name: str, metadata: Mapping[str, Any]) -> Dict[str, str]:
    """EVERY tag a sandbox is CREATED with — all names and values ``str()``:
    inspect-sandboxes' marker, the owner tuple, the task and the sample id
    (``metadata["__sample_id__"]``, an int for BigCodeBench)."""
    sid = metadata.get("__sample_id__") if isinstance(metadata, Mapping) else None
    return {**owner_filter(owner), TAG_TASK: str(task_name), TAG_SAMPLE_ID: "" if sid is None else str(sid)}


def owner_filter(owner: Owner) -> Dict[str, str]:
    """The ``Sandbox.list(tags=...)`` filter: inspect-sandboxes' marker + the
    FULL owner tuple. Modal returns only sandboxes carrying AT LEAST these
    tags, so a sibling cell's or another attempt's sandbox is never listed."""
    return {CREATED_BY_TAG[0]: CREATED_BY_TAG[1],
            TAG_RUN_ID: str(owner.run_id), TAG_CELL: str(owner.cell),
            TAG_FUNCTION_CALL_ID: str(owner.function_call_id)}


def in_scope(tags: Optional[Mapping[str, str]], owner: Owner) -> bool:
    """Whether a sandbox's READ tags name exactly this owner (all three).
    Unreadable tags (``None``) are NOT in scope — they are evidence the sweep
    does not have, and the caller reports them separately."""
    if tags is None:
        return False
    return all(str(tags.get(k)) == v for k, v in owner_filter(owner).items())


def append_event(path: Optional[str | Path], event: str, sandbox_id: str, owner: Optional[Owner],
                 **more: Any) -> Optional[Dict[str, Any]]:
    """Append one durable event to ``sandboxes.jsonl`` (fsync'd). ``path``
    None = no record configured (a hand run): the event is returned, not
    written. An unknown event name refuses."""
    if event not in EVENTS:
        raise ValueError(f"unknown sandbox event {event!r}; events: {list(EVENTS)}")
    rec: Dict[str, Any] = {"ts": time.time(), "event": event, "sandbox_id": str(sandbox_id),
                           "owner": owner.as_dict() if owner is not None else None, **more}
    if path is None:
        return rec
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


def read_events(path: str | Path) -> List[Dict[str, Any]]:
    """Every parseable line of a ``sandboxes.jsonl`` (a torn trailing line is
    skipped, never a crash)."""
    p = Path(path)
    if not p.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def read_tags(sb: Any) -> Optional[Dict[str, str]]:
    """The sandbox's tags, or ``None`` when they CANNOT be read (a failed RPC,
    a sandbox that vanished between the list and the read). ``None`` is
    UNAVAILABLE evidence, distinct from "not ours"."""
    try:
        return {str(k): str(v) for k, v in (sb.get_tags() or {}).items()}
    except Exception:                              # noqa: BLE001 — unreadable, not absent
        return None


def sweep_owned(*, list_sandboxes: Callable[[], Iterable[Any]], owner: Owner, by: str,
                terminate: Optional[Callable[[Any], None]] = None,
                get_tags: Callable[[Any], Optional[Mapping[str, str]]] = read_tags,
                now: Callable[[], float] = time.time,
                sleep: Callable[[float], None] = time.sleep, max_rounds: int = 6,
                settle_s: float = 5.0,
                log: Callable[[str], None] = lambda s: print(s, flush=True),
                record_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Terminate ONLY the live sandboxes IN SCOPE for ``owner`` and VERIFY.

    ``list_sandboxes`` is the caller's listing — in production
    ``Sandbox.list(app_id=..., tags=owner_filter(owner))``, which the server
    already filters, so an out-of-scope sandbox reaching this loop means the
    listing was wider (a test's mixed-owner fake, a hand-run without the tag
    filter): it is LEFT, never terminated. Unreadable tags are LEFT too and
    make the round unverified. Each in-scope termination that a later round
    confirms gone is recorded as ``terminated`` / ``relist`` in
    ``record_path`` (when given). Returns the certificate
    (:data:`SWEEP_RECORD_FIELDS`); ``verified_empty`` False after
    ``max_rounds`` reports ``still_live`` for the caller to escalate."""
    term = terminate if terminate is not None else (lambda sb: sb.terminate())
    terminated: List[str] = []
    left: List[Dict[str, Any]] = []
    could_not: Dict[str, Dict[str, Any]] = {}
    targets: set = set()
    confirmed: set = set()
    still_live: List[str] = []
    unreadable: List[str] = []
    verified_empty = False
    rounds = 0
    n_first = None
    for rounds in range(1, int(max_rounds) + 1):  # noqa: B007 — the count is reported after the loop
        live = list(list_sandboxes())
        if n_first is None:
            n_first = len(live)
        remaining = 0
        left = []
        for sb in live:
            sid = str(getattr(sb, "object_id", sb))
            tags = get_tags(sb)
            if tags is None:
                left.append({"sandbox_id": sid, "why": "tags unreadable — evidence this sweep does not have"})
                continue
            if not in_scope(tags, owner):
                left.append({"sandbox_id": sid,
                             "why": "not this owner's (a wider listing than the owner filter)",
                             "tags": {k: tags.get(k) for k in OWNER_TAG_NAMES}})
                continue
            targets.add(sid)
            try:
                term(sb)
                if sid not in terminated:
                    terminated.append(sid)
                could_not.pop(sid, None)
            except Exception as e:                # noqa: BLE001 — reported by id, retried next round
                prior = could_not.get(sid) or {}
                could_not[sid] = {"sandbox_id": sid, "error": f"{type(e).__name__}: {e}",
                                  "attempts": int(prior.get("attempts") or 0) + 1}
                remaining += 1
        # verify BY OBJECT ID against a FRESH listing
        still, unread = set(), set()
        for sb in list_sandboxes():
            sid = str(getattr(sb, "object_id", sb))
            tags = get_tags(sb)
            if tags is None:
                unread.add(sid)
                still.add(sid)
            elif sid in targets or in_scope(tags, owner):
                still.add(sid)
        for sid in targets - still - confirmed:
            confirmed.add(sid)
            append_event(record_path, EVENT_TERMINATED, sid, owner, verified_by="relist", by=by)
        still_live, unreadable = sorted(still), sorted(unread)
        if remaining == 0 and not still_live:
            verified_empty = True
            break
        sleep(settle_s)
    for sid in sorted(set(terminated) - confirmed):
        append_event(record_path, EVENT_TERMINATE_UNVERIFIED, sid, owner, by=by)
    rec = {"swept_at": now(), "by": by, "owner": owner.as_dict(),
           "ids_terminated": sorted(set(terminated)),
           "left": left, "could_not_terminate": list(could_not.values()), "still_live": still_live,
           "tags_unreadable": unreadable,
           "verified_empty": verified_empty, "rounds": rounds,
           "n_listed_first_round": n_first}
    assert set(rec) == set(SWEEP_RECORD_FIELDS)
    log(f"[sandbox-sweep] by {by} for {owner.cell}@{owner.function_call_id[:12]}: terminated "
        f"{len(rec['ids_terminated'])} owned, left {len(left)}, could not terminate {len(could_not)}, "
        f"still live {len(still_live)} (tags unreadable {len(unreadable)}), verified_empty={verified_empty} "
        f"after {rounds} round(s)")
    return rec


def failed_sweep_record(owner: Owner, *, by: str, error: BaseException,
                        now: Callable[[], float] = time.time) -> Dict[str, Any]:
    """The certificate of a sweep that could not even LIST (no Modal client,
    an RPC failure): every :data:`SWEEP_RECORD_FIELDS` key at its empty value,
    ``verified_empty`` False, plus ``error`` naming the exception — so the
    container's ``finally`` and the external watcher record the same shape and
    neither can pass a failure off as a clean sweep."""
    rec = {"swept_at": now(), "by": by, "owner": owner.as_dict(), "ids_terminated": [], "left": [],
           "could_not_terminate": [], "still_live": [], "tags_unreadable": [], "verified_empty": False,
           "rounds": 0, "n_listed_first_round": None}
    assert set(rec) == set(SWEEP_RECORD_FIELDS)
    return {**rec, "error": f"{type(error).__name__}: {str(error)[:300]}"}
