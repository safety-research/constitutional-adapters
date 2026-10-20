#!/usr/bin/env python
"""seedrep_ladder.py — the streaming at-cap LADDER's control-plane protocol for the seedrep native battery.

This module is PURE PYTHON: it imports ``modal`` lazily and only inside the two adapters that wrap a live
``modal.Dict`` / ``modal.Volume`` (:class:`ModalDictKV`, :class:`RemoteVolume`). Everything else — the key
schema, the records, the producer protocol, the projections, the reconciler, claims / results / assembly —
runs against a tiny interface with an in-memory double, so the whole protocol is exercised by CPU tests
(``tests/test_seedrep_ladder_protocol.py``, ``tests/test_seedrep_ladder_attempts.py``,
``tests/test_seedrep_ladder_control_keys.py``) without a Modal client, a deployment or a GPU.

THE ONE ATOMIC PRIMITIVE. Modal's control plane offers exactly one atomic operation:
``modal.Dict.put(key, value, skip_if_exists=True) -> bool`` — create-if-absent, ``True`` when the pair was added,
``False`` when the key already existed (modal 1.5.4, ``modal/dict.py`` ``_Dict.put``: "Returns True if the
key-value pair was added and False if it wasn't because the key already existed and `skip_if_exists` was set";
the request rides ``DictUpdateRequest(if_not_exists=skip_if_exists)`` and returns ``resp.created``). There is no
compare-and-swap; ``pop`` is a separate call (``KeyError`` when absent unless a default is given); ``get``
returns its default when absent; ``keys()`` / ``items()`` are UNORDERED iterators. A Volume file is a RECORD,
never a lock: containers write into separate mount snapshots and ``commit()`` is last-writer-wins per file, so
two containers can both "succeed" at a local ``O_EXCL`` create of one path. Every election below is therefore
ONE ``skip_if_exists`` put — an acceptance, a claim, an elected result — and nothing relies on Volume ``O_EXCL``.
:class:`MemoryKV` reproduces those semantics exactly and the protocol tests run the same cases against it.

NAMESPACING (the control-plane key schema). ``wave_uuid`` is a uuid4 MINTED AT LAUNCH
and persisted in ``RS/wave.json`` — never derived from plan content, so an identical relaunch is a NEW wave whose
queues and Dicts are unreachable from a dead one. Every Dict is named ``seedrep-<W7>-<wave_uuid8>-<purpose>``
(purposes: :data:`PURPOSES`) and every key inside carries the full six-axis tuple
``(W7, wave_uuid, leg, view|arditi, stage_or_cap, shard)`` (:class:`SixAxisKey`, :func:`key_str`); item-scoped
keys (claims, results) append ``<arm>__s<s>`` and the ``item_key`` after the six axes. :func:`parse_key` refuses
anything shorter.

THE LAYOUT on the volume (NOTHING under ``ladder/`` is ever rewritten):

    seedrep/<W7>/exp/seedrep_battery_modal/<leg>/<view>/<shard>/attempts/<attempt_id>/   (an ATTEMPT ROOT, §4.6)
        ATTEMPT.json, COMPLETE.json (written LAST by the store), queue_inventory.json (§7.2 (ii))
    seedrep/<W7>/<wave_uuid>/ladder/<leg>/<view>/<cap_next>/                              (a QUEUE / LADDER ROOT)
        queue/<arm>__s<s>/<item_key>/<producer_attempt_id>.json    attempt-unique queue records (rev 4, P1-2)
        producers/<shard>__<attempt_id>.CLOSED                    the producer's marker (own or synthesised)
        CLOSED.json                                               the queue's closure (the reconciler's one write)
        materialised/<arm>__s<s>/<item_key>/<attempt_id>.json     attempt-unique draws; the WINNER is Dict-elected
        duplicates/<attempt_id>.jsonl                             a loser's own record (per attempt, never shared)

§7.1 also spells the materialised path ``materialised/<arm>__s<s>/items/<item_key>.json`` ("write-once rung
rows"); §7.3 (rev 3, Sol P1-5) supersedes that with the attempt-unique path above plus the Dict election, and
that is what this module implements.

PRODUCERS AND CLOSURE. The producer set is DECLARED (``launch_plan.json``), never discovered: for
``cap_next = 8192`` the base leg's shards, for ``16384`` the ``8192`` rung's containers (controls: base -> 2048 ->
8192 -> 16384, every transition its own queue). A producer is satisfied by its ACCEPTED attempt and no other.
Container order: (i) store COMPLETE -> ``COMPLETE.json``; (ii) ``queue_inventory.json`` (its own attempt-unique
files, each with its sha) — after which the producer is SEALED: a late item is a producer bug by construction and
:meth:`Producer.enqueue` refuses it; (iii) acceptance = ONE ``accepted.put(key, attempt_id, skip_if_exists=True)``
(``False`` -> superseded: publishes nothing further); (iv) ``producers/<shard>__<attempt_id>.CLOSED``. The
reconciler (:func:`reconcile_queue`, ONE grading host process) treats an accepted attempt whose ``COMPLETE.json`` and
``queue_inventory.json`` are present and whose inventory files all exist with matching shas as CLOSED and
synthesises the idempotent marker (``synthesised_by: reconciler``) when the container died between (iii) and (iv);
a superseded attempt's markers and files are ignored (no two attempts share a path, so a superseded attempt
publishing AFTER acceptance cannot disturb the accepted inventory). ``CLOSED.json`` is written when EVERY declared
producer has a marker from its accepted attempt; a producer with no accepted attempt keeps the queue open
(``PRODUCER_MISSING``). The 16,384 queue's producers are the 8,192 rung's attempts, so closure PROPAGATES rung by
rung. 16,384 is the ceiling: :func:`next_cap` returns ``None`` there and nothing is ever queued above it.

CLAIMS, RESULTS, ASSEMBLY. ``claims.put((queue key, arm_s, item_key), {attempt_id, call_id, ts},
skip_if_exists=True)``: ``False`` = held -> skip. NO automatic stale takeover (read / pop / put is three
operations; two reclaimers can both win): a claim is cleared ONLY by the single reconciler, and only after the
claimant's Modal call is TERMINAL (``lease_state(call_id) in TERMINAL_STATES`` — the ``registry_volume._lease_state``
rule: a call state, never a heartbeat age). Every draw is written to its attempt-unique materialised path, then ONE
``results.put(..., {attempt_id, path, text_sha256, request_id, completion_id}, skip_if_exists=True)`` elects the
result; a loser's file stays as its per-attempt duplicate record plus one line in its OWN ``duplicates/`` file.
The assembly (:func:`load_elected`, :func:`verify_elected`) reads the winner from the Dict (or a snapshot of it,
:func:`snapshot_kv`), verifies the file's ``text_sha256`` and the row's ``(sampling_seed, cap, preset,
base_row_sha256, producer attempt)`` against the ACCEPTED base row, and refuses anything else.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import hashlib
import gzip
import json
import os
import re
import sys
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

__all__ = [
    "SCHEMA_QUEUE", "SCHEMA_INVENTORY", "SCHEMA_MARKER", "SCHEMA_CLOSED", "SCHEMA_MATERIALISED", "SCHEMA_SNAPSHOT",
    "STAGE_BASE", "STAGE_PETRI_RERUN", "STAGES", "PURPOSES", "CEILING", "LADDERS", "TERMINAL_STATES", "QueueFiles",
    "SixAxisKey", "dict_name", "key_str", "item_scoped_key", "parse_key", "item_key", "arm_s",
    "KV", "MemoryKV", "ModalDictKV", "snapshot_kv", "load_kv_snapshot", "SnapshotKV",
    "VolumeFS", "LocalVolume", "SharedVolume", "SnapshotVolume", "RemoteVolume",
    "attempt_root_rel", "ladder_root_rel", "queue_entry_rel", "producer_marker_rel", "closed_rel",
    "materialised_rel", "duplicates_rel", "inventory_rel", "complete_rel",
    "ProducerRef", "QueueSpec", "predecessor_cap", "next_cap", "producer_stage",
    "canonical_json", "sha256_hex", "row_sha256", "item_keys_sha256", "slot", "base_row_identity", "base_row_identity_sha256",
    "same_world", "record_text", "modal_lease_state", "claim_is_this_calls", "sampler_keys", "check_redraw_sampling",
    "RESULTS_ROOT_NAME", "results_root_rel", "shard_dir", "attempt_root_for", "producer_attempt_root", "resolve_producer_root",
    "SCHEMA_ADOPTED", "ADOPTED_DIR", "ADOPTED_MANIFEST", "ADOPTED_INDEX", "attempt_root_marker", "read_source_evidence",
    "evidence_for_items", "adopt_evidence",
    "make_queue_record", "redraw_messages", "static_queue_record", "controls_queue_record", "misalign_queue_record",
    "gsm8k_queue_record",
    "Producer", "ProducerSealed", "accept_attempt", "accepted_attempt", "write_closed_marker", "finish_producer",
    "accepted_producer_attempts", "accepted_queue_entries", "live_queue_entries", "read_closed",
    "reconcile_queue", "PRODUCER_MISSING", "PRODUCER_CLOSED", "PRODUCER_CLOSED_SYNTHESISED", "PRODUCER_UNVERIFIED",
    "claim_item", "elect_result", "write_materialised", "promote_materialised", "record_duplicate", "speculative_rel",
    "Elected", "load_elected", "load_elected_one", "verify_elected", "ElectedMismatch", "LadderRefusal",
    "accepted_attempt_root_or_refuse",
]

# --------------------------------------------------------------------------- #
# constants                                                                     #
# --------------------------------------------------------------------------- #

SCHEMA_QUEUE = "seedrep_queue/1"
SCHEMA_INVENTORY = "seedrep_queue_inventory/1"
SCHEMA_MARKER = "seedrep_producer_closed/1"
SCHEMA_CLOSED = "seedrep_queue_closed/1"
SCHEMA_MATERIALISED = "seedrep_materialised/1"
SCHEMA_SNAPSHOT = "seedrep_kv_snapshot/1"

STAGE_BASE = "base"
STAGE_PETRI_RERUN = "petri-rerun"
#: the ``stage_or_cap`` axis vocabulary: the base leg, the three rung caps, the Petri re-run proxy
STAGES = (STAGE_BASE, "2048", "8192", "16384", STAGE_PETRI_RERUN)
#: the six Dict purposes
PURPOSES = ("accepted", "claims", "results", "heartbeats", "sentinels", "stop")
#: the ceiling of every ladder (LADDER_RULES.md) — recorded, never queued
CEILING = 16384
#: the per-leg-kind ladders. static / controls are the drivers' own (``ladder_decide.LADDERS``, pinned equal by
#: ``tests/test_seedrep_ladder_protocol.py``); misalign 3,000 -> 16,384 (``runner.DEFAULT_MAX_TOKENS["agentic"]``
#: -> the ceiling); GSM8K 8,192 -> 16,384 (``contracts.G8K_CAP`` -> ``G8K_RUNG_CAP``)
LADDERS: dict[str, tuple[int, ...]] = {
    "static": (4096, 8192, 16384),
    "controls": (512, 2048, 8192, 16384),
    "misalign": (3000, 16384),
    "gsm8k": (8192, 16384),
}
#: ``lease_state`` verdicts that AUTHORISE a reclaim (``registry_volume._lease_state``: finished | failed |
#: cancelled are terminal; running | unknown are not — uncertainty fails CLOSED)
TERMINAL_STATES = frozenset({"finished", "failed", "cancelled"})

#: fx1: the env var naming an interpreter that CARRIES the Modal SDK -- the served app's own ``sys.executable``, handed to the
#: rung consumer child (``seedrep_served.RUNG_SDK_PYTHON_ENV`` spells the same literal; pinned by test). ``rung_redraw.SdkBridge`` reads
#: it; :func:`modal_sdk`'s refusal names it.
SDK_PYTHON_ENV = "SEEDREP_MODAL_SDK_PYTHON"


def modal_sdk(*need: str):
    """The Modal SDK, CHECKED for the surface the caller needs (``need``: attribute names; default ``Dict`` + ``Volume``, the consumer's).
    Under an interpreter WITHOUT the ``modal`` distribution but with the repo's ``scripts/`` on ``sys.path`` (the rung container's DRIVER
    venv, ``PYTHONPATH=<tree>/src:<tree>/scripts``) a bare ``import modal`` does not fail: it answers with the ``scripts/modal/`` NAMESPACE
    package -- no ``__init__``, no ``Dict``, no ``Volume``, no ``FunctionCall`` -- and the first attribute access dies ``AttributeError:
    module 'modal' has no attribute 'Volume'`` (all 11 seed-1 rungs). Every SDK access in this module and in the consumer
    goes through here, so that shape is refused BY NAME (:class:`LadderRefusal`), saying where the import landed, what was missing, and how
    a consumer is meant to reach a real SDK (``$SEEDREP_MODAL_SDK_PYTHON`` -> ``rung_redraw.SdkBridge``). Each caller names ITS surface
    (the lease rule: ``FunctionCall`` + ``exception``), so a test double carrying exactly that surface passes, as before."""
    import modal  # the ONE bare import of the protocol module (tests/test_seedrep_fx1_rung_sdk_bridge.py pins it)
    need = need or ("Dict", "Volume")
    missing = [a for a in need if not hasattr(modal, a)]
    if not missing:
        return modal
    where = getattr(modal, "__file__", None) or ", ".join(str(p) for p in (getattr(modal, "__path__", None) or [])) or repr(modal)
    raise LadderRefusal(f"`import modal` under {sys.executable} resolved to {where}, which is not the Modal SDK (missing {missing}; a namespace "
                        f"package: the repo's scripts/modal shadows the name when this interpreter carries no `modal` distribution). A rung "
                        f"consumer reaches the SDK through the interpreter named in ${SDK_PYTHON_ENV} (the served app's own python, via "
                        f"rung_redraw.SdkBridge); a grading host tool runs under a venv that installs it.")


def modal_lease_state(call_id: str) -> str:
    """``registry_volume._lease_state``'s rule, verbatim, over a live Modal client: ``finished`` | ``failed`` |
    ``cancelled`` (terminal) | ``running`` | ``unknown`` — never a guess from a heartbeat age, and an auth /
    connection / not-found error is ``unknown`` (fails CLOSED: no reclaim), never ``failed``."""
    modal = modal_sdk("FunctionCall", "exception")  # lazy: the protocol runs without a Modal client everywhere but here

    try:
        fc = modal.FunctionCall.from_id(call_id)
    except Exception:  # noqa: BLE001
        return "unknown"
    try:
        fc.get(timeout=0)
        return "finished"
    except TimeoutError:
        return "running"
    except modal.exception.NotFoundError:
        return "unknown"
    except (modal.exception.AuthError, modal.exception.ConnectionError):
        return "unknown"
    except Exception:  # noqa: BLE001
        return "failed"   # the call raised: terminal


PRODUCER_MISSING = "PRODUCER_MISSING"
PRODUCER_CLOSED = "CLOSED"
PRODUCER_CLOSED_SYNTHESISED = "CLOSED_SYNTHESISED"
PRODUCER_UNVERIFIED = "ACCEPTED_UNVERIFIED"

_HEX7 = re.compile(r"^[0-9a-f]{7}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_AXIS = re.compile(r"^[A-Za-z0-9._:-]+$")
_SLUG = re.compile(r"^[A-Za-z0-9._@-]+$")
_SEP = "|"


class LadderRefusal(RuntimeError):
    """The protocol refuses: a malformed key, a torn record, a producer bug, a forged file. Never degraded."""


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# the six-axis key schema                                                     #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class SixAxisKey:
    """``(wave_uuid, leg, view, stage, shard)`` — every control-plane key carries all six.

    ``W7`` is the wave sha's first seven hex chars; ``wave_uuid`` the launch-minted uuid4 (``RS/wave.json``);
    ``leg`` the leg id; ``view`` ``s1`` / ``s2`` / ``s3`` / ``arditi``; ``stage`` one of :data:`STAGES`; ``shard``
    the shard label (``"0"``, ``"1"``, … or the literal ``"queue"`` for a queue-scoped key). Validated at
    construction; a malformed axis refuses."""

    W7: str
    wave_uuid: str
    leg: str
    view: str
    stage: str
    shard: str

    def __post_init__(self) -> None:
        if not isinstance(self.W7, str) or not _HEX7.match(self.W7):
            raise LadderRefusal(f"W7 {self.W7!r} is not seven lowercase hex chars")
        if not isinstance(self.wave_uuid, str) or not _UUID.match(self.wave_uuid):
            raise LadderRefusal(f"wave_uuid {self.wave_uuid!r} is not a uuid4 string (RS/wave.json)")
        for name in ("leg", "view", "shard"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v or not _AXIS.match(v) or _SEP in v:
                raise LadderRefusal(f"{name} {v!r} is not a non-empty path-safe axis")
        if self.stage not in STAGES:
            raise LadderRefusal(f"stage {self.stage!r} is not one of {STAGES}")

    @property
    def wave_uuid8(self) -> str:
        return self.wave_uuid.replace("-", "")[:8]

    def with_stage(self, stage: str, shard: Optional[str] = None) -> "SixAxisKey":
        return dataclasses.replace(self, stage=str(stage), shard=self.shard if shard is None else str(shard))


def dict_name(W7: str, wave_uuid: str, purpose: str) -> str:
    """``seedrep-<window>-<wave_uuid8>-<purpose>``; the purpose must be one of :data:`PURPOSES`."""
    if purpose not in PURPOSES:
        raise LadderRefusal(f"Dict purpose {purpose!r} is not one of {PURPOSES}")
    k = SixAxisKey(W7, wave_uuid, "leg", "view", STAGE_BASE, "0")   # validates W7 + wave_uuid
    return f"seedrep-{k.W7}-{k.wave_uuid8}-{purpose}"


def key_str(key: SixAxisKey, *extra: str) -> str:
    """The canonical string form ``W7|wave_uuid|leg|view|stage|shard[|extra…]``: JSON-safe, readable in a Dict
    snapshot, parsed back by :func:`parse_key`. Extra components (``arm__s<s>``, ``item_key``) may not contain
    the separator."""
    parts = [key.W7, key.wave_uuid, key.leg, key.view, key.stage, key.shard]
    for e in extra:
        if not isinstance(e, str) or not e or _SEP in e:
            raise LadderRefusal(f"key component {e!r} is empty or carries the separator {_SEP!r}")
        parts.append(e)
    return _SEP.join(parts)


def item_scoped_key(queue_key: SixAxisKey, arm_s_: str, item_key_: str) -> str:
    """A claim / result key: the queue's six axes + ``<arm>__s<s>`` + ``item_key``."""
    return key_str(queue_key, arm_s_, item_key_)


def parse_key(s: str) -> tuple[SixAxisKey, tuple[str, ...]]:
    """The inverse of :func:`key_str`: the six axes (validated) and the extra components. Anything with fewer
    than six components refuses — a key without every axis is a key a dead wave or another stage could collide
    with."""
    if not isinstance(s, str):
        raise LadderRefusal(f"key {s!r} is not a string")
    parts = s.split(_SEP)
    if len(parts) < 6:
        raise LadderRefusal(f"key {s!r} carries {len(parts)} component(s); the schema is (W7, wave_uuid, leg, view, "
                            f"stage, shard)[, extra…]")
    return SixAxisKey(*parts[:6]), tuple(parts[6:])


def item_key(kind: str, *parts: Any) -> str:
    """The ONE derivation of an item's queue key (a path component and a Dict key component): ``<kind>__<p1>__<p2>…``
    when every part is a plain slug, else ``<kind>__h<sha256(parts)[:24]>`` — deterministic either way, never
    containing ``/`` or the key separator. The record content carries the identity in full regardless."""
    if kind not in LADDERS:
        raise LadderRefusal(f"item kind {kind!r} is not one of {sorted(LADDERS)}")
    strs = [str(p) for p in parts]
    if not strs:
        raise LadderRefusal("item_key needs at least one identity part")
    if all(_SLUG.match(p) for p in strs):
        return "__".join([kind, *strs])
    digest = hashlib.sha256(canonical_json(strs).encode("utf-8")).hexdigest()[:24]
    return f"{kind}__h{digest}"


def arm_s(arm: str, s: int) -> str:
    """``<arm>__s<s>`` — the per-(arm, seed) slot of a queue / materialised dir."""
    if not isinstance(arm, str) or not arm or not _SLUG.match(arm):
        raise LadderRefusal(f"arm {arm!r} is not a plain cell-id slug")
    if isinstance(s, bool) or not isinstance(s, int) or s < 1:
        raise LadderRefusal(f"seed s={s!r} must be a positive int (the literal training seed)")
    return f"{arm}__s{int(s)}"


# --------------------------------------------------------------------------- #
# the KV interface: modal.Dict's four operations and their in-memory double      #
# --------------------------------------------------------------------------- #

_NO_DEFAULT = object()


class KV(Protocol):
    """The subset of ``modal.Dict`` the protocol uses — and ALL it uses."""

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def pop(self, key: str, default: Any = _NO_DEFAULT) -> Any: ...
    def contains(self, key: str) -> bool: ...
    def items(self) -> Iterator[tuple[str, Any]]: ...


class MemoryKV:
    """The in-memory double of ``modal.Dict`` with the SAME semantics as modal 1.5.4 ``modal/dict.py``:

    * ``put(key, value, skip_if_exists=True)`` -> ``False`` and NO write when the key exists, else write + ``True``;
      without the flag it overwrites and returns ``True`` ("added");
    * ``get(key, default=None)`` -> the default when absent;
    * ``pop(key)`` -> ``KeyError`` when absent unless a default is given;
    * ``contains(key)``; ``items()`` UNORDERED over a snapshot of the contents (a live Dict streams them);
    * values are serialised on the wire, so a stored value is a deep copy and a returned value is a deep copy.

    Keys are restricted to strings here (the protocol builds strings; a snapshot must be JSON). Thread-safe."""

    def __init__(self, name: str = "memory") -> None:
        self.name = name
        self._d: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.ops: list[tuple] = []

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        if not isinstance(key, str):
            raise LadderRefusal(f"{self.name}: key {key!r} is not a string")
        with self._lock:
            if skip_if_exists and key in self._d:
                self.ops.append(("put", key, False))
                return False
            self._d[key] = copy.deepcopy(value)
            self.ops.append(("put", key, True))
            return True

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._d:
                return default
            return copy.deepcopy(self._d[key])

    def pop(self, key: str, default: Any = _NO_DEFAULT) -> Any:
        with self._lock:
            if key not in self._d:
                if default is not _NO_DEFAULT:
                    return default
                raise KeyError(f"{key} not in dict {self.name}")
            self.ops.append(("pop", key, True))
            return self._d.pop(key)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._d

    def items(self) -> Iterator[tuple[str, Any]]:
        with self._lock:
            snap = [(k, copy.deepcopy(v)) for k, v in self._d.items()]
        # unordered by contract: a consumer must not rely on insertion order
        snap.sort(key=lambda kv: hashlib.sha256(kv[0].encode()).hexdigest())
        return iter(snap)

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)


class ModalDictKV:
    """A :class:`KV` over a live ``modal.Dict`` (``modal.Dict.from_name(name, create_if_missing=True)``). Only
    the four operations; constructed lazily so this module imports without ``modal``."""

    def __init__(self, dict_obj: Any, name: str) -> None:
        self._d = dict_obj
        self.name = name

    @classmethod
    def from_name(cls, name: str, *, create_if_missing: bool = True) -> "ModalDictKV":
        modal = modal_sdk("Dict")  # lazy: the protocol tests run without a Modal client; a shadowing scripts/modal refuses by name (fx1)
        return cls(modal.Dict.from_name(name, create_if_missing=create_if_missing), name)

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        return bool(self._d.put(key, value, skip_if_exists=skip_if_exists))

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def pop(self, key: str, default: Any = _NO_DEFAULT) -> Any:
        if default is _NO_DEFAULT:
            return self._d.pop(key)
        return self._d.pop(key, default)

    def contains(self, key: str) -> bool:
        return bool(self._d.contains(key))

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(list(self._d.items()))


def snapshot_kv(kv: KV, path: Optional[Path] = None, *, name: str = "") -> dict:
    """A JSON snapshot of a KV (``{schema, name, taken_utc, entries: {key: value}}``) — what the assembly's
    ``--results`` flag reads. Written atomically when ``path`` is given."""
    entries = {k: v for k, v in kv.items()}
    doc = {"schema": SCHEMA_SNAPSHOT, "name": name or getattr(kv, "name", ""), "taken_utc": _utc(),
           "n": len(entries), "entries": entries}
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
        os.replace(tmp, path)
    return doc


class SnapshotKV:
    """A read-only :class:`KV` view over a :func:`snapshot_kv` document (the assembly reads the elected results
    from a snapshot taken on the grading host; ``put`` / ``pop`` refuse — a snapshot is evidence, not the authority)."""

    def __init__(self, doc: Mapping[str, Any]) -> None:
        if doc.get("schema") != SCHEMA_SNAPSHOT or not isinstance(doc.get("entries"), Mapping):
            raise LadderRefusal(f"not a {SCHEMA_SNAPSHOT} document")
        self.name = str(doc.get("name", ""))
        self._d = dict(doc["entries"])

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        raise LadderRefusal("a KV snapshot is read-only")

    def pop(self, key: str, default: Any = _NO_DEFAULT) -> Any:
        raise LadderRefusal("a KV snapshot is read-only")

    def get(self, key: str, default: Any = None) -> Any:
        return copy.deepcopy(self._d.get(key, default))

    def contains(self, key: str) -> bool:
        return key in self._d

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(sorted(self._d.items()))

    def __len__(self) -> int:
        return len(self._d)


def load_kv_snapshot(path: Path) -> SnapshotKV:
    return SnapshotKV(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# the Volume interface: a mount, and the two-snapshot fake                       #
# --------------------------------------------------------------------------- #


class VolumeFS(Protocol):
    """Relative-path file operations over the lane's volume root plus the Volume's two verbs."""

    def read_bytes(self, rel: str) -> bytes: ...
    def write_bytes(self, rel: str, data: bytes, *, exclusive: bool = False) -> None: ...
    def exists(self, rel: str) -> bool: ...
    def listdir(self, rel: str) -> list[str]: ...
    def walk(self, rel: str) -> list[str]: ...
    def read_many(self, rels: Sequence[str]) -> dict[str, bytes]: ...
    def write_many(self, items: Mapping[str, bytes]) -> None: ...
    def commit(self) -> None: ...
    def reload(self) -> None: ...


def _bulk_common_dir(rels: Sequence[str]) -> str:
    """The deepest directory every path in ``rels`` lies under -- the ONE directory a bulk pull fetches (fj32). Refuses an empty set and a
    set with no common directory below the volume root (a bulk read of the whole volume is never what a caller means)."""
    rels = [_norm(r) for r in rels]
    if not rels:
        raise LadderRefusal("read_many: no paths")
    common = os.path.commonpath([str(PurePosixPath(r).parent) for r in rels])
    if common in ("", "."):
        raise LadderRefusal(f"read_many: the {len(rels)} paths share no directory below the volume root (first: {rels[:2]})")
    return common


def _norm(rel: str) -> str:
    p = PurePosixPath(str(rel))
    if p.is_absolute() or ".." in p.parts:
        raise LadderRefusal(f"volume path {rel!r} must be relative and may not climb")
    return str(p)


class LocalVolume:
    """A :class:`VolumeFS` over a directory — the mounted volume inside a container (``/vol``), or a local
    checkout of it (``modal volume get``) on the grading host. ``commit`` / ``reload`` call the callables given
    (``vol.commit`` / ``vol.reload`` in a container; no-ops on a plain directory)."""

    def __init__(self, root: Path, *, commit: Optional[Callable[[], Any]] = None,
                 reload: Optional[Callable[[], Any]] = None) -> None:
        self.root = Path(root)
        self._commit = commit
        self._reload = reload

    def _p(self, rel: str) -> Path:
        return self.root / _norm(rel)

    def read_bytes(self, rel: str) -> bytes:
        return self._p(rel).read_bytes()

    def write_bytes(self, rel: str, data: bytes, *, exclusive: bool = False) -> None:
        p = self._p(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        if exclusive:
            with open(p, "xb") as f:      # a LOCAL O_EXCL: never an election (module docstring)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            return
        tmp = p.with_name(p.name + f".tmp-{uuid.uuid4().hex[:8]}")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def exists(self, rel: str) -> bool:
        return self._p(rel).exists()

    def listdir(self, rel: str) -> list[str]:
        p = self._p(rel)
        return sorted(q.name for q in p.iterdir()) if p.is_dir() else []

    def walk(self, rel: str) -> list[str]:
        p = self._p(rel)
        if not p.is_dir():
            return []
        return sorted(str(q.relative_to(self.root).as_posix()) for q in p.rglob("*") if q.is_file())

    def read_many(self, rels: Sequence[str]) -> dict[str, bytes]:
        return {r: self.read_bytes(r) for r in rels}

    def write_many(self, items: Mapping[str, bytes]) -> None:
        for rel, data in items.items():
            self.write_bytes(rel, data)

    def commit(self) -> None:
        if self._commit is not None:
            self._commit()

    def reload(self) -> None:
        if self._reload is not None:
            self._reload()


class SharedVolume:
    """The COMMITTED state of the fake volume (what every container sees after ``reload``). Per-file
    last-writer-wins on commit; a commit log for the tests."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commit_log: list[tuple[str, tuple[str, ...]]] = []
        self._lock = threading.Lock()

    def snapshot(self, name: str) -> "SnapshotVolume":
        return SnapshotVolume(self, name)

    def read_only_view(self) -> "SnapshotVolume":
        """The grading host reconciler's view: a snapshot that reloads before every read (Volume API semantics)."""
        return SnapshotVolume(self, "reconciler", autoreload=True)


class SnapshotVolume:
    """ONE container's mount snapshot over a :class:`SharedVolume`: writes land in the local snapshot only;
    ``commit()`` publishes the dirty files (last-writer-wins) and, as Modal does on a successful commit, reloads;
    ``reload()`` pulls the committed state (a dirty local file keeps its uncommitted bytes until its own commit).
    ``exclusive=True`` models ``O_EXCL`` against THIS snapshot only — two containers both succeed on one path,
    which is exactly why nothing in the protocol relies on it. ``crash()`` drops every uncommitted write."""

    def __init__(self, shared: SharedVolume, name: str, *, autoreload: bool = False) -> None:
        self.shared = shared
        self.name = name
        self._autoreload = autoreload
        self.local: dict[str, bytes] = dict(shared.files)
        self.dirty: set[str] = set()
        self.crashed = False
        self._lk = threading.RLock()           # a mount is shared by the container's threads (width > 1 draws)

    def _live(self) -> None:
        if self.crashed:
            raise LadderRefusal(f"container {self.name} has crashed; it writes nothing more")
        if self._autoreload:
            self.reload()

    def read_bytes(self, rel: str) -> bytes:
        with self._lk:
            self._live()
            rel = _norm(rel)
            if rel not in self.local:
                raise FileNotFoundError(rel)
            return self.local[rel]

    def write_bytes(self, rel: str, data: bytes, *, exclusive: bool = False) -> None:
        with self._lk:
            self._live()
            rel = _norm(rel)
            if exclusive and rel in self.local:
                raise FileExistsError(rel)
            self.local[rel] = bytes(data)
            self.dirty.add(rel)

    def exists(self, rel: str) -> bool:
        with self._lk:
            self._live()
            rel = _norm(rel)
            return rel in self.local or any(k.startswith(rel + "/") for k in self.local)

    def listdir(self, rel: str) -> list[str]:
        with self._lk:
            self._live()
            rel = _norm(rel)
            prefix = "" if rel in ("", ".") else rel + "/"
            names = set()
            for k in self.local:
                if k.startswith(prefix):
                    names.add(k[len(prefix):].split("/", 1)[0])
            return sorted(names)

    def walk(self, rel: str) -> list[str]:
        with self._lk:
            self._live()
            rel = _norm(rel)
            prefix = "" if rel in ("", ".") else rel + "/"
            return sorted(k for k in self.local if k.startswith(prefix))

    def read_many(self, rels: Sequence[str]) -> dict[str, bytes]:
        with self._lk:
            return {r: self.read_bytes(r) for r in rels}

    def write_many(self, items: Mapping[str, bytes]) -> None:
        with self._lk:
            for rel, data in items.items():
                self.write_bytes(rel, data)

    def commit(self) -> None:
        with self._lk:
            if self.crashed:
                raise LadderRefusal(f"container {self.name} has crashed; it commits nothing more")
            with self.shared._lock:
                for rel in sorted(self.dirty):
                    self.shared.files[rel] = self.local[rel]        # last writer wins, per file
                self.shared.commit_log.append((self.name, tuple(sorted(self.dirty))))
            self.dirty.clear()
            self.reload()

    def reload(self) -> None:
        with self._lk:
            with self.shared._lock:
                merged = dict(self.shared.files)
            for rel in self.dirty:
                merged[rel] = self.local[rel]
            self.local = merged

    def crash(self) -> None:
        """The container dies: uncommitted writes are lost, nothing more is written."""
        with self._lk:
            self.dirty.clear()
            self.crashed = True


def _absent_path_errors() -> tuple:
    """The ONE exception class that means "the volume has no such path": ``modal.Volume.listdir`` raises
    ``FileNotFoundError`` for it (the SDK maps its own ``NotFoundError`` onto it in ``volume.py``); the raw
    ``modal.exception.NotFoundError`` is admitted beside it for a client that does not. An auth, connection, service or
    API failure is NOT in this tuple — it propagates from :meth:`RemoteVolume._entries`, never reads as an empty queue:
    a reconciler that saw "no files" on an expired token would leave every queue open with the fault hidden."""
    errs: list = [FileNotFoundError]
    try:
        modal = modal_sdk("exception")  # lazy; absent outside the grading host / a container (a shadowing scripts/modal is refused, so: FileNotFoundError alone)
        nf = getattr(getattr(modal, "exception", None), "NotFoundError", None)
        if isinstance(nf, type) and issubclass(nf, BaseException):
            errs.append(nf)
    except Exception:  # noqa: BLE001 -- no client here: FileNotFoundError alone
        pass
    return tuple(errs)


REMOTE_SCRATCH_DIR = Path("/tmp/launch_premint/volume_get")     # where a bulk pull lands on the grading host (read-only on the volume)


def _modal_volume_get(name: str, remote_dir: str, local_dir: Path) -> None:
    """ONE ``modal volume get --force <name> <remote_dir> <local_dir>``: the CLI's recursive, concurrent download of a whole directory
    (``modal.cli._download._volume_download``: one recursive ``iterdir`` + concurrent reads), the same primitive `seedrep_served.volume_get`
    uses. Files land under ``<local_dir>/<basename(remote_dir)>/...`` (the CLI keeps the last path component)."""
    import subprocess
    import sys
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", name, remote_dir, str(local_dir)], check=True)


class RemoteVolume:
    """A :class:`VolumeFS` over a live ``modal.Volume`` from OUTSIDE a container (the grading host reconciler): reads
    through ``listdir`` / ``read_file``, writes through ``batch_upload``. Lazy ``modal``; the volume's root is
    ``/``, so ``rel`` paths are used as given. ``commit`` / ``reload`` are no-ops (the Volume API reads the
    committed state).

    fj32 -- the bulk verbs: :meth:`read_many` is ONE ``modal volume get`` of the paths' common directory into a fresh scratch directory under
    ``REMOTE_SCRATCH_DIR`` (needs the volume's ``name``, which :meth:`from_name` records; a handle constructed without one falls back to per-file
    ``read_file``, honestly slow), and :meth:`write_many` is ONE ``batch_upload(force=True)`` context holding every file -- one upload, one
    commit -- where :meth:`write_bytes` opens one context per file. ``bulk_get`` is the pull primitive (tests inject a copy)."""

    def __init__(self, volume: Any, *, prefix: str = "", name: Optional[str] = None, scratch: Optional[Path] = None,
                 bulk_get: Optional[Callable[[str, str, Path], None]] = None) -> None:
        self._v = volume
        self._prefix = prefix.strip("/")
        self.name = name
        self.scratch = Path(scratch) if scratch is not None else REMOTE_SCRATCH_DIR
        self._bulk_get = bulk_get or _modal_volume_get

    @classmethod
    def from_name(cls, name: str, *, prefix: str = "", scratch: Optional[Path] = None) -> "RemoteVolume":
        modal = modal_sdk("Volume")  # lazy; checked (fx1)
        return cls(modal.Volume.from_name(name), prefix=prefix, name=name, scratch=scratch)

    def _abs(self, rel: str) -> str:
        rel = _norm(rel)
        return "/" + "/".join(p for p in (self._prefix, rel) if p and p != ".")

    def read_bytes(self, rel: str) -> bytes:
        return b"".join(self._v.read_file(self._abs(rel).lstrip("/")))

    def write_bytes(self, rel: str, data: bytes, *, exclusive: bool = False) -> None:
        import io
        if exclusive and self.exists(rel):
            raise FileExistsError(rel)
        with self._v.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(bytes(data)), self._abs(rel))

    def read_many(self, rels: Sequence[str]) -> dict[str, bytes]:
        rels = [_norm(r) for r in rels]
        if not rels:
            return {}
        if self.name is None:                      # no name to hand the CLI: per-file reads (correct; the slow path)
            return {r: self.read_bytes(r) for r in rels}
        common = _bulk_common_dir(rels)
        local = self.scratch / f"get-{uuid.uuid4().hex[:12]}"
        local.mkdir(parents=True, exist_ok=True)
        self._bulk_get(self.name, self._abs(common), local)
        base = local / PurePosixPath(common).name
        out: dict[str, bytes] = {}
        missing = []
        for r in rels:
            q = base / PurePosixPath(r).relative_to(common)
            if not q.is_file():
                missing.append(r)
                continue
            out[r] = q.read_bytes()
        if missing:                                # the pull did not deliver a listed file: refuse (never an empty record), keep the scratch
            raise FileNotFoundError(f"bulk pull of {common!r} into {local} delivered {len(out)}/{len(rels)} files; missing {missing[:3]}")
        import shutil
        shutil.rmtree(local, ignore_errors=True)
        return out

    def write_many(self, items: Mapping[str, bytes]) -> None:
        import io
        if not items:
            return
        with self._v.batch_upload(force=True) as batch:    # ONE upload, ONE commit for every file
            for rel, data in items.items():
                batch.put_file(io.BytesIO(bytes(data)), self._abs(rel))

    def _entries(self, rel: str, *, recursive: bool) -> list:
        try:
            return list(self._v.listdir(self._abs(rel).lstrip("/"), recursive=recursive))
        except _absent_path_errors():
            return []          # an absent directory lists as empty; EVERY other failure propagates

    def exists(self, rel: str) -> bool:
        rel = _norm(rel)
        parent = str(PurePosixPath(rel).parent)
        name = PurePosixPath(rel).name
        return any(PurePosixPath(e.path).name == name for e in self._entries(parent, recursive=False))

    def listdir(self, rel: str) -> list[str]:
        return sorted(PurePosixPath(e.path).name for e in self._entries(rel, recursive=False))

    def walk(self, rel: str) -> list[str]:
        base = self._abs("").lstrip("/")
        out = []
        for e in self._entries(rel, recursive=True):
            if getattr(e, "type", None) is not None and str(getattr(e.type, "name", "")).upper() == "DIRECTORY":
                continue
            p = e.path.lstrip("/")
            if base and p.startswith(base + "/"):
                p = p[len(base) + 1:]
            out.append(p)
        return sorted(out)

    def commit(self) -> None:
        return None

    def reload(self) -> None:
        return None


VOLUME_OP_KINDS = ("listdir", "exists", "read", "read_bulk", "upload", "upload_bulk", "commit", "reload")


class CountingVolume:
    """A :class:`VolumeFS` that forwards to ``inner`` and tallies every call by kind (fj32): ``listdir`` = ``walk`` + ``listdir`` (a directory
    listing, recursive or not), ``exists`` (a point lookup: on a RemoteVolume one listdir of the parent), ``read`` = ``read_bytes``,
    ``read_bulk`` = ``read_many``, ``upload`` = ``write_bytes``, ``upload_bulk`` = ``write_many``, ``commit``, ``reload``; ``files_read`` /
    ``files_uploaded`` count the files those calls moved. The replay wraps its volume in one and reports ``ops`` -- the number of volume
    operations a run took, which must not grow with the queue. An ``inner`` without the bulk verbs is served by the per-file loop and counted
    honestly as N single reads / uploads."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.ops: dict[str, int] = {k: 0 for k in VOLUME_OP_KINDS}
        self.files_read = 0
        self.files_uploaded = 0

    def read_bytes(self, rel: str) -> bytes:
        self.ops["read"] += 1
        self.files_read += 1
        return self.inner.read_bytes(rel)

    def read_many(self, rels: Sequence[str]) -> dict[str, bytes]:
        rels = list(rels)
        f = getattr(self.inner, "read_many", None)
        if f is None:
            return {r: self.read_bytes(r) for r in rels}
        self.ops["read_bulk"] += 1
        self.files_read += len(rels)
        return f(rels)

    def write_bytes(self, rel: str, data: bytes, *, exclusive: bool = False) -> None:
        self.ops["upload"] += 1
        self.files_uploaded += 1
        self.inner.write_bytes(rel, data, exclusive=exclusive)

    def write_many(self, items: Mapping[str, bytes]) -> None:
        f = getattr(self.inner, "write_many", None)
        if f is None:
            for rel, data in items.items():
                self.write_bytes(rel, data)
            return
        self.ops["upload_bulk"] += 1
        self.files_uploaded += len(items)
        f(items)

    def exists(self, rel: str) -> bool:
        self.ops["exists"] += 1
        return self.inner.exists(rel)

    def listdir(self, rel: str) -> list[str]:
        self.ops["listdir"] += 1
        return self.inner.listdir(rel)

    def walk(self, rel: str) -> list[str]:
        self.ops["listdir"] += 1
        return self.inner.walk(rel)

    def commit(self) -> None:
        self.ops["commit"] += 1
        self.inner.commit()

    def reload(self) -> None:
        self.ops["reload"] += 1
        self.inner.reload()

    def report(self) -> dict[str, Any]:
        return {"ops": dict(self.ops), "files_read": self.files_read, "files_uploaded": self.files_uploaded}


# --------------------------------------------------------------------------- #
# the layout                                                                  #
# --------------------------------------------------------------------------- #


#: the results root every attempt root hangs off (`lp.RESULTS_ROOT_NAME`): ``seedrep/<run>/exp/seedrep_battery_modal``
RESULTS_ROOT_NAME = "seedrep_battery_modal"


def results_root_rel(W7: str) -> str:
    return f"seedrep/{W7}/exp/{RESULTS_ROOT_NAME}"


def shard_dir(shard: Any) -> str:
    """The shard DIRECTORY of an attempt root — ``shard<NN>`` from a numeric shard label, exactly as the Modal app spells it
    (``lp.attempt_root_for``: ``f"shard{int(shard):02d}"``, what the watcher reads and
    ``AttemptRoot`` parses). A label the app could not spell (non-numeric) REFUSES — there is no root for it anywhere."""
    if isinstance(shard, bool) or not re.fullmatch(r"[0-9]+", str(shard)):
        raise LadderRefusal(f"shard label {shard!r} is not numeric; the app spells attempt roots shard<NN> from a numeric label — no root exists")
    return f"shard{int(str(shard)):02d}"


def attempt_root_for(W7: str, *, leg: str, view: str, stage: Any, shard: Any, attempt_id: str) -> str:
    """THE attempt-root spelling, volume-relative (`lp.attempt_root_for` mirrored line for line — the
    equality is pinned by test against that function read by path): a BASE attempt at
    ``seedrep/<W7>/exp/seedrep_battery_modal/<leg>/<view>/shard<NN>/attempts/<attempt_id>``, a RUNG attempt (stage = its cap) at
    ``…/<leg>/<view>/rungs/<stage>/shard<NN>/attempts/<attempt_id>``. The reconciler's DEFAULT producer-root resolution, the pull
    rule and the consumer's default roots all spell through here; the pre-fix bare-label spelling (``…/<view>/0/attempts/``) is kept
    ONLY as :func:`_bare_attempt_root_rel`, to name it when it is found."""
    root = results_root_rel(W7)
    if str(stage) == STAGE_BASE:
        return f"{root}/{leg}/{view}/{shard_dir(shard)}/attempts/{attempt_id}"
    return f"{root}/{leg}/{view}/rungs/{stage}/{shard_dir(shard)}/attempts/{attempt_id}"


def attempt_root_rel(W7: str, leg: str, view: str, shard: str, attempt_id: str) -> str:
    """A BASE attempt's root (the ``stage = base`` case of :func:`attempt_root_for`; kept for its callers)."""
    return attempt_root_for(W7, leg=leg, view=view, stage=STAGE_BASE, shard=shard, attempt_id=attempt_id)


def producer_attempt_root(p: "ProducerRef", attempt_id: str) -> str:
    """A DECLARED producer's attempt root — its six-axis key's stage (``base`` for a base producer, the feeding rung's cap
    otherwise) and shard, through :func:`attempt_root_for`. The reconciler's default and what every test builds a producer at."""
    k = p.key
    return attempt_root_for(k.W7, leg=k.leg, view=k.view, stage=k.stage, shard=p.shard, attempt_id=attempt_id)


def _bare_attempt_root_rel(W7: str, leg: str, view: str, shard: str, attempt_id: str) -> str:
    """The PRE-FIX default (a bare shard label, stage-blind) — never written by the app; spelled only so a root found there is
    refused BY NAME instead of silently missed."""
    return f"{results_root_rel(W7)}/{leg}/{view}/{shard}/attempts/{attempt_id}"


def resolve_producer_root(vol: VolumeFS, p: "ProducerRef", attempt_id: str) -> str:
    """The reconciler's DEFAULT ``attempt_root_of``: the app's spelling (:func:`producer_attempt_root`) — after looking at the
    pre-fix bare-label root too. A ``COMPLETE.json`` / ``queue_inventory.json`` at the bare root and at the app's root REFUSES
    (two spellings for one shard: the divergence is named, never resolved by preference); at the bare root ONLY refuses as well
    (a root the app never writes is not resolved silently — the pre-fix default looked there and no
    queue ever closed). Neither / the app's root only -> the app's root."""
    canonical = producer_attempt_root(p, attempt_id)
    bare = _bare_attempt_root_rel(p.key.W7, p.key.leg, p.key.view, p.shard, attempt_id)
    if bare == canonical:
        return canonical

    def present(root: str) -> bool:
        return vol.exists(complete_rel(root)) or vol.exists(inventory_rel(root))

    if present(bare):
        if present(canonical):
            raise LadderRefusal(f"producer {p.shard}/{attempt_id}: attempt evidence under TWO spellings — {bare} (a bare shard label) and "
                                f"{canonical} (the app's shard<NN>); refusing to resolve either")
        raise LadderRefusal(f"producer {p.shard}/{attempt_id}: attempt evidence only under {bare} — a bare shard-label root the Modal app never "
                            f"writes (it spells {canonical}); not resolved. Pass attempt_root_of= for a root laid out differently")
    return canonical


def complete_rel(attempt_root: str) -> str:
    return f"{attempt_root}/COMPLETE.json"


def inventory_rel(attempt_root: str) -> str:
    return f"{attempt_root}/queue_inventory.json"


def ladder_root_rel(W7: str, wave_uuid: str, leg: str, view: str, cap_next: int) -> str:
    return f"seedrep/{W7}/{wave_uuid}/ladder/{leg}/{view}/{int(cap_next)}"


def queue_entry_rel(ladder_root: str, arm_s_: str, item_key_: str, producer_attempt_id: str) -> str:
    return f"{ladder_root}/queue/{arm_s_}/{item_key_}/{producer_attempt_id}.json"


def producer_marker_rel(ladder_root: str, shard: str, attempt_id: str) -> str:
    return f"{ladder_root}/producers/{shard}__{attempt_id}.CLOSED"


def closed_rel(ladder_root: str) -> str:
    return f"{ladder_root}/CLOSED.json"


def materialised_rel(ladder_root: str, arm_s_: str, item_key_: str, attempt_id: str) -> str:
    return f"{ladder_root}/materialised/{arm_s_}/{item_key_}/{attempt_id}.json"


def duplicates_rel(ladder_root: str, attempt_id: str) -> str:
    return f"{ladder_root}/duplicates/{attempt_id}.jsonl"


def speculative_rel(ladder_root: str, arm_s_: str, item_key_: str, attempt_id: str) -> str:
    """A TENTATIVE draw (``--speculative``: drawn from a live, not-yet-accepted producer's entry): never elected from
    here; promoted to :func:`materialised_rel` and elected only once its producer attempt is accepted (P1-5)."""
    return f"{ladder_root}/speculative/{arm_s_}/{item_key_}/{attempt_id}.json"


# --------------------------------------------------------------------------- #
# ladders, queue specs, producers                                               #
# --------------------------------------------------------------------------- #


def _rung_index(kind: str, cap: int) -> tuple[tuple[int, ...], int]:
    """``(ladder, index of cap)`` for a kind; refuses an unknown kind or an off-ladder cap."""
    ladder = LADDERS.get(kind)
    if ladder is None:
        raise LadderRefusal(f"ladder kind {kind!r} is not one of {sorted(LADDERS)}")
    cap = int(cap)
    if cap not in ladder:
        raise LadderRefusal(f"{kind}: cap {cap} is not a rung of {ladder}")
    return ladder, ladder.index(cap)


def predecessor_cap(kind: str, cap: int) -> int:
    """The IMMEDIATE predecessor of ``cap`` on the kind's ladder (controls: 512 -> 2,048 -> 8,192 -> 16,384;
    ``escalated_from`` is always this rung, never an earlier one). Refuses the base or an off-ladder
    cap."""
    ladder, i = _rung_index(kind, cap)
    if i == 0:
        raise LadderRefusal(f"{kind}: cap {int(cap)} is the base of {ladder}; it has no predecessor")
    return ladder[i - 1]


def next_cap(kind: str, cap: int) -> Optional[int]:
    """The rung ABOVE ``cap``, or ``None`` at the ceiling (16,384: recorded, never queued)."""
    ladder, i = _rung_index(kind, cap)
    if int(cap) >= CEILING or i + 1 >= len(ladder):
        return None
    return ladder[i + 1]


def producer_stage(kind: str, cap_next: int) -> str:
    """The ``stage`` axis of the queue's PRODUCERS: ``base`` when ``cap_next`` is the first rung above the base,
    else the predecessor cap (the rung whose containers feed this queue)."""
    prev = predecessor_cap(kind, cap_next)
    return STAGE_BASE if prev == LADDERS[kind][0] else str(prev)


@dataclasses.dataclass(frozen=True)
class ProducerRef:
    """One DECLARED producer of a queue: its shard label and its six-axis attempt key."""

    shard: str
    key: SixAxisKey


@dataclasses.dataclass(frozen=True)
class QueueSpec:
    """A queue: ``(wave_uuid, leg, view, kind, cap_next)`` and its declared producers."""

    W7: str
    wave_uuid: str
    leg: str
    view: str
    kind: str
    cap_next: int
    producers: tuple[ProducerRef, ...]

    def __post_init__(self) -> None:
        if self.kind not in LADDERS:
            raise LadderRefusal(f"kind {self.kind!r} is not one of {sorted(LADDERS)}")
        predecessor_cap(self.kind, self.cap_next)     # refuses the base / an off-ladder cap
        if int(self.cap_next) > CEILING:
            raise LadderRefusal(f"cap_next {self.cap_next} exceeds the ceiling {CEILING}")
        if not self.producers:
            raise LadderRefusal("a queue needs at least one declared producer")
        stage = producer_stage(self.kind, self.cap_next)
        for p in self.producers:
            if (p.key.W7, p.key.wave_uuid, p.key.leg, p.key.view) != (self.W7, self.wave_uuid, self.leg, self.view):
                raise LadderRefusal(f"producer {p.shard} key {p.key} is not this queue's (W7, wave_uuid, leg, view)")
            if p.key.stage != stage:
                raise LadderRefusal(f"producer {p.shard} is at stage {p.key.stage!r}; a {self.cap_next} {self.kind} "
                                    f"queue's producers are at stage {stage!r} (the immediate predecessor)")
            if p.key.shard != p.shard:
                raise LadderRefusal(f"producer shard {p.shard!r} != its key's shard {p.key.shard!r}")
        if len({p.shard for p in self.producers}) != len(self.producers):
            raise LadderRefusal("duplicate producer shards")

    @classmethod
    def build(cls, W7: str, wave_uuid: str, leg: str, view: str, kind: str, cap_next: int,
              shards: Sequence[str]) -> "QueueSpec":
        stage = producer_stage(kind, cap_next)
        prods = tuple(ProducerRef(str(s), SixAxisKey(W7, wave_uuid, leg, view, stage, str(s))) for s in shards)
        return cls(W7, wave_uuid, leg, view, kind, int(cap_next), prods)

    @property
    def queue_key(self) -> SixAxisKey:
        return SixAxisKey(self.W7, self.wave_uuid, self.leg, self.view, str(int(self.cap_next)), "queue")

    @property
    def ladder_root(self) -> str:
        return ladder_root_rel(self.W7, self.wave_uuid, self.leg, self.view, self.cap_next)

    @property
    def base_cap(self) -> int:
        return predecessor_cap(self.kind, self.cap_next)

    def next_spec(self, rung_shards: Sequence[str]) -> Optional["QueueSpec"]:
        """The queue THIS rung's containers feed (its producers = this rung's shards), or None at the ceiling."""
        nc = next_cap(self.kind, self.cap_next)
        if nc is None:
            return None
        return QueueSpec.build(self.W7, self.wave_uuid, self.leg, self.view, self.kind, nc, rung_shards)


# --------------------------------------------------------------------------- #
# records                                                                       #
# --------------------------------------------------------------------------- #


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _opt_int(v: Any) -> Optional[int]:
    """``int(v)``, or ``None`` for an absent value — the one normalisation of an optional seed / cap."""
    return None if v is None else int(v)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_sha256(row: Mapping[str, Any]) -> str:
    """``sha256(canonical JSON of the row)`` — the ``base_row_sha256`` every queue record binds to."""
    return sha256_hex(canonical_json(row).encode("utf-8"))


def slot(arm_s_: str, item_key_: str) -> str:
    """``<arm>__s<s>/<item_key>`` — the queue's unit: one ITEM under one ARM. The same item_key legitimately recurs
    under every arm a base container serves, so inventories, closures and projections key on the slot."""
    return f"{arm_s_}/{item_key_}"


def item_keys_sha256(keys: Iterable[str]) -> str:
    """The digest of a set of slots (``inventory.item_keys_sha256`` / ``CLOSED.json.item_keys_sha256``): sha256 over
    the sorted unique ``<arm>__s<s>/<item_key>`` strings joined by newlines."""
    return sha256_hex("\n".join(sorted(set(keys))).encode("utf-8"))


def base_row_identity(*, text: Optional[str], sampling_seed: Optional[int], finish_reason: Optional[str],
                      cap: Optional[int], text_sha256: Optional[str] = None) -> dict:
    """The NORMALISED identity of a base row every queue record binds to (``base_row_sha256`` =
    :func:`row_sha256` of this): the text's sha, the seed, the finish reason and the cap the text was drawn at.
    ONE shape for every leg and every rung, so the enqueuer (a base container reading its store row or a rung
    container reading its own materialised record) and the assembly (reading the ACCEPTED base row or the
    predecessor RUNG STORE's row, whose extra copy-forward keys would otherwise change a whole-row sha) compute
    the same digest. ``text_sha256`` may be given instead of the text (the controls event carries only the sha)."""
    if text is None and text_sha256 is None:
        tsha = None
    else:
        tsha = text_sha256 if text_sha256 is not None else sha256_hex(str(text).encode("utf-8"))
    return {"text_sha256": tsha, "sampling_seed": _opt_int(sampling_seed), "finish_reason": finish_reason, "cap": _opt_int(cap)}


def base_row_identity_sha256(**kw: Any) -> str:
    return row_sha256(base_row_identity(**kw))


_RECORD_REQUIRED = ("schema", "item_key", "kind", "leg", "arm", "s", "identity", "sampling_seed", "base_cap",
                    "cap_next", "base_row_sha256", "base_store", "producer", "W7", "wave_uuid", "enqueued_utc")


def make_queue_record(spec: QueueSpec, *, arm: str, s: int, kind: str, identity: Mapping[str, Any],
                      item_key_: str, sampling_seed: Optional[int], base_cap: int, base_row_sha256: str,
                      base_store: str, producer: ProducerRef, producer_attempt_id: str,
                      redraw: Optional[Mapping[str, Any]] = None) -> dict:
    """The queue entry's content (``{item_key, goal_id|item_id, sample_idx, trial, arm, s, sampling_seed,
    base_cap, base_row_sha256, base_store, producer: {shard, attempt_id}, W, wave_uuid}``), plus
    ``kind`` / ``leg`` / ``cap_next`` and an optional ``redraw`` block — what the rung needs to re-issue the request
    (``{vmodel, messages, sampling}``; the cap and the seed come from the record itself)."""
    if kind != spec.kind:
        raise LadderRefusal(f"record kind {kind!r} != the queue's {spec.kind!r}")
    if int(base_cap) != spec.base_cap:
        raise LadderRefusal(f"record base_cap {base_cap} != the queue's predecessor cap {spec.base_cap}")
    if producer not in spec.producers:
        raise LadderRefusal(f"producer {producer} is not a declared producer of this queue")
    rec = {
        "schema": SCHEMA_QUEUE, "item_key": item_key_, "kind": kind, "leg": spec.leg, "view": spec.view,
        "arm": arm, "s": int(s), "arm_s": arm_s(arm, s), "identity": dict(identity),
        "sampling_seed": _opt_int(sampling_seed), "base_cap": int(base_cap), "cap_next": int(spec.cap_next), "base_row_sha256": str(base_row_sha256),
        "base_store": str(base_store), "producer": {"shard": producer.shard, "attempt_id": producer_attempt_id},
        "W7": spec.W7, "wave_uuid": spec.wave_uuid, "enqueued_utc": _utc(),
    }
    if redraw is not None:
        rec["redraw"] = dict(redraw)
    return rec


def _validate_record(rec: Mapping[str, Any], spec: QueueSpec, *, where: str) -> None:
    missing = [k for k in _RECORD_REQUIRED if k not in rec]
    if rec.get("schema") != SCHEMA_QUEUE or missing:
        raise LadderRefusal(f"{where}: not a {SCHEMA_QUEUE} record (missing {missing})")
    if (rec["W7"], rec["wave_uuid"], rec["leg"], rec.get("view"), int(rec["cap_next"])) != \
            (spec.W7, spec.wave_uuid, spec.leg, spec.view, int(spec.cap_next)):
        raise LadderRefusal(f"{where}: a record of another queue ({rec['W7']}, {rec['wave_uuid']}, {rec['leg']}, "
                            f"{rec.get('view')}, {rec['cap_next']}) under this one's root")
    if rec["kind"] != spec.kind or int(rec["base_cap"]) != spec.base_cap:
        raise LadderRefusal(f"{where}: kind/base_cap {rec['kind']}/{rec['base_cap']} are not this queue's")
    check_redraw_sampling(rec, where=where)


_SAMPLER_KEYS: Optional[tuple[str, ...]] = None


def sampler_keys() -> tuple[str, ...]:
    """The FROZEN sampler key set every re-draw request carries — ``bon_served.SAMPLER_KEYS``, the six vendor terms the
    served target puts on the wire (``static_common.sampler_preset_of`` reads the same tuple, so the assembly's preset
    and a record's block are keyed alike). Lazy: the protocol module imports without the ``sisteer`` package where it
    need not; a container and the grading host have it on ``sys.path``."""
    global _SAMPLER_KEYS
    if _SAMPLER_KEYS is None:
        from sisteer.eval.attacks.bon_served import SAMPLER_KEYS
        _SAMPLER_KEYS = tuple(SAMPLER_KEYS)
    return _SAMPLER_KEYS


def check_redraw_sampling(rec: Mapping[str, Any], *, where: str) -> dict:
    """A DRAWABLE record (every kind but GSM8K, which Inspect draws) carries ``redraw.sampling`` as a
    mapping with EXACTLY the frozen sampler key set. Absent or not a mapping (the wire would run under the server's
    defaults, invisible to the assembly), a key missing (that term would default server-side) or a key beyond the set
    (a term no preset froze) REFUSES — at enqueue, at every read and again before the send. Returns the block."""
    if rec.get("kind") == "gsm8k":
        return {}
    redraw = rec.get("redraw")
    if not isinstance(redraw, Mapping):
        raise LadderRefusal(f"{where}: a drawable {rec.get('kind')!r} record carries no redraw block (vmodel, messages, sampling)")
    if not isinstance(redraw.get("vmodel"), str) or not redraw.get("vmodel") or not isinstance(redraw.get("messages"), list) \
            or not redraw["messages"]:
        raise LadderRefusal(f"{where}: the redraw block lacks vmodel / messages; this leg's rung cannot re-issue the request")
    sampling = redraw.get("sampling")
    if not isinstance(sampling, Mapping):
        raise LadderRefusal(f"{where}: redraw.sampling is {sampling!r}, not the sampler row — a request without it would be drawn "
                            "under the server's defaults, which no assembly could verify")
    want = sampler_keys()
    missing, extra = [k for k in want if k not in sampling], sorted(k for k in sampling if k not in want)
    if missing or extra:
        raise LadderRefusal(f"{where}: redraw.sampling must carry exactly the frozen sampler keys {list(want)}: "
                            f"missing {missing}, extra {extra}")
    return dict(sampling)


def redraw_messages(prompt: str, system: Optional[str]) -> list[dict]:
    """The re-draw's messages EXACTLY as the drivers' target composes them (``ServedChatTarget._payload``): one user
    turn, and for a prompted / stacked arm the arm's AUTHORITATIVE SI as the sole system message through the same
    ``jailbreak.install_authoritative_system`` the target calls. The proxy hash-verifies that prefix on every request
    of a prompted vmodel (H8) and refuses a request without it ``si_prefix_missing`` — so a rung record built for a
    prompted arm without its ``system`` could never be re-issued."""
    user = [{"role": "user", "content": str(prompt)}]
    if not system:
        return user
    # the ONE composer, from its stdlib-only home (launch F-G): `sisteer.eval.jailbreak` imports torch at module top and this
    # runs in the Modal APP process (the image's system python, no torch); lazy: the staged tree's src is on sys.path in a container
    from sisteer.eval.system_prefix import install_authoritative_system
    return install_authoritative_system(user, system)


def static_queue_record(spec: QueueSpec, *, arm: str, s: int, family: str, row: Mapping[str, Any],
                        base_store: str, producer: ProducerRef, producer_attempt_id: str,
                        vmodel: str, sampling: Mapping[str, Any], system: Optional[str] = None) -> dict:
    """A static family's at-cap base row -> its queue record (identity ``{family, goal_id, sample_idx, trial}``;
    the re-draw is the row's ``attack_prompt`` under the arm's ``system`` = the served target's ``.system`` — the SI
    of a prompted / stacked arm, ``None`` for every other arm — composed by :func:`redraw_messages`). Refuses a row
    that is not at cap."""
    if row.get("finish_reason") != "length":
        raise LadderRefusal(f"static row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r}) is not at "
                            "cap (finish_reason != 'length'); only at-cap items are queued")
    ident = {"family": family, "goal_id": row["goal_id"], "sample_idx": int(row["sample_idx"]),
             "trial": int(row.get("trial", 0))}
    seed, cap = int(row["sampling_seed"]), int((row.get("decode") or {}).get("cap", spec.base_cap))
    return make_queue_record(
        spec, arm=arm, s=s, kind="static", identity=ident,
        item_key_=item_key("static", family, row["goal_id"], int(row["sample_idx"])),
        sampling_seed=seed, base_cap=cap,
        base_row_sha256=base_row_identity_sha256(text=row.get("response"), sampling_seed=seed,
                                                 finish_reason=row.get("finish_reason"), cap=cap),
        base_store=base_store, producer=producer, producer_attempt_id=producer_attempt_id,
        redraw={"vmodel": vmodel, "messages": redraw_messages(row["attack_prompt"], system), "sampling": dict(sampling)})


def controls_queue_record(spec: QueueSpec, *, arm: str, s: int, control: str, event: Mapping[str, Any],
                          prompt: str, system: Optional[str], base_store: str, producer: ProducerRef,
                          producer_attempt_id: str, vmodel: str, sampling: Mapping[str, Any],
                          trial: int = 0) -> dict:
    """A sampled control's at-cap event (``controls_events.jsonl``: ``item_id, at_cap, finish_reason,
    sampling_seed, text_sha256, request_id, completion_id`` — the durable controls stream) -> its queue
    record (identity ``{control, item_id, trial}``; the re-draw is the item's prompt under the cell's system
    text). Refuses an event that is not at cap."""
    if not event.get("at_cap"):
        raise LadderRefusal(f"controls event {event.get('item_id')!r} is not at cap; only at-cap items are queued")
    ident = {"control": control, "item_id": str(event["item_id"]), "trial": int(trial)}
    seed = int(event["sampling_seed"])
    return make_queue_record(
        spec, arm=arm, s=s, kind="controls", identity=ident,
        item_key_=item_key("controls", control, event["item_id"], int(trial)),
        sampling_seed=seed, base_cap=spec.base_cap,
        base_row_sha256=base_row_identity_sha256(text=None, text_sha256=event["text_sha256"], sampling_seed=seed,
                                                 finish_reason=event.get("finish_reason"), cap=spec.base_cap),
        base_store=base_store, producer=producer, producer_attempt_id=producer_attempt_id,
        redraw={"vmodel": vmodel, "messages": redraw_messages(prompt, system), "sampling": dict(sampling)})


def misalign_queue_record(spec: QueueSpec, *, arm: str, s: int, row: Mapping[str, Any],
                          messages: Sequence[Mapping[str, str]], base_store: str, producer: ProducerRef,
                          producer_attempt_id: str, sampling: Mapping[str, Any]) -> dict:
    """A misalign ``agentic/<cell>/responses.jsonl`` row that hit the cap and is NOT a repetition loop
    (``truncated_selection``'s zlib rule is the TAILER's and the ASSEMBLY's to apply; this builder refuses only a
    row not at cap) -> its queue record (identity = the row's own keys; the re-draw = the exact messages)."""
    if row.get("finish_reason") != "length":
        raise LadderRefusal(f"misalign row {row.get('item_id')!r}/{row.get('sample_idx')!r} is not at cap")
    ident = {"family": row.get("family"), "behavior": row.get("behavior"), "item_id": str(row["item_id"]),
             "sample_idx": int(row["sample_idx"]), "cell_id": row.get("cell_id"),
             "virtual_model": row.get("virtual_model"), "prompt_sha": row.get("prompt_sha")}
    seed = _opt_int(row.get("seed"))
    return make_queue_record(
        spec, arm=arm, s=s, kind="misalign", identity=ident,
        item_key_=item_key("misalign", row["item_id"], int(row["sample_idx"])),
        sampling_seed=seed, base_cap=spec.base_cap,
        base_row_sha256=base_row_identity_sha256(text=row.get("response"), sampling_seed=seed,
                                                 finish_reason=row.get("finish_reason"), cap=spec.base_cap),
        base_store=base_store, producer=producer, producer_attempt_id=producer_attempt_id,
        redraw={"vmodel": row["virtual_model"], "messages": [dict(m) for m in messages], "sampling": dict(sampling)})


def gsm8k_queue_record(spec: QueueSpec, *, arm: str, s: int, item: Mapping[str, Any], base_store: str,
                       producer: ProducerRef, producer_attempt_id: str) -> dict:
    """A GSM8K filed outcome at cap (``stop_reason == max_tokens``) -> its queue record. No ``redraw`` block: the
    GSM8K rung is the Inspect driver over the queue-derived subset (``inspect_rung.subset_from_materialised_queue``);
    the record carries the item identity and the outcome's sha for the assembly's cross-check."""
    if not item.get("at_cap") or item.get("stop_reason") != "max_tokens":
        raise LadderRefusal(f"GSM8K item {item.get('item_id')!r} is not at cap")
    ident = {"item_id": str(item["item_id"]), "sample_idx": int(item.get("sample_idx", 0))}
    seed = _opt_int(item.get("seed"))
    return make_queue_record(
        spec, arm=arm, s=s, kind="gsm8k", identity=ident,
        item_key_=item_key("gsm8k", item["item_id"], int(item.get("sample_idx", 0))),
        sampling_seed=seed, base_cap=spec.base_cap,
        base_row_sha256=base_row_identity_sha256(text=None, sampling_seed=seed, finish_reason=item.get("stop_reason"), cap=spec.base_cap),
        base_store=base_store, producer=producer, producer_attempt_id=producer_attempt_id)


# --------------------------------------------------------------------------- #
# the producer protocol                                                          #
# --------------------------------------------------------------------------- #


class ProducerSealed(LadderRefusal):
    """An enqueue after the inventory: a late item is a producer bug by construction."""


class Producer:
    """ONE attempt's producer side of ONE queue: attempt-unique queue files, then the inventory (which seals it),
    then — after acceptance — the CLOSED marker. Every write goes through the attempt's own volume view and is
    committed by the caller's sync (``commit_each=True`` commits per write for the tests' fault injection)."""

    def __init__(self, vol: VolumeFS, spec: QueueSpec, producer: ProducerRef, attempt_id: str,
                 attempt_root: str, *, commit_each: bool = False) -> None:
        if producer not in spec.producers:
            raise LadderRefusal(f"{producer} is not a declared producer of {spec.ladder_root}")
        self.vol, self.spec, self.ref, self.attempt_id = vol, spec, producer, str(attempt_id)
        self.attempt_root = attempt_root
        self.commit_each = commit_each
        self.entries: list[dict] = []          # {item_key, arm_s, path, sha256}
        self._keys: set[str] = set()
        self.sealed = False
        self.inventory: Optional[dict] = None
        #: ONE synchronisation for enqueue and sealing: the tailer thread enqueues, the container's
        #: finish seals; the seal and the capture of the entry set happen under this lock BEFORE any blocking commit,
        #: so an enqueue racing the finish either lands in the inventory or raises ProducerSealed — never in between
        self._lock = threading.RLock()
        self.adopted: list[dict] = []        # adoption index entries the inventory lists

    @property
    def inventory_path(self) -> str:
        return inventory_rel(self.attempt_root)

    @property
    def marker_path(self) -> str:
        return producer_marker_rel(self.spec.ladder_root, self.ref.shard, self.attempt_id)

    def enqueue(self, record: Mapping[str, Any]) -> str:
        """Write ``queue/<arm>__s<s>/<item_key>/<attempt_id>.json`` (attempt-unique; refuses a second enqueue of
        the same item by this attempt and any enqueue after the inventory). Returns the volume path."""
        with self._lock:
            if self.sealed:
                raise ProducerSealed(f"{self.ref.shard}/{self.attempt_id}: enqueue of {record.get('item_key')!r} after "
                                     "queue_inventory.json — a late item is a producer bug")
            _validate_record(record, self.spec, where=f"producer {self.ref.shard}/{self.attempt_id}")
            if record["producer"] != {"shard": self.ref.shard, "attempt_id": self.attempt_id}:
                raise LadderRefusal(f"record producer {record['producer']} is not this producer "
                                    f"{{'shard': {self.ref.shard!r}, 'attempt_id': {self.attempt_id!r}}}")
            ik = record["item_key"]
            sl = slot(record["arm_s"], ik)
            if sl in self._keys:
                raise LadderRefusal(f"{self.ref.shard}/{self.attempt_id}: item {sl!r} enqueued twice by one attempt")
            rel = queue_entry_rel(self.spec.ladder_root, record["arm_s"], ik, self.attempt_id)
            data = (canonical_json(record) + "\n").encode("utf-8")
            self.vol.write_bytes(rel, data, exclusive=True)    # ours by construction; a clash IS a bug
            self._keys.add(sl)
            self.entries.append({"item_key": ik, "arm_s": record["arm_s"], "slot": sl, "path": rel, "sha256": sha256_hex(data)})
            if self.commit_each:
                self.vol.commit()
            return rel

    def seal(self) -> list[dict]:
        """Seal the producer and CAPTURE its entry set — under the enqueue lock, before any blocking commit (P2-9).
        Idempotent: a sealed producer returns the same captured set."""
        with self._lock:
            self.sealed = True
            return list(self.entries)

    def write_inventory(self) -> dict:
        """``queue_inventory.json`` in the ATTEMPT ROOT (step (ii)): every entry this attempt enqueued with its sha.
        Seals the producer FIRST (a concurrent enqueue from here on raises ProducerSealed) and captures the set, then
        commits the captured attempt-unique files (they precede the inventory), then the inventory. The
        tailer must be drained / joined before this (``finish_producer(drain=…)``): a late item is a producer bug."""
        with self._lock:
            if self.inventory is not None:
                return self.inventory
            entries = self.seal()
            self.vol.commit()                   # the attempt-unique files precede their inventory
            doc = {"schema": SCHEMA_INVENTORY, "key": key_str(self.ref.key), "queue_root": self.spec.ladder_root,
                   "attempt_id": self.attempt_id, "shard": self.ref.shard, "n": len(entries),
                   "item_keys_sha256": item_keys_sha256(slot(e["arm_s"], e["item_key"]) for e in entries),
                   "entries": entries, "written_utc": _utc()}
            if self.adopted:                     # the evidence this attempt adopted from earlier executions
                doc["adopted"] = [dict(a) for a in self.adopted]
            self.vol.write_bytes(self.inventory_path, (canonical_json(doc) + "\n").encode("utf-8"))
            self.vol.commit()
            self.inventory = doc
            return doc

    def write_closed_marker(self, accepted_kv: KV) -> dict:
        """Step (iv): ``producers/<shard>__<attempt_id>.CLOSED`` — ONLY when the accepted Dict names THIS attempt
        (a superseded attempt publishes nothing)."""
        if self.inventory is None:
            raise LadderRefusal("the CLOSED marker follows the inventory and the acceptance; write_inventory first")
        holder = accepted_attempt(accepted_kv, self.ref.key)
        if holder != self.attempt_id:
            raise LadderRefusal(f"{self.ref.shard}/{self.attempt_id} is not the accepted attempt ({holder!r}); a "
                                "superseded attempt writes no CLOSED marker")
        return write_closed_marker(self.vol, self.spec, self.ref, self.attempt_id, self.inventory, synthesised_by=None)


def write_closed_marker(vol: VolumeFS, spec: QueueSpec, producer: ProducerRef, attempt_id: str,
                        inventory: Mapping[str, Any], *, synthesised_by: Optional[str]) -> dict:
    """The idempotent producer marker ``{n_enqueued, item_keys_sha256, …}``; an existing marker with the same
    content is left alone, a conflicting one refuses."""
    doc = {"schema": SCHEMA_MARKER, "key": key_str(producer.key), "queue_root": spec.ladder_root,
           "shard": producer.shard, "attempt_id": str(attempt_id), "n_enqueued": int(inventory["n"]),
           "item_keys_sha256": inventory["item_keys_sha256"], "synthesised_by": synthesised_by}
    rel = producer_marker_rel(spec.ladder_root, producer.shard, attempt_id)
    if vol.exists(rel):
        old = json.loads(vol.read_bytes(rel).decode("utf-8"))
        core = {k: v for k, v in old.items() if k not in ("synthesised_by", "written_utc")}
        if core != {k: v for k, v in doc.items() if k not in ("synthesised_by", "written_utc")}:
            raise LadderRefusal(f"{rel}: an existing CLOSED marker disagrees ({core} vs {doc}); refusing to overwrite")
        return old
    doc["written_utc"] = _utc()
    vol.write_bytes(rel, (canonical_json(doc) + "\n").encode("utf-8"))
    vol.commit()
    return doc


def accept_attempt(accepted_kv: KV, key: SixAxisKey, attempt_id: str) -> bool:
    """Step (iii): ONE ``accepted.put(key, attempt_id, skip_if_exists=True)``. ``True`` = this attempt is
    accepted; ``False`` = another attempt holds it (the caller marks itself ``superseded_by`` and publishes
    nothing)."""
    return accepted_kv.put(key_str(key), str(attempt_id), skip_if_exists=True)


def accepted_attempt(accepted_kv: KV, key: SixAxisKey) -> Optional[str]:
    v = accepted_kv.get(key_str(key))
    return None if v is None else str(v)


def finish_producer(producer: Producer, accepted_kv: KV, *, crash_after: Optional[str] = None,
                    mark_superseded: Optional[Callable[[str], None]] = None,
                    drain: Optional[Callable[[], Any]] = None) -> dict:
    """The container's steps (ii) -> (iii) -> (iv) in order, after its store's COMPLETE.json (step (i), the
    caller's). ``drain`` (the tailer's stop-and-join: it has read the stream to its end and enqueued every at-cap
    item) runs FIRST, before the seal — the inventory then names every item the stream carried (P2-9).
    ``crash_after`` ∈ {inventory, acceptance} raises :class:`_Crash` at that boundary (fault injection; a real
    container simply dies). Returns ``{state, attempt_id, holder}`` with state ``accepted`` / ``superseded``."""
    if drain is not None:
        drain()
    producer.write_inventory()
    if crash_after == "inventory":
        raise _Crash("after inventory")
    won = accept_attempt(accepted_kv, producer.ref.key, producer.attempt_id)
    if not won:
        holder = accepted_attempt(accepted_kv, producer.ref.key)
        if mark_superseded is not None:
            mark_superseded(holder or "")
        return {"state": "superseded", "attempt_id": producer.attempt_id, "holder": holder}
    if crash_after == "acceptance":
        raise _Crash("after acceptance")
    producer.write_closed_marker(accepted_kv)
    return {"state": "accepted", "attempt_id": producer.attempt_id, "holder": producer.attempt_id}


class _Crash(RuntimeError):
    """A simulated container death at a protocol boundary (tests)."""


# --------------------------------------------------------------------------- #
# adopted evidence: an accepted attempt carries the request pipeline of every elected result it relies on                #
# --------------------------------------------------------------------------- #

SCHEMA_ADOPTED = "seedrep_adopted_evidence/1"
#: the request-pipeline files of an attempt root as the audit locates them — by NAME, exactly one each
#: under the root: the client's two evidence files, the proxy's usage ledger, the engine's log (gzipped at exit)
EVIDENCE_REQUEST_FILE = "request_evidence.jsonl"
EVIDENCE_RESPONSE_FILE = "response_evidence.jsonl"
LEDGER_FILE = "usage_ledger.jsonl"
ENGINE_LOG_NAMES: tuple[str, ...] = ("vllm_server.log.gz", "vllm_server.log")
ADOPTED_DIR = "adopted_from"
ADOPTED_MANIFEST = "adoption.json"
ADOPTED_INDEX = "ADOPTED.json"
ATTEMPT_MARKER = "ATTEMPT.json"
#: vLLM's `--enable-log-requests` line — the id only (`audit_common._RECEIVED_RE`, spelled alike)
_RECEIVED_RE = re.compile(r"Received request ([^\s:]+):")


def _is_engine_id_of(engine_id: Any, external_id: Any) -> bool:
    """``contracts.is_engine_id_of`` — the ONE rule for "this engine-log id is that completion" (the id itself, or the id
    plus exactly one ``-<8 hex>`` suffix)."""
    from sisteer.capeval.tinycap.contracts import is_engine_id_of
    return bool(is_engine_id_of(engine_id, external_id))


def _one_under(vol: VolumeFS, root: str, name: str) -> Optional[str]:
    """Exactly one file called ``name`` under ``root`` on the volume (the ``find_one`` rule: several refuse), or None."""
    hits = sorted(p for p in vol.walk(root) if PurePosixPath(p).name == name)
    if len(hits) > 1:
        raise LadderRefusal(f"{root}: {len(hits)} files called {name!r} (first: {hits[:3]}) — an ambiguous attempt root is not read")
    return hits[0] if hits else None


def _one_local(root: Path, name: str, *, max_depth: int = 10) -> Optional[Path]:
    """The local-filesystem twin of :func:`_one_under` (a bounded walk, like ``walk_files``)."""
    root = Path(root)
    hits: list[Path] = []
    base = len(root.parts)
    stack = [root]
    while stack:
        d = stack.pop()
        if len(d.parts) - base > max_depth or not d.is_dir():
            continue
        for e in sorted(d.iterdir()):
            if e.is_dir() and not e.is_symlink():
                stack.append(e)
            elif e.name == name and e.is_file():
                hits.append(e)
    if len(hits) > 1:
        raise LadderRefusal(f"{root}: {len(hits)} files called {name!r} (first: {[str(h) for h in hits[:3]]}) — an ambiguous attempt root")
    return hits[0] if hits else None


def attempt_root_marker(start: "Path | str") -> Optional[Path]:
    """The nearest directory at or above ``start`` holding ``ATTEMPT.json`` — the attempt root the attempt writes first;
    None when no ancestor carries it."""
    p = Path(start).resolve()
    for d in (p, *p.parents):
        if (d / ATTEMPT_MARKER).is_file():
            return d
    return None


def _jsonl_lines(data: bytes) -> list[str]:
    return [ln for ln in data.decode("utf-8").split("\n") if ln.strip()]


def read_source_evidence(vol: VolumeFS, source_root: str) -> dict:
    """The request pipeline of a (possibly dead) attempt root ON THE VOLUME, as its TreeSyncer left it: the two evidence files,
    the usage ledger and the engine log (``.gz`` preferred; a dead container never gzipped, so the raw synced log is read),
    each located by name exactly once (an absent file is an empty list — the coverage check names what it lacks). Returns
    ``{files: {kind: {path, sha256, n_lines}}, request: [(line, rec)], response: [(line, row)], ledger: [(line, row)],
    engine: [(line, id)]}``."""
    out: dict = {"files": {}, "request": [], "response": [], "ledger": [], "engine": []}
    for kind, name in (("request", EVIDENCE_REQUEST_FILE), ("response", EVIDENCE_RESPONSE_FILE), ("ledger", LEDGER_FILE)):
        rel = _one_under(vol, source_root, name)
        if rel is None:
            continue
        data = vol.read_bytes(rel)
        lines = _jsonl_lines(data)
        out["files"][kind] = {"path": rel, "sha256": sha256_hex(data), "n_lines": len(lines)}
        parsed = []
        for ln in lines:
            try:
                parsed.append((ln, json.loads(ln)))
            except json.JSONDecodeError:
                raise LadderRefusal(f"{rel}: a torn / non-JSON line; the source evidence cannot be adopted") from None
        out[kind] = parsed
    for name in ENGINE_LOG_NAMES:
        rel = _one_under(vol, source_root, name)
        if rel is None:
            continue
        data = vol.read_bytes(rel)
        text = gzip.decompress(data).decode("utf-8", "replace") if name.endswith(".gz") else data.decode("utf-8", "replace")
        out["files"]["engine_log"] = {"path": rel, "sha256": sha256_hex(data), "n_lines": text.count("\n")}
        for ln in text.split("\n"):
            m = _RECEIVED_RE.search(ln)
            if m:
                out["engine"].append((ln, m.group(1)))
        break
    return out


def evidence_for_items(src: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> tuple[dict, list[str]]:
    """The exact source lines that make ``items`` (``{request_id, completion_id, slot}``) auditable, and the problems: per item
    ONE task request record of its request id, its 2xx ledger row whose body digest equals the record's, an authoritative
    completion for it (the ledger row's ``completion_id`` or a response row) naming its completion id, and EXACTLY ONE
    engine-log ``Received request`` line of that completion (``contracts.is_engine_id_of``) — the ``bind_rows`` rule,
    applied before the copy so a partially synced predecessor is refused, never half-adopted."""
    by_rid_rec = {}
    for ln, rec in src["request"]:
        rid = rec.get("request_id")
        if rid in by_rid_rec:
            raise LadderRefusal(f"source request evidence carries request_id {rid!r} twice; not adoptable")
        by_rid_rec[rid] = (ln, rec)
    ledger_by_rid: dict = {}
    for ln, row in src["ledger"]:
        ledger_by_rid.setdefault(row.get("request_id"), []).append((ln, row))
    resp_by_rid: dict = {}
    for ln, row in src["response"]:
        resp_by_rid.setdefault(row.get("request_id"), []).append((ln, row))
    picked = {"request": [], "response": [], "ledger": [], "engine": []}
    problems: list[str] = []
    for it in items:
        rid, cid, where = str(it.get("request_id") or ""), str(it.get("completion_id") or ""), it.get("slot")
        if not rid or not cid:
            problems.append(f"{where}: the elected result names no request_id / completion_id ({rid!r}, {cid!r})")
            continue
        rec = by_rid_rec.get(rid)
        if rec is None or rec[1].get("kind") != "task":
            problems.append(f"{where}: no task request record for request_id {rid!r} in the source's {EVIDENCE_REQUEST_FILE}")
            continue
        rows = ledger_by_rid.get(rid, [])
        ok_rows = [(ln, r) for ln, r in rows if not r.get("rejected") and isinstance(r.get("status"), int) and 200 <= int(r["status"]) < 300]
        if len(ok_rows) != 1:
            problems.append(f"{where}: {len(ok_rows)} accepted ledger row(s) for request_id {rid!r} in the source's {LEDGER_FILE} (one required)")
            continue
        if ok_rows[0][1].get("body_sha256") != rec[1].get("body_sha256"):
            problems.append(f"{where}: the ledger row's body_sha256 differs from the request record's for {rid!r} — two records of one wire")
            continue
        resp = [(ln, r) for ln, r in resp_by_rid.get(rid, []) if r.get("completion_id")]
        completions = {str(ok_rows[0][1].get("completion_id"))} if ok_rows[0][1].get("completion_id") else set()
        completions |= {str(r.get("completion_id")) for _, r in resp}
        if completions != {cid}:
            problems.append(f"{where}: the source's authoritative completion(s) for {rid!r} are {sorted(completions)}, the elected result says {cid!r}")
            continue
        eng = [(ln, e) for ln, e in src["engine"] if _is_engine_id_of(e, cid)]
        if len(eng) != 1:
            problems.append(f"{where}: completion_id {cid!r} matches {len(eng)} engine-log `Received request` line(s) in the source (exactly one required)")
            continue
        picked["request"].append(rec)
        picked["ledger"].extend(ok_rows)
        picked["response"].extend(resp)
        picked["engine"].extend(eng)
    return picked, problems


def _append_missing(path: Path, lines: Sequence[str], present: Callable[[str], bool]) -> int:
    """Append every line whose ``present(line)`` is False (idempotent across a re-entered consumer); fsynced."""
    todo = [ln for ln in lines if not present(ln)]
    if not todo:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ln in todo:
            f.write(ln.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())
    return len(todo)


def adopt_evidence(vol: VolumeFS, *, into_root: "Path | str", evidence_dir: "Path | str", source_root: str, source_attempt: str,
                   items: Sequence[Mapping[str, Any]], adopter: Mapping[str, Any]) -> dict:
    """Make the request pipeline of ``source_attempt``'s elected results (``items``) a VERIFIED part of the adopting attempt's
    root: read the source root on the volume (:func:`read_source_evidence`), require every item covered
    (:func:`evidence_for_items` — anything missing REFUSES, nothing is written), then MERGE the exact source lines into the
    adopting root's own four files — the request / response records into ``evidence_dir`` (where the client writes),
    the ledger rows into the root's ``usage_ledger.jsonl`` (created when the root has none), the engine lines into the root's
    ONE ``vllm_server.log`` (``O_APPEND``; it must exist — an engine wrote it) — idempotently (a re-entered consumer appends
    nothing twice), and write the manifest ``adopted_from/<source>/adoption.json`` (source file digests, the copied blocks'
    digests and counts, every item) plus the root's ``adopted_from/ADOPTED.json`` index. The reader sees the root unchanged:
    ``find_one`` still finds exactly one of each canonical file. Returns the manifest."""
    into_root, evidence_dir = Path(into_root), Path(evidence_dir)
    src = read_source_evidence(vol, source_root)
    picked, problems = evidence_for_items(src, items)
    if problems:
        raise LadderRefusal(f"elected results of attempt {source_attempt!r} (root {source_root}) cannot be adopted by "
                            f"{adopter.get('attempt_id')!r}: {len(problems)} of {len(items)} not covered by the source's request pipeline "
                            f"— first: {problems[0]}")
    ledger_p = _one_local(into_root, LEDGER_FILE) or (into_root / LEDGER_FILE)
    engine_p = _one_local(into_root, ENGINE_LOG_NAMES[1])
    if engine_p is None:
        raise LadderRefusal(f"{into_root}: no {ENGINE_LOG_NAMES[1]} under the attempt root to merge the adopted engine lines into — "
                            "an attempt root without its engine's log cannot be audited")
    req_p, resp_p = evidence_dir / EVIDENCE_REQUEST_FILE, evidence_dir / EVIDENCE_RESPONSE_FILE

    def present_ids(path: Path, key: str) -> set:
        if not path.is_file():
            return set()
        out = set()
        for ln in _jsonl_lines(path.read_bytes()):
            try:
                out.add(str(json.loads(ln).get(key)))
            except json.JSONDecodeError:
                continue
        return out

    have_req, have_led = present_ids(req_p, "request_id"), present_ids(ledger_p, "request_id")
    have_resp = {(str(json.loads(ln).get("request_id")), str(json.loads(ln).get("completion_id"))) for ln in _jsonl_lines(resp_p.read_bytes())} \
        if resp_p.is_file() else set()
    have_eng = {m.group(1) for m in (_RECEIVED_RE.search(ln) for ln in engine_p.read_text(encoding="utf-8", errors="replace").split("\n")) if m}
    n = {
        "request_evidence": _append_missing(req_p, [ln for ln, _ in picked["request"]], lambda ln: str(json.loads(ln).get("request_id")) in have_req),
        "response_evidence": _append_missing(resp_p, [ln for ln, _ in picked["response"]],
                                             lambda ln: (str(json.loads(ln).get("request_id")), str(json.loads(ln).get("completion_id"))) in have_resp),
        "usage_ledger": _append_missing(ledger_p, [ln for ln, _ in picked["ledger"]], lambda ln: str(json.loads(ln).get("request_id")) in have_led),
        "engine_lines": _append_missing(engine_p, [ln for ln, _ in picked["engine"]], lambda ln: _RECEIVED_RE.search(ln).group(1) in have_eng),
    }
    blocks = {k: sha256_hex("\n".join(ln for ln, _ in picked[kk]).encode("utf-8"))
              for k, kk in (("request_evidence", "request"), ("response_evidence", "response"), ("usage_ledger", "ledger"), ("engine_lines", "engine"))}
    manifest = {"schema": SCHEMA_ADOPTED, "adopted_by": dict(adopter), "source": {"attempt_id": str(source_attempt), "attempt_root": str(source_root)},
                "items": [dict(it) for it in items], "source_files": src["files"], "copied_sha256": blocks, "appended_now": n,
                "targets": {"request_evidence": str(req_p), "response_evidence": str(resp_p), "usage_ledger": str(ledger_p), "engine_log": str(engine_p)},
                "written_utc": _utc()}
    mdir = into_root / ADOPTED_DIR / str(source_attempt)
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / ADOPTED_MANIFEST).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    index_p = into_root / ADOPTED_DIR / ADOPTED_INDEX
    index = json.loads(index_p.read_text()) if index_p.is_file() else {"schema": SCHEMA_ADOPTED, "adoptions": []}
    entry = {"attempt_id": str(source_attempt), "manifest": f"{ADOPTED_DIR}/{source_attempt}/{ADOPTED_MANIFEST}",
             "sha256": sha256_hex((canonical_json(manifest) + "\n").encode("utf-8")), "n_items": len(items)}
    index["adoptions"] = [e for e in index["adoptions"] if e.get("attempt_id") != str(source_attempt)] + [entry]
    index["written_utc"] = _utc()
    index_p.write_text(canonical_json(index) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# projections (rungs read ONLY the accepted projection)                        #
# --------------------------------------------------------------------------- #


def accepted_producer_attempts(accepted_kv: KV, spec: QueueSpec) -> dict[str, Optional[str]]:
    """``{shard: accepted attempt_id | None}`` over the queue's DECLARED producers."""
    return {p.shard: accepted_attempt(accepted_kv, p.key) for p in spec.producers}


def _read_json(vol: VolumeFS, rel: str) -> Any:
    try:
        return json.loads(vol.read_bytes(rel).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise LadderRefusal(f"{rel}: torn or not JSON ({type(e).__name__}: {e})") from None


QueueFiles = list[tuple[str, str, str, str]]


def _queue_files(vol: VolumeFS, spec: QueueSpec) -> QueueFiles:
    """Every ``(arm_s, item_key, attempt_id, rel)`` under ``queue/`` — any attempt's. ONE walk per pass: callers
    that need the listing more than once pass it on (``files=``)."""
    out = []
    prefix = f"{spec.ladder_root}/queue/"
    for rel in vol.walk(f"{spec.ladder_root}/queue"):
        tail = rel[len(prefix):].split("/")
        if len(tail) != 3 or not tail[2].endswith(".json"):
            raise LadderRefusal(f"{rel}: not a queue/<arm>__s<s>/<item_key>/<attempt_id>.json path")
        out.append((tail[0], tail[1], tail[2][:-len(".json")], rel))
    return out


def _queue_record(vol: VolumeFS, spec: QueueSpec, rel: str, cache: Optional[dict[str, dict]], *,
                  blobs: Optional[Mapping[str, bytes]] = None) -> dict:
    """The validated record at ``rel``. Queue files are attempt-unique and never rewritten, so a caller that polls
    (the rung consumer, every round for hours) hands in a ``cache`` and pays the read + parse + validation once per
    path. ``blobs`` (fj32): bytes already read in ONE bulk pull -- parsed and validated from there exactly as a read
    would be; a path the pull did not carry is read from the volume."""
    if cache is not None and rel in cache:
        return cache[rel]
    if blobs is not None and rel in blobs:
        try:
            rec = json.loads(blobs[rel].decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise LadderRefusal(f"{rel}: torn or not JSON ({type(e).__name__}: {e})") from None
    else:
        rec = _read_json(vol, rel)
    _validate_record(rec, spec, where=rel)
    if cache is not None:
        cache[rel] = rec
    return rec


def accepted_queue_entries(vol: VolumeFS, accepted_kv: Optional[KV], spec: QueueSpec,
                           accepted: Optional[Mapping[str, Optional[str]]] = None, *, files: Optional[QueueFiles] = None,
                           cache: Optional[dict[str, dict]] = None, blobs: Optional[Mapping[str, bytes]] = None) -> dict[str, dict]:
    """The ACCEPTED PROJECTION: ``{item_key: record}`` reading exactly ``<item_key>/<accepted_attempt_id>.json``
    for every declared producer with an accepted attempt; every other attempt's file is inert. A record whose
    content names another producer / attempt than its path, or another queue, refuses (a forged or misfiled
    entry is never read as an item). Two accepted producers naming one item: on a RUNG-produced queue, kept once
    (the first by declared shard order) when the two records describe the same world (:func:`same_world`) — two
    rung shards forwarding the same elected result (P1-7) — and refused when they differ; on a base-produced queue
    always refused (base shards are disjoint by construction; an agreeing duplicate there is a partition bug)."""
    acc = dict(accepted) if accepted is not None else accepted_producer_attempts(accepted_kv, spec)
    by_attempt = {aid: shard for shard, aid in acc.items() if aid}
    order = {p.shard: i for i, p in enumerate(spec.producers)}
    tolerate = producer_stage(spec.kind, spec.cap_next) != STAGE_BASE
    cands: dict[str, list[tuple[int, str, dict]]] = {}
    for arm_s_, ik, aid, rel in (files if files is not None else _queue_files(vol, spec)):
        if aid not in by_attempt:
            continue                          # inert: not an accepted attempt's file
        rec = _queue_record(vol, spec, rel, cache, blobs=blobs)
        if rec["item_key"] != ik or rec["arm_s"] != arm_s_:
            raise LadderRefusal(f"{rel}: content names item {rec['item_key']!r}/{rec['arm_s']!r}, path says {ik!r}/{arm_s_!r}")
        if rec["producer"] != {"shard": by_attempt[aid], "attempt_id": aid}:
            raise LadderRefusal(f"{rel}: content names producer {rec['producer']}, the path/acceptance says "
                                f"{{'shard': {by_attempt[aid]!r}, 'attempt_id': {aid!r}}}")
        cands.setdefault(slot(arm_s_, ik), []).append((order[by_attempt[aid]], rel, rec))
    out: dict[str, dict] = {}
    for sl, lst in cands.items():
        lst.sort(key=lambda t: t[0])
        first = lst[0][2]
        for _, rel, rec in lst[1:]:
            if not tolerate or not same_world(rec, first):
                raise LadderRefusal(f"{rel}: item {sl} was also enqueued by accepted attempt "
                                    f"{first['producer']['attempt_id']!r}"
                                    + (" with a DIFFERENT binding; shards must be disjoint or agree" if tolerate
                                       else "; base shards must be disjoint"))
        out[sl] = first
    return out


_BINDING_FREE = frozenset({"producer", "enqueued_utc"})


def same_world(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two queue records describe the SAME base row to re-draw — everything but who enqueued them and when
    (identity, item, seed, caps, ``base_row_sha256``, ``base_store``, the redraw block). The ONE predicate behind
    "is this entry / elected result / tentative draw about the record I hold" (duplicate tolerance, the consumer's
    found-result forwarding and speculative promotion)."""
    return {k: v for k, v in a.items() if k not in _BINDING_FREE} == {k: v for k, v in b.items() if k not in _BINDING_FREE}


def live_queue_entries(vol: VolumeFS, spec: QueueSpec, *, files: Optional[QueueFiles] = None,
                       cache: Optional[dict[str, dict]] = None) -> dict[str, list[dict]]:
    """EVERY attempt's entries, ``{arm_s/item_key: [record, …]}`` — the spin-condition count
    (``len(queue) >= S_rung``) and the optional speculative consumption; NEVER the assembly's input."""
    out: dict[str, list[dict]] = {}
    for arm_s_, ik, aid, rel in (files if files is not None else _queue_files(vol, spec)):
        out.setdefault(slot(arm_s_, ik), []).append(_queue_record(vol, spec, rel, cache))
    return out


def read_closed(vol: VolumeFS, spec: QueueSpec) -> Optional[dict]:
    rel = closed_rel(spec.ladder_root)
    if not vol.exists(rel):
        return None
    doc = _read_json(vol, rel)
    if doc.get("schema") != SCHEMA_CLOSED or doc.get("queue_root") != spec.ladder_root:
        raise LadderRefusal(f"{rel}: not this queue's {SCHEMA_CLOSED} record")
    return doc


# --------------------------------------------------------------------------- #
# the reconciler (recoverable closure; stale claims)                           #
# --------------------------------------------------------------------------- #


def _verify_inventory_files(vol: VolumeFS, inv: Mapping[str, Any], spec: QueueSpec, attempt_id: str, *,
                            files: QueueFiles, blobs: Optional[Mapping[str, bytes]] = None) -> list[str]:
    """Every inventory entry exists on the volume (in the pass's ONE ``queue/`` listing) with its sha and is THIS
    attempt's path, and no committed file of the attempt lies outside the inventory; the problems, if any.
    ``blobs`` (fj32): the entries' bytes already read in ONE bulk pull -- checked from there instead of one ``read_bytes`` per entry; an
    entry the pull did not carry is read from the volume as before (never a false problem)."""
    problems = []
    on_volume = {rel for _, _, _, rel in files}
    if inv.get("schema") != SCHEMA_INVENTORY or inv.get("attempt_id") != attempt_id:
        return [f"inventory is not this attempt's ({inv.get('attempt_id')!r} != {attempt_id!r})"]
    if inv.get("queue_root") != spec.ladder_root:
        return [f"inventory names queue root {inv.get('queue_root')!r}, not {spec.ladder_root!r}"]
    entries = inv.get("entries")
    if not isinstance(entries, list) or int(inv.get("n", -1)) != len(entries):
        return ["inventory n / entries disagree"]
    for e in entries:
        rel = e.get("path", "")
        if not rel.startswith(f"{spec.ladder_root}/queue/") or not rel.endswith(f"/{attempt_id}.json"):
            problems.append(f"{rel}: not an attempt-unique queue path of {attempt_id}")
            continue
        if rel not in on_volume:
            problems.append(f"{rel}: missing on the volume")
            continue
        data = blobs.get(rel) if blobs is not None else None
        if data is None:
            data = vol.read_bytes(rel)                # not in the bulk read (or none given): the per-file read, as before
        if sha256_hex(data) != e.get("sha256"):
            problems.append(f"{rel}: sha differs from the inventory")
    if item_keys_sha256(slot(e.get("arm_s", ""), e.get("item_key", "")) for e in entries) != inv.get("item_keys_sha256"):
        problems.append("inventory item_keys_sha256 does not match its entries")
    listed = {e.get("path") for e in entries}
    stray = sorted(rel for _, _, aid, rel in files if aid == attempt_id and rel not in listed)
    if stray:                                  # a file this attempt committed after sealing: the inventory is not its set
        problems.append(f"{len(stray)} committed queue file(s) of {attempt_id} are not in its inventory (first: {stray[:3]})")
    return problems


def reconcile_queue(vol: VolumeFS, accepted_kv: KV, spec: QueueSpec, *, claims_kv: Optional[KV] = None,
                    results_kv: Optional[KV] = None, lease_state: Optional[Callable[[str], str]] = None,
                    attempt_root_of: Optional[Callable[[ProducerRef, str], str]] = None,
                    blobs: Optional[Mapping[str, bytes]] = None) -> dict:
    """ONE reconciler pass over ONE queue (the grading host loop, every 120 s; ONE process per wave — two reconcilers
    could both win a reclaim). Per declared producer, the SAME evidence is verified whether or not a marker exists
    (P1-6): ``COMPLETE.json`` present, the inventory this attempt's and this queue's, every listed file on the
    volume with its sha, no committed queue file of the attempt outside its inventory. Then: ``PRODUCER_MISSING``
    (no accepted attempt), ``CLOSED`` (verified + the accepted attempt's own marker, which must describe its
    inventory), ``CLOSED_SYNTHESISED`` (verified, no marker -> synthesised here), or ``ACCEPTED_UNVERIFIED`` (the
    evidence is incomplete — not closable, reasons listed, a marker notwithstanding). ``CLOSED.json`` is written
    when every producer is CLOSED and the accepted projection equals the accepted inventories' union EXACTLY
    (P2-9; idempotent; a conflicting existing record refuses). Stale claims: with ``claims_kv`` +
    ``lease_state``, a claim whose claimant call is TERMINAL and whose item has no elected result is popped; a
    running / unknown claimant is left alone. Returns the status. ``blobs`` (fj32): the queue files' bytes as ONE bulk read of the volume
    delivered them (the replay's read-back after its upload) -- the inventory verification and the accepted projection take their bytes from
    there instead of one read per file; a path not in it is read from the volume as before."""
    vol.reload()
    root_of = attempt_root_of or (lambda p, aid: resolve_producer_root(vol, p, aid))   # the app's shard<NN> spelling, divergences refused
    acc = accepted_producer_attempts(accepted_kv, spec)
    files = _queue_files(vol, spec)             # ONE listing per pass (a Volume-API round trip on the grading host)
    producers: dict[str, dict] = {}
    for p in spec.producers:
        aid = acc[p.shard]
        st: dict[str, Any] = {"shard": p.shard, "attempt_id": aid, "reasons": []}
        if aid is None:
            st["state"] = PRODUCER_MISSING
            producers[p.shard] = st
            continue
        marker = producer_marker_rel(spec.ladder_root, p.shard, aid)
        root = root_of(p, aid)
        # the evidence is verified REGARDLESS of a marker's presence or origin: COMPLETE.json, the
        # inventory's identity, every listed file with its sha, no committed file outside the inventory
        if not vol.exists(complete_rel(root)):
            st["reasons"].append(f"{complete_rel(root)} missing")
        inv = None
        if not vol.exists(inventory_rel(root)):
            st["reasons"].append(f"{inventory_rel(root)} missing")
        else:
            inv = _read_json(vol, inventory_rel(root))
            st["reasons"].extend(_verify_inventory_files(vol, inv, spec, aid, files=files, blobs=blobs))
        doc = None
        if vol.exists(marker):
            doc = _read_json(vol, marker)
            if doc.get("attempt_id") != aid or doc.get("schema") != SCHEMA_MARKER:
                raise LadderRefusal(f"{marker}: not the accepted attempt's {SCHEMA_MARKER} marker")
            if inv is not None and (int(doc.get("n_enqueued", -1)), doc.get("item_keys_sha256")) != \
                    (int(inv.get("n", -2)), inv.get("item_keys_sha256")):
                raise LadderRefusal(f"{marker}: the marker's (n_enqueued, item_keys_sha256) disagree with the accepted "
                                    "attempt's queue_inventory.json; refusing a marker that does not describe its inventory")
        if st["reasons"]:
            st["state"] = PRODUCER_UNVERIFIED          # a marker does not close what the evidence does not support
        elif doc is not None:
            st["state"] = PRODUCER_CLOSED
            st["marker"] = doc
        else:
            st["marker"] = write_closed_marker(vol, spec, p, aid, inv, synthesised_by="reconciler")
            st["state"] = PRODUCER_CLOSED_SYNTHESISED
        st["inventory"] = inv
        producers[p.shard] = st
    closable = all(s["state"] in (PRODUCER_CLOSED, PRODUCER_CLOSED_SYNTHESISED) for s in producers.values())
    closed = read_closed(vol, spec)
    if closable:
        want = [[p.shard, acc[p.shard]] for p in spec.producers]
        if closed is None:
            entries = accepted_queue_entries(vol, accepted_kv, spec, acc, files=files, blobs=blobs)
            declared = {slot(e["arm_s"], e["item_key"]) for st_ in producers.values() for e in st_["inventory"]["entries"]}
            if set(entries) != declared:       # the projected set IS the accepted inventories' union, exactly (P2-9)
                raise LadderRefusal(f"{spec.ladder_root}: the accepted projection ({len(entries)} slots) is not the accepted "
                                    f"inventories' union ({len(declared)}): extra {sorted(set(entries) - declared)[:3]}, "
                                    f"missing {sorted(declared - set(entries))[:3]}; refusing to close")
            closed = {"schema": SCHEMA_CLOSED, "queue_root": spec.ladder_root, "key": key_str(spec.queue_key),
                      "kind": spec.kind, "cap_next": int(spec.cap_next), "producers": want,
                      "n_items": len(entries), "item_keys_sha256": item_keys_sha256(entries.keys()), "closed_utc": _utc()}
            vol.write_bytes(closed_rel(spec.ladder_root), (canonical_json(closed) + "\n").encode("utf-8"))
            vol.commit()
        elif closed.get("producers") != want:
            raise LadderRefusal(f"{closed_rel(spec.ladder_root)}: an existing CLOSED.json names producers "
                                f"{closed.get('producers')}, the accepted set is {want}; refusing")
    reclaimed: list[str] = []
    left: list[str] = []
    if claims_kv is not None and lease_state is not None:
        prefix = key_str(spec.queue_key) + _SEP
        elected = {k for k, _ in results_kv.items() if k.startswith(prefix)} if results_kv is not None else set()
        states: dict[str, str] = {}                              # one lease lookup per call, not per claim
        for k, v in list(claims_kv.items()):
            if not k.startswith(prefix):
                continue
            if k in elected:
                continue                                      # elected: the claim did its job
            call = str((v or {}).get("call_id", ""))
            if call not in states:
                states[call] = lease_state(call)
            if states[call] in TERMINAL_STATES:
                claims_kv.pop(k, None)                        # the one authorised reclaim
                reclaimed.append(k)
            else:
                left.append(k)
    return {"queue_root": spec.ladder_root, "producers": producers, "closable": closable,
            "queue_closed": closed is not None, "closed": closed, "reclaimed": reclaimed, "claims_left": left}


# --------------------------------------------------------------------------- #
# claims, results, materialised draws, duplicates                              #
# --------------------------------------------------------------------------- #


def claim_item(claims_kv: KV, spec: QueueSpec, arm_s_: str, item_key_: str, *, attempt_id: str,
               call_id: str) -> bool:
    """ONE ``skip_if_exists`` put; ``False`` = someone holds it -> skip the item. No takeover here, ever."""
    return claims_kv.put(item_scoped_key(spec.queue_key, arm_s_, item_key_),
                         {"attempt_id": str(attempt_id), "call_id": str(call_id), "ts": _utc()}, skip_if_exists=True)


def claim_is_this_calls(holder: Optional[Mapping[str, Any]], *, attempt_id: str, call_id: str) -> bool:
    """A HELD claim is ours when it names this attempt — or this Modal FunctionCall: Modal re-executes a
    lost container under the SAME call id while the re-execution gets a NEW ``attempt_id``, and the reconciler
    (correctly) never reclaims a running call's claim — so only the call binds the two executions, and the replacement
    must be able to draw what its predecessor claimed and never wrote. A claim of a DIFFERENT call is another live
    consumer's: held. An empty ``call_id`` (a smoke without Modal) never matches."""
    if not isinstance(holder, Mapping):
        return False
    if holder.get("attempt_id") == str(attempt_id):
        return True
    return bool(call_id) and holder.get("call_id") == str(call_id)


def record_text(record: Mapping[str, Any]) -> str:
    """The generated text of a materialised ``record`` in either leg row shape — ``text`` (``turn_record``: static,
    controls) or ``response`` (the misalign runner's row) — the ONE reading whose sha is ``text_sha256``."""
    text = record.get("text")
    if text is None:
        text = record.get("response")
    if not isinstance(text, str):
        raise LadderRefusal("the record carries no text")
    return text


def write_materialised(vol: VolumeFS, spec: QueueSpec, *, queue_record: Mapping[str, Any], attempt_id: str,
                       call_id: str, record: Mapping[str, Any], request_id: Optional[str], completion_id: Optional[str],
                       response: Any = None, tentative: bool = False, text: Optional[str] = None,
                       shard: Optional[str] = None, attempt_root: Optional[str] = None) -> tuple[str, dict]:
    """The attempt-unique draw file ``materialised/<arm>__s<s>/<item_key>/<attempt_id>.json`` (full rung row
    schema per leg in ``record`` + ``request_id`` + ``completion_id`` + the claimant + ``draw_utc`` + the queue
    record verbatim + the raw ``response``); returns ``(path, document)``. Committed by the caller's sync.
    ``tentative`` (a speculative draw) writes ``speculative/…`` instead, marked ``"speculative": true`` — a file the
    election never names until :func:`promote_materialised` moves it (P1-5)."""
    _validate_record(queue_record, spec, where="materialised: queue_record")
    text = record_text(record) if text is None else text
    doc = {"schema": SCHEMA_MATERIALISED, "key": item_scoped_key(spec.queue_key, queue_record["arm_s"], queue_record["item_key"]),
           "arm_s": queue_record["arm_s"], "item_key": queue_record["item_key"], "cap": int(spec.cap_next),
           "claimant": {"attempt_id": str(attempt_id), "call_id": str(call_id),
                        # the claimant's SLOT and ROOT: a later execution of this slot adopts this draw's request
                        # pipeline from that root — absent on a smoke that names neither
                        **({"shard": str(shard)} if shard is not None else {}),
                        **({"attempt_root": str(attempt_root)} if attempt_root is not None else {})}, "draw_utc": _utc(),
           "request_id": request_id, "completion_id": completion_id,
           "producer_attempt_id": queue_record["producer"]["attempt_id"], "queue_record": dict(queue_record),
           "record": dict(record), "text_sha256": sha256_hex(text.encode("utf-8")), "response": response}
    if tentative:
        doc["speculative"] = True
        rel = speculative_rel(spec.ladder_root, queue_record["arm_s"], queue_record["item_key"], attempt_id)
    else:
        rel = materialised_rel(spec.ladder_root, queue_record["arm_s"], queue_record["item_key"], attempt_id)
    vol.write_bytes(rel, (canonical_json(doc) + "\n").encode("utf-8"), exclusive=True)
    return rel, doc


def promote_materialised(vol: VolumeFS, spec: QueueSpec, doc: Mapping[str, Any], speculative_path: str) -> tuple[str, dict]:
    """A tentative draw whose producer attempt has been ACCEPTED becomes the attempt's canonical materialised file:
    the same document (draw time, request / completion ids, record, text sha) written at :func:`materialised_rel`
    with ``promoted_from`` naming the tentative file; the election may name it from here on (P1-5). The tentative
    file stays as evidence."""
    if not doc.get("speculative"):
        raise LadderRefusal(f"{speculative_path}: not a speculative document; nothing to promote")
    out = {k: v for k, v in doc.items() if k != "speculative"}
    out["promoted_from"] = str(speculative_path)
    out["promoted_utc"] = _utc()
    rel = materialised_rel(spec.ladder_root, doc["arm_s"], doc["item_key"], doc["claimant"]["attempt_id"])
    vol.write_bytes(rel, (canonical_json(out) + "\n").encode("utf-8"), exclusive=True)
    return rel, out


def elect_result(results_kv: KV, spec: QueueSpec, doc: Mapping[str, Any], path: str) -> bool:
    """ONE ``results.put((…key…, item_key), {attempt_id, path, text_sha256, request_id, completion_id},
    skip_if_exists=True)``: ``True`` = THIS draw is the result; ``False`` = a loser."""
    cl = doc["claimant"]
    return results_kv.put(doc["key"], {"attempt_id": cl["attempt_id"], "path": path,
                                       "text_sha256": doc["text_sha256"], "request_id": doc.get("request_id"),
                                       "completion_id": doc.get("completion_id"),
                                       # the claimant's call / slot / root ride on the elected value too (additive): the adopter
                                       # decides "mine to adopt" from the Dict scan, before opening the file
                                       **{k: cl[k] for k in ("call_id", "shard", "attempt_root") if cl.get(k) is not None}},
                          skip_if_exists=True)


def record_duplicate(vol: VolumeFS, spec: QueueSpec, attempt_id: str, doc: Mapping[str, Any], path: str,
                     winner: Mapping[str, Any]) -> str:
    """A loser appends one line to its OWN ``duplicates/<attempt_id>.jsonl`` (per attempt, never a shared file)."""
    rel = duplicates_rel(spec.ladder_root, attempt_id)
    line = canonical_json({"item_key": doc["item_key"], "arm_s": doc["arm_s"], "path": path,
                           "text_sha256": doc["text_sha256"], "winner": dict(winner), "ts": _utc()}) + "\n"
    old = vol.read_bytes(rel) if vol.exists(rel) else b""
    vol.write_bytes(rel, old + line.encode("utf-8"))
    return rel


# --------------------------------------------------------------------------- #
# the assembly's readers                                                       #
# --------------------------------------------------------------------------- #


class ElectedMismatch(LadderRefusal):
    """An elected row that does not match the accepted base row: REFUSED (the assembly re-draws it)."""


@dataclasses.dataclass(frozen=True)
class Elected:
    item_key: str
    arm_s: str
    value: dict          # the Dict's elected value
    doc: dict            # the materialised file
    path: str


def _read_elected(vol: VolumeFS, spec: QueueSpec, arm_s_: str, ik: str, v: Mapping[str, Any]) -> Elected:
    """ONE elected value verified against its file: the file exists at the elected path, is this item's, was written
    by the elected attempt, and its ``text_sha256`` equals both the Dict value's and the sha of the record's text;
    drawn at this queue's cap. Anything else refuses: a torn or forged file is never a result."""
    k = item_scoped_key(spec.queue_key, arm_s_, ik)
    path = str((v or {}).get("path", ""))
    if not path.startswith(f"{spec.ladder_root}/materialised/{arm_s_}/{ik}/") or not vol.exists(path):
        raise LadderRefusal(f"elected result {k!r} names path {path!r}, which is not this item's materialised "
                            "file or is absent on the volume")
    doc = _read_json(vol, path)
    if doc.get("schema") != SCHEMA_MATERIALISED or doc.get("item_key") != ik or doc.get("arm_s") != arm_s_:
        raise LadderRefusal(f"{path}: not this item's {SCHEMA_MATERIALISED} document")
    if doc.get("claimant", {}).get("attempt_id") != v.get("attempt_id") or \
            not path.endswith(f"/{v.get('attempt_id')}.json"):
        raise LadderRefusal(f"{path}: written by {doc.get('claimant')}, the Dict elected {v.get('attempt_id')!r}")
    try:
        text = record_text(doc.get("record") or {})
    except LadderRefusal:
        raise LadderRefusal(f"{path}: the record carries no text") from None
    sha = sha256_hex(text.encode("utf-8"))
    if sha != doc.get("text_sha256") or sha != v.get("text_sha256"):
        raise LadderRefusal(f"{path}: text_sha256 disagrees (file {doc.get('text_sha256')!r}, Dict "
                            f"{v.get('text_sha256')!r}, text {sha!r})")
    if int(doc.get("cap", -1)) != int(spec.cap_next):
        raise LadderRefusal(f"{path}: drawn at cap {doc.get('cap')!r}, this queue's rung is {spec.cap_next}")
    return Elected(ik, arm_s_, dict(v), doc, path)


def load_elected_one(vol: VolumeFS, results_kv: Optional[KV], spec: QueueSpec, arm_s_: str, item_key_: str, *,
                     value: Optional[Mapping[str, Any]] = None) -> Optional[Elected]:
    """The verified elected result of ONE item (:func:`_read_elected` rules), or ``None`` when none is elected. A
    caller that already scanned the Dict passes the ``value`` in hand (no second round trip)."""
    v = value if value is not None else results_kv.get(item_scoped_key(spec.queue_key, arm_s_, item_key_))
    return None if v is None else _read_elected(vol, spec, arm_s_, item_key_, v)


def load_elected(vol: VolumeFS, results_kv: KV, spec: QueueSpec, arm_s_: str) -> dict[str, Elected]:
    """``{item_key: Elected}`` for one ``<arm>__s<s>`` of the queue: the winner from the Dict (or a snapshot), each
    verified by :func:`_read_elected`."""
    prefix = key_str(spec.queue_key, arm_s_) + _SEP
    out: dict[str, Elected] = {}
    for k, v in results_kv.items():
        if not k.startswith(prefix):
            continue
        _, extra = parse_key(k)
        if len(extra) != 2 or extra[0] != arm_s_:
            raise LadderRefusal(f"results key {k!r} is not (queue key, arm_s, item_key)")
        out[extra[1]] = _read_elected(vol, spec, arm_s_, extra[1], v)
    return out


def verify_elected(el: Elected, *, base_row_sha256: str, sampling_seed: Optional[int], cap: int,
                   preset: Optional[Mapping[str, Any]], accepted_producers: Iterable[str]) -> None:
    """The assembly's per-row rule: the elected row's ``(sampling_seed, cap, preset, base_row_sha256,
    producer attempt)`` must match the ACCEPTED base row, else :class:`ElectedMismatch` (refused and re-drawn)."""
    q = el.doc["queue_record"]
    if q.get("base_row_sha256") != base_row_sha256:
        raise ElectedMismatch(f"{el.item_key}: queued against base row {q.get('base_row_sha256')!r}, the accepted "
                              f"base row is {base_row_sha256!r}")
    if q.get("producer", {}).get("attempt_id") not in set(accepted_producers):
        raise ElectedMismatch(f"{el.item_key}: queued by producer attempt {q.get('producer')!r}, which is not an "
                              "accepted producer of this queue")
    if sampling_seed is not None and int(q.get("sampling_seed", -1)) != int(sampling_seed):
        raise ElectedMismatch(f"{el.item_key}: queued seed {q.get('sampling_seed')!r} != the base row's {sampling_seed}")
    rec = el.doc.get("record") or {}
    drawn_seed = rec.get("sampling_seed", rec.get("seed"))          # turn_record's key, else the misalign runner's row key
    if sampling_seed is not None and drawn_seed is not None and int(drawn_seed) != int(sampling_seed):
        raise ElectedMismatch(f"{el.item_key}: drawn with seed {drawn_seed!r} != {sampling_seed}")
    if int(el.doc.get("cap", -1)) != int(cap) or (rec.get("max_tokens") is not None and int(rec["max_tokens"]) != int(cap)):
        raise ElectedMismatch(f"{el.item_key}: drawn at cap {el.doc.get('cap')!r}/{rec.get('max_tokens')!r}, not {cap}")
    if preset is not None:
        sent = (q.get("redraw") or {}).get("sampling")
        if not isinstance(sent, Mapping) or dict(sent) != dict(preset):     # absent = the server's defaults: a mismatch (P2-3)
            raise ElectedMismatch(f"{el.item_key}: drawn under sampling {sent!r} (absent = server defaults), this rung's preset "
                                  f"is {dict(preset)!r}")


def accepted_attempt_root_or_refuse(accepted_kv: KV, key: SixAxisKey) -> str:
    """The pull's rule: the ACCEPTED attempt's root for a shard, or a refusal — a shard with no
    accepted attempt is never pulled."""
    aid = accepted_attempt(accepted_kv, key)
    if aid is None:
        raise LadderRefusal(f"{key_str(key)}: no accepted attempt; refusing to pull an unaccepted shard")
    return attempt_root_for(key.W7, leg=key.leg, view=key.view, stage=key.stage, shard=key.shard, attempt_id=aid)
