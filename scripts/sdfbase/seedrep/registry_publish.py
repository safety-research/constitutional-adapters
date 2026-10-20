#!/usr/bin/env python3
"""The Qwen seed-replicate native battery: publish ACCEPTED attempts' registry spools to the grading host inbox.

Every wave container writes its records into an ATTEMPT-UNIQUE spool (``SISTEER_REGISTRY_SPOOL=<attempt root>/registry/records``),
never into the shared ``/vol/registry/records``. Records reach the registry ONLY through this script, on the grading host,
after the pull, and only for attempts the wave's ``accepted`` Dict names:

* an attempt publishes iff its id is in the accepted map, its own ``ATTEMPT.json`` carries no ``superseded_by``, and its
  ``COMPLETE.json`` exists — a superseded or incomplete attempt publishes NOTHING (kept as evidence, never merged);
* a run directory is published iff ``logical_run_complete.py`` ADMITTED its run AND COMMITTED the completion records
  (its verdict file's ``admission`` block): publication is bound to that admission — the run id, the wave, the COMPLETE
  set of admitted attempts (each with its six axes) and the logical sha of every record file the union check read. An
  attempt the admission does not name (a rung attempt, a same-run-id attempt of another wave) publishes nothing for that
  run; a run dir whose files differ from the admitted shas, or an admitted attempt missing from the pulled tree, REFUSES
  the run outright — the tree changed after the verdict, and nothing of that run is written;
* files are RE-SEQUENCED per ACCEPTED ATTEMPT: ``<seq>-<type>.jsonl[.gz]`` → ``<block·1,000,000 + seq>-<type>.jsonl[.gz]``,
  where ``block`` is a pure function of the attempt id (:func:`attempt_block`: the first 40 bits of its sha256). An attempt
  id is one execution (``MODAL_TASK_ID`` + uuid4), so a base shard and a rung shard of one run, two shards, and
  the attempts of a relaunched wave never share a block, a retry derives the SAME names with no state to persist or race
  on, and none collides with the grading host's own ``RecordWriter`` sequence (< 1e6) in that run directory. The
  filename keeps the ``<digits>-<type>`` grammar the harvester classifies by (``records.RECORD_FILE_RE``); the
  block → attempt/axes provenance rides the publish ledger. Two planned files on one name, or a same-named file with
  DIFFERENT bytes, are refused loudly — never a silent overwrite.

Destination: ``<inbox>/<host>/<campaign>/<run_dir>/<file>`` (``~/registry/inbox/`` by default, host ``seedrep-modal``), the
layout ``registry inbox-merge`` / every harvester cycle moves into ``records/`` (``scripts/registry/inbox_to_log.sh``). The
ledger of what was published lives under ``<inbox>/<host>/_publish_ledger/`` — an underscore-prefixed directory the merge
never touches.

  registry_publish.py --accepted RS/accepted.json --attempts RS/stores [RS/stores2 …] --union-dir RS/registry/union \\
                      [--inbox ~/registry/inbox] [--host seedrep-modal] [--dry-run]

THE SIX AXES ``(W, wave_uuid, container, view, stage, shard)`` — ``container`` is the launch plan's standup leg (the eval leg
for a single-leg container, ``controls`` for the combined xstest + or_bench + mmlu container), ``view`` its engine
(``s1|s2|s3|arditi``) — are read by ONE parser (:func:`parse_axes`) from every place they travel: the launch plan's
``control_key`` (``W/wave_uuid/leg/engine/stage/shard``, which is also the accepted Dict's key), a
``|``-joined key, or a JSON list. ``ATTEMPT.json`` carries them as ``control_key``, or spelled out at its top level
(``leg`` / ``view`` / ``stage`` / ``shard`` / ``W`` / ``wave_uuid``). The accepted map (the orchestrator's export of
the Dict) is ``{"<six-axis key>": "<attempt_id>", …}``: the value binds (an attempt is accepted iff its id is a value), and
every axis the key carries is cross-checked against the attempt's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path and not any(p.endswith("/src") and "sisteer" in os.listdir(p) for p in sys.path if os.path.isdir(p)):
    sys.path.insert(0, str(REPO / "src"))

from sisteer.registry import records as R  # noqa: E402

ATTEMPT_FILE = "ATTEMPT.json"
COMPLETE_FILE = "COMPLETE.json"
SPOOL_REL = os.path.join("registry", "records")          # <attempt root>/registry/records — the SISTEER_REGISTRY_SPOOL the container exports
ITEMS_REL = os.path.join("registry", "items")            # <attempt root>/registry/items/<run_dir>.json — the shard's completed item ids per run
DEFAULT_INBOX = os.path.expanduser("~/registry/inbox")
DEFAULT_HOST = "seedrep-modal"
LEDGER_DIR = "_publish_ledger"
ATTEMPT_BLOCK = 1_000_000
BLOCK_HEX = 10                                            # 40 bits of the attempt id's sha256 → the block
VERDICT_ADMITTED = "ADMITTED"
COMPLETION_COMMITTED = "committed"
COMPLETION_PENDING = "pending"
BASE_STAGE = "base"
AXES: Tuple[str, ...] = ("w7", "wave_uuid", "container", "view", "stage", "shard")
W7_RE = re.compile(r"[0-9a-f]{7,40}")       # a control key's first axis: a 7-character tree sha (or a full sha) — used with fullmatch ONLY: `$` would
                                              # match before a trailing newline, so no anchors and no .match()


class PublishRefused(ValueError):
    """A publication that would violate the accepted-only / admission-bound / collision-free contract: nothing else is written."""


# --------------------------------------------------------------------------------------------
# the six axes
# --------------------------------------------------------------------------------------------


SHARD_RE = re.compile(r"(0|[1-9][0-9]*)")       # a shard spelled as text: canonical ASCII decimal — no sign, padding, whitespace, dot or trailing
                                                 # newline; used with fullmatch ONLY (`^...$` let "1\n" through: `$` matches before a final newline)


def parse_shard(v) -> Optional[int]:
    """The shard axis WITHOUT coercion: a JSON integer that is not a boolean and is non-negative, or a string that is a
    canonical decimal (``SHARD_RE``); anything else — ``1.5``, ``true``, ``-0.5``, ``"01"``, ``" 1"``, ``"+1"`` — is None (unproven), never
    rounded, truncated or stripped into a shard the attempt did not state."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    if isinstance(v, str) and SHARD_RE.fullmatch(v):
        return int(v)
    return None


def _spelled(meta: Mapping[str, object], key: str, p: Path, *, hex7: bool = False) -> Optional[str]:
    """One spelled-out string axis of a legacy ATTEMPT.json: absent -> None (unproven, refused later by the exact rule); PRESENT but not a
    non-empty string (or, for ``W``, not 7..40 hex) -> refused here, like a malformed control_key — never ``str()``-coerced."""
    v = meta.get(key)
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        raise PublishRefused(f"{p}: {key} is {v!r} ({type(v).__name__}), not a non-empty string; a stated axis is read exactly, never coerced")
    if v != v.strip():
        raise PublishRefused(f"{p}: {key} is {v!r}, carrying surrounding whitespace; a stated axis is read exactly, never coerced or trimmed")
    if hex7 and not W7_RE.fullmatch(v):
        raise PublishRefused(f"{p}: W {v!r} is not 7..40 hex characters")
    return v[:7] if hex7 else v

def parse_axes(key) -> Dict[str, object]:
    """The six axes of a control key given as ``W7/wave_uuid/leg/engine/stage/shard`` (the launch plan's ``control_key``), a
    ``|``-joined string, or a JSON list. REFUSES (:class:`PublishRefused`) every other shape — not six parts, an empty axis, a W7
    that is not 7..40 hex characters, a shard that is not a non-negative integer: a key that does not parse never becomes an
    unconstrained acceptance. Every member is read in its STATED type: the first five axes
    must be non-empty strings without surrounding whitespace — a JSON-list ``1234567`` is not the W7 ``"1234567"`` — and only
    the sixth goes through :func:`parse_shard`; the key itself must be a string or a list, never another type."""
    parts: Optional[list] = None
    if isinstance(key, (list, tuple)):
        parts = list(key)                                     # members in their JSON types, never str()-coerced
    elif isinstance(key, str):
        if key != key.strip():                                # never normalised: a key ending in "1 " or "1\n" is not the key for shard 1
            raise PublishRefused(f"control key {key!r}: carries surrounding whitespace; a key is read exactly, never trimmed")
        if key.startswith("["):
            try:
                v = json.loads(key)
            except ValueError:
                v = None
            parts = list(v) if isinstance(v, list) else None
        elif "/" in key:
            parts = key.split("/")
        elif "|" in key:
            parts = key.split("|")
    else:
        raise PublishRefused(f"control key {key!r} ({type(key).__name__}): a key is a string or a JSON list, never another type")
    if parts is None or len(parts) != 6:
        raise PublishRefused(f"control key {key!r}: not six axes W7/wave_uuid/container/view/stage/shard "
                             f"({'no axis separator' if parts is None else f'{len(parts)} parts'})")
    for name, part in zip(("W7", "wave_uuid", "container", "view", "stage"), parts[:5]):
        if not isinstance(part, str):
            raise PublishRefused(f"control key {key!r}: {name} is {part!r} ({type(part).__name__}), not a string; an axis is read in its stated "
                                 "type, never coerced")
        if not part.strip():
            raise PublishRefused(f"control key {key!r}: empty {name}")
        if part != part.strip():
            raise PublishRefused(f"control key {key!r}: {name} {part!r} carries surrounding whitespace; an axis is read exactly, never trimmed")
    if not W7_RE.fullmatch(parts[0]):
        raise PublishRefused(f"control key {key!r}: W7 {parts[0]!r} is not 7..40 hex characters")
    shard = parse_shard(parts[5])
    if shard is None:
        raise PublishRefused(f"control key {key!r}: shard {parts[5]!r} ({type(parts[5]).__name__}) is not a non-negative integer in canonical decimal")
    return {"w7": parts[0][:7], "wave_uuid": parts[1], "container": parts[2], "view": parts[3], "stage": parts[4], "shard": shard}


def axis_mismatch(want: Mapping[str, object], have: Mapping[str, object]) -> Optional[Tuple[str, object, object]]:
    """The first axis the two sides do not agree on EXACTLY — same type, same value — as ``(axis, want, have)``; None when all six
    agree. An axis absent on EITHER side is a mismatch (reported with the None), never a wildcard; the comparison is typed,
    never ``str()``-flattened (both sides are this module's own normalised dicts, so a type difference is a defect)."""
    for ax in AXES:
        w, h = want.get(ax), have.get(ax)
        if w is None or h is None or type(w) is not type(h) or w != h:
            return ax, w, h
    return None


def absent_or_invalid_axes(axes: Mapping[str, object]) -> List[str]:
    """The axes an attempt does NOT prove, in ``AXES`` order: absent (None); a string axis that is not a non-empty ``str`` without
    surrounding whitespace; a ``w7`` that is not 7..40 hex; a ``shard`` that is not a non-negative, non-boolean ``int``. Non-empty means
    the attempt can neither be accepted nor complete a run (fail closed, never compare-where-present; never coerced)."""
    def proven(ax: str) -> bool:
        v = axes.get(ax)
        if v is None:
            return False
        if ax == "shard":
            return isinstance(v, int) and not isinstance(v, bool) and v >= 0
        if not isinstance(v, str) or not v.strip() or v != v.strip():
            return False
        return bool(W7_RE.fullmatch(v)) if ax == "w7" else True
    return [ax for ax in AXES if not proven(ax)]


@dataclass
class AttemptRoot:
    root: Path
    attempt_id: str
    superseded_by: Optional[str]
    axes: Dict[str, object]                    # the six AXES (None where ATTEMPT.json states none)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def container(self):
        return self.axes.get("container")

    leg = container                            # the standup leg IS the container axis

    @property
    def view(self):
        return self.axes.get("view")

    @property
    def stage(self):
        return self.axes.get("stage")

    @property
    def shard(self):
        return self.axes.get("shard")

    @property
    def w7(self):
        return self.axes.get("w7")

    @property
    def wave_uuid(self):
        return self.axes.get("wave_uuid")

    @property
    def spool(self) -> Path:
        return self.root / SPOOL_REL

    @property
    def items_dir(self) -> Path:
        return self.root / ITEMS_REL

    @property
    def complete(self) -> bool:
        return (self.root / COMPLETE_FILE).is_file()

    def run_dirs(self) -> List[Tuple[str, str, Path]]:
        """``[(campaign, run_dir_name, path)]`` of every run directory this attempt's spool holds."""
        out: List[Tuple[str, str, Path]] = []
        if not self.spool.is_dir():
            return out
        for camp in sorted(p for p in self.spool.iterdir() if p.is_dir()):
            for rd in sorted(p for p in camp.iterdir() if p.is_dir()):
                out.append((camp.name, rd.name, rd))
        return out


def read_attempt(root) -> AttemptRoot:
    """``ATTEMPT.json`` of one attempt root: ``attempt_id`` required; the six axes from ``control_key`` (the launch
    plan's grammar) else the spelled-out top-level fields; ``superseded_by`` as the attempt marked itself."""
    root = Path(root)
    p = root / ATTEMPT_FILE
    if not p.is_file():
        raise PublishRefused(f"{root}: no {ATTEMPT_FILE}; not an attempt root")
    with open(p, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict) or meta.get("attempt_id") is None:
        raise PublishRefused(f"{p}: no attempt_id")
    aid = meta["attempt_id"]
    if not isinstance(aid, str) or not aid.strip() or aid != aid.strip():
        raise PublishRefused(f"{p}: attempt_id is {aid!r} ({type(aid).__name__}), not a non-empty string without surrounding whitespace; "
                             "read in its stated type, never coerced")
    if meta.get("control_key") is not None:
        try:
            axes = parse_axes(meta["control_key"])                  # the control_key grammar; a malformed key REFUSES the root, it is never read past
        except PublishRefused as exc:
            raise PublishRefused(f"{p}: {exc}; a malformed control_key is refused, not fallen back from (review r2 finding 1)") from None
    else:
        # the spelled-out legacy shape: every stated field read EXACTLY (`int(1.5)`, `int(True)`, `str(True)` used to coerce a
        # value the attempt never stated into an axis that then matched an accepted key); an absent field is None, refused by the exact rule
        shard = meta.get("shard")
        if shard is not None and parse_shard(shard) is None:
            raise PublishRefused(f"{p}: shard is {shard!r} ({type(shard).__name__}), not a non-negative integer (a JSON integer that is not a "
                                 "boolean, or a canonical decimal string); a stated axis is read exactly, never coerced")
        axes = {"w7": _spelled(meta, "W", p, hex7=True), "wave_uuid": _spelled(meta, "wave_uuid", p), "container": _spelled(meta, "leg", p),
                "view": _spelled(meta, "view", p), "stage": _spelled(meta, "stage", p), "shard": parse_shard(shard) if shard is not None else None}
    sup = meta.get("superseded_by")
    if sup is not None and (not isinstance(sup, str) or not sup.strip()):
        raise PublishRefused(f"{p}: superseded_by is {sup!r} ({type(sup).__name__}); null or the holder's attempt id (a non-empty string), never coerced")
    return AttemptRoot(root, aid, sup, axes, dict(meta))


def discover_attempts(roots: Iterable, max_depth: int = 6) -> List[AttemptRoot]:
    """Every attempt root (a directory holding ``ATTEMPT.json``) under the given roots, walked to a bounded depth — the pulled
    layout ``<leg>/<view>/<shard>/attempts/<attempt_id>/`` and a flattened ``<leg>/<view>/<shard>/`` alike."""
    out: List[AttemptRoot] = []
    seen = set()
    for r in roots:
        r = Path(r)
        if not r.is_dir():
            raise PublishRefused(f"attempts root {r} is not a directory")
        base_depth = len(r.parts)
        for dirpath, dirnames, filenames in os.walk(r):
            if len(Path(dirpath).parts) - base_depth > max_depth:
                dirnames[:] = []
                continue
            if ATTEMPT_FILE in filenames:
                dirnames[:] = []   # an attempt root's subtree is its own; never descend into its stores
                if dirpath not in seen:
                    seen.add(dirpath)
                    out.append(read_attempt(dirpath))
            else:
                dirnames.sort()
    out.sort(key=lambda a: (a.container or "", a.view or "", a.stage or "", a.shard if a.shard is not None else -1, a.attempt_id))
    return out


@dataclass
class Accepted:
    by_id: Dict[str, Dict[str, object]]   # attempt_id -> the key's six axes (every key parsed and validated at load, never None)
    raw: Dict[str, str]

    @property
    def attempt_ids(self) -> set:
        return set(self.by_id)

    def is_accepted(self, a: AttemptRoot) -> Tuple[bool, str]:
        """(accepted?, reason). Accepted iff the id is a value of the map AND the attempt proves all six axes AND each equals its
        key's — an exact match, never compare-where-present (an ATTEMPT.json omitting ``stage`` or ``shard`` is refused by
        name, whatever key names its id)."""
        if a.attempt_id not in self.by_id:
            return False, "not_accepted"
        lacking = absent_or_invalid_axes(a.axes)
        if lacking:
            return False, f"ATTEMPT.json states no {lacking}; an attempt is accepted only on an exact six-axis match with its key"
        hit = axis_mismatch(self.by_id[a.attempt_id], a.axes)
        if hit:
            return False, f"accepted_under_other_{hit[0]}:{hit[1]}"
        return True, "accepted"


def load_accepted(path) -> Accepted:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise PublishRefused(f"{path}: the accepted map must be a JSON object {{key: attempt_id}}")
    by_id: Dict[str, Dict[str, object]] = {}
    for key, aid in raw.items():
        if not isinstance(aid, str) or not aid.strip() or aid != aid.strip():
            raise PublishRefused(f"{path}: key {key!r} names no attempt id ({aid!r}: a non-empty string without surrounding whitespace, never coerced)")
        if aid in by_id:
            raise PublishRefused(f"{path}: attempt {aid!r} is accepted under two keys; an attempt holds exactly one (container, view, stage, shard)")
        try:
            by_id[aid] = parse_axes(key)
        except PublishRefused as exc:
            raise PublishRefused(f"{path}: {exc}; every accepted-map key must carry the six axes — an attempt is never accepted by its id alone "
                                 "(review r2 finding 1)") from None
    return Accepted(by_id, dict(raw))            # JSON object keys are strings by construction; every value was checked above


# --------------------------------------------------------------------------------------------
# names: one block per accepted attempt, a pure function of its id
# --------------------------------------------------------------------------------------------


def attempt_block(attempt_id: str) -> int:
    """The attempt's block: ``1 + sha256(attempt_id)[:40 bits]`` — stable across retries with nothing persisted, distinct per
    execution (an attempt id is one execution), never 0."""
    return 1 + int(hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:BLOCK_HEX], 16)


def block_seq(block: int, seq: int) -> int:
    """The published sequence number of an attempt's ``seq``-th record file: unique across attempts, above any grading-host-local seq."""
    if block < 1 or seq < 1 or seq >= ATTEMPT_BLOCK:
        raise PublishRefused(f"cannot re-sequence block {block} seq {seq}: block >= 1 and 1 <= seq < {ATTEMPT_BLOCK} required")
    return block * ATTEMPT_BLOCK + seq


def published_name(fn: str, block: int) -> str:
    m = R.RECORD_FILE_RE.match(fn)
    if not m:
        raise PublishRefused(f"{fn!r} is not a record file name (<seq>-<type>.jsonl[.gz])")
    return R.record_file_name(block_seq(block, int(m.group("seq"))), m.group("type"), gz=bool(m.group("gz")))


# --------------------------------------------------------------------------------------------
# the admission (logical_run_complete.py's committed verdict)
# --------------------------------------------------------------------------------------------


def record_run_id(rec: Mapping[str, object]) -> Optional[str]:
    """The run a record is bound to, by ITS OWN type's field: ``payload.run_id`` for every record type whose schema
    requires it (``records.REQUIRED_FIELDS`` — run, run_state, run_patch, item_result, run_degeneration, judgment, …); ``payload.extra.run_id``
    for ``artifact`` (the store recorder's completion / rows artifacts, which the artifact table binds that way); None for a record type
    with no run binding at all (object, model, eval, …) — such a record never belongs in a run directory."""
    payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
    rt = rec.get("record_type")                              # read in its stated type: a non-string type binds nothing
    if isinstance(rt, str) and "run_id" in R.REQUIRED_FIELDS.get(rt, ()):
        rid = payload.get("run_id")
    elif rt == "artifact":
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        rid = extra.get("run_id")
    else:
        rid = None
    return rid if isinstance(rid, str) and rid else None


def run_dir_files(rd: Path, run_id: Optional[str] = None) -> Dict[str, str]:
    """``{record file name: logical sha}`` of a run directory — the source contents an admission binds. With ``run_id`` EVERY record
    of every file must bind to ``run_id`` through its type's own field (:func:`record_run_id`): a mixed-run, run-less or run-unbound
    file REFUSES the directory (the old check read the FIRST record only), and the sha comes from the same pass."""
    out: Dict[str, str] = {}
    for p in R.iter_record_files(str(rd)):
        rel = os.path.relpath(p, str(rd))
        if run_id is None:
            out[rel] = R.file_logical_sha(p)
            continue
        h = hashlib.sha256()
        for i, rec, _sha in R.read_record_file(p, hasher=h):
            rid = record_run_id(rec)
            if rid != run_id:
                raise PublishRefused(f"{p}: record {i} ({rec.get('record_type')}) names run {rid!r}, the admission is for {run_id}; a record file "
                                     "with a missing, foreign or unbound run is never published")
        out[rel] = h.hexdigest()
    return out

def read_verdict(union_dir, run_dir_name: str) -> Optional[Dict[str, object]]:
    """``logical_run_complete.py``'s verdict file for a run dir, None when absent."""
    p = Path(union_dir) / f"{run_dir_name}.json"
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        v = json.load(fh)
    return v if isinstance(v, dict) else None


def admission_for(union_dir, run_dir_name: str) -> Tuple[Optional[Dict[str, object]], str]:
    """``(admission block, reason)``: the block iff the verdict reads ADMITTED and its completion is COMMITTED (the completion
    records exist on the grading host spool); otherwise None and why (``absent`` / ``REFUSED`` / ``ADMITTED but completion pending``)."""
    v = read_verdict(union_dir, run_dir_name)
    if v is None:
        return None, "absent"
    verdict = v.get("verdict")
    if verdict != VERDICT_ADMITTED:
        return None, verdict if isinstance(verdict, str) else repr(verdict)
    comp = v.get("completion") if isinstance(v.get("completion"), dict) else {}
    adm = v.get("admission") if isinstance(v.get("admission"), dict) else None
    if adm is None or not isinstance(adm.get("attempts"), dict):
        return None, f"{VERDICT_ADMITTED} but no admission block (a pre-admission verdict file; re-run logical_run_complete.py)"
    if comp.get("state") != COMPLETION_COMMITTED:
        return None, f"{VERDICT_ADMITTED} but completion {comp.get('state') or 'absent'} (the completion records are not on the spool yet)"
    return adm, VERDICT_ADMITTED


def check_admitted_source(adm: Mapping[str, object], a: AttemptRoot, rd: Path) -> Dict[str, str]:
    """The attempt's run dir is EXACTLY what the admission bound: the entry's six axes, the run id on EVERY record, the same files
    with the same logical shas — anything else means the pulled tree changed after the verdict, and the run is refused whole.
    Returns the verified ``{file: logical sha}`` so the copy needs no second hash pass."""
    entry = adm["attempts"][a.attempt_id]   # type: ignore[index]
    hit = axis_mismatch(entry, a.axes)
    if hit:
        raise PublishRefused(f"{rd}: attempt {a.attempt_id} is admitted under {hit[0]}={hit[1]!r} but its ATTEMPT.json says {hit[2]!r}")
    rid = adm.get("run_id")
    if not isinstance(rid, str) or not rid:
        raise PublishRefused(f"{rd}: the admission names run {rid!r}, not a run id")
    have = run_dir_files(rd, rid)                              # every record of every file names the admitted run, or the run refuses
    want = dict(entry.get("files") or {})
    if have != want:
        extra = sorted(set(have) - set(want))
        missing = sorted(set(want) - set(have))
        changed = sorted(f for f in set(have) & set(want) if have[f] != want[f])
        raise PublishRefused(f"{rd}: the record files differ from the admitted set (outside the admission: {extra}; admitted but absent: "
                             f"{missing}; bytes changed: {changed}); the tree changed after the union verdict — re-run "
                             "logical_run_complete.py, never publish past the verdict")
    return have


# --------------------------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------------------------


@dataclass
class PublishReport:
    published: List[Dict[str, object]] = field(default_factory=list)      # {attempt_id, block, <axes>, campaign, run_dir, src, dest, sha}
    duplicates: List[Dict[str, object]] = field(default_factory=list)
    skipped_attempts: List[Dict[str, object]] = field(default_factory=list)   # {attempt_id, reason}
    skipped_run_dirs: List[Dict[str, object]] = field(default_factory=list)   # {attempt_id, run_dir, reason}
    blocks: Dict[str, int] = field(default_factory=dict)                     # attempt_id -> block

    def as_dict(self) -> Dict[str, object]:
        return {"n_published": len(self.published), "n_duplicates": len(self.duplicates), "skipped_attempts": self.skipped_attempts,
                "skipped_run_dirs": self.skipped_run_dirs, "published": self.published, "duplicates": self.duplicates, "blocks": self.blocks}

    def published_attempt_ids(self) -> set:
        return {p["attempt_id"] for p in self.published}


def publishable(attempts: Sequence[AttemptRoot], accepted: Accepted, rep: PublishReport) -> List[AttemptRoot]:
    """The attempts that may publish at all (the accepted-only rule); the others land in ``rep.skipped_attempts`` with the reason."""
    out: List[AttemptRoot] = []
    for a in attempts:
        # the attempt's OWN marker first: an attempt that read the holder and marked itself superseded publishes nothing whatever
        # any map says (a contradiction between the two would be a control-plane bug, and the marker is the more specific reason)
        if a.superseded_by:
            rep.skipped_attempts.append({"attempt_id": a.attempt_id, "reason": f"superseded_by:{a.superseded_by}", "root": str(a.root)})
            continue
        ok, why = accepted.is_accepted(a)
        if not ok:
            rep.skipped_attempts.append({"attempt_id": a.attempt_id, "reason": why, "root": str(a.root)})
            continue
        if not a.complete:
            rep.skipped_attempts.append({"attempt_id": a.attempt_id, "reason": "incomplete:no COMPLETE.json", "root": str(a.root)})
            continue
        out.append(a)
    return out


def admitted_sources(holders: Mapping[Tuple[str, str], Sequence[Tuple[AttemptRoot, Path]]], union_dir, rep: PublishReport
                     ) -> List[Tuple[AttemptRoot, str, str, Path, str]]:
    """``[(attempt, campaign, run_dir, src, logical sha)]`` of every file whose run dir is admitted, COMPLETE, and verified; the
    rest of ``holders`` lands in ``rep.skipped_run_dirs`` with the reason."""
    out: List[Tuple[AttemptRoot, str, str, Path, str]] = []
    for (camp, rd_name), held in sorted(holders.items()):
        adm, why = admission_for(union_dir, rd_name)
        if adm is None:
            for a, _rd in held:
                rep.skipped_run_dirs.append({"attempt_id": a.attempt_id, "campaign": camp, "run_dir": rd_name, "reason": f"union_verdict:{why}"})
            continue
        adm_ids = set(adm["attempts"])   # type: ignore[arg-type]
        in_set = [(a, rd) for a, rd in held if a.attempt_id in adm_ids]
        for a, _rd in held:
            if a.attempt_id not in adm_ids:
                rep.skipped_run_dirs.append({"attempt_id": a.attempt_id, "campaign": camp, "run_dir": rd_name,
                                             "reason": f"attempt_not_admitted:the admission names {sorted(adm_ids)}; this attempt is stage {a.stage!r} "
                                                       f"of wave {a.wave_uuid!r}"})
        missing = sorted(adm_ids - {a.attempt_id for a, _ in in_set})
        if missing:
            # the admission bound the COMPLETE shard set; a shard absent from the pulled tree (or no longer publishable) refuses the run
            for a, _rd in in_set:
                rep.skipped_run_dirs.append({"attempt_id": a.attempt_id, "campaign": camp, "run_dir": rd_name,
                                             "reason": f"admitted_shard_missing:{missing}; the run publishes whole or not at all"})
            continue
        for a, rd in in_set:
            for fn, sha in check_admitted_source(adm, a, rd).items():
                out.append((a, camp, rd_name, rd / fn, sha))
    return out


def publish(attempts: Sequence[AttemptRoot], accepted: Accepted, *, inbox: Path, host: str = DEFAULT_HOST,
            union_dir=None, dry_run: bool = False, ledger: Optional[Path] = None) -> PublishReport:
    """Copy every record file of every admitted (attempt, run dir) into the inbox (see the module docstring for the rule)."""
    if host.startswith("_") or "/" in host:
        raise PublishRefused(f"inbox host {host!r}: must not start with '_' (never merged) or contain '/'")
    if union_dir is None:
        raise PublishRefused("--union-dir is required: records reach the registry only through logical_run_complete.py's committed admission")
    rep = PublishReport()
    inbox = Path(inbox)
    ledger = Path(ledger) if ledger is not None else inbox / host / LEDGER_DIR / "published.jsonl"
    holders: Dict[Tuple[str, str], List[Tuple[AttemptRoot, Path]]] = {}       # run dir -> the publishable attempts holding it
    for a in publishable(attempts, accepted, rep):
        for camp, rd_name, rd in a.run_dirs():
            holders.setdefault((camp, rd_name), []).append((a, rd))
    planned = admitted_sources(holders, union_dir, rep)
    rep.blocks = {a.attempt_id: attempt_block(a.attempt_id) for a, *_ in planned}
    dests = [(a, camp, rd_name, src, sha, inbox / host / camp / rd_name / published_name(src.name, rep.blocks[a.attempt_id]))
             for a, camp, rd_name, src, sha in planned]
    # collision check BEFORE any write: two planned files on one destination is the numbering invariant broken
    by_dest: Dict[Path, List[Path]] = {}
    for *_ignored, src, _sha, dest in dests:
        by_dest.setdefault(dest, []).append(src)
    clashes = {d: s for d, s in by_dest.items() if len(s) > 1}
    if clashes:
        raise PublishRefused("filename collision in the planned publication: " + "; ".join(f"{d} <- {[str(x) for x in s]}" for d, s in clashes.items()))
    for a, camp, rd_name, src, sha, dest in dests:
        entry = {"attempt_id": a.attempt_id, "block": rep.blocks[a.attempt_id], **a.axes, "campaign": camp, "run_dir": rd_name,
                 "src": str(src), "dest": str(dest), "logical_sha256": sha}
        if dest.exists():
            have = R.file_logical_sha(str(dest))
            if have == sha:
                rep.duplicates.append(entry)
                continue
            raise PublishRefused(f"{dest} exists with different bytes than {src} (logical sha {have[:12]} vs {sha[:12]}); "
                                 "a same-named record file with other content is a producer bug, never overwritten")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest)
        rep.published.append(entry)
    if not dry_run and (rep.published or rep.duplicates):
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a", encoding="utf-8") as fh:
            for status, entries in (("published", rep.published), ("duplicate", rep.duplicates)):
                for e in entries:
                    fh.write(json.dumps({**e, "status": status, "utc": R.utc_now()}, sort_keys=True) + "\n")
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--accepted", required=True, help="JSON {six-axis key: attempt_id} exported from the wave's accepted Dict")
    ap.add_argument("--attempts", nargs="+", required=True, help="roots holding the pulled attempt roots (ATTEMPT.json inside)")
    ap.add_argument("--union-dir", required=True, help="logical_run_complete.py's verdict directory (RS/registry/union)")
    ap.add_argument("--inbox", default=DEFAULT_INBOX)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        attempts = discover_attempts(a.attempts)
        acc = load_accepted(a.accepted)
        rep = publish(attempts, acc, inbox=Path(a.inbox), host=a.host, union_dir=Path(a.union_dir), dry_run=a.dry_run,
                      ledger=Path(a.ledger) if a.ledger else None)
    except PublishRefused as exc:
        print(f"registry_publish: REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(rep.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
