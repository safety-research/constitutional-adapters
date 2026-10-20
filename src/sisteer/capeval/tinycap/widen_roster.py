"""The bcb-bfcl-200-widen lane's ROSTER reader (plan S3; spec "Interfaces and contracts"):
yaml-only, the ONE reader of ``configs/capeval/bcb_bfcl_200_widen_roster.yaml`` for the
launch wrapper (``scripts/bcb_bfcl_200_widen_launch.py``), the analysis and the artifact.

The roster names every cell the objective implies, keyed by ENTRY id (the paper's arm key
where one exists) with a NULLABLE ``cell_id`` — the launcher cell id, required to parse only
when the entry is runnable (AlphaSteer has no cell id anywhere, and none is
invented). Every entry carries ``status in {runnable, queued, blocked}``; every non-runnable
one its ``reason`` AND the ``unblock`` trigger, so nothing is silently dropped.
A runnable OBJECT entry names the staged id and its sha256, which
:func:`validate` holds to ``configs/sdfbase_full22_pins.json``; the launcher then wires that
sha into the preflight's identity request as the cell's ``pin_sha256`` (S1-e), so a wrong pin
refuses before any GPU. ``object_generation``: ``final`` or ``batch8-provisional`` — a
provisional object names the key that supersedes it and the trigger that re-runs it.

The strings ``glm46vflash`` / ``glm`` may appear NOWHERE but ``excluded_families``.

The LAUNCH-TIME SNAPSHOT (plan S8): a fitted store's generation label is bound to
the roster AS LAUNCHED, never to the current roster — :func:`snapshot` renders the per-cell
block the wrapper hands the launcher (``--roster-snapshot``), which ``run_configuration`` writes
into the cell's out dir as :data:`SNAPSHOT_FILENAME`; :func:`snapshot_digest` recomputes its
``snapshot_sha256`` over the canonical content, so a relabelled sidecar refuses downstream.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[4]
#: The checked-in roster (repo-relative) and the pins it binds every object sha to.
ROSTER_REL = "configs/capeval/bcb_bfcl_200_widen_roster.yaml"
PINS_REL = "configs/sdfbase_full22_pins.json"
ROSTER_PATH = REPO / ROSTER_REL
PINS_PATH = REPO / PINS_REL

LANE = "bcb-bfcl-200-widen"
#: the SIBLING lane over the FULL item sets — its own roster document
#: (:data:`ROSTER_FULL_REL`), the same reader, the same rules, the two FULL legs
#: (``bigcodebench_full`` / ``bfcl_full``) in place of the 200 ones.
#: A sibling rather than an extension of the 200 roster: a launch snapshot binds a
#: store to the roster FILE's sha (``SNAPSHOT_IMMUTABLE_FIELDS``), so editing the live
#: 200 roster would refuse every in-flight relaunch of the running 200 waves.
LANE_FULL = "bcb-bfcl-full"
ROSTER_FULL_REL = "configs/capeval/bcb_bfcl_full_roster.yaml"
ROSTER_FULL_PATH = REPO / ROSTER_FULL_REL
STATUS_RUNNABLE, STATUS_QUEUED, STATUS_BLOCKED = "runnable", "queued", "blocked"
STATUSES: Tuple[str, ...] = (STATUS_RUNNABLE, STATUS_QUEUED, STATUS_BLOCKED)
#: the generation vocabulary: a batch-8 provisional object names its superseding key. ``batch64``: an
#: object of the batch-64 re-fit lineage (staged ids ``itdoc64__`` / ``derived_itdoc64__``), served BESIDE a batch-8
#: cell under the mixed-set rule — the stamp the analysis carries so a row names the mix; not provisional.
GENERATION_FINAL, GENERATION_BATCH8, GENERATION_BATCH64 = "final", "batch8-provisional", "batch64"
GENERATIONS: Tuple[str, ...] = (GENERATION_FINAL, GENERATION_BATCH8, GENERATION_BATCH64)
#: an entry's PUBLICATION-POOLING declaration: a measured cell whose entry carries
#: ``pooling: {held_out: true, ruled_by, reason}`` is REPORTED — graded, drawn, summarised, exported with its numbers — and
#: its paper rows stamp ``held_out`` so the notebook keeps it out of every pooled category mean. Declared here, never inferred from
#: the numbers; only a runnable steered entry may carry it (the `none` anchor is what the steered cells are paired against).
POOLING_KEY = "pooling"
POOLING_FIELDS: Tuple[str, ...] = ("held_out", "ruled_by", "reason")
#: The two legs, as the launcher's CONTROL atoms (``--controls``) and the roster's keys.
LEG_BCB, LEG_BFCL = "bigcodebench_200", "bfcl_200"
#: The FULL lane's two legs (the full siblings' control atoms).
LEG_BCB_FULL, LEG_BFCL_FULL = C.CONTROL_BIGCODEBENCH_FULL, C.CONTROL_BFCL_FULL
#: The lane's SERVING PATH, declared ONCE here — the roster header carries it (``serving_path: lora_native``) so a
#: reader of the wave sees what launched, the launch wrapper (``scripts/bcb_bfcl_200_widen_launch.py``) composes it
#: as ``--serving-path`` on the launcher AND the sandbox watcher, and :func:`validate` REFUSES a header that
#: disagrees. Before this the wrapper composed no ``--serving-path`` (the launcher's entry-point default carried
#: native since d3422f954) and the registry rows of the earlier qwen waves stamp ``serving_path unverified`` —
#: native was never ESTABLISHED for this lane.
SERVING_PATH_KEY = "serving_path"
SERVING_PATH = C.SERVING_ENGINE_LORA_NATIVE
LEGS: Tuple[str, ...] = (LEG_BCB, LEG_BFCL)
LEGS_FULL: Tuple[str, ...] = (LEG_BCB_FULL, LEG_BFCL_FULL)
#: lane -> its two legs (BCB first, BFCL second); every reader that needs a lane's legs
#: goes through :func:`legs_of_lane` / :func:`lane_legs`, never :data:`LEGS` directly.
LEGS_OF_LANE: Mapping[str, Tuple[str, ...]] = {LANE: LEGS, LANE_FULL: LEGS_FULL}
#: the BFCL leg of each lane (the tool-calling leg the live parser gate guards)
BFCL_LEG_OF_LANE: Mapping[str, str] = {LANE: LEG_BFCL, LANE_FULL: LEG_BFCL_FULL}
#: The excluded family and the substring the reader refuses anywhere else.
EXCLUDED_FAMILY = "glm46vflash"
_GLM_RE = re.compile(r"glm", re.I)
#: A prompted cell over a document-corpus pseudo-instruction is a fabricated baseline
#: (``registry_from_explicit_cells`` refuses it; refused here too, at the roster).
_CORPUS_SI_PREFIXES: Tuple[str, ...] = ("constv2",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")

#: The launch-time snapshot sidecar written into every cell's out dir (plan S8).
SNAPSHOT_FILENAME = "lane_roster.json"
#: Exact key set of the sidecar; ``snapshot_sha256`` is over the canonical JSON of the others.
SNAPSHOT_FIELDS: Tuple[str, ...] = (
    "roster_sha256", "wave", "entry_id", "cell_id", "object_generation", "superseded_by_key",
    "object_sha256", "launched_at", "snapshot_sha256")
SNAPSHOT_CONTENT_FIELDS: Tuple[str, ...] = tuple(f for f in SNAPSHOT_FIELDS if f != "snapshot_sha256")
#: bcb-bfcl-200-widen code review: a store root's `lane_roster.json` is IMMUTABLE once
#: written. A relaunch into the same root is COMPATIBLE when every content field but `launched_at`
#: equals the filed one (the same roster sha, entry, cell, generation, superseding key and object
#: sha — the launch-time facts the label is bound to); its block is then appended to the RETRY
#: record beside the original, never written over it. Any other difference REFUSES without a write.
SNAPSHOT_IMMUTABLE_FIELDS: Tuple[str, ...] = tuple(f for f in SNAPSHOT_CONTENT_FIELDS if f != "launched_at")
SNAPSHOT_RETRY_FILENAME = "lane_roster.retries.jsonl"


class RosterError(ValueError):
    """The roster does not pass its own rules (or names something the code cannot serve)."""


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #


def load_roster(path: str | Path = ROSTER_PATH) -> Dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise RosterError(f"{path}: not a mapping")
    return dict(doc)


def load_pins(path: str | Path = PINS_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def roster_sha256(path: str | Path = ROSTER_PATH) -> str:
    """sha256 of the roster FILE bytes — what a launch-time snapshot records."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def waves(roster: Mapping[str, Any]) -> List[str]:
    return list((roster.get("waves") or {}).keys())


def lane_of(roster: Mapping[str, Any]) -> str:
    """The roster's declared lane (:data:`LANE` or :data:`LANE_FULL`); :func:`validate`
    refuses any other spelling."""
    return str(roster.get("lane") or "")


def legs_of_lane(lane: str) -> Tuple[str, ...]:
    """The two leg atoms of ``lane``; refuses an unknown lane."""
    try:
        return LEGS_OF_LANE[lane]
    except KeyError:
        raise RosterError(f"lane {lane!r} is not one of {sorted(LEGS_OF_LANE)}") from None


def lane_legs(roster: Mapping[str, Any]) -> Tuple[str, ...]:
    """The leg atoms the roster's lane runs (``LEGS`` for the 200 lane, ``LEGS_FULL`` for
    the full lane)."""
    return legs_of_lane(lane_of(roster))


def serving_path(roster: Mapping[str, Any]) -> str:
    """The roster header's declared serving path — :func:`validate` holds it to :data:`SERVING_PATH`."""
    return str(roster.get(SERVING_PATH_KEY) or "")


def wave(roster: Mapping[str, Any], name: str) -> Dict[str, Any]:
    w = (roster.get("waves") or {}).get(name)
    if not isinstance(w, Mapping):
        raise RosterError(f"roster has no wave {name!r}; waves: {waves(roster)}")
    return dict(w)


def entries(roster: Mapping[str, Any], wave_name: str) -> Dict[str, Dict[str, Any]]:
    return {str(k): dict(v or {}) for k, v in (wave(roster, wave_name).get("entries") or {}).items()}


def _status(entry: Mapping[str, Any]) -> str:
    return str(entry.get("status") or "")


def runnable_entries(roster: Mapping[str, Any], wave_name: str) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
    """``[(entry_id, cell_id, object or None)]`` for the wave's runnable entries, ``none``
    FIRST (the same-wave anchor every contrast pairs against), then roster order."""
    out = [(eid, str(e["cell_id"]), (dict(e["object"]) if e.get("object") else None))
           for eid, e in entries(roster, wave_name).items() if _status(e) == STATUS_RUNNABLE]
    out.sort(key=lambda t: (0 if t[1] == C.ROUTE_NONE_CELL else 1))
    return out


def runnable_cells(roster: Mapping[str, Any], wave_name: str) -> List[str]:
    return [cell for _, cell, _ in runnable_entries(roster, wave_name)]


def _leg_status(value: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(value, str):
        return value, {}
    if isinstance(value, Mapping):
        return str(value.get("status") or ""), dict(value)
    return "", {}


def legs(roster: Mapping[str, Any], wave_name: str) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """``{leg atom: (status, detail)}`` — ``runnable`` or ``{status, reason, unblock}``."""
    return {str(k): _leg_status(v) for k, v in (wave(roster, wave_name).get("legs") or {}).items()}


def runnable_legs(roster: Mapping[str, Any], wave_name: str) -> List[str]:
    """The wave's runnable legs as the launcher's control atoms, in the lane's leg order
    (:func:`lane_legs`: ``LEGS`` for the 200 lane, ``LEGS_FULL`` for the full lane)."""
    st = legs(roster, wave_name)
    return [leg for leg in lane_legs(roster) if st.get(leg, ("", {}))[0] == STATUS_RUNNABLE]


def queued_or_blocked(roster: Mapping[str, Any], wave_name: str) -> Dict[str, Dict[str, Any]]:
    """Every non-runnable entry with its status, reason, unblock trigger and (nullable) cell id."""
    return {eid: {"status": _status(e), "cell_id": e.get("cell_id"), "reason": e.get("reason"),
                  "unblock": e.get("unblock"), "paper_arm": e.get("paper_arm")}
            for eid, e in entries(roster, wave_name).items() if _status(e) != STATUS_RUNNABLE}


def paper_lane(roster: Mapping[str, Any], wave_name: str) -> Dict[str, Optional[str]]:
    """``{family, lane_kind, view}`` — the paper notebook's row vocabulary for the wave."""
    w = wave(roster, wave_name)
    return {"family": w.get("family"), "lane_kind": w.get("lane_kind"), "view": w.get("view")}


def object_generation(roster: Mapping[str, Any], wave_name: str, entry_id: str) -> Tuple[Optional[str], Optional[str]]:
    """``(object_generation, superseded_by_key)`` of an entry — ``(None, None)`` for an
    object-free entry (``none`` / prompted: generation-NEUTRAL)."""
    e = entries(roster, wave_name).get(entry_id)
    if e is None:
        raise RosterError(f"wave {wave_name!r} has no entry {entry_id!r}")
    obj = e.get("object") or {}
    return (obj.get("object_generation"), obj.get("superseded_by_key")) if obj else (None, None)


def prerequisites(roster: Mapping[str, Any], wave_name: str) -> List[str]:
    """The wave's prerequisites the orchestrator has NOT yet recorded in ``prerequisites_met``
    — the launcher prints and REFUSES on any."""
    w = wave(roster, wave_name)
    want = [str(p) for p in (w.get("prerequisites") or [])]
    met = {str(p) for p in (w.get("prerequisites_met") or [])}
    return [p for p in want if p not in met]


def pin_of(roster: Mapping[str, Any], wave_name: str, entry_id: str) -> Optional[str]:
    """The roster's object sha256 of an entry (None for an object-free entry) — what the
    launcher wires into the identity request as the cell's ``pin_sha256``."""
    e = entries(roster, wave_name).get(entry_id)
    if e is None:
        raise RosterError(f"wave {wave_name!r} has no entry {entry_id!r}")
    obj = e.get("object") or {}
    return str(obj["sha256"]) if obj.get("sha256") else None


def pins_by_cell(roster: Mapping[str, Any], wave_name: str) -> Dict[str, str]:
    """``{cell_id: object_sha256}`` over the wave's runnable OBJECT entries."""
    return {cell: str(obj["sha256"]) for _, cell, obj in runnable_entries(roster, wave_name) if obj and obj.get("sha256")}


def pooling_of(roster: Mapping[str, Any], wave_name: str, entry_id: str) -> Optional[Dict[str, Any]]:
    """The entry's ``pooling`` block (a copy), or None when it declares nothing."""
    e = entries(roster, wave_name).get(entry_id)
    if e is None:
        raise RosterError(f"wave {wave_name!r} has no entry {entry_id!r}")
    blk = e.get(POOLING_KEY)
    return dict(blk) if isinstance(blk, Mapping) else None


def held_out_cells(roster: Mapping[str, Any], wave_name: str) -> Dict[str, Dict[str, Any]]:
    """``{cell_id: pooling block}`` for every entry of the wave whose ``pooling.held_out`` is true — the cells whose
    readings are reported but kept out of every pooled category mean."""
    out: Dict[str, Dict[str, Any]] = {}
    for e in entries(roster, wave_name).values():
        blk = e.get(POOLING_KEY)
        if isinstance(blk, Mapping) and blk.get("held_out") is True and e.get("cell_id") is not None:
            out[str(e["cell_id"])] = dict(blk)
    return out


def entry_of_cell(roster: Mapping[str, Any], wave_name: str, cell_id: str) -> str:
    hits = [eid for eid, e in entries(roster, wave_name).items() if e.get("cell_id") == cell_id]
    if len(hits) != 1:
        raise RosterError(f"wave {wave_name!r}: {len(hits)} entries name cell {cell_id!r} (want exactly one)")
    return hits[0]


def launch_problems(roster: Mapping[str, Any], *, wave_name: str, cell_id: str,
                    entry_id: Optional[str] = None) -> List[str]:
    """Why ``cell_id`` may NOT launch as a cell of ``wave_name`` under this roster:
    the wave must exist, exactly one entry must name the cell, that entry must be RUNNABLE (a
    blocked / queued entry names its status, reason and unblock trigger), and — when the launch's
    snapshot names an ``entry_id`` — it must be that entry. Empty = may launch."""
    if wave_name not in waves(roster):
        return [f"wave {wave_name!r} is not in the roster ({waves(roster)})"]
    try:
        eid = entry_of_cell(roster, wave_name, cell_id)
    except RosterError as e:
        return [str(e)]
    e = entries(roster, wave_name)[eid]
    out: List[str] = []
    if entry_id is not None and str(entry_id) != eid:
        out.append(f"the snapshot names entry {entry_id!r} for cell {cell_id!r}, the roster's entry is {eid!r}")
    st = _status(e)
    if st != STATUS_RUNNABLE:
        out.append(f"wave {wave_name!r} entry {eid!r} (cell {cell_id!r}) is {st or 'unstated'}, not {STATUS_RUNNABLE}: "
                   f"{e.get('reason') or 'no reason recorded'} — unblock: {e.get('unblock') or 'no trigger recorded'}")
    return out


# --------------------------------------------------------------------------- #
# validation — the roster passes its OWN rules (P2-8); a test loads the        #
# checked-in file against the checked-in pins and contracts.MODELS            #
# --------------------------------------------------------------------------- #


def _walk_strings(node: Any, path: Tuple[str, ...] = ()):
    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from _walk_strings(k, path)
            yield from _walk_strings(v, path + (str(k),))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _walk_strings(v, path + (str(i),))
    elif isinstance(node, str):
        yield path, node


def glm_problems(roster: Mapping[str, Any]) -> List[str]:
    """``glm`` (any case) appears nowhere but the ``excluded_families`` list."""
    out = []
    for path, s in _walk_strings(roster):
        if path[:1] == ("excluded_families",):
            continue
        if _GLM_RE.search(s):
            out.append(f"{'.'.join(path) or '<key>'}: {s!r} names the excluded family (D0(7): GLM appears "
                       "nowhere but excluded_families)")
    return out


def _parse_cell(cell_id: str) -> Optional[str]:
    try:
        parsed = C.parse_launcher_cell(str(cell_id))
    except Exception as e:                       # noqa: BLE001 — every grammar refusal is a roster problem
        return f"cell_id {cell_id!r} does not parse through parse_launcher_cell ({type(e).__name__}: {e})"
    if parsed.kind == C.CELL_KIND_PROMPTED and str(parsed.si).startswith(_CORPUS_SI_PREFIXES):
        return (f"cell_id {cell_id!r} is a prompted cell over a document-corpus pseudo-instruction — its text "
                "is never a system prompt (registry_from_explicit_cells refuses it)")
    return None


def validate(roster: Mapping[str, Any], pins: Mapping[str, Any], models: Mapping[str, Any], *,
             tree: Optional[str | Path] = None) -> List[str]:
    """Every rule of plan S3, as problem strings (empty = the roster passes). ``tree`` is the root the row configs
    and the checked-in live verdict are READ from — the staged tree for the launch sha inside a container, the source
    checkout when None (never this module's ``__file__``, which the image mounts under /root with no
    ``configs/``); a roster with a runnable wave and no readable root is refused loudly, never checked silently."""
    problems: List[str] = []
    root: Optional[Path] = Path(tree) if tree is not None else C.source_checkout_root()
    if root is None and any(any(str(e.get("status")) == STATUS_RUNNABLE for e in (w.get("entries") or {}).values() if isinstance(e, Mapping))
                            for w in (roster.get("waves") or {}).values() if isinstance(w, Mapping)):
        problems.append("no tree to read the row configs and the live verdict from — widen_roster is not imported from a source checkout "
                        "and the caller passed no staged tree; pass tree=<the staged tree for the launch sha>")
    lane = lane_of(roster)
    if lane not in LEGS_OF_LANE:
        problems.append(f"lane {lane!r} != {LANE!r} (the 200 lane) or {LANE_FULL!r} (the full lane)")
        lane_leg_atoms: Tuple[str, ...] = LEGS
    else:
        lane_leg_atoms = LEGS_OF_LANE[lane]
    if list(roster.get("legs") or []) != [C.inspect_task(a) for a in lane_leg_atoms]:
        problems.append(f"legs {roster.get('legs')!r} != the lane's two pinned row tasks "
                        f"{[C.inspect_task(a) for a in lane_leg_atoms]}")
    bfcl_leg = BFCL_LEG_OF_LANE.get(lane, LEG_BFCL)
    if roster.get("sampling") != "protocol":
        problems.append(f"sampling {roster.get('sampling')!r} != 'protocol' (the lane's decoding mode)")
    if roster.get("thinking") != C.THINKING_OFF:
        problems.append(f"thinking {roster.get('thinking')!r} != {C.THINKING_OFF!r} (D0(6))")
    if roster.get("pins") != PINS_REL:
        problems.append(f"pins {roster.get('pins')!r} != {PINS_REL!r}")
    if roster.get(SERVING_PATH_KEY) != SERVING_PATH:
        problems.append(f"{SERVING_PATH_KEY} {roster.get(SERVING_PATH_KEY)!r} != {SERVING_PATH!r} (D86: the lane serves on the "
                        f"native multi-LoRA engine; the wrapper composes --serving-path {SERVING_PATH} on every launch and "
                        "the roster header must say so — one authority, both spellings)")
    if list(roster.get("excluded_families") or []) != [EXCLUDED_FAMILY]:
        problems.append(f"excluded_families {roster.get('excluded_families')!r} != [{EXCLUDED_FAMILY!r}]")
    if list(roster.get("object_generations") or []) != list(GENERATIONS):
        problems.append(f"object_generations {roster.get('object_generations')!r} != {list(GENERATIONS)}")
    override = roster.get("sampling_clamp_override")
    want_from, want_to = C.SAMPLING_CLAMP_OVERRIDE["from"], C.SAMPLING_CLAMP_OVERRIDE["to"]
    if not (isinstance(override, Mapping) and override.get("from") == want_from and override.get("to") == want_to
            and str(override.get("why") or "").strip()):
        problems.append(f"sampling_clamp_override {override!r} is not {{from: {want_from}, to: {want_to}, why: ...}} "
                        "(D13, contracts.SAMPLING_CLAMP_OVERRIDE)")
    problems += glm_problems(roster)
    for wname in waves(roster):
        w = wave(roster, wname)
        where = f"waves.{wname}"
        for key in ("family", "lane_kind"):
            if not w.get(key):
                problems.append(f"{where}: missing {key}")
        ents = entries(roster, wname)
        if not ents:
            problems.append(f"{where}: no entries")
        any_runnable = False
        steered_runnable = False
        none_runnable = False
        seen_cells: Dict[str, str] = {}
        for eid, e in ents.items():
            ew = f"{where}.entries.{eid}"
            st = _status(e)
            if st not in STATUSES:
                problems.append(f"{ew}: status {st!r} not in {list(STATUSES)}")
                continue
            cell = e.get("cell_id")
            if cell is not None:
                if str(cell) in seen_cells:
                    problems.append(f"{ew}: cell_id {cell!r} is also entry {seen_cells[str(cell)]!r}'s")
                seen_cells[str(cell)] = eid
            pool = e.get(POOLING_KEY)
            if pool is not None:
                if not isinstance(pool, Mapping) or set(pool) != set(POOLING_FIELDS) or not isinstance(pool.get("held_out"), bool):
                    problems.append(f"{ew}: {POOLING_KEY} must be {{held_out: bool, ruled_by, reason}} (got {pool!r})")
                else:
                    if pool["held_out"]:
                        for key in ("ruled_by", "reason"):
                            if not str(pool.get(key) or "").strip():
                                problems.append(f"{ew}: a held-out entry needs a non-empty {POOLING_KEY}.{key}")
                    if st != STATUS_RUNNABLE:
                        problems.append(f"{ew}: {POOLING_KEY} on a {st} entry — only a runnable entry's readings can be held out")
                    if cell is not None and _parse_cell(str(cell)) is None and C.parse_launcher_cell(str(cell)).kind == C.CELL_KIND_BASE:
                        problems.append(f"{ew}: {POOLING_KEY} on the `none` anchor — the anchor is what the steered cells are paired against")
            if st != STATUS_RUNNABLE:
                for key in ("reason", "unblock"):
                    if not str(e.get(key) or "").strip():
                        problems.append(f"{ew}: a {st} entry needs a non-empty {key}")
                if cell is not None:
                    bad = _parse_cell(str(cell))
                    if bad:
                        problems.append(f"{ew}: {bad}")
                continue
            any_runnable = True
            if cell is None:
                problems.append(f"{ew}: a runnable entry needs a non-null cell_id")
                continue
            bad = _parse_cell(str(cell))
            if bad:
                problems.append(f"{ew}: {bad}")
                continue
            kind = C.parse_launcher_cell(str(cell)).kind
            if kind == C.CELL_KIND_BASE:
                none_runnable = True
            obj = e.get("object")
            if kind == C.CELL_KIND_FITTED:
                steered_runnable = True
                if not isinstance(obj, Mapping):
                    problems.append(f"{ew}: a runnable fitted cell needs an `object` block (staged_id, sha256, kind, "
                                    "sidecar, object_generation)")
                    continue
                sid, sha = obj.get("staged_id"), str(obj.get("sha256") or "")
                if not sid:
                    problems.append(f"{ew}: object.staged_id missing")
                if not _SHA256.match(sha):
                    problems.append(f"{ew}: object.sha256 {sha!r} is not a sha256 hex")
                pin_entry = ((pins.get(str(w.get("family"))) or {}).get(str(sid)) or {}) if sid else {}
                pinned = pin_entry.get("object_sha256")
                if pinned is None:
                    problems.append(f"{ew}: no pins entry for {w.get('family')}/{sid} in {PINS_REL}")
                elif pinned != sha:
                    problems.append(f"{ew}: object.sha256 {sha[:12]}… != the pins entry {str(pinned)[:12]}… for {sid}")
                gen = obj.get("object_generation")
                if gen not in GENERATIONS:
                    problems.append(f"{ew}: object.object_generation {gen!r} not in {list(GENERATIONS)}")
                if gen == GENERATION_BATCH8:
                    if not str(obj.get("superseded_by_key") or "").strip():
                        problems.append(f"{ew}: a {GENERATION_BATCH8} object names its superseded_by_key")
                    if not str(e.get("rerun_trigger") or "").strip():
                        problems.append(f"{ew}: a {GENERATION_BATCH8} entry names its rerun_trigger")
                for key in ("kind", "sidecar"):
                    if not obj.get(key):
                        problems.append(f"{ew}: object.{key} missing")
                ih = e.get("intervention_hash_expected")
                if ih is not None and not _HEX16.match(str(ih)):
                    problems.append(f"{ew}: intervention_hash_expected {ih!r} is not 16 hex")
            elif obj:
                problems.append(f"{ew}: an object-free cell ({kind}) must not carry an `object` block")
        if steered_runnable and not none_runnable:
            # A STEERED-ONLY wave carries no unsteered leg — its baseline is the u0 run, named by `anchor_wave`: a wave
            # of this roster with the SAME family and served pin whose `none` is runnable
            anchor = w.get("anchor_wave")
            if not anchor:
                problems.append(f"{where}: a wave with a runnable steered entry needs a runnable `none` (the same-wave anchor) or an "
                                "`anchor_wave` naming the unsteered wave whose `none` is its baseline (D1859)")
            elif str(anchor) not in waves(roster):
                problems.append(f"{where}: anchor_wave {anchor!r} is not a wave of this roster")
            elif str(anchor) == wname:
                problems.append(f"{where}: anchor_wave names itself")
            else:
                aw = wave(roster, str(anchor))
                a_ents = entries(roster, str(anchor))
                a_none = [e for e in a_ents.values() if str(e.get("cell_id")) == C.ROUTE_NONE_CELL and _status(e) == STATUS_RUNNABLE]
                if not a_none:
                    problems.append(f"{where}: anchor_wave {anchor!r} has no runnable `none`")
                if aw.get("family") != w.get("family") or aw.get("served") != w.get("served"):
                    problems.append(f"{where}: anchor_wave {anchor!r} serves another family / pin ({aw.get('family')!r} {aw.get('served')!r} vs "
                                    f"{w.get('family')!r} {w.get('served')!r}) — a baseline is the same served model")
                # A steered leg with no runnable baseline leg cannot be certified — every runnable leg of the steered
                # wave must be runnable on its anchor (a queued / blocked anchor leg refuses the steered one by name)
                try:
                    missing_legs = sorted(set(runnable_legs(roster, wname)) - set(runnable_legs(roster, str(anchor))))
                except RosterError:
                    missing_legs = []                      # an unknown lane is reported by the lane check above, once
                if missing_legs:
                    problems.append(f"{where}: legs {missing_legs} are runnable here but not on anchor_wave {anchor!r} — a steered leg needs its "
                                    "baseline leg runnable on the same served model (the comparison is otherwise uncertifiable)")
        elif w.get("anchor_wave") and none_runnable:
            problems.append(f"{where}: carries both a runnable `none` and an anchor_wave — one baseline per wave")
        # legs
        leg_status = legs(roster, wname)
        for leg, (st, detail) in leg_status.items():
            lw = f"{where}.legs.{leg}"
            if leg not in lane_leg_atoms:
                problems.append(f"{lw}: not one of {list(lane_leg_atoms)}")
            if st not in STATUSES:
                problems.append(f"{lw}: status {st!r} not in {list(STATUSES)}")
            elif st != STATUS_RUNNABLE:
                for key in ("reason", "unblock"):
                    if not str(detail.get(key) or "").strip():
                        problems.append(f"{lw}: a {st} leg needs a non-empty {key}")
        # the model row
        slug = w.get("model_row")
        if any_runnable:
            if not slug:
                problems.append(f"{where}: a runnable wave names a model_row")
            elif slug not in models:
                problems.append(f"{where}: model_row {slug!r} is not a registry slug")
            else:
                row = models[slug]
                served = w.get("served") or {}
                if served.get("hf") != row.hf or served.get("rev") != row.rev:
                    problems.append(f"{where}: served {served!r} != row {slug!r}'s {row.hf}@{row.rev}")
                if w.get("objects_dir") != row.objects_dir:
                    problems.append(f"{where}: objects_dir {w.get('objects_dir')!r} != row {slug!r}'s {row.objects_dir!r}")
                if not leg_status:
                    problems.append(f"{where}: a runnable wave names its legs")
                if leg_status.get(bfcl_leg, ("", {}))[0] == STATUS_RUNNABLE:
                    if not row.petri_manifest:
                        problems.append(f"{where}: {bfcl_leg} is runnable but row {slug!r} has no petri_manifest (no verified "
                                        "tool parser — S5 must PASS first)")
                    else:
                        # The roster cannot run BFCL ahead of the LIVE record
                        live = C.tool_parser_live_gate_problem(row.family, tree=root)
                        if live:
                            problems.append(f"{where}: {bfcl_leg} is runnable but {live}")
        elif slug is not None and slug not in models:
            problems.append(f"{where}: model_row {slug!r} is not a registry slug")
        # prerequisites
        met = [str(p) for p in (w.get("prerequisites_met") or [])]
        want = [str(p) for p in (w.get("prerequisites") or [])]
        for p in met:
            if p not in want:
                problems.append(f"{where}: prerequisites_met names {p!r}, which is not a prerequisite")
        # A wave that declares BANK-STAGING prerequisites must declare one for EVERY fitted cell its row's generated
        # config enumerates — the registry build of every standup of the row validates every listed artifact, so a bank
        # missing ONE object refuses the `none` canary too, inside a metered container. The cross-check reads the
        # config the row serves.
        if any_runnable and slug in models and root is not None:
            problems += staging_coverage_problems(w, models[slug], where=where, repo=root)
    return problems


#: the spelling that marks a prerequisite as BANK STAGING (an object landing in the wave's bank): a wave carrying one
#: is held to name every fitted cell its config enumerates in such lines (:func:`staging_coverage_problems`)
BANK_STAGING_PREFIX = "S4 (bank staging):"


def bank_staging_prerequisites(w: Mapping[str, Any]) -> List[str]:
    return [str(p) for p in (w.get("prerequisites") or []) if str(p).startswith(BANK_STAGING_PREFIX)]


def config_fitted_cells(row: Any, repo: Path = REPO) -> Optional[List[str]]:
    """The fitted cells the row's generated config enumerates (``explicit_cells``), or None when the config is not on
    this tree (a matrix lane's overlay written elsewhere) — the caller then has nothing to hold the wave to."""
    cfg = repo / str(getattr(row, "cfg", "") or "")
    if not str(getattr(row, "cfg", "") or "") or not cfg.is_file():
        return None
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    cells = doc.get("explicit_cells")
    if not isinstance(cells, list):
        return None
    out = []
    for c in cells:
        try:
            if C.parse_launcher_cell(str(c)).kind == C.CELL_KIND_FITTED:
                out.append(str(c))
        except Exception:                                    # noqa: BLE001 — the config's grammar is the generator's test's concern
            continue
    return out


def staging_coverage_problems(w: Mapping[str, Any], row: Any, *, where: str, repo: Path = REPO) -> List[str]:
    """Why the wave's bank-staging prerequisites do NOT cover its row's config: a wave with at least
    one ``S4:`` prerequisite must name — by cell id, or by the staged id of an entry on the same row — every fitted
    cell of the config's ``explicit_cells``; a wave declaring none is not held (its bank predates the discipline)."""
    lines = bank_staging_prerequisites(w)
    if not lines:
        return []
    cells = config_fitted_cells(row, repo)
    if cells is None:
        return []
    # whole-token match, never substring: `constv2-mix-full__lora_r4_ce` is a substring of the derived
    # `…__lora_r4_ce_minus_ctl` id, and a line naming only the derived object must not read as covering its parent. A
    # cell is covered by its own id, by a staged id of the form `<lane kind>__<cell>` (the generator's staged_ids_by_cell
    # grammar — the seed bank line names `base64s<s>__<cell>`, the extra `ar_c0p3__arditi_allpos`), or by the staged id
    # of the same wave's entry for it.
    tokens = set(re.findall(r"[A-Za-z0-9_\-]+", "\n".join(lines)))
    staged_of = {str(e.get("cell_id")): str((e.get("object") or {}).get("staged_id") or "")
                 for e in (w.get("entries") or {}).values() if isinstance(e, Mapping) and e.get("cell_id")}

    def covered(c: str) -> bool:
        return c in tokens or any(t.endswith("__" + c) for t in tokens) or bool(staged_of.get(c) and staged_of[c] in tokens)

    missing = [c for c in cells if not covered(c)]
    if not missing:
        return []
    return [f"{where}: the row's config ({getattr(row, 'cfg', '?')}) enumerates fitted cell(s) {missing} that no {BANK_STAGING_PREFIX!r} "
            "staging prerequisite of this wave names — every standup of the row builds its registry over the whole bank, so a "
            "missing object refuses the `none` canary too; declare the staging on THIS wave"]


def validate_or_raise(roster: Mapping[str, Any], pins: Mapping[str, Any], models: Mapping[str, Any], *,
                      tree: Optional[str | Path] = None) -> None:
    problems = validate(roster, pins, models, tree=tree)
    if problems:
        raise RosterError("the roster does not pass its own rules:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# the launch-time snapshot: the fitted store's generation label               #
# --------------------------------------------------------------------------- #


def snapshot_digest(block: Mapping[str, Any]) -> str:
    """sha256 over the canonical JSON of :data:`SNAPSHOT_CONTENT_FIELDS` (every field but the
    digest itself) — recomputed by every reader, so a relabelled sidecar refuses."""
    content = {k: block.get(k) for k in SNAPSHOT_CONTENT_FIELDS}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def snapshot(roster: Mapping[str, Any], wave_name: str, entry_id: str, *, roster_sha256: str,
             launched_at: str) -> Dict[str, Any]:
    """The per-cell block of the launch-time roster snapshot: the roster file's sha256, the wave,
    the entry, its cell id, the object's generation + superseding key + sha (all null for an
    object-free entry), when it launched, and the digest over that content."""
    e = entries(roster, wave_name).get(entry_id)
    if e is None:
        raise RosterError(f"wave {wave_name!r} has no entry {entry_id!r}")
    if _status(e) != STATUS_RUNNABLE:
        raise RosterError(f"wave {wave_name!r} entry {entry_id!r} is {_status(e)}, not runnable — nothing to snapshot")
    gen, sup = object_generation(roster, wave_name, entry_id)
    block = {"roster_sha256": str(roster_sha256), "wave": wave_name, "entry_id": entry_id,
             "cell_id": str(e["cell_id"]), "object_generation": gen, "superseded_by_key": sup,
             "object_sha256": pin_of(roster, wave_name, entry_id), "launched_at": str(launched_at)}
    block["snapshot_sha256"] = snapshot_digest(block)
    assert tuple(block) == SNAPSHOT_FIELDS, tuple(block)
    return block


def snapshot_problems(block: Any) -> List[str]:
    """Why a snapshot block is NOT a valid launch-time record: exact keys, a sha256 roster digest,
    a generation from ``GENERATIONS`` (or null for an object-free cell), a recomputing digest."""
    if not isinstance(block, Mapping):
        return [f"snapshot is {type(block).__name__}, not an object"]
    out = []
    if set(block) != set(SNAPSHOT_FIELDS):
        out.append(f"snapshot keys {sorted(block)} != {sorted(SNAPSHOT_FIELDS)}")
        return out
    if not _SHA256.match(str(block.get("roster_sha256") or "")):
        out.append("roster_sha256 is not a sha256 hex")
    gen = block.get("object_generation")
    if gen is not None and gen not in GENERATIONS:
        out.append(f"object_generation {gen!r} not in {list(GENERATIONS)}")
    if gen == GENERATION_BATCH8 and not block.get("superseded_by_key"):
        out.append(f"a {GENERATION_BATCH8} snapshot names its superseded_by_key")
    sha = block.get("object_sha256")
    if sha is not None and not _SHA256.match(str(sha)):
        out.append("object_sha256 is not a sha256 hex")
    if (gen is None) != (sha is None):
        out.append("object_generation and object_sha256 are both null (object-free) or both set (fitted)")
    for key in ("wave", "entry_id", "cell_id", "launched_at"):
        if not str(block.get(key) or "").strip():
            out.append(f"{key} is empty")
    if block.get("snapshot_sha256") != snapshot_digest(block):
        out.append("snapshot_sha256 does not recompute over the content (a relabelled sidecar)")
    return out


def snapshot_conflicts(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> List[str]:
    """Why ``incoming`` may NOT be filed as a retry of the store whose launch-time record is
    ``existing`` — every :data:`SNAPSHOT_IMMUTABLE_FIELDS` disagreement, named (empty = a
    compatible relaunch: the same roster, entry, cell, generation and object; only the launch
    time moved). Both blocks must be valid snapshots first (:func:`snapshot_problems`)."""
    out = [f"existing record: {p}" for p in snapshot_problems(existing)]
    out += [f"incoming snapshot: {p}" for p in snapshot_problems(incoming)]
    if out:
        return out
    for f in SNAPSHOT_IMMUTABLE_FIELDS:
        if existing.get(f) != incoming.get(f):
            out.append(f"{f}: filed {existing.get(f)!r} != this launch's {incoming.get(f)!r}")
    return out


def parse_snapshot_arg(value: str) -> Dict[str, Dict[str, Any]]:
    """``--roster-snapshot``: a JSON object ``{cell_id: block}`` inline or a path to one;
    every block validated (:func:`snapshot_problems`) and keyed by its own ``cell_id``."""
    text = str(value or "").strip()
    if not text:
        return {}
    if not text.startswith("{"):                   # a path, never stat()'d on an inline document
        p = Path(text).expanduser()
        if not p.is_file():
            raise RosterError(f"--roster-snapshot: {text!r} is neither inline JSON nor a readable file")
        text = p.read_text(encoding="utf-8")
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise RosterError(f"--roster-snapshot: not JSON ({e})") from None
    if not isinstance(doc, Mapping):
        raise RosterError("--roster-snapshot: not an object {cell_id: block}")
    out: Dict[str, Dict[str, Any]] = {}
    for cell, block in doc.items():
        problems = snapshot_problems(block)
        if problems:
            raise RosterError(f"--roster-snapshot[{cell!r}]: " + "; ".join(problems))
        if block["cell_id"] != cell:
            raise RosterError(f"--roster-snapshot[{cell!r}]: the block names cell_id {block['cell_id']!r}")
        out[str(cell)] = dict(block)
    return out


__all__ = [
    "ROSTER_REL", "PINS_REL", "ROSTER_PATH", "PINS_PATH", "LANE", "STATUSES", "GENERATIONS", "LEGS",
    "LANE_FULL", "ROSTER_FULL_REL", "ROSTER_FULL_PATH", "LEG_BCB_FULL", "LEG_BFCL_FULL", "LEGS_FULL", "LEGS_OF_LANE",
    "BFCL_LEG_OF_LANE", "lane_of", "legs_of_lane", "lane_legs",
    "EXCLUDED_FAMILY", "SNAPSHOT_FILENAME", "SNAPSHOT_FIELDS", "SERVING_PATH_KEY", "SERVING_PATH", "RosterError",
    "load_roster", "load_pins", "roster_sha256", "waves", "wave", "serving_path", "entries", "runnable_entries",
    "runnable_cells", "legs", "runnable_legs", "queued_or_blocked", "paper_lane", "object_generation",
    "prerequisites", "pin_of", "pins_by_cell", "entry_of_cell", "glm_problems", "validate",
    "BANK_STAGING_PREFIX", "bank_staging_prerequisites", "config_fitted_cells", "staging_coverage_problems",
    "validate_or_raise", "snapshot", "snapshot_digest", "snapshot_problems", "parse_snapshot_arg",
]
