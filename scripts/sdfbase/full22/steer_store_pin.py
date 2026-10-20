#!/usr/bin/env python3
"""THE SERVING-PATH PIN OF AN EXTERNAL-OBJECT ROW FOLLOWS ITS STORE. STDLIB ONLY: both single-cell launchers
(``gsm8k/g8k_cells.py``, ``math500/cells.py``) read this through the node's python3 before any venv exists, and the
dry run on the grading host reads it without torch.

THE RULE. A matrix row whose object is ANOTHER LANE's (``object_source: external`` — the AlphaSteer / Arditi steer rows) pins
the serving route the standup takes, or leaves it to the kind-aware default (common.sh ``f22_select_serving_path``):

  (a) its object KIND is a lens-only kind (``alphasteer``: no native form)            -> ``lens``, reason ``kind:alphasteer``;
  (b) its kind is natively servable (``static_r`` = Arditi; ``vector_all`` = the all-positions Arditi direction, ``arditi_allpos``)
      AND the row already has a steer store ON RECORD whose engine is the lens (``evidence.json`` ``serving_identity`` reads
      ``proxy_vllm+lens+…``; the served-steer stamp class)                             -> ``lens``, reason ``store:lens``;
  (c) a natively servable kind with NO store on record                                -> None (unpinned), reason ``unpinned:no-store``:
      the kind-aware default then selects ``lora_native`` at ``f22_init`` (both kinds serve on the residual slot);
  (d) a natively servable kind whose store on record is NATIVE-stamped                -> None (unpinned), reason ``store:native``;
  (e) a store carrying BOTH engines' stamps REFUSES BY NAME (never a guess, never the first stamp seen).

The two tuples ``LENS_ONLY_KINDS`` / ``NATIVELY_SERVABLE_KINDS`` PARTITION the steer cells' kinds, and :func:`pin_for` refuses a kind in
neither by name: which kinds read the store is DECLARED, never "whatever is not lens-only". ``arditi_allpos`` has never been run (no
store on either cluster, no pins entry — nothing invented), so rule (c) is its production reading until a store exists.

"On record" has TWO sources, read together (a per-cluster disk read alone cannot honour "never re-run a row that has a store" —
a row's canonical store may live on the OTHER cluster, so a re-launch there would have written a native store for a cell whose
paper record is lens):

  * DISK — the row's OWN out root(s) under the wave root on the cluster the job runs on: the CAP leg's root and, where the launcher
    has one, its rung sibling root (a rung leg must run on the base leg's engine, so the two roots are read together). A superseded
    or set-aside sibling directory (``…-superseded``, ``…-cancelled``) is not the row's out root and is never read.
  * PINS — the repo's admitted store-pins records, which travel with the launch tree (``configs/sdfbase/gsm8k/store_pins_*.json``,
    ``configs/sdfbase/math500/store_pins_default.json``): each admitted leg's ``_meta.serving_engine_by_leg[<leg key>].engine``,
    DERIVED from the pulled store at pin time (``stamp_store_pins_engines.py``: the store's identity + engine stamp, its producer
    tree verified equal to the leg's pin), never typed.

A lens engine from EITHER source pins the lens (``store:lens(disk)`` / ``store:lens(pins)`` / ``store:lens(disk+pins)``); a native
engine from either returns None with ``store:native(<source>)``; a disagreement between the sources, or within a store, refuses by
name; ``unpinned:no-store`` ONLY when neither source knows the row. Nothing here decides for a row whose object is not external.

WHY THE STORE AND NOT THE KIND. Every steer store on record today (16 GSM8K roots, 15 MATH-500 roots on the primary cluster,
2026-09-16) is a lens store, and a finished store is never re-run — so a natively servable kind must keep the lens
wherever its lens store exists, and take the native path only where nothing was served yet. The store is the authority; the
launcher reads it at its first row-resolution step (``*_select_serving_path``), before the out root is created and before any
marker is written.

THE MARKERS (mirrors of the library's vocabulary — a test pins each one equal to its source; the mirror is checked, never
trusted): the serving identity is ``<backend>+<path>+<mode>`` (``sisteer.capeval.manifest.parse_serving_identity``) whose path
segment is ``lens`` | ``lora`` | ``passthrough``; a natively served store also stamps ``route.serving_engine: lora_native``
beside its ``+lora+`` identity (``contracts.SERVED_ENGINE_FIELD`` / ``SERVING_ENGINE_LORA_NATIVE``; the MATH-500 module's
``certification_path_of``), a lens store carries no stamp (the historical record) or ``lens``. Read from the store root's
``evidence.json``, every ``cells/*/manifest.json`` (``route``) and every ``cells/*/evidence.json``.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

#: the object KIND of each steer cell — a mirror of ``sisteer.capeval.petri.registry_adapter.ARM_OBJECT_KINDS`` for the cells the
#: steer wave schedules (pinned equal by tests/test_math500_store_pins_served_steer.py for every cell the adapter knows; the cells
#: it does not know yet are EXACTLY ``PENDING_ADAPTER_REGISTRATIONS`` below, which the same test checks)
KIND_ALPHASTEER, KIND_STATIC_R, KIND_VECTOR_ALL = "alphasteer", "static_r", "vector_all"
STEER_CELL_KINDS: Mapping[str, str] = {"alphasteer": KIND_ALPHASTEER, "arditi": KIND_STATIC_R, "arditi_allpos": KIND_VECTOR_ALL}
#: A DECLARED, CHECKED GAP — not drift: the steer cells whose ``registry_adapter.ARM_OBJECT_KINDS`` entry has not landed on this
#: tree. EMPTY today. It carried ``arditi_allpos`` while that cell's Class B registration was in flight in another change; the
#: registration landed (the adapter maps it to ``vector_all``) and the entry was deleted, which is the whole mechanism working:
#: the mirror test held in BOTH states without an edit, and a separate named test asked for this deletion by cell and by file
#: rather than letting a stale declaration sit here unread. Never add a cell here to silence a mirror disagreement; an entry
#: must carry the kind ``STEER_CELL_KINDS`` gives the cell (refused at import otherwise).
PENDING_ADAPTER_REGISTRATIONS: Mapping[str, str] = {}
#: the kinds with NO native form — a mirror of ``defense_cells.native_serving_disposition(kind, None)[0] == "skip"`` over the
#: steer cells' kinds (pinned): rule (a) pins the lens by kind alone and never reads a store for them
LENS_ONLY_KINDS: tuple[str, ...] = (KIND_ALPHASTEER,)
#: the kinds WITH a native form — served on the residual slot (``static_r`` = Arditi, ``vector_all`` = Arditi all-positions): a
#: mirror of ``defense_cells.NATIVE_LORA_SERVABLE_KINDS`` (and ``RESIDUAL_SLOT_KINDS``) restricted to the steer cells' kinds
#: (pinned). Rules (b)–(e) read the store for these and for NO other kind: :func:`pin_for` refuses a kind in neither tuple by name,
#: and :func:`_check_kind_tables` refuses at import a steer kind that is in neither or in both
NATIVELY_SERVABLE_KINDS: tuple[str, ...] = (KIND_STATIC_R, KIND_VECTOR_ALL)

#: the two serving paths (common.sh F22_SERVING_PATH's vocabulary)
SERVING_PATH_LENS, SERVING_PATH_NATIVE = "lens", "lora_native"

#: the PIN REASONS the launcher logs and the dry run prints (`SERVING_PATH: … reason=…`) — the four the rule names on an external row,
#: a store reason carrying its SOURCE(S) in parentheses: `store:lens(disk)`, `store:lens(pins)`, `store:lens(disk+pins)`, likewise native
REASON_KIND_PREFIX = "kind:"                              # `kind:alphasteer`: rule (a)
REASON_STORE_LENS = "store:lens"                          # rule (b)  -> `store:lens(<sources>)`
REASON_UNPINNED_NO_STORE = "unpinned:no-store"            # rule (c): neither the disk nor the pins know the row
REASON_STORE_NATIVE = "store:native"                      # rule (d)  -> `store:native(<sources>)`
SOURCE_DISK, SOURCE_PINS = "disk", "pins"
SOURCES_SEP = "+"                                          # `disk+pins` when both sources name the (same) engine
#: …and the two a reader outside the rule's scope gets: a row whose object is not external (the pin governs external rows only),
#: and a reading with NO wave root (the grading host `plan`, which knows kinds but cannot read a cluster store — never printed by a launcher)
REASON_UNPINNED_NOT_EXTERNAL = "unpinned:not-external"
REASON_UNPINNED_STORE_UNREAD = "unpinned:store-unread"
#: …and the refusal a pins record earns when it ADMITS a leg (a known store) without recording its engine — never read as "no store"
REASON_REFUSED_RECORD_ENGINE_UNKNOWN = "refused:record-engine-unknown"
LAUNCHER_REASONS: tuple[str, ...] = (REASON_KIND_PREFIX + KIND_ALPHASTEER, REASON_STORE_LENS, REASON_UNPINNED_NO_STORE, REASON_STORE_NATIVE)


def kind_reason(kind: str) -> str:
    return REASON_KIND_PREFIX + kind


def store_reason(engine: str, sources: Sequence[str]) -> str:
    """`store:lens(disk+pins)` and friends: the engine's reason with the sources that named it, in the fixed order disk, pins."""
    order = [s for s in (SOURCE_DISK, SOURCE_PINS) if s in sources]
    return (REASON_STORE_LENS if engine == ENGINE_LENS else REASON_STORE_NATIVE) + "(" + SOURCES_SEP.join(order) + ")"


#: THE PINS RECORD's engine block (under `_meta`, which every store-pins reader skips as documentation — load_store_pins reads only
#: the leg -> tree entries): `_meta.serving_engine_by_leg[<leg key>] = {"engine": lens|lora_native, "producer_tree_sha": …, …}`,
#: written by stamp_store_pins_engines.py from the PULLED store of the leg (provenance beside the engine), never by hand
PINS_META_KEY, PINS_ENGINE_BLOCK, PINS_ENGINE_FIELD = "_meta", "serving_engine_by_leg", "engine"


#: the store's ENGINE MARKERS — mirrors, pinned by test: the identity's path segment (manifest.SERVING_PATHS["proxy_vllm"]), the
#: route's engine stamp (contracts.SERVED_ENGINE_FIELD; contracts.SERVING_ENGINES)
IDENTITY_PATH_LENS, IDENTITY_PATH_LORA, IDENTITY_PATH_PASSTHROUGH = "lens", "lora", "passthrough"
IDENTITY_PATHS: tuple[str, ...] = (IDENTITY_PATH_LENS, IDENTITY_PATH_LORA, IDENTITY_PATH_PASSTHROUGH)
SERVED_ENGINE_FIELD = "serving_engine"
SERVING_ENGINE_LORA_NATIVE, SERVING_ENGINE_LENS = "lora_native", "lens"
SERVING_ENGINES: tuple[str, ...] = (SERVING_ENGINE_LENS, SERVING_ENGINE_LORA_NATIVE)
#: the engine a marker names
ENGINE_LENS, ENGINE_NATIVE = "lens", "native"
#: the files read under a store root
STORE_EVIDENCE, STORE_CELLS_DIR, STORE_CELL_MANIFEST, ROUTE_KEY = "evidence.json", "cells", "manifest.json", "route"
IDENTITY_FIELD = "serving_identity"


@dataclass(frozen=True)
class StoreEngines:
    """What one store root says about the engine that served it: every marker read (file, field, value, engine-or-None)
    and the engines they name (sorted, unique). An EMPTY ``engines`` with readings is a store on record that names no
    engine — refused by the pin, never treated as absent."""
    root: str
    readings: tuple[tuple[str, str, str, Optional[str]], ...]

    @property
    def engines(self) -> tuple[str, ...]:
        return tuple(sorted({e for *_, e in self.readings if e is not None}))

    def describe(self) -> str:
        parts = [f"{f}:{field}={value!r}" + (f" -> {e}" if e else "") for f, field, value, e in self.readings]
        return f"{self.root}: " + ("; ".join(parts) if parts else "no engine marker")


def _refuse(msg: str) -> None:
    raise SystemExit(f"REFUSED (steer store pin): {msg}")


def _check_kind_tables() -> None:
    """The kind tables are consistent, checked at import (the node's python3, the grading host plan): every steer cell's kind is in EXACTLY
    one of ``LENS_ONLY_KINDS`` / ``NATIVELY_SERVABLE_KINDS``, and every pending adapter registration names a steer cell with the kind
    the mirror gives it. A table edit that leaves a kind unclassified stops the launcher before its first row is resolved, rather
    than falling through to a store read under a rule that was never declared for that kind."""
    kinds = set(STEER_CELL_KINDS.values())
    both = kinds & set(LENS_ONLY_KINDS) & set(NATIVELY_SERVABLE_KINDS)
    neither = kinds - set(LENS_ONLY_KINDS) - set(NATIVELY_SERVABLE_KINDS)
    if both or neither:
        _refuse(f"the kind tables do not partition the steer cells' kinds {sorted(kinds)}: in both LENS_ONLY_KINDS and "
                f"NATIVELY_SERVABLE_KINDS {sorted(both)}; in neither {sorted(neither)} — classify every kind in exactly one tuple")
    bad = {c: k for c, k in PENDING_ADAPTER_REGISTRATIONS.items() if STEER_CELL_KINDS.get(c) != k}
    if bad:
        _refuse(f"PENDING_ADAPTER_REGISTRATIONS {bad} does not match STEER_CELL_KINDS "
                f"({ {c: STEER_CELL_KINDS.get(c) for c in bad} }) — a pending registration names a steer cell with the mirror's kind")


_check_kind_tables()


def identity_engine(identity: Any, *, where: str) -> Optional[str]:
    """The engine a serving identity names: ``lens`` / ``native`` for the two steered paths, None for a passthrough
    identity (an unsteered leg — not a steer store's own cell, recorded but naming no engine). A string that is not
    ``<backend>+<path>+<mode>`` with a known path refuses by name."""
    if not isinstance(identity, str) or identity.count("+") != 2 or not all(identity.split("+")):
        _refuse(f"{where}: {IDENTITY_FIELD} {identity!r} is not '<backend>+<path>+<mode>' (manifest.parse_serving_identity)")
    path = identity.split("+")[1]
    if path not in IDENTITY_PATHS:
        _refuse(f"{where}: {IDENTITY_FIELD} {identity!r} names path {path!r}, not one of {list(IDENTITY_PATHS)}")
    return {IDENTITY_PATH_LENS: ENGINE_LENS, IDENTITY_PATH_LORA: ENGINE_NATIVE}.get(path)


def stamp_engine(stamp: Any, *, where: str) -> str:
    """The engine a PRESENT ``route.serving_engine`` stamp names; any other present value refuses by name (a null, a typo,
    an unknown engine — a present unknown stamp is a torn record, never the lens by default)."""
    if stamp not in SERVING_ENGINES:
        _refuse(f"{where}: {ROUTE_KEY}.{SERVED_ENGINE_FIELD} {stamp!r} is not one of {list(SERVING_ENGINES)}")
    return ENGINE_NATIVE if stamp == SERVING_ENGINE_LORA_NATIVE else ENGINE_LENS


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        _refuse(f"{path}: cannot be read as JSON ({type(e).__name__}: {e}) — a store on record that cannot be read is not followed")


def read_store_engines(root: "str | os.PathLike[str]") -> Optional[StoreEngines]:
    """The engine markers of the steer store at ``root`` — None when NO store is on record there (no ``evidence.json``,
    no ``cells/*/manifest.json``, no ``cells/*/evidence.json``: an absent root, or a root holding only ladder / RESULT
    bookkeeping from a job that never served). Every marker present is read; nothing is inferred from a directory's name."""
    root = Path(root)
    readings: list[tuple[str, str, str, Optional[str]]] = []
    files: list[Path] = []
    top = root / STORE_EVIDENCE
    if top.is_file():
        files.append(top)
    files += [Path(p) for p in sorted(glob.glob(str(root / STORE_CELLS_DIR / "*" / STORE_CELL_MANIFEST)))]
    files += [Path(p) for p in sorted(glob.glob(str(root / STORE_CELLS_DIR / "*" / STORE_EVIDENCE)))]
    if not files:
        return None
    for f in files:
        rel = str(f.relative_to(root))
        doc = _load(f)
        if not isinstance(doc, dict):
            _refuse(f"{f}: not a JSON object")
        if f.name == STORE_CELL_MANIFEST:
            route = doc.get(ROUTE_KEY)
            if not isinstance(route, dict):
                _refuse(f"{f}: no {ROUTE_KEY!r} block — a cell manifest names the route that served it")
            if IDENTITY_FIELD in route:
                readings.append((rel, f"{ROUTE_KEY}.{IDENTITY_FIELD}", str(route[IDENTITY_FIELD]),
                                 identity_engine(route[IDENTITY_FIELD], where=str(f))))
            if SERVED_ENGINE_FIELD in route:
                readings.append((rel, f"{ROUTE_KEY}.{SERVED_ENGINE_FIELD}", str(route[SERVED_ENGINE_FIELD]),
                                 stamp_engine(route[SERVED_ENGINE_FIELD], where=str(f))))
        elif IDENTITY_FIELD in doc:
            readings.append((rel, IDENTITY_FIELD, str(doc[IDENTITY_FIELD]), identity_engine(doc[IDENTITY_FIELD], where=str(f))))
    return StoreEngines(root=str(root), readings=tuple(readings))


@dataclass(frozen=True)
class PinsEngine:
    """One admitted leg's engine as the store-pins record carries it: the record file, the leg key, the engine (``lens`` /
    ``native``) and the entry's provenance block verbatim (the pulled store it was read from, its producer tree)."""
    pins_file: str
    leg_key: str
    engine: str
    entry: Mapping[str, Any]

    def describe(self) -> str:
        return f"{self.pins_file}:{PINS_META_KEY}.{PINS_ENGINE_BLOCK}[{self.leg_key}].{PINS_ENGINE_FIELD}={self.entry.get(PINS_ENGINE_FIELD)!r} -> {self.engine}"


def read_pins_engines(pins_files: Sequence["str | os.PathLike[str]"], leg_keys: Sequence[str]) -> tuple[PinsEngine, ...]:
    """The engines the store-pins records name for ``leg_keys`` (a row's admitted legs, any cap): one :class:`PinsEngine` per
    (file, key) that carries an entry. A pins file that cannot be read, a malformed engine block, or an entry whose engine is
    not a serving engine (``lens`` | ``lora_native``) refuses by name; a file with no engine block, or a key with no entry, is
    simply silent (the pins do not know the leg)."""
    out: list[PinsEngine] = []
    for f in pins_files:
        path = Path(f)
        if not path.is_file():
            _refuse(f"store-pins record {path} is not a file — the pin reads the repo's admitted store-pins records (they travel with the launch tree)")
        doc = _load(path)
        if not isinstance(doc, dict):
            _refuse(f"{path}: not a JSON object")
        block = (doc.get(PINS_META_KEY) or {}).get(PINS_ENGINE_BLOCK) if isinstance(doc.get(PINS_META_KEY), dict) else None
        if block is not None and not isinstance(block, dict):
            _refuse(f"{path}: {PINS_META_KEY}.{PINS_ENGINE_BLOCK} is not an object")
        for key in leg_keys:
            entry = (block or {}).get(key)
            if entry is None:
                if key in doc:
                    # ADMITTED (a leg -> tree entry: a KNOWN store) but its engine unrecorded — stamp_store_pins_engines.py --allow-missing
                    # keeps the admission and drops the engine. Never "no store": with the disk absent that would relaunch a known store
                    # natively. Refused by leg name until the record is stamped or the leg set aside.
                    _refuse(f"{REASON_REFUSED_RECORD_ENGINE_UNKNOWN}: {path}: leg {key} is ADMITTED (pinned {str(doc[key])[:8]}) but "
                            f"{PINS_META_KEY}.{PINS_ENGINE_BLOCK} carries no entry for it — a known store whose engine is unrecorded; stamp it "
                            "(stamp_store_pins_engines.py over its pulled store) or set the leg aside; it is never relaunched on a guessed engine")
                continue
            if not isinstance(entry, dict) or PINS_ENGINE_FIELD not in entry:
                _refuse(f"{path}: {PINS_META_KEY}.{PINS_ENGINE_BLOCK}[{key}] carries no {PINS_ENGINE_FIELD!r} (got {entry!r}) — re-stamp it with stamp_store_pins_engines.py")
            out.append(PinsEngine(pins_file=str(path), leg_key=key, engine=stamp_engine(entry[PINS_ENGINE_FIELD], where=f"{path}: {PINS_ENGINE_BLOCK}[{key}]"), entry=entry))
    return tuple(out)


def pin_for(kind: str, roots: Optional[Sequence["str | os.PathLike[str]"]], pins: Sequence[PinsEngine] = ()
            ) -> tuple[Optional[str], str, tuple[StoreEngines, ...]]:
    """The pin of an external row of object ``kind`` whose store(s) would live at ``roots`` (the DISK source) and whose admitted
    legs the store-pins records describe as ``pins`` (the PINS source, :func:`read_pins_engines`): ``(path, reason, stores read
    off the disk)``. ``roots`` None = no wave root to read under (the grading host plan): the disk is unread and only the pins speak —
    a natively servable kind with no pins entry is then ``unpinned:store-unread``, never "no store". A lens-only kind never reads
    either source (rule (a)). A disk store naming no engine, or the two sources (or the roots) together naming BOTH engines,
    refuses by name with every reading listed; ``unpinned:no-store`` only when neither source knows the row. A kind in NEITHER
    ``LENS_ONLY_KINDS`` nor ``NATIVELY_SERVABLE_KINDS`` refuses by name before any source is read (the pin has no rule for it)."""
    if kind in LENS_ONLY_KINDS:
        return SERVING_PATH_LENS, kind_reason(kind), ()
    if kind not in NATIVELY_SERVABLE_KINDS:
        _refuse(f"kind {kind!r} is neither lens-only ({list(LENS_ONLY_KINDS)}) nor natively servable ({list(NATIVELY_SERVABLE_KINDS)}) — "
                "the store-following pin has no rule for it; a steer cell's kind is classified in one of the two tuples, never read as 'not lens-only'")
    stores = () if roots is None else tuple(s for s in (read_store_engines(r) for r in roots) if s is not None)
    disk_engines = sorted({e for s in stores for e in s.engines})
    pins_engines = sorted({p.engine for p in pins})
    if stores and not disk_engines:
        _refuse(f"a {kind} row's store is on record but names NO engine — the pin follows the store and cannot: " + " | ".join(s.describe() for s in stores))
    if not stores and not pins:
        return None, (REASON_UNPINNED_STORE_UNREAD if roots is None else REASON_UNPINNED_NO_STORE), ()
    engines = sorted(set(disk_engines) | set(pins_engines))
    if len(engines) > 1:
        where = " | ".join([s.describe() for s in stores] + [p.describe() for p in pins])
        _refuse(f"a {kind} row's store on record carries BOTH engines' stamps (disk: {disk_engines or 'none'}; pins: {pins_engines or 'none'}) — "
                "a mixed-engine store, refused by name (no leg is written into it until a human sets it aside): " + where)
    sources = ([SOURCE_DISK] if disk_engines else []) + ([SOURCE_PINS] if pins_engines else [])
    if engines == [ENGINE_LENS]:
        return SERVING_PATH_LENS, store_reason(ENGINE_LENS, sources), stores
    return None, store_reason(ENGINE_NATIVE, sources), stores


def steer_cell_kind(cell: str) -> str:
    """The object kind of a steer cell; a cell outside the steer wave's cells refuses by name (the pin has no rule for it)."""
    if cell not in STEER_CELL_KINDS:
        _refuse(f"cell {cell!r} is not a steer cell ({sorted(STEER_CELL_KINDS)}) — the store-following pin has no rule for its kind")
    return STEER_CELL_KINDS[cell]


def pin_lines(path: Optional[str], reason: str) -> str:
    """The CLI's two-line form the launchers read (`row --serving-path --wave-root …`): the path (empty when unpinned), then the reason."""
    return f"{path or ''}\n{reason}\n"
