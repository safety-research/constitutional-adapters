#!/usr/bin/env python
"""Generate the explicit-cells serving configs the bcb-bfcl-200-widen lane's registry rows read
(the two llama8b views and the six qwen35_4b seed / paired-write views) — one per pair of
``contracts.BCBW_MODEL_VIEWS``.

Derivation credited to the tinycap-full22 lane's ``scripts/tinycap_full22_configs.py``
(ac69a7a2 / cd8927d6), SCOPED to this lane: exactly the (target, view) pairs of
``contracts.BCBW_MODEL_VIEWS`` — no GLM pair (zero GLM footprint) and no ``FULL22_*``
table; the donor's ``configs/tinycap/full22_*.yaml``, generator and test are NOT taken.

The roster it reads is the SHARED full22 roster (``configs/sdfbase_full22_roster.yaml``), which
carries every lane target (``LANE_TARGETS``; a roster lacking one refuses by name). History: until
the Llama lane's block landed on the shared roster (2026-09-12) the ``llama8b`` block lived in a lane
SUPPLEMENT (``configs/capeval/bcb_bfcl_200_widen_full22_supplement.yaml``, one derived cell per view,
bodies = lane copies of the Llama branch's cluster configs) merged by this generator; the shared block
was then allowed to supersede a faithful-subset supplement, and the supplement, its merge and the two
body copies were retired once the shared block was on main.

Why these exist. The launcher's registry rows resolve their cells through
``object_identity.route_cell_source`` and the standup through
``registry_adapter.traintest_cell_source``; both build the two-SI CARTESIAN registry
(``si_ids x class_a_arms``) UNLESS the config carries ``explicit_cells`` (S1-e's
``object_identity`` hand-port makes the preflight dispatch like the standup). A
``constv2-*`` cell can therefore never be resolved from ``configs/two_si_*.yaml`` — the
standup would enumerate it ABSENT after paying for the GPU. So each lane row's ``cfg`` is
one of the files this script writes: the cluster's ``configs/sdfbase_full22_<tag>.yaml``
body byte-for-byte as data (model pins, battery keys, out_dir — the driver's hashed
``controls_gen_config`` reads that body), EXCEPT

  * ``explicit_cells`` = [none, prompted__safety_refusal, *the LANE roster's other runnable
    prompted cells for the row, *the view's object cells in roster order] —
    ``f22_roster.served_cells`` is the ONE reader of what a (target, view) serves; the
    prompted safety baseline is listed on EVERY view as the registry's object-free
    same-wave anchor (the lane's gemma instruct-direct wave RUNS it; the other waves list
    it and do not launch it — the launcher plans only ``--cells``); a prompt-only arm the
    paper wants beyond it (the constitution SI, ``prompted__si_constitution``) has no object
    and no pin; it enters the list from the LANE roster's runnable prompted entries
    (``lane_prompted_cells``). The SHARED roster may list such a cell too (the sampled lane
    landed ``prompted__si_constitution: null`` on the gemma, qwen and nemotron views,
    2026-09-12): ADMIT AND HOLD — a shared-roster prompted cell the lane RUNS on the row is
    served once; one the lane does not run is DECLARED in the header as HELD and stays OUT
    of ``explicit_cells``, so the row's next job serves exactly what it serves today, and
    the hold is released by making the lane roster's entry for it runnable
    (``held_prompted_cells``);
  * ``explicit_cells`` ALSO admits the LANE roster's runnable entries DECLARED ``kind: lane_extra`` that the
    shared view does not serve — LANE EXTRAS (the Arditi refusal-steer
    ``arditi_allpos``, staged by that lane as an extra bare-cell dir INSIDE each ``s<s>`` bank and
    pinned under ``qwen35_4b/ar_c0p3__arditi_allpos`` in the pins file), appended AFTER the shared
    view's objects (``lane_extra_cells``); the SAME admit-and-hold rule as the prompted cells — an
    extra whose lane entry is not runnable is DECLARED in the header as HELD and stays OUT of the
    list (``held_extra_cells``), so a bank need not hold the object until the lane runs the cell;
  * ``defenses_dir`` = ``contracts.full22_objects_dir(target, view)`` — ABSOLUTE on the
    Modal volume, so ``resolve_defenses_dir`` rule 1 makes the standup, the driver and
    the preflight read ONE bank (== the row's ``objects_dir``; a test asserts equality);
  * ``multi_layer_arms`` (``cell_controls.MULTI_LAYER_ARMS_KEY``; cd8927d6's derivation)
    for the objects fitted on SEVERAL layers — Nemotron-3-Nano-4B's three LoRA cells on
    the hybrid's four attention layers ``[12, 17, 24, 32]`` (``target_layers`` in their
    staged sidecars) — so F9 requires the declared set to EQUAL the object's layers.

The header records the roster sha256, the body config sha256 and, per object cell, the
staged id, the lane kind (the staged-id prefix) and the object pin
(``configs/sdfbase_full22_pins.json``), so a reader of the file sees what the bank must hold.

THE FRONT-DOOR SCOPE. A models-table key with phase views (reproduce/models.yaml;
``lib/models.py front_door_keys``) serves its one steered wave on Modal through ``reproduce/07_bcb_bfcl.sh``, whose launcher row
(``contracts.MODELS[<key>]``) reads an explicit-cells config too — ``contracts.frontdoor_config_path(key, view)`` =
``configs/tinycap/frontdoor_<key>_<view>.yaml``. It is a PARAMETERISATION of the same render with its OWN row source: the models table
(``launcher_cells`` — the ONE mapping stage 07 launches with — gives the ORDERED ``explicit_cells``: none, then the view's cells in the
table's order as launcher tokens) and the SHARED roster's block for the key (``05_stage_roster.sh --step roster`` prints it; its prompted
cells are the ``prompted__*: null`` entries, its object cells ``f22_roster.served_cells``; the two sources must agree or the render
refuses), ``defenses_dir`` = ``contracts.MODELS[key].objects_dir`` (the key's OWN bank, outside the full22 root), the battery body =
``configs/sdfbase_full22_<key>_<view>.yaml`` (``05 --step configs``) byte-for-byte below the marker, and every object cell's pin from
``configs/sdfbase_full22_pins.json[<key>]`` (``05 --step pins``). Never ``BCBW_SLUG_OF`` or the lane rosters (``lane_prompted_cells``
would hold ``prompted__si_constitution`` out of the list and the lane helpers KeyError on a non-lane key). The render REFUSES by name
when the roster block, the pins block or the battery body is absent or an object cell is unpinned — never an UNPINNED header. The
lane's 19 files are untouched by it; ``--check`` covers every front-door file that exists beside them.

Usage:
  python scripts/bcbw_configs.py                      # write configs/tinycap/bcbw_*.yaml (the lane's 19)
  python scripts/bcbw_configs.py --front-door <key>   # write the front door's configs/tinycap/frontdoor_<key>_<wave>.yaml (refuses by name without its inputs)
  python scripts/bcbw_configs.py --check              # exit 1 on drift of the 19 lane files and of every front-door file that exists
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "src"))
from f22_roster import is_prompted, served_cells, staged_cells  # noqa: E402

from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import widen_roster as W  # noqa: E402
from sisteer.eval.cell_controls import MULTI_LAYER_ARMS_KEY  # noqa: E402

ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
PINS = REPO / "configs" / "sdfbase_full22_pins.json"
OUT_DIR = REPO / "configs" / "tinycap"
#: The marker that separates the cluster config's header from its data; everything
#: from it on is copied verbatim (scripts/sdfbase/make_full22_configs.py writes it).
BODY_MARKER = "# -- everything below:"
#: The one prompted baseline every lane view lists (the harness spells it prompted__safety).
PROMPTED_CELL = C.CELL_PROMPTED_SAFETY
#: The staged-id prefixes (the roster's staging layout, ``<lane>__<cell_id>``): ``derived_itdoc``
#: is the Llama lane's spelling of an INSTRUCT-DIRECT derived object (its ``it`` view).
#: ``itdoc64`` / ``derived_itdoc64`` are the batch-64 re-fit lineage (2026-09-13): the old ``itdoc__`` ids keep
#: the batch-8 objects whose shas historical stores carry, so the re-fit parents and their derived difference take
#: their own prefix, and ``derived_<parent prefix>__`` keeps a label from naming a lineage other than its sha. An
#: unknown prefix refuses here.
#: ``base64s<s>`` / ``derived_base64s<s>`` / ``paired_eb64s<s>`` (2026-09-22): the batch-64
#: THREE-SEED Qwen lineage (the reseed fits at effective batch 64 on Qwen/Qwen3.5-4B-Base, seed s; their LoRA differences;
#: the paired-write PSR differences) — own prefixes, one per seed, so a label names its seed AND its lineage. ``ar_c0p3``:
#: the Arditi all-positions refusal direction at c = 0.3 (configs/arditi_allpos_pins.json's staged id), a LANE
#: EXTRA of the seed banks. ``paired_eb8`` / ``paired_eb64`` (2026-09-23: the all-models
#: one-seed family): the composed paired-write difference ReLU of a non-Qwen family's base-lane constitution and control
#: PSR (kind psr_paired_write; the relu-direct-difference-sweep's multi-model paired lane), the number naming the PARENTS'
#: effective batch — 8 on gemma4e4b / nemotron3nano4b, 64 on llama8b — with no seed suffix (one seed per family).
LANE_KINDS: Tuple[str, ...] = ("base", "itdoc", "si12", "derived", "derived_itdoc", "itdoc64", "derived_itdoc64",
                               "base64s1", "base64s2", "base64s3", "derived_base64s1", "derived_base64s2", "derived_base64s3",
                               "paired_eb64s1", "paired_eb64s2", "paired_eb64s3", "ar_c0p3",
                               "paired_eb8", "paired_eb64",
                               # paper-companion-repo dose 0.75: `d0p75l__<cell>` — a dose-scaled copy (x 0.75,
                               # scripts/sdfbase/scale_psr_dose.py) of a LeCun-lane (`lecun__` / `lecun_derived__`) parent; the `l` is
                               # the lane marker (reproduce/lib/models.py DOSE_LANE_MARKERS), the copy carries its parent's sidecar
                               # paper-companion-repo dose-1.0 random-init ReLU pair: `lecun__<cell>` — the
                               # LeCun-init refits of the s2rl view (bank full22_lecun; the parents of the d0p75l dose copies), served as-is;
                               # `lecun_derived__<cell>` their composition (the view's third cell: in the bank and the registry, never launched)
                               "d0p75l", "d1p25", "lecun", "lecun_derived",
                               # the front-door scope: the worked example's refusal direction is staged `arditi__<rung>__<harness id>`
                               # (reproduce/lib/models.py ARDITI_LANE: the lane is the prefix, the rung rides inside the cell_id), beside `base__` / `derived__`
                               "arditi")
#: demo-import (2026-10-12): a front-door DOSE view's copies are staged `d<tok>[<lane marker>]__<cell_id>` (reproduce/lib/models.py staged_id —
#: `d0p75__constv2-mix-full__lora_r4_ce_minus_ctl` for the worked example's 0.75 rungs; the paper's `d0p75l` / `d1p25` lanes above are two
#: spellings of the same grammar), so a dose lane is admitted by its SHAPE rather than listed per dose token
DOSE_LANE_RE = re.compile(r"^d(?:m?[0-9]+(?:p[0-9]+)?)[a-z]*$")
#: The objects fitted on SEVERAL layers, per target, keyed by config cell id — emitted into the
#: generated config as ``multi_layer_arms``. Source: the staged bank's sidecars
#: (``train_budget_meta.json`` / ``derived_meta.json`` ``target_layers``; the volume survey):
#: Nemotron-3-Nano-4B's LoRA objects target the hybrid's four attention layers. gemma's LoRA
#: objects are all-layer fits (0..41), qwen's derived LoRA targets [3, 7, ..., 31] and llama's
#: derived LoRAs are all-layer fits (0..31; ``derived_meta.json`` ``layers``, rank 4, both views) —
#: the body config's placeholder ``layer: 15`` is a member of each, so none needs a declaration.
MULTI_LAYER_ARMS: Dict[str, Dict[str, List[int]]] = {
    "nemotron3nano4b": {
        "constv2-mix-full__lora_r4_ce": [12, 17, 24, 32],
        "constv2ctl2-full__lora_r4_ce": [12, 17, 24, 32],
        "constv2-mix-full__lora_r4_ce_minus_ctl": [12, 17, 24, 32],
    },
}


#: The targets the lane's six views need in the roster (``contracts.BCBW_MODEL_VIEWS``).
LANE_TARGETS: Tuple[str, ...] = tuple(dict.fromkeys(t for t, _ in C.BCBW_MODEL_VIEWS))


def load_roster(path: Path = ROSTER) -> dict:
    """The shared full22 roster. A roster lacking a lane target REFUSES naming it (a silent gap would fail deep
    inside ``f22_roster`` on a missing target, after the caller has moved on)."""
    roster = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [t for t in LANE_TARGETS if t not in roster["targets"]]
    if missing:
        raise SystemExit(f"{path}: the shared full22 roster lacks the lane target(s) {missing} — nothing is generated without them")
    return roster


def load_pins(path: Path = PINS) -> dict:
    """The full22 object pins. A MISSING file REFUSES (code review): it used to read as ``{}``,
    which printed every cell's header as UNPINNED and let a pin-less config be generated."""
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the full22 object pins file is missing — every lane cell must be pinned "
                                "(configs/sdfbase_full22_pins.json); nothing is generated unpinned")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lane_kind_of_staged_id(staged_id: str) -> str:
    """``base__…`` -> ``base``, ``itdoc__…`` -> ``itdoc``, ``si12__…`` -> ``si12``,
    ``derived__…`` -> ``derived``, ``derived_itdoc__…`` -> ``derived_itdoc``, ``itdoc64__…`` -> ``itdoc64``,
    ``derived_itdoc64__…`` -> ``derived_itdoc64``, the seed lineage ``base64s<s>__`` / ``derived_base64s<s>__`` /
    ``paired_eb64s<s>__`` and the Arditi extra ``ar_c0p3__``; anything else refuses."""
    prefix, sep, _ = staged_id.partition("__")
    if not sep or not (prefix in LANE_KINDS or DOSE_LANE_RE.match(prefix)):
        raise SystemExit(f"staged id {staged_id!r} carries no lane prefix in {list(LANE_KINDS)} and is not a dose lane (`d<tok>[<marker>]__`)")
    return prefix


def object_cells(roster: dict, target: str, view: str) -> List[str]:
    """The view's OBJECT cells in roster order (served_cells minus the prompted baselines)."""
    return [cid for cid, sid in served_cells(roster, target, view).items() if sid is not None]


#: The LANE roster (``configs/capeval/bcb_bfcl_200_widen_roster.yaml``), read for its runnable PROMPTED cells: a
#: prompt-only arm the paper wants (the constitution SI) has no object and no pin, so whether a row SERVES it is
#: the lane roster's call alone; the SHARED full22 roster may DECLARE it on a view (the sampled lane did so on
#: 2026-09-12) — see ``held_prompted_cells``.
LANE_ROSTER = REPO / W.ROSTER_REL
#: the FULL lane roster (configs/capeval/bcb_bfcl_full_roster.yaml) is read BESIDE the 200 roster for the rows that
#: only IT serves — a wave with no 200-roster sibling would otherwise never have its runnable prompted cells enter
#: the row's served list. The two rosters are read as a UNION of waves keyed by name, the 200 roster WINNING a
#: same-named wave (the full roster's copies of the 200 waves differ only where the full lane holds a cell back —
#: gemma4e4b_it's prompted cells, blocked there — so every pre-existing row's list is byte-identical to before;
#: tests/test_bcbw_configs.py pins the byte identity of every generated file). On this tree every full-roster wave has a 200 sibling.
LANE_ROSTER_FULL = REPO / W.ROSTER_FULL_REL


def load_lane_rosters(path_200: Path = LANE_ROSTER, path_full: Path = LANE_ROSTER_FULL) -> dict:
    """The lane rosters as ONE wave table: ``{"waves": {**full.waves, **two_hundred.waves}}`` — the 200 roster wins a same-named
    wave; a wave only the full roster carries joins. Only ``waves`` is read by the helpers below (``widen_roster.waves`` / ``wave`` /
    ``entries`` / ``runnable_cells``), so no header field of either roster is merged or implied."""
    two_hundred = W.load_roster(path_200)
    full = W.load_roster(path_full) if Path(path_full).is_file() else {"waves": {}}
    return {"waves": {**(full.get("waves") or {}), **(two_hundred.get("waves") or {})},
            "_sources": {"200": str(path_200), "full": str(path_full)}}


def _lane(lane_roster: Optional[dict]) -> dict:
    """The lane roster a helper reads: the one the caller loaded, else a fresh load of the UNION (:func:`load_lane_rosters`) —
    ``render_config`` and ``main`` load ONCE and thread it, so a generation parses the files once, not once per helper call."""
    return lane_roster if lane_roster is not None else load_lane_rosters()


def _row_waves(lane: dict, slug: str) -> List[str]:
    """The lane waves whose ``model_row`` is registry slug ``slug``, in roster order — the ONE walk every
    per-row reader of the lane roster starts from."""
    return [w for w in W.waves(lane) if W.wave(lane, w).get("model_row") == slug]


def lane_prompted_cells(target: str, view: str, *, lane_roster: Optional[dict] = None) -> List[str]:
    """The lane roster's RUNNABLE prompted cells (``prompted__*``, in roster order) of the wave(s) whose
    ``model_row`` is the registry slug of ``(target, view)`` — the cells the wrapper launches on that row
    beside the object cells; [] when no wave serves the row."""
    lane = _lane(lane_roster)
    out: List[str] = []
    for wname in _row_waves(lane, C.BCBW_SLUG_OF[(target, view)]):
        for cell in W.runnable_cells(lane, wname):
            if C.parse_launcher_cell(cell).kind == C.CELL_KIND_PROMPTED and cell not in out:
                out.append(cell)
    return out


def shared_prompted_cells(roster: dict, target: str, view: str) -> List[str]:
    """The SHARED roster's prompted cells on the view beyond the safety anchor (``prompted__<si>: null``, roster
    order), each validated against the launcher grammar — a shared-roster typo (an SI with no harness translation)
    refuses HERE, at generation, whether or not the lane runs the cell."""
    out: List[str] = []
    for cid in served_cells(roster, target, view):
        if is_prompted(cid) and cid != PROMPTED_CELL:
            try:
                C.parse_launcher_cell(cid)
            except C.CellNameError as e:
                raise SystemExit(f"{target}/{view}: the shared roster serves prompted cell {cid!r}, which the "
                                 f"launcher grammar refuses — {e}") from e
            out.append(cid)
    return out


def lane_prompted_status(target: str, view: str, cid: str, *, lane_roster: Optional[dict] = None) -> str:
    """Where the LANE roster stands on cell ``cid`` for the row (a prompted cell or a lane extra):
    ``'<status> (wave <w>, entry <e>)'`` when a wave whose ``model_row`` is the row's slug names it, ``'no entry in
    wave <w>'`` when the row's wave(s) never do, ``'no lane wave serves this row'`` otherwise — the standing the
    generated header records beside a HELD cell."""
    lane = _lane(lane_roster)
    waves = _row_waves(lane, C.BCBW_SLUG_OF[(target, view)])
    if not waves:
        return "no lane wave serves this row"
    for w in waves:
        for eid, e in W.entries(lane, w).items():
            if e.get("cell_id") == cid:
                return f"{e.get('status') or 'unstated'} (wave {w}, entry {eid})"
    return "no entry in wave " + " / ".join(waves)


def held_prompted_cells(roster: dict, target: str, view: str, *, lane_roster: Optional[dict] = None) -> List[str]:
    """ADMIT AND HOLD: the shared roster's prompted cells on the view that the LANE roster does not RUN on the
    row. They are DECLARED in the generated header (with the lane roster's standing on each,
    :func:`lane_prompted_status`) and kept OUT of ``explicit_cells``, so the registry — and the row's next job —
    serves exactly what it serves today. The hold is released on the lane's clock, by making the lane roster's
    entry for the cell runnable: it then enters the served list through :func:`lane_prompted_cells`."""
    lane = set(lane_prompted_cells(target, view, lane_roster=lane_roster))
    return [c for c in shared_prompted_cells(roster, target, view) if c not in lane]


#: the lane roster's DECLARATION of a lane extra (an entry's ``kind``): an object another lane stages INSIDE the wave's bank
#: beyond the shared roster's view (the Arditi refusal-steer). Declared, never inferred — a queued placeholder for a
#: cell nobody staged (the gemma waves' ``safety_lora``) is NOT an extra and must not enter a header as one.
LANE_EXTRA_KIND = "lane_extra"


def lane_fitted_entries(target: str, view: str, *, lane_roster: Optional[dict] = None) -> Dict[str, Tuple[str, Optional[dict]]]:
    """``{cell_id: (status, object block or None)}`` for every entry DECLARED ``kind: lane_extra`` on a lane wave whose
    ``model_row`` is the registry slug of ``(target, view)`` — runnable or not, in lane-roster order (the first entry
    naming a cell wins; the lane reader refuses two runnable entries on one cell anyway). A declared extra whose cell
    is not a fitted cell refuses: an extra is an OBJECT the lane stages."""
    lane = _lane(lane_roster)
    out: Dict[str, Tuple[str, Optional[dict]]] = {}
    for wname in _row_waves(lane, C.BCBW_SLUG_OF[(target, view)]):
        for eid, e in W.entries(lane, wname).items():
            if e.get("kind") != LANE_EXTRA_KIND:
                continue
            cid = e.get("cell_id")
            if cid is None or C.parse_launcher_cell(str(cid)).kind != C.CELL_KIND_FITTED:
                raise SystemExit(f"{target}/{view}: lane roster wave {wname!r} entry {eid!r} is declared {LANE_EXTRA_KIND!r} but "
                                 f"names no fitted cell ({cid!r}) — a lane extra is an object the lane stages")
            if cid in out:
                continue
            obj = e.get("object")
            out[str(cid)] = (str(e.get("status") or ""), dict(obj) if isinstance(obj, Mapping) else None)
    return out


def _lane_extras(roster: dict, target: str, view: str, *,
                 lane_roster: Optional[dict] = None) -> Tuple[Dict[str, str], List[str]]:
    """The lane's DECLARED extras of the row that the SHARED roster's view does not serve, partitioned by the lane
    roster's standing — ``({runnable cell_id: staged_id}, [held cell_id])``: one walk, one rule for both halves. A
    runnable extra without an ``object.staged_id`` refuses here (the lane reader requires the block too)."""
    shared = set(served_cells(roster, target, view))
    runnable: Dict[str, str] = {}
    held: List[str] = []
    for cid, (status, obj) in lane_fitted_entries(target, view, lane_roster=lane_roster).items():
        if cid in shared:
            continue
        if status != W.STATUS_RUNNABLE:
            held.append(cid)
            continue
        sid = (obj or {}).get("staged_id")
        if not sid:
            raise SystemExit(f"{target}/{view}: the lane roster runs the extra cell {cid!r} without an object.staged_id — "
                             "a lane extra is pinned through its staged id or it is not served")
        runnable[cid] = str(sid)
    return runnable, held


def lane_extra_cells(roster: dict, target: str, view: str, *, lane_roster: Optional[dict] = None) -> Dict[str, str]:
    """LANE EXTRAS: ``{cell_id: staged_id}`` for the lane roster's RUNNABLE fitted cells of the row that the
    SHARED roster's view does not serve — objects the lane stages INSIDE the wave's bank as extra bare-cell
    dirs (the Arditi refusal-steer ``arditi_allpos`` in the ``s<s>`` banks)."""
    return _lane_extras(roster, target, view, lane_roster=lane_roster)[0]


def held_extra_cells(roster: dict, target: str, view: str, *, lane_roster: Optional[dict] = None) -> List[str]:
    """The lane roster's NON-runnable fitted cells of the row that the shared view does not serve — DECLARED in the
    generated header as HELD (with the lane roster's standing) and kept OUT of ``explicit_cells``, so the bank need
    not hold the object until the lane runs the cell (the prompted cells' admit-and-hold rule, applied to extras)."""
    return _lane_extras(roster, target, view, lane_roster=lane_roster)[1]


def explicit_cells(roster: dict, target: str, view: str, *, lane_roster: Optional[dict] = None) -> List[str]:
    """``[none, prompted__safety_refusal, *the lane roster's other runnable prompted cells, *object cells,
    *lane extras]`` — the generated config's list. The prompted cells beyond the safety anchor are the LANE
    roster's runnable ones for the row (:func:`lane_prompted_cells`), once each; a prompted cell the SHARED roster
    lists on the view is served only when the lane runs it, else HELD (:func:`held_prompted_cells`) — declared,
    never listed here; the lane's runnable fitted extras (:func:`lane_extra_cells`) come last, the held ones are
    declared (:func:`held_extra_cells`)."""
    shared_prompted_cells(roster, target, view)           # the shared roster's prompted cells must parse (loud on a typo)
    prompted = [PROMPTED_CELL] + [c for c in lane_prompted_cells(target, view, lane_roster=lane_roster) if c != PROMPTED_CELL]
    return [C.ROUTE_NONE_CELL, *prompted, *object_cells(roster, target, view),
            *lane_extra_cells(roster, target, view, lane_roster=lane_roster)]


def staged_ids_by_cell(roster: dict, target: str, view: str, *, lane_roster: Optional[dict] = None) -> Dict[str, str]:
    """``{config cell id: staged id}`` over the view's shared object cells AND the lane's runnable extras."""
    return {**staged_cells(roster, target, view), **lane_extra_cells(roster, target, view, lane_roster=lane_roster)}


def pins_by_cell(target: str, view: str, *, roster: Optional[dict] = None,
                 pins: Optional[dict] = None, lane_roster: Optional[dict] = None) -> Dict[str, Optional[str]]:
    """``{config cell id: object_sha256}`` for the view's object cells — the shared view's and the lane's runnable
    extras (None when the pins file has no entry) — what ``object_identity``'s ``pin_match`` compares against."""
    roster = roster if roster is not None else load_roster()
    pins = pins if pins is not None else load_pins()
    tp = pins.get(target, {})
    return {cid: (tp.get(sid) or {}).get("object_sha256")
            for cid, sid in staged_ids_by_cell(roster, target, view, lane_roster=lane_roster).items()}


def _sub_line(body: str, key_regex: str, new_line: str) -> str:
    n = len(re.findall(key_regex, body, flags=re.MULTILINE))
    if n != 1:
        raise SystemExit(f"expected exactly 1 line matching {key_regex!r}, found {n}")
    return re.sub(key_regex, new_line, body, count=1, flags=re.MULTILINE)


def target_served_cells(roster: dict, target: str) -> List[str]:
    """Every config cell id ANY view of ``target`` serves (the union over the shared roster's views, roster order) —
    the set a per-target ``MULTI_LAYER_ARMS`` declaration is held to."""
    out: List[str] = []
    for view in roster["targets"][target]["views"]:
        for cid in served_cells(roster, target, view):
            if cid not in out:
                out.append(cid)
    return out


def multi_layer_block(target: str, cells: List[str], *, target_cells: Optional[List[str]] = None) -> str:
    """The ``multi_layer_arms`` header block for ``target`` (empty when it declares none), emitted for the declared
    cells THIS view serves. The table is per TARGET and a target's views serve different subsets (nemotron's ``s1p``
    view serves one of its three LoRA cells; b64-capability-agentic-queue 2026-09-23), so a declaration is held to the
    cells SOME view of the target serves (``target_cells``, :func:`target_served_cells`; the view's own cells when the
    caller passes none — the pre-2026-09-23 behaviour): a declaration naming a cell NO view serves refuses (a typo in
    the table), one naming a cell another view serves is simply not emitted here."""
    declared = MULTI_LAYER_ARMS.get(target) or {}
    stray = sorted(set(declared) - set(target_cells if target_cells is not None else cells))
    if stray:
        raise SystemExit(f"MULTI_LAYER_ARMS[{target!r}] names cells no view of the target serves: {stray}")
    if not any(c in declared for c in cells):
        return ""
    out = ("\n# -- declared multi-layer objects (cd8927d6 / S1-f): F9 requires each set to EQUAL the object's "
           "target_layers ---\n")
    out += f"{MULTI_LAYER_ARMS_KEY}:\n"
    for cid in cells:
        if cid in declared:
            out += f"  {cid}: [{', '.join(str(int(x)) for x in declared[cid])}]\n"
    return out


def render_config(roster: dict, target: str, view: str, *, pins: Optional[dict] = None,
                  lane_roster: Optional[dict] = None) -> str:
    """The generated config text for one (target, view). ``lane_roster``: the LANE roster, loaded here once when
    the caller did not pass it, and threaded through every helper that reads it."""
    pins = pins if pins is not None else load_pins()
    lane = _lane(lane_roster)
    t = roster["targets"][target]
    v = t["views"][view]
    src = REPO / v["config"]
    src_text = src.read_text(encoding="utf-8")
    if BODY_MARKER not in src_text:
        raise SystemExit(f"{src} has no {BODY_MARKER!r} marker — not a generated cluster config")
    body = src_text[src_text.index(BODY_MARKER):]
    body = re.sub(r"^# -- everything below: .*$",
                  f"# -- everything below: {src.name}'s data, byte-for-byte, EXCEPT defenses_dir --",
                  body, count=1, flags=re.MULTILINE)
    objects_dir = C.full22_objects_dir(target, view)
    body = _sub_line(body, r"^defenses_dir: .*$", f"defenses_dir: {objects_dir}")
    cells = explicit_cells(roster, target, view, lane_roster=lane)
    shared_staged = staged_cells(roster, target, view)
    extras, held_extras = _lane_extras(roster, target, view, lane_roster=lane)
    staged = {**shared_staged, **extras}                   # == staged_ids_by_cell(roster, target, view)
    cell_pins = pins_by_cell(target, view, roster=roster, pins=pins, lane_roster=lane)
    slug = C.BCBW_SLUG_OF[(target, view)]
    row = C.model_row(slug)
    if row.objects_dir != objects_dir or row.cfg != C.bcbw_config_path(target, view):
        raise SystemExit(f"row {slug!r} disagrees with the generator: objects_dir {row.objects_dir!r} vs "
                         f"{objects_dir!r}, cfg {row.cfg!r} vs {C.bcbw_config_path(target, view)!r}")
    target_cells = target_served_cells(roster, target)
    mla_block = multi_layer_block(target, cells, target_cells=target_cells)
    mla_line = ("#   * `multi_layer_arms` = the declared layer set of every object fitted on several layers "
                "(F9 equality).\n" if mla_block else "")
    lane_prompted = [c for c in cells if is_prompted(c) and c != PROMPTED_CELL]
    shared_prompted = shared_prompted_cells(roster, target, view)
    held = held_prompted_cells(roster, target, view, lane_roster=lane)
    cell_lines = []
    for cid in cells:
        if cid in staged:
            sha = cell_pins.get(cid) or "UNPINNED (not in sdfbase_full22_pins.json)"
            extra = "  (a LANE EXTRA: the lane roster's runnable entry; staged INSIDE this bank by that lane)" if cid in extras else ""
            cell_lines.append(f"#       {cid:<48} lane {lane_kind_of_staged_id(staged[cid]):<8} <- {staged[cid]}{extra}")
            cell_lines.append(f"#       {'':<48} object_sha256 {sha}")
        else:
            what = ("the same-wave bare anchor" if cid == C.ROUTE_NONE_CELL
                    else "the SI text as the system prompt; no object"
                    + ("; a runnable prompted cell of the LANE roster (configs/capeval/bcb_bfcl_200_widen_roster.yaml)"
                       if cid in lane_prompted else "")
                    + ("; also declared by the shared roster (configs/sdfbase_full22_roster.yaml)"
                       if cid in shared_prompted else ""))
            cell_lines.append(f"#       {cid:<48} ({what})")
    if held:
        cell_lines.append("#     HELD — declared by the shared roster (configs/sdfbase_full22_roster.yaml) on this view but NOT run by the lane")
        cell_lines.append("#     roster on this row, so NOT in explicit_cells (ADMIT AND HOLD): the registry")
        cell_lines.append("#     serves exactly the list above; the hold is released by making the lane roster's entry for the cell runnable.")
        for cid in held:
            cell_lines.append(f"#       {cid:<48} (HELD; lane roster: {lane_prompted_status(target, view, cid, lane_roster=lane)})")
    if held_extras:
        cell_lines.append("#     HELD LANE EXTRAS — fitted cells the LANE roster (configs/capeval/bcb_bfcl_200_widen_roster.yaml) names on this row")
        cell_lines.append("#     beyond the shared view but does NOT run: NOT in explicit_cells, so this")
        cell_lines.append("#     bank need not hold the object yet; the hold is released by making the lane roster's entry runnable (regenerate).")
        for cid in held_extras:
            cell_lines.append(f"#       {cid:<48} (HELD EXTRA; lane roster: {lane_prompted_status(target, view, cid, lane_roster=lane)})")
    roster_line = "configs/sdfbase_full22_roster.yaml"
    source_lines = f"#   roster  configs/sdfbase_full22_roster.yaml  {sha256_file(ROSTER)}\n"
    source_lines += f"#   body    {v['config']}  {sha256_file(src)}"
    header = f"""# {C.bcbw_config_path(target, view).rsplit('/', 1)[1]} — the bcb-bfcl-200-widen lane's cells on Modal
# the EXPLICIT-cells serving config the tinycap
# launcher's registry row `{slug}` reads (`contracts.MODELS["{slug}"].cfg`). GENERATED by
# scripts/bcbw_configs.py from {roster_line} (target {target}, view {view});
# derivation credited to the tinycap-full22 lane's scripts/tinycap_full22_configs.py, scoped to
# this lane's {len(C.BCBW_MODEL_VIEWS)} (target, view) pairs (contracts.BCBW_MODEL_VIEWS) — never hand-edited;
# tests/test_bcbw_configs.py requires byte-identity.
#
# Sources (sha256 at generation; a change regenerates this file):
{source_lines}
#
# DERIVED from that cluster config: every key below the body marker is its data, byte-for-byte
# (model / model_revision / layer / max_len / the whole battery body, which the driver's hashed
# controls_gen_config reads), EXCEPT —
#
#   * `explicit_cells` = [none, prompted__safety_refusal,{(' ' + ', '.join(lane_prompted) + ' (the lane roster' + ("'s" if len(lane_prompted) == 1 else "'s") + ' runnable prompted cell' + ('' if len(lane_prompted) == 1 else 's') + '),') if lane_prompted else ''} the {len(shared_staged)} object cell(s) of the
#     roster's `{view}` view in roster order{(', the ' + str(len(extras)) + ' lane extra(s) ' + ', '.join(extras)) if extras else ''}] (f22_roster.served_cells is the ONE reader of what a
#     (target, view) serves; the registry is EXACTLY this list — registry_from_explicit_cells
#     validates every object's artifact + its one identity sidecar at build time, on the CPU,
#     before any GPU minute). Lane kind = the roster's staged-id prefix:
{chr(10).join(cell_lines)}
#     Which of these the lane LAUNCHES is the lane roster's call (configs/capeval/bcb_bfcl_200_widen_roster.yaml,
#     runnable entries per wave) — the launcher plans only `--cells`; the rest are the registry's anchors.
#   * `defenses_dir` = {objects_dir} — the row's objects_dir, ABSOLUTE
#     on the Modal volume (resolve_defenses_dir rule 1), so the standup, the driver and the
#     preflight read ONE bank: the staged copy of cluster A's dereferenced
#     $SHARED/exp/objects/sdfbase/full22/{target}/cells_view/{view}, sha-verified against the pins above.
{mla_line}#
# serving target: {t['model']} @ {t['model_revision']}

# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---
explicit_cells:
"""
    for cid in cells:
        header += f"- {cid}\n"
    header += mla_block
    header += "\n"
    return header + body


# ----------------------------------------------------------------------------------------------------------------------------- #
# the FRONT-DOOR scope: the models table + the shared roster's block + the pins block, one config                                        #
# ----------------------------------------------------------------------------------------------------------------------------- #

MODELS_TABLE = REPO / "reproduce" / "models.yaml"
MODELS_READER = REPO / "reproduce" / "lib" / "models.py"
_MODELS_MOD: Dict[str, object] = {}


class FrontDoorRefusal(SystemExit):
    """The front-door render refuses BY NAME (exit 2): an input the committed tree does not carry yet, or two sources that disagree."""

    def __init__(self, key: str, view: Optional[str], msg: str):
        super().__init__(f"REFUSED: front door {key}" + (f"/{view}" if view else "") + f": {msg}")


def models_module(reader: Path = MODELS_READER):
    """reproduce/lib/models.py, the models table's ONE reader (PyYAML + stdlib; its launcher_cells imports this checkout's contracts)."""
    k = str(Path(reader).resolve())
    if k not in _MODELS_MOD:
        spec = importlib.util.spec_from_file_location("_bcbw_front_door_models", reader)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _MODELS_MOD[k] = mod
    return _MODELS_MOD[k]


def frontdoor_keys(table: Path = MODELS_TABLE) -> List[Tuple[str, str]]:
    """``[(key, view)]`` — every front-door model of the table with every PHASE view a launcher row serves (``lib/models.py
    phase_views``: the one steered wave first — the declared wave of the single-phase recipe, never a literal spelled here — then its
    dose views, each with a registry row of its own, ``launcher_slug``; demo-import 2026-10-12)."""
    M = models_module()
    return [(k, v) for k in M.front_door_keys(str(table)) for v in M.phase_views(k, str(table))]


def frontdoor_body_path(key: str, view: str, repo: Path = REPO) -> Path:
    """The battery body ``05_stage_roster.sh --step configs`` renders for the wave: ``configs/sdfbase_full22_<key>_<view>.yaml``."""
    return Path(repo) / "configs" / f"sdfbase_full22_{key}_{view}.yaml"


def frontdoor_inputs(key: str, view: Optional[str] = None, *, repo: Path = REPO, table: Path = MODELS_TABLE,
                     roster_path: Optional[Path] = None, pins_path: Optional[Path] = None) -> Dict[str, object]:
    """Every input of the front-door render, each REFUSED by name when absent: the key's models-table row and its wave; the key's
    ``contracts.MODELS`` row (its ``objects_dir`` — refused when it sits under the full22 root, where ``is_full22_row`` would turn on the
    full22 pins); the shared roster's block for the key with the view; the pins file's block for the key; the battery body with its marker."""
    M = models_module()
    try:
        M.model(key, str(table))
    except KeyError as e:
        raise FrontDoorRefusal(key, view, f"{table} has no row {key!r} ({e})") from None
    if view is None:
        view = M.baseline_view(key, str(table))
    if view not in M.views(key, str(table)):
        raise FrontDoorRefusal(key, view, f"{view!r} is not a view of {key} in {table} (views: {M.views(key, str(table))})")
    slug = M.launcher_slug(key, view, str(table))          # the key for the baseline wave; `<key>-<view>` for a dose view (ONE rule)
    try:
        row = C.model_row(slug)
    except C.UnknownModel as e:
        raise FrontDoorRefusal(key, view, f"contracts.MODELS has no row for the slug {slug!r} ({e}) — the launcher cannot serve the view") from None
    if row.objects_dir.startswith(C.VOL_OBJECTS_FULL22 + "/"):
        raise FrontDoorRefusal(key, view, f"the row's objects_dir {row.objects_dir!r} sits under the full22 root {C.VOL_OBJECTS_FULL22} — a front-door bank "
                                          "is its own (is_full22_row would turn on the full22 pins by cell)")
    roster_path = Path(roster_path) if roster_path else Path(repo) / "configs" / "sdfbase_full22_roster.yaml"
    pins_path = Path(pins_path) if pins_path else Path(repo) / "configs" / "sdfbase_full22_pins.json"
    if not roster_path.is_file():
        raise FrontDoorRefusal(key, view, f"no shared roster at {roster_path}")
    roster = yaml.safe_load(roster_path.read_text(encoding="utf-8")) or {}
    block = (roster.get("targets") or {}).get(key)
    if not isinstance(block, dict):
        raise FrontDoorRefusal(key, view, f"{roster_path} carries no block for the front-door target — 05_stage_roster.sh --step roster prints it "
                                          "(lib/models.py roster-block); paste it under targets:, commit, then generate")
    if view not in (block.get("views") or {}):
        raise FrontDoorRefusal(key, view, f"{roster_path}'s {key} block has no view {view!r} (views: {sorted(block.get('views') or {})})")
    if not pins_path.is_file():
        raise FrontDoorRefusal(key, view, f"no pins file at {pins_path} — every object cell must be pinned; nothing is generated unpinned")
    pins_all = json.loads(pins_path.read_text(encoding="utf-8"))
    if not isinstance(pins_all.get(key), dict) or not pins_all[key]:
        raise FrontDoorRefusal(key, view, f"{pins_path} carries no {key} block — 05_stage_roster.sh --step pins (import_objects.py pins for a committed "
                                          "row, make_full22_configs.py --pins-from for a fitted one) writes it")
    body_path = frontdoor_body_path(key, view, repo)
    if not body_path.is_file():
        raise FrontDoorRefusal(key, view, f"no battery body at {body_path} — 05_stage_roster.sh --step configs renders it (make_full22_configs.py over the "
                                          "roster block); commit it, then generate")
    body_text = body_path.read_text(encoding="utf-8")
    if BODY_MARKER not in body_text:
        raise FrontDoorRefusal(key, view, f"{body_path} has no {BODY_MARKER!r} marker — not a generated cluster config")
    return {"M": M, "key": key, "view": view, "slug": slug, "row": row, "roster": roster, "roster_path": roster_path, "pins": pins_all, "pins_path": pins_path,
            "body_path": body_path, "body_text": body_text, "table": Path(table)}


def frontdoor_cells(inp: Mapping[str, object]) -> Tuple[List[str], Dict[str, str]]:
    """``(explicit_cells, {object cell id: staged id})``: the ORDERED launcher tokens of the wave's cells (``lib/models.py launcher_cells`` —
    none first, then the table's cells in the table's order, the arditi rung stripped), held equal to the roster block's own enumeration
    (``none`` + ``f22_roster.served_cells`` in roster order: the prompted ``prompted__*: null`` entries and the object cells) — a block pasted
    from another table state refuses here instead of serving a different list than the launcher plans."""
    M = inp["M"]; key = str(inp["key"]); view = str(inp["view"]); roster = inp["roster"]
    try:
        tokens = [t for _c, t in M.launcher_cells(key, view, path=str(inp["table"]))]
    except (KeyError, ValueError) as e:
        raise FrontDoorRefusal(key, view, f"the models table's cells do not map to launcher tokens: {e}") from None
    try:
        served = served_cells(roster, key, view)
    except (KeyError, ValueError) as e:
        raise FrontDoorRefusal(key, view, f"the roster block's cells do not read: {e}") from None
    roster_list = [C.ROUTE_NONE_CELL, *served]
    if tokens != roster_list:
        raise FrontDoorRefusal(key, view, f"the roster block's cells {roster_list} are not the models table's launcher tokens {tokens} (05_stage_roster.sh "
                                          "--step roster prints the block the table implies; paste that one)")
    for cid in tokens:
        C.parse_launcher_cell(cid)                      # the launcher's own validator: a token it refuses never reaches a config
    return tokens, dict(staged_cells(roster, key, view))


def frontdoor_pins(inp: Mapping[str, object], staged: Mapping[str, str]) -> Dict[str, str]:
    """``{object cell id: object_sha256}`` from the pins file's block for the key; an unpinned object cell REFUSES by name."""
    key = str(inp["key"]); view = str(inp["view"]); blk = inp["pins"][key]      # type: ignore[index]
    out: Dict[str, str] = {}
    missing = []
    for cid, sid in staged.items():
        sha = (blk.get(sid) or {}).get("object_sha256") if isinstance(blk.get(sid), dict) else None
        if not isinstance(sha, str) or len(sha) != 64:
            missing.append(f"{cid} <- {sid}")
        else:
            out[cid] = sha
    if missing:
        raise FrontDoorRefusal(key, view, f"unpinned object cell(s) in {inp['pins_path']}[{key}]: {missing} — nothing is generated unpinned")
    return out


def frontdoor_cfg_note(key: str, view: str) -> Optional[str]:
    """None when ``contracts.MODELS[key].cfg`` already names the front-door file; else the note the generator prints (the flip lands with
    the committed file — a refusal here would block generating the very file the flip needs)."""
    want = C.frontdoor_config_path(key, view)
    try:
        slug = models_module().launcher_slug(key, view)
    except KeyError:
        slug = key
    try:
        have = C.model_row(slug).cfg
    except C.UnknownModel:
        return f"note: contracts.MODELS has no row {slug!r} — add the view's row (reproduce/lib/models.py launcher_slug) pointing at {want!r} when the file is committed"
    return None if have == want else f"note: contracts.MODELS[{slug!r}].cfg is {have!r}, not {want!r} — point the row at the generated file when it is committed"


def render_frontdoor_config(key: str, view: Optional[str] = None, *, repo: Path = REPO, table: Path = MODELS_TABLE,
                            roster_path: Optional[Path] = None, pins_path: Optional[Path] = None) -> str:
    """The front door's generated config text for ``(key, view)`` — refuses by name on any missing input (:func:`frontdoor_inputs`),
    a roster block that disagrees with the table (:func:`frontdoor_cells`) or an unpinned object cell (:func:`frontdoor_pins`)."""
    inp = frontdoor_inputs(key, view, repo=repo, table=table, roster_path=roster_path, pins_path=pins_path)
    key = str(inp["key"]); view = str(inp["view"]); slug = str(inp["slug"]); row = inp["row"]; roster = inp["roster"]; body_path = inp["body_path"]
    view_flag = "" if slug == key else f" --view {view}"                 # a dose view's row is `<key>-<view>`; the baseline wave's the key's
    cells, staged = frontdoor_cells(inp)
    cell_pins = frontdoor_pins(inp, staged)
    src_text = str(inp["body_text"])
    body = src_text[src_text.index(BODY_MARKER):]
    body = re.sub(r"^# -- everything below: .*$", f"# -- everything below: {body_path.name}'s data, byte-for-byte, EXCEPT defenses_dir --",
                  body, count=1, flags=re.MULTILINE)
    objects_dir = row.objects_dir
    body = _sub_line(body, r"^defenses_dir: .*$", f"defenses_dir: {objects_dir}")
    t = roster["targets"][key]
    target_cells = target_served_cells(roster, key)
    mla_block = multi_layer_block(key, cells, target_cells=target_cells)
    mla_line = ("#   * `multi_layer_arms` = the declared layer set of every object fitted on several layers "
                "(F9 equality).\n" if mla_block else "")
    cell_lines = []
    for cid in cells:
        if cid in staged:
            cell_lines.append(f"#       {cid:<48} lane {lane_kind_of_staged_id(staged[cid]):<8} <- {staged[cid]}")
            cell_lines.append(f"#       {'':<48} object_sha256 {cell_pins[cid]}")
        else:
            what = "the same-wave bare anchor" if cid == C.ROUTE_NONE_CELL else "the SI text as the system prompt; no object (the roster block's prompted__*: null entry)"
            cell_lines.append(f"#       {cid:<48} ({what})")
    n_prompted = sum(1 for c in cells if is_prompted(c))
    path_rel = C.frontdoor_config_path(key, view)
    header = f"""# {path_rel.rsplit('/', 1)[1]} — the FRONT DOOR's cells on Modal (the worked example): the EXPLICIT-cells
# serving config the tinycap launcher's registry row `{slug}` reads (`contracts.MODELS["{slug}"].cfg`; reproduce/07_bcb_bfcl.sh
# --model {key}{view_flag} launches against it). GENERATED by scripts/bcbw_configs.py --front-door {key}{view_flag} from reproduce/models.yaml
# (the `{view}` wave's cells through lib/models.py launcher_cells — the ONE mapping stage 07 launches with) and from
# configs/sdfbase_full22_roster.yaml (target {key}, view {view}: the block 05_stage_roster.sh --step roster prints) — never
# hand-edited; tests/test_bcbw_configs.py requires byte-identity.
#
# Sources (sha256 at generation; a change regenerates this file):
#   table   reproduce/models.yaml  {sha256_file(Path(inp['table']))}
#   roster  configs/sdfbase_full22_roster.yaml  {sha256_file(Path(inp['roster_path']))}
#   body    configs/{body_path.name}  {sha256_file(body_path)}
#   pins    configs/sdfbase_full22_pins.json  {sha256_file(Path(inp['pins_path']))}
#
# DERIVED from that cluster config: every key below the body marker is its data, byte-for-byte
# (model / model_revision / layer / max_len / the whole battery body, which the driver's hashed
# controls_gen_config reads), EXCEPT —
#
#   * `explicit_cells` = [none, the {n_prompted} prompted cell(s), the {len(staged)} object cell(s)] in the models table's order — the
#     launcher tokens of the `{view}` wave's cells (lib/models.py launcher_cells), equal to the roster block's enumeration (none +
#     f22_roster.served_cells); the registry is EXACTLY this list — registry_from_explicit_cells validates every object's artifact
#     + its one identity sidecar at build time, on the CPU, before any GPU minute. Lane kind = the staged id's prefix:
{chr(10).join(cell_lines)}
#     Which of these a run LAUNCHES is stage 07's --cells (the launcher plans only `--cells`); the rest are the registry's anchors.
#   * `defenses_dir` = {objects_dir} — the row's objects_dir, ABSOLUTE on the Modal volume (resolve_defenses_dir rule 1),
#     the front door's OWN bank outside the full22 root: reproduce/tools/stage_demo_bank_modal.py stages <defenses_dir>/<cell>/
#     {{object.pt, <identity sidecar>}} from the committed objects, sha-verified against the pins above.
{mla_line}#
# serving target: {t['model']} @ {t['model_revision']}

# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---
explicit_cells:
"""
    for cid in cells:
        header += f"- {cid}\n"
    header += mla_block
    header += "\n"
    return header + body


def frontdoor_outputs(table: Path = MODELS_TABLE) -> List[Tuple[Path, str, str]]:
    """``[(path, key, wave)]`` for every front-door model of the table (:func:`frontdoor_keys`)."""
    return [(REPO / C.frontdoor_config_path(k, v), k, v) for k, v in frontdoor_keys(table)]


def check_frontdoor(dst: Path, key: str, view: str, *, repo: Path = REPO, table: Path = MODELS_TABLE) -> Tuple[str, str]:
    """``(status, line)`` for one front-door output under ``--check``: ``absent`` (not generated yet — not a drift: its inputs land with
    the integration), ``ok`` (byte-identical to the render), ``drift`` (differs, or the render refuses although the file exists)."""
    if not dst.is_file():
        return "absent", (f"absent: {dst} (the front door's config is generated once its roster block, pins block and battery body are "
                          f"committed — python scripts/bcbw_configs.py --front-door {key})")
    try:
        text = render_frontdoor_config(key, view, repo=repo, table=table)
    except FrontDoorRefusal as e:
        return "drift", f"DRIFT: {dst} (the committed file exists but its render refuses: {e})"
    if dst.read_text(encoding="utf-8") != text:
        return "drift", f"DRIFT: {dst}"
    return "ok", f"ok: {dst}"


def outputs(roster: dict) -> List[Tuple[Path, str, str]]:
    """``[(path, target, view)]`` for the lane's (model, view) pairs (``contracts.BCBW_MODEL_VIEWS``)."""
    out = []
    for target, view in C.BCBW_MODEL_VIEWS:
        if view not in roster["targets"][target]["views"]:
            raise SystemExit(f"roster has no view {view!r} for {target!r}")
        out.append((REPO / C.bcbw_config_path(target, view), target, view))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed file drifts (the 19 lane files + every front-door file that exists)")
    ap.add_argument("--front-door", metavar="KEY", default=None,
                    help="the FRONT-DOOR scope (demo-import D7): write (or --check) configs/tinycap/frontdoor_<KEY>_<wave>.yaml from reproduce/models.yaml + "
                         "the shared roster's block + the pins block; refuses by name when an input is not committed yet")
    ap.add_argument("--view", default=None, help="with --front-door: the wave (default: the key's one steered wave, lib/models.py baseline_view)")
    args = ap.parse_args(argv)
    if args.view and not args.front_door:
        ap.error("--view goes with --front-door")
    rc = 0
    if args.front_door:
        key = args.front_door
        inp_view = args.view
        try:
            text = render_frontdoor_config(key, inp_view)
            view = inp_view or frontdoor_inputs(key, None)["view"]
        except FrontDoorRefusal as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(2) from None
        dst = REPO / C.frontdoor_config_path(key, str(view))
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}")
                rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
            print(f"wrote {dst}")
        note = frontdoor_cfg_note(key, str(view))
        if note:
            print(note)
        return rc
    roster = load_roster()
    pins = load_pins()
    lane = load_lane_rosters()
    for dst, target, view in outputs(roster):
        text = render_config(roster, target, view, pins=pins, lane_roster=lane)
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}")
                rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
            print(f"wrote {dst}")
    if args.check:
        # the front door's files ride the same check once they exist; one not generated yet is said, never a drift
        for dst, key, view in frontdoor_outputs():
            status, line = check_frontdoor(dst, key, view)
            print(line)
            if status == "drift":
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
