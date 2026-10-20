#!/usr/bin/env python3
"""ifeval-canonical-battery — THE IFEVAL ROW BUILDER: ``select_rows``.

The GSM8K lane's ``g8k_views.select_rows`` is NOT task-neutral (it hard-codes ``declared = fitted and header-located``,
the ``@thinking`` spellings, the GSM8K authorisation tuples, the 8,192 / 16,384 rungs, ``thinking_card`` sampling, GSM8K
paths and a family map without Llama). This module REUSES ONLY its neutral enumeration — ``g8k_views.cells_of``
(``[none] + [prompted__safety_refusal iff view == it] + the roster's served cells, roster order``), ``pin_for``,
``parse_views_spec``, ``load_views_yaml``, ``ViewsSpec.hold_for`` — and builds the IFEval rows itself (a source-text test
pins that nothing else of the GSM8K builder is imported):

* ``declared False`` on every row; ``standup_cell == cell`` (no ``@thinking`` — thinking OFF is the fit render, P15).
* ``served_positions_expected``: ``response`` for the header-located cells, ``all`` for every other fitted cell, None for
  bare / prompted; ``expected_stamps`` = ``ifeval_contracts.ifeval_expected_stamps`` (the OFF triple, undeclared).
* ``sampling_mode vendor_preset`` with the model's row BY IMPORT (``VENDOR_PRESET_SAMPLING_BY_MODEL_KEY``), k = 2, the
  ladder ``(2048, 4096, 8192)``.
* EVERY path from ``ifeval_contracts.ifeval_paths(key, view, cell, rung)`` (P37) — a slug is never a path component.
* the per-row object identity the emitted rows must carry (P42): ``object_sha256`` (a stated null on bare / prompted),
  ``route_kind``, ``view``; ``family`` from ``contracts.FAMILY_BY_MODEL_KEY`` (Llama included); ``image nemotron`` iff
  ``nemotron_h``; the launcher slug ``IFEVAL_SLUGS[(key, view)]``.
* the per-row ``pin`` (the follow-up): WHERE the row's object identity is pinned — the roster pin (pins file + sha256 +
  staged id) on a fitted row, a stated null on a bare / prompted row, the per-model record a DEFERRED slot is filled
  from. The cross-eval guard reads object sha + view + pin per row.

THE POLICY (``views.yaml``): ``exclude:`` drops roster cells from the matrix altogether (every ``*psr*`` arm,
``safety_refusal__lora_r4``); ``prompted_extra:`` adds a prompted cell — ``prompted__si_constitution``, the SHORT
constitution SI, whose text sha256 the generator ASSERTS — on every (model, view) that carries a prompted cell, right
after it; ``deferred:`` names the AlphaSteer / Arditi design slots (one per model, ``status: deferred``, no object sha,
appended LAST so filling one never moves a launch row). A deferred row resolves its geometry / paths like a fitted row
and is excluded from the launch set, the contrasts and the estimate; the dispatcher refuses to launch it.

The Llama target comes from the lane's ROSTER OVERLAY ``configs/sdfbase/ifeval/llama8b_roster.yaml`` merged over the
shared roster (targets only) and its PINS OVERLAY ``configs/sdfbase/ifeval/llama8b_pins.json`` merged over the shared
pins — both recorded with their sha256 in the matrix's ``source`` block, so the GSM8K lane's frozen matrix (which pins
the SHARED roster's sha) is untouched.

FROZEN INPUTS (the roster-shadow remedy; the GSM8K lane's P16 pattern): the shared registry MOVED under the
frozen matrix on 2026-09-12 — a ``llama8b`` target was added to ``configs/sdfbase_full22_roster.yaml``
and a ``llama8b`` block to ``configs/sdfbase_full22_pins.json`` (the Llama battery: a DIFFERENT cell set for the
same served model than this lane's overlay), and the shared views now serve ``prompted__si_constitution``
themselves. Each of those makes the live registry refuse this lane's inputs (the overlay adds a target, never shadows
one; a prompted extra the roster already serves is not an extra), and a regeneration that admitted them would change
the ROWS (44 rows / 34 launch). So ``views.yaml`` names ``frozen_inputs:`` — byte-identical copies of the roster
and pins AS COMMITTED AT THE PIN 59cedcca (main's state when the matrix's ``source.roster_sha256`` / ``pins_sha256``
were recorded; ``git show 59cedcca:<path>`` restores an edited copy) under ``configs/sdfbase/ifeval/frozen/59cedcca/``
— and :func:`load_inputs` reads THOSE, their sha256 asserted on every load by the GSM8K lane's own
``frozen_input_paths`` (an edited copy refuses by name). In that copy ``llama8b`` is absent and no view serves the
constitution SI, so the overlay ADDS the target and the prompted extra IS an extra, exactly as at the freeze; both
refusals stay in force as written; a shared-roster commit can no longer move this lane. The matrix records the copies
under ``source.frozen_inputs`` (``source.roster`` / ``source.pins`` stay the REGISTRY paths the copies are copies of,
the GSM8K precedent, so every row's ``pin.source`` is unchanged). Re-freezing at a later pin is a logged decision (copy
the files, update the four values in ``views.yaml``, regenerate, ``--check``), never a side effect of a peer's commit.
A spec without ``frozen_inputs`` (the ``--views`` CLI form over a peer's tree) reads that tree's LIVE registry.

HOLDS (P41): ``views.yaml``'s ``holds:`` stamps the it-view constitution LoRA rows ``status: held`` (+ ``held_reason``);
they stay IN the matrix for stable indices / TAGs and are a HARD EXCLUSION everywhere else.

CONTRASTS (P32): :func:`contrast_resolution` resolves every ``IFEVAL_CONTRASTS`` family over the LAUNCH rows; the
generator's ``--check`` and the launcher's preflight refuse a matrix in which a promised contrast lacks a leg.

THE SIBLING DOCUMENT: ``views_qwen_seedrep.yaml`` -> ``matrix_qwen_seedrep.json`` expresses the batch-64 three-seed
Qwen LANE views ``s1..s3`` / ``s1p..s3p`` (``contracts.IFEVAL_LANE_VIEWS``; TAG ``qwen35_4b_<view>``, slug
``qwen35_4b-sdfbase-<view>``) through the SAME builder, with four grammar additions this module learns and the
frozen 40-row document never uses (its bytes are unchanged): ``anchor_view: {s1p: s1, ...}`` (a view whose
``cell_minus_none`` legs resolve against ANOTHER view's launched ``none`` — the composed row has no own anchor),
``view_cells: {s1p: [<cell>], ...}`` (a view expressing a SUBSET of its neutral enumeration, by name — every
listed cell must be in the enumeration), ``lane_extra:`` (a fitted Class B cell another lane fit and this document
declares STAGED in the view's bank on the volume — the Arditi ``arditi_allpos`` object inside each ``s<s>`` view —
pinned from ITS pins file, appended after the enumerated rows and before the deferred slots), and
``serving_path: lora_native`` (declared on the document and stamped on every row; the dispatcher passes it to the
launcher). The Llama roster / pins overlays are merged ONLY when the document expresses ``llama8b`` (the frozen
document does; the sibling does not — its frozen roster copy already carries the target).
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, HERE.parent / "gsm8k", HERE.parent / "full22", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import if_matrix as IM
from g8k_views import ViewsSpec, cells_of, frozen_input_paths, load_views_yaml, parse_views_spec, pin_for  # the NEUTRAL enumeration + the frozen-copy sha assertion only (P31)
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

ROSTER_REL = Path("configs/sdfbase_full22_roster.yaml")
PINS_REL = Path("configs/sdfbase_full22_pins.json")
VIEWS_REL = Path(IC.IFEVAL_VIEWS_REL)
ROSTER_OVERLAY_REL = Path("configs/sdfbase/ifeval/llama8b_roster.yaml")
PINS_OVERLAY_REL = Path("configs/sdfbase/ifeval/llama8b_pins.json")
GENERATED_BY = "scripts/sdfbase/ifeval/make_ifeval_matrix.py (if_views.select_rows)"
#: the overlay config / harness manifest per TAG the launcher's ``--config`` names (written by make_ifeval_overlays.py)
OVERLAY_CONFIG_REL = C.IFEVAL_OVERLAY_CONFIG_REL
OVERLAY_MANIFEST_REL = C.IFEVAL_OVERLAY_MANIFEST_REL
_SHORT_SI = {"constv2-mix-full": "mix", "constv2ctl2-full": "ctl", "safety_refusal": "si12"}
_SHORT_ARM = {"psr_ce_all": "psr", "lora_r4_ce": "lora", "psr_kl_all": "psr", "lora_r4": "lora",
              "psr_ce_all_minus_ctl_distilled": "dpsr", "lora_r4_ce_minus_ctl": "dlora"}
#: the SHORT prompted spellings of the job id: safety = `prompted`, constitution = `constitution`
_SHORT_PROMPTED = {IC.CELL_PROMPTED: "prompted", IC.CELL_PROMPTED_CONSTITUTION: "constitution"}
#: the prompted_extra placement rules views.yaml may name
ON_VIEWS_WITH_PROMPTED = "views_with_prompted_cell"
ON_ALL_VIEWS = "all_views"
PLACEMENTS = (ON_VIEWS_WITH_PROMPTED, ON_ALL_VIEWS)
#: the fields a deferred slot's per-model record fills (the rule: minimal over-refusal strength)
DEFERRED_RECORD_FIELDS = ("object_sha256", "view", "rung", "selection_rule")
PIN_KIND_ROSTER, PIN_KIND_NONE, PIN_KIND_PROMPTED, PIN_KIND_DEFERRED = "roster", "none", "prompted", "deferred"
#: a `lane_extra:` row's pin kind (pinned from the OTHER lane's pins file, never the full22 pins) and its object source
PIN_KIND_LANE_EXTRA = "lane_extra"
OBJECT_SOURCE_EXTERNAL = "external"
#: the serving paths a document may declare (`serving_path:`); the launcher's own two spellings
SERVING_PATH_LENS, SERVING_PATH_LORA_NATIVE = "lens", "lora_native"
SERVING_PATHS = (SERVING_PATH_LENS, SERVING_PATH_LORA_NATIVE)
#: the fields a `lane_extra:` entry names (the Arditi rows)
LANE_EXTRA_FIELDS = ("key", "view", "cell", "staged_id", "pins", "bank", "sidecar", "reason")
#: the matrix `source` key naming the FROZEN roster / pins copies the rows were derived from (null = the live registry)
FROZEN_INPUTS_KEY = "frozen_inputs"


# --------------------------------------------------------------------------- inputs
def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_inputs_of(spec: ViewsSpec, configs_root: Path) -> Optional[dict[str, Any]]:
    """The IMMUTABLE roster / pins copies ``views.yaml`` names (``frozen_inputs:``, the GSM8K lane's P16 grammar: the two
    paths relative to the configs root + their sha256), their sha256 ASSERTED by the GSM8K lane's own ``frozen_input_paths``
    (an edited copy refuses by name — restore it with ``git show <pin>:<path>``). Returns ``{"roster_path", "pins_path",
    "record"}`` — the copies to read and the ``source.frozen_inputs`` record — or None when the spec names none (the LIVE
    registry of ``configs_root`` is the input: a peer's tree under ``--views``)."""
    paths = frozen_input_paths(spec, Path(configs_root))
    if paths is None:
        return None
    return {"roster_path": paths[0], "pins_path": paths[1],
            "record": {k: spec.frozen_inputs[k] for k in ("roster", "pins", "roster_sha256", "pins_sha256")}}


def load_inputs(configs_root: Path, spec: ViewsSpec) -> tuple[dict, dict, dict[str, Any]]:
    """The generator's inputs for ``spec``: the roster and pins (the FROZEN copies when the spec names ``frozen_inputs``,
    else the live registry) with the lane's overlays merged in, and the ONE ``source`` record — the registry paths the
    copies are copies OF, the sha256 of the bytes actually read (= the pin's), the overlays, and ``frozen_inputs`` (the
    copies' record, or an explicit null when the live registry was read). Returns ``(roster, pins, source)``."""
    frozen = frozen_inputs_of(spec, configs_root)
    # the Llama overlays are THIS lane's additions for the llama8b target; a document that expresses no
    # llama8b (the seed sibling) reads its frozen copies alone — a copy that already carries the target would otherwise
    # refuse under the adds-never-shadows rule, for a model the document never serves
    overlay = "llama8b" in spec.views
    roster, roster_source = load_roster(configs_root, frozen=frozen, overlay=overlay)
    pins, pins_source = load_pins(configs_root, frozen=frozen, overlay=overlay)
    return roster, pins, {**roster_source, **pins_source, FROZEN_INPUTS_KEY: (frozen["record"] if frozen else None)}


def load_roster(configs_root: Path, *, frozen: Optional[Mapping[str, Any]] = None, overlay: bool = True) -> tuple[dict, dict[str, Any]]:
    """The shared roster with the IFEval lane's Llama target overlay merged in (targets only; a target the shared roster
    already names refuses — the overlay adds, never shadows). ``frozen`` (:func:`frozen_inputs_of`) reads the FROZEN copy
    in place of the live registry file; the source record names the registry path either way (what the copy is a copy of)
    and the sha256 of the bytes read. Returns ``(roster, source_record)``."""
    root = Path(configs_root)
    base_path = Path(frozen["roster_path"]) if frozen else root / ROSTER_REL
    overlay_path = root / ROSTER_OVERLAY_REL
    if not base_path.is_file():
        raise SystemExit(f"no roster at {base_path}")
    roster = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    source = {"roster": str(ROSTER_REL), "roster_sha256": sha256_of(base_path), "roster_overlay": None, "roster_overlay_sha256": None}
    if overlay and overlay_path.is_file():
        overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
        for key, target in (overlay.get("targets") or {}).items():
            if key in (roster.get("targets") or {}):
                raise SystemExit(f"{overlay_path}: target {key!r} is already in the shared roster — the overlay adds a "
                                 "target, never shadows one")
            roster.setdefault("targets", {})[key] = target
        source.update(roster_overlay=str(ROSTER_OVERLAY_REL), roster_overlay_sha256=sha256_of(overlay_path))
    return roster, source


def load_pins(configs_root: Path, *, frozen: Optional[Mapping[str, Any]] = None, overlay: bool = True) -> tuple[dict, dict[str, Any]]:
    """The shared pins with the lane's pins overlay merged (a model key both name refuses); ``frozen`` reads the FROZEN
    pins copy in place of the live file (the source record names the registry path and the sha256 of the bytes read)."""
    root = Path(configs_root)
    base_path = Path(frozen["pins_path"]) if frozen else root / PINS_REL
    overlay_path = root / PINS_OVERLAY_REL
    if not base_path.is_file():
        raise SystemExit(f"no pins at {base_path}")
    pins = json.loads(base_path.read_text(encoding="utf-8"))
    source = {"pins": str(PINS_REL), "pins_sha256": sha256_of(base_path), "pins_overlay": None, "pins_overlay_sha256": None}
    if overlay and overlay_path.is_file():
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        for key, entries in overlay.items():
            if key.startswith("_"):
                continue
            if key in pins:
                raise SystemExit(f"{overlay_path}: model key {key!r} is already pinned in the shared pins")
            pins[key] = entries
        source.update(pins_overlay=str(PINS_OVERLAY_REL), pins_overlay_sha256=sha256_of(overlay_path))
    return pins, source


def views_path(configs_root: Path) -> Path:
    return Path(configs_root) / VIEWS_REL


# --------------------------------------------------------------------------- the policy
@dataclass
class IfevalPolicy:
    """What ``views.yaml`` says beyond the neutral enumeration: the roster cells EXCLUDED from the matrix, the prompted
    cells ADDED, the DEFERRED design slots. Empty = the neutral enumeration alone (a matrix with no policy)."""
    exclude: list[dict] = field(default_factory=list)        # [{arms_matching | cells, reason}]
    prompted_extra: list[dict] = field(default_factory=list)  # [{cell, si, text_sha256, text_chars, placement, reason}]
    deferred: list[dict] = field(default_factory=list)        # [{arm, cell, views: {key: view}, record, selection_rule, reason}]
    # the sibling document's grammar; all empty / None on the frozen 40-row document, whose record is byte-identical:
    anchor_view: dict[str, str] = field(default_factory=dict)      # {view: the view whose launched `none` anchors it}
    view_cells: dict[str, list[str]] = field(default_factory=dict)  # {view: [cell, ...]} — the SUBSET of the enumeration expressed
    lane_extra: list[dict] = field(default_factory=list)            # [{key, view, cell, staged_id, pins, bank, sidecar, reason}]
    serving_path: Optional[str] = None                              # `lens` | `lora_native`, stamped on every row when declared

    def exclusion_for(self, cell: str) -> Optional[str]:
        """The reason ``cell`` is excluded from the matrix, or None. ``arms_matching`` matches the ARM half of
        ``<si>__<arm>`` as a substring (``psr`` matches every PSR arm); ``cells`` matches whole ids."""
        si, sep, arm = cell.partition("__")
        for e in self.exclude:
            pat = e.get("arms_matching")
            if pat and sep and pat in arm:
                return e["reason"]
            if cell in (e.get("cells") or ()):
                return e["reason"]
        return None

    def as_record(self) -> dict:
        out = {"exclude": [dict(e) for e in self.exclude], "prompted_extra": [dict(e) for e in self.prompted_extra],
               "deferred": [dict(e) for e in self.deferred]}
        # the extra blocks ride the record ONLY when the document declares them — the frozen document's record is unchanged
        if self.anchor_view:
            out["anchor_view"] = dict(self.anchor_view)
        if self.view_cells:
            out["view_cells"] = {v: list(cs) for v, cs in self.view_cells.items()}
        if self.lane_extra:
            out["lane_extra"] = [dict(e) for e in self.lane_extra]
        if self.serving_path is not None:
            out["serving_path"] = self.serving_path
        return out


def load_ifeval_policy(path: Optional[Path]) -> IfevalPolicy:
    """The policy blocks of ``views.yaml`` (``exclude`` / ``prompted_extra`` / ``deferred``), validated: an exclusion
    names ``arms_matching`` or ``cells`` and a reason; a prompted extra names a ``prompted__<si>`` cell (NEVER a ``_full``
    spelling, which the cell grammar refuses), its SI, the sha256 of the SI text, a placement and a reason; a deferred slot
    names an arm, a cell, a ``{model key: view}`` map over IFEval models / views, a record path template with
    ``{model_key}``, a selection rule and a reason. ``None`` (a ``--views`` CLI spec) = the empty policy."""
    if path is None:
        return IfevalPolicy()
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    pol = IfevalPolicy()
    for e in doc.get("exclude") or []:
        if not isinstance(e, dict) or not str(e.get("reason", "")).strip() or not (e.get("arms_matching") or e.get("cells")):
            raise SystemExit(f"{path}: every `exclude:` entry needs arms_matching or cells, and a reason — got {e!r}")
        rec = {"reason": str(e["reason"]).strip()}
        if e.get("arms_matching"):
            rec["arms_matching"] = str(e["arms_matching"])
        if e.get("cells"):
            rec["cells"] = [str(c) for c in (e["cells"] if isinstance(e["cells"], list) else [e["cells"]])]
        pol.exclude.append(rec)
    for e in doc.get("prompted_extra") or []:
        if not isinstance(e, dict) or not {"cell", "si", "text_sha256", "placement", "reason"} <= set(e):
            raise SystemExit(f"{path}: every `prompted_extra:` entry needs cell, si, text_sha256, placement, reason — got {e!r}")
        cell = str(e["cell"])
        if cell.endswith("_full"):
            # NEVER the `_full` spelling — the short constitution SI is `prompted__si_constitution`; main's grammar
            # admits `prompted__constitution_full` for the Llama fullconst view, so the refusal is explicit here
            raise C.CellNameError(f"{path}: prompted_extra cell {cell!r}: the `_full` spelling is never an IFEval extra (D23)")
        parsed = C.parse_launcher_cell(cell)
        if parsed.kind != "prompted" or parsed.si != str(e["si"]):
            raise SystemExit(f"{path}: prompted_extra cell {cell!r} is not prompted__{e['si']}")
        if e["placement"] not in PLACEMENTS:
            raise SystemExit(f"{path}: prompted_extra {cell!r}: `placement` must be one of {PLACEMENTS}, got {e['placement']!r}")
        pol.prompted_extra.append({"cell": cell, "si": str(e["si"]), "text_sha256": str(e["text_sha256"]),
                                   "text_chars": (int(e["text_chars"]) if e.get("text_chars") is not None else None),
                                   "placement": str(e["placement"]), "reason": str(e["reason"]).strip()})
    for e in doc.get("deferred") or []:
        if not isinstance(e, dict) or not {"arm", "cell", "views", "record", "selection_rule", "reason"} <= set(e):
            raise SystemExit(f"{path}: every `deferred:` entry needs arm, cell, views, record, selection_rule, reason — got {e!r}")
        views = e["views"]
        if not isinstance(views, dict) or not views:
            raise SystemExit(f"{path}: deferred {e['arm']!r}: `views` must map model keys to a PROVISIONAL view")
        for key, view in views.items():
            if key not in IC.IFEVAL_MODELS or view not in IC.IFEVAL_VIEWS:
                raise SystemExit(f"{path}: deferred {e['arm']!r}: {key}/{view} is not an IFEval (model, view)")
        if "{model_key}" not in str(e["record"]):
            raise SystemExit(f"{path}: deferred {e['arm']!r}: the record path must carry {{model_key}}")
        pol.deferred.append({"arm": str(e["arm"]), "cell": str(e["cell"]), "views": {str(k): str(v) for k, v in views.items()},
                             "record": str(e["record"]), "selection_rule": str(e["selection_rule"]).strip(),
                             "reason": str(e["reason"]).strip()})
    # ---- the sibling document's grammar; every block optional, validated when present
    anchor = doc.get("anchor_view") or {}
    if not isinstance(anchor, dict):
        raise SystemExit(f"{path}: `anchor_view:` maps a view to the view whose launched none anchors it")
    for v, a in anchor.items():
        if str(v) not in IC.IFEVAL_VIEWS or str(a) not in IC.IFEVAL_VIEWS or str(v) == str(a):
            raise SystemExit(f"{path}: anchor_view {v!r}: {a!r} — both must be IFEval views ({IC.IFEVAL_VIEWS}) and differ")
        pol.anchor_view[str(v)] = str(a)
    vcells = doc.get("view_cells") or {}
    if not isinstance(vcells, dict):
        raise SystemExit(f"{path}: `view_cells:` maps a view to the list of cells it expresses")
    for v, cs in vcells.items():
        if str(v) not in IC.IFEVAL_VIEWS or not isinstance(cs, list) or not cs or not all(isinstance(c, str) and c for c in cs):
            raise SystemExit(f"{path}: view_cells {v!r}: a non-empty list of cell ids on an IFEval view, got {cs!r}")
        pol.view_cells[str(v)] = [str(c) for c in cs]
    for e in doc.get("lane_extra") or []:
        if not isinstance(e, dict) or set(LANE_EXTRA_FIELDS) - set(e):
            raise SystemExit(f"{path}: every `lane_extra:` entry needs {LANE_EXTRA_FIELDS} — got {e!r}")
        if str(e["key"]) not in IC.IFEVAL_MODELS or str(e["view"]) not in IC.IFEVAL_VIEWS:
            raise SystemExit(f"{path}: lane_extra {e['cell']!r}: {e['key']}/{e['view']} is not an IFEval (model, view)")
        parsed = C.parse_launcher_cell(str(e["cell"]))
        if parsed.kind != C.CELL_KIND_FITTED or parsed.si is not None:
            raise SystemExit(f"{path}: lane_extra {e['cell']!r} is not a bare fitted Class B cell id")
        if str(e["pins"]).startswith("/"):
            raise SystemExit(f"{path}: lane_extra {e['cell']!r}: `pins` is a path relative to the configs root")
        pol.lane_extra.append({k: str(e[k]).strip() for k in LANE_EXTRA_FIELDS})
    sp = doc.get("serving_path")
    if sp is not None:
        if sp not in SERVING_PATHS:
            raise SystemExit(f"{path}: `serving_path:` must be one of {SERVING_PATHS}, got {sp!r}")
        pol.serving_path = str(sp)
    return pol


def si_text_sha256(si_id: str) -> tuple[str, int]:
    """``(sha256 hex, chars)`` of the EXACT SI text — TrainTest's own resolver (``sisteer.eval.jailbreak.resolve_instruction``,
    the driver's ``cell_controls._cell_system`` path), never a copy of the text."""
    from sisteer.eval.jailbreak import resolve_instruction
    text = str(resolve_instruction(si_id).text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)


def assert_prompted_extra(entry: Mapping[str, Any], *, where: str) -> str:
    """The prompted extra's pinned text sha256 against the LIVE SI text (a moved constitution text would silently re-key
    every prompted row); returns the sha."""
    sha, n = si_text_sha256(entry["si"])
    if sha != entry["text_sha256"]:
        raise SystemExit(f"{where}: the {entry['si']} SI text's sha256 is {sha}, not the pinned {entry['text_sha256']} — "
                         "the instruction text moved; re-pin deliberately (views.yaml prompted_extra.text_sha256)")
    if entry.get("text_chars") is not None and n != entry["text_chars"]:
        raise SystemExit(f"{where}: the {entry['si']} SI text is {n} chars, not the pinned {entry['text_chars']}")
    return sha


# --------------------------------------------------------------------------- pure helpers over a row
def cell_short(cell: str) -> str:
    """The job-id token of a cell (the GSM8K lane's spelling): `none` / `prompted` / `constitution`, else
    `<si-short>-<arm-short>`; a Class B id (alphasteer, arditi) is its own token."""
    if cell == IC.CELL_NONE:
        return "none"
    if cell in _SHORT_PROMPTED:
        return _SHORT_PROMPTED[cell]
    if cell.startswith(IC.PROMPTED_PREFIX):
        return "prompted-" + cell[len(IC.PROMPTED_PREFIX):].replace("_", "-")
    si, _, arm = cell.partition("__")
    if si in _SHORT_SI and arm in _SHORT_ARM:
        return f"{_SHORT_SI[si]}-{_SHORT_ARM[arm]}"
    return cell.replace("__", "-")


def harness_cell_id(cell: str) -> str:
    """The harness (manifest) spelling of a cell — the harness's OWN function, never a re-typed table."""
    from sisteer.capeval.petri import registry_adapter as ra
    return ra.harness_cell_id(cell)


def image_for(family: str) -> str:
    return IC.IMAGE_NEMOTRON if family == "nemotron_h" else IC.IMAGE_DEFAULT


def budget_class(cell: str) -> str:
    """The prompt-budget artifact CLASS a cell shares: an object cell renders the same messages as `none`; a prompted cell
    adds ITS system turn (the safety and the constitution SIs are different classes)."""
    return cell if cell.startswith(IC.PROMPTED_PREFIX) else IC.CELL_NONE


def pin_record(*, kind: str, source: Optional[str] = None, source_sha256: Optional[str] = None,
               staged_id: Optional[str] = None, object_sha256: Optional[str] = None, record: Optional[str] = None) -> dict:
    """The per-row ``pin``: where the row's object identity is pinned. ``roster``: the pins file (+ its sha256) and the
    staged id whose ``object_sha256`` the row carries; ``none`` / ``prompted``: a STATED null object; ``deferred``: the
    per-model record the slot is filled from (``object_sha256`` null until then)."""
    out: dict[str, Any] = {"kind": kind, "object_sha256": object_sha256}
    if kind in (PIN_KIND_ROSTER, PIN_KIND_LANE_EXTRA):
        out.update(source=source, source_sha256=source_sha256, staged_id=staged_id)
    elif kind == PIN_KIND_DEFERRED:
        out.update(record=record, record_fields=list(DEFERRED_RECORD_FIELDS), filled=object_sha256 is not None)
    return out


def _spellings(cell: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """``(harness_cell, standup_cell, standup_harness_cell, pending)``: the harness translation of ``cell`` and the
    undeclared standup spellings — or, for a cell the harness vocabulary does not carry yet (a deferred slot's Class B
    id coined on another branch), the launcher's spelling alone with the gap NAMED, never guessed."""
    try:
        sp = C.cell_spellings(cell, declared=False)
        return harness_cell_id(cell), sp.traintest, sp.harness, None
    except Exception as e:  # noqa: BLE001 — the refusal's text IS the pending item
        return None, cell, None, f"harness vocabulary: {type(e).__name__}: {str(e).splitlines()[0][:160]}"


def build_row(*, index: int, key: str, view: str, cell: str, staged: Optional[str], pin: Optional[str],
              held_reason: Optional[str], roster_target: Mapping[str, Any], pin_source: Optional[Mapping[str, Any]] = None,
              si_text_sha256: Optional[str] = None, deferred: Optional[Mapping[str, Any]] = None,
              lane_extra: Optional[Mapping[str, Any]] = None, serving_path: Optional[str] = None) -> dict:
    """ONE matrix row (the IFEval grammar): every path from ``ifeval_paths`` at every rung. ``deferred`` (a slot's
    record block) makes a ``status: deferred`` row: fitted (an object arm) with a null pin until its record lands.
    ``lane_extra`` makes a fitted row whose object ANOTHER lane fit and this document declares staged in the
    view's bank (pin kind ``lane_extra``; ``object_source external``; ``bank_view`` = the source bank). ``serving_path``
 is stamped on the row when the document declares one; absent otherwise (the frozen rows are unchanged)."""
    is_deferred = deferred is not None
    fitted = staged is not None or is_deferred
    header_located = fitted and cell in IC.IFEVAL_HEADER_LOCATED_CELLS
    tag = IC.ifeval_tag(key, view)
    family = C.FAMILY_BY_MODEL_KEY[key]
    hf, rev = IC.IFEVAL_MODEL_PINS[key]
    if roster_target.get("model") != hf or roster_target.get("model_revision") != rev:
        raise SystemExit(f"{tag}: the roster serves {roster_target.get('model')}@{str(roster_target.get('model_revision'))[:8]} "
                         f"but the IFEval pin is {hf}@{rev[:8]} — the two pins disagree")
    if is_deferred:
        harness_cell, standup_cell, standup_harness, pending = _spellings(cell)
    else:
        sp = C.cell_spellings(cell, declared=False)
        harness_cell, standup_cell, standup_harness, pending = harness_cell_id(cell), sp.traintest, sp.harness, None
    row_sampling = IC.vendor_preset_row(key)
    paths = {rung: IC.ifeval_paths(key, view, cell, rung) for rung in IC.IFEVAL_CAP_LADDER}
    p1 = paths[IC.IFEVAL_PASS1_CAP]
    if is_deferred:
        status = IC.STATUS_DEFERRED
        pin_rec = pin_record(kind=PIN_KIND_DEFERRED, record=deferred["record"], object_sha256=pin)
    elif lane_extra is not None:
        status = IC.STATUS_HELD if held_reason else IC.STATUS_LAUNCH
        pin_rec = pin_record(kind=PIN_KIND_LANE_EXTRA, source=(pin_source or {}).get("source"),
                             source_sha256=(pin_source or {}).get("source_sha256"), staged_id=staged, object_sha256=pin)
    elif fitted:
        status = IC.STATUS_HELD if held_reason else IC.STATUS_LAUNCH
        pin_rec = pin_record(kind=PIN_KIND_ROSTER, source=(pin_source or {}).get("source"),
                             source_sha256=(pin_source or {}).get("source_sha256"), staged_id=staged, object_sha256=pin)
    else:
        status = IC.STATUS_HELD if held_reason else IC.STATUS_LAUNCH
        pin_rec = pin_record(kind=(PIN_KIND_PROMPTED if cell.startswith(IC.PROMPTED_PREFIX) else PIN_KIND_NONE))
    prompt_class = IC.ifeval_prompt_class_of_cell(cell) if not is_deferred else C.IFEVAL_PROMPT_CLASS_SHORT
    geometry = {str(c): IC.IFEVAL_GEOMETRY_ENTRY_BY_CLASS[prompt_class][c] for c in IC.IFEVAL_CAP_LADDER}
    row = {
        "index": index, "job_id": f"ife-{key}-{view}-{cell_short(cell)}",
        "model_key": key, "view": view, "tag": tag, "slug": IC.IFEVAL_SLUGS[(key, view)],
        "cell": cell, "cell_slug": p1.cell_slug, "harness_cell": harness_cell,
        "standup_cell": standup_cell, "standup_harness_cell": standup_harness,
        "staged_id": staged, "object_sha256": pin, "pin": pin_rec, "fitted": fitted,
        "route_kind": IC.ifeval_route_kind(cell), "header_located": header_located,
        "served_positions_expected": ("response" if header_located else "all") if fitted else None,
        "declared": False, "thinking": C.THINKING_OFF,
        "sampling_mode": IC.SAMPLING_VENDOR_PRESET, "sampling": IC.sampling_row_dict(row_sampling),
        "samples_per_item": IC.IFEVAL_SAMPLES_PER_ITEM,
        "expected_stamps": list(IC.ifeval_expected_stamps(fitted=fitted, header_located=header_located)),
        "cap": IC.IFEVAL_PASS1_CAP, "caps": list(IC.IFEVAL_CAP_LADDER), "cap_ceiling": IC.IFEVAL_CAP_CEILING,
        "trigger_threshold": IC.IFEVAL_TRIGGER_THRESHOLD, "trigger_provenance": IC.IFEVAL_TRIGGER_PROVENANCE,
        "prompt_class": prompt_class, "si_text_sha256": si_text_sha256,
        "geometry_eval_by_cap": geometry,
        "out_root": p1.out_root,
        "rung_out_roots": {str(r): paths[r].out_root for r in IC.IFEVAL_RUNG_CAPS},
        "subset_paths": {str(r): paths[r].subset for r in IC.IFEVAL_RUNG_CAPS},
        "cap_ladder_paths": {str(r): paths[r].cap_ladder for r in IC.IFEVAL_CAP_LADDER},
        "ladder_state_paths": {str(r): paths[r].ladder_state for r in IC.IFEVAL_CAP_LADDER},
        "budget_paths": {str(r): paths[r].budget for r in IC.IFEVAL_CAP_LADDER},
        "dispatch_paths": {str(r): paths[r].dispatch for r in IC.IFEVAL_CAP_LADDER},
        "configuration_ids": {str(r): paths[r].configuration_id for r in IC.IFEVAL_CAP_LADDER},
        "budget_class": budget_class(cell),
        "config": OVERLAY_CONFIG_REL.format(tag=tag), "manifest": OVERLAY_MANIFEST_REL.format(tag=tag),
        "hf": hf, "rev": rev, "family": family, "image": image_for(family),
        "status": status, "held_reason": held_reason,
        "deferred": None,
    }
    if is_deferred:
        row["deferred"] = {"arm": deferred["arm"], "record": deferred["record"], "record_fields": list(DEFERRED_RECORD_FIELDS),
                           "selection_rule": deferred["selection_rule"], "reason": deferred["reason"],
                           "view_provisional": True, "filled": pin is not None,
                           "pending": [p for p in (f"record {deferred['record']}" if pin is None else None, pending) if p]}
    # the keys below are APPENDED, present only on the rows that carry them — every frozen row is byte-identical
    if lane_extra is not None:
        row["object_source"] = OBJECT_SOURCE_EXTERNAL
        row["bank_view"] = str(lane_extra["bank"])
        row["lane_extra"] = {k: str(lane_extra[k]) for k in LANE_EXTRA_FIELDS}
    if serving_path is not None:
        row["serving_path"] = str(serving_path)
    return row


# --------------------------------------------------------------------------- THE selector
def _pin_source_for(key: str, source: Optional[Mapping[str, Any]]) -> Optional[dict]:
    """Which pins file a model key's object pins were read from (the Llama overlay for llama8b, the shared file else)."""
    if not source:
        return None
    if key == "llama8b" and source.get("pins_overlay"):
        return {"source": source["pins_overlay"], "source_sha256": source.get("pins_overlay_sha256")}
    return {"source": source.get("pins"), "source_sha256": source.get("pins_sha256")}


def select_rows(roster: dict, pins: Mapping, views: Mapping[str, Iterable[str]], *, holds: Iterable[dict] = (),
                policy: Optional[IfevalPolicy] = None, pins_source: Optional[Mapping[str, Any]] = None,
                where: str = "select_rows", configs_root: Optional[Path] = None) -> list[dict]:
    """The IFEval matrix rows for ``views`` (an ordered ``{key: [view, ...]}``) over ``roster`` + ``pins`` under the
    ``policy``: per (key, view) the GSM8K lane's neutral ``cells_of`` in order, minus the EXCLUDED cells, plus each
    ``prompted_extra`` cell right after the view's prompted cell (or, under ``all_views``, after ``none``); every fitted
    cell carries its staged id and pin; a hold stamps ``status: held``. Then the DEFERRED slots, one row per (arm,
    model) at the arm's provisional view, appended LAST. Index = position in the emitted list."""
    policy = policy or IfevalPolicy()
    spec = ViewsSpec(views={k: list(v) for k, v in views.items()}, holds=list(holds))
    extra_sha = {e["cell"]: assert_prompted_extra(e, where=where) for e in policy.prompted_extra}
    safety_sha, _n = si_text_sha256("safety_refusal")
    rows: list[dict] = []
    targets_of: dict[str, Mapping[str, Any]] = {}
    expressed = {(k, v) for k, vs in spec.views.items() for v in vs}
    expressed_views = {v for _k, v in expressed}
    for v, a in policy.anchor_view.items():
        if not {v, a} <= expressed_views:
            raise SystemExit(f"{where}: anchor_view {v!r} -> {a!r}: both views must be expressed by this document")
    for v in policy.view_cells:
        if v not in expressed_views:
            raise SystemExit(f"{where}: view_cells names {v!r}, a view this document does not express")
    for key, vlist in spec.views.items():
        if key not in IC.IFEVAL_MODELS:
            raise SystemExit(f"{where}: {key!r} is not an IFEval model ({IC.IFEVAL_MODELS})")
        for view in vlist:
            if view not in IC.IFEVAL_VIEWS:
                raise SystemExit(f"{where}: {key}/{view}: view {view!r} is not one of {IC.IFEVAL_VIEWS}")
            target = roster["targets"][key] if key in (roster.get("targets") or {}) else None
            targets_of[key] = target or {}
            cells = [(c, s) for c, s in cells_of(roster, key, view) if policy.exclusion_for(c) is None]
            if view in policy.view_cells:
                # a view expressing a SUBSET of its enumeration — by name, in enumeration order; a
                # listed cell the enumeration lacks refuses (a subset is never a way to add a cell)
                want = policy.view_cells[view]
                enumerated = [c for c, _ in cells]
                missing = [c for c in want if c not in enumerated]
                if missing:
                    raise SystemExit(f"{where}: {key}/{view}: view_cells names {missing}, not in the view's enumeration {enumerated}")
                cells = [(c, s) for c, s in cells if c in want]
            names = [c for c, _ in cells]
            for extra in policy.prompted_extra:
                if extra["cell"] in names:
                    raise SystemExit(f"{where}: {key}/{view}: the roster already serves {extra['cell']} — not an extra")
                has_prompted = any(c.startswith(IC.PROMPTED_PREFIX) for c in names)
                if extra["placement"] == ON_VIEWS_WITH_PROMPTED and not has_prompted:
                    continue
                anchor = max([i for i, c in enumerate(names) if c.startswith(IC.PROMPTED_PREFIX)] or [0])
                cells.insert(anchor + 1, (extra["cell"], None))
                names.insert(anchor + 1, extra["cell"])
            for cell, staged in cells:
                fitted = staged is not None
                pin = pin_for(pins, key, staged, where=f"{where}: {IC.ifeval_tag(key, view)}/{cell}") if fitted else None
                si_sha = extra_sha.get(cell) or (safety_sha if cell == IC.CELL_PROMPTED else None)
                rows.append(build_row(index=len(rows), key=key, view=view, cell=cell, staged=staged, pin=pin,
                                      held_reason=spec.hold_for(key, view, cell), roster_target=target or {},
                                      pin_source=(_pin_source_for(key, pins_source) if fitted else None),
                                      si_text_sha256=si_sha, serving_path=policy.serving_path))
    # the lane_extra rows — a fitted Class B cell another lane fit, declared staged in the view's bank, pinned
    # from ITS pins file (read off the configs root; the sha the row carries is that file's `object_sha256`); appended
    # after every enumerated row (list order), before the deferred slots
    for entry in policy.lane_extra:
        key, view = entry["key"], entry["view"]
        if (key, view) not in expressed:
            raise SystemExit(f"{where}: lane_extra {entry['cell']!r}: {key}/{view} is not a view this document expresses")
        if any(r["model_key"] == key and r["view"] == view and r["cell"] == entry["cell"] for r in rows):
            raise SystemExit(f"{where}: lane_extra {entry['cell']!r} is already a row of {key}/{view}")
        if configs_root is None:
            raise SystemExit(f"{where}: a document with `lane_extra:` needs the configs root to read {entry['pins']}")
        pins_path = Path(configs_root) / entry["pins"]
        if not pins_path.is_file():
            raise SystemExit(f"{where}: lane_extra {entry['cell']!r}: no pins file at {pins_path}")
        lane_pins = json.loads(pins_path.read_text(encoding="utf-8"))
        pin = pin_for(lane_pins, key, entry["staged_id"], where=f"{where}: {IC.ifeval_tag(key, view)}/{entry['cell']} (lane_extra)")
        lane_entry = (lane_pins.get(key) or {}).get(entry["staged_id"]) or {}
        # the pins entry's `view` IS the source bank view the row records as bank_view (the GSM8K seed-replicate Arditi rows'
        # spelling: view s<s>, bank ar_c0p3); its `identity_sidecar` is the sidecar the registry validates the object by
        for field_name, doc_key in (("view", "bank"), ("identity_sidecar", "sidecar")):
            got, want = lane_entry.get(field_name), entry[doc_key]
            if got is not None and str(got) != want:
                raise SystemExit(f"{where}: lane_extra {entry['cell']!r}: the pins file records {field_name}={got!r}, "
                                 f"the document's `{doc_key}` says {want!r}")
        rows.append(build_row(index=len(rows), key=key, view=view, cell=entry["cell"], staged=entry["staged_id"], pin=pin,
                              held_reason=spec.hold_for(key, view, entry["cell"]),
                              roster_target=targets_of.get(key) or roster["targets"][key],
                              pin_source={"source": entry["pins"], "source_sha256": sha256_of(pins_path)},
                              lane_extra=entry, serving_path=policy.serving_path))
    for slot in policy.deferred:
        for key, view in slot["views"].items():
            if key not in spec.views:
                continue                                    # a slot for a model this matrix does not express
            if view not in spec.views[key]:
                raise SystemExit(f"{where}: deferred {slot['arm']!r}: {key}/{view} is not a view this matrix expresses")
            record = slot["record"].format(model_key=key)
            rows.append(build_row(index=len(rows), key=key, view=view, cell=slot["cell"], staged=None, pin=None,
                                  held_reason=None, roster_target=targets_of.get(key) or roster["targets"][key],
                                  deferred={**slot, "record": record}, serving_path=policy.serving_path))
    return rows


# --------------------------------------------------------------------------- the contrasts (P32)
def contrast_resolution(rows: list[dict], *, anchor_view: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Every ``IFEVAL_CONTRASTS`` family resolved over the LAUNCH rows: ``cell_minus_none`` needs a launched ``none`` in
    each launched cell's own view — or, under ``anchor_view``, in the view the document names as the
    cell's anchor view (the composed ``s<s>p`` rows pair with ``s<s>/none``; the leg records ``anchor_view``);
    ``none_it_minus_none_main`` exists for every model with BOTH instrument views launched (a one-view model has no
    floor — stated); ``merged_minus_r2048`` is conditional on a promotion (always resolvable: every promoted cell has
    its own pass-1 leg). Held and deferred rows are never legs. Returns the resolution + the unresolved legs (empty =
    the check passes)."""
    anchor_view = dict(anchor_view or {})
    launch = [r for r in rows if r["status"] == IC.STATUS_LAUNCH]
    none_by_view = {(r["model_key"], r["view"]) for r in launch if r["cell"] == IC.CELL_NONE}
    vs_none = [{"model_key": r["model_key"], "view": r["view"], "cell": r["cell"],
                **({"anchor_view": anchor_view[r["view"]]} if r["view"] in anchor_view else {}),
                "anchor": (r["model_key"], anchor_view.get(r["view"], r["view"])) in none_by_view}
               for r in launch if r["cell"] != IC.CELL_NONE]
    models = list(dict.fromkeys(r["model_key"] for r in rows))
    views_of = {m: sorted({r["view"] for r in launch if r["model_key"] == m}) for m in models}
    noise = [{"model_key": m, "views": views_of[m], "resolves": set(views_of[m]) >= set(IC.BANK_VIEWS)} for m in models]
    unresolved = [f"{IC.CONTRAST_VS_NONE}: {c['model_key']}/{c['view']}/{c['cell']} has no launched none anchor"
                  + (f" (anchor view {c['anchor_view']})" if c.get("anchor_view") else "")
                  for c in vs_none if not c["anchor"]]
    return {
        "contrasts": list(IC.IFEVAL_CONTRASTS),
        IC.CONTRAST_VS_NONE: {"n": len(vs_none), "n_resolved": sum(c["anchor"] for c in vs_none), "legs": vs_none},
        IC.CONTRAST_NOISE: {"models_with_floor": [n["model_key"] for n in noise if n["resolves"]],
                            "models_without_floor": [n["model_key"] for n in noise if not n["resolves"]]},
        IC.CONTRAST_TRUNCATION: {"conditional_on": "a promoted cell (its pass-1 leg is its own anchor)"},
        "deferred_legs": [{"model_key": r["model_key"], "view": r["view"], "cell": r["cell"],
                           "anchor": (r["model_key"], anchor_view.get(r["view"], r["view"])) in none_by_view}
                          for r in rows if r["status"] == IC.STATUS_DEFERRED],
        "unresolved": unresolved,
    }


def check_contrasts(rows: list[dict], *, where: str = "matrix", anchor_view: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    res = contrast_resolution(rows, anchor_view=anchor_view)
    if res["unresolved"]:
        raise SystemExit(f"{where}: {len(res['unresolved'])} promised contrast leg(s) do not resolve among the launch rows — "
                         + "; ".join(res["unresolved"]))
    # a deferred slot must have its none anchor in its (provisional) view too, or filling it could never resolve
    orphans = [d for d in res["deferred_legs"] if not d["anchor"]]
    if orphans:
        raise SystemExit(f"{where}: {len(orphans)} deferred slot(s) sit in a view with no launched none anchor — "
                         + "; ".join(f"{d['model_key']}/{d['view']}/{d['cell']}" for d in orphans))
    return res


# --------------------------------------------------------------------------- the document
def matrix_document(rows: list[dict], *, spec: ViewsSpec, source: Mapping[str, Any], views_file: Optional[Path],
                    configs_root: Path, policy: Optional[IfevalPolicy] = None) -> dict:
    """The frozen matrix's document: the rows, the provenance of every input (RELATIVE paths + sha256; no timestamp,
    so a regeneration over the same inputs is byte-identical), the policy, the counts and the contrast resolution."""
    def rel(p):
        return str(Path(p).resolve().relative_to(Path(configs_root).resolve())) if p is not None else None
    policy = policy or IfevalPolicy()
    used = {r["slug"] for r in rows}
    return {
        "schema": IC.MATRIX_SCHEMA,
        "generated_by": GENERATED_BY,
        "experiment": "ifeval-canonical-battery",
        "task": IC.IFEVAL_TASK, "control": IC.CONTROL_IFEVAL, "n_items": IC.IFEVAL_N,
        "samples_per_item": IC.IFEVAL_SAMPLES_PER_ITEM, "thinking": C.THINKING_OFF,
        "sampling_mode": IC.SAMPLING_VENDOR_PRESET, "caps": list(IC.IFEVAL_CAP_LADDER),
        "trigger_threshold": IC.IFEVAL_TRIGGER_THRESHOLD, "trigger_rule": IC.IFEVAL_TRIGGER_RULE,
        "trigger_provenance": IC.IFEVAL_TRIGGER_PROVENANCE, "seed_rule": IC.SEED_RULE_IFEVAL,
        "views_spec": spec.spec_string(),
        "views": spec.views,
        "holds": spec.holds,
        "policy": policy.as_record(),
        "prompt_classes": {cls: {str(c): e for c, e in by_cap.items()} for cls, by_cap in C.IFEVAL_GEOMETRY_ENTRY_BY_CLASS.items()},
        "source": {"views_yaml": rel(views_file), **dict(source)},
        # the slugs THIS document's rows serve, in IFEVAL_SLUGS order (the registry now carries thirteen; the frozen
        # document's seven are exactly the slugs its rows use, so its bytes are unchanged)
        "slugs": {IC.IFEVAL_SLUGS[(k, v)]: {"model_key": k, "view": v} for k, v in IC.IFEVAL_SLUGS if IC.IFEVAL_SLUGS[(k, v)] in used},
        **({"serving_path": policy.serving_path} if policy.serving_path is not None else {}),
        "counts": IM.counts(rows),
        "contrasts": contrast_resolution(rows, anchor_view=policy.anchor_view),
        "rows": rows,
    }


def render_matrix(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


# --------------------------------------------------------------------------- matrix -> driver argv (Appendix A)
def driver_argv_for_row(row: Mapping[str, Any], *, rung: int, python: str, script: str, proxy_url: str,
                        routing_table: str, ledger_url: str, defenses_dir: str, out_root: str, budget: Optional[str],
                        concurrency: int, vllm_log: str, attempt_id: str, judge_model: str,
                        subset_path: Optional[str] = None, repo_root: Optional[Path] = None,
                        top_logprobs: Optional[int] = None) -> list[str]:
    """The ONE driver argv shape for a matrix row at ``rung`` — ``contracts.driver_argv``, the composer the launcher's
    ``_driver_cmd`` calls (plan Appendix A / §8.1), with the IFEval tail the contracts give it: ``--sampling
    vendor_preset``, identity version 2, the cap statement ``ifeval=<cap>``, and on a rung (> pass 1) the OFF setter
    ``ifeval=<cap>`` + ``--item-subset <subset_path>``; ``--expect-object-sha256 <pin>`` on a fitted row; ``--bank-view
    <view>``; ``--oos-prompt-budget <budget>`` (the launcher's container writes the artifact for every IFEval cell and
    passes its path); ``--attempt-id`` LAST (the composer's own keyword). A held or deferred row REFUSES by name (P41).
    ``top_logprobs`` defaults to the row's registry width, the number the launcher passes."""
    if row["status"] == IC.STATUS_HELD:
        raise SystemExit(f"{row['job_id']}: HELD ({row['held_reason']}) — no driver argv is built for a held cell (P41)")
    if row["status"] == IC.STATUS_DEFERRED:
        raise SystemExit(f"{row['job_id']}: DEFERRED ({(row.get('deferred') or {}).get('record')}) — no driver argv is built "
                         "for a design slot whose object sha is not filled")
    if int(rung) not in IC.IFEVAL_CAP_LADDER:
        raise SystemExit(f"rung must be one of {IC.IFEVAL_CAP_LADDER}, got {rung!r}")
    is_rung = int(rung) != IC.IFEVAL_PASS1_CAP
    if is_rung and not subset_path:
        raise SystemExit(f"{row['job_id']}: a rung at {rung} needs its draw-subset path")
    cfg = str((repo_root or REPO) / row["config"]) if repo_root is not None else row["config"]
    width = int(top_logprobs) if top_logprobs is not None else int(C.model_row(row["slug"]).top_logprobs)
    argv = C.driver_argv(
        python=python, script=script, proxy_url=proxy_url, routing_table=routing_table,
        cell=row["standup_cell"], config=cfg, defenses_dir=defenses_dir, out_root=out_root,
        controls=(IC.CONTROL_IFEVAL,), thinking=C.THINKING_OFF, concurrency=int(concurrency),
        judge_model=judge_model, samples=IC.IFEVAL_SAMPLES_PER_ITEM, ledger_url=ledger_url,
        top_logprobs=width, vllm_log=vllm_log,
        expect_object_sha256=(row["object_sha256"] if row["fitted"] else None),
        oos_prompt_budget=budget,
        sampling_mode=IC.SAMPLING_VENDOR_PRESET, identity_version=C.TINYCAP_IDENTITY_VERSION_2,
        max_tokens_by_task={IC.IFEVAL_TASK_NAME: int(rung)},
        off_cap_by_task=({IC.IFEVAL_TASK_NAME: int(rung)} if is_rung else None),
        item_subset=(subset_path if is_rung else None),
        bank_view=row["view"], attempt_id=attempt_id)
    assert "@thinking" not in " ".join(argv)
    return argv
