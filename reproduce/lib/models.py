#!/usr/bin/env python3
"""reproduce/lib/models.py — the ONE reader of `reproduce/models.yaml` (PyYAML + stdlib).

    from models import load_models, model, eligible_layers, lora_modules          # (sys.path: reproduce/lib)
    python reproduce/lib/models.py [--table PATH] keys
    python reproduce/lib/models.py [--table PATH] exists <key>
    python reproduce/lib/models.py [--table PATH] get <key> <dotted.path>        # scalar plain; list space-joined; map JSON
    python reproduce/lib/models.py [--table PATH] views <key>                    # every view (u0 s2 d0p75 d0p75_relu; the declared wave, e.g. battery)
    python reproduce/lib/models.py [--table PATH] phase-views <key>              # the PHASE views: the grammar's plus the row's declared `wave`
    python reproduce/lib/models.py [--table PATH] roster-block <key> <views>     # the target's COMPLETE block for the wave roster
    python reproduce/lib/models.py [--table PATH] launcher-cells <key> <view> <cells|all>   # `<view cell>\t<launcher token>` per cell (07)
    python reproduce/lib/models.py [--table PATH] launcher-slug <key> <view>                 # the launcher's registry slug for the view (07 --model)
    python reproduce/lib/models.py [--table PATH] check                          # schema + pins

The loader REJECTS duplicate mapping keys (review A9: PyYAML keeps the last of two `fit:` blocks silently), and
`check` asserts every model keeps BOTH checkpoint ids and both 40-hex revisions after loading. Everything here is PyYAML +
stdlib, except `launcher-cells`, which imports the Modal launcher's own grammar (src/sisteer/capeval/tinycap/contracts.py,
torch-free) to check every token it prints against what the launcher plans with — never a second copy of that grammar.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.yaml")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
#: set id -> the sorted `lora_modules` list; MODULE_SETS is its key order
LORA_MODULES = {
    "wo": ["o_proj"],
    "qkvo": ["k_proj", "o_proj", "q_proj", "v_proj"],
    "attn_mlp": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
    "mlp": ["down_proj", "gate_proj", "up_proj"],
    "down": ["down_proj"],
}
MODULE_SETS = tuple(LORA_MODULES)
ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}


class DuplicateKeyError(ValueError):
    pass


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DuplicateKeyError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_models(path: Optional[str] = None) -> Dict[str, dict]:
    path = path or os.environ.get("SISTEER_MODELS_TABLE") or DEFAULT_TABLE
    with open(path, encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_StrictLoader)
    if not isinstance(doc, dict) or doc.get("schema") != 1 or not isinstance(doc.get("models"), dict):
        raise ValueError(f"{path}: expected {{schema: 1, models: {{...}}}}")
    return doc["models"]


def model(key: str, path: Optional[str] = None) -> dict:
    models = load_models(path)
    if key not in models:
        raise KeyError(f"model key {key!r} is not in the table (keys: {sorted(models)})")
    return models[key]


def _get(doc: Any, dotted: str) -> Any:
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


def eligible_layers(key: str, module_set: str, path: Optional[str] = None) -> List[int]:
    """FD 6 'every eligible layer': the block indices a module set adapts on this model."""
    m = model(key, path)
    arch = m["architecture"]
    n = int(arch["n_layers"])
    rule = m["layer_sets"][module_set]
    if rule == "attention_layers":
        return [int(i) for i in arch["attention_layers"]]
    if rule == "all":
        return list(range(n))
    if isinstance(rule, list):
        return [int(i) for i in rule]
    raise ValueError(f"{key}: layer_sets.{module_set} = {rule!r} is neither attention_layers, all nor a list")


def lora_modules(key: str, set_id: str, path: Optional[str] = None) -> Dict[str, List[int]]:
    """{module_name: [layers]} for a set id on a model — attention modules on the attention layers, MLP modules on
    every block that has an MLP (the MoE's shared + routed experts included; the probe decides servability)."""
    if set_id not in LORA_MODULES:
        raise KeyError(f"module set {set_id!r} is not one of {MODULE_SETS}")
    out: Dict[str, List[int]] = {}
    for mod in LORA_MODULES[set_id]:
        if mod in ATTENTION_MODULES:
            out[mod] = eligible_layers(key, "qkvo" if set_id != "wo" else "o_proj", path)
        else:
            out[mod] = eligible_layers(key, "mlp" if set_id != "down" else "down", path)
    return out


def per_expert_rank(rank: int, active_experts: int) -> Dict[str, int]:
    """FD 6: shared expert at the full rank; routed experts at max(1, rank // active_experts)."""
    return {"shared_expert": int(rank), "routed_expert": max(1, int(rank) // int(active_experts))}


def views(key: str, path: Optional[str] = None) -> List[str]:
    return list(model(key, path).get("views", {}).keys())


def excluded_families(key: str, view: str, path: Optional[str] = None) -> dict:
    """``{family: reason}`` the table EXCLUDES from a phase view BY RULE (`views.<view>.exclude_families` with its `exclude_reason`;
    the padding ladder never runs on the s2r objects). An explicit, tested exclusion the padding generator,
    the planner and 06's `--family all` all read — never the phase-view grammar's accident. {} when the view
    excludes nothing; a list without its reason refuses by name (a rule names its reason)."""
    vs = model(key, path).get("views") or {}
    if view not in vs:
        raise KeyError(f"{key}: no view {view!r} (views: {sorted(vs)})")
    vd = vs[view] or {}
    fams = vd.get("exclude_families") or []
    if fams and not vd.get("exclude_reason"):
        raise ValueError(f"{key}/{view}: exclude_families {list(fams)} without an exclude_reason — a ruling names its reason")
    return {str(f): str(vd["exclude_reason"]) for f in fams}


#: a PHASE view of the evaluation waves (u0 / s<k> / d<dose>[_relu] / dm<dose>[_relu]) — scripts/sdfbase/make_full22_configs.PHASE_VIEW_RE,
#: mirrored byte for byte (tests/test_mmlu_full_native_registration.py holds the two predicates equal over the table); the paper's own
#: models sit in the table for their pins with `main` / `it` only, which are NOT phase views
#: `s<N>r` = the ReLU-parent subset of phase N (`s2r` = control_relu + constitution_relu at alpha 1; a view outside this grammar
#: renders nothing here while the planner still plans it) — the same grammar in every copy, pinned equal by tests
PHASE_VIEW_RE = re.compile(r"^(u0|s[0-9]+r?p?|s[0-9]+rl|d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")
#: a STEERED-WAVE phase view (`s<N>r`): steered-only — no unsteered `none` leg is generated, planned, registered
#: or scored for it; mirror of scripts/sdfbase/full22/f22_roster.STEERED_WAVE_VIEW_RE (tests hold the copies equal)
STEERED_WAVE_VIEW_RE = re.compile(r"^s[0-9]+rl?$")
UNSTEERED_LEGS = ("none", "none__lora0", "none__lens0")
#: the row-level DECLARED WAVE (demo-import M3; Q2 ruled): `wave: <name>` names the single-phase recipe's ONE steered wave by a name OUTSIDE
#: the phase-view grammar (`battery` on the worked example — never `s<N>`, which spells the paper's fit-seed replicates). A scalar names
#: exactly one wave, so the recipe's shape ("one wave plus dose rungs") is the table's STRUCTURE, not a count the reader enforces. The ONE
#: reader rule lives here (declared_wave / phase_views / is_phase_view): every generator and grader loads this module by importlib and the
#: seven PHASE_VIEW_RE copies stay byte-identical. `check()` refuses a wave naming no view, a dose view, a grammar name, u0 / main / it,
#: the padding generator's hand-listed `obj` / `obj_s<N>`, a `prompted__` name, a wave on a u0 (phased) row, and — on a no-u0 front-door
#: row — any view that is neither grammar nor the declared wave (the silent-drop class: a generator filtering by grammar renders nothing
#: for it while the planner plans it).
WAVE_KEY = "wave"
RESERVED_VIEW_NAMES = ("u0", "main", "it")
_PADDING_HAND_LISTED_FAMILY = re.compile(r"^obj(_s[0-9]+)?$")


def declared_wave_of(m: dict) -> Optional[str]:
    """The row's declared wave (`wave: <name>`), None when it declares none."""
    w = (m or {}).get(WAVE_KEY)
    return str(w) if w is not None and w != "" else None


def declared_wave(key: str, path: Optional[str] = None) -> Optional[str]:
    return declared_wave_of(model(key, path))


def phase_views_of(m: dict) -> List[str]:
    """The PHASE views of a row, in the table's order: the views matching PHASE_VIEW_RE plus the declared wave (the ONE classification rule)."""
    w = declared_wave_of(m)
    return [str(v) for v in (m.get("views") or {}) if PHASE_VIEW_RE.match(str(v)) or (w is not None and str(v) == w)]


def phase_views(key: str, path: Optional[str] = None) -> List[str]:
    return phase_views_of(model(key, path))


def is_phase_view(key: str, view: str, path: Optional[str] = None) -> bool:
    """Whether `view` is a phase view of `key` — the ONE rule for a (key, view) pair a caller holds without the row (the grader's roster
    enumeration, the daemon's padded-run split): a name of the phase-view grammar, for EVERY key (the paper's roster views — `gemma4e4b_s1`,
    `qwen35_4b_d0p1_relu` — are grammar views whether or not the models table carries the key, and the table's paper rows carry `main` / `it`
    only), or the declared wave of a key the table carries. Never narrower than the grammar: a rule that read a table key's ROW alone dropped
    41 paper roster targets from the grader the first time it ran (demo-import M3's sweep caught it)."""
    v = str(view)
    if PHASE_VIEW_RE.match(v):
        return True
    m = load_models(path).get(key)
    return isinstance(m, dict) and declared_wave_of(m) == v


def steered_wave_view(view: str, key: Optional[str] = None, path: Optional[str] = None) -> bool:
    """A view that measures NO unsteered leg — the front-door planner's mirror of f22_roster.steered_wave_view:
    `s<N>r` by name; and every PHASE view except u0 of a model that RUNS IN PHASES — one whose table declares a `u0` view (the
    unsteered-first wave the others are measured against). A front-door model that declares NO u0 view runs the SINGLE-PHASE
    recipe: each of its waves carries its own unsteered legs (`none` + the prompted instructions) and is never a
    steered wave, so nothing drops or refuses them (`single_phase`). With `key`, a model that names no phase view (a paper model's
    `main` / `it`) is never a steered wave."""
    v = str(view)
    if STEERED_WAVE_VIEW_RE.match(v):
        return True
    if key is not None and (key not in front_door_keys(path) or single_phase(key, path)):
        return False
    return bool(PHASE_VIEW_RE.match(v)) and v != "u0"


def single_phase(key: str, path: Optional[str] = None) -> bool:
    """The SINGLE-PHASE recipe: a front-door model whose table declares no `u0` view — every arm is evaluated in
    one steered wave against its own unsteered legs, and the unsteered-first gate (06 --gate) has nothing to gate on (it prints a note
    and admits). A model that declares `u0` runs in phases (u0 first, the steered waves gated behind it) as before."""
    return key in front_door_keys(path) and "u0" not in views(key, path)


def baseline_view(key: str, path: Optional[str] = None) -> str:
    """The wave the unsteered-first gate's inventory is frozen from and the other waves are measured against: `u0` when the table
    declares it (the phased recipe), else — the SINGLE-PHASE recipe — the model's ONE steered wave `s<N>` (its phase view that is
    neither u0 nor a dose view: the wave that carries every arm beside its own unsteered legs). A single-phase table with no such
    view, or more than one, refuses by name: the recipe is one wave plus dose rungs."""
    m = model(key, path)
    vs = [str(v) for v in (m.get("views") or {})]
    if "u0" in vs:
        return "u0"
    declared = declared_wave_of(m)
    waves = [v for v in vs if (PHASE_VIEW_RE.match(v) and not STEERED_WAVE_VIEW_RE.match(v) and not v.startswith(("d", "dm"))) or v == declared]
    if len(waves) != 1:
        raise SystemExit(f"REFUSED: {key}: a table with no u0 view runs the single-phase recipe — exactly ONE steered wave (the row's declared "
                         f"`wave:`, or a grammar `s<N>` view) plus dose views; it declares {waves or 'none'} (views: {vs}; wave: {declared})")
    return waves[0]


def front_door_keys(path: Optional[str] = None) -> List[str]:
    """The FRONT-DOOR models: every table model that names at least one phase view (make_full22_configs.front_door_models' rule: the
    grammar's views or the row's declared wave) — the models the front door stages, evaluates and gates (their mmlu_full job
    registers natively)."""
    return [k for k, m in load_models(path).items() if phase_views_of(m)]


# --- the wave roster's block for a front-door target (05_stage_roster.sh --step roster) ---------------------------------------
#: the battery body every full22 config runs (the nemotron3nano4b / glm46vflash precedent in configs/sdfbase_full22_roster.yaml:
#: a target with no sdfbase transfer lineage of its own takes the qwen35_4b _r3 body's data; make_full22_configs.py substitutes
#: the served model / model_revision from the block, which the printer takes from THIS table's serve.checkpoint)
FRONT_DOOR_BODY_CONFIG = "configs/sdfbase_transfer_qwen35_4b_r3.yaml"
#: the object lane whose `cell_id` spells `<rung>__<harness id>` (ar_c0p3__arditi_allpos): the rung is the STAGED id's, never the
#: roster's config cell (the arditi_allpos roster's own spelling `cells: {arditi_allpos: ar_c0p3__arditi_allpos}`;
#: make_padding_configs.OBJ_ARDITI_CELL "the RUNG is the pin's — never part of the cell id")
ARDITI_LANE = "arditi"
#: the cells of a view that carry no object (no staged id, no pin): the same-wave bare anchor and the prompted SI baselines
NO_OBJECT_PREFIX = "prompted__"


def object_of(m: dict, cell: str) -> dict:
    """The `objects` entry of a view cell of a model; refuses a cell the table names in a view but not under `objects`."""
    objs = m.get("objects") or {}
    if cell not in objs or not isinstance(objs[cell], dict):
        raise KeyError(f"view cell {cell!r} has no `objects` entry (objects: {sorted(k for k, v in objs.items() if isinstance(v, dict))})")
    for f in ("lane", "cell_id"):
        if not objs[cell].get(f):
            raise KeyError(f"objects.{cell}.{f} missing")
    return objs[cell]


def roster_cell_id(m: dict, cell: str) -> str:
    """The wave roster's CONFIG cell id (the code's cell grammar: none | prompted__<si> | <si>__<arm> | a bare Class B id) for a
    view cell of the models table: `none` and `prompted__*` unchanged; an object cell is its object's own `cell_id`
    (`constv2-mix-full__psr_ce_all_minus_ctl_distilled`, …) — EXCEPT an ARDITI_LANE object, whose cell_id spells
    `<rung>__<harness id>`: the roster cell is the harness id (`arditi_allpos`), the rung rides the staged id only.
    (The printer's `<corpus>__<arm>` heuristic mapped refusal_direction onto control_relu's key
    `constv2ctl2-full__psr_ce_all` — a DUPLICATE mapping key. The object's own id is the one spelling.)"""
    if cell == "none" or cell.startswith(NO_OBJECT_PREFIX):
        return cell
    o = object_of(m, cell)
    cid = str(o["cell_id"])
    if o["lane"] == ARDITI_LANE:
        if "__" not in cid:
            raise ValueError(f"objects.{cell}.cell_id {cid!r}: an {ARDITI_LANE}-lane object spells `<rung>__<harness id>`")
        return cid.split("__", 1)[1]
    return cid


# --- the worked example's COMMITTED objects (demo-import): a row that declares `committed_objects_of` imports the paper's committed objects
# (objects/DEMO.json, through reproduce/tools/import_objects.py) in place of stage 04's fits. The value names the paper row whose objects are
# imported — a TABLE KEY resolves to that row's FIT checkpoint model_id (the objects of its fit lane, qwen35_4b -> qwen35_4b_base), a
# CHECKPOINT model_id of any table row is taken as spelled (the refusal direction is a served-checkpoint object: qwen35_4b_it); a view cell
# may name its own (`objects.<cell>.committed_objects_of`), else the row's applies. The resolved id is objects/MANIFEST.json's `model_id`.
COMMITTED_OBJECTS_KEY = "committed_objects_of"


def committed_objects_of(m: dict, cell: Optional[str] = None) -> Optional[str]:
    """The `committed_objects_of` a view cell imports under: the cell's own entry when it declares one, else the row's; None when neither
    does (the fit-your-own path: stage 04 fits, stage 05 stages from the cluster bank)."""
    if cell is not None:
        o = (m.get("objects") or {}).get(cell)
        if isinstance(o, dict) and o.get(COMMITTED_OBJECTS_KEY):
            return str(o[COMMITTED_OBJECTS_KEY])
    v = m.get(COMMITTED_OBJECTS_KEY)
    return str(v) if v else None


def committed_model_id(spelling: str, path: Optional[str] = None) -> str:
    """The registry model_id (objects/MANIFEST.json `model_id`) a `committed_objects_of` spelling names: a table key -> its row's
    fit.checkpoint.model_id; a fit / serve checkpoint model_id of any table row -> itself; anything else refuses by name."""
    models = load_models(path)
    if spelling in models:
        return str(((models[spelling].get("fit") or {}).get("checkpoint") or {}).get("model_id") or "") or _refuse_committed(spelling, models)
    ids = sorted({str(((m.get(side) or {}).get("checkpoint") or {}).get("model_id") or "") for m in models.values() for side in ("fit", "serve")} - {""})
    if spelling in ids:
        return spelling
    return _refuse_committed(spelling, models)


def _refuse_committed(spelling: str, models: Dict[str, dict]) -> str:
    ids = sorted({str(((m.get(side) or {}).get("checkpoint") or {}).get("model_id") or "") for m in models.values() for side in ("fit", "serve")} - {""})
    raise ValueError(f"{COMMITTED_OBJECTS_KEY} {spelling!r} is neither a table key ({sorted(models)}) nor a checkpoint model_id of a table row ({ids}) — "
                     "the committed objects are filed under a paper row's checkpoint id (objects/MANIFEST.json model_id)")


_DOSE_VIEW_TOKEN = re.compile(r"^(?:it_)?d(m?[0-9]+(?:p[0-9]+)?)(?:_[a-z][a-z0-9]*)?$")


def dose_token_of_view(view: str) -> Optional[str]:
    """The dose token a DOSE view's name spells (`d0p75` / `d0p75_relu` -> `0p75`; make_full22_configs._DOSE_VIEW_RE's grammar),
    None for a view that is not a dose view."""
    m = _DOSE_VIEW_TOKEN.match(view)
    return m.group(1) if m else None


#: D0P75RELUL: a DOSE view may serve a copy scaled from a SIBLING LANE's parent (the LeCun-init refit of the
#: composed difference, `difference_relu_lecun`, bank full22_lecun) — the same cell id as the zero-init parent's, so the copy's staged id
#: carries a LANE MARKER right after the dose token (`d0p75l__<cell_id>`, the `d<dose>it__` / `d<dose>it64__` qualifier shape of
#: legacy_spellings.yaml's dose-copy row) and the view's name carries it after its family segment (`d0p75_relul`, as `s2rl` is `s2r`'s
#: LeCun lane). The roster STATES the lane (`object_lane: lecun`, beside `object_family:`); nothing is inferred from the cell's bank.
DOSE_LANE_MARKERS = {"lecun": "l"}


def dose_lane_marker(vd: dict, view: str) -> str:
    """The staged-id marker of a dose view's declared `object_lane` ("" when it declares none); refuses an unknown lane and a view
    whose name does not end in the lane's marker (the name and the key must agree, make_full22_configs.view_object_lane's rule)."""
    lane = vd.get("object_lane")
    if lane is None:
        return ""
    mark = DOSE_LANE_MARKERS.get(str(lane))
    if mark is None:
        raise ValueError(f"view {view!r}: object_lane {lane!r} is not a known dose lane ({sorted(DOSE_LANE_MARKERS)})")
    if not view.endswith(mark):
        raise ValueError(f"view {view!r} declares object_lane {lane!r} but its name does not end in the lane marker {mark!r} (`d<tok>_<family>{mark}`)")
    return mark


def staged_id(m: dict, cell: str, view: Optional[str] = None) -> Optional[str]:
    """The wave bank's STAGED id of a view cell: `<lane>__<cell_id>` (stage_objects.py's bank spelling: the fitted / derived / arditi
    PARENT the 05 stage stages); on a DOSE view (`dose:` declared) the SCALED COPY's id `d<tok>__<cell_id>` (strength/build_strength.py's
    copy lane — the roster's own spelling of every measured dose view, e.g. `d0p25__constv2-mix-full__lora_r4_ce_minus_ctl`; the copy is
    DIFFERENT BYTES from the parent, built and staged by 09_dose_ladder.sh --step build,stage). None for a cell with no object."""
    if cell == "none" or cell.startswith(NO_OBJECT_PREFIX):
        return None
    o = object_of(m, cell)
    if view is not None and "dose" in ((m.get("views") or {}).get(view) or {}):
        tok = dose_token_of_view(view)
        if tok is None:
            raise ValueError(f"view {view!r} declares `dose:` but its name spells no `d<tok>` (make_full22_configs.view_dose refuses it too)")
        return f"d{tok}{dose_lane_marker((m.get('views') or {}).get(view) or {}, view)}__{o['cell_id']}"   # D0P75RELUL: `d0p75l__<cell>` on the LeCun lane
    return f"{o['lane']}__{o['cell_id']}"


def roster_view_cells(m: dict, view: str) -> Dict[str, Optional[str]]:
    """``{config cell id: staged id | None}`` of a phase view, in the table's order, `none` excluded (the generator adds the anchor);
    refuses two view cells that map to ONE roster cell id (the duplicate the printer used to emit)."""
    vd = (m.get("views") or {}).get(view)
    if vd is None:
        raise KeyError(f"no view {view!r} (views: {sorted(m.get('views') or {})})")
    out: Dict[str, Optional[str]] = {}
    for c in vd.get("cells", []):
        if c == "none":
            continue
        cid = roster_cell_id(m, c)
        if cid in out:
            raise ValueError(f"view {view}: cells {c!r} and an earlier one both map to the roster cell id {cid!r} — a duplicate mapping key")
        out[cid] = staged_id(m, c, view)
    return out


# --- the Modal launcher's `--cells` tokens for a view's cells (07_bcb_bfcl.sh) ------------------------------------------------
# The wave roster and the Modal launcher share ONE cell grammar — `none | prompted__<si> | <si>__<arm> | a bare Class B id`
# (contracts.parse_launcher_cell hands the translated harness id to defense_cells.parse_cell_id, the validator the standup runs) — and
# the table's `objects.<cell>.cell_id` is spelled IN that grammar (`<corpus pseudo-SI>__<arm>`: constv2ctl2-full__lora_r4_ce, …; the
# arditi lane's `<rung>__<harness id>`). So the mapping from a view cell (`control_lora`) to the launcher's token is `roster_cell_id`,
# the same rule 05 writes the wave roster's cells with: `none` and `prompted__*` unchanged, an object cell its object's own cell_id, the
# arditi rung stripped. ONE rule, derived from the table's objects block, never a hand-kept name table (paper-companion-repo B14: stage 07
# used to pass the view cells themselves, which the launcher refused at plan time — reproduce/README.md "Known gaps", now closed).
# What the launcher needs BEYOND the token: ONE bank and ONE explicit-cells config per registry ROW (contracts.MODELS[<slug>].objects_dir /
# .cfg — `<objects_dir>/<token>/object.*`; defense_cell.object_path_for keys the directory on the cell id, so the bank directory IS the
# token; no flag overrides either). The launcher's grammar carries no dose, and a dose view's object cell has the SAME cell id as the
# wave's parent, so a dose view is served the way the paper served its dose-scaled rows on Modal (row dirs `qwen35_35b_a3b-f22d0p75_relul`
# / `-f22d1p25`, each a registry row of its own over a bank of the view's copies keyed by cell id): a REGISTRY ROW PER DOSE VIEW,
# `launcher_slug` = `<key>-<view>`, whose bank is `<the key's bank>/<view>/` and whose config is frontdoor_config_path(key, view) (the
# token stays the cell id; the SCALED COPY `d<tok>__<cell_id>` is the cluster bank's spelling of the same bytes, 09_dose_ladder.sh).
# reproduce/tools/stage_demo_bank_modal.py stages the view's committed copies there; reproduce/07_bcb_bfcl.sh --view <dose view> launches
# `--model <slug>` (demo-import: the Known gap "the dose views stay on the cluster" closed, 2026-10-12).


def launcher_slug(key: str, view: str, path: Optional[str] = None) -> str:
    """The Modal launcher's registry slug (contracts.MODELS) that serves a phase view of a table key: the key itself for the baseline
    wave (its row serves the wave's bank, `<objects_dir>`), `<key>-<view>` for every other phase view — a dose view's row, over the
    view's own bank `<objects_dir>/<view>/` of dose-scaled copies keyed by cell id and its own explicit-cells config
    (contracts.frontdoor_config_path(key, view)). ONE rule for the stage (07 `--model`), the bank stager (the bank root) and the config
    generator (scripts/bcbw_configs.py --front-door <key> --view <view>); refuses a view the key does not declare."""
    m = model(key, path)
    if view not in (m.get("views") or {}):
        raise KeyError(f"{key}: no view {view!r} (views: {sorted(m.get('views') or {})})")
    return key if view == baseline_view(key, path) else f"{key}-{view}"


def _launcher_grammar():
    """The launcher's cell grammar, imported lazily from THIS checkout's src (the launching process's own module; torch-free). A
    checkout whose interpreter cannot import it refuses loudly — a token printed unchecked could plan a wave the launcher refuses."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from sisteer.capeval.tinycap import contracts
    except ImportError as exc:                      # pragma: no cover - the venv's job
        raise ValueError(f"the launcher's cell grammar (src/sisteer/capeval/tinycap/contracts.py) does not import with {sys.executable}: {exc}")
    return contracts


def launcher_cells(key: str, view: str, cells: Optional[List[str]] = None, path: Optional[str] = None) -> List[Tuple[str, str]]:
    """``[(view cell, launcher --cells token), …]`` for the cells of a phase view (every cell of the view when `cells` is None), each
    token checked against the launcher's grammar (contracts.parse_launcher_cell). A DOSE view's object cell maps to the same token as the
    wave's parent — its cell id — and is served from the dose view's OWN registry row and bank (`launcher_slug`; module comment above).
    Refuses BY NAME: a cell the view does not list (the launcher runs the roster's cells only; a raw launcher token is not a roster cell),
    a view cell with no `objects` entry (roster_cell_id), a token the launcher's grammar refuses (the table spelled a cell_id outside
    `<si>__<arm>` / the Class B ids), and two cells on one token."""
    m = model(key, path)
    vd = (m.get("views") or {}).get(view)
    if vd is None:
        raise KeyError(f"{key}: no view {view!r} (views: {sorted(m.get('views') or {})})")
    listed = [str(c) for c in (vd.get("cells") or [])]
    wanted = listed if cells is None else [str(c) for c in cells]
    contracts = _launcher_grammar()
    out: List[Tuple[str, str]] = []
    for c in wanted:
        if c not in listed:
            raise ValueError(f"{key}/{view}: {c!r} is not a cell of the view (cells: {listed}) — 07 runs the roster's cells, by their "
                             "models.yaml names")
        tok = roster_cell_id(m, c)
        try:
            contracts.parse_launcher_cell(tok)
        except contracts.CellNameError as exc:
            raise ValueError(f"{key}/{view}: {c!r} -> {tok!r} is outside the launcher's cell grammar (contracts.parse_launcher_cell): {exc}")
        if tok in (t for _c, t in out):
            raise ValueError(f"{key}/{view}: {c!r} and an earlier cell both map to the launcher token {tok!r}")
        out.append((c, tok))
    return out


def roster_block(key: str, views_wanted: List[str], path: Optional[str] = None) -> str:
    """The target's COMPLETE block for configs/sdfbase_full22_roster.yaml (paste under `targets:`): the served checkpoint from
    serve.checkpoint, the battery body (FRONT_DOOR_BODY_CONFIG), the bank, and per phase view its dose keys, its config / manifest
    paths (make_full22_configs.py's spellings) and its cells through roster_cell_id / staged_id. Loads back with a duplicate-key
    refusing loader (test) and renders through make_full22_configs.py without a pins_config (the generator substitutes the model
    pins from the block — the fix-1 owner follow-ups 1 and 3)."""
    m = model(key, path)
    ck = m["serve"]["checkpoint"]
    lines = [f"  {key}:",
             f"    # FRONT DOOR (reproduce/models.yaml `{key}`; 05_stage_roster.sh --step roster): the served checkpoint is the table's; the battery",
             f"    # body is the one every full22 config runs — make_full22_configs.py substitutes model / model_revision into it from this block",
             f"    # and takes the Petri manifest's serving pins from the table (no pins_config: no sdfbase transfer lineage of its own)",
             f"    model: {ck['hf_repo']}",
             f"    model_revision: {ck['revision']}",
             f"    body_config: {FRONT_DOOR_BODY_CONFIG}",
             "    bank: full22",
             "    views:"]
    for v in views_wanted:
        vd = (m.get("views") or {}).get(v)
        if vd is None:
            raise KeyError(f"{key}: no view {v!r} (views: {sorted(m.get('views') or {})})")
        lines.append(f"      {v}:")
        for k in ("dose", "dose_unit", "object_family", "object_lane"):
            if k in vd:
                lines.append(f"        {k}: {vd[k]}")
        lines.append(f"        config: configs/sdfbase_full22_{key}_{v}.yaml")
        lines.append(f"        manifest: configs/capeval/manifests/petri_full22_{key}_{v}.yaml")
        cells = roster_view_cells(m, v)
        lines.append("        cells:" + ("" if cells else " {}"))
        names = {roster_cell_id(m, c): c for c in vd.get("cells", []) if c != "none"}
        for cid, sid in cells.items():
            lines.append(f"          {cid}: {'null' if sid is None else sid}" + (f"   # {names[cid]}" if names[cid] != cid else ""))
    return "\n".join(lines) + "\n"


def check(path: Optional[str] = None) -> List[str]:
    """Every model keeps both checkpoint ids and 40-hex revisions after loading; the pins are well-formed."""
    problems: List[str] = []
    models = load_models(path)
    for key, m in models.items():
        for side in ("fit", "serve"):
            ck = (m.get(side) or {}).get("checkpoint") or {}
            for f in ("hf_repo", "revision", "model_id"):
                if not ck.get(f):
                    problems.append(f"{key}.{side}.checkpoint.{f} missing")
            rev = str(ck.get("revision", ""))
            if rev and not HEX40.match(rev):
                problems.append(f"{key}.{side}.checkpoint.revision {rev!r} is not a 40-hex sha")
        arch = m.get("architecture") or {}
        for f in ("family", "n_layers", "hidden", "attention_layers"):
            if f not in arch:
                problems.append(f"{key}.architecture.{f} missing")
        la = m.get("lora_architecture") or {}
        for f in ("rank", "modules", "per_expert_rank", "selected_by", "selection_ruling_id"):
            if f not in la:
                problems.append(f"{key}.lora_architecture.{f} missing")
        if la.get("selected_by") not in ("paper", "pending_user_pick") and not str(la.get("selected_by", "")).startswith("user_pick"):
            problems.append(f"{key}.lora_architecture.selected_by {la.get('selected_by')!r} unknown")
        for s in ("o_proj", "qkvo", "mlp", "down"):
            if s not in (m.get("layer_sets") or {}):
                problems.append(f"{key}.layer_sets.{s} missing")
        if "run_in_this_work" not in m:
            problems.append(f"{key}.run_in_this_work missing")
        if not (m.get("thinking") or {}):
            problems.append(f"{key}.thinking missing")
        cth = (m.get("serve") or {}).get("chat_template_hash")
        if cth is not None and not re.fullmatch(r"[0-9a-f]{16}", str(cth)):
            problems.append(f"{key}.serve.chat_template_hash {cth!r} is not sha256_hex16 (16 hex) of the pinned chat_template bytes")
        for v in (m.get("views") or {}):
            try:
                roster_view_cells(m, v)
            except (KeyError, ValueError) as exc:
                problems.append(f"{key}.views.{v}: {exc}")
        problems.extend(f"{key}.{p}" for p in declared_wave_problems(m))
        # demo-import: every `committed_objects_of` (the row's, a cell's) names a paper row's checkpoint id (committed_model_id); a cell's own
        # entry without the row's is fine (one imported cell beside fitted ones); a row that imports must be a front-door row with objects
        spellings = [(f"{key}.{COMMITTED_OBJECTS_KEY}", m.get(COMMITTED_OBJECTS_KEY))] + \
            [(f"{key}.objects.{c}.{COMMITTED_OBJECTS_KEY}", o.get(COMMITTED_OBJECTS_KEY)) for c, o in (m.get("objects") or {}).items() if isinstance(o, dict)]
        for where, sp in spellings:
            if sp is None:
                continue
            try:
                committed_model_id(str(sp), path)
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
        if m.get(COMMITTED_OBJECTS_KEY) and not (m.get("objects") or {}):
            problems.append(f"{key}.{COMMITTED_OBJECTS_KEY}: a row with no `objects` block has nothing to import")
    return problems


def declared_wave_problems(m: dict) -> List[str]:
    """The declared-wave refusals of ONE row (demo-import M3): `wave:` must be a scalar naming a view of the row that is not a dose
    view; its name is outside the phase-view grammar and is none of u0 / main / it, `obj` / `obj_s<N>` (make_padding_configs' hand-listed
    families) or a `prompted__` name; a row that declares u0 (the phased recipe) declares no wave; and a no-u0 front-door row's every view
    is either a grammar view or the declared wave (never silently dropped by a generator that filters by grammar)."""
    out: List[str] = []
    vs = m.get("views") or {}
    raw = m.get(WAVE_KEY)
    if raw is not None:
        if not isinstance(raw, str) or not raw:
            out.append(f"{WAVE_KEY}: {raw!r} is not a non-empty scalar name (a scalar names exactly one wave)")
            return out
        w = raw
        if w not in vs:
            out.append(f"{WAVE_KEY}: {w!r} names no view of the row (views: {sorted(vs)})")
        elif "dose" in (vs[w] or {}):
            out.append(f"{WAVE_KEY}: {w!r} is a dose view (`dose:` declared) — the declared wave is the one steered wave, never a ladder rung")
        if PHASE_VIEW_RE.match(w):
            out.append(f"{WAVE_KEY}: {w!r} is a name of the phase-view grammar (PHASE_VIEW_RE) — a declared wave is a name outside it (the grammar's "
                       "`s<N>` spells the paper's fit-seed replicates)")
        if w in RESERVED_VIEW_NAMES:
            out.append(f"{WAVE_KEY}: {w!r} is reserved ({', '.join(RESERVED_VIEW_NAMES)}: the phased recipe's baseline and the paper's roster views)")
        if _PADDING_HAND_LISTED_FAMILY.match(w):
            out.append(f"{WAVE_KEY}: {w!r} collides with make_padding_configs' hand-listed families (obj / obj_s<N>)")
        if w.startswith(NO_OBJECT_PREFIX):
            out.append(f"{WAVE_KEY}: {w!r} carries the prompted-cell prefix {NO_OBJECT_PREFIX!r}")
        if "u0" in vs:
            out.append(f"{WAVE_KEY}: a row that declares u0 runs in PHASES (u0 first, the steered waves gated behind it) — a declared wave is the "
                       "single-phase recipe's; declare one or the other")
    phase = phase_views_of(m)
    if phase and "u0" not in vs:
        for v in vs:
            if str(v) not in phase:
                out.append(f"views.{v}: neither a view of the phase-view grammar nor the declared wave ({declared_wave_of(m)!r}) — a generator "
                           "filtering by the grammar would render nothing for it while the planner plans it; declare it as the row's `wave:` or "
                           "spell it in the grammar")
    return out


def _render(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, (str, int, float)):
        return str(v)
    if isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
        return " ".join(str(x) for x in v)
    return json.dumps(v, sort_keys=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keys")
    p = sub.add_parser("exists"); p.add_argument("key")
    p = sub.add_parser("get"); p.add_argument("key"); p.add_argument("path")
    p = sub.add_parser("views"); p.add_argument("key")
    p = sub.add_parser("phase-views", help="the PHASE views of <key>: the grammar's (u0 / s<N> / d<dose>…) plus the row's declared `wave`"); p.add_argument("key")
    p = sub.add_parser("declared-wave", help="the row's declared wave (`wave:`), exit 1 when it declares none"); p.add_argument("key")
    p = sub.add_parser("excluded-families", help="<family>\t<reason> per family the table excludes from <key>'s <view> by ruling (D889)"); p.add_argument("key"); p.add_argument("view")
    p = sub.add_parser("front-door", help="exit 0 iff <key> is a front-door model (a table model with phase views)"); p.add_argument("key")
    p = sub.add_parser("baseline-view", help="the wave the gate's inventory is frozen from: u0 when declared, else the single-phase recipe's one wave"); p.add_argument("key")
    p = sub.add_parser("single-phase", help="exit 0 iff <key> runs the single-phase recipe (a front-door model with no u0 view)"); p.add_argument("key")
    p = sub.add_parser("roster-block"); p.add_argument("key"); p.add_argument("views", help="comma list of phase views")
    p = sub.add_parser("launcher-cells", help="`<view cell>\t<Modal launcher --cells token>` per cell of <key>'s <view> (07_bcb_bfcl.sh); refuses by name a cell the launcher cannot express")
    p.add_argument("key"); p.add_argument("view"); p.add_argument("cells", help="comma list of the view's cells (models.yaml names), or all")
    p = sub.add_parser("launcher-slug", help="the Modal launcher's registry slug (contracts.MODELS) serving <key>'s <view>: the key for the baseline wave, <key>-<view> for a dose view (07_bcb_bfcl.sh --model)")
    p.add_argument("key"); p.add_argument("view")
    sub.add_parser("check")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "keys":
            print("\n".join(load_models(args.table)))
        elif args.cmd == "exists":
            return 0 if args.key in load_models(args.table) else 1
        elif args.cmd == "front-door":
            return 0 if args.key in front_door_keys(args.table) else 1
        elif args.cmd == "baseline-view":
            print(baseline_view(args.key, args.table))
        elif args.cmd == "single-phase":
            return 0 if single_phase(args.key, args.table) else 1
        elif args.cmd == "get":
            print(_render(_get(model(args.key, args.table), args.path)))
        elif args.cmd == "views":
            print("\n".join(views(args.key, args.table)))
        elif args.cmd == "phase-views":
            print("\n".join(phase_views(args.key, args.table)))
        elif args.cmd == "declared-wave":
            w = declared_wave(args.key, args.table)
            if w is None:
                return 1
            print(w)
        elif args.cmd == "excluded-families":
            for f, why in excluded_families(args.key, args.view, args.table).items():
                print(f"{f}\t{why}")
        elif args.cmd == "roster-block":
            sys.stdout.write(roster_block(args.key, [v for v in args.views.split(",") if v], args.table))
        elif args.cmd == "launcher-cells":
            wanted = None if args.cells == "all" else [c for c in args.cells.split(",") if c]
            for cell, tok in launcher_cells(args.key, args.view, wanted, args.table):
                print(f"{cell}\t{tok}")
        elif args.cmd == "launcher-slug":
            print(launcher_slug(args.key, args.view, args.table))
        elif args.cmd == "check":
            probs = check(args.table)
            for pr in probs:
                print(f"REFUSED: models.yaml: {pr}", file=sys.stderr)
            return 2 if probs else 0
    except (KeyError, DuplicateKeyError, ValueError) as exc:
        print(f"REFUSED: models.yaml: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
