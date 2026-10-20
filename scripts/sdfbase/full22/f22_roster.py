#!/usr/bin/env python
"""full22: the ONE reader of what a (target, view) of the roster SERVES.

A (target, view)'s served cells are its view's ``cells`` PLUS the cells of every lane whose flag
(``lane_flags.<lane>.enabled``) is on and whose ``view`` is this view. Every script that needs the
served set — the config/manifest generator, the bank-view builder, the job-side count guard
(roster_cell_count.py), the validation slice, both submit printers — resolves it HERE, so a lane
flag flip moves all of them together. A reader that inlined ``view["cells"]`` would keep building
a view (or counting a guard reference) WITHOUT the lane while the generated config served it:
exactly the config-vs-view disagreement the short-view guard exists to catch, manufactured by the
roster itself.

PROMPTED BASELINES (2026-09-08, the nemotron3nano4b completion lane). A view may serve a prompted
baseline — ``prompted__<si>``: the SI text as the system prompt, NO object, NO staged dir — by mapping
the cell id to ``null`` in ``views.<view>.cells``. ``served_cells`` returns it with ``None`` as its
staged id (so the CONFIG's explicit_cells and the job-side config count include it), and
``staged_cells`` is the subset every OBJECT consumer reads (the bank view, the pins, the presence
checks): a prompted cell is never looked up in a bank. The grammar is enforced here, once: a
``null`` staged id is legal ONLY on a ``prompted__`` id, and a ``prompted__`` id may carry nothing else.

Import-light on purpose (yaml only): make_bank_view.py's light path must stay torch-free.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import yaml

ROSTER = Path(__file__).resolve().parents[3] / "configs" / "sdfbase_full22_roster.yaml"


class RosterDuplicateKey(ValueError):
    """A mapping key spelled twice in a roster file (PyYAML's safe_load keeps the LAST silently)."""


class _StrictRosterLoader(yaml.SafeLoader):
    pass


def _construct_mapping_strict(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RosterDuplicateKey(f"duplicate mapping key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictRosterLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_strict)


def load_roster(path: Path = ROSTER) -> dict:
    """The roster as data, REFUSING a duplicate mapping key by name and line (RosterDuplicateKey) — PyYAML's safe_load keeps the
    last of two `constv2ctl2-full__psr_ce_all:` entries silently, which is how a printer defect (the fix-1 owner follow-up 3: two
    view cells rendered under one roster cell id) would have become a served view missing an object. Every roster reader loads
    through here."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return yaml.load(text, Loader=_StrictRosterLoader)
    except RosterDuplicateKey as exc:
        raise RosterDuplicateKey(f"{path}: {exc}") from None


#: the CLI spelling of a roster override, shared by every reader the jobs call (roster_view.py,
#: roster_cell_count.py; make_bank_view.py and the generator take the same flag through argparse)
ROSTER_FLAG = "--roster"


def pop_roster_arg(argv: List[str], default: Path = ROSTER) -> Tuple[List[str], Path]:
    """Split ``--roster PATH`` (or ``--roster=PATH``) out of a positional argv: ``(the other
    arguments in order, the roster path)`` — ``default`` (the production roster) when absent.
    The full22 jobs read the roster through ``F22_ROSTER`` (common.sh: ``--roster "$F22_ROSTER"``
    on every reader), so a certification-only roster (llama8b's ``psrcert`` view) is a per-job
    selection and the production roster is never edited for it. Refuses a repeated flag or a
    flag with no path."""
    rest: List[str] = []
    path: Optional[Path] = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == ROSTER_FLAG or a.startswith(ROSTER_FLAG + "="):
            if path is not None:
                raise ValueError(f"{ROSTER_FLAG} given twice")
            if a == ROSTER_FLAG:
                if i + 1 >= len(argv):
                    raise ValueError(f"{ROSTER_FLAG} needs a path")
                path = Path(argv[i + 1]); i += 2
            else:
                val = a.split("=", 1)[1]
                if not val:
                    raise ValueError(f"{ROSTER_FLAG} needs a path")
                path = Path(val); i += 1
            continue
        rest.append(a); i += 1
    return rest, (path if path is not None else default)


def lane_flags(roster: dict) -> Dict[str, dict]:
    return dict(roster.get("lane_flags") or {})


def lane_enabled(roster: dict, lane: str) -> bool:
    flag = lane_flags(roster).get(lane)
    if flag is None:
        raise KeyError(f"roster has no lane_flags.{lane}")
    return bool(flag.get("enabled", False))


def lanes_of(roster: dict, target: str, view: str) -> List[Tuple[str, bool, Dict[str, str]]]:
    """Every lane the target maps for this view: ``(lane, enabled, {config cell id: staged id})``,
    in roster order. A lane the target maps but ``lane_flags`` does not name is an error (a typo
    would otherwise silently never serve)."""
    t = roster["targets"][target]
    out = []
    for lane, cells in (t.get("lanes") or {}).items():
        flag = lane_flags(roster).get(lane)
        if flag is None:
            raise KeyError(f"{target}: lanes.{lane} has no lane_flags.{lane} entry")
        if flag.get("view", "main") != view:
            continue
        out.append((lane, bool(flag.get("enabled", False)), dict(cells or {})))
    return out


PROMPTED_PREFIX = "prompted__"


def is_prompted(cell_id: str) -> bool:
    """A prompted baseline (``prompted__<si>``): the SI as the system prompt, no object."""
    return str(cell_id).startswith(PROMPTED_PREFIX)


#: document-corpus pseudo-instruction prefixes: their text is training data, never a system prompt
#: (defense_cell.registry_from_explicit_cells refuses `prompted__constv2-*` as a fabricated baseline;
#: refused here too, at the grammar, so a roster typo cannot reach a job)
_CORPUS_SI_PREFIXES = ("constv2",)


#: the pin-referencing MAPPING form of a view cell (the builder's ``make_bank_view`` imports
#: it): ``{pins: <pins file>, pin: <target>.<cell>}`` — the cell's object is
#: selected THROUGH a pin in another bank, never by naming a rung here
PIN_FORM_KEYS = ("pins", "pin")


def is_pin_form(sid) -> bool:
    return isinstance(sid, dict) and sorted(sid) == sorted(PIN_FORM_KEYS)


def _check_cell(where: str, cid: str, sid) -> None:
    if isinstance(sid, dict):
        if is_prompted(cid):
            raise ValueError(f"{where}: prompted cell {cid} maps to a pin mapping {sid!r} — a prompted baseline "
                             "serves the SI text, never an object; map it to null")
        if not is_pin_form(sid) or not all(isinstance(sid[k], str) and sid[k] for k in PIN_FORM_KEYS):
            raise ValueError(f"{where}: cell {cid} maps to {sid!r} — the mapping form takes exactly "
                             f"{list(PIN_FORM_KEYS)} (non-empty strings); a rung is never named here")
        return
    if is_prompted(cid) and cid[len(PROMPTED_PREFIX):].startswith(_CORPUS_SI_PREFIXES):
        raise ValueError(f"{where}: {cid} names a document-corpus pseudo-instruction — its text is "
                         "never a system prompt (registry_from_explicit_cells refuses it)")
    if is_prompted(cid) and sid is not None:
        raise ValueError(f"{where}: prompted cell {cid} maps to a staged id {sid!r} — a prompted "
                         "baseline serves the SI text, never an object; map it to null")
    if not is_prompted(cid) and sid is None:
        raise ValueError(f"{where}: cell {cid} maps to null but is not a prompted__<si> baseline — "
                         "an object cell needs its staged id")
    if is_prompted(cid) and not cid[len(PROMPTED_PREFIX):]:
        raise ValueError(f"{where}: malformed prompted cell id {cid!r} (prompted__<si>)")


def served_cells(roster: dict, target: str, view: str) -> Dict[str, Optional[str]]:
    """``{config cell id: staged id}`` the (target, view) SERVES: the view's own cells first, then
    each enabled lane's cells in roster order. A prompted baseline (``prompted__<si>: null``) is
    served with ``None`` as its staged id — see the module docstring; object consumers read
    :func:`staged_cells` instead. Refuses a cell id or staged id that two sources both claim (a
    lane must never shadow a view cell)."""
    v = roster["targets"][target]["views"][view]
    out: Dict[str, Optional[str]] = dict(v.get("cells") or {})
    for cid, sid in out.items():
        _check_cell(f"{target}/{view}", cid, sid)
    for lane, enabled, cells in lanes_of(roster, target, view):
        if not enabled:
            continue
        for cid, sid in cells.items():
            _check_cell(f"{target}/{view} lane {lane}", cid, sid)
            if cid in out or (sid is not None and sid in out.values()):
                raise ValueError(f"{target}/{view}: lane {lane} cell {cid} <- {sid} collides with "
                                 "a cell already served")
            out[cid] = sid
    return out


#: a STEERED-WAVE view — `s<N>r`, the ReLU-parent subset of phase N and its composed difference: steered-only — no unsteered `none`
#: leg is generated, planned, registered or scored for it (the u0 wave is
#: the baseline). Mirrored in reproduce/lib/models.py and make_misalign_manifest.py (tests hold the copies equal). The standup still
#: SERVES the base route implicitly (mt_serving_up.select_cells: load_check.py probes it, the native attestation reads it) — served, never
#: listed, never measured.
STEERED_WAVE_VIEW_RE = re.compile(r"^s[0-9]+rl?$")
#: the models table (reproduce/models.yaml) and its ONE reader — the FRONT-DOOR targets are the table's models that name phase views
MODELS_TABLE = Path(__file__).resolve().parents[3] / "reproduce" / "models.yaml"
MODELS_TABLE_READER = Path(__file__).resolve().parents[3] / "reproduce" / "lib" / "models.py"
_FRONT_DOOR_CACHE: Dict[str, FrozenSet[str]] = {}


def front_door_targets(table_path: "Path | str | None" = None) -> FrozenSet[str]:
    """The FRONT-DOOR targets: the models table's models that name at least one phase view, read through the table's ONE reader
    (reproduce/lib/models.py front_door_keys; SISTEER_MODELS_TABLE overrides the path, as the generator honours it); the empty set when
    the table is absent (the paper's tree before the front door). A table without its reader beside it refuses by name."""
    p = Path(table_path) if table_path is not None else Path(os.environ.get("SISTEER_MODELS_TABLE") or MODELS_TABLE)
    key = str(p.resolve()) if p.exists() else str(p)
    if key not in _FRONT_DOOR_CACHE:
        if not p.is_file():
            _FRONT_DOOR_CACHE[key] = frozenset()
        else:
            if not MODELS_TABLE_READER.is_file():
                raise SystemExit(f"models table {p}: its reader {MODELS_TABLE_READER} is not on this tree — refusing to read it another way")
            import importlib.util
            spec = importlib.util.spec_from_file_location("_f22_roster_models_table", MODELS_TABLE_READER)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            _FRONT_DOOR_CACHE[key] = frozenset(str(k) for k in mod.front_door_keys(str(p)))
            _SINGLE_PHASE_CACHE[key] = frozenset(k for k in _FRONT_DOOR_CACHE[key] if mod.single_phase(k, str(p)))
    return _FRONT_DOOR_CACHE[key]


_SINGLE_PHASE_CACHE: Dict[str, FrozenSet[str]] = {}


def single_phase_targets(table_path: "Path | str | None" = None) -> FrozenSet[str]:
    """The front-door targets on the SINGLE-PHASE recipe (reproduce/lib/models.single_phase): a table row that
    declares NO `u0` view — its one steered wave carries its own unsteered legs, so none of its views is a steered-wave view."""
    p = Path(table_path) if table_path is not None else Path(os.environ.get("SISTEER_MODELS_TABLE") or MODELS_TABLE)
    key = str(p.resolve()) if p.exists() else str(p)
    front_door_targets(table_path)
    return _SINGLE_PHASE_CACHE.get(key, frozenset())


def steered_wave_view(view: str, target: Optional[str] = None, *, table_path: "Path | str | None" = None) -> bool:
    """A view that measures NO unsteered leg: `s<N>r` by name (the ReLU-parent subsets and their composed
    difference), and — for a FRONT-DOOR target that RUNS IN PHASES (its table declares a u0 view) — EVERY view except u0:
    the LoRA wave s1, the residual wave s2, the dose views d<tok> / d<tok>_relu all measure their steered cells against the u0 wave's
    baseline; none, none__lora0 and none__lens0 appear
    in no artefact, plan row, registration or score, while the standup still serves the base route implicitly. A paper target's roster
    views (main, it, s1, d0p1, …) keep their same-wave `none` anchor: the predicate is TARGET-aware because those names collide with
    the phase grammar, so a name-only rule would be wrong on them."""
    if STEERED_WAVE_VIEW_RE.match(str(view)):
        return True
    if target is not None and str(target) in front_door_targets(table_path):
        # the SINGLE-PHASE recipe (models.single_phase): a front-door target that declares no u0 view evaluates
        # every arm in one wave beside its own unsteered legs — none of its views is a steered wave
        return str(view) != "u0" and str(target) not in single_phase_targets(table_path)
    return False


def view_cell_universe(roster: dict, target: str, view: str) -> List[str]:
    """The ORDERED cell list every generated artefact of (target, view) enumerates — the wave config's ``explicit_cells``, the Petri
    manifest's ``cells`` (as harness ids), the sampled rung configs, the misalign pair, the GSM8K matrix rows: ``none`` first + the
    served cells for an ordinary view (the same-wave bare anchor); EXACTLY the served cells, no ``none``, for a steered-wave view
    (a generator's unconditional `none` would re-introduce an unsteered leg there; for a front-door target every view except u0 is one).
    A steered-wave view whose roster lists ``none`` itself refuses by name — the universe is the roster's, never a hand list."""
    served = [c for c in served_cells(roster, target, view)]
    if steered_wave_view(view, target):
        if "none" in served:
            raise ValueError(f"{target}/{view}: a steered-wave view lists `none` in the roster — it measures no unsteered leg (D1152 / D1154); "
                             "drop it from views.{view}.cells")
        return served
    return ["none", *[c for c in served if c != "none"]]


def staged_cells(roster: dict, target: str, view: str) -> Dict[str, str]:
    """The served cells that have a STAGED OBJECT — ``served_cells`` minus the prompted
    baselines. What the bank view builds, the pins bind, and every object presence check
    iterates; a prompted cell has no entry in any of them by construction."""
    return {cid: sid for cid, sid in served_cells(roster, target, view).items()
            if sid is not None}


#: The wave's LEGS, in launch order: `static` = the four
#: single-shot families; `bon` = best-of-N; `capability` = gsm8k + mmlu (the served controls driver);
#: `controls` = the over-refusal controls xstest + or_bench_hard_1k and their cap ladder (the same
#: driver, the other half of its --controls list); `attacks` = the multi-turn trio; `petri` = the audit.
#: A view without a `legs:` key runs ALL of them (every view before this lane); a view that names
#: `legs:` runs exactly those, and `legs_omitted:` must name every other leg WITH its reason, so an
#: omission is recorded in the manifest, never inferred from an absent job.
LEGS: Tuple[str, ...] = ("static", "bon", "capability", "controls", "attacks", "petri")
#: the --controls names each controls-driver leg runs
LEG_CONTROLS: Dict[str, Tuple[str, ...]] = {"capability": ("gsm8k", "mmlu"),
                                            "controls": ("xstest", "or_bench_hard_1k")}


def view_legs(roster: dict, target: str, view: str) -> Tuple[str, ...]:
    """The legs a (target, view) runs — every leg when the view names none; refuses an unknown
    leg, a duplicate, an omission without a reason, or an omission that the view also runs."""
    v = roster["targets"][target]["views"][view]
    legs = v.get("legs")
    omitted = dict(v.get("legs_omitted") or {})
    if legs is None:
        if omitted:
            raise ValueError(f"{target}/{view}: legs_omitted without legs — name the legs that run")
        return LEGS
    legs = tuple(legs)
    bad = [l for l in legs if l not in LEGS]
    if bad or len(set(legs)) != len(legs):
        raise ValueError(f"{target}/{view}: legs {list(legs)} must be distinct names from {list(LEGS)}")
    missing = [l for l in LEGS if l not in legs and l not in omitted]
    if missing:
        raise ValueError(f"{target}/{view}: legs omitted without a reason: {missing} (legs_omitted)")
    both = [l for l in omitted if l in legs]
    if both:
        raise ValueError(f"{target}/{view}: legs_omitted names a leg the view runs: {both}")
    unknown = [l for l in omitted if l not in LEGS]
    if unknown:
        raise ValueError(f"{target}/{view}: legs_omitted names unknown legs {unknown}")
    for l, why in omitted.items():
        if not isinstance(why, str) or not why.strip():
            raise ValueError(f"{target}/{view}: legs_omitted.{l} needs a reason")
    return tuple(l for l in LEGS if l in legs)


def view_legs_omitted(roster: dict, target: str, view: str) -> Dict[str, str]:
    """``{leg: reason}`` for every leg the view does not run (empty for a full-battery view)."""
    view_legs(roster, target, view)   # the grammar check
    return dict(roster["targets"][target]["views"][view].get("legs_omitted") or {})


#: the eval-time TOKEN CAPS a target may carry (``targets.<key>.caps``): one positive int per leg +
#: ``source``. The generator applies ``multiturn_target`` as
#: ``attacks.multiturn_target_max_new_tokens`` (a key read ONLY by the multi-turn path) and asserts every other cap
#: EQUALS the value the body / the roster already carries for that leg; the printers and the attacks sbatch derive the
#: multi-turn serving window from it (geometry.multiturn_window). A target without a caps block is every target before.
CAP_LEGS: Tuple[str, ...] = ("multiturn_target", "attacker", "static", "bon", "controls_ladder", "petri")
#: the multi-turn attack FAMILIES: the multi-turn target cap and its serving windows are PER FAMILY (Crescendo
#: and PAIR at 4,096, ActorAttack at its filed 512, cut from the 4,096 re-run on cost).
#: ``caps.multiturn_target`` is a mapping family -> cap (+ ``source``), ``caps.windows`` a
#: mapping family -> {multiturn, multiturn_rerun, source}; every consumer resolves the family it runs.
MULTITURN_FAMILIES: Tuple[str, ...] = ("crescendo", "pair", "actorattack")


def multiturn_family_caps(v, *, target: str) -> Tuple[Dict[str, int], Optional[str]]:
    """``caps.multiturn_target`` -> (``{family: cap}`` over MULTITURN_FAMILIES, its ``source`` or None): a mapping must name
    EVERY family with a positive int and carry a ``source`` (a family the table does not name cannot run — refused, never
    defaulted); a bare positive int is every family at that cap (the caps block's own source covers it)."""
    if isinstance(v, bool) or (not isinstance(v, int) and not isinstance(v, dict)):
        raise ValueError(f"{target}: caps.multiturn_target={v!r} must be a positive int or a mapping {list(MULTITURN_FAMILIES)} -> int"
                         " + source (a pending entry is not a cap)")
    if isinstance(v, int):
        if v < 1:
            raise ValueError(f"{target}: caps.multiturn_target={v!r} must be a positive int (a pending entry is not a cap)")
        return {f: int(v) for f in MULTITURN_FAMILIES}, None
    unknown = sorted(set(v) - set(MULTITURN_FAMILIES) - {"source"})
    if unknown:
        raise ValueError(f"{target}: caps.multiturn_target carries unknown families {unknown} (the multi-turn families are {list(MULTITURN_FAMILIES)})")
    missing = [f for f in MULTITURN_FAMILIES if f not in v]
    if missing:
        raise ValueError(f"{target}: caps.multiturn_target is missing the families {missing} — every multi-turn family needs its cap")
    for f in MULTITURN_FAMILIES:
        c = v[f]
        if isinstance(c, bool) or not isinstance(c, int) or c < 1:
            raise ValueError(f"{target}: caps.multiturn_target.{f}={c!r} must be a positive int (a pending entry is not a cap)")
    src = v.get("source")
    if not isinstance(src, str) or not src.strip():
        raise ValueError(f"{target}: caps.multiturn_target.source must name where the per-family caps came from")
    return {f: int(v[f]) for f in MULTITURN_FAMILIES}, src


def target_caps(roster: dict, target: str) -> Optional[Dict[str, object]]:
    """``{leg: int, ..., "multiturn_target": {family: int}, "multiturn_target_source": str, "source": str}`` for a target
    with a caps block, None otherwise. Refuses a missing leg, an unknown key, a missing source, a value that is not a
    positive int (a ``pending`` marker is not a cap), or a per-family multi-turn map missing a family."""
    caps = roster["targets"][target].get("caps")
    if caps is None:
        return None
    if not isinstance(caps, dict):
        raise ValueError(f"{target}: caps must be a mapping of {list(CAP_LEGS)} + source")
    unknown = sorted(set(caps) - set(CAP_LEGS) - {"source", "windows", "multiturn_families_omitted"})
    if unknown:
        raise ValueError(f"{target}: caps carries unknown keys {unknown}")
    missing = [l for l in CAP_LEGS if l not in caps]
    if missing:
        raise ValueError(f"{target}: caps is missing {missing}")
    src = caps.get("source")
    if not isinstance(src, str) or not src.strip():
        raise ValueError(f"{target}: caps.source must name where the caps came from")
    for leg in CAP_LEGS:
        v = caps[leg]
        if leg == "multiturn_target":
            continue          # its own grammar, per family (multiturn_family_caps below)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{target}: caps.{leg}={v!r} must be a positive int (a pending entry is not a cap)")
    fam, fam_src = multiturn_family_caps(caps["multiturn_target"], target=target)
    target_windows(roster, target)     # the grammar check of the pins that ride beside the caps (read through target_windows)
    target_multiturn_families(roster, target)   # ... and of the keep-or-drop switch
    out = {k: v for k, v in caps.items() if k not in ("windows", "multiturn_families_omitted")}
    out["multiturn_target"] = fam
    out["multiturn_target_source"] = fam_src or src
    return out


#: the multi-turn SERVING WINDOWS a caps block may PIN (``caps.windows``): the first
#: pass's window and the over-window re-run pass's, each an operator's WIDER value over the derived one
#: (geometry.pin_window refuses a pin below the derived minimum — p99 prompt + cap / measured max call + cap — or off
#: the table's grid, and a re-run window not wider than the first pass); S = the model's budget // L as ever. The
#: generator writes them into the view configs (attacks.multiturn_max_model_len / multiturn_rerun_max_model_len),
#: which target_attacks.sbatch and the re-run pass read; a target without them derives (every earlier target).
CAP_WINDOWS: Tuple[str, ...] = ("multiturn", "multiturn_rerun")


def _window_block(w, *, target: str, where: str) -> Dict[str, object]:
    """ONE family's pin block: ``{multiturn, multiturn_rerun (each a positive int on the 16-token block), source}``, only the
    windows named; refuses an unknown key, a missing source, a bad value, or a block naming no window."""
    if not isinstance(w, dict):
        raise ValueError(f"{target}: {where} must be a mapping of {list(CAP_WINDOWS)} + source")
    unknown = sorted(set(w) - set(CAP_WINDOWS) - {"source"})
    if unknown:
        raise ValueError(f"{target}: {where} carries unknown keys {unknown}")
    pins = [k for k in CAP_WINDOWS if k in w]
    if not pins:
        raise ValueError(f"{target}: {where} names no window ({list(CAP_WINDOWS)})")
    src = w.get("source")
    if not isinstance(src, str) or not src.strip():
        raise ValueError(f"{target}: {where}.source must name the measurement the pins rest on")
    for k in pins:
        v = w[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1 or v % 16:
            raise ValueError(f"{target}: {where}.{k}={v!r} must be a positive int multiple of 16 (a pending entry is not a window)")
    return dict(w)


def target_windows(roster: dict, target: str) -> Dict[str, Dict[str, object]]:
    """``{family: {"multiturn": L, "multiturn_rerun": Lr, "source": str}}`` — the caps block's PINNED multi-turn windows
    PER FAMILY (only the families pinned, only the windows each names) — or ``{}`` for a target that derives them all
    (no ``caps.windows``). A FLAT block ({multiturn, multiturn_rerun, source} with no family key) pins every family alike.
    Refuses an unknown family or key, a missing source (the measurement a pin rests on), a pin that is not a positive int
    on the 16-token block, or a block naming no window; whether a pin is wide enough is geometry's rule (pin_window),
    applied where the window is resolved — at the family's OWN cap."""
    caps = roster["targets"][target].get("caps")
    w = caps.get("windows") if isinstance(caps, dict) else None
    if w is None:
        return {}
    if not isinstance(w, dict):
        raise ValueError(f"{target}: caps.windows must be a mapping family ({list(MULTITURN_FAMILIES)}) -> {list(CAP_WINDOWS)} + source")
    if any(f in w for f in MULTITURN_FAMILIES):
        unknown = sorted(set(w) - set(MULTITURN_FAMILIES))
        if unknown:
            raise ValueError(f"{target}: caps.windows carries unknown families {unknown} (the multi-turn families are "
                             f"{list(MULTITURN_FAMILIES)}; a flat block must not mix with per-family blocks)")
        return {f: _window_block(w[f], target=target, where=f"caps.windows.{f}") for f in MULTITURN_FAMILIES if f in w}
    flat = _window_block(w, target=target, where="caps.windows")
    return {f: dict(flat) for f in MULTITURN_FAMILIES}


def target_multiturn_families(roster: dict, target: str) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """The multi-turn families that RUN for a target and the ones DROPPED with their reasons: the keep-or-drop switch of caps.yaml's
    ``multiturn_families``, rendered into the roster's
    caps block as ``caps.multiturn_families_omitted: {family: reason}``. ``(MULTITURN_FAMILIES, {})`` for a target without a caps
    block or without the key: every family runs. Refuses an unknown family, a non-string / empty / pending reason, or a block that
    drops every family (a multi-turn leg with no family is an omitted leg, not an empty one). The cap map still names every family
    (``multiturn_family_caps``); the printer plans no job for a dropped family and records the omission, never an inferred absence."""
    caps = roster["targets"][target].get("caps")
    om = caps.get("multiturn_families_omitted") if isinstance(caps, dict) else None
    if not om:
        return MULTITURN_FAMILIES, {}
    if not isinstance(om, dict):
        raise ValueError(f"{target}: caps.multiturn_families_omitted must be a mapping family -> the reason it is dropped")
    unknown = sorted(set(om) - set(MULTITURN_FAMILIES))
    if unknown:
        raise ValueError(f"{target}: caps.multiturn_families_omitted names unknown families {unknown} (the multi-turn families are {list(MULTITURN_FAMILIES)})")
    for f, why in om.items():
        if not isinstance(why, str) or not why.strip() or "pending" in why.lower():
            raise ValueError(f"{target}: caps.multiturn_families_omitted.{f} must give a final reason (got {why!r})")
    run = tuple(f for f in MULTITURN_FAMILIES if f not in om)
    if not run:
        raise ValueError(f"{target}: caps.multiturn_families_omitted drops every multi-turn family: omit the attacks leg instead (views.<view>.legs_omitted)")
    return run, {f: om[f] for f in MULTITURN_FAMILIES if f in om}


def target_capability_controls(roster: dict, target: str) -> Tuple[str, ...]:
    """The controls the target's ``capability`` leg runs: ``LEG_CONTROLS['capability']`` unless
    ``capability_controls`` narrows it (``[mmlu]`` where gsm8k was removed). Every dropped control must be named in
    ``capability_omitted`` WITH a reason (the omission is a recorded
    fact, like ``legs_omitted``), and an omitted control may not also be listed."""
    t = roster["targets"][target]
    full = LEG_CONTROLS["capability"]
    sel = t.get("capability_controls")
    omitted = dict(t.get("capability_omitted") or {})
    if sel is None:
        if omitted:
            raise ValueError(f"{target}: capability_omitted without capability_controls — name what runs")
        return full
    sel = tuple(sel)
    bad = [c for c in sel if c not in full]
    if bad or not sel or len(set(sel)) != len(sel):
        raise ValueError(f"{target}: capability_controls {list(sel)} must be a non-empty subset of {list(full)}")
    for c in [c for c in full if c not in sel]:
        why = omitted.get(c)
        if not isinstance(why, str) or not why.strip():
            raise ValueError(f"{target}: capability control {c} dropped without a reason (capability_omitted.{c})")
    both = [c for c in omitted if c in sel]
    if both:
        raise ValueError(f"{target}: capability_omitted names a control the target runs: {both}")
    unknown = [c for c in omitted if c not in full]
    if unknown:
        raise ValueError(f"{target}: capability_omitted names unknown controls {unknown}")
    return tuple(c for c in full if c in sel)


def target_capability_omitted(roster: dict, target: str) -> Dict[str, str]:
    """``{control: reason}`` for every capability control the target does not run (empty for every earlier target)."""
    target_capability_controls(roster, target)   # the grammar check
    return dict(roster["targets"][target].get("capability_omitted") or {})


#: CONTROLS RETIRED FROM THE WAVE'S CONTROLS LEG, WAVE-WIDE (roster top-level ``controls_retired: {control: reason}``;
#: user ruling 2026-09-14: gsm8k). Distinct from a TARGET's ``capability_omitted`` on purpose — that key also flips the
#: sampled lane's ``readouts.gsm8k`` to an omission record (make_sampled_configs.expected_readouts) and the drivers' run
#: records (sisteer.eval.capability_omissions), i.e. it rewrites the FROZEN instrument of runs that already happened (the
#: sixteen live main sampled configs are pinned as data by tests/test_sdfbase_full22_sampled.py). A wave-wide retirement
#: leaves those records alone and drops the control from every list :func:`view_controls` builds — the wave's and the
#: smoke's controls driver, the sampled sbatch's gsm8k gate, the printers. A retired control must be one the leg grammar
#: knows (LEG_CONTROLS); the reason is mandatory.
CONTROLS_RETIRED_KEY = "controls_retired"


def controls_retired(roster: dict) -> Dict[str, str]:
    """``{control: reason}`` retired from the wave's controls leg on EVERY target (roster top-level ``controls_retired``),
    ``{}`` when the roster names none. Grammar: a mapping whose keys are controls of LEG_CONTROLS and whose values are
    non-empty reasons; anything else refuses."""
    blk = roster.get(CONTROLS_RETIRED_KEY)
    if blk is None:
        return {}
    if not isinstance(blk, dict):
        raise ValueError(f"{CONTROLS_RETIRED_KEY} must be a mapping control -> reason")
    known = {c for cs in LEG_CONTROLS.values() for c in cs}
    unknown = sorted(set(blk) - known)
    if unknown:
        raise ValueError(f"{CONTROLS_RETIRED_KEY} names unknown controls {unknown} (the leg grammar knows {sorted(known)})")
    for c, why in blk.items():
        if not isinstance(why, str) or not why.strip():
            raise ValueError(f"{CONTROLS_RETIRED_KEY}.{c} needs a reason")
    return {c: str(blk[c]) for c in blk}


def view_controls(roster: dict, target: str, view: str) -> Tuple[str, ...]:
    """The controls driver's ``--controls`` list for the view: the union of its controls-driver legs,
    in the driver's historical order (gsm8k, mmlu, xstest, or_bench_hard_1k); the capability leg's list is the
    target's (``target_capability_controls`` — mmlu only where the roster removed gsm8k), MINUS the wave-wide
    retirements (:func:`controls_retired`: gsm8k since 2026-09-14)."""
    legs = view_legs(roster, target, view)
    order = ("gsm8k", "mmlu", "xstest", "or_bench_hard_1k")
    want = {c for l in legs
            for c in (target_capability_controls(roster, target) if l == "capability" else LEG_CONTROLS.get(l, ()))}
    want -= set(controls_retired(roster))
    return tuple(c for c in order if c in want)


#: the legs a per-leg ``serving.windows`` block names (the shortconst view): the static
#: standup (the four families + the BoN sub-phase ride it), the controls standup (the greedy MMLU / XSTest / OR-Bench
#: job and the sampled controls ladder's base rung), and the ladder RUNGS (one window per rung, shared by the static
#: and controls ladders — the static max governs, as in the sampled long-SI table)
SERVING_WINDOW_LEGS: Tuple[str, ...] = ("static", "controls")
SERVING_WINDOW_RUNGS: Tuple[int, ...] = (2048, 8192, 16384)


def _window_int(v, *, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1 or v % 16:
        raise ValueError(f"{what}={v!r} must be a positive multiple of the vLLM block size (16)")
    return int(v)


#: THE CONTROLS STANDUP'S WINDOW FOR A gsm8k-BEARING TABLE VIEW (roster top-level ``controls_window: {gsm8k: L,
#: source}``; 2026-09-14). The wave's ONE controls standup (target_static.sbatch: capability:short serving
#: gsm8k + mmlu + xstest + or_bench_hard_1k) stood every table view up at 2,048 — sized (2026-09-03) for the RETIRED
#: 0-shot gsm8k leg's <= 1.1k prompts. The standard Inspect gsm8k control (PW3) renders a 10-SHOT system turn: prompt
#: maxima gemma4e4b 2,310 / qwen35_4b 2,373 / nemotron3nano4b 2,336 / glm46vflash 2,128 bare, up to 2,462 under the
#: prompted__safety_refusal turn (geometry ``capability:gen_long_gsm8k_rung``), + the 512 controls cap = 2,974 > 2,048:
#: vLLM refused 34 of 200 items (smoke 503471) and the wave's own gsm8k control never produced a row on those views.
#: The value is DATA here — roster_view.py runs under plain python3 with no geometry import — and is held to the table
#: by tests/test_f22_gsm8k_controls_window.py (>= geometry.derived_window('capability:gen_long_gsm8k_rung',
#: controls_max_new_tokens, from_field='max_prompt') = 3,072, and == the table's capability:gen_long window, 4,608 —
#: where the GSM8K paired-200 cap-512 leg stood up on all four models and nemotron's hybrid ceiling is MEASURED)
#: and by common.sh f22_gsm8k_window_check at job time. A view's own serving override (one-window / per-leg) is its
#: own pin and is not touched here: it must itself clear the minimum. The cap is never shrunk to fit a window
#: (the standing truncation rule); the window widens and S follows from the budget.
CONTROLS_WINDOW_KEY = "controls_window"


def gsm8k_controls_window(roster: dict) -> Optional[Dict[str, object]]:
    """``{"max_model_len": L, "source": str}`` from the roster's top-level ``controls_window`` block, or ``None``
    when the roster names none. Grammar: keys exactly ``gsm8k`` (a positive multiple of the vLLM block, 16) and
    ``source`` (a non-empty string naming the derivation the window rests on); anything else refuses."""
    blk = roster.get(CONTROLS_WINDOW_KEY)
    if blk is None:
        return None
    if not isinstance(blk, dict):
        raise ValueError(f"{CONTROLS_WINDOW_KEY} must be a mapping {{gsm8k: L, source: str}}")
    unknown = sorted(set(blk) - {"gsm8k", "source"})
    if unknown:
        raise ValueError(f"{CONTROLS_WINDOW_KEY} carries unknown keys {unknown} (only gsm8k + source)")
    if "gsm8k" not in blk:
        raise ValueError(f"{CONTROLS_WINDOW_KEY} names no gsm8k window")
    L = _window_int(blk["gsm8k"], what=f"{CONTROLS_WINDOW_KEY}.gsm8k")
    src = blk.get("source")
    if not isinstance(src, str) or not src.strip():
        raise ValueError(f"{CONTROLS_WINDOW_KEY}.source must name the derivation the window rests on")
    return {"max_model_len": L, "source": src}


def view_serving(roster: dict, target: str, view: str) -> Dict[str, object]:
    """The view's serving overrides — empty for a view that serves at the geometry table's windows.
    Two shapes, never both: ``serving: {max_model_len: L}`` = ONE window for EVERY standup of the view
    (the llama8b fullconst view, 45,056); ``serving: {windows: {static, controls, rungs: {2048, 8192,
    16384}, [multiturn: {family: L}], source}}`` = PER-LEG windows for a prompted-only view that runs
    the whole battery under a long SI prefix (the llama8b shortconst view, 2026-09-12: the sampled
    lane's long-SI mechanism — every window max-prompt-sized for THAT leg's cap; ``multiturn`` pins a
    family's FIRST-PASS window over the target's caps.windows pin, only wider, when the prefix pushes
    the measured maximum past the target pin). S is never hand-picked (the standup's strict identity
    derives it from the model's budget at L). Refuses any other key, a bad value, or a per-leg block
    naming a leg / rung outside SERVING_WINDOW_LEGS / SERVING_WINDOW_RUNGS or missing one."""
    v = roster["targets"][target]["views"][view]
    s = dict(v.get("serving") or {})
    bad = sorted(set(s) - {"max_model_len", "windows"})
    if bad:
        raise ValueError(f"{target}/{view}: serving carries {bad}; only max_model_len (one window for every "
                         "standup) or windows (per leg) may be set (S derives from the budget)")
    if "max_model_len" in s and "windows" in s:
        raise ValueError(f"{target}/{view}: serving names BOTH max_model_len and windows — one window for every "
                         "standup or per-leg windows, never both")
    if "max_model_len" in s:
        _window_int(s["max_model_len"], what=f"{target}/{view}: serving.max_model_len")
    if "windows" in s:
        w = s["windows"]
        if not isinstance(w, dict):
            raise ValueError(f"{target}/{view}: serving.windows must be a mapping {list(SERVING_WINDOW_LEGS)} + rungs "
                             f"{list(SERVING_WINDOW_RUNGS)} + source")
        unknown = sorted(set(w) - set(SERVING_WINDOW_LEGS) - {"rungs", "multiturn", "source"})
        if unknown:
            raise ValueError(f"{target}/{view}: serving.windows carries unknown keys {unknown}")
        missing = [k for k in SERVING_WINDOW_LEGS if k not in w] + (["rungs"] if "rungs" not in w else [])
        if missing:
            raise ValueError(f"{target}/{view}: serving.windows is missing {missing} — every leg the view stands up needs "
                             "its window (a per-leg block is complete or refused)")
        for k in SERVING_WINDOW_LEGS:
            _window_int(w[k], what=f"{target}/{view}: serving.windows.{k}")
        rungs = w["rungs"]
        if not isinstance(rungs, dict) or sorted(rungs) != sorted(SERVING_WINDOW_RUNGS):
            raise ValueError(f"{target}/{view}: serving.windows.rungs must name exactly the rungs {list(SERVING_WINDOW_RUNGS)} "
                             f"(got {sorted(rungs) if isinstance(rungs, dict) else rungs!r})")
        for r in SERVING_WINDOW_RUNGS:
            _window_int(rungs[r], what=f"{target}/{view}: serving.windows.rungs[{r}]")
            if rungs[r] <= r:
                raise ValueError(f"{target}/{view}: serving.windows.rungs[{r}]={rungs[r]} leaves no room for a prompt at the rung's cap")
        mt = w.get("multiturn")
        if mt is not None:
            if not isinstance(mt, dict) or not mt:
                raise ValueError(f"{target}/{view}: serving.windows.multiturn must be a non-empty mapping family -> first-pass window")
            unknown_f = sorted(set(mt) - set(MULTITURN_FAMILIES))
            if unknown_f:
                raise ValueError(f"{target}/{view}: serving.windows.multiturn names unknown families {unknown_f} "
                                 f"(the multi-turn families are {list(MULTITURN_FAMILIES)})")
            for f, L in mt.items():
                _window_int(L, what=f"{target}/{view}: serving.windows.multiturn.{f}")
        src = w.get("source")
        if not isinstance(src, str) or not src.strip():
            raise ValueError(f"{target}/{view}: serving.windows.source must name the measurement the windows rest on")
    return s


def view_windows(roster: dict, target: str, view: str) -> Dict[str, object]:
    """The windows a (target, view) stands its legs up at, NORMALISED from ``view_serving``:
    ``{"static": L|None, "controls": L|None, "controls_gsm8k": L|None, "rungs": {2048: L, 8192: L, 16384: L} | {},
    "source": str|None, "kind": "table" | "one_window" | "per_leg"}``. A table view: static / controls None, rungs {}
    (the sbatch scripts and the printers keep their table defaults) — EXCEPT ``controls_gsm8k``, the roster's
    ``controls_window.gsm8k`` when the view's controls list carries gsm8k (:func:`gsm8k_controls_window`; refused when a
    gsm8k-bearing table view finds none — 2,048 does not hold the 10-shot prompt + the cap), None otherwise; a
    one-window view (fullconst): static = controls = every rung = L (the 16,384 rung's own-window rule stays where it
    lives — common_sampled.sh / submit_sampled.py); a per-leg view (shortconst): the block's values. Both override
    kinds answer ``controls_gsm8k`` None: their controls window is the view's own pin. The sbatch scripts read these
    through roster_view.py (``window <leg>`` / ``rung_window <rung>``) and the printers call this function, so a job
    and its plan cannot disagree."""
    s = view_serving(roster, target, view)
    if "windows" in s:
        w = s["windows"]
        return {"static": int(w["static"]), "controls": int(w["controls"]), "controls_gsm8k": None,
                "rungs": {int(r): int(w["rungs"][r]) for r in SERVING_WINDOW_RUNGS}, "source": w.get("source"), "kind": "per_leg"}
    if "max_model_len" in s:
        L = int(s["max_model_len"])
        return {"static": L, "controls": L, "controls_gsm8k": None,
                "rungs": {int(r): L for r in SERVING_WINDOW_RUNGS}, "source": None, "kind": "one_window"}
    g8 = None
    if "gsm8k" in view_controls(roster, target, view):
        blk = gsm8k_controls_window(roster)
        if blk is None:
            raise ValueError(f"{target}/{view}: runs the gsm8k control but the roster names no {CONTROLS_WINDOW_KEY}.gsm8k — "
                             "capability:short's table window (2,048) does not hold the 10-shot gsm8k prompt + the controls "
                             "cap (D72); widen the window, never the cap")
        g8 = int(blk["max_model_len"])
    return {"static": None, "controls": None, "controls_gsm8k": g8, "rungs": {}, "source": None, "kind": "table"}


def view_multiturn_windows(roster: dict, target: str, view: str) -> Dict[str, Dict[str, object]]:
    """The multi-turn window PINS a (target, view) serves — the target's ``caps.windows`` (target_windows) with the
    view's ``serving.windows.multiturn`` FIRST-PASS overrides laid over it (only wider: a view override below the
    target's pin, or below the family's derived minimum, is refused — the latter by geometry.pin_window where the
    window is resolved). The generator writes THIS map into the view's config (attacks.multiturn_max_model_len), which
    target_attacks.sbatch and the re-run pass read, so a view's override is the served pin, never a printer's note."""
    base = {f: dict(b) for f, b in target_windows(roster, target).items()}
    s = view_serving(roster, target, view)
    mt = (s.get("windows") or {}).get("multiturn") if "windows" in s else None
    for f, L in (mt or {}).items():
        b = base.setdefault(f, {})
        tgt = b.get("multiturn")
        if tgt is not None and int(L) < int(tgt):
            raise ValueError(f"{target}/{view}: serving.windows.multiturn.{f}={L} is below the target's pin {tgt} "
                             "(caps.windows) — a view may only widen a family's first pass")
        rr = b.get("multiturn_rerun")
        if rr is not None and int(L) >= int(rr):
            raise ValueError(f"{target}/{view}: serving.windows.multiturn.{f}={L} is not below the family's re-run pin {rr} — "
                             "the re-run window could carry no goal the first pass refused")
        b["multiturn"] = int(L)
        b["source"] = f"{(s['windows'].get('source') or '').strip()} (view override of the first pass; the target's pin: {b.get('source', '')})".strip()
    return base


def target_xstest_subset(roster: dict, target: str) -> Optional[str]:
    """The target's XSTest selection for the over-refusal control (``xstest_subset: safe`` — the
    llama8b lane: the 250 SAFE prompts); None = the historical 450. Emitted as
    the top-level config key the controls drivers read (cell_controls_run.XSTEST_SUBSET_KEY)."""
    sub = roster["targets"][target].get("xstest_subset")
    if sub is None:
        return None
    if sub != "safe":
        raise ValueError(f"{target}: xstest_subset={sub!r}; only 'safe' or absent")
    return "safe"


def view_prompted_only(roster: dict, target: str, view: str) -> bool:
    """A prompted-only view serves `none` + prompted baselines and no object cell, by design;
    a view that says so and maps an object cell, or maps none and does not say so, is refused."""
    v = roster["targets"][target]["views"][view]
    flag = bool(v.get("prompted_only", False))
    staged = staged_cells(roster, target, view)
    if flag and staged:
        raise ValueError(f"{target}/{view}: prompted_only but maps object cells {sorted(staged)}")
    return flag


def pending_fit_of(roster: dict, target: str, view: Optional[str] = None) -> Dict[str, str]:
    """``{config cell id: staged id}`` the roster names as NOT YET FITTED — the target-level
    ``pending_fit`` map (every roster before 2026-09-10) plus, when ``view`` is given, that view's own
    ``pending_fit`` (two lanes whose config ids collide record their pending fits per view)."""
    t = roster["targets"][target]
    out = dict(t.get("pending_fit") or {})
    if view is not None:
        out.update(t["views"][view].get("pending_fit") or {})
    return out


def pending_decision_of(roster: dict, target: str) -> Dict[str, dict]:
    """``{config cell id: {staged_id, waits_on, state}}`` — enabled by hand, never auto-enabled."""
    out = {}
    for cid, e in (roster["targets"][target].get("pending_decision") or {}).items():
        if not isinstance(e, dict) or not e.get("staged_id") or not e.get("waits_on") or not e.get("state"):
            raise ValueError(f"{target}: pending_decision.{cid} must carry staged_id, waits_on, state")
        out[cid] = dict(e)
    return out


def every_staged_id(roster: dict, target: str) -> List[str]:
    """Every staged id the target names anywhere (views, lanes on or off, pending_* maps) — the
    'exactly once' invariant's domain."""
    t = roster["targets"][target]
    # a prompted baseline maps to null (no staged object) and names nothing here; a PIN-form cell
    # names an object in ANOTHER bank through its pin (the builder re-hashes it there) and no
    # staged id of this bank
    ids = [sid for v in t["views"].values() for sid in (v.get("cells") or {}).values()
           if sid is not None and not is_pin_form(sid)]
    for lane_cells in (t.get("lanes") or {}).values():
        ids += [sid for sid in (lane_cells or {}).values() if sid is not None]
    ids += list((t.get("pending_fit") or {}).values())
    ids += [sid for v in t["views"].values() for sid in (v.get("pending_fit") or {}).values()]
    ids += list((t.get("pending_stage") or {}).values())
    ids += [e["staged_id"] for e in pending_decision_of(roster, target).values()]
    ids += [e["staged_id"] for e in (t.get("not_served") or [])]
    return ids
