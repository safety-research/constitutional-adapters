#!/usr/bin/env python3
"""gsm8k-paired200-paper-objects — THE ONE VIEW-SELECTION INTERFACE: ``select_rows``.

A row is (model key, VIEW, cell). The per-(key, view) cell rule is ONE function, ``cells_of``:

    [none] + [prompted__safety_refusal iff view == it] + the roster's served cells of (key, view), in roster order

read through ``scripts/sdfbase/full22/f22_roster.served_cells`` (the roster's own one reader: the view's ``cells`` plus every
enabled lane's; a prompted baseline the roster maps to ``null`` comes through as a served cell with no object), so a view
the roster lacks refuses BY NAME and a lane flag flip moves this matrix with the configs and the bank views.

THREE consumers call it and none re-derives it: (i) ``make_gsm8k_matrix.py`` (this experiment's FROZEN 42-row
``configs/sdfbase/gsm8k/matrix.json`` from ``views.yaml``, and any peer's matrix from ``--views <spec> --configs-root <tree>``)
and ``make_gsm8k_overlays.py`` (one overlay pair per TAG of a matrix); (ii) production row lookup — ``g8k_cells.py row`` reads
the frozen matrix, stdlib-only; (iii) ``g8k_dry_run.py --views <spec> [--configs-root <tree>]`` runs the SAME selector over
the named tree's roster + pins and drives the SAME argv construction — so a peer's dose view (``it_d0p1`` at their sha)
becomes a printed production row with a generated overlay, the whole production path up to (not including) the standup.

HOLDS: ``views.yaml``'s ``holds:`` block names (key, view, cells) held back and why; the selector stamps
every matching row ``status: held`` (+ ``held_reason``) and every other row ``status: launch``. The matrix keeps its 42
rows, so indices, TAGs and overlays are stable; un-holding is exactly one edit here + regenerate, no other code change.

ADMIT AND HOLD: a ``prompted__<si>`` cell the roster declares on an expressed view that
this matrix does not SCHEDULE (``g8k_cells.SCHEDULED_PROMPTED_CELLS`` — today the matched safety baseline only) is admitted
as a row so the matrix agrees with the roster, and HELD so the launched set does not move: ``select_rows`` emits every
scheduled cell first (the rows above, indices unchanged) and then, TRAILING, one ``status: held`` row per unscheduled
prompted cell in views order then roster order, its reason ``g8k_cells.unscheduled_held_reason`` (a views.yaml hold naming
the same cell is moot: the structural reason wins). Every consumer treats it like any held row; scheduling the cell
(naming it in ``SCHEDULED_PROMPTED_CELLS``, or a later extension mechanism that appends launch rows) releases the hold.

``views_spec`` grammar (the CLI string form): ``key=view[,view…];key=view…`` — e.g. ``gemma4e4b=it,main; qwen35_4b=it,main;
nemotron3nano4b=main; glm46vflash=main`` (this experiment's) or ``gemma4e4b=it_d0p1`` (the peer's dose view).

llama-paper-rows (P7 / P11 / P15 / P16): a views file may declare ``schema: g8k_matrix/2`` + ``wave_root_rel:`` — its rows
carry the family's ARM (``thinking_mode``: ``on`` for the four switch families, ``not_available`` for the no-switch family
``llama``, whose fitted rows are IN SUPPORT — no tuple, no declared cell — and run the model's ``card`` sampling row), and an
``exclude:`` block (fallback (B): cells dropped from the matrix outright, with a reason) — none of which a schema-1 file may
carry, so the published document is byte-identical. ``frozen_inputs:`` names the IMMUTABLE roster / pins copies (+ their
sha256) the published matrix regenerates from: the live registry can grow a fifth target without the frozen document moving.

Import: yaml + the roster module + (at generation time) the harness's own cell-id spelling — NOT stdlib-only; the matrix
it writes is what the sbatch reads through ``g8k_cells.py``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, HERE.parent / "full22"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import g8k_cells as M
if str(HERE.parent / "seedrep") not in sys.path:      # scripts/sdfbase/seedrep: the lane's eval-seed rule (stdlib only)
    sys.path.insert(0, str(HERE.parent / "seedrep"))
import eval_seed as ES  # noqa: E402  (qwen-base-seed-replicates: the replicate seed bases)
import itdoc_bind as B
from f22_roster import served_cells, steered_wave_view

ROSTER_REL = Path("configs/sdfbase_full22_roster.yaml")
PINS_REL = Path("configs/sdfbase_full22_pins.json")
VIEWS_REL = Path("configs/sdfbase/gsm8k/views.yaml")


# --------------------------------------------------------------------------- the spec
@dataclass
class ViewsSpec:
    """Which (key, view) pairs the matrix expresses, in order, and which of their cells are held (schema 1 and 2), plus the
    schema-2 facts: the document's schema and wave root, the cells EXCLUDED outright (fallback (B)); and, for the published
    matrix, the frozen input copies it regenerates from (P16)."""
    views: dict[str, list[str]]                       # key -> [view, ...] (ordered)
    holds: list[dict] = field(default_factory=list)  # [{key, view, cells: [...], reason}]
    schema: str = M.MATRIX_SCHEMA
    wave_root_rel: Optional[str] = None               # schema 2 only (REQUIRED there); None on schema 1 (the published root)
    exclude: list[dict] = field(default_factory=list) # schema 2 only: [{key, view, cells: [...], reason}] — dropped, not held
    trigger_threshold: Optional[float] = None         # schema 2 only (REQUIRED there): the ladder trigger every row carries
    frozen_inputs: Optional[dict] = None              # {roster, pins: <path rel. to the configs root>, roster_sha256, pins_sha256}
    # math-eight-arms-extension E2 (plan P2 / P8): the rows APPENDED after every scheduled row, each block's rows in list order —
    # `extensions:` [{key, view, cells: [prompted__<si>, ...], reason, since}] = prompted routes on an expressed (key, view)
    # (no pin, no roster entry; wave `ext`, controls_ext/), `steering:` [{key, arms: [alphasteer, arditi], reason, since}] = the
    # rung's rows on the PANEL view `main`, generated from the P8 admission record (wave `steer`, controls_steer/)
    extensions: list[dict] = field(default_factory=list)
    steering: list[dict] = field(default_factory=list)
    # math-eight-arms-extension E3: `itdoc:` [{key, view: it, cells: [<the three LoRA arms>], reason, since}] = the instruct-direct
    # batch-64 RE-FIT rows, generated from the it-doc admission record (wave `itdoc`, controls_itdoc/); the mix cell's row REPLACES
    # the held direct-fit row in place, the others append
    itdoc: list[dict] = field(default_factory=list)
    # math-eight-arms-extension E3, schema 2 only: `row_overlays: true` = EVERY row of this matrix serves through its OWN
    # overlay pair (sdfbase_full22_<TAG>.gsm8k_<cell slug>.yaml, explicit_cells [none, <cell>]) — a second matrix cannot own the
    # published matrix's TAG overlays; `object_generation: true` = every fitted row records the object GENERATION its staged id
    # names (`batch8` for the itdoc__ / *it__ ids, `batch64` for the itdoc64__ / derived_itdoc64__ ids) so a wave mixing the two
    # generations (the served batch-8 constitution beside the batch-64 control + difference) names the mix on every row
    row_overlays: bool = False
    object_generation: bool = False
    # schema 2 only: the WAVE every base-pass row of this matrix belongs to (E3: `itdoc` — the roster-served constitution rows of the
    # instruct-direct wave live under controls_itdoc/ beside the record rows and are LAUNCHABLE; without it a base-pass row reads as
    # the COMPLETE `published` wave and is refused by the plan and the batch boundary). None = the published wave (every matrix before E3).
    wave: Optional[str] = None
    # qwen-base-seed-replicates, schema 2 only: `inspect_seed_policy: replicate` = every row of this matrix runs
    # the GSM8K driver under the OPT-IN Inspect replicate seed rule (contracts.SEED_RULE_INSPECT_REPLICATE) at ITS view's base —
    # `inspect_seed_base_by_view: {s1: <base>, ...}`, one base per expressed view, each equal to eval_seed.seed_base(s) (the
    # loader pins the declaration to the rule); the driver flags ride the row (`inspect_seed_policy` / `inspect_seed_base`)
    # into g8k_common.sh g8k_driver_argv. None = the default rule on every row, byte-identical to before.
    inspect_seed_policy: Optional[str] = None
    inspect_seed_base_by_view: Optional[dict] = None
    # qwen-base-seed-replicates (decisions O12a), schema 2 + `row_overlays` only: `standup_companions: {cell, reason}` = every standup
    # of this matrix ALSO registers the named steered cell of the row's view, UNMEASURED (the driver's --cell stays the row's), so the
    # native composer fits ONE `--max-lora-rank` across the wave's standups — the GSM8K reader compares that flag across the two legs
    # of a contrast (tinycap_math500_thinking.LAUNCH_EXCLUDED_KEYS excludes only the cell-set plumbing), and a `none`-only standup
    # (zero adapter r=1 -> cap 1) would be refused against a LoRA r=4 row's standup (cap 8). A row whose view does not stage the
    # companion is HELD (g8k_cells.COMPANION_HELD_MARK) — the provisional pre-staging freeze; the companion's own row carries none.
    standup_companions: Optional[dict] = None
    # The Qwen seed-replicate native battery, schema 2 only: `geometry_class: common_f22` = EVERY row of this matrix — short cells
    # included — serves at the common geometry pair (g8k_cells.GEOMETRY_CLASSES), so every arm of a leg shares one S x L; carried on every
    # row (`geometry_class`) and read by g8k_cells.geometry_for_cell(geometry_class=). None = the per-prompt-class rule, byte-identical.
    geometry_class: Optional[str] = None

    def spec_string(self) -> str:
        return "; ".join(f"{k}={','.join(vs)}" for k, vs in self.views.items())

    def hold_for(self, key: str, view: str, cell: str) -> Optional[str]:
        return _reason_for(self.holds, key, view, cell)

    def excluded(self, key: str, view: str, cell: str) -> Optional[str]:
        return _reason_for(self.exclude, key, view, cell)


def schema2_of(schema: str) -> bool:
    return schema == M.MATRIX_SCHEMA_2


def _reason_for(entries: Iterable[dict], key: str, view: str, cell: str) -> Optional[str]:
    for h in entries:
        if h["key"] == key and h["view"] == view and cell in h["cells"]:
            return str(h["reason"]).strip()
    return None


def parse_views_spec(spec: str) -> dict[str, list[str]]:
    """``key=view[,view…];…`` -> an ordered ``{key: [views]}``; refuses an empty key / view, a repeated key or view."""
    out: dict[str, list[str]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, eq, views = chunk.partition("=")
        key = key.strip()
        if not eq or not key:
            raise SystemExit(f"--views: {chunk!r} is not key=view[,view…]")
        if key in out:
            raise SystemExit(f"--views: key {key!r} is named twice")
        vs = [v.strip() for v in views.split(",") if v.strip()]
        if not vs:
            raise SystemExit(f"--views: key {key!r} names no view")
        if len(set(vs)) != len(vs):
            raise SystemExit(f"--views: key {key!r} names a view twice ({vs})")
        out[key] = vs
    if not out:
        raise SystemExit("--views: the spec names no (key, view) pair")
    return out


def _cell_entries(doc: dict, block: str, out_views: dict, *, path: Path, noun: str) -> list[dict]:
    """A ``holds:`` / ``exclude:`` list of ``{key, view, cells, reason}``: every entry names a (key, view) the ``views:`` block
    expresses, at least one cell, and a non-empty reason — else it refuses by name."""
    out = []
    for h in doc.get(block) or []:
        if not isinstance(h, dict) or not {"key", "view", "cells", "reason"} <= set(h):
            raise SystemExit(f"{path}: every `{block}:` entry needs key, view, cells, reason — got {h!r}")
        if h["view"] not in out_views.get(h["key"], []):
            raise SystemExit(f"{path}: {noun} on {h['key']}/{h['view']} names a (key, view) the `views:` block does not express")
        if not str(h["reason"]).strip():
            raise SystemExit(f"{path}: {noun} on {h['key']}/{h['view']} carries no reason")
        cells = [str(c) for c in (h["cells"] if isinstance(h["cells"], list) else [h["cells"]])]
        if not cells:
            raise SystemExit(f"{path}: {noun} on {h['key']}/{h['view']} names no cell")
        out.append({"key": str(h["key"]), "view": str(h["view"]), "cells": cells, "reason": str(h["reason"]).strip()})
    return out


def load_views_yaml(path: Path) -> ViewsSpec:
    """``views.yaml``: ``views: {key: [view, ...]}`` (ordered) + an optional ``holds:`` list of ``{key, view, cells, reason}``;
    a hold naming a (key, view) outside ``views`` or with no reason refuses by name. Schema 2 (``schema: g8k_matrix/2``)
    REQUIRES ``wave_root_rel:`` and may carry ``exclude:`` (same grammar as holds; a cell both held and excluded refuses);
    a schema-1 file carrying ``wave_root_rel:`` / ``trigger_threshold:`` refuses. A schema-1 file's ``exclude:`` is NOT this
    loader's block and is not read: the IFEval lane's ``views.yaml`` carries one in its OWN grammar (``{arms_matching | cells,
    reason}``, read by ``if_views.load_ifeval_policy``) and reuses only this neutral enumeration (P31) — the loader leaves a
    block it does not own to its owner rather than refusing the file. ``frozen_inputs:`` (``roster``, ``pins`` paths relative
    to the configs root + ``roster_sha256`` / ``pins_sha256``) is admitted on either schema."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    views = doc.get("views")
    if not isinstance(views, dict) or not views:
        raise SystemExit(f"{path}: `views:` must map model keys to a list of views")
    out_views = {str(k): [str(v) for v in (vs if isinstance(vs, list) else [vs])] for k, vs in views.items()}
    schema = str(doc.get("schema") or M.MATRIX_SCHEMA)
    if schema not in M.MATRIX_SCHEMAS:
        raise SystemExit(f"{path}: schema {schema!r} is not one of {M.MATRIX_SCHEMAS}")
    schema2 = schema == M.MATRIX_SCHEMA_2
    wave_root_rel = doc.get("wave_root_rel")
    if schema2 and (not isinstance(wave_root_rel, str) or not wave_root_rel.strip() or wave_root_rel.startswith("/")):
        raise SystemExit(f"{path}: a {M.MATRIX_SCHEMA_2} views file names its wave root (`wave_root_rel:`, relative to the shared root)")
    if not schema2 and wave_root_rel is not None:
        raise SystemExit(f"{path}: `wave_root_rel:` is a {M.MATRIX_SCHEMA_2} field — the published (schema-1) matrix's root is the launcher's")
    trig = doc.get("trigger_threshold")
    if schema2 and not (isinstance(trig, (int, float)) and not isinstance(trig, bool) and 0.0 < float(trig) < 1.0):
        raise SystemExit(f"{path}: a {M.MATRIX_SCHEMA_2} views file declares its ladder trigger (`trigger_threshold:`, a share in (0, 1) — "
                         f"the instrument's, harmonised 2026-09-12, is {M.TRIGGER_THRESHOLD}); got {trig!r}")
    if not schema2 and trig is not None:
        raise SystemExit(f"{path}: `trigger_threshold:` is a {M.MATRIX_SCHEMA_2} field — the published matrix reads the adapter's {M.TRIGGER_THRESHOLD}")
    if schema2 and wave_root_rel.strip() == M.WAVE_ROOT_REL:
        raise SystemExit(f"{path}: a {M.MATRIX_SCHEMA_2} matrix runs under its OWN wave root, never the published one ({M.WAVE_ROOT_REL})")
    holds = _cell_entries(doc, "holds", out_views, path=path, noun="hold")
    # `exclude:` is a schema-2 block (fallback (B)); a schema-1 file's is another lane's (see the docstring) and stays unread here
    exclude = _cell_entries(doc, "exclude", out_views, path=path, noun="exclusion") if schema2 else []
    for e in exclude:
        for c in e["cells"]:
            if _reason_for(holds, e["key"], e["view"], c) is not None:
                raise SystemExit(f"{path}: {e['key']}/{e['view']}/{c} is both held and excluded — a cell is one or the other")
    extensions = _extension_entries(doc, out_views, path=path)
    steering = _steering_entries(doc, out_views, path=path)
    itdoc = _itdoc_entries(doc, out_views, path=path)
    for e in itdoc:                      # a hold naming an it-doc cell is moot AND misleading: the record governs that row's status
        for c in e["cells"]:
            if _reason_for(holds, e["key"], e["view"], c) is not None:
                raise SystemExit(f"{path}: {e['key']}/{e['view']}/{c} is both a `holds:` cell and an `itdoc:` cell — the it-doc admission "
                                 "record governs a re-fit row's status (its `hold_reason`); drop the cell from `holds:`")
    wave = doc.get("wave")
    if wave is not None:
        if not schema2:
            raise SystemExit(f"{path}: `wave:` is a {M.MATRIX_SCHEMA_2} field — the published matrix's base rows are the published wave")
        if not isinstance(wave, str) or wave not in M.OUT_ROOT_PREFIX_BY_WAVE or wave in M.COMPLETED_WAVES:
            raise SystemExit(f"{path}: `wave:` must name a LAUNCHABLE wave of the launcher ({[w for w in M.OUT_ROOT_PREFIX_BY_WAVE if w not in M.COMPLETED_WAVES]}), got {wave!r}")
    flags = {}
    for flag in ("row_overlays", "object_generation"):
        v = doc.get(flag)
        if v is not None and not isinstance(v, bool):
            raise SystemExit(f"{path}: `{flag}:` is a boolean")
        if v and not schema2:
            raise SystemExit(f"{path}: `{flag}:` is a {M.MATRIX_SCHEMA_2} flag — the published (schema-1) matrix's rows serve their TAG overlays")
        flags[flag] = bool(v)
    seed_policy, seed_bases = _inspect_seed_policy_entries(doc, out_views, path=path, schema2=schema2)
    companions = _standup_companions_entry(doc, path=path, schema2=schema2, row_overlays=flags["row_overlays"])
    gclass = doc.get(M.GEOMETRY_CLASS_KEY)
    if gclass is not None:
        if not schema2:
            raise SystemExit(f"{path}: `{M.GEOMETRY_CLASS_KEY}:` is a {M.MATRIX_SCHEMA_2} field — the published matrix's rows serve at their prompt class's entries")
        if not isinstance(gclass, str) or gclass not in M.GEOMETRY_CLASSES:
            raise SystemExit(f"{path}: `{M.GEOMETRY_CLASS_KEY}:` must be one of {sorted(M.GEOMETRY_CLASSES)} (g8k_cells.GEOMETRY_CLASSES), got {gclass!r}")
        for key in out_views:
            if M.longsi_scope(key) != M.GEOMETRY_CLASS_SCOPE[gclass]:
                raise SystemExit(f"{path}: `{M.GEOMETRY_CLASS_KEY}: {gclass}` is scoped to the {M.GEOMETRY_CLASS_SCOPE[gclass]!r} families; "
                                 f"the `views:` block expresses {key!r} ({M.longsi_scope(key)!r})")
    frozen = doc.get("frozen_inputs")
    if frozen is not None:
        need = {"roster", "pins", "roster_sha256", "pins_sha256"}
        if not isinstance(frozen, dict) or set(frozen) != need:
            raise SystemExit(f"{path}: `frozen_inputs:` carries exactly {sorted(need)} — got {sorted(frozen) if isinstance(frozen, dict) else frozen!r}")
        for k in ("roster_sha256", "pins_sha256"):
            if not (isinstance(frozen[k], str) and len(frozen[k]) == 64 and all(ch in "0123456789abcdef" for ch in frozen[k])):
                raise SystemExit(f"{path}: frozen_inputs.{k} is not a 64-hex sha256")
        frozen = {k: str(v) for k, v in frozen.items()}
    return ViewsSpec(views=out_views, holds=holds, schema=schema, wave_root_rel=(wave_root_rel.strip() if schema2 else None),
                     exclude=exclude, frozen_inputs=frozen, trigger_threshold=(float(trig) if schema2 else None),
                     extensions=extensions, steering=steering, itdoc=itdoc, row_overlays=flags["row_overlays"],
                     object_generation=flags["object_generation"], wave=(str(wave) if wave else None),
                     inspect_seed_policy=seed_policy, inspect_seed_base_by_view=seed_bases, standup_companions=companions,
                     geometry_class=(str(gclass) if gclass is not None else None))


def _standup_companions_entry(doc: dict, *, path: Path, schema2: bool, row_overlays: bool) -> Optional[dict]:
    """The ``standup_companions:`` block — ``{cell: <TrainTest cell id>, reason: <text>}``, schema 2
    with ``row_overlays`` only (the companion rides the row's OWN pair; a TAG overlay already lists every cell). The cell is a steered
    object cell: never ``none``, never a prompted baseline. ``None`` when the file declares none."""
    block = doc.get(M.STANDUP_COMPANIONS_KEY)
    if block is None:
        return None
    if not schema2:
        raise SystemExit(f"{path}: `{M.STANDUP_COMPANIONS_KEY}:` is a {M.MATRIX_SCHEMA_2} block")
    if not row_overlays:
        raise SystemExit(f"{path}: `{M.STANDUP_COMPANIONS_KEY}:` needs `row_overlays: true` — the companion is enumerated on the row's OWN overlay pair")
    if not isinstance(block, dict) or set(block) != {"cell", "reason"}:
        raise SystemExit(f"{path}: `{M.STANDUP_COMPANIONS_KEY}:` carries exactly {{cell, reason}} — got {sorted(block) if isinstance(block, dict) else block!r}")
    cell, reason = block.get("cell"), block.get("reason")
    if not isinstance(cell, str) or not cell.strip() or cell.strip() == M.CELL_NONE or M.is_prompted_cell(cell.strip()):
        raise SystemExit(f"{path}: {M.STANDUP_COMPANIONS_KEY}.cell must name a steered object cell (never `none`, never a prompted baseline), got {cell!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise SystemExit(f"{path}: {M.STANDUP_COMPANIONS_KEY}.reason is the recorded why (non-empty)")
    return {"cell": cell.strip(), "reason": str(reason).strip()}


def _inspect_seed_policy_entries(doc: dict, out_views: dict, *, path: Path, schema2: bool) -> tuple:
    """qwen-base-seed-replicates: the views file's `inspect_seed_policy:` (+ `inspect_seed_base_by_view:`) — schema 2 only, the
    ONE admitted policy `replicate`, exactly one int base per expressed view, every view a seed-replicate view (`s<s>`) whose
    base IS eval_seed.seed_base(s) (a mistyped base refuses here, never at a GPU); a base map without the policy, or a policy
    without the map, refuses. Returns ``(policy | None, {view: base} | None)``."""
    policy = doc.get("inspect_seed_policy")
    bases = doc.get("inspect_seed_base_by_view")
    if policy is None and bases is None:
        return None, None
    if not schema2:
        raise SystemExit(f"{path}: `inspect_seed_policy:` is a {M.MATRIX_SCHEMA_2} field — the published matrix runs the default Inspect seed rule")
    if policy != M.INSPECT_SEED_POLICY_REPLICATE:
        raise SystemExit(f"{path}: `inspect_seed_policy:` must be {M.INSPECT_SEED_POLICY_REPLICATE!r} (the one opt-in rule; absent = the default), got {policy!r}")
    if not isinstance(bases, dict) or not bases:
        raise SystemExit(f"{path}: `inspect_seed_policy: {policy}` needs `inspect_seed_base_by_view:` ({{view: base}}) — one base per expressed view")
    expressed = {v for vs in out_views.values() for v in vs}
    if set(bases) != expressed:
        raise SystemExit(f"{path}: `inspect_seed_base_by_view:` names {sorted(bases)} but the expressed views are {sorted(expressed)} — "
                         "exactly one base per expressed view")
    out = {}
    for view, base in bases.items():
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise SystemExit(f"{path}: inspect_seed_base_by_view.{view} must be a non-negative int, got {base!r}")
        try:
            want = ES.seed_base(ES.seed_of_view(view))
        except ValueError as e:
            raise SystemExit(f"{path}: inspect_seed_base_by_view.{view}: {e}") from None
        if base != want:
            raise SystemExit(f"{path}: inspect_seed_base_by_view.{view} = {base} but eval_seed.seed_base({ES.seed_of_view(view)}) = {want} — "
                             "the declaration must equal the rule (scripts/sdfbase/seedrep/eval_seed.py)")
        out[str(view)] = int(base)
    return str(policy), out


def _dated(entry: dict, *, path: Path, block: str) -> str:
    since = str(entry.get("since") or "").strip()
    if not since:
        raise SystemExit(f"{path}: every `{block}:` entry names the date it was scheduled (`since`) — got {entry!r}")
    return since


def _extension_entries(doc: dict, out_views: dict, *, path: Path) -> list[dict]:
    """The ``extensions:`` block (math-eight-arms-extension E2, plan P2): ``[{key, view, cells, reason, since}]`` — every entry on
    a (key, view) the ``views:`` block expresses, every cell a PROMPTED baseline (``prompted__<si>``: a prompted route needs no
    pin and no roster entry; an object cell is a roster matter, never an extension), ``prompted__si_constitution_full`` refused
    by name, a reason and a date. The rows are appended after every scheduled row, in list order."""
    out = []
    for e in doc.get("extensions") or []:
        if not isinstance(e, dict) or not {"key", "view", "cells", "reason", "since"} <= set(e):
            raise SystemExit(f"{path}: every `extensions:` entry needs key, view, cells, reason, since — got {e!r}")
        if e["view"] not in out_views.get(e["key"], []):
            raise SystemExit(f"{path}: extension on {e['key']}/{e['view']} names a (key, view) the `views:` block does not express")
        if not str(e["reason"]).strip():
            raise SystemExit(f"{path}: extension on {e['key']}/{e['view']} carries no reason")
        cells = [str(c) for c in (e["cells"] if isinstance(e["cells"], list) else [e["cells"]])]
        if not cells or len(set(cells)) != len(cells):
            raise SystemExit(f"{path}: extension on {e['key']}/{e['view']} names no cell, or a cell twice ({cells})")
        for c in cells:
            if c == "prompted__si_constitution_full":
                raise SystemExit(f"{path}: {c!r} is not a legal cell of this experiment — the constitution arm is "
                                 f"{M.CELL_PROMPTED_CONST!r} (the SHORT SI); the _full variant is refused by name (D4)")
            if not M.is_prompted_cell(c):
                raise SystemExit(f"{path}: extension cell {c!r} on {e['key']}/{e['view']} is not a prompted baseline — an extension "
                                 "row is a prompted route (no pin, no roster entry); an object cell is the roster's and rides the base pass")
        out.append({"key": str(e["key"]), "view": str(e["view"]), "cells": cells, "reason": str(e["reason"]).strip(),
                    "since": _dated(e, path=path, block="extensions")})
    return out


def _steering_entries(doc: dict, out_views: dict, *, path: Path) -> list[dict]:
    """The ``steering:`` block (math-eight-arms-extension E2, plan P8): ``[{key, arms, reason, since}]`` + an OPTIONAL
    ``view:`` — a registered model whose named view the ``views:`` block expresses, arms among the rung's cells, a reason and a
    date. The rows are generated from the P8 ADMISSION RECORD; an arm the record does not admit for the model refuses at
    generation (a steering row never exists without its record).

    ``view:`` DEFAULTS to ``main`` (arditi-allpos-cap, 2026-09-17), so every entry written before it existed keeps its rows byte
    for byte: the rung's eight-arm slots are the PANEL view's. The all-positions Arditi arm's row sits on the INSTRUCT-DIRECT
    view instead, and the view is not a free choice per entry — ``g8k_cells.EXTERNAL_ROW_VIEW_BY_CELL`` declares it PER ARM and
    an entry naming the other view for an arm refuses BY NAME here, so two arms can never be mixed into one column by editing a
    views file. The view must also be one the ``views:`` block expresses for that model (a model that does not express it has no
    TAG, no campaign config and no bank there)."""
    out = []
    for e in doc.get("steering") or []:
        if not isinstance(e, dict) or not {"key", "arms", "reason", "since"} <= set(e):
            raise SystemExit(f"{path}: every `steering:` entry needs key, arms, reason, since — got {e!r}")
        key = str(e["key"])
        if key not in M.MODEL_KEYS:
            raise SystemExit(f"{path}: steering on {key!r}: not a registered model key ({M.MODEL_KEYS})")
        view = str(e.get("view") or M.VIEW_MAIN)
        if view not in out_views.get(key, []):
            raise SystemExit(f"{path}: steering on {key}/{view}: the `views:` block does not express that view of this model "
                             f"(its views: {out_views.get(key, [])}) — a steering row has no TAG, campaign config or bank on a "
                             "view the model does not express")
        arms = [str(a) for a in (e["arms"] if isinstance(e["arms"], list) else [e["arms"]])]
        bad = [a for a in arms if a not in M.STEER_CELLS]
        if not arms or bad or len(set(arms)) != len(arms):
            raise SystemExit(f"{path}: steering on {key}: arms must be distinct members of {M.STEER_CELLS}, got {arms}")
        for arm in arms:
            want = M.external_row_view_for(arm)
            if view != want:
                raise SystemExit(f"{path}: steering on {key}/{view}: the arm {arm!r} sits on the {want!r} view "
                                 f"(g8k_cells.EXTERNAL_ROW_VIEW_BY_CELL declares the view per arm) — an arm served on another arm's "
                                 "view would file a different intervention into that arm's column")
        if not str(e["reason"]).strip():
            raise SystemExit(f"{path}: steering on {key} carries no reason")
        # OPTIONAL `expected_contrast_refusal:` (arditi-allpos-cap, 2026-09-17): the entry DECLARES that its rows' contrast against
        # the view's `none` anchor is refused by design with this code — one of g8k_cells.EXPECTED_CONTRAST_REFUSAL_CODES, the
        # reader's own refusal constants, never a free string; absent on every entry written before it existed (byte-identical rows)
        refusal = e.get("expected_contrast_refusal")
        if refusal is not None and str(refusal) not in M.EXPECTED_CONTRAST_REFUSAL_CODES:
            raise SystemExit(f"{path}: steering on {key}/{view}: expected_contrast_refusal {refusal!r} is not one of the reader's refusal "
                             f"codes {list(M.EXPECTED_CONTRAST_REFUSAL_CODES)}")
        out.append({"key": key, **({"view": view} if e.get("view") is not None else {}), "arms": arms,
                    "reason": str(e["reason"]).strip(), "since": _dated(e, path=path, block="steering"),
                    **({"expected_contrast_refusal": str(refusal)} if refusal is not None else {})})
    return out


def _itdoc_entries(doc: dict, out_views: dict, *, path: Path) -> list[dict]:
    """The ``itdoc:`` block (math-eight-arms-extension E3): ``[{key, view, cells, reason, since}]`` — a registered model whose
    ``it`` view the ``views:`` block expresses (the instruct-direct view is where a direct-fit object is served), cells among the
    re-fit's three LoRA arms (``g8k_cells.ITDOC_CELLS``, distinct), a reason and a date. The rows are generated from the it-doc
    ADMISSION RECORD; a (key, it, cell) the record does not name refuses at generation (a re-fit row never exists without its
    record — and its record is what holds it)."""
    out = []
    for e in doc.get("itdoc") or []:
        if not isinstance(e, dict) or not {"key", "view", "cells", "reason", "since"} <= set(e):
            raise SystemExit(f"{path}: every `itdoc:` entry needs key, view, cells, reason, since — got {e!r}")
        key, view = str(e["key"]), str(e["view"])
        if key not in M.MODEL_KEYS:
            raise SystemExit(f"{path}: itdoc on {key!r}: not a registered model key ({M.MODEL_KEYS})")
        if view != M.VIEW_IT:
            raise SystemExit(f"{path}: itdoc on {key}/{view}: an instruct-direct re-fit row sits on the {M.VIEW_IT!r} view, nowhere else")
        if view not in out_views.get(key, []):
            raise SystemExit(f"{path}: itdoc on {key}/{view} names a (key, view) the `views:` block does not express")
        cells = [str(c) for c in (e["cells"] if isinstance(e["cells"], list) else [e["cells"]])]
        bad = [c for c in cells if c not in M.ITDOC_CELLS]
        if not cells or bad or len(set(cells)) != len(cells):
            raise SystemExit(f"{path}: itdoc on {key}: cells must be distinct members of {M.ITDOC_CELLS}, got {cells}")
        if not str(e["reason"]).strip():
            raise SystemExit(f"{path}: itdoc on {key} carries no reason")
        out.append({"key": key, "view": view, "cells": cells, "reason": str(e["reason"]).strip(), "since": _dated(e, path=path, block="itdoc")})
    return out


def load_itdoc_admission(configs_root: Path) -> dict:
    """The it-doc admission record of a tree (``g8k_cells.load_itdoc_admission`` over ``<configs root>/configs/sdfbase/itdoc_admission.json``)."""
    return M.load_itdoc_admission(Path(configs_root) / M.ITDOC_ADMISSION_REL)


def itdoc_admission_path(configs_root: Path) -> Path:
    return Path(configs_root) / M.ITDOC_ADMISSION_REL


def load_admission(configs_root: Path) -> dict[tuple[str, str], dict]:
    """The P8 admission record of a tree (``g8k_cells.load_steer_admission`` over ``<configs root>/configs/sdfbase/steer_admission.json``)."""
    return M.load_steer_admission(Path(configs_root) / M.STEER_ADMISSION_REL)


def admission_path(configs_root: Path) -> Path:
    return Path(configs_root) / M.STEER_ADMISSION_REL


def frozen_input_paths(spec: ViewsSpec, configs_root: Path) -> Optional[tuple[Path, Path]]:
    """The frozen roster / pins copies a spec regenerates from, their sha256 ASSERTED against the spec's (an edited copy
    refuses by name); None when the spec names none (the live registry is the input)."""
    if spec.frozen_inputs is None:
        return None
    roster = Path(configs_root) / spec.frozen_inputs["roster"]
    pins = Path(configs_root) / spec.frozen_inputs["pins"]
    for p, want, what in ((roster, spec.frozen_inputs["roster_sha256"], "roster"), (pins, spec.frozen_inputs["pins_sha256"], "pins")):
        if not p.is_file():
            raise SystemExit(f"frozen_inputs: no frozen {what} at {p}")
        got = sha256_of(p)
        if got != want:
            raise SystemExit(f"frozen_inputs: the frozen {what} {p} hashes to {got}, not the declared {want} — an IMMUTABLE input "
                             "copy was edited; restore it (git show <pin>:<path>) rather than moving the frozen matrix")
    return roster, pins


# --------------------------------------------------------------------------- the inputs
def roster_path(configs_root: Path) -> Path:
    return Path(configs_root) / ROSTER_REL


def pins_path(configs_root: Path) -> Path:
    return Path(configs_root) / PINS_REL


def load_roster(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no roster at {path}")
    from f22_roster import load_roster as _strict_load     # the ONE roster loader: a duplicate mapping key refuses by name and line
    return _strict_load(path)


def load_pins(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no pins at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pin_for(pins: Mapping, key: str, staged_id: str, *, where: str) -> str:
    """The staged object's pinned object.pt sha256 — REQUIRED for every fitted row (a missing pin refuses by name: a dose
    view whose object is not staged where the row would run is not a production row)."""
    entry = (pins.get(key) or {}).get(staged_id)
    sha = entry.get("object_sha256") if isinstance(entry, dict) else entry
    if not isinstance(sha, str) or len(sha) != 64:
        raise SystemExit(f"{where}: pins carry no 64-hex object_sha256 for {key}/{staged_id} — the object is not pinned "
                         "(staged + --pins-from) where this row would be served; refusing to emit it as a production row")
    return sha


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- the cell rule
def cells_of(roster: dict, key: str, view: str) -> list[tuple[str, Optional[str]]]:
    """``[(cell, staged_id | None), ...]`` of (key, view): ``none`` first, the prompted safety baseline second iff the view is
    ``it`` (the instruct-direct lane's matched prompted cell — the roster's `main` views of the two chat-checkpoint models
    already serve it), then the roster's served cells in roster order; a key or view the roster lacks refuses by name.
    EVERY served cell is listed, an unscheduled prompted one included — ``select_rows`` is what routes those to the
    trailing held pass."""
    targets = roster.get("targets") or {}
    if key not in targets:
        raise SystemExit(f"roster names no target {key!r} (targets: {sorted(targets)})")
    views = targets[key].get("views") or {}
    if view not in views:
        raise SystemExit(f"roster names no view {key}/{view} (views of {key}: {sorted(views)})")
    served = served_cells(roster, key, view)
    # a STEERED-WAVE view (s<N>r) carries no `none` anchor row — it measures no unsteered leg (the u0 wave's rows are the baseline);
    # the per-TAG overlay then lists exactly the view's cells and the launcher's standup serves the base
    # route implicitly (g8k_cells.launch_cells_for), never as a measured row
    out: list[tuple[str, Optional[str]]] = [] if steered_wave_view(view, key) else [(M.CELL_NONE, None)]
    if view == M.VIEW_IT:
        # the matched prompted baseline is SECOND on every `it` view — whether the roster lists it
        # (the sampled re-run's lane F made the roster say so, listing it LAST under the view's
        # cells, 2026-09-09) or leaves it implicit; the matrix's row order is a fixed contract
        out.append((M.CELL_PROMPTED, served.get(M.CELL_PROMPTED)))
    for cid, sid in served.items():
        if cid == M.CELL_NONE or (view == M.VIEW_IT and cid == M.CELL_PROMPTED):
            continue
        out.append((cid, sid))
    return out


def harness_cell_id(cell: str) -> str:
    """The harness (manifest) spelling of a TrainTest cell — the harness's OWN function, never a re-typed table."""
    from sisteer.capeval.petri import registry_adapter as ra
    return ra.harness_cell_id(cell)


def standup_spellings(cell: str, *, declared: bool) -> tuple[str, str]:
    """``(standup_cell, standup_harness_cell)`` — the ONE spelling the standup boundary reads for this row (review B-P1.1),
    derived by the contract's own door table ``contracts.cell_spellings``: a DECLARED (header-located) cell is
    ``<cell>@thinking`` to the standup's ``--cell``, the SAME spelling keys its ``--expect-object-sha256`` (the parser wants
    the exact requested spelling, variant included), and the fresh routing table's ``lens_served_positions`` is keyed by the
    variant's harness id (``si_…@thinking``); an undeclared cell reads exactly as before, byte for byte."""
    from sisteer.capeval.tinycap import contracts as C
    sp = C.cell_spellings(cell, declared=declared)
    return sp.traintest, sp.harness


def expected_stamps(*, fitted: bool, declared: bool, arm: str = M.THINKING_ON) -> list[str]:
    """The row's expected ``(cache_salt_regime, resumed_prefill_policy, render_regime)`` — the contract's derivation
    ``contracts.g8k_expected_stamps`` (§2.5.11 through ``expected_stamp_regimes``), never a literal (review B-P1.3); the
    ``arm`` (llama-paper-rows P7) selects the no-switch classes for a ``not_available`` row."""
    from sisteer.capeval.tinycap import contracts as C
    return list(C.g8k_expected_stamps(fitted=fitted, declared=declared, arm=arm))


# --------------------------------------------------------------------------- THE selector
def _row(index: int, *, key: str, view: str, cell: str, staged: Optional[str], pins: Mapping, held_reason: Optional[str],
         where: str, family: str, arm: str, sampling_mode: str, schema2: bool, trigger_threshold: Optional[float],
         wave: Optional[str] = None, own_overlay: bool = False, object_generation: bool = False,
         inspect_seed_policy: Optional[str] = None, inspect_seed_base: Optional[int] = None,
         companions: Optional[list] = None, geometry_class: Optional[str] = None) -> dict:
    """ONE matrix row (the field order is the frozen document's — never reorder). llama-paper-rows (P7): the row's ARM is its
    family's; a ``not_available`` row carries ``thinking_mode`` + ``trigger_threshold`` (schema 2 only), runs the model's
    sampling mode (``card``), carries NO tuple when fitted (in support), cannot be declared (no ON header) and expects the
    no-switch stamps. math-eight-arms-extension E2 (plan P2): an EXTENSION row (``wave`` given) lives under its wave-isolated
    out-root prefix (``controls_<wave>/``) and carries ``wave`` + ``extension`` APPENDED after every published field — a
    published row (``wave`` None) is byte for byte what it was."""
    tag = M.tag_for(key, view)
    prefix = M.out_root_prefix_for(wave or M.WAVE_PUBLISHED)
    fitted = staged is not None
    declared = fitted and cell in M.HEADER_LOCATED_CELLS
    if declared and arm == M.THINKING_NOT_AVAILABLE:
        raise SystemExit(f"{where}: {tag}/{cell} is a header-located cell on the no-switch family {family} — there is no "
                         f"thinking-ON header to declare it under (P7); it is not a row of this instrument")
    pin = pin_for(pins, key, staged, where=f"{where}: {tag}/{cell}") if fitted else None
    refusal = M.expected_refusal(key, declared)
    stamps = expected_stamps(fitted=fitted, declared=declared, arm=arm)
    standup_cell, standup_harness = standup_spellings(cell, declared=declared)
    with_tuple = fitted and arm != M.THINKING_NOT_AVAILABLE       # a no-switch fitted row is IN SUPPORT: no tuple
    return {
        "index": index, "job_id": f"g8k-{key}-{view}-{M.cell_short(cell)}",
        "model_key": key, "view": view, "tag": tag, "cell": cell, "cell_slug": M.cell_slug(cell),
        "harness_cell": harness_cell_id(cell), "standup_cell": standup_cell, "standup_harness_cell": standup_harness,
        "staged_id": staged, "object_sha256": pin,
        "fitted": fitted, "route_kind": M.route_kind(cell),
        "served_positions_expected": ("response" if declared else "all") if fitted else None,
        "declared": declared,
        **({"thinking_mode": arm, "trigger_threshold": float(trigger_threshold)} if schema2 else {}),
        "tuple_name": (M.tuple_name(key, declared=declared, rung=M.CAP, cell=cell) if with_tuple else None),
        "tuple_names": ({str(r): M.tuple_name(key, declared=declared, rung=r, cell=cell) for r in M.RUNGS} if with_tuple else None),
        "sampling_mode": sampling_mode, "sampling": M.sampling_row_of(key),
        "expected_stamps": list(stamps),
        "out_root": f"{prefix}/{tag}/{M.cell_slug(cell)}/{sampling_mode}",
        "rung_out_root": f"{prefix}/{tag}/{M.cell_slug(cell)}/{sampling_mode}__rung{M.RUNG_CAP}",
        "time_limit": M.time_limit_for(key, fitted, cell=cell),
        "family": family, "split_mode_candidate": M.SPLIT_MODE_CANDIDATE[family],
        "reasoning_parser": M.reasoning_parser_of(key),
        "gpu_mem_util": M.gpu_mem_util_of(key),
        "budget_class": M.budget_class(cell),
        "cluster_constraint": M.placement_for(key),
        "status": M.STATUS_HELD if held_reason else M.STATUS_LAUNCH, "held_reason": held_reason,
        "expected_refusal": refusal is not None, "expected_refusal_reason": refusal,
        **({"wave": wave, "extension": True} if wave else {}),
        # E3: a row of a `row_overlays` matrix serves through its OWN pair; an `object_generation` matrix names the
        # generation its staged id carries (None on an unfitted row) — both absent on every row of the published matrix
        **({"own_overlay": True} if own_overlay else {}),
        **({"object_generation": M.object_generation_of(staged)} if object_generation and fitted else {}),
        # The Qwen base seed replicates: the OPT-IN Inspect seed policy + this view's base — APPENDED, present only
        # on a matrix that declares the policy (every other matrix's rows byte-identical); the driver flags ride these two fields
        **({"inspect_seed_policy": inspect_seed_policy, "inspect_seed_base": int(inspect_seed_base)} if inspect_seed_policy else {}),
        # qwen-base-seed-replicates (O12a): the STANDUP COMPANIONS — APPENDED, present (possibly empty) on every row of a matrix that
        # declares `standup_companions`, absent everywhere else; `companions` = [(cell, harness, staged_id, pin), ...]
        **({M.STANDUP_COMPANION_CELLS: [c for c, _h, _sid, _pin in companions],
            M.STANDUP_COMPANION_HARNESS_CELLS: [h for _c, h, _sid, _pin in companions],
            M.STANDUP_COMPANION_STAGED_IDS: {c: sid for c, _h, sid, _pin in companions},
            M.STANDUP_COMPANION_PINS: {c: pin for c, _h, _sid, pin in companions}} if companions is not None else {}),
        # The matrix-declared GEOMETRY CLASS — APPENDED LAST, present only on a matrix that
        # declares it (every other matrix's rows byte-identical); geometry_for_cell selects the common entries by it on every row
        **({M.GEOMETRY_CLASS_KEY: geometry_class} if geometry_class is not None else {}),
    }


def _external_row(index: int, *, key: str, cell: str, entry: Mapping, where: str, family: str, arm: str, sampling_mode: str,
                  schema2: bool, trigger_threshold: Optional[float], held_reason: Optional[str] = None,
                  view: str = M.VIEW_MAIN, wave: str = M.WAVE_STEER, own_overlay: bool = False,
                  expected_contrast_refusal: Optional[str] = None) -> dict:
    """ONE steering row (math-eight-arms-extension P8): the EXTERNAL object the ADMISSION RECORD names for
    (model key, cell) on the PANEL view ``main`` — fitted, undeclared, positions ``all``, pinned by the record's object sha256
    (never the full22 pin file), the lane's bank view per cluster role RELATIVE to the shared root, the lane's own view name
    as ``bank_view``, the VERBATIM selection label, the grid-limited / boundary-pick flags, the producer's serving-config hash
    (NOT durable; every leg records the hash of the config IT serves), under the wave-isolated ``controls_steer/`` prefix. A
    SWITCH model's row runs under the rung's ``_steer_<cap>`` tuple; the no-switch family's is IN SUPPORT (no tuple). The
    launcher (``g8k_bank_view``), the standup and the driver hold the served bytes to the pin. A ``holds:`` entry naming
    (key, main, cell) makes the row ``held`` with its reason (E2-12: a staged record the driver's provenance mirror refuses) —
    admitted and pinned, never launched until the hold is lifted."""
    tag = M.tag_for(key, view)
    stamps = expected_stamps(fitted=True, declared=False, arm=arm)
    standup_cell, standup_harness = standup_spellings(cell, declared=False)
    with_tuple = arm != M.THINKING_NOT_AVAILABLE
    prefix = M.out_root_prefix_for(wave)
    placement = (M.PLACEMENT_PRIMARY if (M.placement_for(key) == M.PLACEMENT_PRIMARY or entry["cluster_constraint"] == M.PLACEMENT_PRIMARY)
                 else M.PLACEMENT_ANY)
    row = {
        "index": index, "job_id": f"g8k-{key}-{view}-{M.cell_short(cell)}",
        "model_key": key, "view": view, "tag": tag, "cell": cell, "cell_slug": M.cell_slug(cell),
        "harness_cell": harness_cell_id(cell), "standup_cell": standup_cell, "standup_harness_cell": standup_harness,
        "staged_id": str(entry["staged_id"]), "object_sha256": str(entry["object_sha256"]),
        "fitted": True, "route_kind": M.route_kind(cell),
        "served_positions_expected": "all",
        "declared": False,
        **({"thinking_mode": arm, "trigger_threshold": float(trigger_threshold)} if schema2 else {}),
        # PER CELL (arditi-allpos-cap): the rung's arms keep its `_steer_<cap>` tuple byte-identical; the all-positions
        # arm runs its OWN `_steer_allpos_<cap>` one, because an OOS tuple AUTHORISES a described intervention and this is a
        # different one — prefill+decode, a raw-coefficient dose, an over-budget user override (g8k_cells.external_tuple_name)
        "tuple_name": (M.external_tuple_name(key, cell, rung=M.CAP) if with_tuple else None),
        "tuple_names": ({str(r): M.external_tuple_name(key, cell, rung=r) for r in M.RUNGS} if with_tuple else None),
        "sampling_mode": sampling_mode, "sampling": M.sampling_row_of(key),
        "expected_stamps": list(stamps),
        "out_root": f"{prefix}/{tag}/{M.cell_slug(cell)}/{sampling_mode}",
        "rung_out_root": f"{prefix}/{tag}/{M.cell_slug(cell)}/{sampling_mode}__rung{M.RUNG_CAP}",
        "time_limit": M.time_limit_for(key, True, cell=cell),
        "family": family, "split_mode_candidate": M.SPLIT_MODE_CANDIDATE[family],
        "reasoning_parser": M.reasoning_parser_of(key),
        "gpu_mem_util": M.gpu_mem_util_of(key),
        "budget_class": M.budget_class(cell),
        "cluster_constraint": placement,
        "status": M.STATUS_HELD if held_reason else M.STATUS_LAUNCH, "held_reason": held_reason,
        "expected_refusal": False, "expected_refusal_reason": None,
        "wave": wave, "extension": True,
        # the steering row's OWN facts, straight from the admission record (P8): recorded on the ROW because every table, figure
        # and provenance line reads them from here
        "object_source": M.OBJECT_SOURCE_EXTERNAL, "lane_root": str(entry["lane_root"]),
        "bank_view": str(entry["view"]), "bank_view_rel": dict(entry["bank_view_rel"]), "bank_cell_dir": str(entry["bank_cell_dir"]),
        "selection_rule": str(entry["selection_rule"]), "selection_label": str(entry["selection_label"]),
        "grid_limited": bool(entry["grid_limited"]),
        # E2-B: the canonical ladder's selection outcome — `chosen` under (a)+(b)+(c), or `a4_fallback` (nemotron:
        # condition (a) failed at the rung the selection wanted) — beside the selection label; null on an Arditi row (the battery's pick)
        "selection_outcome": entry.get("selection_outcome"),
        "rung": entry.get("rung"), "lambda": entry.get("lambda"), "alpha": entry.get("alpha"), "served_layer": entry.get("served_layer"),
        "producer": dict(entry.get("producer") or {}), "ceiling_20pp": entry.get("ceiling_20pp"),
        "dose_evidence": dict(entry.get("dose_evidence") or {}), "attestation_expect": dict(entry.get("attestation_expect") or {}),
        "boundary_pick": (entry.get("boundary_pick") or {}).get("status"),
        "alternative_not_pinned": entry.get("alternative_not_pinned"),
        "admission_notes": list(entry.get("notes") or []),
        # arditi-allpos-cap (2026-09-17): the record's SELECTION and FIT PROVENANCE blocks ride the row when the entry carries
        # them — the analysis, every table and the paper's provenance line read them from the ROW (the same reason
        # `selection_label` / `dose_evidence` already do), and for the all-positions arm `external_row_problem` REQUIRES them:
        # its dose is a raw coefficient, its point a declared user override over the rule's budget, its ladder single-trial, and
        # none of that is legible from a label alone. Absent on an entry that carries neither, so every row generated before this
        # existed is byte-identical. APPENDED last: the field order of the frozen documents never moves.
        **({"selection": json.loads(json.dumps(entry["selection"]))} if entry.get("selection") else {}),
        **({"fit_provenance": json.loads(json.dumps(entry["fit_provenance"]))} if entry.get("fit_provenance") else {}),
        **({"own_overlay": True} if own_overlay else {}),
        # the DECLARED expected refusal of this row's CONTRAST (the steering entry's `expected_contrast_refusal`, one of
        # g8k_cells.EXPECTED_CONTRAST_REFUSAL_CODES): the reader files the contrast as refused by design iff its refusal derives
        # exactly this code. APPENDED last; absent on every row generated before it existed
        **({"expected_contrast_refusal": str(expected_contrast_refusal)} if expected_contrast_refusal is not None else {}),
    }
    problem = M.external_row_problem(row)
    if problem:
        raise SystemExit(f"{where}: {tag}/{cell}: {problem}")
    return row


def _itdoc_row(index: int, *, key: str, cell: str, entry: Mapping, where: str, family: str, arm: str, sampling_mode: str,
               schema2: bool, trigger_threshold: Optional[float], job_id: Optional[str] = None, pins: Optional[Mapping] = None) -> dict:
    """ONE it-doc row (math-eight-arms-extension E3): the instruct-direct batch-64 re-fit object the ADMISSION RECORD names for
    (lane, key, it, cell) — fitted, undeclared, positions ``all``, pinned by the record's object sha256 (null while unverified: never
    a placeholder), its RULED staged id, the bank binding the record carries (null until the lane stages the object), the object
    kind and the sidecar that proves it, the superseded direct-fit object (when this row replaces a held one), the known stale
    objects the bank door refuses by label, the pair's QUALIFICATION verbatim, under the wave-isolated ``controls_itdoc/`` prefix.
    ``status`` is the RECORD's: ``launch`` iff the entry carries pin + staged id + a complete bank + no hold reason, else ``held``
    with the record's reason. The it view's all-position tuple names it (by-name authorisation is view-blind; the bytes are bound
    per view by the pin at the bank door, the standup and the driver)."""
    tag = M.tag_for(key, M.VIEW_IT)
    stamps = expected_stamps(fitted=True, declared=False, arm=arm)
    standup_cell, standup_harness = standup_spellings(cell, declared=False)
    with_tuple = arm != M.THINKING_NOT_AVAILABLE
    prefix = M.out_root_prefix_for(M.WAVE_ITDOC)
    bank = entry.get("bank") or {}
    launch = B.entry_launchable(entry)
    # a LAUNCHABLE it-doc row's (target, staged id) must be pinned in the matrix's (frozen) pin file with the record's sha —
    # the gate is the MERGED pins, never a message; a pin file entry that disagrees with the record refuses on a held row too
    if pins is not None:
        pinned = (pins.get(key) or {}).get(str(entry["staged_id"])) if isinstance(pins, Mapping) else None
        pinned_sha = pinned.get("object_sha256") if isinstance(pinned, Mapping) else pinned
        if launch and pinned_sha is None:
            raise SystemExit(f"{where}: {key}/it/{cell}: the record binds a bank for staged id {entry['staged_id']!r} but the pin file names no "
                             "(target, staged id) entry for it — a served object is pinned on main before its row launches (D89)")
        if pinned_sha is not None and pinned_sha != entry.get("object_sha256"):
            raise SystemExit(f"{where}: {key}/it/{cell}: the pin file's {str(pinned_sha)[:12]}… != the record's {str(entry.get('object_sha256'))[:12]}… for "
                             f"staged id {entry['staged_id']!r} — two pins for one object; the record and the pin file disagree")
    placement = (M.PLACEMENT_PRIMARY if (M.placement_for(key) == M.PLACEMENT_PRIMARY or bank.get("cluster_constraint") == M.PLACEMENT_PRIMARY)
                 else M.PLACEMENT_ANY)
    row = {
        "index": index, "job_id": job_id or f"g8k-{key}-{M.VIEW_IT}-{M.cell_short(cell)}",
        "model_key": key, "view": M.VIEW_IT, "tag": tag, "cell": cell, "cell_slug": M.cell_slug(cell),
        "harness_cell": harness_cell_id(cell), "standup_cell": standup_cell, "standup_harness_cell": standup_harness,
        "staged_id": str(entry["staged_id"]), "object_sha256": entry.get("object_sha256"),
        "fitted": True, "route_kind": M.route_kind(cell),
        "served_positions_expected": "all",
        "declared": False,
        **({"thinking_mode": arm, "trigger_threshold": float(trigger_threshold)} if schema2 else {}),
        "tuple_name": (M.tuple_name(key, declared=False, rung=M.CAP, cell=cell) if with_tuple else None),
        "tuple_names": ({str(r): M.tuple_name(key, declared=False, rung=r, cell=cell) for r in M.RUNGS} if with_tuple else None),
        "sampling_mode": sampling_mode, "sampling": M.sampling_row_of(key),
        "expected_stamps": list(stamps),
        "out_root": f"{prefix}/{tag}/{M.cell_slug(cell)}/{sampling_mode}",
        "rung_out_root": f"{prefix}/{tag}/{M.cell_slug(cell)}/{sampling_mode}__rung{M.RUNG_CAP}",
        "time_limit": M.time_limit_for(key, True, cell=cell),
        "family": family, "split_mode_candidate": M.SPLIT_MODE_CANDIDATE[family],
        "reasoning_parser": M.reasoning_parser_of(key),
        "gpu_mem_util": M.gpu_mem_util_of(key),
        "budget_class": M.budget_class(cell),
        "cluster_constraint": placement,
        "status": M.STATUS_LAUNCH if launch else M.STATUS_HELD,
        "held_reason": None if launch else str(entry.get("hold_reason") or "").strip() or None,
        "expected_refusal": False, "expected_refusal_reason": None,
        "wave": M.WAVE_ITDOC, "extension": True,
        # the it-doc row's OWN facts, straight from the admission record (E3): recorded on the ROW because the bank door, every
        # table, figure and provenance line reads them from here
        "object_source": M.OBJECT_SOURCE_ITDOC, "lane": str(entry["lane"]),
        "object_kind": str(entry["object_kind"]), "bank_sidecar": B.SIDECAR_BY_KIND[str(entry["object_kind"])],
        "bank_view": bank.get("view"), "bank_view_rel": (dict(bank["bank_view_rel"]) if bank.get("bank_view_rel") else None),
        "bank_cell_dir": bank.get("bank_cell_dir"),
        "parents_expected": (dict(entry["parents_expected"]) if entry.get("parents_expected") else None),
        "superseded": (dict(entry["superseded"]) if entry.get("superseded") else None),
        "stale_objects": dict(entry.get("stale_objects") or {}),
        "qualification": entry.get("qualification"),
        "qualification_pattern": entry.get("qualification_pattern"),
        "fit_record": (dict(entry["fit_record"]) if entry.get("fit_record") else None),
        "admission_notes": list(entry.get("notes") or []),
        # an it-doc row always serves through its OWN pair and is the batch-64 lane's object by construction
        "own_overlay": True, "object_generation": M.OBJECT_GENERATION_BATCH64,
    }
    problem = M.itdoc_row_problem(row)
    if problem:
        raise SystemExit(f"{where}: {tag}/{cell}: {problem}")
    return row


def select_rows(roster: dict, pins: Mapping, views: Mapping[str, Iterable[str]], *, holds: Iterable[dict] = (),
                exclude: Iterable[dict] = (), schema: str = M.MATRIX_SCHEMA, trigger_threshold: Optional[float] = None,
                where: str = "select_rows", extensions: Iterable[dict] = (), steering: Iterable[dict] = (),
                admission: Optional[Mapping[tuple[str, str], Mapping]] = None, itdoc: Iterable[dict] = (),
                itdoc_admission: Optional[Mapping] = None, row_overlays: bool = False, object_generation: bool = False,
                wave: Optional[str] = None, inspect_seed_policy: Optional[str] = None,
                inspect_seed_base_by_view: Optional[Mapping[str, int]] = None,
                standup_companions: Optional[Mapping[str, str]] = None, geometry_class: Optional[str] = None) -> list[dict]:
    """The matrix rows for ``views`` (an ordered ``{key: [view, ...]}``) over ``roster`` + ``pins``: per (key, view), ``cells_of``
    in order; every fitted cell carries its staged id and pin; the header-located cells (``cells.HEADER_LOCATED_CELLS``)
    are DECLARED and expect ``response``, every other fitted cell expects ``all``; a hold stamps ``status: held``;
    ``expected_refusal`` is False on every row since 2026-09-10 (gemma's header-located cells are servable thinking-ON on the
    declared path) — the field stays, lane A's reader honours it as a tested capability;
    ``standup_cell`` / ``standup_harness_cell`` are the declared-or-plain spellings the standup boundary reads (B-P1.1) and
    ``expected_stamps`` the contract's triple for the row's class (B-P1.3), both by the contract's own functions.
    ``tuple_name`` is the launch row's (the ``_8192`` name); ``tuple_names`` both rungs'. Index = position in the emitted
    list. llama-paper-rows (P7): the row's ARM is its family's (``M.THINKING_MODE_BY_FAMILY``); a ``not_available`` row needs
    a schema-2 matrix (it carries ``thinking_mode``), runs the model's sampling mode (``card``), carries NO tuple when fitted
    (in support), cannot be declared (no ON header), and expects the no-switch stamps; an ``exclude`` entry drops the cell
    from the matrix outright (fallback (B); a schema-2 block). A prompted cell the roster declares that the matrix's
    SCHEMA does not schedule (``g8k_cells.scheduled_prompted_cells(schema)`` — the published matrix the safety baseline
    only, the Llama matrix both prompted baselines) is emitted AFTER every scheduled row (views order, then roster order),
    ``status: held`` with ``g8k_cells.unscheduled_held_reason``, so the scheduled rows' indices and the launched set are
    exactly what they are without it.

    math-eight-arms-extension E2 (plan P2 / P8): after those, the ``extensions`` rows (wave ``ext``, one launch row per
    (key, view, prompted cell), in list order — a cell an extension SCHEDULES is thereby scheduled: it never becomes a
    trailing held row, whether or not the roster declares it) and then the ``steering`` rows (wave ``steer``, one launch
    row per (key, arm) on the panel view ``main``, built from the P8 ``admission`` record — a (key, arm) the record does not
    admit refuses by name). Every appended row's index is its position; nothing before it moves.

    math-eight-arms-extension E3: the ``itdoc`` rows (wave ``itdoc``) are generated from the it-doc ADMISSION RECORD
    (``itdoc_admission``): a (key, it, cell) the BASE PASS emits (the held direct-fit ``constv2-mix-full__lora_r4_ce`` row) is
    REPLACED IN PLACE — same index, TAG and job id (the slot the campaign held for the re-fit; the superseded object recorded on the
    row); every other it-doc cell is appended after the steering rows. A (key, it, cell) the record does not name refuses by name."""
    spec = ViewsSpec(views={k: list(v) for k, v in views.items()}, holds=list(holds), exclude=list(exclude), schema=schema,
                     extensions=list(extensions), steering=list(steering), itdoc=list(itdoc), row_overlays=row_overlays,
                     object_generation=object_generation, wave=wave, inspect_seed_policy=inspect_seed_policy,
                     inspect_seed_base_by_view=(dict(inspect_seed_base_by_view) if inspect_seed_base_by_view else None),
                     standup_companions=(dict(standup_companions) if standup_companions else None), geometry_class=geometry_class)
    if spec.geometry_class is not None:
        # qwen-seedrep-native-battery: a schema-2 fact naming a registered class whose family scope every expressed model shares
        if not schema2_of(schema):
            raise SystemExit(f"{where}: {M.GEOMETRY_CLASS_KEY} is a {M.MATRIX_SCHEMA_2} fact")
        if spec.geometry_class not in M.GEOMETRY_CLASSES:
            raise SystemExit(f"{where}: {M.GEOMETRY_CLASS_KEY} must be one of {sorted(M.GEOMETRY_CLASSES)}, got {spec.geometry_class!r}")
        for key in spec.views:
            if M.longsi_scope(key) != M.GEOMETRY_CLASS_SCOPE[spec.geometry_class]:
                raise SystemExit(f"{where}: {M.GEOMETRY_CLASS_KEY} {spec.geometry_class!r} is scoped to the {M.GEOMETRY_CLASS_SCOPE[spec.geometry_class]!r} "
                                 f"families; {key!r} is {M.longsi_scope(key)!r}")
    if spec.standup_companions is not None:
        # qwen-base-seed-replicates (O12a): a schema-2 + row_overlays fact naming one steered cell (the loader's shape, re-checked here)
        if not schema2_of(schema) or not row_overlays:
            raise SystemExit(f"{where}: standup_companions is a {M.MATRIX_SCHEMA_2} + row_overlays fact")
        ccell = spec.standup_companions.get("cell")
        if not isinstance(ccell, str) or not ccell or ccell == M.CELL_NONE or M.is_prompted_cell(ccell) or not str(spec.standup_companions.get("reason") or "").strip():
            raise SystemExit(f"{where}: standup_companions must be {{cell: <steered cell>, reason: <text>}}, got {spec.standup_companions!r}")
    if spec.inspect_seed_policy is not None:
        # qwen-base-seed-replicates: the policy is a schema-2 fact with ONE base per expressed view (the views loader pins each
        # base to eval_seed; this seam re-checks the shape for a caller that bypasses the loader)
        if not schema2_of(schema):
            raise SystemExit(f"{where}: inspect_seed_policy is a {M.MATRIX_SCHEMA_2} fact")
        if spec.inspect_seed_policy != M.INSPECT_SEED_POLICY_REPLICATE:
            raise SystemExit(f"{where}: inspect_seed_policy must be {M.INSPECT_SEED_POLICY_REPLICATE!r}, got {spec.inspect_seed_policy!r}")
        expressed = {v for vs in spec.views.values() for v in vs}
        if not spec.inspect_seed_base_by_view or set(spec.inspect_seed_base_by_view) != expressed:
            raise SystemExit(f"{where}: inspect_seed_policy needs inspect_seed_base_by_view with exactly one base per expressed view {sorted(expressed)}")
    elif inspect_seed_base_by_view:
        raise SystemExit(f"{where}: inspect_seed_base_by_view without inspect_seed_policy — a base with nowhere to go")

    def _seed_kw(view: str) -> dict:
        out = {} if spec.inspect_seed_policy is None else {"inspect_seed_policy": spec.inspect_seed_policy, "inspect_seed_base": spec.inspect_seed_base_by_view[view]}
        if spec.geometry_class is not None:           # every row of a declaring matrix carries the class (the loader holds them to it)
            out["geometry_class"] = spec.geometry_class
        return out

    def _companion_kw(key: str, view: str, cell: str, roster_cells: list) -> tuple[dict, Optional[str]]:
        """``(row kwargs, hold reason)``: the view's companion resolved through the SAME `cells_of` enumeration the row came from —
        `(cell, staged_id)` of the companion, its pin from the frozen pin file, its harness id; the companion's own row carries none;
        a view that does not stage the companion HOLDS the row by name (the pre-staging freeze) with empty companion fields."""
        if spec.standup_companions is None:
            return {}, None
        ccell = spec.standup_companions["cell"]
        if cell == ccell:
            return {"companions": []}, None
        staged = dict(roster_cells).get(ccell)
        if staged is None:
            return {"companions": []}, (f"{M.COMPANION_HELD_MARK}: `{ccell}` is not a staged cell of {key}/{view} (the roster's `cells` do not name it — "
                                        f"staging pending); regenerate the matrix from the staging freeze, which lifts this hold")
        pin = pin_for(pins, key, staged, where=f"{where}: {M.tag_for(key, view)}/{cell} companion {ccell}")
        return {"companions": [(ccell, harness_cell_id(ccell), staged, pin)]}, None
    extended = {(e["key"], e["view"], c) for e in spec.extensions for c in e["cells"]}
    if spec.steering and admission is None:
        raise SystemExit(f"{where}: `steering` rows are generated from the P8 admission record — pass admission= (g8k_views.load_admission)")
    if spec.itdoc and itdoc_admission is None:
        raise SystemExit(f"{where}: `itdoc` rows are generated from the it-doc admission record — pass itdoc_admission= (g8k_views.load_itdoc_admission)")
    itdoc_cells = {(e["key"], e["view"], c) for e in spec.itdoc for c in e["cells"]}

    def _itdoc_entry(key: str, cell: str) -> Mapping:
        lane = itdoc_admission["lane"]
        entry = B.entry_for(itdoc_admission, lane, key, M.VIEW_IT, cell)
        if entry is None:
            raise SystemExit(f"{where}: itdoc {key}/{M.VIEW_IT}/{cell}: the it-doc admission record names no entry for "
                             f"{B.tuple_label(lane, key, M.VIEW_IT, cell)} — a re-fit row never exists without its record")
        return entry
    replaced: set[tuple[str, str, str]] = set()
    if schema not in M.MATRIX_SCHEMAS:
        raise SystemExit(f"{where}: schema {schema!r} is not one of {M.MATRIX_SCHEMAS}")
    schema2 = schema == M.MATRIX_SCHEMA_2
    if spec.exclude and not schema2:
        raise SystemExit(f"{where}: `exclude` is a {M.MATRIX_SCHEMA_2} block; a schema-1 matrix holds a cell instead")
    if schema2 and trigger_threshold is None:
        raise SystemExit(f"{where}: a {M.MATRIX_SCHEMA_2} matrix declares its ladder trigger (trigger_threshold)")
    if not schema2 and trigger_threshold is not None:
        raise SystemExit(f"{where}: trigger_threshold is a {M.MATRIX_SCHEMA_2} fact; the published matrix reads the adapter's {M.TRIGGER_THRESHOLD}")
    rows: list[dict] = []
    trailing: list[tuple[str, str, str, Optional[str]]] = []
    for key, vlist in spec.views.items():
        family = M.family_of(key)
        if family is None:
            raise SystemExit(f"{where}: {key!r} is not a registered model key ({M.all_model_keys()}) — g8k_cells.FAMILY_BY_MODEL names no family "
                             "and the models table (reproduce/models.yaml) names no phase views for it")
        arm = M.THINKING_MODE_BY_FAMILY[family]
        if arm == M.THINKING_NOT_AVAILABLE and not schema2:
            raise SystemExit(f"{where}: {key} ({family}) runs the no-switch arm {arm!r}, which only a {M.MATRIX_SCHEMA_2} matrix "
                             f"expresses (`thinking_mode` on every row; its own wave root) — never the published schema-1 matrix (P16)")
        sampling_mode = M.sampling_mode_of(key)
        for view in vlist:
            roster_cells = cells_of(roster, key, view)
            for cell, staged in roster_cells:
                if spec.excluded(key, view, cell) is not None:
                    continue
                if (key, view, cell) in extended:
                    continue                    # E2: an extension SCHEDULES the cell — it is emitted in the extension pass below
                if M.is_unscheduled_prompted(cell, schema):
                    trailing.append((key, view, cell, staged))
                    continue
                if (key, view, cell) in itdoc_cells:
                    # E3: the re-fit REPLACES the direct-fit row IN PLACE — the slot the campaign held for it (index, TAG, job id kept)
                    rows.append(_itdoc_row(len(rows), key=key, cell=cell, entry=_itdoc_entry(key, cell), where=where, family=family, arm=arm, pins=pins,
                                           sampling_mode=sampling_mode, schema2=schema2, trigger_threshold=trigger_threshold,
                                           job_id=f"g8k-{key}-{view}-{M.cell_short(cell)}"))
                    replaced.add((key, view, cell))
                    continue
                companion_kw, companion_hold = _companion_kw(key, view, cell, roster_cells)
                rows.append(_row(len(rows), key=key, view=view, cell=cell, staged=staged, pins=pins, own_overlay=spec.row_overlays, object_generation=spec.object_generation, wave=spec.wave,
                                 held_reason=spec.hold_for(key, view, cell) or companion_hold, where=where, family=family, arm=arm,
                                 sampling_mode=sampling_mode, schema2=schema2, trigger_threshold=trigger_threshold, **_seed_kw(view), **companion_kw))
    for key, view, cell, staged in trailing:
        family = M.family_of(key)
        rows.append(_row(len(rows), key=key, view=view, cell=cell, staged=staged, pins=pins, own_overlay=spec.row_overlays, object_generation=spec.object_generation, wave=spec.wave,
                         held_reason=M.unscheduled_held_reason(cell, schema), where=where, family=family,
                         arm=M.THINKING_MODE_BY_FAMILY[family], sampling_mode=M.sampling_mode_of(key), schema2=schema2,
                         trigger_threshold=trigger_threshold, **_seed_kw(view), **({"companions": []} if spec.standup_companions else {})))
    # E2 (plan P2): the EXTENSION rows — prompted routes, no pin, no roster entry — wave `ext`, appended in list order; a cell
    # already emitted for its (key, view) refuses (the matrix names a (key, view, cell) once)
    seen = {(r["model_key"], r["view"], r["cell"]) for r in rows}
    # E3: the IT-DOC rows the base pass did not replace in place — appended in list order (the eight-arm order within an entry), BEFORE
    # the extension and steering rows: a later extension APPENDS behind them (new rows append; a held row's index never moves)
    for e in spec.itdoc:
        key = e["key"]
        family = M.family_of(key)
        arm = M.THINKING_MODE_BY_FAMILY[family]
        for cell in e["cells"]:
            if (key, M.VIEW_IT, cell) in replaced:
                continue
            if (key, M.VIEW_IT, cell) in seen:
                raise SystemExit(f"{where}: itdoc {key}/{M.VIEW_IT}/{cell} is already a row of this matrix")
            rows.append(_itdoc_row(len(rows), key=key, cell=cell, entry=_itdoc_entry(key, cell), where=where, family=family, arm=arm, pins=pins,
                                   sampling_mode=M.sampling_mode_of(key), schema2=schema2, trigger_threshold=trigger_threshold))
            seen.add((key, M.VIEW_IT, cell))
    for e in spec.extensions:
        key, view = e["key"], e["view"]
        family = M.family_of(key)
        if family is None:
            raise SystemExit(f"{where}: extension on {key!r}: not a registered model key ({M.all_model_keys()})")
        arm = M.THINKING_MODE_BY_FAMILY[family]
        if arm == M.THINKING_NOT_AVAILABLE and not schema2:
            raise SystemExit(f"{where}: extension on {key} ({family}) runs the no-switch arm, which only a {M.MATRIX_SCHEMA_2} matrix expresses (P16)")
        for cell in e["cells"]:
            if (key, view, cell) in seen:
                raise SystemExit(f"{where}: extension {key}/{view}/{cell} is already a row of this matrix — an extension adds a cell the base pass does not emit")
            harness_cell_id(cell)               # the harness's own grammar: an unknown prompted id refuses by name here
            # E3: in a matrix that declares its OWN wave and `row_overlays` (the instruct-direct
            # matrix), an extension row is a row of THAT wave under its root and serves through its OWN pair (the published `it` TAG
            # overlay never lists the constitution SI); a matrix without a `wave:` keeps the E2 shape — wave `ext`, the TAG overlay.
            # Since 2026-09-17, on a matrix that declares `standup_companions`, an extension row is a
            # LAUNCH row of a view engine like every base-pass row, so it carries the view's companion through the SAME `_companion_kw`
            # resolution (the loader holds every launch row to exactly the document's companion; `[]` there refused the row) — a view
            # that does not stage the companion HOLDS the prompted row by name too. Every matrix without the block is byte-identical.
            ext_companion_kw, ext_companion_hold = _companion_kw(key, view, cell, cells_of(roster, key, view))
            rows.append(_row(len(rows), key=key, view=view, cell=cell, staged=None, pins=pins, held_reason=spec.hold_for(key, view, cell) or ext_companion_hold,
                             where=where, family=family, arm=arm, sampling_mode=M.sampling_mode_of(key), schema2=schema2,
                             trigger_threshold=trigger_threshold, wave=(spec.wave or M.WAVE_EXT), own_overlay=spec.row_overlays,
                             **_seed_kw(view), **ext_companion_kw))
            seen.add((key, view, cell))
    # E2 (plan P8): the STEERING rows from the admission record — wave `steer`, the panel view `main`
    if spec.inspect_seed_policy is not None and (spec.steering or spec.itdoc):
        raise SystemExit(f"{where}: inspect_seed_policy is defined for the base-pass and extension rows only — this lane schedules no "
                         "steering / itdoc rows (their row builders carry no seed policy)")
    if spec.standup_companions is not None and (spec.steering or spec.itdoc):
        raise SystemExit(f"{where}: standup_companions is defined for the base-pass and extension rows only — this lane schedules no "
                         "steering / itdoc rows (their row builders carry no companion)")
    if spec.geometry_class is not None and (spec.steering or spec.itdoc):
        raise SystemExit(f"{where}: {M.GEOMETRY_CLASS_KEY} is defined for the base-pass and extension rows only — this lane schedules no "
                         "steering / itdoc rows (their row builders carry no geometry class)")
    for e in spec.steering:
        key = e["key"]
        family = M.family_of(key)
        arm = M.THINKING_MODE_BY_FAMILY[family]
        # the entry's view (arditi-allpos-cap): `main` unless the entry named another, and `_steering_entries` has already held
        # it to the ARM's declared view and to a view the model expresses; a matrix declaring its own wave puts its steering rows
        # in that wave (its own out-root prefix under its own root), and a `row_overlays` matrix gives them their own pair
        view = str(e.get("view") or M.VIEW_MAIN)
        if arm == M.THINKING_NOT_AVAILABLE and not schema2:
            raise SystemExit(f"{where}: steering on {key} ({family}) runs the no-switch arm, which only a {M.MATRIX_SCHEMA_2} matrix expresses (P16)")
        for cell in e["arms"]:
            entry = (admission or {}).get((key, cell))
            if entry is None:
                raise SystemExit(f"{where}: steering {key}/{cell}: the admission record names no entry for (model key, cell id) — a steering "
                                 "row never exists without its P8 record (object, dose, selection, pin)")
            if (key, view, cell) in seen:
                raise SystemExit(f"{where}: steering {key}/{view}/{cell} is already a row of this matrix")
            rows.append(_external_row(len(rows), key=key, cell=cell, entry=entry, where=where, family=family, arm=arm,
                                      sampling_mode=M.sampling_mode_of(key), schema2=schema2, trigger_threshold=trigger_threshold,
                                      held_reason=spec.hold_for(key, view, cell), view=view,
                                      wave=(spec.wave or M.WAVE_STEER), own_overlay=spec.row_overlays,
                                      expected_contrast_refusal=e.get("expected_contrast_refusal")))
            seen.add((key, view, cell))
    return rows


def matrix_document(rows: list[dict], *, spec: ViewsSpec, roster_file: Path, pins_file: Path,
                    views_file: Optional[Path], configs_root: Path, roster_rel: Optional[str] = None,
                    pins_rel: Optional[str] = None, admission_file: Optional[Path] = None,
                    itdoc_admission_file: Optional[Path] = None) -> dict:
    """The frozen matrix's document: the rows + the provenance of what they were derived from (paths RELATIVE to the
    configs root and their sha256 — a regenerated matrix over the same inputs is byte-identical: no timestamps). A
    regeneration from FROZEN input copies (P16) passes the copies as ``roster_file`` / ``pins_file`` and the published
    paths as ``roster_rel`` / ``pins_rel``: the document records the pin's tree, the hashes are the copies' (equal by
    construction, asserted by ``frozen_input_paths``). Schema 2 adds ``wave_root_rel`` and ``exclude``; schema 1 nothing."""
    def rel(p):
        return str(Path(p).resolve().relative_to(Path(configs_root).resolve())) if p is not None else None
    schema2 = spec.schema == M.MATRIX_SCHEMA_2
    return {
        "schema": spec.schema,
        **({"wave_root_rel": spec.wave_root_rel, "trigger_threshold": spec.trigger_threshold} if schema2 else {}),
        "generated_by": "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py (g8k_views.select_rows)",
        "views_spec": spec.spec_string(),
        "views": spec.views,
        "holds": spec.holds,
        **({"exclude": spec.exclude} if schema2 else {}),
        # E2: the appended waves' blocks and — iff steering rows exist — the admission record they were generated from
        # (its path relative to the configs root + sha256: a regeneration over the same record is byte-identical)
        **({"extensions": spec.extensions} if spec.extensions else {}),
        **({"steering": spec.steering} if spec.steering else {}),
        **({"itdoc": spec.itdoc} if spec.itdoc else {}),
        **({"row_overlays": True} if spec.row_overlays else {}),
        **({"object_generation": True} if spec.object_generation else {}),
        **({"wave": spec.wave} if spec.wave else {}),
        # qwen-base-seed-replicates: the OPT-IN Inspect seed policy and its per-view bases (schema 2; absent on every other matrix)
        **({"inspect_seed_policy": spec.inspect_seed_policy, "inspect_seed_base_by_view": dict(spec.inspect_seed_base_by_view)}
           if spec.inspect_seed_policy else {}),
        # qwen-base-seed-replicates (O12a): the standup companion every row of this matrix registers (absent on every other matrix)
        **({M.STANDUP_COMPANIONS_KEY: dict(spec.standup_companions)} if spec.standup_companions else {}),
        # the common geometry class every row of this matrix serves at (absent elsewhere)
        **({M.GEOMETRY_CLASS_KEY: spec.geometry_class} if spec.geometry_class else {}),
        "source": {"views_yaml": rel(views_file), "roster": roster_rel or rel(roster_file), "roster_sha256": sha256_of(roster_file),
                   "pins": pins_rel or rel(pins_file), "pins_sha256": sha256_of(pins_file),
                   **({"steer_admission": rel(admission_file), "steer_admission_sha256": sha256_of(admission_file)}
                      if spec.steering and admission_file is not None else {}),
                   **({"itdoc_admission": rel(itdoc_admission_file), "itdoc_admission_sha256": sha256_of(itdoc_admission_file)}
                      if spec.itdoc and itdoc_admission_file is not None else {})},
        "counts": M.counts(rows),
        "rows": rows,
    }


def render_matrix(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"
