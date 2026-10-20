#!/usr/bin/env python
"""The Qwen base seed-replicate lane: EXPLICIT object selection on EVERY producer — the view's ADMITTED cells.

No producer of the lane is ever launched with an empty or `all` cell selector — those select `none` too (target_sampled /
target_static: an empty 5th / 6th positional = every explicit_cells entry INCLUDING `none`; petri_drive: `LEGS=ALL` = every
canonical + control leg incl. `none__lens0`), and a new unsteered measurement is forbidden here (the anchor is the model's
EXISTING `none` store). This tool reads the roster view and prints exactly the view's admitted cells — its steered objects AND the prompted SI
baselines the roster DECLARES on the view (`prompted__<si>: null`, the roster's prompted grammar: the SI as the system prompt on the
served instruct target, no object, no staged id, served on the view's engine beside its objects) — refusing `none`, a prompted cell
that maps to a staged id, every
zero-adapter control and any object that is not the lane's own (a `base__` batch-8 object on a replicate view is the wrong object,
not a cell). The launch plan's MEASURED cells are the objects; the prompted cells ride the plan as `prompted_cells`, never a measured
object of a Slurm producer (nothing of the reseed record changes for a roster without them).

  select_cells.py <view> [--roster <yaml>]                       -> the comma list every submission passes (roster order)
  select_cells.py <view> --require-complete                      -> the same, refusing a view that still declares a `pending_fit`
                                                                    object (neither staged into `cells` nor ruled out of the view)
  select_cells.py <view> --petri-indices <out_root>/cells.json   -> the admitted cells' CANONICAL leg indices (space-separated) of the frozen
                                                                    Petri enumeration; never a `controls` index, never `none`
  select_cells.py <view> --launch-plan <out.json> --wave-sha <W> [--goalset-root <GS>] [--petri-cells-json <cells.json>]
                                                                 -> RS/reseed/launch_plan.json: every producer's argv / env with the
                                                                    admitted cells as measured cells (nothing is launched)
Exit 0 with the value on stdout; a message on stderr and exit 2 on any refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_seed as es  # noqa: E402
from f22_roster import ROSTER as DEFAULT_ROSTER, is_prompted, load_roster, pending_fit_of, served_cells  # noqa: E402

#: the six DECLARED cell ids of every replicate view, in the campaign grammar's order (the roster lists them in this order); a
#: view may carry FEWER (a derived object the admission refused is not on its roster view — decisions O20, 2026-09-17: the
#: distilled PSR difference failed admission for every seed, so every view carries FIVE), never a cell outside this set
PARENT_CELLS = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
                "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce")
DERIVED_CELLS = ("constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl_distilled")
SIX_CELLS = PARENT_CELLS + DERIVED_CELLS
#: the staged-id grammar of the lane (plan naming): parents `base64s<s>__<cell>`, derived `derived_base64s<s>__<cell>`
STAGED_PREFIX = {**{c: "base64s{s}__" for c in PARENT_CELLS}, **{c: "derived_base64s{s}__" for c in DERIVED_CELLS}}
#: the cells that are NEVER a measurement target of the lane
NEVER = ("none",)
CONTROL_SUFFIXES = ("__lens0", "__lora0")          # the scale-0 twins the standups serve for their health checks
#: the multi-turn attacks and the leg list of the coverage table
MULTITURN = ("crescendo", "pair", "actorattack")


class Refuse(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"select_cells: {msg}")


def is_measurement_forbidden(cell_id: str) -> str | None:
    """Why ``cell_id`` may never be a measured cell of the lane (None = it may): the anchor, a prompted baseline, a zero-adapter control."""
    cid = str(cell_id)
    if cid in NEVER:
        return "the unsteered anchor `none` is never re-measured (the model's existing `none` store is the anchor)"
    if is_prompted(cid):
        return "a prompted__* baseline is never a measured cell of the lane"
    if any(cid.endswith(sfx) for sfx in CONTROL_SUFFIXES):
        return "a zero-adapter control (the standup's scale-0 twin) is a health check, never a measurement"
    if cid.startswith("safety_refusal__"):
        return "the safety-SI cells are the campaign's main-view controls, not this lane's objects"
    return None


def staged_id_for(view: str, cell: str) -> str:
    s = es.seed_of_view(view)
    return STAGED_PREFIX[cell].format(s=s) + cell


def select(roster: dict, view: str, *, target: str = es.TARGET, require_complete: bool = False) -> dict:
    """The view's admitted cells ``{cell_id: staged_id}`` in roster order — the steered objects (a staged id each) and the prompted
    SI baselines the roster declares on the view (``prompted__<si>: null`` -> ``None``: roster 978a7a97 declares
    `prompted__safety_refusal` / `prompted__si_constitution` on every seed view, served beside the objects; admitted EXACTLY as
    declared: the expectation is derived from the roster, never a literal six); refuses by name anything the plan forbids.
    ``require_complete`` refuses while the view still declares a ``pending_fit`` object (see the comment below)."""
    if not es.is_seedrep_view(view):
        raise Refuse(f"{view!r} is not a seed-replicate view ({', '.join(es.view_of(s) for s in es.SEEDS)}) — this selector is the lane's")
    targets = roster.get("targets") or {}
    if target not in targets:
        raise Refuse(f"roster names no target {target!r}")
    if view not in (targets[target].get("views") or {}):
        raise Refuse(f"roster names no view {target}/{view} (views: {sorted(targets[target].get('views') or {})}) — the lane's W5 lands it")
    try:
        cells = served_cells(roster, target, view)
    except ValueError as e:
        raise Refuse(f"roster {target}/{view}: {e}") from None
    out: dict = {}
    for cid, sid in cells.items():
        if sid is None and is_prompted(cid):
            # the roster's DECLARED prompted baseline (f22_roster.served_cells hands it over with None; a prompted cell mapping to a
            # staged id was refused there already): admitted as declared — no object, no staged id, no pin
            out[cid] = None
            continue
        why = is_measurement_forbidden(cid)
        if why is not None:
            raise Refuse(f"{target}/{view} lists {cid!r}: {why} — a replicate view carries the six steered objects only")
        if cid not in SIX_CELLS:
            raise Refuse(f"{target}/{view} lists {cid!r}, not one of the lane's six cells {list(SIX_CELLS)}")
        if sid is None:
            raise Refuse(f"{target}/{view}: {cid} maps to null — an object cell needs its staged id")
        if not isinstance(sid, str):
            raise Refuse(f"{target}/{view}: {cid} maps to {sid!r} — the lane stages plain ids ({staged_id_for(view, cid)}), never a pin mapping")
        want = staged_id_for(view, cid)
        if sid != want:
            raise Refuse(f"{target}/{view}: {cid} <- {sid!r} is not this replicate's object ({want}) — a foreign object is never measured as a replicate")
        out[cid] = sid
    if not any(sid is not None for sid in out.values()):
        raise Refuse(f"{target}/{view} lists no steered cell")
    if require_complete:
        # COMPLETE = the roster has RULED on every declared object of the view: each is either staged into `cells` or dropped
        # from the view by its admission record (decisions O20) — an id still under `pending_fit` is neither, so the view is not
        # complete and the refusal names it. Fewer than six admitted cells is NOT incompleteness (the plan's "fewer if an
        # admission failed"); the roster, not this tool, is the authority on which objects a view serves.
        pending = pending_fit_of(roster, target, view)
        if pending:
            raise Refuse(f"{target}/{view} still declares pending_fit {sorted(pending)} (an object neither staged into `cells` nor "
                         f"ruled out of the view; {len(out)} of the six declared cells admitted) — --require-complete refused")
    return out


def petri_indices(cells_json: Path, cells: list) -> list:
    """The frozen Petri enumeration's CANONICAL leg indices of ``cells`` (petri_run print-cells: legs are indexed over
    ``canonical + controls``; a cell in ``controls`` — `none__lens0` and its kin — is never selected; a cell absent refuses)."""
    from sisteer.capeval.petri import registry_adapter as ra
    rec = json.loads(Path(cells_json).read_text(encoding="utf-8"))
    canonical = list(rec.get("canonical") or [])
    controls = list(rec.get("controls") or [])
    legs = canonical + controls
    by_id = {}
    for i, e in enumerate(legs):
        by_id.setdefault(str(e.get("cell_id")), []).append(i)
    out = []
    for cell in cells:
        hid = ra.harness_cell_id(cell)
        idx = by_id.get(hid)
        if not idx:
            raise Refuse(f"{cells_json}: no leg named {hid!r} (cell {cell}) in the frozen enumeration")
        if len(idx) != 1:
            raise Refuse(f"{cells_json}: {hid!r} appears {len(idx)} times")
        if idx[0] >= len(canonical):
            raise Refuse(f"{cells_json}: {hid!r} is a `controls` leg (index {idx[0]}), never a measured one")
        out.append(idx[0])
    for i, e in enumerate(legs):
        cid = str(e.get("cell_id"))
        if i in out and is_measurement_forbidden(cid) is not None:
            raise Refuse(f"{cells_json}: leg {i} ({cid}) would be measured: {is_measurement_forbidden(cid)}")
    return out


def launch_plan(view: str, cells: dict, *, wave_sha: str, goalset_root: str | None = None,
                petri_cells_json: Path | None = None, cluster: str = "cluster_a") -> dict:
    """RS/reseed/launch_plan.json for ONE view: every producer's argv and scalar env with the view's admitted cells as the MEASURED cells.
    A record of what will be submitted, never a submission; the test asserts no producer targets
    `none`, a prompted anchor or a zero-adapter control."""
    s = es.seed_of_view(view)
    tag = es.tag_of(s)
    base = es.seed_base(s)
    # the MEASURED cells of every producer are the view's OBJECTS; a prompted baseline the roster declares (`None` staged id) rides the
    # plan as `prompted_cells` — served on the view's engine, never a measured object of a Slurm producer (the anchor policy below holds)
    prompted = [c for c, sid in cells.items() if sid is None and is_prompted(c)]
    six = [c for c in cells if c not in prompted]          # every other cell is a MEASURED cell — `add` refuses a forbidden one by name
    csv = ",".join(six)
    gs_root = goalset_root or f"$GS/{tag}"
    producers = []

    def add(leg, eval_ids, argv, *, env=None, measured=None, served_only=(), anchor_measured=(), notes=""):
        measured = list(six if measured is None else measured)
        for c in measured:
            why = is_measurement_forbidden(c)
            if why is not None:
                raise Refuse(f"launch plan {leg}: {c}: {why}")
        # decisions O12a: the ONE leg that measures an unsteered anchor of its own — the GSM8K paired-200 matrix's natively served
        # `none` row per view (main's reader refuses the lens-era published card by name, O106) — records it HERE, apart from the
        # six MEASURED cells, so the plan's "no producer measures `none`" reading stays a checked fact everywhere else
        if anchor_measured and leg != "gsm8k":
            raise Refuse(f"launch plan {leg}: only the GSM8K paired-200 matrix carries a native anchor measurement (O12a), got {list(anchor_measured)}")
        producers.append({"leg": leg, "eval_ids": list(eval_ids), "cluster": cluster, "measured_cells": measured,
                          "served_only": list(served_only), "anchor_measured": list(anchor_measured), "argv": list(argv), "env": dict(env or {}),
                          "seed_rule": {e: es.leg_seed_rule(e).seed_rule for e in eval_ids},
                          "seed_base": {e: es.leg_seed_base(e, s) for e in eval_ids}, "notes": notes})
    sampled = "scripts/sdfbase/full22/sampled/target_sampled.sbatch"
    static = "scripts/sdfbase/full22/target_static.sbatch"
    for t in (0, 1):
        add("static_sampled", ("pasttense", "codeattack", "renellm", "grok420"),
            ["sbatch", "--qos=high", "--time=0-08:00:00", "-J", f"reseed-samp-{view}-sb{t}", sampled, es.TARGET, f"static_base_t{t}", wave_sha, "9", csv, view],
            notes="the 5th positional is a LITERAL comma list (never empty, never `all`); rungs on ladder_decide.py's 1 % trigger")
        add("over_refusal", ("xstest", "or_bench_hard_1k"),
            ["sbatch", "--qos=high", "--time=0-08:00:00", "-J", f"reseed-orf-{view}-t{t}", sampled, es.TARGET, f"controls_t{t}", wave_sha, "9", csv, view])
    add("mmlu", ("mmlu",),
        ["sbatch", "--qos=high", "-J", f"reseed-ctl-{view}", static, es.TARGET, view, "controls", wave_sha, "9", csv],
        notes="the 6th positional is a LITERAL comma list")
    add("bon", ("bon",),
        ["sbatch", "--qos=high", "-J", f"reseed-bon-{view}", static, es.TARGET, view, "bon", wave_sha, "9", csv],
        env={"F22_BON_DECODE_PRESET": "anchor_match", "F22_BON_DECODE_SEED_POLICY": es.BON_DECODE_SEED_POLICY,
             "F22_BON_DECODE_SEED_BASE": str(base)},
        notes="SCALAR env on the submitting shell (Slurm's implicit export; never --export=); target_static.sbatch forwards the three")
    add("gsm8k", ("gsm8k",),
        ["sbatch", "-J", "reseed-g8k", "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch", "ARRAY", wave_sha, "8192", "qwen_seedrep"],
        anchor_measured=["none"],
        notes="one row per (view, cell) by matrix construction (views_qwen_seedrep.yaml), PLUS the view's natively served `none` ANCHOR row "
              "(decisions O12a: main's GSM8K reader refuses the lens-era published `none` card against a native cell by name, O106, and keys "
              "the anchor per view — a NEW unsteered measurement, 200 items, recorded in the seed ledger as the view's native anchor); every "
              "standup of the wave co-registers constv2-mix-full__lora_r4_ce unmeasured so the reader's max_lora_rank comparison holds; "
              "the replicate seed policy rides every row; the array's indices are the plan's for THIS view "
              "(g8k_cells.py plan --matrix qwen_seedrep --wave qwen_seedrep)")
    for atk in MULTITURN:
        add("multiturn", (atk,),
            ["sbatch", "-J", f"reseed-mt-{view}-{atk}", "scripts/sdfbase/full22/target_attacks_goalset.sbatch", es.TARGET, view, atk, wave_sha,
             gs_root, "reseed", "temperature=1.0,top_p=1.0,top_k=-1", "9", "4096", "49152", "", "3600", "3600"],
            env={"F22_LOG_REQUESTS": "1"},
            notes=f"a pair exists iff its goal-set file does: {gs_root}/{atk}__<cell>.txt for the six cells ONLY (no none / prompted file)")
    idx = petri_indices(petri_cells_json, six) if petri_cells_json is not None else None
    add("petri", ("petri",),
        ["sbatch", "-J", f"reseed-petri-{view}", "scripts/sdfbase/full22/petri_proxy.sbatch", es.TARGET, view, wave_sha],
        env={"F22_SERVER_DEFAULT_POLICY": "engine_defaults"},
        notes="the grading host driver: petri_drive_full22.sh qwen35_4b <view> cluster_a <W> \"<indices>\" with the six CANONICAL indices "
              + (f"{idx} (from {petri_cells_json})" if idx is not None else "read from the frozen cells.json by select_cells.py --petri-indices — never ALL"))
    producers[-1]["petri_indices"] = idx
    add("misalign", ("misalign_leaking", "misalign_murder"),
        ["sbatch", "-J", f"reseed-misal-{view}", "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch", tag, wave_sha, "full"],
        env={"F22_SERVER_DEFAULT_POLICY": "engine_defaults"}, served_only=["none"],
        notes="lane-scoped manifest (W6b): `none` SERVED for the standup's health checks only, the six MEASURED (measure_none: false)")
    return {"schema": "seedrep_launch_plan/1", "view": view, "tag": tag, "seed": s, "seed_base": base, "wave_sha": wave_sha,
            "target": es.TARGET, "cells": six, "staged_ids": {c: cells[c] for c in six}, "prompted_cells": prompted,
            "anchor_policy": "every steered row pairs against the model's EXISTING main-view `none` stores through anchor_pairing; "
                             "no producer of this plan measures `none`, a prompted baseline or a zero-adapter control — EXCEPT the GSM8K "
                             "paired-200 array (O12a): its matrix carries ONE natively served `none` anchor row per view (`anchor_measured`), "
                             "because main's reader refuses the lens-era published card against a native cell by name (O106)",
            "producers": producers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("view")
    ap.add_argument("--roster", default=str(DEFAULT_ROSTER))
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--petri-indices", metavar="CELLS_JSON", default=None)
    ap.add_argument("--launch-plan", metavar="OUT_JSON", default=None)
    ap.add_argument("--wave-sha", default=None)
    ap.add_argument("--goalset-root", default=None)
    ap.add_argument("--petri-cells-json", default=None)
    ap.add_argument("--cluster", default="cluster_a")
    a = ap.parse_args(argv)
    try:
        roster = load_roster(Path(a.roster))
        cells = select(roster, a.view, require_complete=a.require_complete)
        if a.petri_indices:
            objects = [c for c, sid in cells.items() if sid is not None]     # the canonical legs of the OBJECTS (a prompted arm is not a rung's cell)
            print(" ".join(str(i) for i in petri_indices(Path(a.petri_indices), objects)))
            return 0
        if a.launch_plan:
            if not a.wave_sha:
                raise Refuse("--launch-plan needs --wave-sha")
            plan = launch_plan(a.view, cells, wave_sha=a.wave_sha, goalset_root=a.goalset_root,
                               petri_cells_json=Path(a.petri_cells_json) if a.petri_cells_json else None, cluster=a.cluster)
            out = Path(a.launch_plan)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(plan, indent=1) + "\n")
            print(str(out))
            return 0
        print(",".join(cells))
        return 0
    except SystemExit as e:
        if e.code and isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
