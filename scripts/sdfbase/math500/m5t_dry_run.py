#!/usr/bin/env python3
"""The TWO-STAGE DRY RUN of the math500-thinking-on-full22 launch.
No GPU, no venv build, no bank write. Per matrix row (cells.py M5T_CELLS) it prints:

  * the resolved argv of standup extras, ledger and driver — VERBATIM from the sbatch's own shell functions
    (m5t_common.sh m5t_print_plan, run under M5T_DRY_RUN=1: the same code path the job runs, not a re-typing);
  * `<the row's geometry entry> <its L> x <S>` (S resolved through the real geometry table for the model AT THE
    ROW'S ENTRY — capability:gen_long_math 17,664 for the bare / safety-SI / object cells, capability:gen_long_math_longsi
    24,064 for the constitution-SI cell; math-eight-arms-extension §4.1: "the row's geometry resolves to the row's L"),
    `cap 16384`, `thinking <the row's mode>`, `retry 0`, the sampling row and its source, `--upstream-timeout-s 7200`;
  * the row's WAVE line (wave, status, view, budget class, staged id, SI digest, the sampler read-back opt-in), the
    recorder argv (`RESOLVED_SAMPLER:`, iff opted in) and the row-identity stamp's path (`ROW_IDENTITY:`);
  * the route kind per cell (base / prompted / lens) with the prompted route's system-prefix hash;
  * `lens_served_positions[cell]` read from a FRESHLY EXPORTED routing table (the standup's own cell selection +
    export_routing_table over the given bank views, `--bank-root`; absent key => REFUSED at --stage prod);
  * the family's render fact: `split_mode` + `evidence_status` from contracts.FAMILY_RENDER_FACTS, and
    the verdict of the preflight SEAM `contracts.check_render_fact_certified(family)` at BOTH stages —
    `--stage pre` prints `candidate` and asserts the seam REFUSES a candidate fact by name; `--stage prod`
    demands the seam ADMIT every family it plans (REFUSED: render fact not certified otherwise — the seam's own
    message, never a re-derived compare of the fact's fields; a tree without the seam refuses at prod);
  * the expected stamp triple per cell (and twin), the tuple name per fitted row, `OUT OF SUPPORT` per fitted row;
  * exactly one G-LENS status line (refinement merged: RoutingProxy takes lens_served_positions; Petri smoke record).

Every count the test's greps read is DERIVED from the matrix (cells.counts) and printed as a
COUNTS line; the test compares the greps over this output to those numbers, never to typed literals. Exit 0 =
the stage passes; 2 = a refusal (printed with REFUSED:); nothing GPU-side depends on this script.

The default walk is the MATRIX (cells.M5T_CELLS, 54 rows); the seed wave — a sibling list
(cells.SEEDREP_ROWS, indices 54-86) — is walked with `--wave qwen_seedrep` (any wave name resolves off
cells.LAUNCHER_ROWS), or, when neither `--wave` nor `--only` is given, the submitter-shell variable M5T_DRY_WAVE names the wave
(the mechanism dry_run_prod.sbatch — which takes only the sha — inherits from the submitting shell; never --export=). A seed
row's fresh table is exported over ITS view's campaign bank (`<bank-root>/<key>/cells_view/<view>`) through its OWN overlay pair
with the row's cell AND its standup companions selected (the cells the standup registers); an Arditi seed row over the
external lane's view like the all-positions row.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "math500"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import cells as M
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import itdoc_bind as B  # noqa: E402

STAGES = ("pre", "prod")
SBATCH = REPO / "scripts/sdfbase/math500/math500_thinking.sbatch"


def resolved_s(model_key: str, geometry_eval: str = M.GEOMETRY_EVAL, geometry_l: int = M.GEOMETRY_L) -> int:
    """S the standup will derive for this model at THE ROW'S geometry entry (the real table; S is never passed to the
    standup — `strict_identity`); the entry must resolve to the row's L (17,664 or 24,064), else REFUSED."""
    from capctrl_serving_up import resolve_standup_geometry
    geom = resolve_standup_geometry(geometry_eval, None, None, model_key=model_key)
    if int(geom.max_model_len) != int(geometry_l):
        raise SystemExit(f"REFUSED: {geometry_eval} resolves L={geom.max_model_len}, not the row's {geometry_l}")
    return int(geom.max_num_seqs)


def resolved_s_for_row(row: dict) -> int:
    return resolved_s(row["model_key"], row["geometry_eval"], row["geometry_L"])


def shell_plan(row: dict, *, s: int, shared: Path, python: str) -> str:
    """The sbatch's OWN dry-run printing for one row (m5t_print_plan through math500_thinking.sbatch)."""
    env = {**os.environ, "M5T_DRY_RUN": "1", "M5T_DRY_S": str(s), "M5T_DRY_SHARED": str(shared),
           "M5T_DRY_CPY": python, "USER": os.environ.get("USER", "tester")}
    # math500-steer: each row's dry run starts as the real single-row job does — with NO serving path in its environment — so the
    # ROW's own pin selects the route (m5t_select_serving_path) and a value the ENCLOSING job resolved (dry_run_prod.sbatch's
    # f22_init exports its default before spawning these) never reads as a submitter's conflicting export
    env.pop("F22_SERVING_PATH", None)
    # E3: the optional view positional is keyed the way cells.row_for keys a row (cells.key_view) — a steering row's `view` field names
    # the EXTERNAL lane's view (it / ar_r0p05 / …) while the row itself keys as `main`; only an it-view row names `it` here
    view = M.key_view(row)
    # the dry run of a seed row starts as the real job does — with NO wave selector in its environment (the enclosing dry run's
    # own M5T_DRY_WAVE must not leak into the sbatch stub, which reads none of it, but hygiene first)
    env.pop("M5T_DRY_WAVE", None)
    r = subprocess.run(["bash", str(SBATCH), row["model_key"], row["cell"], row["sampling_mode"], "0" * 40,
                        "1" if row["fitted"] else "0"] + ([view] if view != M.VIEW_MAIN else []),
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: the sbatch dry run failed for row {row['index']}: {r.stdout[-800:]}{r.stderr[-800:]}")
    return "\n".join(line for line in r.stdout.splitlines() if not line.startswith("["))


def render_fact(family: str) -> dict[str, Any]:
    """The family's render fact from lane A's contracts table — ABSENT is recorded (never invented)."""
    try:
        from sisteer.capeval.tinycap import contracts as C
    except Exception as e:
        return {"present": False, "why": f"contracts import failed: {type(e).__name__}: {e}"}
    facts = getattr(C, "FAMILY_RENDER_FACTS", None)
    if not isinstance(facts, dict) or family not in facts:
        return {"present": False, "why": "contracts.FAMILY_RENDER_FACTS lacks this family (lane A's table not merged)"}
    f = facts[family]
    get = (lambda k: getattr(f, k, None)) if not isinstance(f, dict) else (lambda k: f.get(k))
    return {"present": True, "split_mode": get("split_mode"), "evidence_status": get("evidence_status"),
            "off_ids": list(get("off_ids") or []) or None, "on_ids": list(get("on_ids") or []) or None}


def render_fact_seam_verdict(family: str) -> dict[str, Any]:
    """The verdict of the driver's render-fact preflight seam, `contracts.check_render_fact_certified(family)`,
    consulted at BOTH stages (integration I2): `refuses` True with the seam's own message, False when it admits;
    `seam: None, refuses: None` when the tree carries no seam (recorded — `pre` notes it, `prod` fails closed)."""
    try:
        from sisteer.capeval.tinycap import contracts as C
        fn = getattr(C, "check_render_fact_certified", None)
    except Exception:
        fn = None
    if fn is not None:
        try:
            fn(family)
            return {"seam": "contracts.check_render_fact_certified", "refuses": False}
        except Exception as e:
            return {"seam": "contracts.check_render_fact_certified", "refuses": True, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"seam": None, "refuses": None,
            "why": "no render-fact preflight seam on this tree (lane A's contracts / driver) — the assertion cannot run"}


def fresh_table_stamp(row: dict, bank_root: Path | None, tmp: Path, *, shared_root: Path | None = None,
                      cluster_role: str = "primary") -> dict[str, Any]:
    """The stamp the standup's OWN cell selection + export_routing_table write for this row, from a FRESH export
    over the bank views under `bank_root` (`<bank_root>/<key>/cells_view/main`); None when no bank is given.
    A STEERING row (`object_source: external`) serves its OWN overlay pair
    (`sdfbase_full22_<key>.math500_<cell slug>.yaml`, `explicit_cells: [none, <cell>]`) over ANOTHER lane's bank view
    — `<shared_root>/<bank_view_rel[cluster_role]>` — exactly what the job's m5t_resolve_config / m5t_bank_view bind;
    the base overlay's six cells and the full22 `main` view would refuse it (the reason the pair exists), so the export
    for such a row needs `--shared-root` (and the cluster role the view is staged for): absent, it is NOT exported and
    `--stage prod` refuses by name — never the base wave's shape run in its place."""
    if bank_root is None and row.get("object_source") not in (M.OBJECT_SOURCE_EXTERNAL, M.OBJECT_SOURCE_ITDOC):
        return {"exported": False, "why": "no --bank-root: the fresh-table check runs where the bank views exist"}
    external = row.get("object_source") == M.OBJECT_SOURCE_EXTERNAL
    itdoc = row.get("object_source") == M.OBJECT_SOURCE_ITDOC
    # a seed row of the campaign bank (own overlay, lane view, no external source) exports over ITS view's roster bank
    own_overlay = bool(row.get("own_overlay")) and not external and not itdoc
    if itdoc:
        # E3: an it-doc row exports through its OWN pair (the `it` view's campaign config, explicit_cells [none, <cell>]) over the bank
        # entry its record names — NEVER the base overlay over the `main` view (the same cell id there is the base lane's object)
        rel = (row.get("bank_view_rel") or {}).get(cluster_role)
        if row.get("object_sha256") is None or not row.get("bank_view_rel"):
            return {"exported": False, "why": f"it-doc row ({row['cell']}, lane {row.get('lane')}) is not bound to a bank yet ({row.get('held_reason')}) — nothing to export"}
        if shared_root is None:
            return {"exported": False, "why": f"it-doc row ({row['cell']}): the fresh-table export needs --shared-root (its bank view is <shared root>/{rel})"}
        if not rel:
            return {"exported": False, "why": f"it-doc row ({row['cell']}): no bank view on the {cluster_role} cluster (bank_view_rel {row.get('bank_view_rel')!r})"}
    if external:
        rel = (row.get("bank_view_rel") or {}).get(cluster_role)
        if shared_root is None:
            return {"exported": False, "why": f"external row ({row['cell']} from {row.get('lane_root')}): the fresh-table export needs "
                                              f"--shared-root (its bank view is <shared root>/{rel}), not the full22 bank root"}
        if not rel:
            return {"exported": False, "why": f"external row ({row['cell']}): no bank view on the {cluster_role} cluster "
                                              f"(bank_view_rel {row.get('bank_view_rel')!r}) — run it where its object is staged"}
    import importlib.util

    import yaml
    spec = importlib.util.spec_from_file_location("_m5t_dry_mt", REPO / "scripts/sdfbase/mt_serving_up.py")
    mt = importlib.util.module_from_spec(spec)
    sys.modules["_m5t_dry_mt"] = mt
    spec.loader.exec_module(mt)
    import dataclasses

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving.proxy import export_routing_table, load_routing_table
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
    key = row["model_key"]
    # the pair's TAG is the row's PANEL view's, exactly as the launcher's m5t_resolve_config derives it (cells.key_view):
    # `<key>` on a main-keyed row — every steering row, whose `view` names its object BANK — and `<key>_<panel view>`
    # on a row that sits on another panel view (E3's it-doc rows; the all-positions Arditi row, cut from the `it` TAG)
    view = M.key_view(row)
    tag = key if view == M.VIEW_MAIN else f"{key}_{view}"
    if external or itdoc or own_overlay:
        # the row's OWN pair (`<tag>.math500_<cell slug>`) over ITS bank: another lane's view for an external / it-doc row
        # (`<shared root>/<bank_view_rel>`); for a seed row (explicit_cells [none, <cell>] + companions) the roster bank of
        # its VIEW — `<bank_root>/<key>/cells_view/<view>`, what m5t_bank_view's f22_bank_view_check --view exports
        cfg_path = REPO / f"configs/sdfbase/math500/sdfbase_full22_{tag}.math500_{row['cell_slug']}.yaml"
        man_path = REPO / f"configs/capeval/manifests/petri_full22_{tag}.math500_{row['cell_slug']}.yaml"
        bank = Path(bank_root) / key / "cells_view" / view if own_overlay else Path(shared_root) / rel
    else:
        cfg_path = REPO / f"configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml"
        man_path = REPO / f"configs/capeval/manifests/petri_full22_{key}.math500.yaml"
        bank = Path(bank_root) / key / "cells_view" / "main"
    cfg = yaml.safe_load(cfg_path.read_text())
    spec0 = cm.load_petri_spec(man_path)
    # the cells the STANDUP registers: the row's own and, on a seed row, its companions (the standup's repeated --cell; the driver still
    # measures the row's cell alone) — so the fresh table carries the companion's route and stamp beside the row's
    tt_cells = mt.select_cells(cfg["explicit_cells"], [row["cell"]] + [c for c, _h, _sid, _pin in M.row_companions(row)])
    harness = [ra.harness_cell_id(c) for c in tt_cells]
    spec_ = dataclasses.replace(spec0, serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
                                cells=harness)
    src = traintest_cell_source(spec_, repo_root=REPO, config_path=cfg_path, defenses_dir=bank)
    bf = {}
    for k in cm.petri_expected_behavior_keys(spec_):
        bf[k] = (REPO / "configs/capeval/petri" / k / "BEHAVIOR.md").read_bytes()
    prompt_files = resolve_prompt_files(spec_, override_root=REPO / "configs/capeval/petri/prompts")
    cells = dc.enumerate_cells(spec_, src, behavior_files=bf, prompt_files=prompt_files)
    if cells.absent:
        return {"exported": False, "why": f"absent cells over {bank}: {[(a.cell_id, a.reason) for a in cells.absent]}"}
    out = tmp / f"row{row['index']}_routing_table.json"
    export_routing_table(cells, src, out, base_model=cfg["model"], serving_mode="interleaved", base_revision=cfg["model_revision"])
    table = load_routing_table(out)
    routes = {r.cell_id: r for r in table["routes"]}
    stamp = table.get("lens_served_positions")
    prefixes = {s.virtual_model: s.sha16 for s in table["prompted_prefixes"]}
    return {"exported": True, "routes": sorted(routes), "lens_served_positions": stamp, "bank_view": str(bank),
            "config": str(cfg_path.relative_to(REPO)), "own_route": row["harness_cell"] in routes,
            # the prompted route's system-prefix digest, looked up under THIS row's harness cell (prompted__safety /
            # prompted__constitution) — the row's own SI, never the safety cell's by default
            "prompted_prefix_sha16": prefixes.get(f"petri__{row['harness_cell']}") if row["route_kind"] == "prompted" else None,
            "stamp_for_cell": (stamp or {}).get(row["harness_cell"]) if isinstance(stamp, dict) else None}


def g_lens_status(petri_record: str | None) -> str:
    import inspect

    from sisteer.capeval.serving.proxy import RoutingProxy
    merged = "lens_served_positions" in inspect.signature(RoutingProxy.__init__).parameters
    if not merged:
        return "G-LENS: NOT MERGED — fitted jobs held (RoutingProxy takes no lens_served_positions)"
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    return f"G-LENS: refinement merged at {sha[:12]} (RoutingProxy takes lens_served_positions), Petri smoke {petri_record or '<record path not given>'}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--sha", default="0" * 40, help="the wave sha the argv names (cosmetic here)")
    ap.add_argument("--bank-root", default=None, help="<root>/<key>/cells_view/main per model: enables the fresh-table stamp check")
    ap.add_argument("--shared-root", default=None, help="the cluster's shared root: a W4 steering row's external bank view is "
                                                        "<shared root>/<its bank_view_rel[cluster role]> (needed for its fresh-table check)")
    ap.add_argument("--cluster-role", default="primary", choices=("primary", "overflow"),
                    help="which cluster's bank view an external row's export reads (its bank_view_rel key)")
    ap.add_argument("--petri-smoke-record", default=None, help="Evals' Petri smoke record path for the G-LENS line")
    ap.add_argument("--only", nargs=3, metavar=("KEY", "CELL", "SAMPLING"), default=None)
    ap.add_argument("--only-view", default=M.VIEW_MAIN, help=f"the view of the --only row ({M.VIEW_MAIN} by default; {M.VIEW_IT} for an E3 it-doc row; "
                                                          f"a seed / paired view {M.SEEDREP_VIEWS + M.SEEDREP_PAIRED_VIEWS} for a W1 seed row)")
    ap.add_argument("--wave", default=None, help="walk ONE wave's rows off cells.LAUNCHER_ROWS (qwen_seedrep: the seed wave, a sibling list beside "
                                                 "the matrix); default: the whole matrix (cells.M5T_CELLS), or the wave M5T_DRY_WAVE names in the "
                                                 "submitting shell's environment (dry_run_prod.sbatch takes only the sha)")
    ap.add_argument("--python", default=sys.executable, help="the client-venv python the argv names")
    ap.add_argument("--tmp", default="/tmp/m5t-dry", help="scratch root for the printed paths and fresh tables")
    a = ap.parse_args(argv)
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    if a.only is not None and a.wave is not None:
        raise SystemExit("REFUSED: --only and --wave are exclusive")
    if a.only is not None:
        wave, rows = None, [M.row_for(*a.only, view=a.only_view)]
    else:
        wave = a.wave if a.wave is not None else (os.environ.get("M5T_DRY_WAVE") or None)
        rows = M.rows_of_wave(wave) if wave is not None else M.M5T_CELLS
        if not rows:
            raise SystemExit(f"REFUSED: --wave {wave!r} names no row (waves: {list(M.WAVES)})")
    refusals: list[str] = []
    # S per (model, geometry entry): the row's entry must resolve to the row's L (a REFUSAL otherwise, before any row prints)
    s_by_row = {r["index"]: resolved_s_for_row(r) for r in rows}
    print(f"STAGE: {a.stage} rows={len(rows)} sha={a.sha}" + (f" wave={wave}" if wave is not None else ""))
    for (key, ev, gl), s in sorted({(r["model_key"], r["geometry_eval"], r["geometry_L"]): s_by_row[r["index"]] for r in rows}.items()):
        print(f"GEOMETRY_S: model={key} eval={ev} L={gl} S={s}")
    facts = {fam: render_fact(fam) for fam in sorted({r["family"] for r in rows})}
    for fam, f in facts.items():
        status = f.get("evidence_status") if f["present"] else "absent"
        print(f"RENDER_FACT: family={fam} split_mode={f.get('split_mode') if f['present'] else 'absent'} evidence_status={status}"
              + ("" if f["present"] else f" ({f['why']})"))
        # lane A's preflight seam, consulted at BOTH stages: `pre` asserts it refuses a candidate fact, `prod`
        # requires it to admit — the stage's verdict IS the seam's (integration I2), never a string compare here
        pf = render_fact_seam_verdict(fam)
        if a.stage == "pre":
            print(f"PREFLIGHT_REFUSES_CANDIDATE: family={fam} seam={pf['seam']} refuses={pf['refuses']}"
                  + (f" ({pf['why']})" if pf.get("why") else ""))
            if pf["refuses"] is False and status != "certified":
                refusals.append(f"REFUSED: the production preflight ADMITS family {fam} with evidence_status={status} at --stage pre")
        else:
            admits = pf["refuses"] is False
            print(f"PREFLIGHT_ADMITS: family={fam} seam={pf['seam']} admits={admits}"
                  + (f" ({pf['why']})" if pf.get("why") else ""))
            if not admits:
                refusals.append(f"REFUSED: render fact not certified for family {fam} (evidence_status={status}): "
                                + (pf.get("why") or "the production preflight seam did not admit it"))
    for row in rows:
        text = shell_plan(row, s=s_by_row[row["index"]], shared=tmp / "shared", python=a.python)
        print(text)
        if row["status"] == M.STATUS_HELD:
            # E3: a HELD row prints its HELD line and NO argv (the sbatch's dry stub exits there; the batch boundary refuses it in
            # production); nothing is exported for it and its absence is never a prod refusal
            if "HELD:" not in text or "DRIVER:" in text or "STANDUP:" in text:
                refusals.append(f"REFUSED: held row {row['index']} ({row['job_id']}) did not exit HELD with no argv")
            if row.get("object_source") == M.OBJECT_SOURCE_ITDOC:
                print(f"HELD_ITDOC: row={row['index']} {B.tuple_label(row['lane'], row['model_key'], row['view'], row['cell'])} staged_id={row['staged_id']} "
                      f"pin={(row.get('object_sha256') or 'null')[:12]} reason={row['held_reason']}")
            print(f"ROW_FACTS: row={row['index']} wave={row['wave']} status={row['status']} thinking={row['thinking_mode']} "
                  f"geometry={row['geometry_eval']}/{row['geometry_L']} budget_class={row['budget_class']} "
                  f"log_requests={int(bool(row['log_requests']))} staged_id={row['staged_id']} si_sha256={row['si_sha256']}")
            print()
            continue
        st = fresh_table_stamp(row, Path(a.bank_root) if a.bank_root else None, tmp,
                               shared_root=Path(a.shared_root) if a.shared_root else None, cluster_role=a.cluster_role)
        if row.get("object_source") == M.OBJECT_SOURCE_ITDOC:
            # E3: the bank BINDING verdict per launchable it-doc row (itdoc_bind.row_bind; the lane named in every refusal)
            if a.shared_root is None:
                print(f"ITDOC_BIND: row={row['index']} not checked (no --shared-root: the bank entry is named relative to the shared root)")
                if a.stage == "prod":
                    refusals.append(f"REFUSED: --stage prod requires the it-doc bank binding check (give --shared-root) — row {row['index']}")
            else:
                pins_doc = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text(encoding="utf-8"))
                rec = B.row_bind(row, Path(a.shared_root), cluster_role=a.cluster_role, pins_doc=pins_doc)
                print(f"row={row['index']} " + B.bind_verdict_line(rec))
                if rec["verdict"] != B.VERDICT_PASS and a.stage == "prod":
                    refusals.append(f"REFUSED: row {row['index']} ({row['job_id']}) it-doc bank binding {rec['verdict']}: " + "; ".join(rec["problems"]))
        if st["exported"]:
            print(f"FRESH_TABLE: row={row['index']} routes={st['routes']} own_route={st['own_route']} "
                  f"lens_served_positions={json.dumps(st['lens_served_positions'])} prompted_prefix_sha16={st['prompted_prefix_sha16']} "
                  f"config={st['config']} bank_view={st['bank_view']}")
            if not st["own_route"]:
                refusals.append(f"REFUSED: row {row['index']} ({row['cell']}) has no route of its own in the fresh table")
            if row["fitted"]:
                if st["stamp_for_cell"] == ["all"]:
                    print(f"SERVED_POSITIONS: row={row['index']} lens_served_positions[{row['harness_cell']}] == \"all\"")
                elif a.stage == "prod" or st["lens_served_positions"] is None:
                    msg = (f"REFUSED: served-positions stamp absent — stale table or refinement not merged (row {row['index']})"
                           if st["lens_served_positions"] is None else
                           f"REFUSED: lens_served_positions[{row['harness_cell']}] = {st['stamp_for_cell']!r}, not ['all'] (row {row['index']})")
                    print(msg)
                    if a.stage == "prod":
                        refusals.append(msg)
        else:
            print(f"FRESH_TABLE: row={row['index']} not exported ({st['why']})")
            if a.stage == "prod":
                refusals.append(f"REFUSED: --stage prod requires the fresh-table check ({st['why']}) — row {row['index']}")
        print(f"STAMPS: row={row['index']} cell={tuple(row['expected_stamps'])}"
              + (f" twin={tuple(row['twin_expected_stamps'])}" if row["fitted"] else ""))
        print(f"ROUTE_KIND: row={row['index']} {row['route_kind']}")
        print(f"ROW_FACTS: row={row['index']} wave={row['wave']} status={row['status']} thinking={row['thinking_mode']} "
              f"geometry={row['geometry_eval']}/{row['geometry_L']} budget_class={row['budget_class']} "
              f"log_requests={int(bool(row['log_requests']))} staged_id={row['staged_id']} si_sha256={row['si_sha256']}")
        if row["fitted"]:
            print(f"TUPLE: row={row['index']} {row['tuple_name']}")
        print()
    print(g_lens_status(a.petri_smoke_record))
    print("COUNTS: " + json.dumps(M.counts(rows), sort_keys=True))
    for r in refusals:
        print(r)
    print(f"VERDICT: {'PASS' if not refusals else 'REFUSED'} stage={a.stage} refusals={len(refusals)}")
    return 0 if not refusals else 2


if __name__ == "__main__":
    raise SystemExit(main())
