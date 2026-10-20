#!/usr/bin/env python3
"""The TWO-STAGE DRY RUN of the gsm8k-paired200-paper-objects launch. No
GPU, no venv build, no bank write. Per matrix row it prints:

  * the resolved argv of standup extras, ledger and driver — VERBATIM from the sbatch's own shell functions
    (g8k_common.sh g8k_print_plan, run under G8K_DRY_RUN=1: the same code path the job runs, not a re-typing);
  * the rung's geometry entry `x <S>` (S resolved through the real geometry table for the model), `cap 8192` (or the
    rung's 16384), `thinking on`, the sampling row and its source, `--upstream-timeout-s 7200`;
  * `HELD: index=… job_id=… reason=…` and NO argv for a held row — a held row with a DRIVER: line is a refusal;
  * `EXPECTED_REFUSAL:` by name for any row the standup is expected to refuse — NONE today (gemma's two
    header-located cells are servable thinking-ON on the declared path; the line stays as a capability);
  * `lens_served_positions[<standup harness cell>]` read from a FRESHLY EXPORTED routing table over the row's VIEW's bank
    (`<bank_root>/<key>/cells_view/<view>`), exported through the standup's OWN cell selection under the row's ONE standup
    spelling (`standup_cell`: `<cell>@thinking` on a declared row, so the table's key is the variant's harness id
    `si_…@thinking` — the plain harness cell is NOT in that table) against the row's EXPECTED positions
    (`all` / `response`); absent or disagreeing => REFUSED at --stage prod;
  * the family's render fact and the preflight seam's verdict at BOTH stages (as the MATH-500 dry run: `pre` asserts the
    seam REFUSES a candidate fact, `prod` demands it ADMIT every family it plans);
  * the tuple names per fitted row (both rungs), the expected stamps, the route kind, the G-LENS status line.

THE VIEW INTERFACE, END TO END: `--views <spec> --configs-root <tree>` assembles a SCRATCH tree under --tmp from THIS
tree's launcher code + the named tree's roster, pins and campaign configs / manifests, generates that spec's matrix and
overlays there through the ONE selector (make_gsm8k_matrix.py / make_gsm8k_overlays.py), and drives the SAME sbatch argv
construction over it (G8K_REPO_DIR = the scratch tree) — so a peer's dose view (`gemma4e4b=it_d0p1` at their sha) is
printed as a production row with `cells_view/it_d0p1/<cell>` and its pin, and nothing of theirs is written.

Every count the acceptance greps read is DERIVED from the matrix (g8k_cells.counts) and printed as a COUNTS line. Exit 0 = the
stage passes; 2 = a refusal (printed with REFUSED:); nothing GPU-side depends on this script.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import g8k_cells as M
import itdoc_bind as B

STAGES = ("pre", "prod")
SBATCH_REL = "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch"
OVERLAP_SBATCH_REL = "scripts/sdfbase/gsm8k/overlap_off.sbatch"
#: the launcher files a scratch tree needs from THIS tree (the peer's tree supplies the configs)
LAUNCHER_FILES = ("scripts/sdfbase/full22/common.sh", "scripts/sdfbase/full22/served_thinking.sh",
                  "scripts/sdfbase/full22/f22_roster.py",
                  "scripts/sdfbase/full22/serving_path_select.py",   # launch-cells-wiring: the dry-run branch runs f22_select_serving_path over the
                                                                     # row's declared cells (the path f22_init fixes) — the selector must be on the tree
                  
                  "scripts/sdfbase/full22/itdoc_bind.py",          # E3: g8k_cells imports the it-doc binding module (stdlib) at import
                  "scripts/sdfbase/seedrep/eval_seed.py",          # qwen-base-seed-replicates: g8k_views imports the replicate seed rule (stdlib) at import
                  "scripts/sdfbase/full22/steer_store_pin.py",     # arditi-native-pin: g8k_cells imports the store-following pin (stdlib) at import
                  # …and the launcher's admitted store-pins records, the pin's PINS source (this launcher's, not the peer's: a scratch row
                  # whose leg the records admit reads its recorded engine, exactly as on a checkout)
                  "configs/sdfbase/gsm8k/store_pins_default.json", "configs/sdfbase/gsm8k/store_pins_llama.json", "configs/sdfbase/gsm8k/store_pins_itdoc.json",
                  "configs/sdfbase/gsm8k/store_pins_arditi_allpos.json",
                  "scripts/sdfbase/gsm8k/g8k_cells.py", "scripts/sdfbase/gsm8k/g8k_views.py",
                  "scripts/sdfbase/gsm8k/g8k_common.sh", "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch",
                  "scripts/sdfbase/gsm8k/overlap_off.sbatch", "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py",
                  "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py")


def _resolved_s(model_key: str, geometry_eval: str, want_l: int, *, refusal_tail: str = "") -> int:
    """S the standup will derive for this model at `geometry_eval` (the real table; S is never passed to the standup —
    `strict_identity`); refuses by name when the entry does not resolve to the L the launcher expects."""
    from capctrl_serving_up import resolve_standup_geometry
    geom = resolve_standup_geometry(geometry_eval, None, None, model_key=model_key)
    if int(geom.max_model_len) != want_l:
        raise SystemExit(f"REFUSED: {geometry_eval} resolves L={geom.max_model_len}, not {want_l}{refusal_tail}")
    return int(geom.max_num_seqs)


def resolved_s(model_key: str, rung: int, cell: str = M.CELL_NONE, geometry_class: "str | None" = None) -> int:
    """S at the rung's geometry entry for the cell's PROMPT CLASS and the model's FAMILY SCOPE (g8k_cells.geometry_for_cell: the
    shared entries for the short class; the constitution prompted cell's long-SI entries — the Llama family's, or the four full22
    families' `_f22` ones, E2 / P3) — or, on a matrix that declares a `geometry_class` (qwen-seedrep-native-battery), the class's
    common entry for every cell."""
    ev, want_l = M.geometry_for_cell(cell, rung, model_key=model_key, geometry_class=geometry_class)
    return _resolved_s(model_key, ev, want_l, refusal_tail=" (cells.geometry_for_cell)")


def _s_key(row: dict, rung: int) -> tuple:
    """The S-table key of a row at a rung: (model, rung, the prompt-class representative cell, the row's geometry class)."""
    rep = row["cell"] if M.prompt_class(row["cell"]) != M.PROMPT_CLASS_SHORT else M.CELL_NONE
    return (row["model_key"], rung, rep, M.row_geometry_class(row))


def resolved_s_overlap(model_key: str) -> int:
    """S at the overlap leg's geometry entry."""
    return _resolved_s(model_key, M.OVERLAP_GEOMETRY_EVAL, M.OVERLAP_GEOMETRY_L)


def _dry_env(*, s: int, shared: Path, python: str, repo_dir: Path, cluster: str) -> dict:
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": str(s), "G8K_DRY_SHARED": str(shared), "G8K_DRY_CPY": python,
           "G8K_REPO_DIR": str(repo_dir), "G8K_DRY_CLUSTER": cluster, "USER": os.environ.get("USER", "tester")}
    # math500-steer: each row's dry run starts with NO serving path in its environment (the real single-row job's situation), so the
    # ROW's own pin selects the route (g8k_select_serving_path) and the enclosing prod dry run's resolved default never reads as a conflict
    env.pop("F22_SERVING_PATH", None)
    return env


def _shell_plan(argv: list[str], *, s: int, shared: Path, python: str, repo_dir: Path, cluster: str) -> tuple[str, int]:
    """One sbatch's OWN dry-run printing (g8k_print_plan under G8K_DRY_RUN=1): the text (log lines stripped) and the
    exit status — a HELD row exits 0 with only its HELD line, a misplaced row exits 1 with its FATAL / PLACEMENT lines."""
    r = subprocess.run(argv, capture_output=True, text=True, check=False, env=_dry_env(s=s, shared=shared, python=python, repo_dir=repo_dir, cluster=cluster))
    text = "\n".join(line for line in r.stdout.splitlines() if not line.startswith("["))
    if r.returncode != 0:
        text += "\n" + "\n".join(line for line in (r.stdout + r.stderr).splitlines() if "FATAL" in line or "PLACEMENT" in line)
    return text, r.returncode


def shell_plan(row: dict, *, rung: int, s: int, shared: Path, python: str, repo_dir: Path, cluster: str,
               matrix: str = M.MATRIX_DEFAULT_NAME) -> tuple[str, int]:
    """gsm8k_thinking.sbatch's dry-run printing for one row at one rung (the matrix as the sbatch's trailing positional — omitted
    on the default matrix, the launched grammar)."""
    argv = ["bash", str(repo_dir / SBATCH_REL), row["model_key"], row["view"], row["cell"], str(rung), "0" * 40,
            "1" if row["fitted"] else "0", "1" if row["declared"] else "0"] + ([] if matrix == M.MATRIX_DEFAULT_NAME else [matrix])
    return _shell_plan(argv, s=s, shared=shared, python=python, repo_dir=repo_dir, cluster=cluster)


def shell_plan_overlap(leg: dict, *, s: int, shared: Path, python: str, repo_dir: Path, cluster: str) -> tuple[str, int]:
    """overlap_off.sbatch's dry-run printing for one model's leg."""
    return _shell_plan(["bash", str(repo_dir / OVERLAP_SBATCH_REL), leg["model_key"], "0" * 40],
                       s=s, shared=shared, python=python, repo_dir=repo_dir, cluster=cluster)


def render_fact(family: str) -> dict[str, Any]:
    """The family's render fact from the contracts table — ABSENT is recorded (never invented)."""
    try:
        from sisteer.capeval.tinycap import contracts as C
    except Exception as e:  # recorded, never invented
        return {"present": False, "why": f"contracts import failed: {type(e).__name__}: {e}"}
    facts = getattr(C, "FAMILY_RENDER_FACTS", None)
    if not isinstance(facts, dict) or family not in facts:
        return {"present": False, "why": "contracts.FAMILY_RENDER_FACTS lacks this family"}
    f = facts[family]
    get = (lambda k: getattr(f, k, None)) if not isinstance(f, dict) else (lambda k: f.get(k))
    return {"present": True, "split_mode": get("split_mode"), "evidence_status": get("evidence_status")}


def render_fact_seam_verdict(family: str) -> dict[str, Any]:
    """The verdict of the driver's render-fact preflight seam, `contracts.check_render_fact_certified(family)`."""
    try:
        from sisteer.capeval.tinycap import contracts as C
        fn = getattr(C, "check_render_fact_certified", None)
    except Exception:
        fn = None
    if fn is not None:
        try:
            fn(family)
            return {"seam": "contracts.check_render_fact_certified", "refuses": False}
        except Exception as e:  # the seam's own refusal, verbatim
            return {"seam": "contracts.check_render_fact_certified", "refuses": True, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"seam": None, "refuses": None, "why": "no render-fact preflight seam on this tree — the assertion cannot run"}


@functools.lru_cache(maxsize=None)
def _mt_serving_up() -> Any:
    """scripts/sdfbase/mt_serving_up.py as a module, executed ONCE per process (it is a script, not a package module;
    every launched row's fresh-table stamp reads its select_cells through this one object)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_g8k_dry_mt", REPO / "scripts/sdfbase/mt_serving_up.py")
    mt = importlib.util.module_from_spec(spec)
    sys.modules["_g8k_dry_mt"] = mt
    spec.loader.exec_module(mt)
    return mt


def external_bank(row: dict, shared_root: Path | None, cluster: str) -> tuple[Path | None, str | None]:
    """A steering row's EXTERNAL bank view for the dry run's cluster role: `<shared root>/<bank_view_rel[role]>`,
    or why it cannot be named — no --shared-root, or no view for this role (the object is staged on the other cluster alone)."""
    role = "overflow" if cluster != M.PRIMARY_CLUSTER else "primary"
    rel = (row.get("bank_view_rel") or {}).get(role)
    if not rel:
        return None, f"row {row['index']} ({row['job_id']}) names no bank view for the {role} cluster role — its object is staged on the other cluster alone"
    if shared_root is None:
        return None, "no --shared-root: an external row's bank view is named relative to the shared root"
    return Path(shared_root) / rel, None


def fresh_table_stamp(row: dict, bank_root: Path | None, tmp: Path, repo_dir: Path, *, shared_root: Path | None = None,
                      cluster: str = M.PRIMARY_CLUSTER) -> dict[str, Any]:
    """The stamp the standup's OWN cell selection + export_routing_table write for this row, from a FRESH export over the
    bank view of the row's VIEW (`<bank_root>/<key>/cells_view/<view>`); None when no bank is given. A steering row
    exports over its EXTERNAL lane's view (`<shared root>/<bank_view_rel[role]>`) through its OWN overlay pair."""
    if bank_root is None and row.get("object_source") not in M.OBJECT_SOURCES:
        return {"exported": False, "why": "no --bank-root: the fresh-table check runs where the bank views exist"}
    import dataclasses

    import yaml
    mt = _mt_serving_up()
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving.proxy import export_routing_table, load_routing_table
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
    key, view, tag = row["model_key"], row["view"], row["tag"]
    if row.get("object_source") == M.OBJECT_SOURCE_EXTERNAL:
        # the row's OWN overlay pair (explicit_cells [none, <cell>]) over the lane's view. WHICH pair is the launcher's rule,
        # mirrored: `g8k_resolve_config` reads the TAG-keyed pair for a row of a `row_overlays` matrix (its first branch,
        # G8K_OWN_OVERLAY) and the model-keyed steering pair otherwise (make_gsm8k_overlays writes exactly these two) — the
        # all-positions Arditi row is the first external row that is also own-overlay, and exporting the model-keyed pair for it
        # would read the `main` campaign body for a row whose rendering is the `it` TAG's (arditi-allpos-cap, 2026-09-17).
        stem = tag if row.get("own_overlay") else key
        cfg_path = repo_dir / f"configs/sdfbase/gsm8k/sdfbase_full22_{stem}.gsm8k_{row['cell_slug']}.yaml"
        man_path = repo_dir / f"configs/capeval/manifests/petri_full22_{stem}.gsm8k_{row['cell_slug']}.yaml"
        bank, why = external_bank(row, shared_root, cluster)
        if bank is None:
            return {"exported": False, "why": why}
    elif row.get("object_source") == M.OBJECT_SOURCE_ITDOC:
        # E3: the it-doc row's OWN pair (the `it` TAG's, explicit_cells [none, <cell>]) over the bank entry its record names —
        # NEVER the `it` TAG overlay over the roster's view (that would export the superseded batch-8 object under this row's name)
        cfg_path = repo_dir / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k_{row['cell_slug']}.yaml"
        man_path = repo_dir / f"configs/capeval/manifests/petri_full22_{tag}.gsm8k_{row['cell_slug']}.yaml"
        if row.get("object_sha256") is None or not row.get("bank_view_rel"):
            return {"exported": False, "why": f"it-doc row ({row['cell']}, lane {row.get('lane')}) is not bound to a bank yet ({row.get('held_reason')}) — nothing to export"}
        bank, why = external_bank(row, shared_root, cluster)
        if bank is None:
            return {"exported": False, "why": why}
    elif row.get("own_overlay"):
        # E3: a row of a `row_overlays` matrix (the instruct-direct wave's served batch-8 constitution cell) exports through
        # its OWN pair over the CAMPAIGN bank of its view (the roster-served view the TAG branch below reads) — never the published
        # matrix's TAG overlay. The bank binding is the same as the TAG branch's; only the pair differs.
        cfg_path = repo_dir / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k_{row['cell_slug']}.yaml"
        man_path = repo_dir / f"configs/capeval/manifests/petri_full22_{tag}.gsm8k_{row['cell_slug']}.yaml"
        bank = Path(bank_root) / key / "cells_view" / view
    else:
        cfg_path = repo_dir / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k.yaml"
        man_path = repo_dir / f"configs/capeval/manifests/petri_full22_{tag}.gsm8k.yaml"
        bank = Path(bank_root) / key / "cells_view" / view
    cfg = yaml.safe_load(cfg_path.read_text())
    spec0 = cm.load_petri_spec(man_path)
    # the row's ONE standup spelling (the matrix's `standup_cell`, contracts.cell_spellings at generation): a declared row
    # exports `<cell>@thinking`, whose table key is `standup_harness_cell` (review B-P1.1 — the plain cell masked the miss)
    tt_cells = mt.select_cells(cfg["explicit_cells"], [row["standup_cell"]])
    harness = [ra.harness_cell_id(c) for c in tt_cells]
    spec_ = dataclasses.replace(spec0, serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"], cells=harness)
    src = traintest_cell_source(spec_, repo_root=REPO, config_path=cfg_path, defenses_dir=bank)
    bf = {k: (REPO / "configs/capeval/petri" / k / "BEHAVIOR.md").read_bytes() for k in cm.petri_expected_behavior_keys(spec_)}
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
    key_cell = row["standup_harness_cell"]
    return {"exported": True, "bank": str(bank), "routes": sorted(routes), "lens_served_positions": stamp, "stamp_key": key_cell,
            "own_route": key_cell in routes, "prompted_prefix_sha16": prefixes.get("petri__prompted__safety"),
            "stamp_for_cell": (stamp or {}).get(key_cell) if isinstance(stamp, dict) else None}


def g_lens_status(petri_record: str | None) -> str:
    import inspect

    from sisteer.capeval.serving.proxy import RoutingProxy
    merged = "lens_served_positions" in inspect.signature(RoutingProxy.__init__).parameters
    if not merged:
        return "G-LENS: NOT MERGED — fitted jobs held (RoutingProxy takes no lens_served_positions)"
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        sha = "unknown"
    return f"G-LENS: refinement merged at {sha[:12]} (RoutingProxy takes lens_served_positions), Petri smoke {petri_record or '<record path not given>'}"


def scratch_tree(*, configs_root: Path, views_spec: str, tmp: Path) -> Path:
    """A scratch tree = THIS tree's launcher code + the named tree's roster / pins / campaign files, with the spec's matrix and
    overlays generated in place through the ONE selector (run by THIS interpreter — the generators need yaml + the harness's
    cell-id spelling; the client-venv python the argv names is a different thing). Nothing is written under `configs_root`."""
    python = sys.executable
    tree = tmp / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    for rel in LAUNCHER_FILES:
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tree / rel)
    # THIS tree's src, linked (never copied): the selector's import chain (sisteer.capeval manifest / registry_adapter / defense_cells)
    # is reached as f22_select_auto reaches it on a checkout — PYTHONPATH=<tree>/src — and a scratch tree is not a checkout
    (tree / "src").symlink_to(REPO / "src", target_is_directory=True)
    (tree / "configs/capeval/manifests").mkdir(parents=True, exist_ok=True)
    (tree / "configs/sdfbase/gsm8k").mkdir(parents=True, exist_ok=True)
    for rel in ("configs/sdfbase_full22_roster.yaml", "configs/sdfbase_full22_pins.json"):
        src = configs_root / rel
        if not src.is_file():
            raise SystemExit(f"REFUSED: --configs-root {configs_root} carries no {rel}")
        shutil.copy(src, tree / rel)
    r = subprocess.run([python, str(tree / "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py"), "--views", views_spec,
                        "--configs-root", str(tree), "--out", str(tree / "configs/sdfbase/gsm8k/matrix.json")],
                       capture_output=True, text=True, check=False, env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{os.environ.get('PYTHONPATH', '')}"})
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: the matrix for --views {views_spec!r} over {configs_root} did not generate: {r.stdout[-600:]}{r.stderr[-600:]}")
    doc = M.load_matrix(tree / "configs/sdfbase/gsm8k/matrix.json")
    for tag in dict.fromkeys(row["tag"] for row in doc["rows"]):
        for rel in (f"configs/sdfbase_full22_{tag}.yaml", f"configs/capeval/manifests/petri_full22_{tag}.yaml"):
            src = configs_root / rel
            if not src.is_file():
                raise SystemExit(f"REFUSED: --configs-root {configs_root} carries no {rel} (the campaign file the TAG {tag} overlay is cut from)")
            shutil.copy(src, tree / rel)
    r = subprocess.run([python, str(tree / "scripts/sdfbase/gsm8k/make_gsm8k_overlays.py"), "--configs-root", str(tree),
                        "--matrix", str(tree / "configs/sdfbase/gsm8k/matrix.json")], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: the overlays for --views {views_spec!r} did not generate: {r.stdout[-600:]}{r.stderr[-600:]}")
    return tree


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--sha", default="0" * 40, help="the wave sha the argv names (cosmetic here)")
    ap.add_argument("--cluster", default=M.PRIMARY_CLUSTER, choices=sorted(M.CLUSTERS),
                    help="the cluster the rows are dry-run FOR: a `primary` row on the overflow cluster is printed as PLACEMENT-refused")
    ap.add_argument("--bank-root", default=None, help="<root>/<key>/cells_view/<view> per (key, view): enables the fresh-table stamp check")
    ap.add_argument("--shared-root", default=None, help="the shared root a W4 steering row's EXTERNAL bank view is named relative to (E2 / P8); "
                                                        "derived from --bank-root when it is <shared root>/exp/objects/sdfbase/full22")
    ap.add_argument("--petri-smoke-record", default=None, help="Evals' Petri smoke record path for the G-LENS line")
    ap.add_argument("--only", nargs=3, metavar=("KEY", "VIEW", "CELL"), default=None)
    ap.add_argument("--rows", default=None, help="a Slurm index spec of matrix rows to print (default: every row)")
    ap.add_argument("--rung-rows", default=None, help="a Slurm index spec of rows ALSO printed as 16,384 rung rows (a synthetic subset path)")
    ap.add_argument("--overlap", action="store_true", help="also print the four overlap legs (overlap_off.sbatch)")
    ap.add_argument("--views", default=None, help="a views spec (key=view,…;…): dry-run THAT matrix from --configs-root in a scratch tree")
    ap.add_argument("--configs-root", type=Path, default=None, help="the tree whose roster / pins / campaign files the --views matrix is cut from")
    ap.add_argument("--matrix", default=None, help=f"the matrix: a name ({', '.join(M.MATRICES)}, or {M.FRONT_DOOR_MATRIX_PREFIX}<models.yaml key> — the front "
                    "door's derived matrix) or a matrix.json path (default: this tree's frozen published one)")
    ap.add_argument("--python", default=sys.executable, help="the client-venv python the argv names")
    ap.add_argument("--tmp", default="/tmp/g8k-dry", help="scratch root for the printed paths, fresh tables and the scratch tree")
    a = ap.parse_args(argv)
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    if (a.views is None) != (a.configs_root is None):
        raise SystemExit("--views and --configs-root go together")
    if a.views is not None:
        repo_dir = scratch_tree(configs_root=a.configs_root.resolve(), views_spec=a.views, tmp=tmp)
        matrix_path = repo_dir / "configs/sdfbase/gsm8k/matrix.json"
        print(f"SCRATCH_TREE: {repo_dir} (launcher code from this tree; roster / pins / campaign files from {a.configs_root})")
    else:
        repo_dir = REPO
        matrix_path = M.resolve_matrix(a.matrix)
    matrix_name = M.matrix_name_of(matrix_path)
    if a.views is None and matrix_name not in M.MATRICES and matrix_name not in M.front_door_matrices():
        # paper-companion-repo (the fix-1 owner follow-up 6): a front-door model's derived matrix `frontdoor_<key>` is registered by NAME
        # beside MATRICES (g8k_cells.front_door_matrices; every launcher's `case` admits it) and dry-runs like the frozen ones
        raise SystemExit(f"REFUSED: --matrix {matrix_path} is not a registered matrix ({sorted(M.MATRICES)} or {sorted(M.front_door_matrices())}); "
                         "the sbatch takes the matrix by NAME")
    if a.views is not None:
        matrix_name = M.MATRIX_DEFAULT_NAME      # the scratch tree's matrix IS its default (configs/sdfbase/gsm8k/matrix.json)
    doc_all = M.load_matrix(matrix_path)
    rows_all = doc_all["rows"]
    if a.only is not None:
        rows = [M.row_for(*a.only, rows=rows_all)]
    elif a.rows is not None:
        rows = [rows_all[i] for i in M.parse_index_spec(a.rows, len(rows_all))]
    else:
        rows = rows_all
    rung_idx = set(M.parse_index_spec(a.rung_rows, len(rows_all))) if a.rung_rows else set()
    refusals: list[str] = []
    shared_root = Path(a.shared_root) if a.shared_root else None
    if shared_root is None and a.bank_root and Path(a.bank_root).parts[-4:] == ("exp", "objects", "sdfbase", "full22"):
        shared_root = Path(a.bank_root).parents[3]
    s_by = {key: resolved_s(key[0], key[1], key[2], key[3]) for key in sorted({_s_key(row, r) for row in rows for r in M.RUNGS}, key=str)}
    print(f"STAGE: {a.stage} rows={len(rows)} sha={a.sha} cluster={a.cluster} matrix={matrix_path} matrix_name={matrix_name} "
          f"schema={doc_all['schema']} wave_root_rel={M.wave_root_rel_of(doc_all)} thinking_modes={sorted({M.row_thinking(r) for r in rows})}")
    facts = {fam: render_fact(fam) for fam in sorted({row["family"] for row in rows})}
    for fam, f in facts.items():
        status = f.get("evidence_status") if f["present"] else "absent"
        print(f"RENDER_FACT: family={fam} split_mode={f.get('split_mode') if f['present'] else 'absent'} evidence_status={status}"
              + ("" if f["present"] else f" ({f['why']})"))
        pf = render_fact_seam_verdict(fam)
        if a.stage == "pre":
            print(f"PREFLIGHT_REFUSES_CANDIDATE: family={fam} seam={pf['seam']} refuses={pf['refuses']}" + (f" ({pf['why']})" if pf.get("why") else ""))
            if pf["refuses"] is False and status != "certified":
                refusals.append(f"REFUSED: the production preflight ADMITS family {fam} with evidence_status={status} at --stage pre")
        else:
            admits = pf["refuses"] is False
            print(f"PREFLIGHT_ADMITS: family={fam} seam={pf['seam']} admits={admits}" + (f" ({pf['why']})" if pf.get("why") else ""))
            if not admits:
                refusals.append(f"REFUSED: render fact not certified for family {fam} (evidence_status={status}): " + (pf.get("why") or "the production preflight seam did not admit it"))
    shared = tmp / "shared"
    for row in rows:
        misplaced = row["cluster_constraint"] == M.PLACEMENT_PRIMARY and a.cluster != M.PRIMARY_CLUSTER
        for rung in (M.CAP,) + ((M.RUNG_CAP,) if row["index"] in rung_idx else ()):
            text, rc = shell_plan(row, rung=rung, s=s_by[_s_key(row, rung)],
                                  shared=shared, python=a.python, repo_dir=repo_dir,
                                  cluster=a.cluster, matrix=matrix_name)
            print(text)
            if row["status"] == M.STATUS_DEFERRED:
                if rc == 0 and ("DRIVER:" in text or "STANDUP:" in text):
                    refusals.append(f"REFUSED: deferred row {row['index']} ({row['job_id']}) printed argv — a design-only row is never a job")
                continue
            if row["status"] == M.STATUS_HELD:
                if rc != 0 or "HELD:" not in text or "DRIVER:" in text or "STANDUP:" in text or "RESOLVED_SAMPLER:" in text:
                    refusals.append(f"REFUSED: held row {row['index']} ({row['job_id']}) did not exit HELD with no argv (rc={rc})")
                continue
            if misplaced:
                if rc == 0 or "PLACEMENT" not in text:
                    refusals.append(f"REFUSED: row {row['index']} ({row['job_id']}) is `primary` but the dry run on {a.cluster} did not refuse it by name")
                else:
                    print(f"PLACEMENT: row={row['index']} {row['job_id']} refused on {a.cluster} by name (primary-only) — as designed")
                continue
            if rc != 0:
                refusals.append(f"REFUSED: the sbatch dry run failed for row {row['index']} rung {rung}: {text[-600:]}")
                continue
            if "DRIVER:" not in text or "STANDUP:" not in text or "RESOLVED_SAMPLER:" not in text:
                refusals.append(f"REFUSED: row {row['index']} rung {rung} printed no argv (DRIVER / STANDUP / RESOLVED_SAMPLER)")
            if "--enable-log-requests" not in text.split("STANDUP:", 1)[1].split("\n", 1)[0]:
                refusals.append(f"REFUSED: row {row['index']} rung {rung}: the standup argv carries no --enable-log-requests (the resolved-sampler attestation needs it)")
        if row["status"] in (M.STATUS_HELD, M.STATUS_DEFERRED) or misplaced:
            print()
            continue
        st = fresh_table_stamp(row, Path(a.bank_root) if a.bank_root else None, tmp, repo_dir, shared_root=shared_root, cluster=a.cluster)
        if st["exported"]:
            print(f"FRESH_TABLE: row={row['index']} bank={st['bank']} routes={st['routes']} own_route={st['own_route']} "
                  f"lens_served_positions={json.dumps(st['lens_served_positions'])} prompted_prefix_sha16={st['prompted_prefix_sha16']}")
            if not st["own_route"]:
                refusals.append(f"REFUSED: row {row['index']} ({row['cell']}) has no route of its own in the fresh table")
            if row["fitted"]:
                want = row["served_positions_expected"]
                if st["stamp_for_cell"] == [want]:
                    print(f"SERVED_POSITIONS: row={row['index']} lens_served_positions[{st['stamp_key']}] == \"{want}\" (expected)")
                elif a.stage == "prod" or st["lens_served_positions"] is None:
                    msg = (f"REFUSED: served-positions stamp absent — stale table or refinement not merged (row {row['index']})"
                           if st["lens_served_positions"] is None else
                           f"REFUSED: lens_served_positions[{st['stamp_key']}] = {st['stamp_for_cell']!r}, not ['{want}'] "
                           f"(row {row['index']}: served_positions_expected={want})")
                    print(msg)
                    if a.stage == "prod":
                        refusals.append(msg)
        else:
            print(f"FRESH_TABLE: row={row['index']} not exported ({st['why']})")
            if a.stage == "prod":
                refusals.append(f"REFUSED: --stage prod requires the fresh-table check (give --bank-root) — row {row['index']}")
        print(f"STAMPS: row={row['index']} cell={tuple(row['expected_stamps'])}")
        print(f"ROUTE_KIND: row={row['index']} {row['route_kind']}")
        print(f"WAVE_STATUS: row={row['index']} wave={M.row_wave(row)} status={row['status']} object_source={row.get('object_source') or 'none'}")
        if row.get("object_source") == M.OBJECT_SOURCE_ITDOC:
            # E3: the bank BINDING verdict per it-doc row (itdoc_bind.row_bind: manifest entry + bytes + sidecar of the kind + the pin
            # file's (target, staged id) against the record's pin; the lane named in every refusal) — PASS printed explicitly, and at
            # --stage prod a REFUSED binding refuses the run
            role = "overflow" if a.cluster != M.PRIMARY_CLUSTER else "primary"
            if shared_root is None:
                print(f"ITDOC_BIND: row={row['index']} not checked (no --shared-root: the bank entry is named relative to the shared root)")
                if a.stage == "prod":
                    refusals.append(f"REFUSED: --stage prod requires the it-doc bank binding check (give --shared-root) — row {row['index']}")
            else:
                pins_doc = json.loads((repo_dir / "configs/sdfbase_full22_pins.json").read_text(encoding="utf-8"))
                rec = B.row_bind(row, shared_root, cluster_role=role, pins_doc=pins_doc)
                print(f"row={row['index']} " + B.bind_verdict_line(rec))
                if rec["verdict"] != B.VERDICT_PASS and a.stage == "prod":
                    refusals.append(f"REFUSED: row {row['index']} ({row['job_id']}) it-doc bank binding {rec['verdict']}: " + "; ".join(rec["problems"]))
        print(f"THINKING_MODE: row={row['index']} {M.row_thinking(row)} sampling_mode={M.row_sampling_mode(row)}")
        print(f"TRIGGER: row={row['index']} threshold={M.row_trigger_threshold(row)} "
              f"({'declared by the schema-2 matrix' if 'trigger_threshold' in row else 'the published matrix: the adapter default'})")
        if row["fitted"] and row["tuple_names"]:
            print(f"TUPLE: row={row['index']} {row['tuple_names'][str(M.CAP)]} rung={row['tuple_names'][str(M.RUNG_CAP)]}")
        elif row["fitted"]:
            print(f"TUPLE: row={row['index']} none (a {M.row_thinking(row)} fitted row is in support: no out-of-support tuple, no override)")
        print()
    if a.overlap:
        for leg in M.overlap_legs():
            misplaced = leg["cluster_constraint"] == M.PLACEMENT_PRIMARY and a.cluster != M.PRIMARY_CLUSTER
            text, rc = shell_plan_overlap(leg, s=resolved_s_overlap(leg["model_key"]), shared=shared, python=a.python, repo_dir=repo_dir, cluster=a.cluster)
            print(text)
            if misplaced:
                if rc == 0 or "PLACEMENT" not in text:
                    refusals.append(f"REFUSED: overlap leg {leg['job_id']} is `primary` but the dry run on {a.cluster} did not refuse it")
                else:
                    print(f"PLACEMENT: overlap {leg['job_id']} refused on {a.cluster} by name (primary-only) — as designed")
            elif rc != 0 or "DRIVER:" not in text:
                refusals.append(f"REFUSED: the overlap dry run failed for {leg['job_id']}: {text[-600:]}")
            print()
    expected = [r for r in rows if r["status"] == M.STATUS_LAUNCH and r["expected_refusal"]]
    for r in expected:
        print(f"EXPECTED_REFUSAL: row={r['index']} {r['job_id']} — {r['expected_refusal_reason']}")
    held = [r for r in rows if r["status"] == M.STATUS_HELD]
    print(f"HELD_ROWS: {[(r['index'], r['job_id']) for r in held]}")
    for r in held:
        if r.get("object_source") == M.OBJECT_SOURCE_ITDOC:
            print(f"HELD_ITDOC: row={r['index']} {B.tuple_label(r['lane'], r['model_key'], r['view'], r['cell'])} staged_id={r['staged_id']} "
                  f"pin={(r.get('object_sha256') or 'null')[:12]} reason={r['held_reason']}")
    print(f"WAVES: {json.dumps({w: sum(1 for r in rows if M.row_wave(r) == w) for w in dict.fromkeys(M.row_wave(r) for r in rows)})} "
          f"complete={list(M.COMPLETED_WAVES)} (a complete wave's rows dry-run here but are refused by the batch boundary, g8k_assert_launchable)")
    print(g_lens_status(a.petri_smoke_record))
    print("COUNTS: " + json.dumps(M.counts(rows_all), sort_keys=True))
    if doc_all["schema"] == M.MATRIX_SCHEMA_2:
        print(f"WAVE_ROOT: {M.wave_root_rel_of(doc_all)} (schema {M.MATRIX_SCHEMA_2}: the matrix's own root, never the published wave's)")
    for r in refusals:
        print(r)
    print(f"VERDICT: {'PASS' if not refusals else 'REFUSED'} stage={a.stage} refusals={len(refusals)}")
    return 0 if not refusals else 2


if __name__ == "__main__":
    raise SystemExit(main())
