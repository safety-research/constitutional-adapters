#!/usr/bin/env python
"""Grade a GOAL-SUBSET multi-turn run with the campaign's live StrongREJECT
path, keeping per-goal best scores for a PAIRED comparison.

``scripts/grade_gated_served.py`` is the harness-verbatim grader for the gated
families, but it hard-requires a terminal progress record for EVERY
confirmatory goal (259) — correct for a full wave, impossible for a smoke that
replays one shard (29 goals). This driver is the same path with ONE change:
``expected_goal_ids`` is the replayed goal list (``--goal-ids-file``), so the
completeness gate is exact for the subset instead of impossible. Everything
that touches the scores is ``jailbreak_run._gated_defense_metrics`` /
``_summarize_gated`` / ``LiveStrongREJECTGrader`` verbatim (judge
``cfg['grader']['model']`` = claude-haiku-4-5, rubric v1, Low-Prio org — the
same grader the wave's numbers came from), so ASR is basis-identical.

Unlike the sanitized aggregate, ``per_goal`` (goal_id -> best_score,
n_target_forwards, queries_to_first_success) is KEPT in the output: it holds
goal ids and numbers only, and it is what the paired per-goal contrast against
the wave's ``<attack>_aggregate.json`` needs.

Serve VARIANTS: ``--cells`` accepts
``<cell>@all_turns`` / ``<cell>@all_positions`` / ``<cell>@thinking`` — each is
its own transcript directory and its own served identity, graded as its own
cell. A goal whose latest outcome is ``refused:<code>`` (the proxy's per-item
all-turns render check refused the conversation — the experiment's expected
outcome, not an incompleteness) is EXCLUDED unconditionally and counted by
code (``n_refused`` / ``refused_by_code`` / ``refused_goal_ids``);
``--allow-errored-goals`` governs ``error:*`` only. ``n_responses_graded`` is
the row count of the CLEAN mapping ``exclude_errored_goals`` returned (both
rules applied), never an independent subtraction; the ``outcomes`` histogram
counts the raw outcome strings, so ``refused:*`` shows up there by name.

GRADING PROVENANCE (fit-support-transfer code review S-A1). The output carries
an ADDITIVE, versioned ``grading_provenance`` block that binds the grades to
the run they were computed from: the schema literal, the sha256 of the run's
``fst_run_identity.json`` (``--run-identity``; default
``<private-root>/../fst_run_identity.json`` when present, else null), the
sha256 of the ``--goal-ids-file`` bytes, the sha256 of EVERY ``responses.jsonl``
/ ``progress.jsonl`` this grader read (per cell, family and shard — the exact
files ``load_gated_dir`` globbed), the grader model and where it came from,
the StrongREJECT rubric version, and ``dry_run``. Nothing that existed before
moves or changes meaning — the campaign's other waves read this file. The
experiment-2 analyser (``scripts/fst_exp2_analyze.py``) REQUIRES the block,
recomputes every digest, and refuses stale, dry-run or off-grader grades.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import grade_gated_served as ggs  # noqa: E402  (load_gated_dir)
import jailbreak_run as jr  # noqa: E402

from sisteer.eval.defense_cell import sha256_file  # noqa: E402

#: the ADDITIVE provenance block's schema literal (code review S-A1)
GRADING_SCHEMA = "fst_grading/1"
#: the run identity record the run script writes beside the pulled private
#: root (``scripts/modal/fst_multiturn.py`` IDENTITY_FILE): ``<run>/qwen`` is
#: the private root, ``<run>/fst_run_identity.json`` the record
IDENTITY_FILE = "fst_run_identity.json"
#: the StrongREJECT rubric this grader constructs (rubric v1 — the
#: grader's own default, passed explicitly so the output states what it used)
RUBRIC_VERSION = "v1"
#: the ``org`` literal the output has always carried for the Low-Prio pool, and
#: its High-Prio sibling — the campaign's other waves read this field
ORG_LABEL = {"lowprio": "low-prio", "highprio": "high-prio"}
#: the two per-shard files whose bytes the grades derive from
DIGESTED_FILES = (("responses.jsonl", "responses_sha256"),
                  ("progress.jsonl", "progress_sha256"))


def shard_digests(defense_root) -> dict[str, dict]:
    """sha256 of each ``shard*`` dir's ``responses.jsonl`` / ``progress.jsonl``
    under ``defense_root`` — the same glob ``load_gated_dir`` reads, so these
    are the bytes the grades came from; an absent file is ``null``."""
    out: dict[str, dict] = {}
    for sd in sorted(Path(defense_root).glob("shard*")):
        out[sd.name] = {key: (sha256_file(sd / name) if (sd / name).exists() else None)
                        for name, key in DIGESTED_FILES}
    return out


def cell_inputs(defense_root) -> dict:
    """One (cell, family) entry of ``grading_provenance.inputs``."""
    return {"defense_root": str(defense_root), "shards": shard_digests(defense_root)}


def default_run_identity_path(private_root) -> Path:
    return Path(private_root).parent / IDENTITY_FILE


def build_grading_provenance(*, inputs: dict, run_identity_path, grader_model: str,
                             grader_model_source: str, rubric_version,
                             dry_run: bool, goal_ids_file_sha256: str) -> dict:
    """The ``grading_provenance`` block (schema ``fst_grading/1``).
    ``run_identity_path`` is hashed when it exists and recorded ``null``
    otherwise (a run without the record is not experiment 2's);
    ``grader_model_source`` is ``config`` / ``override`` / ``dry_run``."""
    p = Path(run_identity_path) if run_identity_path is not None else None
    present = p is not None and p.exists()
    return {
        "schema": GRADING_SCHEMA,
        "run_identity_path": str(p) if present else None,
        "run_identity_sha256": sha256_file(p) if present else None,
        "goal_ids_file_sha256": goal_ids_file_sha256,
        "inputs": inputs,
        "grader_model": grader_model,
        "grader_model_source": grader_model_source,
        "rubric_version": rubric_version,
        "dry_run": bool(dry_run),
    }


DRIVER_ID = "scripts/grokmt_grade.py"


def registry_worklist_pairs(args, cfg: dict, root: Path, families: list):
    """``--worklist registry``: the (cell, family) pairs come from ``v_pending_judgment`` —
    multi-turn runs (``eval_category = multi_turn``) pending for ``--judge`` (default: each run's instrument
    judge) whose store is a gated root of THIS private run (``<private-root>/<private_run>/<family>/<cell>``);
    pending runs under other roots belong to other invocations and are counted, not graded. ``--cells`` /
    ``--families`` narrow the set. Returns ``(worklist, cells, {(cell, family): row}, n_other)``; the live path
    claims every pair before its first judge call (a dry run claims nothing)."""
    from sisteer.registry import worklist as RW
    wl = RW.RegistryWorklist(args.registry_root, args.judge, args.session, args.campaign, driver=DRIVER_ID, api_pool="lowprio")
    rows = wl.pending(eval_category="multi_turn")
    base = (Path(root) / cfg["private_run"]).expanduser().resolve()
    wanted = {c.strip() for c in args.cells.split(",") if c.strip()} if args.cells else None
    row_of: dict = {}
    for r in rows:
        sp = RW.store_path(r)
        d = Path(sp).expanduser().resolve() if sp else None
        if d is None or d.parent.parent != base:
            continue
        fam, cell = d.parent.name, d.name
        if fam not in families or (wanted is not None and cell not in wanted):
            continue
        row_of[(cell, fam)] = r
    cells = sorted({c for c, _ in row_of})
    print(f"[registry] worklist=registry: {len(row_of)} pending (cell, family) pair(s) under {base}; "
          f"{len(rows) - len(row_of)} pending run(s) belong to other roots or were not requested", flush=True)
    return wl, cells, row_of, len(rows) - len(row_of)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--private-root", required=True,
                    help="holds <private_run>/<attack>/<cell>/shard*/")
    ap.add_argument("--cells", default=None,
                    help="comma-separated DefenseCell ids (required with --worklist from-disk; with --worklist "
                         "registry the pending runs name them and --cells narrows); a served-only serve variant "
                         "'<cell>@all_turns' / '@all_positions' / '@thinking' names its "
                         "own transcript dir and is graded as its own cell")
    ap.add_argument("--families", default="pair,crescendo,actorattack")
    ap.add_argument("--goal-ids-file", required=True,
                    help="the replayed goal ids (one per line) — the completeness gate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True,
                    help="live score cache jsonl (kind 'sr'); resumes across runs")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--org", default="lowprio", choices=("lowprio", "highprio"),
                    help="the rate-limit pool the judge calls go to (same model, same "
                         "rubric; the pool is RECORDED in the output, never assumed). "
                         "Default lowprio — the campaign's routing default.")
    ap.add_argument("--grader-model", default=None,
                    help="override cfg['grader']['model'] (default: the config's)")
    ap.add_argument("--allow-unscored", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-errored-goals", action="store_true",
                    help="EXCLUDE goals whose latest progress outcome is error:* "
                         "(error:PromptOverWindow, error:ActorDiscoveryError, ...) "
                         "instead of refusing; the summary carries n_expected / "
                         "n_graded / n_errored by type. Default: refuse, name the "
                         "goals — resume the run to regenerate them.")
    ap.add_argument("--run-identity", default=None,
                    help="fst_run_identity.json whose sha256 the grading output binds to "
                         "(grading_provenance.run_identity_sha256); default: "
                         "<private-root>/../fst_run_identity.json when present, else null")
    from sisteer.registry import worklist as RW
    RW.add_worklist_arguments(ap)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    registry_wl = None
    registry_row_of: dict = {}
    if args.worklist == "registry":
        try:
            registry_wl, cells, registry_row_of, _ = registry_worklist_pairs(
                args, cfg, jr._private_root(cfg, args.private_root), families)
        except RW.RegistryUnavailable as e:
            raise SystemExit(str(e)) from None
        if not cells:
            raise SystemExit("--worklist registry: no pending (cell, family) pair under this private run; nothing to grade")
    else:
        RW.log_from_disk()
        if not args.cells:
            raise SystemExit("--cells is required with --worklist from-disk")
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    if not cells:
        raise SystemExit("--cells: no cell ids")
    # a malformed or unknown '@<variant>' suffix refuses here, before any file is
    # read (the vocabulary is closed: serve_declaration.SERVE_VARIANTS)
    for c in cells:
        jr._split_defense_id(c)
    if len(set(cells)) != len(cells):
        raise SystemExit(f"--cells: duplicate cell id in {cells}")
    cfg = {**cfg, "defenses": cells}

    goals_all = jr._eval_goals(cfg, confirmatory=True)
    goals = jr._select_goal_ids(goals_all, args.goal_ids_file)
    by_id = {g.goal_id: g for g in goals}
    expected_goal_ids = [g.goal_id for g in goals]
    # the category groups are derived over the FULL goal set (the held-out
    # taxonomy split is a property of the campaign, not of the subset)
    group_of, held_in, held_out, ch = jr._category_groups(cfg, goals_all)
    root = jr._private_root(cfg, args.private_root)

    ids_file_sha256 = sha256_file(Path(args.goal_ids_file))
    if args.run_identity:
        run_identity_path = Path(args.run_identity)
        if not run_identity_path.exists():
            raise SystemExit(f"--run-identity {run_identity_path}: no such file")
    else:
        run_identity_path = default_run_identity_path(root)
        if not run_identity_path.exists():
            print(f"[grade] no run identity record at {run_identity_path} — "
                  "grading_provenance.run_identity_sha256 will be null", flush=True)

    if args.dry_run:
        def grade_fn(items):
            return {k: 0.0 for k in items}
        model = "DRY-RUN-no-api"
        rubric_version = None
        model_source = "dry_run"
        grader = None
    else:
        from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader
        model = args.grader_model or cfg["grader"]["model"]
        model_source = "override" if args.grader_model else "config"
        grader = LiveStrongREJECTGrader(
            model=model, org=args.org, concurrency=args.concurrency,
            cache_path=Path(args.cache), allow_unscored=args.allow_unscored,
            rubric=RUBRIC_VERSION)
        rubric_version = grader.rubric
        grade_fn = grader.grade
        print(f"[grade] judge={model} org={ORG_LABEL[args.org]} pool={grader.pool_stamp} "
              f"concurrency={args.concurrency} "
              f"cache={args.cache} n_goals={len(expected_goal_ids)}", flush=True)

    thresholds = cfg["prereg"]["threshold_grid"]
    canonical = thresholds[len(thresholds) // 2]
    if registry_wl is not None:
        if args.dry_run:
            print(f"[registry] dry run: {len(registry_row_of)} pair(s) would be claimed; nothing claimed, nothing recorded", flush=True)
        else:
            try:
                registry_wl.claim_all([registry_row_of[(c, f)] for c in cells for f in families if (c, f) in registry_row_of])
            except (RW.P.ClaimHeld, RW.P.ClaimNotFolded) as e:   # every claim refusal -- held, not folded, not pending; claim_all released what it took
                raise SystemExit(f"--worklist registry: refusing — {e}") from None
    out_cells: dict = {}
    prov_inputs: dict = {}
    for cell in cells:
        out_fam: dict = {}
        for fam in families:
            if registry_wl is not None and (cell, fam) not in registry_row_of:
                continue     # not pending in the registry: not this invocation's work
            defense_root = jr._gated_defense_root(cfg, root, fam, cell)
            # digest the files BEFORE reading them: the block names the bytes
            # the grades came from (code review S-A1)
            prov_inputs.setdefault(cell, {})[fam] = cell_inputs(defense_root)
            responses, progress = ggs.load_gated_dir(defense_root)
            missing = [gid for gid in expected_goal_ids if gid not in progress]
            if missing:
                raise SystemExit(
                    f"[{fam}] {cell}: {len(missing)}/{len(expected_goal_ids)} replayed "
                    f"goals have no terminal progress record under {defense_root}: "
                    f"{missing[:5]} — refusing to grade partial work")
            extra = [gid for gid in progress if gid not in by_id]
            if extra:
                raise SystemExit(f"[{fam}] {cell}: progress holds {len(extra)} goals "
                                 f"outside the replayed set: {extra[:5]}")
            leaked: list[str] = []
            m = jr._gated_defense_metrics(
                responses, progress, by_id, grade_fn, thresholds=thresholds,
                canonical=canonical, leaked_needles=leaked,
                expected_goal_ids=expected_goal_ids, group_of=group_of,
                allow_errored_goals=args.allow_errored_goals, label=f"{fam}] {cell}")
            # n_responses_graded is the CLEAN mapping's row count as
            # _gated_defense_metrics computed it (errored AND refused goals
            # dropped by exclude_errored_goals) — read, never recomputed here
            assert "n_responses_graded" in m, sorted(m)
            m["outcomes"] = {}
            for gid, p in progress.items():
                m["outcomes"][str(p.get("outcome"))] = m["outcomes"].get(str(p.get("outcome")), 0) + 1
            out_fam[fam] = m
            if registry_wl is not None and not args.dry_run:
                scope = registry_wl.scope(registry_row_of[(cell, fam)], store_uri=str(defense_root))
                with scope:
                    scope.metrics(RW.gated_defense_metric_rows(m, thresholds))
                    scope.complete("complete", n_items_judged=int(m.get("n_responses_graded") or 0))
            asr = m["asr"]
            print(f"[grade] {cell:<28} {fam:<11} n={m['n_goals']:>2} "
                  f"asr@0.5={asr['0.5']:.4f} asr@0.75={asr['0.75']:.4f} "
                  f"mean_best={m['mean_best_score']:.4f} "
                  f"fwd/goal={m['mean_n_target_forwards']:.1f} "
                  f"resp={m['n_responses_graded']} "
                  f"errored={m['n_errored']} refused={m['n_refused']}"
                  + (f" {m['refused_by_code']}" if m["n_refused"] else ""), flush=True)
        out_cells[cell] = out_fam

    out = {"model": model, "org": ORG_LABEL[args.org], "thresholds": thresholds,
           # ADDITIVE: the pool the judge calls ACTUALLY went to (vault pool stamp
           # with the key's sha256 prefix, never the key), the launch concurrency,
           # the SDK retry count and the gate's 429 count — recorded, not assumed
           "grader_run_stats": (grader.run_stats() if grader is not None else None),
           "goal_ids": expected_goal_ids,
           "category_holdout": {"seed": ch["seed"], "frac": ch["frac"],
                                "held_in": held_in, "held_out": held_out},
           "cells": out_cells,
           # ADDITIVE (code review S-A1): binds these grades to the run
           "grading_provenance": build_grading_provenance(
               inputs=prov_inputs, run_identity_path=run_identity_path,
               grader_model=model, grader_model_source=model_source,
               rubric_version=rubric_version, dry_run=args.dry_run,
               goal_ids_file_sha256=ids_file_sha256)}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1))
    if registry_wl is not None and not args.dry_run:
        rep_ = registry_wl.finish()
        print(rep_.line(), flush=True)
    print(f"[grade] wrote {outp}", flush=True)


if __name__ == "__main__":
    # A credential rejection that reaches main -- the shared judge's / grader's typed
    # error -- ends the process with the typed text as its terminal line, never a traceback
    from sisteer import vault  # noqa: E402

    with vault.credential_exit(prefix="[grade] "):
        main()
