"""Grade the SERVED baseline wave's GATED jailbreak families (pasttense,
codeattack, renellm, grok420) with the LIVE StrongREJECT grader — the exact
``jailbreak_run._gated_defense_metrics`` grade/summary path, minus the XSTest
fairness gate and the pyrit-tainted file loader.

Why a separate driver rather than ``jailbreak_run.py aggregate --attack <fam>
--live``:
  1. That entry point ALWAYS runs ``_aggregate_xstest``, which FAILS LOUD
     without ``<root>/<private_run>/<cell>/xstest_responses.jsonl``. The scale-12
     served wave deliberately routes XSTest through the BENIGN leg
     (``cell_controls_run.py``; the JB tree carries only the five attack
     families), so that gate can never pass on this tree.
  2. Its ``--arms`` check rejects two-SI DefenseCell ids (prompted__safety_refusal).
  3. ``_load_gated_responses`` imports ``sisteer.eval.attacks.pyrit_adapters``,
     whose module-level ``import pyrit`` is absent from the grading host grading venv
     (grading needs no pyrit).

Everything that touches the SCORES is the harness verbatim:
``_gated_defense_metrics`` / ``_summarize_gated`` / ``grade_items`` and the same
``load_ordered_jsonl`` reader ``load_persisted_responses`` wraps. Only the
shard file-merge (a dozen lines) is re-implemented, and the xstest slot omitted
— so the ASR numbers are basis-identical to a normal gated aggregate.

Grader: LiveStrongREJECTGrader on the Low-Prio org, rubric v1, judge model from
``cfg["grader"]["model"]`` (claude-haiku-4-5) — never overridden here.
ASR at every threshold in ``cfg["prereg"]["threshold_grid"]`` (0.25/0.5/0.75),
plus per category group (held_in/held_out). Per-family n is emitted, never
assumed equal.

  python scripts/grade_gated_served.py \
    --config configs/two_si.yaml \
    --private-root <cond>/jb/jb_private --cell <cell_id> \
    --families pasttense,codeattack,renellm,grok420 \
    --out <out.json> --cache <cache.jsonl> --concurrency 200 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # jailbreak_run, grade_bon_live
sys.path.insert(0, str(HERE.parent / "src"))

import jailbreak_run as jr  # noqa: E402


def load_gated_dir(defense_root):
    """Replicate jailbreak_run._load_gated_responses WITHOUT importing
    sisteer.eval.attacks.pyrit_adapters (its module-level ``import pyrit`` is
    absent from the grading host grading venv, and grading needs none). Uses the SAME
    ordered-JSONL reader ``load_persisted_responses`` wraps (order_key="order",
    which falls back to the wave's "sample_idx" synonym) and merges progress via
    jr._load_progress. Returns ``({goal_id: [resp,...]}, {goal_id: progress})``.
    """
    from sisteer.eval.jailbreak_artifacts import load_ordered_jsonl

    responses, progress = {}, {}
    for sd in sorted(Path(defense_root).glob("shard*")):
        rp = sd / "responses.jsonl"
        if rp.exists():
            responses.update(load_ordered_jsonl(
                rp, order_key="order", value_key="response"))
        progress.update(jr._load_progress(sd / "progress.jsonl"))
    return responses, progress


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--private-root", required=True,
                    help="<cond>/jb/jb_private (holds <private_run>/<fam>/<cell>/shard*)")
    ap.add_argument("--cell", required=True,
                    help="DefenseCell id whose transcripts to grade (e.g. none, "
                         "prompted__safety_refusal)")
    ap.add_argument("--families", default="pasttense,codeattack,renellm,grok420")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None,
                    help="live score cache jsonl (kind 'sr'); resumes across runs")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--confirmatory", action="store_true", default=True)
    ap.add_argument("--allow-unscored", type=int, default=0,
                    help="max items the judge may leave UNSCORED (dropped, never\n                         scored 0) before failing loud; harness default 0")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API: fake score 0.0 for every item, to validate the "
                         "completeness gate + readers before spending")
    ap.add_argument("--allow-errored-goals", action="store_true",
                    help="EXCLUDE goals whose latest progress outcome is error:* "
                         "(error:PromptOverWindow, error:ActorDiscoveryError, ...) "
                         "instead of refusing; the summary carries n_expected / "
                         "n_graded / n_errored by type. Default: refuse, name the "
                         "goals — resume the run to regenerate them.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cell = args.cell
    cfg = {**cfg, "defenses": [cell]}
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    goals = jr._eval_goals(cfg, confirmatory=args.confirmatory)
    by_id = {g.goal_id: g for g in goals}
    expected_goal_ids = [g.goal_id for g in goals]
    group_of, held_in, held_out, ch = jr._category_groups(cfg, goals)
    root = jr._private_root(cfg, args.private_root)

    if args.dry_run:
        def grade_fn(items):
            return {(d, r, g, resp): 0.0 for (d, r, g, resp) in items}
        model = "DRY-RUN-no-api"
    else:
        from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader
        model = cfg["grader"]["model"]
        cache_path = (Path(args.cache) if args.cache
                      else root / cfg["private_run"] / "live_gated_grade_cache.jsonl")
        grader = LiveStrongREJECTGrader(
            model=model, concurrency=args.concurrency, cache_path=cache_path,
            allow_unscored=args.allow_unscored)
        grade_fn = grader.grade
        print(f"[gated] judge={model} org=low-prio concurrency={args.concurrency} "
              f"cache={cache_path}", flush=True)

    # Loader: replicate jailbreak_run._load_gated_responses WITHOUT importing
    # pyrit_adapters. Same ordered-JSONL reader load_persisted_responses wraps
    # (order_key="order" -> falls back to the wave's "sample_idx"); progress via
    # jr._load_progress. Grade + summary stay jr._gated_defense_metrics verbatim.
    thresholds = cfg["prereg"]["threshold_grid"]
    canonical = thresholds[len(thresholds) // 2]  # 0.5, exactly as _aggregate_gated
    out_families: dict[str, dict] = {}
    for fam in families:
        defense_root = jr._gated_defense_root(cfg, root, fam, cell)
        responses, progress = load_gated_dir(defense_root)
        missing = [gid for gid in expected_goal_ids if gid not in progress]
        if missing:
            raise SystemExit(
                f"[{fam}] {cell}: {len(missing)}/{len(expected_goal_ids)} expected "
                f"goals have no terminal progress record under {defense_root} — "
                "refusing to aggregate partial work (A2/B3).")
        leaked: list[str] = []
        m = jr._gated_defense_metrics(
            responses, progress, by_id, grade_fn, thresholds=thresholds,
            canonical=canonical, leaked_needles=leaked,
            expected_goal_ids=expected_goal_ids, group_of=group_of,
            allow_errored_goals=args.allow_errored_goals, label=f"{fam}] {cell}")
        m.pop("per_goal", None)  # goal-ids only; numbers are what we report
        out_families[fam] = m
        asr = m["asr"]
        if m["n_errored"]:
            print(f"[gated] {fam:<11} EXCLUDED {m['n_errored']} errored goal(s) "
                  f"{m['errored_by_type']} — n_graded={m['n_graded']}/"
                  f"{m['n_expected']}", flush=True)
        print(f"[gated] {fam:<11} n_goals={m['n_goals']:>3} "
              f"asr@0.5={asr['0.5']:.4f} asr@0.75={asr['0.75']:.4f} "
              f"n_jailbroken={m['n_jailbroken']} "
              f"mean_best={m['mean_best_score']:.4f}", flush=True)

    out = {"cell": cell, "model": model, "org": "low-prio",
           "thresholds": thresholds,
           "category_holdout": {"seed": ch["seed"], "frac": ch["frac"],
                                "held_in": held_in, "held_out": held_out},
           "families": out_families}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=1))
    print(f"[gated] wrote {outp}", flush=True)


if __name__ == "__main__":
    # A credential rejection that reaches main -- the shared judge's / grader's typed
    # error -- ends the process with the typed text as its terminal line, never a traceback
    from sisteer import vault  # noqa: E402

    with vault.credential_exit(prefix="[gated] "):
        main()
