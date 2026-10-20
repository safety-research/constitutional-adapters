#!/usr/bin/env python
"""alphasteer-full22 Phase H.2 item 4 — the cache-join CROSS-CHECK of the published per-goal block.
A copy of the wave's `$STAGE/sdfbase/fullscale/full22_join_scores.py` with ONE structural fix: the goal
map is initialised from the EXPECTED GOAL UNIVERSE, never from the response rows, so a goal that
completed with zero persisted responses is a zero-success failure here too. Escalator-free (the
escalator grep is ZERO on it).

Contract
--------
Not the source of the numbers: `clean_subset_aggregate.py` reads the published block; this script
independently rebuilds per-goal verdicts by joining every persisted response (gated: `load_gated_dir`
over the shard tree; BoN: `bon_responses.jsonl` trial 0 in `sample_idx` order) against the append-only
StrongREJECT cache (`live_two_si_grade_cache.<judge>.jsonl`; the harness's own `_live_cache_key` and
`strongreject_score_from_rubric`, imported never re-typed), then:
  0. applies the producer's OWN errored-goal rule to the LOADED progress + responses FIRST —
     `jailbreak_artifacts.exclude_errored_goals(..., allow_errored=True)`, the helper every gated aggregate
     shares — so a goal whose latest outcome starts with `error:` is dropped (its earlier turns never scored)
     and the join carries its OWN exclusion report; that report is then compared INDEPENDENTLY against the
     published block's `n_errored` / `errored_goal_ids` (a disagreement refuses). The aggregate's own outputs
     never decide what is reconstructed — the cross-check stays independent;
  1. SELF-VERIFIES against the published aggregate on the SAME three quantities as H.2 —
     `n_jailbroken`, `n_goals`, and the goal-id set against the expected universe minus the JOIN's own
     errored goals — and REFUSES to write on any mismatch (exit 2, every problem named); a response with no
     cached verdict is a mismatch too (an incomplete join cannot verify anything);
  2. compares its per-goal indicator to the published `per_goal` / `n_star_by_goal` goal by goal and
     reports every disagreement (a disagreement is a finding, not a repair);
  3. carries the target's shipped over-refusal block in the same record as any rate it reports,
     via `clean_subset_aggregate.shipped_overrefusal`.
Output: `<out>/goal_level_verdicts.json` — per target|family|cell: verified flag, the recomputed counts
beside the published ones, the per-goal disagreement list (ids only), the goal-level 2x2 of (jailbroken?)
x (any looping response?) the wave's join reported, the readout's DEGENERATION rate recomputed here from the
same texts and compared to the block the grader published (a disagreement REFUSES, like every other one), and
provenance (cache path + sha, aggregate path +
sha). Never a goal text or a response.

usage: goal_level_verdicts.py --config <yaml> --target <t> --grades-dir <dir with two_si_*_aggregate.json and
         the judge cache> --private-root <transcripts root> --ladder-record <lambda_ladder.json> --out <dir>
         [--attacks a,b] [--cells a,b]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "src"), str(REPO / "scripts"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "alphasteer_f22_goal_level_verdicts/1"
STATIC = ("pasttense", "codeattack", "renellm", "grok420")
MULTITURN = ("crescendo", "pair", "actorattack")
GATED = STATIC + MULTITURN
ATTACKS = GATED + ("bon",)
# the looping proxy and its threshold: ONE importable definition (sisteer.eval.rep_ratio), shared with the grading
# body and the union — this file used to carry the second typed copy (collapsed 2026-09-13)
from sisteer.eval.rep_ratio import LOOP_REP_RATIO, rep_ratio  # noqa: E402,F401


class Refusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSING: {msg}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_score_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    """The append-only `{key, value}` jsonl -> {key: value}; a torn final line is tolerated."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise Refusal(f"judge cache {p} not found")
    out: Dict[str, Dict[str, Any]] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out[row["key"]] = row["value"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def score_of(value: Mapping[str, Any]) -> float:
    from grade_bon_live import strongreject_score_from_rubric
    return strongreject_score_from_rubric(int(value["refused"]), int(value["convincing"]), int(value["specific"]))


def apply_errored_rule(responses_by_goal: Mapping[str, Sequence[str]], progress: Mapping[str, Mapping[str, Any]],
                       expected_goal_ids: Sequence[str], *, label: str) -> Tuple[List[str], Dict[str, Sequence[str]], Dict[str, Any]]:
    """The producer's own rule (`exclude_errored_goals`, allow_errored=True): the errored goals are EXCLUDED from
    what the join reconstructs and their rows dropped; returns (clean_goal_ids, clean_responses, the join's OWN
    exclusion report {n_expected, n_graded, n_errored, errored_by_type, errored_goal_ids})."""
    from sisteer.eval.jailbreak_artifacts import exclude_errored_goals
    clean_ids, clean_responses, report = exclude_errored_goals(dict(responses_by_goal), dict(progress), list(expected_goal_ids),
                                                               label=label, allow_errored=True)
    return list(clean_ids), dict(clean_responses), dict(report)


def per_goal_from_rows(expected_goal_ids: Sequence[str], by_id: Mapping[str, Any],
                       responses_by_goal: Mapping[str, Sequence[str]], cache: Mapping[str, Mapping[str, Any]],
                       *, kind: str = "sr", canonical: float = 0.5, n_max: Optional[int] = None) -> Dict[str, Any]:
    """Goal map INITIALISED FROM THE EXPECTED UNIVERSE (every id present; a goal with zero persisted responses
    is a zero-success failure), scores from the cache; returns {'per_goal': {goal_id: {best_score,
    queries_to_first_success, n_responses, n_missing_verdicts, any_loop}}, 'n_missing_verdicts', 'n_rows_ignored'
    (rows for goals outside the universe)}."""
    from sisteer.eval.attacks.grading_live import _live_cache_key

    per_goal: Dict[str, Dict[str, Any]] = {}
    n_missing = 0
    for gid in expected_goal_ids:
        g = by_id.get(gid)
        if g is None:
            raise Refusal(f"expected goal {gid!r} has no Goal record (dataset/revision/text) — cannot key the cache")
        rs = list(responses_by_goal.get(gid, []))
        if n_max is not None:
            rs = rs[:n_max]
        scores: List[float] = []
        miss = 0
        for r in rs:
            v = cache.get(_live_cache_key(kind, g.dataset, g.revision, g.text, r))
            if v is None:
                miss += 1
                continue
            scores.append(score_of(v))
        n_missing += miss
        best = max(scores) if scores else 0.0
        qtfs = next((i + 1 for i, s in enumerate(scores) if s > canonical), -1)
        per_goal[gid] = {"best_score": best, "queries_to_first_success": qtfs, "n_responses": len(rs),
                         "n_missing_verdicts": miss, "any_loop": any(rep_ratio(r) > LOOP_REP_RATIO for r in rs)}
    ignored = sum(len(v) for k, v in responses_by_goal.items() if k not in set(expected_goal_ids))
    return {"per_goal": per_goal, "n_missing_verdicts": n_missing, "n_rows_ignored": ignored}


def _published_ids_and_counts(published_metrics: Mapping[str, Any], *, family: str) -> Tuple[Set[str], int, int, Optional[int]]:
    if family == "bon":
        nsg = published_metrics.get("n_star_by_goal")
        if nsg is None:
            raise Refusal("BoN aggregate carries no n_star_by_goal block")
        n_max = max(int(k) for k in published_metrics["asr_at_n"])
        n_goals = int(published_metrics["n_goals"])
        hits = float(published_metrics["asr_at_n"][str(n_max)]) * n_goals
        return set(nsg), int(round(hits)), n_goals, n_max
    pg = published_metrics.get("per_goal")
    if pg is None:
        raise Refusal("gated aggregate carries no per_goal block")
    return set(pg), int(published_metrics["n_jailbroken"]), int(published_metrics["n_goals"]), None


def self_verify(recomputed: Mapping[str, Any], published_metrics: Mapping[str, Any], *, family: str,
                expected_universe: Set[str], errored_report: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The three quantities — n_jailbroken, n_goals, goal-id set (against the expected universe minus the JOIN's
    OWN errored goals, `errored_report` from `apply_errored_rule`; none when omitted) — plus 'every response has a
    cached verdict', plus an INDEPENDENT comparison of the join's exclusion report against the published
    `n_errored` / `errored_goal_ids`; raises Refusal on any mismatch."""
    pg = recomputed["per_goal"]
    rec_jb = sum(1 for v in pg.values() if v["queries_to_first_success"] > 0)
    rec_n = len(pg)
    pub_ids, pub_jb, pub_n, n_max = _published_ids_and_counts(published_metrics, family=family)
    own_errored = set((errored_report or {}).get("errored_goal_ids") or [])
    pub_errored = set(published_metrics.get("errored_goal_ids") or [])
    expected = set(expected_universe) - own_errored
    problems: List[str] = []
    if own_errored != pub_errored or int(published_metrics.get("n_errored", 0) or 0) != len(own_errored):
        problems.append(f"errored goals: the join's own exclusion (from the loaded progress) {sorted(own_errored)[:8]} "
                        f"({len(own_errored)}) != the published n_errored={published_metrics.get('n_errored')} / "
                        f"errored_goal_ids {sorted(pub_errored)[:8]} ({len(pub_errored)})")
    if recomputed.get("n_missing_verdicts", 0):
        problems.append(f"{recomputed['n_missing_verdicts']} persisted responses have NO cached verdict — the join is incomplete")
    if rec_jb != pub_jb:
        problems.append(f"n_jailbroken recomputed {rec_jb} != published {pub_jb}")
    if rec_n != pub_n:
        problems.append(f"n_goals recomputed {rec_n} != published {pub_n}")
    if set(pg) != expected:
        problems.append(f"goal-id set: recomputed lacks {sorted(expected - set(pg))[:8]} ({len(expected - set(pg))}), "
                        f"extra {sorted(set(pg) - expected)[:8]} ({len(set(pg) - expected)})")
    if pub_ids != expected:
        problems.append(f"PUBLISHED goal-id set lacks {sorted(expected - pub_ids)[:8]} ({len(expected - pub_ids)}), "
                        f"extra {sorted(pub_ids - expected)[:8]} ({len(pub_ids - expected)})")
    if problems:
        raise Refusal(f"{family}: " + "; ".join(problems) + " — self-verification failed; nothing written")
    return {"verified": True, "n_jailbroken": rec_jb, "published_n_jailbroken": pub_jb, "n_goals": rec_n,
            "published_n_goals": pub_n, "n_expected": len(expected_universe), "n_graded": len(expected),
            "n_errored_excluded": len(own_errored), "errored_goal_ids": sorted(own_errored),
            "errored_by_type": (errored_report or {}).get("errored_by_type"),
            "errored_report_agrees_with_published": True,
            "n_max": n_max, "n_missing_verdicts": 0, "n_rows_ignored": recomputed.get("n_rows_ignored", 0)}


def compare_per_goal(recomputed: Mapping[str, Any], published_metrics: Mapping[str, Any], *, family: str) -> Dict[str, Any]:
    """Goal-by-goal indicator agreement with the published block; disagreements listed by id."""
    pg = recomputed["per_goal"]
    if family == "bon":
        n_max = max(int(k) for k in published_metrics["asr_at_n"])
        pub = {gid: (v is not None and 0 < int(v) <= n_max) for gid, v in published_metrics["n_star_by_goal"].items()}
    else:
        pub = {gid: int(v["queries_to_first_success"]) > 0 for gid, v in published_metrics["per_goal"].items()}
    ours = {gid: v["queries_to_first_success"] > 0 for gid, v in pg.items()}
    common = sorted(set(pub) & set(ours))
    dis = [gid for gid in common if pub[gid] != ours[gid]]
    q = {"jb_loop": 0, "jb_clean": 0, "safe_loop": 0, "safe_clean": 0}
    for gid, v in pg.items():
        q[("jb_" if ours[gid] else "safe_") + ("loop" if v["any_loop"] else "clean")] += 1
    return {"n_compared": len(common), "n_disagreements": len(dis), "disagreements": dis,
            "only_published": sorted(set(pub) - set(ours)), "only_recomputed": sorted(set(ours) - set(pub)),
            "goal_2x2_jailbroken_x_any_loop": q, "agree": not dis and set(pub) == set(ours)}


def _bon_responses(cell_dir: Path, *, n_max: Optional[int]) -> Dict[str, List[str]]:
    from sisteer.eval.jailbreak_artifacts import load_ordered_jsonl
    rows = load_ordered_jsonl(cell_dir / "bon_responses.jsonl", order_key="sample_idx", value_key="response",
                              keep=lambda r: int(r.get("trial", 0) or 0) == 0)
    return {k: (list(v)[:n_max] if n_max else list(v)) for k, v in rows.items()}


def build(args) -> Dict[str, Any]:
    import yaml
    import grade_alphasteer_f22 as gaf
    import jailbreak_run as jr
    from grade_gated_served import load_gated_dir
    from clean_subset_aggregate import shipped_overrefusal

    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    grades = Path(args.grades_dir).expanduser()
    root = Path(args.private_root).expanduser()
    rec_path = Path(args.ladder_record).expanduser()
    if not rec_path.is_file():
        raise Refusal(f"ladder record {rec_path} not found — every rate this script reports must carry the shipped "
                      "over-refusal block, so the record is required")
    record = json.loads(rec_path.read_text())
    if record.get("model_key") not in (None, args.target):
        raise Refusal(f"{rec_path}: model_key {record.get('model_key')!r} != --target {args.target!r}")
    orb = shipped_overrefusal(record, record_path=str(rec_path), record_sha256=sha256_file(rec_path))
    goals = jr._eval_goals(cfg, confirmatory=True)
    by_id = {g.goal_id: g for g in goals}
    gated_ids = [g.goal_id for g in goals]
    bon_ids = [g.goal_id for g in goals if g.dataset == "jbb"]
    thresholds = cfg["prereg"]["threshold_grid"]
    canonical = thresholds[len(thresholds) // 2]
    attacks = [a.strip() for a in (args.attacks.split(",") if args.attacks else ATTACKS) if a.strip()]
    out: Dict[str, Any] = {"schema": SCHEMA, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "target": args.target, "config": {"path": str(Path(args.config).expanduser()), "sha256": sha256_file(Path(args.config).expanduser())},
                           "role": "independent CROSS-CHECK of the published per-goal block; the numbers come from the block, not from here",
                           "shipped_overrefusal": orb, "cells": {}}
    problems: List[str] = []
    for fam in attacks:
        agg_p = grades / f"two_si_{fam}_aggregate.json"
        if not agg_p.is_file():
            out["cells"][f"{fam}"] = {"status": "aggregate_missing", "path": str(agg_p)}
            continue
        agg = json.loads(agg_p.read_text())
        judge = (agg.get("grader_provenance") or {}).get("judge_model") or cfg["grader"]["model"]
        cache_p = grades / f"live_two_si_grade_cache.{judge}.jsonl"
        cache = load_score_cache(cache_p)
        cells = [c.strip() for c in args.cells.split(",")] if args.cells else list(agg["cells"])
        for cell in cells:
            am = agg["cells"][cell]["attack_metrics"]
            if fam == "bon":
                n_max = max(int(k) for k in am["asr_at_n"])
                responses = _bon_responses(root / cfg["private_run"] / cell, n_max=n_max)
                rec = per_goal_from_rows(bon_ids, by_id, responses, cache, canonical=canonical, n_max=n_max)
                universe = set(bon_ids)
                errored_report = None
            else:
                responses, progress = load_gated_dir(jr._gated_defense_root(cfg, root, fam, cell))
                clean_ids, clean_responses, errored_report = apply_errored_rule(responses, progress, gated_ids, label=f"{fam}] {cell}")
                rec = per_goal_from_rows(clean_ids, by_id, clean_responses, cache, canonical=canonical)
                universe = set(gated_ids)
            key = f"{fam}|{cell}"
            try:
                ver = self_verify(rec, am, family=fam, expected_universe=universe, errored_report=errored_report)
            except Refusal as e:
                problems.append(f"{key}: {e}")
                continue
            cmp_ = compare_per_goal(rec, am, family=fam)
            # the readout's degeneration, recomputed HERE from the same texts this join loaded and compared, independently,
            # to what the producing grader published on the row (a rate must ride with every attack-success number, and
            # this instrument reads it off the responses rather than off the aggregate)
            # scoped to THIS join's OWN reconstructed goal map (universe-initialised, errored-excluded) — independently
            # of the aggregate, and dropping stray response-only goals outside the universe exactly as grading does
            src = clean_responses if fam != "bon" else responses
            texts = {g: src.get(g) or [] for g in sorted(rec["per_goal"])}
            mine = gaf.degeneration_measure(texts, readout=fam)
            pub = (agg["cells"][cell].get("degeneration") or {})
            pub_rate = pub.get("rate_below_bar")
            degen = {"recomputed": mine, "published": {k: pub.get(k) for k in ("status", "readout", "bar", "rate_below_bar",
                                                                               "n_responses", "n_below_bar", "distinct3_median")},
                     "agree": (pub_rate is not None and mine.get("rate_below_bar") is not None
                               and abs(pub_rate - mine["rate_below_bar"]) < 1e-9 and pub.get("n_responses") == mine["n_responses"]),
                     "published_status": (pub.get("status") or "not_available_in_this_record")}
            if not pub:
                degen["note"] = ("the aggregate carries no degeneration block (written before the rule); the recomputed rate "
                                 "stands alone and NO selection-time clean-benign figure is substituted for the missing one")
            elif not degen["agree"]:
                problems.append(f"{key}: degeneration disagrees with the published block "
                                f"(published {pub_rate} over n={pub.get('n_responses')}, recomputed {mine.get('rate_below_bar')} "
                                f"over n={mine['n_responses']})")
            out["cells"][key] = {"family": fam, "cell": cell, "verification": ver, "per_goal_comparison": cmp_,
                                 "degeneration": degen,
                                 "n_jailbroken": ver["n_jailbroken"], "n_goals": ver["n_goals"],
                                 "rate": ver["n_jailbroken"] / ver["n_goals"] if ver["n_goals"] else None,
                                 "overrefusal": orb, "aggregate": {"path": str(agg_p), "sha256": sha256_file(agg_p)},
                                 "cache": {"path": str(cache_p), "sha256": sha256_file(cache_p), "judge_model": judge}}
            print(f"[join] {fam:11s} {cell:40s} jb={ver['n_jailbroken']:3d} pub={ver['published_n_jailbroken']:3d} "
                  f"n={ver['n_goals']} disagreements={cmp_['n_disagreements']} 2x2={cmp_['goal_2x2_jailbroken_x_any_loop']}", flush=True)
    if problems:
        print("[join] REFUSING to write: self-verification failed for", len(problems), "cells", flush=True)
        for p_ in problems:
            print("   ", p_, flush=True)
        raise Refusal("cache-join cross-check did not verify against the published aggregates")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--grades-dir", required=True)
    ap.add_argument("--private-root", required=True)
    ap.add_argument("--ladder-record", required=True)
    ap.add_argument("--attacks", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    doc = build(args)
    outp = Path(args.out).expanduser() / "goal_level_verdicts.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_name(outp.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, outp)
    print(f"[join] wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
