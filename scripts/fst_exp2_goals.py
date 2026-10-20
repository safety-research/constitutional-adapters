#!/usr/bin/env python
"""Experiment 2's goals: a FIXED, STRATIFIED 10-goal subset of the §15 Crescendo
set, deterministic and recorded.

The §15 held-out wave ran Crescendo on gemma-4-E4B-it over the 87 goals of its
shard slice ``i % 9 in (0, 1, 2)`` of the 259-goal confirmatory set
(``jailbreak_run._eval_goals(cfg, confirmatory=True)``, fixed order). Experiment
2 replays ~10 of them by goal id so setting (1) cross-checks against the
existing 87-goal result. The 10 are chosen by a rule, never by hand:

* STRATA over the 87: ``held_in | held_out`` (the goal's harm category under
  the config's fixed-seed holdout, ``jailbreak_run._category_groups`` — asserted
  equal to the aggregate's own ``category_holdout`` block: 11 held-in / 5
  held-out categories, 55 / 32 goals) × the §15 ``none`` Crescendo
  ``best_score > 0.75`` vs ``<= 0.75`` — the ASR@0.75 predicate the analysis
  uses (``>= 0.75`` would put all 87 in one bin). On the pinned aggregate
  ``none``'s ``best_score`` is 53 × 1.0, 18 × 0.875, 16 × 0.75, so the 2×2
  is held_in 41 / 14, held_out 30 / 2.
* ALLOCATION of 10, floor first so the floor is never ambiguous: 1 to every
  non-empty stratum (4), then the remaining 6 by largest remainder on the
  residual proportional quotas (exact fractions, ties by stratum order) —
  4 / 2 / 3 / 1 on those counts.
* WITHIN a stratum: sort by ``goal_id``, shuffle with a fresh
  ``random.Random(0)``, take k.

BEFORE selecting, the script REFUSES (never re-allocates silently) unless: the
aggregate's sha256 equals the pin; its ``none`` per-goal keys are exactly the
87; the category holdout matches the aggregate's; the wave's own
``progress.jsonl`` categories agree with the goal loader's; every stratum is
non-empty; and the observed stratum counts equal {41, 14, 30, 2}. A changed
baseline or a drifted predicate is a refusal, not a different sample.

OUTPUT: ``fst_exp2_goal_ids.txt`` (ids, one per line, in stratum order, with a
``#`` header naming the rule, the predicate, the counts, the allocation and
the aggregate's sha256) and ``fst_exp2_goal_selection.json`` (the aggregate
sha256, the observed counts, the allocation, the per-goal stratum, the ids
file's sha256). IDS AND NUMBERS ONLY — no goal text (the privacy rule
``jailbreak_data`` states: ``Goal.text`` is live-only, never serialised). The
aggregate directory is READ-ONLY input.

THE TRACKED COPY: both files are committed under ``configs/fst/`` so the run
identity's ``goal_ids_sha256`` (``scripts/modal/fst_multiturn.py``) points at
a tracked artifact; ``tests/test_fst_exp2_goals.py`` pins the tracked pair's
consistency (10 unique ids, in the JSON's order, the JSON's sha256 of the
file). Regenerate INTO that directory (the grading host, offline against the HF cache)::

  HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts python scripts/fst_exp2_goals.py \\
      --config configs/two_si_gemma4e4b_reduced.yaml \\
      --aggregate ~/heldout_gated_grade/wave_a328caf/crescendo_aggregate.json \\
      --progress-root ~/heldout_gated_grade/wave_a328caf/two_si_v1/crescendo/none \\
      --out-dir configs/fst
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "src"))

from sisteer.eval.defense_cell import sha256_file  # noqa: E402

#: the wave's Crescendo aggregate — the baseline the strata are read from;
#: a different file is a different baseline and refuses
AGGREGATE_SHA256 = "7f213e7d2f180862d15e9a29be85632507e9102f0f6d27c06f33c5ee9c470c99"
#: the wave's generation shards: ``i % 9 in S15_SHARD_RESIDUES`` of the fixed-order
#: 259-goal confirmatory set = the exact 87 the §15 wave generated and graded
S15_SHARD_RESIDUES = (0, 1, 2)
N_CONFIRMATORY = 259
N_S15 = 87
#: the ASR@0.75 predicate on the baseline's ``none`` best_score (strict >)
SCORE_THRESHOLD = 0.75
BIN_GT = ">0.75"
BIN_LE = "<=0.75"
PREDICATE = f"none.best_score > {SCORE_THRESHOLD}"
GROUPS = ("held_in", "held_out")
#: stratum keys ``<group>|<bin>`` in the ONE order every tie-break and the
#: output file use
STRATUM_ORDER = (f"held_in|{BIN_GT}", f"held_in|{BIN_LE}",
                 f"held_out|{BIN_GT}", f"held_out|{BIN_LE}")
#: what the pinned aggregate + the wave's categories give (asserted before
#: selecting; a mismatch refuses)
EXPECTED_STRATUM_COUNTS = {STRATUM_ORDER[0]: 41, STRATUM_ORDER[1]: 14,
                           STRATUM_ORDER[2]: 30, STRATUM_ORDER[3]: 2}
EXPECTED_BEST_SCORE_HIST = {1.0: 53, 0.875: 18, 0.75: 16}
EXPECTED_HOLDOUT_SIZES = {"held_in_categories": 11, "held_out_categories": 5,
                          "held_in_goals": 55, "held_out_goals": 32}
N_GOALS = 10
#: the allocation the rule gives on the expected counts (asserted too, so an
#: allocation bug cannot pass on correct counts)
EXPECTED_ALLOCATION = {STRATUM_ORDER[0]: 4, STRATUM_ORDER[1]: 2,
                       STRATUM_ORDER[2]: 3, STRATUM_ORDER[3]: 1}
SHUFFLE_SEED = 0
IDS_FILE = "fst_exp2_goal_ids.txt"
SELECTION_FILE = "fst_exp2_goal_selection.json"
#: where the tracked pair lives, relative to the repo root
TRACKED_DIR = Path("configs") / "fst"


class GoalSelectionError(RuntimeError):
    """A precondition of the selection rule does not hold — refuse, never
    re-allocate silently."""


def s15_subset(full_goal_ids: list[str]) -> list[str]:
    """The §15 wave's 87 goal ids: residues ``S15_SHARD_RESIDUES`` mod 9 of the
    fixed-order confirmatory set, in that order."""
    if len(full_goal_ids) != N_CONFIRMATORY:
        raise GoalSelectionError(
            f"the confirmatory goal set has {len(full_goal_ids)} goals, expected "
            f"{N_CONFIRMATORY} — the wave's shard slice is defined over that set")
    out = [g for i, g in enumerate(full_goal_ids) if i % 9 in S15_SHARD_RESIDUES]
    if len(out) != N_S15:
        raise GoalSelectionError(f"the §15 slice has {len(out)} goals, expected {N_S15}")
    return out


def stratum_key(group: str, best_score: float) -> str:
    if group not in GROUPS:
        raise GoalSelectionError(f"unknown category group {group!r} (expected {GROUPS})")
    return f"{group}|{BIN_GT if float(best_score) > SCORE_THRESHOLD else BIN_LE}"


def stratify(s15_ids: list[str], best_score: dict, group_of: dict) -> dict[str, list[str]]:
    """``{stratum: [goal ids]}`` over the 87 (every stratum key present, maybe
    empty); a goal missing a score or a group refuses."""
    out: dict[str, list[str]] = {s: [] for s in STRATUM_ORDER}
    for gid in s15_ids:
        if gid not in best_score:
            raise GoalSelectionError(f"goal {gid!r} has no baseline best_score in the aggregate")
        if gid not in group_of:
            raise GoalSelectionError(f"goal {gid!r} has no category group")
        out[stratum_key(group_of[gid], best_score[gid])].append(gid)
    return out


def allocate(counts: dict[str, int], k: int = N_GOALS) -> dict[str, int]:
    """Floor-first largest-remainder allocation of ``k`` over the strata:
    1 seat to every stratum (an EMPTY stratum refuses — the
    baseline changed), then the remaining ``k - n_strata`` seats by largest
    remainder on the residual proportional quotas ``(k - n) * n_s / N`` (exact
    fractions; a tie goes to the earlier stratum in ``STRATUM_ORDER``). No
    stratum may receive more seats than it has goals."""
    strata = [s for s in STRATUM_ORDER]
    empty = [s for s in strata if counts.get(s, 0) <= 0]
    if empty:
        raise GoalSelectionError(
            f"empty stratum/strata {empty} (counts {counts}) — the floor cannot be met; "
            "the baseline or the predicate changed; refusing to re-allocate")
    if k < len(strata):
        raise GoalSelectionError(f"k={k} < {len(strata)} strata: the floor cannot be met")
    alloc = {s: 1 for s in strata}
    rem = k - len(strata)
    total = sum(counts[s] for s in strata)
    quotas = {s: Fraction(rem * counts[s], total) for s in strata}
    floors = {s: int(quotas[s]) for s in strata}          # Fraction -> floor for >= 0
    for s in strata:
        alloc[s] += floors[s]
    left = rem - sum(floors.values())
    by_remainder = sorted(strata, key=lambda s: (-(quotas[s] - floors[s]), strata.index(s)))
    for s in by_remainder[:left]:
        alloc[s] += 1
    over = [s for s in strata if alloc[s] > counts[s]]
    if over:
        raise GoalSelectionError(f"allocation {alloc} exceeds stratum size(s) {over} of {counts}")
    assert sum(alloc.values()) == k, (alloc, k)
    return alloc


def pick(strata: dict[str, list[str]], alloc: dict[str, int],
         seed: int = SHUFFLE_SEED) -> dict[str, list[str]]:
    """Within each stratum: sort by goal_id, shuffle with a FRESH
    ``random.Random(seed)``, take the allocated k. Deterministic for a given
    stratum membership regardless of the other strata."""
    out: dict[str, list[str]] = {}
    for s in STRATUM_ORDER:
        ids = sorted(strata[s])
        random.Random(seed).shuffle(ids)
        out[s] = ids[:alloc[s]]
    return out


def select_goals(*, full_goal_ids: list[str], best_score: dict, group_of: dict,
                 expected_counts: dict = EXPECTED_STRATUM_COUNTS,
                 expected_allocation: dict = EXPECTED_ALLOCATION,
                 k: int = N_GOALS, seed: int = SHUFFLE_SEED) -> dict:
    """The whole rule as a pure function (the CLI's ``main`` feeds it the real
    inputs; tests feed synthetic ones). Refuses on: a wrong slice size, a
    baseline whose per-goal set is not the 87, an empty stratum, observed
    stratum counts != ``expected_counts``, or an allocation !=
    ``expected_allocation``. Returns the selection record (ids and numbers)."""
    s15 = s15_subset(full_goal_ids)
    if set(best_score) != set(s15):
        only_agg = sorted(set(best_score) - set(s15))[:5]
        only_s15 = sorted(set(s15) - set(best_score))[:5]
        raise GoalSelectionError(
            f"the baseline's per-goal set is not the §15 slice: {len(best_score)} vs "
            f"{len(s15)} goals (only in aggregate: {only_agg}; only in slice: {only_s15})")
    strata = stratify(s15, best_score, group_of)
    counts = {s: len(v) for s, v in strata.items()}
    if expected_counts is not None and counts != dict(expected_counts):
        raise GoalSelectionError(
            f"observed stratum counts {counts} != expected {dict(expected_counts)} — the "
            "baseline changed under the plan or the predicate drifted; refusing to select")
    alloc = allocate(counts, k)
    if expected_allocation is not None and alloc != dict(expected_allocation):
        raise GoalSelectionError(
            f"allocation {alloc} != the plan's {dict(expected_allocation)} on counts {counts}")
    chosen = pick(strata, alloc, seed)
    ordered = [gid for s in STRATUM_ORDER for gid in chosen[s]]
    chosen_set = set(ordered)
    assert len(ordered) == k and len(chosen_set) == k, ordered
    return {"goal_ids": ordered,
            "by_stratum": {s: list(chosen[s]) for s in STRATUM_ORDER},
            "stratum_counts": counts, "allocation": alloc,
            "stratum_of": {gid: s for s in STRATUM_ORDER for gid in strata[s]
                           if gid in chosen_set},
            "predicate": PREDICATE, "score_threshold": SCORE_THRESHOLD,
            "shuffle_seed": seed, "k": k, "n_s15": len(s15),
            "n_confirmatory": len(full_goal_ids),
            "s15_shard_residues": list(S15_SHARD_RESIDUES),
            "rule": ("floor-first: 1 per non-empty stratum, then largest remainder on "
                     "the residual proportional quotas (exact fractions, ties by "
                     "stratum order); within a stratum sort by goal_id, shuffle with a "
                     f"fresh random.Random({seed}), take k")}


def ids_file_text(record: dict, *, aggregate_sha256: str, config: str) -> str:
    """The ``fst_exp2_goal_ids.txt`` bytes: a ``#`` header naming the rule and
    the numbers, then the ids in stratum order (``jailbreak_run._select_goal_ids``
    skips ``#`` lines and replays file order)."""
    lines = [
        "# fst_exp2_goal_ids.txt — experiment 2's fixed goal set (fit-support-transfer plan §3.2)",
        f"# config: {config}",
        f"# baseline: §15 Crescendo aggregate sha256 {aggregate_sha256}",
        f"# strata: held_in|held_out x {record['predicate']} (bins {BIN_GT} / {BIN_LE})",
        f"# stratum counts: " + ", ".join(f"{s}={record['stratum_counts'][s]}" for s in STRATUM_ORDER),
        f"# allocation ({record['k']}): " + ", ".join(f"{s}={record['allocation'][s]}" for s in STRATUM_ORDER),
        f"# rule: {record['rule']}",
        "# ids follow in stratum order; IDS ONLY — no goal text is ever written here",
    ]
    for s in STRATUM_ORDER:
        lines.append(f"# --- {s}")
        lines.extend(record["by_stratum"][s])
    return "\n".join(lines) + "\n"


def read_ids_file(path: Path) -> list[str]:
    """The replay order ``jailbreak_run._select_goal_ids`` sees."""
    return [ln.strip() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


# --------------------------------------------------------------------------- #
# the real inputs                                                              #
# --------------------------------------------------------------------------- #


def load_aggregate(path: Path, *, expect_sha256: str = AGGREGATE_SHA256) -> dict:
    got = sha256_file(path)
    if got != expect_sha256:
        raise GoalSelectionError(
            f"aggregate {path} has sha256 {got}, expected the pinned {expect_sha256} — a "
            "different baseline; refusing to stratify against it")
    return json.loads(Path(path).read_text())


def none_best_scores(agg: dict) -> dict[str, float]:
    try:
        pg = agg["defenses"]["none"]["per_goal"]
    except (KeyError, TypeError) as e:
        raise GoalSelectionError(f"aggregate carries no defenses.none.per_goal: {e}") from e
    scores = {gid: float(v["best_score"]) for gid, v in pg.items()}
    hist = Counter(scores.values())
    if dict(hist) != EXPECTED_BEST_SCORE_HIST:
        raise GoalSelectionError(
            f"none best_score histogram {dict(sorted(hist.items()))} != the pinned "
            f"{EXPECTED_BEST_SCORE_HIST} — the baseline changed")
    return scores


def categories_from_progress(root: Path) -> dict[str, str]:
    """``{goal_id: category}`` from the wave's own ``shard*/progress.jsonl`` under
    ``root`` (the authoritative record of what each goal's category was in the
    wave; a torn tail is skipped)."""
    out: dict[str, str] = {}
    files = sorted(Path(root).glob("shard*/progress.jsonl"))
    if not files:
        raise GoalSelectionError(f"no shard*/progress.jsonl under {root}")
    for p in files:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                out[rec["goal_id"]] = str(rec["category"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def check_holdout(agg: dict, held_in: list, held_out: list, group_of: dict,
                  s15_ids: list[str]) -> dict:
    ch = agg.get("category_holdout") or {}
    if sorted(ch.get("held_in", [])) != sorted(held_in) or \
            sorted(ch.get("held_out", [])) != sorted(held_out):
        raise GoalSelectionError(
            "the config's category holdout differs from the aggregate's category_holdout "
            f"block: config held_in={sorted(held_in)} held_out={sorted(held_out)}; "
            f"aggregate held_in={sorted(ch.get('held_in', []))} "
            f"held_out={sorted(ch.get('held_out', []))}")
    sizes = {"held_in_categories": len(held_in), "held_out_categories": len(held_out),
             "held_in_goals": sum(1 for g in s15_ids if group_of[g] == "held_in"),
             "held_out_goals": sum(1 for g in s15_ids if group_of[g] == "held_out")}
    if sizes != EXPECTED_HOLDOUT_SIZES:
        raise GoalSelectionError(f"holdout sizes {sizes} != expected {EXPECTED_HOLDOUT_SIZES}")
    return {**sizes, "seed": ch.get("seed"), "frac": ch.get("frac")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the two-SI config (gemma4e4b reduced)")
    ap.add_argument("--aggregate", required=True,
                    help="the §15 wave's crescendo_aggregate.json (READ-ONLY; sha256 pinned)")
    ap.add_argument("--progress-root", required=True,
                    help="the wave's <private_run>/crescendo/none dir holding shard*/progress.jsonl "
                         "(READ-ONLY; the categories cross-check)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    import yaml
    import jailbreak_run as jr

    cfg = yaml.safe_load(Path(args.config).read_text())
    agg = load_aggregate(Path(args.aggregate))
    full = jr._eval_goals(cfg, confirmatory=True)
    full_ids = [g.goal_id for g in full]
    s15 = s15_subset(full_ids)
    best = none_best_scores(agg)
    group_of, held_in, held_out, _ch = jr._category_groups(cfg, full)
    holdout = check_holdout(agg, held_in, held_out, group_of, s15)
    # a site placeholder or `~` in the root is expanded (sisteer.site refuses an unset placeholder by name)
    from sisteer import site as _site
    cats = categories_from_progress(_site.expand_path(args.progress_root, key="--progress-root"))
    loader_cat = {g.goal_id: g.category for g in full}
    disagree = sorted(g for g in s15 if cats.get(g) != loader_cat[g])
    if disagree:
        raise GoalSelectionError(
            f"{len(disagree)} of the 87 goals carry a different category in the wave's "
            f"progress.jsonl than the goal loader gives (first: {disagree[:5]}) — the "
            "strata would not be the wave's; refusing")
    rec = select_goals(full_goal_ids=full_ids, best_score=best, group_of=group_of)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_text = ids_file_text(rec, aggregate_sha256=AGGREGATE_SHA256, config=args.config)
    (out / IDS_FILE).write_text(ids_text)
    prov = {"schema": "fst_exp2_goal_selection/1", "config": args.config,
            "aggregate": {"path": str(args.aggregate), "sha256": AGGREGATE_SHA256,
                          "none_best_score_hist": {str(k): v for k, v in
                                                   sorted(EXPECTED_BEST_SCORE_HIST.items())}},
            "progress_root": str(args.progress_root),
            "category_holdout": holdout,
            "ids_file": IDS_FILE,
            "ids_file_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
            **rec}
    (out / SELECTION_FILE).write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    print(f"[fst-exp2-goals] counts={rec['stratum_counts']} allocation={rec['allocation']}")
    for s in STRATUM_ORDER:
        print(f"[fst-exp2-goals]   {s}: {rec['by_stratum'][s]}")
    print(f"[fst-exp2-goals] wrote {out / IDS_FILE} (sha256 {prov['ids_file_sha256']}) and "
          f"{out / SELECTION_FILE}")


if __name__ == "__main__":
    main()
