#!/usr/bin/env python3
"""degeneracy_items.py — per-item degeneracy sidecars for the padded static wave: ONE line per goal, computed with the
campaign's OWN statistics (imported, never re-typed), written beside the grades of one (level, family).

EVERY CONFIG FAMILY, ONE WRITER (2026-09-23; the seed replicates 2026-09-24): the prompted-only wave (``pad_static_<key>_p<pad>``,
three cells), the STEERED seed-1 wave (``--config-family obj``: ``pad_static_<key>_obj_p<pad>``, the ten cells of
``make_padding_configs.OBJ_CELLS`` — five fitted s1 objects, the Arditi rung, the paired write, and the same three reference
arms) and the SEED REPLICATES (``--config-family obj_s2`` / ``obj_s3``: ``pad_static_<key>_obj_s<N>_p<pad>``, the seven cells of
``make_padding_configs.OBJ_SEED_CELLS`` — ``none`` and the six fitted objects served from the seed-N fits; no prompted baseline,
no Arditi rung) decompose through the same lines. The families this writer serves ARE the generator's
(``CONFIG_FAMILIES = make_padding_configs.FAMILIES``, and ``--config-family``'s choices are read from it), the run name is
``make_padding_configs.private_run(key, pad, config_family)`` and the default cell set ``make_padding_configs.cells_of(config_family)``
— the ONE spelling, never retyped here, so a seed the generator gains needs no edit in this file. For the steered waves the
writer is NOT optional (target_static_padded_obj.sbatch header): the paired object family degenerates differently by model and its
degeneration flips sign between rubrics, and padded context is where looping and truncation rise, so "the steering decayed"
must stay distinguishable from "the generations degenerated". Zeros are data at p0 too: the zero-point control's sidecars are
written exactly like a padded level's (``padding: null``, ``pad_tokens: 0``), as the prompted-only wave's p0 sidecars were.
A seed replicate's meta names its ``training_seed`` (``make_padding_configs.seed_of``) beside ``config_family``, so its sidecars
can never be read as the seed-1 family's; the seed-1 and prompted-only metas keep the key sets they were filed with.

WHAT IT WRITES (the results-registry owner's shape, 2026-09-22), under ``<grades-root>/<private_run>/<family>/``:
  ``degeneracy_items.<cell>.jsonl``   one line PER GOAL, keyed by the ``goal_id`` that is the per_goal key of
                                      ``two_si_<family>_aggregate.json``; ids, flags and numbers only — never text
  ``degeneracy_items.meta.json``      thresholds, the fully-qualified statistics and their code shas, the reduction rules,
                                      per-cell row and goal counts, every cross-check, the source store, the UTC time

THE STATISTICS ARE THE CAMPAIGN'S. Every per-row number comes out of ``sisteer.eval.degeneracy_union.classify_rows``
(the union: ``sisteer.eval.rep_ratio.rep_ratio`` for the 4-gram loop ratio, ``static_grading._zlib_ratio`` /
``_alnum_count`` for zlib collapse and contentless, ``degeneracy.classify`` for the classifier, ``finish_reason ==
'length'`` for at-cap), which itself refuses to return unless its per-row path agrees with the shared grading body
(``static_grading.degeneration_block``, the body that wrote ``padded_static_summary.json``'s degeneration block) count for
count. The ONE addition is the tail reading of the loop ratio — ``sisteer.capeval.tiers.cap_ladder.tail_rep_ratio``, the
cap ladder's late-onset-loop rule (docs/cap_ladder.md: ``rep_ratio`` over the whole text OR its last 200 whitespace tokens)
— whose own ``rep_ratio`` copy is asserted equal to the canonical one on every row it reads.

THE GRAIN. A goal has several variant rows (2 or 3 in this wave; COUNTED per family, never assumed), so a per-goal line
reduces over them: max for the loop ratios and response_tokens, min for zlib_ratio, ``any`` for every boolean (identical
to thresholding the reduced scalar; asserted per goal), the at-cap row's finish_reason where one exists. ``any`` alone
loses a real case — a goal with one at-cap loop row and one at-cap non-loop row reads ``degenerate`` and hides the genuine
truncation — so every line also carries ``n_rows`` / ``n_at_cap`` / ``n_at_cap_degenerate`` / ``n_at_cap_truncation_only``
(``truncation_only := n_at_cap_truncation_only > 0``) and the full per-row records under ``rows``.

DEFINITIONS (the owner's booleans; the tail reading per the cap ladder):
  loop_rep    := rep_ratio_4gram > 0.5 OR tail_rep_ratio_4gram > 0.5      (loop_rep_whole / loop_rep_tail carried separately;
                                                                            loop_rep_whole is the union's rep_ratio_loop detector,
                                                                            the reading padded_static_summary.json counts)
  loop_zlib   := zlib_ratio < 0.15                                        (empty text is 0.0: collapsed by definition)
  degenerate  := loop_rep OR loop_zlib                                    (classifier / contentless carried beside, NOT inside)
  at_cap      := finish_reason == 'length'                                (the store's declared estimand — the one the padded
                                                                            grades' at_cap_share reads; the union's re-tokenized
                                                                            estimand needs the target tokenizer, absent here)

CROSS-CHECKS, all refusing (exit 2) on a mismatch: the store's shard shas == completion.json's; row and at-cap counts ==
completion.json's; the sidecar's summed per-item counts == padded_static_summary.json's five detector ``n``; the union's
block == the summary's block; the goal set == the aggregate's per_goal keys and n_expected_goals; every union definition
pin verifies. Zeros are data; no cell is special-cased.

usage: degeneracy_items.py [--root <run-state static_padding dir>] [--key qwen35_4b] [--pads 16384,65536]
       (--config-family obj --pads 8192,32768 for the steered family's ADDED rungs: sharded across jobs, read as every shard*;
        --config-family obj_s2 --pads 0,8192,16384,32768,65536 for a seed replicate's five rungs, twelve shards each, read the same way)
                           [--families pasttense,codeattack,renellm,grok420] [--config-family obj|obj_s2|obj_s3]
                           [--cells <subset of the config family's cells; default all of them>]
                           [--stores-subdir stores] [--grades-subdir grades_padded]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
WT = Path(os.environ.get("F22_WORKTREE", str(HERE.parents[4])))   # scripts/sdfbase/full22/static_padding/<this> -> the worktree
for _p in (str(WT / "scripts"), str(WT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(HERE.parent))

from sisteer.capeval.tiers import cap_ladder as cl  # noqa: E402  (tail_rep_ratio / TAIL_WORDS / REP_RATIO_THRESHOLD / DEGENERATE_RULE)
from sisteer.eval import degeneracy as _dg  # noqa: E402
from sisteer.eval import degeneracy_union as du  # noqa: E402
from sisteer.eval import rep_ratio as _rr  # noqa: E402
from sisteer.eval import static_grading as sg  # noqa: E402

import make_padding_configs as gen  # noqa: E402  (private_run / cells_of / seed_of / KEYS / CELLS / OBJ_CELLS / FAMILIES — the ONE spelling)

Refuse = sg.Refuse

SCHEMA = "padded_static_degeneracy_items/1"
META_SCHEMA = "padded_static_degeneracy_items_meta/1"
ITEMS_NAME = "degeneracy_items.{cell}.jsonl"
META_NAME = "degeneracy_items.meta.json"
SUMMARY_NAME = "padded_static_summary.json"
COMPLETION_NAME = "completion.json"
AGGREGATE_NAME = "two_si_{family}_aggregate.json"

DEFAULT_ROOT = "$STATE/romp/candidate-si-refit/run-state/full22/static_padding"   # under the state root; expanded after parsing
DEFAULT_KEY = gen.KEYS[0]
DEFAULT_PADS = (16384, 65536)                     # the padded levels by default; the p0 zero-point is written when named (--pads 0)
DEFAULT_FAMILIES = ("pasttense", "codeattack", "renellm", "grok420")
CELLS = tuple(gen.CELLS)                          # the prompted-only family's cells; every family's are gen.cells_of(family) (ten / seven)
#: the config families this writer serves: THE GENERATOR'S (None, "obj", "obj_s2", "obj_s3" — module docstring), never a pair
#: retyped here; the argparse choices below are the same tuple's non-None members
CONFIG_FAMILIES = tuple(gen.ALL_FAMILY_NAMES)     # + the front door's per-view steered families (a phase view name each)

# --- thresholds: read from their homes, compared where two homes state one number -------------------------------------------
LOOP_REP_RATIO = float(du.LOOP_REP_RATIO)                # 0.5   sisteer.eval.rep_ratio.LOOP_REP_RATIO
TAIL_WORDS = int(cl.TAIL_WORDS)                          # 200   sisteer.capeval.tiers.cap_ladder.TAIL_WORDS
ZLIB_LEVEL = int(du.ZLIB_LEVEL)                          # 6
ZLIB_COLLAPSE_RATIO = float(du.ZLIB_COLLAPSE_RATIO)      # 0.15
ZLIB_HEALTHY_MEDIAN = float(du.ZLIB_HEALTHY_MEDIAN)      # 0.40  reported beside, never counted
CONTENTLESS_ALNUM_MIN = int(du.CONTENTLESS_ALNUM_MIN)    # 10
if float(cl.REP_RATIO_THRESHOLD) != LOOP_REP_RATIO:
    raise RuntimeError(f"the cap ladder's REP_RATIO_THRESHOLD ({cl.REP_RATIO_THRESHOLD}) differs from "
                       f"sisteer.eval.rep_ratio.LOOP_REP_RATIO ({LOOP_REP_RATIO}); one loop rule, two numbers")

#: (the summary's detector block, the sidecar count that reproduces it, the union's key for the same count). The summary's
#: ``at_cap_share`` is the shared body's finish_reason reading, which the union re-labels ``at_cap_finish_reason_share`` (its own
#: ``at_cap_share`` is the canonical re-tokenized estimand, undefined without the target tokenizer).
_UNION_DETECTORS = (("classifier_share", "n_classifier", "classifier_share"),
                    ("rep_ratio_loop_share", "n_loop_rep_whole", "rep_ratio_loop_share"),
                    ("zlib_collapse_share", "n_loop_zlib", "zlib_collapse_share"),
                    ("contentless_share", "n_contentless", "contentless_share"),
                    ("at_cap_share", "n_at_cap", "at_cap_finish_reason_share"))
_ANY_BOOLS = ("at_cap", "loop_rep", "loop_rep_whole", "loop_rep_tail", "loop_zlib", "classifier", "contentless", "degenerate")
_COUNTS = ("n_at_cap", "n_at_cap_degenerate", "n_at_cap_truncation_only", "n_degenerate", "n_loop_rep", "n_loop_rep_whole",
           "n_loop_rep_tail", "n_loop_zlib", "n_classifier", "n_contentless")


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(WT))
    except ValueError:
        return str(p)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def statistics_record() -> dict:
    """The fully-qualified statistics this writer calls, each with the sha256 of the file that defines it in THIS tree."""
    rr_sha, sg_sha, dg_sha, du_sha, cl_sha = (sg.sha256_file(Path(m.__file__)) for m in (_rr, sg, _dg, du, cl))
    return {
        "rep_ratio_4gram": {
            "symbol": "sisteer.eval.rep_ratio.rep_ratio", "path": _rel(_rr.__file__), "sha256": rr_sha,
            "rule": "1 - (distinct whitespace-token 4-grams / all 4-grams); 0.0 under 8 words",
            "threshold": {"symbol": "sisteer.eval.rep_ratio.LOOP_REP_RATIO", "value": LOOP_REP_RATIO, "comparison": ">"},
            "called_through": "sisteer.eval.degeneracy_union.classify_rows (per-row `rep_ratio`; du.rep_ratio is the same function object)",
            "pinned_by_union_as": du.DEFINITION_SOURCES["rep_ratio_loop"]["sha256"],
        },
        "tail_rep_ratio_4gram": {
            "symbol": "sisteer.capeval.tiers.cap_ladder.tail_rep_ratio", "path": _rel(cl.__file__), "sha256": cl_sha,
            "rule": cl.DEGENERATE_RULE,
            "window": {"symbol": "sisteer.capeval.tiers.cap_ladder.TAIL_WORDS", "value": TAIL_WORDS, "unit": "whitespace tokens"},
            "threshold": {"symbol": "sisteer.capeval.tiers.cap_ladder.REP_RATIO_THRESHOLD", "value": float(cl.REP_RATIO_THRESHOLD),
                          "comparison": ">"},
            "note": ("rep_ratio over the last TAIL_WORDS whitespace tokens (the whole text when shorter, so the two readings then "
                     "coincide); cap_ladder.rep_ratio is the origin's byte-for-byte copy and is asserted equal to "
                     "sisteer.eval.rep_ratio.rep_ratio on every row this writer reads"),
        },
        "zlib_ratio": {
            "symbol": "sisteer.eval.static_grading._zlib_ratio", "path": _rel(sg.__file__), "sha256": sg_sha,
            "rule": f"len(zlib.compress(utf8, {ZLIB_LEVEL})) / len(utf8); empty text is 0.0 (collapsed by definition)",
            "threshold": {"symbol": "sisteer.eval.static_grading.ZLIB_COLLAPSE_RATIO", "value": ZLIB_COLLAPSE_RATIO, "comparison": "<"},
            "healthy_median": {"symbol": "sisteer.eval.static_grading.ZLIB_HEALTHY_MEDIAN", "value": ZLIB_HEALTHY_MEDIAN,
                               "role": "reported beside the count (per-cell `zlib_median_ratio`), never part of it"},
            "called_through": "sisteer.eval.degeneracy_union.classify_rows (per-row `zlib_ratio`; du.zlib_ratio is sg._zlib_ratio)",
            "pinned_by_union_as": du.STATIC_GRADING_SHA256,
        },
        "contentless": {
            "symbol": "sisteer.eval.static_grading._alnum_count", "path": _rel(sg.__file__), "sha256": sg_sha,
            "rule": "alphanumeric characters in html.unescape(text)",
            "threshold": {"symbol": "sisteer.eval.static_grading.CONTENTLESS_ALNUM_MIN", "value": CONTENTLESS_ALNUM_MIN, "comparison": "<"},
            "called_through": "sisteer.eval.degeneracy_union.classify_rows (per-row `alnum_count` / `contentless`)",
            "role": "carried beside `degenerate`, not part of it",
        },
        "classifier": {
            "symbol": "sisteer.eval.degeneracy.classify", "path": _rel(_dg.__file__), "sha256": dg_sha,
            "rule": "tail_loop OR line_loop (degenerate == True); reason carried per row as classifier_reason",
            "called_through": "sisteer.eval.degeneracy_union.classify_rows (per-row `classifier`)",
            "pinned_by_union_as": du.DEFINITION_SOURCES["classifier"]["sha256"],
            "role": "carried beside `degenerate`, not part of it",
        },
        "at_cap": {
            "symbol": "sisteer.eval.degeneracy_union.at_cap_finish_reason", "path": _rel(du.__file__), "sha256": du_sha,
            "rule": "finish_reason == 'length' — the store's DECLARED estimand, the one padded_static_summary.json's at_cap_share "
                    "reads (static_grading.degeneration_block); the row's own at_cap field is held equal to it (a disagreement refuses)",
            "at_cap_retokenized": {"computed": False,
                                   "why": "the union's canonical re-tokenized estimand (>= 0.98 x cap, target tokenizer) needs the "
                                          "target's pinned tokenizer snapshot, which is not in this machine's HF cache; not "
                                          "substituted with a sibling tokenizer. Every at-cap row's completion_tokens == cap "
                                          "is recorded per cell instead (`at_cap_rows_completion_tokens_equal_cap`)"},
        },
        "union": {
            "symbol": "sisteer.eval.degeneracy_union.classify_rows(return_rows=True)", "path": _rel(du.__file__), "sha256": du_sha,
            "definition_sha": du.DEFINITION_SHA, "schema": du.SCHEMA,
            "note": ("every per-row number above is lifted from the union's per-row records; the union refuses to return unless "
                     "its per-row path agrees with static_grading.degeneration_block count for count"),
        },
    }


# --------------------------------------------------------------------------- one (level, family, cell) store -> per-goal lines

def build_cell(store_dir: Path, *, label: str, cap: int) -> tuple[list[dict], dict]:
    """Per-goal lines for ONE store, plus the union's cell-level readout (the campaign body's own output)."""
    shard_rows = sg.load_store_rows(store_dir)
    if not shard_rows:
        raise Refuse(f"[{label}] no rows under {store_dir}")
    union = du.classify_rows(shard_rows, label=label, return_rows=True)      # refuses a non-string response
    if union["at_cap_disagreements"]["n"]:
        raise Refuse(f"[{label}] {union['at_cap_disagreements']['n']} rows whose at_cap disagrees with finish_reason: "
                     f"{union['at_cap_disagreements']['examples']}")
    by_goal: dict = collections.defaultdict(list)
    shard_of: dict = collections.defaultdict(set)
    for (shard, r), u in zip(shard_rows, union["rows"]):
        tag = f"{shard}:{r.get('goal_id')}/{r.get('sample_idx')}"
        text = r["response"]
        if cl.rep_ratio(text) != u["rep_ratio"]:
            raise RuntimeError(f"[{label}] {tag}: cap_ladder.rep_ratio != sisteer.eval.rep_ratio.rep_ratio — the tail reading "
                               "would not be the canonical loop ratio")
        fr = r.get("finish_reason")
        if fr is None:
            raise Refuse(f"[{label}] {tag} carries no finish_reason; this leg records one on every row")
        ct = r.get("completion_tokens")
        if ct is None:
            raise Refuse(f"[{label}] {tag} carries no completion_tokens")
        if int(r.get("max_tokens", cap)) != int(cap):
            raise Refuse(f"[{label}] {tag} max_tokens={r.get('max_tokens')} != cap {cap}")
        at_cap = bool(u["at_cap_finish_reason"])
        if at_cap and int(ct) != int(cap):
            raise Refuse(f"[{label}] {tag} is at cap by finish_reason but completion_tokens={ct} != cap {cap}")
        tr = float(cl.tail_rep_ratio(text))
        loop_rep_whole = bool(u["rep_ratio_loop"])
        loop_rep_tail = bool(tr > LOOP_REP_RATIO)
        loop_rep = loop_rep_whole or loop_rep_tail
        loop_zlib = bool(u["zlib_collapse"])
        rec = {
            "sample_idx": r.get("sample_idx"), "variant_key": r.get("variant_key"), "variant_name": r.get("variant_name"),
            "finish_reason": fr, "at_cap": at_cap, "response_tokens": int(ct),
            "rep_ratio_4gram": float(u["rep_ratio"]), "tail_rep_ratio_4gram": tr, "zlib_ratio": float(u["zlib_ratio"]),
            "alnum_count": int(u["alnum_count"]),
            "loop_rep_whole": loop_rep_whole, "loop_rep_tail": loop_rep_tail, "loop_rep": loop_rep, "loop_zlib": loop_zlib,
            "classifier": bool(u["classifier"]), "classifier_reason": u["classifier_reason"], "contentless": bool(u["contentless"]),
            "degenerate": bool(loop_rep or loop_zlib),
        }
        by_goal[r["goal_id"]].append(rec)
        shard_of[r["goal_id"]].add(shard)

    lines: list[dict] = []
    for gid in sorted(by_goal):
        rows = sorted(by_goal[gid], key=lambda x: (x["sample_idx"] is None, x["sample_idx"]))
        shards = sorted(shard_of[gid])
        if len(shards) != 1:
            raise Refuse(f"[{label}] goal {gid} spans shards {shards}; the per-goal `shard` is undefined")
        at_cap_rows = [x for x in rows if x["at_cap"]]
        frs = sorted({x["finish_reason"] for x in rows})
        n_at_cap = len(at_cap_rows)
        n_at_cap_deg = sum(1 for x in at_cap_rows if x["degenerate"])
        line = {
            "goal_id": gid, "shard": shards[0],
            "finish_reason": ("length" if at_cap_rows else (frs[0] if len(frs) == 1 else None)), "finish_reasons": frs,
            "at_cap": n_at_cap > 0,
            "response_tokens": max(x["response_tokens"] for x in rows),
            "rep_ratio_4gram": max(x["rep_ratio_4gram"] for x in rows),
            "tail_rep_ratio_4gram": max(x["tail_rep_ratio_4gram"] for x in rows),
            "zlib_ratio": min(x["zlib_ratio"] for x in rows),
            "loop_rep": any(x["loop_rep"] for x in rows),
            "loop_rep_whole": any(x["loop_rep_whole"] for x in rows),
            "loop_rep_tail": any(x["loop_rep_tail"] for x in rows),
            "loop_zlib": any(x["loop_zlib"] for x in rows),
            "degenerate": any(x["degenerate"] for x in rows),
            "classifier": any(x["classifier"] for x in rows),
            "contentless": any(x["contentless"] for x in rows),
            "truncation_only": (n_at_cap - n_at_cap_deg) > 0,
            "n_rows": len(rows), "n_at_cap": n_at_cap, "n_at_cap_degenerate": n_at_cap_deg,
            "n_at_cap_truncation_only": n_at_cap - n_at_cap_deg,
            "n_degenerate": sum(1 for x in rows if x["degenerate"]),
            "n_loop_rep": sum(1 for x in rows if x["loop_rep"]),
            "n_loop_rep_whole": sum(1 for x in rows if x["loop_rep_whole"]),
            "n_loop_rep_tail": sum(1 for x in rows if x["loop_rep_tail"]),
            "n_loop_zlib": sum(1 for x in rows if x["loop_zlib"]),
            "n_classifier": sum(1 for x in rows if x["classifier"]),
            "n_contentless": sum(1 for x in rows if x["contentless"]),
            "rows": rows,
        }
        # the any-reduced booleans and the reduced scalars are ONE reading
        if line["degenerate"] != (line["loop_rep"] or line["loop_zlib"]):
            raise RuntimeError(f"[{label}] {gid}: degenerate != loop_rep or loop_zlib")
        if line["loop_rep"] != (line["rep_ratio_4gram"] > LOOP_REP_RATIO or line["tail_rep_ratio_4gram"] > LOOP_REP_RATIO):
            raise RuntimeError(f"[{label}] {gid}: loop_rep (any) != max ratio > {LOOP_REP_RATIO}")
        if line["loop_zlib"] != (line["zlib_ratio"] < ZLIB_COLLAPSE_RATIO):
            raise RuntimeError(f"[{label}] {gid}: loop_zlib (any) != min zlib_ratio < {ZLIB_COLLAPSE_RATIO}")
        if line["at_cap"] != (line["n_at_cap"] > 0) or line["truncation_only"] != (line["n_at_cap_truncation_only"] > 0):
            raise RuntimeError(f"[{label}] {gid}: at_cap / truncation_only disagree with their counts")
        lines.append(line)
    return lines, union


def _sums(lines: list[dict]) -> dict:
    out = {"n_rows": sum(l["n_rows"] for l in lines), "n_goals": len(lines),
           "rows_per_goal": dict(sorted(collections.Counter(l["n_rows"] for l in lines).items()))}
    for k in _COUNTS:
        out[k + "_rows"] = sum(l[k] for l in lines)
    out["goals"] = {**{k: sum(1 for l in lines if l[k]) for k in _ANY_BOOLS},
                    "truncation_only": sum(1 for l in lines if l["truncation_only"]),
                    "at_cap_and_degenerate": sum(1 for l in lines if l["at_cap"] and l["degenerate"]),
                    "at_cap_and_not_degenerate": sum(1 for l in lines if l["at_cap"] and not l["degenerate"]),
                    "degenerate_and_truncation_only": sum(1 for l in lines if l["degenerate"] and l["truncation_only"])}
    return out


def cross_check(lines: list[dict], union: dict, *, store_dir: Path, grades_dir: Path, family: str, cell: str, label: str) -> dict:
    """Every check the sidecar must pass against the grades it sits beside; refuses naming the first failures."""
    comp = json.loads((grades_dir / COMPLETION_NAME).read_text())
    summ = json.loads((grades_dir / SUMMARY_NAME).read_text())
    agg = json.loads((grades_dir / AGGREGATE_NAME.format(family=family)).read_text())
    c = comp["cells"][cell][family]
    deg = summ["families"][family]["cells"][cell]["degeneration"]
    per_goal = agg["cells"][cell]["attack_metrics"]["per_goal"]
    s = _sums(lines)
    checks: dict = {}

    shas = {sd.name: sg.sha256_file(sd / "responses.jsonl") for sd in sorted(store_dir.glob("shard*")) if sd.is_dir()}
    checks["store_shards_sha256_equal_completion"] = {"ok": shas == c["responses_sha256"], "shards": sorted(shas)}
    checks["n_rows_equal_completion_n_responses"] = {"sidecar": s["n_rows"], "completion": c["n_responses"],
                                                     "ok": s["n_rows"] == c["n_responses"]}
    checks["n_at_cap_rows_equal_completion_n_at_cap"] = {"sidecar": s["n_at_cap_rows"], "completion": c["n_at_cap"],
                                                         "ok": s["n_at_cap_rows"] == c["n_at_cap"]}
    checks["n_rows_equal_summary_n_rows"] = {"sidecar": s["n_rows"], "summary": deg["n_rows"], "ok": s["n_rows"] == deg["n_rows"]}
    checks["n_goals_equal_summary_n_goals"] = {"sidecar": s["n_goals"], "summary": deg["n_goals"], "ok": s["n_goals"] == deg["n_goals"]}
    for det, key, ukey in _UNION_DETECTORS:
        mine = s[key + "_rows"]
        checks[f"sidecar_sum_{key}_equal_summary_{det}"] = {
            "sidecar": mine, "summary": deg[det]["n"], "denominator": deg[det]["denominator"], "union": union[ukey]["n"],
            "ok": mine == deg[det]["n"] == union[ukey]["n"] and deg[det]["denominator"] == s["n_rows"] == union[ukey]["denominator"]}
    mine_ids, agg_ids = {l["goal_id"] for l in lines}, set(per_goal)
    checks["goal_ids_equal_aggregate_per_goal_keys"] = {
        "n_sidecar": len(mine_ids), "n_aggregate": len(agg_ids), "missing_from_sidecar": sorted(agg_ids - mine_ids)[:6],
        "extra_in_sidecar": sorted(mine_ids - agg_ids)[:6], "ok": mine_ids == agg_ids}
    checks["n_goals_equal_aggregate_n_expected_goals"] = {"sidecar": s["n_goals"], "aggregate": agg["n_expected_goals"],
                                                          "ok": s["n_goals"] == int(agg["n_expected_goals"])}
    checks["at_cap_rows_completion_tokens_equal_cap"] = {"ok": True, "cap": int(summ["cap"]),
                                                         "note": "asserted row by row in build_cell (a violation refuses)"}
    failed = [k for k, v in checks.items() if not v["ok"]]
    if failed:
        raise Refuse(f"[{label}] cross-check failed: {failed}: " + "; ".join(f"{k}={json.dumps(checks[k], default=str)}" for k in failed[:3]))
    return checks


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_family(root: Path, *, key: str = DEFAULT_KEY, pad: int, family: str, cells=None, config_family: str | None = None,
                 stores_subdir: str = "stores", grades_subdir: str = "grades_padded") -> dict:
    """Write every cell's sidecar and the meta for ONE (level, attack family) of ONE config family; returns the meta.
    ``config_family`` None is the prompted-only wave, ``"obj"`` the seed-1 steered one, ``"obj_s<N>"`` a seed replicate (the run
    name, the default cell set and the training seed follow from it through make_padding_configs); ``cells`` may narrow the
    family's cell set, never widen it."""
    if config_family not in CONFIG_FAMILIES:
        raise Refuse(f"config family {config_family!r} is not one this writer serves {list(CONFIG_FAMILIES)}")
    run = gen.private_run(key, int(pad), config_family)
    family_cells = tuple(gen.cells_of(config_family, key))    # a front door's per-view family: the roster's cells of that view of the key
    cells = tuple(cells) if cells else family_cells
    foreign = [c for c in cells if c not in family_cells]
    if foreign:
        raise Refuse(f"[{run}/{family}] --cells names cells outside the {config_family or 'prompted-only'} family's "
                     f"{list(family_cells)}: {foreign}")
    store_root = root / stores_subdir / run / family
    grades_dir = root / grades_subdir / run / family
    for name in (COMPLETION_NAME, SUMMARY_NAME, AGGREGATE_NAME.format(family=family)):
        if not (grades_dir / name).is_file():
            raise Refuse(f"[{run}/{family}] grades incomplete: {grades_dir / name} is missing")
    summ = json.loads((grades_dir / SUMMARY_NAME).read_text())
    cap = int(summ["cap"])
    # the ONE hard stop every producer applies (degeneracy_union.definition_sources_or_refuse): `ok` False is a failed verdict whether the
    # source is present and drifted or an in-repo pin that is ABSENT (a committed file this tree should carry, such as the historical
    # grading driver's evidence copy); only a source outside the repo that is not on this host reads None, and that one is recorded
    # rather than refused
    try:
        verified = du.definition_sources_or_refuse(du.verify_definition_sources())
    except RuntimeError as e:
        failed = getattr(e, "failed", None) or []
        record = getattr(e, "record", None) or {}
        raise Refuse(f"[{run}/{family}] union definition pins do not verify (absent in-repo source, or drift): {failed}: "
                     + "; ".join(f"{k}={json.dumps(record.get(k), default=str)}" for k in failed[:3]) + f" — {e}")

    per_cell: dict = {}
    for cell in cells:
        label = f"{run}/{family}/{cell}"
        store_dir = store_root / cell
        if not store_dir.is_dir():
            raise Refuse(f"[{label}] store missing: {store_dir}")
        lines, union = build_cell(store_dir, label=label, cap=cap)
        checks = cross_check(lines, union, store_dir=store_dir, grades_dir=grades_dir, family=family, cell=cell, label=label)
        out = grades_dir / ITEMS_NAME.format(cell=cell)
        _write_atomic(out, "".join(json.dumps(l, sort_keys=True) + "\n" for l in lines))
        back = [json.loads(l) for l in out.read_text().splitlines()]
        if back != lines:
            raise RuntimeError(f"[{label}] the written sidecar does not read back as what was computed")
        s = _sums(back)                                              # the file's numbers, not memory's
        n = s["n_rows"]
        per_cell[cell] = {
            **s,
            "shares_rows": {"at_cap": s["n_at_cap_rows"] / n, "loop_rep": s["n_loop_rep_rows"] / n,
                            "loop_rep_whole": s["n_loop_rep_whole_rows"] / n, "loop_rep_tail": s["n_loop_rep_tail_rows"] / n,
                            "loop_zlib": s["n_loop_zlib_rows"] / n, "degenerate": s["n_degenerate_rows"] / n,
                            "at_cap_degenerate": s["n_at_cap_degenerate_rows"] / n,
                            "at_cap_truncation_only": s["n_at_cap_truncation_only_rows"] / n,
                            "classifier": s["n_classifier_rows"] / n, "contentless": s["n_contentless_rows"] / n,
                            "denominator": n},
            "zlib_median_ratio": union["zlib_collapse_share"]["median_ratio"],
            "zlib_median_healthy": union["zlib_collapse_share"]["median_healthy"],
            "union_block": {ukey: {k: union[ukey].get(k) for k in ("share", "n", "denominator")}
                            for _, _, ukey in _UNION_DETECTORS},
            "union_definition_sha": union["definition_sha"],
            "cross_checks": checks,
            "file": {"path": str(out), "sha256": sg.sha256_file(out), "n_lines": len(back)},
            "store": str(store_dir),
        }
        print(f"[{label}] rows={n} goals={s['n_goals']} rows/goal={s['rows_per_goal']} at_cap_rows={s['n_at_cap_rows']} "
              f"loop_rep_rows={s['n_loop_rep_rows']} (whole {s['n_loop_rep_whole_rows']}, tail {s['n_loop_rep_tail_rows']}) "
              f"loop_zlib_rows={s['n_loop_zlib_rows']} degenerate_rows={s['n_degenerate_rows']} "
              f"at_cap_truncation_only_rows={s['n_at_cap_truncation_only_rows']} | goals: at_cap={s['goals']['at_cap']} "
              f"degenerate={s['goals']['degenerate']} truncation_only={s['goals']['truncation_only']} -> {out.name}", flush=True)

    head = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True)
    tracked = subprocess.run(["git", "-C", str(WT), "ls-files", "--error-unmatch", str(HERE)], capture_output=True, text=True)
    meta = {
        "schema": META_SCHEMA, "items_schema": SCHEMA, "written_at_utc": _utc_now(),
        "private_run": run, "key": key, "pad_tokens": int(pad), "family": family, "cells": list(cells), "cap": cap,
        # a steered family names itself; the prompted-only family's meta is filed in the registry, so its key set is frozen — and
        # so is the seed-1 family's: only a SEED REPLICATE adds its training_seed (make_padding_configs.seed_of; module docstring)
        **({"config_family": config_family, "config_family_cells": list(family_cells)} if config_family is not None else {}),
        **({"training_seed": gen.seed_of(config_family)} if gen.seed_of(config_family) is not None else {}),
        "target": summ.get("target"), "model": summ.get("model"), "model_revision": summ.get("model_revision"),
        "padding": summ.get("padding"),
        "source_store": str(store_root), "grades_dir": str(grades_dir),
        "files": {cell: per_cell[cell]["file"] for cell in cells},
        "thresholds": {"LOOP_REP_RATIO": LOOP_REP_RATIO, "TAIL_WORDS": TAIL_WORDS, "ZLIB_LEVEL": ZLIB_LEVEL,
                       "ZLIB_COLLAPSE_RATIO": ZLIB_COLLAPSE_RATIO, "ZLIB_HEALTHY_MEDIAN": ZLIB_HEALTHY_MEDIAN,
                       "CONTENTLESS_ALNUM_MIN": CONTENTLESS_ALNUM_MIN},
        "statistics": statistics_record(),
        "definitions": {
            "loop_rep": (f"rep_ratio_4gram > {LOOP_REP_RATIO} OR tail_rep_ratio_4gram > {LOOP_REP_RATIO} — the cap ladder's "
                         "DEGENERATE_RULE (docs/cap_ladder.md: a late-onset loop lives in the tail; under 200 words the readings "
                         "coincide). `loop_rep_whole` is the whole-text reading alone (the union's rep_ratio_loop detector, what "
                         "padded_static_summary.json's rep_ratio_loop_share counts); `loop_rep_tail` the tail reading alone"),
            "loop_zlib": f"zlib_ratio < {ZLIB_COLLAPSE_RATIO} (empty text is 0.0, collapsed by definition)",
            "degenerate": ("loop_rep OR loop_zlib — the registry owner's definition. `classifier` and `contentless` are carried "
                           "beside it and are NOT part of it; the union's five-detector `degenerate` is a wider column and is not emitted"),
            "at_cap": "finish_reason == 'length' (the store's declared estimand; see statistics.at_cap)",
            "truncation_only": "n_at_cap_truncation_only > 0: at least one at-cap row that is NOT degenerate",
        },
        "reduction": {
            "grain": ("one line per goal; the goal's variant rows (2 or 3 per goal in this wave — counted per family under "
                      "rows_per_goal, never assumed) are reduced as below and carried in full under `rows` (ids, flags and numbers; "
                      "no text)"),
            "shard": "the one shard holding every row of the goal (a goal spanning shards refuses)",
            "finish_reason": ("the at-cap row's finish_reason where one exists ('length'); else the rows' single distinct value; "
                              "else null, with `finish_reasons` (always emitted, sorted distinct) listing them"),
            "at_cap": "any row at cap (== n_at_cap > 0)",
            "response_tokens": "max over rows of the server's completion_tokens",
            "rep_ratio_4gram": "max over rows", "tail_rep_ratio_4gram": "max over rows", "zlib_ratio": "min over rows",
            "booleans": ("loop_rep, loop_rep_whole, loop_rep_tail, loop_zlib, classifier, contentless, degenerate: any over rows — "
                         "identical to thresholding the reduced scalar (max ratio > 0.5; min zlib < 0.15), asserted per goal"),
            "counts": {"n_rows": "rows of the goal", "n_at_cap": "rows with finish_reason == 'length'",
                       "n_at_cap_degenerate": "at-cap rows that are degenerate (loop_rep OR loop_zlib)",
                       "n_at_cap_truncation_only": "at-cap rows that are NOT degenerate — the case `any` hides",
                       "n_degenerate / n_loop_rep / n_loop_rep_whole / n_loop_rep_tail / n_loop_zlib / n_classifier / n_contentless":
                           "rows on which that flag fired"},
            "why_counts": ("a goal with one at-cap loop row and one at-cap non-loop row is `degenerate` AND `truncation_only`; the "
                           "any-reduced booleans alone would read it as degenerate and hide the genuine truncation"),
        },
        "per_cell": per_cell,
        "definition_sources_verified": {k: {"present": v.get("present"), "ok": v.get("ok")} for k, v in verified.items() if k != "module"},
        "provenance": {"repo_head": head.stdout.strip() or None, "worktree": str(WT),
                       "script": {"path": _rel(HERE), "sha256": sg.sha256_file(HERE), "tracked": tracked.returncode == 0},
                       "python": sys.version.split()[0]},
        "no_text": "no prompt or response text is stored in the sidecars or this file; ids, flags and numbers only",
    }
    _write_atomic(grades_dir / META_NAME, json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="the run-state static_padding directory (stores/ and grades_padded/ under it)")
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--pads", default=",".join(str(p) for p in DEFAULT_PADS))
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES), help="the ATTACK families")
    steered = [f for f in CONFIG_FAMILIES if f is not None]          # the generator's steered families, in its order (obj, obj_s2, obj_s3, the front door's per-view ones)
    hand_listed = [f for f in gen.FAMILIES if f is not None]         # the hand-listed key's, whose cell counts the name alone gives
    ap.add_argument("--config-family", default=None, choices=steered,
                    help=f"the CONFIG family: omit for the prompted-only wave (pad_static_<key>_p<pad>, {len(gen.CELLS)} cells); "
                         + "; ".join(f"{f} for pad_static_<key>{gen.seg_of(f)}_p<pad> ({len(gen.cells_of(f))} cells"
                                     + (f", training seed {gen.seed_of(f)})" if gen.seed_of(f) is not None else ", the seed-1 steered wave)")
                                     for f in hand_listed)
                         + "; a front-door key's phase view (its cells are the production roster's for that view of --key; D51)")
    ap.add_argument("--cells", default=None, help="a subset of the config family's cells (default: all of them)")
    ap.add_argument("--stores-subdir", default="stores")
    ap.add_argument("--grades-subdir", default="grades_padded")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from sisteer import site as _site
    root = _site.expand_path(args.root, key="--root")   # an unset $STATE refuses by name
    pads = [int(p) for p in args.pads.split(",") if p.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    cells = tuple(c.strip() for c in args.cells.split(",") if c.strip()) if args.cells else None
    for pad in pads:                          # a Refuse is SystemExit(2) and prints itself: nothing to catch
        for family in families:
            meta = write_family(root, key=args.key, pad=pad, family=family, cells=cells, config_family=args.config_family,
                                stores_subdir=args.stores_subdir, grades_subdir=args.grades_subdir)
            print(f"[{meta['private_run']}/{family}] meta -> {root / args.grades_subdir / meta['private_run'] / family / META_NAME}", flush=True)


if __name__ == "__main__":
    main()
