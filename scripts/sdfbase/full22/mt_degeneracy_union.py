#!/usr/bin/env python
"""Canonical 4,096-cap MULTI-TURN degeneracy under the campaign's ONE union classifier (sisteer.eval.degeneracy_union) — a NEW
file beside degeneracy_full22_union.json, never an edit of it.
WHY. Every canonical 4,096 multi-turn cell in the export (decode "sampled", caps (4096,)) reads degenerate_share NULL: the union
was run over the multi-turn families at the RETIRED 512 cap only (degeneracy_full22_union.json: actorattack / crescendo / pair at 512,
the static families at 4,096). The paper needs TRUNCATION separated from LOOPING, so this emits, per cell, the union's `by_reason`
decomposition, `loop_share` (the published word 4-gram rule) and BOTH at-cap estimands separately, at TURN level (rows) and GOAL level.
HOW THIS RUNS. As its OWN process with PYTHONPATH=<degeneracy-union worktree>/src (the validated bytes; the module is deliberately not on
main), refusing when the imported module does not live under --union-src, stamping the sha256 of degeneracy_union.py / rep_ratio.py /
static_grading.py / degeneracy.py, the worktree's git sha and verify_definition_sources() into the output. The target tokenizer is loaded
local_files_only for the canonical re-tokenized at-cap (>= 0.98 x 4,096, add_special_tokens=False). No prompt or response text is stored.
WHAT IS READ. The lane file transfer_stats_full22_cap4096.json names every canonical (view, cell, attack) record and the GRADES root it was
read from; the transcript store is <RS>/<root with grades_ -> transcripts_>/<aggregate.private_run>/<attack>/<cell>/shard*/. Rows come
from responses.jsonl (goal_id, order, response); target_timing.jsonl is joined per turn BY POSITION where its row count equals the
responses' (finish_reason -> the at_cap_finish_reason estimand; producer completion_tokens cross-checked against the re-tokenized count).
MERGED ROOTS (the relu_native stores): shard_r2_* / shard_r3_* sit beside shard*; the LAST progress row per goal (base shards first, then
r2, then r3) wins and that goal's responses are read from the winning shard only.
GOAL LEVEL. `jailbroken` := the grades aggregate's per-goal best_score > meta.threshold (0.5); a goal's degenerate := any of its turns
degenerate (the union), at_cap := any turn at cap (both estimands kept apart), loop := any turn rep_ratio_loop — plus the producer's
goal-level `capped` flag from progress.jsonl and the lane record's n_capped_goals / at_cap_finish_length beside for the cross-check.
Cells are keyed "<view>|<attack>|<cell>" exactly as build_export's UNION_CELLS are, with cap 4096.
usage:
  PYTHONPATH=src <venv python> scripts/sdfbase/full22/mt_degeneracy_union.py \
      --lane-file <RS>/transfer_stats_full22_cap4096.json --rs <RS> \
      --roster configs/sdfbase_full22_roster.yaml --out <RS>/degeneracy_union/degeneracy_mt_cap4096_union.json [--views v1,v2] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
# this script imports sisteer from --union-src only, so the site module is loaded by file path (scripts/site_paths.py)
sys.path.append(str(Path(__file__).resolve().parents[2]))
from site_paths import site_path  # noqa: E402
#: the run-state directory under the site's state root ($STATE); expanded after parsing, an unset one ends the run by name
RS_DEFAULT = "$STATE/romp/candidate-si-refit/run-state/full22"
# The union definition now lives in THIS repo. It used to default to a path inside a
# separate working tree, which made an idle checkout load-bearing for a script on main:
# the four modules load_union() requires existed in no commit reachable from main, so a clone could not run this
# script at all, and the bytes the numbers were computed under were whatever that untracked tree happened to hold.
# degeneracy.py was already here and byte-identical; its three siblings were simply left behind. Landing them makes
# the default self-contained. `--union-src` is KEPT so an audit can still point the run at another tree on purpose,
# and load_union()'s refusals are unchanged — it still demands the path be on sys.path and that every sisteer module
# resolve from inside it, so a foreign import is caught exactly as before.
UNION_SRC_DEFAULT = Path(__file__).resolve().parents[3] / "src"
CAP = 4096
SCHEMA = "degeneracy_union_mt_cap4096/1"
ATTACKS = ("crescendo", "pair", "actorattack")


class Refuse(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"REFUSING: {msg}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_union(union_src: Path):
    """Import the union from --union-src and REFUSE if any sisteer module resolved elsewhere (a pinned-foreign-import would mean
    the numbers were computed under bytes nobody validated). Verbatim the sampled precedent's rule."""
    union_src = union_src.resolve()
    if not (union_src / "sisteer" / "eval" / "degeneracy_union.py").is_file():
        raise Refuse(f"--union-src {union_src} has no sisteer/eval/degeneracy_union.py")
    if str(union_src) not in [str(Path(p).resolve()) for p in sys.path if p]:
        raise Refuse(f"run with PYTHONPATH={union_src} (this process must import sisteer from the union tree only); sys.path[:3]={sys.path[:3]}")
    import sisteer  # noqa: E402
    from sisteer.eval import degeneracy_union as du  # noqa: E402
    from sisteer.eval import static_grading as sg  # noqa: E402
    for mod in (sisteer, du, sg):
        f = Path(mod.__file__).resolve()
        if union_src not in f.parents:
            raise Refuse(f"{mod.__name__} imported from {f}, not from --union-src {union_src}")
    files = {name: union_src / "sisteer" / "eval" / f"{name}.py" for name in ("degeneracy_union", "rep_ratio", "static_grading", "degeneracy")}
    stamp = {"union_src": str(union_src), "files": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in files.items()},
             "DEFINITION_SHA": du.DEFINITION_SHA, "schema_union": du.SCHEMA, "detectors": list(du.DETECTORS),
             "canonical_at_cap": du.CANONICAL_AT_CAP, "at_cap_fraction": du.AT_CAP_RETOKENIZED_FRACTION}
    try:
        stamp["worktree_git_sha"] = subprocess.check_output(["git", "-C", str(union_src), "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        stamp["worktree_git_sha"] = f"<unavailable: {type(e).__name__}>"
    try:
        v = du.verify_definition_sources()
    except Exception as e:  # noqa: BLE001
        raise Refuse(f"verify_definition_sources() failed: {type(e).__name__}: {e}")
    # the ONE hard stop every producer applies (the union tree's degeneracy_union.definition_sources_or_refuse): `ok` False refuses by
    # name whether the source is present and drifted or an in-repo pin that is ABSENT (the historical grading driver's committed
    # evidence copy, say); only a source outside the repo that is not on this host (`ok` None) is stamped rather than refused
    rule = getattr(du, "definition_sources_or_refuse", None)
    if rule is None:
        raise Refuse(f"the union tree's degeneracy_union ({du.__file__}) has no definition_sources_or_refuse — the shared hard stop on its "
                     "definition pins; this process does not stamp an unverified record")
    try:
        stamp["verify_definition_sources"] = rule(v)
    except RuntimeError as e:
        raise Refuse(f"union definition pins do not verify (absent in-repo source, or drift): {getattr(e, 'failed', None) or ''} — {e}")
    return du, stamp


# --------------------------------------------------------------------------------------------- pure helpers (tested)
def base_target_of(view: str) -> str:
    """The roster target a lane view serves: `gemma4e4b_it` / `llama8b_shortconst` -> the base key."""
    for suf in ("_shortconst", "_it"):
        if view.endswith(suf):
            return view[: -len(suf)]
    return view


def transcripts_root_of(grades_root: str) -> str:
    """`grades_<x>` -> `transcripts_<x>`, also under a subdirectory (`relu_native/grades_relu_native_bu_gemma4e4b`)."""
    head, _, name = grades_root.rpartition("/")
    if not name.startswith("grades_"):
        raise Refuse(f"lane record root {grades_root!r} does not start with grades_")
    return (head + "/" if head else "") + "transcripts_" + name[len("grades_"):]


VIEW_SUFFIXES = ("_it", "_shortconst")


def transcripts_root_candidates(grades_root: str) -> list[str]:
    """The direct name first, then the name with the lane's VIEW SUFFIX stripped from the tag: the grades roots carry it
    (grades_obj4096_gemma4e4b_it) and their transcripts roots do not (transcripts_obj4096_gemma4e4b); the convention is not
    uniform (grades_anchor4096r2_qwen35_4b matches directly), so the resolver derives, never assumes."""
    direct = transcripts_root_of(grades_root)
    out = [direct]
    for suf in VIEW_SUFFIXES:
        if direct.endswith(suf):
            out.append(direct[: -len(suf)])
    return out


def resolve_transcripts_root(rs: Path, grades_root: str) -> tuple[Path | None, list[str]]:
    """The first candidate that exists under RS, with every candidate tried (recorded in the cell)."""
    cands = transcripts_root_candidates(grades_root)
    for c in cands:
        if (rs / c).is_dir():
            return rs / c, cands
    return None, cands


_SHARD_RE = re.compile(r"^shard(?:_r(\d+)_)?(\d+)$")


def shard_order(name: str) -> tuple:
    """Base shards first (r0), then shard_r2_*, shard_r3_* ... — the merged-root convention (later runs win)."""
    m = _SHARD_RE.match(name)
    if not m:
        return (999, name)
    return (int(m.group(1) or 0), int(m.group(2)))


def winning_shards(store: Path) -> dict:
    """goal_id -> the shard directory whose progress row for that goal is the LAST in merged order (the loader's rule)."""
    win: dict = {}
    for sd in sorted((p for p in store.glob("shard*") if p.is_dir()), key=lambda p: shard_order(p.name)):
        pp = sd / "progress.jsonl"
        if not pp.is_file():
            continue
        for line in pp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            if g.get("goal_id"):
                win[g["goal_id"]] = sd
    return win


def load_store_rows(store: Path) -> tuple[list, dict, dict]:
    """[(shard_name, row)] over every response row of each goal's WINNING shard, the per-goal progress row, and per-shard timing
    alignment facts. A response row carries response / goal_id / sample_idx (:= order) and, where target_timing.jsonl aligns by
    position with responses.jsonl (equal row counts), finish_reason + completion_tokens from the producer."""
    win = winning_shards(store)
    rows: list = []; progress: dict = {}; timing_facts: dict = {}
    for sd in sorted((p for p in store.glob("shard*") if p.is_dir()), key=lambda p: shard_order(p.name)):
        rp, pp, tp = sd / "responses.jsonl", sd / "progress.jsonl", sd / "target_timing.jsonl"
        if pp.is_file():
            for line in pp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        g = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if g.get("goal_id") and win.get(g["goal_id"]) == sd:
                        progress[g["goal_id"]] = {"capped": bool(g.get("capped")), "outcome": str(g.get("outcome")), "turns": g.get("turns"),
                                                  "n_target_forwards": g.get("n_target_forwards")}
        if not rp.is_file():
            continue
        rlines = [l for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
        tlines = [l for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()] if tp.is_file() else []
        aligned = bool(tlines) and len(tlines) == len(rlines)
        timing_facts[sd.name] = {"responses": len(rlines), "timing_rows": len(tlines), "aligned": aligned}
        for i, l in enumerate(rlines):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            gid = r.get("goal_id")
            if gid in win and win[gid] != sd:
                continue                                   # this goal's last run lives in a later shard
            row = {"response": r.get("response"), "goal_id": gid, "sample_idx": r.get("order")}
            if aligned:
                try:
                    t = json.loads(tlines[i])
                    if t.get("finish_reason") is not None:
                        row["finish_reason"] = t["finish_reason"]
                        row["at_cap"] = (t["finish_reason"] == "length")
                    if t.get("completion_tokens") is not None:
                        row["completion_tokens"] = t["completion_tokens"]
                except json.JSONDecodeError:
                    pass
            rows.append((sd.name, row))
    return rows, progress, timing_facts


def _share(k: int, n: int) -> dict:
    return {"k": k, "n": n, "p": (k / n) if n else None}


def goal_level(per_row: list, progress: dict, jailbroken: dict) -> dict:
    """Goal-level readings from the union's per-row flags: any-turn degenerate / at_cap (both estimands) / loop, the producer's
    `capped` flag, and the defended-collapse split (a goal is defended when its judged best score <= threshold)."""
    g: dict = collections.defaultdict(lambda: {"deg": False, "ac_tok": False, "ac_fr": False, "ac_fr_defined": False, "loop": False, "n": 0})
    for x in per_row:
        e = g[x["goal_id"]]; e["n"] += 1
        e["deg"] |= bool(x.get("degenerate")); e["ac_tok"] |= bool(x.get("at_cap_retokenized")); e["loop"] |= bool(x.get("rep_ratio_loop"))
        if x.get("at_cap_finish_reason") is not None:
            e["ac_fr_defined"] = True; e["ac_fr"] |= bool(x["at_cap_finish_reason"])
    n = len(g)
    out = {"denominator": "goals", "n_goals": n, "canonical_at_cap": "at_cap_retokenized_any_turn",
           "degenerate_any_turn": _share(sum(e["deg"] for e in g.values()), n),
           "at_cap_retokenized_any_turn": _share(sum(e["ac_tok"] for e in g.values()), n),
           "at_cap_finish_reason_any_turn": (_share(sum(e["ac_fr"] for e in g.values()), n) if all(e["ac_fr_defined"] for e in g.values()) and n else {"defined": False}),
           "loop_any_turn": _share(sum(e["loop"] for e in g.values()), n),
           "producer_capped_flag": _share(sum(1 for gid in g if (progress.get(gid) or {}).get("capped")), n),
           "turns_per_goal_mean": round(sum(e["n"] for e in g.values()) / n, 2) if n else None}
    if jailbroken:
        jb = {gid: bool(jailbroken.get(gid)) for gid in g}
        defended = [gid for gid, v in jb.items() if not v]
        out["judged"] = {"n_goals_judged": sum(1 for gid in g if gid in jailbroken), "jailbroken": _share(sum(jb.values()), n),
                         "defended": _share(len(defended), n),
                         "defended_any_turn_degenerate": _share(sum(1 for gid in defended if g[gid]["deg"]), len(defended)),
                         "defended_any_turn_at_cap": _share(sum(1 for gid in defended if g[gid]["ac_tok"]), len(defended)),
                         "jailbroken_any_turn_degenerate": _share(sum(1 for gid, v in jb.items() if v and g[gid]["deg"]), sum(jb.values()))}
    return out


def find_aggregate(rs: Path, grades_root: str, view: str, attack: str) -> Path | None:
    root = rs / grades_root
    cands = sorted(root.glob(f"*/two_si_{attack}_aggregate.json"))
    if not cands:
        return None
    for c in cands:
        if c.parent.name == view:
            return c
    if len(cands) == 1:
        return cands[0]
    for c in cands:
        if c.parent.name == base_target_of(view):
            return c
    return cands[0]


@functools.lru_cache(maxsize=None)
def tokenizer_for(model_id: str):
    from transformers import AutoTokenizer  # noqa: E402
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError as e:
        raise Refuse(f"tokenizer for {model_id!r} is not in the local HF cache ({type(e).__name__}); fetch it first: "
                     f"hf download {model_id} --include 'tokenizer*' special_tokens_map.json — then rerun") from e
    return tok


def tokenizer_snapshot(model_id: str) -> str | None:
    """The cached snapshot the tokenizer was read from (recorded: the pinned revision is not enforced offline)."""
    hub = Path(os.environ.get("HF_HOME", HOME / ".cache/huggingface")) / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    snaps = sorted(hub.glob("*")) if hub.is_dir() else []
    return snaps[-1].name if snaps else None


def one_record(du, *, rs: Path, view: str, cell: str, attack: str, rec: dict, model_id: str, threshold: float) -> dict:
    t0 = time.time()
    key = f"{view}|{attack}|{cell}"
    grades_root = rec.get("root")
    base = {"schema": SCHEMA, "key": key, "view": view, "target": base_target_of(view), "attack": attack, "cell": cell, "cap": CAP,
            "lane_record": {"grades_root": grades_root, "k": rec.get("k"), "n": rec.get("n"), "p": rec.get("p"),
                            "n_capped_goals": rec.get("n_capped_goals"), "at_cap_finish_length": rec.get("at_cap_finish_length"),
                            "degeneracy": rec.get("degeneracy"), "decode_cap_tokens": rec.get("decode_cap_tokens"), "namespace": rec.get("namespace")}}
    if not grades_root:
        return {**base, "status": "pending", "note": "lane record names no grades root"}
    agg_p = find_aggregate(rs, grades_root, view, attack)
    if agg_p is None:
        return {**base, "status": "pending", "note": f"no two_si_{attack}_aggregate.json under {rs / grades_root}"}
    agg = load_json(agg_p)
    private_run = agg.get("private_run")
    troot, cands = resolve_transcripts_root(rs, grades_root)
    if troot is None:
        return {**base, "status": "pending", "note": f"no transcripts root under {rs} for {grades_root} (tried {cands})",
                "aggregate": {"path": str(agg_p), "private_run": private_run}}
    store = troot / str(private_run) / attack / cell
    if not store.is_dir():
        alts = sorted(troot.glob(f"*/{attack}/{cell}"))
        if len(alts) == 1:
            store = alts[0]
        else:
            return {**base, "status": "pending", "note": f"transcript store absent: {store} (root candidates tried {cands}; alternatives under {troot}: {[str(a) for a in alts][:3]})",
                    "aggregate": {"path": str(agg_p), "private_run": private_run}}
    rows, progress, timing_facts = load_store_rows(store)
    if not rows:
        return {**base, "status": "pending", "note": f"no response rows under {store}", "store": {"path": str(store)}}
    tok = tokenizer_for(model_id)
    res = du.classify_rows(rows, cap=CAP, n_tokens=functools.partial(du.count_tokens, tok), label=key, return_rows=True)
    per_row = res.pop("rows")
    prod = [(int(r.get("completion_tokens")), x.get("n_tokens")) for (_, r), x in zip(rows, per_row)
            if str(r.get("completion_tokens", "")).isdigit() and isinstance(x.get("n_tokens"), int)]
    prod_cmp = ({"n_compared": len(prod), "equal": sum(1 for a, b in prod if a == b),
                 "mean_abs_diff": round(sum(abs(a - b) for a, b in prod) / len(prod), 2)} if prod else {"n_compared": 0})
    m = ((agg.get("cells") or {}).get(cell) or {}).get("attack_metrics") or {}
    per_goal = m.get("per_goal") or {}
    jailbroken = {g: float(v.get("best_score", 0.0)) > threshold for g, v in per_goal.items()} if per_goal else {}
    gl = goal_level(per_row, progress, jailbroken)
    for x in per_row:                                       # the per-goal list the union's goal_split wants
        x["jailbroken"] = bool(jailbroken.get(x["goal_id"])) if jailbroken else None
    split = None
    if jailbroken:
        pg: dict = collections.defaultdict(list)
        for x in per_row:
            pg[x["goal_id"]].append({"jailbroken": bool(x["jailbroken"]), "degenerate": bool(x.get("degenerate"))})
        split = du.goal_split(pg)
    ds = res["degenerate_share"]
    rec_out = {**base, "status": "measured", "private_run": private_run, "tokenizer": {"model_id": model_id, "snapshot": tokenizer_snapshot(model_id)},
               "definition_sha": res.get("definition_sha"), "n_rows": len(rows), "n_goals": len(progress) or gl["n_goals"],
               "denominator_rule": {"here": "target replies of each goal's WINNING shard (the last progress row per goal in merged order; superseded goals' rows dropped)",
                                    "lane_at_cap_finish_length": "every target_timing row of every shard (superseded runs included)",
                                    "n_rows_here": len(rows), "n_rows_lane": (rec.get("at_cap_finish_length") or {}).get("n"),
                                    "transcripts_root": str(troot.name), "root_candidates_tried": cands},
               "shards": {sd.name: sha256_file(sd / "responses.jsonl") for sd in sorted(store.glob("shard*")) if (sd / "responses.jsonl").is_file()},
               "timing_alignment": timing_facts, "producer_vs_retokenized_tokens": prod_cmp,
               "canonical_at_cap": du.CANONICAL_AT_CAP,       # at_cap_retokenized: re-tokenized length >= 0.98 x cap, target tokenizer, no special tokens
               "at_cap_estimands": {"at_cap_retokenized": "CANONICAL — a measurement of the response (the only predicate defined across the paper)",
                                    "at_cap_finish_reason": "the producer's declaration (target_timing.jsonl finish_reason == length, joined per turn by position); "
                                                            "NOT the same column as a component's stop-reason at-cap unless it says so"},
               "turn_level": {"denominator": "rows (target replies)",
                              "degenerate_share": ds, "by_reason": ds.get("by_reason"),
                              "loop_share": res["rep_ratio_loop_share"], "classifier_share": res["classifier_share"],
                              "zlib_collapse_share": res["zlib_collapse_share"], "contentless_share": res["contentless_share"],
                              "at_cap_retokenized_share": res["at_cap_retokenized_share"], "at_cap_finish_reason_share": res["at_cap_finish_reason_share"],
                              "at_cap_cross_tab": res.get("at_cap_cross_tab"), "at_cap_disagreements": res.get("at_cap_disagreements"),
                              "truncation_only_share": _share(ds.get("by_reason", {}).get("at_cap", 0), len(rows)),
                              "loop_without_cap_share": _share(sum(v for k, v in (ds.get("by_reason") or {}).items() if "at_cap" not in k), len(rows))},
               "goal_level": gl, "goal_split_union": split,
               "lane_cross_check": {"at_cap_finish_length_lane": (rec.get("at_cap_finish_length") or {}).get("p"),
                                    "at_cap_finish_reason_here": res["at_cap_finish_reason_share"].get("share"),
                                    "n_capped_goals_lane": rec.get("n_capped_goals"),
                                    "producer_capped_goals_here": gl["producer_capped_flag"]["k"]},
               "store": {"path": str(store)}, "aggregate": {"path": str(agg_p), "sha256": sha256_file(agg_p), "threshold": threshold},
               "wall_s": round(time.time() - t0, 1)}
    return rec_out


def markdown(out: dict) -> str:
    L = [f"# Degeneracy under the union classifier — canonical 4,096 multi-turn cells ({out['summary']['measured']} measured, {out['summary']['pending']} pending)",
         "", f"Union `{out['meta']['union']['DEFINITION_SHA'][:12]}` from {out['meta']['union']['union_src']} (worktree {str(out['meta']['union'].get('worktree_git_sha'))[:12]}); "
         "loop = word 4-gram repetition > 0.5; at-cap canonical = re-tokenized >= 0.98 x 4,096, finish_reason estimand beside; degenerate = union of five detectors (at-cap included) — "
         "`by_reason` separates truncation from looping. Turn level = target replies; goal level = any turn.", "",
         "| cell | replies | degenerate | of which at-cap only | loop | at-cap (retok) | at-cap (finish) | goals | goal degenerate | goal at-cap | producer capped |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in sorted(out["cells"].items()):
        if c["status"] != "measured":
            L.append(f"| {k} | *{c['status']}* | | | | | | | | | |"); continue
        t, g = c["turn_level"], c["goal_level"]
        fr = t["at_cap_finish_reason_share"]; frs = f"{100 * fr['share']:.1f}%" if fr.get("share") is not None else "undefined"
        L.append(f"| {k} | {c['n_rows']} | {100 * t['degenerate_share']['share']:.1f}% | {100 * (t['truncation_only_share']['p'] or 0):.1f}% | {100 * t['loop_share']['share']:.2f}% | "
                 f"{100 * t['at_cap_retokenized_share']['share']:.1f}% | {frs} | {g['n_goals']} | {100 * (g['degenerate_any_turn']['p'] or 0):.1f}% | {100 * (g['at_cap_retokenized_any_turn']['p'] or 0):.1f}% | {100 * (g['producer_capped_flag']['p'] or 0):.1f}% |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-file", default=f"{RS_DEFAULT}/transfer_stats_full22_cap4096.json")
    ap.add_argument("--rs", default=str(RS_DEFAULT))
    ap.add_argument("--union-src", default=str(UNION_SRC_DEFAULT))
    ap.add_argument("--roster", required=True, help="configs/sdfbase_full22_roster.yaml (the target model per key, for the tokenizer)")
    ap.add_argument("--views", default="", help="comma list of lane views (default all)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N records (smoke)")
    ap.add_argument("--merge-from", default="", help="a previous output: its measured cells are copied verbatim (provenance kept) and only its "
                                                     "pending cells are re-run — into a NEW --out, never into it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.rs, a.lane_file = str(site_path(a.rs, "--rs")), str(site_path(a.lane_file, "--lane-file"))
    du, stamp = load_union(Path(a.union_src))
    import yaml  # noqa: E402
    roster = yaml.safe_load(Path(a.roster).read_text())
    models = {k: v["model"] for k, v in roster["targets"].items()}
    rs = Path(a.rs); lane_p = Path(a.lane_file); lane = load_json(lane_p)
    threshold = float(lane["meta"].get("threshold", 0.5))
    views = [v for v in a.views.split(",") if v] or list(lane["models"])
    out = {"schema": SCHEMA, "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "cap": CAP,
           "meta": {"union": stamp, "lane_file": {"path": str(lane_p), "sha256": sha256_file(lane_p)}, "rs": str(rs), "threshold": threshold,
                    "roster": {"path": str(a.roster), "sha256": sha256_file(Path(a.roster))}, "tokenizers": {}, "argv": sys.argv},
           "cells": {}, "summary": {}}
    prev = load_json(Path(a.merge_from)) if a.merge_from else None
    prev_cells = (prev or {}).get("cells") or {}
    if prev is not None:
        out["meta"]["merged_from"] = {"path": str(Path(a.merge_from)), "sha256": sha256_file(Path(a.merge_from)),
                                      "measured_copied": sum(1 for c in prev_cells.values() if c.get("status") == "measured"),
                                      "pending_rerun": sum(1 for c in prev_cells.values() if c.get("status") != "measured")}
    n = 0
    for view in views:
        v = lane["models"].get(view)
        if not v:
            raise Refuse(f"view {view!r} not in the lane file")
        model_id = models.get(base_target_of(view))
        if not model_id:
            raise Refuse(f"no roster model for view {view} (base {base_target_of(view)})")
        out["meta"]["tokenizers"][view] = {"model_id": model_id, "snapshot": tokenizer_snapshot(model_id)}
        for cell, c in (v.get("cells") or {}).items():
            for attack in ATTACKS:
                rec = c.get(attack)
                if not isinstance(rec, dict):
                    continue
                key = f"{view}|{attack}|{cell}"
                if prev is not None and prev_cells.get(key, {}).get("status") == "measured":
                    r = dict(prev_cells[key]); r["copied_from_previous_output"] = True
                else:
                    r = one_record(du, rs=rs, view=view, cell=cell, attack=attack, rec=rec, model_id=model_id, threshold=threshold)
                out["cells"][r["key"]] = r
                print(f"[mt-union] {r['key']}: {r['status']}" + (f" rows={r['n_rows']} degenerate={r['turn_level']['degenerate_share']['share']:.4f} "
                      f"at_cap_retok={r['turn_level']['at_cap_retokenized_share']['share']:.4f} loop={r['turn_level']['loop_share']['share']:.4f} ({r['wall_s']}s)"
                      if r["status"] == "measured" else f" — {r.get('note')}"), flush=True)
                n += 1
                if a.limit and n >= a.limit:
                    break
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break
    meas = [c for c in out["cells"].values() if c["status"] == "measured"]
    by_attack: dict = collections.defaultdict(list)
    for c in meas:
        by_attack[c["attack"]].append(c["turn_level"]["degenerate_share"]["share"])
    out["summary"] = {"records": n, "measured": len(meas), "pending": n - len(meas),
                      "turn_degenerate_share_median_by_attack": {k: sorted(v)[len(v) // 2] for k, v in by_attack.items() if v},
                      "cells_over_5pct_degenerate": sum(1 for c in meas if c["turn_level"]["degenerate_share"]["share"] > 0.05),
                      "cells_over_5pct_loop": sum(1 for c in meas if c["turn_level"]["loop_share"]["share"] > 0.05)}
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not a.limit:
        raise Refuse(f"{outp} exists — never overwritten; pick a new --out")
    tmp = outp.with_suffix(outp.suffix + ".tmp"); tmp.write_text(json.dumps(out, indent=1, default=str)); os.replace(tmp, outp)
    outp.with_suffix(".md").write_text(markdown(out))
    print(f"[mt-union] {len(meas)} measured / {n - len(meas)} pending -> {outp} (+ .md)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
