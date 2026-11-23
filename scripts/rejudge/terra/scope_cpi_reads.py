#!/usr/bin/env python3
"""The XSTest RECORD READS of the strength lanes (a supplement to the C/P/I scope): for every run whose over-refusal reading is an arm
block of a strength battery record, one unit per text the record reads, per item, trial and rung store, with the request key the
Sonnet judge's store recorded for that text.

    HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts python scripts/rejudge/terra/scope_cpi_reads.py build --runs RUNS.txt --out SCOPE/reads \
        [--registry ~/registry/registry.sqlite] [--no-shared-rung]
    ... scope_cpi_reads.py union --scope SCOPE/anchors --scope SCOPE/reads --out SCOPE/build     # the manifest build's scope

WHY. ``scope_cpi.py`` takes a strength-lane run's ONE judged store (``judgment.store_uri``: the arm's top-rung store). The record the
paper's rows come from (``battery_curves.json``, ``scripts/sdfbase/full22/strength/battery_record.py``) reads each safe XSTest item
at the FIRST rung (512 -> 2048 -> 8192) where its draw terminated, with the judge's verdict on THAT rung's text, so for a third of the
items the text the record counts is not the text the top-rung store holds (a floor-sample re-draw). This scope holds the texts the
record reads.

WHICH RUNS. Exactly ``--runs`` (e.g. the strength-lane paper runs and their same-standup ``none`` anchors). A run's record block is
named by its ``extra.provenance_legacy`` (``<battery_curves.json>#targets.<T>.cells.<view:cell>.over_refusal.<arm>``); the block's
top store must be the run's ``store_uri`` and the store of its complete Sonnet judgment, and the cell's trial the run's trial.

WHAT THE RECORD READS (``read`` in each unit's note):
  per_item     ``battery_record.per_item_reads`` itself (imported from the checkout; identical in the records' tool versions 7f65eecb
               and c090ff98) over the block's ``stores_by_rung``: each rung's ``gen_stats.json`` xstest block (``xstest_block``) and
               its judge rows (``safe_rows_by_item``). One unit per safe item at its read rung. The reads' census
               (``graded_from_reads``: full / partial refusals, n judged, graded score sum, at-cap, abstentions, reads per rung) must
               equal the record block's, or the run gets no unit.
  shared_rung  the record's ``shared_rung_pair`` beside the primary read (Arditi's rule: ONE whole store per arm at the deepest rung
               both arms hold a judged store; its floor-sample re-draws included): every safe item of that store (Sonnet's abstentions
               kept, as in the per-item read, so a new judge's labels on them are there). A ``none``
               arm pairs with every cell of its view, so it carries one shared read per distinct rung its cells name. The record's
               bootstrap draws for ``shared_rung_pair`` come from the same per-target stream as the primary pairs, so replaying the
               stream for a new judge needs these labels too. ``--no-shared-rung`` leaves them out.
THE TEXT AND THE KEY, as ``scope_cpi.py``: the pinned prompt (``tiers.judged.xstest_rows``), the read store's spool
(``private/gens_xstest_v2.jsonl``, :func:`scope_cpi.read_spool`), the judge row's task, rubric, protocol and model the Sonnet pins,
and ``JudgeRequest.key()`` recomputed and equal to the row's ``request_key`` (a mismatch is a problem and gets no unit).

NOT A FILING SCOPE. A read unit carries its run's (item, trial, sample 0, turn -1): the key of the top-rung unit the paper-row scope
filed, so filing these units would fold over the filed rows. ``report.json`` carries ``not_for_filing`` and ``file_records.py``
refuses such a scope. The reads reach the notebook through ``reads_sidecar.py`` (one row per (run, item, trial, read, rung)).

OUTPUT (``--out``): ``units.tsv.gz`` (the scope columns; ``note`` = {read, rung, top_rung, cell, at_cap, regenerated, subset,
text_sha256, abstention_reason}, ``cell`` indexing the run's ``sources``), ``texts.jsonl.gz`` ({k, rubric, item_id, prompt, response},
as scope_cpi), ``runs.jsonl`` (per run: the record block, its sources per rung with the judge_items and spool sha256, the census
check), ``report.json``. ``union`` writes the scope a manifest build reads: the texts and units of every ``--scope`` (a key's text
must agree), every runs.jsonl row, and ``not_for_filing``.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.terra import cpi as C  # noqa: E402
from rejudge.terra import scope_cpi as SC  # noqa: E402

HOME = os.path.expanduser("~")
EVAL = "xstest"
RUBRIC = C.RUBRIC_OF_EVAL[EVAL]
TASK = C.TASK_OF_EVAL[EVAL]
SPOOL = f"gens_{TASK.split('/', 1)[1]}.jsonl"
BATTERY_RECORD = os.path.join(_SCRIPTS, "sdfbase", "full22", "strength", "battery_record.py")
_FRAG = re.compile(r"^targets\.(?P<t>[^.]+)\.cells\.(?P<cell>.+?)\.over_refusal\.(?P<arm>.+)$")
NOT_FOR_FILING = ("a read unit carries its run's (item, trial, sample, turn), the key of the top-rung unit already filed: filing it "
                  "would fold over that row; the reads reach the notebook through scripts/rejudge/terra/reads_sidecar.py")
UNIT_HEADER = "run_id\titem_id\ttrial\tsample\tturn\tcustom_id\trubric\teval_id\torig_label\torig_status\tnote\n"


def battery_record():
    """The record producer's module, from this checkout (its read rule is imported, never re-implemented)."""
    spec = importlib.util.spec_from_file_location("battery_record_for_reads", BATTERY_RECORD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_provenance(prov: Optional[str]) -> Optional[Tuple[str, str, str, str]]:
    """``(record path, target, cell key, arm)`` from a run's ``extra.provenance_legacy`` (its first ``; `` part), else None."""
    if not prov or "#" not in prov:
        return None
    path, frag = prov.split("; ")[0].split("#", 1)
    m = _FRAG.match(frag.strip())
    if not m or not path.endswith("battery_curves.json"):
        return None
    return SC.expand(path.strip()), m.group("t"), m.group("cell"), m.group("arm")


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def census_problems(block: Mapping[str, Any], g: Mapping[str, Any]) -> List[str]:
    """The reads' census (``graded_from_reads``) against the record block's numbers; empty when they agree."""
    want = {"full_refusal_n": block.get("full_refusal_n"), "partial_refusal_n": block.get("partial_refusal_n"),
            "n_judged": block.get("n_judged"), "n_abstained": block.get("n_abstained"),
            "graded_score_sum": (block.get("graded") or {}).get("graded_score_sum"),
            "at_cap_n_safe": (block.get("graded") or {}).get("at_cap_n_safe"), "read_at_rung": block.get("read_at_rung")}
    have = {"full_refusal_n": g["label_counts_safe"]["full_refusal"], "partial_refusal_n": g["label_counts_safe"]["partial_refusal"],
            "n_judged": g["n_safe"], "n_abstained": g["n_abstained_safe"], "graded_score_sum": g["graded_score_sum"],
            "at_cap_n_safe": g["at_cap_n_safe"], "read_at_rung": g["read_at_rung"]}
    out = []
    for k, w in want.items():
        h = have[k]
        same = (abs(float(w) - float(h)) < 1e-9) if k == "graded_score_sum" and w is not None else (w == h)
        if not same:
            out.append(f"{k}: the reads give {h!r}, the record {w!r}")
    return out


class Reader:
    """Cached, read-only access to the stores (spools, judge rows, sha256s)."""

    def __init__(self, BR):
        self.BR = BR
        self.spools: Dict[str, Dict[str, str]] = {}
        self.rows: Dict[str, Dict[str, dict]] = {}
        self.shas: Dict[str, str] = {}

    def spool(self, store: str) -> Dict[str, str]:
        p = os.path.join(store, "private", SPOOL)
        if p not in self.spools:
            self.spools[p] = SC.read_spool(p)
        return self.spools[p]

    def safe_rows(self, judge_items: str) -> Dict[str, dict]:
        if judge_items not in self.rows:
            rows = self.BR.safe_rows_by_item({"items": judge_items})
            if rows is None:
                raise ValueError(f"{judge_items}: unreadable judge rows")
            self.rows[judge_items] = rows
        return self.rows[judge_items]

    def sha(self, path: str) -> str:
        if path not in self.shas:
            self.shas[path] = SC.sha256_file(path)
        return self.shas[path]


def run_rows(db: str, run_ids: List[str]) -> Tuple[Dict[str, dict], Dict[str, List[dict]], str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sonnet = SC.sonnet_judges(con)[EVAL]
    runs: Dict[str, dict] = {}
    judg: Dict[str, List[dict]] = collections.defaultdict(list)
    for i in range(0, len(run_ids), 500):
        part = run_ids[i:i + 500]
        q = ",".join("?" * len(part))
        for r in con.execute(f"SELECT run_id, campaign, eval_id, lane_source_legacy, store_uri, trial_index, extra_json, model_id, object_id "
                             f"FROM run WHERE run_id IN ({q})", part):
            runs[r["run_id"]] = dict(r)
        for j in con.execute(f"SELECT judgment_id, run_id, judge_id, status, store_uri, n_items_judged, finished_utc FROM judgment "
                             f"WHERE run_id IN ({q})", part):
            judg[j["run_id"]].append(dict(j))
    con.close()
    return runs, judg, sonnet


def build(run_ids: List[str], runs: Mapping[str, dict], judg: Mapping[str, List[dict]], sonnet_judge: str, prompts: Mapping[str, str],
          BR, out: str, shared_rung: bool = True) -> Dict[str, Any]:
    """Write the reads scope (module docstring) into ``out``; returns the report."""
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o700)
    rd = Reader(BR)
    records: Dict[str, dict] = {}
    rubric = C.RUBRICS[RUBRIC]
    rsha, psha = rubric.sha, C.judging.protocol_sha(rubric)
    problems: List[dict] = []
    texts: Dict[str, dict] = {}
    totals: collections.Counter = collections.Counter()
    ufh = gzip.open(os.path.join(out, "units.tsv.gz"), "wt")
    ufh.write(UNIT_HEADER)
    rfh = open(os.path.join(out, "runs.jsonl"), "w")
    for rid in run_ids:
        run = runs.get(rid)
        if run is None or run["eval_id"] != EVAL:
            problems.append({"run_id": rid, "problem": "not an xstest run of the registry"})
            continue
        extra = json.loads(run.get("extra_json") or "{}")
        loc = parse_provenance(extra.get("provenance_legacy"))
        if loc is None:
            problems.append({"run_id": rid, "problem": f"extra.provenance_legacy names no battery record block: {extra.get('provenance_legacy')!r}"})
            continue
        path, target, cell_key, arm = loc
        if path not in records:
            with open(path) as fh:
                records[path] = json.load(fh)
        cells = records[path]["targets"][target]["cells"]
        cell = cells.get(cell_key)
        orb = (cell or {}).get("over_refusal") or {}
        block = orb.get(arm)
        if not block or (cell or {}).get("over_refusal_status") != "graded":
            problems.append({"run_id": rid, "problem": f"{path}#{target}/{cell_key}/{arm}: no graded over-refusal block"})
            continue
        sj = [j for j in judg.get(rid, []) if j["judge_id"] == sonnet_judge and j["status"] == "complete"]
        top = os.path.realpath(block["store"])
        bad = []
        if int(cell.get("trial", 0)) != int(run["trial_index"]):
            bad.append(f"the record cell's trial {cell.get('trial')} is not the run's {run['trial_index']}")
        if os.path.realpath(SC.expand(run["store_uri"] or "")) != top:
            bad.append(f"the record block's top store {block['store']} is not the run's store_uri {run['store_uri']}")
        if len(sj) != 1 or os.path.realpath(SC.expand((sj[0]["store_uri"] or "").split("#", 1)[0])) != top:
            bad.append(f"{len(sj)} complete Sonnet judgment(s); the one judged store must be the record block's top store")
        if bad:
            problems += [{"run_id": rid, "problem": b} for b in bad]
            continue
        sources: List[dict] = []
        src_idx: Dict[Tuple[str, int], int] = {}

        def source(store: str, rung: int, judge_items: str) -> int:
            key = (os.path.realpath(store), int(rung))
            if key not in src_idx:
                src_idx[key] = len(sources)
                sources.append({"trial": int(run["trial_index"]), "rung": int(rung), "cell": store, "judge_items": judge_items,
                                "judge_items_sha256": rd.sha(judge_items), "spool_sha256": rd.sha(os.path.join(store, "private", SPOOL))})
            return src_idx[key]

        # ---- the primary per-item read
        chain = []
        for R in sorted(int(k) for k in block["stores_by_rung"]):
            e = block["stores_by_rung"][str(R)]
            blk, why = BR.xstest_block(Path(e["store"]), R)
            if blk is None:
                bad.append(f"rung {R} store {e['store']}: {why}")
                continue
            chain.append({"rung": R, "store": e["store"], "block": blk, "rows": rd.safe_rows(e["judge_items"]), "judge_items": e["judge_items"]})
        if bad:
            problems += [{"run_id": rid, "problem": b} for b in bad]
            continue
        reads, item_problems = BR.per_item_reads(chain)
        g = BR.graded_from_reads(reads)
        cp = census_problems(block, g)
        if cp or item_problems:
            problems += [{"run_id": rid, "problem": "census: " + p} for p in cp + item_problems]
            continue
        by_rung = {c["rung"]: c for c in chain}
        top_rung = chain[-1]["rung"]
        planned: List[Tuple[str, int, str, dict, dict]] = []      # (read, rung, item, chain link, extra note)
        for iid in sorted(reads):
            r = reads[iid]
            planned.append(("per_item", r["rung"], iid, by_rung[r["rung"]], {"at_cap": r["at_cap"], "regenerated": r["regenerated"]}))
        # ---- the shared-rung reads beside it (one whole store per arm at the rung both arms of a cell hold)
        shared_info: List[dict] = []
        if shared_rung:
            if arm == "none":
                pair_cells = sorted(k for k, c in cells.items() if ((c.get("over_refusal") or {}).get("none") or {}).get("store")
                                    and os.path.realpath(c["over_refusal"]["none"]["store"]) == top)
                for k in pair_cells:
                    if cells[k]["over_refusal"]["none"].get("stores_by_rung") != block.get("stores_by_rung"):
                        problems.append({"run_id": rid, "problem": f"the none block under {k} names other rung stores than under {cell_key}"})
            else:
                pair_cells = [cell_key]
            seen_rungs: Dict[int, str] = {}
            for k in pair_cells:
                sh = cells[k]["over_refusal"].get("shared_rung_pair") or {}
                sb = sh.get(arm) if sh.get("rung") is not None else None
                if not sb or not sb.get("store"):
                    shared_info.append({"cell": k, "rung": None, "status": sh.get("status")})
                    continue
                R = int(sh["rung"])
                shared_info.append({"cell": k, "rung": R, "store": sb["store"]})
                if R in seen_rungs:
                    if os.path.realpath(seen_rungs[R]) != os.path.realpath(sb["store"]):
                        problems.append({"run_id": rid, "problem": f"two shared-rung stores at rung {R}: {seen_rungs[R]} and {sb['store']}"})
                    continue
                seen_rungs[R] = sb["store"]
                rows = rd.safe_rows(sb["judge_items"])
                n_j = sum(1 for x in rows.values() if x.get("status") == "judged")
                n_f = sum(1 for x in rows.values() if x.get("status") == "judged" and x.get("label") == "full_refusal")
                if (n_j, n_f) != (sb.get("n_judged"), sb.get("full_refusal_n")):
                    problems.append({"run_id": rid, "problem": f"shared rung {R}: the store's judged / full-refusal counts {(n_j, n_f)} are not "
                                                               f"the record's {(sb.get('n_judged'), sb.get('full_refusal_n'))}"})
                    continue
                link = {"rung": R, "store": sb["store"], "rows": rows, "judge_items": sb["judge_items"]}
                for iid in sorted(rows):              # every safe row, as the per-item read keeps Sonnet's abstentions: a new judge may label them
                    planned.append(("shared_rung", R, iid, link, {}))
        cnt: collections.Counter = collections.Counter()
        for read, R, iid, link, ex in planned:
            row = link["rows"].get(iid)
            if row is None:
                problems.append({"run_id": rid, "problem": f"{read} {iid} rung {R}: no judge row"})
                continue
            if row.get("task") != TASK or row.get("rubric_sha") != rsha or row.get("protocol_sha") != psha or row.get("judge_model") != C.ORIGINAL_MODEL:
                problems.append({"run_id": rid, "problem": f"{read} {iid} rung {R}: the judge row is not the Sonnet C/P/I judge's"})
                continue
            prompt = prompts.get(iid)
            resp = rd.spool(link["store"]).get(iid)
            if prompt is None or resp is None:
                problems.append({"run_id": rid, "problem": f"{read} {iid} rung {R}: no pinned prompt or no generation in {link['store']}"})
                continue
            key = C.request_key(RUBRIC, prompt, resp)
            if key != row.get("request_key"):
                cnt["key_mismatch"] += 1
                problems.append({"run_id": rid, "problem": f"{read} {iid} rung {R}: recomputed key {key[:12]} != stored {str(row.get('request_key'))[:12]}"})
                continue
            if key not in texts:
                texts[key] = {"k": key, "rubric": RUBRIC, "item_id": iid, "prompt": prompt, "response": resp}
            note = {"read": read, "rung": R, "top_rung": top_rung, "cell": source(link["store"], R, link["judge_items"]), **ex,
                    "subset": "safe", "text_sha256": sha256_text(resp)}
            if row.get("abstention_reason"):
                note["abstention_reason"] = row.get("abstention_reason")
            ufh.write("\t".join([rid, iid, str(run["trial_index"]), "0", "-1", key, RUBRIC, EVAL, str(row.get("label") or ""),
                                 str(row.get("status") or ""), json.dumps(note, separators=(",", ":"))]) + "\n")
            cnt["units"] += 1
            cnt[f"{read}_units"] += 1
            cnt[f"{read}_at_{R}"] += 1
            if read == "per_item" and R != top_rung:
                cnt["per_item_below_top"] += 1
        rec = {"run_id": rid, "eval_id": EVAL, "campaign": run["campaign"], "lane": (run["lane_source_legacy"] or "-") + " (record reads)",
               "rubric": RUBRIC, "model_id": run["model_id"], "object_id": run["object_id"], "trial": int(run["trial_index"]),
               "arm_kind": "none" if arm == "none" else "steered", "sonnet_judge_id": sonnet_judge, "sonnet_judgment_id": sj[0]["judgment_id"],
               "record": path, "record_sha256": rd.sha(path), "block": f"targets.{target}.cells.{cell_key}.over_refusal.{arm}",
               "top_store": block["store"], "top_rung": top_rung, "rungs_read": [c["rung"] for c in chain], "census_matches_record": True,
               "census": {k: g[k] for k in ("n_safe", "graded_score_sum", "at_cap_n_safe", "n_abstained_safe", "label_counts_safe", "read_at_rung")},
               "shared_rung": shared_info, "sources": sources, "counts": dict(cnt), "not_for_filing": True}
        rfh.write(json.dumps(rec, sort_keys=True) + "\n")
        totals.update(cnt)
        totals["runs"] += 1
        totals["runs_" + rec["arm_kind"]] += 1
    ufh.close()
    rfh.close()
    with gzip.open(os.path.join(out, "texts.jsonl.gz"), "wt") as fh:
        for k in sorted(texts):
            fh.write(json.dumps(texts[k], ensure_ascii=True) + "\n")
    report = {"not_for_filing": NOT_FOR_FILING, "runs_requested": len(run_ids), "totals": dict(totals), "distinct_requests": len(texts),
              "shared_rung": bool(shared_rung), "rubric_sha16": rsha, "protocol_sha16": psha, "sonnet_judge": sonnet_judge,
              "battery_record": {"path": os.path.relpath(BATTERY_RECORD, os.path.dirname(_SCRIPTS)), "sha256": SC.sha256_file(BATTERY_RECORD)},
              "records": {p: rd.sha(p) for p in sorted(records)}, "n_problems": len(problems), "problems": problems}
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def union(scopes: List[str], out: str) -> Dict[str, Any]:
    """The manifest build's scope: every scope's texts (one row per key; a key's text must agree), units and runs.jsonl rows, marked
    not for filing (it carries the record reads' units)."""
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o700)
    texts: Dict[str, str] = {}
    n_units = 0
    with gzip.open(os.path.join(out, "units.tsv.gz"), "wt") as dst:
        dst.write(UNIT_HEADER)
        for s in scopes:
            with gzip.open(os.path.join(s, "units.tsv.gz"), "rt") as fh:
                if next(fh) != UNIT_HEADER:
                    raise SystemExit(f"{s}/units.tsv.gz has another header")
                for line in fh:
                    dst.write(line)
                    n_units += 1
    for s in scopes:
        with gzip.open(os.path.join(s, "texts.jsonl.gz"), "rt") as fh:
            for line in fh:
                k = json.loads(line)["k"]
                if k in texts and texts[k] != line:
                    raise SystemExit(f"key {k[:16]} has two different texts across the scopes")
                texts[k] = line
    with gzip.open(os.path.join(out, "texts.jsonl.gz"), "wt") as fh:
        for k in sorted(texts):
            fh.write(texts[k])
    with open(os.path.join(out, "runs.jsonl"), "w") as dst:
        for s in scopes:
            with open(os.path.join(s, "runs.jsonl")) as fh:
                for line in fh:
                    dst.write(line)
    report = {"not_for_filing": "the manifest build's scope (a union that carries the record reads' units): " + NOT_FOR_FILING,
              "scopes": [{"path": os.path.abspath(s), "units_sha256": SC.sha256_file(os.path.join(s, "units.tsv.gz")),
                          "texts_sha256": SC.sha256_file(os.path.join(s, "texts.jsonl.gz"))} for s in scopes],
              "units": n_units, "distinct_requests": len(texts)}
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build", help="the record reads of --runs")
    pb.add_argument("--registry", default=os.path.join(HOME, "registry", "registry.sqlite"))
    pb.add_argument("--runs", required=True, help="run ids, one per line")
    pb.add_argument("--out", required=True)
    pb.add_argument("--no-shared-rung", action="store_true", help="leave out the shared_rung_pair whole-store reads")
    pu = sub.add_parser("union", help="the manifest build's scope from several scopes")
    pu.add_argument("--scope", action="append", required=True)
    pu.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "union":
        rep = union(a.scope, a.out)
        print(json.dumps(rep, indent=1))
        return 0
    from sisteer.capeval import corpora
    from sisteer.capeval.tiers import judged
    prompts = {r["item_id"]: r["prompt"] for r in judged.xstest_rows(corpora.load_pins())}
    run_ids = SC.read_run_ids(a.runs)
    runs, judg, sonnet = run_rows(a.registry, run_ids)
    rep = build(run_ids, runs, judg, sonnet, prompts, battery_record(), a.out, shared_rung=not a.no_shared_rung)
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=1))
    for p in rep["problems"][:30]:
        print("  PROBLEM", p["run_id"], p["problem"])
    return 0 if not rep["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
