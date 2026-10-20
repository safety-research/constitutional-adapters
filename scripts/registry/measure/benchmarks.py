#!/usr/bin/env python3
"""benchmarks.py -- the benchmark recorder for S1's levers.

    python scripts/registry/measure/benchmarks.py [--spec DIR] [--root ~/registry] [--state DIR]
                                                   [--since ISO] [--until ISO] [--r14 FILE] [--write] [--json]

Each LEVER has a target from the plan and is measured from the evidence that exists, never estimated: the source of every number is
named beside it, and a lever with no evidence is ``unmeasured`` (never a guess).

| lever | target | evidence |
|---|---|---|
| ``reader_state_export`` | <= 20 s | the chain logs' ``-- reader-state export`` stage walls (windows.py) |
| ``shard_read`` | <= 90 s per shard | ``pass`` of every successful shard summary under ``<root>/state/cluster_reads/<window>.summaries/`` (per adapter the max) |
| ``array_span`` | <= 120 s | ``-- cluster reads submit`` -> ``-- cluster reads revalidate`` per window |
| ``admission`` | (reported) | ``admission_pre`` + ``admission_post`` per shard summary -- the per-shard walk C5 replaced by the attestation |
| ``attestation_listing`` | <= 60 s per generation | the per-operation attestation record the pull stage writes -- ``stages.pull.attestation`` of every operation's plan under ``<root>/state/cluster_reads/<op>.json`` (a window's and a tick's alike) and the same record on the receipts under ``<state>/reads/receipts/<op>.json`` -- per generation ``listing_s`` (the listing's wall; ``elapsed_s`` = listing + seal + sample beside it); one entry per attestation (a plan and its receipt carry one record), inside the span by the attestation's ``utc`` |
| ``attestation`` | (reported) | the same records' operation-level ``elapsed_s`` (every pinned generation, listed in parallel threads) |
| ``fold`` | (reported; the one ingest) | the ``--- ingest`` stage walls |
| ``drain`` | <= 60 s typical | ``writers quiet`` - start per window |
| ``window_total`` | mean <= 5 / median <= 4 / max <= 12 min (S1) | the completed windows of the span |

R14'S DRILL RECORD IS BASELINE EVIDENCE, NEVER A SAMPLE (``rehearsals/R14/r14-result.json``: ``listing_s`` + ``dir_mode_check_s`` per
generation, a pre-cutover drill on the cluster): it is reported under ``attestation_listing.detail.baseline_r14`` with its own verdict
against the same target, labelled as the baseline, and is never mixed into ``measured_s`` -- so a post-cutover ``--since <cutover>`` run
whose attestations all meet the target reads ``within``, and one with no attestation yet reads ``unmeasured`` (the baseline shown beside).

``--write`` merges the record into ``$SPEC/measure/benchmarks.json`` (``recorded_utc``, the levers, the sources); earlier levers the
new record does not measure are kept. Stdlib only; read-only everywhere else.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC = "$WORK/registry-slurm-only"   # under the site's working-data root; expanded after parsing (scripts/site_paths.py)
DEFAULT_ROOT = os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")
DEFAULT_STATE = os.path.expanduser(os.environ.get("REGISTRY_LANDING_STATE") or "~/.local/state/registry-landing")
TARGETS = {"reader_state_export": {"max_s": 20.0}, "shard_read": {"max_s": 90.0}, "array_span": {"max_s": 120.0}, "attestation_listing": {"max_s": 60.0},
           "attestation": {}, "drain": {"typical_s": 60.0}, "window_total": {"mean_min": 5.0, "median_min": 4.0, "max_min": 12.0}, "admission": {}, "fold": {}}
BASELINE_LABEL = "baseline: the pre-cutover R14 drill on the cluster (listing_s + dir_mode_check_s per generation) -- never a sample of the lever"
#: stamped on every lever; bumped when a lever's evidence rule changes. ``merge_write`` keeps an earlier lever this record leaves
#: ``unmeasured`` ONLY when it was recorded under the same method -- a number computed by a rule since corrected (the attestation lever's
#: R14-forced `over` of the first record) is replaced, not preserved
RECORD_METHOD = "benchmarks/2"


def _windows_module():
    spec = importlib.util.spec_from_file_location("measure_windows_for_benchmarks", os.path.join(HERE, "windows.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "median": None, "mean": None, "max": None, "min": None}
    return {"n": len(vals), "median": round(statistics.median(vals), 1), "mean": round(statistics.mean(vals), 1), "max": round(max(vals), 1),
            "min": round(min(vals), 1)}


def _verdict(stats: Dict[str, Any], key: str, bound: Optional[float]) -> str:
    if not stats.get("n"):
        return "unmeasured"
    if bound is None:
        return "reported"
    v = stats.get(key)
    return "within" if v is not None and float(v) <= bound else "over"


def stage_walls(windows: List[Dict[str, Any]], label_prefix: str) -> List[float]:
    out: List[float] = []
    for w in windows:
        for st in w.get("stages") or []:
            if str(st.get("label", "")).lower().startswith(label_prefix) and st.get("seconds") is not None:
                out.append(float(st["seconds"]))
    return out


def span_walls(windows: List[Dict[str, Any]], start_label: str, end_label: str) -> List[float]:
    """Per window the seconds from the ``start_label`` stamp to the ``end_label`` stamp (both inside the window)."""
    out: List[float] = []
    for w in windows:
        ats: Dict[str, str] = {}
        for st in w.get("stages") or []:
            ats.setdefault(str(st.get("label", "")).lower(), str(st.get("at") or ""))
        a, b = ats.get(start_label), ats.get(end_label)
        if a and b:
            try:
                from datetime import datetime
                ta = datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ")
                tb = datetime.strptime(b, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
            d = (tb - ta).total_seconds()
            if d >= 0:
                out.append(d)
    return out


def shard_timings(root: str, since: Optional[str] = None, until: Optional[str] = None) -> Dict[str, Any]:
    """Per adapter the successful shards' ``pass`` and admission seconds from every ``<window>.summaries/<adapter>/<i>-of-<K>.json``."""
    d = os.path.join(root, "state", "cluster_reads")
    per_adapter: Dict[str, Dict[str, List[float]]] = {}
    windows_seen: List[str] = []
    if not os.path.isdir(d):
        return {"adapters": {}, "windows": [], "dir": d}
    for name in sorted(os.listdir(d)):
        if not name.endswith(".summaries"):
            continue
        window = name[: -len(".summaries")]
        stamp = window[:15]
        if since and stamp < since.replace("-", "").replace(":", "")[:15]:
            continue
        if until and stamp >= until.replace("-", "").replace(":", "")[:15]:
            continue
        base = os.path.join(d, name)
        for adapter in sorted(os.listdir(base)):
            ad = os.path.join(base, adapter)
            if not os.path.isdir(ad):
                continue
            for fn in sorted(os.listdir(ad)):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(ad, fn), "r", encoding="utf-8") as fh:
                        doc = json.load(fh)
                except (OSError, ValueError):
                    continue
                if not isinstance(doc, dict):
                    continue
                t = doc.get("timings_s") or {}
                slot = per_adapter.setdefault(adapter, {"pass": [], "admission": [], "wall": [], "refused": []})
                if doc.get("ok") and t.get("pass") is not None:
                    slot["pass"].append(float(t["pass"]))
                    slot["admission"].append(float(t.get("admission_pre") or 0) + float(t.get("admission_post") or 0))
                    if doc.get("wall_s") is not None:
                        slot["wall"].append(float(doc["wall_s"]))
                else:
                    slot["refused"].append(float(doc.get("wall_s") or 0))
                if window not in windows_seen:
                    windows_seen.append(window)
    return {"adapters": {a: {"pass_s": _stats(v["pass"]), "admission_s": _stats(v["admission"]), "wall_s": _stats(v["wall"]),
                             "refused_shards": len(v["refused"])} for a, v in sorted(per_adapter.items())},
            "windows": windows_seen, "dir": d}


def r14_baseline(r14_path: Optional[str], bound: float) -> Optional[Dict[str, Any]]:
    """R14's drill record as LABELLED BASELINE evidence: per generation ``listing_s`` + ``dir_mode_check_s`` (``total_s``), its own stats and
    verdict against ``bound``; None when there is no record. Never a sample of the lever."""
    if not (r14_path and os.path.isfile(r14_path)):
        return None
    try:
        with open(r14_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {"label": BASELINE_LABEL, "error": f"unreadable: {r14_path}"}
    if not isinstance(doc, dict):
        return {"label": BASELINE_LABEL, "error": f"not a record: {r14_path}"}
    listings = [{"path": l.get("path"), "entries": l.get("entries"), "listing_s": l.get("listing_s"), "dir_mode_check_s": l.get("dir_mode_check_s"),
                 "total_s": round(float(l.get("listing_s") or 0) + float(l.get("dir_mode_check_s") or 0), 3)}
                for l in (doc.get("listings") or []) if isinstance(l, dict)]
    stats = _stats([l["total_s"] for l in listings])
    return {"label": BASELINE_LABEL, "utc": doc.get("utc"), "job_id": doc.get("job_id"), "host": doc.get("host"), "listings": listings,
            "measured_s": stats, "verdict": _verdict(stats, "max", bound), "source": r14_path}


def _attestation_entry(att: Any, source: str, fallback_utc: Optional[str]) -> Optional[Dict[str, Any]]:
    """One attestation record (``cluster_reads.attestation_summary``'s shape: ``utc``, ``job_id``, ``elapsed_s``, ``generations{slug:
    {listing_s, elapsed_s, listing_entries, ok}}``) as the recorder's entry; None when the record is absent or carries no timing."""
    if not isinstance(att, dict) or att.get("present") is False:
        return None
    gens: Dict[str, Dict[str, Any]] = {}
    for slug, g in sorted((att.get("generations") or {}).items()):
        if not isinstance(g, dict):
            continue
        gens[str(slug)] = {"listing_s": g.get("listing_s"), "elapsed_s": g.get("elapsed_s"), "listing_entries": g.get("listing_entries"), "ok": g.get("ok")}
    if att.get("elapsed_s") is None and not any(g.get("listing_s") is not None or g.get("elapsed_s") is not None for g in gens.values()):
        return None
    return {"source": source, "utc": att.get("utc") or fallback_utc, "job_id": att.get("job_id"), "host": att.get("host"), "ok": att.get("ok"),
            "window": att.get("window"), "elapsed_s": att.get("elapsed_s"), "generations": gens}


def attestation_evidence(root: str, state: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None) -> Dict[str, Any]:
    """The POST-CUTOVER attestation evidence: ``stages.pull.attestation`` of every operation's plan file under ``<root>/state/cluster_reads/
    <op>.json`` (the window's and the tick's alike -- both write their plan there, ``cluster_reads.plan_path``) and the same record each
    receipt carries (``<state>/reads/receipts/<op>.json``, ``attestation``), one entry per attestation (deduplicated by job id + stamp: a
    plan and its receipt carry one record), inside the span by the attestation's ``utc`` (fallback: the pull stage's / the receipt's stamp;
    ``since`` inclusive, ``until`` exclusive, ISO-prefix string order as windows.py compares)."""
    entries: Dict[Any, Dict[str, Any]] = {}
    scanned = {"plans": 0, "receipts": 0, "plans_dir": os.path.join(root, "state", "cluster_reads"), "receipts_dir": (os.path.join(state, "reads", "receipts") if state else None)}
    excluded = 0

    def take(entry: Optional[Dict[str, Any]]) -> None:
        nonlocal excluded
        if entry is None:
            return
        utc = str(entry.get("utc") or "")
        if (since and utc < since) or (until and utc >= until):
            excluded += 1
            return
        key = (entry.get("job_id"), entry.get("utc")) if (entry.get("job_id") or entry.get("utc")) else entry["source"]
        entries.setdefault(key, entry)

    d = scanned["plans_dir"]
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, name), "r", encoding="utf-8") as fh:
                    plan = json.load(fh)
            except (OSError, ValueError):
                continue
            if not isinstance(plan, dict):
                continue
            scanned["plans"] += 1
            pull = (plan.get("stages") or {}).get("pull") or {}
            take(_attestation_entry(pull.get("attestation"), f"plan:{name[:-5]}", pull.get("utc")))
    rd = scanned["receipts_dir"]
    if rd and os.path.isdir(rd):
        for name in sorted(os.listdir(rd)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(rd, name), "r", encoding="utf-8") as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict) or rec.get("schema") != "reads_receipt/1":
                continue
            scanned["receipts"] += 1
            take(_attestation_entry(rec.get("attestation"), f"receipt:{name[:-5]}", rec.get("utc")))
    out = sorted(entries.values(), key=lambda e: (str(e.get("utc") or ""), str(e.get("source"))))
    return {"attestations": out, "scanned": scanned, "excluded_by_span": excluded,
            "listing_s": [g["listing_s"] for e in out for g in e["generations"].values() if g.get("listing_s") is not None],
            "generation_s": [g["elapsed_s"] for e in out for g in e["generations"].values() if g.get("elapsed_s") is not None],
            "attestation_s": [e["elapsed_s"] for e in out if e.get("elapsed_s") is not None]}


def record(spec: str, root: str, state: str, since: Optional[str], until: Optional[str], r14: Optional[str], logs: Optional[List[str]] = None) -> Dict[str, Any]:
    W = _windows_module()
    windows_all = W.read_logs(logs or W.DEFAULT_LOGS)
    windows, _aborted = W.select_span(windows_all, since, until)
    done = [w for w in windows if w.get("completed")]
    totals = [float(w["total_s"]) / 60.0 for w in done if w.get("total_s") is not None]
    export = _stats(stage_walls(windows, "reader-state export"))
    span = _stats(span_walls(windows, "cluster reads submit", "cluster reads revalidate"))
    fold = _stats([float(st["seconds"]) for w in windows for st in (w.get("stages") or []) if st.get("class") == "ingest" and st.get("seconds") is not None])
    drain = _stats([float(w["quiet_wait_s"]) for w in windows if w.get("quiet_wait_s") is not None])
    shards = shard_timings(root, since, until)
    passes = [v["pass_s"]["max"] for v in shards["adapters"].values() if v["pass_s"]["n"]]
    adm = [v["admission_s"]["max"] for v in shards["adapters"].values() if v["admission_s"]["n"]]
    att = attestation_evidence(root, state, since, until)
    listing = _stats([float(x) for x in att["listing_s"]])
    attestation = _stats([float(x) for x in att["attestation_s"]])
    baseline = r14_baseline(r14, TARGETS["attestation_listing"]["max_s"])
    wt = {"n": len(totals), "median_min": round(statistics.median(totals), 1) if totals else None, "mean_min": round(statistics.mean(totals), 1) if totals else None,
          "max_min": round(max(totals), 1) if totals else None}
    wt_verdict = ("unmeasured" if not totals else ("within" if (wt["mean_min"] <= TARGETS["window_total"]["mean_min"] and wt["median_min"] <= TARGETS["window_total"]["median_min"]
                                                              and wt["max_min"] <= TARGETS["window_total"]["max_min"]) else "over"))
    levers = {
        "reader_state_export": {"target": TARGETS["reader_state_export"], "measured_s": export, "verdict": _verdict(export, "max", TARGETS["reader_state_export"]["max_s"]),
                                "source": "chain logs: `-- reader-state export` stage wall (windows.py)"},
        "shard_read": {"target": TARGETS["shard_read"], "measured_s": {"per_adapter_max": {a: v["pass_s"]["max"] for a, v in shards["adapters"].items()},
                                                                       "overall": _stats(passes)},
                       "verdict": _verdict(_stats(passes), "max", TARGETS["shard_read"]["max_s"]),
                       "source": f"shard summaries under {shards['dir']} (ok shards' timings_s.pass; windows {shards['windows']})"},
        "array_span": {"target": TARGETS["array_span"], "measured_s": span, "verdict": _verdict(span, "max", TARGETS["array_span"]["max_s"]),
                       "source": "chain logs: `-- cluster reads submit` -> `-- cluster reads revalidate`"},
        "admission": {"target": TARGETS["admission"], "measured_s": {"per_adapter_max": {a: v["admission_s"]["max"] for a, v in shards["adapters"].items()},
                                                                     "overall": _stats(adm)},
                      "verdict": _verdict(_stats(adm), "max", None), "source": "shard summaries: admission_pre + admission_post (the per-shard walk C5 replaced)"},
        "attestation_listing": {"target": TARGETS["attestation_listing"], "measured_s": listing,
                                "verdict": _verdict(listing, "max", TARGETS["attestation_listing"]["max_s"]),
                                "detail": {"per_generation_s": _stats([float(x) for x in att["generation_s"]]), "attestations": att["attestations"],
                                           "scanned": att["scanned"], "excluded_by_span": att["excluded_by_span"], "baseline_r14": baseline},
                                "source": (f"stages.pull.attestation of every operation's plan under {att['scanned']['plans_dir']} and the receipts under "
                                           f"{att['scanned']['receipts_dir']} (per generation listing_s; {len(att['attestations'])} attestation(s) in the span); "
                                           "R14's drill record is the labelled baseline under detail.baseline_r14, never a sample")},
        "attestation": {"target": TARGETS["attestation"], "measured_s": attestation, "verdict": _verdict(attestation, "max", None),
                        "source": "the same attestation records' operation-level elapsed_s (listing + seal + sample over every pinned generation)"},
        "fold": {"target": TARGETS["fold"], "measured_s": fold, "verdict": _verdict(fold, "max", None), "source": "chain logs: the `--- ingest` stage wall"},
        "drain": {"target": TARGETS["drain"], "measured_s": drain, "verdict": _verdict(drain, "median", TARGETS["drain"]["typical_s"]),
                  "source": "chain logs: start -> `writers quiet`"},
        "window_total": {"target": TARGETS["window_total"], "measured_min": wt, "verdict": wt_verdict,
                         "end_definitions": sorted({str(w.get("end_definition")) for w in done}), "source": "chain logs: completed windows of the span"},
    }
    for v in levers.values():
        v["method"] = RECORD_METHOD
    return {"schema": "slurm_only_benchmarks/1", "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "since": since, "until": until,
            "windows_in_span": len(windows), "completed_in_span": len(done), "root": root, "state": state, "levers": levers}


def merge_write(path: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``rec`` into the benchmarks file: levers measured now replace theirs; an earlier lever this record leaves ``unmeasured`` is kept
    when it was recorded under the same ``method`` (:data:`RECORD_METHOD`), and replaced when its rule has since changed."""
    old: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                old = json.load(fh) or {}
        except (OSError, ValueError):
            old = {}
    levers = dict(old.get("levers") or {})
    for k, v in rec["levers"].items():
        if v.get("verdict") == "unmeasured" and k in levers and levers[k].get("verdict") != "unmeasured" and levers[k].get("method") == v.get("method"):
            continue
        levers[k] = v
    out = dict(rec, levers=levers, history=list(old.get("history") or []) + [{"recorded_utc": rec["recorded_utc"], "since": rec["since"], "until": rec["until"]}])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return out


def render(rec: Dict[str, Any]) -> str:
    lines = [f'benchmarks recorded {rec["recorded_utc"]} (span {rec["since"] or "-"} -> {rec["until"] or "-"}: {rec["completed_in_span"]} completed of '
             f'{rec["windows_in_span"]} windows)', "| lever | target | measured | verdict | source |", "|---|---|---|---|---|"]
    for k, v in rec["levers"].items():
        m = v.get("measured_s") if "measured_s" in v else v.get("measured_min")
        if isinstance(m, dict) and "overall" in m:
            txt = f'max per adapter {m["per_adapter_max"]}; overall {m["overall"]}'
        else:
            txt = json.dumps(m, default=str)
        lines.append(f'| {k} | {json.dumps(v["target"])} | {txt} | {v["verdict"]} | {v["source"]} |')
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default=DEFAULT_SPEC, help=f"the spec directory (default {DEFAULT_SPEC}); --write records measure/benchmarks.json there")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--logs", nargs="*", help="chain logs (default windows.py's)")
    ap.add_argument("--since", help="windows starting, and attestations stamped, at or after this UTC stamp")
    ap.add_argument("--until", help="windows starting, and attestations stamped, before this UTC stamp")
    ap.add_argument("--r14", help="R14's drill record (default <spec>/rehearsals/R14/r14-result.json): the labelled baseline, never a sample")
    ap.add_argument("--write", action="store_true", help="merge the record into <spec>/measure/benchmarks.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if "$" in a.spec:   # the default: expanded through the site config (scripts/site_paths.py); an unset $WORK refuses by name
        sys.path.append(os.path.join(HERE, "..", ".."))
        from site_paths import site_path
        a.spec = str(site_path(a.spec, "--spec"))
    r14 = a.r14 or os.path.join(a.spec, "rehearsals", "R14", "r14-result.json")
    rec = record(a.spec, a.root, a.state, a.since, a.until, r14, a.logs)
    if a.write:
        rec = merge_write(os.path.join(a.spec, "measure", "benchmarks.json"), rec)
    if a.json:
        print(json.dumps(rec, indent=1, default=str))
    else:
        sys.stdout.write(render(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
