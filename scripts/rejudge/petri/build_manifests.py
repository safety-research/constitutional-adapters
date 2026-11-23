#!/usr/bin/env python3
"""Write a Petri run directory (``track.json`` + one manifest and body per batch) from the enumerated judge inputs, and a dry-run report.

    PYTHONPATH=src:scripts python scripts/rejudge/petri/build_manifests.py --judge glm-5.3 --enum ENUM --run-dir RUN \
        [--reuse-ledger SAME_JUDGE_LEDGER ...] [--keys KEYS.txt] [--exclude-keys DONE.txt ...] [--max-spend-usd USD] \
        (--require-file PATH ... | --no-submit-gate) [--est-reasoning-tokens N]

``--judge`` names the judge profile (``judges.py``; default ``glm-5.3``): every request is the profile's candidate body
(``bakeoff.models.openrouter_body``), the batches pinned to the profile's provider (DeepInfra).
``--reuse-ledger DIR`` (repeatable) names a ledger of the SAME judge whose verdicts are reused (the judge comparison's run directory of
the model): it must have stopped moving, its ``track.json`` must carry the profile's request settings (``judges.ledger_identity``), and
EVERY request body it sent must be byte-identical to the body this build would send for that key, and every fileable outcome it holds
must have been served by the profile's provider -- otherwise nothing is written. Its keys with a fileable outcome (a verdict or an item
outcome) are then excluded like ``--exclude-keys`` (an infrastructure-terminal key is re-sent), and the dry run's cost model is
calibrated on its verdicts' usage (``--calibrate-from`` names another ledger).

``--keys`` limits the run to the listed judge-input keys (the smoke, the pilot, a probe); otherwise every input of ``ENUM`` is included
(the full paper-row run). ``--exclude-keys`` (repeatable) drops keys another run directory holds -- the full run excludes the smoke's
and the pilot's, whose verdicts the filing reads from their own run directories, so no input is sent (or paid for) twice and no input
has two verdicts to choose between (the filing refuses a key terminal in two ledgers).

Submit gates and the spend ceiling (``track.json``; the shared harvester enforces them, ``common/submitter.py``):
  --require-file PATH           nothing is submitted until PATH exists (repeatable). A run built without ``--keys`` (the full scope)
                                must name one, or pass ``--no-submit-gate``
  --max-spend-usd USD           the run's ceiling, USD incl. the credit fee: the harvester pauses before a submit whose estimate would
                                pass it (the estimate assumes ``--est-reasoning-tokens`` per request; default the profile's)

Nothing is submitted. ``dry_run.json`` / ``dry_run.md`` in the run directory give requests, batches, bytes, tokens and the cost (incl.
the 5.5% credit fee) under the cache and reasoning assumptions, the estimate the spend ceiling reads against the ceiling, plus
OpenRouter's 402 pre-check figure (input + every request's ``max_tokens`` at the output price), which the balance must cover when a
batch is submitted.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sqlite3
import statistics
import sys
from typing import Any, Dict, List, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import state as S  # noqa: E402
from rejudge.petri import judges as J  # noqa: E402
from rejudge.petri import track as T  # noqa: E402



def read_inputs(enum_dir: str, keys=None) -> List[Dict[str, Any]]:
    out = []
    with gzip.open(os.path.join(enum_dir, "inputs.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if "error" in r:
                continue
            if keys is not None and r["key"] not in keys:
                continue
            out.append(r)
    return out


def read_key_file(path: str) -> set:
    """Judge-input keys, one per line (the first field; ``#`` lines skipped)."""
    return {l.split()[0] for l in open(path) if l.strip() and not l.startswith("#")}


def dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in rows:
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        out.append(r)
    return out


def _ceiling_line(rep: Dict[str, Any], mans: List[Dict[str, Any]]) -> str:
    """Whether the manifests' estimates (what the spend ceiling reads) fit under the run's ceiling, and where the harvester would pause."""
    cap = float(rep.get("max_spend_usd") or 0)
    est = rep["ceiling_estimate_usd_with_fee"]
    if cap <= 0:
        return f"- spend ceiling: none (manifest estimate ${est:,.2f} incl. fee)"
    run, stop = 0.0, None
    for m in sorted(mans, key=lambda m: m["batch_key"]):
        run += float(((m.get("est") or {}).get("cost_usd_with_fee")) or 0)
        if run > cap and stop is None:
            stop = m["batch_key"]
    if stop is None:
        return f"- spend ceiling ${cap:,.2f} incl. fee; the manifests' estimate ${est:,.2f} fits (re-asks and a higher real cost can still reach it)"
    return (f"- spend ceiling ${cap:,.2f} incl. fee; the manifests' estimate ${est:,.2f} does NOT fit: the harvester would pause before "
            f"{stop} (cut --keys or raise --max-spend-usd)")


# --------------------------------------------------------------------------------------------
# the judge profile (judges.py): reused ledgers, the calibrated cost model, the dry run
# --------------------------------------------------------------------------------------------
def ledger_requests(ledger_dir: str) -> Dict[str, Dict[str, Any]]:
    """custom_id -> the request body the ledger's FIRST attempt sent (its non-retry manifests; a re-ask re-sends the same body with at
    most a raised cap)."""
    out: Dict[str, Dict[str, Any]] = {}
    for k in BF.list_manifests(ledger_dir):
        man = BF.read_manifest(ledger_dir, k)
        if int(man.get("attempt") or 0) or man.get("parent_batch_key"):
            continue
        for q in BF.body_requests(BF.read_body(ledger_dir, man)):
            out.setdefault(q["custom_id"], q["body"])
    return out


def check_reuse(ledger_dir: str, prof: Mapping[str, Any], rows_by_key: Mapping[str, Mapping[str, Any]], tools: Mapping[str, Any]) -> Dict[str, Any]:
    """Prove a reused ledger holds verdicts of THIS judge for THESE inputs (module docstring) and return the keys to exclude: every key
    with a FILEABLE terminal outcome (a verdict or an item outcome) served by one of the profile's ``served_by`` providers. A key whose
    terminal outcome is an infrastructure failure is NOT excluded, so the full run re-sends it (the filing then lets the new verdict
    settle the old failure). Raises SystemExit, naming the first problem, for: another judge's ledger, a SYNTHETIC one, one that has not
    stopped moving, a manifest whose keys the ledger never registered, a key that is not an enumerated input, a body that differs from
    this build's, or a fileable outcome served by another provider."""
    from rejudge.petri.file_records import FILEABLE, SYNTHETIC_MARK
    try:
        ident = J.ledger_identity(BF.read_track(ledger_dir), prof)
    except ValueError as e:
        raise SystemExit(f"--reuse-ledger {ledger_dir}: {e}")
    if os.path.exists(os.path.join(ledger_dir, SYNTHETIC_MARK)):
        raise SystemExit(f"--reuse-ledger {ledger_dir} holds SYNTHETIC verdicts; never reused")
    if not os.path.exists(os.path.join(ledger_dir, "state.sqlite")):
        raise SystemExit(f"--reuse-ledger {ledger_dir} holds no ledger (state.sqlite): never harvested, no verdicts to reuse")
    ready = S.readiness(ledger_dir)
    if not ready["ready"]:
        raise SystemExit(f"--reuse-ledger {ledger_dir} has not stopped moving ({ {k: ready[k] for k in ('paused', 'live_batches', 'non_terminal')} })")
    con = sqlite3.connect(f"file:{os.path.join(ledger_dir, 'state.sqlite')}?mode=ro", uri=True)
    led = {c: (o, int(t or 0), p) for c, o, t, p in con.execute("SELECT custom_id, outcome, terminal, provider FROM requests")}
    con.close()
    sent = ledger_requests(ledger_dir)
    unregistered = sorted(set(sent) - set(led))
    if unregistered:
        raise SystemExit(f"--reuse-ledger {ledger_dir}: {len(unregistered)} keys of its manifests are not in its ledger (first "
                         f"{unregistered[0][:16]}); a manifest it never registered or submitted")
    missing = sorted(set(sent) - set(rows_by_key))
    if missing:
        raise SystemExit(f"--reuse-ledger {ledger_dir}: {len(missing)} of its keys are not enumerated inputs (first {missing[0][:16]})")
    for cid, body in sorted(sent.items()):
        r = rows_by_key[cid]
        ours = J.arm_body(prof, r["prompt"], tools[r["tool_sha"]])
        if BF.serialize_request(cid, ours) != BF.serialize_request(cid, body):
            diff = sorted(k for k in set(ours) | set(body) if ours.get(k) != body.get(k))
            raise SystemExit(f"--reuse-ledger {ledger_dir}: the body it sent for {cid[:16]} differs from this build's in {diff}; its verdicts "
                             f"are not this judge's for this input")
    reuse = sorted(c for c in sent if led[c][1] and led[c][0] in FILEABLE)
    served = collections.Counter(led[c][2] for c in reuse)
    bad = J.served_problems(prof, served)
    if bad:
        raise SystemExit(f"--reuse-ledger {ledger_dir}: " + "; ".join(bad))
    resend = sorted(c for c in sent if c not in set(reuse))
    headers = {(m.get("model"), m.get("endpoint"), json.dumps(m.get("provider"), sort_keys=True))
               for m in (BF.read_manifest(ledger_dir, k) for k in BF.list_manifests(ledger_dir))}
    return {"ledger": os.path.abspath(ledger_dir), "keys": reuse, "n_keys": len(reuse), "bodies_byte_identical": len(sent),
            "outcomes": dict(collections.Counter(led[c][0] for c in reuse)), "served_by": {str(k): v for k, v in served.items()},
            "infra_terminal_resent": len(resend), "identity": ident, "terminal": ready["terminal"], "requests": ready["requests"],
            "batch_headers": [{"model": a, "endpoint": b, "provider": json.loads(c)} for a, b, c in sorted(headers)]}


def _q(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    return float(xs[min(len(xs) - 1, int(p * len(xs)))]) if xs else 0.0


def measure_ledger(ledger_dir: str, rows_by_key: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The usage of a ledger's verdicts, for the dry run's cost model: the model's prompt tokens per o200k token of the same text (the
    estimate's framing included), the share of prompt tokens read from the provider's cache, completion / reasoning / answer tokens,
    and the reported cost."""
    con = sqlite3.connect(f"file:{os.path.join(ledger_dir, 'state.sqlite')}?mode=ro", uri=True)
    rs = con.execute("SELECT custom_id, usage_json, cost, provider FROM requests WHERE outcome = 'verdict' AND usage_json IS NOT NULL").fetchall()
    con.close()
    ratio, prompt, cached, comp, reas, ans, cost, prov = [], [], [], [], [], [], [], collections.Counter()
    for cid, u, c, pv in rs:
        u = json.loads(u)
        r = rows_by_key.get(cid)
        if r is None:
            continue
        pt = int(u.get("prompt_tokens") or 0)
        ct = int(u.get("completion_tokens") or 0)
        rt = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        ratio.append(pt / T.est_input_tokens(r))
        prompt.append(pt)
        cached.append(int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0))
        comp.append(ct)
        reas.append(rt)
        ans.append(ct - rt)
        cost.append(float(c or 0))
        prov[pv] += 1
    if not prompt:
        raise SystemExit(f"{ledger_dir} holds no verdict with usage to calibrate on")
    return {"ledger": os.path.abspath(ledger_dir), "n": len(prompt), "providers": dict(prov),
            "prompt_tokens_per_o200k": round(statistics.mean(ratio), 4), "cached_share": round(sum(cached) / sum(prompt), 4),
            "completion_mean": round(statistics.mean(comp), 1), "completion_p99": _q(comp, 0.99), "completion_max": max(comp),
            "reasoning_mean": round(statistics.mean(reas), 1), "reasoning_p90": _q(reas, 0.9), "reasoning_p99": _q(reas, 0.99),
            "reasoning_max": max(reas), "answer_mean": round(statistics.mean(ans), 1), "answer_max": max(ans),
            "cost_reported_usd_no_fee": round(sum(cost), 6), "cost_per_request_usd_no_fee": round(sum(cost) / len(cost), 6)}


def arm_scenarios(model: str, rows: List[Mapping[str, Any]], calib: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """USD incl. the credit fee, three readings of the same inputs under the calibrated token counts (the model's prompt tokens =
    o200k x ``prompt_tokens_per_o200k``):
      low      the shared prefix (tool + head) read from the provider's cache by every request but the first of its behaviour and
               prefix (DeepInfra bills cache writes as plain input: ``cache_write_tokens`` 0 throughout the judge comparison); output the
               calibration's mean completion
      central  the calibration's cached share of prompt tokens read from the cache, the rest plain input; output the mean completion
      high     no cache read at all; output the Opus 5 answers' mean (o200k) plus the calibration's p99 reasoning, every request"""
    k = float(calib["prompt_tokens_per_o200k"])
    n = len(rows)
    inp = sum(T.est_input_tokens(r) for r in rows) * k
    pfx = sum(float(r["tokens"]["head"] + r["tokens"]["tool"]) for r in rows) * k
    first: Dict[Any, float] = {}
    for r in rows:
        first.setdefault((r["behavior"], r["prefix_sha"]), float(r["tokens"]["head"] + r["tokens"]["tool"]) * k)
    reads_best = pfx - sum(first.values())
    out_mean = float(calib["completion_mean"]) * n
    out_high = (T.EST_ANSWER_TOKENS + float(calib["reasoning_p99"])) * n
    cached = inp * float(calib["cached_share"])
    sc = {"low": {"input_tokens": inp, "cache_read_tokens": reads_best, "output_tokens": out_mean,
                  "usd_with_fee": BF.estimate_cost(model, inp - reads_best, out_mean, cache_read_tokens=reads_best)},
          "central": {"input_tokens": inp, "cache_read_tokens": cached, "output_tokens": out_mean,
                      "usd_with_fee": BF.estimate_cost(model, inp - cached, out_mean, cache_read_tokens=cached)},
          "high": {"input_tokens": inp, "cache_read_tokens": 0.0, "output_tokens": out_high,
                   "usd_with_fee": BF.estimate_cost(model, inp, out_high)}}
    return {name: {kk: round(v, 4 if kk == "usd_with_fee" else 0) for kk, v in d.items()} for name, d in sc.items()}


def arm_dry_run(run_dir: str, rows: List[Dict[str, Any]], mans: List[Dict[str, Any]], spec: Dict[str, Any], prof: Mapping[str, Any],
                n_excluded: int, reused: List[Mapping[str, Any]], calib: Optional[Mapping[str, Any]],
                calib_rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    model = spec["model"]
    p_in, p_out, p_cr, _ = BF.PRICES_PER_MTOK[model]
    est = T.cache_estimate(rows)
    max_tokens = int(spec["request"]["max_tokens"])
    precheck = BF.estimate_cost(model, (est["uncached"] + est["prefix"]) * (float(calib["prompt_tokens_per_o200k"]) if calib else 1.0),
                                max_tokens * est["n"], with_fee=False)
    scen = arm_scenarios(model, rows, calib) if calib else None
    check = None
    if calib:
        own = arm_scenarios(model, calib_rows, calib)["central"]["usd_with_fee"]
        check = {"inputs": len(calib_rows), "central_model_usd_with_fee": own,
                 "reported_usd_with_fee": round(float(calib["cost_reported_usd_no_fee"]) * (1 + BF.CREDIT_FEE), 4)}
    rep = {"run_dir": run_dir, "judge": prof["name"], "model": model, "endpoint": spec["endpoint"], "provider": spec.get("provider"),
           "request": spec["request"], "truncated_retry_max_tokens": spec.get("truncated_retry_max_tokens"),
           "prices_per_mtok": {"input": p_in, "output": p_out, "cache_read": p_cr}, "credit_fee": BF.CREDIT_FEE,
           "requests": est["n"], "batches": len(mans), "body_bytes": sum(m["body_bytes"] for m in mans),
           "max_batch_bytes": max((m["body_bytes"] for m in mans), default=0), "max_batch_requests": max((m["n_requests"] for m in mans), default=0),
           "per_behavior": dict(collections.Counter(r["behavior"] for r in rows)),
           "tokens_o200k": {"input_total": est["prefix"] + est["uncached"], "prefix_total": est["prefix"], "distinct_prefixes": est["n_prefixes"]},
           "calibration": calib, "calibration_check": check, "cost_usd_with_fee": scen,
           "openrouter_402_precheck_usd_no_fee": round(precheck, 2), "excluded_keys": n_excluded,
           "reused_ledgers": [{k: v for k, v in r.items() if k != "keys"} for r in reused],
           "submit_gate": spec.get("submit_gate"), "max_spend_usd": spec.get("max_spend_usd"), "estimate_per_request": spec.get("estimate"),
           "ceiling_estimate_usd_with_fee": round(sum(((m.get("est") or {}).get("cost_usd_with_fee") or 0) for m in mans), 4),
           "manifests": [{"batch_key": m["batch_key"], "n_requests": m["n_requests"], "body_bytes": m["body_bytes"],
                          "behavior": (m.get("extra") or {}).get("behavior")} for m in mans]}
    json.dump(rep, open(os.path.join(run_dir, "dry_run.json"), "w"), indent=1)
    L = [f"# Dry run: {os.path.basename(run_dir.rstrip('/'))} (Petri judge on {prof['judge_model']}, NOT submitted)", "",
         f"- model `{model}` (`{model}:batch`, ${p_in:.2f} in / ${p_out:.2f} out / ${p_cr:.2f} cache read per MTok, +{BF.CREDIT_FEE:.1%} credit fee) "
         f"via OpenRouter Batch API `{spec['endpoint']}`, provider `{json.dumps(spec.get('provider'))}`",
         f"- request: the candidate `{prof['arm']}`'s body exactly; settings `{json.dumps(spec['request'])}`, tool_choice "
         f"`{spec.get('tool_choice')}`, truncated re-ask cap {spec.get('truncated_retry_max_tokens'):,}",
         f"- **{est['n']:,} requests** in **{len(mans)} batches** ({rep['body_bytes'] / 1e6:,.1f} MB; largest {rep['max_batch_bytes'] / 1e6:,.1f} MB, "
         f"{rep['max_batch_requests']:,} requests)",
         f"- input (o200k): {rep['tokens_o200k']['input_total'] / 1e6:,.2f} M; {n_excluded:,} keys excluded"]
    for r in reused:
        L.append(f"- reused ledger `{r['ledger']}`: {r['n_keys']:,} keys excluded (outcomes {r['outcomes']}, served by {r['served_by']}); "
                 f"all {r['bodies_byte_identical']:,} bodies it sent byte-identical to this build's; {r['infra_terminal_resent']} "
                 f"infrastructure-terminal keys re-sent here; its batch headers {r['batch_headers']}")
    if calib:
        L += [f"- cost model calibrated on {calib['n']} verdicts of `{calib['ledger']}` (providers {calib['providers']}): prompt tokens = o200k x "
              f"{calib['prompt_tokens_per_o200k']}, cached share {calib['cached_share']:.1%}, completion mean {calib['completion_mean']:,.0f} "
              f"(reasoning mean {calib['reasoning_mean']:,.0f}, p90 {calib['reasoning_p90']:,.0f}, p99 {calib['reasoning_p99']:,.0f}; answer max "
              f"{calib['answer_max']:,})",
              f"- model check: the central reading of those {check['inputs']} inputs is ${check['central_model_usd_with_fee']:,.2f}; "
              f"OpenRouter reported ${check['reported_usd_with_fee']:,.2f} for them (both incl. fee)",
              "", "| reading | input M | cache-read M | output M | USD incl. fee |", "|---|---|---|---|---|"]
        for name in ("low", "central", "high"):
            d = scen[name]
            L.append(f"| {name} | {d['input_tokens'] / 1e6:,.2f} | {d['cache_read_tokens'] / 1e6:,.2f} | {d['output_tokens'] / 1e6:,.2f} | "
                     f"${d['usd_with_fee']:,.2f} |")
        L += ["", "low: the shared prefix read from the cache by every request but the first per behaviour and prefix, mean completion; "
              "central: the calibrated cached share, mean completion; high: no cache, output = the Opus 5 answers' mean + the p99 reasoning "
              "for every request. Re-asks (none in the judge comparison) are not included.", ""]
    L += [f"- OpenRouter's 402 pre-check prices the batches at about ${precheck:,.2f} (input + max_tokens {max_tokens:,} x every request at "
          f"the output price, no fee); the balance must cover it at submit",
          f"- submit gate: {(spec.get('submit_gate') or {}).get('requires_files') or 'none'}",
          _ceiling_line(rep, mans), "", "| batch | behaviour | requests | MB |", "|---|---|---|---|"]
    for m in rep["manifests"]:
        L.append(f"| {m['batch_key']} | {m['behavior']} | {m['n_requests']:,} | {m['body_bytes'] / 1e6:,.1f} |")
    open(os.path.join(run_dir, "dry_run.md"), "w").write("\n".join(L) + "\n")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enum", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--keys", help="file of judge-input keys to include (one per line)")
    ap.add_argument("--exclude-keys", action="append", default=[], help="file of judge-input keys to drop (held by another run directory)")
    ap.add_argument("--judge", default=J.DEFAULT, choices=sorted(J.PROFILES), help="the judge profile (judges.py)")
    ap.add_argument("--reuse-ledger", action="append", default=[],
                    help="a ledger of the same judge whose verdicts are reused (e.g. the judge comparison's run directory): checked, then excluded")
    ap.add_argument("--calibrate-from", help="the ledger whose verdicts' usage calibrates the dry run (default the first --reuse-ledger)")
    ap.add_argument("--est-reasoning-tokens", type=float, default=None,
                    help="reasoning tokens per request the cost estimates (and so the spend ceiling) assume (default the judge profile's)")
    ap.add_argument("--max-spend-usd", type=float, default=0.0, help="the run's spend ceiling, USD incl. the credit fee (0 = none)")
    ap.add_argument("--require-file", action="append", default=[], help="a file that must exist before this run submits (repeatable)")
    ap.add_argument("--no-submit-gate", action="store_true", help="build a full-scope run with no submit gate (explicit)")
    ap.add_argument("--prefix", default="b")
    ap.add_argument("--max-requests", type=int)
    args = ap.parse_args(argv)
    prof = J.profile(args.judge)
    est_reasoning = float(prof["est_reasoning_tokens"] if args.est_reasoning_tokens is None else args.est_reasoning_tokens)
    if args.keys is None and not args.require_file and not args.no_submit_gate:
        raise SystemExit("a full-scope run (no --keys) needs --require-file PATH or --no-submit-gate")
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    keys = read_key_file(args.keys) if args.keys else None
    drop = set()
    for f in args.exclude_keys:
        drop |= read_key_file(f)
    if os.path.isdir(run_dir) and BF.list_manifests(run_dir):
        raise SystemExit(f"{run_dir} already holds manifests; a run directory is written once")
    tools = json.load(open(os.path.join(args.enum, "tools.json")))
    reused: List[Dict[str, Any]] = []
    calib = None
    calib_rows: List[Mapping[str, Any]] = []
    if args.reuse_ledger or args.calibrate_from:
        every = {r["key"]: r for r in dedup(read_inputs(args.enum))}
        for led in args.reuse_ledger:                # every check before anything is written: a refused build leaves nothing
            reused.append(check_reuse(os.path.abspath(os.path.expanduser(led)), prof, every, tools))
            drop |= set(reused[-1]["keys"])
        cal = args.calibrate_from or (args.reuse_ledger[0] if args.reuse_ledger else None)
        if cal:
            calib = measure_ledger(os.path.abspath(os.path.expanduser(cal)), every)
            con = sqlite3.connect(f"file:{os.path.join(os.path.abspath(os.path.expanduser(cal)), 'state.sqlite')}?mode=ro", uri=True)
            calib_rows = [every[c] for (c,) in con.execute("SELECT custom_id FROM requests WHERE outcome = 'verdict' AND usage_json IS NOT NULL")
                          if c in every]
            con.close()
    rows = dedup(read_inputs(args.enum, keys))
    if keys is not None and len(rows) != len(keys):
        missing = sorted(keys - {r["key"] for r in rows})
        raise SystemExit(f"{len(missing)} listed keys are not in {args.enum} (first: {missing[:3]})")
    n_before = len(rows)
    rows = [r for r in rows if r["key"] not in drop]
    if not rows:
        raise SystemExit("no inputs left after --exclude-keys")
    extra = {"enum_dir": os.path.abspath(args.enum),
             "keys_file": os.path.abspath(args.keys) if args.keys else None,
             "exclude_keys_files": [os.path.abspath(f) for f in args.exclude_keys]}
    if reused:
        extra["reused_ledgers"] = [{"ledger": r["ledger"], "n_keys": r["n_keys"], "bodies_byte_identical": r["bodies_byte_identical"],
                                    "outcomes": r["outcomes"], "served_by": r["served_by"], "infra_terminal_resent": r["infra_terminal_resent"]}
                                   for r in reused]
    spec = T.init_run(run_dir, extra=extra, max_spend_usd=args.max_spend_usd, est_reasoning_tokens=est_reasoning,
                      requires_files=[os.path.abspath(os.path.expanduser(f)) for f in args.require_file], judge=prof["name"])
    mans = T.add_inputs(run_dir, rows, tools, prefix=args.prefix, max_requests=args.max_requests)
    rep = arm_dry_run(run_dir, rows, mans, spec, prof, n_before - len(rows), reused, calib, calib_rows)
    print(json.dumps({k: rep[k] for k in ("requests", "batches", "body_bytes", "excluded_keys", "cost_usd_with_fee", "max_spend_usd",
                                          "ceiling_estimate_usd_with_fee", "openrouter_402_precheck_usd_no_fee")}, indent=1))
    print(_ceiling_line(rep, mans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
