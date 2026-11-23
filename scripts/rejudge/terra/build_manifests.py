#!/usr/bin/env python3
"""Build one family's run directory of batch manifests from its scope (prepare only: nothing is submitted here).

    HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts python scripts/rejudge/terra/build_manifests.py --family cpi \
        --scope SCOPE/cpi --run-dir TRACK_DIR/full/cpi --require-file TRACK_DIR/GO \
        [--judge minimax-m3] [--keys KEYS.txt] [--exclude-keys KEYS.txt ...] [--max-tokens N] [--max-spend-usd USD] [--workers 8]

Writes ``<run_dir>/units.tsv.gz`` (the scope's units whose key is in this run: the filing's map from a request back to its run / item /
trial / sample), ``track.json``, ``manifests/`` + ``bodies/`` (one request per distinct key, packed under the byte and request caps), and
``dry_run.json`` / ``dry_run.md``: requests, units, batches, bytes, estimated tokens and cost incl. the 5.5% credit fee, per eval and lane,
plus the commands. ``--keys`` restricts the run to a key list (a pilot, a follow-up, or the agentic second rung: ``--max-tokens 4000
--keys <truncated keys>``); ``--exclude-keys`` (repeatable) drops keys judged elsewhere (a pilot's, or the verdicts ``reuse.py`` carries
from a same-protocol run: the filing reads those from their own run directory).

``--require-file PATH`` (repeatable) gates the run's batch submit on those files (``track.json`` ``submit_gate.requires_files``, as the
StrongREJECT build writes it; ``--gate-note`` says why): a source built only to feed a live run names a file that never exists, so its
batches can never be POSTed.

``--judge`` names the judge model's config entry (:mod:`rejudge.terra.judges`; default ``judges.DEFAULT_JUDGE``). With a judge whose
entry states a context (MiniMax-M3: 524,288 tokens), a request whose estimated prompt x the entry's guard ratio plus its largest cap
could exceed it is NOT packed: it is listed in ``over_context.txt`` and the dry run (its units stay unjudged; a judged text is never
cut). With a calibrated entry the dry run adds low / central / high cost bands (``judges.cost_bands``).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.terra import judges as J  # noqa: E402
from rejudge.terra import track as T  # noqa: E402

_G: Dict[str, Any] = {}


def _init(family: str, rq: Dict[str, Any]) -> None:
    _G["family"] = family
    _G["rq"] = rq


def _pack(line: str):
    x = json.loads(line)
    fam = _G["family"]
    try:
        e = T.entry_for(fam, x, _G["rq"])
    except ValueError as exc:
        return ("MISMATCH", x["k"], str(exc), 0.0, 0)
    toks = T.tokens(T.message_text(e["body"]))
    return ("OK", e["custom_id"], BF.serialize_request(e["custom_id"], e["body"]), toks + T.est_overhead_tokens(fam, _G["rq"]),
            len(x.get("response") or x.get("prompt") or ""))


def iter_texts(scope: str, keep: Optional[set], drop: Optional[set]) -> Iterator[str]:
    with gzip.open(os.path.join(scope, "texts.jsonl.gz"), "rt") as fh:
        for line in fh:
            k = line[7:71]          # '{"k": "<64 hex>"' -- json.dumps puts k first
            if (keep is not None and k not in keep) or (drop is not None and k in drop):
                continue
            yield line


def unit_groups(scope: str, keys: Optional[set]) -> Dict[str, str]:
    """custom_id -> "<eval>|<lane>" of the first unit (in scope order) that carries it."""
    lane = {}
    with open(os.path.join(scope, "runs.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            lane[r["run_id"]] = f"{r['eval_id']}|{r.get('lane') or '-'}"
    out: Dict[str, str] = {}
    with gzip.open(os.path.join(scope, "units.tsv.gz"), "rt") as fh:
        next(fh)
        for line in fh:
            p = line.split("\t", 6)
            k = p[5]
            if (keys is None or k in keys) and k not in out:
                out[k] = lane.get(p[0], p[0])
    return out


def render_md(rep: Dict[str, Any]) -> str:
    a = rep["groups"]["all"]
    rd = rep["run_dir"]
    lines = [f"# Dry run: {rd}", "", f"Built {rep['built_utc']}; family {rep['family']}; model {rep['model']}, endpoint {rep['endpoint']}; "
             f"request settings {json.dumps(rep['request'])}.", "",
             f"- requests (distinct judge inputs): {a['requests']}", f"- units they cover: {rep['units']} in {rep['runs']} paper runs",
             f"- batches: {rep['batches']} (caps: {rep['max_bytes']} bytes, {rep['max_requests']} requests; largest {rep['largest_batch_bytes']} bytes)",
             f"- request bytes: {a['bytes']}", f"- estimated input tokens: {a['est_input_tokens']}; output tokens: {a['est_output_tokens']}",
             f"- estimated cost: ${a['est_cost_usd']} at the batch price, ${a['est_cost_usd_with_fee']} with the {BF.CREDIT_FEE:.1%} credit fee",
             f"- the 402 check's worst case (every request at max_tokens): ${rep['max_output_cost_usd_with_fee']} with the fee"]
    if rep.get("judge"):
        lines.insert(3, f"Judge config entry: {rep['judge']} (scripts/rejudge/terra/judges.py); truncated_retry_max_tokens "
                        f"{rep.get('truncated_retry_max_tokens')}; context guard: {rep.get('context')}.")
    if rep.get("excluded_keys"):
        lines.append(f"- keys excluded (judged elsewhere, e.g. verdicts carried from a same-protocol run): {rep['excluded_keys']}")
    if rep.get("over_context") is not None:
        lines.append(f"- requests NOT packed because they could exceed the context: {rep['over_context']} (over_context.txt); "
                     f"largest estimated prompt {rep['largest_est_input_tokens']} tokens (o200k + overhead)")
    if rep.get("cost_bands"):
        cb = rep["cost_bands"]
        lines += ["", "| cost band | input tokens | output tokens | USD incl. fee |", "|---|---:|---:|---:|"]
        lines += [f"| {b} | {cb[b]['input_tokens']} | {cb[b]['output_tokens']} | ${cb[b]['usd_with_fee']} |" for b in ("low", "central", "high")]
        lines += ["", "Bands: " + " ".join(cb["assumptions"].split())]
    lines += ["", "| eval / lane | requests | bytes | est. input tokens | est. cost incl. fee |", "|---|---:|---:|---:|---:|"]
    for g, s in rep["groups"].items():
        if g != "all":
            lines.append(f"| {g} | {s['requests']} | {s['bytes']} | {s['est_input_tokens']} | ${s['est_cost_usd_with_fee']} |")
    lines += ["", "| batch | requests | bytes | est. cost incl. fee |", "|---|---:|---:|---:|"]
    for b in rep["batch_list"]:
        lines.append(f"| {b['batch_key']} | {b['n_requests']} | {b['body_bytes']} | ${b['est_cost_usd_with_fee']} |")
    lines += ["", "Assumptions: " + json.dumps(rep["assumptions"]), "", "## Commands", "",
              "Dry run of the submit (no network):", "", f"    python scripts/rejudge/terra/harvest.py --run-dir {rd} --dry-run", "",
              "The harvester ($OPENROUTER_API_KEY in its environment; a long run belongs in a process that outlives the shell):", "",
              f"    python scripts/rejudge/terra/harvest.py --run-dir {rd} --poll-s 120 --max-hours 72", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=T.FAMILIES, required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--keys")
    ap.add_argument("--exclude-keys", action="append", default=[], help="keys judged elsewhere (repeatable)")
    ap.add_argument("--max-spend-usd", type=float, help="track.json max_spend_usd: the submitter pauses before a batch that would take the "
                    "committed spend (USD incl. the credit fee) over it")
    ap.add_argument("--judge", default=None, help=f"a judges.JUDGES entry (default {J.DEFAULT_JUDGE}): {sorted(J.JUDGES)}")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--max-bytes", type=int)
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefix", default="b")
    ap.add_argument("--require-file", action="append", default=[], help="a file that must exist before this run submits (repeatable)")
    ap.add_argument("--gate-note", default=None, help="why the gate files gate the submit (track.json submit_gate.note)")
    args = ap.parse_args(argv)
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if os.path.exists(os.path.join(run_dir, "manifests")) and os.listdir(os.path.join(run_dir, "manifests")):
        raise SystemExit(f"{run_dir} already has manifests: a run directory is built once")
    keep = {l.strip() for l in open(args.keys) if l.strip()} if args.keys else None
    drop = {l.strip() for f in args.exclude_keys for l in open(f) if l.strip()} if args.exclude_keys else None
    judge = args.judge or J.DEFAULT_JUDGE
    exclude_extra = (args.exclude_keys[0] if len(args.exclude_keys) == 1 else args.exclude_keys) if args.exclude_keys else None
    gate: Dict[str, Any] = {}
    if args.require_file:
        gate["submit_gate"] = {"requires_files": [os.path.abspath(os.path.expanduser(f)) for f in args.require_file]}
        if args.gate_note:
            gate["submit_gate"]["note"] = args.gate_note
    spec = T.init_run(run_dir, args.family, max_tokens=args.max_tokens, judge=judge,
                      extra=dict({"scope": os.path.abspath(args.scope), "keys_file": args.keys, "exclude_keys_file": exclude_extra},
                                 **({"max_spend_usd": float(args.max_spend_usd)} if args.max_spend_usd else {}), **gate))
    rq = spec["request"]
    est_out = T.est_output_tokens(args.family, rq)
    over_context: List[str] = []
    largest = [0.0, None]
    groups_of = unit_groups(args.scope, keep)
    header = BF.batch_header(spec["model"], spec["endpoint"], spec.get("provider"))
    stats: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    mismatches: List[str] = []
    max_tokens = int(rq["max_tokens"])

    def packed() -> Iterator[BF.PackedRequest]:
        with Pool(args.workers, initializer=_init, initargs=(args.family, rq)) as pool:
            for st, k, line, est_in, nchars in pool.imap(_pack, iter_texts(args.scope, keep, drop), chunksize=256):
                if st != "OK":
                    mismatches.append(f"{k}: {line}")
                    continue
                if k in seen:
                    continue
                seen.add(k)
                if est_in > largest[0]:
                    largest[0], largest[1] = est_in, k
                if not J.context_ok(est_in, judge, args.family):
                    over_context.append(f"{k}\t{round(est_in)}")
                    continue
                g = groups_of.get(k, "unmapped")
                for gg in ("all", g):
                    s = stats[gg]
                    s["requests"] += 1
                    s["bytes"] += len(line)
                    s["est_input_tokens"] += est_in
                    s["est_output_tokens"] += est_out
                    s["max_output_tokens"] += max_tokens
                yield BF.PackedRequest(k, line, est_input_tokens=est_in, est_output_tokens=est_out)

    seen: set = set()
    mans = []
    t0 = time.time()
    for i, part in enumerate(BF.chunk(packed(), header, int(args.max_bytes or spec["max_bytes"]), int(args.max_requests or spec["max_requests"]))):
        mans.append(BF.write_manifest(run_dir, f"{args.prefix}{i:05d}", header, part, spec["track"]))
        print(f"  {mans[-1]['batch_key']}: {mans[-1]['n_requests']} requests, {mans[-1]['body_bytes']} bytes", flush=True)
    if mismatches:
        print(f"REFUSED: {len(mismatches)} texts no longer recompute their key (first: {mismatches[0]})")
    with open(os.path.join(run_dir, "over_context.txt"), "w") as fh:
        fh.write("".join(l + "\n" for l in over_context))
    if over_context:
        print(f"NOT PACKED: {len(over_context)} requests could exceed the {J.get(judge).get('context_tokens')}-token context (over_context.txt)")
    keys_in = {k for m in mans for k in m["custom_ids"]}
    n_units = 0
    runs = set()
    with gzip.open(os.path.join(args.scope, "units.tsv.gz"), "rt") as src, gzip.open(os.path.join(run_dir, "units.tsv.gz"), "wt") as dst:
        dst.write(next(src))
        for line in src:
            p = line.split("\t", 6)
            if p[5] in keys_in:
                dst.write(line)
                n_units += 1
                runs.add(p[0])
    model = spec["model"]
    groups = {}
    for g, s in sorted(stats.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        groups[g] = {"requests": s["requests"], "bytes": s["bytes"], "est_input_tokens": round(s["est_input_tokens"]),
                     "est_output_tokens": round(s["est_output_tokens"]),
                     "est_cost_usd": round(BF.estimate_cost(model, s["est_input_tokens"], s["est_output_tokens"], with_fee=False), 2),
                     "est_cost_usd_with_fee": round(BF.estimate_cost(model, s["est_input_tokens"], s["est_output_tokens"]), 2)}
    a = stats["all"]
    rep = {"run_dir": run_dir, "family": args.family, "built_utc": BF.utc_now(), "model": model, "endpoint": spec["endpoint"], "request": rq,
           "units": n_units, "runs": len(runs), "batches": len(mans), "max_bytes": int(args.max_bytes or spec["max_bytes"]),
           "max_requests": int(args.max_requests or spec["max_requests"]), "largest_batch_bytes": max((m["body_bytes"] for m in mans), default=0),
           "groups": groups, "mismatches": len(mismatches),
           "max_output_cost_usd_with_fee": round(BF.estimate_cost(model, a["est_input_tokens"], a["max_output_tokens"]), 2),
           "batch_list": [{"batch_key": m["batch_key"], "n_requests": m["n_requests"], "body_bytes": m["body_bytes"],
                           "est_cost_usd_with_fee": m["est"]["cost_usd_with_fee"]} for m in mans],
           "assumptions": {"tokenizer": "o200k_base estimator; the judge's own tokenizer differs (the calibration's prompt_ratio corrects it in the bands)",
                           "overhead_tokens_per_request": T.est_overhead_tokens(args.family, rq),
                           "output_tokens_per_request": est_out, "reasoning": rq.get("reasoning", rq.get("reasoning_effort")),
                           "cache": "no cache credit in the manifest estimates (the low band applies the measured charge-over-list factor)",
                           "re_asks": "not included in the manifest estimates (the bands include the pilot's measured re-ask share)"},
           "build_seconds": round(time.time() - t0, 1)}
    cfg = J.get(judge)
    pin, pout = BF.PRICES_PER_MTOK[model][:2]
    rep.update({"judge": judge, "batch_model_variant": cfg["batch_variant"], "truncated_retry_max_tokens": spec.get("truncated_retry_max_tokens"),
                "context": {"context_tokens": cfg.get("context_tokens"), "guard_ratio": cfg.get("context_ratio_guard"),
                            "largest_cap": max(max_tokens, int(spec.get("truncated_retry_max_tokens") or 0))},
                "over_context": len(over_context), "largest_est_input_tokens": round(largest[0]), "largest_est_input_key": largest[1],
                "excluded_keys": len(drop or ()), "cost_bands": J.cost_bands(judge, args.family, a["requests"], a["est_input_tokens"], (pin, pout))})
    with open(os.path.join(run_dir, "dry_run.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(os.path.join(run_dir, "dry_run.md"), "w") as fh:
        fh.write(render_md(rep))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("batch_list", "groups")}, indent=1))
    print(json.dumps(groups["all"], indent=1))
    return 2 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
