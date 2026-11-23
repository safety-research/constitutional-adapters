#!/usr/bin/env python3
"""Merge the per-site enumerations and build a run directory of batch manifests (prepare only: nothing is submitted here).

    PYTHONPATH=src python scripts/rejudge/strongreject/build_manifests.py --enum ENUM/local ENUM/cluster_a ENUM/cluster_b \
        --goals SCOPE/goals.json --run-dir RUN_DIR [--judge minimax-m3] [--keys KEYS.txt] [--exclude-keys KEYS.txt] [--max-tokens N] \
        [--reuse-run-dir SAME_PROTOCOL_RUN] [--tokenizer TOKENIZER.json] [--workers 6] (--require-file PATH | --no-submit-gate)

``--judge`` names the model's entry in ``judges.JUDGES`` (default ``judges.DEFAULT``): the model, track name, request settings, reply cap,
re-asks and the estimate assumptions all come from it (``--max-tokens`` overrides the cap).

Writes ``<run_dir>/units.tsv.gz`` (every unit of the merged sites, the filing's map from a request back to its run / item / trial /
sample / turn), ``track.json``, ``manifests/`` + ``bodies/`` (one request per distinct key, packed under the byte and request caps) and
``dry_run.json`` / ``dry_run.md``: requests, batches, bytes, estimated tokens and cost incl. the 5.5% credit fee (the entry's point
estimate, and for a measured entry a low / central / high band and the expected turnaround), per eval category and campaign, plus the
exact submit and harvester commands. ``--keys`` restricts the run to a key list (a pilot, a follow-up); ``--exclude-keys`` drops keys
already judged elsewhere (a pilot's, whose verdicts the filing reads from the pilot's run directory).

REUSE (``--reuse-run-dir``, repeatable): a run directory of the SAME judge protocol (e.g. a judge comparison's run of this model) whose verdicts the
full run takes instead of re-sending the same input. A key is reused only when (a) its outcome there is terminal and is a verdict, or
an item outcome (moderation, judge refusal, parse error, truncation) whose re-ask budget there was at least this run's, and (b) the
request body built here is byte-identical to the one that run sent for it (attempt 0). Reused keys get no request here but keep their
units in ``units.tsv.gz``; ``reuse.json`` lists them, and ``track.json`` ``reuse_run_dirs`` names the directories, which the filing must
be given as ``--run-dir`` too (``file_records.py`` refuses otherwise). Keys not yet terminal there are sent here.

CONTEXT. For an entry with ``context_tokens``, every request's prompt tokens plus its reply cap must fit the context: counted EXACTLY
with the model's own tokenizer when ``--tokenizer`` names it (the entry's ``tokenizer`` pins its sha256 and the constant chat/tool
overhead the provider bills), else the o200k count times the entry's ``input_ratio_high``. A request over it refuses the build (none is
ever truncated).

A run built without ``--keys`` (the full scope) must name ``--require-file PATH`` (repeatable; nothing is submitted until every such file
exists; ``--gate-note`` says why) or pass ``--no-submit-gate`` explicitly. A run directory that exists only as the source of a live run
names a file that is never created, so its batch harvester never submits. ``--max-spend-usd`` sets the run's spend
ceiling (USD incl. the credit fee), or ``--max-spend-x-high X`` sets it to X times the band's high end once the build has priced it: the
harvester pauses before a submit would pass it.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.strongreject import judges as J  # noqa: E402
from rejudge.strongreject import request as R  # noqa: E402
from rejudge.strongreject import track as T  # noqa: E402

#: terminal outcomes about the ITEM that a reuse may carry (an infrastructure outcome is re-sent instead)
REUSABLE_ITEM_OUTCOMES = ("moderation", "judge_refusal", "parse_error", "truncated")

_G: Dict[str, Any] = {}


def _init(goals_path: str, rq: Dict[str, Any], tokenizer_path: Optional[str], overhead: float) -> None:
    _G["goals"] = json.load(open(goals_path))["goals"]
    _G["rq"] = rq
    _G["tok"] = None
    _G["overhead"] = overhead
    if tokenizer_path:
        from tokenizers import Tokenizer
        _G["tok"] = Tokenizer.from_file(tokenizer_path)


def _pack(line: str):
    """One enumerated text -> ``(status, key, request line, est prompt tokens, response chars, exact model prompt tokens or None)``:
    the exact count (the model's tokenizer + the billed overhead) when a tokenizer is loaded, else o200k + the overhead."""
    x = json.loads(line)
    g = _G["goals"][x["goal_id"]]
    k = R.cache_key(x["kind"], x["dataset"], x["revision"], g["text"], x["response"])
    if k != x["k"]:
        return ("MISMATCH", x["k"], None, 0.0, 0, None)
    body = R.request_body(_G["rq"], x["kind"], g["text"], x["response"])
    prompt = body["messages"][0]["content"]
    exact = None
    if _G["tok"] is not None:
        exact = float(len(_G["tok"].encode(prompt, add_special_tokens=False).ids) + _G["overhead"])
        est_in = exact
    else:
        est_in = T._tokens(prompt) + _G["overhead"]
    return ("OK", k, BF.serialize_request(k, body), est_in, len(x["response"]), exact)


def iter_texts(enum_dirs: List[str], keep: Optional[set], drop: Optional[set]) -> Iterator[str]:
    seen = set()
    for d in enum_dirs:
        with gzip.open(os.path.join(d, "texts.jsonl.gz"), "rt") as fh:
            for line in fh:
                k = line[7:71]          # '{"k": "<64 hex>"' -- json.dumps puts k first
                if k in seen or (keep is not None and k not in keep) or (drop is not None and k in drop):
                    continue
                seen.add(k)
                yield line


# --------------------------------------------------------------------------------------------
# reuse of a same-protocol run directory's verdicts
# --------------------------------------------------------------------------------------------
def reuse_candidates(run_dir: str, entry: Mapping[str, Any]) -> Tuple[Dict[str, bytes], Dict[str, str], Dict[str, Any]]:
    """``(lines, outcomes, info)`` of a reuse run directory (read only): the attempt-0 request line it sent per key, the terminal
    outcome per key that may be reused, and what was checked. Refuses (SystemExit) a directory of another model or other request
    settings."""
    rd = os.path.abspath(os.path.expanduser(run_dir))
    spec = BF.read_track(rd)
    want = J.identity_request(entry)
    have = {k: v for k, v in (spec.get("request") or {}).items() if k != "max_tokens"}
    if spec.get("model") != entry["model"] or spec.get("endpoint") != entry["endpoint"] or have != want:
        raise SystemExit(f"--reuse-run-dir {rd}: model {spec.get('model')!r} / request {have} is not the {entry['judge_model']} entry's "
                         f"({entry['model']!r} / {want}); its verdicts are another protocol's")
    their_retry = dict(spec.get("retry") or {})
    our_retry = dict(entry["retry"])
    lines: Dict[str, bytes] = {}
    for key in BF.list_manifests(rd):
        man = BF.read_manifest(rd, key)
        if int(man.get("attempt") or 0) != 0:
            continue
        for r in BF.body_requests(BF.read_body(rd, man)):
            lines.setdefault(r["custom_id"], BF.serialize_request(r["custom_id"], r["body"]))
    outcomes: Dict[str, str] = {}
    skipped: collections.Counter = collections.Counter()
    con = sqlite3.connect(f"file:{os.path.join(rd, 'state.sqlite')}?mode=ro", uri=True, timeout=60)
    try:
        for cid, outcome, terminal in con.execute("SELECT custom_id, outcome, terminal FROM requests"):
            if not terminal:
                skipped["not_terminal"] += 1
            elif outcome == "verdict":
                outcomes[cid] = outcome
            elif outcome in REUSABLE_ITEM_OUTCOMES and int(their_retry.get(outcome, 0)) >= int(our_retry.get(outcome, 0)):
                outcomes[cid] = outcome
            else:
                skipped[f"terminal_{outcome}" + ("_fewer_reasks_there" if outcome in REUSABLE_ITEM_OUTCOMES else "")] += 1
        fp = con.execute("SELECT COUNT(*), COALESCE(SUM(terminal), 0) FROM requests").fetchone()
    finally:
        con.close()
    info = {"run_dir": rd, "track": spec.get("track"), "model": spec.get("model"), "request": spec.get("request"), "retry_there": their_retry,
            "retry_here": our_retry, "requests": int(fp[0]), "terminal": int(fp[1]), "candidates": len(outcomes),
            "candidate_outcomes": dict(collections.Counter(outcomes.values())), "not_candidates": dict(skipped)}
    return lines, outcomes, info


# --------------------------------------------------------------------------------------------
# cost bands
# --------------------------------------------------------------------------------------------
def output_tokens(est: Mapping[str, Any], eval_id: str, band: str = "central") -> float:
    """Completion tokens per call for ``eval_id``: the entry's per-eval measurement (``output_tokens_by_eval[_high]``), else its overall
    figure (``output_tokens[_high]``)."""
    if band == "high":
        by = est.get("output_tokens_by_eval_high") or {}
        return float(by.get(eval_id, est.get("output_tokens_high", est.get("output_tokens", T.EST_OUTPUT_TOKENS))))
    by = est.get("output_tokens_by_eval") or {}
    return float(by.get(eval_id, est.get("output_tokens", T.EST_OUTPUT_TOKENS)))


def cost_bands(model: str, est: Mapping[str, Any], n: float, prompt_tokens: float, out_central: float, out_high: float,
               max_tokens: int, prompt_tokens_high: Optional[float] = None) -> Dict[str, Any]:
    """USD incl. the credit fee for ``n`` requests of ``prompt_tokens`` provider prompt tokens in all (``out_*``: completion tokens in all):
      low      the measured cache reads (``cache_read_tokens_per_request`` at the cache-read price, else ``cache_read_share``), the
               measured output, no re-asks
      central  low x the measured re-asks (1 + ``parse_error_rate`` + ``parse_error_rate`` x ``reask_fail_rate``)
      high     no cache at all, ``prompt_tokens_high`` (default ``prompt_tokens``), the high output figure, the same re-asks
      submit_reservation_usd  what OpenRouter's 402 check prices at submit: every prompt token at the input price and ``max_tokens`` per
                              call (no fee: the balance check is on the batch price)"""
    p_in, p_out, p_cr, _ = BF.PRICES_PER_MTOK[model]
    if est.get("cache_read_tokens_per_request"):
        cached = min(prompt_tokens, float(est["cache_read_tokens_per_request"]) * n)
    else:
        cached = prompt_tokens * float(est.get("cache_read_share", 0.0))
    r1 = float(est.get("parse_error_rate", 0.0))
    r2 = float(est.get("reask_fail_rate", 0.0))
    reask = 1.0 + r1 + r1 * r2
    fee = 1.0 + BF.CREDIT_FEE
    ph = prompt_tokens if prompt_tokens_high is None else prompt_tokens_high
    low = ((prompt_tokens - cached) * p_in + cached * p_cr + out_central * p_out) / 1e6
    high = (ph * p_in + out_high * p_out) / 1e6 * reask
    return {"low_usd_with_fee": round(low * fee, 4), "central_usd_with_fee": round(low * reask * fee, 4), "high_usd_with_fee": round(high * fee, 4),
            "submit_reservation_usd": round((ph * p_in + n * max_tokens * p_out) / 1e6, 4),
            "assumed": {"cache_read_tokens": round(cached), "output_tokens_central": round(out_central), "output_tokens_high": round(out_high),
                        "reask_factor": round(reask, 4)}}


def turnaround_text(entry: Mapping[str, Any], n_batches: int, n_requests: int) -> str:
    t = entry.get("turnaround")
    if not t:
        return "not measured for this model"
    obs = ", ".join(f"{n:,} requests in {m:.0f} min" for n, m in t["observed"])
    biggest = max(n for n, _ in t["observed"])
    return (f"observed for {entry['label']} batches in the paper's runs: {obs}. This run is {n_batches} batches / {n_requests:,} requests, "
            f"{n_requests / biggest:,.0f}x the largest batch measured, so no measurement bounds it below the provider's "
            f"{t['completion_window_h']} h completion window per batch (an expired batch's requests are re-asked, up to 3 times): "
            f"hours if the provider runs the batches side by side, a day or more if it queues them")


def render_md(rep: Dict[str, Any]) -> str:
    """dry_run.md: the plan in one page -- totals, the groups, the batches, the commands."""
    a = rep["groups"].get("all", {})
    rd = rep["run_dir"]
    lines = [f"# Dry run: {rd}", "", f"Built {rep['built_utc']}; judge {rep.get('judge')}; model {rep['model']}, endpoint {rep['endpoint']}; "
             f"request settings {json.dumps(rep['request'])}.", "",
             f"- requests (distinct judge inputs sent): {a.get('requests')}", f"- units they cover (reused keys included): {rep['units']}",
             f"- batches: {rep['batches']} (caps: {rep['max_bytes']} bytes, {rep['max_requests']} requests; largest {rep['largest_batch_bytes']} bytes, "
             f"most requests {rep.get('largest_batch_requests')})",
             f"- request bytes: {a.get('bytes')}",
             f"- estimated prompt tokens: {a.get('est_input_tokens')} ({rep['assumptions'].get('input_tokens')}); output tokens: {a.get('est_output_tokens')}",
             f"- point estimate (no cache, mean output): ${a.get('est_cost_usd')} at the batch price, ${a.get('est_cost_usd_with_fee')} with the "
             f"{BF.CREDIT_FEE:.1%} credit fee"]
    if rep.get("bands"):
        b = rep["bands"]["all"]
        lines += [f"- cost band incl. fee: low ${b['low_usd_with_fee']}, central ${b['central_usd_with_fee']}, high ${b['high_usd_with_fee']} "
                  f"(dry_run.json `bands_assumptions`); OpenRouter's submit-time 402 check prices every batch at ~${b['submit_reservation_usd']} "
                  f"in all (every prompt token + max_tokens per call)"]
    if rep.get("max_spend_usd"):
        lines += [f"- spend ceiling (track.json max_spend_usd): ${rep['max_spend_usd']}"]
    if rep.get("context"):
        c = rep["context"]
        lines += [f"- context: largest prompt {c['largest_prompt_tokens']} tokens ({c['counting']}) + max_tokens {c['max_tokens']} vs the "
                  f"{c['context_tokens']}-token context: {c['over_context']} over"]
    if rep.get("reuse"):
        u = rep["reuse"]
        lines += [f"- reused verdicts (not re-sent): {u['reused']} keys from {', '.join(i['run_dir'] for i in u['dirs'])} "
                  f"({json.dumps(u['reused_outcomes'])}); candidates whose body differed (re-sent): {u['body_mismatch']}"]
    if rep.get("turnaround"):
        lines += [f"- turnaround: {rep['turnaround']}"]
    lines += ["", "| group | requests | reused | bytes | est. prompt tokens | point est. incl. fee | low | central | high |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g, s in rep["groups"].items():
        if g != "all":
            bd = (rep.get("bands") or {}).get(g) or {}
            lines.append(f"| {g} | {s.get('requests', 0)} | {s.get('reused', 0)} | {s.get('bytes', 0)} | {s.get('est_input_tokens', 0)} | "
                         f"${s.get('est_cost_usd_with_fee', 0)} | ${bd.get('low_usd_with_fee', '')} | ${bd.get('central_usd_with_fee', '')} | "
                         f"${bd.get('high_usd_with_fee', '')} |")
    lines += ["", "Assumptions: " + json.dumps(rep["assumptions"]), "", "## Commands", "",
              "Dry run of the submit (no network):", "", f"    python scripts/rejudge/common/harvester.py --run-dir {rd} --dry-run", "",
              "Submit everything and harvest (as a user unit that loads the key itself; scripts/rejudge/README.md):", "",
              f"    python scripts/rejudge/common/harvester.py --run-dir {rd} --poll-s 120 --max-hours 72", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enum", nargs="+", required=True)
    ap.add_argument("--goals", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge", default=J.DEFAULT, choices=sorted(J.JUDGES), help="the model's entry in judges.JUDGES")
    ap.add_argument("--keys")
    ap.add_argument("--exclude-keys")
    ap.add_argument("--max-tokens", type=int, default=None, help="override the entry's reply cap")
    ap.add_argument("--max-bytes", type=int, default=BF.DEFAULT_MAX_BYTES)
    ap.add_argument("--max-requests", type=int, default=BF.DEFAULT_MAX_REQUESTS)
    ap.add_argument("--prefix", default="b")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-spend-usd", type=float, default=0.0, help="the run's spend ceiling, USD incl. the credit fee (0 = none)")
    ap.add_argument("--max-spend-x-high", type=float, default=0.0, help="set the ceiling to this multiple of the band's high end")
    ap.add_argument("--require-file", action="append", default=[], help="a file that must exist before this run submits (repeatable)")
    ap.add_argument("--gate-note", default=None, help="why the gate files gate the submit (track.json submit_gate.note)")
    ap.add_argument("--no-submit-gate", action="store_true", help="build a full-scope run with no submit gate (explicit)")
    ap.add_argument("--reuse-run-dir", action="append", default=[], help="a same-protocol run directory whose terminal verdicts are reused")
    ap.add_argument("--tokenizer", help="the model's tokenizer.json (sha256 pinned by the entry): exact prompt tokens and context check")
    args = ap.parse_args(argv)
    gate_files = list(args.require_file)
    if args.keys is None and not gate_files and not args.no_submit_gate:
        raise SystemExit("a full-scope run (no --keys) needs --require-file PATH or --no-submit-gate")
    entry = J.get(args.judge)
    est_cfg = dict(entry["estimate"])
    tok_cfg = entry.get("tokenizer") or {}
    tokenizer_path = None
    overhead = float(est_cfg.get("overhead_tokens", T.EST_OVERHEAD_TOKENS))
    if args.tokenizer:
        tokenizer_path = os.path.abspath(os.path.expanduser(args.tokenizer))
        if not tok_cfg:
            raise SystemExit(f"--tokenizer given but the {args.judge} entry pins no tokenizer")
        sha = hashlib.sha256(open(tokenizer_path, "rb").read()).hexdigest()
        if sha != tok_cfg["sha256"]:
            raise SystemExit(f"--tokenizer {tokenizer_path} has sha256 {sha}, not the entry's {tok_cfg['sha256']} ({tok_cfg.get('hf_repo')})")
        overhead = float(tok_cfg["prompt_overhead_tokens"])
    rd = os.path.abspath(os.path.expanduser(args.run_dir))
    if os.path.isdir(os.path.join(rd, "manifests")) and os.listdir(os.path.join(rd, "manifests")):
        raise SystemExit(f"{rd} already holds manifests; build into a fresh run directory")
    reuse_lines: Dict[str, bytes] = {}
    reuse_outcome: Dict[str, str] = {}
    reuse_infos: List[Dict[str, Any]] = []
    for rr in args.reuse_run_dir:
        lines_, outs_, info_ = reuse_candidates(rr, entry)
        for k, v in outs_.items():
            if k not in reuse_outcome or (reuse_outcome[k] != "verdict" and v == "verdict"):
                reuse_outcome[k] = v
                reuse_lines[k] = lines_.get(k, b"")
        reuse_infos.append(info_)
        print(f"[build] reuse {info_['run_dir']}: {info_['candidates']} candidates {info_['candidate_outcomes']}, not: {info_['not_candidates']}",
              flush=True)
    os.makedirs(rd, exist_ok=True)
    os.chmod(rd, 0o700)
    keep = {l.strip() for l in open(args.keys) if l.strip()} if args.keys else None
    drop = {l.strip() for l in open(args.exclude_keys) if l.strip()} if args.exclude_keys else None
    extra: Dict[str, Any] = {"tag": args.tag, "max_bytes": args.max_bytes, "max_requests": args.max_requests}
    if reuse_infos:
        extra["reuse_run_dirs"] = [i["run_dir"] for i in reuse_infos]
    spec = T.init_run(rd, max_tokens=args.max_tokens, extra=extra, max_spend_usd=args.max_spend_usd,
                      requires_files=[os.path.abspath(os.path.expanduser(f)) for f in gate_files], judge=args.judge)
    if args.gate_note and spec.get("submit_gate"):
        spec["submit_gate"]["note"] = args.gate_note
        BF.write_track(rd, spec)
    max_tokens = int(spec["request"]["max_tokens"])
    ctx = int(entry.get("context_tokens") or 0)

    # units: the merged map, restricted to the run's keys (reused keys KEEP their units: the filing maps their verdicts)
    t0 = time.time()
    unit_counts: collections.Counter = collections.Counter()
    key_meta: Dict[str, tuple] = {}
    jobs = {}
    jpath = os.path.join(os.path.dirname(os.path.abspath(args.goals)), "jobs.jsonl")
    if os.path.exists(jpath):
        jobs = {j["run_id"]: j for j in (json.loads(l) for l in open(jpath))}
    with gzip.open(os.path.join(rd, "units.tsv.gz"), "wt") as uf:
        uf.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\tkind\tresponse_chars\n")
        for d in args.enum:
            with gzip.open(os.path.join(d, "units.tsv.gz"), "rt") as fh:
                next(fh)
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    k = parts[5]
                    if (keep is not None and k not in keep) or (drop is not None and k in drop):
                        continue
                    uf.write(line)
                    j = jobs.get(parts[0], {})
                    unit_counts[(j.get("category", "?"), j.get("campaign", "?"))] += 1
                    if k not in key_meta:
                        key_meta[k] = (j.get("category", "?"), j.get("campaign", "?"), j.get("eval_id", "?"))
    print(f"[build] units merged: {sum(unit_counts.values())} units, {len(key_meta)} keys in {time.time() - t0:.0f}s", flush=True)

    header = BF.batch_header(spec["model"], spec["endpoint"], spec.get("provider"))
    stats: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    mismatches: List[str] = []
    over_context: List[Tuple[str, float]] = []
    reused: Dict[str, str] = {}
    reuse_mismatch: List[str] = []
    largest_prompt: List[Any] = [0.0, None]
    mans: List[Dict[str, Any]] = []
    out_point = float(est_cfg.get("output_tokens", T.EST_OUTPUT_TOKENS))
    ratio = 1.0 if tokenizer_path else float(est_cfg.get("input_ratio", 1.0))
    ratio_high = 1.0 if tokenizer_path else float(est_cfg.get("input_ratio_high", ratio))

    def packed():
        with Pool(args.workers, initializer=_init, initargs=(args.goals, spec["request"], tokenizer_path, overhead)) as pool:
            for status, k, line, est_in, resp_chars, exact in pool.imap(_pack, iter_texts(args.enum, keep, drop), chunksize=128):
                if status != "OK":
                    mismatches.append(k)
                    continue
                cat, camp, ev = key_meta.get(k, ("?", "?", "?"))
                groups = (f"category:{cat}", f"campaign:{camp}", f"eval:{ev}", "all")
                if k in reuse_outcome:
                    if line == reuse_lines.get(k):
                        reused[k] = reuse_outcome[k]
                        for grp in groups:
                            stats[grp]["reused"] += 1
                        continue
                    reuse_mismatch.append(k)
                prompt_tok = est_in * ratio
                if prompt_tok > largest_prompt[0]:
                    largest_prompt[0], largest_prompt[1] = prompt_tok, k
                ctx_tok = exact if exact is not None else est_in * ratio_high
                if ctx and ctx_tok + max_tokens > ctx:
                    over_context.append((k, ctx_tok))
                    continue
                for grp in groups:
                    s = stats[grp]
                    s["requests"] += 1
                    s["bytes"] += len(line)
                    s["est_input_tokens"] += prompt_tok
                    s["est_input_tokens_high"] += est_in * ratio_high
                    s["est_output_tokens"] += out_point
                    s["out_central"] += output_tokens(est_cfg, ev, "central")
                    s["out_high"] += output_tokens(est_cfg, ev, "high")
                yield BF.PackedRequest(k, line, est_input_tokens=prompt_tok, est_output_tokens=out_point)

    t1 = time.time()
    for i, part in enumerate(BF.chunk(packed(), header, args.max_bytes, args.max_requests)):
        m = BF.write_manifest(rd, f"{args.prefix}{i:05d}", header, part, spec["track"])
        mans.append({k: m[k] for k in ("batch_key", "n_requests", "body_bytes", "est")})
        print(f"[build] {m['batch_key']}: {m['n_requests']} requests, {m['body_bytes'] / 1e6:.1f} MB, est ${m['est']['cost_usd_with_fee']:.2f} "
              f"({time.time() - t1:.0f}s)", flush=True)
    if mismatches:
        print(f"[build] WARNING: {len(mismatches)} texts whose key does not recompute were skipped", flush=True)
    if over_context:
        with open(os.path.join(rd, "over_context.json"), "w") as fh:
            json.dump(over_context, fh)
        raise SystemExit(f"REFUSED: {len(over_context)} requests do not fit the {ctx}-token context with max_tokens {max_tokens} "
                         f"(listed in {rd}/over_context.json); nothing is truncated. Delete the run directory and decide how to judge them.")

    model = spec["model"]

    def cost(s):
        return {"est_cost_usd": round(BF.estimate_cost(model, s["est_input_tokens"], s["est_output_tokens"], with_fee=False), 2),
                "est_cost_usd_with_fee": round(BF.estimate_cost(model, s["est_input_tokens"], s["est_output_tokens"]), 2)}

    measured = "parse_error_rate" in est_cfg          # an entry measured on its provider: the dry run prices a band
    bands: Dict[str, Any] = {}
    if measured:
        for g, s in stats.items():
            if s.get("requests"):
                bands[g] = cost_bands(model, est_cfg, s["requests"], s["est_input_tokens"], s["out_central"], s["out_high"], max_tokens,
                                      prompt_tokens_high=s["est_input_tokens_high"])
    internal = ("out_central", "out_high", "est_input_tokens_high")
    rep: Dict[str, Any] = {
        "run_dir": rd, "built_utc": BF.utc_now(), "judge": args.judge, "model": model, "endpoint": spec["endpoint"], "request": spec["request"],
        "units": sum(unit_counts.values()), "units_by_category_campaign": {f"{a}/{b}": n for (a, b), n in sorted(unit_counts.items())},
        "batches": len(mans), "max_bytes": args.max_bytes, "max_requests": args.max_requests, "key_mismatches": len(mismatches),
        "groups": {g: dict({k: (round(v) if isinstance(v, float) else v) for k, v in s.items() if k not in internal}, **(cost(s) if s.get("requests") else {}))
                   for g, s in sorted(stats.items())},
        "largest_batch_bytes": max((m["body_bytes"] for m in mans), default=0),
        "largest_batch_requests": max((m["n_requests"] for m in mans), default=0), "manifests": mans,
        "assumptions": {"est_output_tokens_per_call": out_point, "est_overhead_tokens_per_call": overhead,
                        "input_tokens": (f"exact: the {tok_cfg.get('hf_repo')} tokenizer.json (sha256 {tok_cfg.get('sha256', '')[:12]}) count of the "
                                         f"prompt + {overhead:.0f} chat/tool overhead tokens (what the provider billed on every reply of the judge comparison)")
                        if tokenizer_path else f"o200k_base count of the prompt text (tiktoken) + {overhead:.0f}, x {ratio}",
                        "reasoning_tokens": 0,
                        "cache": ("implicit caching disabled (prompt_cache_options explicit, no breakpoint)" if spec["request"].get("disable_cache")
                                  else "the provider's implicit prefix cache (none in the point estimate; the band's low and central count it)"),
                        "prices_per_mtok": BF.PRICES_PER_MTOK[model], "credit_fee": BF.CREDIT_FEE}}
    if measured:
        rep["bands"] = bands
        rep["bands_assumptions"] = {"estimate": est_cfg, "doc": cost_bands.__doc__}
        rep["turnaround"] = turnaround_text(entry, len(mans), int(stats["all"].get("requests", 0)))
    if ctx:
        rep["context"] = {"context_tokens": ctx, "max_tokens": max_tokens, "largest_prompt_tokens": round(largest_prompt[0]),
                          "largest_prompt_key": largest_prompt[1], "over_context": len(over_context),
                          "counting": "exact tokenizer count" if tokenizer_path else f"o200k x {ratio_high} (no tokenizer given)"}
    if reuse_infos:
        rep["reuse"] = {"dirs": reuse_infos, "reused": len(reused), "reused_outcomes": dict(collections.Counter(reused.values())),
                        "body_mismatch": len(reuse_mismatch), "body_mismatch_keys": reuse_mismatch[:50],
                        "candidates_not_in_scope": len(set(reuse_outcome) - set(reused) - set(reuse_mismatch))}
        with open(os.path.join(rd, "reuse.json"), "w") as fh:
            json.dump(dict(rep["reuse"], keys=sorted(reused)), fh, indent=1)
    if args.max_spend_x_high and bands.get("all"):
        # rounded UP to the next $10 (never 0, which would mean no ceiling at all)
        spec["max_spend_usd"] = float(10 * math.ceil(args.max_spend_x_high * bands["all"]["high_usd_with_fee"] / 10.0))
        spec["max_spend_note"] = f"{args.max_spend_x_high} x the dry run's high end ${bands['all']['high_usd_with_fee']} (USD incl. the credit fee)"
        BF.write_track(rd, spec)
    if spec.get("max_spend_usd"):
        rep["max_spend_usd"] = spec["max_spend_usd"]
    with open(os.path.join(rd, "dry_run.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(os.path.join(rd, "dry_run.md"), "w") as fh:
        fh.write(render_md(rep))
    a = rep["groups"].get("all", {})
    print(json.dumps({"requests": a.get("requests"), "batches": len(mans), "bytes": a.get("bytes"), "est_input_tokens": a.get("est_input_tokens"),
                      "est_cost_usd": a.get("est_cost_usd"), "est_cost_usd_with_fee": a.get("est_cost_usd_with_fee"), "units": rep["units"],
                      "bands": bands.get("all"), "reused": len(reused), "max_spend_usd": spec.get("max_spend_usd")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
