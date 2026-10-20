"""Fill the judge cache through the Messages BATCH API (the `batch` org: Batch
API only — this script never calls ``messages.create``).

The live path is ``judging.ClaudeJudge.judge_all``; this is its offline
twin for a cache: the SAME requests (``render_request`` +
``translate_request`` over the SAME ``JudgeRequest`` keys, frozen rubric +
protocol), submitted as one batch per (cell, control) — every batch created
BEFORE any is polled, so wall-clock is one turnaround, not N — then polled
together and harvested into the private cache with ``org_used="batch"``.
Results are parsed by the harness's own ``ClaudeJudge._parse`` (forced-tool
verdict; ``stop_reason=refusal`` -> abstention ``refusal``; ``max_tokens``
without a verdict -> abstention ``max_tokens``); an errored result is
classified like the live path's channels: an ``overloaded_error`` /
``api_error`` while the SAME batch had successes -> ``transport_refusal``
(evidence ``batch_siblings``), a content-filter ``invalid_request_error``
-> ``content_filter``; anything else stays unverdicted and this script exits
non-zero. A malformed verdict is re-asked ONCE (``reask``), like the live
path. After the harvest, ``served_capability_judge.py --no-live`` files the
rows and summaries from the cache with zero live calls.

Org: the client is ``sisteer.vault.anthropic_client("batch")`` — the Batch
org's OWN key, resolved by the vault module (its env var, else its vault item),
so nothing is exported by hand at launch and the shell's ``ANTHROPIC_API_KEY``
(High Prio) is never consulted. See :func:`_client` for why that matters.

  batch_judge_fill.py submit  --controls-root R --cache C --targets T [--match M] --state S.json
  batch_judge_fill.py poll    --state S.json [--state S2.json ...] [--once]
  batch_judge_fill.py harvest --controls-root R --cache C --state S.json [--reask]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # scripts/
import served_capability_judge as sj  # noqa: E402

from sisteer.capeval import corpora, judging  # noqa: E402
from sisteer.capeval.judging import JudgeRequest  # noqa: E402
from sisteer.capeval.tiers import judged  # noqa: E402

ORG = "batch"
_TRANSPORT_ERROR_TYPES = {"overloaded_error", "api_error"}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(msg: str) -> None:
    print(f"[batch-fill {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _client():
    """The Batch org's client, built by ``sisteer.vault`` from ``ORG``'s OWN key
    (its env var, else its vault item) — never the shell's ``ANTHROPIC_API_KEY``,
    which is High Prio. Before this the script used whatever key the launcher
    had exported, so a launch that forgot the batch prefix put Batch traffic on
    a live org (against the pool rule) with nothing in the run saying so; a
    missing batch key now exits, named. Base URL pinned; SDK retries stay at 2
    for the control-plane calls (create / retrieve / results); the per-request
    bound is the judge module's constant, never the SDK's 600 s default."""
    from sisteer import vault

    try:
        return vault.anthropic_client(ORG, max_retries=2, timeout=judging.REQUEST_TIMEOUT_S,
                                      base_url=judging.DEFAULT_JUDGE_BASE_URL)
    except vault.VaultSecretError as e:
        raise SystemExit(f"no key for the {ORG!r} org: {e}") from None


def _cells(args) -> List[sj.Cell]:
    pins = corpora.load_pins()
    rows = {"xstest": judged.xstest_rows(pins), "or_bench_hard_1k": judged.or_bench_rows(pins)}
    root = Path(args.controls_root).expanduser()
    targets = [t for t in args.targets.split(",") if t] if getattr(args, "targets", "") else \
        sorted(p.name for p in root.iterdir() if (p / "cells").is_dir())
    return sj.discover(root, targets, getattr(args, "match", "") or "", rows, list(sj.JUDGED_CONTROLS))


def _requests_for(cell: sj.Cell, control: str) -> Dict[str, Tuple[str, JudgeRequest]]:
    """key -> (item_id, JudgeRequest) for every item of the control."""
    spec = cell.specs[control]
    rubric = sj._rubric(control)
    psha = judging.protocol_sha(rubric, temperature=judging.TEMPERATURE, max_tokens=judging.MAX_TOKENS)
    gens = cell.spools[control].entries()
    out = {}
    for r in spec.rows:
        req = JudgeRequest(prompt=r["prompt"], candidate_a=gens[r["item_id"]], candidate_b=None,
                           roles=None, rubric_sha=rubric.sha, judge_model=cell.judge_model,
                           protocol_sha=psha)
        out[req.key()] = (r["item_id"], req)
    return out


def _params(req: JudgeRequest, control: str) -> Dict[str, Any]:
    rubric = sj._rubric(control)
    return judging.translate_request("anthropic", judging.render_request(
        rubric, req, temperature=judging.TEMPERATURE, max_tokens=judging.MAX_TOKENS))


def cmd_submit(args) -> int:
    cache = judging.JudgeCache(Path(args.cache).expanduser())
    client = _client()
    state = {"submitted_at": _utc(), "cache": str(Path(args.cache).expanduser()), "org": ORG,
             "judge_model": None, "protocol_sha": {}, "batches": {}}
    n_total = 0
    for cell in _cells(args):
        state["judge_model"] = cell.judge_model
        for control in cell.controls:
            reqs = _requests_for(cell, control)
            misses = {k: v for k, v in reqs.items() if cache.get(k) is None}
            state["protocol_sha"][control] = judging.protocol_sha(
                sj._rubric(control), temperature=judging.TEMPERATURE, max_tokens=judging.MAX_TOKENS)
            if not misses:
                _log(f"{cell.target}/{cell.cell_id} {control}: fully cached, no batch")
                continue
            requests = [{"custom_id": k, "params": _params(req, control)} for k, (iid, req) in misses.items()]
            batch = client.messages.batches.create(requests=requests)
            state["batches"][batch.id] = {"target": cell.target, "cell": cell.name, "cell_id": cell.cell_id,
                                          "control": control, "n": len(requests),
                                          "keys": {k: iid for k, (iid, _) in misses.items()},
                                          "created_at": _utc(), "status": batch.processing_status}
            n_total += len(requests)
            _log(f"SUBMITTED {batch.id} {cell.target}/{cell.cell_id} {control}: {len(requests)} requests "
                 f"(of {len(reqs)}; {len(reqs) - len(misses)} cached)")
    Path(args.state).write_text(json.dumps(state, indent=1) + "\n")
    _log(f"{len(state['batches'])} batches, {n_total} requests, state -> {args.state}")
    return 0


def _counts(b) -> Dict[str, int]:
    rc = b.request_counts
    return {k: int(getattr(rc, k, 0) or 0) for k in ("processing", "succeeded", "errored", "canceled", "expired")}


def cmd_poll(args) -> int:
    client = _client()
    states = {p: json.loads(Path(p).read_text()) for p in args.state}
    t0 = time.time()
    while True:
        pending = 0
        for p, st in states.items():
            for bid, meta in st["batches"].items():
                b = client.messages.batches.retrieve(bid)
                meta["status"] = b.processing_status
                meta["request_counts"] = _counts(b)
                if b.processing_status != "ended":
                    pending += 1
            Path(p).write_text(json.dumps(st, indent=1) + "\n")
        line = " | ".join(f"{bid[-8:]}:{m['status']}:{m['request_counts']['succeeded']}/{m['n']}"
                          for st in states.values() for bid, m in st["batches"].items())
        _log(f"poll elapsed={int(time.time() - t0)}s pending_batches={pending} {line}")
        if pending == 0 or args.once:
            return 0 if pending == 0 else 3
        time.sleep(args.interval)


def _put_if_absent(cache: judging.JudgeCache, key: str, verdict: Dict[str, Any], counts: Dict[str, int]) -> bool:
    """Insert-only: a verdict that already landed by another route (a Low-Prio live
    fill that beat the batch) keeps its org and content; the batch result is
    counted as ``already_cached`` and dropped."""
    if cache.get(key) is not None:
        counts["already_cached"] = counts.get("already_cached", 0) + 1
        return False
    cache.put(key, verdict, ORG)
    return True


def cmd_harvest(args) -> int:
    cache = judging.JudgeCache(Path(args.cache).expanduser())
    client = _client()
    st = json.loads(Path(args.state).read_text())
    # a parser instance: _parse touches no client, no network
    parser = judging.ClaudeJudge(cache=cache, clients={"lowprio": object()}, org="lowprio")
    totals: Dict[str, int] = {}
    malformed: Dict[str, Dict[str, str]] = {}
    failures: List[str] = []
    for bid, meta in st["batches"].items():
        if meta.get("harvested"):
            continue
        b = client.messages.batches.retrieve(bid)
        if b.processing_status != "ended":
            _log(f"{bid}: still {b.processing_status}; skipping"); continue
        rubric = sj._rubric(meta["control"])
        n_ok = int(_counts(b)["succeeded"])
        counts: Dict[str, int] = {}
        for item in client.messages.batches.results(bid):
            key = item.custom_id
            res = item.result
            kind = getattr(res, "type", None)
            if kind == "succeeded":
                try:
                    verdict = parser._parse(res.message, rubric)
                    if _put_if_absent(cache, key, verdict, counts):
                        counts["judged"] = counts.get("judged", 0) + 1
                except judging.JudgeRefusal as e:
                    if _put_if_absent(cache, key, {"abstained": True, "reason": e.reason, "batch_id": bid}, counts):
                        counts[e.reason] = counts.get(e.reason, 0) + 1
                except judging.JudgingError as e:
                    malformed[key] = {"batch": bid, "why": str(e)[:80]}
                    counts["malformed"] = counts.get("malformed", 0) + 1
            elif kind == "errored":
                err = getattr(res, "error", None)
                inner = getattr(err, "error", None) or err
                etype = getattr(inner, "type", None); emsg = str(getattr(inner, "message", "") or "")
                if etype in _TRANSPORT_ERROR_TYPES and n_ok > 0:
                    if _put_if_absent(cache, key, {"abstained": True, "reason": "transport_refusal",
                                                   "evidence": "batch_siblings", "siblings_succeeded": n_ok,
                                                   "attempts": 1, "batch_id": bid, "error_type": str(etype)}, counts):
                        counts["transport_refusal"] = counts.get("transport_refusal", 0) + 1
                elif etype == "invalid_request_error" and judging._CONTENT_FILTER_MARK in emsg.lower():
                    if _put_if_absent(cache, key, {"abstained": True, "reason": "content_filter", "batch_id": bid}, counts):
                        counts["content_filter"] = counts.get("content_filter", 0) + 1
                else:
                    failures.append(f"{bid} {meta['keys'].get(key)} errored {etype}: {emsg[:100]}")
                    counts["failed"] = counts.get("failed", 0) + 1
            else:
                failures.append(f"{bid} {meta['keys'].get(key)} result {kind}")
                counts["failed"] = counts.get("failed", 0) + 1
        meta["harvest_counts"] = counts
        meta["harvested"] = (counts.get("malformed", 0) == 0 and counts.get("failed", 0) == 0)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        _log(f"HARVESTED {bid} {meta['target']}/{meta['cell_id']} {meta['control']}: {counts}")
    st["malformed"] = malformed
    st["harvested_at"] = _utc()
    Path(args.state).write_text(json.dumps(st, indent=1) + "\n")
    _log(f"totals: {totals}")
    if malformed and args.reask:
        # ONE bounded re-ask, the live path's discipline: same requests, same keys
        cells = {c.name: c for c in _cells(args)}
        by_bc: Dict[Tuple[str, str], List[str]] = {}
        for k, m in malformed.items():
            meta = st["batches"][m["batch"]]
            by_bc.setdefault((meta["cell"], meta["control"]), []).append(k)
        for (cell_name, control), keys in by_bc.items():
            reqs = _requests_for(cells[cell_name], control)
            batch = client.messages.batches.create(
                requests=[{"custom_id": k, "params": _params(reqs[k][1], control)} for k in keys])
            meta0 = next(m for m in st["batches"].values() if m["cell"] == cell_name and m["control"] == control)
            st["batches"][batch.id] = {**{k: meta0[k] for k in ("target", "cell", "cell_id", "control")},
                                       "n": len(keys), "keys": {k: reqs[k][0] for k in keys},
                                       "created_at": _utc(), "status": batch.processing_status, "reask": True}
            _log(f"RE-ASK SUBMITTED {batch.id} {cell_name} {control}: {len(keys)} malformed verdicts")
        st["malformed"] = {}
        Path(args.state).write_text(json.dumps(st, indent=1) + "\n")
        return 3  # poll + harvest again
    if failures or malformed:
        for f in failures:
            _log(f"UNVERDICTED: {f}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("--controls-root", required=True); s.add_argument("--cache", required=True)
    s.add_argument("--targets", default=""); s.add_argument("--match", default=""); s.add_argument("--state", required=True)
    p = sub.add_parser("poll"); p.add_argument("--state", action="append", required=True)
    p.add_argument("--interval", type=int, default=60); p.add_argument("--once", action="store_true")
    h = sub.add_parser("harvest"); h.add_argument("--controls-root", required=True); h.add_argument("--cache", required=True)
    h.add_argument("--targets", default=""); h.add_argument("--match", default=""); h.add_argument("--state", required=True)
    h.add_argument("--reask", action="store_true")
    args = ap.parse_args()
    # A rejected Batch credential -- a 401 / 403 on a create / retrieve / results
    # call, or under SISTEER_SECRETS_MODE=wif the federation identity refused at the token exchange -- ends the command
    # with the typed message as the terminal line (the daemon's `credential` kind), never a traceback; the calls are
    # synchronous, so the first rejection is the last request, and the state file is written only after a whole pass
    from sisteer import vault
    with vault.credential_exit(ORG, prefix="[batch-fill] "):
        return {"submit": cmd_submit, "poll": cmd_poll, "harvest": cmd_harvest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
