"""Submit every pending batch of a run directory UP FRONT, or under the track's ramp (stdlib only).

The Batch API runs batches concurrently server-side, so a driver that submits one batch, waits for it and only then submits the next
throws the parallelism away. :func:`submit_pending` registers every manifest in the ledger and submits every pending one in one pass
(a track with a ``ramp`` block holds its first pass to a canary, then to a limit its measured turnaround sets: THE RAMP, below);
the harvester calls it on every cycle, so anything refused for a reason that clears (a 429 in-progress limit, a 402 balance refusal)
goes out on a later cycle without anyone resubmitting by hand.

Refusals, by kind (:func:`openrouter.classify_error`):
  too_large     413 -> the manifest is SPLIT (into pieces of at most the limit the message states, else halves) and the children
                registered; the parent becomes ``split``. A single request over the cap is refused by name.
  rate_limited  429 -> the batch stays ``pending`` with ``next_attempt_ts`` = now + backoff (``Retry-After`` when sent, else 60 s
                doubling to 15 min); this pass stops submitting (the limit is per account, so the rest would 429 too).
  balance       402 / 403 naming a limit -> ``balance_queued``, retried with backoff (5 min doubling to 1 h); this pass stops; an event
                ``BALANCE`` names the message so progress.json reports it. Nothing is dropped.
  transient     5xx / timeout / connection -> ``submit_unknown``: the POST may have been accepted, so the batch is NOT resubmitted at
                once; :func:`reconcile_unknown` adopts an account batch created in the window with the same model and request count, and
                only after ``unknown_resubmit_s`` with no candidate does the batch go back to ``pending``.
  everything else  401 / 403 (the key, the account, a block), 404 (no eligible batch endpoint), 400 / 422 (a body the API rejects),
                a 403 moderation of the whole batch -> the run PAUSES (``<run_dir>/PAUSED``, the reason named) and this pass STOPS. The
                batch stays ``pending`` with the error recorded. These refusals do not clear by themselves and every batch of a run
                shares the account and the request shape, so re-POSTing each of them (79 batches, 14 GB on the full run) into the same
                refusal would only repeat it; a person fixes the cause and removes ``PAUSED``.
  A batch that holds ONE request and is still too large becomes ``refused`` (never truncated) and the run pauses.

Gates, re-checked before EACH POST (each stops the pass, so nothing is submitted past it; removing the GO file or writing ``PAUSED``
while a pass runs stops it before its next POST):
  PAUSED                      a person's or the harvester's pause (``<run_dir>/PAUSED``)
  submit_gate.requires_files  every listed file must exist (``track.json``; the full run requires the pilot's analysed report)
  max_spend_usd               :func:`committed_spend` + the estimate of the batch about to go must stay under it (``track.json``; USD
                              incl. the 5.5% fee)
  ramp                        (``track.json`` ``ramp``, optional) the first-pass in-flight brake: :func:`ramp_status`

A POST is recorded BEFORE it is made: the batch is ``submit_unknown`` (``next_attempt_ts`` = the moment of the POST) while the bytes
go out, and only the reply moves it to ``submitted`` or back to ``pending``. A process killed mid-POST (a hard stop, a crash, a
reboot) therefore leaves a batch the next run reconciles -- adopted when the account shows it, re-queued only after
``unknown_resubmit_s`` with no candidate -- never one the ledger forgot and a second run would POST again.

THE RAMP (``track.json`` ``ramp``; absent = every pending batch goes out in one pass, as before). First-pass batches (``attempt`` 0,
413 children included) are held back so a run commits its money only as evidence arrives; re-ask batches are never held:
  canary   until ``canary_batches`` first-pass batches (the SMALLEST first: the quickest verdict on cost and parse rate) are harvested
           and ``canary_min_settled_share`` of their requests have an item-level outcome, no other first-pass batch is submitted.
           Then the canary gate (once, recorded as an event): the share of verdicts among the requests' FIRST item-level outcomes must
           be at least ``canary_min_verdict_share``, and the run's spend projected at the canary's measured cost / estimate ratio
           (x ``canary_cost_margin`` for re-asks) must fit under ``max_spend_usd``. A failed gate PAUSES the run with the numbers;
           removing PAUSED accepts the canary and the ramp goes on.
  ramp     after the canary the first-pass work in flight (requests of every submitted / unknown batch) is held under a limit derived
           from the turnaround of the latest ``history`` harvested first-pass batches: each one, with turnaround t and peak work in
           flight Q during its life, allows Q x clamp(target_share x window_hours / t, min_growth, max_growth). The smallest of
           them is the limit. Under a provider that runs batches one after another the turnaround of a new batch then stays near
           ``target_share`` of the 24-hour window (no batch expires waiting); under one that runs them side by side the limit grows
           by up to ``max_growth`` per round. One first-pass batch may always go when none is in flight (largest first after the canary).
  ``<run_dir>/INFLIGHT_LIMIT`` (a person's override, read every pass): an integer number of requests replaces the computed limit
  (and the canary's hold); ``0`` holds every new first-pass batch.
"""
from __future__ import annotations

import calendar
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import batchfile as BF
from .openrouter import BatchClient, OpenRouterError, limit_from_message
from .state import Ledger, pause, paused, utc_now

DEFAULTS = {"submit_spacing_s": 2.0, "unknown_resubmit_s": 1800.0, "max_inflight_batches": 0, "max_spend_usd": 0.0}
RAMP_DEFAULTS = {"canary_batches": 2, "canary_min_settled_share": 0.9, "canary_min_verdict_share": 0.85, "canary_cost_margin": 1.08,
                 "window_hours": 24.0, "target_share": 0.6, "min_growth": 0.5, "max_growth": 4.0, "history": 3}
#: the outcomes that say something about the request itself (the canary's verdict share is read over these)
ITEM_OUTCOMES = ("verdict", "moderation", "judge_refusal", "parse_error", "truncated", "provider_error", "missing")
INFLIGHT_FILE = "INFLIGHT_LIMIT"
CANARY_EVENTS = ("CANARY_PASSED", "CANARY_FAILED", "CANARY_ACCEPTED")


def cost_ratio(ledger: Ledger) -> Optional[float]:
    """Reported cost x (1 + fee) over the manifest estimate, across harvested batches that report a cost (None before any does)."""
    r = ledger.conn.execute("SELECT COALESCE(SUM(cost), 0), COALESCE(SUM(est_cost), 0), COUNT(*) FROM batches "
                            "WHERE state = 'harvested' AND cost IS NOT NULL AND est_cost > 0").fetchone()
    if not r[2] or float(r[1]) <= 0:
        return None
    return float(r[0]) * (1.0 + BF.CREDIT_FEE) / float(r[1])


def committed_spend(ledger: Ledger) -> float:
    """USD incl. the credit fee already spent or committed: the reported ``usage.cost`` of harvested batches x (1 + fee); the manifest
    estimate of a harvested batch that reported NO cost (an expired or failed batch whose billing the API does not state: the safe
    side); and the estimates of the batches in flight or of unknown fate, scaled up by the measured :func:`cost_ratio` when real
    costs run above the estimates (never scaled down)."""
    c = ledger.conn
    reported = float(c.execute("SELECT COALESCE(SUM(cost), 0) FROM batches WHERE state = 'harvested' AND cost IS NOT NULL").fetchone()[0])
    unreported = float(c.execute("SELECT COALESCE(SUM(est_cost), 0) FROM batches WHERE state = 'harvested' AND cost IS NULL").fetchone()[0])
    inflight = float(c.execute("SELECT COALESCE(SUM(est_cost), 0) FROM batches WHERE state IN ('submitted', 'submit_unknown')").fetchone()[0])
    ratio = cost_ratio(ledger)
    return reported * (1.0 + BF.CREDIT_FEE) + unreported + inflight * max(1.0, ratio or 1.0)


# -- the ramp ------------------------------------------------------------------------------------------------------------------------
def _ts(s: Any) -> Optional[float]:
    if not s:
        return None
    try:
        return float(calendar.timegm(time.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


def ramp_config(track: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    r = track.get("ramp")
    return dict(RAMP_DEFAULTS, **r) if isinstance(r, Mapping) and r else None


def inflight_override(run_dir: str) -> Optional[int]:
    """The integer in ``<run_dir>/INFLIGHT_LIMIT`` (a person's override of the ramp's limit, in requests), else None."""
    path = os.path.join(run_dir, INFLIGHT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return max(0, int(fh.read().split("#", 1)[0].strip()))
    except (OSError, ValueError):
        return None


def peak_load(intervals: Sequence[Tuple[float, Optional[float], int]], lo: float, hi: float, now: float) -> int:
    """The largest number of requests in flight at any moment of [lo, hi]: ``intervals`` are (submitted, harvested or None, requests)."""
    points = {lo} | {s for s, _h, _n in intervals if lo <= s < hi}
    best = 0
    for p in points:
        load = sum(n for s, h, n in intervals if s <= p < (h if h is not None else now))
        best = max(best, load)
    return best


def growth_limit(ramp: Mapping[str, Any], intervals: Sequence[Tuple[float, Optional[float], int]],
                 recent: Sequence[Tuple[float, float, int]], now: float) -> Tuple[int, List[Dict[str, Any]]]:
    """The ramp's limit (requests in flight) from the latest harvested first-pass batches ``recent`` = [(submitted, harvested,
    requests)]: the smallest of Q x clamp(target_share x window / t, min_growth, max_growth) over them, Q the peak work in flight
    during the batch's life (at least its own requests), t its turnaround (at least a minute) (module docstring)."""
    target = float(ramp["target_share"]) * float(ramp["window_hours"]) * 3600.0
    out = []
    for s, h, n in recent:
        t = max(60.0, h - s)
        q = max(int(n), peak_load(intervals, s, h, now))
        g = min(float(ramp["max_growth"]), max(float(ramp["min_growth"]), target / t))
        out.append({"turnaround_h": round(t / 3600.0, 3), "peak_inflight_requests": q, "growth": round(g, 3), "limit": int(q * g)})
    return (min(o["limit"] for o in out) if out else 0), out


def _first_item_outcomes(ledger: Ledger, batch_keys: Sequence[str]) -> Dict[str, str]:
    """custom_id -> its first item-level outcome, for every request that had an attempt in one of ``batch_keys``."""
    if not batch_keys:
        return {}
    ph = ",".join("?" * len(batch_keys))
    ip = ",".join("?" * len(ITEM_OUTCOMES))
    q = (f"SELECT a.custom_id, a.outcome, a.attempt FROM attempts a WHERE a.custom_id IN (SELECT custom_id FROM attempts WHERE batch_key IN ({ph})) "
         f"AND a.outcome IN ({ip}) ORDER BY a.custom_id, a.attempt")
    out: Dict[str, str] = {}
    for cid, o, _a in ledger.conn.execute(q, (*batch_keys, *ITEM_OUTCOMES)):
        out.setdefault(cid, o)
    return out


def ramp_status(run_dir: str, ledger: Ledger, track: Mapping[str, Any], act: bool = False, now: Optional[float] = None,
                log: Callable[[str], None] = print) -> Optional[Dict[str, Any]]:
    """Where the ramp stands (None without a ``ramp`` block). ``phase``: ``canary`` (only the canary's first-pass batches may go),
    ``failed`` (the canary gate failed and the run waits on a person), ``ramp`` (``limit_requests`` of work may be in flight). With
    ``act`` (the writer, before a submit pass) the canary gate is evaluated once its batches have settled, its verdict recorded as
    an event (``CANARY_PASSED`` / ``CANARY_FAILED``, the numbers in the message) and a failure PAUSES the run; a failure whose
    PAUSED a person removed is recorded as ``CANARY_ACCEPTED``."""
    ramp = ramp_config(track)
    if ramp is None:
        return None
    now = time.time() if now is None else now
    c = ledger.conn
    rows = c.execute("SELECT batch_key, attempt, state, batch_id, n_requests, est_cost, cost, remote_status, submitted_utc, harvested_utc "
                     "FROM batches").fetchall()
    first = [r for r in rows if int(r["attempt"]) == 0 and r["state"] not in ("split", "refused")]
    sent = sorted([r for r in first if r["batch_id"]], key=lambda r: (_ts(r["submitted_utc"]) or 0.0, r["batch_key"]))
    n_canary = max(1, int(ramp["canary_batches"]))
    canary = sent[:n_canary]
    need = min(n_canary, len(first))
    inflight_req = sum(int(r["n_requests"]) for r in rows if r["state"] in ("submitted", "submit_unknown"))
    fp_unknown = sum(1 for r in first if r["state"] == "submit_unknown")
    events = {str(r[0]): str(r[1]) for r in c.execute(
        f"SELECT kind, message FROM events WHERE kind IN ({','.join('?' * len(CANARY_EVENTS))}) ORDER BY rowid", CANARY_EVENTS)}
    out: Dict[str, Any] = {"canary_batches": n_canary, "canary_keys": [r["batch_key"] for r in canary], "inflight_requests": inflight_req,
                           "first_pass_sent": len(sent) + fp_unknown, "first_pass_total": len(first), "override": inflight_override(run_dir)}
    gate = None
    if "CANARY_PASSED" in events or "CANARY_ACCEPTED" in events:
        gate = "passed" if "CANARY_PASSED" in events else "accepted"
    elif "CANARY_FAILED" in events:
        if paused(run_dir):
            gate = "failed"
        else:
            gate = "accepted"
            if act:
                ledger.event("CANARY_ACCEPTED", "PAUSED was removed after the canary gate failed: the canary is accepted and the ramp goes on")
                log("[ramp] canary accepted by a person (PAUSED removed after a failed gate)")
    else:
        keys = [r["batch_key"] for r in canary]
        firsts = _first_item_outcomes(ledger, keys)
        n_req = sum(int(r["n_requests"]) for r in canary)
        settled = len(firsts) / n_req if n_req else 0.0
        complete = len(canary) >= need and need > 0 and all(r["state"] == "harvested" for r in canary) and settled >= float(ramp["canary_min_settled_share"])
        out["canary"] = {"requests": n_req, "settled_share": round(settled, 4), "complete": complete}
        if complete:
            n_v = sum(1 for o in firsts.values() if o == "verdict")
            vshare = n_v / len(firsts) if firsts else 0.0
            with_cost = [r for r in canary if r["cost"] is not None and (r["est_cost"] or 0) > 0]
            ratio = (sum(float(r["cost"]) for r in with_cost) * (1.0 + BF.CREDIT_FEE) / sum(float(r["est_cost"]) for r in with_cost)) if with_cost else None
            spent = float(c.execute("SELECT COALESCE(SUM(cost), 0) FROM batches WHERE state = 'harvested' AND cost IS NOT NULL").fetchone()[0]) * (1.0 + BF.CREDIT_FEE)
            spent += float(c.execute("SELECT COALESCE(SUM(est_cost), 0) FROM batches WHERE state = 'harvested' AND cost IS NULL").fetchone()[0])
            rest = float(c.execute("SELECT COALESCE(SUM(est_cost), 0) FROM batches WHERE state IN ('pending', 'balance_queued', 'submitted', 'submit_unknown')").fetchone()[0])
            projected = spent + rest * (ratio or 0.0) * float(ramp["canary_cost_margin"])
            cap = float(track.get("max_spend_usd") or 0.0)
            why = []
            if vshare < float(ramp["canary_min_verdict_share"]):
                why.append(f"verdicts are {vshare:.1%} of the canary's first item-level outcomes, under {float(ramp['canary_min_verdict_share']):.0%}")
            if ratio is None:
                why.append("no canary batch reported a cost, so the projection cannot be made")
            elif cap > 0 and projected > cap:
                why.append(f"at the measured cost / estimate ratio {ratio:.3f} (x {float(ramp['canary_cost_margin'])} for re-asks) the run "
                           f"projects to ${projected:,.2f}, over max_spend_usd ${cap:,.2f}")
            counts: Dict[str, int] = {}
            for o in firsts.values():
                counts[o] = counts.get(o, 0) + 1
            metrics = {"batches": keys, "requests": n_req, "first_outcomes": counts, "verdict_share": round(vshare, 4),
                       "cost_ratio": None if ratio is None else round(ratio, 4), "projected_usd": round(projected, 2), "max_spend_usd": cap}
            out["canary"].update(metrics)
            gate = "failed" if why else "passed"
            if act:
                if why:
                    ledger.event("CANARY_FAILED", json.dumps(dict(metrics, why=why)))
                    pause(run_dir, ledger, "canary gate: " + "; ".join(why) + f" ({json.dumps(metrics)}). Read the canary's results; "
                          "remove PAUSED to accept it and let the ramp go on, or leave the run stopped", log)
                else:
                    ledger.event("CANARY_PASSED", json.dumps(metrics))
                    log(f"[ramp] canary passed: {json.dumps(metrics)}")
    if gate is None:
        out["phase"] = "canary"
    elif gate == "failed":
        out["phase"] = "failed"
    else:
        out["phase"] = "ramp"
        out["canary_gate"] = gate
        # (submitted, harvested or None, requests); a batch harvested within the second it went out still spans one second
        intervals = [(s0, None if h0 is None else max(h0, s0 + 1.0), int(r["n_requests"])) for r, s0, h0 in
                     ((r, _ts(r["submitted_utc"]), _ts(r["harvested_utc"])) for r in rows) if s0 is not None]
        recent = sorted([(s0, max(h0, s0 + 1.0), int(r["n_requests"])) for r, s0, h0 in
                         ((r, _ts(r["submitted_utc"]), _ts(r["harvested_utc"])) for r in first if r["state"] == "harvested")
                         if s0 is not None and h0 is not None], key=lambda x: x[1])[-int(ramp["history"]):]
        limit, hist = growth_limit(ramp, intervals, recent, now)
        out["limit_requests"] = limit
        out["history"] = hist
    if out["override"] is not None and out["phase"] != "failed":
        out["limit_requests"] = out["override"]
    return out


def gate_reason(run_dir: str, track: Mapping[str, Any]) -> Optional[str]:
    """Why this run may not submit yet (``track.json`` ``submit_gate.requires_files``), else None."""
    gate = track.get("submit_gate") or {}
    missing = [f for f in (gate.get("requires_files") or []) if not os.path.exists(os.path.expanduser(str(f)))]
    if missing:
        return f"submit gate: {gate.get('note') or 'required files are missing'}; missing {missing}"
    return None


def register_all(run_dir: str, ledger: Ledger) -> int:
    n = 0
    for key in BF.list_manifests(run_dir):
        if ledger.conn.execute("SELECT 1 FROM batches WHERE batch_key = ?", (key,)).fetchone():
            continue
        if ledger.register_manifest(BF.read_manifest(run_dir, key)):
            n += 1
    return n


def split_manifest(run_dir: str, ledger: Ledger, man: Mapping[str, Any], limit: Optional[int]) -> List[str]:
    """Replace a 413-refused manifest by children of at most ``limit`` requests (halves when no limit is known or it is not smaller)."""
    body = BF.read_body(run_dir, man)
    reqs = BF.body_requests(body)
    n = len(reqs)
    if n <= 1:
        raise ValueError(f"batch {man['batch_key']} holds one request and is still too large; refusing (never truncated)")
    size = limit if (limit and limit < n) else math.ceil(n / 2)
    header = BF.batch_header(str(man["model"]), str(man["endpoint"]), man.get("provider"))
    keys = []
    by_id = {r["custom_id"]: r for r in reqs}
    order = [r["custom_id"] for r in reqs]
    # each child carries its share of the parent's estimate (input by bytes, output by requests), so the spend ceiling keeps seeing it
    est = man.get("est") or {}
    in_per_byte = float(est.get("input_tokens") or 0) / max(1, int(man["body_bytes"]))
    out_per_req = float(est.get("output_tokens") or 0) / max(1, n)
    for i in range(0, n, size):
        part = []
        for cid in order[i:i + size]:
            line = BF.serialize_request(cid, by_id[cid]["body"])
            part.append(BF.PackedRequest(cid, line, est_input_tokens=len(line) * in_per_byte, est_output_tokens=out_per_req))
        key = f"{man['batch_key']}.s{i // size}"
        child = BF.write_manifest(run_dir, key, header, part, str(man.get("track")), extra=man.get("extra"),
                                  attempt=int(man.get("attempt") or 0), parent=str(man["batch_key"]))
        ledger.register_manifest(child)
        keys.append(key)
    ledger.update_batch(str(man["batch_key"]), state="split", last_error_kind="too_large")
    ledger.event("SPLIT", f"{man['batch_key']} ({n} requests) -> {len(keys)} children of <= {size}", str(man["batch_key"]))
    return keys


def _backoff(base: float, k: int, cap: float) -> float:
    return min(cap, base * (2 ** max(0, k)))


def submit_pending(run_dir: str, ledger: Ledger, client: Optional[BatchClient], track: Mapping[str, Any], dry_run: bool = False,
                   log: Callable[[str], None] = print, max_submits: Optional[int] = None) -> Dict[str, Any]:
    """One pass: register every manifest, then submit each pending batch whose ``next_attempt_ts`` has come, within the gates and the
    ramp (module docstring). Returns a summary."""
    cfg = dict(DEFAULTS, **{k: track[k] for k in DEFAULTS if k in track})
    register_all(run_dir, ledger)
    now = time.time()
    due = [b for b in ledger.batches(("pending", "balance_queued")) if b["next_attempt_ts"] <= now]
    rs = ramp_status(run_dir, ledger, track, act=not dry_run, log=log)
    if rs is not None:
        # re-asks first (they finish requests already started; the ramp never holds them), then the first pass: the canary smallest
        # first, after it largest first (the longest batches start earliest)
        retries = [b for b in due if int(b["attempt"]) > 0]
        firsts = sorted((b for b in due if int(b["attempt"]) == 0), key=lambda b: (int(b["n_requests"]), b["batch_key"]),
                        reverse=rs["phase"] != "canary")
        due = retries + firsts
    inflight = len(ledger.batches(("submitted", "submit_unknown")))
    summary: Dict[str, Any] = {"due": len(due), "submitted": 0, "split": 0, "stopped": None, "dry_run": dry_run, "ramp": rs}
    if dry_run:
        tot_req = sum(b["n_requests"] for b in due)
        tot_bytes = sum(b["body_bytes"] for b in due)
        tot_cost = sum((b["est_cost"] or 0.0) for b in due)
        summary.update(would_submit=len(due), requests=tot_req, bytes=tot_bytes, est_cost_usd_with_fee=round(tot_cost, 4))
        if rs is not None and rs["phase"] == "canary":
            first = [b for b in due if int(b["attempt"]) == 0][:max(0, rs["canary_batches"] - rs["first_pass_sent"])]
            summary["canary_now"] = {"batches": [b["batch_key"] for b in first], "requests": sum(b["n_requests"] for b in first),
                                     "est_cost_usd_with_fee": round(sum((b["est_cost"] or 0.0) for b in first), 4)}
        log(f"[submit] DRY RUN: would submit {len(due)} batches, {tot_req} requests, {tot_bytes / 1e6:.1f} MB, est ${tot_cost:.2f} incl. fee"
            + (f"; the ramp sends the canary first: {summary['canary_now']}" if summary.get("canary_now") else ""))
        return summary
    assert client is not None
    if due and paused(run_dir):
        summary["stopped"] = "paused"
        return summary
    if due:
        why = gate_reason(run_dir, track)
        if why:
            pause(run_dir, ledger, why, log)
            summary["stopped"] = "gated"
            return summary
    inflight_req = rs["inflight_requests"] if rs is not None else 0
    fp_sent = rs["first_pass_sent"] if rs is not None else 0
    fp_inflight = len([b for b in ledger.batches(("submitted", "submit_unknown")) if int(b["attempt"]) == 0])
    for b in due:
        if max_submits is not None and summary["submitted"] >= max_submits:
            break
        if cfg["max_inflight_batches"] and inflight >= int(cfg["max_inflight_batches"]):
            summary["stopped"] = "max_inflight_batches"
            break
        key = b["batch_key"]
        n_b = int(b["n_requests"])
        if rs is not None and int(b["attempt"]) == 0:
            # the ramp holds this first-pass batch (it stays pending; the next pass looks again)
            if rs["phase"] == "failed":
                summary["stopped"] = "canary_failed"
                continue
            limit = rs.get("limit_requests")
            if limit is None:                                   # the canary, no override
                if fp_sent >= int(rs["canary_batches"]):
                    summary["stopped"] = "ramp_canary"
                    continue
            elif (rs.get("override") == 0) or (fp_inflight > 0 and inflight_req + n_b > int(limit)):
                summary["stopped"] = "ramp_limit"           # a person's INFLIGHT_LIMIT 0, or the limit reached with work in flight
                continue
        # the gates, re-checked before EVERY post: a PAUSED written or a GO file removed while this pass runs stops it here
        p = paused(run_dir)
        if p:
            summary["stopped"] = "paused"
            log(f"[submit] paused while the pass ran ({p.splitlines()[0][:200]}); stopping before {key}")
            break
        why = gate_reason(run_dir, track)
        if why:
            pause(run_dir, ledger, why, log)
            summary["stopped"] = "gated"
            break
        if float(cfg["max_spend_usd"] or 0) > 0:
            committed = committed_spend(ledger)
            if committed + float(b["est_cost"] or 0.0) > float(cfg["max_spend_usd"]):
                pause(run_dir, ledger, f"spend ceiling: ${committed:.2f} spent or committed + ${float(b['est_cost'] or 0):.2f} for {key} "
                                       f"would pass max_spend_usd ${float(cfg['max_spend_usd']):.2f} (track.json); raise it to go on", log)
                summary["stopped"] = "spend_ceiling"
                break
        man = BF.read_manifest(run_dir, key)
        body = BF.read_body(run_dir, man)
        t_submit = time.time()
        k = int(b["submit_attempts"]) + 1
        # recorded BEFORE the bytes go out: if this process dies mid-POST the batch is reconciled, never forgotten and POSTed again
        ledger.update_batch(key, state="submit_unknown", submit_attempts=k, next_attempt_ts=t_submit, last_error_kind="post_started",
                            last_error="POST started; still here means the process ended before the reply: reconcile_unknown adopts the "
                                       "account batch or re-queues this one")
        try:
            resp = client.submit(body)
        except OpenRouterError as e:
            if e.kind == "too_large":
                ledger.update_batch(key, state="pending", last_error_kind=e.kind, last_error=e.message)
                limit = limit_from_message(e.message)
                try:
                    children = split_manifest(run_dir, ledger, man, limit)
                except ValueError as ve:
                    ledger.update_batch(key, state="refused", last_error_kind="too_large_single", last_error=str(ve))
                    ledger.event("REFUSED", str(ve), key)
                    pause(run_dir, ledger, f"{key}: {ve}", log)
                    summary["stopped"] = "refused"
                    break
                summary["split"] += 1
                log(f"[submit] 413 {key}: {e.message!r} -> split into {len(children)}")
                continue
            if e.kind == "rate_limited":
                wait = e.retry_after_s or _backoff(60.0, k - 1, 900.0)
                ledger.update_batch(key, state="pending", last_error_kind=e.kind, last_error=e.message, next_attempt_ts=time.time() + wait)
                ledger.event("RATE_LIMITED", f"429 on submit: {e.message}; next try in {wait:.0f}s", key)
                summary["stopped"] = "rate_limited"
                log(f"[submit] 429 {key}: {e.message!r}; backing off {wait:.0f}s and stopping this pass")
                break
            if e.kind == "balance":
                wait = _backoff(300.0, k - 1, 3600.0)
                ledger.update_batch(key, state="balance_queued", last_error_kind=e.kind, last_error=e.message, next_attempt_ts=time.time() + wait)
                ledger.event("BALANCE", f"HTTP {e.status} on submit (balance / spending limit): {e.message}; queued, next try in {wait:.0f}s", key)
                summary["stopped"] = "balance"
                log(f"[submit] BALANCE {key}: HTTP {e.status} {e.message!r}; queued, retry in {wait:.0f}s; stopping this pass")
                break
            if e.kind == "transient":
                ledger.update_batch(key, state="submit_unknown", last_error_kind=e.kind, last_error=e.message, next_attempt_ts=t_submit)
                ledger.event("SUBMIT_UNKNOWN", f"transient error on submit ({e.message}); the POST may have been accepted", key)
                log(f"[submit] transient {key}: {e.message!r}; marked submit_unknown (not resubmitted until reconciled)")
                inflight += 1
                inflight_req += n_b
                if int(b["attempt"]) == 0:
                    fp_sent += 1
                    fp_inflight += 1
                continue
            # auth / account / block / no endpoint / a body the API rejects: none of these clears by itself, and the next batch shares
            # the account and the request shape -- pause the run and stop the pass (the batch back to pending, the error kept)
            ledger.update_batch(key, state="pending", last_error_kind=e.kind, last_error=e.message)
            ledger.event("SUBMIT_REFUSED", f"HTTP {e.status} {e.kind}: {e.message}", key)
            pause(run_dir, ledger, f"submit of {key} refused, HTTP {e.status} {e.kind}: {e.message[:500]}", log)
            summary["stopped"] = e.kind
            log(f"[submit] HTTP {e.status} {e.kind} on {key}: {e.message!r}; run paused, pass stopped")
            break
        ledger.update_batch(key, state="submitted", batch_id=str(resp.get("id")), submitted_utc=utc_now(), provider=resp.get("provider"),
                            remote_status=resp.get("status"), last_error_kind=None, last_error=None)
        ledger.event("SUBMITTED", f"{resp.get('id')} {man['n_requests']} requests {man['body_bytes']} bytes provider={resp.get('provider')}", key)
        summary["submitted"] += 1
        inflight += 1
        inflight_req += n_b
        if int(b["attempt"]) == 0:
            fp_sent += 1
            fp_inflight += 1
        log(f"[submit] {key} -> {resp.get('id')} ({man['n_requests']} requests, {man['body_bytes'] / 1e6:.1f} MB, status {resp.get('status')})")
        time.sleep(float(cfg["submit_spacing_s"]))
    return summary


def reconcile_unknown(run_dir: str, ledger: Ledger, client: BatchClient, track: Mapping[str, Any], log: Callable[[str], None] = print) -> int:
    """Adopt, for each ``submit_unknown`` batch, the one account batch created within 15 minutes of the failed POST with the same model
    slug prefix and request count that no ledger row names; back to ``pending`` after ``unknown_resubmit_s`` with no candidate."""
    unknown = ledger.batches(("submit_unknown",))
    if not unknown:
        return 0
    cfg = dict(DEFAULTS, **{k: track[k] for k in DEFAULTS if k in track})
    known_ids = {r["batch_id"] for r in ledger.batches() if r["batch_id"]}
    t_min = min(b["next_attempt_ts"] for b in unknown) - 60
    cands = [x for x in client.iter_batches(created_after=str(int(t_min))) if x.get("id") not in known_ids]
    adopted = 0
    for b in unknown:
        man = BF.read_manifest(run_dir, b["batch_key"])
        model = str(man["model"])
        t0 = float(b["next_attempt_ts"])
        hits = [x for x in cands if str(x.get("model", "")).startswith(model) and int((x.get("request_counts") or {}).get("total") or -1) == int(man["n_requests"])
                and abs(float(x.get("created_at") or 0) - t0) < 900]
        if len(hits) == 1:
            x = hits[0]
            ledger.update_batch(b["batch_key"], state="submitted", batch_id=x["id"], remote_status=x.get("status"), provider=x.get("provider"),
                                submitted_utc=utc_now())
            ledger.event("ADOPTED", f"submit_unknown adopted account batch {x['id']}", b["batch_key"])
            cands.remove(x)
            known_ids.add(x["id"])
            adopted += 1
            log(f"[submit] adopted {x['id']} for {b['batch_key']} after an unknown submit")
        elif not hits and time.time() - t0 > float(cfg["unknown_resubmit_s"]):
            ledger.update_batch(b["batch_key"], state="pending", next_attempt_ts=0)
            ledger.event("UNKNOWN_RESUBMIT", "no account batch matched the unknown submit; back to pending", b["batch_key"])
    return adopted
