#!/usr/bin/env python3
"""The persistent harvester of a run directory: submit (everything up front), poll, fetch, parse, retry, report (stdlib only).

    python scripts/rejudge/common/harvester.py --run-dir DIR [--once] [--poll-s 60] [--max-hours 72] [--no-submit] [--dry-run]
    python scripts/rejudge/common/harvester.py --run-dir DIR --set-aside BATCH_KEY --reason TEXT     (an operator's decision; no network)

Each cycle:
  1. ``submit_unknown`` batches are reconciled (:func:`submitter.reconcile_unknown`), then every pending batch is submitted
     (:func:`submitter.submit_pending`: split on 413, back off on 429, queue on balance, PAUSE on any other refusal, the submit gate and
     the spend ceiling).
  2. The in-flight batches' statuses are read from the list endpoint (metadata only); a batch the list does not show is read by id.
  3. Every batch in a final status is fetched by id ONCE (results inline), saved verbatim to ``raw/<batch_key>__<batch_id>.json.gz``,
     and each result is parsed by the TRACK'S PARSER (``track.json`` ``parser`` = ``"module.path:function"``, imported with the
     repository's ``scripts/`` on ``sys.path``). A parser maps one result entry to ``{outcome, verdict, error, status_code,
     finish_reason, usage}``; the outcome vocabulary is the track's, with these shared names, in three budget classes:
       ITEM (the request's own content or the judge's handling of it; they spend the request's re-asks)
        verdict         a valid judgment                                    (terminal)
        moderation      the gateway / provider refused THIS input, on positive evidence (``openrouter.classify_result_error``)
                        (terminal; recorded per item)
        judge_refusal   the model declined (a refusal message instead of the forced call)            (terminal)
        parse_error     a call whose arguments are malformed or fail the schema
        truncated       cut at max_tokens: re-asked ONLY with the raised cap ``track.json`` ``truncated_retry_max_tokens``, else
                        terminal at once (re-sending the same cap cannot succeed)
        provider_error  any other error entry
        missing         a request of a completed batch with no result entry       (set here, not by the parser)
       BATCH (the whole batch ended without naming the request; their own re-asks, never the item's)
        batch_failed / batch_expired / batch_cancelled                            (set here)
       NEVER TERMINAL, NEVER COUNTED (not a property of the request)
        account_error   the provider refused the credential or the account (a dead own-provider key, an account block): the run PAUSES
        batch_failed_auth  a whole batch failed on the account / credential: the run PAUSES
        balance_error   the money ran out mid-batch: re-asked after a backoff (5 min doubling to 1 h)
        batch_failed_other the batch failed NAMING ANOTHER request (validation / screening): this one is re-queued
        held            part of an identical-error flood (below): the run PAUSES until a person rules on it
     ``track.json`` ``retry`` maps an outcome to the number of RE-asks it gets (default: provider_error / missing / batch_* 3,
     parse_error / truncated 1, moderation / judge_refusal 0), counted within the outcome's class. A request with re-asks left goes into a
     new retry manifest (attempt + 1) built from its original body; nothing is dropped and every attempt's outcome stays in ``attempts``.
     A batch that FAILED naming one of its custom_ids (OpenRouter's content screening: ``This request contains prohibited content``)
     records that request as ``moderation`` and re-queues the others.
  4. HEALTH, per harvested batch (``track.json`` ``health``; defaults :data:`HEALTH_DEFAULTS`), each a PAUSE that names its reason:
       identical-error flood   one error signature (outcome, kind, status, type, provider code, normalised message) covering at least
                               ``identical_share`` of the batch's results (and ``min_count``) is HELD -- recorded as ``held``, not as
                               its outcome -- unless the signature is acknowledged (``<run_dir>/ACKNOWLEDGED``, one per line, or
                               ``health.acknowledged``). An unrecognised account failure looks exactly like this, so nothing of it
                               becomes terminal before a person looks. Acknowledge a signature that is a real per-item outcome (genuine
                               moderation of many jailbreak transcripts), then remove ``PAUSED``: the held requests are re-asked once
                               and recorded as what they are.
       truncation              ``truncated`` at least ``truncated_share`` of the results: the cap or the reasoning setting is wrong
       errors                  provider_error + missing + parse_error at least ``error_share``
       any account_error / batch_failed_auth
       a whole-batch ending    (``health.pause_on_batch_end``, default off) a batch that ended ``failed`` / ``expired`` / ``cancelled``
                               leaving requests unanswered: they are recorded as the batch outcome (kept, not terminal) and the run
                               PAUSES instead of re-asking them on its own, so an expiry at scale (a provider slower than the 24-hour
                               window) or a batch failure is read by a person before anything is re-sent; removing PAUSED re-asks them
  5. ``progress.json`` is rewritten (batch states, outcome counts, spend, the key's remaining limit, readiness, the ramp, the latest
     events).
The loop ends with a distinct exit code: 0 DONE (every request terminal, no batch live or set aside, not paused), 4 PAUSED with nothing
in flight, 5 STOPPED UNFINISHED (nothing left this harvester can move, but a batch is set aside / refused or a request has no terminal
outcome), 3 at ``--max-hours`` with work left; one cycle and exit 0 with ``--once``. It takes the run directory's writer lock, so two
harvesters never share a run.

The OpenRouter key is read from ``$OPENROUTER_API_KEY`` and never printed, logged or written anywhere. A ``--dry-run`` needs no key. A
long harvester belongs in a process that outlives the shell that started it (a service manager's unit, a terminal multiplexer), with
the key in that process's environment only (README.md, "Credentials").
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import submitter as SUB  # noqa: E402
from rejudge.common.openrouter import FINAL_STATUSES, BatchClient, OpenRouterError, is_account_failure  # noqa: E402
from rejudge.common.state import (ACK_FILE, LIVE_STATES, Ledger, acknowledged, pause as _pause, paused as _paused,  # noqa: E402
                                  readiness, utc_now)

DEFAULT_RETRY = {"verdict": 0, "moderation": 0, "judge_refusal": 0, "parse_error": 1, "truncated": 1, "provider_error": 3,
                 "missing": 3, "batch_failed": 3, "batch_expired": 3, "batch_cancelled": 3}
#: whole-batch endings that did not name the request: counted against their OWN re-asks, never against the request's
BATCH_OUTCOMES = ("batch_failed", "batch_expired", "batch_cancelled")
#: not a property of the request: never terminal and never counted (the run pauses, backs off, or re-queues instead)
UNCOUNTED_OUTCOMES = ("account_error", "batch_failed_auth", "balance_error", "batch_failed_other", "held")
#: any one of these pauses the run: nothing will succeed until a person fixes the account / credential
PAUSING_OUTCOMES = ("account_error", "batch_failed_auth")
#: the outcomes an identical-error flood is looked for in (truncation has its own share rule)
FLOOD_OUTCOMES = ("moderation", "judge_refusal", "parse_error", "provider_error", "missing")
HEALTH_DEFAULTS = {"min_count": 10, "identical_share": 0.5, "truncated_share": 0.2, "error_share": 0.5, "acknowledged": [],
                   "pause_on_batch_end": False}
BALANCE_BACKOFF = (300.0, 3600.0)
#: exit codes of the loop
EXIT_DONE, EXIT_MAX_HOURS, EXIT_PAUSED, EXIT_UNFINISHED = 0, 3, 4, 5
_CUSTOM_ID_IN_MSG = re.compile(r"'([0-9a-f]{16,128})'|\"([0-9a-f]{16,128})\"|\b([0-9a-f]{64})\b")


def error_signature(p: Mapping[str, Any]) -> str:
    """A short stable id of an outcome's error, for flood detection: outcome, error kind, status, error type, provider code and the
    message with digits and long hex runs normalised away."""
    e = p.get("error") or {}
    if not isinstance(e, Mapping):
        e = {"message": str(e)}
    md = e.get("metadata") if isinstance(e.get("metadata"), Mapping) else {}
    msg = re.sub(r"\d+", "#", re.sub(r"[0-9a-f]{8,}", "H", str(e.get("message") or "").lower()))[:200]
    parts = [str(p.get("outcome")), str(e.get("kind")), str(p.get("status_code") or e.get("status_code") or ""),
             str(e.get("error_type") or md.get("error_type") or ""), str(md.get("provider_code") or ""), msg]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def load_parser(spec: str) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def request_cost(model: str, usage: Optional[Mapping[str, Any]]) -> Optional[float]:
    """USD of one request at the ``:batch`` prices WITHOUT the credit fee, from its usage: prompt tokens minus cached and cache-write
    tokens at the input price, cached at the cache-read price, cache writes at the cache-write price, completion (reasoning included) at
    the output price. ``usage.cost`` when the provider body carries one wins (OpenRouter's own accounting)."""
    if not usage:
        return None
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    if model not in BF.PRICES_PER_MTOK:
        return None
    pt = float(usage.get("prompt_tokens") or 0)
    ct = float(usage.get("completion_tokens") or 0)
    det = usage.get("prompt_tokens_details") or {}
    cached = float(det.get("cached_tokens") or 0)
    cwrite = float(det.get("cache_write_tokens") or 0)
    return BF.estimate_cost(model, max(0.0, pt - cached - cwrite), ct, cached, cwrite, with_fee=False)


class Harvester:
    def __init__(self, run_dir: str, client: Optional[BatchClient], log: Callable[[str], None] = print, dry_run: bool = False):
        self.run_dir = run_dir
        self.track = BF.read_track(run_dir)
        self.client = client
        self.log = log
        self.dry_run = dry_run
        self.ledger = Ledger(run_dir).writer()
        self.parser = load_parser(self.track["parser"])
        self.retry = dict(DEFAULT_RETRY, **(self.track.get("retry") or {}))
        self.health_cfg = dict(HEALTH_DEFAULTS, **(self.track.get("health") or {}))
        self.key_info: Dict[str, Any] = {}
        self.cycles = 0
        os.makedirs(os.path.join(run_dir, "raw"), exist_ok=True)

    # -- polling ------------------------------------------------------------------------------------
    def poll(self) -> List[Tuple[str, str]]:
        """``[(batch_key, batch_id)]`` of in-flight batches now in a final status (remote_status refreshed for the rest)."""
        inflight = {r["batch_id"]: r["batch_key"] for r in self.ledger.batches(("submitted",)) if r["batch_id"]}
        if not inflight:
            return []
        seen: Dict[str, Dict[str, Any]] = {}
        try:
            for b in self.client.iter_batches(max_pages=50):
                if b.get("id") in inflight:
                    seen[b["id"]] = b
                if len(seen) == len(inflight):
                    break
        except OpenRouterError as e:
            self.log(f"[poll] list failed ({e}); reading batches by id")
        final = []
        for bid, key in inflight.items():
            b = seen.get(bid)
            if b is None:
                try:
                    b = self.client.get(bid)
                except OpenRouterError as e:
                    self.log(f"[poll] GET {bid} failed: {e}")
                    continue
            status = b.get("status")
            self.ledger.update_batch(key, remote_status=status, provider=b.get("provider"), request_counts_json=b.get("request_counts"))
            if status in FINAL_STATUSES:
                final.append((key, bid))
        return final

    # -- harvesting -----------------------------------------------------------------------------------
    def fetch_final(self, key: str, bid: str) -> Optional[Dict[str, Any]]:
        """GET a final batch with its results, retrying while a completed batch still shows ``results: null`` (the quickstart's
        ``resultsReady`` rule); saves the reply verbatim."""
        for _ in range(10):
            b = self.client.get(bid)
            if b.get("status") != "completed" or b.get("results") is not None:
                path = os.path.join(self.run_dir, "raw", f"{key}__{bid}.json.gz")
                with gzip.open(path + ".tmp", "wt") as fh:
                    json.dump(b, fh)
                os.replace(path + ".tmp", path)
                return b
            time.sleep(30)
        self.log(f"[harvest] {bid}: completed but results still null after 10 reads; will retry next cycle")
        return None

    def harvest(self, key: str, bid: str) -> Dict[str, int]:
        b = self.fetch_final(key, bid)
        if b is None:
            return {}
        man = BF.read_manifest(self.run_dir, key)
        ids = list(man["custom_ids"])
        idset = set(ids)
        status = b.get("status")
        results = b.get("results") or []
        model = str(self.track["model"])
        parsed: Dict[str, Dict[str, Any]] = {}
        for entry in results:
            cid = entry.get("custom_id")
            if cid not in idset:
                self.log(f"[harvest] {bid}: result for unknown custom_id {str(cid)[:16]}...; ignored")
                continue
            if cid in parsed:
                self.log(f"[harvest] {bid}: a second result for {str(cid)[:16]}...; the first is kept")
                continue
            try:
                p = self.parser(entry)
            except Exception as exc:  # noqa: BLE001 -- a parser crash is recorded, never fatal
                p = {"outcome": "parse_error", "error": {"kind": "parser_exception", "message": f"{type(exc).__name__}: {exc}"}}
            parsed[cid] = dict(p)
        missing = [c for c in ids if c not in parsed]
        msg = str(((b.get("error") or {}).get("message")) or "")
        if missing:
            if status == "completed":
                for cid in missing:
                    parsed[cid] = {"outcome": "missing", "error": {"kind": "missing", "message": "no result entry in a completed batch"}}
            else:
                why = {"failed": "batch_failed", "expired": "batch_expired", "cancelled": "batch_cancelled"}.get(str(status), "batch_failed")
                account = status == "failed" and is_account_failure(msg, batch_level=True)
                named = set()
                if status == "failed" and msg and not account:
                    for m in _CUSTOM_ID_IN_MSG.finditer(msg):
                        cid = next(g for g in m.groups() if g)
                        if cid in idset:
                            named.add(cid)
                if account:
                    why = "batch_failed_auth"
                for cid in missing:
                    if cid in named:
                        o = "moderation" if re.search(r"prohibited|moderat|flagged", msg, re.I) else "provider_error"
                        parsed[cid] = {"outcome": o, "error": {"kind": "batch_failed_named", "message": msg[:2000]}}
                    elif named:
                        # validation stopped at ANOTHER request: this one was never sent, and the failure says nothing about it
                        parsed[cid] = {"outcome": "batch_failed_other", "error": {"kind": "batch_failed_other", "message": msg[:2000],
                                                                                 "batch_status": status}}
                    else:
                        parsed[cid] = {"outcome": why, "error": {"kind": why, "message": msg[:2000], "batch_status": status}}
        held, reasons = self.health(parsed)
        counts: Counter = Counter()
        conn = self.ledger.conn
        for cid in ids:
            p = parsed[cid]
            outcome = str(p.get("outcome"))
            if held and outcome in FLOOD_OUTCOMES:
                sig = error_signature(p)
                if sig in held:
                    p = {"outcome": "held", "usage": p.get("usage"), "status_code": p.get("status_code"), "finish_reason": p.get("finish_reason"),
                         "error": {"kind": "held_identical_error", "signature": sig, "parsed_outcome": outcome, "parsed_error": p.get("error")}}
                    outcome = "held"
            self._record(cid, key, bid, outcome, p, b.get("provider"), model)
            counts[outcome] += 1
        conn.commit()
        usage = b.get("usage") or {}
        self.ledger.update_batch(key, state="harvested", remote_status=status, finished_utc=utc_now(), harvested_utc=utc_now(),
                                 usage_json=usage, cost=usage.get("cost"), request_counts_json=b.get("request_counts"),
                                 provider=b.get("provider"))
        self.ledger.event("HARVESTED", f"{bid} status={status} " + json.dumps(dict(counts)), key)
        self.log(f"[harvest] {key} {bid} status={status} cost={usage.get('cost')} " + json.dumps(dict(counts)))
        if counts.get("batch_failed_auth"):
            self.pause(f"batch {bid} failed on the provider account/credential: {msg[:500]}")
        if counts.get("account_error"):
            sample = next((parsed[c].get("error") or {} for c in ids if parsed[c].get("outcome") == "account_error"), {})
            self.pause(f"{counts['account_error']} request(s) of batch {bid} ({key}) failed on the provider account/credential, e.g. "
                       f"HTTP {sample.get('status_code')}: {str(sample.get('message') or '')[:300]} -- nothing is recorded against those items")
        if counts.get("balance_error"):
            self.ledger.event("BALANCE_ITEM", f"{counts['balance_error']} request(s) of {bid} refused for balance / spending limit; re-asked after a backoff", key)
        for r in reasons:
            self.pause(f"batch {bid} ({key}): {r}")
        unanswered = sum(counts.get(o, 0) for o in BATCH_OUTCOMES + ("batch_failed_other",))
        if self.health_cfg.get("pause_on_batch_end") and status in ("failed", "expired", "cancelled") and unanswered:
            self.pause(f"batch {bid} ({key}) ended {status} with {unanswered} of {len(ids)} requests unanswered "
                       f"({json.dumps({o: counts[o] for o in BATCH_OUTCOMES + ('batch_failed_other',) if counts.get(o)})}; message "
                       f"{msg[:300]!r}). They are kept, not terminal, and re-asked once PAUSED is removed. An expiry means the provider "
                       f"did not finish the batch inside its 24-hour window: before removing PAUSED, consider a lower INFLIGHT_LIMIT "
                       f"(requests in flight) in {self.run_dir}; check the key's usage for what an expired batch billed")
        return dict(counts)

    def health(self, parsed: Mapping[str, Mapping[str, Any]]) -> Tuple[set, List[str]]:
        """``(signatures to hold, pause reasons)`` for one batch's parsed results (module docstring, step 4)."""
        cfg = self.health_cfg
        n = len(parsed)
        if not n:
            return set(), []
        mc = int(cfg["min_count"])
        acks = acknowledged(self.run_dir, cfg.get("acknowledged") or [])
        groups: Counter = Counter()
        sample: Dict[str, Mapping[str, Any]] = {}
        for p in parsed.values():
            if str(p.get("outcome")) in FLOOD_OUTCOMES:
                sig = error_signature(p)
                groups[sig] += 1
                sample.setdefault(sig, p)
        held: set = set()
        reasons: List[str] = []
        for sig, c in groups.most_common():
            if c >= mc and c / n >= float(cfg["identical_share"]) and sig not in acks:
                held.add(sig)
                e = sample[sig].get("error") or {}
                reasons.append(f"identical-error flood: {c} of {n} results ({c / n:.0%}) share signature {sig} (outcome "
                               f"{sample[sig].get('outcome')}, kind {e.get('kind')}, status {sample[sig].get('status_code') or e.get('status_code')}, "
                               f"message {str(e.get('message') or '')[:200]!r}); HELD, not recorded as that outcome. If it is a real per-item "
                               f"outcome, add the line {sig} to {os.path.join(self.run_dir, ACK_FILE)} and remove PAUSED (the held requests are "
                               f"re-asked once and recorded); otherwise fix the cause and remove PAUSED")
        nt = sum(1 for p in parsed.values() if p.get("outcome") == "truncated")
        if nt >= mc and nt / n >= float(cfg["truncated_share"]):
            reasons.append(f"truncation: {nt} of {n} results ({nt / n:.0%}) cut at max_tokens (reasoning not off, or the cap too tight); "
                           f"re-asks wait until a person looks")
        ne = sum(1 for p in parsed.values() if p.get("outcome") in ("provider_error", "missing", "parse_error"))
        if ne >= mc and ne / n >= float(cfg["error_share"]):
            reasons.append(f"errors: {ne} of {n} results ({ne / n:.0%}) are provider errors, missing or unparseable; re-asks wait until a "
                           f"person looks")
        return held, reasons

    def _record(self, cid: str, key: str, bid: str, outcome: str, p: Mapping[str, Any], provider: Optional[str], model: str) -> None:
        """Store one attempt. Terminal: a verdict always; an UNCOUNTED outcome never; otherwise when the attempts of the outcome's class
        (BATCH outcomes, or every other counted outcome) exceed its re-asks."""
        if outcome == "verdict":
            terminal = True
        elif outcome in UNCOUNTED_OUTCOMES:
            terminal = False
        else:
            prior = self.ledger.attempt_outcomes(cid)
            if outcome in BATCH_OUTCOMES:
                n = sum(1 for o in prior if o in BATCH_OUTCOMES) + 1
            else:
                n = sum(1 for o in prior if o not in BATCH_OUTCOMES and o not in UNCOUNTED_OUTCOMES) + 1
            terminal = n > int(self.retry.get(outcome, 0))
        usage = p.get("usage")
        self.ledger.record_outcome(cid, key, bid, outcome, terminal, p.get("verdict"), p.get("error"), p.get("status_code"),
                                   p.get("finish_reason"), usage, request_cost(model, usage), provider, commit=False)

    # -- pause -----------------------------------------------------------------------------------------
    def pause(self, reason: str) -> None:
        """Stop submitting and retrying until a person removes ``<run_dir>/PAUSED`` (after fixing the account / key)."""
        _pause(self.run_dir, self.ledger, reason, self.log)

    def paused(self) -> Optional[str]:
        return _paused(self.run_dir)

    # -- retries -------------------------------------------------------------------------------------
    def build_retries(self, max_requests: Optional[int] = None) -> List[str]:
        """Retry manifests for every non-terminal request whose batch is harvested, built from the original bodies. A ``truncated``
        request is re-sent with the cap raised to ``track.json`` ``truncated_retry_max_tokens``, or made terminal when no higher cap is
        set (the same cap cannot succeed). ``balance_error`` requests go into their own manifests, held back by a backoff. Each retry
        manifest carries a cost estimate (the original manifest's tokens per byte, and per request for the output; the raised cap for a
        truncated re-ask), so the spend ceiling sees it."""
        rows = self.ledger.retryable()
        if not rows:
            return []
        harvested = {r["batch_key"] for r in self.ledger.batches(("harvested",))}
        by_batch: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
        for cid, bkey, attempts, outcome in rows:
            if bkey in harvested:
                by_batch[bkey].append((cid, attempts, outcome))
        if not by_batch:
            return []
        raise_to = int(self.track.get("truncated_retry_max_tokens") or 0)
        packed: Dict[Tuple[int, bool], List[BF.PackedRequest]] = defaultdict(list)
        header = None
        finalized = 0
        for bkey, items in sorted(by_batch.items()):
            man = BF.read_manifest(self.run_dir, bkey)
            header = header or BF.batch_header(str(man["model"]), str(man["endpoint"]), man.get("provider"))
            reqs = {r["custom_id"]: r for r in BF.body_requests(BF.read_body(self.run_dir, man))}
            est = man.get("est") or {}
            in_per_byte = float(est.get("input_tokens") or 0) / max(1, int(man["body_bytes"]))
            out_per_req = float(est.get("output_tokens") or 0) / max(1, int(man["n_requests"]))
            for cid, attempts, outcome in items:
                body = reqs[cid]["body"]
                out_est = out_per_req
                if outcome == "truncated":
                    cap_key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
                    cap = int(body.get(cap_key) or 0)
                    if raise_to <= cap:
                        self.ledger.finalize(cid, f"truncated at {cap_key}={cap}; no higher cap configured (truncated_retry_max_tokens "
                                                  f"{raise_to or 'unset'}), so a re-ask cannot succeed", commit=False)
                        finalized += 1
                        continue
                    body = dict(body, **{cap_key: raise_to})
                    out_est = float(raise_to)
                line = BF.serialize_request(cid, body)
                packed[(attempts, outcome == "balance_error")].append(
                    BF.PackedRequest(cid, line, est_input_tokens=len(line) * in_per_byte, est_output_tokens=out_est))
        if finalized:
            self.ledger.conn.commit()
            self.ledger.event("FINALIZED", f"{finalized} truncated requests made terminal: no higher cap to re-ask with")
        keys = []
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        n_balance_holds = self.ledger.conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'BALANCE_HOLD'").fetchone()[0]
        for (attempts, balance), reqs in sorted(packed.items()):
            for part in BF.chunk(reqs, header, int(self.track.get("max_bytes") or BF.DEFAULT_MAX_BYTES),
                                 int(max_requests or self.track.get("max_requests") or BF.DEFAULT_MAX_REQUESTS)):
                key = self._free_key(f"retry{attempts}{'b' if balance else ''}_{stamp}")
                man = BF.write_manifest(self.run_dir, key, header, part, str(self.track["track"]), attempt=attempts)
                self.ledger.register_manifest(man)
                if balance:
                    wait = min(BALANCE_BACKOFF[1], BALANCE_BACKOFF[0] * (2 ** n_balance_holds))
                    self.ledger.update_batch(key, next_attempt_ts=time.time() + wait)
                    self.ledger.event("BALANCE_HOLD", f"{len(part)} balance-refused requests held {wait:.0f}s before the re-ask", key)
                keys.append(key)
        n = sum(len(v) for v in packed.values())
        if keys:
            self.ledger.event("RETRY", f"{n} requests into {len(keys)} retry batches")
            self.log(f"[retry] {n} requests -> {len(keys)} retry batches")
        return keys

    def _free_key(self, prefix: str) -> str:
        j = 0
        while os.path.exists(os.path.join(self.run_dir, "manifests", f"{prefix}_{j:03d}.json")):
            j += 1
        return f"{prefix}_{j:03d}"

    def set_aside(self, batch_key: str, reason: str) -> None:
        """An operator's decision: a pending / queued batch becomes ``refused`` (kept, reported, never finished) with the reason."""
        row = self.ledger.conn.execute("SELECT state FROM batches WHERE batch_key = ?", (batch_key,)).fetchone()
        if row is None:
            raise SystemExit(f"no batch {batch_key} in {self.run_dir}")
        if row["state"] not in ("pending", "balance_queued"):
            raise SystemExit(f"batch {batch_key} is {row['state']}; only a pending or queued batch can be set aside")
        self.ledger.update_batch(batch_key, state="refused", last_error_kind="set_aside", last_error=reason)
        self.ledger.event("SET_ASIDE", reason, batch_key)

    # -- progress -----------------------------------------------------------------------------------
    def progress(self, done: bool) -> Dict[str, Any]:
        conn = self.ledger.conn
        cost_batches = conn.execute("SELECT COALESCE(SUM(cost), 0) FROM batches WHERE cost IS NOT NULL").fetchone()[0]
        cost_requests = conn.execute("SELECT COALESCE(SUM(cost), 0) FROM requests").fetchone()[0]
        est_pending = conn.execute("SELECT COALESCE(SUM(est_cost), 0) FROM batches WHERE state IN ('pending','balance_queued','submitted','submit_unknown')").fetchone()[0]
        n_req = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        terminal = conn.execute("SELECT COUNT(*) FROM requests WHERE terminal = 1").fetchone()[0]
        events = [dict(r) for r in conn.execute("SELECT ts, kind, batch_key, message FROM events ORDER BY rowid DESC LIMIT 25")]
        balance = [dict(r) for r in conn.execute("SELECT batch_key, last_error, next_attempt_ts FROM batches WHERE state='balance_queued'")]
        ready = readiness(self.run_dir)
        prog = {
            "updated_utc": utc_now(), "run_dir": self.run_dir, "track": self.track.get("track"), "model": self.track.get("model"),
            "cycles": self.cycles, "done": bool(done and ready["ready"]), "busy": not done,
            "readiness": {k: v for k, v in ready.items() if k not in ("run_dir", "fingerprint")}, "batches": self.ledger.state_counts(),
            "requests": {"total": n_req, "terminal": terminal, "outcomes": self.ledger.outcome_counts()},
            "spend_usd": {"batches_reported": round(float(cost_batches), 6), "requests_computed_no_fee": round(float(cost_requests), 6),
                          "est_remaining_with_fee": round(float(est_pending), 4), "committed_with_fee": round(SUB.committed_spend(self.ledger), 4),
                          "max_spend_usd": self.track.get("max_spend_usd")},
            "balance_queued": balance, "paused": self.paused(), "key": self.key_info, "ramp": SUB.ramp_status(self.run_dir, self.ledger, self.track),
            "events": events,
        }
        path = os.path.join(self.run_dir, "progress.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(prog, fh, indent=1, default=str)
        os.replace(path + ".tmp", path)
        return prog

    def busy(self) -> bool:
        """Work left that this harvester can still move: a paused run with nothing in flight is NOT busy (it exits and says why)."""
        if self.paused() and not self.ledger.batches(("submitted", "submit_unknown")):
            return False
        live = self.ledger.batches(LIVE_STATES)
        return bool(live) or bool(self.ledger.retryable())

    def final_state(self) -> Tuple[int, str]:
        """The exit code and line of a loop that has nothing left to move (module docstring). DONE only when every registered request
        is terminal, no batch is live or set aside, and the run is not paused."""
        r = readiness(self.run_dir)
        if r["ready"]:
            return EXIT_DONE, f"done: all {r['requests']} requests terminal, no batch pending, in flight, queued or set aside"
        if r["paused"]:
            return EXIT_PAUSED, (f"stopping: the run is PAUSED and nothing is in flight ({r['paused'].splitlines()[0][:300]}); fix the cause, "
                                 f"remove PAUSED, run again")
        return EXIT_UNFINISHED, (f"stopped UNFINISHED: {len(r['refused_batches'])} batch(es) set aside / refused {r['refused_batches'][:5]}, "
                                 f"{r['requests'] - r['terminal']} of {r['requests']} requests without a terminal outcome {r['non_terminal']}")

    # -- the loop -------------------------------------------------------------------------------------
    def cycle(self, submit: bool = True) -> Dict[str, Any]:
        self.cycles += 1
        out: Dict[str, Any] = {}
        if self.dry_run:
            out["submit"] = SUB.submit_pending(self.run_dir, self.ledger, None, self.track, dry_run=True, log=self.log)
            return out
        if self.cycles % 10 == 1:
            try:
                self.key_info = self.client.key_info()
            except OpenRouterError as e:
                self.key_info = {"error": str(e)}
        SUB.reconcile_unknown(self.run_dir, self.ledger, self.client, self.track, log=self.log)
        paused = self.paused()
        if paused:
            self.log(f"[harvester] paused, not submitting: {paused.splitlines()[0]}")
            submit = False
        if submit:
            out["submit"] = SUB.submit_pending(self.run_dir, self.ledger, self.client, self.track, log=self.log)
        for key, bid in self.poll():
            try:
                out.setdefault("harvested", {})[key] = self.harvest(key, bid)
            except OpenRouterError as e:
                self.log(f"[harvest] {key} {bid}: {e}; next cycle")
        # a harvest above may have paused the run: re-asks wait for the person, so none are built into a paused run
        if submit and self.track.get("auto_retry", True) and not self.paused():
            self.build_retries()
        return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--once", action="store_true", help="one cycle, then exit")
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--max-hours", type=float, default=72.0)
    ap.add_argument("--no-submit", action="store_true", help="poll and harvest only")
    ap.add_argument("--dry-run", action="store_true", help="report what would be submitted; no network")
    ap.add_argument("--set-aside", metavar="BATCH_KEY", help="mark a pending / queued batch refused (an operator's decision) and exit")
    ap.add_argument("--reason", help="why, for --set-aside")
    args = ap.parse_args(argv)
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if args.set_aside:
        if not args.reason:
            raise SystemExit("--set-aside needs --reason")
        h = Harvester(run_dir, None, dry_run=True)
        h.set_aside(args.set_aside, args.reason)
        print(f"set aside {args.set_aside}: {args.reason}")
        return 0
    logf = open(os.path.join(run_dir, "harvester.log"), "a")

    def log(msg: str) -> None:
        line = f"{utc_now()} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    client = None if args.dry_run else BatchClient()
    h = Harvester(run_dir, client, log=log, dry_run=args.dry_run)
    t_end = time.time() + args.max_hours * 3600
    log(f"[harvester] start run_dir={run_dir} track={h.track.get('track')} model={h.track.get('model')} pid={os.getpid()}")
    while True:
        try:
            h.cycle(submit=not args.no_submit)
        except Exception:  # noqa: BLE001 -- one bad cycle is logged; the loop goes on
            log("[harvester] cycle error:\n" + traceback.format_exc())
        busy = h.busy() and not args.dry_run
        h.progress(done=not busy)
        if args.once or args.dry_run:
            return 0
        if not busy:
            code, line = h.final_state()
            log(f"[harvester] {line} (exit {code})")
            return code
        if time.time() > t_end:
            log(f"[harvester] --max-hours reached with work left; exiting {EXIT_MAX_HOURS} (re-run to resume: the ledger is the state)")
            return EXIT_MAX_HOURS
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
