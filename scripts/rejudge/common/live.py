#!/usr/bin/env python3
"""LIVE judging of a batch run directory's requests: the SAME request bodies, sent one by one to OpenRouter's chat completions
endpoint instead of the Batch API, into a run directory of their own whose ledger is the shared :class:`state.Ledger` with the shared
outcome vocabulary, so the progress, the readiness gate and the filing read it as they read a batch ledger.

    init      register the key set: every request of the SOURCE run directory's PENDING batches (state pending / balance_queued, never
              POSTed: no batch id), exactly the bodies its manifests hold (read and sha256-checked from ``bodies/``), with each key's
              prior outcomes (a pending re-ask batch's keys were already answered once) -- read only on the source
    run       send them (asyncio + aiohttp; ``--limit N`` for a pilot), harvest every reply with the TRACK'S PARSER and re-ask with the
              batch track's policy, guarded by a spend ceiling and a global check, until every key is terminal (exit 0)
    handover  move keys the batch run cannot finish (in-flight batches past a deadline, re-asks a stopped batch harvester will never
              build) into the live run: a file in ``handovers/`` that the engine ingests (no lock on the source, none on the live run)
    reopen    make terminal live outcomes non-terminal again (the keys a named provider served, or the ones that break the protocol as
              the routing now stands), so the engine judges them again: :func:`reopen_live` (refused while an engine runs)
    dry-run   keys, cost at the live price mix, duration at the measured (else assumed) latency; no network

WHAT IS SENT. ``POST <base>/chat/completions`` with ``{"model": <track model>, **<the batch request body>, "provider": <routing>}``: the
body is the manifest's request body unchanged (messages, tools, tool_choice, reasoning, max_tokens: what the judge sees), plus the model
the batch header named and OpenRouter's provider-routing object (which providers may serve it: ``track.json`` ``live.routing``). A re-ask
re-sends the body it last sent, a cut reply with the cap raised to ``truncated_retry_max_tokens`` (else made terminal), as
``harvester.build_retries`` does. ``live_keys.body_sha256`` holds the sha256 of each key's compact JSON body at init and every attempt
records the sha256 of what it sent, so identity with the batch manifests is checkable per attempt.

OUTCOMES. Each HTTP reply is wrapped as a batch result entry (``{"response": {"status_code", "body"}}``) and parsed by the track's parser,
so moderation, judge refusals, parse errors, cut replies, account / balance / provider errors are classified exactly as in the batch
track, and the terminal rule is the harvester's (:func:`is_terminal`: a verdict always; the uncounted outcomes never; otherwise when the
attempts of the outcome's budget class, counting the source run's prior attempts, exceed ``track.json`` ``retry``). Transport is handled
before an outcome exists:
  429 / 529             rate limited or overloaded: the same request is retried after the larger of ``Retry-After`` and an exponential
                        backoff (never under ``rate_limit_min_s``, so ``Retry-After: 0`` makes no retry storm; jittered) and the
                        ADAPTIVE LIMIT halves the concurrency when they persist (``rl_halve_after`` within ``rl_window_s``, the first time
                        at once, then at most once per ``rl_halve_spacing_s``), growing back by a tenth of the target every
                        ``rl_recover_s`` without one; never another account, never another key
  408 / 5xx / timeout / connection error   retried in place up to ``transport_retries`` times (jittered exponential backoff); then the
                        attempt is a ``provider_error`` (``kind`` ``transport_*``), re-asked per the policy. A timed-out request may have
                        been billed: its estimate counts as spent
  an engine exception   while sending or reading one request: a counted ``provider_error`` (``kind`` ``engine_exception``, its estimate
                        counted as spent), so the re-ask budget bounds a crash that repeats and a flood of them pauses the run
THE PROTOCOL OF A SERVED REPLY (HTTP 200 with ``choices``; :func:`protocol_violation`): its provider is one the routing ALLOWS
(:func:`allowed_providers`: in ``live.providers``, matched by ``only``, not by ``ignore``, of an allowed quantization), it names the model
asked for (or its dated canonical slug; ``live.served_models`` overrides), its reported cost is within ``price_tolerance`` x what its
provider's list price allows for its usage (a dearer tier of an allowed provider, e.g. Parasail's ``fast``, bears the same provider name),
and it bills no reasoning tokens (the protocol turns reasoning off). A reply that breaks it is OFF-PROTOCOL whatever it parsed to:
recorded ``held`` (uncounted, never terminal, so never filed; ``live_attempts.violation`` names why, the parse is kept in its error).
Every kind but reasoning PAUSES the run (the routing is not doing what was asked; the held keys are re-sent when it starts again); a
reasoning reply is re-sent at once (``max_reasoning_resends`` per key, then the key is parked: exit 5), and a provider the routing still
allows that bills reasoning on more than ``reasoning_share`` of at least ``reasoning_min_replies`` replies pauses the run (dropping it
from the routing clears that guard; raising ``reasoning_share`` keeps it). With ``reasoning_reroute`` (off by default) a key past its
``max_reasoning_resends`` is not parked while an allowed endpoint has not billed reasoning on it: it is re-sent with the run's routing
narrowed to those endpoints (``only`` cut, nothing widened; the body unchanged; a ``REROUTED`` event names them), and parked only once
every allowed endpoint has billed reasoning on it. A key a person decides to leave unjudged is settled with :func:`settle_parked`
(terminal, never filed). ``run_state`` records ``parked_keys`` and ``non_terminal_keys`` at the end.

Replies are parsed as they arrive and recorded in WINDOWS (``flush_n`` replies or ``flush_s`` seconds, one transaction each), and each
window gets the harvester's health check: an identical-error flood is HELD (recorded ``held``, uncounted) and pauses the run, unless the
signature is acknowledged (``ACKNOWLEDGED``, ``health.acknowledged``) or it is a transport failure, which cools the run down (no new
requests for ``cooldown_s``, doubling, then the held keys are re-sent; ``max_cooldowns`` floods in a row pause it); a fifth of a window
cut, half a window failing for other reasons, any account error, a balance error (the money ran out), or an off-protocol reply of a
pausing kind PAUSES it.

STOPPING. Every stop finishes the requests in flight and records them first. Exit 0: every key terminal (or a ``--limit`` pilot's keys
all terminal); 4: PAUSED (``<live dir>/PAUSED`` names the reason, with the numbers for a spend stop; remove it and start again); 3: a
signal (SIGTERM / SIGINT), the ``STOP`` file or ``--max-hours``; 5: nothing left to send but a key without a terminal outcome.
``CONCURRENCY`` (an integer) sets the target concurrency while it runs. A restart resumes from the ledger: a key with a terminal outcome is
never sent again, and one key is never in flight twice (a key stays in flight until its reply is recorded); each key is re-sent with the
body its last attempt went out with (a cut reply's re-ask at the raised cap). ``live_meta`` ``run_state`` records how the run ended
(``end_state`` done / pilot_done / stopped / paused / unfinished / crashed, ``exit_code``, ``limit``); a run that ends any way but through
its finish keeps ``clean_exit`` false with the estimate of what it left unrecorded, which the next start counts as spent. A caller
that restarts the engine without a person (a scheduled job) checks :func:`auto_start_refusal` first: only when the last run was the
FULL run and ended ``done`` (or, where the caller allows it, ``unfinished`` with nothing left but parked keys).

A CUT REPLY OF THE SOURCE'S BATCH that a handover passed on and whose cap cannot be raised is made terminal without sending
(:meth:`LiveRunner._finalize_uncappable`): recorded with the source's batch id, provider, status and harvest time and
``error.from_source_run``, so the filing names it as the batch reply it is (``rejudge.common.merge.read_outcomes``).

SPEND. ``max_spend_usd`` (``track.json``; USD including the 5.5% credit fee) is checked before EVERY request: the reported cost of every
recorded attempt (``usage.cost``; computed from the provider's price when absent; the estimate when the billing is unknown), plus the
requests in flight at their estimates (the most expensive allowed provider, scaled up by the measured cost / estimate ratio), plus this
request's estimate. The GLOBAL check (at start and every ``global_check_s``) adds every other OpenRouter ledger under ``global_root``
(:func:`openrouter_spend`: batch ledgers' reported batch costs x the fee, in-flight batch estimates, other live ledgers) and the spend this
run is projected to add before the next check; at ``global_ceiling_usd`` it stops sending, finishes the requests in flight and PAUSES
with the numbers. ``max_spend_usd``, ``global_root`` (an absolute path of an existing directory) and ``global_ceiling_usd`` have no
defaults: :func:`init_live` refuses a live directory without them, and the engine refuses one that lacks them.

THE KEY is read from ``$OPENROUTER_API_KEY`` (the engine's own environment) and never printed, logged or written; every text the
engine records from a reply or an exception passes :func:`rejudge.common.openrouter.redact` with it first.
"""
from __future__ import annotations

import asyncio
import collections
import dataclasses
import gzip
import hashlib
import json
import math
import os
import random
import re
import signal
import sqlite3
import sys
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import openrouter as OR  # noqa: E402
from rejudge.common.harvester import (BATCH_OUTCOMES, DEFAULT_RETRY, FLOOD_OUTCOMES, HEALTH_DEFAULTS, UNCOUNTED_OUTCOMES,  # noqa: E402
                                      error_signature, load_parser)
from rejudge.common.state import PAUSE_FILE, Ledger, acknowledged, paused as _paused, readiness, utc_now  # noqa: E402

BACKEND = "live"
CHAT_PATH = "/chat/completions"
LIVE_SCHEMA = "rejudge-live/1"
HANDOVER_SCHEMA = "rejudge-live-handover/1"
STOP_FILE = "STOP"
CONCURRENCY_FILE = "CONCURRENCY"
USER_AGENT = "rejudge-live/1"
SOURCE_PENDING_STATES = ("pending", "balance_queued")
SOURCE_INFLIGHT_STATES = ("submitted", "submit_unknown")
EXIT_DONE, EXIT_STOPPED, EXIT_PAUSED, EXIT_UNFINISHED = 0, 3, 4, 5
#: a key's origin in ``live_keys``: registered at init from a pending batch, or handed over later
ORIGINS = ("pending", "reask", "in_flight")
FORBIDDEN = re.compile(r"claude|anthropic", re.I)
#: directories never searched for ledgers by :func:`openrouter_spend` (bulky, or not ledgers)
PRUNE_DIRS = frozenset({"raw", "bodies", "manifests", "records", "inbox", "snapshots", "staging", "tokenizer_dl", "enum", "texts",
                        "__pycache__", "provenance", "handovers"})

#: the engine's settings (``track.json`` ``live``; the CLI overrides a few). ``global_root`` and ``global_ceiling_usd`` have no default
#: (:data:`REQUIRED_LIVE`)
LIVE_DEFAULTS: Dict[str, Any] = {
    "base_url": OR.BASE_URL, "concurrency": 300, "min_concurrency": 8, "request_timeout_s": 240.0, "connect_timeout_s": 30.0,
    "transport_retries": 3, "transport_backoff_s": 2.0, "transport_backoff_max_s": 60.0,
    "rate_limit_min_s": 1.0, "rate_limit_backoff_s": 5.0, "rate_limit_backoff_max_s": 120.0, "rate_limit_max_tries": 40,
    "rl_window_s": 60.0, "rl_halve_after": 10, "rl_halve_spacing_s": 60.0, "rl_recover_s": 300.0,
    "flush_n": 200, "flush_s": 5.0, "progress_s": 30.0, "control_s": 2.0,
    "global_check_s": 180.0,
    "cooldown_s": 60.0, "cooldown_max_s": 900.0, "max_cooldowns": 5,
    "reasoning_share": 0.10, "reasoning_min_replies": 50, "max_reasoning_resends": 3, "reasoning_reroute": False,
    "price_tolerance": 1.2, "cache_write_premium": 1.25, "served_models": None,
    "max_hours": 56.0, "spread_batches": 100,
}
#: the live settings every live directory must state itself: the root its global check walks and the ceiling of that check
REQUIRED_LIVE = ("global_root", "global_ceiling_usd")
#: what a served reply (HTTP 200 with ``choices``) may break; any one makes it OFF-PROTOCOL: recorded ``held`` (never terminal, never
#: counted, never filed) with the violation named in ``live_attempts.violation``. Every kind but ``reasoning_billed`` pauses the run (the
#: routing is not doing what was asked); a reasoning reply is re-sent (``max_reasoning_resends`` per key) while its provider stays under
#: ``reasoning_share``
VIOLATIONS = ("unknown_provider", "off_map_provider", "wrong_model", "price_above_map", "reasoning_billed")
PAUSING_VIOLATIONS = ("unknown_provider", "off_map_provider", "wrong_model", "price_above_map")
#: the error kind of a held off-protocol reply, and of an attempt re-opened by :func:`reopen_live`
OFF_PROTOCOL_KIND = "off_protocol"
REOPENED_KIND = "reopened"
#: how the last run ended (``live_meta`` ``run_state`` ``end_state``) for :func:`auto_start_refusal`: only a FULL run that ended ``done``
#: may be restarted by a handover without a person
AUTO_START_END_STATES = ("done",)

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_keys (
  custom_id TEXT PRIMARY KEY, source_run_dir TEXT NOT NULL, source_batch_key TEXT NOT NULL, origin TEXT NOT NULL, handover_id TEXT,
  prior_outcomes_json TEXT, body_sha256 TEXT, est_input_tokens REAL, est_output_tokens REAL, added_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_keys_batch ON live_keys(source_run_dir, source_batch_key);
CREATE INDEX IF NOT EXISTS idx_live_keys_bk ON live_keys(source_batch_key);
CREATE INDEX IF NOT EXISTS idx_live_keys_origin ON live_keys(origin);
CREATE TABLE IF NOT EXISTS live_attempts (
  custom_id TEXT NOT NULL, attempt INTEGER NOT NULL, started_utc TEXT, finished_utc TEXT NOT NULL, latency_s REAL, http_status INTEGER,
  outcome TEXT NOT NULL, terminal INTEGER NOT NULL, provider TEXT, quantization TEXT, quantization_source TEXT, served_model TEXT,
  generation_id TEXT, max_tokens INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER, reasoning_chars INTEGER,
  cached_tokens INTEGER, cache_write_tokens INTEGER, cost_usd REAL, cost_source TEXT, est_cost_usd REAL, transport_retries INTEGER, rate_limited INTEGER,
  body_sha256 TEXT, error_signature TEXT, served INTEGER, violation TEXT, reopened_json TEXT, PRIMARY KEY (custom_id, attempt)
);
CREATE INDEX IF NOT EXISTS idx_live_attempts_finished ON live_attempts(finished_utc);
CREATE TABLE IF NOT EXISTS live_reopens (
  ts TEXT NOT NULL, custom_id TEXT NOT NULL, attempt INTEGER NOT NULL, selector TEXT NOT NULL, reason TEXT, original_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_spend_adjustments (ts TEXT NOT NULL, usd_no_fee REAL NOT NULL, reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS live_handovers (
  handover_id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL, n_keys INTEGER NOT NULL, n_added INTEGER NOT NULL,
  ingested_utc TEXT NOT NULL, reason TEXT
);
CREATE TABLE IF NOT EXISTS live_meta (k TEXT PRIMARY KEY, v TEXT);
"""


# =====================================================================================================================================
# small helpers
# =====================================================================================================================================
def compact(obj: Any) -> bytes:
    """The batch track's serialization (``batchfile.serialize_request``'s settings): compact, ASCII."""
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def body_sha256(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(compact(body)).hexdigest()


#: columns added to ``live_attempts`` after the first live directories were initialized (added in place by :func:`ensure_schema`)
LATER_ATTEMPT_COLUMNS = (("served", "INTEGER"), ("violation", "TEXT"), ("reopened_json", "TEXT"))


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(live_attempts)")}
    for col, typ in LATER_ATTEMPT_COLUMNS:
        if col not in have:
            conn.execute(f"ALTER TABLE live_attempts ADD COLUMN {col} {typ}")
    conn.commit()


def attempt_columns(conn: sqlite3.Connection) -> Set[str]:
    """``live_attempts``' columns (a directory initialized before :data:`LATER_ATTEMPT_COLUMNS` lacks them until an engine opens it)."""
    return {r[1] for r in conn.execute("PRAGMA table_info(live_attempts)")}


def meta_get(conn: sqlite3.Connection, k: str) -> Optional[Any]:
    try:
        r = conn.execute("SELECT v FROM live_meta WHERE k = ?", (k,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return json.loads(r[0]) if r and r[0] is not None else None


def meta_set(conn: sqlite3.Connection, k: str, v: Any, commit: bool = True) -> None:
    conn.execute("INSERT INTO live_meta(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (k, json.dumps(v, default=str)))
    if commit:
        conn.commit()


def is_live_dir(run_dir: str) -> bool:
    """A run directory written by :func:`init_live` (``track.json`` ``backend`` = live)."""
    try:
        return BF.read_track(run_dir).get("backend") == BACKEND
    except (OSError, ValueError):
        return False


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def _ro(path: str, timeout: float = 60.0) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    con.row_factory = sqlite3.Row
    return con


def is_terminal(outcome: str, prior: Sequence[str], retry: Mapping[str, int]) -> bool:
    """The harvester's terminal rule (``Harvester._record``): a verdict always; an uncounted outcome never; otherwise when the attempts
    of the outcome's budget class (the batch outcomes, or every other counted outcome), this one included, exceed its re-asks."""
    if outcome == "verdict":
        return True
    if outcome in UNCOUNTED_OUTCOMES:
        return False
    if outcome in BATCH_OUTCOMES:
        n = sum(1 for o in prior if o in BATCH_OUTCOMES) + 1
    else:
        n = sum(1 for o in prior if o not in BATCH_OUTCOMES and o not in UNCOUNTED_OUTCOMES) + 1
    return n > int(retry.get(outcome, 0))


def cap_key(body: Mapping[str, Any]) -> str:
    return "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"


def provider_price(providers: Mapping[str, Mapping[str, Any]], name: Optional[str]) -> Optional[Tuple[float, float, float]]:
    """``(input, output, cache read)`` USD / MTok of the provider named in a reply, from ``live.providers``."""
    p = providers.get(name or "") if name else None
    if not p or not p.get("price"):
        return None
    pr = list(p["price"]) + [0.0, 0.0, 0.0]
    return float(pr[0]), float(pr[1]), float(pr[2])


def max_price(providers: Mapping[str, Mapping[str, Any]]) -> Tuple[float, float, float]:
    """The most expensive allowed provider's price per component (the estimate's conservative side)."""
    ps = [provider_price(providers, n) for n in providers]
    ps = [p for p in ps if p]
    if not ps:
        raise ValueError("live.providers holds no price")
    return max(p[0] for p in ps), max(p[1] for p in ps), max(p[2] for p in ps)


def usage_cost(usage: Optional[Mapping[str, Any]], price: Optional[Tuple[float, float, float]]) -> Optional[float]:
    """USD without the fee of one reply from its usage at ``price`` (cached prompt tokens at the cache-read price)."""
    if not usage or price is None:
        return None
    pt = float(usage.get("prompt_tokens") or 0)
    ct = float(usage.get("completion_tokens") or 0)
    det = usage.get("prompt_tokens_details") or {}
    cached = float(det.get("cached_tokens") or 0)
    return (max(0.0, pt - cached) * price[0] + cached * price[2] + ct * price[1]) / 1e6


def item_estimate(est_in: float, est_out: float, price: Tuple[float, float, float]) -> float:
    """USD without the fee of one request at ``price``, no cache discount."""
    return (float(est_in) * price[0] + float(est_out) * price[1]) / 1e6


# =====================================================================================================================================
# the protocol a served reply must keep (the engine holds a reply that breaks it; the filing refuses one)
# =====================================================================================================================================
def _slug_matches(slug: str, entry: str) -> bool:
    """An OpenRouter provider-routing entry names an endpoint (``parasail/fp8``) or a whole provider (``parasail``)."""
    slug, entry = slug.lower(), entry.lower()
    return slug == entry or slug.startswith(entry + "/")


def allowed_providers(providers: Mapping[str, Mapping[str, Any]], routing: Mapping[str, Any]) -> Set[str]:
    """The provider names (as a reply names them) the routing lets serve: in ``live.providers``, its slug matched by ``only`` (when
    given), matched by no ``ignore`` entry, its quantization in ``quantizations`` (when given). A provider dropped from ``only`` or added
    to ``ignore`` stops being allowed without leaving the map."""
    only = [str(s) for s in routing.get("only") or []]
    ignore = [str(s) for s in routing.get("ignore") or []]
    quants = {str(q).lower() for q in routing.get("quantizations") or []}
    out: Set[str] = set()
    for name, p in providers.items():
        slug = str(p.get("slug") or "")
        if only and not any(_slug_matches(slug, o) for o in only):
            continue
        if any(_slug_matches(slug, i) for i in ignore):
            continue
        if quants and str(p.get("quantization") or "").lower() not in quants:
            continue
        out.add(name)
    return out


def served_model_ok(model: str, served: Optional[str], accepted: Optional[Sequence[str]] = None) -> bool:
    """The model a reply names is the one asked for: ``accepted`` (``live.served_models``) when set, else the slug itself or its dated
    canonical form (``minimax/minimax-m3-20260531``); never a prefix match (``minimax/minimax-m3.5`` is another model)."""
    if not served:
        return False
    if accepted:
        return served in set(accepted)
    return served == model or re.fullmatch(re.escape(model) + r"-\d{8}", served) is not None


def usage_cost_bound(usage: Optional[Mapping[str, Any]], price: Tuple[float, float, float], cache_write_premium: float) -> Optional[float]:
    """The most a reply's usage can cost at a provider's LIST price (USD, no fee): uncached prompt tokens at the input price, cached ones
    at the cache-read price, cache writes at ``cache_write_premium`` x the input price (no endpoint lists a write price), completion
    tokens at the output price. Discounts only lower the reported cost."""
    if not usage:
        return None
    pt = float(usage.get("prompt_tokens") or 0)
    ct = float(usage.get("completion_tokens") or 0)
    det = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), Mapping) else {}
    cached = float(det.get("cached_tokens") or 0)
    cw = float(det.get("cache_write_tokens") or 0)
    return (max(0.0, pt - cached - cw) * price[0] + cached * price[2] + cw * price[0] * cache_write_premium + ct * price[1]) / 1e6


def protocol_violation(cfg: Mapping[str, Any], model: str, allowed: Set[str], provider: Optional[str], served_model: Optional[str],
                       usage: Optional[Mapping[str, Any]], reasoning_tokens: Optional[int], cost_usd: Optional[float],
                       cost_source: Optional[str]) -> Optional[Tuple[str, str]]:
    """``(kind, message)`` when a SERVED reply breaks the live protocol (:data:`VIOLATIONS`), else None. ``cfg`` is the live settings
    (``providers``, ``served_models``, ``price_tolerance``, ``cache_write_premium``). The pausing kinds come first, so a reply that breaks
    several is named by the one that pauses."""
    providers = cfg.get("providers") or {}
    if not provider:
        return "unknown_provider", "the reply names no serving provider, so its endpoint and quantization cannot be checked"
    if provider not in allowed:
        return "off_map_provider", (f"served by {provider!r}, which the routing does not allow (allowed: {sorted(allowed)}; quantization "
                                    f"unknown)")
    if not served_model_ok(model, served_model, cfg.get("served_models")):
        return "wrong_model", f"the reply names model {served_model!r}, not {model!r}"
    tol = float(cfg.get("price_tolerance") or 0)
    if tol > 0 and cost_source == "reported" and cost_usd is not None:
        pr = provider_price(providers, provider)
        bound = usage_cost_bound(usage, pr, float(cfg.get("cache_write_premium") or 1.0)) if pr else None
        if bound is not None and float(cost_usd) > bound * tol + 1e-9:
            return "price_above_map", (f"{provider} billed ${float(cost_usd):.6f}, above {tol:g} x the ${bound:.6f} its list price "
                                       f"{list(pr)} allows for this usage: another endpoint of {provider} (e.g. a dearer tier) served it")
    if int(reasoning_tokens or 0) > 0:
        return "reasoning_billed", f"{provider} billed {int(reasoning_tokens)} reasoning tokens (the protocol turns reasoning off)"
    return None


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) else None


def reply_meta(body: Mapping[str, Any]) -> Dict[str, Any]:
    """The serving facts of an HTTP 200 chat-completion body, read defensively (a malformed field is dropped, never raised on):
    ``provider``, ``model``, ``id``, ``usage`` (numbers only, the two details objects always dicts), ``served`` (a completion: ``choices``
    is a list, whatever it holds, so even a malformed completion has its provider checked before its outcome can count),
    ``reasoning_tokens`` and the length of a ``message.reasoning`` text."""
    def s(k: str) -> Optional[str]:
        v = body.get(k)
        return v if isinstance(v, str) and v else None

    u = body.get("usage") if isinstance(body.get("usage"), Mapping) else None
    usage: Optional[Dict[str, Any]] = None
    if u is not None:
        usage = {k: _num(v) for k, v in u.items() if not isinstance(v, Mapping) and _num(v) is not None}
        for dk in ("prompt_tokens_details", "completion_tokens_details"):
            d = u.get(dk) if isinstance(u.get(dk), Mapping) else {}
            usage[dk] = {k: _num(v) for k, v in d.items() if _num(v) is not None}
    choices = body.get("choices")
    ch = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else None
    msg = ch.get("message") if ch is not None and isinstance(ch.get("message"), Mapping) else None
    rtok = _num(((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0.0
    return {"provider": s("provider"), "model": s("model"), "id": s("id"), "usage": usage, "served": isinstance(choices, list),
            "reasoning_tokens": int(rtok), "reasoning_chars": len(msg["reasoning"]) if msg is not None and isinstance(msg.get("reasoning"), str) else 0}


# =====================================================================================================================================
# init: the key set
# =====================================================================================================================================
def default_routing(providers: Mapping[str, Mapping[str, Any]], quantizations: Sequence[str] = ("fp8",),
                    ignore: Sequence[str] = ()) -> Dict[str, Any]:
    """OpenRouter's ``provider`` object (https://openrouter.ai/docs/guides/routing/provider-selection, read 2026-11-11): ``only`` the
    providers' endpoint slugs, ``quantizations``, ``ignore``, fallbacks among them allowed, ``require_parameters`` OFF, ``max_price`` (USD
    per million tokens, inclusive) at the most expensive allowed provider; no ``sort`` / ``order``, so OpenRouter's default load balancing
    spreads the requests among them.

    ``require_parameters`` is off on purpose: every fp8 endpoint of minimax/minimax-m3 declares
    ``supports_tool_choice`` {auto: true, function: false}, so requiring the batch body's forced ``tool_choice`` matched no endpoint (404
    "No endpoints found that support the provided 'tool_choice' value"). The batch run's DeepInfra flex endpoint declares the same
    (function: false), so the batch and bake-off verdicts were produced with the forced choice degraded to ``auto``; leaving the body
    byte-identical and the parameter unrequired reproduces exactly that on the same class of endpoint. Endpoints that DO honour a forced
    function (Venice fp8, CoreWeave fp4) stay excluded so every live verdict is produced like the batch ones."""
    mp = max_price(providers)
    r: Dict[str, Any] = {"only": [str(p["slug"]) for p in providers.values()], "quantizations": list(quantizations),
                         "allow_fallbacks": True, "require_parameters": False, "max_price": {"prompt": mp[0], "completion": mp[1]}}
    if ignore:
        r["ignore"] = list(ignore)
    return r


def _source_fingerprint(src: str) -> str:
    try:
        return readiness(src)["fingerprint"]
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e}"


def init_live(live_dir: str, source_run_dir: str, routing: Mapping[str, Any], providers: Mapping[str, Mapping[str, Any]],
              max_spend_usd: float, live: Optional[Mapping[str, Any]] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Create ``live_dir`` (mode 0700): ``track.json`` (the source's track, model, endpoint, parser, request settings, re-asks and health,
    plus ``backend`` live, ``source_run_dir``, ``max_spend_usd`` and ``live``), and its ledger holding every key of the source's PENDING
    batches with its body sha256, token estimates and prior outcomes. Read only on the source. Refuses a live directory that already
    holds keys (an init is done once), a Claude / Anthropic model or routing slug, a source body whose sha256 or re-serialization differs
    from its manifest, and a key the source carries from a reuse directory."""
    live_dir = os.path.abspath(os.path.expanduser(live_dir))
    src = os.path.abspath(os.path.expanduser(source_run_dir))
    st = BF.read_track(src)
    bad = [v for v in (st.get("model"), json.dumps(routing), json.dumps(list(providers))) if FORBIDDEN.search(str(v or ""))]
    if bad:
        raise SystemExit(f"REFUSED: a Claude / Anthropic name in the model or routing: {bad}")
    if os.path.exists(os.path.join(live_dir, "state.sqlite")):
        con = _ro(os.path.join(live_dir, "state.sqlite"))
        try:
            n = con.execute("SELECT COUNT(*) FROM live_keys").fetchone()[0] if has_table(con, "live_keys") else 0
        finally:
            con.close()
        if n:
            raise SystemExit(f"REFUSED: {live_dir} already holds {n} keys (init runs once; add keys with a handover)")
    cfg = dict(LIVE_DEFAULTS, **(live or {}))
    missing = [k for k in REQUIRED_LIVE if cfg.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"REFUSED: the live settings state no {' / '.join(missing)}: name the root the global check walks and its ceiling "
                         f"(--global-root DIR --global-ceiling-usd USD)")
    cfg["global_root"] = check_global_root(str(cfg["global_root"]))
    if float(cfg["global_ceiling_usd"]) <= 0 or float(max_spend_usd) <= 0:
        raise SystemExit("REFUSED: max_spend_usd and global_ceiling_usd must be positive")
    os.makedirs(live_dir, exist_ok=True)
    os.chmod(live_dir, 0o700)
    cfg.update({"routing": dict(routing), "providers": {k: dict(v) for k, v in providers.items()}})
    spec = {k: st.get(k) for k in ("track", "model", "endpoint", "provider", "parser", "retry", "request", "truncated_retry_max_tokens",
                                   "health", "judge", "estimate", "context_tokens") if k in st}
    spec.update({"tag": (st.get("tag") or "") + "_live", "backend": BACKEND, "live_schema": LIVE_SCHEMA, "source_run_dir": src,
                 "source_track_sha256": hashlib.sha256(open(os.path.join(src, "track.json"), "rb").read()).hexdigest(),
                 "max_spend_usd": float(max_spend_usd), "live": cfg})
    BF.write_track(live_dir, spec)
    model = str(st["model"])
    carried: Set[str] = set()
    rj = os.path.join(src, "reuse.json")
    if os.path.exists(rj):
        carried = set(json.load(open(rj)).get("keys") or [])
    led = Ledger(live_dir).writer()
    try:
        ensure_schema(led.conn)
        con = _ro(os.path.join(src, "state.sqlite"))
        try:
            batches = [(r["batch_key"], int(r["n_requests"]), int(r["attempt"])) for r in con.execute(
                "SELECT batch_key, n_requests, attempt FROM batches WHERE state IN ('pending', 'balance_queued') "
                "AND (batch_id IS NULL OR batch_id = '') ORDER BY attempt, batch_key")]
            prior: Dict[str, List[str]] = collections.defaultdict(list)
            for r in con.execute("SELECT custom_id, outcome FROM attempts ORDER BY custom_id, attempt"):
                prior[r[0]].append(str(r[1]))
            counts: collections.Counter = collections.Counter()
            now = utc_now()
            for bk, n_req, attempt in batches:
                man = BF.read_manifest(src, bk)
                raw = BF.read_body(src, man)                      # sha256 against the manifest
                doc = json.loads(raw.decode("ascii"))
                reqs = doc.pop("requests")
                if doc.get("model") != model or man.get("model") != model or doc.get("endpoint") != st.get("endpoint"):
                    raise SystemExit(f"REFUSED: source batch {bk} names model {doc.get('model')!r} / endpoint {doc.get('endpoint')!r}")
                if BF.serialize_batch(doc, [BF.serialize_request(r["custom_id"], r["body"]) for r in reqs]) != raw:
                    raise SystemExit(f"REFUSED: source batch {bk} does not re-serialize to its own bytes; the bodies sent live would not "
                                     f"be byte-identical")
                if [r["custom_id"] for r in reqs] != list(man["custom_ids"]) or len(reqs) != n_req:
                    raise SystemExit(f"REFUSED: source batch {bk}: body requests != manifest custom_ids")
                owners = {r[0]: int(r[1]) for r in con.execute("SELECT custom_id, terminal FROM requests WHERE batch_key = ?", (bk,))}
                est = man.get("est") or {}
                in_per_byte = float(est.get("input_tokens") or 0) / max(1, int(man["body_bytes"]))
                out_per_req = float(est.get("output_tokens") or 0) / max(1, int(man["n_requests"]))
                base_cap = int((st.get("request") or {}).get("max_tokens") or 0)
                rows_req, rows_key = [], []
                for r in reqs:
                    cid = r["custom_id"]
                    if cid not in owners:
                        counts["skipped_repointed"] += 1          # the key belongs to a later batch of the source
                        continue
                    if owners[cid]:
                        counts["skipped_terminal"] += 1
                        continue
                    if cid in carried:
                        raise SystemExit(f"REFUSED: key {cid[:16]}... of pending batch {bk} is carried from a reuse directory")
                    line = BF.serialize_request(cid, r["body"])
                    cap = int(r["body"].get(cap_key(r["body"])) or 0)
                    e_out = float(cap) if base_cap and cap > base_cap else out_per_req
                    rows_req.append((cid, f"src:{bk}"))
                    rows_key.append((cid, src, bk, "pending", None, json.dumps(prior.get(cid) or []), body_sha256(r["body"]),
                                     len(line) * in_per_byte, e_out, now))
                    counts["keys"] += 1
                    counts["keys_with_prior_attempts"] += 1 if prior.get(cid) else 0
                led.conn.executemany("INSERT INTO requests(custom_id, batch_key) VALUES (?, ?) ON CONFLICT(custom_id) DO NOTHING", rows_req)
                led.conn.executemany("INSERT INTO live_keys(custom_id, source_run_dir, source_batch_key, origin, handover_id, "
                                     "prior_outcomes_json, body_sha256, est_input_tokens, est_output_tokens, added_utc) VALUES "
                                     "(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(custom_id) DO NOTHING", rows_key)
                led.conn.commit()
                counts["batches"] += 1
            info = {"utc": now, "source_run_dir": src, "source_fingerprint": _source_fingerprint(src), "source_batches": counts["batches"],
                    "keys": counts["keys"], "keys_with_prior_attempts": counts["keys_with_prior_attempts"],
                    "skipped_repointed": counts["skipped_repointed"], "skipped_terminal": counts["skipped_terminal"],
                    "calibration": source_calibration(con, src)}
        finally:
            con.close()
        meta_set(led.conn, "init", info)
        led.event("LIVE_INIT", json.dumps({k: v for k, v in info.items() if k != "calibration"}))
    finally:
        led.close()
    log(f"[live] init {live_dir}: {info['keys']} keys from {info['source_batches']} pending batches of {src} "
        f"({info['keys_with_prior_attempts']} with prior attempts)")
    return info


def source_calibration(con: sqlite3.Connection, src: str) -> Dict[str, Any]:
    """Measured tokens of the source's harvested first-pass replies against their manifests' estimates (read only): the dry run scales
    the estimates with it until the live run has replies of its own."""
    est_in = est_out = 0.0
    n_est = 0
    for (bk,) in con.execute("SELECT batch_key FROM batches WHERE state = 'harvested' AND attempt = 0"):
        try:
            man = BF.read_manifest(src, bk)
        except OSError:
            continue
        est_in += float((man.get("est") or {}).get("input_tokens") or 0)
        est_out += float((man.get("est") or {}).get("output_tokens") or 0)
        n_est += int(man["n_requests"])
    pt = ct = cached = rt = 0.0
    n = 0
    first = collections.Counter()
    for r in con.execute("SELECT r.usage_json FROM requests r JOIN batches b ON b.batch_key = r.batch_key "
                         "WHERE b.state = 'harvested' AND b.attempt = 0 AND r.usage_json IS NOT NULL"):
        u = json.loads(r[0])
        n += 1
        pt += float(u.get("prompt_tokens") or 0)
        ct += float(u.get("completion_tokens") or 0)
        cached += float((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        rt += float((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    for r in con.execute("SELECT a.outcome FROM attempts a JOIN batches b ON b.batch_key = a.batch_key WHERE b.attempt = 0 AND a.attempt = 1"):
        first[str(r[0])] += 1
    nf = sum(first.values())
    return {"replies": n, "requests_estimated": n_est,
            "prompt_tokens_per_reply": pt / n if n else None, "completion_tokens_per_reply": ct / n if n else None,
            "cached_tokens_per_reply": cached / n if n else None, "reasoning_tokens_per_reply": rt / n if n else None,
            "est_input_per_request": est_in / n_est if n_est else None, "est_output_per_request": est_out / n_est if n_est else None,
            "first_attempt_outcomes": dict(first), "first_attempt_reask_share": (sum(v for k, v in first.items() if k != "verdict"
                                                                                    and k not in ("moderation", "judge_refusal")) / nf) if nf else None}


# =====================================================================================================================================
# handover: keys the batch run will not finish
# =====================================================================================================================================
def build_handover(live_dir: str, modes: Iterable[str], reason: str, not_before: Optional[str] = None, now: Optional[float] = None,
                   log: Callable[[str], None] = print) -> Optional[str]:
    """Write ``<live dir>/handovers/<stamp>-<modes>.json.gz`` listing the source run's keys that have NO terminal outcome there and are
    not yet live keys, by mode:
      in_flight  their batch is POSTed and not harvested (submitted / submit_unknown): the FALLBACK when those batches have not come
                 back by ``not_before`` (refused before it). A late batch result of such a key is a duplicate the filing resolves:
                 the first fileable outcome stands (``file_records``)
      reasks     their batch is harvested and the key waits on a re-ask that a stopped batch harvester will never build, or their batch
                 is pending but was built after the init
    Each key carries its batch (the body it was last sent, re-sent as the batch's re-ask would be) and its prior outcomes. Read only on
    the source and on the live ledger; the engine ingests the file (:meth:`LiveRunner.ingest_handovers`). Returns the path, or None when
    no key qualifies."""
    live_dir = os.path.abspath(os.path.expanduser(live_dir))
    modes = sorted(set(modes))
    if not modes or any(m not in ("in_flight", "reasks") for m in modes):
        raise SystemExit(f"handover modes must be in_flight and / or reasks, not {modes}")
    t = time.time() if now is None else now
    if not_before:
        nb = time.mktime(time.strptime(not_before.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")) - time.timezone
        if t < nb:
            raise SystemExit(f"REFUSED: not before {not_before} (now {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))}); nothing written")
    spec = BF.read_track(live_dir)
    src = spec["source_run_dir"]
    lcon = _ro(os.path.join(live_dir, "state.sqlite"))
    try:
        existing = {r[0] for r in lcon.execute("SELECT custom_id FROM live_keys")}
    finally:
        lcon.close()
    con = _ro(os.path.join(src, "state.sqlite"))
    try:
        states = {r["batch_key"]: r["state"] for r in con.execute("SELECT batch_key, state FROM batches")}
        keys: List[Dict[str, Any]] = []
        skipped: collections.Counter = collections.Counter()
        cand = []
        for r in con.execute("SELECT custom_id, batch_key, outcome FROM requests WHERE terminal = 0"):
            cid, bk = r["custom_id"], r["batch_key"]
            st = states.get(bk)
            if cid in existing:
                skipped["already_live"] += 1
                continue
            if st in SOURCE_INFLIGHT_STATES:
                origin = "in_flight" if "in_flight" in modes else None
            elif st == "harvested" or st in SOURCE_PENDING_STATES:
                origin = ("reask" if st == "harvested" else "pending") if "reasks" in modes else None
            else:
                origin = None
            if origin is None:
                skipped[f"state_{st}"] += 1
                continue
            cand.append((cid, bk, st, origin))
        prior: Dict[str, List[str]] = collections.defaultdict(list)
        if cand:
            want = {c[0] for c in cand}
            for a in con.execute("SELECT custom_id, outcome FROM attempts ORDER BY custom_id, attempt"):
                if a[0] in want:
                    prior[a[0]].append(str(a[1]))
        for cid, bk, st, origin in cand:
            keys.append({"custom_id": cid, "source_batch_key": bk, "source_state": st, "origin": origin, "prior_outcomes": prior.get(cid, [])})
    finally:
        con.close()
    if not keys:
        log(f"[handover] no key qualifies ({modes}); skipped {dict(skipped)}")
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t))
    hdir = os.path.join(live_dir, "handovers")
    os.makedirs(hdir, exist_ok=True)
    hid = f"{stamp}-{'+'.join(modes)}"
    j = 0
    while os.path.exists(os.path.join(hdir, hid + ".json.gz")):      # two handovers in one second
        j += 1
        hid = f"{stamp}-{'+'.join(modes)}-{j}"
    doc = {"schema": HANDOVER_SCHEMA, "handover_id": hid, "created_utc": utc_now(), "source_run_dir": src, "modes": modes, "reason": reason,
           "not_before": not_before, "source_fingerprint": _source_fingerprint(src), "skipped": dict(skipped),
           "counts": dict(collections.Counter(k["origin"] for k in keys)), "keys": keys}
    path = os.path.join(hdir, hid + ".json.gz")
    with gzip.open(path + ".tmp", "wt") as fh:
        json.dump(doc, fh)
    os.replace(path + ".tmp", path)
    log(f"[handover] {path}: {len(keys)} keys {doc['counts']} (skipped {dict(skipped)})")
    return path


def ingest_handovers(led: Ledger, live_dir: str, log: Callable[[str], None] = print) -> int:
    """Register the keys of every handover file not yet ingested (exactly once per file: ``live_handovers``); a key already live is
    skipped. Each referenced source batch is read (sha256-checked) for the key's body sha256 and estimates. Returns keys added."""
    hdir = os.path.join(live_dir, "handovers")
    if not os.path.isdir(hdir):
        return 0
    done = {r[0] for r in led.conn.execute("SELECT handover_id FROM live_handovers")}
    spec = BF.read_track(live_dir)
    base_cap = int((spec.get("request") or {}).get("max_tokens") or 0)
    added_total = 0
    for f in sorted(os.listdir(hdir)):
        if not f.endswith(".json.gz"):
            continue
        hid = f[:-len(".json.gz")]
        if hid in done:
            continue
        path = os.path.join(hdir, f)
        raw = open(path, "rb").read()
        doc = json.loads(gzip.decompress(raw).decode("utf-8"))
        if doc.get("schema") != HANDOVER_SCHEMA or doc.get("handover_id") != hid:
            log(f"[handover] {path}: not a handover file of this schema; ignored")
            continue
        src = doc["source_run_dir"]
        if os.path.abspath(src) != os.path.abspath(spec["source_run_dir"]):
            raise SystemExit(f"REFUSED: handover {hid} names source {src}, not this live run's {spec['source_run_dir']}")
        by_batch: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
        for k in doc["keys"]:
            if led.conn.execute("SELECT 1 FROM live_keys WHERE custom_id = ?", (k["custom_id"],)).fetchone() is None:
                by_batch[k["source_batch_key"]].append(k)
        try:
            added = _ingest_file(led, hid, path, raw, doc, src, by_batch, base_cap)
        except BaseException:
            led.conn.rollback()                           # a refused file registers none of its keys
            raise
        led.event("HANDOVER", f"{hid}: {added} of {len(doc['keys'])} keys added ({doc.get('counts')}); reason: {doc.get('reason')}")
        log(f"[handover] ingested {hid}: {added} of {len(doc['keys'])} keys added")
        added_total += added
    return added_total


def _ingest_file(led: Ledger, hid: str, path: str, raw: bytes, doc: Mapping[str, Any], src: str,
                 by_batch: Mapping[str, List[Mapping[str, Any]]], base_cap: int) -> int:
    """One handover file's keys into the live ledger, in one transaction (committed here; rolled back by the caller on a refusal)."""
    now = utc_now()
    added = 0
    for bk, ks in sorted(by_batch.items()):
        man = BF.read_manifest(src, bk)
        reqs = {r["custom_id"]: r["body"] for r in BF.body_requests(BF.read_body(src, man))}
        est = man.get("est") or {}
        in_per_byte = float(est.get("input_tokens") or 0) / max(1, int(man["body_bytes"]))
        out_per_req = float(est.get("output_tokens") or 0) / max(1, int(man["n_requests"]))
        for k in ks:
            cid = k["custom_id"]
            body = reqs.get(cid)
            if body is None:
                raise SystemExit(f"REFUSED: handover {hid}: key {cid[:16]}... is not in source batch {bk}")
            cap = int(body.get(cap_key(body)) or 0)
            led.conn.execute("INSERT INTO requests(custom_id, batch_key) VALUES (?, ?) ON CONFLICT(custom_id) DO NOTHING", (cid, f"src:{bk}"))
            led.conn.execute("INSERT INTO live_keys(custom_id, source_run_dir, source_batch_key, origin, handover_id, prior_outcomes_json, "
                             "body_sha256, est_input_tokens, est_output_tokens, added_utc) VALUES (?,?,?,?,?,?,?,?,?,?) "
                             "ON CONFLICT(custom_id) DO NOTHING",
                             (cid, src, bk, k["origin"], hid, json.dumps(k.get("prior_outcomes") or []), body_sha256(body),
                              len(BF.serialize_request(cid, body)) * in_per_byte,
                              float(cap) if base_cap and cap > base_cap else out_per_req, now))
            added += 1
    led.conn.execute("INSERT INTO live_handovers(handover_id, path, sha256, n_keys, n_added, ingested_utc, reason) VALUES (?,?,?,?,?,?,?)",
                     (hid, path, hashlib.sha256(raw).hexdigest(), len(doc["keys"]), added, now, doc.get("reason")))
    led.conn.commit()
    return added


# =====================================================================================================================================
# the protocol of recorded attempts: the filing's check, and re-opening what must not be filed
# =====================================================================================================================================
#: outcomes a served completion produces that a filing would file (the infrastructure ones and the uncounted ones are never filed)
SERVED_OUTCOMES = ("verdict", "moderation", "judge_refusal", "parse_error", "truncated")


def live_settings(live_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """``(track.json, the live settings with the defaults, the providers the routing allows)`` of a live run directory."""
    spec = BF.read_track(live_dir)
    cfg = dict(LIVE_DEFAULTS, **(spec.get("live") or {}))
    return spec, cfg, allowed_providers(cfg.get("providers") or {}, cfg.get("routing") or {})


def attempt_violation(cfg: Mapping[str, Any], model: str, allowed: Set[str], row: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """:func:`protocol_violation` of a RECORDED live attempt (a ``live_attempts`` row as a mapping), judged by the routing as it stands
    now; None for an attempt no provider served or whose outcome a filing never files. ``served`` is the engine's own record (a
    completion: the rows it checked itself); a row from before that column counts as served when it is HTTP 200."""
    served = row.get("served")
    if served is None:
        served = row.get("http_status") == 200
    if not served or row.get("outcome") not in SERVED_OUTCOMES:
        return None
    usage = {"prompt_tokens": row.get("prompt_tokens") or 0, "completion_tokens": row.get("completion_tokens") or 0,
             "prompt_tokens_details": {"cached_tokens": row.get("cached_tokens") or 0, "cache_write_tokens": row.get("cache_write_tokens") or 0}}
    return protocol_violation(cfg, model, allowed, row.get("provider"), row.get("served_model"), usage, row.get("reasoning_tokens"),
                              row.get("cost_usd"), row.get("cost_source"))


def reopen_live(live_dir: str, providers: Sequence[str] = (), off_protocol: bool = False, reason: str = "", apply: bool = False,
                log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Re-open TERMINAL live outcomes so the engine judges those keys again: ``providers`` selects the keys whose terminal attempt one of
    the named providers served (a provider dropped after the pilot), ``off_protocol`` the keys whose terminal attempt breaks the
    protocol as ``track.json``'s routing now stands (:func:`attempt_violation`; the filing refuses those). For each selected key every
    live attempt the selector matches, the terminal one included, is re-classified ``held`` (uncounted, so it spends none of the key's
    re-asks; ``attempts.error_kind`` reopened; the original outcome kept in ``live_attempts.reopened_json`` and in ``live_reopens``),
    and the key's outcome becomes held, not terminal, its verdict cleared. Spend is untouched (those replies were billed). The next
    engine start re-sends each such key with the body its last attempt was sent with. Without ``apply`` it only counts (read only);
    with it, it takes the live run's writer lock, so it is refused while an engine runs."""
    live_dir = os.path.abspath(os.path.expanduser(live_dir))
    if not is_live_dir(live_dir):
        raise SystemExit(f"{live_dir} is not a live run directory")
    provs = set(providers)
    if not provs and not off_protocol:
        raise SystemExit("name --provider and / or --off-protocol")
    spec, cfg, allowed = live_settings(live_dir)
    model = str(spec["model"])
    selector = "+".join(([f"provider={','.join(sorted(provs))}"] if provs else []) + (["off_protocol"] if off_protocol else []))

    def matches(row: Mapping[str, Any]) -> Optional[str]:
        if row.get("outcome") == "held":
            return None
        if provs and row.get("provider") in provs:
            return "provider"
        if off_protocol:
            v = attempt_violation(cfg, model, allowed, row)
            if v:
                return v[0]
        return None

    if apply:
        try:
            led = Ledger(live_dir).writer()
        except Exception as e:  # noqa: BLE001 -- WriterLocked: an engine (or another tool) holds the live run
            raise SystemExit(f"REFUSED: {e}; stop the engine first (SIGTERM, or touch STOP), then re-open")
        ensure_schema(led.conn)
        con = led.conn
    else:
        led = None
        con = _ro(os.path.join(live_dir, "state.sqlite"))
    try:
        cols = attempt_columns(con)
        sel_cols = [c for c in ("attempt", "outcome", "terminal", "http_status", "served", "provider", "served_model", "reasoning_tokens",
                                "cost_usd", "cost_source", "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens", "violation")
                    if c in cols]
        con.row_factory = sqlite3.Row
        picked: List[Tuple[Dict[str, Any], str]] = []
        for row in con.execute(f"SELECT r.custom_id AS cid, r.verdict_json, r.error_json, {', '.join('la.' + c for c in sel_cols)} FROM "
                               f"requests r JOIN live_attempts la ON la.custom_id = r.custom_id AND la.attempt = r.attempts WHERE r.terminal = 1"):
            d = dict(row)
            m = matches(d)
            if m:
                picked.append((d, m))
        counts = collections.Counter(f"{m}:{r['provider']}" for r, m in picked)
        out = {"live_dir": live_dir, "selector": selector, "reason": reason, "keys": len(picked), "by_match": dict(counts),
               "outcomes": dict(collections.Counter(r["outcome"] for r, _ in picked)), "sample": [r["cid"][:16] for r, _ in picked[:5]],
               "applied": False, "attempts_reclassified": 0}
        if not apply or not picked:
            log(f"[reopen] {'would re-open' if not apply else 're-opened'} {len(picked)} terminal keys ({selector}): {dict(counts)}")
            return out
        now = utc_now()
        n_att = 0
        for r, _m in picked:
            cid = r["cid"]
            rows = con.execute(f"SELECT {', '.join(sel_cols)} FROM live_attempts WHERE custom_id = ? ORDER BY attempt", (cid,)).fetchall()
            for a in rows:
                d = dict(a)
                m = matches(d)
                if not m:
                    continue
                orig = {"outcome": d["outcome"], "terminal": d["terminal"], "provider": d.get("provider"), "match": m, "selector": selector,
                        "reason": reason, "ts": now}
                con.execute("UPDATE live_attempts SET outcome = 'held', terminal = 0, violation = COALESCE(violation, ?), reopened_json = ? "
                            "WHERE custom_id = ? AND attempt = ?", (REOPENED_KIND if m == "provider" else m, json.dumps(orig), cid, d["attempt"]))
                con.execute("UPDATE attempts SET outcome = 'held', error_kind = ? WHERE custom_id = ? AND attempt = ?", (REOPENED_KIND, cid, d["attempt"]))
                con.execute("INSERT INTO live_reopens(ts, custom_id, attempt, selector, reason, original_json) VALUES (?,?,?,?,?,?)",
                            (now, cid, d["attempt"], selector, reason, json.dumps(dict(d, verdict_json=r["verdict_json"], error_json=r["error_json"])
                                                                            if d["attempt"] == r["attempt"] else d)))
                n_att += 1
            err = {"kind": REOPENED_KIND, "selector": selector, "reason": reason, "reopened_utc": now, "original_outcome": r["outcome"],
                   "original_verdict": json.loads(r["verdict_json"]) if r["verdict_json"] else None,
                   "original_error": json.loads(r["error_json"]) if r["error_json"] else None, "provider": r["provider"]}
            con.execute("UPDATE requests SET outcome = 'held', terminal = 0, verdict_json = NULL, error_json = ? WHERE custom_id = ?",
                        (json.dumps(err), cid))
        con.commit()
        led.event("REOPEN", f"{len(picked)} terminal keys re-opened, {n_att} attempts re-classified held ({selector}): {reason}"[:4000])
        out.update({"applied": True, "attempts_reclassified": n_att})
        log(f"[reopen] re-opened {len(picked)} terminal keys ({selector}; {n_att} attempts re-classified held): {dict(counts)}")
        return out
    finally:
        if led is not None:
            led.close()
        else:
            con.close()


def parked(con: sqlite3.Connection, cfg: Mapping[str, Any], allowed: Set[str]) -> Dict[str, Dict[str, Any]]:
    """custom_id -> the PARKED keys of a live ledger: no terminal outcome, the latest attempt held for billed reasoning, and more such
    attempts than ``max_reasoning_resends`` (the engine's parking rule), with the providers that billed it and the allowed endpoints that
    have not (what ``reasoning_reroute`` would still try)."""
    if not has_table(con, "live_attempts"):
        return {}
    max_rs = int(cfg.get("max_reasoning_resends") or 0)
    out: Dict[str, Dict[str, Any]] = {}
    q = ("SELECT r.custom_id, (SELECT COUNT(*) FROM live_attempts a WHERE a.custom_id = r.custom_id AND a.violation = 'reasoning_billed'), "
         "(SELECT GROUP_CONCAT(DISTINCT a.provider) FROM live_attempts a WHERE a.custom_id = r.custom_id AND a.violation = 'reasoning_billed') "
         "FROM requests r JOIN live_attempts la ON la.custom_id = r.custom_id AND la.attempt = r.attempts "
         "WHERE r.terminal = 0 AND la.violation = 'reasoning_billed'")
    for cid, n, provs in con.execute(q):
        if int(n) > max_rs:
            billed = sorted(p for p in str(provs or "").split(",") if p)
            out[cid] = {"reasoning_replies": int(n), "billed_by": billed, "not_yet_tried": sorted(allowed - set(billed))}
    return out


def settle_parked(live_dir: str, reason: str = "", apply: bool = False, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """A person's decision to leave the PARKED keys (:func:`parked`) unjudged: each is made terminal as it stands (outcome ``held``,
    ``error.finalized`` names why), so a filing reads it as an infrastructure outcome (no row; the unit stays unjudged and its judgment
    ``partial``) and the engine never sends it again. Spend is untouched. Without ``apply`` it only counts (read only); with it, it takes
    the live run's writer lock, so it is refused while an engine runs. Restart the engine first where ``not_yet_tried`` is not empty: with
    ``reasoning_reroute`` its next start re-sends those keys to the endpoints that have not billed reasoning on them."""
    live_dir = os.path.abspath(os.path.expanduser(live_dir))
    if not is_live_dir(live_dir):
        raise SystemExit(f"{live_dir} is not a live run directory")
    _spec, cfg, allowed = live_settings(live_dir)
    if apply:
        try:
            led = Ledger(live_dir).writer()
        except Exception as e:  # noqa: BLE001 -- WriterLocked: an engine (or another tool) holds the live run
            raise SystemExit(f"REFUSED: {e}; stop the engine first (SIGTERM, or touch STOP), then settle")
        ensure_schema(led.conn)
        con = led.conn
    else:
        led = None
        con = _ro(os.path.join(live_dir, "state.sqlite"))
    try:
        pk = parked(con, cfg, allowed)
        out = {"live_dir": live_dir, "reason": reason, "keys": len(pk), "reroutable": sum(1 for v in pk.values() if v["not_yet_tried"]),
               "by_billed_providers": dict(collections.Counter(",".join(v["billed_by"]) for v in pk.values())),
               "sample": {k[:16]: v for k, v in list(sorted(pk.items()))[:5]}, "applied": False}
        if not apply or not pk:
            log(f"[settle] {'would settle' if not apply else 'settled'} {len(pk)} parked keys ({out['reroutable']} still reroutable)")
            return out
        now = utc_now()
        for cid, v in sorted(pk.items()):
            led.finalize(cid, f"settled {now} by a person's decision: reasoning tokens billed on {v['reasoning_replies']} replies (by "
                              f"{', '.join(v['billed_by'])}); left unjudged{': ' + reason if reason else ''}", commit=False)
        con.commit()
        led.event("SETTLE", f"{len(pk)} parked keys made terminal unjudged ({out['by_billed_providers']}): {reason}"[:4000])
        out["applied"] = True
        log(f"[settle] settled {len(pk)} parked keys: terminal, never filed, never sent again")
        return out
    finally:
        if led is not None:
            led.close()
        else:
            con.close()


def last_run(live_dir: str) -> Dict[str, Any]:
    """The live ledger's ``run_state`` (read only): how the last engine run ended (``end_state``, ``exit_code``, ``limit``, ...)."""
    path = os.path.join(live_dir, "state.sqlite")
    if not os.path.exists(path):
        return {}
    con = _ro(path)
    try:
        return meta_get(con, "run_state") or {}
    finally:
        con.close()


def auto_start_refusal(live_dir: str, allow_parked: bool = False) -> Optional[str]:
    """Why no person-less start (a scheduled restart after a handover) may start the FULL engine now; None when it may. Only a full
    run (no ``--limit``) that ended ``done`` with a clean exit qualifies: after a pilot, a deliberate stop (exit 3: a signal, ``STOP``,
    ``--max-hours``), a pause, an unfinished end or a crash, starting the unlimited run is a person's decision. With ``allow_parked`` a
    full run that ended ``unfinished`` cleanly with nothing left but PARKED keys (``run_state`` ``non_terminal_keys`` ==
    ``parked_keys``: keys that billed reasoning on every try, which a start re-sends once each) qualifies too."""
    if _paused(live_dir):
        return f"{live_dir}/PAUSED exists ({str(_paused(live_dir)).splitlines()[0][:200]})"
    if os.path.exists(os.path.join(live_dir, STOP_FILE)):
        return f"{live_dir}/STOP exists"
    rs = last_run(live_dir)
    if not rs:
        return "no engine run is on record (run the pilot by hand first)"
    if rs.get("limit") is not None:
        return f"the last run was a --limit {rs.get('limit')} pilot ({rs.get('end_state')}); the full run is started by a person"
    if not rs.get("end_state"):
        return f"the last run (pid {rs.get('pid')}, started {rs.get('started_utc')}) has not recorded an end: still running, or killed"
    if (allow_parked and rs.get("end_state") == "unfinished" and rs.get("clean_exit") and int(rs.get("parked_keys") or 0) > 0
            and rs.get("non_terminal_keys") == rs.get("parked_keys")):
        return None
    if rs.get("end_state") not in AUTO_START_END_STATES or not rs.get("clean_exit"):
        return (f"the last full run ended {rs.get('end_state')} (exit {rs.get('exit_code')}, clean_exit {rs.get('clean_exit')}); resuming after "
                f"a stop, a pause, an unfinished end or a crash is a person's decision")
    return None


# =====================================================================================================================================
# spend: this ledger and every other OpenRouter ledger
# =====================================================================================================================================
def live_spend(conn: sqlite3.Connection) -> Dict[str, float]:
    """USD WITHOUT the fee recorded by a live ledger: reported (or computed) costs of its attempts, the estimates of attempts whose billing
    is unknown, and the adjustments (requests in flight when a process ended uncleanly)."""
    if not has_table(conn, "live_attempts"):
        return {"recorded": 0.0, "unknown_estimated": 0.0, "adjustments": 0.0, "total": 0.0}
    rec = float(conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM live_attempts WHERE cost_usd IS NOT NULL").fetchone()[0])
    unk = float(conn.execute("SELECT COALESCE(SUM(est_cost_usd), 0) FROM live_attempts WHERE cost_usd IS NULL AND cost_source = 'unknown_estimated'").fetchone()[0])
    adj = float(conn.execute("SELECT COALESCE(SUM(usd_no_fee), 0) FROM live_spend_adjustments").fetchone()[0]) if has_table(conn, "live_spend_adjustments") else 0.0
    return {"recorded": rec, "unknown_estimated": unk, "adjustments": adj, "total": rec + unk + adj}


def openrouter_spend(root: str, exclude: Iterable[str] = (), prune: Iterable[str] = PRUNE_DIRS) -> Dict[str, Any]:
    """OpenRouter spend of every re-judge ledger under ``root`` except the excluded run directories, USD INCLUDING the credit fee: per
    distinct batch id, the reported cost x (1 + fee), or the manifest estimate of a batch in flight; per live ledger, its recorded spend
    x (1 + fee) plus the in-flight estimate its running engine last wrote. Claude (Anthropic Batch API) ledgers are skipped."""
    excl = [os.path.realpath(os.path.expanduser(e)) for e in exclude]
    prune = frozenset(prune)
    per_batch: Dict[str, float] = {}
    per_dir: Dict[str, float] = {}
    live_usd = 0.0
    errors: List[str] = []
    for dirpath, dirnames, filenames in os.walk(os.path.expanduser(root)):
        dirnames[:] = [d for d in dirnames if d not in prune and not d.startswith("scratch_registry")]
        if "state.sqlite" not in filenames:
            continue
        rp = os.path.realpath(dirpath)
        if any(rp == e or rp.startswith(e + os.sep) for e in excl):
            continue
        try:
            tr = json.load(open(os.path.join(dirpath, "track.json")))
        except (OSError, ValueError):
            tr = {}
        model = str(tr.get("model") or "")
        if model.startswith("claude") or model.startswith("anthropic/"):
            continue
        try:
            con = sqlite3.connect(f"file:{os.path.join(dirpath, 'state.sqlite')}?mode=ro", uri=True, timeout=30)
            try:
                rows = con.execute("SELECT b.batch_id, b.state, b.cost, b.est_cost, (SELECT SUM(r.cost) FROM requests r WHERE r.batch_id = b.batch_id) "
                                   "FROM batches b WHERE b.batch_id IS NOT NULL AND b.batch_id != ''").fetchall()
                lv = live_spend(con)["total"] * (1.0 + BF.CREDIT_FEE)
                inflight = meta_get(con, "run_state") or {}
                if inflight and not inflight.get("clean_exit"):
                    lv += float(inflight.get("inflight_est_with_fee") or 0.0)
            finally:
                con.close()
        except sqlite3.Error as e:
            errors.append(f"{dirpath}: {e}")
            continue
        tot = 0.0
        for bid, st, cost, est, req_cost in rows:
            if st in SOURCE_INFLIGHT_STATES:
                usd = float(est or 0.0)
            else:
                rep = cost if cost is not None else req_cost
                usd = float(rep or 0.0) * (1.0 + BF.CREDIT_FEE)
            per_batch[bid] = max(per_batch.get(bid, 0.0), usd)
            tot += usd
        tot += lv
        live_usd += lv
        per_dir[os.path.relpath(dirpath, os.path.expanduser(root))] = round(tot, 4)
    return {"usd": round(sum(per_batch.values()) + live_usd, 4), "batches": len(per_batch), "live_usd": round(live_usd, 4),
            "per_dir": {k: v for k, v in sorted(per_dir.items()) if v}, "errors": errors}


# =====================================================================================================================================
# the engine
# =====================================================================================================================================
@dataclasses.dataclass
class WorkItem:
    custom_id: str
    body: Dict[str, Any]
    source_batch_key: str
    origin: str
    prior: List[str]
    est_in: float
    est_out: float
    body_sha: str = ""
    new_key: bool = True                    # no live attempt yet
    finalize: Optional[str] = None          # a cut reply whose cap cannot be raised: made terminal, never sent
    #: a narrower provider routing for this key (``reasoning_reroute``: the run's routing with ``only`` cut to the allowed endpoints that
    #: have not billed reasoning on it); None sends the run's routing
    routing: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class Result:
    item: WorkItem
    parsed: Dict[str, Any]
    http_status: Optional[int]
    started_utc: str
    finished_utc: str
    latency_s: float
    provider: Optional[str] = None
    served_model: Optional[str] = None
    generation_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    cost_usd: Optional[float] = None
    cost_source: str = "none"
    est_usd: float = 0.0
    transport_retries: int = 0
    rate_limited: int = 0
    reasoning_reply: bool = False           # reasoning TOKENS billed (the protocol turns reasoning off)
    reasoning_tokens: int = 0
    reasoning_chars: int = 0                # length of a ``message.reasoning`` text (the batch provider returns one on ~3% of replies)
    served: bool = False                    # HTTP 200 with ``choices``: a provider served it, so the protocol is checked
    violation: Optional[Tuple[str, str]] = None     # (kind, message): off-protocol, recorded ``held`` (:func:`protocol_violation`)


class AdaptiveLimit:
    """The concurrency limit: ``acquire`` waits while ``active >= limit``. Persistent rate limiting halves it (``halve_after`` 429 / 529
    replies within ``window_s``, at most once per ``spacing_s``, never below ``minimum``); ``recover`` adds a tenth of the target back after
    ``recover_s`` without one. ``set_target`` (the ``CONCURRENCY`` file) moves both."""

    def __init__(self, target: int, minimum: int, window_s: float, halve_after: int, spacing_s: float, recover_s: float,
                 clock: Callable[[], float] = time.monotonic):
        self.target = int(target)
        self.minimum = max(1, int(minimum))
        self.limit = int(target)
        self.window_s, self.halve_after, self.spacing_s, self.recover_s = float(window_s), int(halve_after), float(spacing_s), float(recover_s)
        self.clock = clock
        self.active = 0
        self.recent: Deque[float] = collections.deque()
        self.last_change = clock() - self.spacing_s        # the first halving needs no wait (a storm at start is halved at once)
        self.last_rl: Optional[float] = None
        self.halvings = 0
        self.n_rate_limited: collections.Counter = collections.Counter()
        self._cond: Optional[asyncio.Condition] = None

    def _c(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    async def acquire(self) -> None:
        c = self._c()
        async with c:
            await c.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def release(self) -> None:
        c = self._c()
        async with c:
            self.active -= 1
            c.notify_all()

    async def _notify(self) -> None:
        c = self._c()
        async with c:
            c.notify_all()

    def on_rate_limited(self, status: int) -> Optional[str]:
        now = self.clock()
        self.n_rate_limited[int(status)] += 1
        self.last_rl = now
        self.recent.append(now)
        while self.recent and now - self.recent[0] > self.window_s:
            self.recent.popleft()
        if len(self.recent) >= self.halve_after and now - self.last_change >= self.spacing_s and self.limit > self.minimum:
            old = self.limit
            self.limit = max(self.minimum, self.limit // 2)
            self.last_change = now
            self.halvings += 1
            self.recent.clear()
            return f"rate limited {self.halve_after}+ times in {self.window_s:.0f}s: concurrency {old} -> {self.limit}"
        return None

    def recover(self) -> Optional[str]:
        now = self.clock()
        if self.limit >= self.target:
            return None
        if (self.last_rl is None or now - self.last_rl >= self.recover_s) and now - self.last_change >= self.recover_s:
            old = self.limit
            self.limit = min(self.target, self.limit + max(1, self.target // 10))
            self.last_change = now
            return f"no rate limiting for {self.recover_s:.0f}s: concurrency {old} -> {self.limit}"
        return None

    def set_target(self, n: int) -> Optional[str]:
        n = max(self.minimum, int(n))
        if n == self.target:
            return None
        old = (self.target, self.limit)
        self.target, self.limit = n, n
        self.last_change = self.clock()
        return f"target concurrency {old[0]} -> {n} (limit was {old[1]})"


class LiveRunner:
    """One live run directory's engine (module docstring). ``run_async`` returns the exit code."""

    def __init__(self, live_dir: str, overrides: Optional[Mapping[str, Any]] = None, limit: Optional[int] = None,
                 log: Callable[[str], None] = print, key_env: str = OR.KEY_ENV, spend_walker: Optional[Callable[..., Dict[str, Any]]] = None,
                 max_spend_usd: Optional[float] = None):
        self.live_dir = os.path.abspath(os.path.expanduser(live_dir))
        self.spec = BF.read_track(self.live_dir)
        if self.spec.get("backend") != BACKEND:
            raise SystemExit(f"{self.live_dir} is not a live run directory (track.json backend {self.spec.get('backend')!r})")
        self.cfg: Dict[str, Any] = dict(LIVE_DEFAULTS, **(self.spec.get("live") or {}))
        self.cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
        self.max_spend = float(max_spend_usd if max_spend_usd is not None else self.spec.get("max_spend_usd") or 0.0)
        if self.max_spend <= 0:
            raise SystemExit("max_spend_usd must be positive (track.json or --max-spend-usd)")
        missing = [k for k in REQUIRED_LIVE if self.cfg.get(k) in (None, "")]
        if missing:
            raise SystemExit(f"REFUSED: {self.live_dir}/track.json live states no {' / '.join(missing)} (init it with --global-root and "
                             f"--global-ceiling-usd)")
        self.model = str(self.spec["model"])
        if FORBIDDEN.search(self.model) or FORBIDDEN.search(json.dumps(self.cfg.get("routing"))):
            raise SystemExit("REFUSED: a Claude / Anthropic name in the model or routing")
        self.routing = dict(self.cfg["routing"])
        self.providers: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in (self.cfg.get("providers") or {}).items()}
        self.allowed = allowed_providers(self.providers, self.routing)
        if not self.allowed:
            raise SystemExit(f"REFUSED: the routing {json.dumps(self.routing)} allows none of live.providers {sorted(self.providers)}")
        self.price_max = max_price(self.providers)
        self.source = str(self.spec["source_run_dir"])
        self.parser = load_parser(str(self.spec["parser"]))
        self.retry = dict(DEFAULT_RETRY, **(self.spec.get("retry") or {}))
        self.health_cfg = dict(HEALTH_DEFAULTS, **(self.spec.get("health") or {}))
        self.raise_to = int(self.spec.get("truncated_retry_max_tokens") or 0)
        self.limit_keys = limit
        self.log_fn = log
        self.key_env = key_env
        self.spend_walker = spend_walker or openrouter_spend
        self.url = str(self.cfg["base_url"]).rstrip("/") + CHAT_PATH
        self.ledger: Optional[Ledger] = None
        # run state
        self.inflight: Set[str] = set()
        self.inflight_est = 0.0                          # USD without the fee, unscaled estimates of the requests in flight
        self.window: List[Result] = []
        self.window_t0 = time.monotonic()
        self.reasks: Deque[WorkItem] = collections.deque()
        self.cur: List[WorkItem] = []
        self.plan: List[Tuple[str, Optional[Set[str]]]] = []     # source batch keys still to load (with the keys allowed, for a pilot)
        self.prefetch: Optional[asyncio.Task] = None
        self.stopping: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.cooldown_until = 0.0
        self.cooldown_n = 0
        self.spent = 0.0                                  # USD without the fee, recorded + completed-not-yet-recorded
        self.cost_reported = 0.0
        self.est_reported = 0.0
        self.terminal_keys = 0
        self.total_keys = 0
        self.sent_keys = 0
        self.new_keys_allowed: Optional[int] = None
        self.completions: Deque[Tuple[float, float, float]] = collections.deque()     # (t, usd no fee, latency)
        self.terminal_ts: Deque[float] = collections.deque()
        self.provider_stats: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
        self.violations: collections.Counter = collections.Counter()     # off-protocol replies held, by kind (this run)
        self.parked: Set[str] = set()                    # keys held for reasoning max_reasoning_resends times: not re-sent by this run
        self.rerouted: Dict[str, List[str]] = {}          # keys re-sent to a narrowed routing (reasoning_reroute): the endpoints left
        self.rl_sleep_n = 0
        self.rl_sleep_min_s: Optional[float] = None
        self._finished = False                           # _finish ran: every request in flight was recorded
        self._flushing: List[Result] = []
        self._end: Dict[str, Any] = {}
        self.global_last: Dict[str, Any] = {}
        self.global_t = 0.0
        self.workers: List[asyncio.Task] = []
        self.lock: Optional[asyncio.Lock] = None
        self.flush_event: Optional[asyncio.Event] = None
        self.t_start = time.time()
        self.limiter = AdaptiveLimit(int(self.cfg["concurrency"]), int(self.cfg["min_concurrency"]), float(self.cfg["rl_window_s"]),
                                     int(self.cfg["rl_halve_after"]), float(self.cfg["rl_halve_spacing_s"]), float(self.cfg["rl_recover_s"]))
        self._session: Any = None
        self._outcomes: Optional[Dict[str, int]] = None
        self._headers: Dict[str, str] = {}
        self._logf = open(os.path.join(self.live_dir, "live.log"), "a")

    # -- logging / pausing ------------------------------------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{utc_now()} {msg}"
        try:
            self.log_fn(line)
        except Exception:  # noqa: BLE001
            pass
        self._logf.write(line + "\n")
        self._logf.flush()

    def event(self, kind: str, msg: str) -> None:
        if self.ledger is not None:
            self.ledger.event(kind, msg)
        self.log(f"[{kind}] {msg}")

    def stop(self, reason: str, code: int) -> None:
        """Stop sending new requests (the requests in flight finish and are recorded); the first reason and code stand."""
        if self.stopping is None:
            self.stopping = reason
            self.exit_code = code
            self.log(f"[live] stopping ({code}): {reason}")

    def pause(self, reason: str) -> None:
        path = os.path.join(self.live_dir, PAUSE_FILE)
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write(f"{utc_now()} {reason}\nRemove this file and start the engine again to resume (the ledger is the state).\n")
            if self.ledger is not None:
                self.ledger.event("PAUSED", reason[:4000])
        self.stop(f"PAUSED: {reason}", EXIT_PAUSED)

    # -- spend ----------------------------------------------------------------------------------------------------------------------------
    def cost_ratio(self) -> float:
        return (self.cost_reported / self.est_reported) if self.est_reported > 0 else 1.0

    def inflight_with_fee(self) -> float:
        return self.inflight_est * max(1.0, self.cost_ratio()) * (1.0 + BF.CREDIT_FEE)

    def spent_with_fee(self) -> float:
        return self.spent * (1.0 + BF.CREDIT_FEE)

    def item_est(self, item: WorkItem) -> float:
        return item_estimate(item.est_in, item.est_out, self.price_max)

    def _unknown_spend(self, usd_no_fee: float, reason: str) -> None:
        """Billing that cannot be known (a request abandoned while it may have run): its estimate counts as spent, durably."""
        self.spent += usd_no_fee
        if self.ledger is not None:
            self.ledger.conn.execute("INSERT INTO live_spend_adjustments(ts, usd_no_fee, reason) VALUES (?,?,?)", (utc_now(), usd_no_fee, reason))
            self.ledger.conn.commit()

    def budget_ok(self, item: WorkItem) -> bool:
        nxt = self.item_est(item) * max(1.0, self.cost_ratio()) * (1.0 + BF.CREDIT_FEE)
        total = self.spent_with_fee() + self.inflight_with_fee() + nxt
        if total > self.max_spend:
            self.pause(f"live spend ceiling: spent ${self.spent_with_fee():,.2f} + in flight ${self.inflight_with_fee():,.2f} + next request "
                       f"${nxt:,.4f} = ${total:,.2f} would pass max_spend_usd ${self.max_spend:,.2f} (USD incl. the {BF.CREDIT_FEE:.1%} fee; "
                       f"{self.terminal_keys} of {self.total_keys} keys terminal). Raise max_spend_usd in {self.live_dir}/track.json (or "
                       f"--max-spend-usd) within the global ceiling, remove PAUSED and start again")
            return False
        return True

    def recent_rate_usd_per_s(self, horizon_s: float = 300.0) -> float:
        now = time.time()
        usd = sum(c for t, c, _ in self.completions if now - t <= horizon_s)
        span = min(horizon_s, max(1.0, now - self.t_start))
        return usd * (1.0 + BF.CREDIT_FEE) / span

    def global_other(self) -> Dict[str, Any]:
        """Every other OpenRouter ledger's spend (:func:`openrouter_spend`; read only, runs in a thread)."""
        return self.spend_walker(os.path.expanduser(str(self.cfg["global_root"])), exclude=[self.live_dir])

    def global_eval(self, other: Mapping[str, Any]) -> Dict[str, Any]:
        """The other ledgers + this run's spend and in-flight estimate + this run's spend projected until the check after next; pauses at
        ``global_ceiling_usd``."""
        projected = self.recent_rate_usd_per_s() * 2 * float(self.cfg["global_check_s"])
        mine = self.spent_with_fee() + self.inflight_with_fee()
        total = float(other["usd"]) + mine + projected
        ceil = float(self.cfg["global_ceiling_usd"])
        out = {"utc": utc_now(), "other_ledgers_usd": round(float(other["usd"]), 2), "this_run_usd": round(mine, 2),
               "projected_next_usd": round(projected, 2), "total_usd": round(total, 2), "ceiling_usd": ceil, "errors": other.get("errors") or [],
               "top_dirs": dict(sorted((other.get("per_dir") or {}).items(), key=lambda kv: -kv[1])[:8])}
        self.global_last = out
        if other.get("errors"):
            self.log(f"[global] could not read some ledgers: {other['errors'][:3]}")
        if total >= ceil:
            self.pause(f"GLOBAL ceiling: the re-judge's OpenRouter spend would reach ${total:,.2f} (other ledgers ${float(other['usd']):,.2f} + "
                       f"this live run ${mine:,.2f} spent and in flight + ${projected:,.2f} projected before the next check) >= ${ceil:,.2f}. "
                       f"Largest ledgers: {out['top_dirs']}")
        return out

    async def global_check(self) -> Dict[str, Any]:
        return self.global_eval(await asyncio.to_thread(self.global_other))

    # -- setup ---------------------------------------------------------------------------------------------------------------------------
    def _open(self) -> None:
        self.ledger = Ledger(self.live_dir).writer()
        ensure_schema(self.ledger.conn)
        conn = self.ledger.conn
        rs = meta_get(conn, "run_state") or {}
        usd = float(rs.get("inflight_est_no_fee") or 0) if rs and not rs.get("clean_exit") else 0.0
        if usd > 0:
            conn.execute("INSERT INTO live_spend_adjustments(ts, usd_no_fee, reason) VALUES (?,?,?)",
                         (utc_now(), usd, f"the run of pid {rs.get('pid')} ended uncleanly with requests in flight; their estimate counts as spent"))
            self.log(f"[live] previous run (pid {rs.get('pid')}) ended uncleanly: ${usd:.4f} of in-flight estimates counted as spent")
        # this run's state replaces the last one's in the same transaction as the adjustment, so the adjustment is never made twice
        meta_set(conn, "run_state", {"pid": os.getpid(), "started_utc": utc_now(), "clean_exit": False, "inflight_est_no_fee": 0.0,
                                     "inflight_est_with_fee": 0.0, "limit": self.limit_keys, "end_state": None, "exit_code": None,
                                     "previous": {k: rs.get(k) for k in ("pid", "end_state", "exit_code", "limit", "clean_exit", "ended_utc")}
                                     if rs else None}, commit=False)
        conn.commit()
        ingest_handovers(self.ledger, self.live_dir, self.log)
        sp = live_spend(conn)
        self.spent = sp["total"]
        r = conn.execute("SELECT COALESCE(SUM(cost_usd), 0), COALESCE(SUM(est_cost_usd), 0) FROM live_attempts WHERE cost_source = 'reported'").fetchone()
        self.cost_reported, self.est_reported = float(r[0]), float(r[1])
        self.total_keys = int(conn.execute("SELECT COUNT(*) FROM live_keys").fetchone()[0])
        self.terminal_keys = int(conn.execute("SELECT COUNT(*) FROM requests r JOIN live_keys k USING (custom_id) WHERE r.terminal = 1").fetchone()[0])
        self.sent_keys = int(conn.execute("SELECT COUNT(DISTINCT custom_id) FROM live_attempts").fetchone()[0])
        for row in conn.execute("SELECT provider, COUNT(*), COALESCE(SUM(cost_usd), 0), COALESCE(SUM(latency_s), 0), "
                                "COALESCE(SUM(reasoning_tokens > 0), 0), COALESCE(SUM(reasoning_chars > 0), 0) FROM live_attempts "
                                "WHERE provider IS NOT NULL GROUP BY provider"):
            s = self.provider_stats[row[0]]
            s["n"], s["cost"], s["latency"], s["reasoning"], s["reasoning_text"] = (float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                                                                                    float(row[5]))

    def _plan(self) -> None:
        """The source batches to work through, in order, and (for a ``--limit`` pilot) which keys: the keys already sent and not terminal,
        plus up to ``limit`` - sent new keys spread over evenly spaced batches."""
        conn = self.ledger.conn
        rows = conn.execute("SELECT k.source_batch_key, COUNT(*) FROM live_keys k JOIN requests r USING (custom_id) WHERE r.terminal = 0 "
                            "GROUP BY k.source_batch_key ORDER BY k.source_batch_key").fetchall()
        if self.limit_keys is None:
            self.plan = [(r[0], None) for r in rows]
            return
        self.new_keys_allowed = max(0, int(self.limit_keys) - self.sent_keys)
        allowed: Dict[str, Set[str]] = collections.defaultdict(set)
        for bk, cid in conn.execute("SELECT k.source_batch_key, k.custom_id FROM live_keys k JOIN requests r USING (custom_id) "
                                    "WHERE r.terminal = 0 AND r.attempts > 0"):
            allowed[bk].add(cid)
        n_new = self.new_keys_allowed
        if n_new > 0 and rows:
            m = max(1, min(n_new, len(rows), int(self.cfg["spread_batches"])))
            picks = sorted({rows[int(i * len(rows) / m)][0] for i in range(m)})
            per = int(math.ceil(n_new / len(picks)))
            left = n_new
            for bk in picks:
                if left <= 0:
                    break
                got = [r[0] for r in conn.execute("SELECT k.custom_id FROM live_keys k JOIN requests r USING (custom_id) WHERE "
                                                  "k.source_batch_key = ? AND r.terminal = 0 AND r.attempts = 0 ORDER BY k.rowid LIMIT ?",
                                                  (bk, min(per, left)))]
                allowed[bk].update(got)
                left -= len(got)
        self.plan = [(bk, ks) for bk, ks in sorted(allowed.items()) if ks]

    def _load_batch(self, bk: str, only: Optional[Set[str]]) -> List[WorkItem]:
        """Work items of one source batch: its non-terminal live keys, each with the body to send next (``harvester.build_retries``'s
        rule for a cut reply, from the cap that attempt was sent with). Runs in a thread (its own read-only connection)."""
        con = _ro(self.ledger.path)
        try:
            rows = con.execute("SELECT k.custom_id, k.source_run_dir, k.origin, k.prior_outcomes_json, k.est_input_tokens, k.est_output_tokens, "
                               "k.body_sha256, r.outcome, r.attempts, (SELECT a.max_tokens FROM live_attempts a WHERE a.custom_id = k.custom_id "
                               "AND a.attempt = r.attempts) FROM live_keys k JOIN requests r USING (custom_id) WHERE k.source_batch_key = ? "
                               "AND r.terminal = 0", (bk,)).fetchall()
        finally:
            con.close()
        rows = [r for r in rows if only is None or r[0] in only]
        if not rows:
            return []
        src = rows[0][1]
        man = BF.read_manifest(src, bk)
        bodies = {r["custom_id"]: r["body"] for r in BF.body_requests(BF.read_body(src, man))}
        items: List[WorkItem] = []
        for cid, _src, origin, prior_json, e_in, e_out, bsha, outcome, attempts, last_cap in rows:
            body = bodies[cid]
            if body_sha256(body) != bsha:
                raise RuntimeError(f"body of {cid[:16]}... in source batch {bk} differs from the one registered (sha256)")
            prior = json.loads(prior_json or "[]")
            latest = outcome if attempts else (prior[-1] if prior else None)
            sent_before = bool(attempts) or origin == "reask"
            item = WorkItem(cid, body, bk, origin, prior, float(e_in or 0), float(e_out or 0), bsha, new_key=not attempts)
            ck = cap_key(body)
            # the body the last attempt was sent with (``harvester.build_retries`` re-sends the request of the batch it was last in): a
            # live attempt may have gone out at the raised cap (a cut reply's re-ask), and a held / re-opened attempt is re-sent as it was
            cap = int(body.get(ck) or 0)
            if attempts and last_cap and int(last_cap) != cap:
                cap = int(last_cap)
                item.body = dict(body, **{ck: cap})
                item.body_sha = body_sha256(item.body)
                item.est_out = float(cap)
            if latest == "truncated" and sent_before:
                # the attempt before was cut: re-ask with the raised cap, from the cap that attempt was sent with
                if self.raise_to > cap:
                    item.body = dict(body, **{ck: self.raise_to})
                    item.body_sha = body_sha256(item.body)
                    item.est_out = float(self.raise_to)
                else:
                    item.finalize = (f"truncated at {ck}={cap}; no higher cap configured (truncated_retry_max_tokens "
                                     f"{self.raise_to or 'unset'}), so a re-ask cannot succeed")
            items.append(item)
        items.reverse()                                  # popped from the end: the batch's order
        return items

    # -- feeding -------------------------------------------------------------------------------------------------------------------------
    def _limit_reached(self) -> bool:
        return self.new_keys_allowed is not None and self.new_keys_allowed <= 0

    async def _next_item(self) -> Optional[WorkItem]:
        assert self.lock is not None
        async with self.lock:
            while True:
                if self.stopping:
                    return None
                now = time.monotonic()
                if self.cooldown_until > now:
                    await asyncio.sleep(min(1.0, self.cooldown_until - now))
                    continue
                item: Optional[WorkItem] = None
                if self.reasks:
                    item = self.reasks.popleft()
                elif self.cur:
                    cand = self.cur[-1]
                    if cand.new_key and self._limit_reached():
                        self.cur.pop()
                        continue
                    item = self.cur.pop()
                elif self.plan or self.prefetch is not None:
                    if self.prefetch is None:
                        bk, only = self.plan.pop(0)
                        self.prefetch = asyncio.create_task(asyncio.to_thread(self._load_batch, bk, only))
                    try:
                        loaded = await self.prefetch
                    except Exception as e:  # noqa: BLE001
                        self.prefetch = None
                        self.pause(f"could not load a source batch: {type(e).__name__}: {e}")
                        return None
                    queued = {it.custom_id for it in self.reasks}
                    self.cur = [it for it in loaded if it.custom_id not in self.inflight and it.custom_id not in queued]
                    self.prefetch = None
                    if self.plan and len(self.cur) < 4 * self.limiter.limit:
                        bk, only = self.plan.pop(0)
                        self.prefetch = asyncio.create_task(asyncio.to_thread(self._load_batch, bk, only))
                    continue
                elif self.inflight or self.window:
                    await asyncio.sleep(0.05)
                    continue
                else:
                    return None
                if item.custom_id in self.inflight:
                    continue                             # never two attempts of one key at once
                row = self.ledger.conn.execute("SELECT terminal FROM requests WHERE custom_id = ?", (item.custom_id,)).fetchone()
                if row is None or row[0]:
                    continue                             # never a key that has a terminal outcome
                if item.finalize:
                    self._finalize_uncappable(item)
                    continue
                if not self.budget_ok(item):
                    return None
                if self.plan and self.prefetch is None and len(self.cur) < 2 * self.limiter.limit:
                    bk, only = self.plan.pop(0)
                    self.prefetch = asyncio.create_task(asyncio.to_thread(self._load_batch, bk, only))
                self.inflight.add(item.custom_id)
                self.inflight_est += self.item_est(item)
                if item.new_key:
                    self.sent_keys += 1
                    if self.new_keys_allowed is not None:
                        self.new_keys_allowed -= 1
                return item

    def _source_reply(self, custom_id: str) -> Dict[str, Any]:
        """The source run's record of a handed-over key's cut reply (read only): its batch id, provider, status, finish reason, usage,
        harvest time and attempts, from the source's ``requests`` row while that still holds the cut reply, else from the latest cut
        attempt of its ``attempts`` (batch id and harvest time only)."""
        con = _ro(os.path.join(self.source, "state.sqlite"))
        con.row_factory = sqlite3.Row
        try:
            r = con.execute("SELECT outcome, batch_id, provider, status_code, finish_reason, usage_json, harvested_utc, attempts FROM requests "
                            "WHERE custom_id = ?", (custom_id,)).fetchone()
            if r is not None and r["outcome"] == "truncated":
                return {"batch_id": r["batch_id"], "provider": r["provider"], "status_code": r["status_code"], "finish_reason": r["finish_reason"],
                        "usage": json.loads(r["usage_json"]) if r["usage_json"] else None, "harvested_utc": r["harvested_utc"],
                        "attempts": r["attempts"], "source_outcome_now": r["outcome"]}
            a = con.execute("SELECT batch_id, harvested_utc, attempt FROM attempts WHERE custom_id = ? AND outcome = 'truncated' "
                            "ORDER BY attempt DESC LIMIT 1", (custom_id,)).fetchone()
            out: Dict[str, Any] = {"source_outcome_now": r["outcome"] if r is not None else None}
            if a is not None:
                out.update({"batch_id": a["batch_id"], "harvested_utc": a["harvested_utc"], "attempts": a["attempt"]})
            return out
        finally:
            con.close()

    def _finalize_uncappable(self, item: WorkItem) -> None:
        """A cut reply whose cap cannot be raised is made terminal without sending (``harvester.build_retries``): a key with live attempts
        through ``Ledger.finalize``; a handed-over key (its cut reply is the source run's BATCH reply) by recording that outcome here as
        terminal WITH the source's provenance (batch id, provider, status, finish reason, usage, harvest time; ``error.from_source_run``),
        so the filing files it as the batch reply it is (``transport`` openrouter_batch), not as a live one. Nothing is sent or billed."""
        if item.new_key:
            so = self._source_reply(item.custom_id)
            err = {"kind": "truncated", "finalized": item.finalize, "from_source_run": True, "source_run_dir": self.source,
                   "source_batch_key": item.source_batch_key, "source_batch_id": so.get("batch_id"), "source_provider": so.get("provider"),
                   "source_harvested_utc": so.get("harvested_utc"), "source_attempts": so.get("attempts"),
                   "source_outcome_now": so.get("source_outcome_now")}
            self.ledger.record_outcome(item.custom_id, f"src:{item.source_batch_key}", so.get("batch_id"), "truncated", True, None, err,
                                       so.get("status_code"), so.get("finish_reason") or "length", so.get("usage"), None, so.get("provider"),
                                       commit=False)
            if so.get("harvested_utc"):
                # the reply's time is the batch harvest's (the filing's judged_utc, and its first-harvest rule for a handed-over key)
                self.ledger.conn.execute("UPDATE requests SET harvested_utc = ? WHERE custom_id = ?", (so["harvested_utc"], item.custom_id))
            self.ledger.conn.commit()
        else:
            self.ledger.finalize(item.custom_id, item.finalize or "truncated; no higher cap")
        self.terminal_keys += 1
        self.terminal_ts.append(time.time())

    # -- one request -----------------------------------------------------------------------------------------------------------------
    def _backoff(self, n: int, base: float, cap: float) -> float:
        return min(cap, base * (2 ** max(0, n - 1))) * (0.5 + random.random())

    def _rl_sleep(self, n: int, retry_after: Optional[float]) -> float:
        """The wait before the ``n``-th re-send of a rate-limited request: the larger of ``Retry-After`` and the exponential backoff, at
        least ``rate_limit_min_s`` and at most ``rate_limit_backoff_max_s``, jittered. A ``Retry-After: 0`` never makes a zero-delay
        retry storm, and the wait grows with every 429 of the same request whatever the header says."""
        floor = float(self.cfg["rate_limit_min_s"])
        cap = max(floor, float(self.cfg["rate_limit_backoff_max_s"]))
        exp = float(self.cfg["rate_limit_backoff_s"]) * (2 ** max(0, n - 1))
        base = min(cap, max(floor, exp, float(retry_after or 0.0)))
        return max(floor, min(cap, base * (0.75 + 0.5 * random.random())))

    async def _send(self, session: Any, item: WorkItem, headers: Mapping[str, str]) -> Optional[Result]:
        """POST one request until it has an outcome (``Result``) or must go back to the queue (None: rate limited past its tries, or the
        run stopping during a backoff)."""
        import aiohttp
        payload = dict({"model": self.model}, **item.body)
        payload["provider"] = item.routing or self.routing
        data = compact(payload)
        sent_sha = body_sha256(item.body)
        t_tr = 0
        n_rl = 0
        started = utc_now()
        t0 = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=float(self.cfg["request_timeout_s"]), sock_connect=float(self.cfg["connect_timeout_s"]))
        while True:
            status: Optional[int] = None
            body: Any = None
            hdrs: Mapping[str, str] = {}
            transport_err: Optional[Tuple[str, str]] = None
            try:
                async with session.post(self.url, data=data, headers=headers, timeout=timeout) as resp:
                    status = resp.status
                    hdrs = {k.lower(): v for k, v in resp.headers.items()}
                    raw = await resp.read()
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else None
                except (ValueError, UnicodeDecodeError):
                    body = {"error": {"message": OR.redact(raw[:2000].decode("utf-8", "replace"), os.environ.get(self.key_env)), "code": status}}
            except asyncio.TimeoutError:
                transport_err = ("transport_timeout", f"no reply within {self.cfg['request_timeout_s']}s")
            except (aiohttp.ClientError, OSError) as e:
                transport_err = ("transport_error", OR.redact(f"{type(e).__name__}: {e}", os.environ.get(self.key_env))[:300])
            berr = body.get("error") if isinstance(body, Mapping) else None
            bcode = berr.get("code") if isinstance(berr, Mapping) and isinstance(berr.get("code"), int) else None
            eff = status if status != 200 or bcode is None else bcode
            if transport_err is None and eff in (429, 529):
                msg = self.limiter.on_rate_limited(int(eff))
                if msg:
                    self.event("CONCURRENCY", msg)
                n_rl += 1
                if n_rl > int(self.cfg["rate_limit_max_tries"]) or self.stopping:
                    return None
                ra = None
                try:
                    ra = float(hdrs.get("retry-after")) if hdrs.get("retry-after") else None
                except ValueError:
                    ra = None
                wait = self._rl_sleep(n_rl, ra)
                self.rl_sleep_n += 1
                self.rl_sleep_min_s = wait if self.rl_sleep_min_s is None else min(self.rl_sleep_min_s, wait)
                await asyncio.sleep(wait)
                continue
            transient = transport_err is not None or (eff is not None and (eff == 408 or eff >= 500))
            if transient:
                t_tr += 1
                if self.stopping:
                    # not recorded (the stop is ours, not the item's); a timed-out request may have been billed
                    if transport_err is not None and transport_err[0] == "transport_timeout":
                        self._unknown_spend(self.item_est(item), "a request timed out while the run was stopping")
                    return None
                if t_tr <= int(self.cfg["transport_retries"]):
                    await asyncio.sleep(self._backoff(t_tr, float(self.cfg["transport_backoff_s"]), float(self.cfg["transport_backoff_max_s"])))
                    continue
            lat = time.monotonic() - t0
            res = Result(item=item, parsed={}, http_status=status, started_utc=started, finished_utc=utc_now(), latency_s=lat,
                         transport_retries=t_tr - (1 if transient else 0), rate_limited=n_rl, est_usd=self.item_est(item))
            if transport_err is not None:
                kind, msg = transport_err
                res.parsed = {"outcome": "provider_error", "error": {"kind": kind, "message": msg, "status_code": None}}
                if kind == "transport_timeout":
                    res.cost_source = "unknown_estimated"     # the provider may have finished and billed it
                res.parsed["sent_sha256"] = sent_sha
                return res
            entry = {"custom_id": item.custom_id, "response": {"status_code": status, "body": body if isinstance(body, Mapping) else
                                                               {"error": {"message": str(body)[:2000], "code": status}}}, "error": None}
            try:
                p = dict(self.parser(entry))
            except Exception as exc:  # noqa: BLE001 -- a parser crash is recorded, never fatal (the harvester's rule)
                p = {"outcome": "parse_error", "error": {"kind": "parser_exception", "message": f"{type(exc).__name__}: {exc}"}}
            res.parsed = p
            res.parsed["sent_sha256"] = sent_sha
            if isinstance(body, Mapping) and status == 200:
                m = reply_meta(body)
                res.provider, res.served_model, res.generation_id, res.usage = m["provider"], m["model"], m["id"], m["usage"]
                res.served, res.reasoning_tokens, res.reasoning_chars = m["served"], m["reasoning_tokens"], m["reasoning_chars"]
                res.reasoning_reply = res.reasoning_tokens > 0
                u = res.usage
                if u and isinstance(u.get("cost"), (int, float)) and not isinstance(u.get("cost"), bool):
                    res.cost_usd, res.cost_source = float(u["cost"]), "reported"
                elif u:
                    pr = provider_price(self.providers, res.provider)
                    res.cost_usd = usage_cost(u, pr or self.price_max)
                    res.cost_source = "computed" if pr else "computed_max_price"
                elif m["served"]:
                    res.cost_source = "unknown_estimated"     # served and billed, but no usage to price it by
                if res.served:
                    res.violation = protocol_violation(self.cfg, self.model, self.allowed, res.provider, res.served_model, res.usage,
                                                       res.reasoning_tokens, res.cost_usd, res.cost_source)
            return res

    async def _worker(self, session: Any, headers: Mapping[str, str]) -> None:
        while True:
            item = await self._next_item()
            if item is None:
                return
            await self.limiter.acquire()
            t0 = time.monotonic()
            started = utc_now()
            try:
                res = await self._send(session, item, headers)
            except Exception as e:  # noqa: BLE001 -- one request's crash must not take the engine down
                # recorded as a COUNTED provider_error (the re-ask budget bounds a crash that repeats; a flood of one pauses the run),
                # its estimate counted as spent: the request may have been answered and billed before the crash
                msg = OR.redact(f"{type(e).__name__}: {e}", os.environ.get(self.key_env))[:300]
                self.log(f"[live] request of {item.custom_id[:12]}... crashed: {msg}")
                res = Result(item=item, parsed={"outcome": "provider_error", "error": {"kind": "engine_exception", "message": msg,
                                                                                        "status_code": None},
                                                "sent_sha256": body_sha256(item.body)},
                             http_status=None, started_utc=started, finished_utc=utc_now(), latency_s=time.monotonic() - t0,
                             cost_source="unknown_estimated", est_usd=self.item_est(item))
            finally:
                await self.limiter.release()
            self.inflight_est = max(0.0, self.inflight_est - self.item_est(item))
            if res is None:
                # back to the front of the queue, not recorded (rate limited past its tries, or the run is stopping)
                self.inflight.discard(item.custom_id)
                if item.new_key:
                    self.sent_keys -= 1
                    if self.new_keys_allowed is not None:
                        self.new_keys_allowed += 1
                if not self.stopping:
                    self.reasks.appendleft(item)
                continue
            usd = res.cost_usd if res.cost_usd is not None else (res.est_usd if res.cost_source == "unknown_estimated" else 0.0)
            self.spent += usd
            if res.cost_source == "reported":
                self.cost_reported += float(res.cost_usd or 0)
                self.est_reported += res.est_usd
            self.completions.append((time.time(), usd, res.latency_s))
            if res.provider:
                s = self.provider_stats[res.provider]
                s["n"] += 1
                s["cost"] += float(res.cost_usd or 0)
                s["latency"] += res.latency_s
                s["reasoning"] += 1 if res.reasoning_reply else 0
                s["reasoning_text"] += 1 if res.reasoning_chars else 0
            self.window.append(res)
            if len(self.window) >= int(self.cfg["flush_n"]) and self.flush_event is not None:
                self.flush_event.set()

    # -- recording: windows with the health check -----------------------------------------------------------------------------------
    def _health(self, window: Sequence[Result]) -> Tuple[Set[str], List[str], bool]:
        """``(signatures to hold, pause reasons, cool down instead)`` for one window (the harvester's ``health`` plus the live guards)."""
        cfg = self.health_cfg
        n = len(window)
        held: Set[str] = set()
        reasons: List[str] = []
        transient_only = True
        if not n:
            return held, reasons, False
        mc = int(cfg["min_count"])
        acks = acknowledged(self.live_dir, cfg.get("acknowledged") or [])
        groups: collections.Counter = collections.Counter()
        sample: Dict[str, Result] = {}
        for r in window:
            if str(r.parsed.get("outcome")) in FLOOD_OUTCOMES:
                sig = error_signature(r.parsed)
                groups[sig] += 1
                sample.setdefault(sig, r)
        for sig, c in groups.most_common():
            if c >= mc and c / n >= float(cfg["identical_share"]) and sig not in acks:
                held.add(sig)
                p = sample[sig].parsed
                e = p.get("error") or {}
                if not (p.get("outcome") == "provider_error" and _transient_error(p, sample[sig].http_status)):
                    transient_only = False
                reasons.append(f"identical-error flood: {c} of {n} replies ({c / n:.0%}) share signature {sig} (outcome {p.get('outcome')}, "
                               f"kind {e.get('kind')}, status {sample[sig].http_status or e.get('status_code')}, message "
                               f"{str(e.get('message') or '')[:200]!r}); HELD, not recorded as that outcome. If it is a real per-item outcome, "
                               f"add the line {sig} to {os.path.join(self.live_dir, 'ACKNOWLEDGED')} and remove PAUSED")
        nt = sum(1 for r in window if r.parsed.get("outcome") == "truncated")
        if nt >= mc and nt / n >= float(cfg["truncated_share"]):
            reasons.append(f"truncation: {nt} of {n} replies ({nt / n:.0%}) cut at max_tokens (reasoning not off, or the cap too tight)")
            transient_only = False
        errs = [r for r in window if r.parsed.get("outcome") in ("provider_error", "missing", "parse_error")]
        if len(errs) >= mc and len(errs) / n >= float(cfg["error_share"]):
            if not all(r.parsed.get("outcome") == "provider_error" and _transient_error(r.parsed, r.http_status) for r in errs):
                transient_only = False
            reasons.append(f"errors: {len(errs)} of {n} replies ({len(errs) / n:.0%}) are provider errors or unparseable")
        cool = bool(reasons) and transient_only
        # guards that always pause
        for r in window:
            o = r.parsed.get("outcome")
            if o == "account_error":
                reasons.append(f"account / credential failure (HTTP {r.http_status}): {str((r.parsed.get('error') or {}).get('message') or '')[:300]}")
                cool = False
                break
        for r in window:
            if r.parsed.get("outcome") == "balance_error":
                reasons.append(f"balance / spending limit (HTTP {r.http_status}): {str((r.parsed.get('error') or {}).get('message') or '')[:300]} "
                               f"-- the key's money or limit ran out; check the key, then remove PAUSED")
                cool = False
                break
        # an off-protocol reply (already recorded held, never terminal, never filed) whose kind says the routing is wrong
        bad = collections.Counter(r.violation[0] for r in window if r.violation and r.violation[0] in PAUSING_VIOLATIONS)
        if bad:
            first = next(r for r in window if r.violation and r.violation[0] in PAUSING_VIOLATIONS)
            reasons.append(f"off-protocol replies {dict(bad)} (e.g. {first.violation[1]}); they are recorded held (not terminal, not filed) "
                           f"and are re-sent once the engine starts again. Fix live.routing / live.providers in {self.live_dir}/track.json "
                           f"(or live.served_models / live.price_tolerance if the reply is in fact right), then remove PAUSED")
            cool = False
        # reasoning billed, per provider the routing still allows (a provider dropped from it no longer counts)
        for prov, s in self.provider_stats.items():
            if prov not in self.allowed:
                continue
            if s["n"] >= int(self.cfg["reasoning_min_replies"]) and s["reasoning"] / s["n"] > float(self.cfg["reasoning_share"]):
                reasons.append(f"provider {prov} billed reasoning tokens on {int(s['reasoning'])} of {int(s['n'])} replies (the protocol turns "
                               f"reasoning off: {self.spec.get('request', {}).get('reasoning')}); those replies were held and re-sent, never "
                               f"filed. Drop it (add its slug {(self.providers.get(prov) or {}).get('slug')!r} to live.routing.ignore, or "
                               f"remove it from live.routing.only) in {self.live_dir}/track.json, or keep it by raising live.reasoning_share "
                               f"(now {self.cfg['reasoning_share']}), then remove PAUSED")
                cool = False
                break
        return held, reasons, cool

    def _hold_off_protocol(self, window: Sequence[Result]) -> None:
        """A served reply that breaks the protocol (:func:`protocol_violation`) is no outcome of this judge, whatever it parsed to: it is
        recorded ``held`` (uncounted, never terminal, never filed), its parse kept in the error for the record."""
        for res in window:
            if not res.violation:
                continue
            kind, msg = res.violation
            p = res.parsed
            res.parsed = {"outcome": "held", "usage": p.get("usage"), "status_code": p.get("status_code"), "finish_reason": p.get("finish_reason"),
                          "error": {"kind": OFF_PROTOCOL_KIND, "violation": kind, "message": msg, "provider": res.provider,
                                    "served_model": res.served_model, "parsed_outcome": p.get("outcome"), "parsed_verdict": p.get("verdict"),
                                    "parsed_error": p.get("error")},
                          "sent_sha256": p.get("sent_sha256")}
            self.violations[kind] += 1

    def _reasoning_reroute(self, item: WorkItem, n_rs: int) -> Optional[Tuple[WorkItem, str]]:
        """``reasoning_reroute``: the key re-sent with the run's routing narrowed to the allowed endpoints that have not billed reasoning
        on it (every attempt in the ledger, this window's included), and the event's text; None when none is left (the key is parked).
        Only ``only`` changes: the body, ``ignore``, the quantizations and ``max_price`` are the run's, so every endpoint it can reach is
        one the protocol allows."""
        billed = {r[0] for r in self.ledger.conn.execute("SELECT DISTINCT provider FROM live_attempts WHERE custom_id = ? AND violation = "
                                                         "'reasoning_billed' AND provider IS NOT NULL", (item.custom_id,))}
        left = sorted(n for n in self.allowed if n not in billed and (self.providers.get(n) or {}).get("slug"))
        if not left:
            return None
        routing = dict(self.routing, only=[str(self.providers[n]["slug"]) for n in left])
        self.rerouted[item.custom_id] = left
        msg = (f"{item.custom_id[:16]}...: reasoning billed on {n_rs} replies (by {', '.join(sorted(billed)) or 'unknown'}); re-sent with "
               f"only {routing['only']} (max_reasoning_resends {self.cfg['max_reasoning_resends']}; reasoning_reroute)")
        return dataclasses.replace(item, new_key=False, routing=routing), msg

    def _flush(self) -> None:
        window, self.window = self.window, []
        self.window_t0 = time.monotonic()
        if not window:
            return
        self._flushing = window                          # until committed: an exit past a failed flush still knows these replies
        self._hold_off_protocol(window)
        held, reasons, cool = self._health(window)
        conn = self.ledger.conn
        reask: List[WorkItem] = []
        held_items: List[WorkItem] = []
        events: List[Tuple[str, str]] = []               # written after the window's commit (Ledger.event commits)
        n_term = 0
        max_rs = int(self.cfg["max_reasoning_resends"])
        for res in window:
            item = res.item
            p = res.parsed
            outcome = str(p.get("outcome"))
            if held and outcome in FLOOD_OUTCOMES and error_signature(p) in held:
                p = {"outcome": "held", "usage": p.get("usage"), "status_code": p.get("status_code"), "finish_reason": p.get("finish_reason"),
                     "error": {"kind": "held_identical_error", "signature": error_signature(p), "parsed_outcome": outcome, "parsed_error": p.get("error")},
                     "sent_sha256": p.get("sent_sha256")}
                outcome = "held"
            prior = list(item.prior) + self.ledger.attempt_outcomes(item.custom_id)
            terminal = is_terminal(outcome, prior, self.retry)
            error = p.get("error")
            next_body: Optional[Dict[str, Any]] = None
            if not terminal and outcome not in UNCOUNTED_OUTCOMES:
                if outcome == "truncated":
                    ck = cap_key(item.body)
                    cap = int(item.body.get(ck) or 0)
                    if self.raise_to <= cap:
                        terminal = True
                        error = dict(error or {}, finalized=f"truncated at {ck}={cap}; no higher cap configured (truncated_retry_max_tokens "
                                                            f"{self.raise_to or 'unset'}), so a re-ask cannot succeed")
                    else:
                        next_body = dict(item.body, **{ck: self.raise_to})
                else:
                    next_body = item.body
            usage = res.usage if res.usage is not None else p.get("usage")
            attempt = self.ledger.record_outcome(item.custom_id, f"src:{item.source_batch_key}", None, outcome, terminal, p.get("verdict"), error,
                                                 p.get("status_code") or res.http_status, p.get("finish_reason"), usage, res.cost_usd,
                                                 res.provider or p.get("provider"), commit=False)
            q = (self.providers.get(res.provider or "") or {}).get("quantization") if res.provider else None
            det = (usage or {}).get("prompt_tokens_details") or {}
            cdet = (usage or {}).get("completion_tokens_details") or {}
            vkind = res.violation[0] if res.violation else None
            conn.execute("INSERT OR REPLACE INTO live_attempts(custom_id, attempt, started_utc, finished_utc, latency_s, http_status, outcome, terminal, "
                         "provider, quantization, quantization_source, served_model, generation_id, max_tokens, prompt_tokens, completion_tokens, "
                         "reasoning_tokens, reasoning_chars, cached_tokens, cache_write_tokens, cost_usd, cost_source, est_cost_usd, transport_retries, "
                         "rate_limited, body_sha256, error_signature, served, violation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (item.custom_id, attempt, res.started_utc, res.finished_utc, round(res.latency_s, 3), res.http_status, outcome, 1 if terminal else 0,
                          res.provider, q, "routing_map" if q else None, res.served_model, res.generation_id, int(item.body.get(cap_key(item.body)) or 0),
                          (usage or {}).get("prompt_tokens"), (usage or {}).get("completion_tokens"), cdet.get("reasoning_tokens"), res.reasoning_chars,
                          det.get("cached_tokens"), det.get("cache_write_tokens"), res.cost_usd, res.cost_source, res.est_usd, res.transport_retries,
                          res.rate_limited, p.get("sent_sha256"), error_signature(p) if outcome != "verdict" else None, 1 if res.served else 0, vkind))
            if terminal:
                n_term += 1
            elif vkind == "reasoning_billed":
                # re-sent at once (the same body), unless this key keeps drawing reasoning: then it waits for a person (exit 5)
                n_rs = conn.execute("SELECT COUNT(*) FROM live_attempts WHERE custom_id = ? AND violation = 'reasoning_billed'",
                                    (item.custom_id,)).fetchone()[0]
                rr = self._reasoning_reroute(item, n_rs) if n_rs > max_rs and self.cfg.get("reasoning_reroute") else None
                if n_rs <= max_rs:
                    reask.append(dataclasses.replace(item, new_key=False))
                elif rr is not None:
                    reask.append(rr[0])
                    events.append(("REROUTED", rr[1]))
                else:
                    self.parked.add(item.custom_id)
                    self.rerouted.pop(item.custom_id, None)
                    events.append(("PARKED", f"{item.custom_id[:16]}...: reasoning billed on {n_rs} replies; not re-sent by this run "
                                             f"(max_reasoning_resends {max_rs}"
                                             + (", and every allowed endpoint has billed reasoning on it" if self.cfg.get("reasoning_reroute") else "")
                                             + "); it stays without a terminal outcome"))
            elif vkind:
                pass                                      # the run pauses (_health); re-sent when the engine starts again
            elif outcome == "held":
                held_items.append(dataclasses.replace(item, new_key=False))
            elif next_body is not None:
                # a re-ask keeps the key's routing: once narrowed (reasoning_reroute) it stays narrowed for the rest of this run
                reask.append(WorkItem(item.custom_id, next_body, item.source_batch_key, item.origin, item.prior, item.est_in,
                                      float(next_body.get(cap_key(next_body)) or 0) if next_body is not item.body else item.est_out,
                                      body_sha256(next_body), new_key=False, routing=item.routing))
        rs = meta_get(conn, "run_state") or {}
        rs.update({"inflight_est_no_fee": round(self.inflight_est, 6), "inflight_est_with_fee": round(self.inflight_with_fee(), 6),
                   "updated_utc": utc_now()})
        meta_set(conn, "run_state", rs, commit=False)
        conn.commit()
        self._flushing = []
        for kind, msg in events:
            self.event(kind, msg)
        self.terminal_keys += n_term
        now = time.time()
        self.terminal_ts.extend([now] * n_term)
        for res in window:
            self.inflight.discard(res.item.custom_id)
        if reasons and cool and self.cooldown_n < int(self.cfg["max_cooldowns"]):
            self.cooldown_n += 1
            cd = min(float(self.cfg["cooldown_max_s"]), float(self.cfg["cooldown_s"]) * (2 ** (self.cooldown_n - 1)))
            self.cooldown_until = time.monotonic() + cd
            self.event("COOLDOWN", f"transient provider failures ({'; '.join(reasons)[:600]}); no new request for {cd:.0f}s "
                                   f"(flood {self.cooldown_n} of {self.cfg['max_cooldowns']} in a row), then the {len(held_items)} held keys are re-sent")
            reask.extend(held_items)
        elif reasons:
            if cool:
                reasons.append(f"{self.cooldown_n} cool-downs in a row did not clear it")
            self.pause(" | ".join(reasons)[:3500])
        else:
            self.cooldown_n = 0
        if not self.stopping:
            self.reasks.extend(reask)

    # -- progress ------------------------------------------------------------------------------------------------------------------
    def progress(self, state: str) -> Dict[str, Any]:
        conn = self.ledger.conn
        now = time.time()
        while self.completions and now - self.completions[0][0] > 3600:
            self.completions.popleft()
        while self.terminal_ts and now - self.terminal_ts[0] > 3600:
            self.terminal_ts.popleft()

        def per_min(ts: Iterable[float], horizon: float) -> float:
            n = sum(1 for t in ts if now - t <= horizon)
            return n / (min(horizon, max(1.0, now - self.t_start)) / 60.0)

        lat = sorted(l for t, _, l in self.completions if now - t <= 300)

        def pct(q: float) -> Optional[float]:
            return round(lat[min(len(lat) - 1, int(q * len(lat)))], 2) if lat else None

        remaining = self.total_keys - self.terminal_keys
        t5 = per_min(self.terminal_ts, 300)
        t60 = per_min(self.terminal_ts, 3600)
        rate = t5 or t60
        tot_n = sum(s["n"] for s in self.provider_stats.values()) or 1.0
        mix = {p: {"replies": int(s["n"]), "share": round(s["n"] / tot_n, 4), "quantization": (self.providers.get(p) or {}).get("quantization"),
                   "allowed": p in self.allowed,
                   "cost_usd_no_fee": round(s["cost"], 4), "mean_latency_s": round(s["latency"] / s["n"], 2) if s["n"] else None,
                   "reasoning_token_replies": int(s["reasoning"]), "reasoning_text_replies": int(s["reasoning_text"])}
               for p, s in sorted(self.provider_stats.items())}
        events = [dict(r) for r in conn.execute("SELECT ts, kind, message FROM events ORDER BY rowid DESC LIMIT 25")]
        outcomes = self._outcomes if self._outcomes is not None else self.ledger.outcome_counts()
        prog = {
            "updated_utc": utc_now(), "live_dir": self.live_dir, "source_run_dir": self.source, "model": self.model, "pid": os.getpid(),
            "state": state, "stopping": self.stopping, "paused": _paused(self.live_dir), "limit_keys": self.limit_keys,
            "keys": {"total": self.total_keys, "terminal": self.terminal_keys, "remaining": remaining, "sent": self.sent_keys},
            "outcomes": outcomes,
            "in_flight": len(self.inflight) - len(self.window), "awaiting_record": len(self.window), "reasks_queued": len(self.reasks),
            "attempts_per_min": {"last_5m": round(per_min((t for t, _, _ in self.completions), 300), 1),
                                 "last_60m": round(per_min((t for t, _, _ in self.completions), 3600), 1)},
            "terminal_keys_per_min": {"last_5m": round(t5, 1), "last_60m": round(t60, 1)},
            "latency_s_last_5m": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99)},
            "concurrency": {"limit": self.limiter.limit, "target": self.limiter.target, "active": self.limiter.active,
                            "halvings": self.limiter.halvings, "rate_limited": dict(self.limiter.n_rate_limited),
                            "rate_limit_waits": self.rl_sleep_n, "rate_limit_min_wait_s": (round(self.rl_sleep_min_s, 3)
                                                                                          if self.rl_sleep_min_s is not None else None),
                            "cooldown_s_left": round(max(0.0, self.cooldown_until - time.monotonic()), 1)},
            "off_protocol_held": dict(self.violations), "parked_keys": len(self.parked), "rerouted_keys": len(self.rerouted),
            "spend_usd": {"spent_with_fee": round(self.spent_with_fee(), 4), "in_flight_with_fee": round(self.inflight_with_fee(), 4),
                          "max_spend_usd": self.max_spend, "cost_per_estimate": round(self.cost_ratio(), 4),
                          "per_terminal_key_with_fee": round(self.spent_with_fee() / self.terminal_keys, 6) if self.terminal_keys else None,
                          "global": self.global_last},
            "provider_mix": mix,
            "eta": {"remaining_keys": remaining, "keys_per_min": round(rate, 1),
                    "hours": round(remaining / rate / 60.0, 2) if rate > 0 else None,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + remaining / rate * 60.0)) if rate > 0 else None},
            "events": events,
        }
        path = os.path.join(self.live_dir, "progress.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(prog, fh, indent=1, default=str)
        os.replace(path + ".tmp", path)
        return prog

    # -- control ---------------------------------------------------------------------------------------------------------------------
    def _controls(self) -> None:
        if _paused(self.live_dir) and not self.stopping:
            self.stop(f"PAUSED file present: {str(_paused(self.live_dir)).splitlines()[0][:300]}", EXIT_PAUSED)
        sp = os.path.join(self.live_dir, STOP_FILE)
        if os.path.exists(sp) and not self.stopping:
            os.replace(sp, sp + f".consumed-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
            self.stop("STOP file", EXIT_STOPPED)
        cp = os.path.join(self.live_dir, CONCURRENCY_FILE)
        if os.path.exists(cp):
            try:
                n = int(open(cp).read().strip())
            except ValueError:
                n = 0
            if n > 0:
                msg = self.limiter.set_target(n)
                if msg:
                    self.event("CONCURRENCY", msg + f" ({cp})")
        msg = self.limiter.recover()
        if msg:
            self.event("CONCURRENCY", msg)
        if time.time() - self.t_start > float(self.cfg["max_hours"]) * 3600 and not self.stopping:
            self.stop(f"--max-hours {self.cfg['max_hours']} reached (start again to resume: the ledger is the state)", EXIT_STOPPED)

    def _source_states(self) -> Tuple[Set[str], Dict[str, Tuple[str, Optional[str]]]]:
        """(source batches holding this run's non-terminal pending-origin keys, every source batch's (state, batch id)); read only, its
        own connections, so it runs in a thread."""
        lcon = _ro(self.ledger.path)
        try:
            # every batch this run took at init or by a pending-origin handover (index scan; a batch whose keys are all terminal here
            # still must not be re-opened there)
            mine = {r[0] for r in lcon.execute("SELECT DISTINCT source_batch_key FROM live_keys WHERE origin = 'pending'")}
        finally:
            lcon.close()
        con = _ro(os.path.join(self.source, "state.sqlite"))
        try:
            st = {r["batch_key"]: (r["state"], r["batch_id"]) for r in con.execute("SELECT batch_key, state, batch_id FROM batches")}
        finally:
            con.close()
        return mine, st

    async def _source_guard(self) -> None:
        """The keys registered from the source's pending batches must stay unsent there: a source batch that left pending (a batch
        harvester re-opened) would judge them twice, so the live run pauses."""
        mine, st = await asyncio.to_thread(self._source_states)
        if not mine:
            return
        moved = sorted(bk for bk in mine if bk not in st or st[bk][0] not in SOURCE_PENDING_STATES or st[bk][1])
        if moved:
            self.pause(f"the source run {self.source} no longer holds {len(moved)} batch(es) whose keys this live run owns as pending "
                       f"(e.g. {[(b, st.get(b)) for b in moved[:5]]}): the batch path re-opened. Stop one of the two before going on")

    async def _monitor(self) -> None:
        t_prog = 0.0
        t_ctl = 0.0
        while True:
            assert self.flush_event is not None
            try:
                await asyncio.wait_for(self.flush_event.wait(), timeout=min(float(self.cfg["control_s"]), float(self.cfg["flush_s"])))
            except asyncio.TimeoutError:
                pass
            self.flush_event.clear()
            now = time.monotonic()
            if self.window and (len(self.window) >= int(self.cfg["flush_n"]) or now - self.window_t0 >= float(self.cfg["flush_s"])):
                self._flush()
            if now - t_ctl >= float(self.cfg["control_s"]):
                t_ctl = now
                self._controls()
                alive = [w for w in self.workers if not w.done()]
                if self._has_work() and not self.stopping and self._session is not None:
                    for _ in range(max(0, self.limiter.limit - len(alive))):
                        self.workers.append(asyncio.create_task(self._worker(self._session, self._headers)))
                    await self.limiter._notify()
            if now - self.global_t >= float(self.cfg["global_check_s"]):
                self.global_t = now
                # each check pauses the run when it cannot be made; a SystemExit (a refusal) is caught too, so the requests in flight
                # are still recorded and the exit is a clean pause, never an escape past them
                try:
                    await self.global_check()
                except (Exception, SystemExit) as e:  # noqa: BLE001
                    self.log(f"[global] check failed: {type(e).__name__}: {e}")
                    self.pause(f"the global spend check failed ({type(e).__name__}: {e}); nothing is sent without it")
                try:
                    await self._source_guard()
                except (Exception, SystemExit) as e:  # noqa: BLE001
                    self.pause(f"the source guard could not read {self.source} ({type(e).__name__}: {e}); nothing is sent without it")
                try:
                    added = ingest_handovers(self.ledger, self.live_dir, self.log)
                    if added:
                        self.total_keys += added
                        self.plan.extend(self._plan_new_handover())
                except (Exception, SystemExit) as e:  # noqa: BLE001
                    self.pause(f"a handover file could not be ingested ({type(e).__name__}: {e}); nothing of it was registered. Move the "
                               f"file out of {os.path.join(self.live_dir, 'handovers')} (or fix its cause), then remove PAUSED")
            if now - t_prog >= float(self.cfg["progress_s"]):
                t_prog = now
                try:
                    self._outcomes = await asyncio.to_thread(self._count_outcomes)
                except sqlite3.Error:
                    self._outcomes = None
                self.progress("stopping" if self.stopping else ("cooldown" if self.cooldown_until > now else "running"))
            if all(w.done() for w in self.workers) and not self.window and (self.stopping or not self._has_work()):
                return

    def _count_outcomes(self) -> Dict[str, int]:
        con = _ro(self.ledger.path)
        try:
            return {str(r[0]): int(r[1]) for r in con.execute("SELECT COALESCE(outcome, 'not_harvested'), COUNT(*) FROM requests GROUP BY 1")}
        finally:
            con.close()

    def _has_work(self) -> bool:
        return bool(self.reasks or self.cur or self.plan or self.prefetch is not None or self.inflight or self.window)

    def _plan_new_handover(self) -> List[Tuple[str, Optional[Set[str]]]]:
        have = {bk for bk, _ in self.plan}
        rows = self.ledger.conn.execute("SELECT DISTINCT k.source_batch_key FROM live_keys k JOIN requests r USING (custom_id) WHERE "
                                        "k.origin IN ('reask', 'in_flight') AND r.terminal = 0 AND r.attempts = 0").fetchall()
        if self.limit_keys is not None:
            return []
        return [(r[0], None) for r in rows if r[0] not in have]

    # -- the run ---------------------------------------------------------------------------------------------------------------------
    async def run_async(self, session_factory: Optional[Callable[[], Any]] = None) -> int:
        key = os.environ.get(self.key_env)
        try:
            if not key:
                raise SystemExit(f"the OpenRouter key is not in ${self.key_env} (export it in this process's environment)")
            if os.path.exists(os.path.join(self.live_dir, STOP_FILE)):
                raise SystemExit(f"{self.live_dir}/STOP exists: remove it to start")
            if _paused(self.live_dir):
                self.log(f"[live] PAUSED, not starting: {str(_paused(self.live_dir)).splitlines()[0][:300]}")
                return EXIT_PAUSED
        except BaseException:
            self._logf.close()
            raise
        if self._logf.closed:
            self._logf = open(os.path.join(self.live_dir, "live.log"), "a")
        try:
            self._open()
        except BaseException:
            if self.ledger is not None:
                self._close_run_state()
                self.ledger.close()
                self.ledger = None
            self._logf.close()
            raise
        self.lock = asyncio.Lock()
        self.flush_event = asyncio.Event()
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json",
                         "User-Agent": USER_AGENT}
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda s=sig: self.stop(f"signal {signal.Signals(s).name}", EXIT_STOPPED))
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        self.log(f"[live] start {self.live_dir}: model {self.model}, {self.total_keys} keys ({self.terminal_keys} terminal), concurrency "
                 f"{self.limiter.target}, max_spend_usd ${self.max_spend:,.2f}, limit {self.limit_keys}, routing {json.dumps(self.routing)}")
        self.event("LIVE_START", f"pid {os.getpid()} limit {self.limit_keys} concurrency {self.limiter.target}")
        try:
            self.global_t = time.monotonic()
            await self.global_check()
            await self._source_guard()
            if self.stopping:
                return self._finish("paused")
            self._plan()
            import aiohttp
            conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, enable_cleanup_closed=True)
            async with (session_factory() if session_factory else aiohttp.ClientSession(connector=conn)) as session:
                self._session = session
                self.workers = [asyncio.create_task(self._worker(session, self._headers)) for _ in range(self.limiter.limit)]
                await self._monitor()
                await asyncio.gather(*self.workers, return_exceptions=True)
                self._session = None
            if self.prefetch is not None:
                self.prefetch.cancel()
            self._flush()
            return self._finish("stopped" if self.stopping else "finished")
        finally:
            if self.ledger is not None:
                self._close_run_state()
                self.ledger.close()
                self.ledger = None
            self._logf.close()

    def _close_run_state(self) -> None:
        """Write how the run ended into ``run_state``. A run that reached :meth:`_finish` recorded every request it sent: a clean exit.
        Any other way out (an exception past the monitor) first tries to record the replies already in hand; whatever is still unrecorded
        (requests in flight, replies that could not be recorded) keeps ``clean_exit`` false with its estimate, which the next start
        counts as spent (and the global walker counts meanwhile)."""
        try:
            pending = 0.0
            if not self._finished:
                try:
                    self.ledger.conn.rollback()           # the partial rows of a flush that failed half way (only _flush defers its commit)
                except sqlite3.Error:
                    pass
                self.window = list(self._flushing) + list(self.window)
                self._flushing = []
                win_usd = sum((r.cost_usd if r.cost_usd is not None else r.est_usd) for r in self.window)
                n_win = len(self.window)
                try:
                    if self.window:
                        self._flush()
                except Exception as e:  # noqa: BLE001
                    try:
                        self.ledger.conn.rollback()       # never commit half a window with the run state below
                    except sqlite3.Error:
                        pass
                    pending += win_usd
                    self.log(f"[live] could not record {n_win} replies at exit: {type(e).__name__}: {e}")
                pending += self.inflight_est
            clean = self._finished or pending <= 0
            rs = meta_get(self.ledger.conn, "run_state") or {}
            rs.update({"clean_exit": clean, "inflight_est_no_fee": 0.0 if clean else round(pending, 6),
                       "inflight_est_with_fee": 0.0 if clean else round(pending * max(1.0, self.cost_ratio()) * (1.0 + BF.CREDIT_FEE), 6),
                       "ended_utc": utc_now(), "exit_code": self._end.get("exit_code"), "end_state": self._end.get("end_state") or "crashed",
                       "limit": self.limit_keys, "parked_keys": self._end.get("parked_keys"),
                       "non_terminal_keys": self._end.get("non_terminal_keys"), "rerouted_keys": self._end.get("rerouted_keys")})
            meta_set(self.ledger.conn, "run_state", rs)
        except sqlite3.Error:
            pass

    def _finish(self, how: str) -> int:
        r = readiness(self.live_dir)
        if self.exit_code is not None:
            code = self.exit_code
        elif r["ready"]:
            code = EXIT_DONE
        elif self.limit_keys is not None and self._limit_done():
            code = EXIT_DONE
        else:
            code = EXIT_UNFINISHED
        state = {EXIT_DONE: "done" if r["ready"] else "pilot_done", EXIT_PAUSED: "paused", EXIT_STOPPED: "stopped"}.get(code, "unfinished")
        self._outcomes = None
        self._finished = True
        self._end = {"exit_code": code, "end_state": state, "parked_keys": len(self.parked),
                     "non_terminal_keys": int(r["requests"]) - int(r["terminal"]), "rerouted_keys": len(self.rerouted)}
        self.progress(state)
        self.event("LIVE_END", f"exit {code} ({state}): {self.terminal_keys} of {self.total_keys} keys terminal; spent "
                               f"${self.spent_with_fee():,.2f} incl. fee; {self.stopping or how}"
                               + (f"; {len(self.parked)} keys parked (reasoning billed on every try)" if self.parked else ""))
        return code

    def _limit_done(self) -> bool:
        n = self.ledger.conn.execute("SELECT COUNT(*) FROM requests r WHERE r.attempts > 0 AND r.terminal = 0").fetchone()[0]
        return int(n) == 0


def _transient_error(p: Mapping[str, Any], status: Optional[int]) -> bool:
    e = p.get("error") or {}
    kind = str(e.get("kind") or "")
    st = status if status is not None else e.get("status_code")
    return kind.startswith("transport") or (isinstance(st, int) and (st == 408 or st >= 500))


# =====================================================================================================================================
# dry run: keys, cost, duration (no network)
# =====================================================================================================================================
def dry_run(live_dir: str, concurrency: Optional[int] = None, assume_latency_s: float = 10.0, latency_band: Tuple[float, float] = (6.0, 20.0),
            global_root: Optional[str] = None) -> Dict[str, Any]:
    """What the run would send and cost (read only): the keys left, the tokens they carry (manifest estimates scaled by the measured
    tokens of the live run's replies, else of the source run's harvested first pass), priced at the routing's provider prices (low: the
    cheapest, central: their mean or the measured cost per reply, high: the dearest with no cache discount, unscaled output estimates and
    the bake-off's re-ask rates), plus the fee; the duration at the live run's measured latency, else ``assume_latency_s``."""
    live_dir = os.path.abspath(os.path.expanduser(live_dir))
    spec = BF.read_track(live_dir)
    cfg = dict(LIVE_DEFAULTS, **(spec.get("live") or {}))
    conc = int(concurrency or cfg["concurrency"])
    providers = cfg["providers"]
    con = _ro(os.path.join(live_dir, "state.sqlite"))
    try:
        info = meta_get(con, "init") or {}
        n_total, n_term = con.execute("SELECT COUNT(*), COALESCE(SUM(r.terminal), 0) FROM live_keys k JOIN requests r USING (custom_id)").fetchone()
        rem = con.execute("SELECT COUNT(*), COALESCE(SUM(k.est_input_tokens), 0), COALESCE(SUM(k.est_output_tokens), 0), "
                          "COALESCE(SUM(json_array_length(k.prior_outcomes_json) > 0), 0) FROM live_keys k JOIN requests r USING (custom_id) "
                          "WHERE r.terminal = 0").fetchone()
        by_origin = dict(con.execute("SELECT k.origin, COUNT(*) FROM live_keys k JOIN requests r USING (custom_id) WHERE r.terminal = 0 "
                                     "GROUP BY 1").fetchall())
        la = con.execute("SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cached_tokens), 0), "
                         "COALESCE(SUM(cost_usd), 0), COALESCE(AVG(latency_s), 0), COALESCE(SUM(est_cost_usd), 0) FROM live_attempts "
                         "WHERE http_status = 200 AND prompt_tokens IS NOT NULL").fetchone()
        est_of_replied = con.execute("SELECT COALESCE(SUM(k.est_input_tokens), 0), COALESCE(SUM(k.est_output_tokens), 0), COUNT(*) FROM live_attempts a "
                                     "JOIN live_keys k USING (custom_id) WHERE a.http_status = 200 AND a.prompt_tokens IS NOT NULL").fetchone()
        n_att, n_keys_att = con.execute("SELECT COUNT(*), COUNT(DISTINCT custom_id) FROM live_attempts").fetchone()
        lat_rows = [r[0] for r in con.execute("SELECT latency_s FROM live_attempts WHERE http_status = 200 ORDER BY latency_s")]
        spent = live_spend(con)["total"] * (1.0 + BF.CREDIT_FEE)
    finally:
        con.close()
    n_rem, est_in, est_out, n_prior = int(rem[0]), float(rem[1]), float(rem[2]), int(rem[3])
    cal = info.get("calibration") or {}
    measured = int(la[0]) >= 50
    if measured:
        in_ratio = float(la[1]) / max(1.0, float(est_of_replied[0]))
        out_ratio = float(la[2]) / max(1.0, float(est_of_replied[1]))
        cached_share = float(la[3]) / max(1.0, float(la[1]))
        reask_factor = float(n_att) / max(1, int(n_keys_att))
        source = f"the live run's own {int(la[0])} replies"
    elif cal.get("replies") and cal.get("est_input_per_request") and cal.get("est_output_per_request"):
        in_ratio = float(cal["prompt_tokens_per_reply"]) / float(cal["est_input_per_request"])
        out_ratio = float(cal["completion_tokens_per_reply"]) / float(cal["est_output_per_request"])
        cached_share = float(cal.get("cached_tokens_per_reply") or 0) / float(cal["prompt_tokens_per_reply"] or 1)
        rs = cal.get("first_attempt_reask_share")
        rs = 0.0535 if rs is None else float(rs)
        reask_factor = 1.0 + rs + rs * 0.0857
        source = f"the source run's {cal.get('replies')} harvested first-pass replies (batch)"
    else:
        in_ratio, out_ratio, cached_share, reask_factor = 1.0, 1.0, 0.0, 1.0 + 0.0535 + 0.0535 * 0.0857
        source = "the manifests' estimates unscaled (no measured reply yet), the bake-off's re-ask rates"
    prices = {n: provider_price(providers, n) for n in providers}
    lo = min(prices.values(), key=lambda p: p[0] + p[1])
    hi = (max(p[0] for p in prices.values()), max(p[1] for p in prices.values()), max(p[0] for p in prices.values()))
    mean = tuple(sum(p[i] for p in prices.values()) / len(prices) for i in range(3))

    def usd(price: Tuple[float, float, float], in_tok: float, out_tok: float, cshare: float, factor: float) -> float:
        cached = in_tok * cshare
        return ((in_tok - cached) * price[0] + cached * price[2] + out_tok * price[1]) / 1e6 * factor * (1.0 + BF.CREDIT_FEE)

    in_tok = est_in * in_ratio
    out_tok = est_out * out_ratio
    bands = {"low": usd(lo, in_tok, out_tok, cached_share, max(1.0, reask_factor - 0.01)),
             "central": usd(mean, in_tok, out_tok, cached_share, reask_factor),
             "high": usd(hi, est_in * max(1.0, in_ratio), est_out * max(1.0, out_ratio), 0.0, max(reask_factor, 1.0 + 0.0535 + 0.0535 * 0.0857) + 0.005)}
    if measured and float(la[4]) > 0:
        per_reply = float(la[4]) / int(la[0])
        per_est = float(la[4]) / max(1e-12, float(la[6]))
        bands["central_measured"] = per_reply * n_rem * reask_factor * (1.0 + BF.CREDIT_FEE)
        bands["measured_cost_per_estimate"] = per_est
    lat_m = (sum(lat_rows) / len(lat_rows)) if len(lat_rows) >= 20 else None
    lat = lat_m or assume_latency_s
    n_req = n_rem * reask_factor

    def hours(l: float) -> float:
        return n_req * l / conc / 3600.0

    out = {"utc": utc_now(), "live_dir": live_dir, "source_run_dir": spec.get("source_run_dir"), "model": spec.get("model"),
           "routing": cfg.get("routing"), "providers": providers, "concurrency": conc,
           "keys": {"total": int(n_total), "terminal": int(n_term), "remaining": n_rem, "remaining_by_origin": by_origin,
                    "remaining_with_prior_attempts": n_prior, "init": {k: v for k, v in info.items() if k != "calibration"}},
           "tokens": {"est_input_remaining": round(est_in), "est_output_remaining": round(est_out), "input_ratio": round(in_ratio, 4),
                      "output_ratio": round(out_ratio, 4), "cached_share": round(cached_share, 4), "reask_factor": round(reask_factor, 4),
                      "scaled_from": source},
           "prices_per_mtok": {"low": lo, "central_mean": [round(x, 4) for x in mean], "high": hi},
           "cost_usd_with_fee": {k: round(v, 2) for k, v in bands.items()}, "spent_so_far_with_fee": round(spent, 4),
           "max_spend_usd": spec.get("max_spend_usd"),
           "duration": {"requests_incl_reasks": round(n_req), "latency_s": round(lat, 2),
                        "latency_source": ("measured mean over the live run's replies" if lat_m else f"ASSUMED {assume_latency_s}s (no live reply yet)"),
                        "hours_at_latency": round(hours(lat), 1), "hours_band": {f"{l:g}s": round(hours(l), 1) for l in latency_band},
                        "requests_per_min": round(conc / lat * 60.0)}}
    if global_root is not None:
        other = openrouter_spend(global_root, exclude=[live_dir])
        out["global"] = {"other_ledgers_usd": other["usd"], "ceiling_usd": cfg.get("global_ceiling_usd"),
                         "headroom_for_this_run_usd": round(float(cfg.get("global_ceiling_usd") or 0) - float(other["usd"]), 2),
                         "top_dirs": dict(sorted(other["per_dir"].items(), key=lambda kv: -kv[1])[:10]), "errors": other["errors"]}
    return out


# =====================================================================================================================================
# the global root
# =====================================================================================================================================
def global_root_of(live_dir: str) -> Optional[str]:
    """The root a live directory's global check walks (``track.json`` ``live.global_root``), expanded; None when it states none."""
    try:
        lv = BF.read_track(live_dir).get("live") or {}
    except (OSError, ValueError):
        lv = {}
    root = lv.get("global_root")
    return os.path.abspath(os.path.expanduser(str(root))) if root else None


def check_global_root(path: str) -> str:
    """A ``--global-root`` as a live directory stores it: an ABSOLUTE path (``~`` expanded) of an EXISTING directory, normalized.
    :class:`SystemExit` otherwise: a relative root would be resolved against whatever directory the engine later starts in, and a root
    that does not exist walks as no ledgers at all, i.e. $0 spent by every other run."""
    root = os.path.expanduser(str(path))
    if not os.path.isabs(root):
        raise SystemExit(f"REFUSED: --global-root {path!r} is not an absolute path")
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        raise SystemExit(f"REFUSED: --global-root {root} is not an existing directory (the global check would walk nothing and count "
                         f"every other run's spend as $0)")
    return root
