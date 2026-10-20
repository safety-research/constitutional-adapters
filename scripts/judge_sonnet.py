"""LLM-judge scoring of recorded eval responses with claude-sonnet-5.

Implements BOTH judge methodologies from the literature, verbatim in
structure, with Sonnet 5 standing in for GPT-4o:

1. Stolfo et al. (2410.12877, App. C) response-quality protocol —
   instruction-BLIND, so it measures collateral quality, not compliance:
   - per base query, the judge generates 5 yes/no questions probing the
     quality of a generic answer (cached per query, shared across arms);
   - per response, the judge answers each question yes / no / N-A with a
     short rationale; quality = fraction "yes" among non-N-A;
   - the answering pass is repeated 3x -> mean and SE across repeats.
   Compliance itself stays programmatic (the harness's IFEval-style
   strict/loose checkers); this script adds McNemar's exact test on the
   paired per-prompt loose (and strict) outcomes of steered-vs-prompted,
   marking p < 0.01 with an asterisk, as in their Table 1.

2. HyperSteer (2506.03292, §4) steering-performance protocol — the judge
   scores each response in {0,1,2} on three dimensions: following the
   base prompt, following the steering prompt, and text fluency;
   steering performance = harmonic mean of the three (0 if any is 0).

Reads the stage-1/1.5 output layout ({out}/{instruction}/{method}/
results.json [+ frontier.json for the judged steered scale]) and judges
four arms per cell: baseline, prompted, steered@frontier-scale, both.
Identical (query, response, task) triples are cached in
{out}/judge_cache.jsonl, so shared baseline/prompted arms across methods
are judged once and reruns are resume-safe.

API: Anthropic Messages API, routed to the LOW-PRIO pool by
default (``Judge(..., org=)`` names another; the key is the org's OWN
environment variable or vault item through sisteer.eval.api_keys /
sisteer.vault — the bare env ANTHROPIC_API_KEY is High Prio and nothing
else, and no key is ever read from a file). On 429s the judge THROTTLES DOWN on
the same org — an AdaptiveLimiter halves the in-flight width and slowly
earns it back; per the 2026-08-21 routing rule it NEVER switches to the
High-Prio org (that rule REPLACED the old High-Prio-on-429 fallback; do
not reinstate it). The client is built with SDK max_retries=1 so retry
layers don't multiply: the outer wait-out loop here owns backoff.
Default concurrency 100 (a real run therefore CROSSES the >50
coordination threshold: a site with a coordination channel
(SISTEER_COORDINATION_CHANNEL) posts at launch — print-and-proceed, never a
gate — and is ready to scale down). Batch API is not allowed on the prio
orgs. A credential the API
REJECTS (401 / 403) stops the run at the first
such answer: the judge latches it, every queued worker raises before its
wire call, the arm's queue is cancelled, no report is written and the typed
``sisteer.vault.JudgeCredentialRejected`` message is the terminal line.
Thinking is disabled by default for latency — the rationale /
justification fields provide the chain of thought; pass --thinking to
re-enable adaptive thinking.

Usage:
  uv run python scripts/judge_sonnet.py --out-dir out/stage1_llama8b \
      [--instructions french json] [--n 50] [--repeats 3] [--concurrency 100]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
import pydantic
from anthropic.lib._parse._transform import transform_schema   # the SDK's own output_format -> json_schema transform (private module; anthropic pinned 1.0.0, where it is byte-identical to 0.120.2's)
from pydantic import BaseModel

from sisteer import vault
from sisteer.data.instructions import REGISTRY

JUDGE_MODEL = "claude-sonnet-5"
#: the structured verdict's output cap, and the ONE larger cap a verdict the model cut off at the first cap is retried at
#: (2026-10-02): five u0 agentic HarmfulVerdict replies overran 1500 tokens at ~750 chars of JSON (a long `reasoning`
#: string) and surfaced as pydantic json_invalid / "EOF while parsing"; 4000 leaves ~2.5x headroom without being unbounded
VERDICT_MAX_TOKENS = 1500
TRUNCATION_RETRY_MAX_TOKENS = 4000
#: the THIRD rung -- 8000 tokens, or the judge model's maximum output when that is lower (the
#: cap applies to every rung: a request above the model's maximum is a 400). The ladder climbs ONLY on a genuine `max_tokens` stop
#: (classify_verdict); a verdict still cut at the last rung is the item's error with its reason (the caller files it errored), never blocked.
TRUNCATION_THIRD_RUNG_MAX_TOKENS = 8000
#: the judge model's maximum output, per the Models API's `max_tokens` field (`client.models.retrieve(id).max_tokens`; 128K for the
#: Sonnet 5 line) -- pinned here so the ladder is a pure function of the model name (no live lookup inside the retry loop); a model
#: absent from the table is uncapped (the three rungs as written)
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {"claude-sonnet-5": 128_000}


def verdict_token_ladder(model: str = JUDGE_MODEL) -> tuple[int, ...]:
    """The max_tokens a verdict is requested at, rung by rung: 1500, 4000, 8000, each capped at the model's maximum output
    (:data:`MODEL_MAX_OUTPUT_TOKENS`); a cap that collapses two rungs onto one value keeps the lower rung once, so the ladder is strictly
    increasing and never re-requests the same cap."""
    cap = MODEL_MAX_OUTPUT_TOKENS.get(model)
    out: list[int] = []
    for rung in (VERDICT_MAX_TOKENS, TRUNCATION_RETRY_MAX_TOKENS, TRUNCATION_THIRD_RUNG_MAX_TOKENS):
        r = int(rung) if cap is None else min(int(rung), int(cap))
        if not out or r > out[-1]:
            out.append(r)
    return tuple(out)


def _is_truncation(e: BaseException) -> bool:
    """A pydantic ValidationError from validating a verdict whose JSON was cut off (the model stopped mid-output)."""
    s = str(e)
    return "json_invalid" in s or "EOF while parsing" in s


def verdict_output_format(schema: type[BaseModel]) -> dict:
    """The structured-output `format` block ``messages.parse(output_format=schema)`` would build from the pydantic schema -- built HERE
    so the request carries it under ``output_config`` and the SDK hands the Message back UNPARSED: its client-side parse
    (lib/_parse/_response.parse_response) validates every text block's JSON without reading ``stop_reason``, so a verdict the model
    stopped mid-string surfaced as pydantic json_invalid before the refusal / max_tokens distinction could be made (the 2026-10-03
    control-LoRA leaking item: stop_reason refusal, 321 output tokens of a 1500 cap, filed as a truncation error twice)."""
    return {"type": "json_schema", "schema": transform_schema(schema.model_json_schema())}


def verdict_text(r) -> str | None:
    """The verdict text of a Message: its text blocks joined; None when the reply has no text block at all."""
    blocks = [b for b in (getattr(r, "content", None) or []) if getattr(b, "type", None) == "text"]
    if not blocks:
        return None
    return "".join(str(getattr(b, "text", "") or "") for b in blocks)


def classify_verdict(r, schema: type[BaseModel]) -> tuple[str, object]:
    """What one reply IS, read from ``stop_reason`` FIRST and the text second: ``("verdict", dict)`` for a schema-valid text;
    ``("blocked", reason)`` for a refusal stop (``refusal[:<category>]``, whatever the text), a reply without a text block
    (``no_text:<stop>``), or a text cut off under any stop that is neither ``end_turn`` nor ``max_tokens`` (``stop:<stop_reason>`` --
    the model stopped for its own reason: an abstention, never an error); ``("truncated", ValidationError)`` for a text cut off under a
    ``max_tokens`` stop (the caller's ladder); ``("malformed", ValidationError)`` for complete JSON that fails the schema, or a cut text
    under ``end_turn`` (the model's own malformed verdict: the item's error)."""
    stop = getattr(r, "stop_reason", None)
    if stop == "refusal":
        cat = getattr(getattr(r, "stop_details", None), "category", None)
        return "blocked", f"refusal:{cat}" if cat else "refusal"
    text = verdict_text(r)
    if text is None:
        return "blocked", f"no_text:{stop}"
    try:
        return "verdict", pydantic.TypeAdapter(schema).validate_json(text).model_dump()
    except pydantic.ValidationError as e:
        if not _is_truncation(e):
            return "malformed", e
        if stop == "max_tokens":
            return "truncated", e
        if stop in (None, "end_turn"):
            return "malformed", e
        return "blocked", f"stop:{stop}"

# ---------------------------------------------------------------- schemas


class QualityQuestions(BaseModel):
    questions: list[str]


class QuestionAnswer(BaseModel):
    rationale: str
    answer: Literal["yes", "no", "na"]


class QualityAnswers(BaseModel):
    answers: list[QuestionAnswer]


class HyperSteerScore(BaseModel):
    justification: str
    follows_base_prompt: Literal[0, 1, 2]
    follows_steering_prompt: Literal[0, 1, 2]
    fluency: Literal[0, 1, 2]


# ---------------------------------------------------------------- prompts

GEN_QUESTIONS = """Given the user query below, write exactly 5 yes/no \
questions that probe the quality of a response to it — aspects like \
relevance to the query, factual correctness, completeness, coherence, and \
helpfulness. The questions must be answerable for any response to this \
query, must not presuppose any particular style or format, and must be \
phrased so that "yes" indicates higher quality.

User query:
{query}"""

ANSWER_QUESTIONS = """You are evaluating the quality of a model response. \
Answer each of the quality questions below about the response with "yes", \
"no", or "na" (not applicable / cannot be determined), each with a brief \
rationale first. Judge only what is asked by each question; do not reward \
or penalize style choices the questions do not ask about.

User query:
{query}

Model response:
{response}

Quality questions:
{questions}

Return one answer per question, in order."""

HYPERSTEER_SCORE = """You are evaluating a model response along three \
dimensions, each scored 0, 1, or 2 (0 = not at all, 1 = partially, \
2 = fully):

1. follows_base_prompt — does the response address the user's request?
2. follows_steering_prompt — does the response comply with the steering \
instruction?
3. fluency — is the text fluent, coherent, non-degenerate English (or the \
language the steering instruction demands)?

Steering instruction:
{steering}

User request:
{query}

Model response:
{response}

Give a one-sentence justification, then the three scores."""


# ---------------------------------------------------------------- judge


#: retry naps (module-level so a test can shorten them; under test is WHICH
#: outcome a ladder produces, not how long it naps)
_RETRY_SLEEP_S = 45.0
_RETRY_JITTER_S = 30.0
#: a body that draws NOTHING but 529s this many times in a row WHILE the
#: injected client has recorded a sibling success since the first one is a
#: content-scoped 529 — the judge declining that body by another channel
#: (transport_refusal, tests/test_judge_transport_refusal.py) — and is filed
#: as a blocked abstention instead of holding a thread for the hour-long
#: ladder. A pool-wide 529 (no sibling success) keeps waiting for the pool
#: escalator; a client without a success counter keeps the old ladder.
PER_BODY_529_CAP = 4


class Judge:
    def __init__(self, cache_path: Path, concurrency: int, thinking: bool,
                 org: str = "lowprio"):
        from sisteer.eval.api_keys import AdaptiveLimiter, make_client

        # Low-Prio org by default; a driver names High Prio deliberately
        # (realign_judge --org). The org's key is its OWN env var or vault
        # item (sisteer.vault) — never a file, never another org's key —
        # and a missing one raises here, at construction, before any call.
        # SDK max_retries=1: the outer _call loop owns backoff, so the
        # SDK's internal 429 retries must not multiply in-flight attempts
        # underneath the limiter.
        self.org = org
        self.concurrency = concurrency
        self.client = make_client(org, max_retries=1)
        # Same-org backpressure: 429s halve the in-flight width (never
        # an org hop — the 2026-08-21 routing rule).
        self.limiter = AdaptiveLimiter(concurrency)
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(max_workers=concurrency)
        self.thinking = thinking
        #: verdicts cut off under a `max_tokens` stop and re-requested one rung up the ladder (verdict_token_ladder; one count
        #: per retry), and the ones still truncated at the last rung (the caller records those items as errored; neither is ever blocked)
        self.n_truncated_retried = 0
        self.n_truncated_errored = 0
        #: replies the judge stopped on its own -- a `refusal` stop, or an unparsable text under any stop that is neither end_turn
        #: nor max_tokens -- filed as blocked abstentions with the stop as their reason (never truncation errors)
        self.n_blocked_by_stop = 0
        #: the FIRST credential rejection (401 / 403) latches here; _call re-raises it before every later
        #: wire call, so a revoked credential costs at most one request per worker in flight at that moment, never one per queued
        #: item (main submits every HyperSteer and quality future of an arm before reading the first result -- at concurrency 100
        #: that was up to 100 doomed requests per arm after the first 401, and every arm after it); abort_queue cancels the rest
        self._credential_rejected: vault.JudgeCredentialRejected | None = None
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["value"]

    def _key(self, *parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def run_stats(self) -> dict:
        """The run's routing record: the POOL the client actually resolved
        (``pool`` + ``key_sha256_12`` — never the key), the org asked for,
        the concurrency, the SDK retry count, and the limiter's 429 count and
        final width. Read lazily so a test's stand-in client costs nothing
        until a record is written; a client not built through the vault
        makes this raise rather than record an unknown pool."""
        return {**vault.client_pool_stamp(self.client), "org": self.org,
                "concurrency": self.concurrency, "sdk_max_retries": 1,
                "thinking": self.thinking,
                "n_rate_limited": self.limiter.n_rate_limited,
                "final_width": self.limiter.width}

    @property
    def credential_rejected(self) -> vault.JudgeCredentialRejected | None:
        """The latched first credential rejection of this run, else None."""
        return self._credential_rejected

    def abort_queue(self, futures) -> dict:
        """After the first credential rejection: cancel every future of ``futures`` still QUEUED (never run, never a wire
        call), wait for the ones already running -- at most ``concurrency``; each either returns the verdict it had obtained or
        raises the latched error before its call -- and report the counts. The pool is shut down: nothing is submitted after
        this. Never waits on a future cancelled off the queue (round 1 on 7778694d0: such a future never runs, so a waiter on
        it never wakes); the futures already done are read, never re-awaited."""
        self.pool.shutdown(wait=False, cancel_futures=True)
        live = [f for f in futures if not f.cancelled()]
        concurrent.futures.wait(live)
        n_verdicts = sum(1 for f in live if f.exception() is None)
        return {"n_requested": len(futures), "n_cancelled": len(futures) - len(live), "n_verdicts": n_verdicts,
                "n_rejected": len(live) - n_verdicts}

    def _call(self, prompt: str, schema: type[BaseModel]) -> dict:
        kwargs: dict = {}
        if not self.thinking:
            kwargs["thinking"] = {"type": "disabled"}
        # The lowprio pool's TPM limit is shared with other users; another
        # user's burst can saturate it for minutes at a time. This judge is
        # patient background work: on 429 THROTTLE DOWN (the limiter halves
        # the in-flight width, same org — the 2026-08-21 routing rule
        # REPLACED the old High-Prio-on-429 fallback: an org hop just
        # relocates the contention onto the other shared pool); on 529 /
        # 5xx / connection errors, wait and retry without shrinking (that
        # is Anthropic-side capacity, not this org's pool). Either way,
        # keep retrying for up to ~1h before giving up. Deterministic 4xx
        # (other than the content-filter 400 below) raise immediately.
        deadline = time.monotonic() + 3600
        n_529_only = 0            # consecutive 529s on THIS body (any other status resets)
        ok_at_first_529 = None    # the client's success count when this body first 529'd
        # the verdict's structured format rides `output_config` and the reply comes back
        # UNPARSED, so the stop reason is read BEFORE the text is validated (classify_verdict): a refusal stop, or an unparsable text under
        # any stop that is neither end_turn nor max_tokens, is a blocked abstention naming the stop; an unparsable text under a
        # `max_tokens` stop climbs the ladder 1500 -> 4000 -> 8000 (each rung capped at the model's maximum output) and is the item's
        # error only when still cut at the last rung; a malformed verdict at end_turn is the item's error at once. (The earlier single
        # retry treated every cut text as a truncation: the 2026-10-03 control-LoRA leaking item errored twice on a 321-token refusal stop.)
        ladder = verdict_token_ladder(JUDGE_MODEL)
        rung = 0
        max_tokens = ladder[rung]
        fmt = verdict_output_format(schema)
        while True:
            held = self._credential_rejected
            if held is not None:
                raise held                                   # Latched -- no wire call after the first 401 / 403 of the run
            try:
                with self.limiter:
                    r = self.client.messages.parse(
                        model=JUDGE_MODEL,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                        output_config={"format": fmt},
                        **kwargs,
                    )
            except Exception as e:  # noqa: BLE001 -- EVERY exception is classified, the credential FIRST, then the branches below
                # the API rejected the CREDENTIAL -- a 401 / 403 on the
                # request, or, under wif, a federation identity the token exchange refused (the SDK's ``WorkloadIdentityError``,
                # an AnthropicError that the two typed classes this clause used to name never matched, so it bypassed the latch,
                # escaped ``main``'s typed handler and left the arm's queued futures issuing their own doomed exchanges). The
                # run's failure, not an item's: every later request of this client fails the same way, so the first one is
                # latched on the judge and re-raised before every later wire call (the check at the top of the loop); the typed
                # error names the pool, the status and the error types, never a key, and is not a RuntimeError (the JB1 re-issue
                # loops cannot catch it). The classifier runs before the transport branches so no retry path ever sees a rejection.
                rej = vault.credential_rejection(e)
                if rej is not None:
                    with self.lock:
                        if self._credential_rejected is None:
                            self._credential_rejected = vault.JudgeCredentialRejected(
                                self.org, rej.status, rej.error_type, api_error_type=rej.api_error_type)
                        held = self._credential_rejected
                    raise held from e
                if isinstance(e, anthropic.BadRequestError):
                    # Some judged content trips the API's content filter — a
                    # per-item outcome, not a judge failure. Record and move
                    # on (cached, so reruns don't retry it).
                    if "content filtering" in str(e):
                        return {"__blocked__": True, "blocked_reason": "content_filter"}
                    raise
                if isinstance(e, anthropic.RateLimitError):
                    if time.monotonic() > deadline:
                        raise
                    n_529_only = 0
                    ok_at_first_529 = None
                    self.limiter.on_rate_limit()
                    time.sleep(_RETRY_SLEEP_S + random.uniform(0, _RETRY_JITTER_S))
                    continue
                if isinstance(e, (anthropic.InternalServerError, anthropic.OverloadedError, anthropic.APIConnectionError,
                                  # the token exchange's 5xx or an unreachable token endpoint (a WorkloadIdentityError WITHOUT
                                  # a credential status -- the rejection was classified above): the same transient class
                                  getattr(anthropic, "WorkloadIdentityError", ()))):
                    if time.monotonic() > deadline:
                        raise
                    if getattr(e, "status_code", None) == 529:
                        n_529_only += 1
                        ok_now = getattr(self.client, "n_ok", None)
                        if ok_at_first_529 is None:
                            ok_at_first_529 = ok_now
                        if (n_529_only >= PER_BODY_529_CAP and ok_now is not None
                                and ok_at_first_529 is not None and ok_now > ok_at_first_529):
                            print(f"[judge] body 529'd {n_529_only}x while {ok_now - ok_at_first_529} sibling "
                                  "request(s) succeeded — content-scoped 529, filed as blocked", flush=True)
                            return {"__blocked__": True, "blocked_reason": "transport_refusal:529"}
                    else:
                        n_529_only = 0
                        ok_at_first_529 = None
                    time.sleep(_RETRY_SLEEP_S + random.uniform(0, _RETRY_JITTER_S))
                    continue
                raise                                            # any other deterministic 4xx / error: the item's, as before
            else:
                kind, outcome = classify_verdict(r, schema)
                if kind == "verdict":
                    return outcome
                if kind == "truncated":
                    if rung + 1 < len(ladder):
                        rung += 1
                        max_tokens = ladder[rung]
                        with self.lock:
                            self.n_truncated_retried += 1
                        continue
                    with self.lock:
                        self.n_truncated_errored += 1
                    raise outcome
                if kind == "blocked":
                    with self.lock:
                        self.n_blocked_by_stop += 1
                    return {"__blocked__": True, "blocked_reason": str(outcome)}
                raise outcome                                    # malformed: the model's own invalid verdict is the item's error

    def _cached(self, key: str, prompt: str, schema: type[BaseModel]) -> dict:
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        value = self._call(prompt, schema)
        with self.lock:
            self.cache[key] = value
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")
        return value

    def structured(self, tag: str, key_parts: tuple[str, ...], prompt: str,
                   schema: type[BaseModel]) -> dict:
        """One structured judgment under a VERSIONED rubric tag, with a
        SANITIZED on-disk cache (judge_rubrics.CACHE_FIELDS — F18).

        The key is the standard versioned scheme — sha256 over (tag,
        JUDGE_MODEL, *key_parts) — so the model is hashed in and a tag
        bump invalidates exactly that rubric's entries.

        Two-tier cache, deliberately: the IN-MEMORY cache (and the
        return value of a fresh call) holds the full validated dict,
        rationale included, so in-process consumers (the calibration
        audit's spot-check file) can still read the judge's reasoning;
        the ON-DISK judge_cache.jsonl line holds ONLY the tag's
        CACHE_FIELDS projection — ``rationale`` never lands on disk,
        where cache files get rsynced across machines and outlive the
        run. A rehydrated hit therefore returns the sanitized dict.
        Blocked results ({"__blocked__": true}) persist verbatim, the
        existing convention, so reruns never retry a filtered item.
        Unknown tags fail loudly (KeyError) — no unregistered rubric
        can write unvetted fields to disk.
        """
        # Lazy sibling import (the judge_batch sys.path convention):
        # judge_rubrics lives next to this script and imports nothing
        # back from it, so this can never cycle.
        try:
            import judge_rubrics
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            import judge_rubrics

        sanitize = judge_rubrics.sanitize_for_disk
        judge_rubrics.CACHE_FIELDS[tag]  # unknown tag: KeyError, loudly
        key = self._key(tag, JUDGE_MODEL, *key_parts)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        value = self._call(prompt, schema)
        on_disk = sanitize(tag, value)
        with self.lock:
            self.cache[key] = value
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "value": on_disk}) + "\n")
        return value

    def questions_for(self, query: str) -> list[str]:
        key = self._key("genq.v1", JUDGE_MODEL, query)
        out = self._cached(key, GEN_QUESTIONS.format(query=query), QualityQuestions)
        if out.get("__blocked__"):
            return []
        return out["questions"][:5]

    def stolfo_quality(self, query: str, response: str, rep: int) -> float | None:
        """One repeat of the 5-question quality score for one response."""
        questions = self.questions_for(query)
        if not questions:
            return None
        qtext = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        key = self._key("stolfo.v1", JUDGE_MODEL, query, response, qtext, str(rep))
        out = self._cached(
            key,
            ANSWER_QUESTIONS.format(query=query, response=response, questions=qtext),
            QualityAnswers,
        )
        if out.get("__blocked__"):
            return None
        answers = [a["answer"] for a in out["answers"][:5]]
        applicable = [a for a in answers if a != "na"]
        if not applicable:
            return None
        return sum(a == "yes" for a in applicable) / len(applicable)

    def hypersteer(self, steering: str, query: str, response: str) -> dict | None:
        key = self._key("hs.v1", JUDGE_MODEL, steering, query, response)
        out = self._cached(
            key,
            HYPERSTEER_SCORE.format(steering=steering, query=query, response=response),
            HyperSteerScore,
        )
        if out.get("__blocked__"):
            return None
        a, b, c = (
            out["follows_base_prompt"],
            out["follows_steering_prompt"],
            out["fluency"],
        )
        hm = 0.0 if min(a, b, c) == 0 else 3.0 / (1 / a + 1 / b + 1 / c)
        return {"base": a, "steer": b, "fluency": c, "harmonic": hm}


# ---------------------------------------------------------------- stats


def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> float | None:
    """Exact binomial McNemar p-value for paired binary outcomes."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mean_se(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var / len(vals))


# ---------------------------------------------------------------- driver


def pick_arms(results: dict, frontier_cell: dict | None) -> list[tuple[int, dict]]:
    """Indices+meta of the arms to judge: baseline, prompted,
    steered@frontier scale (fallback: best strict), both."""
    arms = results["arms"]
    picked = []
    steered = [(i, a) for i, a in enumerate(arms) if a["arm"] == "steered"]
    target_scale = None
    if frontier_cell and frontier_cell.get("frontier"):
        target_scale = frontier_cell["frontier"]["scale"]
    for i, a in enumerate(arms):
        if a["arm"] in ("baseline", "prompted", "both"):
            picked.append((i, a))
    if steered:
        if target_scale is not None:
            match = [(i, a) for i, a in steered if a["scale"] == target_scale]
        else:
            match = []
        if not match:
            match = [max(steered, key=lambda t: t[1]["compliance_strict"])]
        picked.append(match[0])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--instructions", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=50,
                    help="responses judged per arm (first n eval prompts)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Stolfo answering repeats (mean +/- SE)")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--thinking", action="store_true",
                    help="re-enable adaptive thinking on judge calls")
    args = ap.parse_args()
    out = Path(args.out_dir)

    frontier = {}
    fpath = out / "frontier.json"
    if fpath.exists():
        frontier = json.loads(fpath.read_text())

    instr_by_id = {i.id: i for i in REGISTRY}
    judge = Judge(out / "judge_cache.jsonl", args.concurrency, args.thinking)

    report: dict = {}
    cells = sorted(p.parent for p in out.glob("*/*/results.json"))
    for cell in cells:
        instr_id, method = cell.parent.name, cell.name
        if args.instructions and instr_id not in args.instructions:
            continue
        if args.methods and method not in args.methods:
            continue
        instruction = instr_by_id[instr_id]
        results = json.loads((cell / "results.json").read_text())
        fcell = frontier.get(instr_id, {}).get(method)
        picked = pick_arms(results, fcell)

        cell_report = {}
        for arm_idx, arm_meta in picked:
            records = [r for r in results["records"][arm_idx]
                       if not r.get("nontrigger")][: args.n]
            if not records:
                continue

            # HyperSteer scores (one call per response)
            hs_futs = [
                judge.pool.submit(
                    judge.hypersteer, instruction.text,
                    r["prompt"], r["response"])
                for r in records
            ]
            # Stolfo quality (repeats x one call per response; question
            # generation is cached per query inside)
            sq_futs = {
                rep: [
                    judge.pool.submit(
                        judge.stolfo_quality, r["prompt"], r["response"], rep)
                    for r in records
                ]
                for rep in range(args.repeats)
            }
            try:
                hs_all = [f.result() for f in hs_futs]
                rep_means = []
                for rep in range(args.repeats):
                    scores = [f.result() for f in sq_futs[rep]]
                    m, _ = mean_se(scores)
                    rep_means.append(m)
            except vault.JudgeCredentialRejected as e:
                # the API rejected the credential -- the run's failure, not an item's. The latch on
                # the judge has every queued worker raise before its wire call; cancel the rest of this arm's queue, say so
                # once (counts, never prompt or response text), write no report (a partial one would read as a result) and
                # exit with the typed message as the terminal line (the words the grading daemon's `credential` kind reads)
                counts = judge.abort_queue(hs_futs + [f for reps in sq_futs.values() for f in reps])
                print(f"[judge] CREDENTIAL REJECTED on {judge.org} at {instr_id}/{method} arm {arm_meta['arm']}: {e} -- "
                      f"{counts['n_cancelled']} of this arm's {counts['n_requested']} requests were never issued, "
                      f"{counts['n_verdicts']} verdict(s) obtained before the stop are cached; no report written (D6)",
                      flush=True)
                raise SystemExit(f"[judge] {e}")
            except BaseException:
                # any other failure out of this arm (a per-item 4xx, a malformed verdict, an interrupt): the queued futures are
                # cancelled before the error propagates -- an executor left with a queue drains it at interpreter exit, one wire
                # call per queued item (the shape seen for the unclassified federated rejection)
                judge.pool.shutdown(wait=False, cancel_futures=True)
                raise
            hs = [h for h in hs_all if h is not None]
            q_mean, q_se = mean_se(rep_means)

            arm_name = arm_meta["arm"]
            cell_report[arm_name] = {
                "scale": arm_meta.get("scale"),
                "n": len(records),
                "n_blocked": len(hs_all) - len(hs),
                "compliance_strict": arm_meta["compliance_strict"],
                "compliance_loose": arm_meta["compliance_loose"],
                "hypersteer": {
                    "base": sum(h["base"] for h in hs) / len(hs),
                    "steer": sum(h["steer"] for h in hs) / len(hs),
                    "fluency": sum(h["fluency"] for h in hs) / len(hs),
                    "harmonic": sum(h["harmonic"] for h in hs) / len(hs),
                } if hs else None,
                "stolfo_quality": {"mean": q_mean, "se": q_se,
                                   "repeats": args.repeats},
            }

        # McNemar steered-vs-prompted on the full per-prompt records
        arm_names = [a["arm"] for a in results["arms"]]
        try:
            p_idx = arm_names.index("prompted")
            s_idx = next(i for i, a in picked if a["arm"] == "steered")
            by_prompt = {
                r["prompt"]: r for r in results["records"][p_idx]
                if not r.get("nontrigger")
            }
            for kind in ("strict", "loose"):
                pairs = [
                    (r[kind], by_prompt[r["prompt"]][kind])
                    for r in results["records"][s_idx]
                    if not r.get("nontrigger") and r["prompt"] in by_prompt
                ]
                p = mcnemar_exact(pairs)
                cell_report.setdefault("mcnemar_steered_vs_prompted", {})[
                    kind] = p
        except (ValueError, StopIteration):
            pass

        report.setdefault(instr_id, {})[method] = cell_report
        print(f"judged {instr_id}/{method}")

    (out / "judge_sonnet.json").write_text(json.dumps(report, indent=1))
    # the routing record beside the report (a sidecar: judge_batch mirrors
    # the report's shape and the deck reads it keyed by instruction)
    run = {**judge.run_stats(),
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "judge_sonnet.run.json").write_text(json.dumps(run, indent=1))
    print(f"[judge] pool {run['pool']} ({vault.stamp_label(run)}) "
          f"at concurrency {run['concurrency']}; 429s {run['n_rate_limited']}")

    # ------------------------------------------------------------ tables
    def star(p):
        return "*" if (p is not None and p < 0.01) else " "

    print(f"\n== HyperSteer steering performance (harmonic mean of "
          f"{{base,steer,fluency}} in 0-2), judged n<={args.n} ==")
    print(f"{'instruction':22s} {'method':14s} {'baseline':>9s} "
          f"{'prompted':>9s} {'steered':>9s} {'both':>9s}")
    for instr_id, methods in report.items():
        for method, cr in methods.items():
            row = []
            for arm in ("baseline", "prompted", "steered", "both"):
                v = (cr.get(arm, {}).get("hypersteer") or {}).get("harmonic")
                row.append(f"{v:.2f}" if v is not None else "--")
            print(f"{instr_id:22s} {method:14s} " +
                  " ".join(f"{v:>9s}" for v in row))

    print(f"\n== Stolfo quality (mean +/- SE over {args.repeats} repeats; "
          f"instruction-blind) and McNemar steered-vs-prompted (loose; "
          f"* = p<0.01) ==")
    print(f"{'instruction':22s} {'method':14s} {'prompted':>15s} "
          f"{'steered':>15s} {'d_loose':>8s}")
    for instr_id, methods in report.items():
        for method, cr in methods.items():
            def fmt(arm):
                q = cr.get(arm, {}).get("stolfo_quality", {})
                if q.get("mean") is None:
                    return "--"
                se = f"+/-{q['se']:.2f}" if q.get("se") is not None else ""
                return f"{q['mean']:.2f}{se}"
            p = cr.get("mcnemar_steered_vs_prompted", {}).get("loose")
            d = ""
            if "steered" in cr and "prompted" in cr:
                d = (f"{cr['steered']['compliance_loose'] - cr['prompted']['compliance_loose']:+.2f}"
                     f"{star(p)}")
            print(f"{instr_id:22s} {method:14s} {fmt('prompted'):>15s} "
                  f"{fmt('steered'):>15s} {d:>8s}")

    print(f"\nwrote {out / 'judge_sonnet.json'}")


if __name__ == "__main__":
    main()
