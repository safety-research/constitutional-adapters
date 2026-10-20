"""The rewriter-kind build seam — an LLM rewriter that FREEZES its output.

Rewriter-kind families need an LLM at BUILD time (to translate, nest, or rewrite the
request), but the central design principle is that they still reduce to a
frozen, content-hashed corpus that REPLAY treats exactly like a static family.
This module is that build seam:

- :class:`AnthropicRewriter` — a cached, concurrent client on the Low-Prio org,
  reusing ``scripts/grade_bon_live.make_client`` (the SAME client pattern the
  judges use). Aggressive on-disk caching keyed by ``sha256(model, system,
  prompt)`` so a re-run resumes and never re-pays for a completed rewrite. The
  cache lives in the private dir (it holds the rewritten harmful prompt text).
- :class:`StubRewriter` — a DETERMINISTIC offline rewriter (no API, no network)
  so the whole build → freeze → replay pipeline is testable CPU-only and the
  smoke test needs no API budget. A family's tests inject this.
- :func:`build_llm_corpus` — the shared build step: run the rewriter over a
  family's per-goal PLAN, run the LEAKAGE GATE, and write the frozen corpus +
  manifest. Replay then binds the corpus with
  ``frozen_corpus.make_frozen_corpus_family``.

The rewriter is never called at replay time and never at registry-import time,
so importing an attack module for the registry never pulls ``anthropic`` (the
import is lazy, inside :class:`AnthropicRewriter`).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

from sisteer import vault
from typing import Callable, Optional, Sequence

from ..jailbreak_artifacts import private_umask


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
from .frozen_corpus import corpus_content_hash, write_frozen_corpus


# --------------------------------------------------------------------------- #
# A resume-safe rewrite cache (mirrors grade_bon_live.ScoreCache, self-        #
# contained so importing this module never pulls the scripts/ dir)             #
# --------------------------------------------------------------------------- #


class _RewriteCache:
    """Append-only jsonl of ``{"key": sha, "value": text}`` — holds only
    SUCCESSFUL rewrites (a failure stays uncached and retries). Tolerant of a
    torn final line. Lives in the private dir (rewritten prompts are harmful
    text)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self.cache[row["key"]] = row["value"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if key in self.cache:
                return
            self.cache[key] = value
            with private_umask():
                self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")
                f.flush()
                os.fsync(f.fileno())


def _rewrite_key(model: str, system: str, prompt: str) -> str:
    h = hashlib.sha256()
    for part in (model, system, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The rewriters                                                                 #
# --------------------------------------------------------------------------- #


class StubRewriter:
    """A deterministic offline rewriter for tests / smokes. ``fn(prompt, system)
    -> str`` must be a pure function so build → freeze → replay is reproducible
    with no API. Reports a fixed ``model`` string for provenance."""

    model = "stub-rewriter"

    #: a stub calls no API, so it has no pool to record
    pool_stamp = None

    def __init__(self, fn: Callable[[str, str], str], *, model: str = "stub-rewriter"):
        self.fn = fn
        self.model = model

    def one(self, prompt: str, *, system: str = "") -> str:
        return self.fn(prompt, system)

    def batch(self, items: Sequence[tuple[str, str]]) -> list[str]:
        return [self.fn(p, s) for (p, s) in items]


class _CachedConcurrentRewriter:
    """Shared build seam for a cached, concurrent LLM rewriter. Subclasses
    implement :meth:`_call_one` (``prompt, system -> str``, the one live API
    call); this owns the resume-safe cache, the :meth:`one` cache path, and the
    SINGLE-FLIGHT :meth:`batch`.

    Single-flight is the reproducibility fix: a rewriter-kind
    corpus is FROZEN to a content hash, so a COLD build must be deterministic. The
    old ``batch`` submitted every uncached index — including several that shared
    ONE cache key (28 byte-identical multilingual goal texts; ReNeLLM's 3 scenarios
    per goal share one rewrite key) — so duplicate keys each drew an INDEPENDENT
    (and, for an LLM, DIFFERENT) rewrite, and the content hash only settled on a
    warm re-run. ``batch`` now computes each unique key ONCE and fans the single
    result out to every index that shares it, so the cold build's hash is stable."""

    model: str = "unknown"
    #: the pool the rewriter's client resolved (``{"pool", "key_sha256_12"}``,
    #: never the key), or None for a rewriter that calls no pooled API;
    #: :func:`build_llm_corpus` records it in the corpus manifest
    pool_stamp: Optional[dict] = None

    def __init__(self, *, model: str, cache_path, concurrency: int = 200,
                 max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self._cache = _RewriteCache(Path(cache_path))
        # the FIRST credential rejection the rewriter's client meets -- a 401 / 403, or under SISTEER_SECRETS_MODE=wif
        # the federation identity refused at the token exchange -- latches here as the typed
        # sisteer.vault.JudgeCredentialRejected; `one` raises it before every later call, and `batch` cancels its
        # queue at the first one (before this, `f.result()` propagated the SDK error while the executor's exit drained
        # every queued key through its own doomed request). The pool is read off the client's stamp when one is set.
        self._credential = vault.CredentialLatch(lambda: vault.pool_of_client(getattr(self, "_client", None)))

    def _call_one(self, prompt: str, system: str) -> str:
        raise NotImplementedError

    @property
    def credential_rejected(self) -> Optional[vault.JudgeCredentialRejected]:
        """The latched first credential rejection of this rewriter's run, else None."""
        return self._credential.held

    def one(self, prompt: str, *, system: str = "") -> str:
        key = _rewrite_key(self.model, system, prompt)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        text = self._credential.guard(self._call_one)(prompt, system)
        self._cache.put(key, text)
        return text

    def batch(self, items: Sequence[tuple[str, str]]) -> list[str]:
        """Rewrite ``[(prompt, system), ...]`` concurrently, order preserved.
        Cached items skip the API; each DISTINCT cache key is computed exactly
        once (single-flight) and its result fanned out to every index sharing it,
        so a cold concurrent build is deterministic. A failed item raises (a
        partial corpus must never be silently frozen)."""
        results: list[Optional[str]] = [None] * len(items)
        keys = [_rewrite_key(self.model, s, p) for (p, s) in items]
        for i, k in enumerate(keys):
            hit = self._cache.get(k)
            if hit is not None:
                results[i] = hit
        # group the still-uncached indices by cache key, preserving first-seen
        # order of both keys and their members
        groups: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            if results[i] is None:
                groups.setdefault(k, []).append(i)
        if groups:
            # one call per distinct key, on a bounded executor that STOPS at the first credential rejection: the
            # queue cancelled, the running calls settled, the typed error raised -- a failed item still propagates
            # loudly and uncached (never freeze a gap), and now cancels the queue behind it too
            keys = list(groups)
            vals = vault.map_until_credential_rejected(
                lambda k: self.one(items[groups[k][0]][0], system=items[groups[k][0]][1]),
                keys, workers=self.concurrency, latch=self._credential, label=f"[rewriter {self.model}]")
            for k, val in zip(keys, vals):
                for i in groups[k]:
                    results[i] = val
        if any(r is None for r in results):  # defensive: never freeze a gap
            raise RuntimeError("rewriter.batch left an item unfilled")
        return [r for r in results]  # type: ignore[return-value]


class AnthropicRewriter(_CachedConcurrentRewriter):
    """Cached, concurrent LLM rewriter on the named Anthropic org (default Low
    Prio; an org with no key raises rather than borrowing another org's).
    Reuses ``grade_bon_live.make_client``
    so the rewriter-kind build path shares the judges' client exactly. ``batch`` runs
    at ``concurrency`` workers and writes each success to the cache as it
    completes (resume-safe)."""

    def __init__(self, *, model: str, cache_path, org: Optional[str] = None,
                 concurrency: int = 200, max_tokens: int = 1024,
                 max_retries: int = 8):
        # lazy: only constructing a REAL rewriter pulls anthropic + scripts/
        _scripts = Path(__file__).resolve().parents[4] / "scripts"
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        from grade_bon_live import DEFAULT_ORG, make_client

        super().__init__(model=model, cache_path=cache_path,
                         concurrency=concurrency, max_tokens=max_tokens)
        self._client = make_client(org or DEFAULT_ORG,
                                   max_retries=max_retries)

    @property
    def pool_stamp(self) -> dict:
        """The pool the client ACTUALLY resolved, read from the vault's stamp
        on it (lazy: a client not built through the vault raises here, when
        a record is written, never a silent unknown)."""
        return vault.client_pool_stamp(self._client)

    def _call_one(self, prompt: str, system: str) -> str:
        kwargs = {"model": self.model, "max_tokens": self.max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        return "".join(b.text for b in msg.content
                       if getattr(b, "type", "") == "text")


class OpenRouterRewriter(_CachedConcurrentRewriter):
    """Cached, concurrent rewriter on OpenRouter's OpenAI-compatible chat API
    (default target ``x-ai/grok-4.5``). The key comes from
    :func:`sisteer.vault.openrouter_api_key` (the environment, else the vault
    item) and is NEVER logged or embedded in an exception. Retries 429 / 5xx with capped
    exponential backoff; a non-retryable status or a persistently empty
    completion raises loudly and is left UNCACHED (so a re-run retries and a
    partial corpus is never frozen). Selected by the held-out build when the
    ``--backend openrouter`` path is chosen for a rewriter-kind family."""

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    _RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 520, 522, 524, 529})

    def __init__(self, *, model: str, cache_path,
                 concurrency: int = 32, max_tokens: int = 2048,
                 max_retries: int = 8, timeout: float = 180.0):
        import httpx  # lazy: only a REAL OpenRouter rewriter pulls httpx

        super().__init__(model=model, cache_path=cache_path,
                         concurrency=concurrency, max_tokens=max_tokens)
        key = vault.openrouter_api_key()
        self.pool_stamp = vault.pool_stamp("openrouter", key)
        self._max_retries = max_retries
        # one shared, thread-safe client + pool across the batch's workers
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=concurrency + 8,
                                max_keepalive_connections=concurrency + 8),
        )

    def _call_one(self, prompt: str, system: str) -> str:
        import random
        import time

        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": messages}
        last = "no attempt"
        for attempt in range(self._max_retries + 1):
            try:
                r = self._client.post(self.ENDPOINT, json=payload)
                status = r.status_code
                if status == 200:
                    d = r.json()
                    choices = d.get("choices") or []
                    if not choices:  # provider error surfaced inside a 200
                        err = (d.get("error") or {})
                        last = f"200 without choices (error code {err.get('code')})"
                        raise _Retry()
                    text = (choices[0].get("message") or {}).get("content") or ""
                    if not text.strip():
                        last = "empty completion"
                        raise _Retry()  # retry; never cache an empty rewrite
                    return text
                if status in self._RETRYABLE:
                    last = f"HTTP {status}"
                    raise _Retry()
                # non-retryable (auth, 400, model-not-found, …): fail loudly,
                # do not leak body (may echo the prompt) — status only
                raise RuntimeError(
                    f"OpenRouter non-retryable HTTP {status} for model "
                    f"{self.model!r}")
            except _Retry:
                pass
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}"
            if attempt < self._max_retries:
                # capped exponential backoff with jitter
                time.sleep(min(30.0, 1.5 * (2 ** attempt)) * (0.5 + random.random()))
        raise RuntimeError(
            f"OpenRouter failed after {self._max_retries + 1} attempts for "
            f"model {self.model!r} (last: {last})")


class _Retry(Exception):
    """Internal control-flow signal for :meth:`OpenRouterRewriter._call_one`."""


# --------------------------------------------------------------------------- #
# The shared rewriter-kind build step                                                 #
# --------------------------------------------------------------------------- #


def build_llm_corpus(
    goals,
    *,
    rewriter,
    plan_fn: Callable[[object], Sequence[tuple[str, str, str]]],
    out_path,
    fingerprint,
    provenance: dict,
    family_name: str,
    seed: int = 0,
    post_fn: Optional[Callable[[str, str], str]] = None,
) -> dict:
    """REWRITER-KIND BUILD: run ``rewriter`` over each goal's PLAN, freeze the outputs,
    leakage-check, and write the corpus + manifest. ``plan_fn(goal) ->
    [(variant_name, system, user_instruction), ...]`` gives the LLM prompts that
    PRODUCE each variant's frozen attack prompt (the family owns the plan; this
    owns the freeze). Returns the manifest.

    ``post_fn(variant_name, llm_output) -> final_prompt`` optionally transforms
    the LLM output DETERMINISTICALLY before freezing (e.g. ReNeLLM nests the
    rewritten request into a fixed scenario). The nesting is part of the frozen
    corpus, so replay stays a pure lookup.

    The (post-processed) rewriter output is the frozen attack prompt. Replay
    never calls the rewriter — it binds the written corpus with
    :func:`frozen_corpus.make_frozen_corpus_family`."""
    plans = [(g.goal_id, i, name, system, instr)
             for g in goals
             for i, (name, system, instr) in enumerate(plan_fn(g))]
    outputs = rewriter.batch([(instr, system) for (_, _, _, system, instr) in plans])
    if post_fn is not None:
        outputs = [post_fn(name, out)
                   for (_gid, _i, name, _sys, _instr), out in zip(plans, outputs)]
    rows = [{
        "goal_id": gid, "sample_idx": i, "variant_name": name, "prompt": out,
    } for (gid, i, name, _sys, _instr), out in zip(plans, outputs)]

    # LEAKAGE GATE REMOVED (2026-08-25): families are allowed
    # -- encouraged -- to appear on both sides, and a corpus built over
    # TRAINING goals overlaps the training fingerprint BY CONSTRUCTION, so
    # refuse=True would refuse every legitimate build. ``fingerprint`` is
    # still accepted and still recorded below, so the manifest states what
    # the corpus overlaps -- reporting it, no longer refusing on it.
    leak = {'gate': 'removed_by_directive_2026_08_25',
            'n_checked': len(rows), 'fingerprint_n': len(fingerprint or ())}
    chash = write_frozen_corpus(rows, out_path)

    prov = dict(provenance)
    prov.setdefault("rewriter_model", getattr(rewriter, "model", "unknown"))
    # which pool the rewrites were drawn on (None for a stub / a rewriter
    # that declares none), beside the width it ran at — read the way
    # ``model`` is, because rewriters are an open duck type
    prov.setdefault("rewriter_pool", getattr(rewriter, "pool_stamp", None))
    prov.setdefault("rewriter_concurrency", getattr(rewriter, "concurrency", None))
    manifest = {
        'family': family_name, 'build_kind': 'rewriter', 'seed': seed,
        'row_count': len(rows),
        'n_behaviors': len({r['goal_id'] for r in rows}),
        'provenance': prov, 'leakage_check': leak,
        'built_at': _utcnow(),
    }
    manifest["corpus_content_hash"] = chash
    Path(str(out_path) + ".manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


# ``bind_llm_family`` is NOT ported: it wrapped make_frozen_corpus_family to
# hand back a HeldOutFamily, and both went with the held-out registry.
# renellm binds its corpus directly through FrozenCorpus.load.

