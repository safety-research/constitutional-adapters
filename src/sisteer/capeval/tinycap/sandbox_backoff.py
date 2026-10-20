"""Retry with backoff on Modal's sandbox-creation rate limit (tinycap-verify-rate-limit, 2026-09-30).

The fact this module exists for: Modal creates sandboxes at most
``contracts.SANDBOX_CREATE_RATE_LIMIT_PER_S`` (5) per second per workspace and
refuses the rest with ``modal.exception.ResourceExhaustedError`` (gRPC
``RESOURCE_EXHAUSTED``; the message reads ``Sandbox creation rate limit (5/s)
exceeded``). The Modal SDK's own transport retry does NOT retry that status, and
inspect-sandboxes' ``_standard_retry`` around ``Sandbox.create`` gives up after
five attempts inside ~15 s — which is what the gemma (c) row exhausted on
2026-09-30 when nine CPU verifiers, each opening eight sandboxes, started ~40 ms
apart: inspect aborted every task at the first refused create and the whole row
read VERIFY_FAILED with its generations intact.

This module is PURE (no ``modal`` import) so the policy is unit-testable in every
venv; ``sisteer.capeval.tiers.sandbox_owned`` wires it to the real exception types
and to the provider's durable record. The rule it implements:

* only a RATE-LIMIT refusal is retried (``rate_limited(exc)``; anything else
  re-raises at once — an image error or a permission error is not a wait);
* the wait before attempt ``n`` (0-based) is ``min(cap_s, base_s * 2**n)`` plus a
  uniform jitter in ``[0, jitter_s)`` — 1, 2, 4, 8, 16, 30, 30, … s by default;
* the retries stop when the NEXT wait would carry the total past ``total_s``
  (5 min by default): the last refusal is re-raised AS IS, so a truly stuck limit
  still fails loudly with Modal's own message, never a silent fallback to another
  provider or a half-scored task;
* every retry is reported (``on_retry``: the attempt, the wait, the elapsed time,
  the error) so the provider can log it and append a durable event.

Configuration: the four numbers come from :func:`policy_from_env` — the
``TINYCAP_SANDBOX_CREATE_BACKOFF_{BASE,CAP,TOTAL,JITTER}_S`` environment (the
harness's driver env is the natural channel into a verifier container), else the
defaults above.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional

__all__ = ["BackoffPolicy", "DEFAULT_POLICY", "ENV_BASE_S", "ENV_CAP_S", "ENV_TOTAL_S", "ENV_JITTER_S",
           "RATE_LIMIT_TYPE_NAMES", "GRPC_RESOURCE_EXHAUSTED", "policy_from_env", "is_rate_limited",
           "wait_before_attempt", "create_with_backoff", "BackoffExhausted"]

ENV_BASE_S = "TINYCAP_SANDBOX_CREATE_BACKOFF_BASE_S"
ENV_CAP_S = "TINYCAP_SANDBOX_CREATE_BACKOFF_CAP_S"
ENV_TOTAL_S = "TINYCAP_SANDBOX_CREATE_BACKOFF_TOTAL_S"
ENV_JITTER_S = "TINYCAP_SANDBOX_CREATE_BACKOFF_JITTER_S"

#: the exception TYPE NAMES that read as the rate limit when the real class is not importable here
#: (``modal.exception.ResourceExhaustedError`` is both a ``modal.exception.Error`` and a ``grpclib.GRPCError``)
RATE_LIMIT_TYPE_NAMES = ("ResourceExhaustedError",)
#: gRPC's ``RESOURCE_EXHAUSTED`` status — its name and its numeric code (``grpclib.const.Status.RESOURCE_EXHAUSTED == 8``)
GRPC_RESOURCE_EXHAUSTED = ("RESOURCE_EXHAUSTED", 8)


@dataclass(frozen=True)
class BackoffPolicy:
    """The four numbers of the rule (seconds). ``total_s`` bounds the SUM of the waits, not the attempts."""
    base_s: float = 1.0
    cap_s: float = 30.0
    total_s: float = 300.0
    jitter_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("base_s", "cap_s", "total_s", "jitter_s"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v < 0 or v != v:
                raise ValueError(f"sandbox create backoff: {name} must be a non-negative number, got {v!r}")
        if self.base_s <= 0:
            raise ValueError(f"sandbox create backoff: base_s must be > 0, got {self.base_s!r}")
        if self.cap_s < self.base_s:
            raise ValueError(f"sandbox create backoff: cap_s {self.cap_s!r} < base_s {self.base_s!r}")


DEFAULT_POLICY = BackoffPolicy()


def policy_from_env(environ: Optional[Mapping[str, str]] = None) -> BackoffPolicy:
    """The policy the environment names, the defaults for what it leaves unset; a non-numeric value refuses by name."""
    env = os.environ if environ is None else environ

    def _num(var: str, default: float) -> float:
        raw = env.get(var)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"sandbox create backoff: {var}={raw!r} is not a number") from None

    return BackoffPolicy(base_s=_num(ENV_BASE_S, DEFAULT_POLICY.base_s), cap_s=_num(ENV_CAP_S, DEFAULT_POLICY.cap_s),
                         total_s=_num(ENV_TOTAL_S, DEFAULT_POLICY.total_s), jitter_s=_num(ENV_JITTER_S, DEFAULT_POLICY.jitter_s))


def is_rate_limited(exc: BaseException) -> bool:
    """Duck-typed: the exception (or any class in its MRO) is named like Modal's ``ResourceExhaustedError``, or it
    carries a gRPC ``status`` that is ``RESOURCE_EXHAUSTED`` (by name or code). The provider passes a predicate built
    on the real types instead; this one is the modal-free default and the test seam."""
    for klass in type(exc).__mro__:
        if klass.__name__ in RATE_LIMIT_TYPE_NAMES:
            return True
    status = getattr(exc, "status", None)
    if status is None:
        return False
    if getattr(status, "name", None) in GRPC_RESOURCE_EXHAUSTED or getattr(status, "value", None) in GRPC_RESOURCE_EXHAUSTED:
        return True
    return status in GRPC_RESOURCE_EXHAUSTED


def wait_before_attempt(policy: BackoffPolicy, attempt: int, rng: Optional[random.Random] = None) -> float:
    """``min(cap_s, base_s * 2**attempt)`` + uniform jitter in ``[0, jitter_s)``; ``attempt`` is 0-based (the wait
    BEFORE the first retry is attempt 0's)."""
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt!r}")
    base = min(float(policy.cap_s), float(policy.base_s) * (2.0 ** int(attempt)))
    jitter = (rng or random).uniform(0.0, float(policy.jitter_s)) if policy.jitter_s > 0 else 0.0
    return base + jitter


class BackoffExhausted(RuntimeError):
    """Never raised to callers — the LAST refusal is re-raised as itself. Kept as the ``on_exhausted`` marker type."""


async def create_with_backoff(create: Callable[..., Awaitable[Any]], *args: Any, policy: BackoffPolicy = DEFAULT_POLICY,
                              rate_limited: Callable[[BaseException], bool] = is_rate_limited,
                              sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                              rng: Optional[random.Random] = None, clock: Callable[[], float] = time.monotonic,
                              on_retry: Optional[Callable[[int, float, float, BaseException], None]] = None,
                              on_exhausted: Optional[Callable[[int, float, BaseException], None]] = None,
                              **kwargs: Any) -> Any:
    """``await create(*args, **kwargs)``, retried on the rate limit under ``policy``.

    ``on_retry(attempt, wait_s, elapsed_s, exc)`` runs before each sleep; ``on_exhausted(attempts, elapsed_s, exc)``
    runs once when the next wait would pass ``total_s`` — and then the refusal is re-raised unchanged (its type, its
    message: Modal's own ``Sandbox creation rate limit (5/s) exceeded``). A refusal that is not the rate limit
    re-raises at once with no sleep. Returns what ``create`` returns."""
    started = clock()
    attempt = 0
    while True:
        try:
            return await create(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 — classified, then re-raised or retried; never swallowed
            if isinstance(e, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)) or not rate_limited(e):
                raise
            elapsed = clock() - started
            wait = wait_before_attempt(policy, attempt, rng)
            if elapsed + wait > float(policy.total_s):
                if on_exhausted is not None:
                    on_exhausted(attempt, elapsed, e)
                raise
            if on_retry is not None:
                on_retry(attempt, wait, elapsed, e)
            await sleep(wait)
            attempt += 1
