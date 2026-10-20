"""Anthropic client construction by credential label, plus the one shared
same-label backpressure mechanism (AdaptiveLimiter).

The Anthropic credentials go by three labels (``lowprio``, ``highprio`` and
``batch``): names the code asks for, which a site maps to keys of its own.
Judge and live runs default to ``lowprio``. On 429s a run THROTTLES DOWN on
the same label and never switches to another: at judge widths the likeliest
cause of a saturated rate limit is this very run, and moving to another label
only relocates the contention. Do not add a fallback to another label.
:class:`AdaptiveLimiter` below is the throttle every live judge/grader in this
repo shares. (A 529 is provider-side capacity — back off and retry; neither a
label switch nor a width cut helps, so 529s do not shrink the limiter.)

Retry layers must not multiply: callers that run their own patient
retry loop around the limiter should construct their client with a LOW
SDK retry count (``make_client(..., max_retries=1)``) — the SDK's
internal 429 retries are invisible to the limiter and would multiply
in-flight attempts underneath it during a rate-limit event.

Keys come from the vault through :mod:`sisteer.vault` — the label's own
environment variable, else its item in a secret manager read through the
secret-manager CLI. There is no key file and no fall-through between labels.

That last property is the point, and it is a FIX, not a refinement. This
module used to read a ``lowprio`` key file, and when the file was absent it
built a bare SDK client "so CI still works". The SDK then read
ANTHROPIC_API_KEY from the environment — the ``highprio`` label's variable —
so a caller that asked for one label silently ran on another's key, with
nothing in the run saying so. The bare client also inherited
ANTHROPIC_BASE_URL, which in some shells points at a local gateway, so those
requests never reached the API at all. Asking for a label that has no key now
RAISES :class:`sisteer.vault.PoolUnavailable` (a ``VaultSecretError``
carrying ``.pool``); a machine with no credentials gets a loud, named error
instead of another label's key.

Every client this helper returns also carries its POOL STAMP
(``sisteer.vault.client_pool_stamp(client)`` -> ``{"pool", "key_sha256_12"}``,
never the key), so a caller can write which credential label the run actually
used into its run record beside the concurrency it used — the label is
recorded, not inferred from which shell launched the run.
"""

from __future__ import annotations

import threading
import time

from sisteer import vault

#: The credential labels this module builds clients for, named for the vault items behind them
#: (:data:`sisteer.vault.ANTHROPIC_ORGS`). Kept as a module constant because
#: callers and tests enumerate the valid labels.
ORGS = ("lowprio", "highprio")


#: ``make_client``'s default per-request bound: the vault's
#: :data:`sisteer.vault.REQUEST_TIMEOUT_S` (120 s) under the name a sibling
#: branch gave this same default, so the two signatures are textually
#: identical when the branches meet. One number, one source of truth — never
#: a second literal here. WHY it is always set: the SDK's default is 600 s
#: against a judge call that answers in ~1.8 s, a margin that cannot tell a
#: slow call from a dead socket. MEASURED 2026-09-11: two sockets ESTAB with
#: empty send/recv queues stalled a serial 5,953-item judged run and read as
#: pool degradation; a judge that reached the API through THIS helper hung
#: the same way.
DEFAULT_TIMEOUT_S = vault.REQUEST_TIMEOUT_S


def make_client(prio: str = "lowprio", *, max_retries: int = 5,
                timeout: float = DEFAULT_TIMEOUT_S):
    """An ``anthropic.Anthropic`` bound to the named credential label, built
    through :func:`sisteer.vault.anthropic_client`: the label's own key from
    the environment or the vault, the base URL pinned to the real API, and an
    explicit request timeout.

    Defaults to ``lowprio``. Raises
    :class:`sisteer.vault.PoolUnavailable` when that label has no key — it
    never falls back to another label's credential, which is what the old
    file-reading version did silently (module docstring). The client
    carries its pool stamp (:func:`sisteer.vault.client_pool_stamp`).
    """
    if prio not in ORGS:
        raise ValueError(f"prio must be one of {sorted(ORGS)}, got {prio!r}")
    return vault.anthropic_client(prio, max_retries=max_retries,
                                  timeout=timeout)


class AdaptiveLimiter:
    """Bounded SAME-LABEL backpressure for live judge/grader runs.

    A thread-safe gate on in-flight API calls whose width adapts to
    rate-limit pressure — the concrete form of the rule "on 429s, THROTTLE
    DOWN on the same credential label, never switch to another":

    - ``with limiter:`` admits a call while fewer than ``width`` are in
      flight; extra workers block until a slot frees, so a
      ``ThreadPoolExecutor`` wider than the current width simply queues.
    - :meth:`on_rate_limit` (call it from the 429 handler) HALVES the
      width, flooring at ``min_width`` — multiplicative decrease. A
      ``cooldown`` window (default 10s) makes a burst of simultaneous
      429s across N workers coalesce into ONE halving, not N.
    - Recovery is earned under ADMISSION GENERATIONS: every real shrink
      increments a generation, and each admission records the generation
      it entered under. Only clean completions that were BOTH admitted
      under the current post-shrink generation AND finished outside the
      cooldown window count toward recovery — so the tail of the
      pre-shrink wave (calls already in flight when the 429s hit, which
      prove nothing about the reduced width) can never undo the shrink.
    - Recovery is slow and additive: one full successful window at the
      current width (``max(width, recover_floor)`` qualifying
      completions) earns +1, capped at the launch width. A recurring
      event re-halves and resets the streak.

    529s / 5xx / connection errors should NOT shrink the limiter — they
    are provider-side capacity, not this label's rate limit; just sleep and
    retry. Deterministic 4xx must not be retried at all.

    ``n_rate_limited`` and ``width`` are exposed for run stats.
    """

    def __init__(self, width: int, *, min_width: int = 1,
                 cooldown: float = 10.0, recover_floor: int = 16):
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self._launch_width = width
        self._min = max(1, min(min_width, width))
        self._width = width
        self._cooldown = cooldown
        self._recover_floor = max(1, recover_floor)
        self._streak = 0
        self._active = 0
        self._generation = 0            # bumped on every REAL shrink
        self._last_shrink = float("-inf")
        self._cond = threading.Condition()
        self._local = threading.local()
        self.n_rate_limited = 0

    @property
    def width(self) -> int:
        with self._cond:
            return self._width

    def _admissions(self) -> list:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    def __enter__(self) -> AdaptiveLimiter:
        with self._cond:
            while self._active >= self._width:
                self._cond.wait()
            self._active += 1
            # record which generation admitted this call (per-thread LIFO:
            # a thread's with-blocks nest, so pop pairs with push)
            self._admissions().append(self._generation)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        st = self._admissions()
        admitted_gen = st.pop() if st else -1
        with self._cond:
            self._active -= 1
            if exc_type is None:
                self._maybe_recover(admitted_gen)
            self._cond.notify_all()
        return False

    def _maybe_recover(self, admitted_gen: int) -> None:
        """Called under the lock on a clean completion. Earns width back
        only from evidence gathered AT the reduced width: same-generation
        admissions, outside the cooldown, one full window per +1."""
        if self._width >= self._launch_width:
            self._streak = 0
            return
        if admitted_gen != self._generation:
            return          # pre-shrink wave: proves nothing, earns nothing
        if time.monotonic() - self._last_shrink < self._cooldown:
            return          # no recovery during cooldown
        self._streak += 1
        if self._streak >= max(self._width, self._recover_floor):
            self._streak = 0
            self._width += 1            # slow additive recovery
            print(f"[api-limiter] recovered: width back up to "
                  f"{self._width}", flush=True)

    def shrink(self, *, floor: Optional[int] = None, reason: str = "sustained rate limiting") -> int:
        """Halve the width IN PLACE regardless of the cooldown (the misalign judge's Throttler on a SUSTAINED 429 episode, g8-fix):
        the same generation bump and shrink stamp as :meth:`on_rate_limit`, so recovery is earned under the new generation, slow and
        additive, and never above the ORIGINAL launch width -- replacing the limiter with a fresh one set that ceiling to the reduced
        width for good and left queued workers on the old gate. ``floor`` raises the minimum for this shrink (never below the
        constructor's). Returns the new width."""
        with self._cond:
            lo = max(self._min, int(floor)) if floor is not None else self._min
            new = max(lo, self._width // 2)
            if new < self._width:
                self._width = new
                self._generation += 1
                self._last_shrink = time.monotonic()
                self._streak = 0
                print(f"[api-limiter] {reason}: throttling DOWN to {new} concurrent on the SAME org "
                      f"(launch width {self._launch_width} stays the recovery ceiling)", flush=True)
            self._cond.notify_all()
            return self._width

    @property
    def launch_width(self) -> int:
        return self._launch_width

    def on_rate_limit(self) -> None:
        """One 429 observed: halve the width (same credential label — never
        a switch to another), at most once per ``cooldown`` window. Every event resets
        the recovery streak, whether or not it shrinks."""
        with self._cond:
            self.n_rate_limited += 1
            self._streak = 0
            now = time.monotonic()
            if now - self._last_shrink < self._cooldown:
                return  # this burst already shrank the width once
            new = max(self._min, self._width // 2)
            if new < self._width:
                self._width = new
                self._generation += 1
                self._last_shrink = now
                print(f"[api-limiter] 429: throttling DOWN to {new} "
                      "concurrent on the SAME org (the 2026-08-21 rule: "
                      "never hop to another org's pool)", flush=True)
