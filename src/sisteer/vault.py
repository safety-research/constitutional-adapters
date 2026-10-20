"""THE ONE PLACE A CREDENTIAL ENTERS THIS REPOSITORY.

Every API key this repo uses lives in a secret manager's vault and reaches a
process one of two ways: an environment variable the shell's vault bootstrap
already exported, or a bounded subprocess of the site's secret-manager CLI
(``$SISTEER_SECRET_CMD``) against the named vault item. There is no third way. **No key is ever read from a file**, cached to
disk, written to a log, or passed on a command line, and a key that is in
neither place is a typed :class:`VaultSecretError` rather than a silent empty
string or a silent substitute. (Under ``SISTEER_SECRETS_MODE=wif`` there is no
key at all — the clients authenticate through workload identity federation,
the section at the end of this module — and asking for one refuses by name.)

WHY A SHARED MODULE, AND WHY THE CREDENTIAL LABELS' ENV VARS ARE NOT INTERCHANGEABLE.
The Anthropic credentials go by three labels, ``lowprio``, ``highprio`` and ``batch``: names the code asks for, which a
site maps to keys of its own (or all three to one key under ``env`` mode). Before this module each hub read its own
key file and invented its own behaviour for a missing one, and two of those behaviours were wrong in opposite
directions:

* ``eval/api_keys.make_client("lowprio")`` found no key file and fell back to a bare SDK client, which picked up
  ``ANTHROPIC_API_KEY`` from the environment, the ``highprio`` label's variable. A caller that asked for one label
  silently ran on another's key, and nothing in the run said so. The bare client also inherited
  ``ANTHROPIC_BASE_URL``, which in a shell that sets it points at a local gateway, so the requests did not even reach
  the API.
* ``capeval/judging.ClaudeJudge`` found no key file and raised. Loud, and therefore the better of the two, but it
  meant the anthropic judge backend could not run at all.

So the rule this module encodes is that **each credential label resolves its own key**. ``ANTHROPIC_API_KEY`` is the
``highprio`` label's variable and nothing else; asking for ``lowprio`` consults ``ANTHROPIC_API_KEY_LOWPRIO`` and the
``lowprio`` vault item, and can never resolve to another label's key by omission. A caller that wants ``highprio``
must name it. There is no variable whose absence degrades into another label.

Callers that build an Anthropic client must also PIN the base URL
(:data:`ANTHROPIC_BASE_URL` here) rather than let the SDK read
``ANTHROPIC_BASE_URL`` from the environment, for the gateway reason above.
:func:`anthropic_client` does both and is what every hub should use.

THE LABEL IS RECORDED, NOT JUST RESOLVED (2026-09-14). Resolving the right key
is half the fix; the other half is that a run says which credential label (its
"pool") it used, so the accounting can be checked after the fact instead of
inferred from which shell launched it. Every client this module builds therefore carries a
POOL STAMP — ``{"pool": <name>, "key_sha256_12": <first 12 hex of the key's
sha256>}`` — as the attribute :data:`POOL_STAMP_ATTR`; :func:`client_pool_stamp`
reads it back and REFUSES a client that has none (a client built anywhere
else has no recorded pool, and "unknown" is not a record). The stamp never
carries the key: twelve hex characters of its digest identify which pool a
run used without being usable as a credential. Callers write the stamp into
their run record beside the concurrency they used.

A missing pool is :class:`PoolUnavailable` (a :class:`VaultSecretError`
that also carries ``.pool``): named, typed, and never a fallback.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import email.utils
import hashlib
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

#: Bound on one secret-manager CLI call. The CLI is a local subprocess against
#: a service-account token; anything slower than this is a broken session, not
#: a slow one, and a judge run must not hang on it.
SECRET_CMD_TIMEOUT_S = 30.0

#: The site setting that names op mode's secret-manager CLI: one executable (a path, or a name on PATH) that takes a
#: vault item's name as its only argument and prints that item's value on stdout. There is no default command, so op
#: mode with the setting unset refuses by name (:func:`secret_cmd`).
SECRET_CMD_ENV = "SISTEER_SECRET_CMD"

#: The real API. Pinned by :func:`anthropic_client` so a shell's
#: ``ANTHROPIC_BASE_URL`` (some shells set it to a local gateway) cannot
#: silently redirect a judge or grader run.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

#: OpenRouter's Anthropic-compatible skin.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"

#: An explicit per-request timeout on every client this module builds, because
#: the SDK default is 600s and that is a trap at width. Measured 2026-08-28: a
#: 500-thread grade wedged with every worker blocked in a network read, 100
#: sockets open, ZERO progress for 40+ minutes, and no 429 and no exception to
#: diagnose from. 120s is far above a healthy judge call (~1-3s) and turns a
#: wedge into a retry.
REQUEST_TIMEOUT_S = 120.0

#: credential label -> (env var, vault item). The env var is label-specific ON PURPOSE:
#: see the module docstring. ``ANTHROPIC_API_KEY`` names the ``highprio`` label only.
ANTHROPIC_ORGS: Dict[str, Tuple[str, str]] = {
    "highprio": ("ANTHROPIC_API_KEY", "highprio"),
    "lowprio": ("ANTHROPIC_API_KEY_LOWPRIO", "lowprio"),
    "batch": ("ANTHROPIC_API_KEY_BATCH", "batch"),
}

#: The OpenRouter key's env var and vault item.
OPENROUTER_ENV = "OPENROUTER_API_KEY"
OPENROUTER_ITEM = "OpenRouter API Key"

#: Attribute under which every client this module builds carries its pool
#: stamp (see :func:`pool_stamp`). Read it through :func:`client_pool_stamp`.
POOL_STAMP_ATTR = "sisteer_pool_stamp"

#: THREE SECRETS MODES, never a fourth (docs/secrets.md; the same switch the vendored corpus client and
#: reproduce/lib/secrets.sh read). ``op`` (the default): everything above — each credential label its OWN env var,
#: else its vault item, never another pool's key. ``env`` (every other user): the ONE documented ``ANTHROPIC_API_KEY``
#: serves EVERY credential label (lowprio / highprio / batch), required by name, never a file, never the secret-manager CLI;
#: ``OPENROUTER_API_KEY`` likewise. The pool stamp then records ``pool = "env"`` + the key's sha12, so a run record
#: says which contract produced it. The "no silent pool substitution" property is an op-mode property: under env mode
#: a single key IS the contract, so there is nothing to substitute. ``wif`` (since 2026-10-11, after the move to
#: workload identity federation on 2026-10-07 left the API keys rejected with 401): there is NO API
#: key — each credential label authenticates through the SDK's ``WorkloadIdentityCredentials`` under the federation ids of its
#: own file (:func:`wif_env_path`; the section WORKLOAD IDENTITY FEDERATION below), :func:`anthropic_api_key` refuses by
#: name, the clients carry a KEYLESS stamp (``{"pool", "auth": "wif", "ids_sha12"}``), and nothing reads, sets or
#: exports ``ANTHROPIC_*``. op and env are byte-identical to what they were: the wif branch runs only when the switch
#: names it.
SECRETS_MODE_ENV = "SISTEER_SECRETS_MODE"
ENV_MODE_KEY_VAR = "ANTHROPIC_API_KEY"
ENV_MODE_POOL = "env"
WIF_MODE = "wif"
SECRETS_MODES = ("op", "env", WIF_MODE)


def secrets_mode() -> str:
    """``op`` (default), ``env`` or ``wif``; anything else is a typed refusal, never a guess."""
    mode = (os.environ.get(SECRETS_MODE_ENV) or "op").strip() or "op"
    if mode not in SECRETS_MODES:
        raise VaultSecretError(f"{SECRETS_MODE_ENV}={mode!r}: must be one of 'op', 'env' or 'wif'")
    return mode


def secret_cmd(env: Optional[Mapping[str, str]] = None) -> str:
    """The secret-manager CLI op mode runs (``$SISTEER_SECRET_CMD``, read from ``env`` when given, else this process's
    environment). Unset or blank is a typed refusal naming the setting, never a guessed command."""
    cmd = ((os.environ if env is None else env).get(SECRET_CMD_ENV) or "").strip()
    if not cmd:
        raise VaultSecretError(f"{SECRET_CMD_ENV} is not set: op mode reads each secret through the site's secret-manager "
                               f"CLI, which this setting names (config.env); set it, or set {SECRETS_MODE_ENV}=env and "
                               "export the documented variables")
    return cmd


def env_mode_key(org: str) -> str:
    """The single documented key under env mode, for EVERY org role: required by name, never read from a file,
    never fetched from the vault."""
    key = (os.environ.get(ENV_MODE_KEY_VAR) or "").strip()
    if not key:
        raise PoolUnavailable(org, f"{SECRETS_MODE_ENV}=env: {ENV_MODE_KEY_VAR} must be set (one key serves every org "
                                   f"role; requested org={org!r}); never a file, never the secret-manager CLI")
    return key


class VaultSecretError(RuntimeError):
    """A credential was in neither the environment nor the vault.

    The message always names the env var and the vault item that were tried and
    why the vault read failed; it never carries the value, and the underlying
    subprocess exception is deliberately not chained so that no traceback can
    surface captured output.
    """


class PoolUnavailable(VaultSecretError):
    """A named Anthropic pool (``lowprio`` / ``highprio`` / ``batch``) has no
    key in its OWN env var and its vault item could not be read.

    Raised by :func:`anthropic_api_key` and everything built on it. It is the
    typed, named refusal the routing rule requires: a request for one pool is
    never satisfied with another pool's key, and there is no warning-and-
    continue path. ``pool`` names the pool that was asked for.
    """

    def __init__(self, pool: str, message: str):
        super().__init__(message)
        self.pool = pool


class BaseUrlRefused(VaultSecretError):
    """A federated client was asked for a DESTINATION other than the pinned API host. Two faces, one error: under
    ``SISTEER_SECRETS_MODE=wif`` an OVERRIDE from OUTSIDE the process (an environment variable, a config value;
    :func:`pinned_base_url`) naming another host, and an in-process ``base_url=`` argument or an ``http_client`` whose own
    base URL names one, at the federated constructors themselves (:func:`pin_federated_destination`:
    :func:`anthropic_client_wif`, so :func:`anthropic_client` / :func:`anthropic_async_client` under wif, and the inspect
    provider patch's ``install_inspect_wif``). The message names the source, the override, the mode and the pinned host; it
    never carries a credential (there is none under wif)."""


# --- CREDENTIAL REJECTION: the API answered 401 / 403 (grading_diagnosis.md 2026-10-11) ---------------------
#: The HTTP statuses that mean the CREDENTIAL was rejected, not the request: 401 ``authentication_error`` (an invalid or
#: revoked API key -- the legacy lowprio key after the org's move to federation on 2026-10-07 -- or a federation identity
#: the API does not accept) and 403 ``permission_error`` (a credential without the right to the resource). A 429 is the
#: pool, a 529 / 5xx the provider, a 400 the request: none of those is this.
CREDENTIAL_REJECTED_STATUSES = (401, 403)


class CredentialRejection(NamedTuple):
    """What :func:`credential_rejection` read off an SDK exception: the HTTP status, the SDK's error type name and the
    API's own error type (``authentication_error`` / ``permission_error``) when the body carried one."""
    status: int
    error_type: str
    api_error_type: Optional[str]


def credential_rejection(exc: BaseException) -> Optional[CredentialRejection]:
    """``CredentialRejection`` when ``exc`` is the Anthropic SDK's credential rejection -- ``anthropic.AuthenticationError``
    (401) / ``anthropic.PermissionDeniedError`` (403), any ``APIStatusError`` whose status is one of
    :data:`CREDENTIAL_REJECTED_STATUSES`, or, under federation, ``anthropic.WorkloadIdentityError`` with such a status --
    else None. Reads the status and the API's error type only (never a key, never a token, never the request). The one
    classifier every judge driver uses, so a 401 means the same thing everywhere.

    THE FEDERATED SHAPE: under ``SISTEER_SECRETS_MODE=wif`` the credential is the
    federation identity, and the API refuses THAT at the token exchange (``POST /v1/oauth/token``: the rule, the service
    account or the workspace no longer matches), which the SDK raises from the request that needed the bearer as
    ``WorkloadIdentityError`` carrying ``status_code`` (401 / 403) and a REDACTED body -- an ``AnthropicError``, not an
    ``APIStatusError``. Without this clause a revoked federation identity slipped past every credential-rejection stop
    (the controls judge, the misalign preflight, the live graders, judge_sonnet, the lane finisher's probe) and the
    drivers retried a rejected credential exactly as they did the legacy key. The exchange's 5xx and a transport
    failure (no status) are not this.

    A rejection ALREADY TYPED (a :class:`JudgeCredentialRejected` a latch below the caller raised -- the federated clients' own
    retry layer, :class:`LatchedRetryMessages`, raises the held typed error from its per-attempt check; round 12 on fd06740ae)
    classifies as what it records: its status, the SDK error type it was typed from and the API's error type, so a driver's
    ``except Exception`` ladder that classifies the credential FIRST reads it as the rejection it is, never as "any other
    error"."""
    if isinstance(exc, JudgeCredentialRejected):
        return CredentialRejection(exc.status, exc.error_type, exc.api_error_type)
    status = getattr(exc, "status_code", None)
    try:
        import anthropic
        typed = isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError))
        by_status = isinstance(exc, anthropic.APIStatusError) and status in CREDENTIAL_REJECTED_STATUSES
        federated = (isinstance(exc, getattr(anthropic, "WorkloadIdentityError", ()))      # an SDK without federation: never
                     and status in CREDENTIAL_REJECTED_STATUSES)
    except ImportError:  # pragma: no cover - the SDK is a dependency of every caller; the status alone still classifies
        typed, by_status, federated = False, status in CREDENTIAL_REJECTED_STATUSES, False
    if not (typed or by_status or federated):
        return None
    api_type = None
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        err = body.get("error")
        if isinstance(err, Mapping) and isinstance(err.get("type"), str):
            api_type = err["type"]
    try:
        status_int = int(status) if status is not None else (401 if "Authentication" in type(exc).__name__ else 403)
    except (TypeError, ValueError):
        status_int = 401
    return CredentialRejection(status_int, type(exc).__name__, api_type)


class JudgeCredentialRejected(Exception):
    """The API REJECTED the credential the vault resolved for ``pool`` -- HTTP 401 ``authentication_error`` or 403
    ``permission_error`` (:func:`credential_rejection`) -- raised by every judge driver on the FIRST such answer and never
    retried (grading_diagnosis.md, 2026-10-11): a driver stops on the first 401. Auth is per client: every later request
    of the same client fails the same way until the credential is fixed, so a driver that keeps issuing them hides the one
    cause under thousands of per-item errors (the demo smoke: 14 x 6 x 777 static + 6 x 7,500 BoN requests, 10,000
    agentic, then 0-verdict judgments filed). Raised by
    ``grade_bon_live._forced_tool_grade`` (so ``LiveStrongREJECTGrader`` stops and cancels its queue), the misalign judge's
    ``PoolClient`` (its preflight before any claim, and its judge loop) and ``capeval.judging.ClaudeJudge`` (a
    ``JudgingError`` subclass there, latched on the shared client).

    Deliberately NOT a RuntimeError, and its text never says UNSCORED: the JB1 re-issue loops (``except RuntimeError`` +
    "UNSCORED" in the message) therefore never catch it and never request the same doomed items six more times. The
    message names the pool, the status, the SDK's error type and the API's own error type -- never a key, never a token,
    never prompt or response text -- and the grading daemon classifies a tail carrying those words as its `credential`
    kind (``grade_daemon.credential_rejected``: AuthenticationError / PermissionDeniedError / authentication_error /
    Error code: 401): the family held, ONE problem, nothing quarantined, nothing retried."""

    def __init__(self, pool: str, status: int, error_type: str, *, api_error_type: Optional[str] = None,
                 message: Optional[str] = None):
        self.pool = str(pool)
        self.status = int(status)
        self.error_type = str(error_type)
        self.api_error_type = api_error_type
        mode = (os.environ.get(SECRETS_MODE_ENV) or "op").strip() or "op"
        fix = ("the federation ids file / the VM identity" if mode == WIF_MODE else
               f"{ENV_MODE_KEY_VAR}" if mode == "env" else "the pool's vault item")
        super().__init__(message or (
            f"the API rejected the {self.pool} pool's credential: {self.error_type} (HTTP {self.status}"
            f"{', ' + api_error_type if api_error_type else ''}) under {SECRETS_MODE_ENV}={mode} -- stopping at the first "
            f"credential rejection (D6): no retry, no further request on this credential; fix {fix} and re-run"))


# --- THE SHARED STOP (the WIF driver sweep, 2026-10-13) --------------------------------------------------------------
# Three consecutive review rounds each found another driver that met a 401 / 403 and went on issuing: judge_sonnet's
# queue, the misalign judge's, the corpus generator's live engine, then judge_overrefusal's `pool.map` (the executor
# drains the whole queue after the first rejection) and build_probe_set's `_call_api` (a 401 retried like a 5xx, then
# recorded per item, then a verify report written over it). The rule is the same everywhere -- classify EVERY exception
# for the credential FIRST, latch the first rejection for the run, raise it before every later wire call, cancel the
# queued work and wait only for the futures already running, write no partial artifact, exit with the typed message --
# so it lives here ONCE: a latch, a queue cancel, a bounded map and a typed exit. A driver that reaches Claude from more
# than one worker builds its stop from these rather than re-deriving it (tests/test_wif_driver_inventory.py names every
# request site and the mechanism that stops it).


def is_credential_rejection(exc: BaseException) -> bool:
    """Whether ``exc`` is the API's credential rejection (:func:`credential_rejection`): a 401 / 403 on the request, or
    the federation identity refused at the token exchange. Never a 429 / 5xx / connection error."""
    return credential_rejection(exc) is not None


def pool_of_client(client: Any) -> str:
    """The pool a client's vault stamp names (wrappers that forward attributes are transparent), else ``unknown`` -- a
    label for a message, never a routing decision."""
    try:
        return str(client_pool_stamp(client)["pool"])
    except Exception:  # noqa: BLE001 -- a stand-in client carries no stamp
        return "unknown"


class CredentialLatch:
    """The run-level latch of a driver that issues Claude requests from several workers.

    ``check()`` BEFORE every wire call raises the held rejection, so a worker that reaches its request after the first
    401 / 403 of the run issues nothing. ``rejection(exc)`` AFTER a failed call classifies ``exc`` (:func:`credential_rejection`,
    the federated shape included) and, for a rejection, holds the FIRST one filed as a typed :class:`JudgeCredentialRejected`
    (``cls`` for a driver whose callers catch their own error family, e.g. capeval.judging's JudgingError subclass) under a
    check-and-set -- two workers meeting a 401 in the same instant hold ONE object (round 3 on 545c56eda) -- and returns
    it; None for every other error, which stays the caller's retry loop's or item's business. ``pool`` is the pool's
    name or a callable that resolves it when the first rejection is filed (a client's stamp: :func:`pool_of_client`)."""

    def __init__(self, pool: Any = "unknown", *, cls: Any = None):
        self._pool = pool
        self._cls = cls or JudgeCredentialRejected
        self._held: Optional[JudgeCredentialRejected] = None
        self._lock = threading.Lock()

    @property
    def held(self) -> Optional[JudgeCredentialRejected]:
        """The latched first rejection of this run, else None."""
        return self._held

    def check(self) -> None:
        held = self._held
        if held is not None:
            raise held

    def rejection(self, exc: BaseException, *, pool: Any = None) -> Optional[JudgeCredentialRejected]:
        rej = credential_rejection(exc)
        if rej is None:
            return None
        with self._lock:
            if self._held is None:
                if isinstance(exc, self._cls):
                    self._held = exc                  # an already-typed rejection of this latch's own family (a layer below): held as is
                else:
                    p = pool if pool is not None else self._pool
                    if callable(p):
                        try:
                            p = p()
                        except Exception:  # noqa: BLE001 -- a message label never refuses
                            p = "unknown"
                    if isinstance(exc, JudgeCredentialRejected):
                        p = exc.pool              # typed by another family's latch (the client's, below a driver's own): its pool, re-typed as ours
                    self._held = self._cls(str(p), rej.status, rej.error_type, api_error_type=rej.api_error_type)
            return self._held

    def hold(self, rejected: JudgeCredentialRejected) -> JudgeCredentialRejected:
        """Hold an ALREADY-TYPED rejection a caller met -- a wrapper layer's (``sisteer.throttle.ThrottledClient``), a
        canary's (the AlphaSteer driver's judge-shedding probe) -- under the same check-and-set as :meth:`rejection`: the
        first held wins and is returned, so every later :meth:`check` raises that one object."""
        with self._lock:
            if self._held is None:
                self._held = rejected
            return self._held

    def guard(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` with the stop around it: the latch checked before the call, every exception classified after it (a
        rejection raised as the held typed error, anything else re-raised as is)."""
        def guarded(*args: Any, **kwargs: Any) -> Any:
            self.check()
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 -- classified, the credential first
                held = self.rejection(e)
                if held is not None:
                    raise held from e
                raise
        return guarded


#: the attribute under which a CLIENT carries the run's :class:`CredentialLatch`. The one-shot
#: graders' retry layers -- the throttled wrapper's 429 loop (``sisteer.throttle``), ``grade_bon_live._forced_tool_grade``'s
#: bounded loop, the BoN twin's ``BoundedJudgeClient`` -- each SLEEP between attempts, and before round 11 the latch was checked
#: once per item: a worker asleep in a backoff when a sibling's request met the run's first 401 / 403 woke and issued another
#: request on the rejected credential (as many as the grader's concurrency, per layer). The latch now rides the CLIENT every
#: layer is handed, and each layer checks it BEFORE EVERY attempt -- the first and each wake. A wrapper that forwards public
#: attributes (``ThrottledClient``, ``BoundedJudgeClient``) exposes the latch of the client it wraps, so a wrapper swap over the
#: same inner client keeps the same latch. (``capeval.judging`` holds its OWN client attribute, the held error itself, for the
#: ClaudeJudge family; no client is shared between a ClaudeJudge and a one-shot grader.)
CLIENT_LATCH_ATTR = "sisteer_credential_latch"


def client_credential_latch(client: Any, *, create: bool = False) -> Optional[CredentialLatch]:
    """The :class:`CredentialLatch` ``client`` carries (:data:`CLIENT_LATCH_ATTR`; a forwarding wrapper is transparent), else
    None -- or, with ``create``, a new one ATTACHED to it (its pool resolved from the client's stamp when the first rejection is
    filed). A client that refuses attributes (a slotted stand-in) gets a latch the caller alone holds: returned, not attached."""
    latch = getattr(client, CLIENT_LATCH_ATTR, None)
    if isinstance(latch, CredentialLatch):
        return latch
    if not create:
        return None
    latch = CredentialLatch(lambda: pool_of_client(client))
    attach_credential_latch(client, latch)
    return latch


def attach_credential_latch(client: Any, latch: CredentialLatch) -> bool:
    """Attach ``latch`` to ``client`` under :data:`CLIENT_LATCH_ATTR` when it carries none yet; True when attached (False: the
    client already carries one, or refuses attributes)."""
    if isinstance(getattr(client, CLIENT_LATCH_ATTR, None), CredentialLatch):
        return False
    try:
        setattr(client, CLIENT_LATCH_ATTR, latch)
    except Exception:  # noqa: BLE001 -- a client that refuses attributes: the caller holds the latch alone
        return False
    return True


def cancel_queue(executor: concurrent.futures.Executor, futures: Iterable[concurrent.futures.Future]) -> Dict[str, int]:
    """After the first credential rejection: cancel every future still QUEUED (never run, never a wire call), wait for the
    ones already running -- at most the executor's width; each returns its result or raises the latch before its call --
    and report the counts. Never waits on a future cancelled off the queue (round 1 on 7778694d0: such a future never
    runs, so a waiter on it never wakes). The executor accepts nothing after this."""
    futures = list(futures)
    executor.shutdown(wait=False, cancel_futures=True)
    live = [f for f in futures if not f.cancelled()]
    concurrent.futures.wait(live)
    n_ok = sum(1 for f in live if f.exception() is None)
    return {"n_requested": len(futures), "n_cancelled": len(futures) - len(live), "n_settled": len(live),
            "n_ok": n_ok, "n_failed": len(live) - n_ok}


def map_until_credential_rejected(fn: Callable[[Any], Any], items: Sequence[Any], *, workers: int, latch: CredentialLatch,
                                  label: str = "", on_result: Optional[Callable[[Any, Any], None]] = None,
                                  say: Callable[[str], None] = lambda s: print(s, flush=True)) -> List[Any]:
    """``[fn(item) for item in items]`` on a bounded executor of ``workers`` threads, STOPPING at the first credential
    rejection: ``fn`` runs under ``latch.guard`` (checked before each call, every failure classified), the first
    rejection cancels the queue (:func:`cancel_queue`), is said once with counts -- never an item's text -- and is raised
    as the typed error; any OTHER exception out of an item cancels the queue too and propagates as is (an executor left
    with a queue drains it at exit, one wire call per queued item). ``on_result(item, result)`` runs as each item
    completes (a flush-as-you-go cache); the results come back in ``items``' order."""
    items = list(items)
    results: Dict[int, Any] = {}
    guarded = latch.guard(fn)
    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(guarded, it): i for i, it in enumerate(items)}
        try:
            for f in concurrent.futures.as_completed(futs):
                i = futs[f]
                results[i] = f.result()
                n_done += 1
                if on_result is not None:
                    on_result(items[i], results[i])
        except JudgeCredentialRejected as e:
            counts = cancel_queue(ex, futs)
            say(f"{label + ': ' if label else ''}CREDENTIAL REJECTED after {n_done}/{len(items)} result(s): {e} -- "
                f"{counts['n_cancelled']} queued request(s) cancelled and {counts['n_settled']} other(s) settled ({counts['n_ok']} with "
                f"a result, the rest raising the latch before their call); nothing else is requested (D6)")
            raise
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    return [results[i] for i in range(len(items))]


@contextlib.contextmanager
def credential_exit(pool: Any = "unknown", *, prefix: str = "") -> Iterator[None]:
    """For a driver's ``main``: a credential rejection that reaches it -- the typed :class:`JudgeCredentialRejected` a
    latch raised, or the SDK's own exception out of a single synchronous call (a batch create, a poll, a probe) -- ends
    the process as ``SystemExit`` whose message is the typed text (the terminal line the grading daemon classifies as
    its `credential` kind), never a traceback and never a report written over it. Every other exception passes."""
    try:
        yield
    except JudgeCredentialRejected as e:
        raise SystemExit(f"{prefix}{e}")
    except Exception as e:  # noqa: BLE001 -- classified: the credential only
        rej = credential_rejection(e)
        if rej is None:
            raise
        p = pool() if callable(pool) else pool
        raise SystemExit(f"{prefix}{JudgeCredentialRejected(str(p), rej.status, rej.error_type, api_error_type=rej.api_error_type)}")


def resolve_secret(*, env: str, item: str,
                   timeout: float = SECRET_CMD_TIMEOUT_S) -> str:
    """The secret behind ``env``/``item``: the environment variable when set
    and non-empty, else the vault item read through the secret-manager CLI
    that ``$SISTEER_SECRET_CMD`` names, as ``<command> <item>``.

    Raises :class:`VaultSecretError` when neither source yields a value: the
    setting unset, the command not found or not executable, exiting non-zero
    (item missing, vault session expired), not answering within ``timeout``
    seconds, or answering empty. Never reads a file; never logs the value, and
    the messages name the setting rather than the command it holds.
    """
    from_env = (os.environ.get(env) or "").strip()
    if from_env:
        return from_env
    prefix = (f"{env} is unset and the vault item {item!r} could not be "
              "read: ")
    try:
        cmd = secret_cmd()
    except VaultSecretError as exc:
        raise VaultSecretError(prefix + str(exc)) from None
    try:
        out = subprocess.check_output(
            [cmd, item], text=True, stderr=subprocess.PIPE,
            timeout=timeout)
    except (FileNotFoundError, PermissionError):
        raise VaultSecretError(
            prefix + f"the secret-manager CLI that {SECRET_CMD_ENV} names is not an executable "
            "(not found, or not on PATH)") from None
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or "").strip().splitlines()
        first = err[0][:200] if err else "no stderr"
        raise VaultSecretError(
            prefix + f"the secret-manager CLI ({SECRET_CMD_ENV}) exited {exc.returncode} ({first})") from None
    except subprocess.TimeoutExpired:
        raise VaultSecretError(
            prefix + f"the secret-manager CLI ({SECRET_CMD_ENV}) did not answer within {timeout:g}s") from None
    value = (out or "").strip()
    if not value:
        raise VaultSecretError(prefix + f"the secret-manager CLI ({SECRET_CMD_ENV}) returned an empty value")
    return value


def anthropic_api_key(org: str = "lowprio", *,
                      timeout: float = SECRET_CMD_TIMEOUT_S) -> str:
    """The named Anthropic org's key. ``org`` is one of
    :data:`ANTHROPIC_ORGS`; an unknown org is a ``ValueError``, never a guess.

    Each credential label reads its OWN env var, so a missing ``lowprio`` key raises instead of
    quietly resolving to the ``highprio`` key in ``ANTHROPIC_API_KEY`` — the
    misroute described in the module docstring.
    """
    try:
        env, item = ANTHROPIC_ORGS[org]
    except KeyError:
        raise ValueError(
            f"org must be one of {sorted(ANTHROPIC_ORGS)}, got {org!r}") from None
    mode = secrets_mode()
    if mode == WIF_MODE:
        # by name, never a substitute: under federation there is no key to hand out, and a caller that wanted one
        # would otherwise reach for the shell's ANTHROPIC_API_KEY — the very read the mode exists to end
        raise PoolUnavailable(org, f"{SECRETS_MODE_ENV}={WIF_MODE}: pool {org!r} has NO API key — it authenticates through "
                                   f"workload identity federation (the ids of {wif_env_path(org)}); build the client with "
                                   "anthropic_client(org) / anthropic_async_client(org), whose pool stamp is keyless "
                                   "({'pool', 'auth': 'wif', 'ids_sha12'}); nothing reads ANTHROPIC_API_KEY*, never the secret-manager CLI")
    if mode == "env":
        return env_mode_key(org)
    try:
        return resolve_secret(env=env, item=item, timeout=timeout)
    except VaultSecretError as exc:
        raise PoolUnavailable(org, f"pool {org!r} unavailable: {exc}") from None


def pool_stamp(pool: str, key: str) -> Dict[str, str]:
    """The record of which pool a client uses: ``{"pool": pool,
    "key_sha256_12": <first 12 hex chars of sha256(key)>}``.

    Never the key. Twelve hex characters (48 bits) of the digest are enough
    to tell the three pools' keys apart in a run record and useless as a
    credential; the value is safe to write into any sidecar or log line.
    """
    if not key:
        raise ValueError("pool_stamp needs a non-empty key to digest")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return {"pool": pool, "key_sha256_12": digest}


def _stamp(client: Any, pool: str, key: str) -> Any:
    setattr(client, POOL_STAMP_ATTR, pool_stamp(ENV_MODE_POOL if secrets_mode() == "env" else pool, key))
    return client


def client_pool_stamp(client: Any) -> Dict[str, str]:
    """The pool stamp a vault-built client carries (a fresh copy).

    Raises ``ValueError`` for a client that carries none — one constructed
    outside :func:`anthropic_client` / :func:`anthropic_async_client` /
    :func:`anthropic_client_wif` has no recorded pool, and a run record must
    say which pool it used or fail loudly, never write "unknown". Wrappers
    that forward public attributes (``sisteer.throttle.ThrottledClient``)
    are transparent to this read. Two shapes are records: the keyed one
    (``pool`` + ``key_sha256_12``, op / env) and the KEYLESS one a federated
    client carries (``pool`` + ``auth: "wif"`` + ``ids_sha12``; there is no
    key to digest). :func:`stamp_label` prints either for a log line.
    """
    stamp = getattr(client, POOL_STAMP_ATTR, None)
    if not isinstance(stamp, dict) or not stamp.get("pool") or not (
            stamp.get("key_sha256_12")
            or (stamp.get("auth") == WIF_AUTH and stamp.get("ids_sha12"))):
        raise ValueError(
            f"{type(client).__name__} carries no pool stamp: it was not built "
            "through sisteer.vault.anthropic_client / anthropic_async_client / "
            "anthropic_client_wif (or sisteer.eval.api_keys.make_client), so its "
            "pool is unrecorded")
    return dict(stamp)


def stamp_label(stamp: Mapping[str, Any]) -> str:
    """One log-line phrase for a pool stamp of either shape — ``key sha256/12 <hex>`` for a keyed client,
    ``auth wif ids sha12 <hex>`` for a federated one — so a driver's status line never indexes a key the stamp
    does not carry. Never the key, never an id."""
    if stamp.get("auth") == WIF_AUTH:
        return f"auth {WIF_AUTH} ids sha12 {stamp.get('ids_sha12')}"
    return f"key sha256/12 {stamp.get('key_sha256_12')}"


# --- WORKLOAD IDENTITY FEDERATION (no API key; SISTEER_SECRETS_MODE=wif) -----------------------------------------
#: THE WIF PATH, a vault MODE rather than a per-grader flag. The pool's federation ids live in the workload-identity
#: env file (``~/.anthropic/wif-<stem>.env``) as ``KEY=VALUE`` lines, are read off that FILE and passed EXPLICITLY to
#: the SDK's ``WorkloadIdentityCredentials`` — never placed in ``os.environ`` (an exported ANTHROPIC_* would shadow
#: federation for every child); the client authenticates with a Google identity token minted at call time
#: (``google.oauth2.id_token.fetch_id_token`` for the Anthropic audience: the VM's attached service account) under the
#: GCP federation rule (``ANTHROPIC_FEDERATION_RULE_ID_GCP``). The ROUTING RULE is that an org names its own file,
#: never another's, and a file stem is not always the org name (:data:`WIF_ORG_FILES`); an empty or incomplete file is
#: a typed refusal by name, never a fallback to another file. The pool stamp records ``{"pool": <org>, "auth": "wif",
#: "ids_sha12": sha256(<ids file bytes>)[:12]}`` — the digest of the FILE, so a run record (and the grading daemon's
#: Petri attestation) can say WHICH federation identity graded without carrying a token or an id.
WIF_AUTH = "wif"
WIF_ENV_DIR_ENV = "SISTEER_WIF_ENV_DIR"
WIF_ENV_DIR_DEFAULT = "~/.anthropic"
#: org name -> the file stem under the WIF env dir (``wif-<stem>.env``)
WIF_ORG_FILES: Dict[str, str] = {"lowprio": "loprio", "highprio": "hiprio", "batch": "batch"}
#: the ids a WIF env file must carry (the GCP federation rule; the VM path reads the ``_GCP`` rule id)
WIF_REQUIRED_KEYS = ("ANTHROPIC_FEDERATION_RULE_ID_GCP", "ANTHROPIC_ORGANIZATION_ID",
                     "ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID")
ANTHROPIC_WIF_AUDIENCE = "https://api.anthropic.com"
#: Modal's container marker (the same name the registry's substrate ladder and the Modal rungs read). The GCP identity-token
#: path (:func:`google_identity_token` under the ``_GCP`` rule) cannot work in a Modal container -- its identity is a sandbox
#: token under the ``_MODAL`` rule, a constructor this module does not have -- so under wif on Modal the vault REFUSES BY
#: NAME instead of letting the SDK fail opaquely at the first request; no front-door judge runs on Modal, and
#: scripts/modal/seedrep_petri_drive.py keeps its own path.
MODAL_MARKER_ENV = "MODAL_TASK_ID"


def on_modal(env: Optional[Mapping[str, str]] = None) -> bool:
    """True inside a Modal container (its task id is in the environment), as the codebase detects it elsewhere."""
    return bool(((os.environ if env is None else env).get(MODAL_MARKER_ENV) or "").strip())


def wif_env_path(org: str = "lowprio", *, env: Optional[Mapping[str, str]] = None) -> str:
    """The WIF env file of ``org`` (``$SISTEER_WIF_ENV_DIR`` or ``~/.anthropic``, ``wif-<stem>.env``); an unknown org
    is a ``ValueError``, never a guess. ``env`` resolves the directory for ANOTHER process's environment (the grading
    daemon computes the digest a Petri drive must attest under the environment it hands that drive: its
    ``SISTEER_WIF_ENV_DIR`` and ``HOME``); the default is this process's, exactly as before."""
    try:
        stem = WIF_ORG_FILES[org]
    except KeyError:
        raise ValueError(f"org must be one of {sorted(WIF_ORG_FILES)}, got {org!r}") from None
    if env is None:
        base = os.path.expanduser(os.environ.get(WIF_ENV_DIR_ENV) or WIF_ENV_DIR_DEFAULT)
    else:
        base = env.get(WIF_ENV_DIR_ENV) or os.path.join(env.get("HOME") or os.path.expanduser("~"), WIF_ENV_DIR_DEFAULT[2:])
    return os.path.join(base, f"wif-{stem}.env")


def wif_ids(org: str = "lowprio", *, path: Optional[str] = None) -> Dict[str, str]:
    """The federation ids of ``org`` read off its env file: every :data:`WIF_REQUIRED_KEYS` present and non-empty, else
    :class:`PoolUnavailable` naming the file and what it lacks (an EMPTY file — ``wif-lowprio.env`` on this machine — is
    refused by name, never read as another pool's). The values are ids, not secrets; nothing is logged."""
    p = path or wif_env_path(org)
    if not os.path.isfile(p):
        raise PoolUnavailable(org, f"pool {org!r} unavailable over WIF: {p} does not exist")
    ids: Dict[str, str] = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            ids[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in WIF_REQUIRED_KEYS if not ids.get(k)]
    if missing:
        raise PoolUnavailable(org, f"pool {org!r} unavailable over WIF: {p} is {'empty' if not ids else 'incomplete'} — "
                                   f"lacks {missing} (each pool reads only its own ids file, never another pool's)")
    return {k: ids[k] for k in WIF_REQUIRED_KEYS}


def wif_ids_sha12(org: str = "lowprio", *, path: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    """sha256 of the org's ids FILE (its bytes), first 12 hex: the keyless half of a federated client's pool stamp and
    the value the grading daemon compares a Petri drive's attestation against. The file is validated first
    (:func:`wif_ids`), so an empty or incomplete file never yields a digest that looks like a record. ``env`` resolves
    the file under another process's environment (:func:`wif_env_path`)."""
    p = path or wif_env_path(org, env=env)
    wif_ids(org, path=p)
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def wif_org_of_ids_sha12(sha12: str, *, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The org whose CURRENT ids file digests to ``sha12`` (:func:`wif_ids_sha12` over every org of :data:`WIF_ORG_FILES`;
    an org whose file is absent, empty or incomplete simply does not match), else None. How a reader VALIDATES a drive's
    attested ``ids_sha12`` instead of trusting the org label written beside it (the Petri reader trusted
    ``auditor_org``)."""
    for org in WIF_ORG_FILES:
        try:
            if wif_ids_sha12(org, env=env) == sha12:
                return org
        except (PoolUnavailable, OSError):
            continue
    return None


def wif_pool_stamp(org: str = "lowprio", *, path: Optional[str] = None) -> Dict[str, str]:
    """``{"pool": org, "auth": "wif", "ids_sha12": <sha256(ids file)[:12]>}`` — never a token, never an id."""
    return {"pool": org, "auth": WIF_AUTH, "ids_sha12": wif_ids_sha12(org, path=path)}


def google_identity_token() -> str:
    """A fresh Google identity token for the Anthropic audience (the VM's attached service account); the SDK calls
    this at each exchange and caches the exchanged bearer itself."""
    import google.auth.transport.requests
    import google.oauth2.id_token

    return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), ANTHROPIC_WIF_AUDIENCE)


def wif_credentials(org: str = "lowprio", *, token_provider: Any = None, env_path: Optional[str] = None) -> Any:
    """The SDK's ``WorkloadIdentityCredentials`` for ``org``'s pool: the federation ids of ``~/.anthropic/wif-<stem>.env``
    (:func:`wif_ids`) passed EXPLICITLY, the identity-token provider ``token_provider`` (default
    :func:`google_identity_token`; injectable for tests and for a future Modal constructor). The ONE seam every federated
    client of this repository is built on — :func:`anthropic_client_wif` here, and the inspect_ai provider patch
    (``sisteer.eval.anthropic_wif.install_inspect_wif``) that hands the Petri legs' auditor / judge SDK the same
    credentials — so there is no second reader of the ids files. With the default provider, a Modal container REFUSES BY
    NAME (:func:`on_modal`): the GCP id-token path cannot work there."""
    from anthropic import WorkloadIdentityCredentials

    if token_provider is None and on_modal():
        raise PoolUnavailable(org, f"{SECRETS_MODE_ENV}={WIF_MODE} inside a Modal container ({MODAL_MARKER_ENV} is set): the GCP "
                                   "identity-token path (google.oauth2.id_token under ANTHROPIC_FEDERATION_RULE_ID_GCP) cannot work "
                                   "there and no Modal constructor exists yet (a follow-up: a sandbox identity token under the _MODAL "
                                   "rule, one org Secret per function) — no front-door judge runs on Modal; "
                                   "scripts/modal/seedrep_petri_drive.py keeps its own path")
    path = env_path or wif_env_path(org)
    ids = wif_ids(org, path=path)
    return WorkloadIdentityCredentials(
        identity_token_provider=(token_provider or google_identity_token),
        federation_rule_id=ids["ANTHROPIC_FEDERATION_RULE_ID_GCP"],
        organization_id=ids["ANTHROPIC_ORGANIZATION_ID"],
        service_account_id=ids["ANTHROPIC_SERVICE_ACCOUNT_ID"],
        workspace_id=ids["ANTHROPIC_WORKSPACE_ID"])


def anthropic_client_wif(org: str = "lowprio", *, max_retries: int = 5, timeout: float = REQUEST_TIMEOUT_S,
                         base_url: Optional[str] = None, token_provider: Any = None,
                         env_path: Optional[str] = None, async_client: bool = False, http_client: Any = None) -> Any:
    """An ``anthropic.Anthropic`` (``AsyncAnthropic`` with ``async_client``) authenticated through WORKLOAD IDENTITY
    FEDERATION for ``org``'s pool — no API key is read, set or stored anywhere: the SDK's ``WorkloadIdentityCredentials``
    (:func:`wif_credentials`) exchanges a Google identity token (``token_provider``, default :func:`google_identity_token`;
    injectable for tests) under the federation ids of ``~/.anthropic/wif-<stem>.env`` (:func:`wif_ids`), passed
    explicitly. With ``credentials=`` the SDK consults no environment variable for a CREDENTIAL, so a shell's
    ``ANTHROPIC_API_KEY`` is never read -- but its constructor still reads ``ANTHROPIC_CUSTOM_HEADERS`` and
    ``ANTHROPIC_WEBHOOK_SIGNING_KEY`` whatever it was given, and a static ``X-Api-Key`` / ``Authorization`` header parsed
    off the former rides EVERY request and makes the SDK's ``AccessTokenAuth`` skip the token exchange (the federated
    provider is never invoked; the request carries the stale header and is refused). So the inherited ``ANTHROPIC_*`` names
    are DROPPED first, by name (:func:`scrub_inherited_anthropic_environment`) -- whatever the
    mode switch says: this constructor IS the wif path. The request timeout is explicit, exactly as :func:`anthropic_client`;
    the pool stamp is :func:`wif_pool_stamp` — keyless. On Modal with the default provider: :class:`PoolUnavailable` by name
    (:func:`wif_credentials`).

    THE DESTINATION IS THE PIN, BY CONSTRUCTION: ``base_url`` may only spell the pinned API host
    (:data:`ANTHROPIC_BASE_URL`; a trailing slash aside) and an ``http_client``'s own base URL likewise -- anything else is
    :class:`BaseUrlRefused` BY NAME before the ids are read (:func:`pin_federated_destination`). The constructor used to forward
    an explicit ``base_url`` to the SDK, which built a credentialed client against it: the exchanged bearer and every request's
    contents could reach that host. An ``http_client`` carrying a static ``X-Api-Key`` / ``Authorization`` header is refused too
    (:func:`refuse_static_auth_headers`).

    AND ENFORCED WHERE THE BYTES LEAVE: the SDK is never handed a caller's httpx client -- its base URL
    says nothing about where a request hook, a mount, a subclass's ``send`` or a transport of its own would carry the bearer. The SDK
    client is built on the vault's OWN httpx client (:func:`federated_http_client`): no hook, no mount, no auth, no environment proxy,
    the SDK's own defaults, and its one transport the pin guard (:class:`PinnedTransport`) over httpx's HTTP transport, which refuses
    by name any request not addressed to the pin at the last step before the wire. The test seam is a bare ``http_client`` over a
    ``MockTransport`` and nothing else (:func:`transport_seam`): its transport, headers and timeout travel into the vault's client;
    anything more about it is refused by name.

    THE SDK'S OWN RETRIES ARE OFF, REPLAYED ABOVE THE LATCH (round 12): the SDK client is built with ``max_retries=0`` and the
    ``max_retries`` asked for here is the budget of :class:`LatchedRetryMessages` (:func:`federate_sdk_retries`), which retries
    what the SDK would have, as often and after the same delays, but checks the client's credential latch before every attempt
    -- so a worker asleep in a retry when a sibling met the run's first 401 / 403 wakes into the stop, never into another
    request. ``client.max_retries`` therefore reads 0 on a federated client; :func:`federated_retry_budget` reads the budget. The
    layer also refuses by name a static ``X-Api-Key`` / ``Authorization`` in any request's ``extra_headers`` (round 13).

    AND ITS COPIES STAY FEDERATED: the client is an instance of the vault's own subclass of the SDK's class
    (:func:`federated_sdk_class`), whose ``copy`` / ``with_options`` build a copy on the same credentials, token cache and pinned httpx
    client with the SDK's retries off, the SAME latch, the stamp and the layer (:func:`federated_copy`), refusing by name an
    ``http_client``, another base URL, a static header, a credential change or a middleware -- the SDK's own copy inherited the
    credentials and took a caller's httpx client, and carried the exchanged bearer wherever that client sent it."""
    import anthropic

    subject = f"the federated {'async ' if async_client else ''}client for pool {org!r} (sisteer.vault.anthropic_client_wif)"
    pinned = pin_federated_destination(base_url, http_client=http_client, subject=subject)   # first: a refused destination builds and reads nothing
    refuse_static_auth_headers(http_client=http_client, subject=subject)
    own = federated_http_client(http_client, async_client=async_client, timeout=timeout, subject=subject)   # round 13: the vault's own, over the pin guard
    path = env_path or wif_env_path(org)
    creds = wif_credentials(org, token_provider=token_provider, env_path=path)
    scrub_inherited_anthropic_environment()           # before the SDK's constructor reads the process environment
    cls = federated_sdk_class(anthropic.AsyncAnthropic if async_client else anthropic.Anthropic)   # round 14: its copies stay federated
    extra = {"http_client": own}                      # never the caller's object (round 13)
    client = cls(credentials=creds, base_url=pinned, max_retries=0, timeout=timeout, **extra)
    setattr(client, POOL_STAMP_ATTR, wif_pool_stamp(org, path=path))
    federate_sdk_retries(client, max_retries=max_retries)
    setattr(client, FEDERATED_DRESS_ATTR, _default_federated_dress)   # round 14: how a copy of this client is dressed (the stamp, the layer)
    return client


def pinned_base_url(override: Optional[str], *, source: str, default: str = ANTHROPIC_BASE_URL,
                    subject: str = "a federated client", hint: Optional[str] = None) -> str:
    """The ONE mode-aware rule for a base URL that reaches a client this module builds from OUTSIDE the process -- an
    environment variable or a config value, named by ``source`` (``CAPEVAL_JUDGE_BASE_URL``, a config key) -- as opposed to
    an in-process ``base_url=`` argument, which stays each constructor's test seam (a double of the service). Under ``op`` /
    ``env``: the stripped ``override`` else ``default``, byte for byte as every caller resolved it before. Under
    ``SISTEER_SECRETS_MODE=wif`` the ``default`` is PINNED: a federated client's bearer is minted for that host (the identity
    token's audience, :data:`ANTHROPIC_WIF_AUDIENCE`) and must not reach another, so an override naming any other host is
    :class:`BaseUrlRefused` BY NAME -- never silently ignored (the variable is a deliberate seam; a surprise there is a loud
    one) -- and one spelling the pinned host, a trailing slash aside, is the pin. Consult it BEFORE building the client, so a
    refused override builds nothing. A fourth mode value is :func:`secrets_mode`'s refusal. ``subject`` and ``hint`` are the
    caller's words for the message (which client, and where a test injects one instead).

    Round 3 on 545c56eda: ``capeval.judging.judge_base_url`` had this rule and the misalign judge's own copy of that
    function did not -- it handed an inherited ``CAPEVAL_JUDGE_BASE_URL`` to :func:`anthropic_client`, whose federated
    constructor honors an explicit URL. The rule lives here once; each reader types the refusal as its own
    (``JudgingError``, ``SystemExit``), and ``tests/test_base_url_pin_sources.py`` audits that every reader of such a
    variable calls it and that no constructor is fed an outside value directly."""
    override = (override or "").strip()
    if secrets_mode() == WIF_MODE:
        if override and override.rstrip("/") != default.rstrip("/"):
            raise BaseUrlRefused(
                f"{source}={override!r} under {SECRETS_MODE_ENV}={WIF_MODE}: {subject} is pinned to {default} (its bearer is "
                f"minted for that host and must not reach another); unset the override" + (f" -- {hint}" if hint else ""))
        return default
    return override or default


def pin_federated_destination(base_url: Optional[str] = None, *, http_client: Any = None,
                              subject: str = "a federated client") -> str:
    """THE destination of a federated client, decided at the CONSTRUCTOR (Critical): the pinned API
    host (:data:`ANTHROPIC_BASE_URL`), and nothing else, whatever the caller passed. An in-process ``base_url=`` naming another
    host -- the SDK would build a credentialed client against it and the exchanged bearer plus every request's contents would
    reach that host -- or an ``http_client`` whose own base URL names one, is :class:`BaseUrlRefused` BY NAME before anything is
    built (never silently re-pointed: a caller's argument is deliberate, and a surprise there is a loud one, the rule of
    :func:`pinned_base_url` and :func:`refuse_static_auth_headers`). One spelling the pinned host, a trailing slash aside, is the
    pin. :func:`pinned_base_url` is the per-OUTSIDE-value face of the same rule (an environment variable, a config value, consulted
    by the readers of such a value before they call a constructor); this is the per-CONSTRUCTOR face, which no caller can route
    around: :func:`anthropic_client_wif` (so :func:`anthropic_client` / :func:`anthropic_async_client` under wif) and the inspect
    provider patch (``sisteer.eval.anthropic_wif.install_inspect_wif``, both of its factories) consult it first.

    THE TEST SEAM IS A TRANSPORT, NOT A URL: a test that wants the request to go nowhere hands the constructor an ``http_client``
    (an ``httpx.Client`` / ``AsyncClient`` over an ``httpx.MockTransport``), and the request leaves the SDK addressed to the pinned
    host through that transport. Before round 12 an in-process ``base_url=`` was "each constructor's test seam", which is exactly
    the seam the finding named: the constructor accepted a non-Anthropic destination. ``subject`` is the caller's words for the
    message. Returns the pin.

    This is the BY-NAME face of the destination rule (what the caller spelled). The ENFORCED face is round 13's
    (:func:`federated_http_client` / :class:`PinnedTransport`): an http_client's base URL is not its destination, so the SDK is never
    handed the caller's client at all, and the pin is checked again on every request at the transport, where the bytes leave."""
    pin = ANTHROPIC_BASE_URL
    found = []
    url = (base_url or "").strip() if isinstance(base_url, str) or base_url is None else str(base_url).strip()
    if url and url.rstrip("/") != pin.rstrip("/"):
        found.append(f"base_url={url!r}")
    if http_client is not None:
        own = getattr(http_client, "base_url", None)
        own = ("" if own is None else str(own)).strip()
        if own and own.rstrip("/") != pin.rstrip("/"):
            found.append(f"http_client.base_url={own!r}")
    if found:
        raise BaseUrlRefused(
            f"{subject}: {' and '.join(found)} names a destination other than the pinned API host {pin} -- a federated client's "
            f"bearer is minted for that host (the identity token's audience) and must not reach another, so the constructor refuses "
            f"BY NAME under {SECRETS_MODE_ENV}={WIF_MODE} (it is the federated path whatever the switch says); a test injects an "
            f"http_client over a mock transport against the pinned host instead of a URL; refusing to build")
    return pin


# --- THE DESTINATION IS ENFORCED WHERE THE BYTES LEAVE (Critical) -------------------------------
# Round 12 read a caller-supplied `http_client`'s `base_url` and then handed that caller-owned object to the credentialed SDK. An
# httpx client's base URL is not its destination: a request event hook rewrites `request.url` AFTER the SDK's auth flow has put the
# exchanged bearer on the request (measured on bcd10c0eb: `httpx.Client(base_url=<the pin>, event_hooks={"request": [hook]})` passed
# the check and the hook carried the bearer to another host), a mount routes a URL pattern through another transport, a subclass
# overrides `send`, a transport of the caller's own goes wherever it likes. So under federation the SDK is NEVER handed a caller's
# httpx client. The vault builds the httpx client itself (:func:`federated_http_client`): the SDK's own defaults (the timeout, its
# connection limits, `follow_redirects`, its TCP keepalive socket options), no event hook, no mount, no auth, no environment proxy,
# and ONE transport -- :class:`PinnedTransport`, which refuses by name any request whose scheme / host / port are not the pin's at
# the last in-process step before a byte leaves (so an in-SDK rewrite -- a middleware model argument, a cross-host redirect the API
# would answer -- is refused there too, with the bearer already on the request and nowhere to go) -- over httpx's own HTTP transport.
# The offline test seam is the TRANSPORT alone: a caller's `http_client` is accepted only as a bare carrier of a `MockTransport`
# (:func:`transport_seam`: exactly the SDK flavour's `Client` / `AsyncClient`, its one transport a `MockTransport`, no mount, no hook,
# no auth, its base URL the pin or none, no static credential header), and even then only its transport, headers and timeout travel
# into the vault's own client; anything else about it is refused BY NAME, never silently dropped. The SDK's httpx flavour is read
# off the installed SDK (:func:`sdk_httpx`: `httpx` on anthropic 0.x, `httpx2` on 1.x) so the classes compared are the SDK's own.

#: the TCP keepalive options the SDK's own default httpx client sets on its transport (`anthropic.DefaultHttpxClient`), replicated
#: on the vault's: a long streamed judge call must not be dropped by an idle NAT. Kept equal to the SDK's by test.
FEDERATED_SOCKET_OPTIONS: Tuple[Tuple[int, int, Any], ...] = tuple(
    [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)]
    + ([(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL"), 60)] if getattr(socket, "TCP_KEEPINTVL", None) is not None
       else [(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPALIVE", 0x10), 60)] if sys.platform == "darwin" else [])
    + ([(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT"), 5)] if getattr(socket, "TCP_KEEPCNT", None) is not None else [])
    + ([(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE"), 60)] if getattr(socket, "TCP_KEEPIDLE", None) is not None else []))
_DEFAULT_PORTS = {"https": 443, "http": 80}
_FEDERATED_CLIENT_CLASSES: Dict[Tuple[str, bool], Any] = {}


def sdk_httpx() -> Any:
    """The httpx flavour the installed SDK is built on -- ``httpx`` on anthropic 0.x, ``httpx2`` on 1.x -- read off the SDK itself:
    the ``Client`` the SDK accepts as ``http_client`` (an exact ``isinstance``), the ``MockTransport`` a test seam must be, and the
    transports the vault builds all belong to that module, never to whichever ``httpx`` happens to import first."""
    from anthropic import _base_client as bc

    return getattr(bc, "httpx", None) or bc.httpx2


class PinnedTransport:
    """THE final in-process boundary of a federated client: the one transport of the vault's own httpx client, which forwards a request
    to ``inner`` (httpx's HTTP transport; a test's ``MockTransport``) only when its scheme, host and port are the pin's
    (:data:`ANTHROPIC_BASE_URL`), and otherwise raises :class:`BaseUrlRefused` BY NAME -- naming the destination it refused and the
    pin, never a header, a path or a body -- before any byte leaves. Runs AFTER everything that can rewrite a request's URL inside the
    process (the SDK's URL merge and middleware, httpx's auth flow, event hooks, a redirect's follow-up request), so what it sees is
    what would have been sent. Duck-typed over both httpx flavours' transport protocols (sync and async), as ``MockTransport`` is."""

    def __init__(self, inner: Any, *, subject: str = "a federated client", pin: str = ANTHROPIC_BASE_URL):
        self.inner = inner
        self.subject = subject
        self.pin = pin
        url = sdk_httpx().URL(pin)
        self._pin_key = (url.scheme, url.host, url.port or _DEFAULT_PORTS.get(url.scheme))

    def check(self, url: Any) -> None:
        key = (url.scheme, url.host, url.port or _DEFAULT_PORTS.get(url.scheme))
        if key != self._pin_key:
            where = f"{url.scheme}://{url.host}" + (f":{url.port}" if url.port else "")
            raise BaseUrlRefused(
                f"{self.subject}: a request addressed to {where} reached the federated client's transport -- the pin is {self.pin} and "
                f"a federated client's bearer is minted for that host (the identity token's audience) and must not reach another; "
                f"something between the SDK's constructor and the wire re-pointed the request (an http_client hook or mount, an SDK "
                f"middleware, a cross-host redirect); refusing before any byte leaves")

    def handle_request(self, request: Any) -> Any:
        self.check(request.url)
        return self.inner.handle_request(request)

    async def handle_async_request(self, request: Any) -> Any:
        self.check(request.url)
        return await self.inner.handle_async_request(request)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()

    async def aclose(self) -> None:
        aclose = getattr(self.inner, "aclose", None)
        if aclose is not None:
            await aclose()

    def __enter__(self) -> "PinnedTransport":
        enter = getattr(self.inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc: Any) -> None:
        leave = getattr(self.inner, "__exit__", None)
        if leave is not None:
            leave(*exc)

    async def __aenter__(self) -> "PinnedTransport":
        enter = getattr(self.inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        leave = getattr(self.inner, "__aexit__", None)
        if leave is not None:
            await leave(*exc)


def _federated_client_class(mod: Any, async_client: bool) -> Any:
    """The vault's own httpx client class for the SDK's flavour ``mod``: exactly the module's ``Client`` / ``AsyncClient`` plus the
    ``__del__`` the SDK's own default client wrapper has (closed when collected). Built once per flavour."""
    key = (mod.__name__, bool(async_client))
    cls = _FEDERATED_CLIENT_CLASSES.get(key)
    if cls is not None:
        return cls
    if async_client:
        class FederatedAsyncHttpxClient(mod.AsyncClient):
            """The httpx client of a federated ``AsyncAnthropic`` (:func:`sisteer.vault.federated_http_client`): the vault's own."""

            def __del__(self) -> None:
                if getattr(self, "is_closed", True):
                    return
                try:
                    import asyncio
                    asyncio.get_running_loop().create_task(self.aclose())
                except Exception:  # noqa: BLE001 -- no running loop: the pool is collected with the object, as the SDK's is
                    pass
        cls = FederatedAsyncHttpxClient
    else:
        class FederatedHttpxClient(mod.Client):
            """The httpx client of a federated ``Anthropic`` (:func:`sisteer.vault.federated_http_client`): the vault's own."""

            def __del__(self) -> None:
                if getattr(self, "is_closed", True):
                    return
                try:
                    self.close()
                except Exception:  # noqa: BLE001 -- as the SDK's wrapper
                    pass
        cls = FederatedHttpxClient
    _FEDERATED_CLIENT_CLASSES[key] = cls
    return cls


def transport_seam(http_client: Any = None, *, async_client: bool = False, subject: str = "a federated client") -> Any:
    """What a caller-supplied ``http_client`` may be under federation: NOTHING but a bare carrier of an offline ``MockTransport`` --
    exactly the SDK flavour's ``Client`` (``AsyncClient`` for an async door; never a subclass, whose ``send`` is its own), its one
    transport a ``MockTransport`` (never httpx's HTTP transport or a transport of the caller's own), no mount, no event hook, no auth.
    Returns that transport (None when no client was given); anything else is :class:`BaseUrlRefused` BY NAME, naming each thing found
    and never a value. The client's base URL is :func:`pin_federated_destination`'s to refuse and its headers
    :func:`refuse_static_auth_headers`'s; both run before this. The vault never hands the caller's object to the SDK whatever this
    returns (:func:`federated_http_client` builds its own client over the transport), so the check is a loud refusal of an argument
    that would otherwise be silently ignored, and the transport is the only thing a test can inject."""
    if http_client is None:
        return None
    mod = sdk_httpx()
    want = mod.AsyncClient if async_client else mod.Client
    found: List[str] = []
    if type(http_client) is not want:
        found.append(f"its type is {type(http_client).__module__}.{type(http_client).__qualname__}, not exactly {want.__module__}.{want.__name__}")
    transport = getattr(http_client, "_transport", None)
    if type(transport) is not mod.MockTransport:
        found.append(f"its transport is {type(transport).__module__}.{type(transport).__qualname__}, not {mod.__name__}.MockTransport")
    mounts = getattr(http_client, "_mounts", None) or {}
    if mounts:
        found.append(f"it carries {len(mounts)} mount(s)")
    hooks = getattr(http_client, "event_hooks", None) or {}
    n_hooks = sum(len(v or ()) for v in hooks.values()) if hasattr(hooks, "values") else 0
    if n_hooks:
        found.append(f"it carries {n_hooks} event hook(s) ({', '.join(k for k, v in hooks.items() if v)})")
    if getattr(http_client, "auth", None) is not None:
        found.append("it carries an auth")
    if found:
        raise BaseUrlRefused(
            f"{subject}: the http_client passed is not the offline test seam -- {'; '.join(found)} -- and under "
            f"{SECRETS_MODE_ENV}={WIF_MODE} the SDK is never handed a caller's httpx client (a request hook, a mount, a subclass's own "
            f"send or a transport of the caller's own can re-point the exchanged bearer off the pinned host {ANTHROPIC_BASE_URL} however "
            f"the client's base URL reads); the vault builds its own client over the pin guard, and a test injects a bare "
            f"{mod.__name__}.{want.__name__}(transport={mod.__name__}.MockTransport(handler)) whose transport alone is used; refusing to build")
    return transport


def federated_http_client(http_client: Any = None, *, async_client: bool = False, timeout: Any = None,
                          subject: str = "a federated client") -> Any:
    """THE httpx client every federated SDK client is built on -- the vault's own, never a caller's (the section comment above): the
    SDK flavour's ``Client`` / ``AsyncClient`` (:func:`_federated_client_class`) with ``base_url`` the pin, ``follow_redirects`` as the
    SDK's default, ``timeout`` the caller's (else the donor's own when it set one, else the SDK's default), no event hook, no mount, no
    auth, no environment proxy (an explicit transport: httpx consults none), and its ONE transport :class:`PinnedTransport` over
    httpx's own HTTP transport with the SDK's connection limits and keepalive socket options (:data:`FEDERATED_SOCKET_OPTIONS`) --
    or, when ``http_client`` is the offline seam (:func:`transport_seam`), over that seam's ``MockTransport``, with the seam's headers
    (a static credential among them already refused) carried along. Called by :func:`anthropic_client_wif` and by both factories of the
    inspect provider patch; the SDK's constructor then receives this client as its ``http_client`` and nothing of the caller's."""
    import anthropic

    mod = sdk_httpx()
    seam = transport_seam(http_client, async_client=async_client, subject=subject)
    if seam is None:
        limits = anthropic.DEFAULT_CONNECTION_LIMITS
        inner = (mod.AsyncHTTPTransport if async_client else mod.HTTPTransport)(limits=limits, socket_options=list(FEDERATED_SOCKET_OPTIONS))
    else:
        inner = seam
    if timeout is None or not isinstance(timeout, (int, float, mod.Timeout)):
        donor_timeout = getattr(http_client, "timeout", None) if seam is not None else None
        timeout = donor_timeout if donor_timeout is not None and donor_timeout != mod.Client().timeout else anthropic.DEFAULT_TIMEOUT
    kwargs: Dict[str, Any] = {"base_url": ANTHROPIC_BASE_URL, "timeout": timeout, "follow_redirects": True,
                              "transport": PinnedTransport(inner, subject=subject)}
    if seam is not None:
        kwargs["headers"] = http_client.headers
    return _federated_client_class(mod, async_client)(**kwargs)


def destination_refusal(exc: BaseException) -> Optional[BaseUrlRefused]:
    """The :class:`BaseUrlRefused` the pin guard raised inside the SDK's request, found through the ``__cause__`` chain (the SDK wraps
    anything its httpx client raises as ``APIConnectionError(...) from err``); None when ``exc`` is not that. The latched retry layer
    re-raises it typed and never retries it (a retry would only meet the guard again)."""
    seen: set = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, BaseUrlRefused):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__
    return None


# --- THE SDK'S OWN RETRIES, REPLAYED ABOVE THE LATCH (Major) -----------------------------------
# An SDK-internal retry of a 408 / 409 / 429 / 5xx / connection error sleeps and re-sends BELOW every latch of this repository:
# a worker inside one when a sibling's request met the run's first 401 / 403 issued up to `max_retries` more requests on the
# rejected credential, and no stop of ours could reach it (round 11 closed that on the live grader's path alone; the sweep's
# residual 10 named it on judge_sonnet 1, judge_overrefusal 1, the misalign PoolClient 1, build_probe_set 2, ClaudeJudge 2,
# grokmt_analyze 3, the rewriter 8 and the inspect provider's clients). The ruling is that EVERY driver stops at the first
# rejection, so the rule lives in the ONE constructor: under federation the SDK client is built with `max_retries=0` and the
# count the caller asked for becomes the budget of THIS layer, which replays the SDK's own policy -- the same statuses
# (`x-should-retry`, 408, 409, 429, every 5xx, connection errors, `RetryableError`, walking `__cause__` as the SDK does), the
# same delays (`retry-after-ms` / `retry-after` up to 60 s, else 0.5 s doubling to 8 s with the SDK's jitter) -- but checks the
# client's latch BEFORE EVERY ATTEMPT, the first and each wake, and classifies the credential FIRST after every failure. So a
# driver's robustness contract is unchanged (the same errors retried the same number of times) and a sleeper wakes into the
# typed stop, never into another request. op / env are untouched: the keyed branches build the SDK client with the caller's
# `max_retries` and no layer, byte for byte as before.
SDK_RETRY_INITIAL_DELAY_S = 0.5
SDK_RETRY_MAX_DELAY_S = 8.0
SDK_RETRY_AFTER_CAP_S = 60.0
SDK_RETRY_STATUSES = (408, 409, 429)          # and every 5xx


def sdk_would_retry(exc: BaseException) -> Tuple[bool, Any]:
    """Whether the SDK's own request loop would retry ``exc`` (``anthropic._base_client._should_retry_exception`` /
    ``_should_retry``, replayed): a ``RetryableError``, an ``APIStatusError`` whose response says ``x-should-retry: true`` (or not
    ``false``) and whose status is 408 / 409 / 429 / >= 500, an ``APIConnectionError`` (a timeout included) -- each also found
    through the ``__cause__`` chain, as the SDK walks it. Returns ``(retry, response)``, the response for the retry-after headers.
    A credential rejection is never asked about: the caller classifies it first."""
    import anthropic

    retryable = getattr(anthropic, "RetryableError", None)
    seen: set = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if retryable is not None and isinstance(cur, retryable):
            return True, None
        if isinstance(cur, anthropic.APIStatusError):
            resp = getattr(cur, "response", None)
            headers = getattr(resp, "headers", None)
            flag = headers.get("x-should-retry") if headers is not None and hasattr(headers, "get") else None
            if flag == "true":
                return True, resp
            if flag == "false":
                return False, resp
            status = getattr(cur, "status_code", None)
            if status is None:
                status = getattr(resp, "status_code", None)
            return (isinstance(status, int) and (status in SDK_RETRY_STATUSES or status >= 500)), resp
        if isinstance(cur, anthropic.APIConnectionError):
            return True, None
        cur = cur.__cause__
    return False, None


def _retry_after_s(headers: Any) -> Optional[float]:
    """The SDK's ``_parse_retry_after_header``: ``retry-after-ms`` (milliseconds), else ``retry-after`` as seconds, else as a date."""
    if headers is None or not hasattr(headers, "get"):
        return None
    try:
        return float(headers.get("retry-after-ms", None)) / 1000
    except (TypeError, ValueError):
        pass
    retry_header = headers.get("retry-after")
    try:
        return float(retry_header)
    except (TypeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_tz(retry_header) if retry_header else None
    except Exception:  # noqa: BLE001 -- an unparsable date is no header
        parsed = None
    if parsed is None:
        return None
    return float(email.utils.mktime_tz(parsed) - time.time())


def sdk_retry_delay(nb_retries: int, response: Any = None) -> float:
    """The SDK's ``_calculate_retry_timeout``: the server's retry-after when it is between 0 and 60 s, else
    ``min(0.5 * 2 ** nb_retries, 8.0)`` scaled by the SDK's jitter (``1 - 0.25 * random()``); never negative."""
    after = _retry_after_s(getattr(response, "headers", None))
    if after is not None and 0 < after <= SDK_RETRY_AFTER_CAP_S:
        return after
    sleep_s = min(SDK_RETRY_INITIAL_DELAY_S * (2.0 ** min(int(nb_retries), 1000)), SDK_RETRY_MAX_DELAY_S)
    t = sleep_s * (1 - 0.25 * random.random())
    return t if t >= 0 else 0.0


class _LatchedRetry:
    """The shared half of the federated clients' retry layer: the inner SDK resource, the CLIENT whose latch is consulted (resolved
    at EVERY attempt, so a latch a driver attaches to the client later -- the live grader's ``ThrottledClient`` attaches the
    grader's -- is the one read), and the retry budget the caller asked the vault for. Every other attribute forwards to the
    resource (``with_raw_response``, ...). The ``max_retries`` here is what the SDK's would have been: a record, and the budget."""

    #: the resource this layer stands for, in the refusal's words (``messages`` / ``messages.batches``)
    resource = "messages"

    def __init__(self, inner: Any, client: Any, max_retries: int):
        self._inner = inner
        self._client = client
        self.max_retries = max(0, int(max_retries))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_inner"], name)

    def _latch(self) -> CredentialLatch:
        return client_credential_latch(self._client, create=True)

    def _refuse_request(self, kwargs: Mapping[str, Any], what: str) -> None:
        """BEFORE every request of a federated client (Major): a caller's per-request ``extra_headers``
        carrying a static ``X-Api-Key`` / ``Authorization`` is refused BY NAME (:func:`refuse_static_auth_headers`) -- the SDK's
        ``AccessTokenAuth`` skips the token exchange for a request already carrying one, so the static header would be sent in place
        of the exchanged bearer (measured: zero exchanges, the stale header on the wire, the 401 it earns). Round 11 put this check on
        the inspect provider's wrapper alone; it is the layer's now, so every door of every federated client has it."""
        refuse_static_auth_headers(extra_headers=kwargs.get("extra_headers"),
                                   subject=f"a request of the federated client for pool {pool_of_client(self._client)!r} ({self.resource}.{what})")

    def _after_failure(self, latch: CredentialLatch, exc: BaseException, nb: int) -> Tuple[bool, float]:
        """After a failed attempt: a credential rejection is held on the client's latch and NEVER retried (``retry`` False -- the
        caller re-raises it as the SDK raised it, so the driver's own classification of that one call is unchanged; every sibling's
        next attempt raises the typed error from the check); anything else is retried iff the SDK would have and the budget allows,
        after the SDK's delay. Returns ``(retry, delay_s)``."""
        if latch.rejection(exc) is not None:
            return False, 0.0
        retry, resp = sdk_would_retry(exc)
        if not retry or nb >= self.max_retries:
            return False, 0.0
        return True, sdk_retry_delay(nb, resp)

    def _attempt(self, call: Callable[[], Any], kwargs: Optional[Mapping[str, Any]] = None, what: str = "request") -> Any:
        self._refuse_request(kwargs or {}, what)     # round 13: a static per-request credential header, before anything
        latch = self._latch()
        nb = 0
        while True:
            latch.check()                            # BEFORE EVERY attempt: the first, and each wake from the sleep below
            try:
                return call()
            except Exception as e:                   # noqa: BLE001 -- classified: the credential first, then the SDK's own policy
                refused = destination_refusal(e)
                if refused is not None:              # round 13: the pin guard spoke inside the SDK's request -- typed, never retried
                    raise refused from e
                retry, delay = self._after_failure(latch, e, nb)
                if not retry:
                    raise
                nb += 1
                time.sleep(delay)                    # the SDK's own sleep, above the latch (a test patches time.sleep as it does the SDK's)

    async def _attempt_async(self, call: Callable[[], Any], kwargs: Optional[Mapping[str, Any]] = None, what: str = "request") -> Any:
        import asyncio

        self._refuse_request(kwargs or {}, what)
        latch = self._latch()
        nb = 0
        while True:
            latch.check()
            try:
                return await call()
            except Exception as e:                   # noqa: BLE001 -- classified as above
                refused = destination_refusal(e)
                if refused is not None:
                    raise refused from e
                retry, delay = self._after_failure(latch, e, nb)
                if not retry:
                    raise
                nb += 1
                await asyncio.sleep(delay)


class _LatchedRetryBatches(_LatchedRetry):
    """``messages.batches`` of a federated client with the same layer around its control-plane calls (the Batch judges' submit /
    poll / harvest: single synchronous calls whose SDK retries used to be the SDK's; now this layer's, latch-checked)."""

    resource = "messages.batches"

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.create(*args, **kwargs), kwargs, "create")

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.retrieve(*args, **kwargs), kwargs, "retrieve")

    def list(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.list(*args, **kwargs), kwargs, "list")

    def cancel(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.cancel(*args, **kwargs), kwargs, "cancel")

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.delete(*args, **kwargs), kwargs, "delete")

    def results(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.results(*args, **kwargs), kwargs, "results")


class _AsyncLatchedRetryBatches(_LatchedRetry):
    resource = "messages.batches"

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.create(*args, **kwargs), kwargs, "create")

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.retrieve(*args, **kwargs), kwargs, "retrieve")

    def list(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.list(*args, **kwargs), kwargs, "list")

    def cancel(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.cancel(*args, **kwargs), kwargs, "cancel")

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.delete(*args, **kwargs), kwargs, "delete")

    def results(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.results(*args, **kwargs), kwargs, "results")


class _LatchedRetryStream:
    """The stream manager of :meth:`LatchedRetryMessages.stream`: the request goes out at ``__enter__``, which is where the layer's
    attempts run (a fresh SDK manager per attempt); the entered stream is the SDK's own. A static per-request credential header was
    refused when ``stream`` was called, before this manager existed (and is refused again at the attempt)."""

    def __init__(self, owner: "_LatchedRetry", args: tuple, kwargs: dict):
        self._owner, self._args, self._kwargs, self._mgr = owner, args, kwargs, None

    def __enter__(self) -> Any:
        def enter():
            mgr = self._owner._inner.stream(*self._args, **self._kwargs)
            return mgr, mgr.__enter__()
        self._mgr, stream = self._owner._attempt(enter, self._kwargs, "stream")
        return stream

    def __exit__(self, *exc: Any) -> Any:
        return self._mgr.__exit__(*exc)

    async def __aenter__(self) -> Any:
        async def enter():
            mgr = self._owner._inner.stream(*self._args, **self._kwargs)
            return mgr, await mgr.__aenter__()
        self._mgr, stream = await self._owner._attempt_async(enter, self._kwargs, "stream")
        return stream

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._mgr.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or self.__dict__.get("_mgr") is None:
            raise AttributeError(name)
        return getattr(self.__dict__["_mgr"], name)


class LatchedRetryMessages(_LatchedRetry):
    """The ``messages`` resource of a federated SYNC client -- ``create`` / ``parse`` / ``count_tokens`` / ``stream`` and
    ``batches`` -- with the SDK's own retry policy replayed above the client's credential latch (the section comment above).
    Installed by :func:`federate_sdk_retries` on every client :func:`anthropic_client_wif` builds and on the inspect provider
    patch's clients; never on a keyed client."""

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.create(*args, **kwargs), kwargs, "create")

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.parse(*args, **kwargs), kwargs, "parse")

    def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt(lambda: self._inner.count_tokens(*args, **kwargs), kwargs, "count_tokens")

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse_request(kwargs, "stream")      # at the call, before a manager exists (and again at the attempt)
        return _LatchedRetryStream(self, args, kwargs)

    @property
    def batches(self) -> Any:
        b = self.__dict__.get("_batches")
        if b is None:
            b = self.__dict__["_batches"] = _LatchedRetryBatches(self._inner.batches, self._client, self.max_retries)
        return b


class AsyncLatchedRetryMessages(_LatchedRetry):
    """The async twin of :class:`LatchedRetryMessages` (``AsyncAnthropic``: build_probe_set's client, the inspect provider's)."""

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.create(*args, **kwargs), kwargs, "create")

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.parse(*args, **kwargs), kwargs, "parse")

    def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.count_tokens(*args, **kwargs), kwargs, "count_tokens")

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse_request(kwargs, "stream")
        return _LatchedRetryStream(self, args, kwargs)

    @property
    def batches(self) -> Any:
        b = self.__dict__.get("_batches")
        if b is None:
            b = self.__dict__["_batches"] = _AsyncLatchedRetryBatches(self._inner.batches, self._client, self.max_retries)
        return b


def federate_sdk_retries(client: Any, *, max_retries: int) -> Any:
    """Put the latched retry layer on a federated SDK client: its ``messages`` resource becomes :class:`LatchedRetryMessages`
    (:class:`AsyncLatchedRetryMessages` for an ``AsyncAnthropic``) carrying ``max_retries`` as the budget. The client MUST have
    been built with the SDK's own retries off (``max_retries=0``); anything else is refused by name -- the layer above a retrying
    SDK would be two retry layers, the lower one below the latch, the very thing this replaces. Returns the layer."""
    import anthropic

    sdk = getattr(client, "max_retries", None)
    if sdk != 0:
        raise ValueError(f"federate_sdk_retries: the SDK client carries max_retries={sdk!r}; a federated client is built with the "
                         "SDK's own retries OFF (max_retries=0) and its retries replayed here, above the credential latch (round 12)")
    cls = AsyncLatchedRetryMessages if isinstance(client, anthropic.AsyncAnthropic) else LatchedRetryMessages
    layer = cls(client.messages, client, max_retries)
    client.messages = layer
    return layer


def federated_retry_budget(client: Any) -> Optional[int]:
    """The retry budget the latched layer of a federated client carries (the count its builder asked the vault for; the SDK's own
    ``max_retries`` on such a client is 0), read through forwarding wrappers (``ThrottledClient``'s and the inspect patch's
    ``messages`` wrappers keep the layer as their ``_inner``); None for a keyed client, whose SDK ``max_retries`` IS its budget."""
    m = getattr(client, "messages", None)
    seen: set = set()
    while m is not None and id(m) not in seen:
        if isinstance(m, _LatchedRetry):
            return m.max_retries
        seen.add(id(m))
        m = getattr(m, "_inner", None)
    return None


# --- A COPY OF A FEDERATED CLIENT IS FEDERATED (Critical) --------------------------------------------
# Rounds 12 and 13 put everything that makes a client federated ON THE INSTANCE the SDK's base class returned: the vault's own httpx
# client over the pin guard as the constructor's `http_client`, the latched retry layer as `client.messages`, the stamp and the latch as
# attributes. The SDK's `copy` / `with_options` (and `with_middleware`, which is a copy) build a NEW client from the constructor's
# arguments -- the credentials and the token cache inherited, a replacement `http_client` accepted -- and nothing of the instance's
# dressing travels (measured on dbd7473df: `with_options(http_client=<a hooked client whose base URL is the pin>)` on a client of
# `vault.anthropic_client("lowprio")` built without a word and its one request reached the recorder at another host with the exchanged
# bearer; the copy carried no latch and no layer, so a rejection met on it stopped nothing). So every federated SDK client is an instance of the vault's
# OWN subclass of the SDK's class (:func:`federated_sdk_class`: `FederatedAnthropic` / `FederatedAsyncAnthropic`; the inspect factories
# build the same class over the provider's), whose `copy` / `with_options` are :func:`federated_copy` -- the copy built DIRECTLY on the
# parent's credentials, token cache and the vault's own httpx client (exactly what the SDK's own copy passes for a base client), with
# the SDK's retries OFF, and dressed as its parent was: the parent's latch (the SAME object, attached first), then the parent's dress
# (:data:`FEDERATED_DRESS_ATTR`: the vault's -- the stamp and the latched retry layer -- or the inspect patch's, which adds its abort
# wrapper) -- and whose `with_middleware` refuses by name. A copy takes `timeout`, `max_retries` (its budget; else the parent's),
# `default_headers` / `set_default_headers` (a static credential header refused, :func:`refuse_static_auth_headers`), `default_query` /
# `set_default_query` and the pin's own spelling of `base_url` (:func:`pin_federated_destination`); an `http_client` is refused by name
# as the destination class it is (round 13's rule holds for a copy: the SDK is never handed a caller's httpx client), and every other
# argument of the SDK's `copy` -- a credential (`api_key`, `auth_token`, `credentials`, `config`, `profile`), a `webhook_key`, a
# `middleware`, `_extra_kwargs`, an unknown name -- by name too. The subclass is also what makes the bypass harmless: the installed SDK
# forwards credentials through `copy` only for its EXACT base classes (`anthropic._client._is_base_client`), so the SDK's own `copy`
# called on the class builds a copy with no credentials and no token cache -- there is no bearer such a copy could carry anywhere.
# (The same gate skips the SDK's environment auto-discovery for a subclass, which a federated client never wanted.) op / env build the
# SDK's own class, whose copies are the SDK's own, byte for byte.


class FederatedCopyRefused(VaultSecretError):
    """A ``copy`` / ``with_options`` / ``with_middleware`` argument a federated client does not take (:func:`federated_copy`)."""


#: the attribute under which a federated client carries the callable that dresses a COPY of it as it was dressed itself --
#: ``dress(copy, parent, budget)`` -- the vault's own (:func:`_default_federated_dress`: the parent's pool stamp, the latched retry layer
#: with ``budget``) or the inspect provider patch's (the install's latch, the layer, its abort wrapper). The parent's latch is attached to
#: the copy BEFORE the dress runs, by :func:`federated_copy` itself; the dress travels to the copy, so a copy of a copy is dressed too.
FEDERATED_DRESS_ATTR = "sisteer_federated_dress"
#: the arguments of the SDK's ``copy`` a federated copy accepts (``base_url`` as the pin's own spelling only; ``http_client`` as None
#: only -- "keep the parent's", the SDK's own reading of None); every other name is refused by name
FEDERATED_COPY_OPTIONS: Tuple[str, ...] = ("timeout", "max_retries", "default_headers", "set_default_headers", "default_query",
                                           "set_default_query", "base_url", "http_client")
_FEDERATED_SDK_CLASSES: Dict[Any, Any] = {}
_NOT_GIVEN = object()


def _copy_subject(client: Any) -> str:
    return (f"a copy of the federated client for pool {pool_of_client(client)!r} (built by sisteer.vault.anthropic_client / "
            f"anthropic_async_client / anthropic_client_wif, or the inspect provider patch)")


def federated_sdk_class(cls: Any) -> Any:
    """The class every federated SDK client is an instance of: the vault's own subclass of ``cls`` (``anthropic.Anthropic`` /
    ``anthropic.AsyncAnthropic``; the inspect provider's class), named ``Federated<cls>``, whose ``copy`` / ``with_options`` are
    :func:`federated_copy` and whose ``with_middleware`` is :class:`FederatedCopyRefused` by name (the section comment above). Built
    once per class; a federated class is returned as it is; anything that is not a class is a ``TypeError`` by name (a stand-in for the
    SDK's class is never silently wrapped)."""
    if not isinstance(cls, type):
        raise TypeError(f"federated_sdk_class: {cls!r} is not a class (the SDK's client class, or the inspect provider's, is expected)")
    if cls in _FEDERATED_SDK_CLASSES.values():
        return cls
    found = _FEDERATED_SDK_CLASSES.get(cls)
    if found is not None:
        return found

    class Federated(cls):  # type: ignore[misc,valid-type]
        def copy(self, **overrides: Any) -> Any:
            return federated_copy(self, overrides)

        with_options = copy

        def with_middleware(self, *middleware: Any) -> Any:
            raise FederatedCopyRefused(
                f"{_copy_subject(self)}: with_middleware() -- a copy of a federated client carries exactly its parent's request pipeline "
                f"(the vault's own httpx client over the pin guard, the latched retry layer, its latch); a middleware is refused BY NAME "
                f"under {SECRETS_MODE_ENV}={WIF_MODE}; build another client through the vault for anything else")

    Federated.__name__ = f"Federated{cls.__name__}"
    Federated.__qualname__ = Federated.__name__
    Federated.__module__ = __name__
    Federated.__doc__ = (f"A federated ``{cls.__module__}.{cls.__qualname__}`` (sisteer.vault.federated_sdk_class): its ``copy`` / "
                         f"``with_options`` keep the federation (sisteer.vault.federated_copy); ``with_middleware`` is refused by name.")
    _FEDERATED_SDK_CLASSES[cls] = Federated
    return Federated


def _default_federated_dress(copy: Any, parent: Any, budget: int) -> None:
    """How the vault dresses a copy of a client it built: the parent's pool stamp, then the latched retry layer with ``budget`` (the
    parent's latch is already on the copy, attached by :func:`federated_copy`)."""
    stamp = getattr(parent, POOL_STAMP_ATTR, None)
    if isinstance(stamp, dict):
        setattr(copy, POOL_STAMP_ATTR, dict(stamp))
    federate_sdk_retries(copy, max_retries=budget)


def federated_copy(client: Any, overrides: Mapping[str, Any]) -> Any:
    """``copy`` / ``with_options`` of a federated client (the section comment above): a client of the same federated class, built
    DIRECTLY on the parent's credentials, token cache (one exchange serves both) and the vault's own httpx client over the pin guard
    (the same object), its base URL the pin, the SDK's retries OFF, the parent's headers and query merged with the copy's as the SDK's
    own copy merges them, then the parent's latch (the same object) and the parent's dress (:data:`FEDERATED_DRESS_ATTR`) with the
    budget asked (``max_retries``, else the parent's). Refused BY NAME before anything is built: an ``http_client``
    (:class:`BaseUrlRefused`), a ``base_url`` naming another host (:func:`pin_federated_destination`), a static credential header in
    ``default_headers`` / ``set_default_headers`` (:func:`refuse_static_auth_headers`), and any other argument
    (:class:`FederatedCopyRefused`, naming it)."""
    subject = _copy_subject(client)
    given = dict(overrides)
    unknown = sorted(k for k in given if k not in FEDERATED_COPY_OPTIONS)
    if unknown:
        raise FederatedCopyRefused(
            f"{subject}: copy() / with_options() with {', '.join(unknown)} -- a copy of a federated client carries its parent's credentials, "
            f"token cache, destination and request pipeline, and takes only timeout, max_retries, default_headers / set_default_headers, "
            f"default_query / set_default_query and the pinned host as base_url; a credential (api_key, auth_token, credentials, config, "
            f"profile), a webhook key, a middleware or an argument the SDK does not know is refused BY NAME under {SECRETS_MODE_ENV}={WIF_MODE}; "
            f"build another client through the vault for anything else")
    if given.get("http_client") is not None:
        raise BaseUrlRefused(
            f"{subject}: copy() / with_options(http_client=...) -- under {SECRETS_MODE_ENV}={WIF_MODE} the SDK is never handed a caller's httpx "
            f"client (a request hook, a mount, a subclass's own send or a transport of its own can carry the exchanged bearer off the pinned "
            f"host {ANTHROPIC_BASE_URL} however the client's base URL reads; round 13's rule holds for a copy): the copy keeps its parent's own "
            f"client over the pin guard, and a test injects its mock transport at the CONSTRUCTOR; refusing to copy")
    pin_federated_destination(given.get("base_url"), subject=subject)
    if given.get("default_headers") is not None and given.get("set_default_headers") is not None:
        raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")     # the SDK's own rule
    new_headers = given.get("default_headers") if given.get("default_headers") is not None else given.get("set_default_headers")
    refuse_static_auth_headers(default_headers=new_headers, subject=subject)
    if getattr(client, "api_key", None) is not None or getattr(client, "auth_token", None) is not None:
        raise FederatedCopyRefused(f"{subject}: the parent carries a static credential beside its federated one; refusing to copy")
    asked = given.get("max_retries", _NOT_GIVEN)
    if asked is _NOT_GIVEN or asked is None:
        budget = federated_retry_budget(client)
        if budget is None:
            raise FederatedCopyRefused(f"{subject}: the parent carries no latched retry layer (it was not built by the vault); refusing to copy")
    else:
        budget = max(0, int(asked))
    headers = dict(getattr(client, "_custom_headers", None) or {})
    if given.get("default_headers") is not None:
        headers = {**headers, **dict(given["default_headers"])}
    elif given.get("set_default_headers") is not None:
        headers = dict(given["set_default_headers"])
    params = dict(getattr(client, "_custom_query", None) or {})
    if given.get("default_query") is not None:
        params = {**params, **dict(given["default_query"])}
    elif given.get("set_default_query") is not None:
        params = dict(given["set_default_query"])
    scrub_inherited_anthropic_environment()           # the SDK's constructor reads ANTHROPIC_CUSTOM_HEADERS whatever it is given (round 10)
    copy = type(client)(
        credentials=client.credentials, _token_cache=getattr(client, "_token_cache"),      # what the SDK's own copy passes for a base client
        base_url=client.base_url, timeout=given.get("timeout", client.timeout), max_retries=0,
        http_client=getattr(client, "_client"),                                           # the vault's own, over the pin guard: the same object
        default_headers=headers, default_query=params, webhook_key=client.webhook_key,
        middleware=getattr(client, "_middleware", None),
        _strict_response_validation=getattr(client, "_strict_response_validation", False))
    attach_credential_latch(copy, client_credential_latch(client, create=True))           # ONE latch for parent and copy
    dress = getattr(client, FEDERATED_DRESS_ATTR, None) or _default_federated_dress
    dress(copy, client, budget)
    setattr(copy, FEDERATED_DRESS_ATTR, dress)
    return copy


#: THE PROCESS-ENVIRONMENT SCRUB under ``wif``: every name with this prefix is DROPPED -- by NAME, never a value read --
#: before a federated client is built. The prefix, not a list: round 9 dropped five names and the
#: SDK's constructor reads others whatever credential it was given -- ``ANTHROPIC_CUSTOM_HEADERS`` (parsed into EVERY request's
#: headers; a static ``X-Api-Key`` / ``Authorization`` there makes ``AccessTokenAuth`` skip the token exchange, so the federated
#: provider is never invoked and the stale header is sent) and ``ANTHROPIC_WEBHOOK_SIGNING_KEY``; its auto-discovery reads
#: ``ANTHROPIC_PROFILE`` / ``ANTHROPIC_CONFIG_DIR`` and the identity trio; and nothing under wif has a use for any of them (the
#: federation ids come off a FILE). Known inherited names are listed in :data:`WIF_INHERITED_ENV_NAMES` for the tests' synthetic
#: shell; the RULE is the prefix. The shell spelling of the same rule is :data:`WIF_SHELL_SCRUB`.
WIF_SCRUB_PREFIX = "ANTHROPIC_"
#: bash's name-only expansion of every variable with the prefix (``${!PREFIX@}`` lists NAMES; no value is read or printed) -- the
#: ONE scrub line every generated judging unit, launcher and the Petri drive carry under wif (reproduce/lib/secrets.sh,
#: launch_judge.sh, finish_lane.sh, grade_sampled.sh, grade_battery.sh, petri_drive_full22.sh); a no-op when nothing matches, under
#: ``set -u`` too. ``tests/test_wif_process_environment.py`` pins the shell lines to this string and RUNS each under a synthetic
#: shell carrying every name of :data:`WIF_INHERITED_ENV_NAMES`.
WIF_SHELL_SCRUB = 'unset "${!' + WIF_SCRUB_PREFIX + '@}"'
#: the ``ANTHROPIC_*`` names a Claude Code or vault shell is KNOWN to carry (the credential and routing variables: an OAuth
#: bearer under ``ANTHROPIC_AUTH_TOKEN``, a local gateway under ``ANTHROPIC_BASE_URL``, env mode's one key and a vault shell's
#: per-org keys) plus the ones the SDK reads outside its credential guard (``ANTHROPIC_CUSTOM_HEADERS``,
#: ``ANTHROPIC_WEBHOOK_SIGNING_KEY``) and its auto-discovery selectors (``ANTHROPIC_PROFILE``, ``ANTHROPIC_CONFIG_DIR``). A documented
#: SUBSET of what the scrub drops -- the tests' synthetic shell, never the rule.
WIF_INHERITED_ENV_NAMES: Tuple[str, ...] = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                                            "ANTHROPIC_API_KEY_LOWPRIO", "ANTHROPIC_API_KEY_BATCH",
                                            "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_WEBHOOK_SIGNING_KEY",
                                            "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR")


#: the request headers that carry a STATIC credential: the SDK's ``AccessTokenAuth`` skips the
#: federated token exchange on a request already carrying one, so an EXPLICIT ``default_headers`` / ``http_client`` header /
#: per-request ``extra_headers`` naming either -- a caller's argument this time, not an inherited variable (round 10's scrub) --
#: would ride a federated client's every request in place of the exchanged bearer (``client.api_key`` None, the provider never
#: invoked). Header names are case-insensitive: compared lower-cased.
STATIC_AUTH_HEADERS: Tuple[str, ...] = ("x-api-key", "authorization")


def static_auth_header_names(headers: Any) -> Tuple[str, ...]:
    """The names in ``headers`` (a mapping, an ``httpx.Headers``; None = none) that are a static credential header
    (:data:`STATIC_AUTH_HEADERS`, compared lower-cased), in the caller's own spelling and order. Never a value."""
    if headers is None:
        return ()
    try:
        names = list(headers.keys()) if hasattr(headers, "keys") else list(headers)
    except Exception:  # noqa: BLE001 -- not a mapping: nothing to name
        return ()
    return tuple(str(k) for k in names if isinstance(k, str) and k.lower() in STATIC_AUTH_HEADERS)


class StaticAuthRefused(ValueError):
    """An explicit static credential header offered to a federated client (:func:`refuse_static_auth_headers`)."""


def refuse_static_auth_headers(*, default_headers: Any = None, http_client: Any = None, extra_headers: Any = None,
                               subject: str = "a federated client") -> None:
    """REFUSE BY NAME an explicit static credential header on its way into a federated client -- the SDK constructor's
    ``default_headers``, the headers of an explicit ``http_client`` (httpx merges a client's headers into every request it
    builds) or one request's ``extra_headers`` -- as :class:`StaticAuthRefused` (a ``ValueError``), before anything is built
    or sent. Under federation the credential is the identity token's exchange; a static ``X-Api-Key`` / ``Authorization``
    beside ``credentials=`` is a contradiction (which credential did the caller mean?), and the SDK resolves it silently in the
    static header's favour (:data:`STATIC_AUTH_HEADERS`). Never dropped silently: the argument is a caller's deliberate one,
    and a surprise there is a loud one (the same rule as :func:`pinned_base_url`'s refusal of an outside URL). Nothing in this
    repository passes such a header to a federated client; the inspect provider patch
    (``sisteer.eval.anthropic_wif.install_inspect_wif``) refuses it in both factories and before every request, and since round
    13 the latched retry layer of EVERY federated client (:class:`_LatchedRetry`) refuses a request's
    ``extra_headers`` carrying one before the wire, on every door -- ``messages.{create,parse,count_tokens,stream}`` and
    ``messages.batches.*``, sync and async -- which the vault's own clients lacked (measured: zero exchanges, the static header
    sent)."""
    found = []
    for where, headers in (("default_headers", default_headers),
                           ("http_client.headers", getattr(http_client, "headers", None) if http_client is not None else None),
                           ("extra_headers", extra_headers)):
        names = static_auth_header_names(headers)
        if names:
            found.append(f"{where}: {', '.join(names)}")
    if found:
        raise StaticAuthRefused(
            f"{subject}: an explicit static credential header was passed ({'; '.join(found)}) -- under {SECRETS_MODE_ENV}={WIF_MODE} "
            f"the credential is the federation identity's exchanged bearer, and the SDK's AccessTokenAuth skips that exchange for a "
            f"request already carrying X-Api-Key / Authorization (the static header would be sent, the identity never used); an API "
            f"key or bearer has no place beside credentials= -- drop the header; refusing to build / send")


def scrub_inherited_anthropic_environment(env: Any = None) -> Tuple[str, ...]:
    """Drop EVERY inherited ``ANTHROPIC_*`` name (:data:`WIF_SCRUB_PREFIX`) from ``env`` (default ``os.environ``) -- a key
    iteration and a delete, NO value read -- and return the names dropped, sorted (never a value). The ONE scrub every
    federated constructor runs before the SDK's constructor reads the process environment: :func:`anthropic_client_wif`
    (so :func:`anthropic_client` / :func:`anthropic_async_client` under wif), the inspect provider patch
    (``sisteer.eval.anthropic_wif.install_inspect_wif``'s two factories) and :func:`pin_process_environment`'s wif branch;
    the corpus generator carries its twin (``corpus_generator.client.scrub_inherited_anthropic_environment``). Idempotent.
    Nothing under wif has a use for any ``ANTHROPIC_*`` variable: the federation ids come off a file, the base URL is pinned
    in-process, and the SDK is given its credentials explicitly. ``env`` is the test seam (a recording mapping)."""
    env = os.environ if env is None else env
    present = tuple(sorted(k for k in list(env) if isinstance(k, str) and k.startswith(WIF_SCRUB_PREFIX)))   # NAMES: __iter__ only
    for k in present:
        del env[k]
    return present


def pin_process_environment(env: Any = None) -> Tuple[str, ...]:
    """The ONE process-environment pin of a grader -- the per-PROCESS face of the base-URL rule whose per-client face is
    :func:`pinned_base_url` -- called before the first client of the process is built. Under ``op`` / ``env`` it is,
    byte for byte, what four graders used to do in their own copies: export the API host as
    ``ANTHROPIC_BASE_URL`` (a shell can tunnel SDK calls through a local gateway there) and drop the shell's
    ``ANTHROPIC_API_KEY`` (the ``highprio`` label's) and ``ANTHROPIC_AUTH_TOKEN`` (an OAuth bearer), so a KEYED client built anywhere in
    the process reaches the API with the org's OWN key. Under ``SISTEER_SECRETS_MODE=wif`` NOTHING is set or exported
    (nothing under wif reads, sets or exports ``ANTHROPIC_*``; the federated client
    pins the host in-process, :func:`anthropic_client_wif`) and every inherited ``ANTHROPIC_*`` name is dropped
    (:func:`scrub_inherited_anthropic_environment`: by NAME -- a key iteration and a delete, no value read, over the whole
    prefix rather than a fixed list of names). Returns the names dropped (never a value). ``env`` is the test seam (a
    recording mapping); the default is ``os.environ``."""
    env = os.environ if env is None else env
    if secrets_mode() == WIF_MODE:
        return scrub_inherited_anthropic_environment(env)
    env["ANTHROPIC_BASE_URL"] = ANTHROPIC_BASE_URL
    dropped = []
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if env.pop(k, None) is not None:
            dropped.append(k)
    return tuple(dropped)


def secret_sha12(item: str, *, env: Optional[Mapping[str, str]] = None, run=subprocess.run,
                 timeout: float = 60.0) -> Tuple[Optional[str], Optional[str]]:
    """sha256[:12] of a vault item WITHOUT the value entering this process: hashed in a shell PIPELINE (``<command> <item> | tr -d
    '\\n' | sha256sum``, pipefail) -- what the grading daemon holds a Petri drive's attested key hash against. The command is the
    secret-manager CLI that ``SISTEER_SECRET_CMD`` names in ``env`` (another process's environment, such as the one the daemon hands
    a drive), else in this process's environment when ``env`` is empty; a bare name is looked up on that environment's ``PATH``.
    Returns ``(sha12, None)`` or ``(None, reason)`` when the fetch fails (the setting unset, the command missing, a non-zero exit,
    an empty value, a timeout); never raises, never logs the value, and a reason names the setting rather than the command it
    holds. Lives here because this module is the ONE sanctioned caller of the secret-manager CLI
    (tests/test_no_credential_files.py); ``run`` is the subprocess seam for tests."""
    env = dict(env or {})
    src: Mapping[str, str] = env or os.environ
    try:
        cmd = secret_cmd(src)
    except VaultSecretError as exc:
        return None, str(exc)
    if os.sep not in cmd:
        cmd = shutil.which(cmd, path=src.get("PATH") or os.environ.get("PATH")) or cmd
    if not (os.path.isfile(cmd) and os.access(cmd, os.X_OK)):     # checked here, so no shell error carries the path it holds
        return None, f"{SECRET_CMD_ENV} does not name an executable (not found, or not on PATH)"
    try:
        r = run(["bash", "-c", 'set -o pipefail; "$1" "$2" | tr -d "\\n" | sha256sum', "bash", cmd, item], capture_output=True, text=True,
                timeout=timeout, env=env or None, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{SECRET_CMD_ENV} {item}: {type(exc).__name__}"
    digest = (r.stdout or "").split()[0] if (r.stdout or "").strip() else ""
    if r.returncode != 0 or len(digest) != 64 or digest == hashlib.sha256(b"").hexdigest():
        return None, f"{SECRET_CMD_ENV} {item} failed (rc {r.returncode}): {(r.stderr or '').strip()[-200:]}"
    return digest[:12], None


def openrouter_api_key(*, timeout: float = SECRET_CMD_TIMEOUT_S) -> str:
    """The OpenRouter API key from the environment or the vault (env mode: the
    environment only, required by name — never the secret-manager CLI)."""
    if secrets_mode() == "env":
        key = (os.environ.get(OPENROUTER_ENV) or "").strip()
        if not key:
            raise VaultSecretError(f"{SECRETS_MODE_ENV}=env: {OPENROUTER_ENV} must be set; never a file, never the secret-manager CLI")
        return key
    return resolve_secret(env=OPENROUTER_ENV, item=OPENROUTER_ITEM,
                          timeout=timeout)


def anthropic_client(org: str = "lowprio", *, max_retries: int = 5,
                     timeout: float = REQUEST_TIMEOUT_S,
                     base_url: Optional[str] = None,
                     vault_timeout: float = SECRET_CMD_TIMEOUT_S,
                     http_client: Any = None) -> Any:
    """An ``anthropic.Anthropic`` bound to ``org``'s key, with the base URL
    PINNED (default :data:`ANTHROPIC_BASE_URL`) and an explicit per-request
    ``timeout`` (default :data:`REQUEST_TIMEOUT_S`).

    All three halves matter. The key comes from the org's own env var or vault
    item, so the client cannot be built on another org's credential. The base
    URL is passed explicitly, so the SDK never reads ``ANTHROPIC_BASE_URL``
    from the shell — in some sessions that points at a local gateway, and
    every request hangs instead of reaching the API. And the request timeout is
    always set, because the SDK's 600s default turns a few hung sockets into a
    silently idle pool.

    Under ``SISTEER_SECRETS_MODE=wif`` the client is :func:`anthropic_client_wif`
    for the same org, retries and timeout — federated, keyless, stamped
    ``{"pool", "auth": "wif", "ids_sha12"}``, its destination the pin by
    construction (a ``base_url`` naming another host is refused by name; round
    12 on fd06740ae), the SDK's own retries off and replayed above the
    credential latch — and no key is resolved at all. op and env take the
    branch below exactly as before: the SDK client with the caller's
    ``max_retries``, an explicit ``base_url`` honored, no layer. ``http_client``
    (an httpx client; the test seam over a mock transport) is passed to the
    SDK only when given, in every mode.
    """
    if secrets_mode() == WIF_MODE:
        return anthropic_client_wif(org, max_retries=max_retries, timeout=timeout, base_url=base_url, http_client=http_client)
    import anthropic

    key = anthropic_api_key(org, timeout=vault_timeout)
    return _stamp(anthropic.Anthropic(
        api_key=key, base_url=base_url or ANTHROPIC_BASE_URL,
        max_retries=max_retries, timeout=timeout,
        **({"http_client": http_client} if http_client is not None else {})), org, key)


def anthropic_async_client(org: str = "lowprio", *, max_retries: int = 5,
                           timeout: float = REQUEST_TIMEOUT_S,
                           base_url: Optional[str] = None,
                           vault_timeout: float = SECRET_CMD_TIMEOUT_S,
                           http_client: Any = None) -> Any:
    """The ``anthropic.AsyncAnthropic`` twin of :func:`anthropic_client`: the
    same key resolution (the org's own env var, else its vault item, else
    :class:`PoolUnavailable`), the same pinned base URL, the same explicit
    request timeout, and the same pool stamp. Async callers build through
    this rather than keeping a private copy of the resolution — a private
    copy is where the last misroute lived. Under ``SISTEER_SECRETS_MODE=wif``
    the twin is :func:`anthropic_client_wif` with ``async_client=True`` (the
    same destination pin and latched retries).
    """
    if secrets_mode() == WIF_MODE:
        return anthropic_client_wif(org, max_retries=max_retries, timeout=timeout, base_url=base_url, async_client=True,
                                    http_client=http_client)
    import anthropic

    key = anthropic_api_key(org, timeout=vault_timeout)
    return _stamp(anthropic.AsyncAnthropic(
        api_key=key, base_url=base_url or ANTHROPIC_BASE_URL,
        max_retries=max_retries, timeout=timeout,
        **({"http_client": http_client} if http_client is not None else {})), org, key)
