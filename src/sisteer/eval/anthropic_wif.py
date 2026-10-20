"""sisteer.eval.anthropic_wif -- the Petri legs' inspect_ai Anthropic provider routed through workload identity federation.

THE ONE CREDENTIAL SEAM IS THE VAULT (``sisteer.vault``): under ``SISTEER_SECRETS_MODE=wif`` every Claude client of this repository
authenticates with ``WorkloadIdentityCredentials`` built by :func:`sisteer.vault.wif_credentials` over the org's ids read off the
workload-identity env file -- never an API key, never an ``ANTHROPIC_*`` read. The Petri legs do not build their own clients:
inspect_ai's Anthropic provider (``inspect_ai.model._providers.anthropic``) builds ``AsyncAnthropic(api_key=<env key>, ...)`` itself
inside ``AnthropicAPI.__init__`` and refuses to start without a key. This module (re-based on the vault) therefore

* REPLACES the provider's client factory for the plain Anthropic API (``AnthropicAPI._create_client``):
  the federated client is built DIRECTLY -- ``AsyncAnthropic(credentials=<the vault's>, base_url=<the pin>, **model_args)`` -- so the
  provider's own resolution never runs for it: its ``os.environ.get("ANTHROPIC_AUTH_TOKEN")`` (the Claude Code OAuth bearer branch),
  its ``model_base_url(..., "ANTHROPIC_BASE_URL")`` and its ``ANTHROPIC_API_KEY`` fallback are not consulted at all (before this the
  factory below received and DISCARDED those values: the client came out right, but an inherited bearer had been read and handed
  through a constructor under a mode that forbids any ``ANTHROPIC_*`` credential read). Bedrock / vertex / azure / foundry model
  names keep the provider's own factory and auth.
* rebinds the provider's ``AsyncAnthropic`` name to a factory that DROPS ``api_key`` / ``auth_token`` and passes ``credentials=`` --
  the backstop for any other construction path in the provider, and the carrier of the install's markers (org, pin): an explicit
  static key would shadow federation (the SDK warns and uses the key), which is why the sentinel below never reaches the SDK.
* PINS the client's base URL to the vault's (:data:`sisteer.vault.ANTHROPIC_BASE_URL`, the plain Anthropic API) -- the provider
  resolves its URL from ``ANTHROPIC_BASE_URL`` / ``INSPECT_EVAL_MODEL_BASE_URL`` (an inherited environment may point those at a local
  gateway), and a federated client must never carry federally authenticated judge traffic to an inherited endpoint; the
  provider's ``oauth-2025-04-20`` beta header (its ``ANTHROPIC_AUTH_TOKEN`` branch's) is stripped in the backstop factory too, since the
  provider reads its per-request betas off the client's default headers.
* hands the provider's constructor the sentinel key it insists on when the caller passed none, for the plain Anthropic API only
  (bedrock / vertex / azure / foundry model names keep their own auth) -- which is also what keeps ``ModelAPI.__init__``'s api-key
  override pass from reading ``ANTHROPIC_API_KEY`` off the environment (it reads the variable only when no key was passed).

* LATCHES the first credential rejection of the process (the WIF driver sweep): inspect retries RETRYABLE statuses
  only and runs its eval under ``fail_on_error=False``, so a 401 / 403 -- or the federation identity refused at the token exchange,
  the SDK's ``WorkloadIdentityError`` -- was ONE sample's hard error and the eval went on to the next scenario's own doomed request,
  one per launched scenario per leg process. Every federated client the factory builds wraps its ``messages`` resource
  (:class:`_LatchedMessages`): the first rejection any of them meets is re-raised AS the SDK raised it (inspect's own classification
  of that one call is unchanged: no retry, a hard error) and held in the process-wide latch; every later ``create`` / ``stream`` /
  ``count_tokens`` of every federated client raises the typed ``sisteer.vault.JudgeCredentialRejected`` BEFORE its wire call.
* ABORTS the eval at that first rejection: the latch alone left inspect walking every remaining scenario as an instant
  typed error under ``fail_on_error=False``, every later eval group launching, and the runner converting the logs and FILING the
  scenarios judged before the rejection as if the leg had ended well -- the leg's terminal line was then the INCOMPLETE / hard-error
  diagnosis ("fix the REQUEST"), not the credential. Now the first rejection also fires inspect's own task-cancel handle for every
  task running in the process (:func:`abort_inspect_tasks`: ``inspect_ai._control.cancel.cancel_task(..., action="cancel")``, the
  ``TaskCancel("abort")`` the display's cancel dialog and ``inspect ctl`` fire): the in-flight samples are interrupted, the queued
  ones never start, the completed ones stay in the log, the log is finalised with an error status noting the cancel, and an eval_set
  does not retry an aborted task. ``petri_runner._inspect_driver`` then reads :func:`credential_rejected` after the eval_set, launches
  no later group, converts no log and raises the typed error, so the leg files no row, its ledger names the credential and its
  process ends with the typed line; a re-drive after the fix resumes from the samples the logs kept (inspect reuses completed
  samples of an errored log). :func:`uninstall_inspect_wif` clears the latch and the abort record with the install.
* REFUSES BY NAME an EXPLICIT static credential header (:func:`sisteer.vault.refuse_static_auth_headers`):
  a ``default_headers`` / ``http_client`` model argument of ``get_model`` carrying ``X-Api-Key`` / ``Authorization`` (either letter
  case), in both factories before any client is built, and a request's ``extra_headers`` carrying one, before the wire -- the SDK
  keeps such a header beside ``credentials=`` (``api_key`` None, the credentials set) and its ``AccessTokenAuth`` then skips the
  token exchange, so the leg would send the static credential in place of the exchanged bearer and stop on the 401 it earns. Round
  10's scrub covers the INHERITED face of the same SDK behaviour (``ANTHROPIC_CUSTOM_HEADERS``); this is the caller's-argument face.
* PINS THE DESTINATION BY CONSTRUCTION and REPLAYS THE SDK'S RETRIES ABOVE THE LATCH: the install's
  ``base_url`` may only spell the pinned API host (:func:`sisteer.vault.pin_federated_destination`: any other host is refused by name
  before anything is installed -- it used to be a seam that moved every federated client's bearer to the host it named), an
  ``http_client`` model argument whose own base URL names another host is refused in both factories, and every federated client is
  built with the SDK's own retries OFF (``max_retries=0``) and the count asked for carried by the vault's latched retry layer
  (:func:`sisteer.vault.federate_sdk_retries`: the same statuses, counts and delays as the SDK's, the install's latch checked before
  every attempt), so a sample's request asleep in a retry when another sample's met the first 401 / 403 wakes into the stop, never
  into another request (the SDK's internal retry slept and re-sent below :class:`_LatchedMessages`).
* NEVER HANDS THE SDK A CALLER'S HTTPX CLIENT: an ``http_client`` model argument's base URL says nothing
  about where its request hooks, mounts, a subclass's ``send`` or a transport of its own would carry the exchanged bearer (measured: a
  request hook re-pointed a federated request, bearer and all, to another host through round 12's check). Both factories build every
  federated client on the vault's OWN httpx client (:func:`sisteer.vault.federated_http_client`: no hook, no mount, no auth, no
  environment proxy, the SDK's defaults, and its one transport the pin guard :class:`sisteer.vault.PinnedTransport` over httpx's HTTP
  transport, refusing by name any request not addressed to the pin at the last step before the wire -- an SDK middleware model
  argument that rewrites the URL, or a cross-host redirect, is refused there too); an ``http_client`` model argument is accepted only
  as a bare carrier of a ``MockTransport`` whose transport alone is used (:func:`sisteer.vault.transport_seam`), anything more about
  it refused by name. The vault's latched layer under :class:`_LatchedMessages` refuses a static per-request ``extra_headers``
  credential on every door too, so the round-11 check here is no longer the only one.
* BUILDS THE VAULT'S FEDERATED CLASS, WHOSE COPIES STAY FEDERATED: both factories build
  :func:`sisteer.vault.federated_sdk_class` over the provider's class, so a ``copy`` / ``with_options`` of a provider client is
  :func:`sisteer.vault.federated_copy` -- the same credentials, token cache and pinned httpx client, the SDK's retries off, the install's
  latch (the one object), the layer and this module's wrapper (the dress the factory leaves on every client under
  :data:`sisteer.vault.FEDERATED_DRESS_ATTR`), an ``http_client`` / another base URL / a credential change / a middleware refused by
  name -- where the SDK's own copy inherited the credentials, took a caller's httpx client and carried the bearer wherever it sent it,
  with no latch and no layer.

Idempotent (a second install replaces the first, never stacks); :func:`uninstall_inspect_wif` restores the originals. The install
must run BEFORE the first ``get_model("anthropic/...")`` of the process -- the provider builds its client in its constructor -- which
is why ``sisteer.capeval.tiers.petri_runner._inspect_driver`` calls :func:`install_wif_if_requested` first thing. THE MODE IS
AUTHORITATIVE (:func:`leg_org`): under ``SISTEER_SECRETS_MODE=wif`` the leg REQUIRES the drive's routing switch
``SISTEER_ANTHROPIC_AUTH=wif:<org>`` and refuses by name without it -- a leg run outside petri_drive_full22.sh (a direct
``scripts/petri_run.py``, a resumed leg) would otherwise fall through to the provider's ``ANTHROPIC_API_KEY`` read while the mode said
federation; under ``op`` / ``env`` a present switch is a contradiction and refuses too. The org is the vault's spelling (``lowprio`` /
``highprio`` / ``batch``); the ids-file stems ``loprio`` / ``hiprio`` are accepted as aliases of the same two pools.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, Mapping, Optional

from sisteer import vault

#: the provider's api-key requirement is met with this sentinel, which the federated factory discards before the SDK sees it
WIF_SENTINEL_KEY = "sisteer-wif-federated-no-key"
#: the ids-file stems as org aliases (a drive may spell the org `loprio`); the vault's names pass through unchanged
ORG_ALIASES: Dict[str, str] = {"loprio": "lowprio", "hiprio": "highprio"}
#: the leg's switch: ``wif:<org>`` exported by petri_drive_full22.sh under SISTEER_SECRETS_MODE=wif; unset = the key path
WIF_AUTH_ENV = "SISTEER_ANTHROPIC_AUTH"
#: the process-wide credential latch of the federated provider clients (module docstring): set by the install, cleared by the
#: uninstall; read by every :class:`_LatchedMessages` the factory wraps
_LATCH: Optional[vault.CredentialLatch] = None


def credential_rejected() -> Optional[vault.JudgeCredentialRejected]:
    """The first credential rejection a federated provider client of this process met, latched, else None."""
    return _LATCH.held if _LATCH is not None else None


#: the first-rejection ABORT of the running inspect tasks (module docstring): what :func:`abort_inspect_tasks` did when
#: the latch took the process's first rejection -- the task ids aborted, those skipped (already finished / already cancelling) and
#: any error of the control call -- for the driver's one CREDENTIAL REJECTED line and the tests; None until then, cleared by the
#: uninstall. ``_ABORT_LATCH`` is the latch the abort was fired for, so it fires ONCE per install.
_ABORT: Optional[Dict[str, Any]] = None
_ABORT_LATCH: Optional[vault.CredentialLatch] = None
_ABORT_LOCK = threading.Lock()


def inspect_abort_result() -> Optional[Dict[str, Any]]:
    """What the first rejection's abort of the running inspect tasks did (:func:`abort_inspect_tasks`), else None."""
    return _ABORT


def abort_inspect_tasks(reason: str = "") -> Dict[str, Any]:
    """Abort EVERY inspect task running in this process through inspect's own task-cancel handle --
    ``inspect_ai._control.cancel.cancel_task(task_id, action="cancel")``, the ``TaskCancel("abort")`` the in-process display's cancel
    dialog and ``inspect ctl task cancel`` fire: the in-flight samples are interrupted (their transcripts kept in the log as cancelled
    samples), the queued ones never start, the completed ones are kept, the log is finalised with an error status noting the cancel,
    and an eval_set does NOT retry an aborted task (inspect_ai 0.3.251, ``_eval/run.py``: "an abort or external cancellation is never
    retried"). Best effort and never raises -- the credential error that caused it is the one the caller re-raises. Returns
    ``{"aborted": [task ids], "skipped": [...], "errors": [...], "reason": reason}``."""
    out: Dict[str, Any] = {"aborted": [], "skipped": [], "errors": [], "reason": str(reason)}
    try:
        from inspect_ai._control.cancel import cancel_task
        from inspect_ai._control.eval_state import get_eval_states, latest_eval_for_task
    except Exception as e:  # noqa: BLE001 -- inspect absent, or its control API moved: nothing to abort, said so
        out["errors"].append(f"inspect control API unavailable: {type(e).__name__}: {e}")
        return out
    seen = set()
    for state in get_eval_states():
        tid = str(getattr(state, "task_id", "") or "")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        try:
            latest = latest_eval_for_task(tid) or state
            if getattr(latest, "completed_at", None) is not None or getattr(latest, "task_cancel", None) is None:
                out["skipped"].append(f"{tid}: not running")
                continue
            res = cancel_task(tid, action="cancel")
            if isinstance(res, dict) and res.get("ok") and res.get("changed"):
                out["aborted"].append(tid)
            else:
                why = (res.get("error") or res.get("reason") or "no change") if isinstance(res, dict) else "not in this process"
                out["skipped"].append(f"{tid}: {why}")
        except Exception as e:  # noqa: BLE001 -- one task's failed cancel never blocks the others, nor the error being raised
            out["errors"].append(f"{tid}: {type(e).__name__}: {e}")
    return out


def _abort_once(latch: vault.CredentialLatch, held: vault.JudgeCredentialRejected) -> None:
    """The process's first rejection (the one that set ``latch``): abort the running inspect tasks ONCE and say so."""
    global _ABORT, _ABORT_LATCH
    with _ABORT_LOCK:
        if _ABORT_LATCH is latch:
            return
        _ABORT_LATCH = latch
        result = _ABORT = abort_inspect_tasks(str(held))
    print(f"[anthropic_wif] CREDENTIAL REJECTED: {held} -- aborting the running inspect task(s) through inspect's own cancel handle: "
          f"{len(result['aborted'])} aborted, {len(result['skipped'])} skipped, {len(result['errors'])} error(s); the in-flight samples "
          f"are cancelled, the queued ones never start, the completed ones stay in the log; nothing is filed (D6)", flush=True)


class _LatchedMessages:
    """The SDK's ``messages`` resource of a federated provider client (since round 12 the vault's latched retry layer over it) with
    the credential stop around its three entry points (the ones inspect's provider calls: ``create``, ``stream``, ``count_tokens``):
    the latch checked BEFORE each call -- held means the typed error without a request -- and every failure classified AFTER it
    (``vault.credential_rejection``): a rejection met on the wire is held for the process (the first fires the inspect abort, once)
    and re-raised AS the SDK raised it (inspect classifies that one call as it always did); every later call of every federated
    client is the held typed error before the wire. Every other attribute forwards to the resource (``batches``, ``with_raw_response``, ...)."""

    def __init__(self, inner: Any, latch: vault.CredentialLatch):
        self._inner = inner
        self._latch = latch

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_inner"], name)

    def _before(self, kwargs: Mapping[str, Any]) -> None:
        """Before every request: an explicit static credential header on THIS request (``extra_headers``: inspect's
        ``GenerateConfig.extra_headers`` reaches the SDK call as such) is refused by name -- the SDK would send it in place
        of the exchanged bearer (the constructor-time faces are the two factories') -- then the
        latch."""
        vault.refuse_static_auth_headers(extra_headers=kwargs.get("extra_headers"),
                                         subject="a request of the inspect Anthropic provider's federated client")
        self._latch.check()

    def _after(self, exc: BaseException) -> None:
        held = self._latch.rejection(exc)
        if held is not None:
            _abort_once(self._latch, held)           # the process's FIRST rejection: the running inspect tasks are aborted, once (a no-op after)
        raise exc                                    # a rejection as the SDK raised it, now held (every later request stops typed at _before); or not one
        # (round 12 on fd06740ae: the vault's latched retry layer under this wrapper holds a rejection before it reaches here, so
        # "was the latch empty before" no longer tells the first rejection from a concurrent one; both are re-raised as the SDK
        # raised them, and the typed error comes from the check before the next request -- the abort is idempotent per latch)

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        self._before(kwargs)
        try:
            return await self._inner.create(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- classified
            self._after(e)

    async def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        self._before(kwargs)
        try:
            return await self._inner.count_tokens(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- classified
            self._after(e)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        self._before(kwargs)
        return _LatchedStream(self._inner.stream(*args, **kwargs), self)


class _LatchedStream:
    """The stream manager of :meth:`_LatchedMessages.stream`: the request goes out at ``__aenter__``, which is where the
    rejection surfaces and is classified; the entered stream is the SDK's own."""

    def __init__(self, inner: Any, owner: _LatchedMessages):
        self._inner = inner
        self._owner = owner

    async def __aenter__(self) -> Any:
        self._owner._latch.check()
        try:
            return await self._inner.__aenter__()
        except Exception as e:  # noqa: BLE001 -- classified
            self._owner._after(e)

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_inner"], name)


def org_name(org: str) -> str:
    """The vault's spelling of an org (``loprio`` -> ``lowprio``, ``hiprio`` -> ``highprio``); an unknown org is the vault's
    ``ValueError``, never a guess."""
    name = ORG_ALIASES.get(str(org).strip().lower(), str(org).strip().lower())
    if name not in vault.WIF_ORG_FILES:
        raise ValueError(f"unknown Claude API org {org!r} (one of {sorted(vault.WIF_ORG_FILES)}; aliases {sorted(ORG_ALIASES)})")
    return name


def parse_auth_switch(raw: Optional[str]) -> Optional[str]:
    """``wif:<org>`` -> the vault's org name; None / empty -> None (the key path); anything else refuses by name (a leg must never
    fall through to a key it was not meant to use)."""
    value = (raw or "").strip()
    if not value:
        return None
    if not value.startswith("wif:") or len(value) <= 4:
        raise ValueError(f"{WIF_AUTH_ENV}={value!r}: expected wif:<org> (lowprio | highprio | batch)")
    return org_name(value.split(":", 1)[1])


def leg_org(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The org a Petri leg federates as, with the VAULT'S MODE authoritative (round 1 on 7778694d0): under ``SISTEER_SECRETS_MODE=wif``
    the routing switch :data:`WIF_AUTH_ENV` (``wif:<org>``, exported by petri_drive_full22.sh from F22_AUDITOR_ORG) is REQUIRED and its
    org is returned -- absent, the leg refuses by name rather than let inspect's provider read ``ANTHROPIC_API_KEY`` (or fail for want
    of one) while the mode selected federation; under ``op`` / ``env`` the switch must be ABSENT (a present one contradicts the mode:
    refused by name) and None is returned (the key path, byte for byte as before). A malformed switch or an unknown mode refuses with the
    vault's / the parser's words. ``ValueError`` throughout, for the runner to type as its own."""
    e = os.environ if env is None else env
    mode = (e.get(vault.SECRETS_MODE_ENV) or "op").strip() or "op"
    if mode not in vault.SECRETS_MODES:
        raise ValueError(f"{vault.SECRETS_MODE_ENV}={mode!r}: must be one of {', '.join(repr(m) for m in vault.SECRETS_MODES)}")
    raw = e.get(WIF_AUTH_ENV)
    org = parse_auth_switch(raw)
    if mode == vault.WIF_MODE:
        if org is None:
            raise ValueError(f"{vault.SECRETS_MODE_ENV}={vault.WIF_MODE} but {WIF_AUTH_ENV} is unset: the leg's inspect Anthropic provider would "
                             f"fall through to its own ANTHROPIC_API_KEY read (or fail for want of one) while the mode selected federation; "
                             f"export {WIF_AUTH_ENV}=wif:<org> (lowprio | highprio | batch; petri_drive_full22.sh exports it from F22_AUDITOR_ORG) "
                             f"-- refusing before the first get_model")
        return org
    if org is not None:
        raise ValueError(f"{WIF_AUTH_ENV}={str(raw).strip()!r} under {vault.SECRETS_MODE_ENV}={mode}: federation is the wif mode's; a routing switch "
                         f"under the key modes is a contradiction (which credential did the caller mean?) -- unset one of them")
    return None


def _provider_module():
    from inspect_ai.model._providers import anthropic as prov
    return prov


def _plain_service(model_name: str) -> bool:
    """True for the plain Anthropic API (``claude-...``); bedrock / vertex / azure / foundry model names keep their own auth."""
    parts = str(model_name).split("/")
    return len(parts) < 2 or parts[0] not in ("bedrock", "vertex", "azure", "foundry")


def install_inspect_wif(org: str = "lowprio", *, token_provider: Optional[Callable[[], str]] = None, env_path: Optional[str] = None,
                        base_url: Optional[str] = None) -> Any:
    """Route every inspect_ai Anthropic provider client built AFTER this call through federation for ``org`` (the vault's credentials:
    :func:`sisteer.vault.wif_credentials`; a Modal container refuses by name there), against the PINNED base URL
    (:data:`sisteer.vault.ANTHROPIC_BASE_URL` -- exactly :func:`sisteer.vault.anthropic_client_wif`'s pin: the provider's own URL, resolved
    from ``ANTHROPIC_BASE_URL`` / ``INSPECT_EVAL_MODEL_BASE_URL`` or a model argument, is discarded, so an inherited gateway never receives
    federated traffic; ``base_url`` may only spell that pin -- any other host is :class:`sisteer.vault.BaseUrlRefused` BY NAME before
    anything is installed, and an ``http_client`` model argument whose own base URL names another host is refused in both factories:
    :func:`sisteer.vault.pin_federated_destination`). Both factories run the vault's scrub
    (:func:`sisteer.vault.scrub_inherited_anthropic_environment`) before the SDK's constructor: it reads ``ANTHROPIC_CUSTOM_HEADERS``
    whatever credential it is given, and a static ``X-Api-Key`` / ``Authorization`` there would ride every request and skip the token
    exchange -- and both REFUSE BY NAME an explicit ``default_headers`` / ``http_client`` carrying either (the SDK keeps a caller's static
    header beside ``credentials=`` just the same). Both build the SDK client with its own retries OFF and replay them above the install's
    latch (:func:`sisteer.vault.federate_sdk_retries` with the ``max_retries`` model argument, else the SDK's default, as the budget --
    inspect's own retry layer sits above this module's latched wrapper, so no layer re-sends below the latch). Both build the SDK client
    on the VAULT'S OWN httpx client over the pin guard (:func:`sisteer.vault.federated_http_client`; an ``http_client`` model argument is
    a bare mock-transport donor or refused by name), so the destination is enforced at the transport whatever a model argument or a
    redirect would re-point. Returns the credentials. Idempotent: a second install replaces the first, never stacks."""
    global _LATCH, _ABORT, _ABORT_LATCH
    name = org_name(org)
    # round 12 on fd06740ae: the destination is the pin BY CONSTRUCTION -- an install asked for another host is refused by name
    # BEFORE anything is read or installed (the install's `base_url=` used to be a test seam that moved every federated client's
    # bearer to whatever host it named; a test injects an `http_client` over a mock transport through get_model's model args instead)
    pinned_url = vault.pin_federated_destination(base_url, subject=f"the inspect Anthropic provider's federated clients for {name!r} "
                                                                   "(install_inspect_wif)")
    creds = vault.wif_credentials(name, token_provider=token_provider, env_path=env_path)
    prov = _provider_module()
    real_cls = getattr(prov.AsyncAnthropic, "__sisteer_wif_real__", None) or prov.AsyncAnthropic
    fed_cls = vault.federated_sdk_class(real_cls)       # round 14: the vault's federated subclass of the provider's class -- its copies stay federated
    latch = _LATCH = vault.CredentialLatch(name)        # one per install: a re-install (a new org) starts a fresh latch
    with _ABORT_LOCK:                                   # ... and a fresh abort record (a stand-alone latch's abort never outlives it)
        _ABORT = None
        _ABORT_LATCH = None

    def _sdk_default_retries() -> int:
        import anthropic
        return int(getattr(anthropic, "DEFAULT_MAX_RETRIES", 2))

    def _federated(client, requested_retries: int):
        """The two stops every federated provider client carries (round 12): the install's process-wide latch attached to the
        client (so the vault's retry layer reads THIS latch, never a second one), the SDK's own retries replayed above it
        (:func:`sisteer.vault.federate_sdk_retries`: the client was built with ``max_retries=0``; ``requested_retries`` -- the
        model argument, else the SDK's default -- is the layer's budget), then this module's latched wrapper over that."""
        vault.attach_credential_latch(client, latch)
        vault.federate_sdk_retries(client, max_retries=requested_retries)
        client.messages = _LatchedMessages(client.messages, latch)     # the credential stop around every request of this client
        # round 14: a copy of this client (vault.federated_copy) is dressed the same way -- the install's latch, the layer, this wrapper
        setattr(client, vault.FEDERATED_DRESS_ATTR, lambda copy, parent, budget: _federated(copy, budget))
        return client

    def federated_async_anthropic(*args, **kwargs):
        kwargs.pop("api_key", None)                    # the sentinel (or an inherited key): a static credential would shadow federation
        kwargs.pop("auth_token", None)                 # the provider's ANTHROPIC_AUTH_TOKEN branch (Claude Code OAuth tokens): not WIF
        kwargs["base_url"] = pinned_url                # the provider's env-derived URL (ANTHROPIC_BASE_URL -- a local gateway) never wins
        # round 12: an explicit http_client whose own base URL names another host is refused by name (the model-args door)
        vault.pin_federated_destination(http_client=kwargs.get("http_client"),
                                        subject="the inspect Anthropic provider's federated client (the AsyncAnthropic factory)")
        headers = kwargs.get("default_headers")
        if isinstance(headers, dict) and headers.get("anthropic-beta") == "oauth-2025-04-20":
            headers = {k: v for k, v in headers.items() if k != "anthropic-beta"}      # the OAuth beta is for Claude Code tokens, not WIF
            kwargs["default_headers"] = headers or None
        # round 11: an EXPLICIT static credential header -- a caller's `default_headers` / `http_client` (a model argument of
        # get_model) -- is refused by name: the SDK keeps it beside credentials= and its AccessTokenAuth then skips the exchange
        vault.refuse_static_auth_headers(default_headers=kwargs.get("default_headers"), http_client=kwargs.get("http_client"),
                                         subject="the inspect Anthropic provider's federated client (the AsyncAnthropic factory)")
        # round 13: the SDK is never handed a caller's httpx client -- the vault's own, over the pin guard, with a caller's bare
        # mock-transport client accepted as the offline seam's transport donor and anything more about it refused by name
        kwargs["http_client"] = vault.federated_http_client(kwargs.get("http_client"), async_client=True, timeout=kwargs.get("timeout"),
                                                            subject="the inspect Anthropic provider's federated client (the AsyncAnthropic factory)")
        requested = kwargs.pop("max_retries", None)    # round 12: the SDK's own retries OFF; the count asked for is the latched layer's budget
        vault.scrub_inherited_anthropic_environment()  # round 10: the SDK's constructor reads ANTHROPIC_CUSTOM_HEADERS whatever it is given
        client = fed_cls(*args, credentials=creds, max_retries=0, **kwargs)
        return _federated(client, _sdk_default_retries() if requested is None else int(requested))

    federated_async_anthropic.__sisteer_wif_real__ = real_cls      # type: ignore[attr-defined]
    federated_async_anthropic.__sisteer_wif_org__ = name           # type: ignore[attr-defined]
    federated_async_anthropic.__sisteer_wif_base_url__ = pinned_url  # type: ignore[attr-defined]
    prov.AsyncAnthropic = federated_async_anthropic

    # THE PROVIDER'S OWN CLIENT FACTORY IS BYPASSED for the plain API: `AnthropicAPI._create_client` reads
    # ANTHROPIC_AUTH_TOKEN (its OAuth bearer branch) and ANTHROPIC_BASE_URL (model_base_url) before it reaches the factory above,
    # which then discarded them -- a read of an inherited credential under a mode that forbids one. The federated client is built
    # here directly from the vault's credentials and the pin; the other services keep the provider's factory and their own auth.
    real_create = getattr(prov.AnthropicAPI._create_client, "__sisteer_wif_real__", None) or prov.AnthropicAPI._create_client

    def create_client_federated(self):
        if not _plain_service(getattr(self, "model_name", "")):
            return real_create(self)                 # bedrock / vertex / azure / foundry: the provider's own resolution
        extra = {k: v for k, v in dict(getattr(self, "model_args", None) or {}).items()
                 if k not in ("api_key", "auth_token", "credentials", "config", "profile", "base_url")}
        subject = f"the inspect Anthropic provider's federated client for {getattr(self, 'model_name', '?')!r} (model_args)"
        # round 12: an explicit http_client model argument whose own base URL names another host is refused by name
        vault.pin_federated_destination(http_client=extra.get("http_client"), subject=subject)
        # round 11: the model arguments reach the SDK's constructor as they are -- an explicit `default_headers` / `http_client`
        # carrying X-Api-Key / Authorization would be kept beside credentials= and sent in place of the exchanged bearer
        vault.refuse_static_auth_headers(default_headers=extra.get("default_headers"), http_client=extra.get("http_client"),
                                         subject=subject)
        # round 13: the vault's own httpx client over the pin guard; an http_client model argument is a transport donor at most
        extra["http_client"] = vault.federated_http_client(extra.get("http_client"), async_client=True, timeout=extra.get("timeout"),
                                                           subject=subject)
        requested = extra.pop("max_retries", None)     # round 12: the SDK's own retries OFF; the count asked for is the latched layer's budget
        vault.scrub_inherited_anthropic_environment()  # round 10: an inherited static header must not ride the federated client
        client = fed_cls(credentials=creds, base_url=pinned_url, max_retries=0, **extra)
        return _federated(client, _sdk_default_retries() if requested is None else int(requested))

    create_client_federated.__sisteer_wif_real__ = real_create      # type: ignore[attr-defined]
    prov.AnthropicAPI._create_client = create_client_federated

    real_init = getattr(prov.AnthropicAPI.__init__, "__sisteer_wif_real__", None) or prov.AnthropicAPI.__init__

    def init_with_sentinel(self, model_name, base_url=None, api_key=None, *args, **kwargs):
        if api_key is None and _plain_service(model_name):
            api_key = WIF_SENTINEL_KEY                 # the provider refuses to build without one; the factory above discards it
        return real_init(self, model_name, base_url, api_key, *args, **kwargs)

    init_with_sentinel.__sisteer_wif_real__ = real_init  # type: ignore[attr-defined]
    prov.AnthropicAPI.__init__ = init_with_sentinel
    return creds


def uninstall_inspect_wif() -> bool:
    """Restore the provider's own ``AsyncAnthropic``, client factory and constructor, and clear the credential latch and the abort
    record; True when something was installed."""
    global _LATCH, _ABORT, _ABORT_LATCH
    _LATCH = None
    with _ABORT_LOCK:
        _ABORT = None
        _ABORT_LATCH = None
    prov = _provider_module()
    changed = False
    real_cls = getattr(prov.AsyncAnthropic, "__sisteer_wif_real__", None)
    if real_cls is not None:
        prov.AsyncAnthropic = real_cls; changed = True
    real_create = getattr(prov.AnthropicAPI._create_client, "__sisteer_wif_real__", None)
    if real_create is not None:
        prov.AnthropicAPI._create_client = real_create; changed = True
    real_init = getattr(prov.AnthropicAPI.__init__, "__sisteer_wif_real__", None)
    if real_init is not None:
        prov.AnthropicAPI.__init__ = real_init; changed = True
    return changed


def inspect_wif_org() -> Optional[str]:
    """The org the inspect provider is currently routed to by :func:`install_inspect_wif`, else None."""
    try:
        return getattr(_provider_module().AsyncAnthropic, "__sisteer_wif_org__", None)
    except Exception:
        return None


def inspect_wif_base_url() -> Optional[str]:
    """The base URL every federated provider client is pinned to by :func:`install_inspect_wif`, else None."""
    try:
        return getattr(_provider_module().AsyncAnthropic, "__sisteer_wif_base_url__", None)
    except Exception:
        return None
