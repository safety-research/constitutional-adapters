"""The ONLY Anthropic client constructor in this codebase.

THREE SECRETS MODES, never a fourth (``SISTEER_SECRETS_MODE``, the same switch
the surrounding front door reads — ``reproduce/lib/secrets.sh`` and
``sisteer.vault`` read it too):

* ``op`` (the default): the key comes from the site's secret-manager CLI,
  ``$SISTEER_SECRET_CMD <item>`` (no default command: unset is a refusal by
  name) — never the environment, never a file. In this mode ``ANTHROPIC_API_KEY`` in the env is the ``highprio``
  label's variable, which this generator refuses, and every credential label keeps its OWN key
  (guard 3 below).
* ``env`` (everyone else): ONE documented variable, ``ANTHROPIC_API_KEY``,
  serves EVERY credential label — batch, smoke and live all resolve to it, it is required
  by name, the secret-manager CLI is never called and no file is ever read. The
  per-label refusal is not applicable: a single key IS the contract.
* ``wif`` (a GCP VM whose attached service account is federated with the
  Anthropic API): there is NO API key. Each kind's credential label
  authenticates through the SDK's ``WorkloadIdentityCredentials`` under the
  label's federation ids read off ``~/.anthropic/wif-<stem>.env`` (``KEY=VALUE``
  lines: the ``batch`` label's workload-identity env file for ``kind="batch"``, the
  ``lowprio`` label's for ``smoke`` / ``live`` — an empty or incomplete file
  refuses by name, never a fallback to another file; ``SISTEER_WIF_ENV_DIR``
  overrides the directory),
  the identity token fetched from the VM's metadata server
  (``google.oauth2.id_token``) at each exchange. The ids are identifiers, not
  secrets; nothing reads, sets or exports ``ANTHROPIC_*``; the secret manager is
  never consulted; :func:`read_key` refuses by name. The file format and the
  label -> file map are ``sisteer.vault``'s (which this package does not import:
  it is a workspace member of its own). The mode was admitted by the front door
  for every stage while this module still refused it, so a wif site could not
  run stage 01's generation at all.

Three guards, all mandatory in ``op`` mode because that machine's inherited
environment is hostile to correct routing (guard 2 holds in every mode;
guard 3 under ``wif`` compares IDENTITIES):

1. Key comes from the site's secret-manager CLI (``$SISTEER_SECRET_CMD``) —
   never the environment and never a file (three secrets modes and never a fourth, no
   key in a file). Tests inject fakes through the ``runner=`` seam (``op``)
   and the ``token_provider=`` seam (``wif``).
2. ``base_url`` is pinned to the real API. ``ANTHROPIC_BASE_URL`` in the env
   may point at a local gateway that silently swallows SDK calls.
3. The resolved key is compared against the highprio vault item's value and
   construction REFUSES on a match, whatever path delivered it. Under ``wif``
   the kind's federation ids are compared against the ``highprio`` label's ids
   file when that file is present and complete, and construction refuses on a
   match the same way.

``kind="batch"`` -> the ``batch`` credential's key (bulk generation, Batch API only).
``kind="smoke"`` -> the lowprio key (<=5-request live smoke tests only).
``kind="live"``  -> the lowprio key, async client, for the live fixup
campaigns only (live.py): batch discipline does not apply to those two
campaigns, and they run on lowprio, never highprio.
Under ``wif`` the same kinds name the same credential labels, keyless.

max_retries=4 covers poll/harvest/list plumbing. Batch ``create`` calls must
run with ``client.with_options(max_retries=0)`` — transparent SDK retries on
create are exactly how a double-submit happens (enforced in batch.py, asserted
by tests). Live runner clients are built with max_retries=0 too: live.py owns
its retry loop deliberately, so a 429 can throttle the limiter DOWN instead of
the SDK silently re-firing at full width.

Under ``wif`` the SDK's OWN retries are OFF on every client: the installed SDK
re-sends the FIRST 401 of a federated client with a freshly exchanged bearer,
and a sibling engine's poll asleep in an SDK backoff re-sent after the first
rejection -- so the count asked for is carried by the member's latched retry
layer (:func:`federate_sdk_retries`: the SDK's own policy replayed above the
client's credential latch), the clients are the member's own subclass of the
SDK's whose ``with_options`` / ``copy`` stay federated
(:func:`federated_sdk_class`; the ``max_retries=0`` creator shares its parent's
credentials, token cache, transport and latch), and ``op`` / ``env`` are byte
for byte as above.

And under ``wif`` the DESTINATION IS ENFORCED WHERE THE BYTES LEAVE and a
STATIC PER-REQUEST CREDENTIAL HEADER IS REFUSED ON EVERY DOOR: every federated
client is built on the member's OWN httpx2
client whose one transport is the pin guard (:func:`federated_http_client` /
:class:`PinnedTransport`: a request addressed anywhere but the API host --
the installed SDK's ``batches.results`` takes its destination off the
retrieve response's ``results_url``, and a cross-host redirect would be
followed -- is :class:`DestinationRefused` by name before any byte leaves,
re-raised typed by the latched layer and never retried), and the layer
refuses a static ``X-Api-Key`` / ``Authorization`` in a request's
``extra_headers`` before every attempt of every door
(:func:`refuse_static_auth_headers`; the SDK would send it in place of the
exchanged bearer and never exchange). The vault's rules, carried
as twins here and pinned equal to the vault's by test.
"""

from __future__ import annotations

import email.utils
import hashlib
import os
import random
import shutil
import socket
import ssl
import subprocess
import sys
import time
from typing import Any, Callable, Literal, NamedTuple

import anthropic
import certifi
import httpx2

API_BASE_URL = "https://api.anthropic.com"

# One secret-manager item per kind (one item per credential label: batch /
# lowprio / highprio). Read at use time through the secret-manager CLI;
# nothing is cached to disk. The item NAMES follow the front door's config keys
# (SISTEER_OP_ITEM_ANTHROPIC_{BATCH,LOWPRIO,HIGHPRIO}) so a site that names its
# vault items differently needs no code change.
VAULT_ITEMS = {
    "batch": os.environ.get("SISTEER_OP_ITEM_ANTHROPIC_BATCH", "batch"),
    "smoke": os.environ.get("SISTEER_OP_ITEM_ANTHROPIC_LOWPRIO", "lowprio"),
    "live": os.environ.get("SISTEER_OP_ITEM_ANTHROPIC_LOWPRIO", "lowprio"),
}
HIGHPRIO_VAULT_ITEM = os.environ.get("SISTEER_OP_ITEM_ANTHROPIC_HIGHPRIO", "highprio")
#: The site setting that names the secret-manager CLI op mode runs (``<command> <item>`` prints that item's value). There is
#: no default command, so op mode with the setting unset refuses by name (:func:`secret_cmd`); the front door
#: (``reproduce/lib/secrets.sh``) and ``sisteer.vault`` read the same setting.
SECRET_CMD_ENV = "SISTEER_SECRET_CMD"

#: The secrets-mode switch and the ONE variable env mode reads.
SECRETS_MODE_ENV = "SISTEER_SECRETS_MODE"
ENV_MODE_KEY_VAR = "ANTHROPIC_API_KEY"
WIF_MODE = "wif"
SECRETS_MODES = ("op", "env", WIF_MODE)

#: wif mode (module docstring): the label's federation ids live in ``$SISTEER_WIF_ENV_DIR`` (default ``~/.anthropic``) as
#: ``wif-<stem>.env`` — the ``batch`` label's file for ``batch``, the ``lowprio`` label's for ``smoke`` / ``live``, the stems being
#: ``sisteer.vault.WIF_ORG_FILES``'s (pinned equal by test; this member imports no sisteer); an empty or incomplete file
#: refuses by name; the ``highprio`` label's file is what guard 3 compares against.
#: The four ids a file must carry (the GCP federation rule) and the audience the identity token is minted for.
WIF_ENV_DIR_ENV = "SISTEER_WIF_ENV_DIR"
WIF_ENV_DIR_DEFAULT = "~/.anthropic"
WIF_POOLS = {"batch": "batch", "smoke": "lowprio", "live": "lowprio"}
WIF_POOL_STEMS = {"lowprio": "loprio", "highprio": "hiprio", "batch": "batch"}
WIF_FILES = {kind: f"wif-{WIF_POOL_STEMS[pool]}.env" for kind, pool in WIF_POOLS.items()}
HIGHPRIO_WIF_FILE = f"wif-{WIF_POOL_STEMS['highprio']}.env"
WIF_REQUIRED_KEYS = ("ANTHROPIC_FEDERATION_RULE_ID_GCP", "ANTHROPIC_ORGANIZATION_ID",
                     "ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID")
ANTHROPIC_WIF_AUDIENCE = API_BASE_URL
#: Modal's container marker (``sisteer.vault.MODAL_MARKER_ENV``, the name the registry's substrate ladder reads too): the GCP
#: identity-token path (:func:`google_identity_token`, the ``_GCP`` federation rule) does not exist in a Modal container -- its
#: identity is a sandbox token under a ``_MODAL`` rule this member has no constructor for -- so under wif on Modal every constructor
#: REFUSES BY NAME before any request (:func:`wif_credentials`), the vault's own rule re-implemented here because this member imports
#: no sisteer; otherwise construction succeeded there and the first request failed late, opaquely.
MODAL_MARKER_ENV = "MODAL_TASK_ID"

ClientKind = Literal["batch", "smoke", "live"]


class ClientGuardError(RuntimeError):
    pass


#: The statuses that mean the CREDENTIAL was rejected (``sisteer.vault.CREDENTIAL_REJECTED_STATUSES``): 401 ``authentication_error``,
#: 403 ``permission_error``. A 429 is the pool, a 529 / 5xx the provider, a 400 the request: none of those is this.
CREDENTIAL_REJECTED_STATUSES = (401, 403)


class CredentialRejection(NamedTuple):
    """What :func:`credential_rejection` read off an SDK exception: the HTTP status, the SDK's error type name and the API's
    own error type (``authentication_error`` / ``permission_error``) when the body carried one."""
    status: int
    error_type: str
    api_error_type: str | None


def credential_rejection(exc: BaseException) -> CredentialRejection | None:
    """``CredentialRejection`` when ``exc`` is the Anthropic SDK's credential rejection -- ``anthropic.AuthenticationError``
    (401) / ``anthropic.PermissionDeniedError`` (403), any ``APIStatusError`` whose status is one of
    :data:`CREDENTIAL_REJECTED_STATUSES`, or, under federation, ``anthropic.WorkloadIdentityError`` with such a status (the
    token exchange refused the identity: an ``AnthropicError``, NOT an ``APIStatusError``, whose redacted body is an OAuth
    error string or a non-JSON text as often as the API's error object) -- else None. The SAME rule as ``sisteer.vault
    .credential_rejection`` (this member imports no sisteer); reads the status and the API's error type only -- never a key,
    a token or the request. The exchange's 5xx, a transport failure with no status and a plain connection error are None:
    the retry loops' business, never a stop."""
    status = getattr(exc, "status_code", None)
    typed = isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError))
    by_status = isinstance(exc, anthropic.APIStatusError) and status in CREDENTIAL_REJECTED_STATUSES
    federated = (isinstance(exc, getattr(anthropic, "WorkloadIdentityError", ()))       # an SDK without federation: never
                 and status in CREDENTIAL_REJECTED_STATUSES)
    if not (typed or by_status or federated):
        return None
    api_type = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("type"), str):
            api_type = err["type"]
    try:
        status_int = int(status) if status is not None else (401 if "Authentication" in type(exc).__name__ else 403)
    except (TypeError, ValueError):
        status_int = 401
    return CredentialRejection(status_int, type(exc).__name__, api_type)


def credential_fix_hint(mode: str | None = None) -> str:
    """What fixes a rejected credential under ``mode`` (default: the configured mode, read leniently -- a refusal message must
    never itself refuse): the federation ids file / the VM identity under wif, the one env key under env, the vault item under op."""
    mode = (mode or os.environ.get(SECRETS_MODE_ENV) or "op").strip() or "op"
    if mode == WIF_MODE:
        return "the federation ids file / the VM identity"
    if mode == "env":
        return ENV_MODE_KEY_VAR
    return "the pool's vault item"


def secrets_mode() -> str:
    """``op`` (default), ``env`` or ``wif``; anything else is a typed refusal."""
    mode = os.environ.get(SECRETS_MODE_ENV, "op").strip() or "op"
    if mode not in SECRETS_MODES:
        raise ClientGuardError(f"{SECRETS_MODE_ENV}={mode!r}: must be one of 'op', 'env' or 'wif'")
    return mode


def env_mode_key(kind: ClientKind) -> str:
    """The single documented key under env mode, for EVERY kind (batch, smoke,
    live): required by name, never read from a file, never fetched from the
    vault."""
    key = os.environ.get(ENV_MODE_KEY_VAR, "").strip()
    if not key:
        raise ClientGuardError(
            f"{SECRETS_MODE_ENV}=env: {ENV_MODE_KEY_VAR} must be set (one key serves "
            f"every role; requested kind={kind!r})"
        )
    return key


def secret_cmd() -> str:
    """The secret-manager CLI op mode runs (``$SISTEER_SECRET_CMD``); unset or blank is a refusal naming the setting, never a
    guessed command."""
    cmd = (os.environ.get(SECRET_CMD_ENV) or "").strip()
    if not cmd:
        raise ClientGuardError(f"{SECRET_CMD_ENV} is not set: op mode reads each key through the site's secret-manager CLI, which "
                               f"this setting names (config.env); set it, or set {SECRETS_MODE_ENV}=env and export {ENV_MODE_KEY_VAR}")
    return cmd


def vault_secret(item: str, runner=None) -> str:
    """One secret from the secret manager via the secret-manager CLI (stdout
    only — the value is never logged, echoed, or written anywhere). ``runner``
    is the subprocess seam tests inject a fake through; its default is resolved
    at CALL time (``subprocess.run`` as the module sees it then), so a patched
    ``subprocess.run`` is what every constructor reaches."""
    cmd = secret_cmd()
    if shutil.which(cmd) is None:
        raise ClientGuardError(f"the secret-manager CLI that {SECRET_CMD_ENV} names is not an executable (not found, or not on PATH)")
    run = runner if runner is not None else subprocess.run
    proc = run([cmd, item], capture_output=True, text=True)
    if proc.returncode != 0:
        raise ClientGuardError(
            f"the secret-manager CLI ({SECRET_CMD_ENV}) {item!r} failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}"
        )
    return proc.stdout.strip()


def read_key(kind: ClientKind, runner=None) -> str:
    """The key for ``kind``.

    Under ``SISTEER_SECRETS_MODE=env``: the one documented ``ANTHROPIC_API_KEY``
    for every kind (no file, no vault).

    Under ``op`` (default): the vault item for ``kind`` — and nothing else
    (never a fourth path; no key file is ever read, see the module docstring).
    The value is compared against the
    HIGHPRIO vault item and refused on a match.

    Under ``wif``: REFUSED by name — no key exists; the constructors federate
    (:func:`wif_credentials`)."""
    mode = secrets_mode()
    if mode == "env":
        return env_mode_key(kind)
    if mode == WIF_MODE:
        raise ClientGuardError(
            f"{SECRETS_MODE_ENV}={WIF_MODE}: the {WIF_POOLS[kind]} pool (kind={kind!r}) has NO API key — the client "
            f"authenticates through workload identity federation (the ids of {wif_env_path(kind)}); build it with "
            f"build_client / build_async_client, nothing reads {ENV_MODE_KEY_VAR}, never the secret-manager CLI")
    key = vault_secret(VAULT_ITEMS[kind], runner)
    if not key:
        raise ClientGuardError(f"vault item {VAULT_ITEMS[kind]!r} for kind={kind!r} is empty")
    source = f"vault item {VAULT_ITEMS[kind]!r}"
    highprio = vault_secret(HIGHPRIO_VAULT_ITEM, runner)
    highprio_src = f"vault item {HIGHPRIO_VAULT_ITEM!r}"

    if highprio and key == highprio:
        raise ClientGuardError(
            f"REFUSING to build a client: the {kind!r} key ({source}) "
            f"resolves to the HIGHPRIO org key ({highprio_src}). This "
            f"project never runs on highprio."
        )
    return key


# --- workload identity federation (SISTEER_SECRETS_MODE=wif; no API key) ----------------------------------------------------
def wif_env_path(kind: ClientKind) -> str:
    """The federation ids file of ``kind``'s pool: ``$SISTEER_WIF_ENV_DIR`` (default ``~/.anthropic``) / ``wif-<stem>.env``."""
    if kind not in WIF_FILES:
        raise ClientGuardError(f"unknown client kind {kind!r} (one of {sorted(WIF_FILES)})")
    base = os.path.expanduser(os.environ.get(WIF_ENV_DIR_ENV) or WIF_ENV_DIR_DEFAULT)
    return os.path.join(base, WIF_FILES[kind])


def _read_ids_file(path: str) -> dict:
    """``KEY=VALUE`` lines (``#`` comments and blanks skipped); the file is read whole and ONLY the required ids are kept."""
    ids: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            ids[k.strip()] = v.strip().strip('"').strip("'")
    return ids


def wif_ids(kind: ClientKind, *, path: str | None = None) -> dict:
    """The four federation ids of ``kind``'s pool, every one present and non-empty — else a :class:`ClientGuardError` naming
    the file and what it lacks (an EMPTY file is a refusal by name, never a fallback to another file)."""
    p = path or wif_env_path(kind)
    if not os.path.isfile(p):
        raise ClientGuardError(f"{SECRETS_MODE_ENV}={WIF_MODE}: the {WIF_POOLS[kind]} pool (kind={kind!r}) is unavailable: "
                               f"{p} does not exist")
    ids = _read_ids_file(p)
    missing = [k for k in WIF_REQUIRED_KEYS if not ids.get(k)]
    if missing:
        raise ClientGuardError(f"{SECRETS_MODE_ENV}={WIF_MODE}: the {WIF_POOLS[kind]} pool (kind={kind!r}) is unavailable: "
                               f"{p} is {'empty' if not ids else 'incomplete'} — lacks {missing} (each pool reads only "
                               "its own ids file, never another pool's)")
    return {k: ids[k] for k in WIF_REQUIRED_KEYS}


def wif_ids_sha12(kind: ClientKind, *, path: str | None = None) -> str:
    """sha256 of the ids file's bytes, first 12 hex — the keyless pool stamp's digest (what the harness's provenance records
    as ``ids_sha12``); the file is validated first (:func:`wif_ids`) so an empty file never yields a digest that looks like a
    record."""
    p = path or wif_env_path(kind)
    wif_ids(kind, path=p)
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def wif_pool_stamp(kind: ClientKind) -> dict:
    """``{"pool": <batch|lowprio>, "auth": "wif", "ids_sha12": ...}`` — never a token, never an id."""
    return {"pool": WIF_POOLS[kind], "auth": WIF_MODE, "ids_sha12": wif_ids_sha12(kind)}


def google_identity_token() -> str:
    """A fresh Google identity token for the Anthropic audience (the VM's attached service account); the SDK calls this at
    each exchange and caches the exchanged bearer itself. google-auth rides the harness venv (the generator's interpreter);
    its absence is a refusal by name, never a fallback to a key."""
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token
    except ImportError as exc:  # pragma: no cover - the harness venv carries google-auth
        raise ClientGuardError(f"{SECRETS_MODE_ENV}={WIF_MODE} needs google-auth for the identity token: {exc}") from None
    return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), ANTHROPIC_WIF_AUDIENCE)


def on_modal(env: dict | None = None) -> bool:
    """True inside a Modal container (its task id is in the environment), as the codebase detects it elsewhere."""
    return bool(((os.environ if env is None else env).get(MODAL_MARKER_ENV) or "").strip())


#: the ``ANTHROPIC_*`` prefix every federated constructor scrubs from the process environment first (the vault's
#: ``WIF_SCRUB_PREFIX``, re-stated here because this member imports no sisteer): the SDK's constructor reads
#: ``ANTHROPIC_CUSTOM_HEADERS`` and ``ANTHROPIC_WEBHOOK_SIGNING_KEY`` whatever credential it is given, a static ``X-Api-Key`` /
#: ``Authorization`` header parsed off the former rides EVERY request and makes the SDK skip the token exchange (the federated
#: provider is never invoked; the stale header is sent and refused), and nothing under wif has a use for any of them.
WIF_SCRUB_PREFIX = "ANTHROPIC_"


def scrub_inherited_anthropic_environment(env: dict | None = None) -> tuple[str, ...]:
    """Drop EVERY inherited ``ANTHROPIC_*`` name from ``env`` (default ``os.environ``) -- a key iteration and a delete, NO value
    read -- and return the names dropped, sorted (never a value). The twin of ``sisteer.vault.scrub_inherited_anthropic_environment``
    (round 10 on a278e4ec0); run by :func:`build_client` / :func:`build_async_client` under wif before the SDK's constructor.
    Idempotent. ``env`` is the test seam (a recording mapping)."""
    e = os.environ if env is None else env
    present = tuple(sorted(k for k in list(e) if isinstance(k, str) and k.startswith(WIF_SCRUB_PREFIX)))   # NAMES: __iter__ only
    for k in present:
        del e[k]
    return present


def wif_credentials(kind: ClientKind, *, token_provider: Callable[[], str] | None = None):
    """The SDK's ``WorkloadIdentityCredentials`` for ``kind``'s pool: the ids of its file passed EXPLICITLY, the identity
    token from ``token_provider`` (default :func:`google_identity_token`; tests inject a stub — the ``wif`` twin of the
    ``runner=`` seam). Guard 3 under wif: when the highprio pool's file is present and complete and ``kind``'s ids are ITS
    ids, construction refuses — this project never runs on highprio, whichever file delivered the identity. With the default
    provider, a Modal container REFUSES BY NAME (:func:`on_modal`, before the ids file is even read): the GCP identity-token
    path cannot work there, and the first request would otherwise fail late and opaquely; an injected ``token_provider`` (the
    future Modal constructor) passes."""
    if token_provider is None and on_modal():
        raise ClientGuardError(
            f"{SECRETS_MODE_ENV}={WIF_MODE} inside a Modal container ({MODAL_MARKER_ENV} is set): the {WIF_POOLS[kind]} pool "
            f"(kind={kind!r}) cannot federate there -- the GCP identity-token path (google.oauth2.id_token under "
            f"ANTHROPIC_FEDERATION_RULE_ID_GCP) does not exist in a Modal sandbox and this generator has no Modal constructor "
            f"(a sandbox identity token under the _MODAL rule, one org Secret per function, is a follow-up); the corpus generator "
            f"runs on the GCP VM -- refusing before any request")
    ids = wif_ids(kind)
    hi_path = os.path.join(os.path.dirname(wif_env_path(kind)), HIGHPRIO_WIF_FILE)
    if os.path.isfile(hi_path):
        hi = _read_ids_file(hi_path)
        if all(hi.get(k) for k in WIF_REQUIRED_KEYS) and {k: hi[k] for k in WIF_REQUIRED_KEYS} == ids:
            raise ClientGuardError(
                f"REFUSING to build a client: the {kind!r} pool's federation ids ({wif_env_path(kind)}) are the HIGHPRIO "
                f"pool's ({hi_path}). This project never runs on highprio.")
    return anthropic.WorkloadIdentityCredentials(
        identity_token_provider=token_provider or google_identity_token,
        federation_rule_id=ids["ANTHROPIC_FEDERATION_RULE_ID_GCP"],
        organization_id=ids["ANTHROPIC_ORGANIZATION_ID"],
        service_account_id=ids["ANTHROPIC_SERVICE_ACCOUNT_ID"],
        workspace_id=ids["ANTHROPIC_WORKSPACE_ID"])


def build_client(
    kind: ClientKind,
    max_retries: int = 4,
    *,
    token_provider: Callable[[], str] | None = None,
) -> anthropic.Anthropic:
    if secrets_mode() == WIF_MODE:
        # credentials= is an explicit credential to the SDK: it consults no ANTHROPIC_* variable for a CREDENTIAL (api_key stays
        # None) -- but its constructor reads ANTHROPIC_CUSTOM_HEADERS / ANTHROPIC_WEBHOOK_SIGNING_KEY whatever it is given, and a
        # static X-Api-Key / Authorization parsed off the former rides every request and skips the token exchange, so the inherited
        # ANTHROPIC_* names are dropped by name FIRST (round 10); base_url pinned (guard 2). Round 14: the SDK's OWN retries are OFF
        # (it re-sends the first 401 of a federated client with a fresh exchange) and `max_retries` is the latched layer's budget; the
        # client is the member's federated subclass, so batch.py's with_options(max_retries=0) creator stays federated -- the same
        # credentials, token cache, transport and latch, the layer at budget 0. Round 15: the SDK is handed the MEMBER'S OWN httpx client
        # over the pin guard (never its default client, whose transport sends wherever a results_url or a redirect points)
        creds = wif_credentials(kind, token_provider=token_provider)
        scrub_inherited_anthropic_environment()
        client = federated_sdk_class(anthropic.Anthropic)(
            credentials=creds,
            base_url=API_BASE_URL,
            max_retries=0,
            http_client=federated_http_client(subject=_federated_subject(kind)),
        )
        federate_sdk_retries(client, max_retries=max_retries)
        return client
    key = read_key(kind)
    # Explicit api_key + base_url: the SDK then ignores ANTHROPIC_API_KEY and
    # ANTHROPIC_BASE_URL entirely (guard 2 + 3).
    return anthropic.Anthropic(
        api_key=key,
        base_url=API_BASE_URL,
        max_retries=max_retries,
    )


def build_async_client(
    kind: ClientKind = "live",
    max_retries: int = 0,
    timeout: float = 1800.0,
    *,
    token_provider: Callable[[], str] | None = None,
) -> anthropic.AsyncAnthropic:
    """Async client for the live wave engine — same three guards.

    max_retries defaults to 0 ON PURPOSE: live.py owns the bounded retry loop
    (429 halves its concurrency limiter; SDK-transparent retries would refire
    at full width and deepen a rate-limit event). The long timeout covers
    streamed multi-thousand-token story generations."""
    if secrets_mode() == WIF_MODE:
        creds = wif_credentials(kind, token_provider=token_provider)
        scrub_inherited_anthropic_environment()       # round 10: see build_client
        client = federated_sdk_class(anthropic.AsyncAnthropic)(   # round 14: the SDK's own retries OFF, the count the layer's budget
            credentials=creds,
            base_url=API_BASE_URL,
            max_retries=0,
            timeout=timeout,
            # round 15: the live runner's transport (its plain ssl context, its wave-sized limits) INSIDE the pin guard
            http_client=federated_http_client(async_client=True, timeout=timeout, inner=live_async_transport(),
                                              subject=_federated_subject(kind, async_client=True)),
        )
        federate_sdk_retries(client, max_retries=max_retries)
        return client
    key = read_key(kind)
    return anthropic.AsyncAnthropic(
        api_key=key,
        base_url=API_BASE_URL,
        max_retries=max_retries,
        timeout=timeout,
        http_client=live_http_client(timeout),
    )


# --- THE SDK'S OWN RETRIES ARE OFF UNDER wif, REPLAYED ABOVE THE CLIENT'S CREDENTIAL LATCH; A COPY STAYS FEDERATED --------------
# (Major.) ``build_client("batch")`` gave the SDK its default four retries under wif, and the installed SDK's
# ``Anthropic._should_retry`` (0.120.2 and 1.0.0 alike) INVALIDATES a federated token cache on a 401 and RETRIES THE FIRST ONE: a poll /
# list / results request answered 401 was re-sent with a freshly exchanged bearer -- a second token exchange and a second request on the
# rejected identity -- before ``batch.WaveEngine._guard`` saw anything (measured on dbd7473df: 2 requests, 2 exchanges per rejected
# call); and sibling engines share one client across threads (``pipeline.make_engine`` per thread), so a poll asleep in the SDK's 429
# backoff when a sibling's poll met the 401 woke into a third request. The vault closed the same class on its own clients in round 12
# (``sisteer.vault.federate_sdk_retries``); this member imports no sisteer, so it carries the twin: under wif the SDK client is built with
# ``max_retries=0`` and the count asked for is the budget of :class:`LatchedRetryMessages` (:func:`federate_sdk_retries`), which replays
# the SDK's own policy -- the same statuses (:func:`sdk_would_retry`), delays (:func:`sdk_retry_delay`) and count -- over
# ``messages.{create,parse,count_tokens,stream}`` and ``messages.batches.{create,retrieve,list,cancel,delete,results}``, checking the
# client's credential latch (:data:`CLIENT_LATCH_ATTR`: the typed halt the engines hold on the client) BEFORE EVERY ATTEMPT and
# classifying the credential FIRST after every failure (held on the client, raised as the SDK raised it, never retried). op / env hand the
# SDK the caller's count and put no layer on the client, byte for byte as before.
# And the client is the member's own subclass of the SDK's (:func:`federated_sdk_class`), whose ``copy`` / ``with_options`` keep the
# federation (:func:`federated_copy`: batch.py's ``with_options(max_retries=0)`` creator shares its parent's credentials, token cache,
# transport and latch, with the layer at budget 0) and refuse by name an ``http_client``, another base URL, a static credential header, a
# credential change or a middleware (round 14's Critical on the vault: the SDK's own copy inherited the credentials, took a caller's httpx
# client, and nothing of the instance's protections travelled). The installed SDK forwards credentials through ``copy`` only for its EXACT
# base classes (``anthropic._client._is_base_client``), so the SDK's own copy of the subclass carries no credentials at all.

#: the attribute under which the FIRST credential rejection is latched on the CLIENT (``batch`` re-exports
#: it): sibling engines over one client -- `corpusgen run --corpus all`, the extension sub-DAGs, `onpolicy score` (pipeline.make_engine
#: builds one WaveEngine per thread) -- each used to issue its own doomed create after the first engine's rejection; the halt rides the
#: client they share. Read and held through :func:`held_credential_rejection` / :func:`hold_credential_rejection`, which resolve a
#: federated copy to the client it was copied from (:data:`COPY_ROOT_ATTR`), so parent and every copy share ONE stop.
CLIENT_LATCH_ATTR = "_corpusgen_credential_rejected"
#: a federated copy's pointer to the client it was copied from (the latch lives there)
COPY_ROOT_ATTR = "_corpusgen_copy_root"
SDK_RETRY_INITIAL_DELAY_S = 0.5
SDK_RETRY_MAX_DELAY_S = 8.0
SDK_RETRY_AFTER_CAP_S = 60.0
SDK_RETRY_STATUSES = (408, 409, 429)          # and every 5xx
#: the arguments of the SDK's ``copy`` a federated copy accepts (``base_url`` as the pin only; ``http_client`` as None only -- "keep the
#: parent's", the SDK's own reading of None); every other name is refused by name
FEDERATED_COPY_OPTIONS = ("timeout", "max_retries", "default_headers", "set_default_headers", "default_query", "set_default_query",
                          "base_url", "http_client")
STATIC_AUTH_HEADERS = ("x-api-key", "authorization")
_FEDERATED_SDK_CLASSES: dict = {}
_NOT_GIVEN = object()


def latch_root(client: Any) -> Any:
    """The client a latch is read from and held on: ``client`` itself, or -- for a federated copy -- the client it was copied from."""
    seen: set = set()
    while True:
        root = getattr(client, COPY_ROOT_ATTR, None)
        if root is None or id(root) in seen:
            return client
        seen.add(id(client))
        client = root


def held_credential_rejection(client: Any) -> Any:
    """The typed halt (``batch.WaveCredentialRejected``) latched on ``client`` -- on the client it was copied from, for a federated
    copy -- else None."""
    return getattr(latch_root(client), CLIENT_LATCH_ATTR, None)


def hold_credential_rejection(client: Any, halt: Any) -> Any:
    """Hold ``halt`` on ``client`` (its root) when nothing is held there yet; returns what is held after -- the FIRST wins. A client that
    refuses attributes holds nothing (the caller's own latch still holds)."""
    root = latch_root(client)
    held = getattr(root, CLIENT_LATCH_ATTR, None)
    if held is not None:
        return held
    try:
        setattr(root, CLIENT_LATCH_ATTR, halt)
    except (AttributeError, TypeError):
        pass
    return halt


def sdk_would_retry(exc: BaseException) -> tuple[bool, Any]:
    """Whether the SDK's own request loop would retry ``exc`` (``anthropic._base_client._should_retry_exception`` / ``_should_retry``,
    replayed; the twin of ``sisteer.vault.sdk_would_retry``): a ``RetryableError``, an ``APIStatusError`` whose response says
    ``x-should-retry: true`` (or not ``false``) and whose status is 408 / 409 / 429 / >= 500, an ``APIConnectionError`` (a timeout
    included) -- each also found through the ``__cause__`` chain, as the SDK walks it. Returns ``(retry, response)``. A credential
    rejection is never asked about: the layer classifies it first (the SDK's own 401 re-send of a federated client is exactly what this
    layer replaces)."""
    retryable = getattr(anthropic, "RetryableError", None)
    seen: set = set()
    cur: BaseException | None = exc
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


def _retry_after_s(headers: Any) -> float | None:
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
    """The shared half of the latched retry layer: the inner SDK resource, the CLIENT whose latch is consulted (through its root, so a
    copy reads its parent's) and the retry budget the caller asked for. Every other attribute forwards to the resource."""

    resource = "messages"

    def __init__(self, inner: Any, client: Any, max_retries: int):
        self._inner = inner
        self._client = client
        self.max_retries = max(0, int(max_retries))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_inner"], name)

    def _check(self) -> None:
        held = held_credential_rejection(self._client)
        if held is not None:
            raise held                                   # BEFORE EVERY attempt: the first, and each wake from a retry sleep

    def _refuse_request(self, kwargs: Any, what: str) -> None:
        """BEFORE every request of a federated client (Major): a caller's per-request ``extra_headers``
        carrying a static ``X-Api-Key`` / ``Authorization`` is refused BY NAME (:func:`refuse_static_auth_headers`) -- the SDK's
        ``AccessTokenAuth`` skips the token exchange for a request already carrying one, so the static header would be sent in place
        of the exchanged bearer (measured: zero exchanges, the stale header on the wire). The vault's round-13 rule, on this member's
        every door."""
        refuse_static_auth_headers(extra_headers=kwargs.get("extra_headers"),
                                   subject=f"a request of the corpus generator's federated client ({self.resource}.{what})")

    def _after_failure(self, exc: BaseException, nb: int) -> tuple[bool, float]:
        """After a failed attempt: a credential rejection is held on the client (the typed halt, ``batch.credential_halt``) and NEVER
        retried -- the caller re-raises it as the SDK raised it, so the engine's own classification of that one call is unchanged, and
        every sibling's next attempt raises the held halt from the check; anything else is retried iff the SDK would have and the budget
        allows, after the SDK's delay. Returns ``(retry, delay_s)``."""
        from .batch import credential_halt

        halt = credential_halt(exc)
        if halt is not None:
            hold_credential_rejection(self._client, halt)
            return False, 0.0
        retry, resp = sdk_would_retry(exc)
        if not retry or nb >= self.max_retries:
            return False, 0.0
        return True, sdk_retry_delay(nb, resp)

    def _attempt(self, call: Callable[[], Any], kwargs: Any = None, what: str = "request") -> Any:
        self._refuse_request(kwargs or {}, what)     # round 15: a static per-request credential header, before anything
        nb = 0
        while True:
            self._check()
            try:
                return call()
            except Exception as e:                       # noqa: BLE001 -- classified: the credential first, then the SDK's own policy
                refused = destination_refusal(e)
                if refused is not None:                  # round 15: the pin guard spoke inside the SDK's request -- typed, never retried
                    raise refused from e
                retry, delay = self._after_failure(e, nb)
                if not retry:
                    raise
                nb += 1
                time.sleep(delay)                        # the SDK's own sleep, above the latch

    async def _attempt_async(self, call: Callable[[], Any], kwargs: Any = None, what: str = "request") -> Any:
        import asyncio

        self._refuse_request(kwargs or {}, what)
        nb = 0
        while True:
            self._check()
            try:
                return await call()
            except Exception as e:                       # noqa: BLE001 -- classified as above
                refused = destination_refusal(e)
                if refused is not None:
                    raise refused from e
                retry, delay = self._after_failure(e, nb)
                if not retry:
                    raise
                nb += 1
                await asyncio.sleep(delay)


class _LatchedRetryBatches(_LatchedRetry):
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
    """The stream manager of ``messages.stream``: the request goes out at ``__enter__`` / ``__aenter__``, which is where the layer's
    attempts run (a fresh SDK manager per attempt); the entered stream is the SDK's own. A static per-request credential header was
    refused when ``stream`` was called, before this manager existed (and is refused again at the attempt; round 15)."""

    def __init__(self, owner: _LatchedRetry, args: tuple, kwargs: dict):
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
    """The ``messages`` resource of a federated SYNC client -- ``create`` / ``parse`` / ``count_tokens`` / ``stream`` and ``batches`` --
    with the SDK's own retry policy replayed above the client's credential latch (the section comment above). Installed by
    :func:`federate_sdk_retries` on every client :func:`build_client` builds under wif; never on a keyed client."""

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
    """The async twin of :class:`LatchedRetryMessages` (:func:`build_async_client`'s client under wif: the live engine's ``stream``)."""

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.create(*args, **kwargs), kwargs, "create")

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.parse(*args, **kwargs), kwargs, "parse")

    def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        return self._attempt_async(lambda: self._inner.count_tokens(*args, **kwargs), kwargs, "count_tokens")

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse_request(kwargs, "stream")      # at the call, before a manager exists (and again at the attempt)
        return _LatchedRetryStream(self, args, kwargs)

    @property
    def batches(self) -> Any:
        b = self.__dict__.get("_batches")
        if b is None:
            b = self.__dict__["_batches"] = _AsyncLatchedRetryBatches(self._inner.batches, self._client, self.max_retries)
        return b


def federate_sdk_retries(client: Any, *, max_retries: int) -> Any:
    """Put the latched retry layer on a federated SDK client: its ``messages`` resource becomes :class:`LatchedRetryMessages`
    (:class:`AsyncLatchedRetryMessages` for an ``AsyncAnthropic``) carrying ``max_retries`` as the budget. The client MUST have been
    built with the SDK's own retries off (``max_retries=0``); anything else is refused by name -- the layer above a retrying SDK would
    be two retry layers, the lower one below the latch, the very thing this replaces. Returns the layer."""
    sdk = getattr(client, "max_retries", None)
    if sdk != 0:
        raise ClientGuardError(f"federate_sdk_retries: the SDK client carries max_retries={sdk!r}; a federated client is built with the "
                               "SDK's own retries OFF (max_retries=0) and its retries replayed here, above the credential latch (round 14)")
    cls = AsyncLatchedRetryMessages if isinstance(client, anthropic.AsyncAnthropic) else LatchedRetryMessages
    layer = cls(client.messages, client, max_retries)
    client.messages = layer
    return layer


def federated_retry_budget(client: Any) -> int | None:
    """The retry budget the latched layer of a federated client carries (the count its builder asked for; the SDK's own ``max_retries``
    on such a client is 0); None for a keyed client, whose SDK ``max_retries`` IS its budget."""
    m = getattr(client, "messages", None)
    seen: set = set()
    while m is not None and id(m) not in seen:
        if isinstance(m, _LatchedRetry):
            return m.max_retries
        seen.add(id(m))
        m = getattr(m, "_inner", None)
    return None


def _copy_subject(client: Any) -> str:
    return "a copy of the corpus generator's federated client (corpus_generator.client.build_client / build_async_client)"


def federated_sdk_class(cls: Any) -> Any:
    """The class every federated client of this member is an instance of: the member's own subclass of ``cls`` (``anthropic.Anthropic``
    / ``anthropic.AsyncAnthropic``), named ``Federated<cls>``, whose ``copy`` / ``with_options`` are :func:`federated_copy` and whose
    ``with_middleware`` is :class:`ClientGuardError` by name. Built once per class; a federated class is returned as it is; anything that
    is not a class is a ``TypeError`` by name."""
    if not isinstance(cls, type):
        raise TypeError(f"federated_sdk_class: {cls!r} is not a class (the SDK's client class is expected)")
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
            raise ClientGuardError(
                f"{_copy_subject(self)}: with_middleware() -- a copy of a federated client carries exactly its parent's request pipeline; a "
                f"middleware is refused BY NAME under {SECRETS_MODE_ENV}={WIF_MODE}; build another client for anything else")

    Federated.__name__ = f"Federated{cls.__name__}"
    Federated.__qualname__ = Federated.__name__
    Federated.__module__ = __name__
    Federated.__doc__ = (f"A federated ``{cls.__module__}.{cls.__qualname__}`` (corpus_generator.client.federated_sdk_class): its ``copy`` / "
                         f"``with_options`` keep the federation (federated_copy); ``with_middleware`` is refused by name.")
    _FEDERATED_SDK_CLASSES[cls] = Federated
    return Federated


def federated_copy(client: Any, overrides: dict) -> Any:
    """``copy`` / ``with_options`` of a federated client (the section comment above): a client of the same federated class, built
    DIRECTLY on the parent's credentials, token cache (one exchange serves both) and httpx client (the same object), its base URL the
    pin, the SDK's retries OFF, the parent's headers and query merged with the copy's as the SDK's own copy merges them, pointing at the
    parent as its latch root (one stop for parent and copies), with the latched layer at the budget asked (``max_retries``, else the
    parent's). Refused BY NAME (:class:`ClientGuardError`) before anything is built: an ``http_client``, a ``base_url`` naming another
    host, a static ``X-Api-Key`` / ``Authorization`` in ``default_headers`` / ``set_default_headers``, and any other argument."""
    subject = _copy_subject(client)
    given = dict(overrides)
    unknown = sorted(k for k in given if k not in FEDERATED_COPY_OPTIONS)
    if unknown:
        raise ClientGuardError(
            f"{subject}: copy() / with_options() with {', '.join(unknown)} -- a copy of a federated client carries its parent's credentials, "
            f"token cache, destination and request pipeline, and takes only timeout, max_retries, default_headers / set_default_headers, "
            f"default_query / set_default_query and {API_BASE_URL} as base_url; a credential (api_key, auth_token, credentials, config, "
            f"profile), a webhook key, a middleware or an argument the SDK does not know is refused BY NAME under {SECRETS_MODE_ENV}={WIF_MODE}")
    if given.get("http_client") is not None:
        raise ClientGuardError(
            f"{subject}: copy() / with_options(http_client=...) -- under {SECRETS_MODE_ENV}={WIF_MODE} the SDK is never handed a caller's httpx "
            f"client through a copy (a request hook, a mount, a subclass's own send or a transport of its own can carry the exchanged bearer "
            f"off {API_BASE_URL} however the client's base URL reads); the copy keeps its parent's client; refusing to copy")
    base_url = given.get("base_url")
    if base_url is not None and str(base_url).strip().rstrip("/") != API_BASE_URL:
        raise ClientGuardError(
            f"{subject}: copy() / with_options(base_url={str(base_url)!r}) names a destination other than the pinned API host {API_BASE_URL} -- "
            f"a federated client's bearer is minted for that host and must not reach another; refusing to copy")
    if given.get("default_headers") is not None and given.get("set_default_headers") is not None:
        raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")     # the SDK's own rule
    new_headers = given.get("default_headers") if given.get("default_headers") is not None else given.get("set_default_headers")
    refuse_static_auth_headers(default_headers=new_headers, subject=f"{subject}: copy() / with_options()")   # round 15: the ONE rule
    if getattr(client, "api_key", None) is not None or getattr(client, "auth_token", None) is not None:
        raise ClientGuardError(f"{subject}: the parent carries a static credential beside its federated one; refusing to copy")
    asked = given.get("max_retries", _NOT_GIVEN)
    if asked is _NOT_GIVEN or asked is None:
        budget = federated_retry_budget(client)
        if budget is None:
            raise ClientGuardError(f"{subject}: the parent carries no latched retry layer; refusing to copy")
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
        http_client=getattr(client, "_client"),                                           # the parent's httpx client: the same object
        default_headers=headers, default_query=params, webhook_key=client.webhook_key,
        middleware=getattr(client, "_middleware", None),
        _strict_response_validation=getattr(client, "_strict_response_validation", False))
    setattr(copy, COPY_ROOT_ATTR, latch_root(client))                                     # ONE latch for parent and copy
    federate_sdk_retries(copy, max_retries=budget)
    return copy


# --- THE DESTINATION IS ENFORCED WHERE THE BYTES LEAVE; A STATIC PER-REQUEST CREDENTIAL HEADER IS REFUSED ON EVERY DOOR ------------
# Under wif the SDK client was built on the SDK's OWN default httpx client (the live
# async client on ``live_http_client``'s), whose transport sends wherever the request is addressed -- and the installed SDK's
# ``messages.batches.results`` takes its request's destination OFF THE RETRIEVE RESPONSE (``batch.results_url``; the SDK's
# ``_prepare_url`` keeps an absolute URL as it is), so a batch answer naming another host made the SDK send a NEW authenticated GET
# there, the exchanged bearer on it (measured with the real SDK over a mock wire: the retrieve to the pin, then ``GET
# other-host.example/.../results`` with ``Authorization: Bearer <the exchanged bearer>``); a cross-host redirect the API answered would
# have been followed the same way. And the latched layer forwarded a caller's per-request ``extra_headers`` unchanged: a static
# ``Authorization`` / ``X-Api-Key`` there makes the SDK's ``AccessTokenAuth`` skip the token exchange (``_has_static_credential``), so the
# static header rode the request and the identity was never used (measured: provider 0, exchanges 0, the static header on the wire, at
# ``messages.create`` and ``messages.batches.retrieve``). The vault closed both on its own clients
# (``sisteer.vault.PinnedTransport`` / ``federated_http_client`` / ``destination_refusal``; ``_LatchedRetry._refuse_request`` /
# ``refuse_static_auth_headers``); this member imports no sisteer, so it carries the twins -- ONE rule, pinned equal to the vault's by
# ``tests/test_client_wif_transport_pin_and_request_headers.py``, which imports the vault's module from the repository beside this
# member. Under wif every client :func:`build_client` / :func:`build_async_client` builds is handed the MEMBER'S OWN httpx2 client
# (:func:`federated_http_client`: ``base_url`` the pin, the SDK's defaults -- or, for the live async client, the live runner's plain ssl
# context and wave-sized limits INSIDE the guard -- no hook, no mount, no auth, no environment proxy) whose ONE transport is
# :class:`PinnedTransport`: it forwards a request to httpx's HTTP transport only when the scheme, host and port are the pin's and
# otherwise raises :class:`DestinationRefused` BY NAME at the last in-process step before a byte leaves, after the SDK's URL merge (an
# absolute ``results_url``), httpx's auth flow and a redirect's follow-up request; the latched layer re-raises it typed and never retries
# it (:func:`destination_refusal`: never held as a credential rejection, never counted against the budget); a copy shares its parent's
# client, so the guard travels (:func:`federated_copy`). And the layer refuses a static ``X-Api-Key`` / ``Authorization`` in any request's
# ``extra_headers`` before every attempt of every door (:meth:`_LatchedRetry._refuse_request` -> :func:`refuse_static_auth_headers` ->
# :class:`StaticAuthRefused`) -- ``messages.{create,parse,count_tokens,stream}`` (at the ``stream()`` call too, before a manager exists)
# and ``messages.batches.*``, sync and async -- and the same function refuses a static header in a copy's ``default_headers`` (the
# earlier inline check, one rule now). The constructors still take no ``http_client`` (the member's clients have no door for a caller's httpx
# client; a copy refuses one); the tests swap the guard's INNER transport for a mock on the very client built. op / env are byte for
# byte as before: the SDK's own default client, the live runner's ``live_http_client``, the SDK's own handling of ``extra_headers``.


class DestinationRefused(ClientGuardError):
    """A request of a federated client addressed anywhere but the pinned API host, refused by :class:`PinnedTransport` before any byte
    leaves (the member's twin of ``sisteer.vault.BaseUrlRefused``). The message names the destination and the pin, never a header, a
    path or a body."""


class StaticAuthRefused(ClientGuardError, ValueError):
    """An explicit static credential header offered to a federated client (:func:`refuse_static_auth_headers`): the twin of
    ``sisteer.vault.StaticAuthRefused`` -- a ``ValueError``, as the SDK's own header refusals are -- and a :class:`ClientGuardError`."""


#: the TCP keepalive options the SDK's own default httpx client sets on its transport (``anthropic.DefaultHttpxClient``), replicated on
#: the member's sync client (the vault's ``FEDERATED_SOCKET_OPTIONS``; kept equal by test): a long batch poll must not be dropped by an
#: idle NAT
FEDERATED_SOCKET_OPTIONS: tuple = tuple(
    [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)]
    + ([(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL"), 60)] if getattr(socket, "TCP_KEEPINTVL", None) is not None
       else [(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPALIVE", 0x10), 60)] if sys.platform == "darwin" else [])
    + ([(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT"), 5)] if getattr(socket, "TCP_KEEPCNT", None) is not None else [])
    + ([(socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE"), 60)] if getattr(socket, "TCP_KEEPIDLE", None) is not None else []))
_DEFAULT_PORTS = {"https": 443, "http": 80}
#: the live runner's connection limits, sized for a 200-wide wave (the SDK's defaults keep only 100 alive, so half of every burst would
#: re-handshake): ``live_http_client``'s under op / env, the guard's inner transport's under wif
LIVE_LIMITS = httpx2.Limits(max_connections=1000, max_keepalive_connections=256)


def _federated_subject(kind: ClientKind, *, async_client: bool = False) -> str:
    return (f"the corpus generator's federated {'async ' if async_client else ''}client for the {WIF_POOLS.get(kind, kind)} pool "
            f"(kind={kind!r}; corpus_generator.client.{'build_async_client' if async_client else 'build_client'})")


class PinnedTransport:
    """THE final in-process boundary of a federated client of this member: the one transport of the member's own httpx client, which
    forwards a request to ``inner`` (httpx2's HTTP transport; a test's ``MockTransport``) only when its scheme, host and port are the
    pin's (:data:`API_BASE_URL`), and otherwise raises :class:`DestinationRefused` BY NAME -- naming the destination it refused and the
    pin, never a header, a path or a body -- before any byte leaves. Runs AFTER everything that can rewrite a request's URL inside the
    process (the SDK's URL merge -- an absolute ``results_url`` off a batch response -- and middleware, httpx's auth flow, event hooks, a
    redirect's follow-up request), so what it sees is what would have been sent. The twin of ``sisteer.vault.PinnedTransport``, over both
    httpx2 transport protocols (sync and async), as ``MockTransport`` is."""

    def __init__(self, inner: Any, *, subject: str = "a federated client of the corpus generator", pin: str = API_BASE_URL):
        self.inner = inner
        self.subject = subject
        self.pin = pin
        url = httpx2.URL(pin)
        self._pin_key = (url.scheme, url.host, url.port or _DEFAULT_PORTS.get(url.scheme))

    def check(self, url: Any) -> None:
        key = (url.scheme, url.host, url.port or _DEFAULT_PORTS.get(url.scheme))
        if key != self._pin_key:
            where = f"{url.scheme}://{url.host}" + (f":{url.port}" if url.port else "")
            raise DestinationRefused(
                f"{self.subject}: a request addressed to {where} reached the federated client's transport -- the pin is {self.pin} and "
                f"a federated client's bearer is minted for that host (the identity token's audience) and must not reach another; "
                f"something between the SDK's constructor and the wire re-pointed the request (a batch response's results_url, a "
                f"cross-host redirect, an SDK middleware); refusing before any byte leaves")

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


class FederatedHttpxClient(httpx2.Client):
    """The httpx client of a federated ``Anthropic`` of this member (:func:`federated_http_client`): the member's own, closed when
    collected as the SDK's own default client wrapper is."""

    def __del__(self) -> None:
        if getattr(self, "is_closed", True):
            return
        try:
            self.close()
        except Exception:  # noqa: BLE001 -- as the SDK's wrapper
            pass


class FederatedAsyncHttpxClient(httpx2.AsyncClient):
    """The httpx client of a federated ``AsyncAnthropic`` of this member (:func:`federated_http_client`): the member's own."""

    def __del__(self) -> None:
        if getattr(self, "is_closed", True):
            return
        try:
            import asyncio
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: BLE001 -- no running loop: the pool is collected with the object, as the SDK's is
            pass


def live_async_transport() -> httpx2.AsyncHTTPTransport:
    """The live runner's HTTP transport -- :func:`live_http_client`'s settings as a TRANSPORT: the plain ssl context
    (:func:`live_ssl_context`) and the wave-sized limits (:data:`LIVE_LIMITS`) -- what sits INSIDE the pin guard of the federated
    async client under wif."""
    return httpx2.AsyncHTTPTransport(verify=live_ssl_context(), limits=LIVE_LIMITS)


def federated_http_client(*, async_client: bool = False, timeout: Any = None, inner: Any = None,
                          subject: str = "a federated client of the corpus generator") -> Any:
    """THE httpx client every federated SDK client of this member is built on -- the member's own, never a caller's (the section comment
    above): :class:`FederatedHttpxClient` / :class:`FederatedAsyncHttpxClient` with ``base_url`` the pin, ``follow_redirects`` as the
    SDK's default, ``timeout`` the caller's (else the SDK's default), no event hook, no mount, no auth, no environment proxy (an explicit
    transport: httpx consults none), and its ONE transport :class:`PinnedTransport` over ``inner`` -- httpx2's HTTP transport with the
    SDK's connection limits and keepalive socket options (:data:`FEDERATED_SOCKET_OPTIONS`) by default; the live async client passes
    :func:`live_async_transport`. The SDK's constructor receives this client as its ``http_client`` and nothing of a caller's (the
    constructors take none). The twin of ``sisteer.vault.federated_http_client``."""
    if inner is None:
        inner = (httpx2.AsyncHTTPTransport if async_client else httpx2.HTTPTransport)(
            limits=anthropic.DEFAULT_CONNECTION_LIMITS, socket_options=list(FEDERATED_SOCKET_OPTIONS))
    if timeout is None:
        timeout = anthropic.DEFAULT_TIMEOUT
    cls = FederatedAsyncHttpxClient if async_client else FederatedHttpxClient
    return cls(base_url=API_BASE_URL, timeout=timeout, follow_redirects=True, transport=PinnedTransport(inner, subject=subject))


def destination_refusal(exc: BaseException) -> DestinationRefused | None:
    """The :class:`DestinationRefused` the pin guard raised inside the SDK's request, found through the ``__cause__`` chain (the SDK
    wraps anything its httpx client raises as ``APIConnectionError(...) from err``); None when ``exc`` is not that. The latched retry
    layer re-raises it typed and never retries it (a retry would only meet the guard again). The twin of
    ``sisteer.vault.destination_refusal``."""
    seen: set = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, DestinationRefused):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__
    return None


def static_auth_header_names(headers: Any) -> tuple[str, ...]:
    """The names in ``headers`` (a mapping, an ``httpx.Headers``; None = none) that are a static credential header
    (:data:`STATIC_AUTH_HEADERS`, compared lower-cased), in the caller's own spelling and order. Never a value. The twin of
    ``sisteer.vault.static_auth_header_names``."""
    if headers is None:
        return ()
    try:
        names = list(headers.keys()) if hasattr(headers, "keys") else list(headers)
    except Exception:  # noqa: BLE001 -- not a mapping: nothing to name
        return ()
    return tuple(str(k) for k in names if isinstance(k, str) and k.lower() in STATIC_AUTH_HEADERS)


def refuse_static_auth_headers(*, default_headers: Any = None, http_client: Any = None, extra_headers: Any = None,
                               subject: str = "a federated client of the corpus generator") -> None:
    """REFUSE BY NAME an explicit static credential header on its way into a federated client -- a copy's ``default_headers`` /
    ``set_default_headers``, the headers of an ``http_client`` (none is ever taken; the face is kept for the twin's signature) or one
    request's ``extra_headers`` -- as :class:`StaticAuthRefused`, before anything is built or sent. Under federation the credential is
    the identity token's exchange; a static ``X-Api-Key`` / ``Authorization`` beside ``credentials=`` is a contradiction the SDK resolves
    silently in the static header's favour (``AccessTokenAuth._has_static_credential``). Never dropped silently: a caller's argument is
    deliberate, and a surprise there is a loud one. The twin of ``sisteer.vault.refuse_static_auth_headers``."""
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


def live_ssl_context() -> ssl.SSLContext:
    """A PLAIN stdlib ssl context over the certifi bundle for the live runner.

    The SDK's default httpx2 client verifies through ``truststore`` (the
    system trust store), and httpcore2's async path hands TLS ``wrap_bio`` to
    worker threads — so a wave that opens ~200 connections at once has several
    threads reconfiguring one truststore OpenSSL context concurrently, which
    corrupts the heap ("double free or corruption", "realloc(): invalid old
    size"; the r2 edit phase died three times at launch, 2026-09-03, every
    aborting thread inside truststore._openssl._configure_context). The
    stdlib context has no per-handshake reconfiguration, so the race is gone;
    api.anthropic.com chains to a public CA, so certifi suffices."""
    return ssl.create_default_context(cafile=certifi.where())


def live_http_client(timeout: float) -> httpx2.AsyncClient:
    """The async transport for the live runner: plain ssl (above) and
    connection limits sized for a 200-wide wave (the SDK's defaults keep only
    100 alive, so half of every burst would re-handshake)."""
    return httpx2.AsyncClient(
        verify=live_ssl_context(),
        timeout=timeout,
        limits=LIVE_LIMITS,
    )
