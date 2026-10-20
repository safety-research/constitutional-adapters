"""The ONE W&B routing validator every live-logging entry point shares.

A run may log ONLY to the entity named by ``WANDB_ENTITY``, and only when
the key W&B will actually use resolves to that same entity. The incident
this guards against: a key whose default entity was another member's
account silently routed a whole wave's runs there — ``wandb.init`` with no
entity, or with an entity the key cannot write to, does not refuse, it
re-routes. So a missing/empty ``WANDB_ENTITY`` or a key whose default
entity differs is a CONFIGURATION FAILURE: :class:`WandbEntityError` is
raised and nothing runs. It never degrades to "telemetry off" — an
operator who wants W&B off does not ask for it.

Two callers, one validator: the Modal wrappers
(``scripts/modal/sdfbase_fits.py`` ``require_wandb_entity`` → here, at
container start) and the direct CLI paths
(``scripts/sdfbase/psr_residual_distill.py`` ``_wandb`` →
:func:`wandb_init_checked`, before any model load). API errors while
resolving the key's default entity are retried a bounded number of times
(:data:`DEFAULT_TRIES`) and then ALSO raise: an unverifiable key is not a
verified key.
"""

from __future__ import annotations

import os
import time

#: Bounded retries on API/transport errors; a run that cannot verify its
#: key in three attempts refuses.
DEFAULT_TRIES = 3
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_BACKOFF_S = 2.0

_sleep = time.sleep  # patched by tests; never called on the first attempt


class WandbEntityError(RuntimeError):
    """W&B routing is misconfigured or unverifiable — refuse to run."""


def require_wandb_entity(*, tries: int = DEFAULT_TRIES,
                         timeout: float = DEFAULT_TIMEOUT_S,
                         backoff_s: float = DEFAULT_BACKOFF_S,
                         hint: str | None = None, env=None) -> str:
    """Return the entity every ``wandb.init`` must pass explicitly, or
    raise :class:`WandbEntityError`:

    - ``WANDB_ENTITY`` unset / empty / whitespace → refuse (the key's
      default account would silently receive the runs);
    - the key's own default entity (``wandb.Api().default_entity`` — the
      account the key belongs to) differs from ``WANDB_ENTITY`` → refuse
      (a foreign key; an explicit ``entity=`` would not save it: W&B
      accepts the init and files the run wherever the key can write);
    - the API cannot be reached / the key is rejected ``tries`` times in a
      row → refuse, naming the last error (never "assume it's fine").

    ``hint`` is appended to the unset message (the Modal wrappers name the
    secret to mount). ``env`` defaults to ``os.environ``."""
    env = os.environ if env is None else env
    ent = (env.get("WANDB_ENTITY") or "").strip()
    if not ent:
        raise WandbEntityError(
            "WANDB_ENTITY is unset (or empty) — refusing to start rather than "
            "log to whatever account the W&B key defaults to"
            + (f"; {hint}" if hint else ""))
    tries = max(1, int(tries))
    import wandb

    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            default = wandb.Api(timeout=timeout).default_entity
        except Exception as e:  # noqa: BLE001 — retried, then refused
            last = e
            if attempt < tries:
                _sleep(backoff_s * attempt)
            continue
        if default != ent:
            raise WandbEntityError(
                f"the W&B key's default entity is {default!r} but "
                f"WANDB_ENTITY={ent!r} — a foreign key; refusing to start "
                "rather than log to another account")
        return ent
    raise WandbEntityError(
        f"could not verify the W&B key's default entity against "
        f"WANDB_ENTITY={ent!r} after {tries} tries "
        f"({type(last).__name__}: {last}) — refusing to start rather than "
        "log with an unverified key")


def wandb_init_checked(*, project: str, name: str, config: dict | None = None,
                       tries: int = DEFAULT_TRIES,
                       backoff_s: float = DEFAULT_BACKOFF_S, **init_kwargs):
    """:func:`require_wandb_entity`, then ``wandb.init`` with THAT entity
    passed explicitly (never the key's default), retried ``tries`` times on
    errors and then raised as :class:`WandbEntityError` — a live-logging
    run that cannot open its run does not silently continue unlogged."""
    ent = require_wandb_entity(tries=tries, backoff_s=backoff_s)
    import wandb

    last: Exception | None = None
    for attempt in range(1, max(1, int(tries)) + 1):
        try:
            return wandb.init(project=project, entity=ent, name=name,
                              config=dict(config or {}), **init_kwargs)
        except Exception as e:  # noqa: BLE001 — retried, then refused
            last = e
            if attempt < tries:
                _sleep(backoff_s * attempt)
    raise WandbEntityError(
        f"wandb.init(entity={ent!r}, project={project!r}, name={name!r}) "
        f"failed {tries} times ({type(last).__name__}: {last}) — refusing to "
        "run unlogged after asking for live W&B")


#: The prefix of a purpose's neutral default project, for a site that names no project: ``sisteer-<purpose>``.
NEUTRAL_PREFIX = "sisteer-"


def _project_key(purpose: str) -> str:
    """The site key of one purpose's project: ``SISTEER_WANDB_PROJECT_<PURPOSE>``, upper-cased, ``-`` written ``_``."""
    return "SISTEER_WANDB_PROJECT_" + purpose.upper().replace("-", "_")


def wandb_project(purpose: str, *, env=None, site=None) -> str:
    """The W&B project one purpose of logging writes to (``tinycap``, ``seedrep``, ``distill``, ...):

    1. ``SISTEER_WANDB_PROJECT_<PURPOSE>``: the environment, then the site config
    2. ``WANDB_PROJECT`` in the environment
    3. ``SISTEER_WANDB_PROJECT``: the environment, then the site config
    4. ``sisteer-<purpose>``

    The purpose's own key comes first, so an exported ``WANDB_PROJECT`` never redirects a purpose the site names. A host
    with no site config file reads the environment only (a file named by ``$SISTEER_CONFIG`` and missing refuses, as
    every site read does). ``env`` defaults to ``os.environ``; ``site`` to the host's site config."""
    if not purpose or not isinstance(purpose, str):
        raise ValueError(f"wandb_project: a purpose is a non-empty string, got {purpose!r}")
    from sisteer import site as _site

    env = os.environ if env is None else env
    site = _site.load_site(env=env) if site is None else site
    got = _site.setting(_project_key(purpose), site=site, env=env)
    if got:
        return got
    if env.get("WANDB_PROJECT"):
        return env["WANDB_PROJECT"]
    return _site.setting("SISTEER_WANDB_PROJECT", site=site, env=env) or NEUTRAL_PREFIX + purpose
