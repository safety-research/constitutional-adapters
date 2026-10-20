"""``modal_owned`` — inspect-sandboxes' ``modal`` provider with an OWNER on
every sandbox.

A thin subclass of ``inspect_sandboxes.modal.ModalSandboxEnvironment``
(0.5.0), registered under the unqualified name ``modal_owned``
(``contracts.BCB_SANDBOX_PROVIDER``), which the BigCodeBench thin task passes
to the stock ``bigcodebench(sandbox=...)`` constructor. Three things differ
from the parent, and nothing else:

1. **Create-time tags, all strings.** The parent creates the sandbox and then
   ``set_tags({"created_by": "inspect-ai"})`` in a SECOND RPC — and Modal's
   ``set_tags`` REPLACES the whole tag set. So ``sample_init`` is re-implemented
   (the parent's compose handling verbatim) to pass ``tags=owner_tags(...)`` to
   ``Sandbox.create`` itself: the owner tuple ``(run_id, cell,
   function_call_id)`` read from the container's environment
   (``sandbox_scope.owner_from_env``), the task and the sample id, every
   value ``str()`` (BigCodeBench's sample ids are ints; a non-str tag value
   is a protobuf ``TypeError`` before the RPC). ``set_tags`` is never called.
   A container that declares NO owner refuses: an untagged sandbox cannot be
   swept by anyone.
2. **A 3 h lifetime** (``contracts.BCB_SANDBOX_TIMEOUT_S``), not the parent's
   24 h — the ceiling on a leak from a hard-killed container.
3. **A durable record.** Every create / observed termination appends to the
   ``sandboxes.jsonl`` named by ``TINYCAP_SANDBOX_RECORD`` (fsync'd), so the
   external watcher and the analysis can account for every sandbox without
   the process that made it. ``terminated`` is written ONLY after
   ``poll()`` returned an exit code (``verified_by: poll``) or the sweep's
   re-list no longer showed the id (``relist``); a requested-but-unconfirmed
   termination is ``terminate_unverified``; a cleanup skipped because the
   sample was interrupted is ``cleanup_skipped_interrupted``.

4. **Backoff on the sandbox-creation rate limit** (tinycap-verify-rate-limit,
   2026-09-30). Modal creates at most ``contracts.SANDBOX_CREATE_RATE_LIMIT_PER_S``
   (5) sandboxes per second per workspace and refuses the rest with
   ``modal.exception.ResourceExhaustedError`` (gRPC ``RESOURCE_EXHAUSTED``). The
   parent's ``_create_sandbox`` is wrapped by tenacity's ``_standard_retry`` —
   five attempts, waits capped at 10 s, ~15 s in all — which DOES retry that
   error (it excludes only ``RemoteError``) but gives up long before a burst of
   verifiers clears: nine verifiers x eight sandboxes on 2026-09-30 exhausted it
   at the first sample and inspect aborted every task. ``_create_sandbox`` is
   therefore OVERRIDDEN here: it calls the parent's (its transient retry kept)
   under ``sandbox_backoff.create_with_backoff`` — exponential backoff with
   jitter (1, 2, 4, … s, cap 30 s, total 5 min by default;
   ``TINYCAP_SANDBOX_CREATE_BACKOFF_*_S`` override), each retry logged AND
   appended to the record as ``create_rate_limited`` (attempt, wait, elapsed),
   the last refusal re-raised unchanged once the bound is spent. Only the rate
   limit is retried; nothing falls back to another provider.

Everything else — exec with the timeout-retry ladder, file I/O, the app lookup
(``inspect_modal_sandbox``), the create's transient retry — is the parent's,
unchanged.
"""

from __future__ import annotations

import asyncio
import os
from logging import getLogger
from typing import Any, Dict, Optional

import modal
import modal.exception

try:  # grpclib rides modal; a venv carrying a partial modal (a test stub) has none — the raw-status branch is then moot
    import grpclib.const
except ImportError:  # pragma: no cover — production always has it
    grpclib = None  # type: ignore[assignment]
from inspect_ai.util import (ComposeConfig, SandboxEnvironment, SandboxEnvironmentConfigType,
                             is_compose_yaml, is_dockerfile, parse_compose_yaml, sandboxenv,
                             trace_message)
from inspect_sandboxes._util.naming import make_sandbox_name
from inspect_sandboxes.modal._compose import _MODAL_PORT_KEYS, convert_compose_to_modal_params
from inspect_sandboxes.modal._modal import MODAL_APP_NAME, ModalSandboxEnvironment, running_sandboxes
from typing_extensions import override

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import sandbox_backoff as BK
from sisteer.capeval.tinycap import sandbox_scope as scope

logger = getLogger(__name__)

__all__ = ["OwnedModalSandboxEnvironment", "PROVIDER_NAME", "SANDBOX_APP_NAME",
           "SANDBOX_TIMEOUT_S", "current_owner", "record_path", "is_sandbox_rate_limit"]

#: the sleep the create backoff awaits — a module global so a test can replace it (``asyncio.sleep`` in production)
_backoff_sleep = asyncio.sleep


def is_sandbox_rate_limit(exc: BaseException) -> bool:
    """Modal's ``ResourceExhaustedError`` (the SDK's mapping of gRPC ``RESOURCE_EXHAUSTED``), or a raw
    ``grpclib.GRPCError`` carrying that status (the pre-mapping form) — the ONE refusal the create backoff retries."""
    if isinstance(exc, modal.exception.ResourceExhaustedError):
        return True
    if grpclib is None:
        return False
    return isinstance(exc, grpclib.GRPCError) and exc.status == grpclib.const.Status.RESOURCE_EXHAUSTED

PROVIDER_NAME = C.BCB_SANDBOX_PROVIDER
SANDBOX_APP_NAME = MODAL_APP_NAME
SANDBOX_TIMEOUT_S = int(C.BCB_SANDBOX_TIMEOUT_S)
assert SANDBOX_APP_NAME == C.BCB_SANDBOX_APP_NAME, (SANDBOX_APP_NAME, C.BCB_SANDBOX_APP_NAME)


def current_owner() -> scope.Owner:
    """The owner tuple the container declared (``TINYCAP_OWNER_*``); a run
    that declares none REFUSES here — an untagged sandbox is one nobody can
    sweep after a hard death."""
    owner = scope.owner_from_env()
    if owner is None:
        raise RuntimeError(
            f"{PROVIDER_NAME}: no sandbox owner in the environment "
            f"({', '.join(scope.OWNER_ENV_VARS)}) — the launcher sets all three for every "
            "container; a sandbox created without an owner could not be swept after a hard death, "
            "so it is not created at all")
    return owner


def record_path() -> Optional[str]:
    """The durable ``sandboxes.jsonl`` the container named, or None (a hand run)."""
    p = os.environ.get(scope.SANDBOX_RECORD_ENV)
    return p or None


@sandboxenv(name=PROVIDER_NAME)
class OwnedModalSandboxEnvironment(ModalSandboxEnvironment):
    """The parent provider with the owner tuple on every create-time tag, and a
    rate-limit backoff around the create."""

    @override
    @staticmethod
    async def _create_sandbox(command: list[str], kwargs: Dict[str, Any]) -> modal.Sandbox:
        """The parent's create (its tenacity transient retry intact) under the rate-limit backoff — see the module
        docstring's point 4. Each retry is logged and appended to the record (``create_rate_limited``); the last
        refusal is re-raised as itself once ``policy.total_s`` is spent."""
        policy = BK.policy_from_env()
        owner = scope.owner_from_env()
        record = record_path()
        name = kwargs.get("name")

        def _on_retry(attempt: int, wait: float, elapsed: float, exc: BaseException) -> None:
            msg = (f"{PROVIDER_NAME}: sandbox create {name!r} refused by the rate limit "
                   f"({C.SANDBOX_CREATE_RATE_LIMIT_PER_S}/s): {type(exc).__name__}: {str(exc)[:200]} — "
                   f"retry {attempt + 1} in {wait:.1f} s ({elapsed:.1f} s elapsed of {policy.total_s:g} s)")
            logger.warning(msg)
            trace_message(logger, "modal", msg)
            scope.append_event(record, scope.EVENT_CREATE_RATE_LIMITED, "", owner, attempt=int(attempt),
                               wait_s=round(float(wait), 3), elapsed_s=round(float(elapsed), 3), name=name,
                               error=f"{type(exc).__name__}: {str(exc)[:200]}")

        def _on_exhausted(attempts: int, elapsed: float, exc: BaseException) -> None:
            msg = (f"{PROVIDER_NAME}: sandbox create {name!r} still refused by the rate limit after {attempts} retr"
                   f"{'y' if attempts == 1 else 'ies'} and {elapsed:.1f} s (bound {policy.total_s:g} s) — giving up: "
                   f"{type(exc).__name__}: {str(exc)[:200]}")
            logger.error(msg)
            trace_message(logger, "modal", msg)

        return await BK.create_with_backoff(ModalSandboxEnvironment._create_sandbox, command, kwargs, policy=policy,
                                            rate_limited=is_sandbox_rate_limit, sleep=_backoff_sleep,
                                            on_retry=_on_retry, on_exhausted=_on_exhausted)

    @override
    @classmethod
    async def sample_init(cls, task_name: str, config: SandboxEnvironmentConfigType | None,
                          metadata: dict[str, str]) -> dict[str, SandboxEnvironment]:
        owner = current_owner()                       # refuse BEFORE any RPC
        app = await cls._lookup_app(SANDBOX_APP_NAME)
        # -- the parent's compose handling, verbatim (0.5.0 _modal.py sample_init)
        sandbox_kwargs: Dict[str, Any] = {
            "app": app,
            "name": make_sandbox_name(task_name, metadata),
            "timeout": SANDBOX_TIMEOUT_S,
        }
        command: list[str] = []
        modal_params = None
        if config is None:
            trace_message(logger, "modal", f"Using default Modal image for task '{task_name}'")
        elif is_dockerfile(config):
            sandbox_kwargs["image"] = modal.Image.from_dockerfile(config)
        elif is_compose_yaml(config):
            compose_config = parse_compose_yaml(config, multiple_services=False)
            modal_params = convert_compose_to_modal_params(compose_config, config)
            command = modal_params.command
            sandbox_kwargs.update(modal_params.kwargs)
        elif isinstance(config, ComposeConfig):
            modal_params = convert_compose_to_modal_params(config, None)
            command = modal_params.command
            sandbox_kwargs.update(modal_params.kwargs)
        else:
            raise ValueError(f"Unrecognized config: {config}. Expected a compose file (*.yaml/*.yml), "
                             "Dockerfile, ComposeConfig object, or None.")
        if modal_params is not None and modal_params.volumes:
            raise ValueError(f"{PROVIDER_NAME}: the compose declares Modal volumes — a BigCodeBench "
                             "sandbox mounts nothing (spec: no secrets or volumes in sandboxes)")
        if sandbox_kwargs.get("secrets"):
            raise ValueError(f"{PROVIDER_NAME}: the compose declares secrets — refused")
        # -- the ONE difference at create: the owner tuple in the create-time
        #    tags (all str), and the 3 h lifetime; never a second set_tags RPC
        sandbox_kwargs["tags"] = scope.owner_tags(owner, task_name, metadata)
        sandbox_kwargs["timeout"] = SANDBOX_TIMEOUT_S
        sandbox = await cls._create_sandbox(command, sandbox_kwargs)
        running_sandboxes().append(sandbox.object_id)
        scope.append_event(record_path(), scope.EVENT_CREATED, sandbox.object_id, owner,
                           task=str(task_name), sample_id=sandbox_kwargs["tags"][scope.TAG_SAMPLE_ID],
                           name=sandbox_kwargs["name"], timeout_s=SANDBOX_TIMEOUT_S,
                           app=SANDBOX_APP_NAME,
                           image=str(getattr(config, "name", config)) if config is not None else None)
        has_tunnels = any(sandbox_kwargs.get(key) for key in _MODAL_PORT_KEYS)
        return {"default": cls(sandbox, has_tunnels=has_tunnels)}

    @override
    @classmethod
    async def sample_cleanup(cls, task_name: str, config: SandboxEnvironmentConfigType | None,
                             environments: dict[str, SandboxEnvironment], interrupted: bool) -> None:
        owner = scope.owner_from_env()
        if not environments:
            return
        if interrupted:
            # the parent returns early too; the difference is that the skip is
            # RECORDED (task_cleanup / the sweep owns these ids now)
            for env in environments.values():
                sid = cls._get_sandbox_id(getattr(env, "sandbox", None))
                scope.append_event(record_path(), scope.EVENT_CLEANUP_SKIPPED_INTERRUPTED, sid, owner,
                                   task=str(task_name))
            return
        for env in environments.values():
            sandbox = None
            try:
                sandbox = env.as_type(ModalSandboxEnvironment).sandbox
                await cls._terminate_sandbox(sandbox)      # terminate() + poll() != None, retried
            except Exception as e:                          # noqa: BLE001 — the parent's contract
                sid = cls._get_sandbox_id(sandbox)
                scope.append_event(record_path(), scope.EVENT_TERMINATE_UNVERIFIED, sid, owner,
                                   task=str(task_name), error=f"{type(e).__name__}: {str(e)[:200]}")
                trace_message(logger, "modal",
                              f"Error terminating Modal sandbox {sid} for task '{task_name}': {e}. "
                              "Will retry in task_cleanup.")
            else:
                # _terminate_sandbox RAISES while poll() is None, so reaching
                # here means termination was OBSERVED
                scope.append_event(record_path(), scope.EVENT_TERMINATED, sandbox.object_id, owner,
                                   verified_by="poll", task=str(task_name))

    @override
    @classmethod
    async def task_cleanup(cls, task_name: str, config: SandboxEnvironmentConfigType | None,
                           cleanup: bool) -> None:
        """The parent's task-level cleanup over the ContextVar list, each
        observed termination recorded."""
        if not cleanup:
            return
        owner = scope.owner_from_env()
        failed: list[str] = []
        for sandbox_id in running_sandboxes().copy():
            try:
                sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
                await cls._terminate_sandbox(sandbox)
                scope.append_event(record_path(), scope.EVENT_TERMINATED, sandbox_id, owner,
                                   verified_by="poll", task=str(task_name), by="task_cleanup")
            except Exception as e:                          # noqa: BLE001
                failed.append(sandbox_id)
                scope.append_event(record_path(), scope.EVENT_TERMINATE_UNVERIFIED, sandbox_id, owner,
                                   task=str(task_name), by="task_cleanup",
                                   error=f"{type(e).__name__}: {str(e)[:200]}")
                logger.error(f"Failed to terminate sandbox {sandbox_id}: {e}")
        if failed:
            logger.warning(f"Failed to cleanup {len(failed)} sandbox(es). Failed IDs: {', '.join(failed)}")
        running_sandboxes().clear()
