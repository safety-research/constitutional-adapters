"""The inspect_ai TARGET-role client wiring.

The audit's target role is an OpenAI-compatible client pointed at the
routing proxy's ``base_url`` over the SSH tunnel: inspect_ai runs with
``--model-role target=openai/<vmodel>`` and ``OPENAI_BASE_URL`` set to
the tunnelled localhost port. This module builds that wiring —
the model string, the env mapping for the inspect process, and (when the
``petri`` group is installed) the AsyncOpenAI client itself.

Containment stance: the proxy has NO open port — it is reached through an
SSH tunnel, so the client-side base_url is a LOOPBACK address. A
non-loopback URL is refused unless the caller explicitly says it is
cluster-internal (the on-cluster gate hitting the compute node directly),
so a mis-set env var can never quietly send audit traffic over an open
network path.

``openai`` / ``inspect_ai`` are petri-group (grading-host) deps — both are
import-guarded; everything else here is stdlib and CPU-tested.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Dict
from urllib.parse import urlparse

from sisteer.capeval.petri.defense_cells import VMODEL_PREFIX


class TargetClientError(RuntimeError):
    """A target-wiring refusal — loud, never a silent default."""


#: The dummy API key the proxy ignores (or checks as a shared token —
#: RoutingProxy(api_token=…)); inspect's openai provider requires SOME key.
DEFAULT_API_KEY = "petri-proxy"


@dataclass(frozen=True)
class TargetClientConfig:
    """One cell's target-role client wiring."""

    virtual_model: str  # petri__<...>
    base_url: str  # http://localhost:<tunnelport>/v1
    api_key: str = DEFAULT_API_KEY

    @property
    def inspect_model(self) -> str:
        """The inspect_ai model string for --model-role target."""
        return f"openai/{self.virtual_model}"

    def env(self) -> Dict[str, str]:
        """The env for the inspect process."""
        return {"OPENAI_BASE_URL": self.base_url,
                "OPENAI_API_KEY": self.api_key}


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def target_client_config(virtual_model: str, base_url: str, *,
                         api_key: str = DEFAULT_API_KEY,
                         allow_cluster_internal: bool = False
                         ) -> TargetClientConfig:
    """Build + validate one cell's target wiring.

    ``allow_cluster_internal=True`` is for the ON-CLUSTER gate/smoke
    hitting the compute node's internal interface directly; the grading host
    audit path always goes through the tunnel (loopback)."""
    if (not isinstance(virtual_model, str)
            or not virtual_model.startswith(VMODEL_PREFIX)):
        raise TargetClientError(
            f"virtual model {virtual_model!r} does not carry the "
            f"{VMODEL_PREFIX!r} prefix — the target role only ever points "
            "at proxy virtual models (one vmodel per served leg)")
    # canonicalize to exactly ONE trailing /v1: the ProxyClientEngine's
    # base_url already ends in /v1 (the canonical OpenAI form), and the
    # runner's legacy wiring appends another — collapse the duplicate
    # rather than shipping '/v1/v1' to the openai client.
    base = str(base_url).rstrip("/")
    while base.endswith("/v1/v1"):
        base = base[: -len("/v1")]
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise TargetClientError(f"base_url {base_url!r} is not a valid "
                                "http(s) URL")
    if not parsed.path.endswith("/v1"):
        raise TargetClientError(
            f"base_url {base_url!r} must end in /v1 (the OpenAI-compatible "
            "root the openai client expects)")
    if not _is_loopback(parsed.hostname) and not allow_cluster_internal:
        raise TargetClientError(
            f"base_url host {parsed.hostname!r} is not loopback — the "
            "proxy is reached over the SSH tunnel (D0 constraint 5); pass "
            "allow_cluster_internal=True only for on-cluster gate/smoke "
            "traffic")
    return TargetClientConfig(virtual_model=virtual_model,
                              base_url=base, api_key=api_key)


def make_async_client(config: TargetClientConfig) -> Any:
    """An ``openai.AsyncOpenAI`` client over the config (petri-group
    venv only — import-guarded)."""
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise TargetClientError(
            "the openai package is not installed in this venv — the "
            "target client runs in the grading-host petri group "
            "(pyproject `petri`), not the CPU test venv") from e
    return AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
