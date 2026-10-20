"""Inspect-model auditor fallback chain for Petri.

The auditor is an instrument, not a harm oracle. A provider-side refusal is
therefore never converted into a score. When a call fails while sibling calls
to the same model are completing, :class:`LiveSiblingDiscriminator` classifies
that failure as content-correlated and :class:`AuditorFallbackController`
reissues the *same* generate call, with the full message list and unchanged
system prompt/tools/config, to the next configured auditor. Outages and
undetermined failures do not switch models.

The selected leg persists for the rest of that sample. Provenance is written to
Inspect's per-sample store under :data:`AUDITOR_CHAIN_STORE_KEY`, so the eval log
records every switch without re-running or replacing the sample. The default
chain is Sonnet 5 -> Grok 4.6 -> the local abliterated Qwen.

UNVALIDATED PREMISE: Grok 4.6 will accept the bodies that made Sonnet 5 shed.
The adjacent multi-turn attacker measurement does not establish that premise:
Grok was only 6.4% compliant there and about 40% of its outputs were
JSON-wrapped refusals that the attacker harness forwarded. Auditor and attacker
are different roles, and carrying a transcript may behave differently from
creating adversarial turns, but the observed refusal base rate is high enough
that Grok may refuse the same content and merely relabel the shed. Before this
leg carries inferential load, replay a small set of known Sonnet-shed bodies
through Grok as the auditor and report its acceptance rate. Until then, Grok
and Qwen continuations are reported as different, unvalidated instruments,
never interchangeable rollouts.

The OpenRouter leg's credential is never a file.
:func:`resolve_openrouter_api_key` takes it from the environment
(``OPENROUTER_API_KEY``, which the shell's vault bootstrap exports) or else
asks the site's secret-manager CLI (``$SISTEER_SECRET_CMD``); a key that is in neither
place is a typed error, never a silent empty string.

This module stays CPU-importable. Inspect is imported only by
:func:`make_inspect_auditor_chain`, preserving the Petri driver's host-only
optional-dependency boundary.
"""

from __future__ import annotations

import copy
import os
import subprocess
import time
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

AUDITOR_CHAIN_STORE_KEY = "petri_auditor_chain"
#: the JUDGE role's chain record, one key per role so the two provenance records never collide in one
#: sample store (Grok 4.6 is the fallback for BOTH roles). Same shape as the auditor's.
JUDGE_CHAIN_STORE_KEY = "petri_judge_chain"
#: inspect stop reasons that ARE a refusal returned as output rather than raised: the Anthropic provider
#: hands a safety-classifier decline back as ``stop_reason == "content_filter"`` (its stop_details.type is
#: "refusal"), and inspect_scout.generate_answer -- the judge's call path -- raises RefusalError on exactly
#: that stop reason. A chain built with ``switch_on_refusal_output`` treats it as a content-correlated
#: failure of that leg (no discriminator needed: a refusal is per-content by construction) and moves on.
REFUSAL_STOP_REASONS = ("content_filter",)
#: the first-party provider of the per-request refusal fallback's Claude legs; a leg OFF it is the terminal non-Claude judge
#: (Grok 4.6 over OpenRouter) and gets the non-answer rule below
FIRST_PARTY_PROVIDER = "anthropic/"
#: the terminal non-Claude leg's non-answer evidence, on the served output's metadata and its refusal attempt record
NON_ANSWER_KEY = "non_answer"
#: inspect's provider prefixes a model id may carry (``<provider>/<model>``): stripped ONCE by :func:`bare_model_id`, so an id
#: whose MODEL part has a slash of its own (OpenRouter's ``x-ai/grok-4.6``) reads the same whether recorded as
#: ``openrouter/x-ai/grok-4.6`` (the manifest's spelling) or ``x-ai/grok-4.6`` (the response's ``model``). ONE source for the
#: runner (petri_runner._bare_model_id) and this wrapper.
INSPECT_PROVIDER_PREFIXES: Tuple[str, ...] = (
    "anthropic/", "openrouter/", "openai/", "openai-api/", "google/", "vertex/", "bedrock/", "azureai/", "mistral/", "together/",
    "groq/", "fireworks/", "perplexity/", "cloudflare/", "hf/", "vllm/", "sglang/", "ollama/", "llama-cpp-python/", "grok/", "mockllm/")
#: a DATED BUILD suffix on a model id (OpenRouter's canonical slug ``x-ai/grok-4.6-20260810``; Anthropic's snapshots): the id
#: without it is the build's ALIAS. The alias and its pinned build are ONE judge for provenance and attribution.
BUILD_SUFFIX_RE = re.compile(r"-\d{8}$")
#: the terminal non-Claude leg's build attestation on the hand-off record
BUILD_ATTESTATION_KEY = "build_attestation"
#: the attestation CLASSES (Grok-fix-3). `exact`: the provider's response names the pinned dated build itself --
#: the ONLY class that is a VERIFIED build. `declared_unverified`: the request pinned the dated slug and the response echoed the
#: mutable alias or named nothing -- declared-only; the build is NOT attested by the provider, and nothing downstream may call such a
#: judgment an attested immutable build (OpenRouter's observed behaviour for every dated request, 2026-10-01). `mismatch`: the response
#: names another model or another build -- RefusalFallbackBuildError, a hard error, never a score.
BUILD_CLASS_EXACT = "exact"
BUILD_CLASS_DECLARED = "declared_unverified"
BUILD_CLASS_MISMATCH = "mismatch"
BUILD_CLASSES = (BUILD_CLASS_EXACT, BUILD_CLASS_DECLARED, BUILD_CLASS_MISMATCH)


def bare_model_id(name: Any) -> str:
    """``anthropic/claude-opus-5`` -> ``claude-opus-5``; ``openrouter/x-ai/grok-4.6`` AND ``x-ai/grok-4.6`` -> ``x-ai/grok-4.6``;
    a bare id is returned as is. Only an inspect PROVIDER prefix is stripped: a slash inside the model part is the model's."""
    text = str(name)
    for prefix in INSPECT_PROVIDER_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def model_build_family(name: Any) -> str:
    """The judge FAMILY of a model id: its bare id without one trailing dated-build suffix --
    ``openrouter/x-ai/grok-4.6-20260810``, ``x-ai/grok-4.6-20260810`` and ``x-ai/grok-4.6`` all -> ``x-ai/grok-4.6``;
    ``claude-opus-4-8`` -> itself. The declared alias and the pinned dated build are the SAME judge in provenance
    (refusal_fallback_allowed) and attribution (refusing_judge_model); the served id itself is kept raw as provenance."""
    return BUILD_SUFFIX_RE.sub("", bare_model_id(name))


def is_dated_build(name: Any) -> bool:
    """Whether a model id names an IMMUTABLE dated build (``…-YYYYMMDD``) rather than a mutable alias."""
    return BUILD_SUFFIX_RE.search(bare_model_id(name)) is not None
#: the key under which a CLIENT-side per-request refusal fallback (:func:`make_refusal_fallback_model`) stamps the
#: hand-off on the served output's ``metadata``: the requested (default) model, the model that served, every refused
#: attempt's stop evidence, and ``client_side: True``
REFUSAL_FALLBACK_SERVED_KEY = "petri_refusal_fallback"
#: the LEADING marker of a :class:`RefusalFallbackLegError`'s text: the converter
#: (petri_runner.ladder_exhaustion) reads it off the sample error and the chain's leg-failure text and classifies
#: the sample a HARD error whatever the status — a declared fallback leg has no ladder of its own (one re-issue)
FALLBACK_LEG_ERROR_MARKER = "refusal-fallback leg error"
#: the chain record's ``final_failure.kind`` for a sample a fallback leg's own error ended: not content, never a switch
FALLBACK_LEG_HARD_ERROR_KIND = "hard_error"
_HTTP_STATUS_IN_TEXT = re.compile(r"(?:error code|status(?: code)?|http|apistatuserror|statuscode)[:=\s]+(\d{3})\b", re.IGNORECASE)
_ERROR_TYPE_IN_TEXT = re.compile(r"\\*['\"]type\\*['\"]\s*:\s*\\*['\"]([a-z_]+_error)\\*['\"]")
DEFAULT_GROK_MODEL = "openrouter/x-ai/grok-4.6"
DEFAULT_QWEN_MODEL = "openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated"
#: The abliterated-Qwen leg was REMOVED from the default chain 2026-09-15: across every audit on
#: disk -- 231 cells, 21,399 rows carrying an auditor_chain record -- it was reached ZERO times, while grok-4.6
#: took 27 rows (9 conversations, all content_correlated refusals). Keeping it cost a GPU: the driver refused to
#: start unless an OpenAI-compatible endpoint served it, and the documented way to provide one was a tunnel to a
#: live attacker-pool member, so a pool had to exist for a leg that never ran. DEFAULT_QWEN_MODEL and the
#: leg-building branch below are kept so archived auditor_chain records still parse and classify.
DEFAULT_FALLBACK_MODELS = (DEFAULT_GROK_MODEL,)
#: where the OpenRouter key comes from, in order: this environment variable
#: (exported from the vault by the shell bootstrap), else the vault item below
#: read through the secret-manager CLI (bounded by the timeout). There is no file.
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_VAULT_ITEM = "OpenRouter API Key"
SECRET_CMD_TIMEOUT_S = 30.0
QWEN_BASE_URL_ENV = "PETRI_QWEN_AUDITOR_BASE_URL"
QWEN_API_KEY_ENV = "PETRI_QWEN_AUDITOR_API_KEY"

CHAIN_SONNET_ONLY = "sonnet_only"
CHAIN_CONTINUED_GROK = "continued_by_grok"
CHAIN_CONTINUED_QWEN = "continued_by_qwen"
CHAIN_REFUSED_ALL = "auditor_refused_all_legs"
CHAIN_UNAVAILABLE = "auditor_unavailable"
CHAIN_KINDS = (
    CHAIN_SONNET_ONLY,
    CHAIN_CONTINUED_GROK,
    CHAIN_CONTINUED_QWEN,
    CHAIN_REFUSED_ALL,
    CHAIN_UNAVAILABLE,
)


class AuditorFallbackError(ValueError):
    """A malformed chain, missing local endpoint, or corrupt provenance."""


class AuditorSecretError(AuditorFallbackError):
    """The OpenRouter key is in neither the environment nor the vault."""


@dataclass(frozen=True)
class AuditorLeg:
    """One configured auditor model and its concise report label."""

    model: Any
    model_name: str
    label: str


@dataclass(frozen=True)
class LiveDiscriminatorEvidence:
    """Sibling evidence available when one generate call exhausts.

    Interval comparisons use monotonic timestamps. ``window_start`` /
    ``window_end`` and sibling timestamp tuples are wall-clock values retained
    only for human-readable provenance.
    """

    kind: str
    model: str
    window_start: float
    window_end: float
    monotonic_start: float
    monotonic_end: float
    siblings_succeeded: int
    siblings_failed: int
    sibling_success_timestamps: Tuple[float, ...]
    sibling_failure_timestamps: Tuple[float, ...]
    basis: str
    window_start_iso: str = ""
    window_end_iso: str = ""


class LiveSiblingDiscriminator:
    """Process-local live discriminator shared by all copies of one role.

    A success counts only when it belongs to another sample and completed
    inside the failed call's monotonic interval. A failed sibling counts as
    outage evidence only when its own exception independently classified as an
    outage; an overlapping content-correlated or undetermined failure is not
    promoted into outage evidence. The operations contain no awaits, so updates
    are atomic with respect to Inspect's asyncio event loop.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        # ``clock`` remains injectable for CPU tests; the controller supplies
        # timestamps explicitly and production defaults to monotonic time.
        self.clock = clock
        self._successes: Dict[
            str, list[tuple[str, float, float]]
        ] = {}
        self._failures: Dict[
            str, list[tuple[str, float, float, str, float]]
        ] = {}

    def record_success(
        self, model: str, sample_key: str, completed: float,
        wall_completed: Optional[float] = None,
    ) -> None:
        self._successes.setdefault(model, []).append(
            (sample_key, completed,
             completed if wall_completed is None else wall_completed))

    def classify_failure(
        self, model: str, sample_key: str, started: float, completed: float,
        *, wall_started: Optional[float] = None,
        wall_completed: Optional[float] = None,
        independent_outage: bool = False,
    ) -> LiveDiscriminatorEvidence:
        wall_started = started if wall_started is None else wall_started
        wall_completed = completed if wall_completed is None else wall_completed
        successes = sorted(
            (mono, wall) for key, mono, wall
            in self._successes.get(model, ())
            if key != sample_key and started <= mono <= completed
        )
        failures = sorted(
            (other_start, other_wall) for (
                key, other_start, other_end, other_kind, other_wall
            ) in self._failures.get(model, ())
            if key != sample_key and other_kind == "outage"
            and other_start <= completed and other_end >= started
        )
        if successes:
            kind = "content_correlated"
            basis = (
                f"{len(successes)} sibling call(s) completed while this call "
                "kept failing; the model pool was serving"
            )
        elif independent_outage or failures:
            kind = "outage"
            basis = (
                "the failed call carried independent outage evidence"
                if independent_outage else
                "no sibling completed and independently outage-typed "
                f"sibling call(s) overlapped: {len(failures)}"
            )
        else:
            kind = "undetermined"
            basis = (
                "no sibling completion or independently outage-typed "
                "overlapping sibling was visible when the call ended"
            )
        self._failures.setdefault(model, []).append(
            (sample_key, started, completed, kind, wall_started)
        )
        return LiveDiscriminatorEvidence(
            kind=kind,
            model=model,
            window_start=wall_started,
            window_end=wall_completed,
            monotonic_start=started,
            monotonic_end=completed,
            siblings_succeeded=len(successes),
            siblings_failed=len(failures),
            sibling_success_timestamps=tuple(wall for _, wall in successes),
            sibling_failure_timestamps=tuple(wall for _, wall in failures),
            basis=basis,
            window_start_iso=datetime.fromtimestamp(
                wall_started, tz=timezone.utc).isoformat(),
            window_end_iso=datetime.fromtimestamp(
                wall_completed, tz=timezone.utc).isoformat(),
        )


def _exception_is_independent_outage(exc: Exception) -> bool:
    """Conservative live outage signature; content-like overloads excluded."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "overloaded" in text or "status 529" in text or "error code: 529" in text:
        return False
    markers = (
        "ratelimit", "rate limit", "status 429", "error code: 429",
        "connection error", "connection reset", "connection refused",
        "connect timeout", "read timeout", "timed out",
        "status 502", "status 503", "status 504",
        "error code: 502", "error code: 503", "error code: 504",
        "service unavailable", "bad gateway", "gateway timeout",
    )
    return any(marker in text for marker in markers)


def auditor_label(model_name: str) -> str:
    """Stable concise label used in provenance and reports."""
    low = model_name.lower()
    if "sonnet-5" in low:
        return "sonnet-5"
    if "grok-4.6" in low:
        return "grok-4.6"
    if "opus-5" in low:
        return "opus-5"
    if "opus-4-8" in low:
        return "opus-4-8"
    if "qwen" in low and "abliterated" in low:
        return "qwen-abl"
    return model_name.split("/", 1)[-1][:48]


def _store_get(store: Any, key: str) -> Any:
    return store.get(key) if hasattr(store, "get") else None


def _store_set(store: Any, key: str, value: Any) -> None:
    if hasattr(store, "set"):
        store.set(key, value)
    elif isinstance(store, MutableMapping):
        store[key] = value
    else:
        raise AuditorFallbackError("sample store is not writable")


def _display_chain(used: Sequence[Mapping[str, Any]]) -> str:
    if not used:
        return ""
    text = str(used[0]["label"])
    for leg in used[1:]:
        text += f" → {leg['label']} @turn {int(leg['switch_turn'])}"
    return text


def chain_kind(chain: Optional[Mapping[str, Any]], *, refused: bool = False) -> str:
    """Mutually-exclusive report bucket for one sample's provenance."""
    if not chain:
        return CHAIN_UNAVAILABLE if refused else CHAIN_SONNET_ONLY
    used = list(chain.get("used") or ())
    labels = [str(x.get("label", "")) for x in used if isinstance(x, Mapping)]
    final = chain.get("final_failure")
    if refused:
        if isinstance(final, Mapping) and final.get("kind") == "content_correlated" \
                and len(used) == len(chain.get("configured") or ()):
            return CHAIN_REFUSED_ALL
        return CHAIN_UNAVAILABLE
    if any(label == "qwen-abl" for label in labels):
        return CHAIN_CONTINUED_QWEN
    if any(label == "grok-4.6" for label in labels):
        return CHAIN_CONTINUED_GROK
    return CHAIN_SONNET_ONLY


class AuditorFallbackController:
    """Provider-agnostic control plane used by the Inspect model wrapper."""

    def __init__(
        self,
        legs: Sequence[AuditorLeg],
        *,
        discriminator: Optional[LiveSiblingDiscriminator] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        state_getter: Optional[Callable[[], tuple[str, Any]]] = None,
        role: str = "auditor",
        store_key: str = AUDITOR_CHAIN_STORE_KEY,
        switch_on_refusal_output: bool = False,
    ) -> None:
        if not legs:
            raise AuditorFallbackError(f"{role} chain must contain a primary leg")
        names = [leg.model_name for leg in legs]
        if len(names) != len(set(names)):
            raise AuditorFallbackError(f"{role} chain repeats a model: {names}")
        self.legs = tuple(legs)
        #: which role this chain serves and under which store key its provenance record lives. The auditor's
        #: defaults are byte-identical to the pre-2026-09-19 controller; the judge chain (make_inspect_judge_chain)
        #: names its own key so the two records never collide in one sample store.
        self.role = role
        self.store_key = store_key
        #: switch legs when a leg RETURNS a refusal (REFUSAL_STOP_REASONS) instead of raising. OFF for the auditor
        #: (its instrument is unchanged: a returned refusal is passed through as before, and only a raised,
        #: content-correlated failure switches). ON for the judge, whose stalls are exactly these returned
        #: refusals: claude-opus-5 declining a transcript with a safety-classifier refusal, which
        #: inspect_scout.generate_answer turns into a terminal RefusalError -- with nowhere to fall back to.
        self.switch_on_refusal_output = switch_on_refusal_output
        self.clock = clock
        self.wall_clock = wall_clock
        self.discriminator = discriminator or LiveSiblingDiscriminator(clock=clock)
        self.state_getter = state_getter or _inspect_sample_state

    def _initial_state(self) -> Dict[str, Any]:
        configured = [
            {"model": leg.model_name, "label": leg.label} for leg in self.legs
        ]
        return {
            "version": 1,
            "configured": configured,
            "current_index": 0,
            "auditor_turns": 0,
            "used": [{**configured[0], "switch_turn": 1}],
            "switches": [],
            "display": self.legs[0].label,
        }

    def _state(self, store: Any) -> Dict[str, Any]:
        state = _store_get(store, self.store_key)
        if state is None:
            state = self._initial_state()
            _store_set(store, self.store_key, state)
            return state
        if not isinstance(state, Mapping):
            raise AuditorFallbackError(f"recorded {self.store_key} is not a mapping")
        state = copy.deepcopy(dict(state))
        expected = self._initial_state()["configured"]
        if state.get("configured") != expected:
            raise AuditorFallbackError(
                f"recorded {self.store_key} configuration differs from this run"
            )
        return state

    async def generate(
        self,
        input: Any,
        tools: Any = (),
        tool_choice: Any = None,
        config: Any = None,
        cache: Any = None,
    ) -> Any:
        """Generate one auditor turn, switching only on a content refusal.

        Every fallback receives the identical ``input``, ``tools``,
        ``tool_choice`` and ``config`` objects. The selected fallback remains
        current on later turns of this same sample.
        """
        sample_key, store = self.state_getter()
        state = self._state(store)
        turn = int(state.get("auditor_turns", 0)) + 1
        state["auditor_turns"] = turn
        index = int(state.get("current_index", 0))

        while True:
            leg = self.legs[index]
            started = self.clock()
            wall_started = self.wall_clock()
            try:
                kwargs = {"tools": tools, "tool_choice": tool_choice}
                if config is not None:
                    kwargs["config"] = config
                if cache is not None:
                    kwargs["cache"] = cache
                output = await leg.model.generate(input, **kwargs)
            except Exception as exc:
                completed = self.clock()
                wall_completed = self.wall_clock()
                evidence = self.discriminator.classify_failure(
                    leg.model_name, sample_key, started, completed,
                    wall_started=wall_started, wall_completed=wall_completed,
                    independent_outage=_exception_is_independent_outage(exc),
                )
                attempt = {
                    "model": leg.model_name,
                    "label": leg.label,
                    "turn": turn,
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    "discriminator": asdict(evidence),
                }
                # a declared per-request fallback leg's OWN error (RefusalFallbackLegError) is transport/HTTP,
                # not content — it names the leg that failed and never earns a sticky switch, whatever the primary's
                # siblings did; the sample's error reaches the runner's hard-error ledger (never a Grok-completed
                # sample, never a scored result)
                leg_error = isinstance(exc, RefusalFallbackLegError)
                if leg_error:
                    attempt["fallback_leg"] = exc.leg_model
                    attempt["hard_error"] = True
                    attempt["status"] = exc.status
                state.setdefault("failures", []).append(attempt)
                can_switch = (
                    not leg_error
                    and evidence.kind == "content_correlated"
                    and index + 1 < len(self.legs)
                )
                if not can_switch:
                    state["current_index"] = index
                    state["final_failure"] = {
                        **attempt,
                        "kind": FALLBACK_LEG_HARD_ERROR_KIND if leg_error else evidence.kind,
                        "all_legs_exhausted": (
                            not leg_error
                            and evidence.kind == "content_correlated"
                            and index + 1 == len(self.legs)
                        ),
                    }
                    state["display"] = _display_chain(state.get("used") or ())
                    _store_set(store, self.store_key, state)
                    raise

                next_index = index + 1
                next_leg = self.legs[next_index]
                switch = {
                    "from": leg.label,
                    "to": next_leg.label,
                    "turn": turn,
                    "reason": "content_correlated",
                    "discriminator": asdict(evidence),
                }
                state.setdefault("switches", []).append(switch)
                state.setdefault("used", []).append({
                    "model": next_leg.model_name,
                    "label": next_leg.label,
                    "switch_turn": turn,
                })
                index = next_index
                state["current_index"] = index
                state.pop("final_failure", None)
                state["display"] = _display_chain(state["used"])
                _store_set(store, self.store_key, state)
                continue
            else:
                completed = self.clock()
                wall_completed = self.wall_clock()
                stop_reason = getattr(output, "stop_reason", None)
                if (self.switch_on_refusal_output
                        and stop_reason in REFUSAL_STOP_REASONS):
                    # a refusal RETURNED by this leg (the judge's failure mode): content-correlated by
                    # construction, so switch without consulting the discriminator; the last leg's refusal is
                    # returned as-is (the caller files it as a terminal refusal exactly as today), recorded.
                    attempt = {
                        "model": leg.model_name,
                        "label": leg.label,
                        "turn": turn,
                        "error": f"refusal_output: stop_reason={stop_reason}",
                        "discriminator": None,
                    }
                    state.setdefault("failures", []).append(attempt)
                    if index + 1 < len(self.legs):
                        next_leg = self.legs[index + 1]
                        state.setdefault("switches", []).append({
                            "from": leg.label,
                            "to": next_leg.label,
                            "turn": turn,
                            "reason": "refusal_output",
                            "discriminator": None,
                        })
                        state.setdefault("used", []).append({
                            "model": next_leg.model_name,
                            "label": next_leg.label,
                            "switch_turn": turn,
                        })
                        index += 1
                        state["current_index"] = index
                        state.pop("final_failure", None)
                        state["display"] = _display_chain(state["used"])
                        _store_set(store, self.store_key, state)
                        continue
                    state["current_index"] = index
                    state["final_failure"] = {
                        **attempt, "kind": "content_correlated",
                        "all_legs_exhausted": True,
                    }
                    state["display"] = _display_chain(state.get("used") or ())
                    _store_set(store, self.store_key, state)
                    metadata = dict(getattr(output, "metadata", None) or {})
                    metadata[self.store_key] = copy.deepcopy(state)
                    try:
                        output.metadata = metadata
                    except Exception:
                        pass
                    return output
                self.discriminator.record_success(
                    leg.model_name, sample_key, completed, wall_completed
                )
                state["current_index"] = index
                state.pop("final_failure", None)
                state["display"] = _display_chain(state.get("used") or ())
                _store_set(store, self.store_key, state)
                metadata = dict(getattr(output, "metadata", None) or {})
                metadata[self.store_key] = copy.deepcopy(state)
                try:
                    output.metadata = metadata
                except Exception:
                    pass
                return output


def _inspect_sample_state() -> tuple[str, Any]:
    from inspect_ai.solver._task_state import sample_state

    state = sample_state()
    if state is None:
        raise AuditorFallbackError(
            "auditor fallback model was called outside an Inspect sample"
        )
    return str(state.uuid), state.store


def resolve_openrouter_api_key(*, timeout: float = SECRET_CMD_TIMEOUT_S) -> str:
    """The OpenRouter API key from :func:`sisteer.vault.openrouter_api_key` —
    the environment variable when set and non-empty, else the vault item
    through the secret-manager CLI. Never a file, never logged.

    Raises :class:`AuditorSecretError` when neither source yields a key, so
    this module's callers keep catching one auditor-shaped exception. The
    resolution itself lives in :mod:`sisteer.vault` so the timeout, the
    message discipline (name the sources, never the value) and the
    no-chained-traceback rule are decided in ONE place for every credential
    in the repository.
    """
    from sisteer import vault

    try:
        return vault.resolve_secret(env=OPENROUTER_API_KEY_ENV,
                                    item=OPENROUTER_VAULT_ITEM,
                                    timeout=timeout)
    except vault.VaultSecretError as exc:
        raise AuditorSecretError(str(exc)) from None


def resolve_inspect_auditor_models(
    primary_model: str,
    fallback_models: Sequence[str],
    *,
    role: str = "auditor",
) -> Tuple[AuditorLeg, ...]:
    """Resolve configured model strings without exporting provider secrets.
    ``role`` is the inspect model role every leg is stamped with ("auditor",
    or "judge" for :func:`make_inspect_judge_chain`).

    The OpenRouter leg's key is passed to ``get_model`` in-process from
    :func:`resolve_openrouter_api_key`; nothing is written to the environment
    or to disk.
    """
    from inspect_ai.model import get_model

    legs: list[AuditorLeg] = []
    for model_name in (primary_model, *fallback_models):
        kwargs: Dict[str, Any] = {"memoize": False}
        if model_name.startswith("openrouter/"):
            kwargs["api_key"] = resolve_openrouter_api_key()
        elif model_name == DEFAULT_QWEN_MODEL:
            base_url = os.environ.get(QWEN_BASE_URL_ENV, "").strip()
            if not base_url:
                raise AuditorFallbackError(
                    f"{DEFAULT_QWEN_MODEL} needs {QWEN_BASE_URL_ENV} pointing "
                    "at its local OpenAI-compatible endpoint"
                )
            kwargs["base_url"] = base_url.rstrip("/")
            kwargs["api_key"] = os.environ.get(
                QWEN_API_KEY_ENV, "petri-local"
            )
            kwargs["responses_api"] = False
        model = get_model(model_name, **kwargs)
        model._set_role(role)
        legs.append(AuditorLeg(model, model_name, auditor_label(model_name)))
    return tuple(legs)


def is_refusal_output(output: Any) -> bool:
    """A refusal RETURNED as output (:data:`REFUSAL_STOP_REASONS`): the safety-classifier decline the Anthropic provider
    maps to ``stop_reason == "content_filter"`` (``stop_details.type == "refusal"`` on Claude 4.7+), and its own
    content-filtering error form (the same stop reason, no stop_details)."""
    return getattr(output, "stop_reason", None) in REFUSAL_STOP_REASONS


def _stop_details_record(output: Any) -> Optional[Dict[str, Any]]:
    choices = getattr(output, "choices", None) or []
    details = getattr(choices[0], "stop_details", None) if choices else None
    if details is None:
        return None
    dump = getattr(details, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {"type": getattr(details, "type", None), "category": getattr(details, "category", None)}


def _refusal_attempt(model_name: str, output: Any) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"model": model_name, "stop_reason": str(getattr(output, "stop_reason", None)),
                           "stop_details": _stop_details_record(output)}
    md = getattr(output, "metadata", None) or {}
    if isinstance(md, Mapping) and md.get(NON_ANSWER_KEY):
        rec[NON_ANSWER_KEY] = md[NON_ANSWER_KEY]          # Why a terminal non-Claude leg's output was no answer
    return rec


def _forced_tool_name(tool_choice: Any) -> Optional[str]:
    """The function a ``tool_choice`` FORCES (inspect's ToolFunction(name=...), the judge's answer tool); None for
    "auto" / "none" / "any" / absent."""
    name = getattr(tool_choice, "name", None)
    return str(name) if isinstance(name, str) and name else None


def _output_tool_names(output: Any) -> list:
    choices = getattr(output, "choices", None) or []
    msg = getattr(choices[0], "message", None) if choices else None
    return [str(getattr(c, "function", "") or "") for c in (getattr(msg, "tool_calls", None) or [])]


def non_answer_reason(output: Any, tool_choice: Any) -> Optional[str]:
    """Why a terminal non-Claude leg's RETURNED output is NO answer, or None when it is one: the forced answer tool was
    not called (``tool_choice`` named it -- inspect_scout's structured judge forces its answer tool -- and the message carries
    no call to it: a textual refusal, an empty reply), or the provider marked the message an OpenAI-style refusal
    (``stop_details.type == "refusal"``, the shape the OpenAI-compatible provider gives ``message.refusal``). A returned
    content_filter is already a refusal (:func:`is_refusal_output`) and is not re-marked. No text heuristic: the answer
    tool is the contract."""
    if output is None or is_refusal_output(output):
        return None
    details = _stop_details_record(output)
    if isinstance(details, Mapping) and details.get("type") == "refusal":
        return "stop_details.type refusal"
    forced = _forced_tool_name(tool_choice)
    if forced is not None and forced not in _output_tool_names(output):
        return f"forced answer tool {forced!r} not called"
    return None


def _mark_non_answer_as_refusal(output: Any, why: str) -> None:
    """Mark a terminal non-Claude leg's non-answer a RETURNED refusal: ``stop_reason`` content_filter on the first choice (what
    inspect's ModelOutput.stop_reason reads -- the property has no setter -- and what inspect_scout's generate_retry_refusals
    switches on, so the terminal judge refusal is filed exactly as for a Claude leg and B4 attributes it to this leg, its event
    being the last judge event), on the output itself where the object allows (test doubles), and the reason under metadata."""
    for choice in (getattr(output, "choices", None) or [])[:1]:
        try:
            choice.stop_reason = "content_filter"
        except Exception:
            pass
    if getattr(output, "stop_reason", None) != "content_filter":
        try:
            output.stop_reason = "content_filter"
        except Exception:
            pass
    md = dict(getattr(output, "metadata", None) or {})
    md[NON_ANSWER_KEY] = why
    try:
        output.metadata = md
    except Exception:
        pass


def _stamp_refusal_fallback(output: Any, record: Mapping[str, Any]) -> None:
    md = dict(getattr(output, "metadata", None) or {})
    md[REFUSAL_FALLBACK_SERVED_KEY] = {**dict(record), "client_side": True}
    output.metadata = md


class RefusalFallbackLegError(RuntimeError):
    """A declared per-request fallback leg FAILED on its one re-issue: transport/HTTP, not content — a
    429 at the run's concurrency, a 400 for a request the fallback does not accept, a connection error. It names WHICH
    leg failed (``leg_model``; the primary is ``primary_model``), the HTTP status / error type read off the cause, and
    the refused attempts that preceded it. Its text LEADS with :data:`FALLBACK_LEG_ERROR_MARKER` so the marker survives
    every head clip (the chain's 400-char leg-failure text, the shed's 400-char error) and the converter classifies the
    sample a hard error whatever the status. The sticky chain never rescues it (:class:`AuditorFallbackController`):
    the sample's error reaches the runner's hard-error ledger — an unresolved casualty, never a Grok-completed sample."""

    def __init__(self, *, primary_model: str, leg_model: str, cause: BaseException,
                 attempts: Sequence[Mapping[str, Any]] = ()) -> None:
        text = f"{type(cause).__name__}: {cause}"
        statuses = _HTTP_STATUS_IN_TEXT.findall(text)
        types = _ERROR_TYPE_IN_TEXT.findall(text)
        self.primary_model = str(primary_model)
        self.leg_model = str(leg_model)
        self.cause = cause
        self.status: Optional[int] = int(statuses[-1]) if statuses else None
        self.error_type: Optional[str] = types[-1] if types else None
        self.attempts = [dict(a) for a in attempts]
        http = (f"HTTP {self.status}" + (f" {self.error_type}" if self.error_type else "")) if self.status is not None \
            else "a transport error (no HTTP status)"
        super().__init__(
            f"{FALLBACK_LEG_ERROR_MARKER}: the declared per-request fallback {self.leg_model} (for {self.primary_model}) "
            f"failed with {http} — transport/HTTP, not content; one re-issue, no ladder, never a sticky switch: "
            f"{text[:600]}")


class RefusalFallbackBuildError(RefusalFallbackLegError):
    """The terminal non-Claude leg answered from a build OTHER than the pinned one: the request named the
    immutable dated build (``openrouter/x-ai/grok-4.6-20260810``) and the response's ``model`` names neither that build nor
    its alias. A hard error (the marker leads the text; the converter files it as one), never a score, never a sticky switch."""

    def __init__(self, *, primary_model: str, leg_model: str, served_model: Any, attempts: Sequence[Mapping[str, Any]] = ()) -> None:
        cause = RuntimeError(f"served build mismatch: the request pinned {leg_model}, the response names {str(served_model)!r} "
                             f"(neither the pinned build nor its alias {model_build_family(leg_model)!r}) -- another build or another model")
        super().__init__(primary_model=primary_model, leg_model=leg_model, cause=cause, attempts=attempts)
        self.served_model = str(served_model)


def _served_model_of(output: Any) -> str:
    """The model id a RETURNED output names: the assistant message's ``model`` (what the OpenAI-compatible provider copies
    from the response's ``model``), else the output's ``model``; '' when neither is set."""
    choices = getattr(output, "choices", None) or []
    msg = getattr(choices[0], "message", None) if choices else None
    model = getattr(msg, "model", None) or getattr(output, "model", None)
    return str(model) if model else ""


def build_claim(leg_model: str, served: str, verdict: str) -> str:
    """The plain-words statement a record or a paper-facing attribution may make about the judgment's build (Grok-fix-3): never
    "attested immutable build" unless the provider's response named the pinned build itself."""
    pin = bare_model_id(leg_model)
    if verdict == BUILD_CLASS_EXACT:
        return f"judge pinned to {pin}; the provider's response named the pinned build (verified)"
    if verdict == BUILD_CLASS_DECLARED:
        how = f"the response echoed the alias {served}" if served else "the response named no model"
        return f"judge pinned to {pin}, build not attested by the provider ({how}; declared-only, unverified)"
    return f"judge pinned to {pin}; the provider served {served!r} -- another build or model (refused, never a score)"


def attest_build(leg_model: str, output: Any) -> Dict[str, Any]:
    """The terminal non-Claude leg's build attestation. The REQUEST pins the
    immutable dated build: OpenRouter accepts the canonical slug (``x-ai/grok-4.6-20260810``) as a request model id and routes it to
    that build's endpoints (its /models/<slug>/endpoints listing names only that build; verified by one small request, 2026-10-01,
    served by xAI). The RESPONSE's ``model`` echoes the ALIAS (``x-ai/grok-4.6``), never the build, and carries no fingerprint, so the
    response is evidence of the build ONLY when it names the pinned build itself: ``exact`` (verified). The alias echo, or no model
    named at all, is ``declared_unverified`` -- the request was pinned, the provider attested nothing; it is recorded and attributed as
    declared-only, never as an attested immutable build. Anything else is another model or another build: ``mismatch`` ->
    :class:`RefusalFallbackBuildError`, a hard error, never a score. ``build_verified`` and ``claim`` say the same in plain words."""
    served = _served_model_of(output)
    pin = bare_model_id(leg_model); family = model_build_family(leg_model)
    served_bare = bare_model_id(served) if served else ""
    if served_bare == pin:
        verdict = BUILD_CLASS_EXACT
    elif served_bare == family or not served_bare:
        verdict = BUILD_CLASS_DECLARED
    else:
        verdict = BUILD_CLASS_MISMATCH
    return {"requested": str(leg_model), "pin": pin, "family": family, "served": served, "verdict": verdict,
            "build_verified": verdict == BUILD_CLASS_EXACT, "claim": build_claim(leg_model, served, verdict)}


def make_refusal_fallback_model(primary: Any, primary_name: str, fallback_models: Sequence[str], *, role: str) -> Any:
    """The role's DEFAULT model with its PER-REQUEST refusal fallback served CLIENT-side: ``generate`` issues to the
    default; a request it declines with a RETURNED refusal
    (:func:`is_refusal_output`) is re-issued ONCE per declared fallback, identical ``input`` / ``tools`` / ``tool_choice``
    / ``config`` / ``cache``, and the first non-refusal is returned with the hand-off stamped on its ``metadata``
    (:data:`REFUSAL_FALLBACK_SERVED_KEY`: requested, served, the refused attempts' stop evidence). The next request is at
    the default again. Every fallback refusing returns the LAST refusal as-is (the caller files it as today), stamped
    ``outcome: all_refused``. Each inner ``generate`` is an inspect model event of its own model, so the provenance
    readers (``sample_role_serving_models``, the resume check, the footing tally) need no change: the fallback's
    event names the fallback.

    Why client-side: the server-side form — ``GenerateConfig.fallback_models`` -> ``extra_body["fallbacks"]`` + the
    ``server-side-fallback`` beta — is REJECTED by the API for claude-sonnet-5 (HTTP 400 invalid_request_error,
    deterministic: every relaunched sample of the 2026-10-01 second pass, re-confirmed by one live request), while
    claude-opus-5 accepts it. Routing by model would be a capability table the API can change under us; the client-side
    form sends no such parameter for any role.

    ``primary`` is the resolved inspect ``Model`` of the default; ``fallback_models`` are the manifest's ids (provider
    prefix included, as ``get_model`` wants them).

    The list may END with ONE non-Claude leg -- Grok 4.6 over OpenRouter, the judge's terminal
    leg (manifest.PETRI_REFUSAL_FALLBACK_TERMINAL_*). It is re-issued the IDENTICAL request (the same rubric prompt, tools,
    forced answer tool) and its structured answer is the score, served and stamped like any leg; a Grok refusal or a
    non-answer (:func:`non_answer_reason`) is marked a returned refusal so the terminal judge refusal is filed under Grok
    (petri_runner.refusing_judge_model); a Grok HTTP/transport error raises :class:`RefusalFallbackLegError` as for
    any leg (a hard error, never a shed). The OpenRouter key is resolved in-process
    (:func:`resolve_openrouter_api_key`), never exported. The leg is declared as the IMMUTABLE dated build
    (the manifest validator refuses a mutable alias), the request pins it, and :func:`attest_build` checks every served id
    against the pin or its alias -- a mismatch is :class:`RefusalFallbackBuildError` (hard, never a score)."""
    from inspect_ai.model import Model, get_model

    fallbacks: list = []
    for name in fallback_models:
        get_kwargs: Dict[str, Any] = {"memoize": False}
        if str(name).startswith("openrouter/"):
            # the terminal non-Claude leg rides OpenRouter; its key is passed in-process (the environment or the vault,
            # never a file, never exported), exactly as resolve_inspect_auditor_models passes the auditor chain's Grok leg
            get_kwargs["api_key"] = resolve_openrouter_api_key()
        model = get_model(name, **get_kwargs)
        setter = getattr(model, "_set_role", None)
        if callable(setter):
            setter(role)
        fallbacks.append((str(name), model))

    class RefusalFallbackModel(Model):
        async def generate(
            self, input: Any, tools: Any = (), tool_choice: Any = None,
            config: Any = None, cache: Any = None,
        ) -> Any:
            kwargs: Dict[str, Any] = {"tools": tools, "tool_choice": tool_choice}
            if config is not None:
                kwargs["config"] = config
            if cache is not None:
                kwargs["cache"] = cache
            output = await self._petri_primary.generate(input, **kwargs)
            if not is_refusal_output(output) or not self._petri_fallbacks:
                return output
            attempts = [_refusal_attempt(self._petri_primary_name, output)]
            attestation: Optional[Dict[str, Any]] = None
            for name, model in self._petri_fallbacks:
                try:
                    served = await model.generate(input, **kwargs)
                except Exception as exc:
                    # the fallback leg's OWN failure, named as such — never re-raised as the primary's, never
                    # a sticky switch, never a shed (the converter reads the marker)
                    raise RefusalFallbackLegError(primary_model=self._petri_primary_name, leg_model=name, cause=exc,
                                                  attempts=attempts) from exc
                if not name.startswith(FIRST_PARTY_PROVIDER):
                    # the terminal non-Claude leg's SERVED id must be the pinned build or its alias -- any other
                    # id is another build/model and a hard error (never a score, never "accepted as a refusal"); attested FIRST
                    attestation = attest_build(name, served)
                    if attestation["verdict"] == BUILD_CLASS_MISMATCH:
                        raise RefusalFallbackBuildError(primary_model=self._petri_primary_name, leg_model=name,
                                                        served_model=attestation["served"], attempts=attempts)
                    # the terminal non-Claude leg's non-answer (the forced answer tool uncalled, an OpenAI-style refusal)
                    # is a REFUSAL of that leg -- marked content_filter so the chain below returns it as the LAST refusal and
                    # inspect_scout files the terminal judge refusal; never a NaN without refusal evidence
                    why = non_answer_reason(served, tool_choice)
                    if why:
                        _mark_non_answer_as_refusal(served, why)
                if not is_refusal_output(served):
                    _stamp_refusal_fallback(served, {"requested": self._petri_primary_name, "served": name,
                                                     "outcome": "served", "attempts": attempts,
                                                     **({BUILD_ATTESTATION_KEY: attestation} if attestation else {})})
                    return served
                attempts.append(_refusal_attempt(name, served))
                output = served
            _stamp_refusal_fallback(output, {"requested": self._petri_primary_name, "served": None,
                                             "outcome": "all_refused", "attempts": attempts,
                                             **({BUILD_ATTESTATION_KEY: attestation} if attestation else {})})
            return output

    wrapped = RefusalFallbackModel(primary.api, primary.config, getattr(primary, "model_args", None))
    wrapped._petri_primary = primary
    wrapped._petri_primary_name = str(primary_name)
    wrapped._petri_fallbacks = fallbacks
    setter = getattr(wrapped, "_set_role", None)
    if callable(setter):
        setter(role)
    return wrapped


def with_primary_refusal_fallback(legs: Sequence[AuditorLeg], fallback_models: Sequence[str], *,
                                  role: str = "auditor") -> Tuple[AuditorLeg, ...]:
    """The same legs with the PRIMARY leg's model wrapped in its CLIENT-side per-request refusal fallback
    (:func:`make_refusal_fallback_model`); the sticky chain's later legs are untouched. A first-party primary only (the
    per-request fallback is a first-party-to-first-party hand-off); another provider's primary is left as resolved."""
    legs = tuple(legs)
    if not legs or not legs[0].model_name.startswith("anthropic/") or not fallback_models:
        return legs
    primary = legs[0]
    wrapped = make_refusal_fallback_model(primary.model, primary.model_name, fallback_models, role=role)
    return (AuditorLeg(wrapped, primary.model_name, primary.label), *legs[1:])


def make_inspect_auditor_chain(
    primary_model: str,
    fallback_models: Sequence[str] = DEFAULT_FALLBACK_MODELS,
    *,
    legs: Optional[Sequence[AuditorLeg]] = None,
    discriminator: Optional[LiveSiblingDiscriminator] = None,
    state_getter: Optional[Callable[[], tuple[str, Any]]] = None,
    role: str = "auditor",
    store_key: str = AUDITOR_CHAIN_STORE_KEY,
    switch_on_refusal_output: bool = False,
    primary_refusal_fallback: Sequence[str] = (),
) -> Any:
    """Return a concrete ``inspect_ai.model.Model`` role wrapper (the auditor
    by default; :func:`make_inspect_judge_chain` for the judge).
    ``primary_refusal_fallback`` (the manifest's per-request refusal fallback
    ids) reaches the primary leg only, served client-side
    (:func:`with_primary_refusal_fallback`)."""
    from inspect_ai.model import Model

    resolved = tuple(legs) if legs is not None else resolve_inspect_auditor_models(
        primary_model, fallback_models, role=role
    )
    if primary_refusal_fallback:
        resolved = with_primary_refusal_fallback(resolved, primary_refusal_fallback, role=role)
    controller = AuditorFallbackController(
        resolved, discriminator=discriminator, state_getter=state_getter,
        role=role, store_key=store_key,
        switch_on_refusal_output=switch_on_refusal_output,
    )

    class ChainedAuditorModel(Model):
        async def generate(
            self, input: Any, tools: Any = (), tool_choice: Any = None,
            config: Any = None, cache: Any = None,
        ) -> Any:
            return await self._petri_controller.generate(
                input, tools=tools, tool_choice=tool_choice,
                config=config, cache=cache,
            )

    primary = resolved[0].model
    wrapped = ChainedAuditorModel(primary.api, primary.config, primary.model_args)
    wrapped._petri_controller = controller
    wrapped._set_role(role)
    return wrapped


def make_inspect_judge_chain(
    primary_model: str,
    fallback_models: Sequence[str],
    *,
    legs: Optional[Sequence[AuditorLeg]] = None,
    discriminator: Optional[LiveSiblingDiscriminator] = None,
    state_getter: Optional[Callable[[], tuple[str, Any]]] = None,
    primary_refusal_fallback: Sequence[str] = (),
) -> Any:
    """The JUDGE role's chain: Grok 4.6 falls back for both roles. Same
    controller, three differences, all deliberate:

    * role ``"judge"``: inspect_petri's judge resolves its model with
      ``get_model(role="judge")``, so the wrapper reaches it through
      ``model_roles`` with no change to the vendored package;
    * its own provenance record, :data:`JUDGE_CHAIN_STORE_KEY`;
    * ``switch_on_refusal_output=True``: the judge's failure mode is a refusal
      RETURNED (``stop_reason == "content_filter"``, the safety-classifier
      decline inspect_scout turns into a terminal RefusalError), not raised.
      The auditor chain keeps that switch OFF so its instrument is unchanged.

    inspect's own ``GenerateConfig.fallback_models`` cannot do this job: for
    the Anthropic provider it is a SERVER-side fallback (``extra_body
    ["fallbacks"]``, first-party Claude models only), so it can never name
    an OpenRouter model.

    UNVALIDATED PREMISE, as for the auditor leg: that Grok 4.6 scores the
    transcripts Opus 5 declines, in the structured answer shape the judge
    asks for. A switch is recorded per sample under the store key, so the
    read-out can count and inspect every judge continuation.
    """
    return make_inspect_auditor_chain(
        primary_model, fallback_models, legs=legs,
        discriminator=discriminator, state_getter=state_getter,
        role="judge", store_key=JUDGE_CHAIN_STORE_KEY,
        switch_on_refusal_output=True, primary_refusal_fallback=primary_refusal_fallback,
    )
