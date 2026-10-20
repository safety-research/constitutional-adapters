"""The served client's ``max_tokens`` clamp against the serving window — the
ONE parser + policy both served clients (``eval/attacks/served_target.py``,
``eval/attacks/bon_served.py``) apply.

WHY (the per-eval geometry ruling, 2026-09-03). A served eval now runs at a
window sized to the p99 of its own prompt+completion (``geometry.py``), so a
tail of requests will not fit. vLLM REJECTS — HTTP 400, never a truncation —
any request whose ``prompt + max_tokens`` exceeds ``max_model_len``. At the
multiturn_attacks p99 window that is 1.37% of attack calls, while only 0.91%
are truly over the window: the other ~0.46% would fit if the client asked for
fewer output tokens. Before this module a 400 propagated as an exception and
took the whole goal (or the whole BoN run) down with it.

THE POLICY, applied by both clients at the request boundary:

1. A 400 whose body carries vLLM's context-length text (the pinned 0.25.1
   shapes, the same markers ``petri_runner.CONTEXT_OVERFLOW_MARKERS`` keys
   on) is parsed for the server's window, the prompt's token count and the
   requested ``max_tokens`` (:func:`parse_context_overflow`).
2. If ``window - prompt_tokens`` leaves at least :data:`MIN_CLAMPED_COMPLETION`
   tokens, retry ONCE with ``max_tokens`` clamped to that room, and record
   ``max_tokens_clamped_from`` on the per-call record.
3. Otherwise — or if the clamped retry is rejected again — the call is a
   TYPED ERROR, never a response: :class:`PromptOverWindow` (carrying
   prompt_tokens, the requested max_tokens, the threaded window and the
   server's) is raised from the client. The multi-turn runners catch it PER
   GOAL and record the goal as errored (``outcome: "error:PromptOverWindow"``
   in progress.jsonl — the same field and terminal record shape as an
   attacker-side ``error:ActorDiscoveryError``), so the analyzers' existing
   errored-goal rules apply unchanged and ``--resume`` treats the goal as
   NOT done; BoN leaves the sample slot UNCOVERED and files a typed error row
   in a separate ledger so the completeness gate reports the shortfall; the
   static-family adapter files it under its own error policy. An empty
   string NEVER stands in for a failed call anywhere (adversarial review of
   the first cut, 2026-09-03: a ``""`` turn was being persisted as an
   ordinary response and graded as a refusal, biasing ASR downward, and a
   resume never regenerated it).

THE WINDOW. The SERVER-STATED window parsed from the rejection is
AUTHORITATIVE for the clamped retry — it is the fact about what the server
will accept. The window the driver threads in (the geometry entry the
launcher resolved) is PROVENANCE plus a cross-check: on a disagreement the
client logs loudly and records ``window_mismatch`` (threaded, server) on the
call record and in its manifest, and it NEVER shrinks the clamp below the
server's value (the first cut used ``min(threaded, server)``; with a
standup launched at 10,240 and a client assuming 8,704, that manufactured
prompt_over_window failures for calls the server would have served). Only
when the rejection carries no window number does the threaded value stand
in; with neither, the call fails as PromptOverWindow rather than guess.

THE FLOOR-ONLY FORM. vLLM's PRODUCTION rejection — the one the live Petri
audits produced 284 times — does not state the prompt length: "... you
requested 1024 output tokens and your prompt contains AT LEAST 15361 input
tokens ...". That number is ``window - requested + 1``, a LOWER BOUND, so
the exact room is unknowable from the message and the clamp cannot be
computed from it. Policy: the plan comes back ``action == "resolve"``; the
client obtains the EXACT prompt length by POSTing the identical ``messages``
to the server's ``/tokenize`` endpoint THROUGH the routing proxy (a read-only
pass-through with the same virtual-model resolution and no steering —
``RoutingProxy.handle_tokenize``), then ``room = window - prompt_tokens``:
one clamped retry if room >= 64, else the typed error. The call record
carries ``prompt_tokens_source``: ``"error_message"`` when the rejection
stated the length, ``"tokenize_endpoint"`` when it was resolved. If
``/tokenize`` is unavailable (404, connection error, no transport) the
client does NOT guess: the typed error is raised with
``reason == "floor_only_unresolvable"``. The floor is never treated as
the true prompt length. The production template is ONE string across all
284 live rejections ("... your prompt contains at least N input tokens,
for a total of at least M tokens. Please reduce the length of the
messages or completion."); the parser also accepts the "your request has
at least N input tokens" wording, since vLLM's text drifts across
versions. Live-verified 2026-09-03 on vLLM 0.25.1: ``POST /tokenize``
(chat form, ``add_generation_prompt``) returns ``{"count",
"max_model_len", "tokens", "token_strs"}`` behind the VLLM_API_KEY Bearer
token, and ``count`` includes the generation header — the same render the
chat request is measured by.

Anything that is NOT a context-length 400 (a 409 salt refusal, a 5xx, a
transport error) is not this module's business and propagates unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

#: The fewest completion tokens a clamped retry may ask for. Below this the
#: turn is not worth serving truncated — it is filed as prompt_over_window.
MIN_CLAMPED_COMPLETION = 64

#: The error tag of a call whose prompt cannot fit the window (the value of
#: ``error`` on ledger / timing records; the exception is PromptOverWindow).
PROMPT_OVER_WINDOW = "prompt_over_window"

#: pinned vLLM 0.25.1 context-length rejection texts (renderers/params.py
#: and v1/engine/input_processor.py) — the same markers petri_runner uses.
_CONTEXT_MARKERS = ("maximum context length is", "longer than the maximum model length")

#: a 400 must be evidenced: the refusal's own http_status, or the status the
#: transport wrote into its message (ProxyClientEngine: "returned 400: …";
#: the drivers' aiohttp posters: "proxy 400: …"; inspect/openai:
#: "Error code: 400 - …").
_STATUS_400 = re.compile(r"(?:returned|proxy|Error code:|HTTP|status)\s*:?\s*400\b")

_RE_WINDOW = re.compile(r"maximum context length is (\d+) tokens|"
                        r"maximum model length of (\d+)")
_RE_PROMPT = re.compile(r"prompt contains (\d+) input tokens|"
                        r"decoder prompt \(length (\d+)\)")
#: the production floor-only form: "prompt contains at least N input tokens"
#: (N = max_model_len - requested + 1, a LOWER BOUND on the prompt length);
#: also the "request has at least N input tokens" wording (vLLM's text drifts
#: across versions — 0.26 may differ)
_RE_PROMPT_FLOOR = re.compile(r"(?:prompt contains|request has) at least (\d+) input tokens")
_RE_REQUESTED = re.compile(r"you requested (\d+) output tokens")
_RE_TOTAL = re.compile(r"for a total of (\d+) tokens")


#: ``ClampPlan.action`` values
RETRY = "retry"
FAIL = "fail"
RESOLVE = "resolve"          # floor-only rejection: fetch the exact prompt length first

#: ``ClampPlan.prompt_tokens_source`` / call-record values
SOURCE_ERROR_MESSAGE = "error_message"
SOURCE_TOKENIZE = "tokenize_endpoint"

#: the typed error's reason when a floor-only rejection could not be resolved
FLOOR_ONLY_UNRESOLVABLE = "floor_only_unresolvable"


@dataclass(frozen=True)
class ContextOverflow:
    """The facts parsed out of one context-length 400. ``prompt_floor`` is the
    production form's LOWER BOUND ("at least N input tokens"); it is never a
    prompt length."""

    server_window: Optional[int]
    prompt_tokens: Optional[int]
    requested_max_tokens: Optional[int]
    total_tokens: Optional[int]
    excerpt: str
    prompt_floor: Optional[int] = None

    @property
    def floor_only(self) -> bool:
        return self.prompt_tokens is None and self.prompt_floor is not None


def _first_int(m: Optional["re.Match[str]"]) -> Optional[int]:
    if m is None:
        return None
    for g in m.groups():
        if g is not None:
            return int(g)
    return None


def parse_context_overflow(err: Any, *, http_status: Optional[int] = None
                           ) -> Optional[ContextOverflow]:
    """``None`` unless ``err`` (an exception or its text) is a context-length
    400: BOTH a 400 indication (``http_status`` / ``err.http_status`` == 400,
    or a 400 written into the text) AND a pinned context-length marker are
    required — a rate limit, a refusal, or a generic 5xx must never be read
    as an overflow (the same precision contract as petri_runner's classifier).
    """
    text = err if isinstance(err, str) else str(err)
    status = http_status if http_status is not None else getattr(err, "http_status", None)
    if status is not None and status != 400:
        return None
    if status is None and not _STATUS_400.search(text):
        return None
    low = text.lower()
    hit = next((low.find(m) for m in _CONTEXT_MARKERS if m in low), -1)
    if hit < 0:
        return None
    return ContextOverflow(
        server_window=_first_int(_RE_WINDOW.search(text)),
        prompt_tokens=_first_int(_RE_PROMPT.search(text)),
        requested_max_tokens=_first_int(_RE_REQUESTED.search(text)),
        total_tokens=_first_int(_RE_TOTAL.search(text)),
        excerpt=" ".join(text[hit:hit + 200].split()),
        prompt_floor=_first_int(_RE_PROMPT_FLOOR.search(text)),
    )


@dataclass(frozen=True)
class ClampPlan:
    """What the client does next: ``action == RETRY`` with ``max_tokens``
    clamped, ``action == FAIL`` (raise PromptOverWindow), or ``action ==
    RESOLVE`` (a floor-only rejection: fetch the exact prompt length through
    the proxy's ``/tokenize`` and re-plan with :meth:`with_exact_prompt_tokens`,
    or give up with :meth:`unresolvable`). ``window`` is the threaded launch
    window (provenance), ``effective_window`` the one the plan used — the
    server's stated window when the rejection carries one. ``prompt_tokens``
    is the length the plan used and ``prompt_tokens_source`` where it came
    from (``error_message`` / ``tokenize_endpoint`` / None)."""

    action: str
    max_tokens: Optional[int]
    reason: str
    overflow: ContextOverflow
    window: Optional[int]
    effective_window: Optional[int]
    requested_max_tokens: int = 0
    min_completion: int = MIN_CLAMPED_COMPLETION
    prompt_tokens: Optional[int] = None
    prompt_tokens_source: Optional[str] = None
    detail: str = ""

    @property
    def needs_exact_prompt_tokens(self) -> bool:
        return self.action == RESOLVE

    def with_exact_prompt_tokens(self, count: int) -> "ClampPlan":
        """Re-plan a floor-only rejection with the EXACT prompt length the
        proxy's ``/tokenize`` returned (source ``tokenize_endpoint``)."""
        return clamp_plan(self.overflow, requested_max_tokens=self.requested_max_tokens,
                          window=self.window, min_completion=self.min_completion,
                          prompt_tokens=int(count), prompt_tokens_source=SOURCE_TOKENIZE)

    def unresolvable(self, detail: str) -> "ClampPlan":
        """A floor-only rejection whose exact prompt length could not be
        obtained (no tokenize transport, 404, connection error): FAIL with the
        typed reason — never a guess from the floor."""
        return ClampPlan(action=FAIL, max_tokens=None, reason=FLOOR_ONLY_UNRESOLVABLE,
                         overflow=self.overflow, window=self.window,
                         effective_window=self.effective_window,
                         requested_max_tokens=self.requested_max_tokens,
                         min_completion=self.min_completion, prompt_tokens=None,
                         prompt_tokens_source=None, detail=detail)

    @property
    def window_mismatch(self) -> bool:
        sw = self.overflow.server_window
        return self.window is not None and sw is not None and sw != self.window

    def failed(self, reason: str) -> "ClampPlan":
        """This plan re-filed as terminal (the single retry was rejected too)."""
        return ClampPlan(action=FAIL, max_tokens=None, reason=reason,
                         overflow=self.overflow, window=self.window,
                         effective_window=self.effective_window,
                         requested_max_tokens=self.requested_max_tokens,
                         min_completion=self.min_completion,
                         prompt_tokens=self.prompt_tokens,
                         prompt_tokens_source=self.prompt_tokens_source,
                         detail=self.detail)

    def mismatch_record(self) -> Optional[dict]:
        """``{"threaded", "server"}`` when the launcher's window disagrees with
        the server's (for the call record and the manifest), else None."""
        if not self.window_mismatch:
            return None
        return {"threaded": self.window, "server": self.overflow.server_window}

    def record(self) -> dict:
        """The per-call facts of a FAILED call (the typed error's payload),
        for the timing log / error ledger — never a responses row."""
        return {
            "error": PROMPT_OVER_WINDOW,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tokens_source": self.prompt_tokens_source,
            "prompt_floor": self.overflow.prompt_floor,
            "completion_tokens": 0,
            "window": self.window,
            "server_window": self.overflow.server_window,
            "effective_window": self.effective_window,
            "window_mismatch": self.mismatch_record(),
            "window_reason": self.reason,
            "window_reason_detail": self.detail,
            "overflow_excerpt": self.overflow.excerpt,
        }


class PromptOverWindow(RuntimeError):
    """The TYPED error for a call whose prompt cannot fit the serving window
    even with ``max_tokens`` clamped (or whose single clamped retry was
    rejected again). Raised by the served clients; caught PER GOAL by the
    multi-turn runners (``outcome: "error:PromptOverWindow"``), per sample by
    BoN (typed error ledger row, slot left uncovered), per chunk by the
    static-family adapter (its error policy). Carries the parsed facts."""

    def __init__(self, *, prompt_tokens: Optional[int], requested_max_tokens: int,
                 window: Optional[int], server_window: Optional[int],
                 effective_window: Optional[int], reason: str,
                 excerpt: str = "", prompt_tokens_source: Optional[str] = None,
                 prompt_floor: Optional[int] = None, detail: str = "") -> None:
        self.prompt_tokens = prompt_tokens
        self.prompt_tokens_source = prompt_tokens_source
        self.prompt_floor = prompt_floor
        self.requested_max_tokens = int(requested_max_tokens)
        self.window = window
        self.server_window = server_window
        self.effective_window = effective_window
        self.reason = reason
        self.detail = detail
        self.excerpt = excerpt
        shown = (f"prompt {prompt_tokens}" if prompt_tokens is not None
                 else f"prompt >= {prompt_floor} (floor only)")
        super().__init__(
            f"{PROMPT_OVER_WINDOW}: {shown} + max_tokens "
            f"{requested_max_tokens} does not fit the serving window "
            f"(server {server_window}, threaded {window}): {reason}"
            + (f" — {detail}" if detail else ""))

    @classmethod
    def from_plan(cls, plan: "ClampPlan", *, requested_max_tokens: int
                  ) -> "PromptOverWindow":
        return cls(prompt_tokens=plan.prompt_tokens,
                   requested_max_tokens=requested_max_tokens,
                   window=plan.window, server_window=plan.overflow.server_window,
                   effective_window=plan.effective_window, reason=plan.reason,
                   excerpt=plan.overflow.excerpt,
                   prompt_tokens_source=plan.prompt_tokens_source,
                   prompt_floor=plan.overflow.prompt_floor, detail=plan.detail)

    @property
    def window_mismatch(self) -> Optional[dict]:
        if self.window is None or self.server_window is None \
                or self.window == self.server_window:
            return None
        return {"threaded": self.window, "server": self.server_window}

    def record(self) -> dict:
        """The typed error's facts for a ledger / timing row."""
        return {
            "error": PROMPT_OVER_WINDOW,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tokens_source": self.prompt_tokens_source,
            "prompt_floor": self.prompt_floor,
            "requested_max_tokens": self.requested_max_tokens,
            "completion_tokens": 0,
            "window": self.window,
            "server_window": self.server_window,
            "effective_window": self.effective_window,
            "window_mismatch": self.window_mismatch,
            "window_reason": self.reason,
            "window_reason_detail": self.detail,
            "overflow_excerpt": self.excerpt,
        }


def clamp_plan(overflow: ContextOverflow, *, requested_max_tokens: int,
               window: Optional[int] = None,
               min_completion: int = MIN_CLAMPED_COMPLETION,
               prompt_tokens: Optional[int] = None,
               prompt_tokens_source: Optional[str] = None) -> ClampPlan:
    """The policy (module docstring): retry once with ``max_tokens = window -
    prompt_tokens`` when that leaves ``>= min_completion``, else fail
    (PromptOverWindow). The SERVER-STATED window is authoritative; the
    launcher's threaded ``window`` is provenance + a cross-check
    (``ClampPlan.window_mismatch``) and stands in only when the rejection
    carries no window number. The clamp is never shrunk below the server's
    value. ``prompt_tokens`` is the EXACT prompt length when the caller
    obtained it (``/tokenize``); otherwise the rejection's own number is
    used, and a FLOOR-ONLY rejection yields ``action == RESOLVE`` — the floor
    is never used as the length."""
    eff = overflow.server_window if overflow.server_window is not None else window
    if prompt_tokens is None:
        prompt_tokens = overflow.prompt_tokens
        prompt_tokens_source = (SOURCE_ERROR_MESSAGE if prompt_tokens is not None else None)
    base = dict(overflow=overflow, window=window, effective_window=eff,
                requested_max_tokens=int(requested_max_tokens),
                min_completion=int(min_completion), prompt_tokens=prompt_tokens,
                prompt_tokens_source=prompt_tokens_source)
    if eff is None:
        return ClampPlan(action=FAIL, max_tokens=None, **base,
                         reason="no window known (none in the rejection, none threaded)")
    if prompt_tokens is None:
        if overflow.floor_only:
            return ClampPlan(action=RESOLVE, max_tokens=None, **base,
                             reason=(f"floor-only rejection (prompt >= "
                                     f"{overflow.prompt_floor}): the exact prompt length "
                                     "must come from /tokenize"))
        return ClampPlan(action=FAIL, max_tokens=None, **base,
                         reason="prompt token count not parseable from the rejection")
    room = eff - prompt_tokens
    if room < min_completion:
        return ClampPlan(action=FAIL, max_tokens=None, **base,
                         reason=(f"prompt {prompt_tokens} leaves {room} of "
                                 f"window {eff} (< {min_completion} minimum completion)"))
    if room >= requested_max_tokens:
        # prompt + requested fits the window we know of, yet the server
        # rejected it: the real window is smaller than anything we can see
        return ClampPlan(action=FAIL, max_tokens=None, **base,
                         reason=(f"rejection inconsistent with window {eff}: prompt "
                                 f"{prompt_tokens} + max_tokens "
                                 f"{requested_max_tokens} fits it"))
    return ClampPlan(action=RETRY, max_tokens=int(room), **base,
                     reason=(f"clamped max_tokens {requested_max_tokens} -> {room} "
                             f"(prompt {prompt_tokens} via {prompt_tokens_source}, "
                             f"window {eff})"))


def tokenize_payload(request: Mapping[str, Any]) -> dict:
    """The ``/tokenize`` request for the IDENTICAL messages of a rejected chat
    request (same virtual model — the proxy resolves it the same way — and the
    generation prompt appended, exactly as the chat render counts it)."""
    out = {"model": request["model"], "messages": request["messages"],
           "add_generation_prompt": True}
    if request.get("tools"):
        out["tools"] = request["tools"]
    return out


def exact_prompt_tokens(tokenize_response: Any) -> int:
    """The prompt length from a vLLM ``/tokenize`` response (``count``);
    ``ValueError`` on any other shape — never a guess."""
    if isinstance(tokenize_response, Mapping):
        count = tokenize_response.get("count")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            return count
    raise ValueError(f"/tokenize response carries no integer 'count': "
                     f"{str(tokenize_response)[:200]}")


def plan_for_exception(exc: BaseException, *, requested_max_tokens: int,
                       window: Optional[int] = None,
                       min_completion: int = MIN_CLAMPED_COMPLETION
                       ) -> Optional[ClampPlan]:
    """``None`` when ``exc`` is not a context-length 400 (the caller re-raises);
    else the :class:`ClampPlan` for it."""
    ov = parse_context_overflow(exc)
    if ov is None:
        return None
    return clamp_plan(ov, requested_max_tokens=requested_max_tokens,
                      window=window, min_completion=min_completion)


def overflow_facts(plan: ClampPlan) -> dict:
    """Serializable diagnostics of a plan (for evidence / manifests)."""
    d = asdict(plan)
    d["window_mismatch"] = plan.window_mismatch
    return d


__all__ = [
    "MIN_CLAMPED_COMPLETION", "PROMPT_OVER_WINDOW", "RETRY", "FAIL", "RESOLVE",
    "SOURCE_ERROR_MESSAGE", "SOURCE_TOKENIZE", "FLOOR_ONLY_UNRESOLVABLE",
    "ContextOverflow", "ClampPlan", "PromptOverWindow", "parse_context_overflow",
    "clamp_plan", "plan_for_exception", "tokenize_payload", "exact_prompt_tokens",
    "overflow_facts",
]
