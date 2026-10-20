"""Continuation-loglikelihood scorer through the served path.

THE INSTRUMENT. One prefill-only chat request per candidate continuation:
the candidate is the FINAL assistant message, sent with
``continue_final_message: true`` + ``add_generation_prompt: false``,
declared with ``petri_response_prefill: true`` on lens routes (E1/E6 — an
undeclared continued message on a lens route is a proxy 400; the declaration
is FORBIDDEN on a maskless route), ``max_tokens: 1``, ``prompt_logprobs: 1``.
The scored span is the candidate's tokens after the assistant header in the
SERVER's own render (``/v1/tokenize`` of the same messages), never a
client-side concatenation; the continuation logprob is the sum of the
actual token's ``prompt_logprobs`` entry over that span.

THE HTTP NORMALIZATION BOUNDARY (P1 #2). vLLM 0.25.1's
``ChatCompletionResponse.prompt_logprobs`` is ``list[dict[int, Logprob] |
None]`` IN PROCESS; over JSON every per-position dict has STRING keys
(token ids as base-10 strings) and each value is an object
``{"logprob": float, "rank": int|null, "decoded_token": str|null}``;
position 0 is ``null``. :func:`normalize_prompt_logprobs` is the ONLY code
that touches that raw shape. It converts ``str → int`` keys and reads
``"logprob"``, and REFUSES — ``PrefillLogprobError``, a per-item hard error,
never a silently mis-scored candidate — a non-list container, a key that is
not a canonical base-10 integer string (an int key is ALSO refused: the
wire shape is strings, and accepting both would hide a schema change), a
value without a numeric finite ``"logprob"``, a duplicate key, or a
container whose length differs from the tokenize ids. Everything
downstream sees ``list[dict[int, float] | None]``.

THE SPAN RULE (P37). The assistant header must occur EXACTLY ONCE in the
render (missing → refuse; twice → refuse: a single-turn prefill render has
one model header, so a second one means this is not the shape the
instrument scores — refused rather than disambiguated by rfind). When the
caller supplies ``expected_candidate_ids`` (the tokenizer's encoding of the
template-rendered, TRIMMED candidate — gemma-4's template trims assistant
content), the candidate is the render's TAIL and must equal them exactly;
tokens between the header and that tail are SCAFFOLD — empty under the
flag-only render, the empty thought block ``<|channel>thought\\n \\n<channel|>``
under the diagnostic render (P6 B) — and are scored only when the caller
says ``scaffold_ok=True`` (so a flag-only request whose render grew an
unexpected thought block is refused, not silently scored). Without
expected ids the span is everything after the header.

OTHER REFUSALS (all hard, all per item, collected into the run's error
list — the ``_fanout_responses`` discipline): empty span;
``len(prompt_logprobs) != len(ids)``; the actual token id absent at any
span position (vLLM includes the actual token's entry with
``prompt_logprobs=1``);
``prompt_token_ids`` (when the response carries it) differing from the
tokenize ids.

Base-lane importable: stdlib + the proxy's declaration key only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sisteer.capeval.serving.proxy import RESPONSE_PREFILL_KEY
from sisteer.capeval.tinycap import contracts as C  # the thinking-ON declaration predicate

__all__ = [
    "PromptLogprobs",
    "PrefillLogprobError",
    "ContinuationSpan",
    "CandidateScore",
    "PREFILL_REQUEST_KEYS",
    "TOKENIZE_REQUEST_KEYS",
    "REASONING_DIAG_VALUE",
    "build_prefill_request",
    "build_tokenize_request",
    "normalize_prompt_logprobs",
    "prompt_token_ids_of",
    "continuation_span",
    "continuation_logprob",
    "score_candidate",
    "rendered_candidate",
]

#: The normalized shape: one entry per prompt position, ``None`` at
#: position 0 (and anywhere vLLM emits null), else ``{token_id: logprob}``.
PromptLogprobs = List[Optional[Dict[int, float]]]


class PrefillLogprobError(RuntimeError):
    """A refusal of the continuation-logprob instrument for ONE item — the
    raw response, the render, or the span did not meet the contract. Never
    caught into a default value; collected per item and aborts the task's
    aggregate (the item reruns on resume)."""


@dataclass(frozen=True)
class ContinuationSpan:
    """The scored positions of one prefill render: ``range(start, stop)``
    over the tokenize ids; ``header_at`` is the index of the (single)
    assistant-header occurrence; ``scaffold`` the number of render tokens
    between the header's end and ``start`` (0 under the flag-only render)."""

    start: int
    stop: int
    header_at: int
    scaffold: int = 0

    @property
    def positions(self) -> range:
        return range(self.start, self.stop)

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CandidateScore:
    """Per-candidate facts a loglikelihood row records: ``ll_sum`` = Σ
    logprob over the span;
    ``n_cont_tokens`` = span length; ``n_chars`` = the ``acc_norm``
    denominator supplied by the caller (lm-eval's ``len(choice)`` rule);
    ``span_start`` = the span's first position; ``prompt_tokens`` = the
    response's ``usage.prompt_tokens`` (None when absent);
    ``token_ids`` = the span's actual token ids, in order."""

    ll_sum: float
    n_cont_tokens: int
    n_chars: int
    span_start: int
    prompt_tokens: Optional[int]
    token_ids: Tuple[int, ...]


#: Exactly the keys a prefill-scoring request body carries;
#: ``petri_response_prefill`` present iff ``response_prefill`` (lens route);
#: ``return_token_ids`` present iff requested (the 0.25.1 response class has
#: ``prompt_token_ids``); ``petri_thinking_on`` (contract §2.5.4) present iff
#: the body sends ``enable_thinking: true`` on a
#: lens route — ``response_prefill`` IS the lens bit here. NEVER ``n``,
#: ``stream``, or any of ``proxy.FORBIDDEN_CLIENT_KEYS``.
PREFILL_REQUEST_KEYS: Tuple[str, ...] = (
    "model", "messages", "continue_final_message", "add_generation_prompt",
    "max_tokens", "temperature", "seed", "prompt_logprobs", "logprobs",
    "chat_template_kwargs", RESPONSE_PREFILL_KEY, "return_token_ids",
    C.OOS_THINKING_DECLARATION_KEY,
)
#: The ``/v1/tokenize`` body for the SAME render: the proxy forwards these
#: five and returns ``{count, max_model_len, tokens, token_strs}``.
TOKENIZE_REQUEST_KEYS: Tuple[str, ...] = (
    "model", "messages", "continue_final_message", "add_generation_prompt",
    "chat_template_kwargs",
)
#: The assistant-message ``reasoning`` value of the diagnostic render (P6 B).
REASONING_DIAG_VALUE = " "

_INT_KEY = re.compile(r"^(0|[1-9][0-9]*)$")


def _refuse(msg: str) -> "PrefillLogprobError":
    return PrefillLogprobError(f"prefill_logprob: {msg}")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def rendered_candidate(candidate: str) -> str:
    """What the chat template renders for an assistant message whose
    content is ``candidate``: gemma-4 (and the Llama-3 family) ``trim`` the
    content, so lm-eval's ``" " + choice`` continuation renders as
    ``choice``. The expected-span ids are the tokenizer's encoding of THIS
    string (never of the raw continuation with its delimiter)."""
    if not isinstance(candidate, str):
        raise _refuse(f"candidate must be a str, got {type(candidate).__name__}")
    return candidate.strip()


def _messages(user_text: str, candidate: str, system: Optional[str],
              *, thought_block_diag: bool) -> List[Dict[str, Any]]:
    if not isinstance(user_text, str) or not user_text:
        raise _refuse("user_text must be a non-empty str")
    if system is not None and (not isinstance(system, str) or not system):
        raise _refuse("system must be None or a non-empty str")
    if not rendered_candidate(candidate):
        raise _refuse("candidate renders EMPTY after the template's trim — an empty "
                      "span cannot be scored")
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    asst: Dict[str, Any] = {"role": "assistant", "content": candidate}
    if thought_block_diag:
        asst["reasoning"] = REASONING_DIAG_VALUE
    msgs.append(asst)
    return msgs


def build_prefill_request(*, virtual_model: str, user_text: str, candidate: str,
                          system: Optional[str], response_prefill: bool,
                          enable_thinking: bool, seed: int = 0,
                          thought_block_diag: bool = False,
                          return_token_ids: bool = True) -> Dict[str, Any]:
    """The prefill-only chat body for ONE candidate (module docstring):
    ``messages = [system?, user, assistant=candidate]`` (the ``reasoning``
    field set to ``" "`` on the assistant message iff ``thought_block_diag``),
    ``continue_final_message: true``, ``add_generation_prompt: false``,
    ``max_tokens: 1``, ``temperature: 0.0``, ``seed``, ``prompt_logprobs: 1``,
    ``logprobs: false``, ``chat_template_kwargs: {"enable_thinking": b}``,
    ``petri_response_prefill: true`` iff ``response_prefill`` (lens routes
    ONLY — the proxy refuses it on a maskless route), ``petri_thinking_on:
    true`` iff ``response_prefill and enable_thinking`` (the lens route's
    thinking-ON declaration, ``contracts.thinking_on_declaration`` —
    contract §2.5.4; NOTE the contract's §2.5.6 also lists the key beside
    ``petri_response_prefill`` as ``thinking_on_mismatch``, a contradiction the
    plan resolves toward the uniform predicate),
    ``return_token_ids: true`` iff requested. Keys ⊆ :data:`PREFILL_REQUEST_KEYS`;
    ``system`` None ⇒ no system message (compiled cells, the ``none`` route)."""
    if not isinstance(virtual_model, str) or not virtual_model:
        raise _refuse("virtual_model must be a non-empty str")
    arm = C.thinking_mode(enable_thinking)
    body: Dict[str, Any] = {
        "model": virtual_model,
        "messages": _messages(user_text, candidate, system,
                              thought_block_diag=bool(thought_block_diag)),
        "continue_final_message": True,
        "add_generation_prompt": False,
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": int(seed),
        "prompt_logprobs": 1,
        "logprobs": False,
        **C.render_fragment(lens=bool(response_prefill), arm=arm),
    }
    if response_prefill:
        body[RESPONSE_PREFILL_KEY] = True
    if return_token_ids:
        body["return_token_ids"] = True
    assert set(body) <= set(PREFILL_REQUEST_KEYS), sorted(set(body) - set(PREFILL_REQUEST_KEYS))
    return body


def build_tokenize_request(*, virtual_model: str, messages: Sequence[Mapping[str, Any]],
                           enable_thinking: bool) -> Dict[str, Any]:
    """The ``/v1/tokenize`` body for the prefill render of ``messages``
    (the SAME list the prefill request sends): ``continue_final_message:
    true``, ``add_generation_prompt: false``, the same
    ``chat_template_kwargs``. Keys == :data:`TOKENIZE_REQUEST_KEYS`."""
    if not isinstance(virtual_model, str) or not virtual_model:
        raise _refuse("virtual_model must be a non-empty str")
    msgs = [dict(m) for m in messages]
    if not msgs or msgs[-1].get("role") != "assistant":
        raise _refuse("tokenize render must end with the assistant candidate message")
    body = {
        "model": virtual_model,
        "messages": msgs,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    assert set(body) == set(TOKENIZE_REQUEST_KEYS)
    return body


def normalize_prompt_logprobs(resp_json: Mapping[str, Any], *,
                              expected_len: Optional[int] = None) -> PromptLogprobs:
    """The HTTP normalization boundary (module docstring). Reads
    ``resp_json["prompt_logprobs"]``; returns one ``dict[int, float] | None``
    per position. Refuses (``PrefillLogprobError``): a missing or non-list
    ``prompt_logprobs``; ``expected_len`` given and ``len != expected_len``;
    a per-position entry that is neither ``None`` nor a mapping; a key that
    is not a canonical base-10 integer string (int keys refused too); a
    value that is not a mapping with a ``"logprob"`` that is a finite
    number (bool excluded); a duplicate integer key. The ``rank`` /
    ``decoded_token`` fields are ignored, never required."""
    if not isinstance(resp_json, Mapping):
        raise _refuse(f"response must be a JSON object, got {type(resp_json).__name__}")
    if "prompt_logprobs" not in resp_json:
        raise _refuse("response carries no 'prompt_logprobs' — the server did not "
                      "honor prompt_logprobs=1 (or the proxy dropped it); refusing")
    raw = resp_json["prompt_logprobs"]
    if not isinstance(raw, list):
        raise _refuse(f"'prompt_logprobs' must be a list aligned with prompt positions, "
                      f"got {type(raw).__name__}")
    if expected_len is not None and len(raw) != expected_len:
        raise _refuse(f"'prompt_logprobs' has {len(raw)} positions but the tokenize render "
                      f"has {expected_len} ids — the response is not this render's")
    # This loop runs once per prompt position per candidate (~7k positions
    # for a prompted-cell render), on the driver's event loop: the exact
    # `dict` / `str` / `float` cases take the cheap path, and anything else
    # falls through to the SAME refusals as before (measured 3x faster; the
    # accepted set and every refusal are unchanged — see the tests).
    out: PromptLogprobs = []
    for pos, entry in enumerate(raw):
        if entry is None:
            out.append(None)
            continue
        if type(entry) is not dict and not isinstance(entry, Mapping):
            raise _refuse(f"position {pos}: entry must be null or an object, "
                          f"got {type(entry).__name__}")
        norm: Dict[int, float] = {}
        for k, v in entry.items():
            # canonical base-10 (ASCII digits, no leading zero unless exactly "0")
            # on the fast path; anything else goes through the regex it mirrors
            if (not (type(k) is str and k.isascii() and k.isdigit()
                     and (k == "0" or k[0] != "0"))
                    and (not isinstance(k, str) or not _INT_KEY.match(k))):
                raise _refuse(f"position {pos}: key {k!r} is not a base-10 token-id string "
                              "(the wire shape is string keys; anything else is a schema "
                              "change, refused)")
            tid = int(k)
            if tid in norm:
                raise _refuse(f"position {pos}: duplicate token id {tid}")
            if (type(v) is not dict and not isinstance(v, Mapping)) or "logprob" not in v:
                raise _refuse(f"position {pos}: token {tid}: value must be an object with "
                              f"'logprob', got {v!r}")
            lp = v["logprob"]
            if (type(lp) is not float and not _is_number(lp)) or not math.isfinite(lp):
                raise _refuse(f"position {pos}: token {tid}: 'logprob' must be a finite "
                              f"number, got {lp!r}")
            norm[tid] = float(lp)
        out.append(norm)
    return out


def prompt_token_ids_of(resp_json: Mapping[str, Any]) -> Optional[List[int]]:
    """``resp_json["prompt_token_ids"]`` as ints when present (the 0.25.1
    response class carries it when ``return_token_ids`` was honored), else
    ``None``. A present-but-malformed field (non-list, non-int elements)
    is a refusal, not ``None``."""
    if not isinstance(resp_json, Mapping):
        raise _refuse(f"response must be a JSON object, got {type(resp_json).__name__}")
    v = resp_json.get("prompt_token_ids")
    if v is None:
        return None
    if not isinstance(v, list) or not all(_is_int(t) for t in v):
        raise _refuse("'prompt_token_ids' is present but not a list of ints — refusing "
                      "rather than ignoring a malformed echo")
    return [int(t) for t in v]


def _find_all(ids: Sequence[int], needle: Sequence[int]) -> List[int]:
    """Every start index at which ``needle`` occurs in ``ids`` (overlaps
    included). The first-token test before the slice compare keeps this
    linear in practice over a ~7k-token render (a slice per position was
    the scorer's second-largest per-candidate cost)."""
    ids = ids if type(ids) is list else list(ids)
    needle = needle if type(needle) is list else list(needle)
    n, m = len(ids), len(needle)
    if not needle:
        return list(range(n + 1))  # every position matches the empty needle (callers refuse it)
    first = needle[0]
    return [i for i in range(n - m + 1) if ids[i] == first and ids[i:i + m] == needle]


def continuation_span(ids: Sequence[int], header_ids: Sequence[int], *,
                      expected_candidate_ids: Optional[Sequence[int]] = None,
                      scaffold_ok: bool = False) -> ContinuationSpan:
    """The span rule (module docstring, P37). ``header_ids`` is
    ``generation_header_ids(tok)`` of the SERVED tokenizer (gemma-4:
    ``[105, 4368, 107]``), read from the model, never typed."""
    ids = list(ids)
    header = list(header_ids)
    if not header or not all(_is_int(t) for t in header):
        raise _refuse(f"header_ids must be a non-empty list of ints, got {header_ids!r}")
    if not all(_is_int(t) for t in ids):
        raise _refuse("tokenize ids must be ints")
    hits = _find_all(ids, header)
    if not hits:
        raise _refuse(f"assistant header {header} not found in the {len(ids)}-token render — "
                      "the request did not render as a continued assistant turn")
    if len(hits) > 1:
        raise _refuse(f"assistant header {header} occurs {len(hits)} times (at {hits}) — a "
                      "single-turn prefill render has exactly one; refusing to guess which "
                      "occurrence starts the candidate")
    h = hits[0]
    header_end = h + len(header)
    if header_end >= len(ids):
        raise _refuse("empty continuation span: the render ends at the assistant header "
                      "(no candidate tokens to score)")
    if expected_candidate_ids is None:
        return ContinuationSpan(start=header_end, stop=len(ids), header_at=h, scaffold=0)
    exp = list(expected_candidate_ids)
    if not exp or not all(_is_int(t) for t in exp):
        raise _refuse(f"expected_candidate_ids must be a non-empty list of ints, got {exp!r}")
    start = len(ids) - len(exp)
    if start < header_end or ids[start:] != exp:
        raise _refuse(f"render tail {ids[header_end:]} does not end with the tokenizer's "
                      f"encoding of the rendered candidate {exp} — the server's render of "
                      "the candidate is not the candidate (template trim / merge across "
                      "the header boundary); refusing to score a different string")
    scaffold = start - header_end
    if scaffold and not scaffold_ok:
        raise _refuse(f"{scaffold} render token(s) {ids[header_end:start]} sit between the "
                      "assistant header and the candidate — a flag-only render has none "
                      "(the empty thought block appears only under the diagnostic render, "
                      "which must be declared with scaffold_ok=True); refusing")
    return ContinuationSpan(start=start, stop=len(ids), header_at=h, scaffold=scaffold)


def continuation_logprob(plp: PromptLogprobs, ids: Sequence[int],
                         span: ContinuationSpan) -> float:
    """``Σ_{p ∈ span} plp[p][ids[p]]``. Refuses ``len(plp) != len(ids)``, an
    empty span, a ``None`` entry at a span position, or the actual token id
    absent from the entry (``prompt_logprobs=1`` always includes the actual
    token). Returns a finite float (a non-finite sum is a refusal — the
    store refuses NaN at write and this must fail earlier, per item)."""
    ids = list(ids)
    if len(plp) != len(ids):
        raise _refuse(f"prompt_logprobs has {len(plp)} positions but the render has "
                      f"{len(ids)} ids")
    if len(span) <= 0 or span.start < 0 or span.stop > len(ids):
        raise _refuse(f"span {span} is empty or outside the {len(ids)}-token render")
    total = 0.0
    for p in span.positions:
        entry = plp[p]
        if entry is None:
            raise _refuse(f"position {p}: prompt_logprobs entry is null inside the scored "
                          "span — the server returned no distribution there")
        tid = ids[p]
        if tid not in entry:
            raise _refuse(f"position {p}: the actual token {tid} is absent from the "
                          f"prompt_logprobs entry (keys {sorted(entry)[:8]}) — cannot form "
                          "an exact continuation logprob")
        total += entry[tid]
    if not math.isfinite(total):
        raise _refuse(f"continuation logprob is not finite ({total!r})")
    return float(total)


def _usage_prompt_tokens(resp_json: Mapping[str, Any]) -> Optional[int]:
    usage = resp_json.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, Mapping):
        raise _refuse("'usage' is present but not an object")
    v = usage.get("prompt_tokens")
    if v is None:
        return None
    if not _is_int(v):
        raise _refuse(f"'usage.prompt_tokens' must be an int, got {v!r}")
    return int(v)


def score_candidate(*, resp_json: Mapping[str, Any], tokenize_ids: Sequence[int],
                    header_ids: Sequence[int], n_chars: int,
                    expected_candidate_ids: Optional[Sequence[int]] = None,
                    scaffold_ok: bool = False) -> CandidateScore:
    """One candidate end to end: normalize (``expected_len = len(tokenize_ids)``),
    cross-check ``prompt_token_ids_of(resp_json)`` against ``tokenize_ids``
    when present (mismatch = refusal), take the span, sum, and record
    ``usage.prompt_tokens``. ``n_chars`` is the ``acc_norm`` denominator
    supplied by the caller (lm-eval's ``len(choice)``; the scorer does not
    decode) and must be a positive int."""
    if not _is_int(n_chars) or n_chars < 1:
        raise _refuse(f"n_chars must be a positive int, got {n_chars!r}")
    ids = list(tokenize_ids)
    if not ids:
        raise _refuse("tokenize_ids is empty")
    plp = normalize_prompt_logprobs(resp_json, expected_len=len(ids))
    echo = prompt_token_ids_of(resp_json)
    if echo is not None and echo != ids:
        raise _refuse(f"response prompt_token_ids ({len(echo)} ids) != the tokenize render "
                      f"({len(ids)} ids) — the response was not produced from this render")
    span = continuation_span(ids, header_ids, expected_candidate_ids=expected_candidate_ids,
                             scaffold_ok=scaffold_ok)
    ll = continuation_logprob(plp, ids, span)
    return CandidateScore(
        ll_sum=ll,
        n_cont_tokens=len(span),
        n_chars=int(n_chars),
        span_start=span.start,
        prompt_tokens=_usage_prompt_tokens(resp_json),
        token_ids=tuple(ids[span.start:span.stop]),
    )
