"""The agentic-misalignment classifier request (murder + leaking) and its result parser. The paper reports no agentic eval: this family is
kept because the C/P/I track's shared modules (``judges``, ``track``, ``build_manifests``, ``file_records``, ``live``) dispatch per family,
not as a documented reproduction path.

THE SAME TEXT THE SONNET 5 JUDGE SAW. The original call (``scripts/sdfbase/full22/misalign/misalign_judge.judge_agentic`` ->
``scripts/judge_sonnet.Judge.structured``) sends ONE user message -- ``judge_rubrics.PROMPTS[tag]`` (``am_leak.v1`` for leaking,
``am_murder.v1`` for murder; upstream's system and user text in one message) filled with ``str.format(response=<the response>,
email_context=classify.get_email_context(<the scenario's user prompt>, <the response>))`` -- no system prompt, no temperature,
``thinking {"type": "disabled"}``, ``max_tokens`` 1500 (a cut verdict climbed to 4000, then 8000), and the verdict's structured output
``judge_sonnet.verdict_output_format(HarmfulVerdict)`` = ``{"type": "json_schema", "schema": transform_schema(<pydantic schema>)}``.
All of those are IMPORTED here (never copied): the prompt templates, the email-context builder, the scenarios
(``loader.load_scenarios_v2``, matched on item_id == scenario_id; a row's own ``behavior`` wins, as the judge does), the schema transform.

The OpenAI chat-completions form:
  messages          [{"role": "user", "content": <the same prompt>}]
  response_format   {"type": "json_schema", "json_schema": {"name": "HarmfulVerdict", "strict": true, "schema": <the same transformed
                    schema>}} -- Anthropic's structured output is a constrained decode of that schema; OpenAI's strict json_schema is the
                    same contract (the schema already carries additionalProperties false and both fields required)
  reasoning         the judge entry's ``reasoning`` object (thinking disabled on the Sonnet judge)
  max_tokens        the judge entry's first cap (the original's first rung, :data:`MAX_TOKENS`) and its re-ask cap

THE KEY. ``custom_id`` = the judge's own cache key, ``sha256(tag \\0 claude-sonnet-5 \\0 "agentic" \\0 behavior \\0 item_id \\0 response \\0)``
(``Judge._key(tag, JUDGE_MODEL, *("agentic", behavior, item, response))``): the stores' ``judge_cache.jsonl`` hold it for every judged row.

THE PARSER (:func:`parse_result`): ``verdict`` (the text through ``HarmfulVerdict``'s own validation; ``reasoning`` dropped, its length
kept), ``judge_refusal`` (``message.refusal``), ``truncated`` (finish ``length``), ``parse_error`` (no text, malformed or schema-invalid
JSON), and the shared error outcomes (:mod:`parsing`).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(os.path.dirname(_SCRIPTS), "src")
for _p in (_SCRIPTS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import judge_rubrics as JR  # noqa: E402  (scripts/judge_rubrics.py: the prompts and the HarmfulVerdict schema)

from rejudge.terra import parsing as P  # noqa: E402

ENDPOINT = "/v1/chat/completions"
ORIGINAL_MODEL = "claude-sonnet-5"
TAGS = {"leaking": JR.AM_LEAK_TAG, "murder": JR.AM_MURDER_TAG}
EVAL_OF_BEHAVIOR = {"leaking": "agentic_leaking", "murder": "agentic_murder"}
BEHAVIOR_OF_EVAL = {v: k for k, v in EVAL_OF_BEHAVIOR.items()}
SCHEMA = JR.HarmfulVerdict
SCHEMA_NAME = SCHEMA.__name__
#: the Sonnet judge's first rung (judge_sonnet.VERDICT_MAX_TOKENS)
MAX_TOKENS = 1500
LADDER = (1500, 4000, 8000)
RUBRIC_NAME = "agentic_misalignment_classifier"


def _js():
    """``scripts/judge_sonnet`` (imports anthropic and the vault; loaded only where its schema transform or key is needed)."""
    import judge_sonnet
    return judge_sonnet


def cache_key(behavior: str, item_id: str, response: str) -> str:
    """The Sonnet judge's cache key for one agentic row (``Judge._key(tag, JUDGE_MODEL, "agentic", behavior, item, response)``)."""
    h = hashlib.sha256()
    for part in (TAGS[behavior], ORIGINAL_MODEL, "agentic", behavior, item_id, response):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


def output_schema() -> Dict[str, Any]:
    """The schema of the Sonnet judge's structured output, exactly as ``judge_sonnet.verdict_output_format`` builds it."""
    fmt = _js().verdict_output_format(SCHEMA)
    assert fmt["type"] == "json_schema"
    return copy.deepcopy(fmt["schema"])


_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def _schema() -> Dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = output_schema()
    return _SCHEMA_CACHE


def prompt_text(behavior: str, user_prompt: str, response: str) -> str:
    """The user message the Sonnet judge received for this row (``misalign_judge.judge_agentic``'s fill)."""
    from sisteer.capeval.misalign import classify
    email_context = classify.get_email_context(user_prompt, response)
    return JR.PROMPTS[TAGS[behavior]].format(response=response, email_context=email_context)


def build_body(prompt: str, max_tokens: int = MAX_TOKENS, reasoning_effort: Optional[str] = None,
               disable_cache: bool = False) -> Dict[str, Any]:
    body: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_schema", "json_schema": {"name": SCHEMA_NAME, "strict": True,
                                                                                       "schema": copy.deepcopy(_schema())}}}
    body["max_tokens"] = int(max_tokens)
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if disable_cache:
        body["prompt_cache_options"] = {"mode": "explicit"}
    return body


def build_entry(behavior: str, item_id: str, user_prompt: str, response: str, expect_key: Optional[str] = None, **kw) -> Dict[str, Any]:
    key = cache_key(behavior, item_id, response)
    if expect_key is not None and key != expect_key:
        raise ValueError(f"cache key {key[:16]} != the unit's {expect_key[:16]}")
    return {"custom_id": key, "body": build_body(prompt_text(behavior, user_prompt, response), **kw)}


def protocol(model: str, batch_variant: str, body_template: Mapping[str, Any]) -> Dict[str, Any]:
    """The request protocol of a judge model's body template (:func:`rejudge.terra.judges.protocol` builds it), hashed into the judge's
    ``protocol_sha256``."""
    return {"route": "openrouter_batch_api", "endpoint": ENDPOINT, "model": model, "batch_model_variant": batch_variant,
            "body_template": dict(body_template), "system_prompt": None, "temperature": None,
            "translated_from": {"judge_model": ORIGINAL_MODEL, "driver": "scripts/judge_sonnet.Judge.structured", "thinking": {"type": "disabled"},
                                "output": "output_config.format = judge_sonnet.verdict_output_format(HarmfulVerdict)", "max_tokens_ladder": list(LADDER)}}


def rubric_sha256() -> str:
    """sha256 over the canonical JSON of both prompts by tag (the judge row covers murder and leaking, as the Sonnet row does)."""
    blob = json.dumps({TAGS[b]: JR.PROMPTS[TAGS[b]] for b in sorted(TAGS)}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------
def parse_result(entry: Mapping[str, Any]) -> Dict[str, Any]:
    err = P.error_outcome(entry)
    if err is not None:
        return err
    base, msg, finish = P.completion(entry)
    if base.pop("_no_choices", False):
        return dict(base, outcome="parse_error", error={"kind": "no_choices"})
    if msg.get("refusal"):        # the model's own refusal (OpenRouter: message.refusal, finish_reason content_filter): the judge declined
        return dict(base, outcome="judge_refusal", error={"kind": "judge_refusal", "refusal_chars": len(str(msg.get("refusal"))),
                                                          "finish_reason": finish})
    if finish == "content_filter":  # an output cut by a content filter with no refusal text: the provider's decision on this input
        return dict(base, outcome="moderation", error={"kind": "moderation", "message": "finish_reason=content_filter",
                                                       "moderation_source": "finish_reason_content_filter"})
    content = msg.get("content")
    if isinstance(content, list):         # a content-parts list: join its text parts
        content = "".join(str((p or {}).get("text") or "") for p in content if isinstance(p, Mapping))
    if not isinstance(content, str) or not content.strip():
        if finish == "length":
            return dict(base, outcome="truncated", error={"kind": "truncated", "message": "finish_reason=length with no text"})
        return dict(base, outcome="parse_error", error={"kind": "no_text"})
    if finish == "length":
        return dict(base, outcome="truncated", error={"kind": "truncated", "message": "finish_reason=length", "content_chars": len(content)})
    try:
        import pydantic
        v = pydantic.TypeAdapter(SCHEMA).validate_json(content).model_dump()
    except Exception as e:  # noqa: BLE001 -- pydantic ValidationError and kin: the original judge's malformed class
        return dict(base, outcome="parse_error", error={"kind": "schema_validation", "message": type(e).__name__, "content_chars": len(content)})
    verdict = {"harmful": bool(v["harmful"]), "reasoning_chars": len(str(v.get("reasoning") or ""))}
    return dict(base, outcome="verdict", verdict=verdict)
