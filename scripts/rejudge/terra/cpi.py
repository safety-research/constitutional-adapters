"""The OR-Bench hard / XSTest C/P/I judge request (OpenAI chat completions through OpenRouter's Batch API and the live slot) and its
result parser. The judge model and its model-specific settings are a config entry in :mod:`rejudge.terra.judges`.

THE SAME TEXT THE SONNET 5 JUDGE SAW, BY CONSTRUCTION. The original call (``sisteer.capeval.judging``: ``ClaudeJudge`` over
``render_request(rubric, JudgeRequest(...))``, driven by ``tiers.judged._judge_cpi`` and ``scripts/served_capability_judge.py``) is
rendered here by the SAME function with the SAME arguments -- the frozen rubric (``ORBENCH_RUBRIC_V1`` sha 15f29334703c31a8 /
``XSTEST_RUBRIC_V1`` sha 288e8c08b0c7bbfa), ``judge_model`` claude-sonnet-5, the default ``protocol_sha`` b8d28441cf71caec -- and only
then translated. The ``custom_id`` is that request's own ``JudgeRequest.key()``, which every judged store records per item
(``judge_items.jsonl`` ``request_key``): :mod:`scope_cpi` recomputes it from the store's text and refuses an item whose key differs.

The Anthropic params -> OpenAI chat completions:
  system (SYSTEM_FRAMING)     -> messages[0] {"role": "system", "content": <the same text>}
  messages[0] (the rubric filled with the request and the response) -> messages[1], unchanged
  tools [compliance_rating]   -> tools [{"type": "function", "function": {name, description, parameters: <the same input_schema>}}]
  tool_choice {type: tool, name} -> {"type": "function", "function": {"name": "compliance_rating"}}  (the forced call)
                              + parallel_tool_calls false (Anthropic returns one tool_use)
  thinking {"type": "disabled"} -> the judge entry's ``reasoning`` object (reasoning off at the model's lowest setting; MiniMax-M3:
                              ``{"enabled": false}``)
  temperature: none sent (judging.TEMPERATURE is None: claude-sonnet-5 rejects sampling params), so none is sent here either
  max_tokens 1024             -> the judge entry's cap (MiniMax-M3: 1024, the Sonnet judge's own)
  + ``prompt_cache_options`` and ``parallel_tool_calls`` only where the judge entry sends them (MiniMax-M3: neither)

THE PARSER (:func:`parse_result`): ``verdict`` (the label through ``judging._sanitize_verdict`` -- the original judge's own validation:
enum membership, required fields, the free-text ``reasoning`` dropped and only its length kept), ``moderation`` / ``account_error``
/ ``provider_error`` for error entries (:mod:`parsing`), ``judge_refusal`` (``message.refusal``), ``truncated`` (finish ``length``),
``parse_error`` (no call, the wrong tool, malformed JSON, a label outside the enum -- the welded-label shape included).
"""
from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(os.path.dirname(_SCRIPTS), "src")
for _p in (_SCRIPTS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sisteer.capeval import judging  # noqa: E402
from sisteer.capeval.judging import JudgeRequest  # noqa: E402

from rejudge.terra import parsing as P  # noqa: E402

ENDPOINT = "/v1/chat/completions"
#: the model whose requests (and cache keys) are re-sent
ORIGINAL_MODEL = judging.JUDGE_MODEL
#: rubric name -> the frozen rubric; the eval each judges
RUBRICS = {"or_bench_cpi": judging.ORBENCH_RUBRIC_V1, "xstest_cpi": judging.XSTEST_RUBRIC_V1}
EVAL_OF_RUBRIC = {"or_bench_cpi": "or_bench_hard", "xstest_cpi": "xstest"}
RUBRIC_OF_EVAL = {v: k for k, v in EVAL_OF_RUBRIC.items()}
#: the store's control name and task per eval
CONTROL_OF_EVAL = {"or_bench_hard": "or_bench_hard_1k", "xstest": "xstest"}
TASK_OF_EVAL = {"or_bench_hard": "judged/or_bench_hard_1k", "xstest": "judged/xstest_v2"}
ORIGINAL_MAX_TOKENS = judging.MAX_TOKENS
TOOL_NAME = judging.ORBENCH_RUBRIC_V1.tool_name
assert TOOL_NAME == judging.XSTEST_RUBRIC_V1.tool_name


def original_request(rubric_name: str, prompt: str, response: str) -> Tuple[str, Dict[str, Any]]:
    """``(key, Anthropic params)`` of the Sonnet judge's request for (request, response) under ``rubric_name``: the store's request key
    and the exact params ``ClaudeJudge`` sent (``judging.render_request`` at the default protocol)."""
    rubric = RUBRICS[rubric_name]
    req = JudgeRequest(prompt=prompt, candidate_a=response, candidate_b=None, roles=None, rubric_sha=rubric.sha,
                       judge_model=ORIGINAL_MODEL, protocol_sha=judging.protocol_sha(rubric))
    return req.key(), judging.render_request(rubric, req)


def request_key(rubric_name: str, prompt: str, response: str) -> str:
    return original_request(rubric_name, prompt, response)[0]


def translate(params: Mapping[str, Any], max_tokens: int, reasoning_effort: Optional[str] = None,
              disable_cache: bool = False) -> Dict[str, Any]:
    """The Anthropic params -> one chat-completions body (no ``model``: the batch names it), ``parallel_tool_calls`` false (the judge
    entry drops it where the endpoint does not list it). Refuses params it cannot translate faithfully (a sampling parameter, thinking
    other than disabled, more than one tool, a non-forced tool choice)."""
    if "extra_body" in params or "temperature" in params:
        raise ValueError("the original request carried a sampling parameter; this translation sends none")
    if dict(params.get("thinking") or {}) != {"type": "disabled"}:
        raise ValueError(f"thinking {params.get('thinking')!r}: only the disabled form maps to reasoning effort none")
    tools = params.get("tools") or []
    if len(tools) != 1 or (params.get("tool_choice") or {}).get("type") != "tool":
        raise ValueError("expected exactly one tool and a forced tool choice")
    t = tools[0]
    body: Dict[str, Any] = {"messages": [{"role": "system", "content": params["system"]}] + [dict(m) for m in params["messages"]]}
    body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                       "parameters": copy.deepcopy(dict(t["input_schema"]))}}]
    body["tool_choice"] = {"type": "function", "function": {"name": params["tool_choice"]["name"]}}
    body["parallel_tool_calls"] = False
    body["max_tokens"] = int(max_tokens)
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if disable_cache:
        body["prompt_cache_options"] = {"mode": "explicit"}
    return body


def protocol(model: str, batch_variant: str, body_template: Mapping[str, Any]) -> Dict[str, Any]:
    """The request protocol of a judge model's body template (everything but the filled user message and max_tokens:
    :func:`rejudge.terra.judges.protocol` builds the template) -- hashed into the judge's ``protocol_sha256``."""
    return {"route": "openrouter_batch_api", "endpoint": ENDPOINT, "model": model, "batch_model_variant": batch_variant,
            "body_template": dict(body_template), "temperature": None, "translated_from": {
                "judge_model": ORIGINAL_MODEL, "protocol_sha16": judging.protocol_sha(judging.ORBENCH_RUBRIC_V1),
                "protocol_code_version": judging.PROTOCOL_CODE_VERSION, "thinking": dict(judging.THINKING),
                "max_tokens": judging.MAX_TOKENS}}


def rubric_sha256(rubric_name: str) -> str:
    """The FULL sha256 of the rubric's canonical JSON (name, version, text, tool schema): ``Rubric.sha`` is its first 16 hex."""
    import hashlib
    from sisteer.capeval.manifest import canonical_json
    r = RUBRICS[rubric_name]
    full = hashlib.sha256(canonical_json({"name": r.name, "version": r.version, "text": r.text,
                                           "tool_schema": dict(r.tool_schema)}).encode("utf-8")).hexdigest()
    assert full[:16] == r.sha, (full, r.sha)
    return full


# --------------------------------------------------------------------------------------------
# parsing one batch result entry
# --------------------------------------------------------------------------------------------
def parse_result(entry: Mapping[str, Any], rubric_name: str = "or_bench_cpi") -> Dict[str, Any]:
    """One ``results[]`` entry -> ``{outcome, verdict, error, status_code, finish_reason, usage}``. Both C/P/I rubrics share the tool
    and its validation, so one parser reads either."""
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
    calls = msg.get("tool_calls") or []
    if not calls:
        content = msg.get("content")
        if finish == "length":
            return dict(base, outcome="truncated", error={"kind": "truncated", "message": "finish_reason=length with no complete call"})
        return dict(base, outcome="parse_error", error={"kind": "no_tool_call",
                                                        "content_chars": len(content) if isinstance(content, str) else None})
    match = [c for c in calls if ((c or {}).get("function") or {}).get("name") == TOOL_NAME]
    if not match:
        return dict(base, outcome="parse_error", error={"kind": "wrong_tool", "names": [((c or {}).get("function") or {}).get("name") for c in calls]})
    if finish == "length":
        return dict(base, outcome="truncated", error={"kind": "truncated", "message": "finish_reason=length"})
    args_text = (match[0].get("function") or {}).get("arguments")
    try:
        args = json.loads(args_text) if isinstance(args_text, str) else args_text
    except (ValueError, TypeError) as e:
        return dict(base, outcome="parse_error", error={"kind": "malformed_json", "message": str(e)[:200]})
    if not isinstance(args, Mapping):
        return dict(base, outcome="parse_error", error={"kind": "malformed_json", "message": "arguments are not an object"})
    try:
        v = judging._sanitize_verdict(args, RUBRICS[rubric_name])
    except judging.JudgingError as e:
        kind = "welded_label" if isinstance(e, judging.JudgeMalformedLabel) else "schema_validation"
        return dict(base, outcome="parse_error", error={"kind": kind, "message": str(e)[:300]})
    verdict = {"label": v["label"], "reasoning_chars": len(str(args.get("reasoning") or "")), "n_tool_calls": len(calls)}
    return dict(base, outcome="verdict", verdict=verdict)
