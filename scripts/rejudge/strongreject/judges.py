"""The StrongREJECT track's judge table: which model re-judges the StrongREJECT inputs, and every parameter that belongs to that model.

The judge TEXT never changes: the Haiku 4.5 judge's rubric (``STRONGREJECT_PROMPT`` / ``_V2``), its ``StrongREJECTRating`` tool and its
score formula, imported from ``scripts/judge_batch.py`` by ``request.py``. What a model changes is a config entry here, read by every
stage, not a code fork:

    build_manifests.py --judge minimax-m3   ``track.json`` takes the entry (model, track name, request settings, reply cap, re-asks,
                                            health thresholds, estimate assumptions); the bodies are built from it
    file_records.py --judge minimax-m3      files under the entry's ``judge_model``, its protocol (the body template its request
                                            settings produce) hashed into ``protocol_sha256``; it refuses a ledger whose ``track.json``
                                            names another model or other request settings than the entry's

An entry's ``request`` holds the request settings (``request.request_body`` turns them into a body; every key is required):

    mode                  ``tool`` (the ``StrongREJECTRating`` function, ``tool_choice`` naming it, as the Haiku judge's Anthropic tool
                          choice did) | ``json_schema`` (``response_format`` strict, the same schema)
    reasoning_effort      ``reasoning: {"effort": ...}`` when set
    reasoning             a whole ``reasoning`` OBJECT that replaces the effort one (``{"enabled": false}``) when set
    disable_cache         ``prompt_cache_options: {"mode": "explicit"}`` with no breakpoint (OpenAI's implicit-cache switch) when true
    parallel_tool_calls   ``False`` (sent) | ``None`` (omitted: the ``:batch`` variant does not list the parameter)
    max_tokens            the reply cap (kept OUT of the judge's identity, so a re-ask with ``truncated_retry_max_tokens`` is the same
                          judge)

``estimate`` is the cost model the dry run prices with (USD at ``batchfile.PRICES_PER_MTOK``, plus the 5.5% credit fee;
``build_manifests.cost_bands``): ``input_ratio`` (the provider's prompt tokens per o200k token of the filled prompt plus
``overhead_tokens``; unused when ``tokenizer`` pins the model's own tokenizer, which counts them exactly), ``output_tokens`` /
``output_tokens_by_eval`` (completion tokens per call), ``cache_read_tokens_per_request`` or ``cache_read_share`` (what the provider bills
at the cache-read price; implicit caching only), ``parse_error_rate`` and ``reask_fail_rate`` (the first-pass share re-asked, and the share
of those that fail again), with ``*_high`` variants for the dry run's high end; ``turnaround`` the batch times observed.

THE MINIMAX-M3 ENTRY (the paper's StrongREJECT judge). ``minimax/minimax-m3:batch`` on OpenRouter's Batch API, served by DeepInfra (fp8,
tag ``deepinfra/fp8/flex``; its only batch provider). Its request settings are the ones the judge comparison sent and harvested on this
model (3,271 requests of the pilot's stratified draw, in batches of 2,000 / 1,271, every batch ``completed``, none failed); a full run
may carry that comparison's verdicts (``build_manifests.py --reuse-run-dir``, which refuses a directory of other request settings):
  * the forced call: the ``:batch`` variant lists ``tools`` and ``tool_choice`` (catalogue 2026-11-10), the original judge's tool choice
    is kept. DeepInfra's endpoint lists ``supports_tool_choice`` ``{auto: true, required: false, function: false}``, so the force is not
    honoured: the model calls the tool by itself (it writes a short rationale, then the call) and on 5.35% of the comparison's first
    replies (175 / 3,271) answered in text instead (``no_tool_call``, finish ``stop``); a re-ask answered 160 of them with the call, so
    they are re-asked (``retry`` below). ``json_schema`` strict is not an option: the ``:batch`` variant does not list
    ``structured_outputs``. ``parallel_tool_calls`` is not listed, so it is omitted
  * reasoning OFF with the object ``{"enabled": false}``: MiniMax-M3's reasoning is optional and has no effort levels
    (``reasoning.mandatory`` false); the Haiku 4.5 judge ran without extended thinking (a forced Anthropic tool call rules it out).
    Every reply of the comparison reports 0 reasoning tokens
  * no cache fields: DeepInfra caches prompt prefixes by itself (``supports_implicit_caching``; the comparison's replies bill 39% of
    their prompt tokens at the cache-read price, 640 per reply on 3,448 of 3,449, the shared tool + template prefix; no cache-write
    tokens, no write price listed); ``prompt_cache_options`` is OpenAI's
  * no temperature, as the Haiku judge sent none (both defaults are 1.0)
  * ``max_tokens`` 1500, the Haiku judge's own cap: the comparison's 3,271 first replies have mean 375 completion tokens (the rationale
    text counts), p99 773, p99.9 1,153; 3 (0.09%) were cut at 1,500 and are re-asked once at 4,000 (``truncated_retry_max_tokens``)
  * re-asks: a parse error twice (three attempts in all), a cut reply once at the raised cap, a transient provider failure three times;
    a moderation rejection or a judge refusal never (recorded per item)
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402

#: the track's re-asks: a transient provider failure up to 3 times, a malformed call once, a cut call once WITH THE RAISED CAP, a
#: moderation rejection or a judge refusal never. The account / balance / flood outcomes are never terminal and never counted
#: (common/harvester.py)
BASE_RETRY = {"verdict": 0, "moderation": 0, "judge_refusal": 0, "parse_error": 1, "truncated": 1, "provider_error": 3, "missing": 3,
              "batch_failed": 3, "batch_expired": 3, "batch_cancelled": 3}
#: per-batch health pauses (common/harvester.py HEALTH_DEFAULTS)
BASE_HEALTH = {"min_count": 10, "identical_share": 0.5, "truncated_share": 0.2, "error_share": 0.5, "acknowledged": []}

DEFAULT = "minimax-m3"
#: ``harvester.error_signature`` of ``request.parse_result`` on a 200 reply with no tool call (``parse_error`` / ``no_tool_call``): every
#: such reply of the judge comparison carries it (``tests/test_rejudge_strongreject_judges.py`` recomputes it)
NO_TOOL_CALL_SIGNATURE = "31aba4654a23"

#: the judge record's ``notes`` (``{version}`` = v1 / v2): the text of the judge rows on record
_MINIMAX_NOTES = ("MiniMax-M3 re-judge of the paper's StrongREJECT {version} verdicts through OpenRouter's Batch API "
                  "(minimax/minimax-m3:batch, DeepInfra fp8; 2026-11-11): the same user message, tool name, description and schema the "
                  "Haiku 4.5 judge saw, the Anthropic tool call translated to OpenAI chat completions with tool_choice naming "
                  "StrongREJECTRating (DeepInfra does not honour a named tool choice, so a reply that answers in text instead is a parse "
                  "error and is re-asked), reasoning off ({{\"enabled\": false}}, mirroring Haiku without extended thinking), no "
                  "temperature (provider default 1.0, as Haiku's), the provider's implicit prefix cache left on. A SECOND judge beside the "
                  "Claude ones: its judgments never supersede theirs and the runs' primary judge is unchanged.")

JUDGES: Dict[str, Dict[str, Any]] = {
    "minimax-m3": {
        "judge_model": "minimax-m3", "label": "MiniMax-M3", "model": "minimax/minimax-m3", "track": "strongreject_minimax_m3",
        "endpoint": "/v1/chat/completions", "provider": None, "canonical_slug": "minimax/minimax-m3-20260531",
        "provider_note": "DeepInfra (fp8, tag deepinfra/fp8/flex), the only provider of minimax/minimax-m3:batch on 2026-11-10/11",
        "request": {"mode": "tool", "reasoning_effort": None, "reasoning": {"enabled": False}, "disable_cache": False,
                    "parallel_tool_calls": None, "max_tokens": 1500},
        # a reply in text instead of the call is a real per-item outcome of this provider (a 200 with finish ``stop``: never an account
        # failure), so its signature is pre-acknowledged: a re-ask batch where most fail again is recorded, not held for a person. The
        # error-share pause (half a batch unparseable) still stands
        "truncated_retry_max_tokens": 4000, "retry": dict(BASE_RETRY, parse_error=2),
        "health": dict(BASE_HEALTH, acknowledged=[NO_TOOL_CALL_SIGNATURE]),
        # the batch variant's context (catalogue 2026-11-10: context_length 524,288; DeepInfra max_completion_tokens 131,072)
        "context_tokens": 524288,
        # the model's own tokenizer (HF MiniMaxAI/MiniMax-M3 tokenizer.json): its count of the filled prompt + 673 equals the prompt
        # tokens DeepInfra billed on all 3,449 replies of the judge comparison (both rubrics), so the dry run counts input tokens exactly
        "tokenizer": {"hf_repo": "MiniMaxAI/MiniMax-M3", "file": "tokenizer.json",
                      "sha256": "bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e", "prompt_overhead_tokens": 673},
        # measured on the judge comparison (3,271 keys of the pilot's stratified draw, half of each stratum Haiku-harmful; 3,449 replies):
        #   output_tokens        completion tokens per call, post-stratified to the paper-row population (stratum x Haiku-harm class
        #                        weights); ``_high`` the draw's own mean (harm-oversampled, longer)
        #   cache_read_tokens_per_request  3,448 of 3,449 replies billed exactly 640 prompt tokens at the cache-read price (the shared tool +
        #                        template prefix), none written
        #   parse_error_rate     first-pass replies that answered in text (175 / 3,271); reask_fail_rate: of those re-asked once, the
        #                        share that answered in text again (15 / 175); the cost per reply billed was $0.000586 before the fee
        #   input_ratio(_high)   only without the tokenizer: billed prompt tokens per (o200k count + 300), all evals (per-eval maximum)
        "estimate": {"input_ratio": 1.277, "input_ratio_high": 1.392, "overhead_tokens": 300.0,
                     "output_tokens": 338.4, "output_tokens_high": 374.6,
                     "output_tokens_by_eval": {"actorattack": 384.0, "codeattack": 257.7, "crescendo": 379.9, "grok420": 336.6, "pair": 298.9,
                                               "pasttense": 330.6, "renellm": 286.8, "strongreject_bon": 311.9},
                     "output_tokens_by_eval_high": {"actorattack": 419.2, "codeattack": 312.6, "crescendo": 411.9, "grok420": 363.9, "pair": 343.7,
                                                    "pasttense": 379.7, "renellm": 339.3, "strongreject_bon": 371.5},
                     "cache_read_tokens_per_request": 640, "parse_error_rate": 0.0535, "reask_fail_rate": 0.0857},
        "seed_judge_id": "strongreject_{version}_minimax_m3_openrouter", "notes": _MINIMAX_NOTES,
        # observed batch turnarounds on OpenRouter (DeepInfra, minimax-m3; submit to harvest, minutes): 2,000 requests in 24 and 1,271 in
        # 35 (side by side), re-asks of 109 in 8 and 69 in 11, and 1,213 / 457 C/P/I and classifier requests in 24 / 12
        "turnaround": {"observed": [[2000, 24.3], [1271, 35.3], [109, 8.1], [69, 11.1], [1213, 24.2], [457, 12.1]], "completion_window_h": 24},
    },
}


def get(name: Optional[str] = None) -> Dict[str, Any]:
    """A deep copy of the entry ``name`` (default :data:`DEFAULT`); KeyError naming the known entries otherwise."""
    name = name or DEFAULT
    if name not in JUDGES:
        raise KeyError(f"unknown StrongREJECT judge {name!r}; known: {sorted(JUDGES)}")
    return copy.deepcopy(JUDGES[name])


def by_track(track: str) -> Optional[str]:
    """The entry whose ``track`` (the run directory's ``track.json`` ``track``) is ``track``, else None."""
    for name, e in JUDGES.items():
        if e["track"] == track:
            return name
    return None


def identity_request(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The request settings that are part of the judge's identity: everything but the reply cap."""
    rq = dict(entry["request"])
    rq.pop("max_tokens", None)
    return rq


def price(entry: Mapping[str, Any]):
    """``(input, output, cache read, cache write)`` USD / MTok at the ``:batch`` price (``batchfile.PRICES_PER_MTOK``)."""
    return BF.PRICES_PER_MTOK[entry["model"]]
