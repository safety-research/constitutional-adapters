"""The judge models the C/P/I track can send its judge inputs to: one CONFIG ENTRY per model (:data:`JUDGES`), never a code fork.

Every body starts from the track's own builders -- :func:`rejudge.terra.cpi.original_request` + :func:`rejudge.terra.cpi.translate`
(the exact params the Sonnet 5 C/P/I judge sent, recomputed from the store's text and checked against its request key) and
:func:`rejudge.terra.agentic.build_body` (the agentic prompt the scope stores) -- so the judge TEXT, rubric and output schema are the
original judge's whatever the model. A config entry only says what changes per model:

  openrouter_model / batch_variant   the batch header's model (OpenRouter's Batch API serves its ``:batch`` variant) and that variant
  provider                           a provider pin for the batch header, or None
  families.<family>.output           the request variant: cpi ``forced_function`` (the original forced tool call as a function call
                                     with ``tool_choice`` naming it) or ``tool_auto`` (the same tool, ``tool_choice`` "auto"); agentic
                                     ``json_schema_strict`` (the verdict's structured output as a strict ``response_format``)
  families.<family>.reasoning        the ``reasoning`` object sent (the original judges ran with thinking DISABLED), or None
  families.<family>.prompt_cache_options / parallel_tool_calls   sent when not None (parallel_tool_calls only with a tool: C/P/I)
  families.<family>.max_tokens       the first cap; ``truncated_retry_max_tokens`` the cap of the one re-ask of a cut call (the shared
                                     harvester's ``track.json`` key), None = a cut call is terminal at once
  families.<family>.est_* / calibration   the dry run's token and cost model (measured on the judge comparison's runs of the model)
  context_tokens, max_requests       the batch variant's context (a request whose prompt plus retry cap could exceed it is never sent)
                                     and the per-batch request cap of the manifests (``families.<family>.max_requests`` overrides it)

A run directory records ``request.judge`` and every setting explicitly (:func:`request_settings`); :func:`normalize` refuses a request
block that names no judge. ``tests/test_rejudge_request_goldens.py`` pins the bodies byte for byte and
``tests/test_rejudge_paper_judges.py`` the judge rows on record.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(os.path.dirname(_SCRIPTS), "src")
for _p in (_SCRIPTS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rejudge.terra import agentic as AG  # noqa: E402
from rejudge.terra import cpi as CPI  # noqa: E402

FAMILIES = ("cpi", "agentic")
CPI_OUTPUTS = ("forced_function", "tool_auto")
AGENTIC_OUTPUTS = ("json_schema_strict",)
DEFAULT_JUDGE = "minimax-m3"

JUDGES: Dict[str, Dict[str, Any]] = {
    # MiniMax-M3, the paper's C/P/I judge. Every setting below is what the judge comparison sent and verified on the Batch API (1,213
    # C/P/I + 457 classifier keys): accepted by the batch endpoint, served by DeepInfra fp8, every C/P/I key answered (7 first answers
    # without a call, re-asked once, all answered), 456 of 457 classifier keys answered (4 cut at 1500, re-asked at 8000: 3 answered,
    # 1 cut again). Reasoning is optional on minimax/minimax-m3:batch (catalogue reasoning.mandatory false) and is OFF ({"enabled":
    # false}: 0 reasoning tokens billed on 1,681 results), mirroring the Sonnet judges' disabled thinking.
    "minimax-m3": {
        "label": "MiniMax-M3",
        "judge_model": "minimax-m3",
        "openrouter_model": "minimax/minimax-m3",      # the batch header's model (results name minimax/minimax-m3)
        "batch_variant": "minimax/minimax-m3:batch",
        "canonical_slug": "minimax/minimax-m3-20260531",
        "provider": None,                             # DeepInfra (fp8, flex tier) is the only :batch provider (endpoints, 2026-11-10)
        "served_by": "DeepInfra fp8 (flex tier), the only provider of minimax/minimax-m3:batch",
        "track_suffix": "minimaxm3",
        "context_tokens": 524288,                     # the :batch variant's context_length (catalogue 2026-11-10)
        #: a request is never sent when its estimated prompt x this ratio plus its largest cap could exceed the context: the highest
        #: MiniMax-M3 prompt / o200k-estimate ratio measured on the judge comparison is 1.266 (StrongREJECT), 1.184 C/P/I, 0.945 agentic
        "context_ratio_guard": 1.35,
        "max_requests": 10000,                        # per batch: a failed or expired batch re-asks at most 10k; byte cap 180 MB as ever
        "families": {
            "cpi": {
                "output": "forced_function",
                # the Sonnet judge's own cap (judging.MAX_TOKENS); the comparison's largest answer was 586 tokens of 1,220, none cut
                "max_tokens": 1024,
                "reasoning": {"enabled": False},
                "prompt_cache_options": None,         # an OpenAI knob; DeepInfra caches implicitly (cache reads billed at 0.2x)
                "parallel_tool_calls": None,          # not a supported parameter of the :batch endpoint
                "truncated_retry_max_tokens": None,   # the Sonnet C/P/I judge abstained on a cut call; the comparison cut none
                "est_output_tokens": 137.6, "est_overhead_tokens": 220.0,
                "calibration": {
                    "source": "the judge comparison's first attempts (1,213 C/P/I requests)",
                    "prompt_ratio": 1.1841,           # prompt_tokens / (o200k(messages) + overhead)
                    "completion_tokens": {"low": 121.37, "central": 137.64, "high": 236.0},   # pop.-weighted mean, pilot mean, pilot p90
                    "charged_over_list": 0.8216,      # OpenRouter's charge / the tokens at list batch price (implicit cache reads)
                    "reask_share": 0.0058,            # 7 parse errors (no call) of 1,213, each re-asked once
                    "truncated_share": 0.0,
                },
            },
            "agentic": {
                "output": "json_schema_strict",
                "max_tokens": 1500,                   # the Sonnet judge's first rung (judge_sonnet.VERDICT_MAX_TOKENS)
                "reasoning": {"enabled": False},
                "prompt_cache_options": None,
                "parallel_tool_calls": None,
                "truncated_retry_max_tokens": 8000,   # the Sonnet judge's top rung
                # a request cap the C/P/I batches never share: after a timed-out POST the submitter adopts the one account batch of the same
                # model and request count created within 15 minutes, so two run directories must not mint batches of equal counts
                "max_requests": 9000,
                "est_output_tokens": 436.8, "est_overhead_tokens": 300.0,
                "calibration": {
                    "source": "the judge comparison's first attempts (457 classifier requests)",
                    "prompt_ratio": 0.9448,
                    "completion_tokens": {"low": 404.98, "central": 436.77, "high": 657.8},
                    "charged_over_list": 0.9213,
                    "reask_share": 0.0088,            # 4 cut at 1500 of 457
                    "truncated_share": 0.0088,
                },
            },
        },
        "seed_suffix": "minimaxm3_openrouter",
        #: the judge record's ``notes`` per family (``{rubric}`` = or_bench_cpi / xstest_cpi): the text of the judge rows on record
        "notes": {
            "cpi": ("MiniMax-M3 re-judge of the paper's {rubric} v1 verdicts through OpenRouter's Batch API (minimax/minimax-m3:batch, served "
                    "by DeepInfra fp8; staged 2026-11-11): the request capeval.judging.render_request builds for the Sonnet 5 judge (same system "
                    "framing, rubric text, compliance_rating tool and schema), the forced Anthropic tool call sent as an OpenAI function call "
                    "with tool_choice naming it (DeepInfra's batch endpoint lists only the none / auto tool choices and evidently treats the "
                    "named choice as auto: 7 of the bake-off's 1,213 first answers carried no call and were re-asked once), reasoning off "
                    "(reasoning.enabled false, mirroring the Sonnet judge's disabled thinking), no sampling parameter (as the Sonnet judge), "
                    "max_tokens 1024 (the Sonnet judge's cap). A SECOND judge beside the Sonnet one: its judgments never supersede the Sonnet "
                    "judgments and the runs' primary judge is unchanged."),
            "agentic": ("MiniMax-M3 re-judge of the paper's agentic-misalignment classifier verdicts (murder + leaking) through OpenRouter's Batch "
                        "API (minimax/minimax-m3:batch, served by DeepInfra fp8; staged 2026-11-11): the same single user message the Sonnet 5 "
                        "judge received (judge_rubrics.PROMPTS[am_leak.v1 | am_murder.v1] filled with the response and "
                        "classify.get_email_context), the same HarmfulVerdict schema as a strict json_schema response format, reasoning off "
                        "(reasoning.enabled false, mirroring the disabled thinking), no sampling parameter, max_tokens 1500 with one re-ask of a "
                        "cut verdict at 8000 (the Sonnet judge's first and top rungs). A SECOND judge beside the Sonnet one; the runs' primary "
                        "judge is unchanged."),
        },
        #: the judge record's ``settings_evidence``: the judge comparison's files, kept outside the repository (named ``<path>``, as the
        #: published judge rows name them), and what they show
        "evidence": {
            "request_format": "<path>, <path> (protocol; bodies/ hold the bytes sent)",
            "outcomes": "<path>, <path>",
            "agreement": "<path> (C/P/I kappa3 w 0.894, agentic call kappa w 0.731, final kappa w 0.903)",
            "endpoint": "<path> (DeepInfra deepinfra/fp8/flex; supports_tool_choice none/auto)",
        },
    },
}


def get(judge: Optional[str]) -> Dict[str, Any]:
    name = judge or DEFAULT_JUDGE
    try:
        return JUDGES[name]
    except KeyError:
        raise ValueError(f"judge {name!r}; one of {sorted(JUDGES)}") from None


def family_cfg(judge: Optional[str], family: str) -> Dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(f"family {family!r}; one of {FAMILIES}")
    return get(judge)["families"][family]


def track_name(judge: Optional[str], family: str) -> str:
    """``<family>_<suffix>`` (``cpi_minimaxm3``, ``agentic_minimaxm3``): the run directories' ``track`` and the records namespace."""
    return f"{family}_{get(judge)['track_suffix']}"


def parser_spec(judge: Optional[str], family: str) -> str:
    fc = family_cfg(judge, family)
    if family == "agentic":
        if fc["output"] not in AGENTIC_OUTPUTS:
            raise ValueError(f"agentic output {fc['output']!r}; one of {AGENTIC_OUTPUTS}")
        return "rejudge.terra.agentic:parse_result"
    if fc["output"] not in CPI_OUTPUTS:
        raise ValueError(f"cpi output {fc['output']!r}; one of {CPI_OUTPUTS}")
    return "rejudge.terra.cpi:parse_result"


def request_settings(judge: Optional[str], family: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """The ``track.json`` ``request`` block: the judge entry's name and every setting, explicitly."""
    judge = judge or DEFAULT_JUDGE
    fc = family_cfg(judge, family)
    mt = int(max_tokens or fc["max_tokens"])
    return {"judge": judge, "max_tokens": mt, "output": fc["output"], "reasoning": copy.deepcopy(fc.get("reasoning")),
            "prompt_cache_options": copy.deepcopy(fc.get("prompt_cache_options")), "parallel_tool_calls": fc.get("parallel_tool_calls")}


def normalize(rq: Mapping[str, Any]) -> Dict[str, Any]:
    """A ``track.json`` ``request`` block -> the explicit settings :func:`cpi_body` / :func:`agentic_body` read. Refuses (ValueError) a
    block that names no judge entry: a body is never built from settings the block does not state."""
    if "judge" not in rq:
        raise ValueError(f"request block {dict(rq)} names no judge entry (request.judge); build the run directory with --judge")
    return {"judge": rq["judge"], "max_tokens": int(rq["max_tokens"]), "output": rq["output"], "reasoning": rq.get("reasoning"),
            "prompt_cache_options": rq.get("prompt_cache_options"), "parallel_tool_calls": rq.get("parallel_tool_calls")}


def _finish(body: Dict[str, Any], s: Mapping[str, Any]) -> Dict[str, Any]:
    if s.get("reasoning") is not None:
        body["reasoning"] = copy.deepcopy(dict(s["reasoning"]))
    if s.get("prompt_cache_options") is not None:
        body["prompt_cache_options"] = copy.deepcopy(dict(s["prompt_cache_options"]))
    return body


def cpi_body(params: Mapping[str, Any], s: Mapping[str, Any]) -> Dict[str, Any]:
    """The Sonnet judge's C/P/I params -> one chat-completions body under the settings ``s`` (:func:`normalize`)."""
    out = s.get("output") or "forced_function"
    body = CPI.translate(params, max_tokens=int(s["max_tokens"]), reasoning_effort=None, disable_cache=False)
    if s.get("parallel_tool_calls") is None:
        body.pop("parallel_tool_calls", None)
    else:
        body["parallel_tool_calls"] = bool(s["parallel_tool_calls"])
    if out == "tool_auto":
        body["tool_choice"] = "auto"
    elif out != "forced_function":
        raise ValueError(f"cpi output {out!r}; one of {CPI_OUTPUTS}")
    return _finish(body, s)


def agentic_body(prompt: str, s: Mapping[str, Any]) -> Dict[str, Any]:
    out = s.get("output") or "json_schema_strict"
    if out not in AGENTIC_OUTPUTS:
        raise ValueError(f"agentic output {out!r}; one of {AGENTIC_OUTPUTS}")
    # no tools in the agentic request, so parallel_tool_calls has no meaning there and is never sent
    body = AG.build_body(prompt, max_tokens=int(s["max_tokens"]), reasoning_effort=None, disable_cache=False)
    return _finish(body, s)


def build_entry(family: str, x: Mapping[str, Any], rq: Mapping[str, Any]) -> Dict[str, Any]:
    """``{custom_id, body}`` for one row of the scope's ``texts.jsonl.gz`` under a ``track.json`` request block. A C/P/I key is
    recomputed from the text and must equal the scope's (the text is the text the Sonnet judge saw)."""
    s = normalize(rq)
    if family == "cpi":
        key, params = CPI.original_request(x["rubric"], x["prompt"], x["response"])
        if key != x["k"]:
            raise ValueError(f"request key {key[:16]} != the store's {x['k'][:16]}: the text is not the text the judge saw")
        return {"custom_id": key, "body": cpi_body(params, s)}
    if family != "agentic":
        raise ValueError(f"family {family!r}")
    return {"custom_id": x["k"], "body": agentic_body(x["prompt"], s)}


def protocol(judge: Optional[str], family: str) -> Dict[str, Any]:
    """The request protocol hashed into the judge row's ``protocol_sha256``: the track's protocol record (:func:`cpi.protocol` /
    :func:`agentic.protocol`, which name the original judge it was translated from) with this judge's model, body template (no texts)
    and batch provider pin. max_tokens stays out of it (recorded in the judge's extra), so a re-ask at a larger cap is the same judge."""
    cfg = get(judge)
    s = normalize(request_settings(judge, family, max_tokens=1))
    if family == "cpi":
        _, params = CPI.original_request("or_bench_cpi", "{prompt}", "{response}")
        body = cpi_body(params, s)
        body.pop("max_tokens")
        body["messages"][1] = {"role": "user", "content": "<the rubric's text, str.format(prompt=, response=)>"}
        base = CPI.protocol(cfg["openrouter_model"], cfg["batch_variant"], body)
    else:
        body = agentic_body("<judge_rubrics.PROMPTS[tag].format(response=, email_context=)>", s)
        body.pop("max_tokens")
        base = AG.protocol(cfg["openrouter_model"], cfg["batch_variant"], body)
    base["provider"] = copy.deepcopy(cfg.get("provider"))
    return base


def estimate_input_tokens(text_tokens: float, judge: Optional[str], family: str) -> float:
    """The dry run's input estimate of one request: o200k tokens of its messages + the framing overhead (the estimator the calibration's
    ``prompt_ratio`` was measured against)."""
    return float(text_tokens) + float(family_cfg(judge, family)["est_overhead_tokens"])


def context_ok(est_input: float, judge: Optional[str], family: str) -> bool:
    """False when ``est_input`` x the guard ratio plus the request's largest cap (first or retry) could exceed the model's context."""
    cfg = get(judge)
    ctx = cfg.get("context_tokens")
    if not ctx:
        return True
    fc = family_cfg(judge, family)
    cap = max(int(fc["max_tokens"]), int(fc.get("truncated_retry_max_tokens") or 0))
    return est_input * float(cfg.get("context_ratio_guard") or 1.0) + cap <= int(ctx)


def cost_bands(judge: Optional[str], family: str, n_requests: int, est_input_tokens: float, prices) -> Optional[Dict[str, Any]]:
    """Low / central / high USD (incl. the credit fee) for ``n_requests`` whose summed estimator input is ``est_input_tokens``, from the
    family's calibration (None without one). ``prices`` = (input, output) USD / MTok at the batch price.

      low      measured prompt ratio, population-weighted completion mean, the pilot's charge-over-list factor (implicit cache reads),
               the measured re-ask share
      central  measured prompt ratio, the pilot's completion mean, list batch price (no cache credit), the measured re-ask share
      high     prompt ratio x 1.10, the pilot's completion p90, list batch price, 3x the measured re-ask share, and every cut call's
               re-ask billed at its full retry cap
    """
    from rejudge.common.batchfile import CREDIT_FEE
    fc = family_cfg(judge, family)
    cal = fc.get("calibration")
    if not cal:
        return None
    pin, pout = prices
    retry_cap = int(fc.get("truncated_retry_max_tokens") or 0)
    per_in = est_input_tokens / max(1, n_requests)
    out: Dict[str, Any] = {}
    for band in ("low", "central", "high"):
        ratio = cal["prompt_ratio"] * (1.10 if band == "high" else 1.0)
        comp = cal["completion_tokens"][band]
        reask = cal["reask_share"] * (3.0 if band == "high" else 1.0)
        tin = n_requests * per_in * ratio * (1.0 + reask)
        tout = n_requests * comp * (1.0 + reask)
        if band == "high" and retry_cap:
            tout += n_requests * cal.get("truncated_share", 0.0) * 3.0 * retry_cap
        usd = (tin * pin + tout * pout) / 1e6
        if band == "low":
            usd *= cal["charged_over_list"]
        out[band] = {"input_tokens": round(tin), "output_tokens": round(tout), "usd_with_fee": round(usd * (1.0 + CREDIT_FEE), 2)}
    out["assumptions"] = cost_bands.__doc__.split("\n\n", 1)[1].strip()
    return out
