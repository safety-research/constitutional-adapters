"""The Petri judge candidate table: what the chosen judge model is sent besides the judge text, and how its replies are read. (The
module's path is the judge comparison's: the judge rows on record and the run directories' ``track.json`` parser specs name it.)

THE JUDGE TEXT IS THE ORIGINAL'S. The candidate is sent the Opus 5 judge's stored prompt string and its one ``answer`` tool exactly as
:mod:`rejudge.petri.enumerate_inputs` read them (``request.request_parts``), byte for byte; only the model and the model-specific
parameters below change. The body is :func:`rejudge.petri.request.build_body` unsplit and without the OpenAI-only cache fields
(``split=False, cache_mode=None``): one user message holding the original string, the tool as an OpenAI function, ``tool_choice``
forcing ``answer``.

THE REASONING MIRROR. The Opus 5 judge omitted ``thinking`` and forced the tool call, so it answered WITHOUT reasoning (no stored judge
response holds a thinking block; ``request.py``). The candidate therefore runs at the LOWEST reasoning setting it accepts:
  z-ai/glm-5.3           reasoning mandatory: ``low`` is the floor (max / high / low; OpenRouter catalogue 2026-11-10)
``parallel_tool_calls`` is dropped because the model's ``:batch`` variant does not list it among its supported parameters (the parser
reads the first ``answer`` call either way). A Claude model is never a candidate (:func:`is_claude`).

PRICES are the ``:batch`` prices per MTok (input, output, cache read, cache write) from the OpenRouter catalogue of 2026-11-10; a price
absent from the catalogue (cache write) is the input price. They are registered into ``batchfile.PRICES_PER_MTOK`` on import so the
shared harvester and submitter can estimate and cap spend for the model (the harvester imports this module as the run's parser).
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.petri import request as R  # noqa: E402

#: the candidates. ``backend``: openrouter (the shared Batch API library)
CANDIDATES: Dict[str, Dict[str, Any]] = {
    "glm-5.3": {"role": "candidate", "backend": "openrouter", "model": "z-ai/glm-5.3", "reasoning_effort": "low",
                "max_tokens": 16000, "truncated_retry_max_tokens": 32000, "drop_parallel_tool_calls": True,
                "prices": (0.45, 2.00, 0.10, 0.45), "why": "reasoning is mandatory on GLM-5.3: low is its floor"},
}
#: candidates never run as a judge: every Claude / Anthropic model
SET_ASIDE = frozenset(n for n, c in CANDIDATES.items()
                      if c["backend"] == "anthropic" or str(c["model"]).startswith(("claude", "anthropic/")))


def is_claude(name: str) -> bool:
    """True for a candidate whose judge is a Claude / Anthropic model (never run; see ``SET_ASIDE``)."""
    c = CANDIDATES[name]
    return name in SET_ASIDE or c["backend"] == "anthropic" or str(c["model"]).lower().startswith(("claude", "anthropic/"))


def register_prices() -> None:
    """Make the candidates' ``:batch`` prices known to the shared library (estimates, the spend ceiling, per-request cost when a result
    carries no ``usage.cost``). Never overrides a price the library already holds."""
    for c in CANDIDATES.values():
        if c["backend"] == "openrouter":
            BF.PRICES_PER_MTOK.setdefault(c["model"], tuple(c["prices"]))


register_prices()

#: the shared harvester's parser for the candidates (``track.json`` ``parser``): the Petri track's chat-completions parser, which
#: OpenRouter's normalised responses fit for every provider; importing this module registers the prices first
parse_result = R.parse_result


def openrouter_body(prompt: str, tool: Mapping[str, Any], cand: Mapping[str, Any], max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """The chat-completions body for a candidate: the Petri track's body unsplit, without the OpenAI cache fields, with this model's
    reasoning floor and cap; ``parallel_tool_calls`` dropped where the ``:batch`` variant does not list it. A candidate's ``reasoning``
    object replaces the ``{"effort": ...}`` one."""
    body = R.build_body(prompt, tool, max_tokens=int(max_tokens or cand["max_tokens"]), reasoning_effort=cand["reasoning_effort"],
                        split=False, cache_prefix=False, cache_mode=None)
    if cand.get("reasoning") is not None:
        body["reasoning"] = copy.deepcopy(dict(cand["reasoning"]))
    if cand.get("drop_parallel_tool_calls"):
        body.pop("parallel_tool_calls", None)
    return body
