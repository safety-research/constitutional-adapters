"""The Petri judge's profiles: which model re-judges the paper's Petri transcripts, the request each transcript becomes, and the
registry judge row its verdicts file under. The judge TEXT is the original's in every profile (the Opus 5 judge's stored prompt string
and ``answer`` tool, read by :mod:`rejudge.petri.enumerate_inputs`); a profile only says which model reads it and with which
model-specific parameters.

  glm-5.3      the paper's Petri judge, chosen from the Petri judge comparison (own-dimension MAE 0.82 against the stored Opus 5
               verdicts, Spearman 0.91, per-run rho 0.97, 0 refusals, 238 of 238 parsed). Every request is EXACTLY the comparison's
               ``glm-5.3`` request: ``bakeoff.models.openrouter_body`` over ``bakeoff.models.CANDIDATES["glm-5.3"]`` -- the stored
               prompt as ONE user message, the stored tool as an OpenAI function, ``tool_choice`` forcing ``answer``,
               ``parallel_tool_calls`` dropped (GLM-5.3's ``:batch`` variant does not list it), ``reasoning: {"effort": "low"}``
               (reasoning is mandatory on GLM-5.3 and ``low`` is its floor), ``max_tokens`` 16,000, no cache fields; a truncated reply
               is re-asked once at 32,000; the parser is ``bakeoff.models:parse_result`` (whose import registers the ``:batch``
               prices). The candidate's table entry is the single source: this profile copies nothing from it.

PROVIDER (glm-5.3). Pinned to DeepInfra with ``{"only": ["deepinfra"], "allow_fallbacks": false}`` at the batch's top level, the two
provider options OpenRouter's Batch API accepts (``order`` and ``sort`` are refused there). Why pinned, and why both options:
  - the comparison's 238 verdicts, which a full run may carry and which the choice rests on, were all served by DeepInfra. Another
    provider serves another deployment of the weights (on OpenRouter's endpoint list DeepInfra's GLM-5.3 is fp4 and Morph's fp8), which
    is not the judge that was measured;
  - unpinned, a batch goes to the cheapest batch provider and providers at the same price share traffic, and a batch on a slow provider
    can sit in progress for hours;
  - ``only`` alone already keeps a batch off every provider outside the list; ``allow_fallbacks: false`` adds the documented outcome when
    DeepInfra has no capacity: the batch FAILS instead of an undocumented wait or move. The harvester re-asks a failed batch up to three
    times (the pin travels with every re-ask), then records ``batch_failed``, an infrastructure outcome that is not filed: its runs file
    ``partial`` until a follow-up run directory re-sends exactly those keys, whose verdicts the filing lets settle the failure
    (``file_records.read_outcomes``). The harvester does not back off between those re-asks, so a capacity outage longer than a few
    poll cycles ends in that follow-up rather than in a wait;
  - a provider list that does not serve the model in batch mode is a 404 at submit, which pauses the run before anything is spent.

IDENTITY. The judge row's protocol is the request (endpoint, model slug, the body template without the texts and without
``max_tokens``) plus, for a pinned profile, the provider the verdicts were SERVED by (``served_by``). The routing pin itself is not part
of it: the comparison's ledger was built unpinned and served by DeepInfra, the full run is pinned to DeepInfra, and their verdicts are one
judge. The filing enforces ``served_by`` per filed outcome (``file_records.py`` refuses a verdict served by any other provider), so the
judge id's claim holds for every row filed under it, whatever the routing was.

LEDGERS (glm-5.3). Two kinds of run directory hold verdicts of this judge: the full run's (``track`` ``petri_glm53``, built by
``build_manifests.py --judge glm-5.3``) and the comparison's own (``track`` ``petri_bakeoff`` with ``arm`` ``glm-5.3``), whose verdicts
the full run may carry: ``build_manifests.py --reuse-ledger`` excludes their keys and refuses unless every one of the ledger's request
bodies is byte-identical to the body the full run would send for that key and every verdict it holds was served by DeepInfra. A
follow-up run directory (``build_manifests.py --judge glm-5.3 --keys ...``) is a third ledger of the same kind as the full run's.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.petri import request as R  # noqa: E402
from rejudge.petri.bakeoff import models as M  # noqa: E402

DEFAULT = "glm-5.3"

PROFILES: Dict[str, Dict[str, Any]] = {
    "glm-5.3": {
        "track": "petri_glm53", "model": M.CANDIDATES["glm-5.3"]["model"], "endpoint": R.ENDPOINT, "judge_model": "glm-5.3",
        "parser": "rejudge.petri.bakeoff.models:parse_result", "arm": "glm-5.3",
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False}, "served_by": ["DeepInfra"],
        # reasoning tokens per request the manifests' estimates (so the spend ceiling) assume: the comparison's p90 (310) over its 238
        # verdicts (mean 186, p99 948); the answer length assumed is the Opus 5 answers' mean (2,281; GLM's measured mean 1,389)
        "est_reasoning_tokens": 310.0,
        # the full run's ledger, and the comparison's ledger whose verdicts it may carry (track -> the track.json fields that must match)
        "ledger_tracks": {"petri_glm53": {"judge": "glm-5.3"}, "petri_bakeoff": {"arm": "glm-5.3", "bakeoff": True}},
        "seed_judge_id": "petri_judge_glm53_openrouter", "canonical_slug": "z-ai/glm-5.3-20260816",
        # the judge record's ``chosen`` (and its notes' "chosen by ..."): the text of the judge row on record
        "chosen": "the authors, 2026-11-11, from the Petri judge bake-off (own-dimension MAE 0.82, Spearman 0.91, per-run rho 0.97, "
                  "0 refusals, 238/238 parsed against the stored Opus 5 verdicts)",
    },
}


def profile(name: Optional[str] = None) -> Dict[str, Any]:
    name = name or DEFAULT
    if name not in PROFILES:
        raise ValueError(f"unknown Petri judge {name!r}; one of {sorted(PROFILES)}")
    p = dict(PROFILES[name], name=name)
    if M.is_claude(p["arm"]):
        raise ValueError(f"{name}: a Claude model is never the Petri re-judge (this tool re-judges with non-Claude models only)")
    cand = M.CANDIDATES[p["arm"]]
    if cand["backend"] != "openrouter" or cand["model"] != p["model"]:
        raise ValueError(f"{name}: candidate {p['arm']} is not an OpenRouter candidate of {p['model']}")
    return p


def candidate(prof: Mapping[str, Any]) -> Mapping[str, Any]:
    return M.CANDIDATES[prof["arm"]]


def arm_request(prof: Mapping[str, Any]) -> Dict[str, Any]:
    """``track.json`` ``request`` of a profile: the dict the comparison's run directory recorded for the candidate."""
    cand = candidate(prof)
    return {"max_tokens": cand["max_tokens"], "reasoning_effort": cand["reasoning_effort"], "split": False, "cache_prefix": False,
            "cache_mode": None, "parallel_tool_calls": None if cand.get("drop_parallel_tool_calls") else False,
            **({"reasoning": dict(cand["reasoning"])} if cand.get("reasoning") is not None else {})}


def arm_shape(prof: Mapping[str, Any]) -> Dict[str, Any]:
    """The request form besides ``request``, as the comparison's ``track.json`` records it (absent there = these defaults)."""
    cand = candidate(prof)
    return {"tool_choice": cand.get("tool_choice", "forced"), "output": cand.get("output", "tool")}


def arm_body(prof: Mapping[str, Any], prompt: str, tool: Mapping[str, Any]) -> Dict[str, Any]:
    """One request body of a profile: exactly the comparison's (``bakeoff.models.openrouter_body``)."""
    return M.openrouter_body(prompt, tool, candidate(prof))


def ledger_identity(track: Mapping[str, Any], prof: Mapping[str, Any]) -> Dict[str, Any]:
    """The identity-bearing settings of a profile's ledger (``track.json``): the request without ``max_tokens`` and the request form.
    Refuses (ValueError) a ledger of another track or model, a probe (``id_suffix``), or one whose settings are not the profile's."""
    want_tracks: Mapping[str, Mapping[str, Any]] = prof["ledger_tracks"]
    t = track.get("track")
    if t not in want_tracks:
        raise ValueError(f"track.json track={t!r}, not one of {sorted(want_tracks)}; not a ledger of the {prof['name']} judge")
    for k, v in want_tracks[t].items():
        if track.get(k) != v:
            raise ValueError(f"track.json {k}={track.get(k)!r}, not {v!r}; not a ledger of the {prof['name']} judge")
    for k in ("model", "endpoint", "parser"):
        if track.get(k) != prof[k]:
            raise ValueError(f"track.json {k}={track.get(k)!r}, not {prof[k]!r}; not a ledger of the {prof['name']} judge")
    if track.get("id_suffix"):
        raise ValueError(f"a probe run directory (id_suffix {track['id_suffix']!r}) is never a verdict ledger")
    rq = dict(track.get("request") or {})
    want = arm_request(prof)
    got = {k: rq.get(k) for k in want if k != "max_tokens"}
    exp = {k: v for k, v in want.items() if k != "max_tokens"}
    shape = {k: track.get(k, d) for k, d in (("tool_choice", "forced"), ("output", "tool"))}
    if got != exp or shape != arm_shape(prof):
        raise ValueError(f"request settings {dict(got, **shape)} are not the {prof['arm']} candidate's {dict(exp, **arm_shape(prof))}")
    if track.get("truncated_retry_max_tokens") != candidate(prof)["truncated_retry_max_tokens"]:
        raise ValueError(f"truncated_retry_max_tokens {track.get('truncated_retry_max_tokens')} is not the candidate's "
                         f"{candidate(prof)['truncated_retry_max_tokens']}")
    return dict(exp, **shape)


def served_problems(prof: Mapping[str, Any], providers: Mapping[Optional[str], int]) -> List[str]:
    """Why outcomes served by these providers (provider -> count) may not file under ``prof``'s judge (empty = they may)."""
    allowed = prof.get("served_by")
    if not allowed:
        return []
    bad = {p: n for p, n in providers.items() if p not in allowed}
    return [f"{n} outcome(s) served by {p!r}, not by {allowed} (the {prof['name']} judge is the model as served there)" for p, n in sorted(bad.items(), key=str)]
