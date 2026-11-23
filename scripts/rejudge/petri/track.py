"""The Petri track's run-directory setup: ``track.json`` and the manifests built from distinct judge inputs.

    from rejudge.petri.track import init_run, add_inputs
    init_run(run_dir, judge="glm-5.3")                         # track.json (model, endpoint, parser, retry policy, limits, health,
                                                               # truncation cap, spend ceiling, submit gate, estimate assumptions)
    add_inputs(run_dir, inputs, tools, prefix="b")             # inputs: rows of enumerate_inputs' inputs.jsonl.gz

``add_inputs`` builds one request per distinct judge input (the profile's body, :func:`rejudge.petri.judges.arm_body`; ``custom_id`` =
``request.judge_input_key``, recomputed from the text and refused on a mismatch with the row's ``key``), orders them by behaviour and cacheable prefix so that the
requests sharing a prefix sit together in one batch (the cache entry the first of them writes is the one the rest read), never puts
two behaviours in one batch, and packs them under the byte and request caps (``batchfile.chunk``), one manifest per batch. A
manifest's ``extra`` carries its cache-aware token estimate (:func:`cache_estimate`): the shared prefix tokens per distinct prefix,
the rest per request.
"""
from __future__ import annotations

import collections
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.petri import judges as J  # noqa: E402
from rejudge.petri import request as R  # noqa: E402

#: re-asks: a transient provider failure up to 3 times; a malformed or schema-invalid reply once (the original judge re-asked a schema
#: failure up to 3 times in-conversation, inspect_scout max_attempts=3; here the re-ask is a fresh request from the same body); a cut
#: reply once WITH THE RAISED CAP below (re-sending the same cap cannot succeed); a moderation rejection or a judge refusal never
#: (recorded per item). The account / balance / flood outcomes are never terminal and never counted (common/harvester.py)
RETRY = {"verdict": 0, "moderation": 0, "judge_refusal": 0, "parse_error": 1, "truncated": 1, "provider_error": 3, "missing": 3,
         "batch_failed": 3, "batch_expired": 3, "batch_cancelled": 3}
#: per-batch health pauses (common/harvester.py HEALTH_DEFAULTS): an identical-error flood of half a batch is held for a person; a fifth
#: of a batch cut at its cap already says the reasoning setting or the cap is wrong (the longest Opus answer is 4,645 tokens)
HEALTH = {"min_count": 10, "identical_share": 0.5, "truncated_share": 0.2, "error_share": 0.5, "acknowledged": []}
#: expected visible answer per call (o200k): the Opus 5 answers' mean over the 9,417 production transcripts
EST_ANSWER_TOKENS = 2281.0
#: expected reasoning per call when a ``track.json`` estimate names none (the profile's ``est_reasoning_tokens`` is what init_run writes)
EST_REASONING_TOKENS = 1000.0
#: chat framing beyond the counted text (message wrapper, the function rendered for the model; the tool JSON itself is counted)
EST_FRAMING_TOKENS = 40.0


def init_run(run_dir: str, provider: Optional[Mapping[str, Any]] = None, extra: Optional[Mapping[str, Any]] = None,
             max_spend_usd: float = 0.0, requires_files: Optional[List[str]] = None, est_reasoning_tokens: Optional[float] = None,
             judge: Optional[str] = None) -> Dict[str, Any]:
    """``track.json`` of a profile of :mod:`rejudge.petri.judges` (default ``judges.DEFAULT``): its model, parser, track name, provider
    pin, request settings and truncation cap, so its requests are the candidate's exactly (``provider`` and those fields in ``extra`` are
    refused). ``max_spend_usd`` (USD incl. the credit fee; 0 = no ceiling) pauses the run before a submit would pass it;
    ``requires_files`` gates every submit on those files existing; ``estimate`` is what the manifests' cost estimates assume per request
    (the spend ceiling reads them; ``est_reasoning_tokens`` defaults to the profile's)."""
    prof = J.profile(judge)
    arm_fields = {"track": prof["track"], "model": prof["model"], "endpoint": prof["endpoint"], "parser": prof["parser"],
                  "provider": dict(prof["provider"]) if prof["provider"] else None, "request": J.arm_request(prof),
                  "truncated_retry_max_tokens": J.candidate(prof)["truncated_retry_max_tokens"], "judge": prof["name"],
                  "arm": prof["arm"], **J.arm_shape(prof)}
    clash = sorted(set(extra or {}) & set(arm_fields))
    if provider is not None or clash:
        raise ValueError(f"the {prof['name']} judge's provider pin and request fields come from its profile; refused: "
                         f"{(['provider'] if provider is not None else []) + clash}")
    est_r = float(prof["est_reasoning_tokens"] if est_reasoning_tokens is None else est_reasoning_tokens)
    spec = {"track": arm_fields["track"], "model": arm_fields["model"], "endpoint": arm_fields["endpoint"], "provider": arm_fields["provider"],
            "parser": arm_fields["parser"], "retry": RETRY, "max_bytes": BF.DEFAULT_MAX_BYTES, "max_requests": BF.DEFAULT_MAX_REQUESTS,
            "submit_spacing_s": 2.0, "request": arm_fields["request"], "truncated_retry_max_tokens": arm_fields["truncated_retry_max_tokens"],
            "health": dict(HEALTH), "max_spend_usd": float(max_spend_usd),
            "estimate": {"answer_tokens": EST_ANSWER_TOKENS, "reasoning_tokens": est_r, "framing_tokens": EST_FRAMING_TOKENS}}
    spec.update(arm_fields)
    if requires_files:
        spec["submit_gate"] = {"requires_files": list(requires_files), "note": "every listed file must exist before this run submits"}
    if extra:
        spec.update(extra)
    os.makedirs(run_dir, exist_ok=True)
    os.chmod(run_dir, 0o700)
    BF.write_track(run_dir, spec)
    return spec


def output_per_request(spec: Mapping[str, Any]) -> float:
    """The output tokens a manifest's estimate assumes per request (answer + reasoning, ``track.json`` ``estimate``)."""
    e = spec.get("estimate") or {}
    return float(e.get("answer_tokens", EST_ANSWER_TOKENS)) + float(e.get("reasoning_tokens", EST_REASONING_TOKENS))


def est_input_tokens(row: Mapping[str, Any]) -> float:
    t = row["tokens"]
    return float(t["head"] + t["rest"] + t["tool"]) + EST_FRAMING_TOKENS


def cache_estimate(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Token totals for a set of requests under the prefix cache: ``prefix`` = the shared prefix (tool + head) tokens of each request,
    ``uncached`` = the rest, ``distinct_prefix`` = one prefix's tokens per distinct prefix (the minimum written), ``n``."""
    prefix = uncached = 0.0
    per_prefix: Dict[str, float] = {}
    n = 0
    for r in rows:
        t = r["tokens"]
        p = float(t["head"] + t["tool"])
        prefix += p
        uncached += float(t["rest"]) + EST_FRAMING_TOKENS
        per_prefix[r["prefix_sha"]] = p
        n += 1
    return {"n": n, "prefix": prefix, "uncached": uncached, "distinct_prefix": sum(per_prefix.values()), "n_prefixes": len(per_prefix)}


def cost_range(model: str, est: Mapping[str, float], output_per_request: float) -> Dict[str, float]:
    """USD incl. the credit fee for a cache estimate: ``best`` (every prefix written once, read by the rest), ``no_cache`` (caching
    off), ``worst`` (every request writes its prefix and none reads)."""
    out = output_per_request * est["n"]
    best = BF.estimate_cost(model, est["uncached"], out, cache_read_tokens=est["prefix"] - est["distinct_prefix"],
                            cache_write_tokens=est["distinct_prefix"])
    none = BF.estimate_cost(model, est["uncached"] + est["prefix"], out)
    worst = BF.estimate_cost(model, est["uncached"], out, cache_write_tokens=est["prefix"])
    return {"best": round(best, 4), "no_cache": round(none, 4), "worst": round(worst, 4)}


def build_requests(rows: Iterable[Mapping[str, Any]], tools: Mapping[str, Mapping[str, Any]], spec: Mapping[str, Any]) -> Iterable[BF.PackedRequest]:
    prof = J.profile(spec.get("judge"))
    seen = set()
    for r in rows:
        tool = tools[r["tool_sha"]]
        k = R.judge_input_key(r["prompt"], tool)
        if r.get("key") and r["key"] != k:
            raise ValueError(f"input {r['phys']}: key {r['key'][:16]} does not recompute ({k[:16]}): the text changed since it was enumerated")
        cid = k
        if cid in seen:
            continue
        seen.add(cid)
        body = J.arm_body(prof, r["prompt"], tool)
        yield BF.PackedRequest(cid, BF.serialize_request(cid, body), est_input_tokens=est_input_tokens(r),
                               est_output_tokens=output_per_request(spec),
                               meta={"behavior": r["behavior"], "prefix_sha": r["prefix_sha"], "tokens": r["tokens"]})


def add_inputs(run_dir: str, inputs: Iterable[Mapping[str, Any]], tools: Mapping[str, Mapping[str, Any]], prefix: str = "b",
               max_bytes: Optional[int] = None, max_requests: Optional[int] = None,
               batch_extra: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Manifests for ``inputs`` (rows with ``prompt``, ``tool_sha``, ``key``, ``prefix_sha``, ``behavior``, ``tokens``): one batch
    holds one behaviour, its requests ordered by prefix. Returns the manifests written."""
    spec = BF.read_track(run_dir)
    header = BF.batch_header(spec["model"], spec["endpoint"], spec.get("provider"))
    by_beh: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for r in inputs:
        by_beh[r["behavior"]].append(r)
    mans = []
    existing = len(BF.list_manifests(run_dir))
    i = 0
    for beh in sorted(by_beh):
        rows = sorted(by_beh[beh], key=lambda r: (r["prefix_sha"], r["key"]))
        reqs = build_requests(rows, tools, spec)
        for part in BF.chunk(reqs, header, int(max_bytes or spec["max_bytes"]), int(max_requests or spec["max_requests"])):
            est = cache_estimate({"tokens": p.meta["tokens"], "prefix_sha": p.meta["prefix_sha"]} for p in part)
            extra = {"behavior": beh, "cache_estimate": est,
                     "cost_range_usd_with_fee": cost_range(spec["model"], est, output_per_request(spec))}
            if batch_extra:
                extra.update(batch_extra)
            key = f"{prefix}{existing + i:05d}_{beh}"
            mans.append(BF.write_manifest(run_dir, key, header, part, str(spec["track"]), extra=extra))
            i += 1
    return mans
