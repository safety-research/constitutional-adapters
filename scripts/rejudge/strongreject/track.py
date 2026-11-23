"""The StrongREJECT track's run-directory setup: ``track.json`` and the manifests built from distinct judge inputs.

    from rejudge.strongreject.track import init_run, add_inputs
    init_run(run_dir, judge="minimax-m3")                 # writes track.json (model, endpoint, parser, retry policy, limits) from the
                                                          # judges.JUDGES entry (default: judges.DEFAULT)
    add_inputs(run_dir, inputs, prefix="b")               # inputs: iterable of {k, kind, dataset, revision, goal_id, goal, response}

``add_inputs`` builds one request per input (``request.build_entry``; ``custom_id`` = the live grader's cache key, which it recomputes
and refuses on a mismatch with the input's ``k``), estimates its tokens (o200k via tiktoken when importable, else chars / 4) and packs
the requests into batches under the byte and request caps (``batchfile.chunk``), one manifest per batch.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.strongreject import judges as J  # noqa: E402
from rejudge.strongreject import request as R  # noqa: E402

#: the estimate's fallbacks, for an entry whose ``estimate`` lacks the key: the expected visible output per call (o200k tokens) and the
#: chat framing + the function rendered for the model, beyond the prompt text
EST_OUTPUT_TOKENS = 122.0
EST_OVERHEAD_TOKENS = 300.0

_ENC = None


def _tokens(text: str) -> float:
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001
            _ENC = False
    if _ENC:
        if len(text) > 200_000:      # BPE is slow on huge degenerate runs; a ratio is close enough for an estimate
            return len(text) / 3.5
        return float(len(_ENC.encode_ordinary(text)))
    return len(text) / 4.0


def init_run(run_dir: str, max_tokens: Optional[int] = None, mode: Optional[str] = None, disable_cache: Optional[bool] = None,
             provider: Optional[Mapping[str, Any]] = None, extra: Optional[Mapping[str, Any]] = None, max_spend_usd: float = 0.0,
             requires_files: Optional[List[str]] = None, judge: Optional[str] = None) -> Dict[str, Any]:
    """``track.json`` from the ``judges.JUDGES`` entry ``judge`` (default ``judges.DEFAULT``): model, track name, request settings, the
    raised cap for a cut reply, re-asks, health thresholds and the estimate assumptions. ``max_tokens`` / ``mode`` / ``disable_cache`` /
    ``provider`` override the entry. ``max_spend_usd`` (USD incl. the credit fee; 0 = no ceiling) pauses the run before a submit would
    pass it; ``requires_files`` gates every submit on those files existing."""
    judge = judge or J.DEFAULT
    e = J.get(judge)
    rq = dict(e["request"])
    if max_tokens is not None:
        rq["max_tokens"] = int(max_tokens)
    if mode is not None:
        rq["mode"] = mode
    if disable_cache is not None:
        rq["disable_cache"] = bool(disable_cache)
    spec = {"track": e["track"], "model": e["model"], "endpoint": e["endpoint"], "provider": provider if provider is not None else e["provider"],
            "parser": "rejudge.strongreject.request:parse_result", "retry": dict(e["retry"]), "max_bytes": BF.DEFAULT_MAX_BYTES,
            "max_requests": BF.DEFAULT_MAX_REQUESTS, "submit_spacing_s": 2.0, "request": rq,
            "truncated_retry_max_tokens": e["truncated_retry_max_tokens"], "health": dict(e["health"]), "max_spend_usd": float(max_spend_usd),
            "judge": judge, "estimate": dict(e["estimate"])}
    if e.get("context_tokens"):
        spec["context_tokens"] = int(e["context_tokens"])
    if requires_files:
        spec["submit_gate"] = {"requires_files": list(requires_files), "note": "every listed file must exist before this run submits"}
    if extra:
        spec.update(extra)
    BF.write_track(run_dir, spec)
    return spec


def estimate_of(spec: Mapping[str, Any]) -> Dict[str, float]:
    """The estimate assumptions of a ``track.json`` (its own ``estimate``, else the fallbacks above)."""
    est = spec.get("estimate") or {}
    return {"overhead_tokens": float(est.get("overhead_tokens", EST_OVERHEAD_TOKENS)), "output_tokens": float(est.get("output_tokens", EST_OUTPUT_TOKENS)),
            "input_ratio": float(est.get("input_ratio", 1.0))}


def add_inputs(run_dir: str, inputs: Iterable[Mapping[str, Any]], prefix: str = "b", max_bytes: Optional[int] = None,
               max_requests: Optional[int] = None, goals: Optional[Mapping[str, Mapping[str, str]]] = None) -> List[Dict[str, Any]]:
    """Manifests for ``inputs`` (each with ``k``, ``kind``, ``dataset``, ``revision``, ``response`` and either ``goal`` text or a
    ``goal_id`` found in ``goals``). Returns the manifests written."""
    spec = BF.read_track(run_dir)
    rq = spec["request"]
    est = estimate_of(spec)
    header = BF.batch_header(spec["model"], spec["endpoint"], spec.get("provider"))

    def packed():
        seen = set()
        for x in inputs:
            goal = x.get("goal")
            if goal is None:
                goal = goals[x["goal_id"]]["text"]
            k = R.cache_key(x["kind"], x["dataset"], x["revision"], goal, x["response"])
            if x.get("k") and x["k"] != k:
                raise ValueError(f"input key {x['k'][:16]} does not recompute ({k[:16]}): the text changed since it was enumerated")
            if k in seen:
                continue
            seen.add(k)
            body = R.request_body(rq, x["kind"], goal, x["response"])
            prompt = body["messages"][0]["content"]
            yield BF.PackedRequest(k, BF.serialize_request(k, body),
                                   est_input_tokens=(_tokens(prompt) + est["overhead_tokens"]) * est["input_ratio"],
                                   est_output_tokens=est["output_tokens"])

    mans = []
    existing = len(BF.list_manifests(run_dir))
    for i, part in enumerate(BF.chunk(packed(), header, int(max_bytes or spec["max_bytes"]), int(max_requests or spec["max_requests"]))):
        key = f"{prefix}{existing + i:05d}"
        mans.append(BF.write_manifest(run_dir, key, header, part, spec["track"]))
    return mans
