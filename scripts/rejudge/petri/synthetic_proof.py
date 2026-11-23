#!/usr/bin/env python3
"""SCRATCH-PROOF ONLY: drive a Petri run directory through the REAL harvester against a FAKE Batch API that returns SYNTHETIC verdicts,
so the filing path (ledger -> records -> inbox -> fold) can be proven on a scratch registry while the provider route is blocked.

    PYTHONPATH=src:scripts python scripts/rejudge/petri/synthetic_proof.py --run-dir RUN --i-know-this-is-synthetic

The run directory is marked ``SYNTHETIC`` first; ``file_records.py`` refuses a marked run directory without ``--allow-synthetic`` and
never files one into ``~/registry``. Every reply is synthetic: deterministic pseudo-random scores from the custom_id, the texts say
SYNTHETIC, and the outcomes cover the vocabulary -- about 1 in 37 inputs a 403 moderation rejection (OpenRouter's moderation
metadata), 1 in 59 an OpenAI policy block (``provider_code: cyber_policy``, "blocked by our safety system": moderation), 1 in 61 a
``message.refusal``, 1 in 67 a refusal returned with ``finish_reason: content_filter`` (a judge refusal, not moderation), 1 in 53 a
schema failure on the first attempt (a score of 11) answered on the re-ask, 1 in 43 a transient 500 on the first attempt answered on
the re-ask, 1 in 71 text instead of the call on every attempt (a terminal parse_error).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common.harvester import Harvester  # noqa: E402
from rejudge.petri import request as R  # noqa: E402

MARK = "SYNTHETIC"


def synthetic_entry(req: Dict[str, Any], attempt: int) -> Dict[str, Any]:
    cid = req["custom_id"]
    h = int(hashlib.sha256(cid.encode()).hexdigest(), 16)
    if h % 37 == 0:
        return {"custom_id": cid, "response": None,
                "error": {"code": 403, "message": "SYNTHETIC: input flagged by moderation (scratch proof)", "metadata": {"reasons": ["synthetic"]}}}
    if h % 59 == 0:
        return {"custom_id": cid, "response": None,
                "error": {"code": 400, "message": "SYNTHETIC: Invalid prompt: your prompt was blocked by our safety system (scratch proof)",
                          "metadata": {"provider_name": "OpenAI", "provider_code": "cyber_policy", "error_type": "invalid_request"}}}
    if h % 43 == 0 and attempt == 0:
        return {"custom_id": cid, "response": None, "error": {"code": 500, "message": "SYNTHETIC: upstream error (scratch proof)"}}
    usage = {"prompt_tokens": 12000, "completion_tokens": 3000, "prompt_tokens_details": {"cached_tokens": 4400, "cache_write_tokens": 0},
             "completion_tokens_details": {"reasoning_tokens": 700}, "cost": 0.0}
    if h % 67 == 0:
        msg = {"role": "assistant", "content": None, "refusal": "SYNTHETIC refusal with a content_filter finish (scratch proof)"}
        return {"custom_id": cid, "error": None, "response": {"status_code": 200, "body": {"model": "SYNTHETIC", "usage": usage,
                "choices": [{"finish_reason": "content_filter", "message": msg}]}}}
    if h % 61 == 0:
        msg = {"role": "assistant", "content": None, "refusal": "SYNTHETIC refusal (scratch proof)"}
        return {"custom_id": cid, "error": None, "response": {"status_code": 200, "body": {"model": "SYNTHETIC", "usage": usage,
                "choices": [{"finish_reason": "stop", "message": msg}]}}}
    if h % 71 == 0:
        msg = {"role": "assistant", "content": "SYNTHETIC text instead of the call"}
        return {"custom_id": cid, "error": None, "response": {"status_code": 200, "body": {"model": "SYNTHETIC", "usage": usage,
                "choices": [{"finish_reason": "stop", "message": msg}]}}}
    tool = R.canonical_tool(req["body"]["tools"][0])
    own = R.own_dimension(tool)
    args = {"highlights": "SYNTHETIC highlights (scratch proof)", "summary": "SYNTHETIC summary (scratch proof)",
            own: 1 + (h >> 8) % 10, "eval_awareness": 1 + (h >> 16) % 10, "scenario_realism": 1 + (h >> 24) % 10,
            "justification": "SYNTHETIC justification (scratch proof)"}
    if h % 53 == 0 and attempt == 0:
        args[own] = 11
    msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_synthetic", "type": "function",
                                                                 "function": {"name": "answer", "arguments": json.dumps(args)}}]}
    return {"custom_id": cid, "error": None, "response": {"status_code": 200, "body": {"model": "SYNTHETIC", "provider": "SYNTHETIC",
            "usage": usage, "choices": [{"finish_reason": "tool_calls", "message": msg}]}}}


class SyntheticAPI:
    """A fake Batch API: a batch is validating on its first listing, in progress on the next, then completed with one result per
    request from ``entry_fn(request, attempt)`` (default :func:`synthetic_entry`; ``attempt`` counts that custom_id's earlier results)."""

    def __init__(self, entry_fn=None, provider: str = "SYNTHETIC"):
        self.provider = provider
        self.batches: Dict[str, Dict[str, Any]] = {}
        self.seen: Dict[str, int] = {}
        self.entry_fn = entry_fn or synthetic_entry
        self.submits = 0

    def submit(self, body: bytes) -> Dict[str, Any]:
        self.submits += 1
        d = json.loads(body)
        bid = f"synthetic-{len(self.batches):04d}"
        self.batches[bid] = {"id": bid, "status": "validating", "model": d["model"], "provider": self.provider, "created_at": time.time(),
                             "request_counts": {"total": len(d["requests"]), "completed": 0, "failed": 0}, "_reqs": d["requests"]}
        return {k: v for k, v in self.batches[bid].items() if not k.startswith("_")}

    def _advance(self, b: Dict[str, Any]) -> None:
        if b["status"] == "validating":
            b["status"] = "in_progress"
            return
        if b["status"] != "in_progress":
            return
        res = []
        for r in b["_reqs"]:
            n = self.seen.get(r["custom_id"], 0)
            self.seen[r["custom_id"]] = n + 1
            res.append(self.entry_fn(r, n))
        b.update(status="completed", results=res, usage={"cost": 0.0}, completed_at=time.time())

    def iter_batches(self, statuses=None, created_after=None, max_pages=1000):
        for b in self.batches.values():
            self._advance(b)
            yield {k: v for k, v in b.items() if not k.startswith("_") and k != "results"}

    def get(self, bid: str) -> Dict[str, Any]:
        return {k: v for k, v in self.batches[bid].items() if not k.startswith("_")}

    def key_info(self) -> Dict[str, Any]:
        return {"synthetic": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--i-know-this-is-synthetic", action="store_true", required=True)
    ap.add_argument("--provider", default="SYNTHETIC", help="the provider name the fake batches report (a profile with served_by checks it)")
    args = ap.parse_args(argv)
    open(os.path.join(args.run_dir, MARK), "w").write("verdicts in this run directory are SYNTHETIC (scratch proof); never file them live\n")
    h = Harvester(args.run_dir, SyntheticAPI(provider=args.provider), log=print)
    for _ in range(args.cycles):
        h.cycle()
        if not h.busy():
            break
    h.progress(done=not h.busy())
    print(json.dumps(h.ledger.outcome_counts()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
