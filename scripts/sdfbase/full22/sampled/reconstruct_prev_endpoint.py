#!/usr/bin/env python
"""Reconstruct the endpoint record a LATER ladder rung needs from a standup whose ``endpoint.json`` is gone.

``mt_serving_up.py`` UNLINKS ``endpoint.json`` at teardown, so the sampled ladder's
16,384 rung — a separate job that must read the 8,192 standup's recorded proxy hop through ``f22s_prev_endpoint`` — found
nothing and refused (eight legs). ``f22_standup`` now keeps a byte-identical copy at READY (``endpoint.recorded.json``); for
a standup that ended BEFORE that copy existed, this script rebuilds the fields the transport approval reads
(``hop_timeout_guard.read_endpoint_upstream_timeout``: ``upstream_timeout_s``) from the records that DO survive:

  * ``<standup dir>/standup_jobs.jsonl`` — the standup's own record (job_id, key, view, mode, eval, sha, geometry);
  * the proxy hop default IN FORCE on this tree, ``sisteer.capeval.serving.proxy.HOP_TIMEOUT_S`` — the value
    ``mt_serving_up.py`` records as ``upstream_timeout_s`` when ``--upstream-timeout-s`` is not passed, which the sampled
    phases never pass (``target_sampled.sbatch`` calls ``f22_standup`` without ``F22_STANDUP_EXTRA``).

The file it writes (``endpoint.reconstructed.json``) carries ``proxy_upstream_timeout_s: null`` (the flag was not given) and a
``reconstructed`` block naming both sources, so a reader can tell it from a READY record. REFUSES (rc 2, nothing written)
when the standup record is missing or malformed, or when the record says a ``--upstream-timeout-s`` override was in play
(``upstream_timeout_override`` present) — then the hop cannot be known and the rung must not pretend it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

RECONSTRUCTED_NAME = "endpoint.reconstructed.json"
STANDUP_RECORD = "standup_jobs.jsonl"
#: the standup-record fields copied verbatim (the geometry the transport approval and the ladder record read)
COPIED = ("job_id", "node", "key", "view", "mode", "eval", "sha", "lens_reduction_mode", "max_num_seqs", "max_model_len",
          "attested_max_num_seqs", "attested_max_model_len", "batched_tokens", "cluster")


class Refusal(SystemExit):
    def __init__(self, why: str):
        super().__init__(2)
        self.why = why
        print(f"reconstruct_prev_endpoint: REFUSING — {why}", file=sys.stderr)


def hop_timeout_default() -> float:
    from sisteer.capeval.serving.proxy import HOP_TIMEOUT_S  # the same symbol mt_serving_up.py records
    return float(HOP_TIMEOUT_S)


def read_standup_record(standup_dir: Path) -> dict:
    p = standup_dir / STANDUP_RECORD
    if not p.is_file():
        raise Refusal(f"{p}: no standup record — nothing to reconstruct from")
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise Refusal(f"{p}: empty standup record")
    try:
        rec = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise Refusal(f"{p}: last line is not JSON ({exc})")
    if not isinstance(rec, dict) or "max_model_len" not in rec or "job_id" not in rec:
        raise Refusal(f"{p}: last record lacks job_id / max_model_len")
    if rec.get("upstream_timeout_override") is not None:
        raise Refusal(f"{p}: the standup recorded an --upstream-timeout-s override ({rec['upstream_timeout_override']}); "
                      "the hop in force cannot be reconstructed from the default")
    return rec


def reconstruct(standup_dir: Path) -> dict:
    rec = read_standup_record(standup_dir)
    hop = hop_timeout_default()
    out = {k: rec.get(k) for k in COPIED}
    out.update({
        "proxy_upstream_timeout_s": None,                    # the flag as given: not given (the sampled phases pass none)
        "upstream_timeout_s": hop,                           # the value in force = the tree's default
        "reconstructed": {
            "why": "the standup's endpoint.json is unlinked at teardown (mt_serving_up.py); a later rung job needs its recorded proxy hop",
            "from_record": str(standup_dir / STANDUP_RECORD),
            "hop_source": "sisteer.capeval.serving.proxy.HOP_TIMEOUT_S (the default mt_serving_up.py records when --upstream-timeout-s is not passed)",
            "at": datetime.now(timezone.utc).isoformat(),
            "decision": "sdfbase-llama8b D228q (2026-09-12)",
        },
    })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--standup-dir", type=Path, required=True, help="the previous rung's standup dir (holds standup_jobs.jsonl)")
    ap.add_argument("--force", action="store_true", help="rewrite an existing reconstructed record")
    a = ap.parse_args(argv)
    sd = a.standup_dir.resolve()
    if (sd / "endpoint.json").is_file():
        print(sd / "endpoint.json"); return 0                # the READY record is still there: nothing to reconstruct
    if (sd / "endpoint.recorded.json").is_file():
        print(sd / "endpoint.recorded.json"); return 0       # the durable copy: authoritative, byte-identical to READY
    target = sd / RECONSTRUCTED_NAME
    if target.is_file() and not a.force:
        print(target); return 0
    doc = reconstruct(sd)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(target)
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
