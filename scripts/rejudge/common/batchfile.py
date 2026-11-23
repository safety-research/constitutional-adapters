"""Batch bodies, size-aware chunking, manifests on disk, and cost estimates (stdlib only).

A RUN DIRECTORY (one per track and scope, e.g. ``<root>/strongreject/full``) holds::

    track.json                   the track spec (model, endpoint, parser, prices, retry policy, limits): :func:`write_track`
    manifests/<batch_key>.json   one manifest per batch (schema ``rejudge-batch-manifest/1``)
    bodies/<batch_key>.json.gz   the exact bytes POSTed for that batch, gzip'd; ``body_sha256`` in the manifest is over the
                                 UNCOMPRESSED bytes, so a submit re-reads, re-hashes and refuses a body that changed
    state.sqlite                 :mod:`state` (submission ledger + verdict cache)
    raw/<batch_key>__<id>.json.gz  every GET of a finished batch, verbatim (results included)
    progress.json                written by the harvester

A batch body is ``{"endpoint", "model", ["provider"], "completion_window", "requests": [...]}`` with ``requests`` LAST: OpenRouter reads
the body as a stream and returns 400 when ``requests`` comes first. Requests are serialized with ``ensure_ascii=True`` (a response that
holds a lone surrogate still serializes; the decoded text the model sees is unchanged) and compact separators.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

MANIFEST_SCHEMA = "rejudge-batch-manifest/1"
TRACK_SCHEMA = "rejudge-track/1"
#: 200 MB is OpenRouter's documented per-batch body cap; stay under it with margin (the server counts the bytes it receives)
DEFAULT_MAX_BYTES = 180 * 1000 * 1000
#: the per-batch request cap is unpublished ("generally follow the batch limits of the upstream providers"; OpenAI's is 50,000);
#: a 413 states the real one and the submitter splits on it
DEFAULT_MAX_REQUESTS = 50_000
#: OpenRouter's fee on credit purchases (Standard plan)
CREDIT_FEE = 0.055

#: USD per 1M tokens for the ``:batch`` variants, read from https://openrouter.ai/api/v1/models on 2026-11-11
#: (input, output, cache read, cache write); reasoning tokens bill as output. ``minimax/minimax-m3:batch`` is served by DeepInfra (fp8),
#: its only batch provider, which lists no cache-write price. A track whose model is not here registers its own prices on import (the
#: Petri judge's: ``rejudge.petri.bakeoff.models.register_prices``)
PRICES_PER_MTOK: Dict[str, Tuple[float, float, float, float]] = {
    "minimax/minimax-m3": (0.224, 0.88, 0.0448, 0.0),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def serialize_request(custom_id: str, body: Mapping[str, Any]) -> bytes:
    """One ``{custom_id, body}`` entry as compact ASCII JSON bytes."""
    return json.dumps({"custom_id": custom_id, "body": body}, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def batch_header(model: str, endpoint: str, provider: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    h: Dict[str, Any] = {"endpoint": endpoint, "model": model}
    if provider:
        h["provider"] = dict(provider)
    h["completion_window"] = "24h"
    return h


def serialize_batch(header: Mapping[str, Any], request_lines: Sequence[bytes]) -> bytes:
    """The POST body: the header's fields in order, then ``"requests": [...]`` last."""
    head = json.dumps(dict(header), ensure_ascii=True, separators=(",", ":")).encode("ascii")
    assert head.endswith(b"}")
    return head[:-1] + b',"requests":[' + b",".join(request_lines) + b"]}"


def header_overhead(header: Mapping[str, Any]) -> int:
    return len(serialize_batch(header, []))


def estimate_cost(model: str, input_tokens: float, output_tokens: float, cache_read_tokens: float = 0.0,
                  cache_write_tokens: float = 0.0, with_fee: bool = True) -> float:
    """USD at the ``:batch`` prices, optionally with the 5.5% credit fee. ``input_tokens`` excludes cache reads and writes."""
    p_in, p_out, p_cr, p_cw = PRICES_PER_MTOK[model]
    usd = (input_tokens * p_in + output_tokens * p_out + cache_read_tokens * p_cr + cache_write_tokens * p_cw) / 1e6
    return usd * (1.0 + CREDIT_FEE) if with_fee else usd


@dataclass
class PackedRequest:
    custom_id: str
    line: bytes
    est_input_tokens: float = 0.0
    est_output_tokens: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def chunk(requests: Iterable[PackedRequest], header: Mapping[str, Any], max_bytes: int = DEFAULT_MAX_BYTES,
          max_requests: int = DEFAULT_MAX_REQUESTS) -> Iterator[List[PackedRequest]]:
    """Greedy packing in input order: a batch closes when the next request would push the serialized body past ``max_bytes`` or the
    count past ``max_requests``. A single request larger than ``max_bytes`` on its own is refused (ValueError), never truncated."""
    over = header_overhead(header)
    cur: List[PackedRequest] = []
    size = over
    for r in requests:
        add = len(r.line) + (1 if cur else 0)
        if over + len(r.line) > max_bytes:
            raise ValueError(f"request {r.custom_id} alone is {len(r.line)} bytes, over the {max_bytes}-byte batch cap")
        if cur and (size + add > max_bytes or len(cur) >= max_requests):
            yield cur
            cur, size = [], over
            add = len(r.line)
        cur.append(r)
        size += add
    if cur:
        yield cur


def write_manifest(run_dir: str, batch_key: str, header: Mapping[str, Any], reqs: Sequence[PackedRequest], track: str,
                   extra: Optional[Mapping[str, Any]] = None, attempt: int = 0, parent: Optional[str] = None) -> Dict[str, Any]:
    """Write ``bodies/<batch_key>.json.gz`` and ``manifests/<batch_key>.json``; refuse to overwrite either (a manifest is immutable
    once written). Returns the manifest."""
    body = serialize_batch(header, [r.line for r in reqs])
    mdir, bdir = os.path.join(run_dir, "manifests"), os.path.join(run_dir, "bodies")
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)
    mpath, bpath = os.path.join(mdir, batch_key + ".json"), os.path.join(bdir, batch_key + ".json.gz")
    if os.path.exists(mpath) or os.path.exists(bpath):
        raise FileExistsError(f"manifest or body for batch {batch_key} already exists in {run_dir}")
    tmp = bpath + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(body)
    os.replace(tmp, bpath)
    model = str(header["model"])
    est_in = float(sum(r.est_input_tokens for r in reqs))
    est_out = float(sum(r.est_output_tokens for r in reqs))
    man: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "track": track, "batch_key": batch_key, "model": model, "endpoint": header["endpoint"],
        "provider": header.get("provider"), "n_requests": len(reqs), "body_bytes": len(body), "body_sha256": sha256_bytes(body),
        "body_path": os.path.relpath(bpath, run_dir), "custom_ids": [r.custom_id for r in reqs],
        "est": {"input_tokens": est_in, "output_tokens": est_out,
                "cost_usd": round(estimate_cost(model, est_in, est_out, with_fee=False), 6) if model in PRICES_PER_MTOK else None,
                "cost_usd_with_fee": round(estimate_cost(model, est_in, est_out), 6) if model in PRICES_PER_MTOK else None},
        "attempt": attempt, "parent_batch_key": parent, "created_utc": utc_now(),
    }
    if extra:
        man["extra"] = dict(extra)
    tmpm = mpath + ".tmp"
    with open(tmpm, "w") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmpm, mpath)
    return man


def read_manifest(run_dir: str, batch_key: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "manifests", batch_key + ".json")) as fh:
        return json.load(fh)


def read_body(run_dir: str, manifest: Mapping[str, Any], verify: bool = True) -> bytes:
    """The exact bytes to POST; refuses (ValueError) when their sha256 is not the manifest's."""
    with gzip.open(os.path.join(run_dir, str(manifest["body_path"])), "rb") as fh:
        body = fh.read()
    if verify and sha256_bytes(body) != manifest["body_sha256"]:
        raise ValueError(f"body of {manifest['batch_key']} does not hash to its manifest's body_sha256")
    return body


def body_requests(body: bytes) -> List[Dict[str, Any]]:
    """The ``requests`` array of a batch body."""
    return json.loads(body.decode("ascii"))["requests"]


def write_track(run_dir: str, spec: Mapping[str, Any]) -> None:
    """``track.json``: ``{schema, track, model, endpoint, provider, parser ("module.path:function"), retry {outcome: max attempts},
    max_inflight_batches, max_bytes, max_requests, submit_spacing_s, ...}``. Refuses to change an existing file's model / endpoint /
    parser (a run directory is one track)."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "track.json")
    spec = dict(spec, schema=TRACK_SCHEMA)
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
        for k in ("track", "model", "endpoint", "parser"):
            if old.get(k) != spec.get(k):
                raise ValueError(f"track.json in {run_dir} names {k}={old.get(k)!r}; refusing to change it to {spec.get(k)!r}")
    with open(path + ".tmp", "w") as fh:
        json.dump(spec, fh, indent=1)
    os.replace(path + ".tmp", path)


def read_track(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "track.json")) as fh:
        return json.load(fh)


def list_manifests(run_dir: str) -> List[str]:
    mdir = os.path.join(run_dir, "manifests")
    return sorted(f[:-5] for f in os.listdir(mdir) if f.endswith(".json")) if os.path.isdir(mdir) else []
