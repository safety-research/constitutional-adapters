#!/usr/bin/env python
"""anchor_sampler_preflight.py — is a standup's EFFECTIVE sampler the reused anchor's?

    anchor_sampler_preflight.py <leg> <view> <standup-dir> --anchor <record> [--anchor-kind auto|resolved_record|vllm_log|
        resolved_json|client_record] [--anchor-server-policy engine_defaults|contract_override:<model id>]
        [--request-id <id>] [--client-cap N ...] [--probe-cap N ...] [--record <resolved_sampler.json>] [--out-root DIR]

VERDICTS. ``AGREES`` (exit 0): the standup's engine-resolved client tuple equals the anchor's engine-resolved tuple over
:data:`SAMPLER_KEYS` (temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty; ``top_k`` in {None, -1, 0} =
disabled) — the leg's rows are ``comparable``. ``DIVERGED`` (exit 1): both tuples were read and differ — the leg is BLOCKED
and, were it run, every paired row would carry ``anchor_sampler_verdict: non_comparable(<diff>)``. ``UNKNOWN`` (exit 2): a
side could not be read (no anchor record, a client-only anchor record with no way to resolve it, no logged request, a
mixed client class) — a refusal, NEVER "assumed equal". The record lands at
``<out-root>/<leg>_<view>.json`` (default ``RS/reseed/anchor_preflight/``, ``RS`` = the wave lanes' run-state root).

WHAT IS COMPARED — and what is NOT. Both sides are the sampler the ENGINE RESOLVED after merging its defaults over what
the client omitted (``record_resolved_sampler.py``'s reading of vLLM's ``Received request ...: params: SamplingParams(...)``
lines, emitted under ``--enable-log-requests``). The keys a client SENT are never the verdict's basis: two clients that
both omit every sampling key decode identically only if their two servers resolve an omitted key identically, and the
Qwen server-side vendor pin (``contract_override``) made a 2026-09 standup resolve 0.7 / 0.8 / 20 where every
historical Qwen Petri / BoN / misalignment anchor resolved vLLM's neutral 1.0 / 1.0 / off. ``anchor_pairing.sampler_check(
None, None, not_sent=True)`` — "neither side sends a key, the server's generation config governs on both" — is therefore
NOT a source of this verdict: "not sent" describes the clients, and the question is about the engines.

THE STANDUP SIDE (the EFFECTIVE sampler). In order: (1) ``--request-id <id>`` — the ONE request the caller fired (the
proxies' health-class probe shaped like the run's requests; the response's ``id`` is the id vLLM logs), read from
``<standup-dir>/vllm_server.log`` (or ``--vllm-log``); the load check's greedy requests at the same cap never enter.
(2) ``--record`` or ``<standup-dir>/resolved_sampler.json`` — a ``record_resolved_sampler.py`` record: its client class's
single distinct tuple. (3) the log's CLIENT class (``--client-cap`` = the run's cap; probe caps default to the served
target's 48) — exactly one distinct tuple, else UNKNOWN (a mixed class is not a sampler). The standup's
``server_default_policy`` / ``override_generation_config`` are read off ``endpoint.json`` (or its ``.recorded`` /
``.snapshot`` copies) and recorded beside the verdict — provenance, not evidence.

THE ANCHOR SIDE (``--anchor``, kind auto-detected unless ``--anchor-kind``): a ``record_resolved_sampler.py`` record
(``verdict`` + ``distinct_resolved``: one distinct client tuple); a vLLM server log (the anchor standup's; ``--anchor-request-id``
/ ``--anchor-client-cap`` scope it); a JSON object carrying ``resolved_sampler`` (an explicit engine-resolved tuple someone
recorded from the anchor's log, with its ``source``); or a CLIENT record — ``attacker_engine.json`` (``target_decode``),
``bon_served_run.json`` (``sampler`` / ``decode``), the misalign runner's ``meta.json`` (``params``) — which states what the
anchor SENT, and resolves only under an explicit ``--anchor-server-policy``: ``engine_defaults`` fills every omitted key
with vLLM's neutral default (the historical Qwen standups: no generation_config.json, no override), ``contract_override:<model>``
fills the contracts' pinned row (``SERVER_DEFAULT_SAMPLING_BY_MODEL``) then the neutral defaults. Without it a client record
alone is UNKNOWN: the anchor's engine-resolved tuple is not readable from what its client sent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (REPO / "src", REPO / "scripts" / "sdfbase"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import record_resolved_sampler as rrs  # noqa: E402  (scripts/sdfbase; torch-free)

SCHEMA = "anchor_sampler_preflight/1"
SAMPLER_KEYS: Tuple[str, ...] = tuple(rrs.SAMPLER_KEYS)
VLLM_NEUTRAL_DEFAULTS: Dict[str, Any] = dict(rrs.VLLM_NEUTRAL_DEFAULTS)
VERDICTS = ("AGREES", "DIVERGED", "UNKNOWN")
EXIT = {"AGREES": 0, "DIVERGED": 1, "UNKNOWN": 2}
ANCHOR_KINDS = ("auto", "resolved_record", "vllm_log", "resolved_json", "client_record")
#: the reseed directory under the run-state root every wave lane shares ($RESEED_RS, else a built-in path)
#: (the built-in path sits under the site's state root, $STATE, expanded through sisteer.site where it is written)
RS_DEFAULT = Path(os.environ.get("RESEED_RS", "$STATE/romp/candidate-si-refit/run-state/full22")) / "reseed"
OUT_ROOT_DEFAULT = RS_DEFAULT / "anchor_preflight"
#: the keys a CLIENT record states its sent sampler under, in the order tried
CLIENT_RECORD_KEYS: Tuple[str, ...] = ("target_decode", "sampler", "decode", "params", "sampling", "sent")


class PreflightError(Exception):
    """A side that cannot be read — the UNKNOWN verdict's reason."""


#: ``top_k`` OFF as one comparable value: vLLM logs ``top_k=0`` for "off", clients send ``-1`` for it — both read as this
TOP_K_OFF = -1.0


def normalise(rec: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The comparable projection over SAMPLER_KEYS: ``top_k`` in {-1, 0} -> :data:`TOP_K_OFF` (disabled); numbers as
    floats; an ABSENT key stays None — and None is never "equal" to None: :func:`verdict_of` reads it as UNKNOWN
    (resolution is the readers' job; a key a reader could not state is not evidence)."""
    if rec is None:
        return None
    out: Dict[str, Any] = {}
    for k in SAMPLER_KEYS:
        v = rec.get(k) if isinstance(rec, Mapping) else None
        if k == "top_k" and v in (-1, 0):
            v = TOP_K_OFF
        elif isinstance(v, bool):
            v = float(v)
        elif isinstance(v, (int, float)):
            v = float(v)
        out[k] = v
    return out


def diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{"key": k, "effective": a.get(k), "anchor": b.get(k)} for k in SAMPLER_KEYS if a.get(k) != b.get(k)]


def _resolve_sent(sent: Mapping[str, Any], *, policy: str) -> Dict[str, Any]:
    """A CLIENT tuple completed the way the anchor's ENGINE completed it under ``policy``."""
    out = {k: sent[k] for k in SAMPLER_KEYS if k in sent and sent[k] is not None}
    if policy == "engine_defaults":
        fill = dict(VLLM_NEUTRAL_DEFAULTS)
    elif policy.startswith("contract_override:"):
        model = policy.split(":", 1)[1]
        from sisteer.capeval.tinycap import contracts as C  # lazy
        row = C.SERVER_DEFAULT_SAMPLING_BY_MODEL.get(model)
        if row is None:
            raise PreflightError(f"--anchor-server-policy {policy}: contracts pin no server-default row for {model!r} "
                                 f"(pinned: {sorted(C.SERVER_DEFAULT_SAMPLING_BY_MODEL)})")
        fill = dict(VLLM_NEUTRAL_DEFAULTS)
        for k in C.SERVER_DEFAULTABLE_SAMPLING_KEYS:
            v = getattr(row, k, None)
            if v is not None:
                fill[k] = v
    else:
        raise PreflightError(f"--anchor-server-policy must be engine_defaults or contract_override:<model id>, got {policy!r}")
    for k in SAMPLER_KEYS:
        out.setdefault(k, fill[k])
    return out


# ----------------------------------------------------------------------------- the standup side

def _log_requests(log: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if not log.is_file():
        raise PreflightError(f"no server log at {log} — the standup must run under --enable-log-requests (F22_LOG_REQUESTS=1)")
    rec = rrs.read_log(log)
    if rec["log_requests_muted"]:
        raise PreflightError(f"{log}: --enable-log-requests is set but muted (log level above INFO) — no request is logged")
    return list(rec["resolved"])


def _one_distinct(resolved: Sequence[Tuple[str, Mapping[str, Any]]], *, what: str) -> Dict[str, Any]:
    tuples = {json.dumps({k: p.get(k) for k in SAMPLER_KEYS}, sort_keys=True) for _rid, p in resolved}
    if not tuples:
        raise PreflightError(f"{what}: no logged request to read")
    if len(tuples) > 1:
        raise PreflightError(f"{what}: {len(tuples)} distinct resolved tuples — a mixed class is not one sampler: "
                             + "; ".join(sorted(tuples)))
    return json.loads(next(iter(tuples)))


def effective_sampler(standup: Path, *, request_id: Optional[str], record: Optional[Path], vllm_log: Optional[Path],
                      client_caps: Sequence[int], probe_caps: Optional[Sequence[int]]) -> Dict[str, Any]:
    """``{sampler, source, n_requests}`` — see the module docstring's three readings."""
    log = vllm_log or (standup / "vllm_server.log")
    if request_id:
        resolved = [(rid, p) for rid, p in _log_requests(log) if rid == request_id]
        if not resolved:
            raise PreflightError(f"{log}: no `Received request {request_id}` line — the probe's request was not logged "
                                 "(is --enable-log-requests on? did the request reach the engine?)")
        return {"sampler": {k: resolved[0][1].get(k) for k in SAMPLER_KEYS}, "source": f"vllm log, request {request_id}",
                "n_requests": len(resolved), "max_tokens": resolved[0][1].get("max_tokens")}
    rec_path = record or (standup / "resolved_sampler.json")
    if rec_path.is_file():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        return {**_from_resolved_record(rec, what=str(rec_path)), "source": f"record_resolved_sampler record {rec_path}"}
    resolved = _log_requests(log)
    pcaps = set(str(c) for c in (probe_caps if probe_caps is not None else [rrs.default_probe_cap()]))
    classes = rrs.partition(resolved, set(str(c) for c in client_caps), pcaps)
    client = classes["client"]
    return {"sampler": _one_distinct(client, what=f"{log} client class (caps {sorted(client_caps) or 'every non-probe'})"),
            "source": f"vllm log client class ({log})", "n_requests": len(client)}


def _from_resolved_record(rec: Mapping[str, Any], *, what: str) -> Dict[str, Any]:
    if not isinstance(rec, Mapping) or "distinct_resolved" not in rec or "verdict" not in rec:
        raise PreflightError(f"{what}: not a record_resolved_sampler record (needs verdict + distinct_resolved)")
    blocks = list(rec.get("distinct_resolved") or [])
    if len(blocks) != 1:
        raise PreflightError(f"{what}: the client class holds {len(blocks)} distinct resolved tuples (need exactly 1)")
    s = dict(blocks[0].get("sampler") or {})
    return {"sampler": {k: s.get(k) for k in SAMPLER_KEYS}, "n_requests": int(blocks[0].get("n") or 0),
            "record_verdict": rec.get("verdict")}


def standup_provenance(standup: Path) -> Dict[str, Any]:
    """``server_default_policy`` / ``override_generation_config`` off the standup's endpoint record (any of its copies)."""
    for name in ("endpoint.json", "endpoint.recorded.json", "endpoint.snapshot.json"):
        p = standup / name
        if p.is_file():
            try:
                ep = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            return {"endpoint": str(p), "server_default_policy": ep.get("server_default_policy"),
                    "override_generation_config": ep.get("override_generation_config"), "engine": ep.get("engine"),
                    "model": ep.get("model"), "model_revision": ep.get("model_revision")}
    snaps = sorted(standup.parent.glob("endpoint.snapshot*.json")) if standup.parent.is_dir() else []
    for p in snaps:
        try:
            ep = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        return {"endpoint": str(p), "server_default_policy": ep.get("server_default_policy"),
                "override_generation_config": ep.get("override_generation_config"), "engine": ep.get("engine"),
                "model": ep.get("model"), "model_revision": ep.get("model_revision")}
    return {"endpoint": None, "server_default_policy": None, "override_generation_config": None}


# ----------------------------------------------------------------------------- the anchor side

def _detect_kind(path: Path) -> Tuple[str, Any]:
    if not path.exists():
        raise PreflightError(f"anchor record {path} does not exist — a missing anchor is a refusal, never 'assumed equal'")
    text = path.read_text(encoding="utf-8", errors="replace")
    if rrs._REQ.search(text) or "Received request" in text:
        return "vllm_log", None
    try:
        doc = json.loads(text)
    except ValueError:
        raise PreflightError(f"anchor record {path} is neither a vLLM server log nor JSON")
    if isinstance(doc, Mapping) and "distinct_resolved" in doc and "verdict" in doc:
        return "resolved_record", doc
    if isinstance(doc, Mapping) and isinstance(doc.get("resolved_sampler"), Mapping):
        return "resolved_json", doc
    if isinstance(doc, Mapping) and any(isinstance(doc.get(k), Mapping) for k in CLIENT_RECORD_KEYS):
        return "client_record", doc
    raise PreflightError(f"anchor record {path}: unrecognised shape (keys {sorted(doc)[:12] if isinstance(doc, Mapping) else type(doc).__name__})")


def anchor_sampler(path: Path, *, kind: str, server_policy: Optional[str], request_id: Optional[str],
                   client_caps: Sequence[int], probe_caps: Optional[Sequence[int]]) -> Dict[str, Any]:
    """``{sampler, source, kind, ...}`` — the anchor's ENGINE-RESOLVED tuple, or a PreflightError naming why not."""
    detected, doc = (_detect_kind(path) if kind == "auto" else (kind, None))
    if doc is None and detected != "vllm_log":
        if not path.exists():
            raise PreflightError(f"anchor record {path} does not exist — a missing anchor is a refusal, never 'assumed equal'")
        doc = json.loads(path.read_text(encoding="utf-8"))
    if detected == "resolved_record":
        r = _from_resolved_record(doc, what=str(path))
        return {**r, "kind": detected, "path": str(path), "source": f"record_resolved_sampler record {path}"}
    if detected == "vllm_log":
        resolved = _log_requests(path)
        if request_id:
            resolved = [(rid, p) for rid, p in resolved if rid == request_id]
            if not resolved:
                raise PreflightError(f"{path}: no `Received request {request_id}` line")
            return {"sampler": {k: resolved[0][1].get(k) for k in SAMPLER_KEYS}, "kind": detected, "path": str(path),
                    "source": f"anchor vllm log, request {request_id}", "n_requests": len(resolved)}
        pcaps = set(str(c) for c in (probe_caps if probe_caps is not None else [rrs.default_probe_cap()]))
        client = rrs.partition(resolved, set(str(c) for c in client_caps), pcaps)["client"]
        return {"sampler": _one_distinct(client, what=f"{path} client class"), "kind": detected, "path": str(path),
                "source": f"anchor vllm log client class ({path})", "n_requests": len(client)}
    if detected == "resolved_json":
        s = dict(doc["resolved_sampler"])
        missing = [k for k in SAMPLER_KEYS if k not in s]
        if missing:
            raise PreflightError(f"{path}: resolved_sampler lacks {missing} — an engine-resolved tuple states every key")
        return {"sampler": {k: s.get(k) for k in SAMPLER_KEYS}, "kind": detected, "path": str(path),
                "source": f"explicit resolved_sampler in {path} ({doc.get('source', 'no source stated')})"}
    if detected == "client_record":
        sent_key = next((k for k in CLIENT_RECORD_KEYS if isinstance(doc.get(k), Mapping)), None)
        if sent_key is None:
            raise PreflightError(f"{path}: no client sampler block under {CLIENT_RECORD_KEYS}")
        sent = dict(doc[sent_key])
        if server_policy is None:
            raise PreflightError(f"{path} is a CLIENT record ({sent_key}: {sent}) — it states what the anchor SENT, not what "
                                 "its engine resolved; pass its server log / a record_resolved_sampler record, or state the "
                                 "anchor server's policy with --anchor-server-policy engine_defaults | contract_override:<model>")
        return {"sampler": _resolve_sent(sent, policy=server_policy), "kind": detected, "path": str(path),
                "source": f"client record {path} [{sent_key}] resolved under {server_policy}", "sent": sent,
                "anchor_server_policy": server_policy}
    raise PreflightError(f"--anchor-kind {kind!r} is not one of {ANCHOR_KINDS}")


# ----------------------------------------------------------------------------- the verdict

def verdict_of(effective: Optional[Mapping[str, Any]], anchor: Optional[Mapping[str, Any]], *, error: Optional[str]) -> Dict[str, Any]:
    if error is not None or effective is None or anchor is None:
        return {"verdict": "UNKNOWN", "row_marker": f"non_comparable(unknown: {error or 'a side is unreadable'})", "diff": None}
    e, a = normalise(effective), normalise(anchor)
    unstated = sorted({k for k in SAMPLER_KEYS if e.get(k) is None} | {k for k in SAMPLER_KEYS if a.get(k) is None})
    if unstated:
        # a key a side does not STATE is not evidence of anything — least of all of agreement
        return {"verdict": "UNKNOWN", "diff": None, "effective_normalised": e, "anchor_normalised": a,
                "row_marker": f"non_comparable(unknown: sampler key(s) not stated on a side: {', '.join(unstated)})"}
    d = diff(e, a)
    if d:
        marker = "non_comparable(" + ", ".join(f"{x['key']}: {x['effective']} vs anchor {x['anchor']}" for x in d) + ")"
        return {"verdict": "DIVERGED", "row_marker": marker, "diff": d, "effective_normalised": e, "anchor_normalised": a}
    return {"verdict": "AGREES", "row_marker": "comparable", "diff": [], "effective_normalised": e, "anchor_normalised": a}


def run(leg: str, view: str, standup: Path, *, anchor: Path, anchor_kind: str = "auto", anchor_server_policy: Optional[str] = None,
        request_id: Optional[str] = None, anchor_request_id: Optional[str] = None, record: Optional[Path] = None,
        vllm_log: Optional[Path] = None, client_caps: Sequence[int] = (), anchor_client_caps: Sequence[int] = (),
        probe_caps: Optional[Sequence[int]] = None, out_root: Path = OUT_ROOT_DEFAULT) -> Dict[str, Any]:
    eff: Optional[Dict[str, Any]] = None
    anc: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    try:
        eff = effective_sampler(standup, request_id=request_id, record=record, vllm_log=vllm_log,
                                client_caps=client_caps, probe_caps=probe_caps)
    except PreflightError as e:
        errors.append(f"standup: {e}")
    try:
        anc = anchor_sampler(anchor, kind=anchor_kind, server_policy=anchor_server_policy, request_id=anchor_request_id,
                             client_caps=anchor_client_caps, probe_caps=probe_caps)
    except PreflightError as e:
        errors.append(f"anchor: {e}")
    v = verdict_of(eff["sampler"] if eff else None, anc["sampler"] if anc else None,
                   error=("; ".join(errors) if errors else None))
    rec = {"schema": SCHEMA, "leg": leg, "view": view, **v,
           "effective": ({**eff, "standup_dir": str(standup), **standup_provenance(standup)} if eff
                         else {"standup_dir": str(standup), **standup_provenance(standup)}),
           "anchor": (anc if anc else {"path": str(anchor), "kind": anchor_kind}),
           "errors": errors, "sampler_keys": list(SAMPLER_KEYS), "vllm_neutral_defaults": VLLM_NEUTRAL_DEFAULTS,
           "basis": "engine-resolved tuples on both sides (record_resolved_sampler); the keys a client SENT are never the basis — "
                    "sampler_check(not_sent=True) is not a verdict source here",
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out_root.mkdir(parents=True, exist_ok=True)
    safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]+", "_", s)  # noqa: E731
    out = out_root / f"{safe(leg)}_{safe(view)}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    tmp.rename(out)
    rec["record_path"] = str(out)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("leg"); ap.add_argument("view"); ap.add_argument("standup_dir", type=Path)
    ap.add_argument("--anchor", type=Path, required=True, help="the reused anchor's record (see the module docstring for the shapes)")
    ap.add_argument("--anchor-kind", choices=ANCHOR_KINDS, default="auto")
    ap.add_argument("--anchor-server-policy", default=None, help="for a CLIENT anchor record: engine_defaults | contract_override:<model id>")
    ap.add_argument("--anchor-request-id", default=None)
    ap.add_argument("--anchor-client-cap", type=int, action="append", default=[])
    ap.add_argument("--request-id", default=None, help="the probe request's id (the response's `id`): the ONE logged request compared")
    ap.add_argument("--record", type=Path, default=None, help="a record_resolved_sampler.py record for the standup (default <standup-dir>/resolved_sampler.json)")
    ap.add_argument("--vllm-log", type=Path, default=None, help="the standup's server log (default <standup-dir>/vllm_server.log)")
    ap.add_argument("--client-cap", type=int, action="append", default=[])
    ap.add_argument("--probe-cap", type=int, action="append", default=None)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    a = ap.parse_args(argv)
    from sisteer import site as _site
    a.out_root = _site.expand_path(a.out_root, key="--out-root")   # an unset $STATE refuses by name
    rec = run(a.leg, a.view, a.standup_dir, anchor=a.anchor, anchor_kind=a.anchor_kind, anchor_server_policy=a.anchor_server_policy,
              request_id=a.request_id, anchor_request_id=a.anchor_request_id, record=a.record, vllm_log=a.vllm_log,
              client_caps=a.client_cap, anchor_client_caps=a.anchor_client_cap, probe_caps=a.probe_cap, out_root=a.out_root)
    eff = rec["effective"].get("sampler"); anc = rec["anchor"].get("sampler")
    print(f"[anchor-preflight] {rec['verdict']} {a.leg}/{a.view}: {rec['row_marker']}")
    print(f"[anchor-preflight]   effective: {eff} ({rec['effective'].get('source')}; server_default_policy={rec['effective'].get('server_default_policy')})")
    print(f"[anchor-preflight]   anchor:    {anc} ({rec['anchor'].get('source')})")
    for e in rec["errors"]:
        print(f"[anchor-preflight]   UNREADABLE: {e}")
    print(f"[anchor-preflight] -> {rec['record_path']}")
    return EXIT[rec["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
