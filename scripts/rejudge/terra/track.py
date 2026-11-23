"""The C/P/I track's run directories: ``track.json`` and the manifests built from distinct judge inputs, one run directory per FAMILY
(the harvester reads one parser per run directory, and the two families' verdicts differ):

    cpi      OR-Bench hard + XSTest (the C/P/I tool call; parser ``rejudge.terra.cpi:parse_result``)
    agentic  the agentic-misalignment classifier, murder + leaking (strict json_schema; parser ``rejudge.terra.agentic:parse_result``)

    from rejudge.terra.track import init_run, add_inputs
    init_run(run_dir, "cpi")                       # track.json (model, endpoint, parser, retry policy, limits, request settings)
    add_inputs(run_dir, texts_rows, prefix="b")    # rows of the scope's texts.jsonl.gz

``add_inputs`` builds one request per distinct key (CPI: rebuilt from the Sonnet judge's own params, the scope's key recomputed and
checked, so a text that no longer recomputes refuses), estimates its tokens (o200k via tiktoken, else chars / 4) and packs the requests
into batches under the byte and request caps (``batchfile.chunk``), one manifest per batch.

THE JUDGE MODEL is a config entry (:mod:`rejudge.terra.judges`): ``init_run(run_dir, family, judge="minimax-m3")`` writes that model,
its parser, its request settings (``request.judge`` names the entry) and its retry cap for a cut call (``truncated_retry_max_tokens``);
``judge=None`` is ``judges.DEFAULT_JUDGE``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.terra import judges as J  # noqa: E402

FAMILIES = ("cpi", "agentic")
#: re-asks, counted within the outcome's class (rejudge.terra.harvest): a transient provider failure up to 3 times, a malformed or cut
#: verdict once (the Sonnet judges re-asked a malformed verdict once, judging.ClaudeJudge's re-ask, and climbed a cut agentic verdict to a
#: larger cap), a moderation rejection or a judge refusal never (recorded per item), a whole-batch ending 3 times on its own count; an
#: ACCOUNT failure (account_error per request, batch_failed_auth per batch) and a held result are never terminal, never counted, and pause
#: the run (the entries below for them only keep the shared harvester's table complete)
RETRY = {"verdict": 0, "moderation": 0, "judge_refusal": 0, "parse_error": 1, "truncated": 1, "provider_error": 3, "missing": 3,
         "batch_failed": 3, "batch_expired": 3, "batch_cancelled": 3, "batch_failed_auth": 1000, "account_error": 1000, "held": 1000}
#: the moderation-share hold (rejudge.terra.harvest): a batch whose results are at least this share moderation (and at least min_count)
#: is held and the run pauses until a person acknowledges it
HEALTH = {"moderation_share": 0.05, "min_count": 10, "acknowledged": []}


def est_output_tokens(family: str, rq: Optional[Mapping[str, Any]] = None) -> float:
    return float(J.family_cfg(J.normalize(rq)["judge"] if rq else None, family)["est_output_tokens"])


def est_overhead_tokens(family: str, rq: Optional[Mapping[str, Any]] = None) -> float:
    return float(J.family_cfg(J.normalize(rq)["judge"] if rq else None, family)["est_overhead_tokens"])

_ENC = None


def tokens(text: str) -> float:
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


def module(family: str):
    if family == "cpi":
        from rejudge.terra import cpi as M
    elif family == "agentic":
        from rejudge.terra import agentic as M
    else:
        raise ValueError(f"family {family!r}; one of {FAMILIES}")
    return M


def init_run(run_dir: str, family: str, max_tokens: Optional[int] = None, provider: Optional[Mapping[str, Any]] = None,
             extra: Optional[Mapping[str, Any]] = None, judge: Optional[str] = None) -> Dict[str, Any]:
    """``track.json`` of a new run directory: the settings of the :data:`rejudge.terra.judges.JUDGES` entry ``judge`` (default
    ``judges.DEFAULT_JUDGE``); ``max_tokens`` / ``provider`` override the entry's."""
    M = module(family)
    judge = judge or J.DEFAULT_JUDGE
    cfg = J.get(judge)
    fc = J.family_cfg(judge, family)
    spec = {"track": J.track_name(judge, family), "family": family, "judge": judge, "model": cfg["openrouter_model"],
            "batch_model_variant": cfg["batch_variant"], "endpoint": M.ENDPOINT, "provider": provider if provider is not None else cfg.get("provider"),
            "parser": J.parser_spec(judge, family), "retry": RETRY, "max_bytes": BF.DEFAULT_MAX_BYTES,
            "max_requests": int(fc.get("max_requests") or cfg.get("max_requests") or BF.DEFAULT_MAX_REQUESTS), "submit_spacing_s": 2.0,
            "max_inflight_batches": 0,
            "health": dict(HEALTH), "request": J.request_settings(judge, family, max_tokens)}
    if fc.get("truncated_retry_max_tokens"):
        spec["truncated_retry_max_tokens"] = int(fc["truncated_retry_max_tokens"])
    if extra:
        spec.update(extra)
    os.makedirs(run_dir, exist_ok=True)
    os.chmod(run_dir, 0o700)
    BF.write_track(run_dir, spec)
    return spec


def entry_for(family: str, x: Mapping[str, Any], rq: Mapping[str, Any]) -> Dict[str, Any]:
    """``{custom_id, body}`` for one row of the scope's ``texts.jsonl.gz`` under a ``track.json`` request block
    (:func:`rejudge.terra.judges.build_entry`)."""
    return J.build_entry(family, x, rq)


def message_text(body: Mapping[str, Any]) -> str:
    return "\n".join(str(m.get("content") or "") for m in body["messages"])


def add_inputs(run_dir: str, inputs: Iterable[Mapping[str, Any]], prefix: str = "b", max_bytes: Optional[int] = None,
               max_requests: Optional[int] = None) -> List[Dict[str, Any]]:
    spec = BF.read_track(run_dir)
    family = spec["family"]
    rq = spec["request"]
    header = BF.batch_header(spec["model"], spec["endpoint"], spec.get("provider"))

    def packed():
        seen = set()
        for x in inputs:
            if x["k"] in seen:
                continue
            seen.add(x["k"])
            e = entry_for(family, x, rq)
            yield BF.PackedRequest(e["custom_id"], BF.serialize_request(e["custom_id"], e["body"]),
                                   est_input_tokens=tokens(message_text(e["body"])) + est_overhead_tokens(family, rq),
                                   est_output_tokens=est_output_tokens(family, rq))

    mans = []
    existing = len(BF.list_manifests(run_dir))
    for i, part in enumerate(BF.chunk(packed(), header, int(max_bytes or spec["max_bytes"]), int(max_requests or spec["max_requests"]))):
        key = f"{prefix}{existing + i:05d}"
        mans.append(BF.write_manifest(run_dir, key, header, part, spec["track"]))
    return mans
