#!/usr/bin/env python3
"""Read every in-scope Petri transcript's ORIGINAL judge input from its stored inspect sample: the exact prompt and tool the Opus 5
judge was sent, keyed for dedup, with the Claude verdict and the token counts for the estimate.

    PYTHONPATH=src:scripts python scripts/rejudge/petri/enumerate_inputs.py --units SCOPE/units.tsv.gz --out ENUM \
        [--verify-rebuild] [--procs 10] [--limit N]

Per distinct physical transcript (``phys`` in ``units.tsv.gz``): the sample is read from its ``.eval`` log (attachments resolved); the
judge's input is the stored API request of its FIRST judge model event (``ModelEvent.call.request``: ``messages[0].content`` and
``tools[0]``, :func:`rejudge.petri.request.request_parts`) -- the bytes the Opus 5 judge received. Every later judge event of the
sample (a refusal re-ask, a schema re-ask, the Opus 4.8 / Grok 4.6 refusal fallback) is checked to carry the SAME first user message
and tool. ``--verify-rebuild`` also regenerates the prompt and the answer schema with the ORIGINAL builders -- inspect_petri's
``audit_judge`` scanner over ``inspect_scout.transcript_from_eval_sample(sample)``, the behaviour loaded by petri_bloom from the
audit's own behaviour directory (the eval header's ``task_args.behavior``), its model call captured before any network -- and records
whether they equal the stored request (``rebuild``: ``match`` / ``mismatch:<where>`` / ``no_stored_timeline`` (a log that stores no
timeline, where an offline rebuild cannot know which timeline the live scorer read; 30 of 4,918 on 2026-11-09, 28 of them
native-path qwen35_4b seed audits -- their stored request is what the judge received) / ``error:<why>``). On 2026-11-09 the other 4,888 matched byte for byte.

Outputs (``ENUM``):
  inputs.jsonl.gz   one row per transcript: phys, key (``request.judge_input_key``), prefix_sha, tool_sha, behaviour, own dimension,
                    the prompt, tokens (o200k: head, rest, tool JSON), the judge events' served models, the Claude verdict stored in
                    the sample (the three scores, refusal), rebuild
  tools.json        tool_sha -> the tool (name, description, input_schema), as stored
  report.json       counts: transcripts, distinct keys, distinct prefixes, rebuild outcomes, errors
The outputs hold transcript text: keep ENUM private and out of the repository.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.petri import request as R  # noqa: E402

_ENC = None
_BEHAVIORS: Dict[str, Any] = {}


def tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001
            _ENC = False
    return len(_ENC.encode(text, disallowed_special=())) if _ENC else int(len(text) / 4.47)


def read_units(path: str) -> Dict[str, Dict[str, Any]]:
    """phys -> {path, scenario, epoch, behavior, n_units}"""
    out: Dict[str, Dict[str, Any]] = {}
    with gzip.open(path, "rt") as fh:
        cols = next(fh).rstrip("\n").split("\t")
        for line in fh:
            d = dict(zip(cols, line.rstrip("\n").split("\t")))
            p = out.setdefault(d["phys"], {"path": d["path"], "scenario": d["scenario"], "epoch": int(d["epoch"]), "behavior": d["behavior"], "n_units": 0})
            p["n_units"] += 1
    return out


def judge_events(sample) -> List[Any]:
    return [e for e in sample.events if getattr(e, "event", None) == "model" and getattr(e, "role", None) == "judge"]


def stored_request(evs: List[Any]) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, int]]:
    """The first judge event's request that parses as the judge input; the PROBLEMS found checking the others against it (a different
    prompt, or a different tool from another Anthropic event); and NOTES that are not problems: ``errored_attempt`` (an attempt that
    failed in transport and stored no request -- inspect retried it, max_retries 14) and ``fallback_tool_rendering`` (the Grok 4.6
    refusal fallback, sent through an OpenAI-compatible provider that drops the integer bounds from the schema; its prompt is checked)."""
    first = None
    problems: List[str] = []
    notes: Dict[str, int] = collections.Counter()
    for i, e in enumerate(evs):
        req = (e.call.request if e.call else None) or {}
        msgs = req.get("messages") or []
        if not msgs:
            notes["errored_attempt"] += 1
            continue
        one = dict(req, messages=msgs[:1])               # a re-ask appends the bad call and its error after the judge input
        try:
            prompt, tool = R.request_parts(one)
        except ValueError as exc:
            problems.append(f"event {i} ({e.model}): {exc}")
            continue
        if first is None:
            first = {"prompt": prompt, "tool": tool, "event_index": i, "model": e.model, "request_keys": sorted(req.keys())}
            continue
        if prompt != first["prompt"]:
            problems.append(f"event {i} ({e.model}) carries a different prompt than event {first['event_index']}")
        elif R.tool_sha(tool) != R.tool_sha(first["tool"]):
            if str(e.model).startswith("anthropic/"):
                problems.append(f"event {i} ({e.model}) carries a different tool than event {first['event_index']}")
            else:
                notes["fallback_tool_rendering"] += 1
    return first, problems, dict(notes)


def rebuild(path: str, sample, header) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Regenerate (prompt, answer schema) with the original builders; ``(status, rebuilt)``."""
    from pathlib import Path

    import inspect_petri._judge.judge as J
    from inspect_scout._llm_scanner.structured import structured_schema
    from inspect_scout._transcript.eval_log import transcript_from_eval_sample
    from petri_bloom._behavior.behavior import behavior as load_behavior
    from petri_bloom._evaluation.evaluation import _build_judge_instructions

    if not sample.timelines:
        # a log that stores no timeline: transcript_from_eval_sample builds one from the events, whose first timeline is not
        # necessarily the target's (the live scorer had the sample's own timelines) -- a mismatch here is the offline rebuild's
        return "no_stored_timeline", None
    bdir = (header.eval.task_args or {}).get("behavior")
    if not bdir:
        return "error:no task_args.behavior in the eval header", None
    if bdir not in _BEHAVIORS:
        try:
            _BEHAVIORS[bdir] = load_behavior(path=Path(bdir))
        except Exception as exc:  # noqa: BLE001
            _BEHAVIORS[bdir] = exc
    bhv = _BEHAVIORS[bdir]
    if isinstance(bhv, Exception):
        return f"error:behaviour {bdir}: {type(bhv).__name__}: {str(bhv)[:200]}", None
    captured: Dict[str, Any] = {}

    class _Stop(Exception):
        pass

    async def _capture(prompt, model=None, config=None, answer=None, extract_refs=None):
        captured["prompt"], captured["answer"] = prompt, answer
        raise _Stop()

    saved = (J.generate_answer, J._resolve_judge_model)
    J.generate_answer, J._resolve_judge_model = _capture, (lambda model=None: None)
    try:
        scan = J.audit_judge(dimensions=bhv.dimensions, instructions=_build_judge_instructions(bhv))
        t = transcript_from_eval_sample(sample, eval_id=header.eval.eval_id, log_location=path, model=None)
        try:
            asyncio.run(scan(t))
        except _Stop:
            pass
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}: {str(exc)[:200]}", None
    finally:
        J.generate_answer, J._resolve_judge_model = saved
    if "prompt" not in captured:
        return "error:the scanner made no model call", None
    schema = structured_schema(captured["answer"]).model_dump(exclude_none=True)
    return "ok", {"prompt": captured["prompt"], "schema": schema}


def one(job: Tuple[str, Dict[str, Any], bool]) -> Dict[str, Any]:
    phys, u, verify = job
    from inspect_ai.log import read_eval_log, read_eval_log_sample
    out: Dict[str, Any] = {"phys": phys, "path": u["path"], "scenario": u["scenario"], "epoch": u["epoch"], "behavior": u["behavior"]}
    try:
        s = read_eval_log_sample(u["path"], id=u["scenario"], epoch=u["epoch"], resolve_attachments=True)
    except Exception as exc:  # noqa: BLE001
        return dict(out, error=f"read: {type(exc).__name__}: {str(exc)[:300]}")
    evs = judge_events(s)
    first, problems, notes = stored_request(evs)
    out["n_judge_events"] = len(evs)
    out["judge_models"] = [e.model for e in evs]
    out["consistency_problems"] = problems
    out["event_notes"] = notes
    sc = (s.scores or {}).get("audit_judge")
    val = sc.value if sc is not None else None
    md = (sc.metadata or {}) if sc is not None else {}
    out["claude"] = {"scores": val if isinstance(val, dict) else None, "refusal": bool(md.get("refusal")),
                     "served": next((e.model for e in reversed(evs) if e.output and e.output.choices and
                                     (e.output.choices[0].message.tool_calls or [])), None)}
    if first is None:
        return dict(out, error="no judge event carries a parseable judge request")
    prompt, tool = first["prompt"], first["tool"]
    try:
        head, rest = R.split_prompt(prompt)
        own = R.own_dimension(tool)
    except ValueError as exc:
        return dict(out, error=f"input: {exc}")
    out.update(key=R.judge_input_key(prompt, tool), prefix_sha=R.prefix_sha(prompt, tool), tool_sha=R.tool_sha(tool), own_dimension=own,
               prompt=prompt, tool=tool, source_event=first["event_index"], source_model=first["model"],
               tokens={"head": tokens(head), "rest": tokens(rest), "tool": tokens(json.dumps(tool, ensure_ascii=False))},
               chars={"head": len(head), "rest": len(rest)})
    if verify:
        try:
            header = read_eval_log(u["path"], header_only=True)
            st, rb = rebuild(u["path"], s, header)
        except Exception as exc:  # noqa: BLE001
            st, rb = f"error:{type(exc).__name__}: {str(exc)[:200]}", None
        if rb is not None:
            schema = tool["input_schema"]
            if rb["prompt"] != prompt:
                st = "mismatch:prompt"
            elif rb["schema"].get("properties") != schema.get("properties") or rb["schema"].get("required") != schema.get("required"):
                st = "mismatch:schema"
            else:
                st = "match"
        out["rebuild"] = st
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-rebuild", action="store_true")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.chmod(args.out, 0o700)
    units = read_units(args.units)
    jobs = [(p, u, args.verify_rebuild) for p, u in sorted(units.items())]
    if args.limit:
        jobs = jobs[:args.limit]
    t0 = time.time()
    tools: Dict[str, Any] = {}
    counts: collections.Counter = collections.Counter()
    keys: collections.Counter = collections.Counter()
    prefixes: collections.Counter = collections.Counter()
    errors: List[Dict[str, Any]] = []
    tmp = os.path.join(args.out, "inputs.jsonl.gz.tmp")
    with gzip.open(tmp, "wt") as fh, mp.get_context("fork").Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if "error" in r:
                errors.append({"phys": r["phys"], "error": r["error"]})
                counts["errors"] += 1
            else:
                tools.setdefault(r["tool_sha"], r["tool"])
                keys[r["key"]] += 1
                prefixes[r["prefix_sha"]] += 1
                counts["rebuild|" + str(r.get("rebuild", "not_checked")).split(":")[0]] += 1
                if r["consistency_problems"]:
                    counts["with_consistency_problems"] += 1
                for nk, nv in (r.get("event_notes") or {}).items():
                    counts["transcripts_with_" + nk] += 1
                    counts["events_" + nk] += nv
            r.pop("tool", None)
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")
            counts["transcripts"] += 1
            if k % 500 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, os.path.join(args.out, "inputs.jsonl.gz"))
    # key order kept: the schema's property order is part of the judge input (request._ordered_json)
    json.dump(tools, open(os.path.join(args.out, "tools.json"), "w"), indent=0)
    rep = {"counts": dict(counts), "distinct_keys": len(keys), "duplicate_keys": {k: v for k, v in keys.items() if v > 1},
           "distinct_prefixes": len(prefixes), "prefix_sizes": sorted(prefixes.values()), "distinct_tools": len(tools),
           "errors": errors[:200], "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("counts", "distinct_keys", "distinct_prefixes", "distinct_tools", "elapsed_s")}, indent=1))
    print("duplicate keys:", len(rep["duplicate_keys"]), "errors:", len(errors))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
