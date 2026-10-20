#!/usr/bin/env python
"""record_resolved_sampler.py — WHAT THE ENGINE RESOLVED, read off the server's
own log, and compared against what the client sent.

    record_resolved_sampler.py --vllm-log <vllm_server.log> --out <record.json>
        [--sent-static <responses.jsonl> ...] [--sent-controls <meta.jsonl> ...]
        [--expect temperature=0.7 --expect top_p=0.8 ...]

WHY THIS EXISTS. The resolved sampler is a property of the SERVING STACK, not
of the model: vLLM fills every sampling key a request omits, from the model's
``generation_config.json`` when it ships one and from its own neutral defaults
when it does not. Two lanes on two stacks can therefore resolve the same
omitted key differently and produce a divergence no summary statistic shows.
So the resolved values are recorded as data and exchanged, the way a
classifier hash is.

WHAT IS AUTHORITATIVE HERE, and what is not:

* ``resolved`` — parsed from vLLM's own ``Received request <id>: params:
  SamplingParams(...)`` INFO lines (0.25.1 ``request_logger.py``), which the
  server emits only under ``--enable-log-requests``. This is the engine's
  object AFTER its defaults were merged, i.e. the real answer. The prompt is a
  separate DEBUG record and is NOT in these lines, so nothing harmful travels.
* ``generation_config_override`` — vLLM logs "Default vLLM sampling parameters
  have been overridden by the model's `generation_config.json`: {...}" ONLY
  when such a file exists and differs. Its ABSENCE is therefore positive
  evidence that no key could fall back to a vendor file. Recorded either way,
  because "absent" is a measurement here and not a missing value.
* ``sent`` — the sampler each CLIENT row recorded. Not authoritative about the
  engine; it is the other half of the comparison.

THE COMPARISON. Every distinct resolved tuple is reported, not just the modal
one: one control decoding differently from the rest is exactly the defect this
is for. A key that resolves to something other than what was sent is a
``divergence`` and makes the record's ``verdict`` ``DIVERGED``; a run with no
resolved lines at all is ``UNREADABLE`` (never "fine"), because the whole
point is that silence must not read as agreement.

vLLM 0.25.1's own neutral defaults, for reading a divergence against:
temperature 1.0, top_p 1.0, top_k 0, min_p 0.0, repetition_penalty 1.0
(``entrypoints/openai/chat_completion/protocol.py`` ``_DEFAULT_SAMPLING_PARAMS``).

NOT COVERED, stated rather than implied: ``max_tokens`` is COMPUTED by the
server (``get_max_tokens``) and can be clamped below what was asked, so a
difference there is not a sampler divergence and is reported separately; keys
absent from ``SamplingParams.__repr__`` cannot be read this way at all; and
this reads the ENGINE's line, so a rewrite between client and engine by the
routing proxy would show up here as a divergence without naming the proxy as
its cause.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

#: the sampler keys this record compares (bon_served.SAMPLER_KEYS order)
SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "repetition_penalty")
#: vLLM 0.25.1's neutral fallbacks — what an OMITTED key becomes when the
#: model ships no generation_config.json
VLLM_NEUTRAL_DEFAULTS = {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                         "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}

_FLAG_ATTEST = re.compile(r"`--enable-log-requests` is set")
_REQ = re.compile(r"Received request ([^\s:]+): params: SamplingParams\((.*?)\), "
                  r"lora_request:", re.S)
_OVERRIDE = re.compile(
    r"Default vLLM sampling parameters have been overridden by (.+?): `(.+?)`")


def _parse_params(body: str) -> dict:
    """``SamplingParams(k=v, ...)`` -> the SAMPLER_KEYS (+ seed, max_tokens)
    it states. Values are read as Python literals; anything that is not a
    literal is kept as its source text rather than evaluated."""
    out: dict = {}
    for m in re.finditer(r"(\w+)=", body):
        key = m.group(1)
        if key not in (*SAMPLER_KEYS, "seed", "max_tokens", "n"):
            continue
        rest = body[m.end():]
        # up to the next ", <word>=" at this nesting level
        nxt = re.search(r",\s+\w+=", rest)
        raw = (rest[:nxt.start()] if nxt else rest).strip().rstrip(",")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def read_log(path: Path) -> dict:
    text = Path(path).read_text(errors="replace")
    resolved = [(rid, _parse_params(body)) for rid, body in _REQ.findall(text)]
    overrides = [{"source": a, "params": b} for a, b in _OVERRIDE.findall(text)]
    return {
        "vllm_log": str(path),
        "n_resolved_lines": len(resolved),
        "resolved": resolved,
        # vLLM 0.25.1 request_logger.py:28 attests the flag at server start, BEFORE
        # any request exists; a readback that runs at standup reads this, not silence
        "log_requests_flag_attested": bool(_FLAG_ATTEST.search(text)),
        "generation_config_override": {
            "present": bool(overrides),
            "lines": overrides,
            "reading": (
                "a generation_config.json governs the defaults for any key a "
                "request omits" if overrides else
                "ABSENT: the served snapshot ships no generation_config.json "
                "(or it does not differ), so an omitted key falls to vLLM's "
                "own neutral defaults — and this lane omits none"),
        },
    }


def _sampler_of(row: dict) -> "dict | None":
    """The sampler a CLIENT row recorded: the controls meta sidecar's
    ``sampler`` block, or a static row's ``decode`` block minus its
    non-sampler metadata."""
    if isinstance(row.get("sampler"), dict):
        return {k: row["sampler"][k] for k in SAMPLER_KEYS if k in row["sampler"]}
    d = row.get("decode")
    if isinstance(d, dict):
        return {k: d[k] for k in SAMPLER_KEYS if k in d}
    return None


def _gen_stats_samplers(path: Path) -> list:
    """The canonical controls store records its decode ONCE per control in
    ``gen_stats.json[control]["decode"]`` (the ported ``_run_judged_served``;
    the per-item meta rows carry the seed and finish facts, not the sampler).
    Returns one (sampler, n_items) per sampled control block."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for control, blk in d.items():
        if not isinstance(blk, dict) or not isinstance(blk.get("decode"), dict):
            continue
        s = {k: blk["decode"][k] for k in SAMPLER_KEYS if k in blk["decode"]}
        if s:
            out.append((s, int(blk.get("n_total") or blk.get("n") or 1), blk["decode"].get("cap")))
    return out


def _cap_of(row: dict):
    """The cap a CLIENT row asked for: a static row's ``max_tokens`` (else its
    ``decode.cap``), a controls meta row's ``sampler``/``decode`` cap."""
    for k in ("max_tokens", "cap"):
        if row.get(k) is not None:
            return str(row[k])
    d = row.get("decode") if isinstance(row.get("decode"), dict) else row.get("sampler")
    if isinstance(d, dict) and d.get("cap") is not None:
        return str(d["cap"])
    return None


def read_sent(paths) -> dict:
    counts: Counter = Counter()
    caps: Counter = Counter()
    n_rows = 0
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            continue
        if p.name == "gen_stats.json":
            for s, n, cap in _gen_stats_samplers(p):
                counts[json.dumps(s, sort_keys=True)] += n
                n_rows += n
                if cap is not None:
                    caps[str(cap)] += n
            continue
        for line in p.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = _sampler_of(row) if isinstance(row, dict) else None
            if s is None:
                continue
            n_rows += 1
            counts[json.dumps(s, sort_keys=True)] += 1
            cap = _cap_of(row)
            if cap is not None:
                caps[cap] += 1
    return {"n_rows": n_rows,
            "distinct": [{"sampler": json.loads(k), "n": v}
                         for k, v in counts.most_common()],
            "distinct_max_tokens": [{"max_tokens": k, "n": v} for k, v in caps.most_common()]}


def _probe_cap():
    """The served target's probe cap, from the code that fires the probes."""
    try:
        from sisteer.eval.attacks.served_target import PROBE_MAX_TOKENS
        return int(PROBE_MAX_TOKENS)
    except Exception:  # noqa: BLE001 — no sisteer on path: no probe class, everything is client
        return None


def _mmlu_cap():
    """The MMLU letter prefill's cap, from the driver that sends it: ONE greedy token,
    teacher-forced, `logprobs` on. It is a CLIENT leg but its rows record no decode block, so the
    readback cannot learn its cap from --sent-*; keyed on the producer's constant instead."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_scr_const", Path(__file__).resolve().parents[2] / "scripts" / "served_capability_run.py")
        # read the constant WITHOUT importing the driver (it pulls the whole serving stack)
        import re as _re
        src = (Path(__file__).resolve().parents[2] / "scripts" / "served_capability_run.py").read_text()
        m = _re.search(r"^MMLU_MAX_TOKENS = (\d+)$", src, _re.M)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


#: what an MMLU prefill request carries (served_capability_run._mmlu_payload): temperature 0.0,
#: seed 0, max_tokens MMLU_MAX_TOKENS, and NO other sampler key — so every other key resolves to
#: vLLM's neutral default, exactly like the liveness probes.
MMLU_SENT = {"temperature": 0.0}
MMLU_REFERENCE_SOURCE = ("served_capability_run._mmlu_payload: the letter prefill sends "
                         "temperature=0.0, seed=0, max_tokens=MMLU_MAX_TOKENS and no other sampler "
                         "key, so the rest resolve to vLLM's neutral defaults")


#: what a probe REQUEST carries (bites_policy.neutral_probe_payload /
#: harmful_probe_payload): ``temperature: 0.0, seed: 0, max_tokens``, and NO
#: other sampler key — so every other key resolves to vLLM's own neutral default.
PROBE_SENT = {"temperature": 0.0}
PROBE_REFERENCE_SOURCE = ("sisteer.eval.attacks.bites_policy.neutral_probe_payload / "
                          "harmful_probe_payload: temperature=0.0, seed=0, max_tokens="
                          "served_target.PROBE_MAX_TOKENS; every other sampler key omitted, "
                          "so it resolves to vLLM's neutral default")


def probe_reference() -> dict:
    ref = {k: VLLM_NEUTRAL_DEFAULTS[k] for k in SAMPLER_KEYS if k in VLLM_NEUTRAL_DEFAULTS}
    ref.update(PROBE_SENT)
    return ref


def partition(resolved, client_caps, probe_cap, mmlu_cap=None):
    """Requests by CLASS, keyed on facts the producers declare — never on how a
    row looks. ``client``: a ``max_tokens`` some client row recorded (the rows
    under --sent-*). ``probe``: ``max_tokens == served_target.PROBE_MAX_TOKENS``,
    the steering-bites / hook-attestation probes the standup and every static
    driver fire before scoring anything (job 487306: 180 of 5,360, greedy at
    48). ``unclassified``: neither — reported, never folded into a class.
    A single-class comparison flagged the probes as four diverged keys, i.e.
    the instrument's own calibration requests as a defect in the thing measured
    (2026-09-12); only a WITHIN-class mismatch is a divergence. Refuses when a
    client cap equals the probe cap: the classes would be indistinguishable."""
    if probe_cap is not None and str(probe_cap) in client_caps:
        raise ValueError(f"a client row recorded max_tokens={probe_cap}, the probe cap "
                         "(served_target.PROBE_MAX_TOKENS) — the request classes cannot be "
                         "told apart, so no verdict is possible")
    classes = {"client": [], "probe": [], "mmlu": [], "unclassified": []}
    for rid, p in resolved:
        cap = str(p.get("max_tokens"))
        if cap in client_caps:
            classes["client"].append((rid, p))
        elif probe_cap is not None and cap == str(probe_cap):
            classes["probe"].append((rid, p))
        elif mmlu_cap is not None and cap == str(mmlu_cap):
            classes["mmlu"].append((rid, p))
        elif not client_caps and probe_cap is not None and cap != str(probe_cap):
            classes["client"].append((rid, p))   # no client rows given: everything non-probe is client
        elif not client_caps:
            classes["client"].append((rid, p))
        else:
            classes["unclassified"].append((rid, p))
    return classes


def compare(resolved, sent_distinct, expect: dict) -> dict:
    """Distinct resolved tuples, and every key that resolved to something
    other than what was sent (or than ``--expect``)."""
    counts: Counter = Counter()
    caps: Counter = Counter()
    for _rid, p in resolved:
        counts[json.dumps({k: p[k] for k in SAMPLER_KEYS if k in p},
                          sort_keys=True)] += 1
        if "max_tokens" in p:
            caps[str(p["max_tokens"])] += 1
    distinct = [{"sampler": json.loads(k), "n": v} for k, v in counts.most_common()]
    # the reference to compare against: --expect wins, else the single sampler
    # the client rows agree on (more than one and there is nothing to compare
    # against — that is itself reported)
    reference = dict(expect) if expect else None
    ref_source = "--expect" if expect else None
    if reference is None and len(sent_distinct) == 1:
        reference = sent_distinct[0]["sampler"]
        ref_source = "the client rows' recorded sampler"
    divergences = []
    if reference is not None:
        for block in distinct:
            got = block["sampler"]
            for k, want in reference.items():
                if k not in got:
                    divergences.append(
                        {"key": k, "sent": want, "resolved": None,
                         "note": "the engine's params line does not state this "
                                 "key — it is outside SamplingParams.__repr__, "
                                 "so it cannot be read back this way",
                         "n_requests": block["n"]})
                elif float(got[k]) != float(want):
                    divergences.append(
                        {"key": k, "sent": want, "resolved": got[k],
                         "n_requests": block["n"],
                         "equals_vllm_neutral_default":
                             (k in VLLM_NEUTRAL_DEFAULTS
                              and float(got[k]) == float(VLLM_NEUTRAL_DEFAULTS[k]))})
    return {"distinct_resolved": distinct,
            "distinct_max_tokens": [{"max_tokens": k, "n": v}
                                    for k, v in caps.most_common()],
            "reference": reference, "reference_source": ref_source,
            "divergences": divergences}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-log", required=True, type=Path, action="append",
                    help="a vLLM server log (repeatable)")
    ap.add_argument("--sent-static", type=Path, action="append", default=[],
                    help="a static family responses.jsonl (repeatable)")
    ap.add_argument("--sent-controls", type=Path, action="append", default=[],
                    help="a judged control store's gen_stats.json (the canonical layout: the "
                         "decode block per control) or a legacy per-item meta.jsonl (repeatable)")
    ap.add_argument("--expect", action="append", default=[],
                    help="key=value the resolved sampler must equal (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    expect = {}
    for kv in a.expect:
        k, _, v = kv.partition("=")
        if k not in SAMPLER_KEYS:
            print(f"REFUSED: --expect {kv!r}: {k!r} is not a sampler key "
                  f"{SAMPLER_KEYS}", file=sys.stderr)
            return 2
        expect[k] = float(v)

    logs = [read_log(p) for p in a.vllm_log if Path(p).exists()]
    missing = [str(p) for p in a.vllm_log if not Path(p).exists()]
    resolved_all = [r for lg in logs for r in lg["resolved"]]
    sent = read_sent(list(a.sent_static) + list(a.sent_controls))
    client_caps = {str(c["max_tokens"]) for c in sent.get("distinct_max_tokens", [])}
    probe_cap = _probe_cap()
    mmlu_cap = _mmlu_cap()
    try:
        classes = partition(resolved_all, client_caps, probe_cap, mmlu_cap)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    per_class = {}
    for name, rows in classes.items():
        if name == "client":
            cmp_ = compare(rows, sent["distinct"], expect)
        elif name == "probe":
            cmp_ = compare(rows, [{"sampler": probe_reference(), "n": len(rows)}], {})
            cmp_["reference_source"] = PROBE_REFERENCE_SOURCE if rows else None
        elif name == "mmlu":
            ref = {k: VLLM_NEUTRAL_DEFAULTS[k] for k in SAMPLER_KEYS if k in VLLM_NEUTRAL_DEFAULTS}
            ref.update(MMLU_SENT)
            cmp_ = compare(rows, [{"sampler": ref, "n": len(rows)}], {})
            cmp_["reference_source"] = MMLU_REFERENCE_SOURCE if rows else None
        else:
            cmp_ = compare(rows, [], {})
        if not rows:
            v = "EMPTY"
        elif name == "unclassified":
            v = "UNCLASSIFIED"
        elif cmp_["reference"] is None:
            v = "UNCOMPARED"
        elif cmp_["divergences"]:
            v = "DIVERGED"
        else:
            v = "AGREES"
        per_class[name] = {"n": len(rows), "verdict": v, **cmp_}
    n_client = per_class["client"]["n"]
    n_probe = per_class["probe"]["n"]
    n_mmlu = per_class["mmlu"]["n"]
    n_uncl = per_class["unclassified"]["n"]
    diverged = [n for n, c in per_class.items() if c["verdict"] == "DIVERGED"]

    if not logs:
        verdict = "UNREADABLE"
        why = f"no vLLM log was readable (missing: {missing})"
    elif not resolved_all and any(lg["log_requests_flag_attested"] for lg in logs):
        verdict = "FLAG_SET_NO_REQUESTS_YET"
        why = ("the server attests `--enable-log-requests` (request_logger.py:28) "
               "but has logged no request yet — this readback ran before any "
               "request was sent (a standup-time call). It says nothing about "
               "the sampler either way; read again after the legs")
    elif not resolved_all:
        verdict = "UNREADABLE"
        why = ("the server logged no `Received request ... params: "
               "SamplingParams(...)` line and never attested the flag — it was "
               "launched WITHOUT --enable-log-requests (or VLLM_LOGGING_LEVEL is "
               "above INFO), so what resolved is unrecorded. Silence here is NOT "
               "agreement")
    elif diverged:
        verdict = "DIVERGED"
        why = (f"within-class mismatch in {diverged}: "
               + "; ".join(f"{n}: {len(per_class[n]['divergences'])} key(s) resolved to a value "
                           f"other than its reference ({per_class[n]['reference_source']})"
                           for n in diverged) + " — see `classes`")
    elif n_uncl:
        verdict = "UNCLASSIFIED_REQUESTS"
        why = (f"{n_uncl} request(s) match neither a client cap {sorted(client_caps)} nor the "
               f"probe cap {probe_cap}; nothing is folded into a class it does not belong to")
    elif n_client and per_class["client"]["verdict"] == "UNCOMPARED":
        verdict = "UNCOMPARED"
        why = ("client requests were read, but there is no single reference to "
               "compare them against (pass --expect, or point --sent-* at rows "
               "that agree on one sampler)")
    else:
        verdict = "AGREES"
        why = (f"every class is internally uniform and matches its reference: "
               f"client {n_client} request(s) = the rows' recorded sampler"
               + (f"; probe {n_probe} request(s) at max_tokens {probe_cap} = greedy + neutral "
                  f"defaults ({PROBE_REFERENCE_SOURCE.split(':')[0]})" if n_probe else ""))

    rec = {"verdict": verdict, "why": why,
           "classes": per_class,
           "class_rule": ("client: max_tokens in the caps the --sent-* rows recorded; probe: "
                          "max_tokens == sisteer.eval.attacks.served_target.PROBE_MAX_TOKENS "
                          f"({probe_cap}); mmlu: max_tokens == served_capability_run.MMLU_MAX_TOKENS "
                          f"({mmlu_cap}), the teacher-forced letter prefill; a class is compared "
                          "only against ITS reference"),
           "n_client_requests": n_client, "n_probe_requests": n_probe,
           "n_mmlu_requests": n_mmlu,
           "n_unclassified_requests": n_uncl,
           "n_logs": len(logs), "missing_logs": missing,
           "n_resolved_requests": len(resolved_all),
           "vllm_neutral_defaults": VLLM_NEUTRAL_DEFAULTS,
           "generation_config_override": [lg["generation_config_override"]
                                          for lg in logs],
           "sent": sent,
           # the client class at top level too: what earlier readers cite
           **{k: per_class["client"][k] for k in
              ("distinct_resolved", "distinct_max_tokens", "reference", "reference_source", "divergences")},
           "logs": [{k: v for k, v in lg.items() if k != "resolved"}
                    for lg in logs]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=False, default=str))
    print(f"[resolved-sampler] {verdict}: {why}")
    for name in ("client", "probe", "mmlu", "unclassified"):
        for block in per_class[name]["distinct_resolved"]:
            print(f"[resolved-sampler]   {name:12s} n={block['n']:6d}  {block['sampler']}")
    for lg in logs:
        g = lg["generation_config_override"]
        print(f"[resolved-sampler]   generation_config override: {g['present']} ({g['reading']})")
    print(f"[resolved-sampler] -> {a.out}")
    return {"AGREES": 0, "DIVERGED": 1, "UNREADABLE": 3,
            "FLAG_SET_NO_REQUESTS_YET": 4}.get(verdict, 3)  # UNCOMPARED / UNCLASSIFIED -> 3


if __name__ == "__main__":
    raise SystemExit(main())
