#!/usr/bin/env python
"""Measure the ATTACKER's request geometry from recorded attacker/scorer calls.

The shared attacker pool (scripts/sbatch/attacker_pool.sbatch) needs a
``max_model_len`` sized the way every served window is sized
(``sisteer.capeval.serving.geometry`` — the 2026-09-03 ruling): the p99 of
prompt + completion over REAL calls, unless the clamp rate at that window
exceeds 5%, in which case p99(prompt) + the requested ``max_tokens``. The
served-target table cannot supply this number: it measures the TARGET's
requests (a gemma/llama prompt + a 2,048 completion); the attacker's requests
are the PyRIT adversarial chat (system prompt + the growing attack
transcript, up to 1,024 new tokens) and the in-loop scorer's rubric prompts
(8 new tokens) — a different distribution on a different tokenizer.

Input: the drivers' per-call ledgers, ``attacker_calls.jsonl`` and
``scorer_calls.jsonl`` (``RecordingAttackerClient`` rows: ``role``,
``messages``, ``reply``, ``params``), found under one or more roots
(``<root>/**/<attack>/<cell>/shardNN/{attacker,scorer}_calls.jsonl``). The
vLLM attacker client records no ``usage`` (``meta`` is empty), so lengths are
re-tokenized here with the ATTACKER's exact tokenizer: prompt = the chat
template rendered the way the server renders it (``add_generation_prompt``,
``enable_thinking=False`` — the client's ``chat_template_kwargs``); completion
= the recorded reply (+1 EOS), clamped to the requested cap. A scorer row's
reply is stored truncated to 200 chars, so its completion column is its
``max_tokens`` (8) — an upper bound, and immaterial next to its prompt.

Output: a JSON report (``--out``) and a markdown table on stdout (the format
of ``docs/served_geometry_p99.md``) — counts and percentiles only, never
prompt text. Run where the snapshot is cached (a cluster login node, niced —
tokenizing a few thousand rows is seconds of CPU, no GPU):

  nice -n 19 <vllm-venv>/bin/python scripts/attacker_pool_profile.py \\
      --tokenizer <snapshot dir> --root <SHARED>/exp/<lane>/transcripts_results \\
      --out <SHARED>/exp/attacker-pool/profile.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

#: candidate windows round up to this (the doc's convention; the vLLM block is 16)
ROUND_TO = 256
CLAMP_RATE_CEILING = 0.05
ATTACKER_MAX_TOKENS = 1024     # cfg attacks.attacker_max_new_tokens (every two_si config)
SCORER_MAX_TOKENS = 8          # cfg attacks.scorer_max_new_tokens


def next_multiple(n: int, m: int = ROUND_TO) -> int:
    return -(-int(n) // m) * m


def pct(xs: list[int], q: float) -> int:
    """Nearest-rank percentile (the doc's p50/p90/p99 convention)."""
    if not xs:
        return 0
    s = sorted(xs)
    k = max(1, math.ceil(q * len(s)))
    return int(s[k - 1])


def size_window(prompt: list[int], total: list[int], max_tokens: int) -> dict:
    """The sizing rule of ``geometry.sizing_problem`` applied to one arm's
    measured calls: L_total = next multiple >= p99(total); clamp rate at it =
    share of calls with prompt + max_tokens > L_total; above the ceiling the
    window becomes next multiple >= p99(prompt) + max_tokens."""
    if not total:
        return {"L": None, "sized_by": None, "clamp_rate_at_total_sizing": None}
    p99_total, p99_prompt = pct(total, 0.99), pct(prompt, 0.99)
    L_total = next_multiple(p99_total)
    clamp_total = sum(1 for p in prompt if p + max_tokens > L_total) / len(prompt)
    if clamp_total > CLAMP_RATE_CEILING:
        L = next_multiple(p99_prompt + max_tokens)
        sized_by = "p99_prompt_plus_max_tokens"
    else:
        L, sized_by = L_total, "p99_total"
    return {
        "L": L, "sized_by": sized_by,
        "L_total_sizing": L_total,
        "clamp_rate_at_total_sizing": round(clamp_total, 4),
        "clamp_rate_at_L": round(sum(1 for p in prompt if p + max_tokens > L) / len(prompt), 4),
        "frac_over_L": round(sum(1 for t in total if t > L) / len(total), 4),
    }


def summarize(rows: list[dict], max_tokens: int) -> dict:
    prompt = [r["prompt_tokens"] for r in rows]
    comp = [r["completion_tokens"] for r in rows]
    total = [p + c for p, c in zip(prompt, comp)]
    out = {
        "n_calls": len(rows),
        "p50_total": pct(total, 0.5), "p90_total": pct(total, 0.9),
        "p99_total": pct(total, 0.99), "max_total": max(total, default=0),
        "p99_prompt": pct(prompt, 0.99), "max_prompt": max(prompt, default=0),
        "p99_completion": pct(comp, 0.99), "max_completion": max(comp, default=0),
        "requested_max_tokens": max_tokens,
        "n_at_cap": sum(1 for c in comp if c >= max_tokens),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "mean_latency_s": (round(sum(r["latency_s"] for r in rows) / len(rows), 2)
                           if rows else None),
    }
    out.update(size_window(prompt, total, max_tokens))
    return out


def _attack_of(path: Path) -> str:
    # <...>/<attack>/<cell>/shardNN/<role>_calls.jsonl
    try:
        return path.parents[2].name
    except IndexError:
        return "unknown"


def _iter_rows(path: Path):
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue        # torn tail


def tokenize_rows(files: list[Path], tok, *, max_tokens_by_role: dict) -> dict:
    """``{(role, attack): [row, ...]}`` with per-call token counts."""
    grouped: dict = defaultdict(list)
    n_files = 0
    for f in files:
        n_files += 1
        attack = _attack_of(f)
        for rec in _iter_rows(f):
            role = rec.get("role") or ("scorer" if f.name.startswith("scorer") else "attacker")
            msgs = rec.get("messages") or []
            if not msgs:
                continue
            try:
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False)
            except Exception:  # noqa: BLE001 — a malformed row is skipped, counted
                grouped[("skipped", attack)].append({"prompt_tokens": 0, "completion_tokens": 0,
                                                     "latency_s": 0.0, "error": "render"})
                continue
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            n_prompt = len(ids)
            cap = int((rec.get("params") or {}).get("max_new_tokens")
                      or max_tokens_by_role.get(role, ATTACKER_MAX_TOKENS))
            reply = rec.get("reply")
            if rec.get("error") or reply is None:
                n_comp = 0
            elif role == "scorer":
                n_comp = cap                        # reply stored truncated; cap is the bound
            else:
                n_comp = min(cap, len(tok(reply, add_special_tokens=False)["input_ids"]) + 1)
            grouped[(role, attack)].append({
                "prompt_tokens": n_prompt, "completion_tokens": n_comp,
                "latency_s": float(rec.get("latency_s") or 0.0), "error": rec.get("error")})
    return grouped, n_files


def build_report(grouped: dict, *, max_tokens_by_role: dict, n_files: int,
                 tokenizer: str) -> dict:
    rep = {"tokenizer": tokenizer, "n_files": n_files, "rows": {}, "pooled": {}}
    by_role: dict = defaultdict(list)
    for (role, attack), rows in sorted(grouped.items()):
        if role == "skipped":
            rep.setdefault("skipped", {})[attack] = len(rows)
            continue
        rep["rows"][f"{role}/{attack}"] = summarize(rows, max_tokens_by_role[role])
        by_role[role].extend(rows)
    everything: list = []
    for role, rows in by_role.items():
        rep["pooled"][role] = summarize(rows, max_tokens_by_role[role])
        everything.extend(rows)
    # the pool serves BOTH roles from one server: its window is the max of the
    # per-role rule windows (a scorer call is short-completion but long-prompt).
    Ls = [v["L"] for v in rep["pooled"].values() if v.get("L")]
    rep["pool_max_model_len"] = max(Ls) if Ls else None
    rep["pool_sized_by"] = {r: v["sized_by"] for r, v in rep["pooled"].items()}
    rep["n_calls_total"] = len(everything)
    return rep


def markdown(rep: dict) -> str:
    hdr = ("| role / attack | n calls | p50 | p90 | p99 | max | p99 prompt | max prompt | "
           "p99 comp | max comp | at cap | max_tokens | L | sized by | clamp @L | frac > L |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    lines = [hdr]

    def row(name, s):
        return (f"| {name} | {s['n_calls']:,} | {s['p50_total']:,} | {s['p90_total']:,} | "
                f"{s['p99_total']:,} | {s['max_total']:,} | {s['p99_prompt']:,} | {s['max_prompt']:,} | "
                f"{s['p99_completion']:,} | {s['max_completion']:,} | {s['n_at_cap']:,} | "
                f"{s['requested_max_tokens']:,} | {s['L']:,} | {s['sized_by']} | "
                f"{100 * s['clamp_rate_at_L']:.2f}% | {100 * s['frac_over_L']:.2f}% |")
    for role, s in rep["pooled"].items():
        lines.append(row(f"**{role} (ALL attacks)**", s))
    for key, s in rep["rows"].items():
        lines.append(row(key, s))
    lines.append("")
    lines.append(f"pool max_model_len (max over roles of the rule's window): "
                 f"**{rep['pool_max_model_len']:,}** ({rep['pool_sized_by']}); "
                 f"{rep['n_calls_total']:,} calls from {rep['n_files']} ledger files")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True,
                    help="the attacker snapshot dir (or hub id, if online)")
    ap.add_argument("--root", action="append", required=True,
                    help="a directory searched recursively for {attacker,scorer}_calls.jsonl "
                         "(repeatable)")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--attacker-max-tokens", type=int, default=ATTACKER_MAX_TOKENS)
    ap.add_argument("--scorer-max-tokens", type=int, default=SCORER_MAX_TOKENS)
    ap.add_argument("--no-scorer", action="store_true", help="attacker ledgers only")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for root in args.root:
        files += sorted(Path(root).rglob("attacker_calls.jsonl"))
        if not args.no_scorer:
            files += sorted(Path(root).rglob("scorer_calls.jsonl"))
    if not files:
        print(f"no *_calls.jsonl under {args.root}", file=sys.stderr)
        return 2
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    caps = {"attacker": args.attacker_max_tokens, "scorer": args.scorer_max_tokens}
    grouped, n_files = tokenize_rows(files, tok, max_tokens_by_role=caps)
    rep = build_report(grouped, max_tokens_by_role=caps, n_files=n_files,
                       tokenizer=str(args.tokenizer))
    rep["roots"] = [str(r) for r in args.root]
    print(markdown(rep))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
