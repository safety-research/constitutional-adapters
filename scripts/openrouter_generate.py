#!/usr/bin/env python3
"""Generate the Qwen thinking arm through OpenRouter instead of our own vLLM.

WHY. We only need the OUTPUTS. Standing up a 27B hybrid-Mamba model ourselves
cost a day of footguns (Mamba cache blocks, chat-template injection, token
caps) and the thinking arm still came out with ~20% of rows carrying no
answer. A hosted endpoint returns reasoning and answer as SEPARATE FIELDS, so
the whole class of "did the trace close" problems disappears.

WHAT IS DELIBERATELY PINNED, and why each matters:
  * ONE provider, allow_fallbacks FALSE. OpenRouter will happily route across
    providers running fp4 / fp8 / bf16 / unknown quantisations. A dataset
    assembled from a mix is a dataset whose rows were produced by different
    models -- the same confound as generating half a pool on HF and half on
    vLLM (+25.8pp, measured on this project).
  * the SAME prompt inventory as every other model in this set, so the
    cross-model comparison still rests on identical prompts.
  * provider, quantisation and finish_reason are RECORDED on every row, so a
    reader can audit what produced the number instead of trusting this
    docstring.

COMPARABILITY CAVEAT, stated in the data and not only here: rows produced this
way are NOT stack-identical to the locally generated Qwen thinking-off pool.
Thinking-on vs thinking-off across the two therefore conflates reasoning with
serving stack. Generate BOTH arms here if that contrast is the point.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

API = "https://openrouter.ai/api/v1/chat/completions"


def load_key() -> str:
    """The OpenRouter key from the environment or the vault — never a file."""
    from sisteer.vault import openrouter_api_key
    return openrouter_api_key()


def si_texts(si_ids):
    from sisteer.data.instructions import REGISTRY, SAFETY_SI
    by = {i.id: i for i in REGISTRY}
    by.setdefault(SAFETY_SI.id, SAFETY_SI)
    out = {"none": None}
    for s in si_ids:
        out[s] = by[s].text
    return out


def row_key(p, cond):
    return f"{p['family']}|{p['goal_id']}|{p.get('sample_idx')}|{p.get('variant_name')}|{cond}"


async def one(session, sem, key, model, provider, prompt, cond, si, think,
              max_tokens, out_f, lock, stats, retries=5):
    import aiohttp
    body = {
        "model": model,
        "provider": {"order": [provider], "allow_fallbacks": False},
        "messages": ([{"role": "system", "content": si}] if si else [])
                    + [{"role": "user", "content": prompt["text"]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if not think:
        body["reasoning"] = {"enabled": False}
    async with sem:
        for attempt in range(retries):
            try:
                async with session.post(API, json=body, headers={
                        "Authorization": f"Bearer {key}"}) as r:
                    if r.status in (429, 500, 502, 503, 529):
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    d = await r.json()
                if "error" in d:
                    await asyncio.sleep(min(2 ** attempt, 30)); continue
                ch = d["choices"][0]
                m = ch["message"]
                u = d.get("usage", {}) or {}
                rec = {
                    # --- schema the rest of the pipeline already reads ---
                    "attack": prompt["family"], "goal_id": prompt["goal_id"],
                    "attack_prompt": prompt["text"],
                    "response": m.get("content") or "",
                    "sample_idx": prompt.get("sample_idx"),
                    "variant_name": prompt.get("variant_name"),
                    "trial": prompt.get("trial", 0), "candidate_idx": 0,
                    "category": prompt.get("category"),
                    "category_group": prompt.get("category_group"),
                    "source": prompt.get("source"), "side": prompt.get("side"),
                    "si_id": (cond if cond != "none" else None),
                    "si_condition": cond, "graded": False, "score": None,
                    # --- provenance: what actually produced this row ---
                    "reasoning": m.get("reasoning") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "gen_provider": d.get("provider"), "gen_model": d.get("model"),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {})
                                        .get("reasoning_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "cached_tokens": (u.get("prompt_tokens_details") or {})
                                     .get("cached_tokens"),
                    "cost_usd": u.get("cost"),
                }
                async with lock:
                    out_f.write(json.dumps(rec) + "\n")
                    stats["n"] += 1
                    stats["cost"] += float(u.get("cost") or 0)
                    stats["cached"] += int((u.get("prompt_tokens_details") or {})
                                           .get("cached_tokens") or 0)
                    stats["ptok"] += int(u.get("prompt_tokens") or 0)
                    if ch.get("finish_reason") == "length":
                        stats["truncated"] += 1
                return
            except Exception:                                    # noqa: BLE001
                await asyncio.sleep(min(2 ** attempt, 30))
        async with lock:
            stats["failed"] += 1


async def main_async(a):
    import aiohttp
    key = load_key()
    inv = json.loads(Path(a.inventory).expanduser().read_text())["prompts"]
    sis = si_texts([s for s in a.sis.split(",") if s])
    conds = (["none"] if not a.no_none else []) + [s for s in a.sis.split(",") if s]

    out_path = Path(a.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and a.resume:
        for line in out_path.open():
            try:
                r = json.loads(line)
                done.add(f"{r['attack']}|{r['goal_id']}|{r.get('sample_idx')}|"
                         f"{r.get('variant_name')}|{r.get('si_condition')}")
            except Exception:                                    # noqa: BLE001
                continue
        print(f"[or] resuming: {len(done)} rows already present", flush=True)

    todo = [(p, c) for c in conds for p in inv if row_key(p, c) not in done]
    print(f"[or] {len(todo)} calls to make "
          f"({len(inv)} prompts x {len(conds)} conditions, minus done)", flush=True)
    if not todo:
        print("[or] nothing to do"); return

    stats = {"n": 0, "cost": 0.0, "failed": 0, "truncated": 0,
             "cached": 0, "ptok": 0}
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=a.timeout)
    with out_path.open("a") as out_f:
        conn = aiohttp.TCPConnector(limit=a.concurrency + 20)
        async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
            tasks = [one(s, sem, key, a.model, a.provider, p, c, sis[c],
                         a.think, a.max_tokens, out_f, lock, stats)
                     for p, c in todo]
            for i in range(0, len(tasks), 2000):
                await asyncio.gather(*tasks[i:i + 2000])
                el = time.time() - t0
                rate = stats["n"] / max(el, 1) * 60
                hit = stats["cached"] / max(stats["ptok"], 1)
                print(f"[or] {stats['n']}/{len(todo)}  ${stats['cost']:.2f}  "
                      f"{rate:.0f}/min  cache-hit {hit:.0%}  "
                      f"trunc {stats['truncated']}  failed {stats['failed']}",
                      flush=True)
    print(f"[or] DONE {stats['n']} rows, ${stats['cost']:.2f}, "
          f"{stats['failed']} failed, {stats['truncated']} hit the token cap",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--provider", default="Alibaba",
                    help="PINNED, fallbacks disabled: mixing providers mixes "
                         "quantisations and stacks within one dataset")
    ap.add_argument("--sis", default="safety_refusal,si_constitution,si_irrelevant")
    ap.add_argument("--no-none", action="store_true")
    ap.add_argument("--think", action="store_true", default=True)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--resume", action="store_true", default=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
