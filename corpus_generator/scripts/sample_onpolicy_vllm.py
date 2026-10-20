#!/usr/bin/env python3
"""Cluster-side on-policy wave-C sampler (vLLM, offline LLM() engine).

Self-contained: stdlib + vllm only (runs in the cluster serving venv, NOT the
repo venv). Reads the sha-stamped prompt-set artifact extracted by
`corpusgen onpolicy extract`, renders each (system, user) conversation through the
model's OWN chat template, samples ONE response per prompt with wave C's
sampling params (read from the prompt-set manifest — temperature/top_p/
max_tokens; never hardcoded here), and writes:

  <out-dir>/samples.jsonl          {prompt_id, text, finish_reason, ...}
  <out-dir>/SAMPLES_MANIFEST.json  full sampler provenance

Resumable: already-sampled prompt_ids are skipped on restart (samples.jsonl
is append-only, flushed per chunk so the log/file grow visibly for watchers).

Determinism: each request gets seed = sha256(f"{base_seed}:{prompt_id}") so a
re-run (any batching order, any restart point) reproduces the same draws.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def request_seed(base_seed: int, prompt_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{prompt_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-set", required=True, type=Path)
    ap.add_argument("--model", required=True, help="HF repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0, help="base seed for per-request seeds")
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=128, help="prompts per flush chunk")
    ap.add_argument(
        "--hybrid", action="store_true",
        help="Qwen3.5-style hybrid attention: mamba_cache_mode=align so "
             "prefix caching works (requires chunked prefill, left on)",
    )
    args = ap.parse_args()

    manifest_path = args.prompt_set.with_name(args.prompt_set.stem + ".manifest.json")
    pm = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = sha256_file(args.prompt_set)
    if actual_sha != pm["prompt_set_sha256"]:
        sys.exit(
            f"prompt set sha256 {actual_sha[:12]} != manifest "
            f"{pm['prompt_set_sha256'][:12]} — refusing to sample from a "
            f"modified prompt set"
        )
    sampling_cfg = pm["sampling"]
    log(
        f"prompt set {args.prompt_set} ok (sha {actual_sha[:12]}, "
        f"n={pm['n_prompts']}); sampling params from manifest: {sampling_cfg}"
    )

    prompts = [
        json.loads(line)
        for line in args.prompt_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "samples.jsonl"
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["prompt_id"])
        log(f"resume: {len(done)} prompts already sampled in {out_path}")
    todo = [p for p in prompts if p["prompt_id"] not in done]
    if not todo:
        log("nothing to do — all prompts sampled")
    log(f"phase=engine-init model={args.model} rev={args.revision} todo={len(todo)}")

    from vllm import LLM, SamplingParams
    import vllm

    llm_kwargs = dict(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,  # shared injection prefixes across prompts
        limit_mm_per_prompt={"image": 0, "video": 0},  # text-only budgets
        seed=args.seed,
    )
    if args.hybrid:
        # prefix caching on hybrid attention needs the aligned mamba cache;
        # align REQUIRES chunked prefill, so leave chunked prefill at its
        # default (on) — 328401 died on the contradictory pair
        llm_kwargs["mamba_cache_mode"] = "align"
    try:
        llm = LLM(**llm_kwargs)
    except TypeError as e:
        # older vllm without a kwarg (e.g. mamba_cache_mode): drop prefix
        # caching (an optimization only — sampling params are unaffected)
        log(f"engine kwargs rejected ({e}); retrying without prefix caching")
        llm_kwargs.pop("mamba_cache_mode", None)
        llm_kwargs["enable_prefix_caching"] = False
        llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    log(f"phase=engine-ready vllm={vllm.__version__}")

    # Render through the model's OWN chat template. enable_thinking=False is
    # passed unconditionally: templates that know the flag (Qwen) disable
    # thinking traces; templates that don't simply ignore an unused variable.
    def render(p: dict) -> str:
        conv = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        return tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    chat_mode = "system_role"
    try:
        render(prompts[0])
    except Exception as e:  # template refuses a system role -> fold into user
        log(f"system-role template failed ({e}); folding system into user turn")
        chat_mode = "folded_system"

        def render(p: dict) -> str:  # noqa: F811
            conv = [{"role": "user", "content": f"{p['system']}\n\n{p['user']}"}]
            return tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

    # pre-render + length-check EVERYTHING before burning GPU time
    rendered = {p["prompt_id"]: render(p) for p in todo}
    max_out = int(sampling_cfg["max_tokens"])
    too_long = []
    max_prompt_tokens = 0
    for pid, text in rendered.items():
        n = len(tokenizer.encode(text))
        max_prompt_tokens = max(max_prompt_tokens, n)
        if n + max_out > args.max_model_len:
            too_long.append((pid, n))
    log(f"max prompt tokens (rendered): {max_prompt_tokens}")
    if too_long:
        sys.exit(
            f"{len(too_long)} prompts exceed max_model_len {args.max_model_len} "
            f"minus max_tokens {max_out} (first: {too_long[:3]}) — raise "
            f"--max-model-len"
        )

    started = time.time()
    n_done = 0
    gen_tokens = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i : i + args.chunk]
            params = [
                SamplingParams(
                    temperature=float(sampling_cfg["temperature"]),
                    top_p=float(sampling_cfg["top_p"]),
                    max_tokens=max_out,
                    seed=request_seed(args.seed, p["prompt_id"]),
                )
                for p in chunk
            ]
            outs = llm.generate(
                [rendered[p["prompt_id"]] for p in chunk], params, use_tqdm=False
            )
            for p, o in zip(chunk, outs):
                comp = o.outputs[0]
                out_f.write(
                    json.dumps(
                        {
                            "prompt_id": p["prompt_id"],
                            "text": comp.text,
                            "finish_reason": comp.finish_reason,
                            "prompt_tokens": len(o.prompt_token_ids),
                            "completion_tokens": len(comp.token_ids),
                            "seed": request_seed(args.seed, p["prompt_id"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                gen_tokens += len(comp.token_ids)
            out_f.flush()
            os.fsync(out_f.fileno())
            n_done += len(chunk)
            el = time.time() - started
            log(
                f"progress {n_done}/{len(todo)} sampled "
                f"({gen_tokens} gen tokens, {gen_tokens / max(el, 1e-9):.0f} tok/s)"
            )

    # ---- manifest over the FULL samples file (all rows, resumed included)
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    finish = {}
    think_tags = 0
    empty = 0
    comp_tokens = []
    for r in rows:
        finish[r["finish_reason"]] = finish.get(r["finish_reason"], 0) + 1
        comp_tokens.append(r["completion_tokens"])
        if "<think>" in r["text"]:
            think_tags += 1
        if not r["text"].strip():
            empty += 1
    comp_tokens.sort()

    def pct(q: float) -> int:
        return comp_tokens[min(len(comp_tokens) - 1, int(q * len(comp_tokens)))] if comp_tokens else 0

    manifest = {
        "engine": "vllm",
        "vllm_version": vllm.__version__,
        "model": args.model,
        "revision": args.revision,
        "chat_mode": chat_mode,
        "enable_thinking": False,
        "sampling": {
            **sampling_cfg,
            "seed_scheme": f"sha256('{args.seed}:'+prompt_id)[:4] & 0x7fffffff",
            "base_seed": args.seed,
        },
        "engine_config": {
            k: llm_kwargs[k]
            for k in llm_kwargs
            if k in ("max_model_len", "max_num_seqs", "gpu_memory_utilization",
                     "enable_prefix_caching", "mamba_cache_mode", "dtype")
        },
        "prompt_set": str(args.prompt_set),
        "prompt_set_sha256": actual_sha,
        "n_prompts": len(prompts),
        "n_sampled": len(rows),
        "finish_reasons": finish,
        "empty_responses": empty,
        "think_tag_responses": think_tags,
        "completion_tokens": {
            "total": sum(comp_tokens),
            "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
            "max": comp_tokens[-1] if comp_tokens else 0,
        },
        "max_prompt_tokens_this_run": max_prompt_tokens,
        "elapsed_s": round(time.time() - started, 1),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "sampled_at": utcnow(),
    }
    (args.out_dir / "SAMPLES_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    log(f"DONE n_sampled={len(rows)} finish={finish} empty={empty} think={think_tags}")


if __name__ == "__main__":
    main()
