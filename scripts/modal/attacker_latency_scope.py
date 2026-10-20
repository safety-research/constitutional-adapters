"""Narrow bf16/FP8 and prefix-cache measurement for the production attacker.

One H200, three sequential vLLM standups: bf16/cache-on, bf16/cache-off,
and FP8/cache-on.  It reads recorded production attacker calls from the private
``twosi-fits`` Modal volume and writes aggregate metrics only; no attack wave,
target call, or judge is involved.

Run with an explicit Modal app invocation name (Modal 1.5 spells this ``--name``):

  ~/venvs/modal/bin/python -m modal run --name attacker-latency-scope-<stamp> \
    scripts/modal/attacker_latency_scope.py --run scope-<stamp>

``--mode cache`` runs the prefix-cache EQUIVALENCE probe instead (bf16, cache
off vs on-warm against an off-vs-off-repeat noise floor; ``--with-fp8`` adds an
FP8/cache-on leg read the same way, so bf16-on vs FP8-on isolates the weight
precision at the production cache setting; see ``run_cache_equivalence``):

  ~/venvs/modal/bin/python -m modal run --name attacker-cache-equivalence-<stamp> \
    scripts/modal/attacker_latency_scope.py --mode cache --run cacheq-<stamp> [--with-fp8]

Readout discipline (both modes): two top-k readouts are compared with
``sisteer.eval.attacks.distribution_agreement`` -- per position, top-k
MEMBERSHIP agreement, per-token logprob deltas over the common tokens, the
captured mass on each side, and the argmax; the coarsened top-k KL is reported
only as the LOWER bound it is.  The RAW per-position readouts are written to the
result file so the analysis can be re-derived on the grading host.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import modal

APP_NAME = "attacker-latency-scope"
app = modal.App(APP_NAME)
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"
MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
SNAPSHOT = "27502c8717fd5a2f8c0c77188c10c243fd4f672e"
MODEL_PATH = (
    f"{VOL_PATH}/grokmt/hf_attacker/models--huihui-ai--"
    f"Huihui-Qwen3.6-27B-abliterated/snapshots/{SNAPSHOT}"
)
RECORDED_ROOT = Path(f"{VOL_PATH}/grokmt/gen/grokmt-0902T2138")
OUT_ROOT = Path(f"{VOL_PATH}/attacker_latency_scoping")
MAX_MODEL_LEN = 49152
MAX_NUM_SEQS = 64
MAX_BATCHED_TOKENS = 131072
N_PROMPTS_DEFAULT = 200
GREEDY_MAX_TOKENS = 128
KL_PROMPTS = 5
KL_CONTINUATION_TOKENS = 32
KL_TOP_K = 64
PREFIX_MAX_TOKENS = 128
PREFIX_REPEATS = 3
PREFIX_BATCHES = (1, 8, 32)

_HERE = Path(__file__).resolve()
# Modal imports the source as /root/attacker_latency_scope.py in the container;
# the client-side image build sees its real <repo>/scripts/modal path.
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/repo")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431",
        "torch==2.11.0+cu129",
        "torchvision==0.26.0+cu129",
        "torchaudio==2.11.0+cu129",
        "aiohttp==3.13.3",
        "huggingface_hub==1.5.0",
        "transformers==5.14.1",
        "tokenizers==0.22.2",
        "ninja==1.13.0",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    .run_commands(
        "python -c \"import torch, vllm, transformers, tokenizers; "
        "assert torch.__version__ == '2.11.0+cu129'; "
        "assert vllm.__version__.startswith('0.25.1'); "
        "assert transformers.__version__ == '5.14.1'; "
        "assert tokenizers.__version__ == '0.22.2'; "
        "print('IMAGE OK', vllm.__version__, torch.__version__)\"",
        "which vllm",
    )
    .env({
        "HF_HOME": f"{VOL_PATH}/hf",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": "/repo/src",
        "VLLM_CACHE_ROOT": f"{VOL_PATH}/vllm_cache/attacker_latency_vllm0251",
    })
    .add_local_file(
        str(REPO / "src/sisteer/eval/attacks/attacker_serving.py"),
        remote_path="/repo/src/sisteer/eval/attacks/attacker_serving.py",
    )
    .add_local_file(
        str(REPO / "configs/attacker_serving.json"),
        remote_path="/repo/configs/attacker_serving.json",
    )
    .add_local_file(
        str(REPO / "src/sisteer/eval/attacks/distribution_agreement.py"),
        remote_path="/repo/src/sisteer/eval/attacks/distribution_agreement.py",
    )
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p90": round(_percentile(values, 0.90), 4),
    }


def _chat_token_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Normalize transformers 5 BatchEncoding chat-template output to token ids."""
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError(f"expected one rendered chat, got batch size {len(encoded)}")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(isinstance(x, int) for x in encoded):
        raise RuntimeError(f"chat template returned unsupported token container: {type(encoded)!r}")
    return encoded


def _wait_health(port: int, proc: subprocess.Popen, log_path: Path, timeout: float = 1800) -> float:
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-6000:] if log_path.exists() else ""
            raise RuntimeError(f"vLLM died before health (rc={proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if 200 <= response.status < 300:
                    return time.time() - start
        except Exception:  # noqa: BLE001 - readiness polling
            pass
        time.sleep(3)
    raise RuntimeError(f"vLLM not healthy after {timeout}s; log={log_path}")


def _start_server(quantization: str, prefix_caching: bool, log_path: Path):
    from sisteer.eval.attacks.attacker_serving import AttackerServingOptions

    port = _free_port()
    options = AttackerServingOptions(quantization, prefix_caching)
    command = [
        "vllm", "serve", MODEL_PATH,
        "--served-model-name", MODEL,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_BATCHED_TOKENS),
        "--max-logprobs", str(KL_TOP_K),
        *options.vllm_argv(),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log = log_path.open("w")
    print("[standup]", " ".join(command), flush=True)
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    ready_s = _wait_health(port, process, log_path)
    from sisteer.eval.attacks.attacker_serving import parse_engine_log, verify_engine

    engine = parse_engine_log(log_path.read_text(errors="replace"))
    problems = verify_engine(options, engine)
    if problems:
        _stop_server(process)
        raise RuntimeError("unverified vLLM standup: " + "; ".join(problems))
    return process, port, ready_s, command, engine


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=90)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(8)


def _attack_from_path(path: Path) -> str:
    for attack in ("pair", "crescendo", "actorattack"):
        if attack in path.parts:
            return attack
    return "unknown"


def _load_recorded_prompts(n_prompts: int) -> tuple[list[dict], list[dict], dict]:
    candidates: list[dict] = []
    files = sorted(RECORDED_ROOT.glob("qwen/**/attacker_calls.jsonl"))
    fallback = False
    if not files:
        files = sorted(RECORDED_ROOT.glob("grok/**/attacker_calls.jsonl"))
        fallback = True
    for path in files:
        attack = _attack_from_path(path)
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("role") != "attacker" or not isinstance(row.get("messages"), list):
                continue
            params = row.get("params") or {}
            candidates.append({
                "attack": attack,
                "messages": row["messages"],
                "temperature": float(params.get("temperature", 0.5 if attack == "crescendo" else 1.0)),
                "top_p": float(params.get("top_p", 0.9)),
                "max_tokens": int(params.get("max_new_tokens", 1024)),
                "source": str(path.relative_to(Path(VOL_PATH))),
                "line": line_number,
            })
    # Exact-message de-duplication, then deterministic round-robin across attacks.
    unique: dict[str, dict] = {}
    for row in candidates:
        key = json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
        unique.setdefault(key, row)
    by_attack: dict[str, list[dict]] = defaultdict(list)
    for row in unique.values():
        by_attack[row["attack"]].append(row)
    selected: list[dict] = []
    order = [a for a in ("pair", "crescendo", "actorattack", "unknown") if by_attack[a]]
    cursor = 0
    while len(selected) < n_prompts and any(by_attack.values()):
        attack = order[cursor % len(order)]
        if by_attack[attack]:
            selected.append(by_attack[attack].pop(0))
        cursor += 1
        if cursor > 10 * max(1, len(unique)):
            break
    if len(selected) < n_prompts:
        raise RuntimeError(f"recorded prompt source has only {len(selected)} unique calls; need {n_prompts}")
    selected = selected[:n_prompts]
    source_counts = Counter(row["source"] for row in selected)
    return selected, list(unique.values()), {
        "root": str(RECORDED_ROOT),
        "used_grok_prompt_fallback": fallback,
        "n_candidate_rows": len(candidates),
        "n_unique_messages": len(unique),
        "n_selected": len(selected),
        "selected_by_attack": dict(Counter(row["attack"] for row in selected)),
        "selected_source_files": dict(source_counts),
    }


async def _post(session, port: int, path: str, payload: dict) -> tuple[dict, float]:
    start = time.perf_counter()
    async with session.post(f"http://127.0.0.1:{port}{path}", json=payload) as response:
        text = await response.text()
        elapsed = time.perf_counter() - start
        if response.status >= 300:
            raise RuntimeError(f"HTTP {response.status} {path}: {text[:1000]}")
        return json.loads(text), elapsed


async def _chat_many(port: int, prompts: list[dict], *, greedy: bool, concurrency: int) -> list[dict]:
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1200)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(index: int, row: dict) -> dict:
            payload = {
                "model": MODEL,
                "messages": row["messages"],
                "temperature": 0.0 if greedy else row["temperature"],
                "top_p": 1.0 if greedy else row["top_p"],
                "max_tokens": GREEDY_MAX_TOKENS if greedy else row["max_tokens"],
                "seed": 730000 + index,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            async with semaphore:
                data, elapsed = await _post(session, port, "/v1/chat/completions", payload)
            choice = data["choices"][0]
            return {
                "index": index,
                "text": (choice["message"].get("content") or "").strip(),
                "latency_s": elapsed,
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage") or {},
            }

        return await asyncio.gather(*(one(i, row) for i, row in enumerate(prompts)))


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _schema_result(attack: str, messages: list[dict], text: str) -> tuple[bool, str, dict[str, str]]:
    if attack in {"pair", "crescendo"}:
        try:
            value = json.loads(_strip_fence(text))
        except json.JSONDecodeError:
            return False, "adversarial_chat", {}
        required = {"next_message", "rationale", "last_response_summary"}
        valid = isinstance(value, dict) and set(value) == required and all(
            isinstance(value.get(key), str) for key in required
        )
        return valid, "adversarial_chat", ({key: value[key] for key in required} if valid else {})
    joined = "\n".join(str(m.get("content", "")) for m in messages)
    if '"actors"' in joined and "identify" in joined.lower():
        try:
            value = json.loads(_strip_fence(text))
        except json.JSONDecodeError:
            return False, "actor_discovery", {}
        actors = value.get("actors") if isinstance(value, dict) else None
        valid = isinstance(actors, list) and bool(actors) and all(
            isinstance(actor, dict)
            and isinstance(actor.get("name"), str)
            and isinstance(actor.get("queries"), list)
            and all(isinstance(query, str) for query in actor["queries"])
            for actor in actors
        )
        if not valid:
            return False, "actor_discovery", {}
        names = " ".join(actor["name"] for actor in actors)
        queries = " ".join(query for actor in actors for query in actor["queries"])
        return True, "actor_discovery", {"actor_names": names, "queries": queries}
    return bool(text.strip()), "actor_rewrite", {"rewritten_question": text.strip()}


_REFUSAL = re.compile(
    r"\b(?:i (?:cannot|can't|won't|will not)|unable to (?:help|assist|comply)|"
    r"cannot (?:help|assist|provide)|decline to|not able to assist|against (?:my|the) (?:policy|guidelines))\b",
    re.I,
)


def _mode_metrics(tokenizer, prompts: list[dict], sampled: list[dict]) -> dict:
    schema_counts: Counter = Counter()
    finish = Counter()
    refusals = 0
    field_tokens: Counter = Counter()
    output_tokens = 0
    latencies: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    per_attack: dict[str, dict] = {}
    attack_rows: dict[str, list[int]] = defaultdict(list)
    attack_schema: dict[str, Counter] = defaultdict(Counter)
    attack_refusals: Counter = Counter()

    for row, result in zip(prompts, sampled):
        valid, schema, fields = _schema_result(row["attack"], row["messages"], result["text"])
        schema_counts[(schema, valid)] += 1
        attack_schema[row["attack"]][valid] += 1
        finish[result["finish_reason"]] += 1
        is_refusal = bool(_REFUSAL.search(result["text"]))
        refusals += is_refusal
        attack_refusals[row["attack"]] += is_refusal
        ids = tokenizer.encode(result["text"], add_special_tokens=False)
        output_tokens += len(ids)
        for key, value in fields.items():
            field_tokens[(schema, key)] += len(tokenizer.encode(value, add_special_tokens=False))
        latencies.append(result["latency_s"])
        prompt_tokens.append(int(result["usage"].get("prompt_tokens") or 0))
        completion_tokens.append(int(result["usage"].get("completion_tokens") or 0))
        attack_rows[row["attack"]].append(len(latencies) - 1)

    for attack, indices in attack_rows.items():
        schema_n = attack_schema[attack][True] + attack_schema[attack][False]
        schema_ok = attack_schema[attack][True]
        per_attack[attack] = {
            "n": len(indices),
            "latency_s": _summary([latencies[i] for i in indices]),
            "prompt_tokens_mean": round(statistics.mean(prompt_tokens[i] for i in indices), 2),
            "completion_tokens_mean": round(statistics.mean(completion_tokens[i] for i in indices), 2),
            "schema_valid_rate": round(schema_ok / schema_n, 4) if schema_n else None,
            "refusal_rate": round(attack_refusals[attack] / len(indices), 4),
        }
    schema_report = {}
    for schema in sorted({key[0] for key in schema_counts}):
        total = schema_counts[(schema, True)] + schema_counts[(schema, False)]
        schema_report[schema] = {
            "n": total,
            "valid": schema_counts[(schema, True)],
            "valid_rate": round(schema_counts[(schema, True)] / total, 4) if total else None,
        }
    fields_report = defaultdict(dict)
    for (schema, key), count in field_tokens.items():
        schema_total = sum(v for (s, _), v in field_tokens.items() if s == schema)
        fields_report[schema][key] = {
            "tokens": count,
            "fraction_of_parsed_field_tokens": round(count / schema_total, 4) if schema_total else None,
            "fraction_of_all_output_tokens": round(count / output_tokens, 4) if output_tokens else None,
        }
    return {
        "n": len(sampled),
        "latency_s": _summary(latencies),
        "prompt_tokens": _summary([float(x) for x in prompt_tokens]),
        "completion_tokens": _summary([float(x) for x in completion_tokens]),
        "finish_reason": dict(finish),
        "schema": schema_report,
        "refusal_count": refusals,
        "refusal_rate": round(refusals / len(sampled), 4),
        "field_token_fractions": dict(fields_report),
        "per_attack": per_attack,
    }


def _greedy_agreement(tokenizer, bf16: list[dict], fp8: list[dict]) -> dict:
    if len(bf16) != len(fp8):
        raise RuntimeError(f"greedy legs have {len(bf16)} vs {len(fp8)} results; refusing to "
                           "compare the overlap of a partial leg")
    exact = 0
    common_prefix = 0
    aligned_equal = 0
    aligned_total = 0
    edit_ratios: list[float] = []
    import difflib

    for a, b in zip(bf16, fp8):
        left = tokenizer.encode(a["text"], add_special_tokens=False)
        right = tokenizer.encode(b["text"], add_special_tokens=False)
        exact += left == right
        for x, y in zip(left, right):
            if x != y:
                break
            common_prefix += 1
        aligned_equal += sum(x == y for x, y in zip(left, right))
        aligned_total += max(len(left), len(right))
        edit_ratios.append(difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio())
    n = len(bf16)
    total_bf16 = sum(len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in bf16)
    return {
        "n": n,
        "exact_sequence_rate": round(exact / n, 4),
        "common_prefix_tokens_over_bf16_tokens": round(common_prefix / total_bf16, 4) if total_bf16 else None,
        "same_token_at_aligned_position_rate": round(aligned_equal / aligned_total, 4) if aligned_total else None,
        "mean_token_sequence_match_ratio": round(statistics.mean(edit_ratios), 4),
        "greedy_max_tokens": GREEDY_MAX_TOKENS,
    }


def _prefix_pairs(prompts: list[dict]) -> list[tuple[dict, dict]]:
    by_messages = {
        json.dumps(row["messages"], sort_keys=True, ensure_ascii=False): row for row in prompts
    }
    pairs = []
    for full in prompts:
        messages = full["messages"]
        for prefix_len in range(len(messages) - 2, 1, -2):
            key = json.dumps(messages[:prefix_len], sort_keys=True, ensure_ascii=False)
            if key in by_messages:
                pairs.append((by_messages[key], full))
                break
    # De-duplicate full calls, then interleave attack families. Sorted source
    # paths put every Crescendo call before PAIR; returning insertion order
    # would make the fixed-size latency sample Crescendo-only.
    seen = set()
    by_attack: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for parent, full in pairs:
        key = json.dumps(full["messages"], sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            by_attack[full["attack"]].append((parent, full))
    order = [name for name in ("pair", "crescendo", "actorattack", "unknown")
             if by_attack[name]]
    out: list[tuple[dict, dict]] = []
    cursor = 0
    while any(by_attack.values()):
        attack = order[cursor % len(order)]
        if by_attack[attack]:
            out.append(by_attack[attack].pop(0))
        cursor += 1
    return out


async def _prefix_benchmark(port: int, pairs: list[tuple[dict, dict]]) -> dict:
    import aiohttp

    needed = PREFIX_REPEATS * sum(PREFIX_BATCHES)
    if len(pairs) < needed:
        raise RuntimeError(f"prefix benchmark needs {needed} parent/full pairs, found {len(pairs)}")
    cursor = 0
    report = {}
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch in PREFIX_BATCHES:
            rep_latencies = []
            prompt_tokens = []
            for rep in range(PREFIX_REPEATS):
                chunk = pairs[cursor:cursor + batch]
                cursor += batch
                warm_payloads = []
                full_payloads = []
                for i, (parent, full) in enumerate(chunk):
                    base = {
                        "model": MODEL,
                        "temperature": full["temperature"],
                        "top_p": full["top_p"],
                        "seed": 910000 + cursor + i,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    warm_payloads.append(base | {"messages": parent["messages"], "max_tokens": 1})
                    full_payloads.append(base | {"messages": full["messages"], "max_tokens": PREFIX_MAX_TOKENS})
                await asyncio.gather(*(_post(session, port, "/v1/chat/completions", p) for p in warm_payloads))
                start = time.perf_counter()
                results = await asyncio.gather(*(_post(session, port, "/v1/chat/completions", p) for p in full_payloads))
                wall = time.perf_counter() - start
                rep_latencies.append(wall)
                prompt_tokens.extend(int((data.get("usage") or {}).get("prompt_tokens") or 0) for data, _ in results)
            report[str(batch)] = {
                "repeats": PREFIX_REPEATS,
                "batch_wall_s": _summary(rep_latencies),
                "per_request_wall_s": _summary([x / batch for x in rep_latencies]),
                "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 2),
                "max_tokens": PREFIX_MAX_TOKENS,
                "method": "warm exact shorter recorded conversation, then time its longer recorded continuation",
            }
    return report


async def _teacher_forced_distributions(port: int, tokenizer, prompts: list[dict],
                                        bf16_greedy: list[dict]) -> list[list[dict[int, float]]]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=1200)
    rows = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for prompt, generated in zip(prompts[:KL_PROMPTS], bf16_greedy[:KL_PROMPTS]):
            base_ids = _chat_token_ids(tokenizer, prompt["messages"])
            continuation = tokenizer.encode(generated["text"], add_special_tokens=False)[:KL_CONTINUATION_TOKENS]
            payload = {
                "model": MODEL,
                "prompt": list(base_ids) + list(continuation),
                "add_special_tokens": False,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": KL_TOP_K,
                "return_token_ids": True,
            }
            data, _ = await _post(session, port, "/v1/completions", payload)
            distributions = data["choices"][0]["prompt_logprobs"]
            sliced = distributions[len(base_ids):len(base_ids) + len(continuation)]
            compact = []
            for dist in sliced:
                item = {}
                for token_id, value in (dist or {}).items():
                    item[int(token_id)] = float(value["logprob"] if isinstance(value, dict) else value)
                compact.append(item)
            rows.append(compact)
    return rows


def _teacher_forced_kl(bf16: list[list[dict[int, float]]], fp8: list[list[dict[int, float]]]) -> dict:
    """Per-position agreement of the two teacher-forced top-k readouts (see
    ``distribution_agreement``): membership, common-token deltas, captured
    mass, argmax, and the coarsened KL labelled as the lower bound it is."""
    from sisteer.eval.attacks.distribution_agreement import compare

    out = compare(bf16, fp8)          # raises AlignmentError on a shape mismatch: fail loud
    rows = out["summary"]["rows"]
    out["summary"].update({
        "n_prompts": len(bf16),
        # a prompt whose continuation had no token was never compared: listed, not dropped
        "n_prompts_compared": rows["n_rows_compared"],
        "n_prompts_empty": rows["n_rows_empty"],
        "top_k": KL_TOP_K,
        "continuation_tokens_per_prompt_cap": KL_CONTINUATION_TOKENS,
        "readout": "prompt_logprobs on the bf16 greedy continuation (prefill-side)",
    })
    return out


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16,
              memory=131072, timeout=3600)
def run_scope(run_id: str, n_prompts: int = N_PROMPTS_DEFAULT) -> dict:
    from transformers import AutoTokenizer
    from sisteer.eval.attacks.attacker_serving import ATTACKER_ENABLE_THINKING, parse_engine_log

    if ATTACKER_ENABLE_THINKING:
        raise RuntimeError("measurement requires production attacker thinking mode to remain disabled")
    if not Path(MODEL_PATH).is_dir():
        raise RuntimeError(f"attacker snapshot missing: {MODEL_PATH}")
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts, all_prompts, source = _load_recorded_prompts(n_prompts)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    source["prompt_tokens_with_attacker_tokenizer"] = _summary([
        float(len(_chat_token_ids(tokenizer, row["messages"]))) for row in prompts
    ])
    source["thinking_mode"] = "disabled (enable_thinking=False)"
    source["attacker_model"] = MODEL
    source["snapshot"] = SNAPSHOT

    pairs = _prefix_pairs(all_prompts)
    # Exact tokenizer-level reusable-prefix estimate from recorded growing
    # conversations. Prefix caching stores whole blocks, so this is an upper
    # bound before block alignment; the measured on/off delta is the referee.
    reusable = []
    for parent, full in pairs:
        parent_ids = _chat_token_ids(tokenizer, parent["messages"])
        full_ids = _chat_token_ids(tokenizer, full["messages"])
        common = 0
        for left, right in zip(parent_ids, full_ids):
            if left != right:
                break
            common += 1
        reusable.append((common, len(full_ids)))
    source["growing_conversation_prefix_profile"] = {
        "n_parent_full_pairs": len(reusable),
        "weighted_reusable_token_fraction_upper_bound": (
            round(sum(a for a, _ in reusable) / sum(b for _, b in reusable), 4) if reusable else None),
        "mean_reusable_tokens": (round(statistics.mean(a for a, _ in reusable), 2) if reusable else None),
        "mean_full_prompt_tokens": (round(statistics.mean(b for _, b in reusable), 2) if reusable else None),
        "source": "exact parent/full message prefixes in recorded PyRIT attacker calls; attacker tokenizer",
    }
    result = {
        "run_id": run_id,
        "app_name": APP_NAME,
        "model": MODEL,
        "snapshot": SNAPSHOT,
        "vllm_pin": "0.25.1+cu129",
        "gpu": "H200",
        "source": source,
        "constants": {
            "n_prompts": n_prompts,
            "greedy_max_tokens": GREEDY_MAX_TOKENS,
            "kl_prompts": KL_PROMPTS,
            "kl_continuation_tokens": KL_CONTINUATION_TOKENS,
            "kl_top_k": KL_TOP_K,
            "prefix_batches": PREFIX_BATCHES,
            "prefix_repeats": PREFIX_REPEATS,
            "prefix_max_tokens": PREFIX_MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": MAX_BATCHED_TOKENS,
        },
        "thinking_mode": "disabled",
        "kv_cache_dtype": "auto (bf16 model dtype; no FP8 KV cache)",
        "timeline": {},
    }
    start = time.time()

    def stamp(name):
        result["timeline"][name] = round(time.time() - start, 2)
        print(f"[t+{result['timeline'][name]:.1f}s] {name}", flush=True)

    bf16_sampled = bf16_greedy = bf16_dist = None
    prefix = {}
    modes = {}
    for quantization, cache_on in (("none", True), ("none", False), ("fp8", True)):
        key = f"{quantization}_{'cache_on' if cache_on else 'cache_off'}"
        log_path = out_dir / f"vllm_{key}.log"
        proc = None
        try:
            proc, port, ready_s, command, engine = _start_server(quantization, cache_on, log_path)
            stamp(f"{key}_ready")
            modes.setdefault(key, {})["standup"] = {
                "ready_s": round(ready_s, 3),
                "command": command,
                "engine": engine,
            }
            if quantization == "none":
                prefix[key] = asyncio.run(_prefix_benchmark(port, pairs))
                stamp(f"{key}_prefix_done")
            if cache_on:
                sampled = asyncio.run(_chat_many(port, prompts, greedy=False, concurrency=32))
                greedy = asyncio.run(_chat_many(port, prompts, greedy=True, concurrency=32))
                metrics = _mode_metrics(tokenizer, prompts, sampled)
                distributions = asyncio.run(
                    _teacher_forced_distributions(port, tokenizer, prompts, bf16_greedy or greedy)
                )
                modes[key]["sampled"] = metrics
                if quantization == "none":
                    bf16_sampled, bf16_greedy, bf16_dist = sampled, greedy, distributions
                else:
                    modes[key]["greedy_agreement_vs_bf16"] = _greedy_agreement(
                        tokenizer, bf16_greedy, greedy
                    )
                    modes[key]["teacher_forced"] = _teacher_forced_kl(
                        bf16_dist, distributions
                    )
                    modes[key]["teacher_forced_raw_readouts"] = {
                        "keys": "token ids (attacker tokenizer)",
                        "bf16": [[{str(k): v for k, v in d.items()} for d in row] for row in bf16_dist],
                        "fp8": [[{str(k): v for k, v in d.items()} for d in row] for row in distributions],
                    }
                stamp(f"{key}_divergence_done")
            # Capture final cache hit-rate line after requests, not only standup.
            modes[key]["standup"]["engine_after_requests"] = parse_engine_log(
                log_path.read_text(errors="replace")
            )
        finally:
            if proc is not None:
                _stop_server(proc)
            stamp(f"{key}_stopped")

    delta = {}
    for batch in PREFIX_BATCHES:
        on = prefix["none_cache_on"][str(batch)]["batch_wall_s"]["mean"]
        off = prefix["none_cache_off"][str(batch)]["batch_wall_s"]["mean"]
        delta[str(batch)] = {
            "cache_on_batch_wall_s": on,
            "cache_off_batch_wall_s": off,
            "saved_s": round(off - on, 4),
            "speedup_x": round(off / on, 4) if on else None,
            "latency_reduction_fraction": round((off - on) / off, 4) if off else None,
        }
    result["modes"] = modes
    result["prefix_cache"] = {
        "pairs_available": len(pairs),
        "cache_on": prefix["none_cache_on"],
        "cache_off": prefix["none_cache_off"],
        "delta": delta,
    }
    result["total_wall_s"] = round(time.time() - start, 2)
    result_path = out_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    VOL.commit()
    print("[result]", result_path, flush=True)
    print(json.dumps({
        "run_id": run_id,
        "total_wall_s": result["total_wall_s"],
        "thinking_mode": result["thinking_mode"],
        "prefix_delta": delta,
        "bf16": modes["none_cache_on"]["sampled"],
        "fp8": modes["fp8_cache_on"]["sampled"],
        "greedy": modes["fp8_cache_on"]["greedy_agreement_vs_bf16"],
        "teacher_forced": modes["fp8_cache_on"]["teacher_forced"]["summary"],
    }, indent=1), flush=True)
    return {"result_path": str(result_path), "summary": result}


# --------------------------------------------------------------------------- #
# Cache-equivalence probe: does the (experimental-for-hybrids) align-mode      #
# prefix cache change what the attacker computes?                              #
# --------------------------------------------------------------------------- #

CACHE_PAIRS_DEFAULT = 16
CACHE_POSITIONS = 16
CACHE_TOP_K = KL_TOP_K
# Hard bound on the probe (45 min): the cold standup is ~11 min (weights from the
# volume), each further standup ~3-5 min, each reading ~2 min; three legs fit
# with margin, and the staged checkpoints keep whatever finished on a timeout.
CACHE_PROBE_TIMEOUT_S = 2700


async def _next_token_dists(port: int, items: list[tuple[list[int], list[int]]],
                            *, concurrency: int = 32) -> list[list[dict[str, float]]]:
    """For each ``(prefix_ids, continuation_ids)``: the top-k NEXT-token
    distribution at every continuation position ``t``, obtained by generating
    ONE token from ``prefix + continuation[:t]``. Each request re-reads the whole
    prefix, so on a prefix-caching server it exercises the cached prefix state
    on the real decode path (a ``prompt_logprobs`` request would not certify
    that: it is a prefill-side readout). Keys are vLLM's token strings, shared
    across legs because the tokenizer is."""
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def at(prefix, continuation, t):
            payload = {
                "model": MODEL,
                "prompt": list(prefix) + list(continuation[:t]),
                "add_special_tokens": False,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": CACHE_TOP_K,
            }
            async with semaphore:
                data, _ = await _post(session, port, "/v1/completions", payload)
            top = data["choices"][0]["logprobs"]["top_logprobs"][0]
            return {str(k): float(v) for k, v in top.items()}

        async def one(prefix, continuation):
            return list(await asyncio.gather(
                *(at(prefix, continuation, t) for t in range(len(continuation)))))

        return await asyncio.gather(*(one(p, c) for p, c in items))


def _coarsened_kl(left_rows, right_rows) -> dict:
    """Per-position agreement of two next-token readouts (``distribution_agreement``):
    membership identity, common-token logprob deltas, captured mass, argmax, and
    the coarsened top-k KL reported as the LOWER bound it is.  Returns
    ``{"summary", "positions"}``; the caller keeps the raw readouts too."""
    from sisteer.eval.attacks.distribution_agreement import AlignmentError, compare, inconclusive_summary

    try:
        return compare(left_rows, right_rows)
    except AlignmentError as exc:
        # a leg that returned fewer rows/positions cannot certify anything: the
        # contrast is recorded as inconclusive with the reason, never as the
        # verdict of whatever overlap survived
        return {"summary": inconclusive_summary(str(exc), left_rows=len(left_rows),
                                                right_rows=len(right_rows)),
                "positions": []}


async def _warm_parents(port: int, pairs: list[tuple[dict, dict]], *, concurrency: int = 8) -> None:
    """Populate the cache the way production does: the SHORTER recorded
    conversation (the parent turn) is requested first, one token, so its
    blocks are resident when the longer continuation arrives."""
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(parent):
            payload = {"model": MODEL, "messages": parent["messages"], "max_tokens": 1,
                       "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
            async with semaphore:
                await _post(session, port, "/v1/chat/completions", payload)

        await asyncio.gather(*(one(parent) for parent, _ in pairs))


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16,
              memory=131072, timeout=CACHE_PROBE_TIMEOUT_S)
def run_cache_equivalence(run_id: str, n_pairs: int = CACHE_PAIRS_DEFAULT,
                          include_fp8: bool = False) -> dict:
    """One H200, two standups (bf16, prefix cache OFF, then ON with
    ``--mamba-cache-mode align``) plus, with ``include_fp8``, a third (FP8
    weights, cache ON) read the same way. For ``n_pairs`` recorded growing-turn pairs:

    1. OFF: greedy 128-token continuation of the longer conversation (the
       reference text) and the top-k next-token distributions at its first
       ``CACHE_POSITIONS`` positions -- TWICE, so OFF-vs-OFF is the numerical
       noise floor (batch composition alone moves bits at batch > 1);
    2. ON: the parent turn is requested first (production order), then the
       same greedy decode and the same per-position distributions on the
       longer conversation, i.e. through the cached prefix state.

    3. FP8 (optional): the same warm-then-read sequence on the FP8/cache-on
       engine; bf16-ON vs FP8-ON isolates the weight precision at the
       production cache setting (OFF vs FP8-ON is reported too).

    Reported per contrast: the ``distribution_agreement`` summary (membership,
    deltas, captured mass, argmax, verdict; coarsened KL as a lower bound) and
    greedy agreement; plus the RAW per-position readouts, so the analysis can
    be re-derived.  The result file is (re)written after every leg and the
    volume committed, so a timeout still leaves the legs that finished."""
    from transformers import AutoTokenizer
    from sisteer.eval.attacks.attacker_serving import parse_engine_log

    if not Path(MODEL_PATH).is_dir():
        raise RuntimeError(f"attacker snapshot missing: {MODEL_PATH}")
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _, all_prompts, source = _load_recorded_prompts(N_PROMPTS_DEFAULT)
    pairs = _prefix_pairs(all_prompts)[:n_pairs]
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only {len(pairs)} growing-turn pairs available; need {n_pairs}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    full_rows = [full for _, full in pairs]
    full_ids = [_chat_token_ids(tokenizer, full["messages"]) for full in full_rows]
    reusable = []
    for (parent, _), ids in zip(pairs, full_ids):
        parent_ids = _chat_token_ids(tokenizer, parent["messages"])
        common = 0
        for a, b in zip(parent_ids, ids):
            if a != b:
                break
            common += 1
        reusable.append((common, len(ids)))
    result = {
        "run_id": run_id, "app_name": APP_NAME, "model": MODEL, "snapshot": SNAPSHOT,
        "vllm_pin": "0.25.1+cu129", "gpu": "H200", "quantization": "none (bf16)",
        "constants": {"n_pairs": n_pairs, "positions_per_pair": CACHE_POSITIONS,
                      "top_k": CACHE_TOP_K, "greedy_max_tokens": GREEDY_MAX_TOKENS,
                      "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
                      "max_num_batched_tokens": MAX_BATCHED_TOKENS,
                      "include_fp8": include_fp8, "timeout_s": CACHE_PROBE_TIMEOUT_S},
        "readout": ("one greedy token from prefix + continuation[:t] via /v1/completions "
                    "with logprobs=top_k; keys are vLLM token strings"),
        "status": "running",
        "pairs": {
            "by_attack": dict(Counter(full["attack"] for full in full_rows)),
            "mean_full_prompt_tokens": round(statistics.mean(b for _, b in reusable), 2),
            "weighted_reusable_token_fraction_upper_bound": round(
                sum(a for a, _ in reusable) / sum(b for _, b in reusable), 4),
        },
        "source": {k: source[k] for k in ("root", "n_candidate_rows", "n_unique_messages")},
        "timeline": {},
    }
    start = time.time()

    result_path = out_dir / "cache_equivalence.json"

    def stamp(name):
        result["timeline"][name] = round(time.time() - start, 2)
        print(f"[t+{result['timeline'][name]:.1f}s] {name}", flush=True)

    def checkpoint():
        result["total_wall_s"] = round(time.time() - start, 2)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        VOL.commit()

    legs: dict = {}
    readouts: dict = {}
    contrasts: dict = {}
    # ---- OFF: reference text + two readings of the distributions --------------
    proc = None
    try:
        proc, port, ready_s, command, engine = _start_server("none", False, out_dir / "vllm_cache_off.log")
        stamp("off_ready")
        off_greedy_1 = asyncio.run(_chat_many(port, full_rows, greedy=True, concurrency=32))
        continuations = [
            tokenizer.encode(r["text"], add_special_tokens=False)[:CACHE_POSITIONS] for r in off_greedy_1
        ]
        items = list(zip(full_ids, continuations))
        off_dists_1 = asyncio.run(_next_token_dists(port, items))
        stamp("off_reading_1_done")
        off_greedy_2 = asyncio.run(_chat_many(port, full_rows, greedy=True, concurrency=32))
        off_dists_2 = asyncio.run(_next_token_dists(port, items))
        stamp("off_reading_2_done")
        legs["cache_off"] = {"ready_s": round(ready_s, 3), "command": command, "engine": engine,
                             "engine_after_requests": parse_engine_log(
                                 (out_dir / "vllm_cache_off.log").read_text(errors="replace"))}
        readouts["bf16_off_1"], readouts["bf16_off_2"] = off_dists_1, off_dists_2
        contrasts["noise_floor_off_vs_off_repeat"] = {
            "distributions": _coarsened_kl(off_dists_1, off_dists_2),
            "greedy_agreement": _greedy_agreement(tokenizer, off_greedy_1, off_greedy_2),
        }
        result.update(legs=legs, contrasts=contrasts, raw_readouts=readouts)
        checkpoint()
    finally:
        if proc is not None:
            _stop_server(proc)
        stamp("off_stopped")
    # ---- ON: warm the parents, then read through the cached prefix ------------
    proc = None
    try:
        proc, port, ready_s, command, engine = _start_server("none", True, out_dir / "vllm_cache_on.log")
        stamp("on_ready")
        asyncio.run(_warm_parents(port, pairs))
        stamp("on_parents_warm")
        on_greedy = asyncio.run(_chat_many(port, full_rows, greedy=True, concurrency=32))
        on_dists = asyncio.run(_next_token_dists(port, items))
        stamp("on_reading_done")
        legs["cache_on"] = {"ready_s": round(ready_s, 3), "command": command, "engine": engine,
                            "engine_after_requests": parse_engine_log(
                                (out_dir / "vllm_cache_on.log").read_text(errors="replace"))}
        readouts["bf16_on_warm"] = on_dists
        contrasts["cache_effect_off_vs_on_warm"] = {
            "distributions": _coarsened_kl(off_dists_1, on_dists),
            "greedy_agreement": _greedy_agreement(tokenizer, off_greedy_1, on_greedy),
        }
        checkpoint()
    finally:
        if proc is not None:
            _stop_server(proc)
        stamp("on_stopped")
    # ---- FP8 (optional): same warm-then-read through the FP8/cache-on engine ---
    if include_fp8:
        proc = None
        try:
            proc, port, ready_s, command, engine = _start_server("fp8", True, out_dir / "vllm_fp8_cache_on.log")
            stamp("fp8_ready")
            asyncio.run(_warm_parents(port, pairs))
            stamp("fp8_parents_warm")
            fp8_greedy = asyncio.run(_chat_many(port, full_rows, greedy=True, concurrency=32))
            fp8_dists = asyncio.run(_next_token_dists(port, items))
            stamp("fp8_reading_done")
            legs["fp8_cache_on"] = {"ready_s": round(ready_s, 3), "command": command, "engine": engine,
                                    "engine_after_requests": parse_engine_log(
                                        (out_dir / "vllm_fp8_cache_on.log").read_text(errors="replace"))}
            readouts["fp8_on_warm"] = fp8_dists
            contrasts["fp8_effect_bf16_on_vs_fp8_on"] = {
                "distributions": _coarsened_kl(on_dists, fp8_dists),
                "greedy_agreement": _greedy_agreement(tokenizer, on_greedy, fp8_greedy),
            }
            contrasts["fp8_vs_reference_off_vs_fp8_on"] = {
                "distributions": _coarsened_kl(off_dists_1, fp8_dists),
                "greedy_agreement": _greedy_agreement(tokenizer, off_greedy_1, fp8_greedy),
            }
            checkpoint()
        finally:
            if proc is not None:
                _stop_server(proc)
            stamp("fp8_stopped")

    result["interpretation"] = (
        "Each contrast carries a per-position verdict from distribution_agreement: "
        "'equivalent_within_tolerance' needs identical top-k membership, bounded common-token "
        "deltas and a bounded residual at EVERY position; 'divergent' names the positions that "
        "differ with a bounded residual; 'inconclusive' means skipped positions or an unbounded "
        "tail. The off-vs-off repeat is the same-serving floor for every quantity. The coarsened "
        "KL is a lower bound and is not the basis of any verdict."
    )
    result["status"] = "complete"
    checkpoint()
    print("[result]", result_path, flush=True)
    brief = {name: {"distributions": c["distributions"]["summary"],
                    "greedy_agreement": c["greedy_agreement"]} for name, c in contrasts.items()}
    brief["total_wall_s"] = result["total_wall_s"]
    print(json.dumps(brief, indent=1), flush=True)
    return {"result_path": str(result_path), "summary": {k: v for k, v in result.items()
                                                        if k != "raw_readouts"}}


@app.local_entrypoint()
def main(run: str, n_prompts: int = N_PROMPTS_DEFAULT, mode: str = "scope",
         n_pairs: int = CACHE_PAIRS_DEFAULT, with_fp8: bool = False):
    if not run:
        raise SystemExit("--run is required")
    if mode == "scope":
        answer = run_scope.remote(run, n_prompts)
    elif mode == "cache":
        answer = run_cache_equivalence.remote(run, n_pairs, with_fp8)
    else:
        raise SystemExit(f"--mode must be scope|cache, got {mode!r}")
    print(json.dumps({
        "result_path": answer["result_path"],
        "run_id": answer["summary"]["run_id"],
        "total_wall_s": answer["summary"]["total_wall_s"],
    }, indent=2))
