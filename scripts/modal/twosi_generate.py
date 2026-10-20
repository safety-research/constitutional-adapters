"""Qwen3.8-27B thinking-arm generation on Modal — the 8192-token regeneration.

WHY MODAL, after a day of pod failures: RunPod had no 8-GPU blocks, its 4-GPU
creates failed or provisioned dead (three pods RUNNING with an empty runtime
and no SSH), and every pod that DID come up paid ~20 min of idle GPU rebuilding
an env and re-downloading 54 GiB of weights. Modal fixes exactly those:

  * the vLLM image is a CACHED layer -- built once, free thereafter.
  * the model lives on the persistent `twosi-fits` Volume, ALREADY STAGED by
    twosi_fits.py (hf/hub/models--Qwen--Qwen3.8-27B), so containers read it
    instead of downloading it. HF_HUB_OFFLINE=1 makes that a hard guarantee
    rather than a hope: an accidental download attempt FAILS instead of
    silently costing 54 GiB of GPU-idle time.
  * .map() fans out to <=16 concurrent H200s on demand, no NCCL, no bootstrap.

The Volume+image pattern is borrowed from twosi_fits.py.

WHY 8192: the first attempt at 4096 left ~20% of responses with an
UNTERMINATED <think> block -- no answer at all -- and unevenly by condition
(15.6%-27.6%). Those rows are excluded rather than scored, so the arm was
biased. The unfinished responses had median length EXACTLY 4096, i.e. cut at
the wall rather than stopped naturally, so more budget is the fix.

SHARDING: one GPU per container, one shard of the prompt list each, sized so a
container runs ~10-30 min and amortises its ~1 min cold start. Each shard
writes its own jsonl to the Volume and commits -- returning 89k long
generations through the client would be a giant payload for no reason.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("twosi-generate")
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"

# cu129, matching the clusters and the earlier pods exactly, so this arm is not
# quietly produced by a different build than the datasets it is compared to.
image = (
    # nvidia/cuda DEVEL base, not debian_slim: vLLM's memory profiler shells
    # out to nvcc and dies with
    #     RuntimeError: Could not find nvcc and default
    #     cuda_home='/usr/local/cuda' doesn't exist
    # on a slim image -- AFTER loading 54GB of weights, so it costs a full
    # cold start per shard to discover. twosi_fits.py hit the same wall on
    # this model and uses the same base.
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "vllm==0.26.0",
        "torch==2.11.0+cu129",
        "transformers==5.15.0",
        "numpy",                    # NOT implicit; missing it fails at import
        "huggingface_hub",
        "pyyaml",
        # NOTE: pip, not uv -- `--index-strategy unsafe-best-match` is a uv
        # flag and pip rejects it outright ("no such option"), failing the
        # image build before anything runs.
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    # STEP 2, AND IT IS NOT OPTIONAL. PyPI's `vllm==0.26.0` is the cu13-linked
    # build, so step 1 gives a torch that is cu129 and a vLLM that is not, and
    # the container dies at import with:
    #     ImportError: libcudart.so.13: cannot open shared object file
    # The torch extra-index fixes torch only; it does nothing for vLLM's own
    # compiled binary. Overlay the cu129 build by DIRECT URL -- the wheels.vllm
    # .ai index is keyed by commit sha and its hrefs are relative, so the sha
    # has to be read out rather than the /0.26.0/cu129/ path pasted (that 404s).
    # --no-deps so it swaps ONLY the binary and leaves step 1's resolved tree.
    .pip_install(
        "https://wheels.vllm.ai/568afb3a13806beb53bb2e6bd518269357b237c0/"
        "vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl",
        extra_options="--no-deps --force-reinstall",
    )
    # Fail the BUILD, not the first GPU container: an image that imports is
    # the only evidence that counts, and discovering this at run time costs a
    # cold start and a GPU-minute per shard instead of one build.
    .run_commands(
        "nvcc --version | tail -1",
        "python -c \"import torch, vllm; "
        "assert 'cu129' in torch.__version__, torch.__version__; "
        "print('IMAGE OK', vllm.__version__, torch.__version__)\""
    )
    .env({
        "HF_HOME": f"{VOL_PATH}/hf",
        # the model is on the Volume; a download attempt is a BUG here, and
        # this turns it into a loud failure instead of 54 GiB of idle GPU
        "HF_HUB_OFFLINE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        # torch.compile cache ON THE VOLUME, not container-local. vLLM spends
        # ~60-100s in Dynamo bytecode transform + CUDA graph capture before
        # the first token, and that is CPU work with the GPU sitting idle.
        # Container-local (~/.cache/vllm) means EVERY shard pays it; on the
        # volume the first shard warms it and the rest read it. At 16 shards
        # that is ~20 GPU-minutes of pure idle recovered.
        "VLLM_CACHE_ROOT": f"{VOL_PATH}/vllm_cache",
    })
)

# Per-model settings, selected by --model-key. Kept as a table rather than
# CLI floats so a dataset's identity is a NAME, and two runs of the same key
# cannot silently differ.
#   tf: the transformers pin the model needs. gemma-4 is a HETEROGENEOUS
#       (per-layer) architecture and 5.15.0 raises
#       AmbiguousGlobalPerLayerAttributeError on its head_dim; 5.14.1 loads it.
#   effort: only reasoning models use it.
#   max_tokens: a cap costs nothing when unhit -- generation stops at EOS --
#       so these are set for safety, not economy. The small models simply
#       never approach them.
MODELS = {
    "qwen38_think": dict(hf="Qwen/Qwen3.8-27B", tf="5.15.0",
                         rev="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                         effort="medium", max_tokens=9216),
    "e2b":          dict(hf="google/gemma-4-E2B-it", tf="5.14.1",
                         rev=None, effort=None, max_tokens=4096),
    "llama32_1b":   dict(hf="unsloth/Llama-3.2-1B-Instruct", tf="5.15.0",
                         rev=None, effort=None, max_tokens=4096),
    # The 3B sibling, to test whether repetition degeneracy is a SIZE effect.
    # Same ungated unsloth mirror as the 1B, so the chat-template
    # scaffold-stripping already verified there applies unchanged. Revision
    # PINNED: leaving rev=None broke the offline volume cache once (no refs/
    # entry to resolve against with HF_HUB_OFFLINE=1).
    "llama32_3b":   dict(hf="unsloth/Llama-3.2-3B-Instruct", tf="5.15.0",
                         rev="006f5dcd1393c3add266de40994ba96225e9689d",
                         effort=None, max_tokens=4096),
    # Qwen3.5: the plain repos ARE the instruct models (no -Instruct suffix
    # exists in this line; -Base is the separate pretrain release). Hybrid
    # linear+full attention with a vision tower
    # (Qwen3_5ForConditionalGeneration) -- supported by BOTH pins of the
    # shared image, verified against the tagged sources before staging
    # (vllm v0.26.0 registry + transformers v5.15.0 models/qwen3_5).
    #
    # THINKING IS ON BY DEFAULT in the chat template:
    #     {%- if enable_thinking is defined and enable_thinking is false %}
    # i.e. the default render ends with an OPEN <think> block. This is NOT
    # Qwen3.8's reasoning_effort knob -- the template never reads
    # reasoning_effort, so `effort` would be a silent no-op here.
    #
    # Four variants, a thinking-vs-not contrast at two sizes. The ONLY things
    # that differ between an off and a think sibling are the thinking itself
    # and the room the thinking needs:
    #   *_off    tpl_kwargs {"enable_thinking": False} pre-closes the block
    #            (`<think>\n\n</think>\n\n`), system turn carries only our SI
    #            -- the thinking-OFF render, matching the qwen38_off pool.
    #            Same pin as sisteer.models.load_tokenizer, so fits/evals
    #            render these prompts the way they were generated.
    #   *_think  the template default: the prompt ENDS with an open <think>.
    #            max_tokens 24576 and max_model_len 32768. Qwen3.5 has NO
    #            reasoning_effort knob (the template reads only enable_thinking),
    #            so traces run long: a 9216 smoke answered only 61% (9B) / 36%
    #            (4B) of rows, every failure length-capped at exactly 9216 and
    #            worst in the long-prompt SIs -- an uneven loss that biases the
    #            SI contrast. A 9B probe at 24576 set the cap for both
    #            sizes. Worst rendered prompt on THIS tokenizer is 7,386
    #            tokens (re-measured, 200 longest, think render), so
    #            7,386 + 24,576 = 31,962 <= 32,768 (slack 806).
    #            think_open=True also makes _shard_body (a) ASSERT every
    #            render ends with the open block and (b) STORE responses
    #            with the prompt-opened "<think>\n" prepended, so the
    #            stored text is the complete assistant turn. Without (b),
    #            every downstream gate of the form `if "<think>" in
    #            response` -- both grade entry points' _strip_think and
    #            engine.strip_think's unterminated-trace rule -- is a
    #            silent no-op, and the judge scores the reasoning trace
    #            along with the answer. That exact false negative shipped
    #            on qwen38_think8k (verified against its grade cache:
    #            300/300 sampled keys hash the RAW trace+answer).
    # _shard_body ASSERTS the off renders actually closed the block (a
    # silent template no-op would ship a confounded pool that looks fine).
    "qwen35_9b_off":   dict(hf="Qwen/Qwen3.5-9B", tf="5.15.0",
                            rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                            effort=None, max_tokens=4096,
                            tpl_kwargs={"enable_thinking": False}),
    "qwen35_9b_think": dict(hf="Qwen/Qwen3.5-9B", tf="5.15.0",
                            rev="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                            effort=None, max_tokens=24576,
                            max_model_len=32768, think_open=True),
    "qwen35_4b_off":   dict(hf="Qwen/Qwen3.5-4B", tf="5.15.0",
                            rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                            effort=None, max_tokens=4096,
                            tpl_kwargs={"enable_thinking": False}),
    "qwen35_4b_think": dict(hf="Qwen/Qwen3.5-4B", tf="5.15.0",
                            rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                            effort=None, max_tokens=24576,
                            max_model_len=32768, think_open=True),
}
# gemma-4 (E4B and E2B) is a HETEROGENEOUS per-layer architecture: 5.15.0
# raises AmbiguousGlobalPerLayerAttributeError on its head_dim, 5.14.1 loads
# it. A second image rather than downgrading the shared one, so the datasets
# already produced on 5.15.0 keep the stack that produced them.
image_tf514 = image.pip_install("transformers==5.14.1",
                                extra_options="--no-deps --force-reinstall")

MODEL = "Qwen/Qwen3.8-27B"
# PIN THE REVISION, and not merely for provenance -- it is what makes the
# offline volume cache usable at all. The staged cache has snapshots/<sha> but
# NO refs/ dir, so resolving the implicit "main" needs a network round trip;
# with HF_HUB_OFFLINE=1 that fails with LocalEntryNotFoundError even though
# every weight file is sitting right there. Naming the sha skips ref
# resolution and reads the snapshot directly.
# Same sha twosi_fits.py pins, so this arm and its fits use one checkout.
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# The output dir carries the SETTINGS that produced the rows. Resume keyed on
# prompt identity alone silently skipped every row when the effort and token
# cap changed, and reported the OLD rows as the new run's result -- a config
# change that reads as a finished job is the worst kind of no-op. Changing
# effort or cap now changes the path, so resume can only ever match like with
# like.
EFFORT = "medium"
MAX_TOKENS = 9216
# A smoke test must never write where a production run reads. Settings-keyed
# paths stopped a CONFIG change from resuming onto stale rows, but a 24-prompt
# smoke and an 89k run share settings by design -- so smoke9's 96 rows landed
# in the full run's shards 0 and 1 (+44, +48). Harmless in content, wrong in
# accounting. Scale of the run is part of the path now.
OUT_SUFFIX = os.environ.get("TWOSI_OUT_SUFFIX", "")
OUT_DIR = (f"{VOL_PATH}/gen/qwen38_think_{EFFORT}_{MAX_TOKENS}"
           + (f"_{OUT_SUFFIX}" if OUT_SUFFIX else ""))


@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=60 * 60 * 2,
              cpu=4.0)
def stage_model(model_key: str) -> dict:
    """Download a model to the Volume ONCE, on a CPU container.

    gpu-util item 2, and the reason HF_HUB_OFFLINE=1 is set on the GPU path:
    an unstaged model must fail LOUDLY at once rather than have sixteen H200s
    each pull tens of GiB while billing. That guard fired exactly as intended
    -- Llama-3.2-1B was not on the volume and the run died in seconds.

    CPU container on purpose: downloading is network work with no business
    holding a GPU.
    """
    import os

    # ORDER MATTERS. huggingface_hub reads HF_HUB_OFFLINE into a module
    # CONSTANT at import time, so setting the env var after importing it does
    # nothing -- the first version of this function did exactly that and died
    # with LocalEntryNotFoundError while trying to download. Set the env
    # BEFORE the import, and override the already-resolved constant too, so
    # this works regardless of whether something else imported hub first.
    os.environ["HF_HUB_OFFLINE"] = "0"
    import huggingface_hub.constants as _hc
    _hc.HF_HUB_OFFLINE = False
    from huggingface_hub import snapshot_download

    spec = MODELS[model_key]
    path = snapshot_download(spec["hf"], revision=spec.get("rev"),
                             cache_dir=f"{VOL_PATH}/hf/hub")
    VOL.commit()
    files = sorted(os.listdir(path))
    return {"model": spec["hf"], "n_files": len(files),
            "has_config": "config.json" in files,
            "has_tokenizer": any(f.startswith("tokenizer") for f in files),
            "weights": sum(1 for f in files if f.endswith(".safetensors"))}


# One engine per container, reused across inputs.
#
# WHY THIS EXISTS. .map() sends 16 inputs; max_containers=16 is a CEILING, not
# a promise. When Modal can only get N<16 H200s, each container is handed a
# SECOND input after its first returns -- and a vLLM engine built as a local
# does not release its VRAM when that local goes out of scope. The second
# build then meets its own predecessor still holding the card:
#     Free memory on device cuda:0 (13.36/139.8 GiB) is less than desired
#     GPU memory utilization (0.9, 125.82 GiB)
# 0.90 x 139.8 = 125.8 used, 14.0 left, 13.36 reported -- the "neighbour" is
# the previous input. Measured: the Llama-1B run got ~5 containers and wrote
# exactly 5 of 16 shards, every container dying on its second input.
#
# Reusing is strictly better than freeing: same model and settings for every
# shard in a run, so the second input skips a full load and a Dynamo warm.
_ENGINE: dict = {}


def _get_engine(key: tuple, build):
    """Return this container's engine, building it at most once per config."""
    if key in _ENGINE:
        print(f"[shard] reusing cached engine {key[0]}", flush=True)
        return _ENGINE[key]
    if _ENGINE:
        # A different config in the same container: the old engine must go,
        # or it holds VRAM the new one is about to ask for.
        import gc
        import torch
        _ENGINE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    _ENGINE[key] = build()
    return _ENGINE[key]


def _shard_body(payload: dict) -> dict:
    """One shard: build an engine, render every prompt, generate, commit."""
    import time
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    idx, total = payload["idx"], payload["total"]
    prompts = payload["prompts"]
    sis = payload["sis"]                     # {cond: system_text or None}
    t0 = time.time()

    out_p = Path(payload["out_dir"]) / f"shard_{idx:03d}.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_p.exists():                        # resume: a re-run costs nothing
        for line in out_p.open():
            try:
                r = json.loads(line)
                done.add((r["attack"], r["goal_id"], r.get("sample_idx"),
                          r.get("variant_name"), r.get("si_condition")))
            except Exception:                 # noqa: BLE001
                continue

    spec = payload["spec"]
    mid, rev, eff = spec["hf"], spec.get("rev"), spec.get("effort")
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    # FILL THE CARD. There is no 2^n batch constraint here -- that is a
    # TRAINING concern. vLLM continuously batches and sizes batches from
    # whatever KV cache exists, so the only real knob is how much of the 141GB
    # H200 we hand it. 0.95 leaves ~80GB of cache after 54GB of weights
    # instead of ~72GB at 0.90, and with 8192-token outputs this job is
    # cache-bound: more cache is directly more concurrent sequences.
    # No memory probe is needed or wanted -- a probe answers "what batch fits",
    # and vLLM answers that itself, per step.
    # Per-spec max_model_len: the qwen35_*_think variants need 18432, because
    # their worst rendered prompt is 7,388 tokens and 7,388 + 9,216 output
    # exceeds the 16,384 default (measured; see the MODELS comment). The
    # engine cache key carries the value, so an off and a think shard landing
    # in one container cannot share the wrong engine.
    mml = int(spec.get("max_model_len", 16384))
    llm = _get_engine(
        (mid, rev, mml, 0.90),
        lambda: LLM(model=mid, revision=rev, dtype="bfloat16",
                    max_model_len=mml,
                    gpu_memory_utilization=0.90, enable_prefix_caching=True,
                    enforce_eager=False, seed=0))
    # max_num_seqs left at vLLM's default: on a 141GB H200 the default fits
    # this hybrid Mamba model (one cache block per decode sequence). An 80GB
    # card needs <=~300 or the engine REFUSES to start -- it does not auto-cap.
    # 9216, not 8192: the longest SI is 6,678 tokens and max_model_len is
    # 16,384, so 6,808 + 9,216 = 16,024 fits with headroom on every condition.
    # Generating ABOVE 8k is deliberate -- greedy decode is prefix-
    # deterministic, so an 8k dataset can be recovered later by truncation
    # while the reverse is impossible. The longer cap strictly dominates.
    params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=spec["max_tokens"])
    # publish the compile cache the engine just warmed, so sibling and later
    # shards skip Dynamo entirely. Idempotent: same model, same shapes.
    try:
        VOL.commit()
    except Exception:                                            # noqa: BLE001
        pass

    n_written = 0
    # SI-block by SI-block so the long SI prefix is prefilled once per block
    # and every request in that block hits the prefix cache.
    for cond, si_text in sis.items():
        todo = [p for p in prompts
                if (p["family"], p["goal_id"], p.get("sample_idx"),
                    p.get("variant_name"), cond) not in done]
        if not todo:
            continue
        msgs = [([{"role": "system", "content": si_text}] if si_text else [])
                + [{"role": "user", "content": p["text"]}] for p in todo]
        # reasoning_effort="medium" does TWO things, and the second is the
        # scientifically important one:
        #  1. shorter traces than the xhigh default, so fewer responses run
        #     out of budget before producing an answer;
        #  2. it injects NO reasoning directive. xhigh prepends "Reasoning
        #     effort is set to xhigh. Please think carefully..." and low
        #     prepends its own; medium leaves the system turn carrying ONLY
        #     our SI. Verified: the medium render is byte-identical to the
        #     thinking-OFF render up to the think block, differing solely in
        #     whether <think> is left open.
        # That makes this arm a CLEAN single-variable contrast against the
        # thinking-off Qwen pool, instead of the two-variable "as shipped"
        # comparison (reasoning AND an extra directive) it would otherwise be.
        rendered = [tok.apply_chat_template(m, tokenize=False,
                                            add_generation_prompt=True,
                                            **({"reasoning_effort": eff}
                                               if eff else {}),
                                            **spec.get("tpl_kwargs", {}))
                    for m in msgs]
        # VERIFY the render, do not assume it. Qwen3.5's template turns
        # thinking ON unless enable_thinking=False is passed; a template that
        # silently ignored the kwarg (renamed knob, new revision) would ship
        # a confounded pool that LOOKS fine. The correct OFF render still
        # contains "<think>" -- as an immediately-closed empty block -- so
        # the off invariant is "no UNCLOSED think block", not "no <think>".
        # The think invariant is the mirror image: every prompt must END with
        # the open block the response prefix below claims it has.
        if spec.get("tpl_kwargs", {}).get("enable_thinking") is False:
            bad = [s for s in rendered
                   if s.count("<think>") != s.count("</think>")]
            assert not bad, (
                f"enable_thinking=False did not close the think block on "
                f"{len(bad)}/{len(rendered)} renders; refusing to generate a "
                f"confounded pool. Tail: {bad[0][-200:]!r}")
        if spec.get("think_open"):
            bad = [s for s in rendered if not s.rstrip().endswith("<think>")]
            assert not bad, (
                f"think_open=True but {len(bad)}/{len(rendered)} renders do "
                f"not end with an open <think> block; the stored-response "
                f"prefix would be a lie. Tail: {bad[0][-200:]!r}")
        # The prompt-opened think tag is not part of vLLM's completion, so
        # without this the stored response of a thinking run contains only
        # "</think>" and every `if "<think>" in response` gate downstream
        # (grading's _strip_think at both entry points, strip_think's
        # unterminated-trace -> "" rule) silently never fires -- the judge
        # then scores the reasoning trace. Prepend the tag the prompt opened
        # so the stored text is the COMPLETE assistant turn.
        resp_prefix = "<think>\n" if spec.get("think_open") else ""
        outs = llm.generate(rendered, params)
        with out_p.open("a") as f:
            for p, o in zip(todo, outs):
                out = o.outputs[0]
                f.write(json.dumps({
                    "attack": p["family"], "goal_id": p["goal_id"],
                    "attack_prompt": p["text"], "response": resp_prefix + out.text,
                    "sample_idx": p.get("sample_idx"),
                    "variant_name": p.get("variant_name"),
                    "trial": p.get("trial", 0), "candidate_idx": 0,
                    "category": p.get("category"),
                    "category_group": p.get("category_group"),
                    "source": p.get("source"), "side": p.get("side"),
                    "si_id": (cond if cond != "none" else None),
                    "si_condition": cond, "graded": False, "score": None,
                    "finish_reason": out.finish_reason,
                    "n_tokens": len(out.token_ids),
                }) + "\n")
                n_written += 1
        VOL.commit()                          # per block, not per shard

    rows = sum(1 for _ in out_p.open()) if out_p.exists() else 0
    # plain types ONLY: a str-subclass (e.g. torch.__version__) fails to
    # deserialize in the local client.
    return {"idx": int(idx), "written": int(n_written), "rows": int(rows),
            "secs": float(time.time() - t0)}


@app.function(image=image, gpu="H200", volumes={VOL_PATH: VOL},
              timeout=60 * 60 * 6, max_containers=16)
def generate_shard(payload: dict) -> dict:
    return _shard_body(payload)


@app.function(image=image_tf514, gpu="H200", volumes={VOL_PATH: VOL},
              timeout=60 * 60 * 6, max_containers=16)
def generate_shard_tf514(payload: dict) -> dict:
    """Identical work on the transformers-5.14.1 image, for gemma-4."""
    return _shard_body(payload)


@app.local_entrypoint()
def main(inventory: str = "", model_key: str = "qwen38_think",
         stage: bool = False, shards: int = 16,
         limit: int = 0, out_suffix: str = "",
         sis: str = "safety_refusal,si_constitution,si_irrelevant",
         no_none: bool = False, max_tokens: int = 0):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from sisteer.data.instructions import REGISTRY, SAFETY_SI

    if model_key not in MODELS:
        raise SystemExit(f"--model-key must be one of {sorted(MODELS)}")
    spec = MODELS[model_key]
    if max_tokens:
        # Top-up runs re-generate a SUBSET at a bigger cap.
        #
        # Do NOT assume the untouched rows come back identical: vLLM batches
        # continuously, so reduction order -- and therefore argmax on a
        # near-tie -- depends on what else is in the batch. Measured on this
        # very path: 59% of uncapped rows came back textually different
        # (diverging after a shared prefix, usually reconverging). What makes
        # a top-up sound is replacing every condition of a touched prompt
        # TOGETHER, not bitwise reproducibility.
        spec = dict(spec, max_tokens=int(max_tokens))

    if stage:
        r = stage_model.remote(model_key)
        print(f"[stage] {r['model']}: {r['weights']} weight shards, "
              f"config={r['has_config']} tokenizer={r['has_tokenizer']}")
        if not (r["has_config"] and r["has_tokenizer"] and r["weights"]):
            raise SystemExit("[stage] INCOMPLETE -- refusing to call it staged")
        return
    if not inventory:
        raise SystemExit("--inventory is required unless --stage")

    by = {i.id: i for i in REGISTRY}
    by.setdefault(SAFETY_SI.id, SAFETY_SI)
    conds = {} if no_none else {"none": None}
    for x in [y for y in sis.split(",") if y]:
        conds[x] = by[x].text

    inv = json.loads(Path(inventory).expanduser().read_text())["prompts"]
    if limit:
        step = len(inv) / limit
        inv = [inv[int(i * step)] for i in range(limit)]

    # path carries model AND settings: a dataset's identity is its path, and
    # resume can then only ever match rows produced the same way.
    out_dir = (f"{VOL_PATH}/gen/{model_key}_{spec['max_tokens']}"
               + (f"_{spec['effort']}" if spec.get("effort") else "")
               + (f"_{out_suffix}" if out_suffix else ""))

    # ROUND-ROBIN, not contiguous: the inventory is family-ordered and the
    # benign tail generates far longer answers, so contiguous slices strand
    # half the containers idle (measured, 4 of 8 finished while 4 ran on).
    payloads = [{"idx": i, "total": shards, "sis": conds, "spec": spec,
                 "out_dir": out_dir, "prompts": inv[i::shards]}
                for i in range(shards)]
    fn = generate_shard_tf514 if spec["tf"] == "5.14.1" else generate_shard
    print(f"[gen] {model_key} ({spec['hf']}, tf{spec['tf']}): {len(inv)} prompts "
          f"x {len(conds)} conditions = {len(inv)*len(conds)} generations "
          f"over {shards} H200 shards -> {out_dir}")
    tot = 0
    for r in fn.map(payloads):
        tot += r["rows"]
        print(f"[gen] shard {r['idx']:3d}: {r['rows']:6d} rows "
              f"in {r['secs']/60:.1f} min", flush=True)
    print(f"[gen] TOTAL {tot} rows at {out_dir}")
