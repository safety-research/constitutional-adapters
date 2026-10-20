"""One-node engine-init + generate probe: Qwen3.5-9B on the PINNED vllm 0.25.1+cu129 venv.

Qwen3.5 checkpoints have only ever been loaded under vllm 0.26.0 (offline pod
gate + a Modal app); the pinned serving venv (sisteer-petri-vllm-6b72fc94,
vllm 0.25.1) has never loaded one. 0.26.0 is not adoptable (its cutlass-dsl
chain forces protobuf>=6.30.2 against evalplus's protobuf<6 — the reason
0.25.1 was pinned, see configs/capeval/vllm-requirements.txt). So this probe
answers, cheaply and decisively, BEFORE any wave work: does qwen3_5 serve on
the pinned stack at all, and at what no-chunk geometry?

Phases (each run as its OWN process by the sbatch — engine teardown between
geometries is not trusted within one process):
  A    modest geometry (4096 x 8), engine defaults otherwise — the decisive
       "does it load and generate" test. Hybrid attn auto-disables prefix
       caching in 0.25.1 (model.py is_prefix_caching_supported), no flag needed.
  B<N> the wave's no-chunk identity at 16384 x N:
       max_num_batched_tokens = N x 16384 EXACTLY, chunked prefill off,
       prefix caching off. Bracket N empirically; the sbatch greps the
       engine's "Maximum concurrency for ... tokens per request" read-back —
       max_num_seqs is a ceiling, not a grant.
  C<N> blocker-1 recon: enable_prefix_caching=True + mamba_cache_mode="align"
       (what qwen3_5.py's own error message prescribes) at the same identity.
       If this inits+generates, the wave can keep prefix caching for qwen via
       align mode; if not, qwen serves caching-off and the gate assertion goes
       model-aware.

Pass evidence (a check that cannot fail is worthless): real generated token
ids/text asserted non-empty and non-degenerate, printed PROBE_GEN and written
to --out JSON. Exit is nonzero on ANY failure; exit 0 only after assertions.
"""

import argparse
import json
import sys
import time

MODEL = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # provenance.json model_revision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, help="A | B<seqs> | C<seqs>")
    ap.add_argument("--out", required=True, help="evidence JSON path")
    args = ap.parse_args()
    phase = args.phase

    kw = dict(
        model=MODEL,
        revision=REV,
        tokenizer_revision=REV,
        dtype="bfloat16",
        enforce_eager=True,  # the production steering path serves eager
        gpu_memory_utilization=0.90,
        # text-only serving: zero out the multimodal budget so profiling
        # doesn't reserve for images this wave never sends
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if phase == "A":
        kw.update(max_model_len=4096, max_num_seqs=8)
    elif phase[:1] in ("B", "C") and phase[1:].isdigit():
        n = int(phase[1:])
        kw.update(
            max_model_len=16384,
            max_num_seqs=n,
            max_num_batched_tokens=n * 16384,  # the no-chunk identity, exactly
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
        )
        if phase[:1] == "C":
            kw.update(enable_prefix_caching=True, mamba_cache_mode="align")
    else:
        sys.exit(f"unknown phase {phase!r}")

    print(f"PROBE_PHASE_START {phase} kwargs={json.dumps({k: v for k, v in kw.items() if k != 'model'}, default=str)}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(**kw)
    init_s = time.time() - t0

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    msgs = [
        [{"role": "user", "content": "Name the first five prime numbers."}],
        [{"role": "user", "content": "Write two sentences about why the sky is blue."}],
        [{"role": "user", "content": "List three creative uses for a paperclip."}],
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    ]
    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t1

    evid = {
        "phase": phase,
        "model": MODEL,
        "revision": REV,
        "engine_kwargs": {k: v for k, v in kw.items() if k != "model"},
        "init_s": round(init_s, 1),
        "gen_s": round(gen_s, 1),
        "generations": [],
    }
    total_new = 0
    for o in outs:
        text = o.outputs[0].text
        ids = list(o.outputs[0].token_ids)
        total_new += len(ids)
        evid["generations"].append(
            {"prompt_tail": o.prompt[-120:], "n_tokens": len(ids), "text": text}
        )
        print(f"PROBE_GEN [{phase}] ({len(ids)} tok): {text!r}", flush=True)

    # positive artifacts only a working engine could produce
    assert len(outs) == len(prompts), f"{len(outs)} outputs for {len(prompts)} prompts"
    assert all(g["n_tokens"] > 0 for g in evid["generations"]), "empty generation"
    assert any(
        len(set(o.outputs[0].token_ids)) > 5 for o in outs
    ), "degenerate output (no generation with >5 distinct tokens)"

    with open(args.out, "w") as f:
        json.dump(evid, f, indent=2)
    print(
        f"PROBE_PHASE_OK {phase} total_new_tokens={total_new} init_s={init_s:.1f} gen_s={gen_s:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
