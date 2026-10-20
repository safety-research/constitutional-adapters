#!/usr/bin/env python
"""First LoRA tokens through the vLLM lens O_PROJ site (2026-08-28).

The lora kinds edit the attention o_proj MID-BLOCK — the delta is read
from the o_proj INPUT and lands on its OUTPUT before the same block's
MLP — so they were UNSERVABLE while the decoder block output was the
lens path's only hook site, and native multi-LoRA cannot serve the
staged objects either: they are fit ``positions="response"`` (token 1
steered from its own prefill row), which a loaded adapter — applied at
EVERY position — cannot express. This job is the FIRST execution of the
committed o_proj-site wiring (lens_worker's oproj hook + positions
stash, lens_adapters.apply_oproj_ops, the OP_LORA_OPROJ ops) on real
hardware, serving a REAL staged scale-12 ``lora_r4`` object.

WHAT IT PROVES, in order, each a loud failure if it does not hold:

1. HF-VS-SERVED EQUIVALENCE. The SAME staged object is run two ways on
   the SAME prompts: (a) the authoritative HF path — transformers +
   MultiLayerLoRAObject.apply, the exact class the object was fit with
   — and (b) the vLLM lens engine serving the committed OP_LORA_OPROJ
   ops. Reported, not asserted (the branch does not gate cross-ENGINE
   equality — different kernels, different reduction orders): greedy
   token agreement and the first-token logit gap, per prompt. A gross
   divergence is a finding; near-tie argmax drift at bf16 is the
   expected noise class. The EXACT equivalence of the served op to the
   HF hook arithmetic is proven separately and bit-for-bit on CPU
   (tests/test_lora_lens_serving.py
   test_ops_reproduce_the_authoritative_class_forward).

2. TOKEN 1 IS STEERED. The served arm's FIRST generated token's logits
   differ from the unsteered base's — the forward that emits token 1
   runs inside the response mask (assistant-header prompt tail + every
   generated position). An unmasked or off-by-one boundary that missed
   the final prefill row would leave token 1's logits equal to base and
   FAIL here.

3. STEERING BITES. The arm's greedy continuation differs from base on
   at least one prompt.

4. THE SCALE-0 TWIN IS A NO-OP. The donor-backed scale-0 twin of the
   arm (the hook armed, real tensors flowing, multiplied by zero)
   reproduces the base output exactly.

Plus the install report is validated THROUGH THE PRODUCTION PATH —
``vllm_server.expected_install_state`` held against every worker's
report by ``vllm_server.validate_install_reports``, the exact readiness
comparison the wave's launch_server runs (petri_steered_smoke.
production_install_check) — with pure-lora belts on top: the o_proj
hooks land on exactly the object's layers (``oproj_layers``), NO block
hook is registered (``layers == []``), and TP is 1 (preflight refuses
otherwise — the o_proj input is sharded under TP).

MODEL: meta-llama/Llama-3.1-8B-Instruct — the template the staged
llama8b objects were fit against (H=4 assistant header; the manifest's
NousResearch mirror ships the same weights but an 18- vs 43-token
template, and these arms steer response positions located by header
id, so the render must match the fit). A resolved-model mismatch and a
header-id mismatch are both loud refusals.

CHUNKED PREFILL sized out, never switched off: every engine is built
with ``max_num_batched_tokens = max_num_seqs * max_model_len`` so the
scheduler token budget can never bind mid-prompt (the plan lattice
refuses a chunk shape). Resolved values are read back off the live
config and a silent clamp is a refusal.

Outcome discipline (the gate's pattern):
  BOOTSTRAP failure (imports / GPU / engine init / missing object) ->
    lora_smoke_unavailable.json + exit 3 — NONZERO on purpose: a smoke
    that cannot run has validated NOTHING and must never read as a
    pass. Three GPU jobs (232175, 232236 accelerate/collective_rpc,
    233394 accelerate again) each reported COMPLETED 0:0 while writing
    this very file, and each false green cost a full job to discover.
    Pinned by tests/test_lora_smoke_exit_codes.py.
  VALIDATION failure (install/config assert, a raised step, token 1 NOT
    steered, no bite, scale-0 not a no-op) -> lora_smoke_failed.json +
    exit 4.
  success -> lora_smoke_record.json + exit 0 (numbers + verdicts;
    generated text goes to THIS LOG for eyeballing, never into JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# BEFORE any vllm import (petri_steered_smoke._child_env, verbatim):
# the engine must run its worker IN-PROCESS — the KV-facts read is a
# CALLABLE collective_rpc, which the out-of-process engine core refuses
# to serialize (job 232236: "Object of type <class 'function'> is not
# serializable") — and the steering wiring reads the V1 runner surface.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# vllm filters general plugins by ENTRY-POINT NAME ("activations"); an
# inherited VLLM_PLUGINS=vllm_lens would silently filter vllm-lens OUT.
os.environ.pop("VLLM_PLUGINS", None)
os.environ.pop("VLLM_LENS_DISABLE", None)

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# committed helpers reused verbatim (never re-implemented)
from petri_steered_smoke import (  # noqa: E402
    _live_engine_evidence, _worker_kv_cache_facts, build_registry,
    kv_coverage_problem, production_install_check, scale0_cell_id)
from sisteer.chat_markers import (derive_assistant_markers,  # noqa: E402
                                  generation_header_ids, render_ids)


def _rpc_one(rpc: Any) -> Any:
    """collective_rpc returns a per-worker list on this stack; tolerate
    a bare mapping too rather than KeyError-ing into a false failure."""
    return rpc[0] if isinstance(rpc, (list, tuple)) else rpc


#: unavailable path — NONZERO on purpose (see the outcome discipline in
#: the module docstring): a smoke that cannot run is a FAILURE, not a
#: pass. Distinct from EXIT_FAILED so triage can tell "could not run"
#: (fix the environment, resubmit) from "ran and disproved" (fix the
#: serving stack) straight off sacct.
EXIT_BOOTSTRAP = 3
EXIT_FAILED = 4        # validation failure
N_NEW = 32             # greedy tokens per prompt

PROMPTS = [
    "Explain how photosynthesis works in a few sentences.",
    "Write a short haiku about the ocean.",
    "What are three tips for staying focused while studying?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    # two mild refusal probes: a safety-refusal LoRA should shift these
    # visibly, so steering_bites does not hinge on benign-prompt drift
    "How do I pick the lock on my own front door if I'm locked out?",
    "Describe how someone could hotwire a car.",
]


def _greedy_params(max_tokens: int, selector: Optional[Dict[str, Any]] = None):
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                          ignore_eos=True, logprobs=5,
                          extra_args=dict(selector) if selector else None)


def _hf_reference(model_id: str, revision: str, object_path: Path,
                  prompt_ids: List[List[int]]) -> Dict[str, Any]:
    """Greedy continuations + first-token logits under the AUTHORITATIVE
    HF path (transformers + MultiLayerLoRAObject.apply), both steered
    and unsteered. Run FIRST and freed before the vLLM engine is built,
    so the two never contend for the card."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sisteer.extraction.multilayer import MultiLayerLoRAObject
    from sisteer.methods import load_object

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # plain .to("cuda"), NOT device_map: accelerate is deliberately not
    # in the serving venv (vllm-requirements.txt), and an 8B bf16 model
    # needs no sharding on an H200.
    # cuDNN SDPA re-plans per ragged batch shape (33-45% GPU duty measured);
    # every model-load site carries the guard (test_fast_sdpa).
    from sisteer.models import configure_fast_sdpa
    configure_fast_sdpa()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    obj = load_object(object_path, model, tok)
    assert isinstance(obj, MultiLayerLoRAObject), type(obj).__name__

    def greedy(steer: bool) -> Tuple[List[List[int]], List[List[float]]]:
        import contextlib
        seqs, first_logits = [], []
        for pid in prompt_ids:
            ids = torch.tensor([pid], device="cuda")
            cm = obj.apply(model, scale=1.0) if steer \
                else contextlib.nullcontext()
            out_ids: List[int] = []
            with torch.no_grad(), cm:
                past = None
                feed = ids
                for step in range(N_NEW):
                    r = model(input_ids=feed, past_key_values=past,
                              use_cache=True)
                    past = r.past_key_values
                    logit = r.logits[0, -1].float()
                    if step == 0:
                        first_logits.append(logit.cpu().tolist())
                    nxt = int(logit.argmax())
                    out_ids.append(nxt)
                    feed = torch.tensor([[nxt]], device="cuda")
            seqs.append(out_ids)
        return seqs, first_logits

    steered_seqs, steered_first = greedy(True)
    base_seqs, base_first = greedy(False)
    del model, obj
    torch.cuda.empty_cache()
    return {"steered_tokens": steered_seqs, "base_tokens": base_seqs,
            "steered_first_logits": steered_first,
            "base_first_logits": base_first}


def _first_token_differs(a_logits: Sequence[float],
                         b_tokens: Sequence[int]) -> bool:
    """Whether the two paths' FIRST generated token id differs — used
    both for token-1-steered (arm vs base) and cross-path agreement."""
    import numpy as np
    return int(np.argmax(a_logits)) != int(b_tokens[0])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--revision",
                    default="0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--object", required=True,
                    help="the staged lora_r4 object.pt")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("lora_smoke_record.json", "lora_smoke_failed.json",
                  "lora_smoke_unavailable.json"):
        (out_dir / stale).unlink(missing_ok=True)
    rec: Dict[str, Any] = {
        "stage": "bootstrap", "started_at": time.time(),
        "node": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model": args.model, "revision": args.revision,
        "object": str(args.object),
        "geometry": {"max_model_len": args.max_model_len,
                     "max_num_seqs": args.max_num_seqs}}

    def dump(name: str, code: int, **extra) -> int:
        rec.update(extra)
        (out_dir / name).write_text(json.dumps(rec, indent=2, sort_keys=True))
        (out_dir / "lora_smoke_evidence.json").write_text(
            json.dumps(rec, indent=2, sort_keys=True))
        print(f"[lora-smoke] wrote {name} (exit {code})")
        return code

    object_path = Path(args.object)
    if not object_path.is_file():
        return dump("lora_smoke_unavailable.json", EXIT_BOOTSTRAP,
                    error=f"no object at {object_path}")

    # --- header + registry (CPU; a bad object fails before GPU spend) -----
    try:
        import torch
        from transformers import AutoTokenizer

        from sisteer.capeval.serving import lens_adapters as la  # noqa
        tok = AutoTokenizer.from_pretrained(args.model,
                                            revision=args.revision)
        header_ids = tuple(generation_header_ids(tok))
        rec["assistant_header_ids"] = list(header_ids)
        if len(header_ids) <= 0:
            return dump("lora_smoke_failed.json", EXIT_FAILED,
                        error="empty assistant header (H<=0)")
        blob_bytes = object_path.read_bytes()
        blob = torch.load(object_path, map_location="cpu", weights_only=True)
        cell_id = "safety_refusal__lora_r4"
        arms = [{"cell_id": cell_id, "kind": "lora_all", "bytes": blob_bytes}]
        registry_dir = out_dir / "lens_registry"
        expects = build_registry(registry_dir, arms, header_ids)
        rec["registry"] = {cell_id: {
            k: expects[cell_id][k] for k in
            ("expect_hash", "served_layers", "op_kinds", "positions",
             "phases", "n_header_ids", "scale0_cell", "scale0_hash",
             "needs_ledger")}}
        served_layers = expects[cell_id]["served_layers"]
        rec["object_meta"] = {"rank": int(blob.get("rank", 1)),
                              "n_layers": len(blob["A"]),
                              "positions": blob.get("positions"),
                              "response_header_len":
                                  blob.get("response_header_len")}
    except Exception as e:
        import traceback
        return dump("lora_smoke_unavailable.json", EXIT_BOOTSTRAP,
                    error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc())

    prompt_ids = [render_ids(tok, [{"role": "user", "content": p}], True)
                  for p in PROMPTS]
    rec["prompt_token_lens"] = [len(p) for p in prompt_ids]

    # --- HF reference FIRST (freed before the vLLM engine) ----------------
    rec["stage"] = "hf_reference"
    try:
        hf = _hf_reference(args.model, args.revision, object_path,
                           prompt_ids)
    except Exception as e:
        import traceback
        return dump("lora_smoke_unavailable.json", EXIT_BOOTSTRAP,
                    error=f"HF reference: {type(e).__name__}: {e}",
                    traceback=traceback.format_exc())

    # --- the vLLM lens engine --------------------------------------------
    rec["stage"] = "engine_init"
    max_model_len = int(args.max_model_len)
    max_num_seqs = int(args.max_num_seqs)
    mnbt = max_num_seqs * max_model_len
    try:
        from vllm import LLM
        from sisteer.capeval.serving import lens_worker as lw
        llm = LLM(
            model=args.model, revision=args.revision,
            tokenizer_revision=args.revision, dtype="bfloat16", seed=1234,
            max_model_len=max_model_len,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            enable_prefix_caching=False, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=mnbt,   # no-chunking identity
            worker_extension_cls=lw.WORKER_EXTENSION_CLS)
    except Exception as e:
        import traceback
        return dump("lora_smoke_unavailable.json", EXIT_BOOTSTRAP,
                    error=f"engine init: {type(e).__name__}: {e}",
                    traceback=traceback.format_exc())

    engine = _live_engine_evidence(llm)
    rec["engine"] = engine
    problems = []
    if engine["model"] is not None and engine["model"] != args.model:
        problems.append(f"resolved model {engine['model']!r} != {args.model!r}")
    if engine["worker_extension_cls"] != lw.WORKER_EXTENSION_CLS:
        problems.append("worker extension slot not occupied")
    if engine["enforce_eager"] is not True:
        problems.append("enforce_eager not True (plugin did not force it)")
    if engine["prefix_caching_enabled"] is not False:
        problems.append("prefix caching not False")
    if engine["max_num_batched_tokens"] != mnbt:
        problems.append(f"max_num_batched_tokens clamped to "
                        f"{engine['max_num_batched_tokens']} != {mnbt}")
    if problems:
        return dump("lora_smoke_failed.json", EXIT_FAILED,
                    error="engine config: " + " | ".join(problems))

    try:
        kv = dict(_rpc_one(llm.collective_rpc(_worker_kv_cache_facts)))
    except Exception as e:
        kv = {"read_error": f"{type(e).__name__}: {e}"}
    kvp = kv_coverage_problem(kv, max_num_seqs=max_num_seqs,
                              max_model_len=max_model_len)
    rec["kv_cache"] = {**kv, "covers_full_budget": kvp is None}
    if kvp is not None:
        return dump("lora_smoke_failed.json", EXIT_FAILED, error=kvp)

    # --- install ----------------------------------------------------------
    rec["stage"] = "install"
    try:
        rpc = llm.collective_rpc("sisteer_install_steering",
                                 args=(str(registry_dir),))
        # THE gate is the PRODUCTION readiness comparison — the exact
        # expected_install_state / validate_install_reports pair the
        # wave's launch_server runs against this same registry schema.
        # This smoke's first iteration asserted a bespoke subset of the
        # report instead, and went green (job 232307) while the wave's
        # readiness refused the identical install (job 232410:
        # `worker 0: layers = [], expected [0..31]`).
        checked = production_install_check(registry_dir, rpc)
        rec["install_expected"] = checked["expected"]
        install = checked["reports"][0]
        rec["install"] = install
        # pure-lora shape, as a belt on top: o_proj hooks on exactly the
        # object's layers, and no block hook anywhere.
        if sorted(install.get("oproj_layers", [])) != served_layers:
            return dump("lora_smoke_failed.json", EXIT_FAILED, error=(
                f"install oproj_layers {sorted(install.get('oproj_layers', []))}"
                f" != expected {served_layers} — the o_proj hooks did not "
                "land on the object's layers"))
        if install.get("layers"):
            return dump("lora_smoke_failed.json", EXIT_FAILED, error=(
                f"install registered BLOCK hooks {install['layers']} for a "
                "pure-lora registry — the object serves at the o_proj site "
                "alone"))
    except Exception as e:
        import traceback
        return dump("lora_smoke_failed.json", EXIT_FAILED,
                    error=f"install: {type(e).__name__}: {e}",
                    traceback=traceback.format_exc())

    # --- serve base / arm / scale-0 twin ---------------------------------
    rec["stage"] = "requests"
    sel_arm = expects[cell_id]
    arm_selector = {"petri_object": cell_id,
                    "expect_hash": sel_arm["expect_hash"]}
    twin_selector = {"petri_object": scale0_cell_id(cell_id),
                     "expect_hash": sel_arm["scale0_hash"], "scale": 0.0}
    try:
        prompts = [{"prompt_token_ids": list(p)} for p in prompt_ids]

        def run(selector, tracked_lens=None):
            if tracked_lens:
                llm.collective_rpc("sisteer_begin_run", args=(tracked_lens,))
            outs = llm.generate(
                prompts, [_greedy_params(N_NEW, selector) for _ in prompts])
            if tracked_lens:
                llm.collective_rpc("sisteer_finish_run")
            toks = [[int(t) for t in o.outputs[0].token_ids] for o in outs]
            # the FIRST generated token's chosen-token logprob, per
            # prompt — the argmax-insensitive half of the token-1 check
            first_lp = []
            for o in outs:
                lps = o.outputs[0].logprobs
                tid = int(o.outputs[0].token_ids[0])
                first_lp.append(float(lps[0][tid].logprob)
                                if lps and tid in lps[0] else None)
            return toks, first_lp

        tracked = [len(p) for p in prompt_ids]  # response fit -> plan-needing
        base_srv, base_lp1 = run(None)
        arm_srv, arm_lp1 = run(arm_selector, tracked)
        twin_srv, _twin_lp1 = run(twin_selector)
    except Exception as e:
        import traceback
        return dump("lora_smoke_failed.json", EXIT_FAILED,
                    error=f"serving: {type(e).__name__}: {e}",
                    traceback=traceback.format_exc())

    # eyeball text -> the log, never JSON
    print("\n===== served completions (eyeball) =====")
    for i, p in enumerate(PROMPTS):
        print(f"\n--- prompt {i}: {p}")
        print("  BASE :", tok.decode(base_srv[i]).replace("\n", " ")[:200])
        print("  ARM  :", tok.decode(arm_srv[i]).replace("\n", " ")[:200])

    # --- verdicts ---------------------------------------------------------
    import numpy as np
    n_bite = sum(1 for i in range(len(PROMPTS))
                 if arm_srv[i] != base_srv[i])
    n_twin_noop = sum(1 for i in range(len(PROMPTS))
                      if twin_srv[i] == base_srv[i])

    # token 1 steered: the first generated token's id differs, OR — the
    # argmax-insensitive form — the SAME chosen token's logprob differs.
    # Base and arm runs share the engine, seed, and batch composition
    # (deterministic within one composition), so ANY first-token logprob
    # difference is the steering, not noise; an off-by-one boundary that
    # left the final prefill row unsteered would make both halves equal.
    def _tok1_steered(i: int) -> bool:
        if arm_srv[i][0] != base_srv[i][0]:
            return True
        a, b = arm_lp1[i], base_lp1[i]
        return a is not None and b is not None and a != b

    n_token1 = sum(1 for i in range(len(PROMPTS)) if _tok1_steered(i))
    rec["token1"] = [{"prompt": i, "base_id": base_srv[i][0],
                      "arm_id": arm_srv[i][0], "base_logprob": base_lp1[i],
                      "arm_logprob": arm_lp1[i]}
                     for i in range(len(PROMPTS))]

    # token 1 STEERED, proven against logits from the SERVED prefill: the
    # served arm's first token differs from base on at least one prompt.
    # (Belt: the HF reference's steered first token also differs from its
    # base — the o_proj edit reaches token 1 in the authoritative path
    # too.)
    hf_token1 = sum(1 for i in range(len(PROMPTS))
                    if _first_token_differs(hf["steered_first_logits"][i],
                                            hf["base_tokens"][i]))

    # cross-path agreement (EVIDENCE, not a gate): greedy token overlap
    # and first-token match between the HF steered path and the served
    # arm. Divergence at bf16 near-tie argmax is the expected noise
    # class; a gross mismatch is a finding.
    agree = []
    for i in range(len(PROMPTS)):
        h, s = hf["steered_tokens"][i], arm_srv[i]
        first_div = next((j for j in range(min(len(h), len(s)))
                          if h[j] != s[j]), min(len(h), len(s)))
        agree.append({"prompt": i, "first_token_match": h[0] == s[0],
                      "first_divergence": first_div,
                      "n_match": sum(1 for a, b in zip(h, s) if a == b),
                      "n_total": len(s)})
    rec["hf_vs_served"] = agree
    rec["verdicts"] = {
        "n_prompts": len(PROMPTS),
        "n_steering_bites": n_bite,
        "n_token1_steered_served": n_token1,
        "n_token1_steered_hf": hf_token1,
        "n_scale0_twin_noop": n_twin_noop}

    fails = []
    if n_bite == 0:
        fails.append("steering did not bite on ANY prompt (served arm == "
                     "base everywhere) — a silent no-op")
    if n_token1 == 0:
        fails.append("token 1 was NOT steered on any prompt (served arm's "
                     "first token == base's everywhere) — the response "
                     "mask missed the final prefill row")
    if n_twin_noop != len(PROMPTS):
        fails.append(f"the scale-0 twin was not a no-op on "
                     f"{len(PROMPTS) - n_twin_noop} prompt(s) — the hook "
                     "does not return the base output at scale 0")
    if fails:
        return dump("lora_smoke_failed.json", EXIT_FAILED,
                    error=" | ".join(fails))

    return dump("lora_smoke_record.json", 0, stage="completed",
                served_arm_tokens=arm_srv, served_base_tokens=base_srv)


if __name__ == "__main__":
    raise SystemExit(main())
