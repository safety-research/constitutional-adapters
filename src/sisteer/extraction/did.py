"""DiD — difference-in-differences steering.

The steering vector isolates the CONTEXT-SPECIFIC excess of the system-
instruction (SI) direction: how the SI shifts the residual stream in an
adversarial ("jailbreak") input context, over and above how it shifts it in
a benign input context.

    v_cond = ( E[h | jb,     SI] - E[h | jb,     ¬SI] )
           - ( E[h | benign, SI] - E[h | benign, ¬SI] )

at the pinned layer. Each inner difference is a CAA-style SI-effect vector
on one input distribution, so DiD is literally the second difference of two
CAA vectors — one on adversarial user prompts, one on benign ones. DiD
combines BiDPO and CAA: BiDPO contributes the difference-of-differences
contrast; CAA contributes the pooling.

POOLING CHOICE (CAA completion_mean or prompt-last-token): we use **CAA
freeform completion_mean** (the repo's `extract_caa` convention,
extraction/diffmean.py). Rationale — DiD is CAA-flavored by construction,
so CAA's completion pooling is the natural default; the steering vector is
applied all-positions during generation (standard CAA / `vector`-arm
semantics), and completion_mean reads the SI's effect where it manifests
in the model's actual output, which is what an all-positions additive
vector reproduces. Each of the four conditions is generated FRESH under
its own (present/absent) system prompt and pooled over the model's
completion tokens; per-condition n is recorded. (Prompt-last-token — the
instr_diff convention — was the considered alternative; it would avoid
generating completions but reads the SI effect only at the prompt
boundary rather than across the generated response.)

DATA MAPPING (SI steering adaptation):
- benign user context = the arm's standard A1 pair prompts (alpaca-derived).
- jb user context = HarmBench VAL-split standard behaviours
  (eval/jailbreak_data.load_harmbench), deliberately the VAL split so the
  extraction never touches the TEST eval set used for jailbreak evaluation.
- SI present / absent = the instruction as a system prompt vs no system
  prompt (models.build_chat_prompt), exactly as every other arm's contrast.
Completions are pooled and DISCARDED (only per-condition activation means
persist); no harmful text is stored. On plain IFEval DiD may UNDERPERFORM
CAA by construction — the benign SI-effect is subtracted out — which is the
point, not a bug: its natural showcase is adversarial contexts (a follow-up
under the longctx adversarial secondary or in jailbreak defense).

The trained artifact is a plain SteeringVector at the pinned layer (with the
canary-calibrated base scale baked in), applied all-positions like CAA —
downstream it is an ordinary VectorObject.
"""

from __future__ import annotations

import torch

from ..generate import encode_kwargs, generate_batch
from ..models import build_chat_prompt
from ..vectors import SteeringVector
from .calibrate import DEFAULT_SCALE_LADDER, calibrate_base_scale
from .capture import _pool, capture_layer_outputs
from .traindata import EncodedSeq, collate

CONSTRUCTION = "difference-in-differences (user construction 2026-08-16)"


def did_combine(
    e_jb_si: torch.Tensor,
    e_jb_no: torch.Tensor,
    e_benign_si: torch.Tensor,
    e_benign_no: torch.Tensor,
) -> torch.Tensor:
    """v_cond = (E[jb,SI] - E[jb,¬SI]) - (E[benign,SI] - E[benign,¬SI])."""
    return (e_jb_si - e_jb_no) - (e_benign_si - e_benign_no)


@torch.no_grad()
def _condition_pooled(
    model,
    tok,
    prompts: list[str],
    system: str | None,
    layer: int,
    *,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    max_len: int = 768,
) -> tuple[torch.Tensor, int]:
    """E[h | context, SI-present-or-absent]: render each prompt with (or
    without) the instruction as system, greedily generate the model's
    completion in that condition, and pool the layer-``layer`` residual over
    the completion tokens (completion_mean). Returns (mean [d], n_used).

    Rows whose completion is empty carry no completion tokens and are
    dropped from the mean (they would pool to ~0 and bias the estimate).
    Prompt and completion are tokenized SEPARATELY and concatenated at the
    token level so the completion boundary is exact and the chat template's
    BOS is never doubled (encode_kwargs).
    """
    device = next(model.parameters()).device
    rendered = [build_chat_prompt(tok, p, system=system) for p in prompts]
    completions = generate_batch(
        model, tok, rendered, batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    prev_side = tok.padding_side
    tok.padding_side = "right"
    pooled_chunks: list[torch.Tensor] = []
    n_used = 0
    try:
        for i in range(0, len(rendered), batch_size):
            r_chunk = rendered[i : i + batch_size]
            c_chunk = completions[i : i + batch_size]
            seqs: list[EncodedSeq] = []
            for r, c in zip(r_chunk, c_chunk):
                if not c.strip():
                    continue
                p_ids = tok(r, **encode_kwargs(tok, r))["input_ids"]
                c_ids = tok(c, add_special_tokens=False)["input_ids"]
                if not c_ids:
                    continue
                ids = (p_ids + c_ids)[:max_len]
                if len(ids) <= len(p_ids):  # completion fully truncated away
                    continue
                seqs.append(EncodedSeq(ids=ids, prompt_len=len(p_ids)))
            if not seqs:
                continue
            batch = collate(seqs, tok.pad_token_id)
            with capture_layer_outputs(model, [layer]) as cap:
                model(input_ids=batch.input_ids.to(device),
                      attention_mask=batch.attention_mask.to(device))
            pooled = _pool(
                cap[layer], batch.attention_mask.to(device),
                "completion_mean", batch.prompt_lens.to(device),
            )
            pooled_chunks.append(pooled.float().cpu())
            n_used += len(seqs)
    finally:
        tok.padding_side = prev_side
    if not pooled_chunks:
        raise RuntimeError(
            "DiD condition captured no usable (non-empty-completion) rows — "
            f"system={'present' if system else 'absent'}, {len(prompts)} prompts"
        )
    acts = torch.cat(pooled_chunks, dim=0)  # [N, d]
    return acts.mean(dim=0), n_used


def _cluster_separation(a: torch.Tensor, b: torch.Tensor,
                        direction: torch.Tensor) -> dict:
    """How well ``direction`` separates two pooled sets — a d'-style read on
    the DiD vector against the jb SI-present/absent means (informational
    diagnostic, no gate)."""
    u = direction / (direction.norm() + 1e-8)
    return {
        "proj_gap": float((a - b) @ u),
        "dir_norm": float(direction.norm()),
    }


def train_did(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    log_fn=None,
    jb_prompts: list[str] | None = None,
    jb_split: str = "val",
    n_jb: int | None = None,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    calib_prompts: int = 16,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    epochs: int | None = None,  # train_kwargs.default compat (documented no-op)
    max_len: int | None = None,  # train_kwargs.default compat (documented no-op)
):
    """Registry trainer: capture four conditions, form the DiD vector,
    calibrate its base scale on the canary coherence procedure, and bake the
    scale into a SteeringVector wrapped as a VectorObject (all-positions).

    ``jb_prompts`` is a test seam: production leaves it None and loads
    HarmBench VAL behaviours; a caller (unit tests) may inject synthetic jb
    prompts to avoid the external dependency. epochs/max_len are accepted
    only because train_kwargs.default carries them for the gradient arms.
    """
    from ..methods import VectorObject  # local: methods imports this module

    del epochs  # closed-form (no gradient training); documented no-op
    benign_prompts = [p.prompt for p in pairs]
    if not benign_prompts:
        raise ValueError("DiD extraction got an empty pairs list")

    if jb_prompts is None:
        from ..eval.jailbreak_data import load_harmbench

        jb_prompts = [g.text for g in load_harmbench(split=jb_split)]
    if not jb_prompts:
        raise RuntimeError(
            f"DiD found no jb prompts (HarmBench split={jb_split!r}); the "
            "adversarial inner difference cannot be formed"
        )
    if n_jb is not None:
        jb_prompts = jb_prompts[:n_jb]

    mlen = max_len or 768
    cap_kw = dict(max_new_tokens=max_new_tokens, batch_size=batch_size,
                  max_len=mlen)
    e_benign_si, n_bs = _condition_pooled(
        model, tok, benign_prompts, instruction.text, layer, **cap_kw)
    e_benign_no, n_bn = _condition_pooled(
        model, tok, benign_prompts, None, layer, **cap_kw)
    e_jb_si, n_js = _condition_pooled(
        model, tok, jb_prompts, instruction.text, layer, **cap_kw)
    e_jb_no, n_jn = _condition_pooled(
        model, tok, jb_prompts, None, layer, **cap_kw)

    v_raw = did_combine(e_jb_si, e_jb_no, e_benign_si, e_benign_no)
    if float(v_raw.norm()) < 1e-6:
        print("[did] WARNING: DiD vector is near-zero — the SI shifts the jb "
              "and benign contexts almost identically (no context-specific "
              "excess to steer with)")

    # Calibrate the base scale on the benign, SI-absent renderings (the eval
    # distribution: unprompted benign inputs), all-positions like CAA.
    device = next(model.parameters()).device
    v_dev = v_raw.to(device)
    calib_rendered = [build_chat_prompt(tok, p, system=None)
                      for p in benign_prompts[:calib_prompts]]

    def _steered_generate(scale: float) -> list[str]:
        return generate_batch(
            model, tok, calib_rendered, sv={layer: v_dev}, scale=scale,
            positions="all", batch_size=batch_size,
            max_new_tokens=min(max_new_tokens, 128),
        )

    calib = calibrate_base_scale(
        model, tok, calib_rendered, _steered_generate,
        scale_ladder=scale_ladder, rep_threshold=rep_threshold,
        ppl_guard_factor=ppl_guard_factor,
        max_new_tokens=min(max_new_tokens, 128), batch_size=batch_size,
        log_fn=log_fn, arm="did",
    )
    base_scale = calib["base_scale"]

    metadata = {
        "construction": CONSTRUCTION,
        "pooling": "completion_mean",
        "jb_source": f"harmbench:{jb_split}",
        "base_scale": base_scale,
        "n_benign_si": n_bs, "n_benign_no": n_bn,
        "n_jb_si": n_js, "n_jb_no": n_jn,
        "n_benign_prompts": len(benign_prompts),
        "n_jb_prompts": len(jb_prompts),
        "raw_did_norm": float(v_raw.norm()),
        "jb_si_effect_norm": float((e_jb_si - e_jb_no).norm()),
        "benign_si_effect_norm": float((e_benign_si - e_benign_no).norm()),
        **_cluster_separation(e_jb_si, e_jb_no, v_raw),
        **{k: v for k, v in calib.items() if k != "warnings"},
        "warnings": calib["warnings"],
    }
    # Bake the calibrated base scale into the stored vector: the eval factor
    # grid then multiplies it (base_scale x factor), matching the antidote
    # arm's "factor grid multiplies the stored base scale" treatment while
    # keeping DiD a plain all-positions VectorObject downstream.
    sv = SteeringVector(
        vectors={layer: (v_raw * base_scale).float().cpu()},
        method="did",
        model_name=getattr(model.config, "_name_or_path", ""),
        instruction_id=instruction.id,
        metadata=metadata,
    )
    return VectorObject(sv)
