"""Batched generation, optionally under steering."""

from __future__ import annotations

import torch

from .hooks import steering
from .vectors import SteeringVector


def encode_kwargs(tok, rendered: str) -> dict:
    """Tokenizer kwargs that avoid a DOUBLE BOS on chat-templated strings.

    A chat template renders the BOS as literal text, so encoding the
    result with the default add_special_tokens=True prepends a SECOND BOS:
    skip_bos then masks BOS#1 while the norm-relative addition lands on
    BOS#2's massive activation — exactly what the batch=1 rule exists to
    prevent — and every position shifts by one. Plain strings (no BOS
    text) keep the default so the tokenizer still adds their only BOS.
    """
    bos = getattr(tok, "bos_token", None)
    if bos and rendered.startswith(bos):
        return {"add_special_tokens": False}
    return {}


def assert_left_padded(attention_mask, *, what: str = "batched generation") -> None:
    """Raise unless a multi-row batch is LEFT-padded (every row carries a real
    token in its final column). Batched generation is correct only under left
    padding — for HF generation generally, and specifically for response-mode
    steering whose mask boundary is (prompt_width - H): that is the assistant
    header ONLY when real tokens are right-aligned. Under RIGHT/mixed padding a
    short row's final H columns are PADDING, so its mask lands in padding and the
    row is served BASELINE while generation succeeds — an invisible failure a
    one-word padding_side change would reintroduce. Measured from the ACTUAL
    mask, not a tokenizer setting elsewhere."""
    am = attention_mask
    if am is None or getattr(am, "dim", lambda: 0)() != 2 or am.shape[0] <= 1:
        return  # single row / no mask: no padding ambiguity
    if not bool(am[:, -1].all()):
        raise RuntimeError(
            f"{what} must be LEFT-padded, but the built batch has padding in a "
            "row's final column. Response-mode steering's mask boundary "
            "(prompt_width - H) would then land in padding and silently serve "
            "BASELINE, and HF batched generation is incorrect under right "
            "padding regardless. Set tokenizer.padding_side='left'.")


# ENGINE NOTE (2026-08-25). This is HF `model.generate`, and it is the ONLY
# generation path for the per-family none/prompted conditions:
#   two_si_run.py generate -> run_static -> defense.generate -> here.
# There is no vLLM branch anywhere in that chain. `generate-pos` (the
# K-candidate positive side) DOES run on vLLM after the engine port, so a
# pool built today is vLLM on the positive side and HF on the none/prompted
# side. That is homogeneous WITHIN each layer -- which is what the
# acceptance comparison needs -- but it is not "all vLLM", and the
# difference is invisible from either CLI. Stated here because this
# function is where someone looking for the engine would land.
@torch.no_grad()
def generate_batch(
    model,
    tok,
    rendered_prompts: list[str],
    sv: SteeringVector | None = None,
    scale: float = 1.0,
    positions: str = "all",
    skip_bos: bool = False,
    norm_relative: bool = False,
    project_targets: dict[int, float] | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[str]:
    """Generate completions for already-rendered prompt strings.

    Left padding so all rows share the same prompt width, which keeps
    position-dependent steering modes consistent across the batch.
    """
    device = next(model.parameters()).device
    prev_side = tok.padding_side
    tok.padding_side = "left"
    outputs: list[str] = []
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    try:
        for i in range(0, len(rendered_prompts), batch_size):
            chunk = rendered_prompts[i : i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      **encode_kwargs(tok, chunk[0])).to(device)
            # tripwire at the batch-build point: we set padding_side="left"
            # above, so this fires only if that coupling is ever broken.
            assert_left_padded(enc.get("attention_mask"),
                               what="batched steered generation")
            prompt_width = enc["input_ids"].shape[1]
            with steering(
                model,
                sv,
                scale=scale,
                positions=positions,
                prompt_len=prompt_width,
                skip_bos=skip_bos,
                norm_relative=norm_relative,
                project_targets=project_targets,
            ) as handle:
                if handle is not None:
                    handle.reset(prompt_len=prompt_width)
                out = model.generate(**enc, **gen_kwargs)
            for row in out[:, prompt_width:]:
                outputs.append(tok.decode(row, skip_special_tokens=True).strip())
    finally:
        tok.padding_side = prev_side
    return outputs


def generate_with_object(
    model,
    tok,
    rendered_prompts: list[str],
    obj=None,
    scale: float = 1.0,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[str]:
    """generate_batch under an intervention OBJECT (obj=None -> plain).

    Token-kind sentinels are injected into every prompt, and obj.apply is
    entered PER CHUNK: one apply() around several generate calls leaves a
    position-tracking handle (VectorObject skip_bos) counting across
    chunks, so only the first chunk's BOS would be masked.
    """
    if obj is None:
        return generate_batch(
            model, tok, rendered_prompts, batch_size=batch_size,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature,
        )
    if (getattr(obj, "kind", "") or "").startswith("token"):
        rendered_prompts = [obj.inject(r) for r in rendered_prompts]
    outputs: list[str] = []
    for i in range(0, len(rendered_prompts), batch_size):
        chunk = rendered_prompts[i : i + batch_size]
        with obj.apply(model, scale=scale):
            outputs.extend(
                generate_batch(
                    model, tok, chunk, batch_size=batch_size,
                    max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=temperature,
                )
            )
    return outputs
