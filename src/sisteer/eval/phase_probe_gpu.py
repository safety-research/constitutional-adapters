"""The GPU forward-pair that feeds the serving-phase probe — the ONE
model-specific call site behind :mod:`sisteer.eval.phase_probe`'s pure
classification. Kept OUT of ``phase_probe.py`` so that module stays model-free and
unit-testable; kept here (not inlined in the sweep driver) so the sweep and the
standalone validation script share ONE forward-pair.

WHAT IT MEASURES. One forward with the object's ctx APPLIED and one WITHOUT, over
the SAME fixed synthetic sequence with a KNOWN prompt/generated split, diffing
hidden states per position. The set of positions whose hidden state CHANGED is the
object's applied footprint — measured, not asserted (:mod:`phase_probe`).

WHY PREFILL + TEACHER-FORCED DECODE (not model.generate). To measure the object's
footprint the two forwards must run over the IDENTICAL token sequence — otherwise
steering changes the generated tokens and the decode-position diff conflates "the
object steered here" with "a different token was fed here". So we TEACHER-FORCE a
fixed continuation: a prefill over the prompt, then the fixed continuation tokens
fed one-at-a-time through the KV cache (each a real seq_len==1 decode step). This
replicates serving EXACTLY — the position-mask boundary is derived at the prefill
as ``prompt_width - H`` (steering the assistant HEADER, so the position that
produces the first generated token is steered) and every KV-cached decode step is
steered — while keeping the two sequences token-identical so the diff is purely the
object's footprint. It is what lets the probe PROVE token 1 is steered under
position-mask serving rather than assume it.

The CONTENT boundary (the footprint's prompt/generated split) is ``prompt_width -
H`` — the assistant HEADER counts as RESPONSE, since a position-mask serve steers
it to shape token 1. H is located via :mod:`sisteer.chat_markers`
(``derive_assistant_markers``), the SAME primitive the fit and Petri lanes bind to,
so our boundary agrees with theirs by construction rather than by a hand-rolled
suffix arithmetic.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional, Tuple

from ..chat_markers import (derive_assistant_markers, generation_header_ids,
                            render_ids)
from ..models import get_decoder_layers
from .phase_probe import PhaseVerificationError

#: A fixed, benign synthetic probe (no harmful content — this only measures which
#: positions an intervention touches, never grades an answer).
_PROBE_USER = "Explain, in one sentence, why the sky appears blue."
_PROBE_CONTINUATION = (
    "The sky appears blue because air scatters short-wavelength blue light "
    "from the sun more than other colors.")


def build_probe_sequence(tok) -> Tuple:
    """A fixed synthetic ``[user prompt + assistant continuation]`` token sequence
    with a KNOWN split. Returns ``(prompt_ids, cont_ids, prompt_width,
    header_len, content_boundary, seq_len)``.

    ``prompt_ids`` is the rendered user turn WITH the assistant generation prompt
    (so it ends with the assistant header, exactly as serving prefills it);
    ``cont_ids`` is the fixed teacher-forced continuation (no special tokens).
    ``header_len`` (H) and thus ``content_boundary = prompt_width - H`` come from
    :func:`sisteer.chat_markers.generation_header_ids` — the GENERATION-PROMPT
    header (the tokens ``add_generation_prompt`` appends), which is the SAME
    width the serve-time mask uses (``hooks.response_header_len`` /
    ``jailbreak._set_response_positions``). It is DELIBERATELY not
    ``derive_assistant_markers().header``: the located header is anchored at
    the previous turn's terminator and, on gemma-4, absorbs a leading ``\n``
    that is PROMPT content (present in both add_generation_prompt renders,
    outside the fit-time mask) — so measuring the header region against the
    located 4-id header while the object steers the 3-id generation header made
    a clean response serve read header='partial' and refused every gemma
    fitted/transferred cell as MISLABELLED (pt829 wave, 2026-08-28; the same
    located-vs-appended disagreement 03184be fixed for the hf_engine arm-time
    guard and generation_header_ids documents — the probe was the one call site
    still substituting the locator's 4). On llama-3.1 located == appended (H=4),
    so this is a no-op there. A zero H is a hard error (the boundary would
    collapse and the measurement would be meaningless)."""
    markers = derive_assistant_markers(tok)
    header_len = len(generation_header_ids(tok, markers))
    prompt_ids = render_ids(tok, [{"role": "user", "content": _PROBE_USER}],
                            add_generation_prompt=True)
    prompt_width = len(prompt_ids)
    content_boundary = prompt_width - header_len
    if header_len <= 0 or content_boundary <= 0:
        raise PhaseVerificationError(
            f"probe sequence has H={header_len}, prompt_width={prompt_width} — "
            "cannot locate an assistant-header boundary, so the measured footprint "
            "would be meaningless. Fix the tokenizer/template (chat_markers) before "
            "probing.")
    cont_ids = tok(_PROBE_CONTINUATION, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        raise PhaseVerificationError("probe continuation tokenized empty")
    seq_len = prompt_width + len(cont_ids)
    return (prompt_ids, cont_ids, prompt_width, header_len, content_boundary,
            seq_len)


def _capture_layer_index(obj, model) -> int:
    """The decoder layer to capture at. The LAST decoder layer — every earlier
    intervention (a residual add at layer L, a weight edit, a gate) propagates
    into it, and causal attention keeps CONTENT positions (strictly before the
    header) unchanged there, so a position-mask serve still reads content=none.
    Robust across object kinds; the object's own steered layer is a subset of what
    the final layer reflects."""
    return len(get_decoder_layers(model)) - 1


def decode_probe_capture(model, tok, obj, *, scale: float = 1.0,
                         layer_index: Optional[int] = None) -> Tuple:
    """Build the serving-phase forward-pair for ``obj`` on ``model``/``tok``.

    Returns ``(prompt_len, seq_len, header_len, ignore_positions, capture)``:
    ``prompt_len`` is the FULL prompt/prefill width (``measure_footprint``'s
    convention: the header is the prompt's last ``header_len`` positions — passing
    the content boundary here slid the header window onto unsteered content and
    mislabelled a clean response serve as MIXED; caught by the first GPU
    validation, 2026-08-21), ``ignore_positions`` is the true-BOS index (0)
    excluded under ``skip_bos`` so an all-position serve still reads full content
    coverage, and
    ``capture(applied: bool)`` runs ONE prefill + teacher-forced-decode pass over
    the fixed probe sequence and returns that pass's per-position hidden states at
    the captured layer (a list of ``[d]`` CPU tensors, length ``seq_len``), WITH
    the object's ctx (``applied=True``) or WITHOUT (``applied=False``).

    ``obj`` is a compiled intervention object exposing ``apply(model, scale)`` (a
    context manager). For a residual-hook object the yielded handle is
    ``reset(prompt_len=prompt_width)``-ed before the prefill, exactly as
    ``generate.py`` does, so the position-mask boundary is derived identically to
    serving."""
    import torch

    (prompt_ids, cont_ids, prompt_width, header_len, content_boundary,
     seq_len) = build_probe_sequence(tok)
    device = next(model.parameters()).device
    cap_idx = layer_index if layer_index is not None else _capture_layer_index(obj, model)
    layer = get_decoder_layers(model)[cap_idx]
    prompt_t = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    def capture(applied: bool):
        captured = []

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h.detach().float().cpu())

        hk = None
        try:
            ctx = obj.apply(model, scale=scale) if applied else nullcontext()
            with torch.no_grad():
                with ctx as steer_handle:
                    # register the capture AFTER the ctx installed its hooks:
                    # forward hooks run in registration order, so a capture
                    # registered first at a STEERED layer records PRE-steering
                    # hidden and the whole probe reads no-op (caught on the tiny
                    # CPU repro where the steered layer IS the last layer).
                    hk = layer.register_forward_hook(hook)
                    if (applied and steer_handle is not None
                            and hasattr(steer_handle, "reset")):
                        steer_handle.reset(prompt_len=prompt_width)
                    out = model(input_ids=prompt_t, use_cache=True)
                    past = out.past_key_values
                    for t in cont_ids:
                        step = torch.tensor([[int(t)]], device=device,
                                            dtype=torch.long)
                        out = model(input_ids=step, past_key_values=past,
                                    use_cache=True)
                        past = out.past_key_values
        finally:
            if hk is not None:
                hk.remove()
        # captured[0] = prefill [1, prompt_width, d]; captured[1:] = K x [1, 1, d].
        per_pos = [captured[0][0, i] for i in range(captured[0].shape[1])]
        for step_h in captured[1:]:
            per_pos.append(step_h[0, -1])
        if len(per_pos) != seq_len:
            raise PhaseVerificationError(
                f"probe capture produced {len(per_pos)} positions, expected "
                f"{seq_len} — the forward pair did not cover the fixed sequence")
        return per_pos

    return prompt_width, seq_len, header_len, (0,), capture


def token1_position(prompt_width: int, header_len: int) -> int:
    """The position whose logits produce the FIRST generated token (token 1): the
    LAST prompt/prefill position, ``prompt_width - 1``. Under position-mask serving
    it is the final assistant-header token and IS steered — the probe proves this by
    checking this index is in the measured changed set. (``header_len`` accepted for
    call-site clarity.)"""
    return prompt_width - 1
