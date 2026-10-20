"""Residual-stream steering hooks.

Adds per-layer vectors to decoder-block outputs. Supports:
- position modes: "all" (default; prompt + generation), "response" /
  "prompt" (relative to a prompt length), "front" (ActAdd-style
  per-position injection at absolute positions 0..P-1), "last"
  (Jailbreak Antidote, arXiv 2410.02298: the CURRENT last token only —
  during a multi-token prefill the final real prompt token per row,
  during KV-cached decode every incremental forward's single position);
- interventions: plain addition (h + scale*v) or Stolfo et al. Eq. 2
  projection-targeting (h + scale*(target - h.u)*u with u the unit
  vector), selected per handle via `project_targets`;
- KV-cached generation (a position tracker counts tokens across
  incremental forwards) and full-sequence training forwards (per-sample
  prompt lengths);
- a header-steered response boundary: pass ``response_header_len=H``
  (see the module-level ``response_header_len()``) and the handle
  derives the boundary at each prefill as prompt_width - H, so the
  trailing assistant-header tokens are steered and the FIRST generated
  token is sampled from steered logits;
- gradient flow to the vector (used by BiDPO / distillation trainers);
- optional norm-relative scaling and BOS exclusion (HyperSteer applies
  vectors at all positions except BOS).

This module is also the ONE home of the fit-time position mask for
intervention objects that do NOT apply through ``steering()`` (PSR / LoRA
— single- and all-layer — / LoReFT / LoFiT hooks): :class:`PositionMaskedFit`
gives such an object a
vectorless :class:`SteeringHandle` whose only job is position tracking and
``_position_mask`` — the single place a mask is computed — so the object's
own hooks and the trainers' loss gathers (``distill.loss_gather_masks``)
consume the SAME mask and can never disagree.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
from torch import nn

from .models import apply_chat_template_folding, get_decoder_layers
from .vectors import SteeringVector

POSITION_MODES = ("all", "response", "prompt", "front", "last")

#: The fit's mask-leg vocabulary (the train half of the train x test 2x2):
#: "all" is each arm's own historical fit; "response" is the decode-only
#: leg (header-steered when a response_header_len accompanies it).
FIT_POSITION_MODES = ("all", "response")


def response_header_len(tok) -> int:
    """H — trailing assistant-header tokens. Boundary = prompt_width - H.

    Rendered-diff first: the ``add_generation_prompt=True`` render minus
    the ``False`` render counts the appended assistant header. The marker
    fallback below is MANDATORY, not defensive: the NousResearch
    Meta-Llama-3.1-8B-Instruct mirror — the two-SI target pin
    since 2026-08-20 (configs/two_si.yaml @ d10aef79; the runs completed
    BEFORE then ran meta-llama/Llama-3.1-8B-Instruct, whose template
    honors the flag, so THEY derived H = 4 through this diff path — every
    recorded header of those artifacts says meta-llama) — ships a
    stripped chat template that appends the assistant
    header UNCONDITIONALLY and ignores ``add_generation_prompt``: the
    two renders come back identical, the diff is 0, and a diff-only
    implementation silently collapses the boundary to prompt_width — the
    whole assistant header lands on the prompt side and the FIRST
    GENERATED TOKEN is sampled from unsteered logits (for a refusal
    defense, the decisive token). On the new pin the marker fallback IS
    the live path, and it still derives H = 4.
    tests/test_mask_parity.py pins both paths.
    """
    if not getattr(tok, "chat_template", None):
        return 0
    # encode_kwargs lives in .generate, which imports steering() from this
    # module — import lazily to keep the module graph acyclic.
    from .generate import encode_kwargs

    msgs = [{"role": "user", "content": "x"}]
    full = apply_chat_template_folding(tok, msgs, add_generation_prompt=True)
    head = apply_chat_template_folding(tok, msgs, add_generation_prompt=False)
    n_full = len(tok(full, **encode_kwargs(tok, full))["input_ids"])
    n_head = len(tok(head, **encode_kwargs(tok, head))["input_ids"])
    h = max(n_full - n_head, 0)
    if h:
        return h
    # Flag-ignoring template (the NousResearch mirror): locate the LAST
    # header-start marker and count tokens from it. "<|turn>" is gemma-4's
    # turn token (gemma-2/3 use "<start_of_turn>") -- its pinned template
    # honors add_generation_prompt TODAY, so the diff path serves it, but a
    # flag-ignoring gemma-4 mirror would otherwise fall through a marker
    # list missing its turn token and silently collapse the boundary to
    # prompt_width (the exact configuration the warning above records).
    # "<|assistant|>" is GLM-4.x's role marker — same precautionary story:
    # the pinned GLM-4.6V template honors the flag (diff path derives H=5
    # under the thinking-off pin), a flag-ignoring mirror must not fall
    # through.
    for marker in ("<|start_header_id|>", "<|im_start|>", "<start_of_turn>",
                   "<|turn>", "<|assistant|>"):
        cut = full.rfind(marker)
        if cut < 0:
            continue
        n_before = len(tok(full[:cut], **encode_kwargs(tok, full[:cut]))["input_ids"])
        return max(n_full - n_before, 0)
    return 0


# prompt-width policies for trackers that derive the prompt width from the
# PREFILL EVENT itself (the first multi-token forward) instead of a plumbed
# prompt_len — generate_with_object enters apply() before tokenizing, so a
# plumbed width cannot exist there:
#   None       — prompt_len is whatever the caller set (SteeringHandle's mode)
#   "prefill"  — prompt_len := prefill seq_len (the padded prompt width)
#   "anchor"   — prompt_len := prefill seq_len - 1 (upstream persona-vector
#                position -1 semantics: the LAST prompt token counts as
#                response)
PROMPT_WIDTH_POLICIES = (None, "prefill", "anchor")


def _hidden_from_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


class PositionTracker:
    """Shared position/mask bookkeeping for hooks that act per-position on
    residual states across KV-cached generation (pure refactor out of
    SteeringHandle, so every position-aware hook derives its masks from
    the SAME machinery and none can drift).

    Owns: the global position counter keyed on the prefill event (a
    multi-token forward resets it; decode steps keep counting), the
    model-boundary attention-mask (and optionally input_ids) grab, per-row
    true-BOS resolution under left padding, the position-mode mask, and —
    for subclasses that opt in — the prompt width derived authoritatively
    from the prefill event.

    Subclasses register the two tracking hooks themselves (the tracker owns
    the callbacks, not the registration): layer-0 forward pre-hook ->
    `_track_positions`, model forward pre-hook -> `_track_model_inputs`.
    """

    def __init__(
        self,
        positions: str = "all",
        prompt_len: int = 0,
        skip_bos: bool = False,
        response_header_len: int | None = None,
        track_input_ids: bool = False,
        prompt_width_policy: str | None = None,
        use_cache_position: bool = False,
    ):
        if positions not in POSITION_MODES:
            raise ValueError(f"positions must be one of {POSITION_MODES}")
        if prompt_width_policy not in PROMPT_WIDTH_POLICIES:
            raise ValueError(
                f"prompt_width_policy must be one of {PROMPT_WIDTH_POLICIES}"
            )
        self.positions = positions
        self.prompt_len = prompt_len
        self.skip_bos = skip_bos
        # When set, prompt_len is DERIVED per prefill as (prefill_width - H),
        # H = response_header_len: the assistant-header scaffolding is a
        # CONSTANT-length suffix, and generation's forced LEFT padding
        # right-aligns every row, so the last H columns are the header for
        # the whole batch and one scalar boundary is exact per row.
        # OFF-DISTRIBUTION BY CONSTRUCTION for the historical all-position
        # fits: their training steered the prompt too and only restricted
        # the LOSS to completion tokens. (An object fit under the
        # PositionMaskedFit "response" leg IS on-distribution here.)
        self.response_header_len = response_header_len
        self._track_input_ids = track_input_ids
        self._prompt_width_policy = prompt_width_policy
        # Prefill detection via the layer kwargs' cache_position (the exact
        # event: index 0 means the sequence starts here), with the seq_len>1
        # shape heuristic only as fallback (review I-P2). Opt-in per
        # subclass: reading cache_position[0] costs a device sync per
        # forward, which the plain SteeringHandle should not pay.
        self._use_cache_position = use_cache_position
        self._prompt_lens: torch.Tensor | None = None  # [B], training mode
        self._seen = 0  # tokens consumed so far in an incremental generate
        self._cur_offset = 0
        self._attn_mask: torch.Tensor | None = None  # [B, S] 2D padding mask
        self._cache_len: int | None = None  # KV length at the model boundary
        self._input_ids: torch.Tensor | None = None  # this forward's ids

    # -- position bookkeeping -------------------------------------------------

    def _track_model_inputs(self, module, args, kwargs):
        mask = kwargs.get("attention_mask")
        self._attn_mask = mask if (mask is not None and mask.dim() == 2) else None
        # The KV cache's length at the model boundary is the AUTHORITATIVE
        # global offset of the forward about to run (0 for a prefill /
        # full-sequence forward, prompt+generated-so-far for a decode
        # step) — captured here because generate passes the cache through
        # the model call, and preferred over the local token counter in
        # _track_positions. This is what keeps positions exact when the
        # handle is installed MID-generation: generated-only serving
        # (2026-08-20) enters the steering ctx only for the decode phase,
        # where a fresh counter would restart at 0 and misread the first
        # decode step as the BOS position (skip_bos then silently masked
        # the first generated token).
        pkv = kwargs.get("past_key_values")
        get_len = getattr(pkv, "get_seq_length", None)
        self._cache_len = int(get_len()) if get_len is not None else None
        if self._track_input_ids:
            ids = kwargs.get("input_ids", args[0] if args else None)
            self._input_ids = ids if (ids is not None and ids.dim() == 2) else None
        return None

    def _track_positions(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        seq_len = hidden.shape[1]
        cache_position = (
            kwargs.get("cache_position") if self._use_cache_position else None
        )
        if cache_position is not None and cache_position.numel():
            # The exact prefill event (review I-P2): cache position 0 means
            # this forward starts the sequence — catching one-token prompts
            # (seq_len == 1 prefills) and not mistaking a chunked-prefill
            # continuation (seq_len > 1, cache_position[0] > 0) for a fresh
            # sequence.
            is_prefill = int(cache_position[0]) == 0
        elif self._cache_len is not None:
            # The KV length captured at the model boundary is the same
            # event read the other way: 0 means this forward starts the
            # sequence (and >0 that it continues one — including a handle
            # installed MID-generation, whose first tracked forward is a
            # decode step and must NOT be mistaken for a prefill).
            is_prefill = self._cache_len == 0
        else:
            is_prefill = seq_len > 1
        if self.response_header_len is not None and is_prefill and seq_len > 1:
            # last H columns of THIS prefill are the assistant header
            self.prompt_len = max(seq_len - self.response_header_len, 0)
        if is_prefill:
            # A multi-token forward is a PREFILL (or a full-sequence training
            # forward): the start of a new sequence. Resetting here keys the
            # counter on that event, so a handle held open across several
            # generate calls / training batches keeps positions exact —
            # without it, skip_bos only masked the FIRST forward's BOS.
            # (KV-cached decode steps are seq_len == 1 and keep counting.)
            self._seen = 0
            if self._prompt_width_policy == "prefill":
                self.prompt_len = seq_len
            elif self._prompt_width_policy == "anchor":
                self.prompt_len = seq_len - 1
            self._on_prefill(seq_len)
        if self._cache_len is not None:
            # authoritative global positions from the KV cache length
            # captured at the model boundary (see _track_model_inputs)
            self._cur_offset = self._cache_len
            self._seen = self._cache_len + seq_len
        else:
            self._cur_offset = self._seen
            self._seen += seq_len
        return None

    def _on_prefill(self, seq_len: int) -> None:
        """Subclass hook, fired at every prefill event (after the width
        policy applied, before the counter advances)."""

    def reset(self, prompt_len: int | None = None) -> None:
        """Reset the generation position counter (call before each generate)."""
        self._seen = 0
        self._cur_offset = 0
        self._prompt_lens = None
        self._attn_mask = None
        self._cache_len = None
        self._input_ids = None
        if prompt_len is not None:
            self.prompt_len = prompt_len

    def set_prompt_lens(self, lens: torch.Tensor) -> None:
        """Per-sample prompt lengths for a single full-sequence forward."""
        if self.positions == "last":
            # A full-sequence TRAINING forward has no "current last token":
            # the mode is defined over prefill + incremental decode only
            # (Jailbreak Antidote is a runtime intervention). Refuse loudly
            # rather than steer some position the method never specified.
            raise RuntimeError(
                "positions='last' cannot honor a full-sequence training "
                "forward (set_prompt_lens); it is defined for prefill + "
                "KV-cached decode only"
            )
        self._prompt_lens = lens

    # -- masking ---------------------------------------------------------------

    def _bos_positions(self, batch: int, device) -> torch.Tensor:
        """[B] global index of the first REAL (non-pad) position per row.

        argmax over the 2D attention mask finds the first 1, which is the
        true BOS under LEFT padding (index pad_len) and under right padding
        (index 0) alike. Without a usable mask (unbatched/unpadded
        forwards) it falls back to index 0, which is exact there.
        """
        am = self._attn_mask
        if am is not None and am.shape[0] == batch:
            return am.argmax(dim=1).to(device)
        return torch.zeros(batch, dtype=torch.long, device=device)

    def _position_mask(self, batch: int, seq_len: int, device) -> torch.Tensor | None:
        """[B, S] float mask of positions to steer, or None for all."""
        need_global = (
            self.positions != "all"
            or self.skip_bos
            or self.response_header_len is not None
        )
        if not need_global:
            return None
        idx = torch.arange(
            self._cur_offset, self._cur_offset + seq_len, device=device
        )  # [S] global positions (valid when no per-sample lens are set)

        if self.positions == "last":
            # Jailbreak Antidote semantics (arXiv 2410.02298): steer the
            # CURRENT last token only. A 1-token forward is a KV-cached
            # decode step — its single position IS the sequence's last
            # token, so every step is steered. A multi-token forward is a
            # prefill: steer each row's last REAL position (from the 2D
            # padding mask when available — exact under right padding;
            # without a usable mask, seq_len-1, exact for unbatched and
            # left-padded forwards, which is what generate.py produces).
            if self._prompt_lens is not None:  # belt for set_prompt_lens
                raise RuntimeError(
                    "positions='last' cannot honor per-sample prompt "
                    "lengths (full-sequence training forward)"
                )
            if seq_len == 1:
                mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
            else:
                mask = torch.zeros(
                    batch, seq_len, dtype=torch.bool, device=device
                )
                am = self._attn_mask
                if am is not None and tuple(am.shape) == (batch, seq_len):
                    last = (
                        seq_len - 1 - am.flip(1).long().argmax(dim=1)
                    ).to(device=device, dtype=torch.long)
                else:
                    last = torch.full(
                        (batch,), seq_len - 1, dtype=torch.long, device=device
                    )
                mask[torch.arange(batch, device=device), last] = True
            if self.skip_bos:
                mask = mask & (
                    idx[None, :] != self._bos_positions(batch, device)[:, None]
                )
            return mask.float()

        if self._prompt_lens is not None:
            local = torch.arange(seq_len, device=device)[None, :]
            lens = self._prompt_lens.to(device)[:, None]
            if self.positions == "response":
                mask = local >= lens
            elif self.positions == "prompt":
                mask = local < lens
            else:
                mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
            mask = mask.expand(batch, seq_len).clone()
            if self.skip_bos:
                # Full-sequence forward: mask each row's first real position
                # (position 0 when right-padded, pad_len when left-padded).
                mask = mask & (local != self._bos_positions(batch, device)[:, None])
            return mask.float()

        if self.positions == "response":
            mask = idx >= self.prompt_len
        elif self.positions == "prompt":
            mask = idx < self.prompt_len
        else:  # "all" (here only when skip_bos) or "front" scalar fallback
            mask = torch.ones_like(idx, dtype=torch.bool)
        mask = mask[None, :].expand(batch, seq_len)
        if self.skip_bos:
            # Per-row true-BOS exclusion in GLOBAL coordinates: exact during
            # the prefill; decode steps sit past every BOS and are unaffected.
            mask = mask & (idx[None, :] != self._bos_positions(batch, device)[:, None])
        return mask.float()


class SteeringHandle(PositionTracker):
    """Live steering hooks on a model. Use via the `steering()` context manager.

    For KV-cached generation with position-dependent modes, call
    `reset(prompt_len)` before each `generate` call. For full-sequence
    (training) forwards, call `set_prompt_lens(lens)` with a [B] tensor;
    positions are then absolute within each forward.
    """

    def __init__(
        self,
        model: nn.Module,
        vectors: dict[int, torch.Tensor],
        scale: float | torch.Tensor = 1.0,
        positions: str = "all",
        prompt_len: int = 0,
        norm_relative: bool = False,
        skip_bos: bool = False,
        project_targets: dict[int, float] | None = None,
        response_header_len: int | None = None,
    ):
        super().__init__(positions=positions, prompt_len=prompt_len,
                         skip_bos=skip_bos,
                         response_header_len=response_header_len)
        if positions == "last" and any(
            torch.is_tensor(v) and v.dim() == 2 for v in vectors.values()
        ):
            # A [P, d] per-position matrix is the ActAdd front-injection
            # carrier; under "last" it would silently take the 2-D hook
            # path and ignore the mode. Loud, never silent.
            raise ValueError(
                "positions='last' requires [d]-shaped vectors; got a "
                "per-position [P, d] matrix (ActAdd-style), which this "
                "mode cannot honor"
            )
        self.model = model
        self.vectors = vectors
        self.scale = scale
        self.norm_relative = norm_relative
        self.project_targets = project_targets or {}
        self._handles: list = []

        layers = get_decoder_layers(model)
        if positions != "all" or skip_bos or response_header_len is not None:
            # The tracker must fire once per forward pass, before any steered
            # layer. Layer 0 always runs, whether or not it is steered.
            self._handles.append(
                layers[0].register_forward_pre_hook(
                    self._track_positions, with_kwargs=True
                )
            )
            # skip_bos needs the 2D padding mask: under generate.py's forced
            # LEFT padding the true BOS sits at (pad_len), not at global
            # index 0, so the first REAL position must be found per row. The
            # 2D mask only exists at the model boundary — the per-layer mask
            # is a 4D causal tensor or None — so grab it there.
            self._handles.append(
                model.register_forward_pre_hook(
                    self._track_model_inputs, with_kwargs=True
                )
            )
        for idx, vec in vectors.items():
            self._handles.append(
                layers[idx].register_forward_hook(self._make_hook(idx, vec))
            )

    # -- the hook itself -------------------------------------------------------

    def _make_hook(self, layer_idx: int, vec: torch.Tensor):
        def hook(module, args, output):
            hidden = _hidden_from_output(output)
            batch, seq_len, _ = hidden.shape
            v = vec.to(device=hidden.device, dtype=hidden.dtype)

            if v.dim() == 2:  # [P, d] per-position (ActAdd front injection)
                start = self._cur_offset if self.positions == "front" else 0
                p = v.shape[0]
                lo, hi = start, min(start + seq_len, p)
                if lo >= hi:
                    return output
                add = torch.zeros_like(hidden)
                add[:, lo - start : hi - start, :] = v[lo:hi] * self.scale
                return _replace_hidden(output, hidden + add)

            mask = self._position_mask(batch, seq_len, hidden.device)

            if layer_idx in self.project_targets:
                # Stolfo Eq. 2: move each position's projection onto u to the
                # target value observed under the explicit instruction.
                u = (v / (v.norm() + 1e-8)).float()
                proj = hidden.float() @ u  # [B, S]
                delta = (self.project_targets[layer_idx] - proj) * self.scale
                addition = delta.unsqueeze(-1) * u.view(1, 1, -1)
                addition = addition.to(hidden.dtype)
            elif self.norm_relative:
                unit = v / (v.norm() + 1e-8)
                h_norm = hidden.norm(dim=-1, keepdim=True)  # [B, S, 1]
                addition = unit * (h_norm * self.scale)
            else:
                addition = (v * self.scale).view(1, 1, -1)

            if mask is None:
                return _replace_hidden(output, hidden + addition)
            return _replace_hidden(
                output, hidden + addition * mask.unsqueeze(-1).to(hidden.dtype)
            )

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@contextlib.contextmanager
def steering(
    model: nn.Module,
    sv: SteeringVector | dict[int, torch.Tensor] | None,
    scale: float | torch.Tensor = 1.0,
    positions: str = "all",
    prompt_len: int = 0,
    norm_relative: bool = False,
    skip_bos: bool = False,
    project_targets: dict[int, float] | None = None,
    response_header_len: int | None = None,
) -> Iterator[SteeringHandle | None]:
    """Context manager applying a steering vector. `sv=None` → no-op passthrough
    (lets eval code treat baseline and steered runs uniformly)."""
    if sv is None:
        yield None
        return
    vectors = sv.vectors if isinstance(sv, SteeringVector) else sv
    handle = SteeringHandle(
        model,
        vectors,
        scale=scale,
        positions=positions,
        prompt_len=prompt_len,
        norm_relative=norm_relative,
        skip_bos=skip_bos,
        project_targets=project_targets,
        response_header_len=response_header_len,
    )
    try:
        yield handle
    finally:
        handle.remove()


class PositionMaskedFit:
    """Mixin: the fit-time position-mask leg for hook objects that do not
    apply through ``steering()`` (PSR gates, rank-1 LoRA single- and
    all-layer, LoReFT, LoFiT). A steering()-backed object
    (multilayer.MultiLayerVectorObject) may also take the mixin for its
    ``set_fit_positions`` bookkeeping while routing the mask through its
    OWN live steering handle — same single mask home either way
    (``SteeringHandle._position_mask``).

    The mask leg is a FIT configuration, set by the trainer/builder via
    :meth:`set_fit_positions` (default ``"all"`` — the arm's historical
    fit, bit-identical to the pre-mixin behavior). Under ``"response"``
    the object's ``apply()`` must:

    1. create the tracker via :meth:`_fit_mask_handle` — a VECTORLESS
       :class:`SteeringHandle` (``vectors={}`` steers nothing) that
       registers only the position-tracking hooks, so the mask comes from
       ``SteeringHandle._position_mask``, the single place a mask is
       computed, never a reimplementation;
    2. gate its own per-position delta through :meth:`_fit_masked`
       (``torch.where`` — masked-out positions stay BIT-identical to the
       unsteered hidden states, not "+ 0.0");
    3. yield the tracker handle from ``apply()`` so the trainer can set
       per-row boundaries (``handle.set_prompt_lens(steer_prompt_
       boundaries(batch, H))``) and tie the loss to the SAME mask
       (``distill.loss_gather_masks``) — the mask→loss wiring.

    Objects whose intervention has no per-position notion on generated
    tokens (the soft-token arms: a PROMPT-embedded sentinel embedding)
    must NOT take this mixin — their trainers refuse a masked fit loudly
    instead (an arm that silently pretends to be masked is worse than one
    that refuses).
    """

    fit_positions: str = "all"
    fit_response_header_len: int | None = None

    def set_fit_positions(self, positions: str,
                          response_header_len: int | None = None) -> None:
        if positions not in FIT_POSITION_MODES:
            raise ValueError(
                f"fit positions must be one of {FIT_POSITION_MODES}, got "
                f"{positions!r}")
        self.fit_positions = positions
        self.fit_response_header_len = (
            None if positions == "all" else response_header_len)

    def _fit_mask_handle(self, model) -> SteeringHandle | None:
        """The live mask tracker for one ``apply()`` context, or ``None``
        in the default all-positions fit. The caller owns removal."""
        if self.fit_positions == "all":
            return None
        return SteeringHandle(
            model, {}, positions=self.fit_positions,
            response_header_len=self.fit_response_header_len)

    @staticmethod
    def _fit_masked(handle: SteeringHandle | None, steered: torch.Tensor,
                    original: torch.Tensor) -> torch.Tensor:
        """``steered`` where the handle's mask is on, ``original`` (bit-
        identical, via ``torch.where``) where it is off; ``steered``
        unchanged when no handle/mask is live."""
        if handle is None:
            return steered
        b, s = original.shape[0], original.shape[1]
        mask = handle._position_mask(b, s, original.device)
        if mask is None:
            return steered
        return torch.where(mask.bool().unsqueeze(-1), steered, original)
