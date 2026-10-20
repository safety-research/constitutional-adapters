"""Runtime patches for bugs in the pinned vLLM that the served path cannot
live with. Each patch is idempotent, targeted to the exact condition that is
wrong, and applied at import of THIS module (which both worker extensions —
:mod:`lens_worker`'s and :class:`SisteerPatchExtension` below — import) so it
lands in the vLLM WORKER process before the model is constructed (``vllm serve`` resolves
``--worker-extension-cls`` in the worker ahead of ``load_model``; ``CustomOp``
binds its dispatch method at ``__init__``, so a class-level replacement must
precede construction to take effect).

M-RoPE interleaved layout (vLLM 0.25.1, still present at 0.26.0)
-----------------------------------------------------------------
``MRotaryEmbedding.forward_cuda`` routes 2-D (t/h/w) positions to
``triton_mrope``, whose kernel takes ``mrope_interleaved`` but has NO
``is_neox_style`` parameter: it always rotates first-half against second-half
channels (the neox layout). Every M-RoPE model before GLM-4.1V is neox-style,
so this went unnoticed; the GLM-4V line is ``is_neox_style=False`` (GPT-J
pairs ``(2i, 2i+1)``), so every attention layer applied the wrong channel
pairing. Because q and k share the wrong pairing the relative-position
property survives and the error grows with distance rather than exploding —
measured on GLM-4.6V-Flash as 0.05 nats at position 1, 4–7 nats by position
20, and a teacher-forced NLL that RISES with position (3.35 → 3.87 over
1.2–4k-token documents) while HF's falls (2.58 → 1.50). ``forward_native``
honours ``is_neox_style``; with the rotary forced onto it vLLM matches HF to
bf16 noise (mean |ΔNLL| 0.02 in every position bin, 32 documents). The fix
below takes that route ONLY for ``is_neox_style=False`` with 2-D positions;
neox M-RoPE models and 1-D positions keep the fused kernel.

SCOPE — which engines were ever wrong. ``forward_cuda`` is reached only when
the rotary CustomOp is ENABLED, and vLLM enables custom ops by default only
when running without Inductor (``enforce_eager=True``, or ``custom_ops``
listing it); a compiled engine (the default ``enforce_eager=False`` with the
Inductor backend, ``custom_ops=['none']``) binds ``forward_native`` at
``__init__`` and never touched the kernel. So: the lens/served path (vllm-lens
forces eager) and any eager in-process engine were wrong and are fixed by
this patch; compiled pool generation was never affected and the patch is
inert there (a peer verified byte-identical completions patched vs
unpatched on a compiled 0.26.0 engine). That is why the attestation reports
the BOUND DISPATCH (``dispatch``/``kernel_reachable``) and a routed-call
counter, not merely that the class patch is installed.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_MARK = "_sisteer_mrope_interleaved_patch"


def patch_mrope_interleaved_layout() -> bool:
    """Route non-neox 2-D-position rotary through ``forward_native``.
    Returns True when the patch is (or already was) in place, False when vLLM
    is not importable here (a non-worker process; nothing to patch)."""
    try:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding)
    except ImportError:
        return False
    if getattr(MRotaryEmbedding, _MARK, False):
        return True
    orig = MRotaryEmbedding.forward_cuda

    def forward_cuda(self, positions, query, key=None, offsets=None):
        if positions.ndim == 2 and not self.is_neox_style:
            # counted so a report taken AFTER a forward can show the native
            # branch actually ran, not just that the wrapper is installed
            self._sisteer_routed_native = getattr(
                self, "_sisteer_routed_native", 0) + 1
            return self.forward_native(positions, query, key, offsets)
        return orig(self, positions, query, key, offsets)

    forward_cuda._sisteer_orig = orig            # for tests / attestation
    forward_cuda.__doc__ = orig.__doc__
    MRotaryEmbedding.forward_cuda = forward_cuda
    setattr(MRotaryEmbedding, _MARK, True)
    log.info("sisteer vllm_patches: MRotaryEmbedding interleaved-layout "
             "patch applied (non-neox 2-D positions -> forward_native)")
    return True


def mrope_patch_report(model) -> dict:
    """What the running model's rotary actually is: patch state plus the
    layer-0 rotary class / layout / mrope_section, so a served run can attest
    that a non-neox M-RoPE model is not on the buggy kernel."""
    try:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            MRotaryEmbedding)
        patched = bool(getattr(MRotaryEmbedding, _MARK, False))
    except ImportError:
        patched = False
    lm = getattr(model, "language_model", model)
    lm = getattr(lm, "model", lm)
    rep: dict = {"patch_applied": patched}
    try:
        # THE FIRST ATTENTION LAYER, not layer 0: hybrid models (Qwen3.5 —
        # gated-deltanet layers carrying `linear_attn`, Nemotron-H — Mamba2
        # layers) interleave recurrent layers that have no `self_attn` and no
        # rotary; layer 0 of Qwen3.5-4B is one, so reading it errored the
        # report and the standup REFUSED to serve a model whose rotary was
        # fine (canary 84787, 2026-09-09). The rotary lives on the attention
        # layers; report which one was read.
        layers = list(lm.layers)
        if not layers:
            raise AttributeError("the model has no decoder layers")
        # vLLM names the attention submodule differently per family:
        # `self_attn` (Qwen, Gemma, GLM text), `mixer` (Nemotron-H, whose
        # attention layers are NemotronHAttention behind the same attribute
        # its Mamba2 / MLP layers use), `attn` elsewhere. A rotary-bearing
        # attention module is any of those with a `rotary_emb`.
        found = None
        n_attn = 0
        for i, layer in enumerate(layers):
            for name in ("self_attn", "mixer", "attn"):
                mod = getattr(layer, name, None)
                if mod is None:
                    continue
                if hasattr(mod, "rotary_emb"):
                    n_attn += 1
                    if found is None:
                        found = (i, name, mod.rotary_emb)
        rep.update(n_layers=len(layers), n_layers_with_rotary=n_attn)
        if found is None:
            # Nemotron-H: attention layers WITHOUT a rotary (positions come
            # from the recurrent layers) — provably nothing for the M-RoPE
            # patch to act on. A report, not an error: the attestation
            # refuses errors, and "no rotary" is the proof it wants.
            rep.update(has_rotary=False, rotary_layer_index=None,
                       rotary_attr=None, rotary_cls=None, needs_patch=False,
                       note="no attention module carries a rotary_emb "
                            "(e.g. Nemotron-H attention without RoPE)")
            return rep
        idx, attr, rot = found
        rep.update(has_rotary=True, rotary_layer_index=idx, rotary_attr=attr)
        # the method CustomOp bound at __init__ — the truth about which branch
        # a forward takes: forward_native (custom op disabled: compiled
        # engines; the kernel is unreachable and the patch inert), or
        # forward_cuda (eager engines: our wrapper when patched, the buggy
        # kernel when not)
        dispatch = getattr(getattr(rot, "_forward_method", None),
                           "__name__", None)
        rep.update(rotary_cls=type(rot).__name__,
                   is_neox_style=bool(getattr(rot, "is_neox_style", True)),
                   mrope_section=list(getattr(rot, "mrope_section", None)
                                      or []),
                   rotary_dim=int(getattr(rot, "rotary_dim", 0)),
                   dispatch=dispatch,
                   kernel_reachable=(dispatch == "forward_cuda"),
                   routed_native_calls=int(getattr(
                       rot, "_sisteer_routed_native", 0)))
        rep["needs_patch"] = (rep["rotary_cls"] == "MRotaryEmbedding"
                              and not rep["is_neox_style"]
                              and rep["kernel_reachable"])
    except Exception as e:  # noqa: BLE001 — a report, never a gate
        rep["error"] = f"{type(e).__name__}: {e}"
    return rep


#: ``worker_extension_cls`` for engines built OUTSIDE the lens server — the
#: plain pool sampler, attackers, gates. vLLM v1 runs the worker in its own
#: process, so a patch applied in the parent never reaches the model; naming
#: this class makes the worker IMPORT this module (applying the patches) before
#: the model is built, and adds the attestation RPC. lens_worker's extension
#: imports this module too, so both routes end up patched the same way.
PATCH_EXTENSION_CLS = ("sisteer.capeval.serving.vllm_patches"
                       ".SisteerPatchExtension")


class SisteerPatchExtension:
    """Mixed into the vLLM worker (``worker_extension_cls``)."""

    def sisteer_mrope_report(self) -> dict:
        return mrope_patch_report(self.model_runner.model)


def engine_kwargs_with_patch(kwargs: dict) -> dict:
    """A copy of ``vllm.LLM`` kwargs that reaches the worker patched: adds
    ``worker_extension_cls`` when the caller has none. An existing one (the
    lens extension) is kept — every harness extension module imports this
    one, so the patches land either way."""
    out = dict(kwargs)
    out.setdefault("worker_extension_cls", PATCH_EXTENSION_CLS)
    return out


class UnpatchedEngineError(RuntimeError):
    """A non-neox M-RoPE model is running on vLLM's neox-only mrope kernel."""


def attest_engine(llm) -> dict:
    """Ask the worker which rotary the model actually runs on and FAIL LOUDLY
    if a non-neox M-RoPE model (GLM-4V) is on the buggy kernel — a served wave
    or an eager pool sampled from the mis-rotated engine is the wrong
    distribution, not a noisy one. The report names the BOUND dispatch, so
    "patch installed but the kernel path is unreachable here" (compiled
    engines) reads as exactly that rather than as a fix at work; call it
    again after a forward to see ``routed_native_calls`` move on an eager
    engine. Returns the worker's report (rank 0)."""
    reps = llm.collective_rpc("sisteer_mrope_report")
    rep = dict(reps[0]) if reps else {}
    if rep.get("needs_patch") and not rep.get("patch_applied"):
        raise UnpatchedEngineError(
            "non-neox M-RoPE model is running on vLLM's neox-only "
            f"triton_mrope kernel (GLM-4V layout bug): {rep}")
    return rep


#: Applied at IMPORT on purpose. vLLM resolves ``worker_extension_cls`` in the
#: worker process ahead of ``load_model`` by importing the class's module, and
#: ``CustomOp`` binds its dispatch at ``__init__`` — so the import of this
#: module is the one moment a class-level patch reaches every rotary instance.
#: False only outside a vLLM process (nothing to patch there). The attestation
#: (``attest_engine``) is what proves it landed: the first verification run
#: had the patch applied only by lens_worker's import, and the plain-sampler
#: extension came up unpatched — caught by the attestation, loudly.
MROPE_PATCH_APPLIED = patch_mrope_interleaved_layout()
