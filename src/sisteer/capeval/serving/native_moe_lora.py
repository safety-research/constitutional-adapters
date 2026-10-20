"""THE ROUTED-EXPERT LoRA BUFFER PATCH: vLLM's fused-MoE LoRA buffers sized by the
ROUTED rank the served adapters need, not the server-wide ``--max-lora-rank``.

THE PROBLEM (the serving probe on a routed-experts model). vLLM
0.25.1 wraps every ``mlp.experts`` module of a MoE model with ``FusedMoE3DWithLoRA``
(``vllm/lora/layers/fused_moe.py``) and allocates, per MoE layer and adapter slot,
``w13_lora_a [max_loras, E, R, d]``, ``w13_lora_b [max_loras, E, 2I, R]``,
``w2_lora_a [max_loras, E, R, I]`` and ``w2_lora_b [max_loras, E, d, R]`` with
``E`` the model's routed-expert count and ``R = lora_config.max_lora_rank`` — the
SERVER-WIDE cap — in ``_create_lora_a_weights`` / ``_create_lora_b_weights``
(:157-217), whatever the adapters' actual per-expert rank. The harness's adapters put
the full rank ``r`` on the dense modules (attention, the shared expert) and
``r_e = max(1, r // active_experts)`` on each routed expert (FD 6; a difference
adapter ``2·r_e``, ``lora_export._export_general_subtraction``), and the cap is
fitted to the LARGEST dense rank on the server (``vllm_server.fitted_max_lora_rank``)
— so on a model with ``k`` active experts an expert set of rank ``r`` served under a
cap ``R >= r`` has its expert buffers padded from the ``r_e = r // k`` rows it needs
to ``R``, a ``k·R/r``-fold excess: ``E·R·(d + 2I + I + d)·2 B`` per MoE layer and slot
(``standup_manifest``'s expert-adapter bytes law), which with ``E`` in the hundreds
is of order ``E·(2d + 3I)·R`` bytes — tens of GB for a few slots at a cap in the
hundreds — enough to starve the KV cache at the long production windows and OOM /
refuse every full-rank expert-set launch (and leave the low-rank ones short of the
KV coverage the no-chunking invariant needs at the short window).

THE FIX. The kernel path derives R from the BUFFERS, never from the config:
``punica_gpu.add_lora_w13`` / ``add_lora_w2`` read ``max_lora_rank =
lora_a_stacked[0].shape[-2]``; ``_fused_moe_lora`` / the one-shot and small-batch
kernels read ``rank = A0.shape[2]`` and tile ``BLOCK_R = max(next_pow2(rank), 16)``
with ``rank_mask = offs_r < actual_rank``; ``set_lora`` copies an adapter's
``[E, r_e, in]`` factors into the leading ``r_e`` rows (:385-427). So allocating the
expert buffers at the ROUTED rank the served adapters need — the max over adapters
of their per-expert rank — changes no arithmetic (the padding rows were exact zeros,
masked and summed as zeros in fp32) and shrinks the buffers by the cap-to-routed
ratio; a smaller R even unlocks the ``max_lora_rank <= 64`` small-batch GEMV fast
path for decode. Dense modules keep ``max_lora_rank``. Established read-only against
the 0.25.1 source before this module was written; nothing here forks the engine.

THE THREE CLASS PATCHES (installed at IMPORT when the launch env carries
``SISTEER_MOE_LORA_RANK=<R>`` — the composer sets it iff the launch's adapted class
wraps the routed experts, from the registered adapters' own ``rank_pattern`` /
factor files (``vllm_server.fitted_moe_lora_rank``), so an o_proj / qkvo launch's
env, buffers and graph are byte-for-byte the certified ones; the import happens in
the worker process through ``--worker-extension-cls`` BEFORE ``load_model``,
``native_worker``'s fact 6):

1. ``FusedMoEWithLoRA._create_lora_a_weights`` / ``_create_lora_b_weights`` and
   ``FusedMoE3DWithLoRA._create_lora_b_weights`` (each class's OWN definitions —
   the 3-D class inherits ``_create_lora_a_weights``): the original runs
   unchanged (dtype, device, the fully-sharded ``divide``) over a VIEW of the
   ``LoRAConfig`` whose ``max_lora_rank`` reads the routed R; every other attribute
   delegates. Under ``fully_sharded_loras`` with ``tp_size > 1`` a routed R the TP
   group cannot divide is refused by name before vLLM's ``divide`` asserts.
2. ``set_lora`` on both classes: an adapter whose routed factors carry a rank ABOVE
   the composed R is REFUSED by name (:class:`MoeLoraRankError`, the message the
   probe's ``classify_failure`` recognises) — never silently truncated (vLLM's own
   copy would raise a shape error deep in ``copy_``, or, worse, a wider slice would
   read as a fit). The incoming rank is read off the factors themselves (``lora_a
   [E, r, in]`` dim 1 and ``lora_b [E, out, r]`` dim 2, every slice).
3. ``LoRAModelManager.get_dummy_lora_warmup_rank``: the profiling / graph-capture
   dummy adapters are created at ``min(max_lora_rank, 8)`` for EVERY module
   (``lora_model_runner_mixin.maybe_setup_dummy_loras`` @0.25.1), so a routed R
   below 8 would make the expert wrapper's own warmup overflow its buffers; the
   result is clamped to the routed R (the dummies are zeros of any rank; dense
   modules' kernels read their stride off their own buffers, so the clamp changes
   no captured shape). Under fully-sharded TP the original's divisibility
   adjustment is applied to the clamped request and a result above R refuses.

IMPORT-TIME CONTRACT. Importable with neither vllm nor torch present (the composer
module is torch-free and pins :data:`ENV_FLAG` by value). With the flag unset the
import touches nothing (``PATCH_REPORT.armed`` False). With the flag set and vllm
absent, or set to a value that is not a positive int, the import RAISES — a worker
asked to size its buffers must never come up at the padded size with a quiet log.
The worker's ``sisteer_scheduler_config`` report carries :func:`report`'s three
keys, and the launcher's readiness holds every MoE wrapper's buffer R
(``worker_reports.lora_execution_moe_wrappers.*.max_lora_rank``) to the composed
value (``vllm_server.validate_moe_lora_rank``).

None of the four paper models routes experts; the path stays as an OPTION of the
serving library for a checkpoint that does (the fit / export side — ``lora_modules.
per_expert_rank``, ``lora_export``'s ``experts`` adapter class — writes the adapters
this module serves).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: the launch-env knob the composer sets (``vllm_server.MOE_LORA_RANK_ENV`` pins the
#: same string by test): the routed-expert rank the fused-MoE LoRA buffers are
#: allocated at. Unset = vLLM's own sizing (the server-wide cap).
ENV_FLAG = "SISTEER_MOE_LORA_RANK"
#: the one-shot kernel's hard ceiling on the buffer rank (``fused_moe_lora_op.py``
#: ``assert rank <= 128``): a routed R above it could never run
KERNEL_MAX_RANK = 128
#: the report keys the worker extension folds into ``sisteer_scheduler_config``
REPORT_ARMED_KEY = "moe_lora_rank_armed"
REPORT_RANK_KEY = "moe_lora_rank_composed"
REPORT_CLASSES_KEY = "moe_lora_rank_patched_classes"
#: the refusal's wording — ``serving_probe.MOE_RANK_REFUSAL_RE`` matches it
REFUSAL_PHRASE = "routed-expert LoRA rank {incoming} exceeds the composed " + ENV_FLAG + "={routed}"
#: the attribute a wrapped function carries (a second install is a no-op)
_WRAPPED_ATTR = "__sisteer_moe_lora_rank__"
#: the fused-MoE wrapper methods patched on each class that DEFINES them
_ALLOC_METHODS = ("_create_lora_a_weights", "_create_lora_b_weights")
_SET_LORA = "set_lora"
_DUMMY_RANK = "get_dummy_lora_warmup_rank"


class MoeLoraRankError(RuntimeError):
    """A routed-rank sizing that cannot stand: a malformed knob, an adapter above the
    composed R, a TP group that cannot divide it, a missing vLLM surface."""


def routed_rank_from_env(environ: Mapping[str, str]) -> Optional[int]:
    """The composed routed rank, or ``None`` when :data:`ENV_FLAG` is unset / empty.
    A set value that is not an int in ``[1, KERNEL_MAX_RANK]`` RAISES (the composer
    wrote it; a worker must not guess around it)."""
    raw = environ.get(ENV_FLAG)
    if raw is None or not str(raw).strip():
        return None
    try:
        r = int(str(raw).strip())
    except ValueError:
        raise MoeLoraRankError(f"{ENV_FLAG}={raw!r} is not an int — the composer sets this knob; refusing to arm")
    if r < 1 or r > KERNEL_MAX_RANK:
        raise MoeLoraRankError(
            f"{ENV_FLAG}={r} is outside [1, {KERNEL_MAX_RANK}] (the fused-MoE LoRA kernel's rank ceiling); refusing to arm")
    return r


class _RankView:
    """A read-only view of a ``LoRAConfig`` whose ``max_lora_rank`` is the routed R;
    every other attribute delegates to the real config (dtype, fully_sharded_loras,
    max_loras, …), so vLLM's own allocation code runs unchanged."""

    __slots__ = ("_cfg", "_rank")

    def __init__(self, cfg: Any, rank: int) -> None:
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_rank", int(rank))

    @property
    def max_lora_rank(self) -> int:
        return self._rank

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"the routed-rank LoRAConfig view is read-only ({name})")


def _factor_rank(tensors: Sequence[Any], *, a_side: bool) -> int:
    """The largest rank carried by a list of expert factors: ``lora_a`` tensors are
    ``[E, r, in]`` (dim 1), ``lora_b`` tensors ``[E, out, r]`` (dim 2)."""
    ranks: List[int] = []
    for t in tensors:
        shape = tuple(int(x) for x in getattr(t, "shape", ()))
        if len(shape) != 3:
            raise MoeLoraRankError(
                f"a routed-expert LoRA factor has shape {shape}, not the 3-D [E, r, in] / [E, out, r] vLLM 0.25.1 "
                "stacks (lora_weights.pack_moe); the routed rank cannot be read, refusing")
        ranks.append(shape[1] if a_side else shape[2])
    return max(ranks) if ranks else 0


@dataclass
class PatchReport:
    armed: bool = False
    routed_rank: Optional[int] = None
    patched_classes: List[str] = field(default_factory=list)
    patched_methods: Dict[str, List[str]] = field(default_factory=dict)
    manager_patched: bool = False


@dataclass(frozen=True)
class PatchTargets:
    """The classes to patch: the fused-MoE LoRA wrapper classes (each patched for the
    methods it DEFINES) and the LoRA model manager (the dummy warmup rank)."""

    moe_classes: Tuple[type, ...]
    manager_cls: type


def _qual(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _wrap_alloc(cls: type, name: str, routed_rank: int) -> None:
    orig = cls.__dict__[name]

    def wrapped(self, max_loras, lora_config, *args, **kwargs):  # noqa: ANN001 — vLLM's signature
        tp = int(getattr(self, "tp_size", 1) or 1)
        if bool(getattr(self, "fully_sharded", False)) and tp > 1 and routed_rank % tp:
            raise MoeLoraRankError(
                f"{ENV_FLAG}={routed_rank} is not divisible by tp_size={tp} under fully_sharded_loras — the fused-MoE "
                "LoRA A buffers shard the rank axis across the TP group (fused_moe.py _create_lora_a_weights); refusing "
                "to allocate")
        return orig(self, max_loras, _RankView(lora_config, routed_rank), *args, **kwargs)

    wrapped.__name__ = getattr(orig, "__name__", name)
    wrapped.__qualname__ = getattr(orig, "__qualname__", name)
    wrapped.__doc__ = getattr(orig, "__doc__", None)
    setattr(wrapped, _WRAPPED_ATTR, routed_rank)
    setattr(cls, name, wrapped)


def _wrap_set_lora(cls: type, routed_rank: int) -> None:
    orig = cls.__dict__[_SET_LORA]

    def wrapped(self, index, lora_a, lora_b, *args, **kwargs):  # noqa: ANN001 — vLLM's signature
        a_list = list(lora_a) if isinstance(lora_a, (list, tuple)) else [lora_a]
        b_list = list(lora_b) if isinstance(lora_b, (list, tuple)) else [lora_b]
        incoming = max(_factor_rank(a_list, a_side=True), _factor_rank(b_list, a_side=False))
        if incoming > routed_rank:
            raise MoeLoraRankError(
                REFUSAL_PHRASE.format(incoming=incoming, routed=routed_rank)
                + f" on {type(self).__name__} slot {index}: the fused-MoE LoRA buffers were allocated at the routed rank "
                "the registered adapters declared (rank_pattern), and a wider adapter is never truncated into them; "
                "re-compose the launch with every served adapter's routed rank")
        return orig(self, index, lora_a, lora_b, *args, **kwargs)

    wrapped.__name__ = getattr(orig, "__name__", _SET_LORA)
    wrapped.__qualname__ = getattr(orig, "__qualname__", _SET_LORA)
    wrapped.__doc__ = getattr(orig, "__doc__", None)
    setattr(wrapped, _WRAPPED_ATTR, routed_rank)
    setattr(cls, _SET_LORA, wrapped)


def _wrap_dummy_rank(manager_cls: type, routed_rank: int) -> None:
    orig = manager_cls.__dict__[_DUMMY_RANK]

    def wrapped(self, default_rank, *args, **kwargs):  # noqa: ANN001 — vLLM's signature
        asked = min(int(default_rank), routed_rank)
        got = int(orig(self, asked, *args, **kwargs))
        if got > routed_rank:
            raise MoeLoraRankError(
                f"the dummy LoRA warmup rank {got} (adjusted for fully-sharded TP from {asked}) exceeds "
                f"{ENV_FLAG}={routed_rank}: the expert wrappers' warmup adapters would not fit their buffers; refusing")
        return got

    wrapped.__name__ = getattr(orig, "__name__", _DUMMY_RANK)
    wrapped.__qualname__ = getattr(orig, "__qualname__", _DUMMY_RANK)
    wrapped.__doc__ = getattr(orig, "__doc__", None)
    setattr(wrapped, _WRAPPED_ATTR, routed_rank)
    setattr(manager_cls, _DUMMY_RANK, wrapped)


def install_patches(targets: PatchTargets, routed_rank: int) -> PatchReport:
    """Patch every fused-MoE class for the allocation / ``set_lora`` methods it
    DEFINES in its own ``__dict__`` (an inherited method is patched on the class that
    owns it, once) and the manager's dummy warmup rank. Idempotent for the same R;
    a different R on an already-patched class is a refusal (one engine, one R)."""
    routed_rank = int(routed_rank)
    if routed_rank < 1 or routed_rank > KERNEL_MAX_RANK:
        raise MoeLoraRankError(f"routed rank {routed_rank} outside [1, {KERNEL_MAX_RANK}]")
    rep = PatchReport(armed=True, routed_rank=routed_rank)
    for cls in targets.moe_classes:
        done: List[str] = []
        for name in _ALLOC_METHODS + (_SET_LORA,):
            fn = cls.__dict__.get(name)
            if fn is None:
                continue
            prior = getattr(fn, _WRAPPED_ATTR, None)
            if prior is not None:
                if int(prior) != routed_rank:
                    raise MoeLoraRankError(
                        f"{_qual(cls)}.{name} is already patched for routed rank {prior}; refusing to re-patch at {routed_rank}")
                done.append(name)
                continue
            if name == _SET_LORA:
                _wrap_set_lora(cls, routed_rank)
            else:
                _wrap_alloc(cls, name, routed_rank)
            done.append(name)
        if not done:
            raise MoeLoraRankError(
                f"{_qual(cls)} defines none of {_ALLOC_METHODS + (_SET_LORA,)} — not the pinned vLLM 0.25.1 "
                "FusedMoEWithLoRA surface; refusing to arm a patch that would size nothing")
        rep.patched_classes.append(_qual(cls))
        rep.patched_methods[_qual(cls)] = done
    # every wrapper class must END UP with patched allocation + set_lora somewhere in its MRO
    for cls in targets.moe_classes:
        for name in _ALLOC_METHODS + (_SET_LORA,):
            if getattr(getattr(cls, name, None), _WRAPPED_ATTR, None) != routed_rank:
                raise MoeLoraRankError(f"{_qual(cls)}.{name} is not patched after install")
    mgr = targets.manager_cls
    fn = mgr.__dict__.get(_DUMMY_RANK)
    if fn is None:
        raise MoeLoraRankError(
            f"{_qual(mgr)} defines no {_DUMMY_RANK} — not the pinned vLLM 0.25.1 LoRAModelManager surface; the "
            "warmup dummies would overflow the routed buffers, refusing")
    prior = getattr(fn, _WRAPPED_ATTR, None)
    if prior is None:
        _wrap_dummy_rank(mgr, routed_rank)
    elif int(prior) != routed_rank:
        raise MoeLoraRankError(f"{_qual(mgr)}.{_DUMMY_RANK} is already patched for {prior}; refusing {routed_rank}")
    rep.manager_patched = True
    return rep


def vllm_patch_targets() -> PatchTargets:
    """The pinned wheel's classes (raises :class:`MoeLoraRankError` when vllm is not
    importable or a class is missing)."""
    try:
        from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
        from vllm.lora.model_manager import LoRAModelManager
    except ImportError as e:
        raise MoeLoraRankError(
            f"{ENV_FLAG} is set but vllm is not importable here ({e}); the fused-MoE LoRA buffer patch cannot be "
            "installed and a worker asked to size its buffers must not come up at the padded size") from e
    return PatchTargets(moe_classes=(FusedMoEWithLoRA, FusedMoE3DWithLoRA), manager_cls=LoRAModelManager)


def install(routed_rank: int, targets: Optional[PatchTargets] = None) -> PatchReport:
    """Arm the patch at ``routed_rank`` and publish :data:`PATCH_REPORT`."""
    global PATCH_REPORT
    if targets is None:
        targets = vllm_patch_targets()
    rep = install_patches(targets, routed_rank)
    PATCH_REPORT = rep
    log.info("sisteer native_moe_lora: fused-MoE LoRA buffers ARMED at routed rank %d (%s)",
             routed_rank, ", ".join(rep.patched_classes))
    return rep


def reset_for_tests() -> None:
    global PATCH_REPORT
    PATCH_REPORT = PatchReport()


def report() -> Dict[str, Any]:
    """The worker's facts for ``sisteer_scheduler_config``: whether the patch is
    armed in THIS process, the composed routed rank, the patched classes. The
    launcher cross-checks them against the launch env and the MoE wrappers' buffer
    shapes (``vllm_server.validate_moe_lora_rank``)."""
    return {REPORT_ARMED_KEY: bool(PATCH_REPORT.armed),
            REPORT_RANK_KEY: PATCH_REPORT.routed_rank,
            REPORT_CLASSES_KEY: list(PATCH_REPORT.patched_classes)}


def _arm_from_env() -> PatchReport:
    r = routed_rank_from_env(os.environ)
    if r is None:
        return PatchReport(armed=False)
    return install(r)


PATCH_REPORT = _arm_from_env()

__all__ = ["ENV_FLAG", "KERNEL_MAX_RANK", "REPORT_ARMED_KEY", "REPORT_RANK_KEY", "REPORT_CLASSES_KEY",
           "REFUSAL_PHRASE", "MoeLoraRankError", "PatchReport", "PatchTargets", "routed_rank_from_env",
           "install_patches", "install", "vllm_patch_targets", "report", "reset_for_tests", "PATCH_REPORT"]
