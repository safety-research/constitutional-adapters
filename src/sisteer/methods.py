"""Stage-2 arm registry: one interface every suite iterates over.

The 12 stage-2 method arms plus the two
divergence-ablation arms (vector_rkl / vector_js — distill_soft's exact
cell with the self-distillation target swapped), each as an ArmSpec with a
factory (fresh untrained object), a trainer (pairs in, trained
InterventionObject out), and the suites the arm belongs to. Vector-based
arms are wrapped in the VectorObject adapter so SteeringVector and
object arms are indistinguishable downstream: everything exposes
`apply(model, scale)` / `inject(prompt)` / `save(path)`, and
`load_object` re-hydrates any saved artifact by its stored kind.

Downstream contract: always call `obj.inject(rendered_prompt)` on every
prompt the intervention should act on — identity for all non-token
kinds, sentinel insertion for the two token kinds.

Suite membership:
- ifeval: all 12 arms;
- composition: vector (residual additive sum) + token_faithful (<and>);
- ladder: vector (distill_soft) only.
(lifbench was removed from the campaign: its instruction-after-context
design measures retrieval, not instruction decay — so no arm carries it.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .data.instructions import Instruction
from .data.pairs import (
    Pair,
    docs_to_ce_examples,
    pairs_to_distill_examples,
    pairs_to_pref_examples,
)
from .hooks import steering
from .models import hidden_size, model_device, num_layers
from .vectors import SteeringVector
from .extraction.adasteer import AdaSteerObject, train_adasteer
from .extraction.alphasteer import AlphaSteerObject, train_alphasteer
from .extraction.alphakl import AlphaKLObject, train_alphakl
from .extraction.antidote import AntidoteObject, train_antidote
from .extraction.bidpo import train_bidpo
from .extraction.bidpo_lora import train_bidpo_lora
# CastObject / CircuitBreakerObject are imported for _KIND_LOADERS ONLY:
# the arms left the registry entirely on 2026-08-20 — no ArmSpec, nothing
# trains or enumerates them — but load_object MUST keep hydrating the
# already-fitted on-disk artifacts (another consumer uses the existing
# fitted object cells as a serving/kind-coverage target, and two of those
# cells are cast / circuit_breaker artifacts). Their trainers are
# deliberately NOT imported here; extraction/cast.py and
# extraction/circuit_breaker.py keep them for direct callers and tests.
from .extraction.caa_static_r import (
    PAPER_LLAMA31_LAYERS,
    CaaStaticRObject,
    train_caa_static_r_content,
)
from .extraction.cast import CastObject
from .extraction.circuit_breaker import CircuitBreakerObject
from .extraction.did import train_did
from .extraction.diffmean import DiffMeanObject, train_diffmean
from .extraction.distill import train_distill
from .extraction.lofit import LoFiTObject, train_lofit
from .extraction.multilayer import (LoRASubtractionObject,
                                    MultiLayerLoRAObject,
                                    MultiLayerVectorObject)
from .extraction.objects import (
    RankOneLoRAObject,
    SoftTokenObject,
    train_object_ce,
    train_object_kl,
)
from .extraction.paired_write import PSRPairedWriteObject
from .extraction.psr import PSRObject, init_psr_directions, train_psr
from .extraction.reft import (
    LoReFTObject,
    ReFTr1Object,
    train_loreft,
    train_reft_r1,
)
# StaticRObject is imported for _KIND_LOADERS ONLY (the same reason
# CastObject / CircuitBreakerObject are above): no trainer for the
# `static_r` kind exists on this tree (the Arditi objects are fit and staged
# outside this repo and the arm is SERVED-ONLY,
# defense_cells.SERVED_ONLY_CLASS_B_IDS), but load_object MUST hydrate those
# on-disk artifacts — the fp32 serving oracle reaches them through
# arm_for_object -> load_object, and without the entry every Arditi cell raised
# "unknown intervention kind".
from .extraction.static_r import StaticRObject
from .extraction.tokens import (
    FAITHFUL_LOSS_SCALE,
    FAITHFUL_TEMPERATURE,
    AppendedSoftTokenObject,
)

# -- suites --------------------------------------------------------------------

S_IFEVAL = "ifeval"
S_COMPOSITION = "composition"
S_LADDER = "ladder"
# The two-SI comparison's own suite tag. diffmean is matrix-only (its cells
# derive from configs/two_si.yaml's class_a_arms, NOT from arms_for_suite),
# so it carries S_TWO_SI ALONE and deliberately joins NO _CORE suite —
# keeping it out of every stage-2 ifeval positional array decode. (cast and circuit_breaker used to share this tag; both left
# the registry entirely on 2026-08-20 — see the import note above.)
S_TWO_SI = "two_si"

_CORE = frozenset({S_IFEVAL})


# -- SteeringVector -> object-protocol adapter -----------------------------------


class VectorObject:
    """Adapter: a trained SteeringVector behind the object protocol.

    Works for single-layer and multi-layer (vector_all) artifacts alike —
    hooks.steering already takes a multi-layer dict with one shared scale.
    """

    kind = "vector"

    def __init__(self, sv: SteeringVector, positions: str = "all",
                 skip_bos: bool = True, norm_relative: bool = False,
                 response_header_len: int | None = None):
        self.sv = sv
        self.positions = positions
        self.skip_bos = skip_bos
        self.norm_relative = norm_relative
        # Generation-time boundary for positions="response" (H = assistant-header
        # token length); set by the jailbreak defense factory, None otherwise.
        self.response_header_len = response_header_len

    @property
    def trainable(self) -> list:
        return []  # a trained artifact; training happens in the vector trainers

    def n_params(self) -> int:
        return sum(v.numel() for v in self.sv.vectors.values())

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def apply(self, model, scale: float = 1.0):
        return steering(
            model, self.sv, scale=scale, positions=self.positions,
            skip_bos=self.skip_bos, norm_relative=self.norm_relative,
            response_header_len=self.response_header_len,
        )

    def to_(self, device) -> None:
        self.sv.vectors = {
            l: v.to(device) for l, v in self.sv.vectors.items()
        }

    def save(self, path: Path) -> None:
        # Persist the ADAPTER's own state in the sidecar metadata (review
        # H-P1-2): load_object used to rebuild every safetensors artifact
        # with the defaults (positions="all", skip_bos=True), so a resume
        # silently converted actadd_front's faithful front-injection into
        # a mode-all arm.
        self.sv.metadata = {
            **(self.sv.metadata or {}),
            "vector_object": {"positions": self.positions,
                              "skip_bos": self.skip_bos,
                              "norm_relative": self.norm_relative},
        }
        self.sv.save(path)


# -- trainers (uniform signature: model, tok, instruction, pairs, *, layer, ...) --


def _make_train_vector(loss: str):
    """Fixed-vector distillation trainer at the given soft target: "soft"
    = forward KL (the stage-1 distill_soft cell), "rkl" / "js" = the
    divergence ablation — same engine, pairs, and hyperparameters, only
    the divergence differs (see extraction/distill.soft_divergence)."""

    def _train(model, tok, instruction: Instruction, pairs: list[Pair],
               *, layer: int, log_fn=None, **kw) -> VectorObject:
        exs = pairs_to_distill_examples(tok, pairs, instruction)
        sv = train_distill(model, tok, exs, layer, loss=loss,
                           instruction_id=instruction.id, log_fn=log_fn, **kw)
        return VectorObject(sv)

    return _train


_train_vector = _make_train_vector("soft")


def _train_bidpo_arm(model, tok, instruction: Instruction, pairs: list[Pair],
                     *, layer: int, log_fn=None, **kw) -> VectorObject:
    prefs = pairs_to_pref_examples(tok, pairs)
    sv = train_bidpo(model, tok, prefs, layer,
                     instruction_id=instruction.id, log_fn=log_fn, **kw)
    return VectorObject(sv)


def _train_bidpo_lora_arm(model, tok, instruction: Instruction,
                          pairs: list[Pair], *, layer: int, log_fn=None,
                          **kw) -> RankOneLoRAObject:
    # bidpo's pair semantics (pairs_to_pref_examples: pos -> chosen,
    # neg -> rejected, prompt rendered WITHOUT the instruction), lora_r1's
    # substrate. Like _train_lora, no instruction_id travels into the
    # artifact — the trained object is byte-compatible with lora_r1's
    # (kind "lora"), and the fit report carries the provenance.
    prefs = pairs_to_pref_examples(tok, pairs)
    return train_bidpo_lora(model, tok, prefs, layer, log_fn=log_fn, **kw)


def _train_caa(model, tok, instruction: Instruction, pairs: list[Pair],
               *, layer: int, log_fn=None, **_ignored) -> VectorObject:
    """CAA promoted to a standalone arm: fitted on the S2
    P1 pairs via the SAME pairs→CAA-inputs preparation PSR's
    init_psr_directions uses — our CAA arm is exactly the PSR
    direction-init estimator. Pooled completion_mean, Braun diagnostics in
    metadata (diff_of_means adds them). Fit is forward-passes only, so the
    config's distill train_kwargs (epochs/max_len/batch_size/...) do not
    apply and are deliberately ignored — batch_size is PINNED to 8, the
    init_psr_directions default (review H-P2-2: the config's distill
    batch_size=2 leaked in and broke the "exactly the PSR init estimator"
    identity, since pooled activations are not bitwise batch-invariant)."""
    from .data.pairs import pairs_to_caa_inputs
    from .extraction.diffmean import extract_caa

    prompts, pos, neg = pairs_to_caa_inputs(tok, pairs)
    sv = extract_caa(model, tok, prompts, pos, neg, [layer],
                     position="completion_mean", batch_size=8,
                     instruction_id=instruction.id)
    return VectorObject(sv)


def _train_actadd(model, tok, instruction: Instruction, pairs: list[Pair],
                  *, layer: int, log_fn=None, **_ignored) -> VectorObject:
    """ActAdd per its published recipe (single contrast pair — the S2
    pairs are NOT consumed):
    prompt_plus = the SI text, prompt_minus = a space-padded neutral,
    paper layer = harness layer + 1 (resid_pre(l) == resid_post(l-1)),
    POOLED single-vector form so it occupies the standard vector slot
    (mode all, skip_bos, absolute scale grid); see the actadd_front
    sensitivity arm for the faithful front-injection form."""
    from .extraction.actadd import extract_actadd

    sv = extract_actadd(model, tok, prompt_plus=instruction.text,
                        prompt_minus=" ", paper_layers=[layer + 1],
                        instruction_id=instruction.id, pool=True)
    return VectorObject(sv)


def _train_actadd_front(model, tok, instruction: Instruction,
                        pairs: list[Pair], *, layer: int, log_fn=None,
                        **_ignored) -> VectorObject:
    """The FAITHFUL ActAdd form ([P, d] front injection) — a sensitivity
    row (one SI) beside the pooled actadd arm."""
    from .extraction.actadd import extract_actadd

    sv = extract_actadd(model, tok, prompt_plus=instruction.text,
                        prompt_minus=" ", paper_layers=[layer + 1],
                        instruction_id=instruction.id, pool=False)
    return VectorObject(sv, positions="front", skip_bos=False)


def _train_kl_object(obj, model, tok, instruction, pairs, *, log_fn=None, **kw):
    exs = pairs_to_distill_examples(tok, pairs, instruction)
    train_object_kl(model, tok, obj, exs, log_fn=log_fn, **kw)
    return obj


def _train_vector_all(model, tok, instruction, pairs, *, layer=None,
                      log_fn=None, layers=None, **kw):
    obj = MultiLayerVectorObject(model, layers=layers)
    return _train_kl_object(obj, model, tok, instruction, pairs,
                            log_fn=log_fn, **kw)


def _train_lora(model, tok, instruction, pairs, *, layer, log_fn=None, **kw):
    obj = RankOneLoRAObject(model, layer=layer)
    return _train_kl_object(obj, model, tok, instruction, pairs,
                            log_fn=log_fn, **kw)


def _train_lora_all(model, tok, instruction, pairs, *, layer=None,
                    log_fn=None, layers=None, lora_rank: int = 1, **kw):
    # lora_rank (twosi re-tune@32): the lora arm's searched rank {1, 4, 16}.
    # Consumed HERE at construction — the config key name matches the
    # train_kwargs key end-to-end so the winner folds without translation.
    obj = MultiLayerLoRAObject(model, layers=layers, rank=int(lora_rank))
    return _train_kl_object(obj, model, tok, instruction, pairs,
                            log_fn=log_fn, **kw)


def _train_token_bos(model, tok, instruction, pairs, *, layer=None,
                     log_fn=None, sentinel=None, **kw):
    obj = SoftTokenObject(model, tok, sentinel=sentinel)
    return _train_kl_object(obj, model, tok, instruction, pairs,
                            log_fn=log_fn, **kw)


def _train_token_faithful(model, tok, instruction, pairs, *, layer=None,
                          log_fn=None, sentinel=None, **kw):
    # Radevski semantic init (2601.05062 §3; Table 4 shows init drives
    # composition accuracy): the token starts at the MEAN of the
    # instruction's own token embeddings, not the vocab mean.
    ids = tok(instruction.text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(
            f"instruction {instruction.id!r} tokenized to no tokens — "
            "cannot build the semantic soft-token init"
        )
    init = model.get_input_embeddings().weight.detach().float()[ids].mean(0)
    obj = AppendedSoftTokenObject(model, tok, sentinel=sentinel, init=init)
    kw.setdefault("temperature", FAITHFUL_TEMPERATURE)
    kw.setdefault("loss_scale", FAITHFUL_LOSS_SCALE)
    return _train_kl_object(obj, model, tok, instruction, pairs,
                            log_fn=log_fn, **kw)


def _make_train_psr(loss: str, all_layers: bool):
    def _train(model, tok, instruction, pairs, *, layer=None, log_fn=None,
               layers=None, probe_cb=None, probe_every=None, **kw):
        if layers is None:
            layers = (list(range(num_layers(model))) if all_layers
                      else [layer])
        # The direction trains under BOTH losses: the paper trains z under
        # MSE (their code: a trainable Linear — the old mean-diff freeze
        # was a hybrid the paper never ran). The mean-diff below is
        # the direction's INIT only — and only for the SINGLE-layer arms
        # (one layer's vector is the standard informed start, and for the
        # paper-faithful mse arm `psr` the init is the point). EVERY
        # all-layer variant — psr_mse_all included since 2026-08-20 —
        # ZERO-inits like every other all-layer arm (multilayer.py):
        # stacking ~32 full-strength diffmean vectors
        # under the wide-open init gate blew a fit's step-0 KL to 9.28 vs
        # a 0.148 no-op floor, and the first AdamW step (lr 5e-3) drove
        # the deep-layer gates permanently shut — ReLU dead, zero gradient
        # to gate AND direction; the KL path then had no gate regularizer,
        # unlike mse. Measured on the saved blobs: a contiguous deep block
        # frozen at gate_b=0.924214 (AdamW's one-nonzero-gradient impulse
        # response) on both SIs — 18/32 layers on safety_refusal, 11/32 on
        # si_constitution — final losses 4.82/5.72 ~ 30x worse than a
        # no-op. Zero-init makes
        # step 0 an exact no-op with the gate alive at g==1. psr_mse_all
        # had survived the aggressive init only because of its 0.1
        # gate_reg — protection the KL arms lacked — so MSE-vs-KL
        # conflated the objective with its init AND its regularizer; now
        # both objectives get both treatments (zero-init here, the shared
        # regularizer inside train_psr/train_object_kl) and the
        # comparison isolates the objective. See extraction/psr.py's
        # module docstring ("Confound removed").
        obj = PSRObject(model, layers, train_direction=True)
        if len(layers) == 1:
            obj.set_directions(init_psr_directions(model, tok, pairs, layers))
        exs = pairs_to_distill_examples(tok, pairs, instruction)
        # The probe seam travels EXPLICITLY (F7): the object is built
        # right here, so the production driver / tuning adapter cannot
        # hand a monitor the object up front — the trainer hands the LIVE
        # obj to probe_cb(step, obj) instead. Named parameters (not just
        # **kw passthrough) so the seam is declared in this trainer's
        # signature for both psr_kl and psr_mse arms; a TrainingDiverged
        # raised mid-fit propagates through train_psr, so `obj` is never
        # returned for a diverged fit.
        train_psr(model, tok, obj, exs, loss=loss, log_fn=log_fn,
                  probe_cb=probe_cb, probe_every=probe_every, **kw)
        return obj

    return _train


def _ce_probe_log_fn(log_fn, probe_cb, probe_every, obj):
    """The CE arms' mid-fit probe seam, riding ``train_object_ce``'s
    per-optimizer-step ``log_fn`` event (the psr KL path's kl_log_fn
    pattern): the trainer logs exactly once per optimizer step, right
    after ``opt.step()``, with the 1-based step — step 0 is the startup
    record, never probed — so ``probe_cb(step, obj)`` fires on the live
    object without a second trainer seam. Gated on the per-step
    ``.../ce_loss`` key so end-of-epoch held-out records (which reuse the
    current step ordinal) never double-fire a probe."""
    if (probe_cb is None) != (probe_every is None):
        raise ValueError(
            "probe_cb and probe_every come together — a callback with no "
            "cadence (or a cadence with no callback) is silently inert; "
            f"got probe_cb={'set' if probe_cb is not None else None}, "
            f"probe_every={probe_every!r}")
    if probe_cb is None:
        return log_fn
    pe = int(probe_every)
    if pe < 1:
        raise ValueError(f"probe_every must be >= 1, got {probe_every!r}")

    def wrapped(record, step=None, _inner=log_fn):
        if _inner is not None:
            _inner(record, step)
        if (isinstance(step, int) and step >= 1 and step % pe == 0
                and any(k.endswith("/ce_loss") for k in record)):
            probe_cb(step, obj)

    return wrapped


def _record_gate_liveness(diagnostics, log_fn, liveness: dict) -> None:
    """Per-layer PSR gate activity into the fit diagnostics (-> fit_record)
    and, when a log stream exists, W&B — for EVERY PSR CE fit, guard pass
    or fail (a gate-dead PSR is a near-no-op that PASSES the CE
    floor by design, so its health must be legible from the record)."""
    per_layer = {str(l): float(v) for l, v in sorted(liveness.items())}
    if diagnostics is not None:
        diagnostics["gate_liveness"] = per_layer
    if log_fn is not None:
        rec = {f"psr_ce/gate_live_frac/L{l}": v for l, v in per_layer.items()}
        rec["psr_ce/gate_live_frac_min"] = min(per_layer.values())
        log_fn(rec)


def _train_psr_ce(model, tok, instruction, pairs, *, layer=None, log_fn=None,
                  layers=None, gate_reg: float = 0.1, probe_cb=None,
                  probe_every=None, max_len: int = 768, diagnostics=None,
                  init_scheme: str = "zero", **kw):
    """The gated-PSR document-CE arm (sdf-base-steering §5.2): the exact
    ``psr_kl_all`` object (all-layer PSRObject, train_direction=True,
    zero-init) trained by :func:`train_object_ce` on document tokens.
    ``instruction`` is a corpus-cell pseudo-instruction and is DELIBERATELY
    ignored — the doc-mode builder renders no SI leg, ever. ``gate_reg``
    (default 0.1) is the same anti-dead-gate regularizer both existing PSR
    objectives carry.

    Dead-gate guard, load-bearing for the science (not hygiene): a
    gate-dead PSR is a near-no-op, scores ce_ratio ~ 1.0 and therefore
    PASSES the 1.1 CE floor by design — it would silently write "no
    benefit at this scale" into the PSR scaling curve. So after
    train_object_ce's best-epoch restoration, ``check_psr_gates`` (the ONE
    guard implementation) rules on the weights that would ship and
    REFUSES publication on dead layers; per-layer gate activity is
    recorded in the diagnostics (guard pass or fail) for the fit_record."""
    del instruction  # corpus data cell, not a system instruction (§5.2)
    from .extraction.psr import (GUARD_PROBE_FLOOR, DeadGateError,
                                 check_psr_gates, psr_gate_liveness)

    # The object comes from the SHARED factory (extraction.object_factory)
    # so the search trial, the reconnaissance run and
    # this production fit build the identical object: all-layer PSR,
    # train_direction=True, the paper's zero init unless the run override
    # selects FD 7's random scheme (``init_scheme``; the seed that draws it
    # is the fit seed, read with .get so the trainer still receives it).
    from .extraction.object_factory import build_object

    obj = build_object(model, object_kind="relu_steer", layers=layers,
                       init_scheme=init_scheme,
                       seed=int(kw.get("seed", 0) or 0))
    exs = docs_to_ce_examples(tok, pairs)
    if diagnostics is None:
        diagnostics = {}
    train_object_ce(model, tok, obj, exs,
                    log_fn=_ce_probe_log_fn(log_fn, probe_cb, probe_every,
                                            obj),
                    log_tag="psr_ce_all", reg_coeff=gate_reg,
                    max_len=max_len, diagnostics=diagnostics, **kw)
    # min_probe: the training set IS the probe pool — exhaustive, not thin
    # (the train_psr convention for deliberately small fits).
    try:
        liveness = check_psr_gates(
            model, tok, obj=obj, examples=exs, max_len=max_len,
            tag="psr_ce_all", min_probe=min(GUARD_PROBE_FLOOR, len(exs)))
    except DeadGateError:
        # The refusal stands — but the per-layer record must survive it
        # (guard pass or fail), so measure liveness once more for the
        # diagnostics before re-raising. One extra no-grad pass, on the
        # failure path only; the VERDICT stays check_psr_gates's alone.
        try:
            _record_gate_liveness(
                diagnostics, log_fn,
                psr_gate_liveness(model, tok, obj=obj, examples=exs,
                                  max_len=max_len, n_probe=len(exs)))
        except Exception:  # noqa: BLE001 — never mask the DeadGateError
            pass
        raise
    _record_gate_liveness(diagnostics, log_fn, liveness)
    return obj


def _train_lora_ce(model, tok, instruction, pairs, *, layer=None,
                   log_fn=None, layers=None, lora_rank: int = 4,
                   lora_modules=None, per_expert_rank=None,
                   init_scheme: str = "zero",
                   probe_cb=None, probe_every=None, diagnostics=None, **kw):
    """The rank-4 all-layer LoRA document-CE arm (sdf-base-steering §5.2):
    the exact ``lora_r4`` object (MultiLayerLoRAObject, B zero-init,
    o_proj of every full-attention layer — the constructor's
    ``oproj_layers`` filter handles hybrid-attention models) trained by
    :func:`train_object_ce` on document tokens. ``instruction`` is a
    corpus-cell pseudo-instruction and is DELIBERATELY ignored (no SI leg
    is ever rendered). ``lora_rank`` defaults to 4 — the arm's name — and
    stays a train_kwargs key so configs read end-to-end. The config seed
    (``train_kwargs.default.seed`` -> ``kw["seed"]``, default 0) seeds BOTH
    the A-factor init here and the trainer's per-epoch shuffle."""
    del instruction  # corpus data cell, not a system instruction (§5.2)
    # The seed rides ``kw`` for train_object_ce's epoch order; it also seeds
    # the LoRA A-factor init (2026-09-15): before this the init generator
    # was manual_seed(0 + l) for EVERY config, so two configs differing in
    # ``seed`` alone shared their initialisation and differed in data order
    # only. Read with .get — never popped, the trainer still receives it —
    # and defaulted to 0, so every existing config's object is bit-identical
    # to before. Built through the SHARED factory
    # (extraction.object_factory) so this production fit, the
    # search trial and the recon run construct the identical object.
    # ``lora_modules`` (default the paper's ["o_proj"] — the legacy,
    # bit-identical path), ``per_expert_rank`` and ``init_scheme`` are the
    # LoRA architecture search's axes (FD 6, FD 7) and ride the per-arm
    # train_kwargs block like ``lora_rank`` does.
    from .extraction.object_factory import build_object

    obj = build_object(model, object_kind="lora", layers=layers,
                       lora_rank=int(lora_rank), lora_modules=lora_modules,
                       per_expert_rank=per_expert_rank, init_scheme=init_scheme,
                       seed=int(kw.get("seed", 0) or 0))
    exs = docs_to_ce_examples(tok, pairs)
    if diagnostics is None:
        diagnostics = {}
    train_object_ce(model, tok, obj, exs,
                    log_fn=_ce_probe_log_fn(log_fn, probe_cb, probe_every,
                                            obj),
                    log_tag="lora_r4_ce", diagnostics=diagnostics, **kw)
    return obj
def _make_train_transferred(arm_name: str):
    """The psr_xfer_* arms are CROSS-MODEL TRANSFERRED artifacts: their
    object.pt is produced by the PSR transfer procedure (a source model's
    PSR unit mapped into the target model's basis), never trained by this
    repo's trainers. Registering a real trainer would let a refit silently
    overwrite a transferred artifact with a natively-fit one under the
    transferred arm's name — the exact identity confusion the distinct
    arm names exist to prevent. Serving/hydration needs no trainer:
    the artifacts are ordinary PSRObject blobs (kind 'psr'), loaded via
    methods.load_object like any psr_kl_all object."""

    def _refuse(model, tok, instruction, pairs, **kw):
        raise RuntimeError(
            f"arm {arm_name!r} is a cross-model TRANSFERRED artifact "
            "(psr-transfer lane): it is never trained in-repo. Stage its "
            "object.pt (+ fit-provenance sidecar naming the TARGET model) "
            "under <defenses_dir>/<si>__" + arm_name + "/ instead.")

    return _refuse


#: The DERIVED arms (sdf-base-steering round 2): decomposition products of
#: two FITTED parents, composed on CPU by a builder that lives on the
#: sdf-base lineage (``scripts/sdfbase/lora_subtract.py`` on derived-full /
#: full22-newmodels) — never trained here, never built here. The ONE record
#: of which registered arms are derived: ``is_derived_arm`` reads it, the
#: safety-defense builder skips them through it, and a test pins that every
#: member's ArmSpec refuses both ``build`` and ``train``. Ported onto main
#: 2026-09-10 (bcb-bfcl-200) for the LoRA arm; since the math500 merge
#: (2026-09-11) it names BOTH derived arms of the sdf-base lineage: the
#: distilled PSR residual (kind ``psr``) and the full-corpus lora_r4_ce fit
#: MINUS its register-matched control (kind ``lora_subtraction``).
#: ... and since relu-differenced-native-qwen (2026-09-17) the UNDISTILLED paired write
#: (kind ``psr_paired_write``, composed by the same builder from the same two parents).
DERIVED_ARMS: tuple[str, ...] = ("psr_ce_all_minus_ctl_distilled", "lora_r4_ce_minus_ctl",
                                 "psr_ce_all_minus_ctl")


def is_derived_arm(arm: str) -> bool:
    """Whether ``arm`` is a DERIVED arm (:data:`DERIVED_ARMS`): registered
    for hydration + serving only, with a build AND a train that refuse."""
    return arm in DERIVED_ARMS


def _make_train_derived(arm_name: str, builder: str):
    """The DERIVED arms (sdf-base-steering round 2) are decomposition
    products of two FITTED parents, composed on CPU by ``builder`` — never
    trained by this repo's trainers. Same refusal, for the same reason, as
    :func:`_make_train_transferred`: a real trainer here would let a refit
    silently overwrite a published derived artifact with a natively-fit
    object under the derived name, and the whole point of a derived id is
    that its bytes are provably a function of its parents (whose shas ride
    in ``derived_meta.json`` AND inside the blob's ``provenance``). Serving
    needs no trainer: the artifacts hydrate through ``load_object``'s
    ordinary ``_KIND_LOADERS`` dispatch and serve through their kind's
    committed lens adapter."""

    def _refuse(model, tok, instruction, pairs, **kw):
        raise RuntimeError(
            f"arm {arm_name!r} is a DERIVED object (sdf-base-steering round "
            f"2): it is composed from two fitted parents by {builder}, never "
            "trained in-repo. Build it there and stage its object.pt + "
            f"derived_meta.json under <defenses_dir>/<si>__{arm_name}/ "
            "instead.")

    return _refuse


def _make_build_derived(arm_name: str, builder: str):
    """A derived arm's BUILD refuses too. Unlike psr_xfer_* (whose build
    constructs the fresh PSRObject a transfer lands in), a derived object
    has no meaningful untrained form: ``lora_subtraction`` cannot exist
    without two hydrated parents, and a "fresh" residual-distilled PSR
    would be indistinguishable from a real one. Refusing keeps the only
    route to these artifacts the builder that records their provenance."""

    def _refuse(model, tok, layer, **kw):
        raise RuntimeError(
            f"arm {arm_name!r} has no untrained form: a derived object is "
            f"composed from two fitted parents by {builder}. There is "
            "nothing to construct here; load the published artifact.")

    return _refuse


def _train_loreft_arm(model, tok, instruction, pairs, *, layer, log_fn=None,
                      rank: int = 4, **kw):
    obj = LoReFTObject(model, [layer], rank=rank)
    exs = pairs_to_distill_examples(tok, pairs, instruction)
    train_loreft(model, tok, obj, exs, log_fn=log_fn, **kw)
    return obj


def _train_antidote_arm(model, tok, instruction: Instruction, pairs: list[Pair],
                        *, layer, log_fn=None, **kw) -> AntidoteObject:
    # closed-form extraction + alpha calibration; the shared
    # train_kwargs.default (epochs/batch_size/max_len) pass through and
    # train_antidote documents which are honored (batch_size) vs no-ops.
    return train_antidote(model, tok, instruction, pairs, layer=layer,
                          log_fn=log_fn, **kw)


def _train_lofit_arm(model, tok, instruction: Instruction, pairs: list[Pair],
                     *, layer, log_fn=None, **kw) -> LoFiTObject:
    # LoFiT's two-phase trainer. The pinned `layer` is passed through
    # and DELIBERATELY IGNORED inside train_lofit (head selection ranges
    # over ALL layers — the method's identity; recorded in the metadata).
    exs = pairs_to_distill_examples(tok, pairs, instruction)
    return train_lofit(model, tok, exs, layer=layer,
                       instruction_id=instruction.id, log_fn=log_fn, **kw)


def _train_reft_r1_arm(model, tok, instruction, pairs, *, layer, log_fn=None,
                       topk: int = 8, seed: int = 0, **kw):
    # topk/seed are CONSTRUCTOR args (train_reft_r1 does not take topk), so
    # config train_kwargs like {topk: 8, l1_lambda: 0.005} split cleanly
    # instead of crashing the trainer with an unexpected kwarg.
    obj = ReFTr1Object(model, layer, topk=topk, seed=seed)
    exs = pairs_to_distill_examples(tok, pairs, instruction)
    train_reft_r1(model, tok, obj, exs, log_fn=log_fn, seed=seed, **kw)
    return obj


def _train_caa_static_r_arm(model, tok, instruction, pairs, *, layer,
                            log_fn=None, benign_prompts=None,
                            harmful_prompts=None, benign_prompts_path=None,
                            harmful_prompts_path=None, min_prompts=None,
                            epochs=None, max_len=None, seed=None, **kw):
    """caa_static_r is CONTENT-fit (Class B, SI-agnostic): ``instruction``
    and ``pairs`` are DELIBERATELY ignored — the SI must never enter this
    fit. The pinned ``layer`` is ALSO deliberately ignored (the lofit
    precedent): the arm's identity is the AlphaSteer-matched TEN-layer
    deployment (extraction/caa_static_r.py PAPER_LLAMA31_LAYERS; a
    ``layers`` kwarg overrides for tests/small models). Production goes
    through scripts/fit_class_b.py; this shim mirrors the
    an earlier development branch's static_r registry shim in ALSO accepting the
    two prompt-file paths directly (loaded through
    ``build_content_contrast`` so the sha256 provenance and the
    ``min_prompts`` undersized-fit floor ride along), or explicit prompt
    lists (tests). ``epochs``/``max_len``/``seed`` are documented no-ops
    (closed-form, deterministic)."""
    del instruction, pairs, layer, epochs, max_len, seed
    if benign_prompts is None or harmful_prompts is None:
        if benign_prompts_path is None or harmful_prompts_path is None:
            raise ValueError(
                "caa_static_r is a Class B content fit: pass benign_prompts "
                "+ harmful_prompts (lists) or benign_prompts_path + "
                "harmful_prompts_path (the pinned JSON prompt files) — the "
                "(instruction, pairs) surface is ignored by design. "
                "Production fits go through scripts/fit_class_b.py.")
        from .data.content_sets import build_content_contrast

        contrast = build_content_contrast(
            benign_prompts_path=benign_prompts_path,
            harmful_prompts_path=harmful_prompts_path,
            min_prompts=min_prompts)
        benign_prompts, harmful_prompts = contrast.benign, contrast.harmful
        contrast_meta = dict(contrast.meta)
    else:
        contrast_meta = None
    obj = train_caa_static_r_content(
        model, tok, benign_prompts, harmful_prompts, log_fn=log_fn, **kw)
    if contrast_meta is not None:
        obj.metadata["content_contrast"] = contrast_meta
    return obj


# -- the registry ---------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    name: str
    build: Callable  # (model, tok, layer, **kw) -> fresh untrained object
    train: Callable  # (model, tok, instruction, pairs, *, layer, **kw) -> object
    suites: frozenset[str]


def _make_train_alphakl(stage: int):
    """One registry trainer per ladder stage: the arm name pins the stage
    (a config train_kwargs `stage` cannot silently run a different rung
    under an arm named for another), everything else flows through.

    INTERPRETATION GATE: the trained object's metadata records
    `probe_auc` and `gate_benign_mean`/`gate_malicious_mean` on a held-out
    split. `r` is diff-in-means at the LAST prompt token, and after a
    ~5,900-token SI it can encode length/position features instead of SI
    content — that failure is legible as a chance-level probe_auc, NOT a
    crash. Read those fields before trusting any cell; a chance-level
    probe means the cell must not be interpreted.

    RESOURCE DEFAULT: the stage-3 (fisher) trainer defaults
    batch_size to 2 here. Its forward-AD passes force math-backend SDPA
    (flash has no forward-mode AD), so attention memory is O(T^2), and the
    bf16 logits and tangent are each [B, T, V] — at the 5,908-token
    constitution SI used here the source branch's batch_size=8 default OOMs.
    A config train_kwargs batch_size still overrides."""

    def _train(model, tok, instruction, pairs, *, layer, log_fn=None, **kw):
        kw.pop("stage", None)
        if stage == 3:
            kw.setdefault("batch_size", 2)
        return train_alphakl(model, tok, instruction, pairs, layer=layer,
                             stage=stage, log_fn=log_fn, **kw)

    return _train


ARMS: dict[str, ArmSpec] = {
    spec.name: spec
    for spec in [
        ArmSpec(
            "vector",
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="distill_soft",
                               model_name="", instruction_id="")),
            _train_vector,
            _CORE | {S_COMPOSITION, S_LADDER},
        ),
        ArmSpec(
            "vector_all",
            lambda model, tok, layer, **kw: MultiLayerVectorObject(model, **kw),
            _train_vector_all,
            _CORE,
        ),
        ArmSpec(
            "lora_r1",
            lambda model, tok, layer, **kw: RankOneLoRAObject(model, layer, **kw),
            _train_lora,
            _CORE,
        ),
        ArmSpec(
            # LEGACY SPELLING of lora_r4 (below) — kept registered so cell
            # ids and configs that still name it keep resolving, and so the
            # upstream fitting lane (which emits arm=lora_r1_all until it
            # switches) still trains it. New work names lora_r4; the two
            # specs share build/train/kind ("lora_all") byte-for-byte, and
            # LEGACY_ARM_ALIASES below records the equivalence.
            "lora_r1_all",
            lambda model, tok, layer, **kw: MultiLayerLoRAObject(model, **kw),
            _train_lora_all,
            _CORE,
        ),
        # lora_r4: the arm's post-re-tune name (rank 4 won the per-rank
        # comparison on every model; 'all' implicit). Same machinery —
        # rank rides the kwargs; legacy lora_r1_all artifacts keep their
        # name and stay servable.
        ArmSpec(
            "lora_r4",
            lambda model, tok, layer, **kw: MultiLayerLoRAObject(model, **kw),
            _train_lora_all,
            _CORE,
        ),
        ArmSpec(
            "token_bos",
            lambda model, tok, layer, **kw: SoftTokenObject(model, tok, **kw),
            _train_token_bos,
            _CORE,
        ),
        ArmSpec(
            "token_faithful",
            lambda model, tok, layer, **kw: AppendedSoftTokenObject(
                model, tok, **kw),
            _train_token_faithful,
            _CORE | {S_COMPOSITION},
        ),
        ArmSpec(
            "psr",
            lambda model, tok, layer, **kw: PSRObject(
                model, [layer], train_direction=True, **kw),
            _make_train_psr("mse", all_layers=False),
            _CORE,
        ),
        ArmSpec(
            "psr_kl",
            lambda model, tok, layer, **kw: PSRObject(
                model, [layer], train_direction=True, **kw),
            _make_train_psr("kl", all_layers=False),
            _CORE,
        ),
        ArmSpec(
            "psr_kl_all",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _make_train_psr("kl", all_layers=True),
            _CORE,
        ),
        ArmSpec(
            "psr_mse_all",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _make_train_psr("mse", all_layers=True),
            _CORE,
        ),
        ArmSpec(
            "psr_kl_hard",
            lambda model, tok, layer, **kw: PSRObject(
                model, [layer], train_direction=True, **kw),
            _make_train_psr("kl_hard", all_layers=False),
            _CORE,
        ),
        ArmSpec(
            "psr_kl_all_hard",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _make_train_psr("kl_hard", all_layers=True),
            _CORE,
        ),
        ArmSpec(
            "loreft",
            lambda model, tok, layer, **kw: LoReFTObject(model, [layer], **kw),
            _train_loreft_arm,
            _CORE,
        ),
        ArmSpec(
            "reft_r1",
            lambda model, tok, layer, **kw: ReFTr1Object(model, layer, **kw),
            _train_reft_r1_arm,
            _CORE,
        ),
        ArmSpec(
            "bidpo",
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="bidpo",
                               model_name="", instruction_id="")),
            _train_bidpo_arm,
            _CORE,
        ),
        # Divergence ablation: the vector arm with the distillation
        # target swapped — reverse KL (mode-seeking) and Jensen-Shannon
        # (symmetric) against forward KL's mass-covering "soft". Appended
        # AFTER the 12 plan arms: the sbatch array decodings index the
        # first 12 positionally, and that order is load-bearing.
        ArmSpec(
            "vector_rkl",
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="distill_rkl",
                               model_name="", instruction_id="")),
            _make_train_vector("rkl"),
            _CORE,
        ),
        ArmSpec(
            "vector_js",
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="distill_js",
                               model_name="", instruction_id="")),
            _make_train_vector("js"),
            _CORE,
        ),
        # Three defense methods brought in from the lofit-arm branch for the
        # two-SI comparison: APPENDED after the 17 pre-existing arms
        # so every positional decode (the sbatch DEFENSES arrays, the
        # si_idx*12 + arm_idx stage-2 decodes) keeps its meaning — antidote,
        # then alphasteer, then lofit. All three carry this lane's _CORE
        # suite set (the lofit-arm branch registered alphasteer/lofit as
        # IFEval-only; this campaign runs them on the same suites as the
        # comparable arms, e.g. psr_kl_all).
        #
        # Jailbreak Antidote (arXiv 2410.02298) — pooled-PCA direction
        # at the pinned layer, top-5% |component| mask (no renorm),
        # h' = h + alpha (d ⊙ m) at the LAST token only (hooks "last"
        # mode), absolute alpha = base_alpha (calibrated) x factor.
        ArmSpec(
            "antidote",
            lambda model, tok, layer, **kw: AntidoteObject(
                SteeringVector(vectors={}, method="antidote",
                               model_name="", instruction_id="",
                               metadata={"base_alpha": 1.0})),
            _train_antidote_arm,
            _CORE,
        ),
        # AlphaSteer (arXiv 2506.07022) — closed-form null-space
        # projected ridge matrix, input-dependent prefill-only edit at the
        # pinned layer, canary-calibrated base scale x the eval factor grid.
        ArmSpec(
            "alphasteer",
            lambda model, tok, layer, **kw: AlphaSteerObject(
                model, layer,
                torch.zeros(hidden_size(model), hidden_size(model)),
                base_scale=1.0),
            train_alphasteer,
            _CORE,
        ),
        # LoFiT (arXiv 2406.01563) — two-phase localized fine-tuning:
        # per-head multiplicative scalers under L1 select the top-~3% of
        # attention heads ACROSS ALL LAYERS, then per-head additive biases
        # v_h train on those heads' pre-o_proj outputs (base_scale fixed
        # at 1.0 — the paper applies at strength 1; the factor grid sweeps
        # around it). KEY DELIBERATE DEVIATION from the other arms: NO
        # layer-15 pin — the all-layer head selection is the method's
        # identity and what makes it a complement to ReFT (loreft:
        # residual positions at the pinned layer; lofit: sparse head
        # outputs wherever selected).
        ArmSpec(
            "lofit",
            lambda model, tok, layer, **kw: LoFiTObject(model, heads=[], **kw),
            _train_lofit_arm,
            _CORE,
        ),
        # The AlphaSteer -> KL-constrained steering ladder, ported
        # additively from extraction/alphakl.py — ONE rank-1 gated
        # deployment (h += scale * gate(w.h) * r, EVERY position, prompt
        # and decode), three probe-fitting rungs that strictly nest: hard
        # null space, soft covariance penalty, and Fisher/KL weighting
        # under an eps-nats/position benign budget.
        # alphakl_null is deliberately included beyond the two upper rungs:
        # alphasteer is prefill-only, last-token-broadcast and
        # UNCLAMPED, so it does not share the ladder's deployment — the
        # stage-1 rung is the deployment-matched baseline without which an
        # alphasteer-vs-alphakl_soft difference conflates the penalty with
        # the deployment. Appended AFTER lofit (indices 20/21/22) so every
        # existing positional decode keeps its meaning; registered on the
        # full _CORE suite set (matching psr_kl_all), not the IFEval-only
        # set of the source implementation.
        #
        # clamp_gate stays True by default, on purpose: the unclamped gate
        # can go NEGATIVE — anti-SI steering on adversarial projections —
        # and adaptive jailbreaks run against these objects, so
        # that channel is a live attack surface. `train_kwargs clamp_gate:
        # false` is the deliberate unclamped parity mode.
        #
        # Before interpreting any alphakl cell, read the trained object's
        # metadata probe_auc / gate_benign_mean / gate_malicious_mean
        # (see _make_train_alphakl) — a chance-level probe at the
        # 5,908-token SI means length/position leaked into r and the cell
        # must not be interpreted.
        ArmSpec(
            "alphakl_null",   # stage 1: hard null space (deployment-matched)
            lambda model, tok, layer, **kw: AlphaKLObject(
                model, [layer], {layer: torch.zeros(hidden_size(model))},
                {layer: torch.zeros(hidden_size(model))}),
            _make_train_alphakl(1),
            _CORE,
        ),
        ArmSpec(
            "alphakl_soft",   # stage 2: soft covariance penalty (beta
                              # derived from eps unless explicitly pinned)
            lambda model, tok, layer, **kw: AlphaKLObject(
                model, [layer], {layer: torch.zeros(hidden_size(model))},
                {layer: torch.zeros(hidden_size(model))}),
            _make_train_alphakl(2),
            _CORE,
        ),
        ArmSpec(
            "alphakl_fisher",  # stage 3: Fisher/KL weighting (beta from eps)
            lambda model, tok, layer, **kw: AlphaKLObject(
                model, [layer], {layer: torch.zeros(hidden_size(model))},
                {layer: torch.zeros(hidden_size(model))}),
            _make_train_alphakl(3),
            _CORE,
        ),
        # cast and circuit_breaker (the two-SI net-new arms of 2026-08-18)
        # were REMOVED from the registry entirely on 2026-08-20 — not just
        # dropped from the trained set. They used to sit here at positions
        # 23-24; diffmean below moved up two positions with them gone.
        # Their implementations (extraction/cast.py,
        # extraction/circuit_breaker.py) and their _KIND_LOADERS entries
        # below survive so load_object keeps hydrating the already-fitted
        # artifacts — see the import note at the top of this module.
        #
        # diffmean (added 2026-08-20): the diff-in-means / CAA-family
        # floor — Stolfo et al. (ICLR 2025, 2410.12877) instruction-difference
        # steering, equivalently the alphakl arms' static actuation direction r with
        # the gate deleted. CLASS A, one fixed RAW vector r = mean(h | SI
        # present) - mean(h | SI absent) at the LAST prompt token of the
        # pinned layer, applied additively at every position (BOS excluded,
        # the plain-vector convention) — no learned gate, no probe, no ridge,
        # no null-space projection, on purpose: the interpretable floor every
        # learned arm must beat. Appended LAST so every existing positional
        # decode keeps its meaning, and tagged S_TWO_SI ONLY (its cells
        # derive from configs/two_si.yaml's class_a_arms, never from
        # arms_for_suite).
        ArmSpec(
            "diffmean",
            lambda model, tok, layer, **kw: DiffMeanObject(
                model, {layer: torch.zeros(hidden_size(model))}, **kw),
            train_diffmean,
            frozenset({S_TWO_SI}),
        ),
        # bidpo_lora: the bidirectional-DPO
        # objective applied to lora_r1's EXACT substrate — the build lambda
        # is verbatim lora_r1's, on purpose (same RankOneLoRAObject, same
        # rank/target module/layer semantics), and ONLY the trainer differs.
        # DPO appeared once in the matrix, vector-substrate only, so
        # bidpo-vs-lora_r1 conflated objective with substrate; this cell
        # de-confounds it (lora_r1 vs bidpo_lora isolates the objective,
        # bidpo vs bidpo_lora the substrate — see extraction/bidpo_lora.py).
        # Appended LAST, tagged S_TWO_SI ONLY (its cells derive from
        # configs/two_si.yaml's class_a_arms, never arms_for_suite), like
        # diffmean above. Its artifact is lora_r1's (kind "lora", object.pt
        # via the object_artifact_name default), so hydration needs no new
        # loader entry.
        ArmSpec(
            "bidpo_lora",
            lambda model, tok, layer, **kw: RankOneLoRAObject(model, layer, **kw),
            _train_bidpo_lora_arm,
            frozenset({S_TWO_SI}),
        ),
        # caa_static_r: the CAA / diff-in-means
        # floor as a CLASS B (SI-agnostic, fit-once) arm — the
        # an earlier development branch's static_r diff-in-means FIT (reused per
        # the user's instruction; vendored with attribution in
        # extraction/caa_static_r.py) under the AlphaSteer-MATCHED
        # deployment, per that lane's 2026-08-21 correction (their
        # `static_r_matched`): the paper's TEN layers, resid_pre,
        # prefill-only, RAW per-layer lam·r_l broadcast to ALL prompt
        # positions — NOT the withdrawn single-layer/last-token/unit-norm
        # form, which measured BELOW the unsteered baseline and is not a
        # floor. With the deployment matched, the only difference from
        # AlphaSteer is FIXED direction vs learned input-dependent d×d map.
        # Deliberately NOT named `static_r`: on the peer's branch that name
        # stays the preserved last-token form (their index 28), and the
        # deployment difference is the load-bearing part. Production fit:
        # scripts/fit_class_b.py (the AlphaSteer authors' sha-pinned
        # 14,000/2,000 lists; the ten-layer set is this arm's documented
        # exemption from the Class B single-layer pin). Appended LAST,
        # tagged S_TWO_SI ONLY (its cell derives from configs/two_si.yaml's
        # class_b_arms, never arms_for_suite), like diffmean/bidpo_lora
        # above.
        ArmSpec(
            "caa_static_r",
            lambda model, tok, layer, **kw: CaaStaticRObject(
                model, {l: torch.zeros(hidden_size(model))
                        for l in PAPER_LLAMA31_LAYERS}),
            _train_caa_static_r_arm,
            frozenset({S_TWO_SI}),
        ),
        # two jailbreak-defense / contrast methods adapted to
        # SI steering in the stage-2 evaluations (added 2026-08-16),
        # merged in APPENDED LAST (after circuit_breaker, indices 25/26) so
        # every existing positional decode on BOTH lanes (si_idx*12 +
        # arm_idx, the sbatch DEFENSES arrays, eval/jailbreak.COMPILED =
        # tuple(ARMS)) keeps its meaning. Scope preserved from that lane:
        # the IFEval matrix ONLY (suites = {S_IFEVAL}). Each
        # stores a canary-calibrated base scale that the eval factor grid
        # multiplies, exactly like the antidote arm.
        ArmSpec(
            "did",  # difference-in-differences
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="did",
                               model_name="", instruction_id="")),
            train_did,
            frozenset({S_IFEVAL}),
        ),
        ArmSpec(
            "adasteer",  # AdaSteer (arXiv 2504.09466)
            lambda model, tok, layer, **kw: AdaSteerObject(
                model, layer,
                {k: torch.zeros(hidden_size(model)) for k in
                 ("v_RD", "v_HD", "mu_a_rd", "mu_b_rd", "mu_a_hd", "mu_b_hd")}
                | {"rlaw": {"w": 0.0, "b": 0.0},
                   "hlaw": {"w": 0.0, "b": 0.0}},
                base_scale=1.0),
            train_adasteer,
            frozenset({S_IFEVAL}),
        ),
        # -- stage-2 cheap-estimator arms: ifeval suite only;
        # composition membership deliberately not granted.
        # APPENDED LAST (after adasteer) at the stage2-evals merge, keeping
        # every pre-existing positional decode (si_idx*12 + arm_idx, the
        # sbatch DEFENSES arrays, eval/jailbreak.COMPILED = tuple(ARMS))
        # meaningful; that lane addresses these arms by NAME.
        ArmSpec(
            "caa",
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="caa",
                               model_name="", instruction_id="")),
            _train_caa,
            frozenset({S_IFEVAL}),
        ),
        ArmSpec(
            "actadd",
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="actadd_pooled",
                               model_name="", instruction_id="")),
            _train_actadd,
            frozenset({S_IFEVAL}),
        ),
        ArmSpec(
            "actadd_front",
            lambda model, tok, layer, **kw: VectorObject(
                SteeringVector(vectors={}, method="actadd",
                               model_name="", instruction_id=""),
                positions="front", skip_bos=False),
            _train_actadd_front,
            frozenset({S_IFEVAL}),
        ),
        # psr_xfer_* (2026-08-28, psr-transfer behavioral eval): CROSS-MODEL
        # TRANSFERRED PSR units — the SAME artifact shape psr_kl_all stores
        # (PSRObject blob, kind 'psr': per-layer gated units on BLOCK
        # OUTPUTS), but the bytes come from the transfer procedure (a source
        # model's unit mapped into the target's basis), never from an
        # in-repo fit. Distinct arm names on purpose: (a) the transferred
        # object must never collide on disk with the target's native
        # <si>__psr_kl_all/ fit, and (b) a results row must carry its
        # transfer identity. Naming: psr_xfer_from_<source>; the _rw1
        # suffix is the ridge-ablation variant of the same transfer. The
        # trainer REFUSES (see _make_train_transferred) — serving needs
        # only build/load, and both ride the existing psr machinery.
        # Appended LAST so every earlier arm's arm-major index is unchanged.
        ArmSpec(
            "psr_xfer_from_e2b",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _make_train_transferred("psr_xfer_from_e2b"),
            _CORE,
        ),
        ArmSpec(
            "psr_xfer_from_e2b_rw1",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _make_train_transferred("psr_xfer_from_e2b_rw1"),
            _CORE,
        ),
        ArmSpec(
            "psr_xfer_from_e4b",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _make_train_transferred("psr_xfer_from_e4b"),
            _CORE,
        ),
        ArmSpec(
            "psr_xfer_from_e4b_rw1",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _make_train_transferred("psr_xfer_from_e4b_rw1"),
            _CORE,
        ),
        # -- sdf-base-steering document-CE arms (2026-09-01):
        # the campaign's two object classes trained by plain hard-target LM
        # cross-entropy on DOCUMENT tokens (data_mode: documents pools,
        # docs_to_ce_examples — never pairs_to_distill_examples, which
        # chat-renders). Object geometry is IDENTICAL to the instruct
        # campaign's arms — psr_ce_all builds exactly psr_kl_all's object,
        # lora_r4_ce exactly lora_r4's — so the instruct-transfer
        # comparison is object-matched; only the loss and data change.
        # psr_ce_all is deliberately NOT a reuse of psr_kl_all_hard: that
        # arm routes through train_psr's KL delegation (five refusal
        # blocks, a dead teacher budget) and its name misstates the
        # objective. APPENDED LAST, keeping every existing positional
        # decode (the sbatch arrays, si_idx*12 + arm_idx, COMPILED =
        # tuple(ARMS)); tagged S_TWO_SI only — their cells come from the
        # sdfbase configs' si_ids x explicit arms, never arms_for_suite.
        # At the sdfbase-mt merge (2026-09-04) they sit AFTER main's
        # psr_xfer_* arms: still appended last, positional pins updated.
        # Lockstep: neither joins _LOCKSTEP_COMPATIBLE (CE on document
        # tokens — no teacher forward at all), so both train SOLO.
        ArmSpec(
            "psr_ce_all",
            lambda model, tok, layer, **kw: PSRObject(
                model, list(range(num_layers(model))), train_direction=True,
                **kw),
            _train_psr_ce,
            frozenset({S_TWO_SI}),
        ),
        ArmSpec(
            "lora_r4_ce",
            lambda model, tok, layer, **kw: MultiLayerLoRAObject(model, **kw),
            _train_lora_ce,
            frozenset({S_TWO_SI}),
        ),
        # -- sdf-base-steering DERIVED arms (round 2, 2026-09-06): the two
        # decomposition products of the full-corpus fit MINUS its
        # register-matched control fit, registered so the wave can SERVE
        # them. Neither is trained here — both refuse (_make_train_derived
        # / _make_build_derived); their artifacts are composed from two
        # fitted parents by the builders named in the refusals and carry a
        # derived_meta.json sidecar (never train_budget_meta.json) binding
        # the parent identities, which train_budget.derived_object_problems
        # verifies before any load.
        #
        # NEITHER IS A NEW SERVED SHAPE. `psr_ce_all_minus_ctl_distilled`
        # IS a PSRObject (blob kind 'psr' — the residual is distilled back
        # into psr_ce_all's own geometry), so it serves through the
        # committed psr adapter unchanged. `lora_r4_ce_minus_ctl` stores
        # BOTH parents' (A, B) factors bit-exact and applies each parent's
        # own o_proj hook at +scale / -scale (multilayer.
        # LoRASubtractionObject.apply), so it serves as two OP_LORA_OPROJ
        # ops per layer at the existing o_proj site — a new blob SCHEMA,
        # not a new op, hook site or reduction class.
        #
        # NOT in defense_cell.CLASS_A_ARMS on purpose: these cells are
        # enumerated explicitly (explicit_cells), never crossed with the
        # SI list, and CLASS_A_ARMS membership would fabricate cells and
        # move measurement_inputs_sha for every config. Appended LAST so
        # every earlier arm's arm-major index is unchanged.
        ArmSpec(
            "psr_ce_all_minus_ctl_distilled",
            _make_build_derived("psr_ce_all_minus_ctl_distilled",
                                "scripts/sdfbase/psr_residual_distill.py"),
            _make_train_derived("psr_ce_all_minus_ctl_distilled",
                                "scripts/sdfbase/psr_residual_distill.py"),
            frozenset({S_TWO_SI}),
        ),
        ArmSpec(
            "lora_r4_ce_minus_ctl",
            _make_build_derived("lora_r4_ce_minus_ctl",
                                "scripts/sdfbase/lora_subtract.py"),
            _make_train_derived("lora_r4_ce_minus_ctl",
                                "scripts/sdfbase/lora_subtract.py"),
            frozenset({S_TWO_SI}),
        ),
        # psr_ce_all_minus_ctl (relu-differenced-native-qwen, 2026-09-17):
        # the UNDISTILLED paired write — PSRPairedWriteObject (kind
        # 'psr_paired_write'), both parents' (w, b, d) stored bit-exact and
        # applied as `+u_pos(z) - u_neg(z)` on the shared pre-write state
        # (extraction/paired_write.py). Built by the SAME builder as the
        # distilled arm (psr_residual_distill.py writes it into
        # derived/psr_paired_write/ before distilling); never trained here.
        # Served natively on the residual slot's second gate term; the lens
        # has no adapter for it. Appended LAST (arm-major indices unchanged).
        ArmSpec(
            "psr_ce_all_minus_ctl",
            _make_build_derived("psr_ce_all_minus_ctl",
                                "scripts/sdfbase/psr_residual_distill.py"),
            _make_train_derived("psr_ce_all_minus_ctl",
                                "scripts/sdfbase/psr_residual_distill.py"),
            frozenset({S_TWO_SI}),
        ),
    ]
}


def arms_for_suite(suite: str) -> list[str]:
    return [name for name, spec in ARMS.items() if suite in spec.suites]


# Arms whose trained artifact is a SteeringVector container (safetensors blob
# + a .json sidecar); every other arm saves a plain torch blob. Mirrors
# stage2_run.VECTOR_SAFETENSOR_ARMS so the jailbreak defense loader and the
# main S2 pipeline agree on the on-disk filename for a compiled object.
# antidote + did join: each artifact is a SteeringVector whose
# sidecar carries its method (base_alpha for antidote). load_object
# re-hydrates by the stored method (AntidoteObject for antidote, plain
# VectorObject for did / vector / bidpo), never the filename.
VECTOR_SAFETENSOR_ARMS = frozenset({"vector", "bidpo", "antidote", "did"})

# Arms whose artifact HYDRATES to a plain VectorObject. VECTOR_SAFETENSOR_ARMS
# above names the on-disk CONTAINER; hydration diverges for antidote, whose
# safetensors container re-hydrates to AntidoteObject (load_object dispatches
# on the STORED method, never the filename), so the capeval engine must serve
# it under kind 'sisteer_object' — kind 'steering' asserts a VectorObject and
# refuses AntidoteObject (evals screen g8 failure, 2026-08-20).
PLAIN_VECTOR_ARMS = VECTOR_SAFETENSOR_ARMS - frozenset({"antidote"})


# LEGACY arm-name spellings -> the canonical arm name (renamed 2026-08-28:
# name the rank, leave "all" implicit — lora_r1_all's staged objects are
# rank 4, so the canonical name is lora_r4). BOTH names stay registered in
# ARMS (same class, trainer, and stored kind), and this table is the ONE
# record of the equivalence, consumed by:
#   - defense_cell.object_path_for / legacy_cell_id — a canonical cell id
#     resolves the LEGACY directory (`<si>__lora_r1_all/object.pt`) when the
#     canonical one holds no artifact, because the upstream fit still emits
#     the legacy name
#   - the provenance checks (two_si_run._check_budget_sidecar,
#     registry_adapter._check_meta_json) — a sidecar recording the legacy
#     spelling of the SAME cell is accepted AS THE RECORDED ALIAS; any other
#     cell_id/arm mismatch still refuses. The acceptance is deliberate and
#     table-driven, never a substring heuristic.
LEGACY_ARM_ALIASES: dict[str, str] = {"lora_r1_all": "lora_r4"}
#: canonical arm -> its legacy spelling (inverse of LEGACY_ARM_ALIASES).
CANONICAL_TO_LEGACY_ARM: dict[str, str] = {
    v: k for k, v in LEGACY_ARM_ALIASES.items()}


def canonical_arm(arm: str) -> str:
    """The canonical spelling of an arm name (identity for non-aliased
    arms). ``canonical_arm('lora_r1_all') == 'lora_r4'``."""
    return LEGACY_ARM_ALIASES.get(arm, arm)


def object_artifact_name(arm: str) -> str:
    """The on-disk filename for a compiled arm's artifact. ``load_object``
    dispatches on file CONTENT, not suffix, so the extension is
    cosmetic — but keeping it honest (``.safetensors`` for the vector/bidpo
    SteeringVector containers, ``.pt`` for torch-blob objects) means the
    sidecar path SteeringVector.save writes (``object.json``) lands correctly."""
    return "object.safetensors" if arm in VECTOR_SAFETENSOR_ARMS else "object.pt"


# -- artifact loading -------------------------------------------------------------

# Keyed by each class's own `kind` — the same string its save() writes.
# CastObject and CircuitBreakerObject stay in this table DELIBERATELY even
# though their arms left ARMS (2026-08-20): removal from the registry stops
# training/enumeration, but already-fitted artifacts exist on disk and a
# peer lane's serving/kind-coverage canary hydrates them through
# load_object. Dropping these entries would turn those artifacts into
# "unknown intervention kind" errors.
_KIND_LOADERS = {
    cls.kind: cls.load_blob
    for cls in (
        SoftTokenObject,
        AppendedSoftTokenObject,
        RankOneLoRAObject,
        MultiLayerVectorObject,
        MultiLayerLoRAObject,
        # Derived (never trained) task-vector difference of two lora_all
        # fits — sdf-base-steering round-2 (scripts/sdfbase/lora_subtract
        # .py builds them into a derived/ namespace); registered so
        # probe_score --object and every other load_object consumer can
        # hydrate the artifacts.
        LoRASubtractionObject,
        PSRObject,
        # Derived (never trained) PSR paired write — the coherent virtual
        # teacher of the round-2 functional-residual read (extraction/
        # paired_write.py; scripts/sdfbase/psr_residual_distill.py builds
        # them into derived/psr_paired_write/). Registered so probe_score
        # --object / ce_score --object and every other load_object
        # consumer can hydrate the artifacts.
        PSRPairedWriteObject,
        LoReFTObject,
        ReFTr1Object,
        AlphaSteerObject,
        AdaSteerObject,
        LoFiTObject,
        AlphaKLObject,
        CastObject,
        CircuitBreakerObject,
        DiffMeanObject,
        CaaStaticRObject,
        # the served-only Arditi arm's class: vendored as
        # extraction/static_r.py, kind "static_r" — NOT CaaStaticRObject's
        # "caa_static_r" (a different blob: it records neither steer_positions
        # nor site, and static_r's loader defaults would read it as a
        # one-layer, last-prompt-token block-OUTPUT edit)
        StaticRObject,
    )
}


def _is_safetensors_file(path: Path) -> bool:
    """Content sniff for the safetensors container: an 8-byte little-endian
    header length followed by the JSON header's opening brace. Dispatching
    on CONTENT rather than the filename suffix means a mis-suffixed
    artifact can never be silently mis-hydrated."""
    try:
        with path.open("rb") as f:
            head = f.read(9)
    except OSError:
        return False
    if len(head) < 9:
        return False
    header_len = int.from_bytes(head[:8], "little")
    return head[8:9] in (b"{", b" ") and 0 < header_len < 1 << 34


def _hydrate_object(path: Path, model, tok):
    """load_object's dispatch + construction, WITHOUT the device move —
    the ONE implementation both entry points below share, so a
    model-free hydration can never drift from the served one.

    ``model=None`` is the model-free path (capeval's post-load resolved
    spec, capeval/serving/resolved_spec.py): the safetensors branch
    never reads the model, so vector artifacts hydrate fully; torch-blob
    constructors are built around the live model (hidden size, decoder
    layers), so blob kinds refuse loudly rather than guessing shapes.

    An absent path that names an object ``objects/WITHHELD.json`` lists
    (a paper object this repository does not ship) is refused by name with
    the recorded reason, never as a bare missing file."""
    if not path.exists():
        from .extraction.committed_objects import refuse_if_withheld
        refuse_if_withheld(path)
    if _is_safetensors_file(path):
        sv = SteeringVector.load(path)
        # Dispatch on the STORED method (content, never suffix):
        # an antidote artifact must re-hydrate with its intrinsic
        # positions="last" + base_alpha semantics, not as a plain
        # all-positions vector (which would silently run the wrong method).
        if sv.method == "antidote":
            return AntidoteObject(sv)
        # Reconstruct the adapter state save() persisted (review
        # H-P1-2): without this, actadd_front's positions="front"/
        # skip_bos=False silently reloaded as the mode-all defaults on
        # resume. Artifacts saved before the fix carry no block and
        # keep the defaults.
        vo = (sv.metadata or {}).get("vector_object") or {}
        return VectorObject(sv,
                            positions=vo.get("positions", "all"),
                            skip_bos=vo.get("skip_bos", True),
                            norm_relative=vo.get("norm_relative", False))
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "kind" not in blob:
        raise ValueError(
            f"{path} is neither a safetensors artifact nor a torch blob "
            "with a 'kind' field — refusing to guess the object class"
        )
    kind = blob["kind"]
    if kind not in _KIND_LOADERS:
        raise ValueError(f"unknown intervention kind {kind!r} in {path}")
    if model is None:
        raise ValueError(
            f"{path} is a torch blob (kind {kind!r}); hydrating it needs "
            "the loaded model — model-free hydration serves safetensors "
            "vector artifacts only")
    return _KIND_LOADERS[kind](blob, model, tok)


def load_object(path: str | Path, model, tok):
    """Re-hydrate any saved intervention artifact, dispatching on the
    file's stored kind/magic (never on its filename suffix).

    Argument order — the PATH comes FIRST, then model, then tok. This is
    the one sisteer entry point whose leading argument is not
    ``(model, tok, ...)``, and a ported call died twice guessing the order,
    so prefer the keyword form, which makes the order impossible to get
    wrong: ``load_object(path, model=model, tok=tok)``.

    - ``path``: the saved artifact on disk (``object.pt`` torch blob or
      ``object.safetensors`` container). On the cluster, artifacts live on
      SHARED storage (``$SHARED/...``), never ``/home``.
    - ``model``: the ALREADY-LOADED HF causal LM the object will steer —
      used for device placement and kind-specific re-hydration.
    - ``tok``: that model's tokenizer.

    A safetensors container (sniffed by content) -> a SteeringVector
    artifact (the vector / bidpo / gkd arms), wrapped in VectorObject.
    Anything else -> a torch blob whose "kind" field picks the object
    class. Loaded objects are moved to the MODEL's device (leaving blobs
    on CPU makes PSR silently slow and errors LoReFT on GPU) — whatever
    that device is: cuda, Apple's mps, or cpu
    (``models.model_device``). The tensors keep their stored float32: every
    object's hook casts the model's activations up to the object's dtype
    for its own arithmetic and casts the result back to the activation's
    dtype, so a bf16 / fp16 model on any device is served by the same fp32
    object math the paper's CUDA runs used.
    """
    if not isinstance(path, (str, Path)):
        # The historic misuse is passing (model, tok, path) — the model
        # lands here and used to die deep in Path() with a message that
        # named neither the argument nor the right order.
        raise TypeError(
            "load_object takes the artifact PATH first — "
            "load_object(path, model, tok) — but path got a "
            f"{type(path).__name__}. Call it as "
            "load_object(path, model=model, tok=tok) so the order cannot "
            "be confused with the (model, tok, ...) convention elsewhere "
            "in sisteer.")
    obj = _hydrate_object(Path(path), model, tok)
    obj.to_(model_device(model))
    return obj


def load_object_cpu(path: str | Path):
    """Hydrate a SAFETENSORS artifact on CPU without a loaded model — the
    same dispatch and constructors as load_object (shared code, minus the
    device move), for consumers that need the object's RESOLVED FIELDS
    rather than a servable object. capeval's post-load intervention hash
    (capeval/serving/resolved_spec.py) is derived through this on engines
    and verifiers that never hold an HF model. Torch-blob kinds refuse
    loudly: their constructors read model dims/decoder layers."""
    return _hydrate_object(Path(path), None, None)
