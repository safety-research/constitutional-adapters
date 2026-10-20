"""Shared-pre-write-state combinations of PSR objects — the coherent virtual
teacher of the functional-residual read, and the servable
``psr_paired_write`` kind that carries it.

The estimand. Two PSR parents on ONE base model — the corpus fit ``P_mix``
and the register-matched control fit ``P_ctl`` — each write, at decoder
layer ℓ, ``u_obj,ℓ(z) = ReLU(w·z + b)·d`` on the layer's output residual
state. The functional residual "what the corpus fit does beyond the
control fit, evaluated on the same states" is the PAIRED write

    V_ℓ(z_ℓ) = u_mix,ℓ(z_ℓ) − u_ctl,ℓ(z_ℓ);      h_ℓ ← z_ℓ + V_ℓ(z_ℓ),

both gates reading the IDENTICAL pre-write state z_ℓ, one combined hook
per layer adding the difference, layer ℓ+1 seeing z_ℓ + V_ℓ(z_ℓ). Run as
a single forward this yields one self-consistent trajectory h^V(x): the
coherent virtual teacher, and the exact PSR analogue of the LoRA
task-vector subtraction (ΔW_mix·x − ΔW_ctl·x at the shared input x).

Why one COMBINED hook, never two sequential ones. The PSR gate reads the
very tensor it writes: a second hook on the same layer would evaluate its
gate on the FIRST hook's post-write state, so ``+mix then −ctl`` and
``−ctl then +mix`` are different functions, and neither is the estimand.
(``LoRASubtractionObject`` composes two hooks order-free ONLY
because LoRA hooks read the module INPUT, which neither hook edits.) The
declared application order is a constant on this module
(:data:`APPLICATION_ORDER`) and rides into every artifact; the planted
suite (tests/test_residual_teacher.py) runs BOTH orders on a toy model to
prove the implementation is the declared shared-state one and to
document the order sensitivity. :func:`sequential_paired_apply` exists for
that diagnostic and is never servable.

Two classes:

- :class:`SharedStateWriteObject` — the general Σ_i c_i·u_i(z) on the
  shared pre-write state, for ANY signed list of PSR parents. Not a
  persisted kind: it is the construction the calibration controls need
  (``B ⊕ C`` = ``[(+1, B), (+1, C)]``; the planted teacher ``R(B ⊕ C, B)``
  = ``[(+1, B), (+1, C), (−1, B)]``, whose write is EXACTLY ``u_C(z)`` up
  to fp cancellation — the real-model teacher identity check).
- :class:`PSRPairedWriteObject` — kind ``psr_paired_write``: exactly two
  parents, ``+pos − neg``, both parents' parameters stored BIT-EXACT,
  ``trainable == []``, ``scale`` multiplying the paired write (never the
  gate thresholds — the honest dose knob), provenance (both parents' cell
  ids / object shas / budget-sidecar shas / model id+revision / layers /
  fit legs) riding INSIDE the blob and in the ``derived_meta.json``
  sidecar the builder writes (scripts/sdfbase/psr_residual_distill.py),
  hydrated through the ordinary ``methods.load_object`` path. Structurally
  the sibling of ``multilayer.LoRASubtractionObject``.

Position legs: the campaign's document fits steer EVERY position
(``positions_at_fit="all"``) and the paired write reads the layer output
at every position, so masked parents are REFUSED rather than half-
supported (a masked parent would need its ``_fit_masked`` gating folded
into the combined write — undefined for a difference of two masks).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..hooks import PositionMaskedFit, _hidden_from_output, _replace_hidden
from ..models import get_decoder_layers
from .distill import DistillExample, _encode_aligned
from .psr import PSRObject
from .traindata import collate

#: The declared application order, recorded in every artifact. Both gates
#: read the identical pre-write layer output z; one combined hook adds
#: u_pos(z) − u_neg(z); the next layer sees z + V(z). Tests pin that the
#: implementation matches THIS and differs from either sequential order.
APPLICATION_ORDER = "shared_pre_write_state"

#: The two sequential (two-hook) orders — DIAGNOSTIC labels only, for the
#: order-sensitivity test and the calibration report. Never servable.
SEQUENTIAL_ORDERS = ("pos_then_neg", "neg_then_pos")


def _check_parent(tag: str, obj) -> None:
    if not isinstance(obj, PSRObject):
        raise TypeError(
            f"shared-state write: {tag} parent is {type(obj).__name__}, "
            "not a PSRObject — the paired write is defined on PSR gates")
    if obj.fit_positions != "all":
        raise ValueError(
            f"shared-state write: {tag} parent carries fit_positions="
            f"{obj.fit_positions!r} — only all-position parents are "
            "defined (a masked parent's write is gated by its own mask, "
            "which a difference of two writes cannot honor)")
    if obj._needs_tracking():
        raise ValueError(
            f"shared-state write: {tag} parent has a serve-time position "
            f"restriction (positions={obj.positions!r}, skip_bos="
            f"{obj.skip_bos}, response_header_len="
            f"{obj.response_header_len}) — parents must be plain "
            "all-position objects")


class SharedStateWriteObject(PositionMaskedFit):
    """Σ_i c_i · u_i(z) on the SHARED pre-write layer output z, for a
    signed list of PSR parents on identical layer sets. See the module
    docstring. ``terms`` = ``[(coef, PSRObject), ...]``; every parent's
    gate is evaluated on the same z, the signed writes are summed in
    float32, and ONE addition lands on the hidden state.

    Not a persisted kind (:attr:`kind` is a construction label): the
    calibration controls build these in-process; the servable two-parent
    case is :class:`PSRPairedWriteObject`.

    Gate telemetry: after a forward under :meth:`apply`, ``_last_gate``
    maps term index → {layer → [B, S] post-ReLU gate} and ``_last_pre``
    the matching pre-activations, all evaluated on the pre-write state
    (cleared at context exit, like PSRObject's stash)."""

    kind = "psr_shared_state_write"

    def __init__(self, terms: list[tuple[float, PSRObject]]):
        if not terms:
            raise ValueError("shared-state write needs at least one term")
        self.terms: list[tuple[float, PSRObject]] = []
        layers = None
        decoder = None
        for i, (coef, obj) in enumerate(terms):
            _check_parent(f"term {i}", obj)
            if layers is None:
                layers = list(obj.layers)
                decoder = obj._decoder
            elif list(obj.layers) != layers:
                raise ValueError(
                    f"shared-state write: term {i} layer set {obj.layers} "
                    f"differs from term 0's {layers} — the combined write "
                    "is only defined layerwise on ONE layer set")
            elif obj._decoder is not decoder:
                raise RuntimeError(
                    f"shared-state write: term {i} was hydrated against a "
                    "different model's decoder stack — every parent must "
                    "be built on the SAME model")
            self.terms.append((float(coef), obj))
        self.layers = list(layers)
        self._decoder = decoder
        self._last_gate: dict[int, dict[int, torch.Tensor]] = {}
        self._last_pre: dict[int, dict[int, torch.Tensor]] = {}

    @property
    def trainable(self) -> list[nn.Parameter]:
        return []  # a combination of frozen parents — nothing trains here

    def n_params(self) -> int:
        return 0

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def paired_write(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """The combined write Σ_i c_i·ReLU(w_i·z + b_i)·d_i at ``layer`` on
        the pre-write state ``hidden`` ([B, S, d]), in float32; stashes
        every term's gate/pre-activation for telemetry."""
        add = None
        for i, (coef, obj) in enumerate(self.terms):
            pre = obj.gate_pre(layer, hidden)          # [B, S], on z
            g = F.relu(pre)
            self._last_gate.setdefault(i, {})[layer] = g
            self._last_pre.setdefault(i, {})[layer] = pre
            d = obj.directions[layer].to(device=hidden.device)
            term = (coef * g).unsqueeze(-1) * d.float()  # [B, S, d]
            add = term if add is None else add + term
        return add

    def _make_hook(self, layer: int, scale: float):
        def hook(_mod, _args, output):
            if scale == 0.0:  # exact no-op, bit-identical to unsteered
                return output
            hidden = _hidden_from_output(output)
            add = self.paired_write(layer, hidden)
            return _replace_hidden(output, hidden + (scale * add).to(hidden.dtype))

        return hook

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        """ONE combined forward hook per layer, on the decoder-layer output
        (the tensor every PSR parent writes). ``scale`` multiplies the whole
        paired write; the gate thresholds are never touched."""
        if get_decoder_layers(model) is not self._decoder:
            raise RuntimeError(
                "shared-state write: apply() got a model whose decoder "
                "stack is not the one the parents were hydrated against")
        handles = [self._decoder[l].register_forward_hook(
            self._make_hook(l, float(scale))) for l in self.layers]
        try:
            yield None
        finally:
            for h in handles:
                h.remove()
            self._last_gate = {}
            self._last_pre = {}

    def to_(self, device) -> None:
        for _c, obj in self.terms:
            obj.to_(device)


class PSRPairedWriteObject(SharedStateWriteObject):
    """The servable coherent virtual teacher V = +pos − neg (kind
    ``psr_paired_write``), see the module docstring. Both parents ride
    bit-exact; ``provenance`` (the builder's parent identity block) rides
    in the blob and is re-validated by every consumer through the shared
    derived-object verdict (``train_budget.derived_object_problems``)."""

    kind = "psr_paired_write"

    def __init__(self, pos: PSRObject, neg: PSRObject,
                 provenance: dict | None = None):
        _check_parent("pos", pos)
        _check_parent("neg", neg)
        if list(pos.layers) != list(neg.layers):
            raise ValueError(
                f"psr_paired_write: parent layer sets differ ({pos.layers} "
                f"vs {neg.layers}) — the paired write is only defined "
                "layerwise on the same set")
        for l in pos.layers:
            if pos.directions[l].shape != neg.directions[l].shape:
                raise ValueError(
                    f"psr_paired_write: layer {l} direction shapes differ "
                    f"({tuple(pos.directions[l].shape)} vs "
                    f"{tuple(neg.directions[l].shape)})")
        super().__init__([(1.0, pos), (-1.0, neg)])
        self.pos = pos
        self.neg = neg
        self.provenance = dict(provenance or {})
        self.application_order = APPLICATION_ORDER

    # -- persistence ----------------------------------------------------------

    @staticmethod
    def _side(obj: PSRObject, layers) -> dict:
        return {
            "directions": {str(l): obj.directions[l].detach().float().cpu()
                           for l in layers},
            "gate_w": {str(l): obj.gate_w[l].detach().float().cpu()
                       for l in layers},
            "gate_b": {str(l): obj.gate_b[l].detach().float().cpu()
                       for l in layers},
        }

    @classmethod
    def _blob(cls, layers: list[int], pos_side: dict, neg_side: dict,
              provenance: dict | None) -> dict:
        """The ``psr_paired_write`` artifact dict — the ONE shape ``save``
        and ``compose_blob`` both write. Honest provenance rides IN the
        artifact (the builder fills it; an empty dict marks an ad-hoc/test
        object); ``positions`` is always ``"all"`` (masked parents are
        refused)."""
        return {"kind": cls.kind,
                "layers": layers,
                "application_order": APPLICATION_ORDER,
                "pos": pos_side,
                "neg": neg_side,
                "provenance": dict(provenance or {}),
                "positions": "all",
                "response_header_len": None}

    def save(self, path: Path) -> None:
        torch.save(self._blob([int(l) for l in self.layers],
                              self._side(self.pos, self.layers),
                              self._side(self.neg, self.layers),
                              self.provenance), path)

    @staticmethod
    def _parent_blob(blob: dict, side: str) -> dict:
        sub = blob[side]
        return {"kind": "psr",
                "layers": blob["layers"],
                "train_direction": True,
                "positions": "all",
                "response_header_len": None,
                "directions": sub["directions"],
                "gate_w": sub["gate_w"],
                "gate_b": sub["gate_b"]}

    @classmethod
    def compose_blob(cls, pos_blob: dict, neg_blob: dict,
                     provenance: dict | None = None) -> dict:
        """The paired-write ARTIFACT from two saved ``psr`` blobs — pure
        tensor bookkeeping, NO model needed (the builder runs on CPU).
        Parent tensors ride bit-exact (``.clone()`` only); every
        structural precondition of ``__init__`` is validated on the blobs,
        and ``load_blob`` re-validates through PSRObject's own loader."""
        for tag, b in (("pos", pos_blob), ("neg", neg_blob)):
            if b.get("kind") != "psr":
                raise ValueError(
                    f"psr_paired_write: {tag} parent kind is "
                    f"{b.get('kind')!r}, not 'psr'")
            if b.get("positions", "all") != "all":
                raise ValueError(
                    f"psr_paired_write: {tag} parent carries fit "
                    f"positions={b.get('positions')!r} — only all-position "
                    "parents are defined")
        layers = sorted(int(k) for k in pos_blob["directions"])
        if sorted(int(k) for k in neg_blob["directions"]) != layers:
            raise ValueError(
                "psr_paired_write: parent layer sets differ "
                f"({layers} vs {sorted(int(k) for k in neg_blob['directions'])})")
        if sorted(int(l) for l in pos_blob["layers"]) != layers:
            raise ValueError(
                "psr_paired_write: pos parent's layers key contradicts its "
                "direction tensors")
        for l in layers:
            for tag in ("directions", "gate_w", "gate_b"):
                a, b = pos_blob[tag][str(l)], neg_blob[tag][str(l)]
                if a.shape != b.shape:
                    raise ValueError(
                        f"psr_paired_write: layer {l} {tag} shapes differ "
                        f"({tuple(a.shape)} vs {tuple(b.shape)})")

        def _side_of_blob(b):
            return {tag: {str(l): b[tag][str(l)].clone() for l in layers}
                    for tag in ("directions", "gate_w", "gate_b")}
        return cls._blob(layers, _side_of_blob(pos_blob),
                         _side_of_blob(neg_blob), provenance)

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "PSRPairedWriteObject":
        order = blob.get("application_order")
        if order != APPLICATION_ORDER:
            raise ValueError(
                f"psr_paired_write artifact declares application_order="
                f"{order!r}; this implementation serves only "
                f"{APPLICATION_ORDER!r} — refusing to serve a different "
                "composition under this kind")
        pos = PSRObject.load_blob(cls._parent_blob(blob, "pos"), model, tok)
        neg = PSRObject.load_blob(cls._parent_blob(blob, "neg"), model, tok)
        return cls(pos, neg, provenance=blob.get("provenance") or {})


# --------------------------------------------------------------------------- #
# Diagnostics: the sequential (two-hook) orders, and gate liveness            #
# --------------------------------------------------------------------------- #


@contextmanager
def sequential_paired_apply(pos: PSRObject, neg: PSRObject, model,
                            scale: float = 1.0, order: str = "pos_then_neg"):
    """DIAGNOSTIC ONLY — the two-hook composition the paired write is NOT:
    each parent's own hook registered on the same layer, in ``order``, so
    the second gate reads the first hook's POST-write state. Used by the
    planted suite to show the order sensitivity and to prove the servable
    object is the declared shared-state one; never a servable kind."""
    if order not in SEQUENTIAL_ORDERS:
        raise ValueError(f"order must be one of {SEQUENTIAL_ORDERS}")
    _check_parent("pos", pos)
    _check_parent("neg", neg)
    if list(pos.layers) != list(neg.layers):
        raise ValueError("sequential paired apply: parent layer sets differ")
    decoder = get_decoder_layers(model)
    first, second = ((pos, scale), (neg, -scale))
    if order == "neg_then_pos":
        first, second = second, first
    handles = []
    for l in pos.layers:
        for obj, s in (first, second):
            handles.append(decoder[l].register_forward_hook(
                obj._make_hook(l, float(s), None)))
    try:
        yield None
    finally:
        for h in handles:
            h.remove()
        pos._last_gate = {}
        pos._last_pre = {}
        neg._last_gate = {}
        neg._last_pre = {}


def paired_gate_liveness(model, tok, *, obj: SharedStateWriteObject,
                         examples: list[DistillExample], max_len: int = 8192,
                         chunk: int = 4) -> dict:
    """Per-term, per-layer gate liveness of a shared-state write object —
    the fraction of valid positions at which each parent's gate is OPEN on
    the pre-write state the combined hook actually used — plus ``any``:
    the fraction of positions where the paired write is nonzero (some term
    open). Mirrors :func:`psr.psr_gate_liveness` (no-grad student forwards
    with the object applied, chunked)."""
    device = next(model.parameters()).device
    obj.to_(device)
    open_ct = {i: {l: 0 for l in obj.layers} for i in range(len(obj.terms))}
    any_ct = {l: 0 for l in obj.layers}
    valid_ct = 0
    for i0 in range(0, len(examples), chunk):
        exs = examples[i0:i0 + chunk]
        encoded = [_encode_aligned(tok, ex, max_len)[1] for ex in exs]
        batch = collate(encoded, tok.pad_token_id).to(device)
        with torch.no_grad(), obj.apply(model, scale=1.0):
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
            gates = {i: {l: obj._last_gate[i][l] for l in obj.layers}
                     for i in range(len(obj.terms))}
        valid = batch.attention_mask.bool()
        valid_ct += int(valid.sum())
        for l in obj.layers:
            any_open = torch.zeros_like(valid)
            for i in gates:
                op = gates[i][l] > 0
                open_ct[i][l] += int((op & valid).sum())
                any_open |= op
            any_ct[l] += int((any_open & valid).sum())
    denom = max(valid_ct, 1)
    return {"terms": {i: {l: open_ct[i][l] / denom for l in obj.layers}
                      for i in open_ct},
            "any": {l: any_ct[l] / denom for l in obj.layers},
            "n_valid_positions": valid_ct}
