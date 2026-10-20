"""Condition registry -> lens worker registry: the cutover shim.

The seam between the unified sweep (Evals' lane) and the vLLM lens
serving path (this lane). The sweep enumerates CONDITIONS — a defense
object at a serving coefficient and a mask variant; this shim turns each
into a preloadable, hash-verified worker-registry entry plus the
immutable request selector the driver attaches to a request.

THE INPUT IS A PLAIN DATA CONTRACT, NOT A FOREIGN CLASS. Each condition is
an ordinary mapping so the two lanes can evolve independently — the sweep's
``Condition`` internals can move and this registry format can move without
either breaking the other:

    {
      "cell_id":       str,   # the served id; the selector's petri_object
      "artifact_path": str,   # the fit object's bytes on disk
      "resolved_spec": {..},  # the F9 resolved spec; carries "positions"
                              #   ("response" selects the cross-phase mask)
      "coefficient":   float, # the serving coefficient c (a MULTIPLIER on
                              #   the object's baked scale, not an absolute)
                              #   — finite and STRICTLY POSITIVE (c == 0
                              #   silently serves a passthrough as an arm;
                              #   c < 0 silently serves the negated edit)
      "object_sha256": str,   # the ORIGINAL fit artifact's sha (provenance)
      "sidecar_path":  str | None,   # the .json sidecar, when the kind has one
      "sidecar_sha256": str,  # REQUIRED iff sidecar_path is given: the
                              #   sidecar bytes' sha, verified BEFORE the
                              #   bytes are adapted or folded — the sidecar
                              #   carries served semantics (antidote's
                              #   base_alpha), so unbound sidecar bytes
                              #   would be an unverified half of the object
      "expect_served_coeff": float | None,  # OPTIONAL cross-check, below
    }

THE CONVENTION CROSS-CHECK (``expect_served_coeff``). The fold and its
verifier both assume the coefficient MULTIPLIES the baked scale — so the
verifier cannot falsify the assumption it shares with the fold (a green
check that cannot go red, the failure class this programme keeps finding).
The only thing that can catch a convention mismatch is a value computed
OUTSIDE that assumption: the driver, from its OWN understanding of what
the served edit should be, declares the absolute served coefficient it
expects; the shim asserts its independently-folded artifact reproduces
that and REFUSES otherwise. Two independent computations of the same
absolute agreeing is real evidence; a multiply-vs-replace divergence
becomes a refused fold rather than a silent factor-of-S serve. It is
defined only for the coeff-locus kinds, where "the coefficient" is a
single unambiguous scalar; a distributed-magnitude kind (psr / vector /
vector_all, whose coefficient lives spread across a tensor) refuses the
field, since there is no scalar for the driver to have declared.

THE COEFFICIENT IS FOLDED INTO A DERIVED ARTIFACT, never carried as a
request knob (the H3 contract: a request carries only immutable ids; the
selector's scale is a cross-check, never a control). A condition at
coefficient ``c`` becomes an artifact whose deployment at selector-scale
1.0 reproduces the original artifact's deployment at scale ``c``. Every
op in the vocabulary applies its coefficient OUTSIDE its gate — the edit
is ``scale·coeff·gate(h)·v`` for gated ops, ``scale·coeff·v`` for
additive — so the edit is LINEAR in ``c`` for every kind, and the fold is
always "multiply the object's baked scale by ``c``", written at whichever
slot each kind holds it (:data:`_COEFF_LOCUS`). Each fold is VERIFIED, not
trusted: the derived artifact is re-adapted and asserted to be exactly
``c`` times the original op, the same bit-exact discipline as the rest of
the serving lane. A kind with no linear scale locus (``loreft``, whose
affine edit does not factor through a single tensor) REFUSES ``c != 1.0``
rather than emitting a wrong artifact.

TWO SHAS, USED DIFFERENTLY:

- the ORIGINAL sha keys FIT PROVENANCE and the train/serve template
  check — it is the artifact whose sidecar records model/revision/
  template. It is recorded, never used as the served identity. Checking
  the derived sha against fit provenance would refuse everything.
- the DERIVED sha is recorded as SERVED — it is the bytes that actually
  run, and it is what the intervention hash and the selector's
  ``expect_hash`` bind to. A worker refuses at load if the derived bytes
  do not rehash to it. A row recording only the original is lying about
  what executed; a row recording only the derived cannot be traced to a
  fit.

THE EFFECTIVE COEFFICIENT is ``c``, recorded per condition. The derived
artifact serves at selector-scale 1.0, so a naive stamp would read 1.0
and every point in a coefficient sweep would look identical; the manifest
records the ``c`` that was folded in, and the post-bake 1.0 is the
selector scale, labelled as such and never as the coefficient.

SCOPE: this shim emits the ARM cells. The base (unsteered) leg is the
driver's concern — served by omitting the selector, which the lens engine
runs as passthrough on the same serving path, so a lens-served arm pairs
against a lens-served base without this shim minting a base cell. The
worker registry writing/loading itself lives in ``lens_steer`` and is
reused verbatim for hash re-verification.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls


class RegistryShimError(RuntimeError):
    """A shim refusal — a malformed condition, an unfoldable coefficient,
    or a fold that did not reproduce ``c`` times the intervention. Loud,
    never a silently-wrong served artifact."""


#: Where each kind carries the scale the coefficient multiplies. "coeff"
#: means the adapted op's ``coeff`` field (the kinds that bake base_scale
#: / base_alpha); "vector" means the op's magnitude tensor, whose scale
#: the op's ``coeff`` leaves at 1.0. A kind absent here has no linear
#: scale locus and refuses any ``c != 1.0`` (loreft: its affine edit does
#: not factor through one tensor, so no byte fold reproduces c·edit).
_COEFF_LOCUS: Dict[str, str] = {
    "alphasteer": "coeff",
    "alphakl": "coeff",
    "static_r": "coeff",
    "cosine_gated_r": "coeff",
    "lowrank_gated": "coeff",
    "loreft_both": "coeff",
    "antidote": "coeff",
    "psr": "vector",
    "vector": "vector",
    "vector_all": "vector",
    # the o_proj LoRA kinds: the edit is (x @ A) @ Bᵀ, linear in B, so
    # the fold scales the WRITE basis — B, and only B (scaling A too
    # would fold c², and scaling the op's coeff instead would leave the
    # artifact bytes identical across coefficients, losing the
    # served-vs-fit hash distinction the derived sha exists for).
    "lora": "lora_b",
    "lora_all": "lora_b",
}
#: kinds served on the lens path but with NO linear coefficient fold —
#: servable at their trained strength (c == 1.0) only.
_UNFOLDABLE = ("loreft",)

REGISTRY_INDEX = ls.REGISTRY_INDEX
#: the atomically published pointer file at a registry ROOT: its content
#: is the current build's directory name under :data:`REGISTRY_BUILDS`.
REGISTRY_POINTER = "CURRENT"
#: where the immutable versioned builds live under a registry root.
REGISTRY_BUILDS = "builds"


@dataclass(frozen=True)
class ConditionResult:
    """One condition's serving identity, for the driver's row stamp."""

    cell_id: str
    kind: str
    #: the immutable request selector — {petri_object, expect_hash}
    selector: Dict[str, str]
    #: the original fit artifact's sha (provenance / fit-template key)
    original_sha256: str
    #: the served object's sha over ARTIFACT + SIDECAR — what actually
    #: runs. It must span both parts, because a sidecar-carrying kind
    #: (antidote) folds the coefficient into the SIDECAR and leaves the
    #: artifact untouched: an artifact-only sha would read identical to
    #: the fit even at a different coefficient, silently losing the
    #: served-vs-fit distinction. For a kind with no sidecar this equals
    #: the artifact sha.
    derived_sha256: str
    #: the coefficient folded in — the value to stamp, NOT the post-bake
    #: selector scale of 1.0
    effective_coefficient: float
    #: True when the derived bytes differ from the original (c != 1.0)
    coefficient_was_folded: bool


@dataclass(frozen=True)
class ShimResult:
    #: the RESOLVED immutable build directory — what consumers hand to
    #: the serving side (``WorkerObjectRegistry.load`` takes it as-is).
    registry_dir: str
    conditions: Tuple[ConditionResult, ...]
    assistant_header_ids: Optional[Tuple[int, ...]]
    #: the registry ROOT (holds ``CURRENT`` + ``builds/``) — what a
    #: later rebuild targets; ``resolve_registry_dir(root_dir)`` yields
    #: whatever build is currently published.
    root_dir: Optional[str] = None
    #: the served template's assistant turn-end scaffold recorded in the
    #: registry (the all-turns mask's second marker); None when no
    #: condition serves ``positions="response_all_turns"`` and none was
    #: derivable.
    assistant_turn_end_ids: Optional[Tuple[int, ...]] = None
    #: the served template's DECLARED thinking-ON header recorded in the
    #: registry (the second marker of ``@thinking`` conditions); None when
    #: no condition declares the thinking-ON render.
    assistant_header_ids_thinking_on: Optional[Tuple[int, ...]] = None


# --- coefficient folding ------------------------------------------------------


def _finite_c(c: Any, what: str = "coefficient") -> float:
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise RegistryShimError(f"{what} must be a number, got {c!r}")
    c = float(c)
    if c != c or c in (float("inf"), float("-inf")):
        raise RegistryShimError(f"{what} must be finite, got {c!r}")
    return c


def _finite_positive_c(c: Any) -> float:
    c = _finite_c(c)
    if c <= 0.0:
        raise RegistryShimError(
            f"coefficient must be strictly positive, got {c!r} — c == 0 "
            "folds every op to an exact no-op (the arm would serve as a "
            "passthrough while its row claims a defense), and c < 0 serves "
            "the NEGATED edit; neither is a sweep point, both are almost "
            "certainly a driver bug. An unsteered leg is served by omitting "
            "the selector, never by a zero coefficient")
    return c


def _load_blob(kind: str, artifact_bytes: bytes) -> Dict[str, Any]:
    try:
        blob = torch.load(io.BytesIO(bytes(artifact_bytes)),
                          map_location="cpu", weights_only=True)
    except Exception as e:
        raise RegistryShimError(
            f"kind {kind!r}: artifact did not load as a torch blob: {e}"
        ) from e
    if not isinstance(blob, dict):
        raise RegistryShimError(f"kind {kind!r}: artifact is not a dict blob")
    return blob


def _save_blob(blob: Mapping[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(dict(blob), buf)
    return buf.getvalue()


def _scale_layer_map(kind: str, group_name: str, group: Any,
                     c: float) -> Dict[Any, torch.Tensor]:
    if not isinstance(group, Mapping) or not group:
        raise RegistryShimError(
            f"kind {kind!r}: {group_name!r} is not a non-empty mapping — "
            "cannot fold the coefficient into it")
    return {k: (v * c) for k, v in group.items()}


def _fold_coefficient(kind: str, artifact_bytes: bytes,
                      sidecar_bytes: bytes, c: float
                      ) -> Tuple[bytes, bytes]:
    """Return (derived_artifact_bytes, derived_sidecar_bytes) whose
    deployment at selector-scale 1.0 equals the original's at scale ``c``.

    ``c == 1.0`` is the identity fold — the bytes are returned unchanged,
    so a condition at trained strength has ``derived_sha == original_sha``
    and needs no re-verification. Any other ``c`` writes the scale locus
    and the caller verifies the result reproduces ``c``× the op."""
    if c == 1.0:
        return bytes(artifact_bytes), bytes(sidecar_bytes)
    if kind in _UNFOLDABLE:
        raise RegistryShimError(
            f"kind {kind!r} has no linear coefficient locus (its edit does "
            f"not factor through a single scaled tensor), so c={c} cannot be "
            "folded into a derived artifact. Serve it at its trained "
            "strength (c=1.0), or extend the adapter to carry an explicit "
            "coefficient before sweeping it")
    if kind not in _COEFF_LOCUS:
        raise RegistryShimError(
            f"kind {kind!r} is not a coefficient-foldable lens kind "
            f"(foldable: {sorted(_COEFF_LOCUS)}; unfoldable-at-c!=1: "
            f"{list(_UNFOLDABLE)}) — refusing to guess where its scale lives")

    if kind == "antidote":
        # base_alpha lives in the JSON sidecar; the artifact is unchanged
        # and the sidecar (which is hashed into the identity) carries the
        # fold, so the derived object still hashes differently.
        if not sidecar_bytes:
            raise RegistryShimError(
                "kind 'antidote': coefficient fold needs the base_alpha "
                "sidecar, which is absent")
        try:
            side = json.loads(bytes(sidecar_bytes).decode("utf-8"))
            meta = side["metadata"]
            meta["base_alpha"] = float(meta["base_alpha"]) * c
        except (KeyError, ValueError, TypeError,
                json.JSONDecodeError) as e:
            raise RegistryShimError(
                f"kind 'antidote': could not fold c into sidecar "
                f"metadata.base_alpha: {e}") from e
        return (bytes(artifact_bytes),
                json.dumps(side).encode("utf-8"))

    if kind == "vector":
        # SteeringVector safetensors container: scale every layer tensor.
        try:
            from safetensors.torch import load as _st_load, save as _st_save
            tensors = _st_load(bytes(artifact_bytes))
        except Exception as e:
            raise RegistryShimError(
                f"kind 'vector': artifact did not load as safetensors: {e}"
            ) from e
        scaled = {name: (t * c) for name, t in tensors.items()}
        return _st_save(scaled), bytes(sidecar_bytes)

    blob = _load_blob(kind, artifact_bytes)
    if _COEFF_LOCUS[kind] == "coeff":
        if "base_scale" not in blob:
            raise RegistryShimError(
                f"kind {kind!r}: no 'base_scale' in the blob to fold c into")
        blob["base_scale"] = float(blob["base_scale"]) * c
    elif _COEFF_LOCUS[kind] == "lora_b":
        # scale the WRITE basis (B) alone: the served delta is
        # (x @ A) @ (cB)ᵀ = c · (x @ A) @ Bᵀ, exactly linear in c.
        if kind == "lora_all":
            blob["B"] = _scale_layer_map("lora_all", "B", blob.get("B"), c)
        else:  # "lora": single-layer tensors, not a layer map
            B = blob.get("B")
            if not torch.is_tensor(B):
                raise RegistryShimError(
                    "kind 'lora': no tensor 'B' in the blob to fold c into")
            blob["B"] = B * c
    else:  # "vector" locus in a torch blob
        if kind == "psr":
            blob["directions"] = _scale_layer_map(
                "psr", "directions", blob.get("directions"), c)
        elif kind == "vector_all":
            blob["vectors"] = _scale_layer_map(
                "vector_all", "vectors", blob.get("vectors"), c)
        else:  # pragma: no cover — locus map and this branch must agree
            raise RegistryShimError(
                f"kind {kind!r} has a 'vector' locus but no fold rule")
    return _save_blob(blob), bytes(sidecar_bytes)


def _op_signature(op: la.HookOp) -> Tuple:
    """The op's identity apart from its magnitude — everything the fold
    must leave untouched. Used to assert the derived op differs from the
    original ONLY in its coefficient locus."""
    tensors = tuple(
        None if getattr(op, name) is None
        else tuple(getattr(op, name).shape)
        for name in la._ALL_TENSOR_FIELDS)
    return (op.op, op.layer, op.positions, op.skip_bos, op.phase,
            op.threshold, op.header_ids, op.turn_end_ids,
            op.thinking_header_ids, tensors)


def _adapt_condition(kind: str, resolved_spec: Mapping[str, Any],
                     art: bytes, side: bytes,
                     header_ids: Optional[Tuple[int, ...]],
                     turn_end_ids: Optional[Tuple[int, ...]],
                     thinking_header_ids: Optional[Tuple[int, ...]]
                     ) -> Tuple[la.HookOp, ...]:
    """Adapt a condition's bytes exactly as the worker registry load will:
    the committed adapter, then the declared thinking-ON header attached to
    a declaring spec (``lens_adapters.attach_thinking_header`` — handed the
    registry's ON header iff the spec declares, like
    ``WorkerObjectRegistry.load``), so the op signature the fold verifier
    compares covers the second header too."""
    ops = la.get_adapter(kind)(dict(resolved_spec), art, side,
                               header_ids=header_ids, turn_end_ids=turn_end_ids)
    return la.attach_thinking_header(
        ops, resolved_spec,
        thinking_header_ids if la.declares_thinking(resolved_spec) else None)


def _verify_fold(kind: str, resolved_spec: Mapping[str, Any],
                 orig_art: bytes, orig_side: bytes,
                 deriv_art: bytes, deriv_side: bytes, c: float,
                 header_ids: Optional[Tuple[int, ...]], *,
                 turn_end_ids: Optional[Tuple[int, ...]] = None,
                 thinking_header_ids: Optional[Tuple[int, ...]] = None) -> None:
    """Assert the derived artifact is exactly ``c``× the original — same
    ops, same structure, scaled only at the coefficient locus. The check
    is independent of the fold: it re-adapts both artifacts through the
    committed adapter and compares the resulting ops, so a fold that
    touched the wrong field, or scaled a gate, or dropped a layer, is
    caught here rather than at serve time."""
    ops_o = _adapt_condition(kind, resolved_spec, orig_art, orig_side,
                             header_ids, turn_end_ids, thinking_header_ids)
    ops_d = _adapt_condition(kind, resolved_spec, deriv_art, deriv_side,
                             header_ids, turn_end_ids, thinking_header_ids)
    if len(ops_o) != len(ops_d):
        raise RegistryShimError(
            f"kind {kind!r}: fold changed the op count ({len(ops_o)} -> "
            f"{len(ops_d)}) — a coefficient fold must not add or drop ops")
    locus = _COEFF_LOCUS[kind]
    for i, (a, b) in enumerate(zip(ops_o, ops_d)):
        if locus == "coeff":
            if _op_signature(a) != _op_signature(b):
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: fold changed op structure, not "
                    "just its coefficient")
            for name in la._ALL_TENSOR_FIELDS:
                ta, tb = getattr(a, name), getattr(b, name)
                if ta is not None and not torch.equal(ta, tb):
                    raise RegistryShimError(
                        f"kind {kind!r} op {i}: a coeff-locus fold changed "
                        f"tensor {name!r}; only the scalar coeff may move")
            if abs(b.coeff - c * a.coeff) > 1e-6 * max(1.0, abs(c * a.coeff)):
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: derived coeff {b.coeff} != c*orig "
                    f"{c * a.coeff} (c={c}, orig coeff={a.coeff})")
        elif locus == "lora_b":  # ONLY the write basis scales
            if _op_signature(a) != _op_signature(b):
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: fold changed op structure "
                    "(op/layer/positions/phase/header/shapes), not just "
                    "its write-basis magnitude")
            if abs(b.coeff - a.coeff) > 1e-9:
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: a lora_b fold moved coeff "
                    f"({a.coeff} -> {b.coeff}); it must stay put")
            for name in la._ALL_TENSOR_FIELDS:
                if name == "lowrank_b":
                    continue  # the one field the fold is allowed to scale
                ta, tb = getattr(a, name), getattr(b, name)
                if ta is not None and not torch.equal(ta, tb):
                    raise RegistryShimError(
                        f"kind {kind!r} op {i}: a lora_b fold changed "
                        f"tensor {name!r} — scaling the READ basis too "
                        "would fold c², a different edit; only "
                        "'lowrank_b' may move")
            ba, bb = a.lowrank_b, b.lowrank_b
            if ba is None or bb is None:
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: expected a write basis to "
                    "scale")
            if not torch.allclose(bb, c * ba, rtol=1e-5, atol=1e-6):
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: derived write basis != "
                    f"c*orig (c={c})")
        else:  # "vector" locus: op.coeff stays put, ONLY the vector scales
            if _op_signature(a) != _op_signature(b):
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: fold changed op structure "
                    "(op/layer/positions/phase/threshold/header/shapes), "
                    "not just its vector magnitude")
            if abs(b.coeff - a.coeff) > 1e-9:
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: a vector-locus fold moved coeff "
                    f"({a.coeff} -> {b.coeff}); it must stay put")
            for name in la._ALL_TENSOR_FIELDS:
                if name == "vector":
                    continue  # the one field the fold is allowed to scale
                ta, tb = getattr(a, name), getattr(b, name)
                if ta is not None and not torch.equal(ta, tb):
                    raise RegistryShimError(
                        f"kind {kind!r} op {i}: a vector-locus fold changed "
                        f"tensor {name!r} — a gate or basis that scales "
                        "with c changes the op's MATH, not its magnitude; "
                        "only 'vector' may move")
            va, vb = a.vector, b.vector
            if va is None or vb is None:
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: expected a vector to scale")
            if not torch.allclose(vb, c * va, rtol=1e-5, atol=1e-6):
                raise RegistryShimError(
                    f"kind {kind!r} op {i}: derived vector != c*orig vector "
                    f"(c={c})")


def _assert_served_coeff(kind: str, resolved_spec: Mapping[str, Any],
                         deriv_art: bytes, deriv_side: bytes,
                         header_ids: Optional[Tuple[int, ...]],
                         expect: float, *,
                         turn_end_ids: Optional[Tuple[int, ...]] = None
                         ) -> None:
    """Cross-check the folded artifact against the ABSOLUTE served
    coefficient the driver declared — the one guard that can catch a
    multiply-vs-replace convention mismatch, because ``expect`` is
    computed outside the fold's own assumption. Coeff-locus kinds only:
    a distributed-magnitude kind has no single scalar the driver could
    have declared, so it refuses the field rather than pretend to check
    it."""
    if _COEFF_LOCUS.get(kind) != "coeff":
        raise RegistryShimError(
            f"kind {kind!r}: expect_served_coeff is defined only for kinds "
            "whose coefficient is a single scalar (the coeff-locus kinds); "
            f"{kind!r} carries its magnitude in a tensor, so there is no "
            "scalar served coefficient to cross-check. Drop the field for "
            "this condition — the c*vector fold verify already covers it")
    ops = la.get_adapter(kind)(dict(resolved_spec), deriv_art, deriv_side,
                               header_ids=header_ids, turn_end_ids=turn_end_ids)
    for i, op in enumerate(ops):
        if abs(op.coeff - expect) > 1e-6 * max(1.0, abs(expect)):
            raise RegistryShimError(
                f"kind {kind!r} op {i}: the folded artifact serves at coeff "
                f"{op.coeff}, but the driver declared expect_served_coeff="
                f"{expect}. The fold and the driver disagree on what the "
                "served coefficient should be — the likely cause is a "
                "multiply-vs-replace convention mismatch on the coefficient. "
                "Refusing rather than serving an edit off by that factor")


# --- building the registry ----------------------------------------------------


_REQUIRED_KEYS = ("cell_id", "artifact_path", "resolved_spec",
                  "coefficient", "object_sha256")


def _read_condition(cond: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [k for k in _REQUIRED_KEYS if k not in cond]
    if missing:
        raise RegistryShimError(
            f"condition is missing required key(s) {missing}: "
            f"{sorted(cond)} — the data contract is "
            f"{list(_REQUIRED_KEYS)} (+ sidecar_path with its "
            "sidecar_sha256, + optional expect_served_coeff)")
    unknown = sorted(set(cond) - set(_REQUIRED_KEYS)
                     - {"sidecar_path", "sidecar_sha256",
                        "expect_served_coeff"})
    if unknown:
        raise RegistryShimError(
            f"condition {cond['cell_id']!r} carries unknown key(s) {unknown} "
            "— refusing rather than silently ignoring a field the driver "
            "meant to pass")
    # the sidecar travels hash-bound or not at all: a sidecar-carrying
    # kind (antidote) keeps served SEMANTICS (base_alpha) in the sidecar,
    # so unbound sidecar bytes would be an unverified half of the object.
    if cond.get("sidecar_path") and "sidecar_sha256" not in cond:
        raise RegistryShimError(
            f"condition {cond['cell_id']!r} carries sidecar_path but no "
            "sidecar_sha256 — the artifact's bytes are hash-bound "
            "(object_sha256) and the sidecar's must be too; the shim "
            "verifies BOTH before adapting anything")
    if cond.get("sidecar_sha256") and not cond.get("sidecar_path"):
        raise RegistryShimError(
            f"condition {cond['cell_id']!r} carries sidecar_sha256 without "
            "a sidecar_path — a hash of nothing binds nothing; drop the "
            "field or name the sidecar it covers")
    spec = cond["resolved_spec"]
    if not isinstance(spec, Mapping) or "kind" not in spec:
        raise RegistryShimError(
            f"condition {cond['cell_id']!r}: resolved_spec must be a mapping "
            "carrying at least 'kind'")
    # the two serve-declaration keys are legal spec content — a condition
    # serving an object out of its fit support declares it IN the hashed
    # spec (serve_declaration: serve_out_of_support / serve_experiment) —
    # and are validated here, so a
    # malformed declaration refuses before any bytes are read or folded
    try:
        la.cell_declarations(spec)
    except la.AdapterError as e:
        raise RegistryShimError(
            f"condition {cond['cell_id']!r}: {e}") from e
    return dict(cond)


def build_lens_registry(
    conditions: Sequence[Mapping[str, Any]],
    out_dir: Any,
    *,
    assistant_header_ids: Optional[Sequence[int]] = None,
    tokenizer: Any = None,
    assistant_turn_end_ids: Optional[Sequence[int]] = None,
    assistant_header_ids_thinking_on: Optional[Sequence[int]] = None,
) -> ShimResult:
    """Write the worker registry for a set of lens-served conditions and
    return each one's serving identity.

    ``assistant_header_ids_thinking_on`` (or the same ``tokenizer``, through
    ``chat_markers.generation_header_ids_under(tok, enable_thinking=True)``)
    is required iff any condition's spec DECLARES the thinking-ON render
    (``serve_out_of_support: ["thinking"]``): the declared render's header
    is the second boundary marker of every response op of that condition.
    Derived only when a condition declares it; a template whose thinking
    switch does not move the header (gemma-4) refuses as vacuous, and one
    with no switch refuses outright.

    ``assistant_header_ids`` (or a ``tokenizer`` to derive them from) are
    required iff any condition serves ``positions="response"`` — the
    cross-phase mask is located by matching those ids, so a response
    condition without them cannot be served. They are recorded once for
    the whole registry (one registry, one served template).

    ``assistant_turn_end_ids`` (or the same ``tokenizer``, through
    ``chat_markers.assistant_turn_end_ids`` — the terminator plus the glue
    before the next turn's opener) are required iff any condition serves
    ``positions="response_all_turns"`` — the all-turns mask ends every
    earlier assistant turn's span after that scaffold, located by matching
    those ids (lens_adapters, THE ALL-TURNS MASK). A successor-bounded
    template (GLM-4.x) has no terminator, so such a condition refuses on
    it. Recorded once per registry like the header.

    The written registry is re-loaded through the committed
    :meth:`lens_steer.WorkerObjectRegistry.load` before returning, so
    every content hash, intervention hash, selector binding, adapter
    translation, and (when present) the assistant-header cross-check is
    exercised here rather than first failing inside a GPU worker.

    REBUILDS ARE CRASH- AND CONCURRENCY-SAFE by construction. Every blob
    and the index are written read-only (0440), so a build is never
    edited in place. ``out_dir`` is a registry ROOT, not the registry:
    each build lands whole in an IMMUTABLE versioned directory
    (``out_dir/builds/<build-id>/``, staged under a dot-name invisible
    to resolution until verified), and the only mutable object at the
    root is the POINTER file ``out_dir/CURRENT``, published atomically
    (write-fsync-rename). Consequences, each load-bearing:

    - a CONCURRENT READER that resolved earlier keeps its immutable
      build — nothing it holds is ever renamed away or deleted; a
      reader resolving mid-publish sees either the old pointer or the
      new one, never a torn or missing registry;
    - a CRASH anywhere leaves the previous ``CURRENT`` (and its build)
      fully intact — there is no instant at which the canonical
      registry does not exist; a crashed staging dir is dot-named junk
      that resolution can never reach;
    - old builds are NEVER deleted by the builder — pruning is the
      owner's explicit act, so no build a live worker resolved can be
      pulled out from under it.

    Consumers resolve the root ONCE via :func:`resolve_registry_dir`
    (which also accepts an already-resolved build dir, or a flat
    ``export_worker_registry`` dir) and pass the resolved immutable path
    to serving. A non-empty target that is neither a registry root nor
    empty refuses before any mutation; a FLAT legacy build at the root
    refuses with a migration note rather than mixing layouts."""
    out_dir = Path(out_dir)
    _check_registry_root(out_dir)
    builds = out_dir / REGISTRY_BUILDS
    builds.mkdir(parents=True, exist_ok=True)
    build_id = (f"build-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                f"-{os.getpid()}-{time.time_ns()}")
    staging = builds / f".building-{build_id}"
    staging.mkdir()  # fresh by construction (pid+ns unique); never reused

    try:
        result = _build_registry_into(
            staging, conditions,
            assistant_header_ids=assistant_header_ids, tokenizer=tokenizer,
            assistant_turn_end_ids=assistant_turn_end_ids,
            assistant_header_ids_thinking_on=assistant_header_ids_thinking_on)
        # DURABILITY ORDER (each step must complete before the next, or
        # a storage crash could leave CURRENT naming a build whose bytes
        # the disk never kept):
        #   1. every generated file + every nested dir + the build dir
        #      itself (bottom-up), while still dot-staged;
        _fsync_tree(staging)
        final = builds / build_id
        # dot-staged -> visible, one atomic rename. Resolution only ever
        # follows CURRENT, so even the visible dir serves nothing until
        # the pointer names it.
        os.rename(staging, final)
        #   2. builds/ AFTER the rename — the rename lives in this dir's
        #      entries and is not durable until the dir is synced;
        _fsync_path(builds)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # If THIS step fails (or the process dies first), CURRENT still
    # names the previous build: the new one is an unreferenced immutable
    # orphan, the canonical registry never wavers, and a retry publishes
    # a fresh build id. (Should a storage crash nonetheless leave a
    # CURRENT that resolution refuses — torn root, pointer to a lost
    # build — resolve_registry_dir refuses LOUDLY, never guesses, and
    # the documented recovery is exactly this function: a rebuild into
    # the same root repairs the pointer.)
    #   3. the pointer publish: tmp write -> tmp fsync -> atomic replace
    #      -> root dir fsync (_publish_pointer).
    _publish_pointer(out_dir, build_id)
    return ShimResult(registry_dir=str(final),
                      conditions=result.conditions,
                      assistant_header_ids=result.assistant_header_ids,
                      root_dir=str(out_dir),
                      assistant_turn_end_ids=result.assistant_turn_end_ids,
                      assistant_header_ids_thinking_on=result.assistant_header_ids_thinking_on)


def _check_registry_root(out_dir: Path) -> None:
    """Refuse-before-mutation: the target must be absent, empty, or a
    registry root this shim owns (CURRENT/builds/, plus crashed publish
    temps)."""
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise RegistryShimError(
            f"registry target {out_dir} exists and is not a directory")
    entries = {p.name for p in out_dir.iterdir()}
    if not entries:
        return
    if REGISTRY_INDEX in entries:
        raise RegistryShimError(
            f"registry target {out_dir} holds a FLAT registry "
            f"({REGISTRY_INDEX} at its root — the pre-pointer layout, or "
            "an export_worker_registry dir). The shim builds immutable "
            f"versioned dirs under {REGISTRY_BUILDS}/ behind an atomically "
            f"published {REGISTRY_POINTER} pointer, and mixing the two "
            "layouts in one directory would make resolution ambiguous — "
            "point the build at a fresh root, or remove the flat build "
            "deliberately first")
    allowed = {REGISTRY_POINTER, REGISTRY_BUILDS}
    stray = sorted(n for n in entries if n not in allowed
                   and not n.startswith(f".{REGISTRY_POINTER}.tmp-"))
    if stray:
        raise RegistryShimError(
            f"registry target {out_dir} carries {stray} — not a registry "
            "root this shim owns, so building into it could destroy files "
            "that are not its to touch; refusing before any mutation")


def _fsync_path(path: Any) -> None:
    """fsync one file or directory by path (directories carry their
    entries' names — an unsynced dir can lose a renamed-in child on
    power loss even when the child's own data is synced)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    """fsync every file and every directory under ``root``, children
    before parents (bottom-up), then ``root`` itself — after this, the
    build's bytes AND its directory structure are durable, so the
    rename/publish that follows can never point at a build the storage
    has not actually kept."""
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            _fsync_path(os.path.join(dirpath, name))
        _fsync_path(dirpath)


def _publish_pointer(root: Path, build_id: str) -> None:
    """Atomically publish ``build_id`` as the root's CURRENT build:
    write a temp, fsync it, then ``os.replace`` over the pointer, then
    fsync the root dir (durability of the replace itself) — readers see
    the old content or the new, never a torn file, and a crash before
    the replace leaves the old pointer untouched."""
    tmp = root / f".{REGISTRY_POINTER}.tmp-{os.getpid()}-{time.time_ns()}"
    tmp.write_text(build_id + "\n")
    _fsync_path(tmp)
    os.replace(tmp, root / REGISTRY_POINTER)
    _fsync_path(root)


def resolve_registry_dir(path: Any) -> Path:
    """Resolve what a consumer was handed into the immutable directory
    ``WorkerObjectRegistry.load`` takes — ONCE, at consumption time, so
    every worker of one server loads the same build even if a rebuild
    publishes mid-launch. Accepts either:

    - an already-resolved build dir / a flat exported registry
      (``registry.json`` directly inside), returned as-is; or
    - a registry ROOT (``CURRENT`` + ``builds/``), following the
      pointer.

    Anything else — including a root whose pointer names a build that
    does not exist — refuses loudly. That refusal is also the
    storage-crash fallback: a torn/orphaned pointer is never "repaired"
    by guessing among builds/ (two orphans would be ambiguous); the
    recovery is :func:`build_lens_registry` into the same root, which
    publishes a fresh, valid pointer over the broken one."""
    path = Path(path)
    if (path / REGISTRY_INDEX).is_file():
        return path
    pointer = path / REGISTRY_POINTER
    if pointer.is_file():
        build_id = pointer.read_text().strip()
        if (not build_id or build_id != Path(build_id).name
                or build_id.startswith(".")):
            raise RegistryShimError(
                f"registry pointer {pointer} names an invalid build id "
                f"{build_id!r} — refusing to resolve outside "
                f"{REGISTRY_BUILDS}/")
        build = path / REGISTRY_BUILDS / build_id
        if not (build / REGISTRY_INDEX).is_file():
            raise RegistryShimError(
                f"registry pointer {pointer} names build {build_id!r} but "
                f"{build / REGISTRY_INDEX} does not exist — pointer and "
                "builds disagree (a partially copied root?); refusing")
        return build
    raise RegistryShimError(
        f"{path} is neither a registry build (no {REGISTRY_INDEX}) nor a "
        f"registry root (no {REGISTRY_POINTER}) — nothing to serve")


def _build_registry_into(
    out_dir: Path,
    conditions: Sequence[Mapping[str, Any]],
    *,
    assistant_header_ids: Optional[Sequence[int]] = None,
    tokenizer: Any = None,
    assistant_turn_end_ids: Optional[Sequence[int]] = None,
    assistant_header_ids_thinking_on: Optional[Sequence[int]] = None,
) -> ShimResult:
    """Write + verify one registry build inside ``out_dir`` (a fresh
    private build dir — :func:`build_lens_registry` owns the swap)."""
    header = _resolve_header(conditions, assistant_header_ids, tokenizer)
    turn_end = _resolve_turn_end(conditions, assistant_turn_end_ids,
                                 tokenizer)
    on_header = _resolve_thinking_header(conditions, assistant_header_ids_thinking_on,
                                         tokenizer, header)
    entries: List[Dict[str, Any]] = []
    results: List[ConditionResult] = []
    seen: set = set()

    for raw in conditions:
        cond = _read_condition(raw)
        cell_id = str(cond["cell_id"])
        if cell_id in seen:
            raise RegistryShimError(f"duplicate cell_id {cell_id!r}")
        seen.add(cell_id)
        spec = dict(cond["resolved_spec"])
        kind = str(spec["kind"])
        if kind in la.UNSERVABLE_KIND_REASONS:
            raise RegistryShimError(
                f"condition {cell_id!r}: kind {kind!r} is not served on the "
                f"lens path — {la.UNSERVABLE_KIND_REASONS[kind]} This is the "
                "LENS registry shim; route this kind to its own committed "
                "path (soft tokens -> serving/soft_token.py's prompt-embeds; "
                "weight-level LoRA -> serving/lora_export.py's native "
                "multi-LoRA), where its coefficient story — including the "
                "soft-token scale-0 passthrough, which a folded zero "
                "embedding would NOT reproduce — is that path's to handle")
        c = _finite_positive_c(cond["coefficient"])

        art_path = Path(cond["artifact_path"])
        if not art_path.is_file():
            raise RegistryShimError(
                f"condition {cell_id!r}: no artifact at {art_path}")
        orig_art = art_path.read_bytes()
        got_sha = cm.sha256_hex16(orig_art)
        want_sha = str(cond["object_sha256"])[:16]
        if got_sha != want_sha:
            raise RegistryShimError(
                f"condition {cell_id!r}: artifact at {art_path} hashes to "
                f"{got_sha}, but object_sha256 says {want_sha} — the bytes "
                "on disk are not the object the condition names")

        side_path = cond.get("sidecar_path")
        orig_side = b""
        if side_path:
            side_file = Path(side_path)
            if not side_file.is_file():
                raise RegistryShimError(
                    f"condition {cell_id!r}: no sidecar at {side_file}")
            orig_side = side_file.read_bytes()
            got_side_sha = cm.sha256_hex16(orig_side)
            want_side_sha = str(cond["sidecar_sha256"])[:16]
            if got_side_sha != want_side_sha:
                raise RegistryShimError(
                    f"condition {cell_id!r}: sidecar at {side_file} hashes "
                    f"to {got_side_sha}, but sidecar_sha256 says "
                    f"{want_side_sha} — the sidecar bytes on disk are not "
                    "the ones the condition names; refusing BEFORE "
                    "adapting or folding them")

        deriv_art, deriv_side = _fold_coefficient(kind, orig_art, orig_side, c)
        if c != 1.0:
            _verify_fold(kind, spec, orig_art, orig_side, deriv_art,
                         deriv_side, c, header, turn_end_ids=turn_end,
                         thinking_header_ids=on_header)
        if cond.get("expect_served_coeff") is not None:
            _assert_served_coeff(
                kind, spec, deriv_art, deriv_side, header,
                _finite_c(cond["expect_served_coeff"], "expect_served_coeff"),
                turn_end_ids=turn_end)

        ih = cm.petri_intervention_hash(
            spec, artifact_bytes=deriv_art, sidecar_bytes=deriv_side)
        art_rel = _write_blob(out_dir, f"{cell_id}/artifact.bin", deriv_art)
        side_rel = (_write_blob(out_dir, f"{cell_id}/sidecar.bin", deriv_side)
                    if deriv_side else None)

        entries.append({
            "cell_id": cell_id, "kind": kind, "resolved_spec": spec,
            "expect_hash": ih, "artifact": art_rel, "sidecar": side_rel,
            "artifact_sha16": cm.sha256_hex16(deriv_art),
            "artifact_intervention_hash": ih,
            "forced_scale": None, "donor_cell": None,
            "validated_by": "registry_shim._verify_fold",
        })
        results.append(ConditionResult(
            cell_id=cell_id, kind=kind,
            selector={"petri_object": cell_id, "expect_hash": ih},
            original_sha256=got_sha,
            derived_sha256=cm.sha256_hex16(deriv_art + deriv_side),
            effective_coefficient=c, coefficient_was_folded=(c != 1.0)))

    index = {"version": 1,
             "assistant_header_ids": list(header) if header else None,
             "assistant_turn_end_ids": (list(turn_end) if turn_end
                                        else None),
             ls.REGISTRY_THINKING_HEADER_KEY: (list(on_header) if on_header
                                               else None),
             "entries": entries}
    index_path = out_dir / REGISTRY_INDEX
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    os.chmod(index_path, stat.S_IRUSR | stat.S_IRGRP)

    # Re-load through the committed path: re-verifies every hash, the
    # selector binding, the adapter translation, and the header — so a
    # bad entry fails here, not first inside a GPU worker.
    ls.WorkerObjectRegistry.load(out_dir)

    return ShimResult(registry_dir=str(out_dir),
                      conditions=tuple(results),
                      assistant_header_ids=header,
                      assistant_turn_end_ids=turn_end,
                      assistant_header_ids_thinking_on=on_header)


def _served_positions_of(spec: Any) -> Any:
    """The position MODE a condition's spec asks to serve — for a STACKED
    condition the mode lives on the OBJECT leg's spec
    (``resolved_spec["object"]["positions"]``; the wrapper carries none of
    its own), exactly as ``lens_steer._assert_ops_express_spec_positions``
    unwraps it. The two marker gates below decide from THIS, never from the
    top-level key alone: the 2026-09-07 sixth review's P2 was a stacked
    all-turns condition read as response-only, given no scaffold at build,
    and refused at reload.

    ONE implementation: ``defense_cells.served_positions_of``, beside the
    wrapper it unwraps — the proxy's routing-table stamp
    (``proxy.lens_served_positions``) reads the SAME function, so the
    header a build must carry and the header-locatedness the thinking-ON
    gate decides from can never disagree about a stacked cell."""
    return dc.served_positions_of(spec)


def _resolve_header(conditions: Sequence[Mapping[str, Any]],
                    assistant_header_ids: Optional[Sequence[int]],
                    tokenizer: Any) -> Optional[Tuple[int, ...]]:
    wants_response = any(
        _served_positions_of(c.get("resolved_spec")) in la.RESPONSE_MASK_MODES
        for c in conditions)
    if assistant_header_ids is not None:
        return tuple(int(i) for i in assistant_header_ids)
    if tokenizer is not None:
        # the GENERATION-PROMPT header (the fit-time width's id-form) —
        # never the terminator-anchored located header, which absorbs
        # prompt-side scaffold on gemma-4-style templates and would
        # steer outside the fit support (chat_markers docstring).
        from sisteer.chat_markers import generation_header_ids
        return tuple(generation_header_ids(tokenizer))
    if wants_response:
        raise RegistryShimError(
            "a condition serves a header-located mask (positions in "
            f"{list(la.RESPONSE_MASK_MODES)}) but no assistant_header_ids "
            "and no tokenizer were given — the mask is located by matching "
            "the header's token ids, so it cannot be served without them")
    return None


def _resolve_turn_end(conditions: Sequence[Mapping[str, Any]],
                      assistant_turn_end_ids: Optional[Sequence[int]],
                      tokenizer: Any) -> Optional[Tuple[int, ...]]:
    """The assistant turn-end scaffold to record — the all-turns mask's
    second marker (lens_adapters, THE ALL-TURNS MASK): the given ids; else,
    ONLY when a condition serves ``positions="response_all_turns"``, the
    served template's ``chat_markers.assistant_turn_end_ids`` (the
    terminator plus the glue before the next turn's opener, LOCATED in a
    multi-turn probe render) with the role-header shape certified; else
    None — a response-only build records no scaffold metadata. An
    all-turns condition with none resolvable refuses HERE, at build time,
    never per request."""
    wants_all_turns = any(
        _served_positions_of(c.get("resolved_spec"))
        == la.POSITIONS_RESPONSE_ALL_TURNS
        for c in conditions)
    if assistant_turn_end_ids is not None:
        ids = tuple(int(i) for i in assistant_turn_end_ids)
        if not ids:
            raise RegistryShimError(
                "assistant_turn_end_ids is empty — a turn with no "
                "locatable end has no span; pass None for a registry that "
                "serves no response_all_turns condition")
        return ids
    if not wants_all_turns:
        # a response-only build records NO scaffold metadata: it is unused
        # by every op it serves, and older builds that derived it anyway
        # tripped the (stricter) all-turns startup certification on
        # templates the response mode serves fine (the fifth review's
        # Nemotron regression) — the worker now gates that check on loaded
        # ops, and the shim stops emitting the metadata
        return None
    if tokenizer is not None:
        from sisteer.chat_markers import (ChatMarkerError,
                                          assistant_turn_end_ids as _turn_end,
                                          turn_opener_id as _opener)
        scaffold = _turn_end(tokenizer)
        if scaffold is None:
            raise RegistryShimError(
                "a condition serves positions='response_all_turns' "
                "(every assistant turn's span) but the served chat "
                "template is SUCCESSOR-BOUNDED — assistant turns end "
                "bare, with no terminator token (GLM-4.x) — so an "
                "earlier turn's end cannot be located; this template "
                "cannot serve the all-turns mask")
        # the historical fail-closed rule validates a successor role
        # header by the generation header's shape; the template's user
        # header must have that shape (successor_role_header_ids)
        try:
            _opener(tokenizer)
        except ChatMarkerError as e:
            raise RegistryShimError(
                "a condition serves positions='response_all_turns' but "
                f"the served template fails the turn-opener check: {e}; "
                "this template cannot serve the all-turns mask") from e
        return tuple(scaffold)
    if wants_all_turns:
        raise RegistryShimError(
            "a condition serves positions='response_all_turns' (every "
            "assistant turn's span) but no assistant_turn_end_ids and no "
            "tokenizer were given — an earlier turn's span ends at its "
            "turn-end scaffold, located by matching its token ids, so the "
            "mask cannot be served without them")
    return None


def _resolve_thinking_header(conditions: Sequence[Mapping[str, Any]],
                             assistant_header_ids_thinking_on: Optional[Sequence[int]],
                             tokenizer: Any,
                             header: Optional[Tuple[int, ...]]
                             ) -> Optional[Tuple[int, ...]]:
    """The DECLARED thinking-ON header to record: the given ids; else,
    ONLY when a condition's spec declares the
    thinking-ON render, the served template's
    ``chat_markers.generation_header_ids_under(tok, enable_thinking=True)``;
    else None — a registry with no declaring condition records no second
    header (and refuses one handed to it: a header nothing serves is
    drift). A declaring condition with none resolvable refuses HERE, at
    build; a template with no thinking switch refuses too. An ON header
    EQUAL to the OFF header is the header-invariant template's case
    (gemma-4) and is recorded as the second header (the rule 2026-09-10)."""
    def _declares(c):
        spec = c.get("resolved_spec")
        try:
            return isinstance(spec, Mapping) and la.declares_thinking(spec)
        except la.AdapterError:
            return False  # _read_condition refuses the malformed spec itself
    wants = any(_declares(c) for c in conditions)
    if assistant_header_ids_thinking_on is not None:
        ids = tuple(int(i) for i in assistant_header_ids_thinking_on)
        if not ids:
            raise RegistryShimError(
                "assistant_header_ids_thinking_on is empty — a collapsed "
                "thinking-ON header disables the declared render's steered "
                "region; pass None for a registry with no declaring condition")
        if not wants:
            raise RegistryShimError(
                "assistant_header_ids_thinking_on was given but no condition "
                "declares the thinking-ON render (serve_out_of_support: "
                "['thinking']) — a header nothing serves is refused, never "
                "recorded quietly")
        return ids
    if not wants:
        return None
    if tokenizer is None:
        raise RegistryShimError(
            "a condition declares the thinking-ON render (serve_out_of_support: "
            "['thinking']) but no assistant_header_ids_thinking_on and no "
            "tokenizer were given — the declared render's boundary is located "
            "by matching the ON header's token ids, so the condition cannot be "
            "served without them")
    from sisteer.chat_markers import ChatMarkerError, generation_header_ids_under
    try:
        ids = tuple(generation_header_ids_under(tokenizer, enable_thinking=True))
    except ChatMarkerError as e:
        raise RegistryShimError(
            "a condition declares the thinking-ON render but the served template "
            f"has no thinking-ON render to declare against: {e}") from e
    return ids


def _write_blob(out_dir: Path, rel: str, data: bytes) -> str:
    p = out_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP)
    return rel
