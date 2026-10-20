"""ONE definition of the post-load (F9) resolved intervention spec.

Why this module exists: two matched runs (pod job
capeval-c6-h100-b, cluster job 216163) both completed and were both
REJECTED by scripts/capeval_c6_report.py with ``results.json
arm_intervention_hash != expected`` on every HF cell. The engines were
hashing the SAME artifact + sidecar bytes under two different spec
shapes: HFEngine hashed the POST-LOAD spec read off the loaded object
({kind, object_kind, layers, positions, skip_bos, norm_relative, coeff,
converter_version}) while vllm_lens hashed the PRE-LOAD static cell spec
(runtime.static_intervention_spec: {kind, layer, coeff, positions,
norm_relative, skip_bos}) — different key sets, so they could never
agree by construction. The served interventions matched (capability
scores agreed within a few items per task on both platforms); only the
provenance bookkeeping diverged.

HF's shape is canonical — it is the reference engine and its handle hash
is what historical results.json carry — so this module makes BOTH
engines derive it from one implementation:

* :func:`resolved_spec_from_object` — moved verbatim from
  hf_engine's ``_resolved_spec``.
* :func:`resolve_and_hash` — HFEngine.arm's whole §6.1 post-load
  sequence (kind validation → sidecar/F9 layer validation → manifest
  overrides → resolved spec → hash), moved verbatim, over an
  already-hydrated object. HFEngine runs it on the object it loaded to
  serve; the model-free path below runs it on the SAME construction
  (methods.load_object_cpu shares methods.load_object's hydration code,
  minus the device move). Because every step is literally shared code, a
  change to load_object's defaults moves both engines at once instead of
  reintroducing the drift one level down.
* :func:`post_load_intervention_hash` — the model-free derivation for
  the vllm_lens engine (which holds no HF model) and for the
  verifier: hydrate via load_object_cpu, then :func:`resolve_and_hash`.

Residual, by construction: TORCH-BLOB artifacts (kind ``sisteer_object``
— psr, alphasteer, loreft, alphakl, …) cannot hydrate without the loaded
model (their constructors read hidden size / decoder layers), so the
vllm_lens handle hash for those arms stays the static cell hash
(runtime.cell_intervention_hash) and does NOT equal HF's post-load hash
for the same cell. The verifier pins kind "steering" (safetensors
vectors), which this module fully unifies; a matched blob-arm comparison
needs a model-side derivation first. :func:`hashes_post_load` names that
boundary with the same content sniff load_object dispatches on, so
callers branch on the artifact's actual container format, never a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from sisteer.capeval import manifest as cm
from sisteer.capeval.manifest import AblationSpec
from sisteer.capeval.serving import (GENERATED_POSITIONS,
                                     RESPONSE_POSITIONS)


def object_layers(obj: Any) -> Optional[List[int]]:
    """Layer indices a loaded object can report about itself, or None."""
    from sisteer.methods import VectorObject  # heavy module; cycle-free

    if isinstance(obj, VectorObject):
        return list(obj.sv.layers)
    if getattr(obj, "layers", None):
        try:
            return [int(x) for x in obj.layers]
        except (TypeError, ValueError):
            return None
    if getattr(obj, "layer", None) is not None:
        return [int(obj.layer)]
    return None


def resolved_spec_from_object(spec: AblationSpec, obj: Any, coeff: float,
                              steer_positions: str = "all",
                              serve_header_len: Optional[int] = None,
                              ) -> Dict[str, Any]:
    """The canonical RESOLVED intervention spec (F9): every effective
    field after load_object has applied its defaults — what actually
    runs, not what was typed. ``positions`` is the OBJECT's own position
    mode inside the ctx; ``steer_positions`` is the engine-phase policy
    (which forwards enter the ctx at all), recorded ONLY when non-default
    so every existing all-positions resolved spec — and every
    intervention_hash derived from one — stays byte-identical.

    ``serve_header_len`` (non-None only for a response-mask arm) records
    the mask explicitly: blob objects express the mask through the
    PositionMaskedFit leg (``fit_positions``) rather than a ``positions``
    attribute, so without the override a response-masked blob would
    resolve ``positions: "all"`` and hash identically to the unmasked
    serve of the same artifact — two different interventions, one
    identity. Both keys appear only under the response request, so every
    existing hash is untouched."""
    out = {
        "kind": spec.kind,
        "object_kind": getattr(obj, "kind", type(obj).__name__),
        "layers": object_layers(obj),
        "positions": getattr(obj, "positions", "all"),
        "skip_bos": bool(getattr(obj, "skip_bos", False)),
        "norm_relative": bool(getattr(obj, "norm_relative", False)),
        "coeff": float(coeff),
        "converter_version": None,  # no PEFT converters in v0
    }
    if steer_positions != "all":
        out["steer_positions"] = steer_positions
    if serve_header_len is not None:
        out["positions"] = RESPONSE_POSITIONS
        out["response_header_len"] = int(serve_header_len)
    return out


def resolve_and_hash(spec: AblationSpec, obj: Any, coeff: float,
                     err_cls: Type[Exception],
                     arm_response_mask: Optional[
                         Callable[[str, Any], int]] = None,
                     ) -> Tuple[Dict[str, Any], str]:
    """HFEngine.arm's §6.1 post-load sequence over an already-hydrated
    object: one code path, two validations, the engine-phase position
    policy, then the F9 resolved spec and intervention hash. Returns
    ``(resolved_spec, intervention_hash)``.

    ``err_cls`` keeps each caller's refusal contract (EngineError on the
    HF engine, the verifier's failure type elsewhere); sidecar-layer
    mismatches keep raising ManifestError from
    validate_layer_against_sidecar, exactly as before the move.

    ``arm_response_mask(name, obj) -> int`` is the ENGINE-HELD leg of the
    sequence: arming positions='response' needs the served tokenizer and
    the JB mask seam (HFEngine._arm_response_mask), which a model-free
    caller does not hold. A response-mask spec with no armer refuses
    loudly rather than hashing an unmasked serve under a response label.
    """
    from sisteer.methods import VectorObject  # heavy module; cycle-free

    artifact = Path(spec.artifact)
    is_vector = isinstance(obj, VectorObject)
    if spec.kind == "steering" and not is_vector:
        raise err_cls(
            f"arm {spec.name!r}: kind 'steering' requires a safetensors "
            f"SteeringVector artifact, but {artifact} hydrated to "
            f"{type(obj).__name__} — use kind 'sisteer_object'")
    if spec.kind == "sisteer_object" and is_vector:
        raise err_cls(
            f"arm {spec.name!r}: kind 'sisteer_object' requires a torch "
            f"blob, but {artifact} is a SteeringVector container — use "
            "kind 'steering'")

    # F9 layer validation against the JSON sidecar; a blob without a
    # sidecar file falls back to the object's own layer attributes.
    sidecar_file = cm.sidecar_path(artifact)
    sidecar_bytes = b""
    if sidecar_file.is_file():
        sidecar_bytes = sidecar_file.read_bytes()
        cm.validate_layer_against_sidecar(
            spec, json.loads(sidecar_bytes.decode("utf-8")))
    elif is_vector:
        raise err_cls(
            f"arm {spec.name!r}: steering artifact {artifact} has no "
            f"JSON sidecar at {sidecar_file} — cannot validate the "
            "manifest layer (F9)")
    else:
        obj_layers = object_layers(obj)
        # A DECLARED layer set (spec.layers) must EQUAL the object's layers; the
        # single-layer membership rule stays for every other arm (ONE rule,
        # manifest.f9_layer_problem; cd8927d6 hand-ported by bcb-bfcl-200-widen
        # S1-f). The hash below already carries the object's own layers.
        if obj_layers is not None and getattr(spec, "layers", None):
            problem = cm.f9_layer_problem(spec, obj_layers,
                                          observed_what="the loaded object's layer(s)")
            if problem:
                raise err_cls(problem + " (the declaration is the serving config's "
                              "`multi_layer_arms: {<cell>: [...]}`, read by "
                              "cell_controls.arm_spec_for_cell)")
        elif obj_layers is not None and spec.layer not in obj_layers:
            raise err_cls(
                f"arm {spec.name!r}: manifest layer {spec.layer} does not "
                f"match the loaded object's layer(s) {obj_layers} (F9). "
                "If the object is GENUINELY multi-layer, this is a NAMED "
                "DESIGN GAP, not a config typo: the benign arm spec "
                "assumes single-layer serving and cannot express a "
                "multi-layer object (spec.layer feeds only this check + "
                "the intervention-identity hash; the object applies at "
                "its own layers). Do not 'fix' it by picking one of the "
                "object's layers — that fabricates the identity hash. "
                "See cell_controls.arm_spec_for_cell (coordinator "
                "2026-08-25, caa_static_r). A GENUINELY multi-layer object is "
                "served by DECLARING its whole layer set — the serving config's "
                "`multi_layer_arms: {<cell>: [...]}` (cd8927d6; bcb-bfcl-200-widen "
                "S1-f) — which F9 then requires to EQUAL the object's layers.")

    # positions="generated" is an ENGINE-PHASE policy, not an object
    # position mode (2026-08-20): the arm ctx is entered for decode
    # forwards only, the prefill runs outside it, and the object keeps
    # its own default position semantics INSIDE the ctx.
    #
    # positions="response" (2026-08-28) is the OTHER decode-only
    # serving — the POSITION MASK the JB/lens lanes already use: the
    # assistant-header positions at the tail of the prompt PLUS every
    # generated position are steered, so the prefill forward that emits
    # token 1 runs STEERED. The mask is armed ON THE OBJECT through the
    # one JB seam (engine._arm_response_mask ->
    # jailbreak._set_response_positions), passed in here as
    # ``arm_response_mask``. The two decode-only modes are INCOMPATIBLE
    # mechanisms (sweep.DECODE_MECHANISMS) and an arm resolves to at
    # most one.
    steer_positions = "all"
    serve_header_len = None
    positions = spec.positions
    if positions == GENERATED_POSITIONS:
        steer_positions = GENERATED_POSITIONS
        positions = "all"
    elif positions == RESPONSE_POSITIONS:
        if arm_response_mask is None:
            raise err_cls(
                f"arm {spec.name!r}: positions='response' needs the "
                "engine-held mask seam (a tokenizer + the loaded object) "
                "to arm the response mask — this caller supplied no "
                "arm_response_mask, so the mask cannot be armed and the "
                "hash would label an unmasked serve as masked; refusing")
        serve_header_len = arm_response_mask(spec.name, obj)
        positions = "all"  # the mask lives on the object now

    # honor the manifest's positions/norm_relative on vector arms
    # (skip_bos stays the S2-exact load_object default); blob objects
    # carry their own trained internals, so a non-default request on
    # one is a refusal rather than a silent no-op.
    if is_vector:
        if serve_header_len is None:
            obj.positions = positions
        obj.norm_relative = bool(spec.norm_relative)
    elif positions != "all" or spec.norm_relative:
        raise err_cls(
            f"arm {spec.name!r}: positions/norm_relative overrides are "
            "not honored by torch-blob objects — remove them from the "
            "manifest entry")

    resolved = resolved_spec_from_object(spec, obj, coeff,
                                         steer_positions=steer_positions,
                                         serve_header_len=serve_header_len)
    ih = cm.intervention_hash(resolved,
                              artifact_bytes=artifact.read_bytes(),
                              sidecar_bytes=sidecar_bytes)
    return resolved, ih


def hashes_post_load(spec: AblationSpec) -> bool:
    """True when ``spec.artifact`` hydrates model-free — a safetensors
    container, decided by the SAME content sniff load_object dispatches
    on — so :func:`post_load_intervention_hash` is computable here.
    Torch blobs return False: the caller keeps the static cell hash and
    the module docstring's residual note applies."""
    from sisteer.methods import _is_safetensors_file  # cycle-free

    return bool(spec.artifact) and _is_safetensors_file(Path(spec.artifact))


def post_load_intervention_hash(spec: AblationSpec, coeff: float,
                                err_cls: Type[Exception] = ValueError) -> str:
    """The post-load (F9) intervention hash EXACTLY as HFEngine.arm
    derives it, computed without a loaded model: hydrate the artifact
    through methods.load_object's own construction path
    (load_object_cpu), then run the shared :func:`resolve_and_hash`.
    Refuses (via load_object_cpu) on torch-blob artifacts — check
    :func:`hashes_post_load` first."""
    from sisteer.methods import load_object_cpu  # heavy module; cycle-free

    if not spec.artifact:
        raise err_cls(f"arm {spec.name!r}: kind {spec.kind!r} requires an "
                      "artifact path")
    artifact = Path(spec.artifact)
    if not artifact.is_file():
        raise err_cls(f"arm {spec.name!r}: artifact does not exist: "
                      f"{artifact}")
    obj = load_object_cpu(artifact)
    return resolve_and_hash(spec, obj, coeff, err_cls)[1]
