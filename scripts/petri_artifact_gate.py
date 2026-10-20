#!/usr/bin/env python
"""The per-kind exact-equality serving gate against REAL trained
artifacts (first run against real weights).

WHY THIS EXISTS. The lens MECHANISM is GPU-verified (the vllm-lens x
vllm 0.25.1 probe, job 165167), and the per-kind adapters are proven
equivalent to the authoritative object classes on SYNTHETIC artifacts
(tests/test_petri_lens_adapters.py). Neither has ever touched a real
TRAINED object. This gate closes that gap: it points the same
exact-equality bar at the published safety-defense artifacts
(psr_kl_all / psr_mse_all / vector / lora_r1) on the real pinned
target model.

WHAT IT ASSERTS, per artifact kind:

- **psr (psr_kl_all, psr_mse_all)** — the full battery:
  1. *Adapter-vs-object equivalence on a real forward.* The artifact is
     hydrated through the authoritative ``sisteer.methods.load_object``;
     the SAME bytes are adapted via ``lens_adapters.get_adapter("psr")``;
     a real forward (prefill + one KV-cached decode step, the frozen
     gate prompts rendered under the PINNED chat template) runs under
     the object's own ``apply(model, scale=1.0)`` with every decoder
     layer's hidden states captured immediately before and after the
     object's hooks; and ``apply_hook_ops`` must map each pre-tensor to
     the post-tensor BIT-EXACTLY — the synthetic per-kind test's bar
     (test_petri_lens_adapters._assert_equivalent), with real weights.
     DEAD-CLOSED GATES ARE EXPECTED, NOT FAILURES: a layer whose ReLU
     gate outputs exactly zero on every token contributes exactly zero,
     and the adapter must reproduce that exact zero (bit-identity).
     Non-vacuity is the tolerant form — at least ONE layer changed
     somewhere (``any``, never ``all``); the per-layer live/dead map is
     REPORTED as evidence (verified: nothing on the adapt/apply path
     requires a per-layer non-zero — validation is finiteness/shape
     only, so no loosening was needed).
  2. *Token-level exact equality on the fp32 reference oracle.* Greedy
     generation over the frozen gate prompt set through the SAME
     HF-engine code path (``openai_shim.reference_generate``): the
     adapter's ops (installed as forward hooks with real
     ``HookContext``s) must reproduce the object's own ``apply()``
     generation token-exactly, and the scale-0 leg (armed hooks, zero
     add) must reproduce the undefended base EXACTLY (H5's no-op bar).
  3. *Scale-0 bit-identity* on the captured hidden states
     (``apply_hook_ops(..., 0.0) is hidden`` — Phase-0 assertion 8's
     contract, on real ops).

- **vector** — the artifact on disk is a SINGLE-LAYER SteeringVector
  safetensors container (kind "vector", method "distill_soft", layer
  15), NOT a MultiLayerVectorObject "vector_all" torch blob, so the
  vector_all adapter equivalence is NOT RUNNABLE on these bytes by
  design. What this gate asserts: ``load_object`` hydrates it
  (VectorObject — the authoritative loader accepts the artifact), the
  vector_all adapter REFUSES the foreign safetensors bytes loudly, and
  the REGISTERED "vector" adapter (lens_adapters._adapt_vector — the
  kind is served on the lens path now) translates these exact bytes
  into ops that reproduce ``VectorObject.apply(model, scale)``
  BIT-EXACTLY on a real forward over the frozen gate prompts, at every
  scale in VECTOR_EQUIVALENCE_SCALES plus the scale-0 identity — the
  psr battery's equivalence bar, pinning layer set, positions/BOS
  behavior, sign, and the vector itself. The skipped vector_all
  equivalence is recorded as ``not_runnable`` with the reason — never a
  silent pass or a fake failure.

- **lora_r1** — must REFUSE on the lens path with the precise
  documented reason (weight-level o_proj edit; the committed route is
  the genuine PEFT export + vLLM native multi-LoRA), while
  ``load_object`` hydrates it fine — proving the refusal is about the
  PATH, not artifact corruption. No lora lens adapter exists or is
  added here (default-deny).

SERVED-ON-PROXY-vLLM STATUS. The committed proxy-vLLM serving path's
live wiring is DEFERRED at this commit
(``petri_serving_gate._bootstrap_live_stack`` raises NotImplementedError;
the worker HookContext wiring does not exist yet), so the
served-vs-fp32 comparison ON THAT PATH cannot run and is recorded
``unavailable`` with this reason — per artifact and top-level, never
silently skipped and never conflated with a failure. The token-level
checks above run on the one serving-relevant path that IS available
(the audit's own fp32 reference oracle with the adapter's ops armed) and
are labelled exactly that. NO no-op certificates are written here: the
certificate store belongs to the committed-backend gate
(scripts/petri_serving_gate.py), and this gate must never publish a
certificate the proxy path did not earn.

OUTCOME DISCIPLINE (the same pattern as the probe/gate):

- BOOTSTRAP failure (imports, manifest, model/tokenizer load, missing
  artifact files) -> ``artifact_gate_unavailable.json`` + exit 0 — an
  escalated launch blocker, never a silent skip;
- VALIDATION failure (pinned chat-template hash mismatch — the
  meta-llama/NousResearch guard — an equivalence mismatch, a token
  inequality, a refusal that did not happen or happened for the wrong
  reason, an artifact of the wrong kind/sha) ->
  ``artifact_gate_failed.json`` + exit NONZERO;
- success -> ``artifact_gate_record.json``.
  ``artifact_gate_evidence.json`` is always written alongside.

Run on the cluster via ``scripts/sbatch/petri_artifact_gate.sbatch``
from the immutable commit-addressed checkout. The pure logic below
(equivalence runner, live/dead map, refusal checks, token-leg
comparison, marker discipline) is CPU-tested with a tiny model and
fakes (tests/test_petri_artifact_gate.py); only the model load and the
real-artifact I/O need the GPU node.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # script-mode import path
    sys.path.insert(0, str(REPO_ROOT / "src"))


def load_gate_module():
    """The frozen gate prompt set + comparator live in
    scripts/petri_serving_gate.py — import it by path (scripts/ is not a
    package) so the prompt set and its hash can never fork."""
    existing = sys.modules.get("_petri_serving_gate")
    if existing is not None:  # the test suite already loaded it
        return existing
    spec = importlib.util.spec_from_file_location(
        "_petri_serving_gate",
        Path(__file__).resolve().with_name("petri_serving_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    # registered BEFORE exec: dataclass processing resolves the module
    # through sys.modules, and an unregistered module crashes it
    sys.modules["_petri_serving_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- outcome markers (the probe's fail-loud/record discipline) -----------------

UNAVAILABLE_MARKER = "artifact_gate_unavailable.json"
FAILURE_MARKER = "artifact_gate_failed.json"
RECORD_MARKER = "artifact_gate_record.json"
EVIDENCE_FILE = "artifact_gate_evidence.json"

#: why the committed proxy-vLLM served comparison cannot run at this
#: commit (recorded verbatim, per artifact and top-level).
PROXY_VLLM_UNAVAILABLE_REASON = (
    "the committed proxy-vLLM serving path's live wiring is deferred at "
    "this commit (plan §11 step 3: petri_serving_gate._bootstrap_live_stack "
    "raises NotImplementedError; the worker-side HookContext wiring from "
    "vllm-lens forward metadata is not landed) — the served-vs-fp32 "
    "comparison on that path is UNAVAILABLE, not failed; the token-level "
    "checks here run on the fp32 reference oracle with the adapter's ops "
    "armed (hf_shim), which is the serving-relevant path available today")


def _write_marker(out_dir: Path, name: str,
                  payload: Mapping[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / name
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                 default=str))
    return marker


def record_unavailable(out_dir: Path, reason: str, **extra: Any) -> Path:
    """Bootstrap failure: fail loud, record, exit clean (0)."""
    return _write_marker(out_dir, UNAVAILABLE_MARKER, {
        "status": "artifact gate unavailable", "reason": reason,
        "recorded_at": time.time(), **extra})


def record_failure(out_dir: Path, reason: str, **extra: Any) -> Path:
    """Validation failure: assertions ran and failed — recorded AND exit
    nonzero, never downgraded to 'unavailable'."""
    return _write_marker(out_dir, FAILURE_MARKER, {
        "status": "artifact gate FAILED", "reason": reason,
        "recorded_at": time.time(), **extra})


def clear_stale_markers(out_dir: Path) -> None:
    """FIRST thing a run does: delete every outcome marker a previous
    run left in ``out_dir``. Each run writes exactly ONE of the three,
    so without this a re-run into the same directory accumulates
    contradictory verdicts — a stale ``artifact_gate_record.json`` next
    to a fresh failure reads as a success to anything keying on the
    success marker's existence, which is precisely how a failed gate
    gets treated as a passed one."""
    out_dir = Path(out_dir)
    for name in (UNAVAILABLE_MARKER, FAILURE_MARKER, RECORD_MARKER):
        stale = out_dir / name
        if stale.exists():
            stale.unlink()


# --- pre/post capture on a real forward (the synthetic test's harness, ---------
# --- generalized to a real tokenized prompt) ------------------------------------


def capture_pre_post(model, apply_ctx, input_ids, attention_mask):
    """Prefill + ONE KV-cached greedy decode step; capture every decoder
    layer's output hidden states immediately BEFORE (pre) and AFTER
    (post) the context's hooks (forward hooks fire in registration
    order). Returns ``(pre, post, prompt_len)`` with
    ``pre[layer] == [prefill, decode]`` tensors. The decode token is the
    prefill's greedy argmax under the SAME context — a real continuation,
    not a random id."""
    import torch

    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)
    pre: Dict[int, List[Any]] = {l: [] for l in range(len(decoder))}
    post: Dict[int, List[Any]] = {l: [] for l in range(len(decoder))}

    def mk(store, l):
        def hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[l].append(h.detach().clone())
        return hook

    prompt_len = int(input_ids.shape[1])
    pre_handles = [decoder[l].register_forward_hook(mk(pre, l))
                   for l in range(len(decoder))]
    try:
        with apply_ctx:
            post_handles = [decoder[l].register_forward_hook(mk(post, l))
                            for l in range(len(decoder))]
            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=True)
                    next_id = out.logits[:, -1, :].argmax(dim=-1,
                                                          keepdim=True)
                    next_mask = torch.ones(
                        (attention_mask.shape[0], prompt_len + 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device)
                    model(input_ids=next_id, attention_mask=next_mask,
                          past_key_values=out.past_key_values,
                          use_cache=True)
            finally:
                for h in post_handles:
                    h.remove()
    finally:
        for h in pre_handles:
            h.remove()
    return pre, post, prompt_len


@dataclass(frozen=True)
class EquivalenceOutcome:
    """One artifact's adapter-vs-object equivalence over the prompt set.
    ``passed`` is bit-exact equality at every layer/phase/prompt plus the
    tolerant non-vacuity bar (``any`` live layer — dead-closed layers are
    evidence, never failures) plus scale-0 bit-identity."""

    passed: bool
    prompts: int
    layers_checked: Tuple[int, ...]
    op_layers: Tuple[int, ...]
    live_layers: Tuple[int, ...]
    dead_layers: Tuple[int, ...]  # op layers, exactly-zero on every token
    scale0_bit_identical: bool
    mismatches: Tuple[str, ...]
    per_layer: Mapping[str, Mapping[str, Any]]

    def as_json(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["note"] = (
            "dead = the op contributed exactly zero at every captured "
            "token of every gate prompt (prefill + one decode step); the "
            "adapter must and did reproduce that exact zero bit-exactly. "
            "Non-vacuity is any(live), never all(live).")
        return d


def run_equivalence_check(model, obj_apply_ctx_fn, ops, encoded_prompts,
                          *, scale: float = 1.0,
                          gate_stats: bool = True) -> EquivalenceOutcome:
    """The synthetic per-kind equivalence test, on real inputs: for every
    encoded prompt, capture pre/post under the object's own apply() at
    ``scale`` and assert ``apply_hook_ops`` at the SAME scale reproduces
    the object's transformation BIT-EXACTLY at every decoder layer,
    prefill AND decode. Running it at more than one scale is how a
    scale-handling divergence between the object's deployment and the
    adapter's (which coincide at any single scale only by luck) is
    caught.

    ``obj_apply_ctx_fn(scale)`` returns a fresh apply() context (the
    fresh-ctx-per-call discipline); ``ops`` must already be materialized
    to the model's device/dtype. Failures are COLLECTED (never an early
    exit) so the record names every mismatch."""
    import torch

    from sisteer.capeval.serving import lens_adapters as la

    op_layers = sorted({op.layer for op in ops})
    ops_by_layer = {l: [op for op in ops if op.layer == l]
                    for l in op_layers}
    mismatches: List[str] = []
    live: Dict[int, bool] = {l: False for l in op_layers}
    stats: Dict[int, Dict[str, float]] = {
        l: {"max_abs_delta": 0.0, "gate_rows_seen": 0.0,
            "gate_rows_active": 0.0} for l in op_layers}
    n_layers_checked: Optional[int] = None
    scale0_ok = True
    scale = float(scale)

    for p_idx, (input_ids, attention_mask) in enumerate(encoded_prompts):
        pre, post, prompt_len = capture_pre_post(
            model, obj_apply_ctx_fn(scale), input_ids, attention_mask)
        n_layers_checked = len(pre)
        contexts = (
            (0, la.HookContext(la.PHASE_PREFILL, tuple(range(prompt_len)))),
            (1, la.HookContext(la.PHASE_DECODE, (prompt_len,))),
        )
        for l in range(len(pre)):
            for phase_idx, ctx in contexts:
                pre_t, post_t = pre[l][phase_idx], post[l][phase_idx]
                mine = la.apply_hook_ops(pre_t, ops, scale, layer=l, ctx=ctx)
                if not torch.equal(mine, post_t):
                    delta = (mine.float() - post_t.float()).abs().max()
                    mismatches.append(
                        f"prompt {p_idx} layer {l} phase {ctx.phase} scale "
                        f"{scale}: adapter ops do not reproduce the "
                        f"object's own transformation (max "
                        f"|adapter-object| = {float(delta):.3e})")
                if l not in ops_by_layer:
                    # no op here: the object must not have touched it,
                    # and the adapter must return the SAME tensor.
                    if not torch.equal(pre_t, post_t):
                        mismatches.append(
                            f"prompt {p_idx} layer {l} phase {ctx.phase}: "
                            "the object modified a layer the adapter has "
                            "no op for")
                    if mine is not pre_t:
                        mismatches.append(
                            f"prompt {p_idx} layer {l} phase {ctx.phase}: "
                            "apply_hook_ops did not return the identical "
                            "tensor at an op-less layer")
                    continue
                # scale-0 bit-identity on the real captured tensor
                if la.apply_hook_ops(pre_t, ops, 0.0, layer=l,
                                     ctx=ctx) is not pre_t:
                    scale0_ok = False
                    mismatches.append(
                        f"prompt {p_idx} layer {l} phase {ctx.phase}: "
                        "scale-0 apply_hook_ops is not bit-identical "
                        "(Phase-0 assertion 8's contract)")
                if not torch.equal(pre_t, post_t):
                    live[l] = True
                st = stats[l]
                st["max_abs_delta"] = max(
                    st["max_abs_delta"],
                    float((post_t.float() - pre_t.float())
                          .abs().max().item()))
                if gate_stats:
                    for op in ops_by_layer[l]:
                        if op.op != la.OP_RELU_GATED_VECTOR:
                            continue
                        import torch.nn.functional as F
                        g = F.relu(pre_t.float() @ op.gate_w + op.gate_b)
                        st["gate_rows_seen"] += float(g.numel())
                        st["gate_rows_active"] += float(
                            (g > 0).sum().item())
        del pre, post

    live_layers = tuple(l for l in op_layers if live[l])
    dead_layers = tuple(l for l in op_layers if not live[l])
    if not live_layers:
        mismatches.append(
            "VACUOUS: no layer changed on any gate prompt — the tolerant "
            "non-vacuity bar is any(layer changed); an all-dead object "
            "proves nothing about the adapter (and is a red flag for a "
            "trained artifact)")
    per_layer: Dict[str, Dict[str, Any]] = {}
    for l in op_layers:
        st = stats[l]
        entry: Dict[str, Any] = {
            "live": live[l],
            "max_abs_delta": st["max_abs_delta"],
        }
        if st["gate_rows_seen"]:
            entry["gate_active_fraction"] = (
                st["gate_rows_active"] / st["gate_rows_seen"])
        per_layer[str(l)] = entry
    return EquivalenceOutcome(
        passed=not mismatches,
        prompts=len(encoded_prompts),
        layers_checked=tuple(range(n_layers_checked or 0)),
        op_layers=tuple(op_layers),
        live_layers=live_layers,
        dead_layers=dead_layers,
        scale0_bit_identical=scale0_ok,
        mismatches=tuple(mismatches),
        per_layer=per_layer)


# --- the adapter-ops arm handle (HF forward hooks with real HookContexts) ------


@contextmanager
def adapter_ops_ctx(model, ops, scale: float):
    """Install the adapter's ops as forward hooks on the decoder layers,
    building a real :class:`HookContext` per forward: the first forward a
    layer sees inside this ctx is the prefill (positions ``0..T-1``);
    every later forward must be a single-row KV-cached decode step at the
    tracked global position. Anything else (chunked prefill, batch > 1)
    refuses loudly — the same shapes the serving gate keeps impossible.

    Returning the SAME tensor leaves the module output untouched
    (bit-identical no-op for scale 0 / dead layers)."""
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)
    op_layers = sorted({op.layer for op in ops})
    unknown = [l for l in op_layers if l >= len(decoder)]
    if unknown:
        raise la.AdapterError(
            f"adapter ops name decoder layer(s) {unknown} but the model "
            f"has {len(decoder)} layers")
    next_pos: Dict[int, int] = {}
    handles = []

    def mk(l):
        def hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() != 3 or h.shape[0] != 1:
                raise la.AdapterError(
                    f"adapter-ops hook at layer {l}: expected a batch-1 "
                    f"[1, T, d] hidden, got {tuple(h.shape)} — this gate "
                    "generates one prompt at a time, by construction")
            T = int(h.shape[-2])
            if l not in next_pos:
                ctx = la.HookContext(la.PHASE_PREFILL, tuple(range(T)))
                next_pos[l] = T
            else:
                if T != 1:
                    raise la.AdapterError(
                        f"adapter-ops hook at layer {l}: multi-row forward "
                        f"(T={T}) after the prefill — a chunked prefill / "
                        "prefix reuse is unrepresentable by design")
                ctx = la.HookContext(la.PHASE_DECODE, (next_pos[l],))
                next_pos[l] += 1
            new_h = la.apply_hook_ops(h, ops, scale, layer=l, ctx=ctx)
            if new_h is h:
                return None  # untouched — leave the module output alone
            if isinstance(output, tuple):
                return (new_h,) + tuple(output[1:])
            return new_h
        return hook

    try:
        for l in op_layers:
            handles.append(decoder[l].register_forward_hook(mk(l)))
        yield
    finally:
        for hd in handles:
            hd.remove()


class AdapterOpsArmHandle:
    """Duck-compatible with what ``HFEngine.generate(arm=...)`` needs
    (``ctx()`` + ``transform_prompt``), so the adapter-served leg runs
    through the SAME reference_generate code path as the object leg —
    fresh ctx per call, position state reset each entry."""

    def __init__(self, *, model: Any, ops: Sequence[Any], scale: float,
                 name: str):
        self._model = model
        self._ops = tuple(ops)
        self.coeff = float(scale)
        self.name = name
        self.kind = "lens_adapter_ops"
        self.ctx_entries = 0

    def ctx(self):
        self.ctx_entries += 1
        return adapter_ops_ctx(self._model, self._ops, self.coeff)

    def transform_prompt(self, s: str) -> str:
        return s


# --- token-level exact equality over the frozen gate prompt set ----------------


def token_equality_report(gate_mod, legs: Mapping[str, Callable[..., Any]]
                          ) -> Dict[str, Any]:
    """Greedy token ids for every leg over the FROZEN gate prompt set,
    then the two H5 exact-equality comparisons:

    - ``adapter`` vs ``object`` — the adapter's ops must reproduce the
      object's own generation token-exactly;
    - ``adapter_scale0`` vs ``base`` — the armed-hook zero leg must
      reproduce the undefended base EXACTLY.

    ``legs`` maps leg name -> ``fn(messages) -> token id list``. Pass
    criteria are exact equality and nothing else (numeric fields are
    diagnostics — H5)."""
    required = {"base", "object", "adapter", "adapter_scale0"}
    missing = sorted(required - set(legs))
    if missing:
        raise ValueError(f"token_equality_report needs legs {sorted(required)}"
                         f" — missing {missing}")
    tokens: Dict[str, List[List[int]]] = {name: [] for name in legs}
    for messages in gate_mod.GATE_PROMPT_SET:
        for name, fn in legs.items():
            tokens[name].append([int(t) for t in fn(list(messages))])

    def compare(a: str, b: str) -> List[Dict[str, Any]]:
        out = []
        for i in range(len(gate_mod.GATE_PROMPT_SET)):
            res = gate_mod.exact_token_equality(
                tokens[a][i], tokens[b][i], prompt_index=i)
            out.append(dataclasses.asdict(res))
        return out

    adapter_vs_object = compare("adapter", "object")
    scale0_vs_base = compare("adapter_scale0", "base")
    steered_differs = sum(
        1 for i in range(len(gate_mod.GATE_PROMPT_SET))
        if tokens["object"][i] != tokens["base"][i])
    return {
        "passed": (all(c["passed"] for c in adapter_vs_object)
                   and all(c["passed"] for c in scale0_vs_base)),
        "adapter_vs_object": adapter_vs_object,
        "adapter_scale0_vs_base": scale0_vs_base,
        "prompts": len(gate_mod.GATE_PROMPT_SET),
        "max_tokens": gate_mod.GATE_MAX_TOKENS,
        # evidence only, never a pass criterion: on how many gate prompts
        # the steered object changed the base's greedy output at all.
        "object_differs_from_base_on_n_prompts": steered_differs,
    }


# --- default-deny refusal checks (real bytes, real registry) -------------------


def check_lens_kind_refusal(kind: str, must_mention: Sequence[str]
                            ) -> Dict[str, Any]:
    """``get_adapter(kind)`` must refuse, and the refusal must carry the
    documented reason (every ``must_mention`` substring)."""
    from sisteer.capeval.serving import lens_adapters as la

    try:
        la.get_adapter(kind)
    except la.AdapterError as e:
        msg = str(e)
        missing = [s for s in must_mention if s not in msg]
        return {"passed": not missing, "refused": True, "message": msg,
                "missing_expected_phrases": missing}
    return {"passed": False, "refused": False,
            "message": f"kind {kind!r} was UNEXPECTEDLY served an adapter "
                       "— default-deny did not hold"}


def check_adapter_refuses_bytes(kind: str, resolved_spec: Mapping[str, Any],
                                artifact_bytes: bytes,
                                sidecar_bytes: bytes = b"") -> Dict[str, Any]:
    """A registered adapter must REFUSE these bytes (wrong payload for
    the kind) — loudly, via AdapterError, never a guessed translation."""
    from sisteer.capeval.serving import lens_adapters as la

    try:
        fn = la.get_adapter(kind)
    except la.AdapterError as e:
        return {"passed": False, "refused": False,
                "message": f"no adapter for kind {kind!r} to test: {e}"}
    try:
        ops = fn(dict(resolved_spec), artifact_bytes, sidecar_bytes)
    except la.AdapterError as e:
        return {"passed": True, "refused": True, "message": str(e)}
    return {"passed": False, "refused": False,
            "message": f"adapter for kind {kind!r} ACCEPTED foreign bytes "
                       f"and produced {len(ops)} op(s) — it must refuse"}


# --- artifact table -------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactSpec:
    """One ``--artifact name:expected_kind:sha_prefix`` argument."""

    name: str
    expected_kind: str  # psr | steering_vector | lora
    sha_prefix: str

    @classmethod
    def parse(cls, raw: str) -> "ArtifactSpec":
        parts = raw.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"--artifact {raw!r} is not '<name>:<expected_kind>:"
                "<sha256-prefix>'")
        name, kind, prefix = parts
        if kind not in ("psr", "steering_vector", "lora"):
            raise ValueError(f"--artifact {raw!r}: unknown expected_kind "
                             f"{kind!r}")
        return cls(name=name, expected_kind=kind, sha_prefix=prefix.lower())


def artifact_paths(root: Path, name: str) -> Tuple[Path, Optional[Path]]:
    """The artifact file (+ sidecar when present) inside one artifact
    dir: object.pt or object.safetensors — exactly one must exist."""
    d = Path(root) / name
    pt, st = d / "object.pt", d / "object.safetensors"
    present = [p for p in (pt, st) if p.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"{d}: expected exactly one of object.pt / object.safetensors, "
            f"found {[p.name for p in present]}")
    sidecar = d / "object.json"
    return present[0], (sidecar if sidecar.is_file() else None)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the GPU driver -------------------------------------------------------------


def _encode_gate_prompts(shim, gate_mod):
    """The frozen gate prompts, rendered under the PINNED chat template
    and tokenized exactly as the reference engine does (no double BOS) —
    returned as (input_ids, attention_mask) pairs on the model device."""
    import torch

    from sisteer.generate import encode_kwargs

    tok = shim.engine.tokenizer
    device = next(shim.engine.model.parameters()).device
    encoded = []
    for messages in gate_mod.GATE_PROMPT_SET:
        rendered = shim._render(list(messages))
        ids = tok(rendered, **encode_kwargs(tok, rendered))["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids)
        encoded.append((input_ids, mask))
    return encoded


def _check_psr_artifact(shim, gate_mod, *, name: str, path: Path,
                        artifact_bytes: bytes) -> Dict[str, Any]:
    """The full battery for one real psr artifact."""
    import torch

    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.extraction.psr import PSRObject
    from sisteer.methods import load_object

    model = shim.engine.model
    result: Dict[str, Any] = {"kind": "psr"}

    # 1. authoritative load
    obj = load_object(path, model, shim.engine.tokenizer)
    if not isinstance(obj, PSRObject):
        result["passed"] = False
        result["error"] = (f"load_object hydrated {type(obj).__name__}, "
                           "expected PSRObject")
        return result
    result["loaded_class"] = type(obj).__name__
    result["object_meta"] = {"layers": list(obj.layers),
                             "train_direction": bool(obj.train_direction)}

    # 2. adapt the SAME bytes; materialize once to the model device/dtype
    ops = la.get_adapter("psr")(
        {"kind": "psr", "layers": list(obj.layers)}, artifact_bytes, b"")
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    ops = la.materialize_ops(ops, device=device, dtype=dtype)
    result["adapter"] = {"n_ops": len(ops),
                         "ops": sorted({op.op for op in ops}),
                         "op_layers": sorted({op.layer for op in ops})}

    # 3. adapter-vs-object equivalence on a real forward, all gate prompts
    encoded = _encode_gate_prompts(shim, gate_mod)
    equiv = run_equivalence_check(
        model, lambda scale: obj.apply(model, scale=scale), ops, encoded)
    result["adapter_equivalence"] = equiv.as_json()

    # 4. token-level exact equality on the fp32 reference oracle
    object_arm = shim.arm_for_object(str(path), scale=1.0,
                                     name=f"{name}__object")
    adapter_arm = AdapterOpsArmHandle(model=model, ops=ops, scale=1.0,
                                      name=f"{name}__adapter_ops")
    adapter0_arm = AdapterOpsArmHandle(model=model, ops=ops, scale=0.0,
                                       name=f"{name}__adapter_ops0")
    max_tokens = gate_mod.GATE_MAX_TOKENS
    legs = {
        "base": lambda m: shim.reference_generate(m, max_tokens=max_tokens),
        "object": lambda m: shim.reference_generate(
            m, max_tokens=max_tokens, arm=object_arm),
        "adapter": lambda m: shim.reference_generate(
            m, max_tokens=max_tokens, arm=adapter_arm),
        "adapter_scale0": lambda m: shim.reference_generate(
            m, max_tokens=max_tokens, arm=adapter0_arm),
    }
    with torch.no_grad():
        result["hf_fp32_token_equality"] = token_equality_report(gate_mod,
                                                                 legs)

    # 5. the committed served path: recorded unavailable, never silent
    result["served_proxy_vllm"] = {"status": "unavailable",
                                   "reason": PROXY_VLLM_UNAVAILABLE_REASON}

    result["passed"] = bool(equiv.passed
                            and result["hf_fp32_token_equality"]["passed"])
    return result


#: the scales the vector equivalence runs at. Two NON-ZERO scales on
#: purpose: at any single scale a multiply-vs-replace (or sign / double-
#: scaling) divergence between VectorObject's deployment and the
#: adapter's can coincide by luck; two scales cannot both coincide. The
#: scale-0 identity is asserted inside every run_equivalence_check pass.
VECTOR_EQUIVALENCE_SCALES = (1.0, 0.5)


def _check_vector_artifact(shim, gate_mod, *, path: Path,
                           artifact_bytes: bytes,
                           sidecar_bytes: bytes) -> Dict[str, Any]:
    """The real 'vector' artifact is a single-layer SteeringVector
    container, NOT a vector_all blob. Asserted here:

    - the authoritative ``load_object`` hydrates it (VectorObject);
    - the REGISTERED 'vector' adapter (the kind is served on the lens
      path now — the old absent-by-default expectation is stale)
      translates these exact bytes;
    - SEMANTIC EQUIVALENCE against the authoritative object: the
      adapter's ops reproduce ``VectorObject.apply(model, scale)``
      BIT-EXACTLY at every decoder layer, prefill and decode, over the
      frozen gate prompts, at EVERY scale in
      :data:`VECTOR_EQUIVALENCE_SCALES` — which pins layer set,
      positions/BOS behavior, sign, and the vector itself, not merely
      that ops exist. Scale-0 bit-identity is asserted in the same
      runner (Phase-0 assertion 8's contract);
    - the vector_all adapter still default-denies the foreign
      safetensors bytes; its own equivalence stays recorded
      ``not_runnable`` with the reason."""
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.methods import VectorObject, load_object

    result: Dict[str, Any] = {"kind": "steering_vector"}
    obj = load_object(path, shim.engine.model, shim.engine.tokenizer)
    result["loaded_class"] = type(obj).__name__
    result["loaded_kind"] = getattr(obj, "kind", None)
    load_ok = isinstance(obj, VectorObject)
    if not load_ok:
        result["error"] = (f"load_object hydrated {type(obj).__name__}, "
                           "expected VectorObject")
    sidecar = json.loads(sidecar_bytes.decode("utf-8")) if sidecar_bytes \
        else {}
    result["sidecar"] = {k: sidecar.get(k) for k in
                         ("method", "model_name", "layers", "metadata")}

    # default-deny still holds where it should: the vector_all adapter
    # refuses these foreign safetensors bytes (they are not its blob).
    result["vector_all_adapter_refusal"] = check_adapter_refuses_bytes(
        "vector_all", {"kind": "vector_all"}, artifact_bytes, b"")
    # ... and kind 'vector' is now REGISTERED (lens_adapters
    # register_adapter("vector", _adapt_vector)): the gate's expectation
    # moved with it. The adapter must translate THESE real bytes — a
    # refusal here would mean the served arm silently lost its adapter.
    try:
        ops = la.get_adapter("vector")(
            {"kind": "vector"}, artifact_bytes, sidecar_bytes)
        result["kind_vector_adapter"] = {
            "passed": len(ops) >= 1, "registered": True,
            "n_ops": len(ops),
            "ops": sorted({op.op for op in ops}),
            "op_layers": sorted({op.layer for op in ops}),
        }
    except la.AdapterError as e:
        result["kind_vector_adapter"] = {
            "passed": False, "registered": True, "refused": True,
            "message": str(e)}
        ops = None

    # adapter-vs-object SEMANTIC equivalence on a real forward — the
    # same bar the psr battery holds, per scale.
    equivalence_ok = False
    if load_ok and ops:
        model = shim.engine.model
        mat = la.materialize_ops(ops,
                                 device=next(model.parameters()).device,
                                 dtype=next(model.parameters()).dtype)
        encoded = _encode_gate_prompts(shim, gate_mod)
        per_scale: Dict[str, Any] = {}
        for scale in VECTOR_EQUIVALENCE_SCALES:
            outcome = run_equivalence_check(
                model, lambda s: obj.apply(model, scale=s), mat, encoded,
                scale=scale)
            per_scale[str(scale)] = outcome.as_json()
        equivalence_ok = all(o["passed"] for o in per_scale.values())
        result["vector_equivalence"] = {
            "passed": equivalence_ok,
            "scales": list(VECTOR_EQUIVALENCE_SCALES),
            "per_scale": per_scale,
        }
    else:
        result["vector_equivalence"] = {
            "passed": False,
            "error": ("not run — the authoritative load or the adapter "
                      "translation failed above, and an equivalence over "
                      "nothing certifies nothing")}

    result["vector_all_equivalence"] = {
        "status": "not_runnable",
        "reason": (
            "the artifact on disk is a SINGLE-LAYER SteeringVector "
            "safetensors container (kind 'vector', method "
            f"{sidecar.get('method')!r}, layers {sidecar.get('layers')!r}) "
            "— not a MultiLayerVectorObject 'vector_all' torch blob, so "
            "the vector_all adapter takes different bytes by design and "
            "the requested equivalence cannot run against THIS artifact; "
            "the adapter's refusal of these bytes is asserted instead "
            "(default-deny on real bytes). The registered 'vector' "
            "adapter's semantic equivalence against the authoritative "
            "VectorObject is checked above (vector_equivalence)"),
    }
    result["passed"] = bool(load_ok
                            and result["vector_all_adapter_refusal"]["passed"]
                            and result["kind_vector_adapter"]["passed"]
                            and equivalence_ok)
    return result


def _check_lora_artifact(shim, *, path: Path) -> Dict[str, Any]:
    """lora_r1 SERVES on the lens path (the o_proj module site,
    HOOK_SITE_OPROJ — landed 2026-08-28): the committed adapter must
    translate the REAL bytes into ops carrying exactly the hydrated
    object's own A/B tensors, while the authoritative loader still
    hydrates the artifact fine. The refusal-with-export-pointer this
    check used to assert is retired WITH the refusal; the PEFT export
    (serving/lora_export.py) remains the committed path for
    all-position fits."""
    import torch as _torch

    from sisteer.capeval.serving import lens_adapters as _la
    from sisteer.extraction.objects import RankOneLoRAObject
    from sisteer.methods import load_object

    result: Dict[str, Any] = {"kind": "lora"}
    obj = load_object(path, shim.engine.model, shim.engine.tokenizer)
    result["loaded_class"] = type(obj).__name__
    load_ok = isinstance(obj, RankOneLoRAObject)
    if not load_ok:
        result["error"] = (f"load_object hydrated {type(obj).__name__}, "
                           "expected RankOneLoRAObject")
    served: Dict[str, Any] = {"passed": False}
    try:
        blob = _torch.load(path, map_location="cpu", weights_only=True)
        ops = _la.get_adapter("lora")(
            {"kind": "lora"}, path.read_bytes(), b"")
        (op,) = ops
        checks = {
            "one_op": True,
            "op_kind": op.op == _la.OP_LORA_OPROJ,
            "site": _la.op_site(op) == _la.HOOK_SITE_OPROJ,
            "layer": op.layer == int(blob["layer"]),
            # the op's read basis is A ([1, d_in] — the k=1 transpose)
            # and its write basis B ([d_out, 1]) — EXACTLY the
            # authoritative object's own tensors, no rescale, no swap
            # (on square o_proj shapes alone a silent transpose would
            # pass every shape check)
            "read_basis_is_A": _torch.equal(
                op.lowrank_a, blob["A"].float().reshape(1, -1)),
            "write_basis_is_B": _torch.equal(
                op.lowrank_b, blob["B"].float().reshape(-1, 1)),
            "coeff_is_unit": op.coeff == 1.0,
        }
        served = {"passed": all(checks.values()), "checks": checks}
    except Exception as e:  # loud in the record, never a crash
        served = {"passed": False,
                  "error": f"{type(e).__name__}: {e}"}
    result["lens_served"] = served
    result["served_route"] = (
        "lens o_proj module site (HOOK_SITE_OPROJ, lens_adapters "
        "OP_LORA_OPROJ) — the mid-block edit is exact at the module and "
        "it is the only committed path that can serve a "
        "positions='response' fit; the genuine PEFT export (two-SI D13, "
        "serving/lora_export.py) remains committed for all-position fits")
    result["passed"] = bool(load_ok and served["passed"])
    return result


class _Bootstrap(RuntimeError):
    """Environment cannot come up — recorded unavailable, exit 0."""


def cmd_gate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # a run's first act: no marker from a PREVIOUS run may survive into
    # this one — exactly one of the three is written per run, and a
    # stale success beside a fresh failure would read as a pass.
    clear_stale_markers(out_dir)
    evidence: Dict[str, Any] = {"stage": "bootstrap",
                                "started_at": time.time()}

    def dump_evidence() -> None:
        (out_dir / EVIDENCE_FILE).write_text(
            json.dumps(evidence, indent=2, sort_keys=True, default=str))

    # ---- bootstrap: anything failing here is UNAVAILABLE (exit 0) ----
    try:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            raise _Bootstrap(f"torch/transformers not importable: {e}")
        from sisteer.capeval import manifest as cm
        from sisteer.capeval.serving.openai_shim import OpenAIShimEngine

        gate_mod = load_gate_module()
        spec = cm.load_petri_spec(Path(args.manifest))
        artifacts = [ArtifactSpec.parse(a) for a in args.artifact]
        root = Path(args.artifacts_root)
        loaded_files: Dict[str, Tuple[Path, bytes, Optional[Path], bytes]] = {}
        for a in artifacts:
            try:
                path, sidecar_path = artifact_paths(root, a.name)
            except FileNotFoundError as e:
                raise _Bootstrap(str(e))
            blob = path.read_bytes()
            side = sidecar_path.read_bytes() if sidecar_path else b""
            loaded_files[a.name] = (path, blob, sidecar_path, side)
        import importlib.metadata as md
        evidence["packages"] = {
            p: md.version(p) for p in ("torch", "transformers")}
        evidence["stage"] = "model_load"
        dump_evidence()
        shim = OpenAIShimEngine(
            spec.target_model, revision=spec.target_revision,
            identity=cm.serving_identity("hf_shim", "apply",
                                         spec.serving_mode))
        try:
            shim.load()  # fp32, load-bearing (the reference oracle)
        except Exception as e:
            raise _Bootstrap(
                f"the fp32 reference model failed to load: "
                f"{type(e).__name__}: {e}")
    except _Bootstrap as e:
        evidence["bootstrap_error"] = str(e)
        dump_evidence()
        marker = record_unavailable(out_dir, str(e))
        print(f"artifact gate unavailable — recorded at {marker}: {e}",
              file=sys.stderr)
        return 0

    # ---- from here on, any failure is a VALIDATION failure (exit 1) ----
    import torch

    device = next(shim.engine.model.parameters()).device
    dtype = next(shim.engine.model.parameters()).dtype
    self_bytes = Path(__file__).resolve().read_bytes()
    provenance: Dict[str, Any] = {
        "checkout_sha": args.checkout_sha,
        "script_sha256": sha256_hex(self_bytes),
        "sbatch_sha256": (sha256_hex(Path(args.sbatch_path).read_bytes())
                          if args.sbatch_path else None),
        "model": spec.target_model,
        "revision": spec.target_revision,
        "serving_mode": spec.serving_mode,
        "reference_oracle": ("hf_shim+apply fp32 greedy, batch=1 "
                             "(openai_shim.reference_generate)"),
        "gate_prompt_set_hash": gate_mod.gate_prompt_set_hash(),
        "max_tokens": gate_mod.GATE_MAX_TOKENS,
        "device": (torch.cuda.get_device_name(device)
                   if device.type == "cuda" else str(device)),
        "model_dtype": str(dtype),
        "artifacts_root": str(args.artifacts_root),
        "packages": evidence["packages"],
    }
    evidence["provenance"] = provenance
    evidence["stage"] = "validation"
    failures: List[str] = []

    # the pinned-template guard (holds the meta-llama pin; refuses the
    # NousResearch mirror's different default-system-block template)
    from sisteer.capeval import manifest as cm
    tok = shim.engine.tokenizer
    template = getattr(tok, "chat_template", None) or ""
    computed_hash = cm.sha256_hex16(template.encode("utf-8"))
    provenance["chat_template_hash"] = {
        "pinned": spec.chat_template_hash, "computed": computed_hash,
        "match": computed_hash == spec.chat_template_hash,
        "tokenizer_name_or_path": getattr(tok, "name_or_path", None),
    }
    if computed_hash != spec.chat_template_hash:
        failures.append(
            f"chat_template_hash mismatch: pinned "
            f"{spec.chat_template_hash!r} != computed {computed_hash!r} — "
            "the loaded tokenizer is NOT the pinned meta-llama template "
            "(a NousResearch-mirror substitution injects a different "
            "default system block); refusing to gate against the wrong "
            "render")

    results: Dict[str, Any] = {}
    if not failures:
        for a in artifacts:
            path, blob, sidecar_path, side = loaded_files[a.name]
            full_sha = sha256_hex(blob)
            entry_provenance = {
                "path": str(path),
                "artifact_sha256": full_sha,
                "sidecar_path": str(sidecar_path) if sidecar_path else None,
                "sidecar_sha256": sha256_hex(side) if side else None,
                "expected_sha_prefix": a.sha_prefix,
            }
            if not full_sha.startswith(a.sha_prefix):
                results[a.name] = {
                    **entry_provenance, "passed": False,
                    "error": (f"artifact sha256 {full_sha[:16]} does not "
                              f"start with the expected {a.sha_prefix!r} — "
                              "wrong bytes at the offered path")}
                failures.append(f"{a.name}: artifact sha mismatch")
                continue
            try:
                if a.expected_kind == "psr":
                    res = _check_psr_artifact(
                        shim, gate_mod, name=a.name, path=path,
                        artifact_bytes=blob)
                elif a.expected_kind == "steering_vector":
                    res = _check_vector_artifact(
                        shim, gate_mod, path=path, artifact_bytes=blob,
                        sidecar_bytes=side)
                else:
                    res = _check_lora_artifact(shim, path=path)
            except Exception as e:
                import traceback
                res = {"passed": False,
                       "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()}
            results[a.name] = {**entry_provenance, **res}
            if not results[a.name].get("passed"):
                failures.append(
                    f"{a.name}: {results[a.name].get('error') or 'assertions failed'}")
            evidence["artifacts"] = results
            dump_evidence()  # durable after every artifact

    evidence["artifacts"] = results
    evidence["failures"] = failures
    evidence["finished_at"] = time.time()
    dump_evidence()

    summary = {
        "provenance": provenance,
        "proxy_vllm_serving": {"status": "unavailable",
                               "reason": PROXY_VLLM_UNAVAILABLE_REASON},
        "artifacts": {
            name: {
                "passed": bool(r.get("passed")),
                "artifact_sha256": r.get("artifact_sha256"),
                "live_layers": (r.get("adapter_equivalence") or {}
                                ).get("live_layers"),
                "dead_layers": (r.get("adapter_equivalence") or {}
                                ).get("dead_layers"),
            } for name, r in results.items()},
        "recorded_at": time.time(),
    }
    if failures:
        marker = record_failure(out_dir, "; ".join(failures), **summary)
        print(f"artifact gate FAILED — recorded at {marker}:\n  "
              + "\n  ".join(failures), file=sys.stderr)
        return 1
    marker = _write_marker(out_dir, RECORD_MARKER,
                           {"status": "completed", **summary})
    print(f"artifact gate completed — record at {marker}")
    for name, r in results.items():
        eq = r.get("adapter_equivalence") or {}
        print(f"  {name}: PASS"
              + (f" (live layers {eq.get('live_layers')}, dead "
                 f"{eq.get('dead_layers')})" if eq else ""))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate",
                       help="the real-artifact exact-equality gate")
    g.add_argument("--manifest", required=True)
    g.add_argument("--artifacts-root", required=True,
                   help="dir holding one subdir per artifact")
    g.add_argument("--artifact", action="append", required=True,
                   help="<name>:<expected_kind>:<sha256-prefix>; kinds: "
                        "psr | steering_vector | lora")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--checkout-sha", default=None,
                   help="the immutable checkout's commit (provenance)")
    g.add_argument("--sbatch-path", default=None,
                   help="the submitting sbatch script (hashed into "
                        "provenance — it rides along untracked)")
    g.set_defaults(fn=cmd_gate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
