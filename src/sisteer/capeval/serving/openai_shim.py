"""The fp32/greedy reference oracle.

An OpenAI-compatible shim over the existing
``serving/hf_engine.py::HFEngine``, with EXACTLY one role — the §10
serving-correctness gate's reference:

- :meth:`OpenAIShimEngine.reference_generate` produces the exact greedy
  output-token ids the gate compares the proxy-served output against
  (exact token equality — the pass criterion), for the
  undefended base and, via :meth:`arm_for_object` (``methods.load_object``
  + the object's own ``apply(model, scale)`` under the per-call ctx
  discipline the capability harness already trusts), for each sampled
  defense object in the per-kind canary.

It is NEVER an audit serving path (B1): a kind the
proxy-vLLM primary cannot serve is a HARD §10 gate failure that blocks
the audit — there is no recorded-exception HF fallback and no hf-served
cell. ``manifest.validate_petri_spec`` refuses
``serving_backend != proxy_vllm`` structurally, and enumerate_cells
derives every audit cell's identity from that backend, so no audit
traffic can route here.

fp32 is LOAD-BEARING: the reference role is defined as the fp32 greedy
HF output, so :meth:`load` always loads in ``torch.float32`` — there is
no dtype knob to quietly weaken the reference.

v1 limitation, refused loudly (never degraded): ``tools`` /
``tool_choice`` requests — the reference role needs only the plain chat
surface; a tool-call parser would be a recorded decision if the gate
ever needs armed references over tool transcripts.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Mapping, Optional

from sisteer.capeval import manifest as cm


class ShimError(RuntimeError):
    """An OpenAI-shim refusal — loud, never a silent default."""


#: OpenAI request keys the shim understands; anything else refuses.
_ALLOWED_REQUEST_KEYS = frozenset({
    "model", "messages", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "stop", "stream", "n", "seed", "user",
    "tools", "tool_choice",
})


class _ObjectArmHandle:
    """A minimal arm handle over a ``methods.load_object`` object: the
    fresh-ctx-per-call ``apply(model, scale)`` discipline, duck-
    compatible with what ``HFEngine.generate(arm=…)`` needs."""

    def __init__(self, *, obj: Any, model: Any, scale: float, name: str):
        self._obj = obj
        self._model = model
        self.coeff = float(scale)
        self.name = name
        self.kind = getattr(obj, "kind", type(obj).__name__)
        self.ctx_entries = 0

    def ctx(self):
        self.ctx_entries += 1
        return self._obj.apply(self._model, scale=self.coeff)

    def transform_prompt(self, s: str) -> str:
        return self._obj.inject(s) if hasattr(self._obj, "inject") else s


class OpenAIShimEngine:
    """The hf_shim engine behind ``runtime.make_engine`` (identity
    ``hf_shim+apply+<mode>``)."""

    def __init__(self, model_id: str, *, revision: Optional[str],
                 identity: str, hf_engine: Optional[Any] = None):
        ident = cm.parse_serving_identity(identity)
        if ident.backend != "hf_shim":
            raise ShimError(f"OpenAIShimEngine serves hf_shim identities "
                            f"only, got {identity!r}")
        if not model_id:
            raise ShimError("OpenAIShimEngine needs a model id")
        self.model_id = model_id
        self.revision = revision
        self.identity = identity
        self.name = identity  # rows stamp engine = the serving identity
        #: injectable (tests / preloaded) — but never a dtype bypass:
        #: ``load()`` VALIDATES a preloaded engine against the fp32
        #: reference contract before accepting it (fp32 is load-bearing,
        #: module docstring).
        self._engine = hf_engine

    @classmethod
    def from_manifest(cls, manifest: Any) -> "OpenAIShimEngine":
        model = (getattr(manifest, "base", None)
                 or getattr(manifest, "target_model", None))
        revision = (getattr(manifest, "base_revision", None)
                    or getattr(manifest, "target_revision", None))
        if not model:
            raise ShimError(
                f"engine {manifest.engine!r}: the manifest names no model "
                "(base/target_model) — the HF shim never guesses which "
                "weights to load")
        return cls(model, revision=revision, identity=manifest.engine)

    def load(self) -> None:
        """One fp32 model load per process (the reference role's contract
        — fp32 is not configurable here). A PRELOADED engine never
        bypasses that contract: it is validated fp32 before acceptance."""
        if self._engine is None:
            import torch

            from sisteer.capeval.serving.hf_engine import HFEngine

            self._engine = HFEngine(self.model_id, revision=self.revision,
                                    dtype=torch.float32)
            self._engine.load()
            return
        self._validate_preloaded_fp32(self._engine)

    @staticmethod
    def _validate_preloaded_fp32(engine: Any) -> None:
        """Refuse a preloaded engine that is not the fp32 reference: any
        declared inference dtype and every floating-point model
        parameter/buffer must be ``torch.float32`` — a bf16 engine handed
        in via the injection seam would silently compute every exact
        served-vs-reference comparison against the wrong reference.
        (Tensor-free test fakes pass vacuously: they carry nothing that
        could violate fp32.)"""
        import torch

        for attr in ("dtype", "torch_dtype", "_dtype"):
            declared = getattr(engine, attr, None)
            if declared is not None and declared != torch.float32:
                raise ShimError(
                    f"preloaded engine declares inference dtype {declared} "
                    f"(via {attr!r}) — the reference role is DEFINED as the "
                    "fp32 greedy HF output (module docstring); load() builds "
                    "its own fp32 engine rather than accepting a weaker one")
        model = getattr(engine, "model", None)
        if model is None:
            return
        for kind, it in (("parameter", getattr(model, "parameters", None)),
                         ("buffer", getattr(model, "buffers", None))):
            if not callable(it):
                continue
            for t in it():
                if (isinstance(t, torch.Tensor) and t.is_floating_point()
                        and t.dtype != torch.float32):
                    raise ShimError(
                        f"preloaded engine is not the fp32 reference: a "
                        f"model {kind} is {t.dtype} — exact served "
                        "comparisons and no-op certificates must never be "
                        "computed against a non-fp32 reference (H5)")

    @property
    def engine(self) -> Any:
        if self._engine is None:
            raise ShimError("engine.load() has not run — no HF engine")
        return self._engine

    # -- the armed reference (per-kind canary) object path --------------------

    def arm_for_object(self, object_path: str, *,
                       scale: float, name: str) -> _ObjectArmHandle:
        """Hydrate a defense object via ``methods.load_object`` (all
        object kinds — the one loader the harness trusts) and bind it at
        ``scale`` under the per-call apply/ctx discipline."""
        from sisteer.methods import load_object

        obj = load_object(object_path, self.engine.model,
                          self.engine.tokenizer)
        return _ObjectArmHandle(obj=obj, model=self.engine.model,
                                scale=scale, name=name)

    # -- OpenAI translation ----------------------------------------------------

    def _render(self, messages: Any) -> str:
        if not isinstance(messages, list) or not messages:
            raise ShimError("chat request needs a non-empty messages list")
        clean: List[Dict[str, str]] = []
        for m in messages:
            if (not isinstance(m, Mapping) or not isinstance(
                    m.get("role"), str)
                    or not isinstance(m.get("content"), str)):
                raise ShimError(
                    f"the HF shim renders plain {{role, content:str}} "
                    f"messages only, got {m!r} — tool/parts messages are a "
                    "refused v1 limitation (module docstring)")
            clean.append({"role": m["role"], "content": m["content"]})
        tok = self.engine.tokenizer
        # THE REFERENCE MUST RENDER THE FIT'S REGIME: the tokenizer
        # (models.load_model_and_tokenizer via HFEngine) carries the
        # thinking-off pin for reasoning templates, and dropping it here
        # would compare a thinking-ON reference against a thinking-OFF
        # serve — different prompts, so exact-equality canaries and any
        # HF-reference leg would diverge for reasons that have nothing
        # to do with serving fidelity (Qwen3.5 onboarding, 2026-08-30).
        # Same pattern as chat_markers.render_ids.
        return tok.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=True,
            **getattr(tok, "_sisteer_template_kwargs", {}))

    def _gen_cfg(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        max_tokens = request.get("max_tokens",
                                 request.get("max_completion_tokens"))
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ShimError(
                "the HF shim requires an explicit positive integer "
                "max_tokens — it never guesses a generation budget "
                f"(got {max_tokens!r})")
        temperature = request.get("temperature", 0)
        cfg: Dict[str, Any] = {"max_new_tokens": max_tokens, "chat": False}
        if temperature in (0, 0.0, None):
            cfg["greedy"] = True
        else:
            cfg["greedy"] = False
            cfg["temperature"] = float(temperature)
            if "top_p" in request:
                cfg["top_p"] = float(request["top_p"])
        stop = request.get("stop")
        if stop is not None:
            cfg["until"] = ([stop] if isinstance(stop, str) else list(stop))
        return cfg

    def chat_completions(self, request: Mapping[str, Any], *,
                         arm: Optional[Any] = None) -> Dict[str, Any]:
        """OpenAI ``/v1/chat/completions`` over ``HFEngine.generate``
        under ``arm.ctx()`` (the per-call apply discipline). ``arm=None``
        = the undefended base / a passthrough control repeat."""
        unknown = sorted(set(request) - _ALLOWED_REQUEST_KEYS)
        if unknown:
            raise ShimError(f"unknown chat-completions key(s) {unknown} "
                            f"(the shim understands "
                            f"{sorted(_ALLOWED_REQUEST_KEYS)})")
        if request.get("stream"):
            raise ShimError("stream=true is not supported by the HF shim")
        if request.get("tools") or request.get("tool_choice"):
            raise ShimError(
                "the HF shim does not serve tool calls (v1 limitation, "
                "module docstring) — the reference role needs only the "
                "plain chat surface; extending it is a recorded decision, "
                "never a silent degradation")
        if request.get("n", 1) != 1:
            raise ShimError("the HF shim serves n=1 only")
        prompt = self._render(request.get("messages"))
        if arm is not None:
            prompt = arm.transform_prompt(prompt)
        cfg = self._gen_cfg(request)
        out = self.engine.generate([prompt], arm=arm, gen_cfg=cfg)[0]

        tok = self.engine.tokenizer
        prompt_tokens = len(tok(prompt)["input_ids"])
        completion_tokens = len(out.gen_token_ids)
        return {
            "id": f"petri-hfshim-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", self.model_id),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out.text},
                "finish_reason": ("stop" if out.ended_with_eos else "length"),
            }],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
            "petri": {
                "serving_identity": self.identity,
                "engine": "hf_shim",
                "arm": getattr(arm, "name", None),
            },
        }

    # -- the gate's reference role ----------------------------------------------

    def reference_generate(self, messages: Any, *, max_tokens: int,
                           arm: Optional[Any] = None) -> List[int]:
        """The §10 gate's fp32/greedy reference: the RAW generated token
        ids for a fixed chat input (exact-equality comparisons happen on
        token ids, never on decoded text)."""
        prompt = self._render(messages)
        if arm is not None:
            prompt = arm.transform_prompt(prompt)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ShimError(f"reference_generate needs a positive "
                            f"max_tokens, got {max_tokens!r}")
        out = self.engine.generate(
            [prompt], arm=arm,
            gen_cfg={"greedy": True, "chat": False,
                     "max_new_tokens": max_tokens})[0]
        return list(out.gen_token_ids)
