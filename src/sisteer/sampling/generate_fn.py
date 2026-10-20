"""A vLLM-backed ``generate_fn`` for the none/prompted generation path.

WHY THIS EXISTS. Two generation paths feed the pool and they ran on
different engines, invisibly:

  generate-pos (K-candidate positive side)  -> vLLM (sampling.engine)
  none / prompted (per-family replay)       -> HF model.generate

The HF/vLLM acceptance gap on this workload was measured at +25.8pp
[20.9, 30.7] against a +/-5pp equivalence margin, and it is engine-level
rather than a sampler setting anyone can match. A pool whose two halves come
from different engines carries that gap as a confound. This closes the
asymmetry from the none side.

WHY IT IS SAFE HERE AND NOT EVERYWHERE. ``LLMHFSteered.generate`` dispatches
to ``_generate_fn`` when one is injected, bypassing
``generate_with_object``. For the ``none`` and ``prompted`` cells that loses
nothing: ``_obj`` is None for both (it returns the object only for COMPILED
modes), so there is no steering object, no forward hook, and nothing for
vLLM to fail to run -- the work is plain generation over rendered chat
prompts.

For a COMPILED cell it would lose EVERYTHING, silently: the object's forward
hooks never run, vLLM emits undefended text, and the row lands labelled as a
defended generation. That is a defense evaluating as an undefended baseline
while looking like data, so ``for_cell`` REFUSES rather than warning.
"""
from __future__ import annotations

from typing import Sequence


class VllmGenerateUnsupported(RuntimeError):
    """Raised when a vLLM generate_fn is requested for a cell that needs
    HF forward hooks."""


class VllmGenerateFn:
    """Callable matching ``LLMHFSteered``'s ``generate_fn`` contract.

    Engine settings come from :func:`sampling.engine.vllm_engine_kwargs`, the
    same call :class:`sampling.engine.VllmSampler` uses, so the two halves of
    a pool are produced by one engine configuration and not merely the same
    library. This used to be two hand-kept copies with a docstring asserting
    they matched; they stopped matching and nothing noticed.
    """

    def __init__(self, model: str, *, revision: str | None = None,
                 dtype: str | None = None, max_model_len: int | None = None,
                 gpu_memory_utilization: float | None = None, seed: int = 0,
                 enforce_eager: bool | None = None):
        # LAZY. The engine is built on the FIRST CALL, not here: this object
        # is constructed alongside the defense, before anyone knows whether
        # there is work to do, and a resumed shard with every goal already
        # done would otherwise claim a GPU and load weights to generate
        # nothing. It also keeps `import vllm` off the path of every caller
        # that merely builds a defense (tests included).
        from .engine import vllm_engine_kwargs

        self.seed = seed
        self._llm = None
        # Built from the SHARED definition, not restated here. This class
        # exists so both halves of the pool come from one engine
        # configuration; restating the settings is how that guarantee gets
        # quietly broken, and it already was once (0.90 vs 0.85, 2026-08-26).
        self._cfg = vllm_engine_kwargs(
            model, revision=revision, seed=seed, dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager)

    # GUARDRAIL FOR ANYONE WIDENING for_cell(). These two settings are the
    # OPPOSITE of what a hooked path needs, and they are correct here only
    # because none/prompted carry no object:
    #   enable_prefix_caching=True  -- matches sampling.engine, so both pool
    #     halves share an engine CONFIGURATION and not merely a library.
    #   enforce_eager=False         -- CUDA-graph replay is allowed.
    # The steered-vLLM backend on an earlier development branch
    # (src/sisteer/vllm_backend.py) requires enforce_eager=TRUE and
    # enable_prefix_caching=FALSE, because **CUDA-graph replay bypasses
    # Python forward hooks entirely**. So widening for_cell to a compiled
    # cell without also flipping these would not raise -- it would run, and
    # emit unsteered text. Refusing the cell is what keeps that impossible.

    @property
    def llm(self):
        if self._llm is None:
            import os

            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            from vllm import LLM

            try:
                self._llm = LLM(**self._cfg)
                from sisteer.capeval.serving import vllm_patches as _vp
                _vp.attest_engine(self._llm)      # loud, never silent
            except Exception as e:
                # FAIL THE RUN, not the goal. The caller (run_static) catches
                # per-GOAL exceptions and records them into progress.jsonl
                # with no log line -- so a construction failure here becomes
                # one "error" per goal, the engine is retried 209 times, and
                # the job exits 0 having written no responses at all. That is
                # a total failure reported as a complete run. Measured on
                # 2026-08-26 (job 206660): rc=0, goals_run=209/209, zero rows.
                # SystemExit is not caught by that handler, so it stops here.
                self._construct_failed = True
                raise SystemExit(
                    f"[vllm-generate-fn] ENGINE CONSTRUCTION FAILED, aborting "
                    f"the run rather than degrading to one error per goal: "
                    f"{type(e).__name__}: {e}\n"
                    f"  config: {self._cfg}\n"
                    f"  A ValueError here is most often "
                    f"gpu_memory_utilization too high for the free memory on "
                    f"this device -- vLLM raises it AFTER printing its engine "
                    f"config banner, which makes it look like a runtime "
                    f"failure rather than a bad argument."
                ) from e
        return self._llm

    @classmethod
    def for_cell(cls, cell_mode: str, model: str, **kw) -> "VllmGenerateFn":
        """Build one for ``cell_mode``, refusing the modes it would silently
        break."""
        if cell_mode not in ("none", "prompted"):
            raise VllmGenerateUnsupported(
                f"cell mode {cell_mode!r} carries a steering object, whose "
                f"forward hooks a vLLM engine cannot run. Injecting a vLLM "
                f"generate_fn here would emit UNDEFENDED text and label it "
                f"as a defended generation -- an undefended baseline wearing "
                f"a defense's name. Use the HF path for compiled cells.")
        return cls(model, **kw)

    def __call__(self, rendered: Sequence[str], *, do_sample: bool = False,
                 temperature: float = 1.0, sampling_seed: int | None = None,
                 max_new_tokens: int = 512) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=1,
            temperature=(float(temperature) if do_sample else 0.0),
            top_p=(0.9 if do_sample else 1.0),
            max_tokens=int(max_new_tokens),
            seed=(int(sampling_seed) if sampling_seed is not None else None),
        )
        # prompts are ALREADY chat-rendered by the caller (build_chat_prompt),
        # so this is a raw completion call -- rendering twice would double the
        # template and is the classic way this seam goes wrong.
        outs = self.llm.generate(list(rendered), params)
        return [o.outputs[0].text for o in outs]
