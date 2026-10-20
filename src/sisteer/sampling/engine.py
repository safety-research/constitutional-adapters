"""The GPU half: sample every (prompt, SI, candidate) cell on vLLM.

Two properties of this workload decide the whole design.

1. **The SI is a ~6.5k-token system prefix shared by every prompt under that
   SI.** ``si_constitution`` is 4,896 words and ``si_irrelevant`` 4,930; at
   ~1.33 tok/word that is ~6.5k tokens of prefill that would otherwise be
   recomputed for every single request. vLLM's automatic prefix caching
   (``enable_prefix_caching=True``) prefills the shared block ONCE per engine
   and reuses its KV for every subsequent request, which is why this lane
   groups requests BY SI and submits each SI's block contiguously — the cache
   is LRU, so interleaving SIs would evict the prefix between neighbours.

2. **Nothing here is steered.** The pool's ``pos``/``neg`` are PROMPTED-teacher
   completions (``si_condition`` "none"/"prompted"), so no hook, no steering
   vector, and therefore no reason to pay HF-transformers generation speed.
   The engine-matched rule (capability deltas pair rows of one engine only)
   does not bind: no row produced here is
   ever compared against a steered row generated elsewhere — these rows are
   TRAINING pairs, not eval deltas.

Data parallelism is one independent engine per GPU in its own process, each
taking a contiguous shard of the prompt list. Llama-3.1-8B in bf16 is ~16GB,
so tensor parallelism would only add collectives to a model that already fits;
DP scales the throughput of an embarrassingly parallel workload near-linearly.
"""

from __future__ import annotations

import re

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

#: The undefended condition — no system instruction at all.
NONE_SI = "none"


@dataclass(frozen=True)
class GenSpec:
    """One SI's sampling plan: the system text and how many candidates to draw.

    ``k`` > 1 is rejection sampling (the irrelevant SI barely refuses on its
    own, so its ``pos`` is drawn at raised K until it reaches thwart parity
    with the safety SIs). ``k`` > 1 forces sampling on — K greedy candidates
    would be K identical strings.

    ``sample`` overrides that k-inferred greedy decision when set. It exists for
    generate-pos's cost-staged rounds: a K>1 scheme may draw candidate 0 first
    (``--candidates 1``), grade it, then generate the delta — but candidate 0
    must be a DRAW from the same distribution as candidates 1..K-1, not the
    greedy mode, or this SI's accepted pos is systematically the modal
    completion while its peers' are sampled. ``sample=True`` forces a real draw
    even at k==1; ``None`` (the default) keeps the ``k == 1`` inference that the
    anti-neg / sample-all lanes rely on.

    ``top_p`` is the nucleus-sampling cutoff for the SAMPLED passes (ignored
    when greedy). Default 1.0 preserves the historical anti-neg / sample-all
    behaviour; the pos teacher passes the store's teacher value (0.9) so its
    rejection-sampled completions come from the SAME distribution as the HF
    store/run rows they are pooled with. ``top_k`` (default -1 = disabled)
    likewise matches the HF-inherited value (50) on the pos path."""

    si_id: str
    system_text: str | None
    k: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    sample: bool | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"{self.si_id}: k must be >= 1, got {self.k}")


def _greedy(spec: GenSpec) -> bool:
    """Greedy (deterministic argmax) iff this is a genuine single draw: ``k ==
    1`` with no explicit override. ``spec.sample`` forces the decision either
    way — see ``GenSpec.sample``."""
    return spec.k == 1 if spec.sample is None else (not spec.sample)


def candidate_seed(base_seed: int, spec: GenSpec,
                   candidate_idx: int) -> "int | None":
    """The per-candidate sampling seed: ``None`` for a greedy draw (no RNG),
    else ``base_seed + 1000 * candidate_idx``.

    Two properties this guarantees, both load-bearing for reproducing a pool:
    the seed is DISTINCT per candidate index, so K draws of one prompt are K
    different samples rather than K copies; and it is REPRODUCIBLE from
    ``base_seed`` and the candidate index ALONE. vLLM seeds each request's
    sampler from this value independently of the request's position in the
    batch, so — unlike a process-global ``torch.manual_seed`` — resuming a run
    regenerates candidate ``ci`` with exactly the seed a fresh run gave it.
    (Token-level byte reproducibility across DIFFERENT batchings is still not
    guaranteed: vLLM's fused kernels are batch-composition sensitive, the same
    caveat the HF path carries across batch sizes.)"""
    return None if _greedy(spec) else base_seed + 1000 * candidate_idx


def _row(p, si_id: str, candidate_idx: int, response: str) -> dict:
    """One store row in the schema ``build_two_si_pool._row_to_graded`` reads.

    ``si_condition`` is "none" for the undefended pass, "prompted" for any
    SI-prompted teacher. The explicit ``si_id`` field is what
    ``_row_to_graded`` keys on for SI rows, so it is always written.
    ``score``/``graded`` mark the row EXPLICITLY ungraded rather than
    absent-by-omission — the three-state rule that lets a reader tell
    "generated, awaiting grade" from "never generated"."""
    return {
        "attack": p.family,
        "goal_id": p.goal_id,
        "sample_idx": int(p.sample_idx),
        "trial": int(p.trial),
        "si_condition": "none" if si_id == NONE_SI else "prompted",
        "si_id": None if si_id == NONE_SI else si_id,
        "candidate_idx": int(candidate_idx),
        "category": p.category,
        # generate-pos carries the store row's category_group through so its
        # rows match the HF path's schema exactly; the anti-neg / sample-all
        # prompts have no such attribute and keep the historical None (their
        # split is derived at read time — see test_anti_neg_sharding).
        "category_group": getattr(p, "category_group", None),
        "source": p.source,
        "side": p.side,
        "variant_name": p.variant_name,
        "attack_prompt": p.text,
        "response": response,
        "score": None,
        "graded": False,
    }


def cell_key_fields(family, goal_id, variant_key, si_id,
                    candidate_idx) -> str:
    """The resume-key format, in ONE place: the write path (`cell_key`),
    the read-back (`load_done_keys`) and build_two_si_pool's plan-side
    mask/verification all spell it through here — a format drift between
    any pair silently regenerates every cell on resume."""
    return f"{family}|{goal_id}|{variant_key}|{si_id}|{int(candidate_idx)}"


def cell_key(p, si_id: str, candidate_idx: int) -> str:
    """The resume key for one generated cell."""
    return cell_key_fields(p.family, p.goal_id, p.variant_key, si_id,
                           candidate_idx)


def load_done_keys(path: str | Path) -> set[str]:
    """Cell keys already on disk. Tolerates a torn tail from a SIGKILL append
    (the last line of an unflushed JSONL), matching ``_iter_store_rows``."""
    done: set[str] = set()
    path = Path(path)
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            vk = f"{int(r['sample_idx'])}:{int(r.get('trial', 0))}"
            si = r.get("si_id") or NONE_SI
            done.add(cell_key_fields(r["attack"], r["goal_id"], vk, si,
                                     r.get("candidate_idx", 0)))
    return done


def template_extra_kwargs(tok) -> dict:
    """Per-model ``apply_chat_template`` kwargs needed to render a NEUTRAL
    prompt — one carrying our system text and nothing else.

    Some chat templates inject their own system directive. Qwen3.8 prepends
    "Reasoning effort is set to xhigh. Please think carefully through the
    task…" AHEAD of the caller's system content and opens a ``<think>``
    block, so with the default render:

      * the ``none`` condition is not undefended, it is thinking-instructed;
      * every SI sits on top of a second, stronger system directive it did
        not ask for, confounding the SI contrast this pool exists to measure;
      * responses carry reasoning traces, so the grader scores a different
        object than it does for other models.

    ``enable_thinking=False`` removes the preamble and closes the think block.
    Probed per tokenizer rather than matched on model id, so a model that
    stops needing it stops getting it, and a new model that needs it is
    handled without a code change.
    """
    probe = [{"role": "system", "content": _NEUTRALITY_PROBE},
             {"role": "user", "content": "x"}]
    # MERGE (stage3 -> twosweep): the BASELINE render must be the render the
    # harness actually uses, i.e. with the tokenizer marker applied, not the
    # raw template default. Two reasons, and the first is the substantive one:
    #   * asking "does this template inject its own system text?" is only
    #     meaningful against the prompt we will really send. If load_tokenizer
    #     has already pinned enable_thinking=False, a raw baseline measures a
    #     configuration that never ships.
    #   * it also keeps the invariant twosweep's tests assert — that EVERY
    #     render through this tokenizer pins thinking — true of the probe too,
    #     rather than carving out an exception for it in each test.
    from sisteer.models import apply_chat_template_folding
    try:
        base = apply_chat_template_folding(tok, probe)
    except Exception:                                        # noqa: BLE001
        return {}
    if not _injects_extra_system(base):
        return {}
    try:
        alt = tok.apply_chat_template(probe, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
    except Exception:                                        # noqa: BLE001
        return {}
    return {"enable_thinking": False} if not _injects_extra_system(alt) else {}


_NEUTRALITY_PROBE = "\u0001SI_PROBE\u0001"


def _injects_extra_system(rendered: str) -> bool:
    """True when the template put text of its own in the system turn."""
    head = rendered.split(_NEUTRALITY_PROBE)[0]
    # strip role/BOS scaffolding; anything left is the template's own prose
    for tokstr in ("<|im_start|>", "<|turn>", "<bos>", "system", "\n", " "):
        head = head.replace(tokstr, "")
    return len(head) > 0


#: How a reasoning model's think block is handled. Not a style knob -- each
#: value is a DIFFERENT EXPERIMENT, and mixing them within one pool would
#: confound the SI contrast the pool exists to measure.
#:
#:   "off"        think block pre-closed; system turn carries ONLY our SI.
#:                The comparable-across-models default.
#:   "isolated"   think block OPEN, system turn STILL only our SI. Thinking is
#:                the single variable vs "off" -- the valid contrast.
#:   "as_shipped" whatever the template does by default. For Qwen3.8 that is
#:                thinking ON *plus* its own "Reasoning effort is set to
#:                xhigh..." directive PREPENDED to our SI. Answers "is the
#:                model as deployed safer", NOT "does reasoning change
#:                susceptibility" -- two variables move at once.
THINK_MODES = ("off", "isolated", "as_shipped")


def apply_think_mode(rendered_off: str, rendered_default: str,
                     mode: str) -> str:
    """Turn the two template renders into the prompt for ``mode``.

    ``isolated`` is built from the OFF render by reopening its pre-closed
    think block, which is how the system turn stays clean: the template only
    injects its reasoning directive on the thinking-on path, so we take the
    clean turn and add the think block ourselves rather than accepting both.
    """
    if mode not in THINK_MODES:
        raise ValueError(f"think_mode must be one of {THINK_MODES}, got {mode!r}")
    if mode == "off":
        return rendered_off
    if mode == "as_shipped":
        return rendered_default
    for closed, opened in (("<think>\n\n</think>\n\n", "<think>\n"),
                           ("<think></think>", "<think>"),
                           ("<think>\n</think>\n", "<think>\n")):
        if closed in rendered_off:
            return rendered_off.replace(closed, opened, 1)
    raise ValueError(
        "think_mode='isolated' requires a template whose thinking-off render "
        "contains a pre-closed think block to reopen; this one does not, so "
        "there is no way to enable thinking WITHOUT also accepting whatever "
        "else the default render injects. Use 'as_shipped' and state the "
        "confound, or 'off'.")


def strip_think(text: str) -> str:
    """The answer, with any reasoning trace removed.

    Graders must score the ANSWER, or a thinking arm is being measured with a
    different instrument than every non-thinking arm and the numbers stop
    being comparable. Keep the raw text too -- harmful content inside a trace
    that the answer then refuses is a real finding, just a different one.
    """
    import re
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # TEMPLATE-OPENED TRACE. Some chat templates (Qwen3.x with thinking on)
    # emit the OPENING <think> as part of the PROMPT, so the response carries
    # only the closing tag. The regex above then matches nothing, the
    # unterminated check below does not fire, and this function silently
    # returns trace+answer -- the judge scores the reasoning.
    #
    # That is not hypothetical: it shipped on the qwen38_think8k pool, where
    # 0 of 37,564 responses contained "<think>" and 37,385 contained
    # "</think>", and strip_think was a no-op on every one of them.
    #
    # Generators should store the complete assistant turn (prepend the
    # prompt-opened "<think>") so the branch above handles it. This is the
    # safety net for text that was not stored that way.
    if "<think>" not in out and "</think>" in out:
        out = out.split("</think>", 1)[1]
    # an unterminated trace means the budget ran out BEFORE any answer: that
    # is an empty answer, not a refusal, and must not be scored as compliance
    # or as a thwart.
    if "<think>" in out:
        return ""
    return out.strip()


#: Llama-3.2's chat template injects a scaffold ahead of the caller's system
#: text, and it embeds TODAY'S DATE:
#:     Cutting Knowledge Date: December 2023
#:     Today Date: 27 Aug 2026
#:
#:     <our SI>
#: Two problems, and the second is worse than the first. It is extra system
#: content the SI contrast did not ask for -- the same confound as Qwen3.8's
#: reasoning directive. And the date makes the prompt NON-REPRODUCIBLE: the
#: identical run tomorrow renders different bytes, so a prefix cache misses,
#: a pool built today cannot be extended next week, and nothing is comparable
#: across days. Llama-3.1 via the Nous mirror has no such scaffold, so
#: stripping it makes the 1B render match the 8B exactly.
_LLAMA_SCAFFOLD = re.compile(
    r"Cutting Knowledge Date: [^\n]*\nToday Date: [^\n]*\n\n")


def strip_template_scaffold(rendered: str) -> str:
    """Remove template-injected system boilerplate that is not our SI."""
    return _LLAMA_SCAFFOLD.sub("", rendered, count=1)


def build_chat(tok, system_text: str | None, user_text: str,
               extra: dict | None = None, *,
               template_default: bool = False) -> str:
    """Render one chat prompt. The system turn carries the SI, so the shared
    prefix is a literal PREFIX of every rendered string under that SI — which
    is the precondition for prefix caching to hit.

    UNION (stage3 -> twosweep merge). Three independent behaviours, all kept:

    - rendering goes through ``sisteer.models.apply_chat_template_folding``
      (twosweep), which folds the system turn into the first user turn for
      templates that REJECT the system role (the Gemma shape) instead of
      crashing;
    - the render is wrapped in ``strip_template_scaffold`` (stage3), which
      removes Llama-3.2's "Cutting Knowledge Date / Today Date" preamble --
      that text is NOT reproducible across days, so leaving it in makes rows
      generated on different days incomparable;
    - ``extra`` is the per-model neutrality kwargs from
      :func:`template_extra_kwargs`; pass the SAME value for every condition
      of a pool, or the conditions stop being comparable.

    The tokenizer MARKER applies by default. load_tokenizer sets
    ``enable_thinking: False`` on reasoning-capable templates, and twosweep
    requires this render to honour it and to come out byte-identical to
    ``models.build_chat_prompt`` -- the canonical path whose prefix hash the
    provenance sidecar records (F8). Breaking that silently invalidates
    provenance, so "no ``extra``" does NOT mean "no kwargs".

    ``template_default=True`` is the explicit opt-out, and exists for exactly
    one caller: ``apply_think_mode`` builds ``isolated``/``as_shipped`` from
    the TEMPLATE-DEFAULT (thinking-on) render, so it must be able to ask for
    a render the marker has not disabled. Overloading "no extra" to mean that
    would have quietly turned every thinking render thinking-OFF while still
    looking like it ran."""
    from sisteer.models import apply_chat_template_folding

    msgs = ([{"role": "system", "content": system_text}] if system_text else [])
    msgs.append({"role": "user", "content": user_text})
    return strip_template_scaffold(
        apply_chat_template_folding(tok, msgs, extra=extra,
                                    use_marker=not template_default))


# ONE definition of the engine configuration, shared by both halves of the
# pool. VllmSampler builds the POS side; VllmGenerateFn (sampling.generate_fn)
# builds the none/neg side. The whole point of the vLLM neg-side work is that
# those halves come from the same engine CONFIGURATION and not merely the same
# library -- and that claim was maintained by hand in two docstrings until
# 2026-08-26, when a fix to one default (0.90 -> 0.85) silently broke it. A
# shared constructor makes the invariant structural instead of aspirational.
VLLM_ENGINE_DEFAULTS = dict(
    dtype="bfloat16",
    max_model_len=16384,
    gpu_memory_utilization=0.85,
    enforce_eager=False,
    enable_prefix_caching=True,
    # None = vLLM's own default (1024). Must be lowered for a HYBRID
    # Mamba/attention model whose weights leave little cache: each decode
    # sequence needs one Mamba cache block, and vLLM REFUSES to start when
    # max_num_seqs exceeds the blocks that fit. Qwen3.8-27B on an 80GB H100
    # leaves ~15 GiB of cache = 310 blocks against a default of 1024:
    #   ValueError: max_num_seqs (1024) exceeds available Mamba cache
    #   blocks (310) ... CUDA graph capture cannot proceed
    # A dense model of the same size does not hit this, so it cannot be
    # inferred from parameter count alone.
    max_num_seqs=None,
)


def vllm_engine_kwargs(model: str, *, revision: str | None = None, seed: int = 0,
                       **overrides) -> dict:
    """Engine kwargs for ``vllm.LLM``, identical for both pool halves.

    ``enable_prefix_caching`` and ``enforce_eager=False`` are correct ONLY for
    unhooked generation. A steered/compiled path needs the opposite of both,
    because CUDA-graph replay bypasses Python forward hooks -- see the
    guardrail note in sampling/generate_fn.py.
    """
    unknown = set(overrides) - set(VLLM_ENGINE_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown vLLM engine setting(s): {sorted(unknown)}")
    kw = dict(VLLM_ENGINE_DEFAULTS)
    kw.update({k: v for k, v in overrides.items() if v is not None})
    if kw.get("max_num_seqs") is None:
        kw.pop("max_num_seqs")          # let vLLM choose
    # Every engine built from these kwargs reaches the worker with the
    # harness's vLLM patches applied (vllm_patches: GLM-4V's non-neox M-RoPE
    # would otherwise run on vLLM's neox-only triton kernel and sample the
    # wrong distribution). Callers attest with vllm_patches.attest_engine.
    from sisteer.capeval.serving import vllm_patches as _vp
    return _vp.engine_kwargs_with_patch(
        dict(model=model, revision=revision, seed=seed, **kw))


class VllmSampler:
    """One vLLM engine, sampling SI-block by SI-block so the prefix cache holds.

    ``max_model_len`` must cover the longest SI (~6.5k tokens) plus the longest
    attack prompt plus ``max_new_tokens``; WildJailbreak's adversarial rows run
    long, so the default is generous rather than tight."""

    def __init__(self, model: str, *, revision: str | None = None,
                 dtype: str | None = None, max_model_len: int | None = None,
                 gpu_memory_utilization: float | None = None, seed: int = 0,
                 max_new_tokens: int = 512, enforce_eager: bool | None = None,
                 think_mode: str = "off", max_num_seqs: int | None = None):
        # Engine settings default to None so VLLM_ENGINE_DEFAULTS is the ONE
        # place they are written; an explicit value here still wins. Before
        # this the signature restated 0.90 and the neg side restated 0.85,
        # and nothing detected the split.
        # v1 multiprocessing off: this class IS the per-GPU process, and the
        # driver pins it with CUDA_VISIBLE_DEVICES before import.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # vLLM JIT-compiles its sampler and shells out to `ninja` on the
        # first generate(); a venv python invoked by ABSOLUTE PATH leaves
        # its bin/ off PATH, so that child dies with
        # FileNotFoundError: 'ninja'. Owned HERE so no caller can forget it.
        import sys

        bindir = str(Path(sys.executable).parent)
        if bindir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get(
                "PATH", "")
        from sisteer.models import load_tokenizer
        from vllm import LLM

        self.max_new_tokens = max_new_tokens
        self.seed = seed
        # UNION (stage3 -> twosweep merge).
        # Tokenizer: the CANONICAL construction (twosweep F8) --
        # sisteer.models.load_tokenizer, the same call every fit and every
        # HF-path render uses, never a raw AutoTokenizer -- so the anti-neg /
        # sample-all renders carry the same template semantics as the
        # pos/benign legs. It also sets the enable_thinking marker; build_chat
        # deliberately does not read that marker (see its docstring), so the
        # think modes below still get a true template-default render.
        self.tok = load_tokenizer(model, revision=revision)
        # Per-model chat-template neutrality (stage3), resolved ONCE and
        # applied to every condition alike. Qwen3.8 otherwise prepends its own
        # reasoning-effort system directive ahead of ours, which would make
        # the `none` condition thinking-instructed rather than undefended.
        # Empty for Llama and Gemma, so their renders are byte-identical.
        if think_mode not in THINK_MODES:
            raise ValueError(f"think_mode must be one of {THINK_MODES}")
        self.think_mode = think_mode
        self._tmpl_extra = template_extra_kwargs(self.tok)
        if think_mode != "off" and max_new_tokens < 2048:
            # A reasoning trace plus an answer does not fit in 512 tokens.
            # Truncating inside the trace yields NO answer, which grades as
            # non-compliance and manufactures "thinking is safer". Raise it
            # here rather than trusting every caller to remember.
            print(f"[sample] think_mode={think_mode}: raising max_new_tokens "
                  f"{max_new_tokens} -> 4096 so the answer is not truncated "
                  f"inside the reasoning trace", flush=True)
            max_new_tokens = 4096
            self.max_new_tokens = 4096
        if self._tmpl_extra:
            print(f"[sample] chat-template neutrality: {self._tmpl_extra} "
                  f"(this template injects its own system text otherwise)",
                  flush=True)
        # Engine kwargs via the SHARED constructor (stage3), so this half of
        # the pool cannot drift from the neg side on a default again.
        self.llm = LLM(**vllm_engine_kwargs(
            model, revision=revision, seed=seed, dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager, max_num_seqs=max_num_seqs,
        ))   # prefix caching is THE optimisation (see module doc)
        from sisteer.capeval.serving import vllm_patches as _vp
        self.engine_report = _vp.attest_engine(self.llm)   # loud, never silent

    def _sampling_params(self, spec: GenSpec, candidate_idx: int):
        from vllm import SamplingParams

        # k == 1 is the deterministic teacher pass (greedy, reproducible given
        # the prompt — the contract run_static relies on). k > 1 (or an
        # explicit spec.sample) is rejection sampling, where greedy would
        # return K identical strings. The greedy flag and the per-candidate
        # seed live in _greedy / candidate_seed so both are unit-testable
        # without importing vLLM.
        greedy = _greedy(spec)
        return SamplingParams(
            temperature=0.0 if greedy else spec.temperature,
            # top_p matters only when sampling; the HF teacher path inherits the
            # model generation_config's top_p (0.9 for Llama-3.1) while this
            # engine historically forced 1.0 — a wider distribution that made
            # sampled completions drift/thwart differently. spec.top_p lets the
            # pos teacher match the HF/store config (see GenSpec.top_p).
            top_p=1.0 if greedy else spec.top_p,
            # top_k, like top_p, matters only when sampling. HF generate
            # inherits top_k=50 (the transformers GenerationConfig default,
            # since the model's generation_config omits top_k); the pos teacher
            # matches it. Default -1 leaves the historical vLLM behaviour
            # (disabled) for anti-neg / sample-all.
            top_k=-1 if greedy else spec.top_k,
            max_tokens=self.max_new_tokens,
            seed=candidate_seed(self.seed, spec, candidate_idx),
        )

    def run(self, prompts: Sequence, specs: Iterable[GenSpec],
            out_path: str | Path, *, done: set[str] | None = None,
            progress_every: int = 2000) -> dict:
        """Generate every missing cell, appending rows as each block finishes.

        Blocks are (SI, candidate_idx) pairs: all prompts for one SI at one
        candidate index go to the engine together, so the SI prefix is
        prefilled once and every request in the block hits the cache."""
        done = done if done is not None else load_done_keys(out_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = skipped = 0
        stats: dict[str, int] = {}

        for spec in specs:
            for ci in range(spec.k):
                todo = [p for p in prompts
                        if cell_key(p, spec.si_id, ci) not in done]
                skipped += len(prompts) - len(todo)
                if not todo:
                    continue
                if self.think_mode == "off":
                    rendered = [build_chat(self.tok, spec.system_text, p.text,
                                           self._tmpl_extra) for p in todo]
                else:
                    rendered = [
                        apply_think_mode(
                            build_chat(self.tok, spec.system_text, p.text,
                                       self._tmpl_extra),
                            build_chat(self.tok, spec.system_text, p.text,
                                       template_default=True),
                            self.think_mode)
                        for p in todo]
                outs = self.llm.generate(
                    rendered, self._sampling_params(spec, ci))
                # Append after the block so a crash mid-block loses only that
                # block's work; resume regenerates exactly the missing cells.
                with out_path.open("a") as f:
                    for p, o in zip(todo, outs):
                        f.write(json.dumps(
                            _row(p, spec.si_id, ci, o.outputs[0].text)) + "\n")
                        written += 1
                    f.flush()
                    os.fsync(f.fileno())
                key = f"{spec.si_id}:{ci}"
                stats[key] = len(todo)
                print(f"[sample] {key}: +{len(todo)} rows "
                      f"(total written {written}, skipped {skipped})",
                      flush=True)
        return {"written": written, "skipped": skipped, "per_block": stats,
                "out_path": str(out_path)}
