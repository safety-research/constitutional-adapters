"""Model loading and architecture-generic access to decoder layers."""

from __future__ import annotations

import os
import sys
import types
import warnings
from collections.abc import Mapping

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def configure_fast_sdpa() -> None:
    """Disable cuDNN's SDPA backend — the ONE call every model-load site must
    make on this stack (torch 2.11 cu128, H100/H200).

    cuDNN re-plans the attention graph ON CPU for every new (batch, seqlen)
    shape. Two workloads trigger it constantly: ragged training batches (each
    a new pad width) and autoregressive decode (seqlen grows every step). The
    result is CPU-bound work with the GPU near-idle — measured twice on this
    project: a fit wave at 33-45% GPU duty, and eval generation pinned 99% CPU
    with GPUs 0-56% (the gpu-util skill's headline footgun). Falling back to
    flash/mem-efficient SDPA removes the per-shape planning; the numeric
    difference is the usual bf16-kernel-variance class.

    THIS is the enforcement the skill documentation could not provide: a
    documented footgun recurs (it did — training fix didn't reach the eval
    generation path); an enforced default called at every load site does not.
    Idempotent, CUDA-independent (the flag is a global preference only
    consulted when CUDA SDPA runs), env-overridable (SISTEER_KEEP_CUDNN_SDP=1)
    for anyone reproducing a cuDNN-baseline run. Pinned by tests/test_fast_sdpa.py.
    """
    if os.environ.get("SISTEER_KEEP_CUDNN_SDP") == "1":
        return
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:  # older/newer torch without the toggle — nothing to do
        pass

# Attribute paths to the decoder-block ModuleList across common architectures.
# llama/qwen/mistral/gemma: model.model.layers; gpt2: model.transformer.h;
# pythia/gpt-neox: model.gpt_neox.layers; opt: model.model.decoder.layers;
# multimodal wrappers that expose a text stack (gemma-4's
# Gemma4ForConditionalGeneration): model.language_model.layers and the
# older nesting language_model.model.layers.
# backbone.layers: the NemotronH (Nemotron-3 Nano) hybrid Mamba2/MLP/
# attention stack AS THE CHECKPOINT NAMES IT (weight keys
# backbone.layers.N.mixer.*, and the module path NVIDIA's remote-code
# modeling + vLLM keep). transformers 5.14.1's NATIVE NemotronHForCausalLM
# remaps that prefix to plain model.layers (base_model_prefix "model";
# measured on the meta device 2026-09-02: children ['model', 'lm_head'],
# model.layers = 42 NemotronHBlocks), so the FIRST entry already serves it
# there and this one is the guard for loaders that keep the checkpoint
# naming. tests/test_model_onboarding.py pins both spellings.
_LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "backbone.layers",
)


def decoder_layers_path(model: nn.Module) -> str:
    """The ``_LAYER_PATHS`` entry that resolves this model's decoder-block
    ModuleList — the dotted attribute path, not the modules. It is the
    HF-side prefix every decoder module name carries (``model.layers`` on
    llama; ``model.language_model.layers`` on the VLM wrappers gemma-4 /
    GLM-4.6V, whose decoder sits under ``language_model``), which is what
    a PEFT adapter's tensor names must spell out — so consumers that emit
    module names (the vLLM LoRA export) derive them from HERE, off the
    real module tree, never from a second hardcoded string. Same search,
    same refusal as :func:`get_decoder_layers`."""
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return path
    raise ValueError(
        f"Could not locate decoder layers on {type(model).__name__}; "
        f"tried paths {_LAYER_PATHS}"
    )


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    obj = model
    for attr in decoder_layers_path(model).split("."):
        obj = getattr(obj, attr)
    return obj


def num_layers(model: nn.Module) -> int:
    return len(get_decoder_layers(model))


#: The attention output projection's (parent attribute, projection
#: attribute) per decoder-block layout, tried in order:
#:   * ``self_attn.o_proj`` — the Llama family (llama/qwen/gemma/mistral,
#:     the pinned vLLM classes and the HF classes the hooks attach to);
#:   * ``mixer.o_proj``     — NemotronH's ATTENTION blocks. Every NemotronH
#:     block wraps one ``mixer``; only the attention mixer carries
#:     ``o_proj`` (HF: q/k/v/o_proj; vLLM: qkv_proj + o_proj). Mamba2
#:     mixers project out through ``out_proj`` and MLP mixers through
#:     ``down_proj`` — different attribute NAMES, matched exactly, so
#:     neither can ever be mistaken for an attention o_proj.
_OPROJ_SITES = (("self_attn", "o_proj"), ("mixer", "o_proj"))
#: What makes a ``mixer`` an ATTENTION mixer: an ``o_proj`` fed by the
#: query/key/value projections — HF spells them ``q_proj``/``k_proj``/
#: ``v_proj``, vLLM packs them as ``qkv_proj``. Mamba2 (in_proj/out_proj)
#: and MLP (up_proj/down_proj) mixers carry neither spelling.
_ATTENTION_SHAPES = (("q_proj", "k_proj", "v_proj"), ("qkv_proj",))


def attention_oproj_site(block: nn.Module) -> str | None:
    """The dotted attribute path of ``block``'s attention output
    projection (``"self_attn.o_proj"`` / ``"mixer.o_proj"``), or ``None``
    when the block has none — the SITE half of a decoder module name, the
    way :func:`decoder_layers_path` is the PREFIX half (the LoRA export
    joins the two off the real module tree). Exact attribute lookups over
    :data:`_OPROJ_SITES`, never a substring match, so a Mamba
    ``mixer.out_proj`` is not an o_proj."""
    for parent_name, attr in _OPROJ_SITES:
        parent = getattr(block, parent_name, None)
        if parent is None:
            continue
        if parent_name == "mixer" and not any(
                all(hasattr(parent, name) for name in shape)
                for shape in _ATTENTION_SHAPES):
            # NemotronHBlock hangs whichever mixer the layer is — Mamba2,
            # MLP or attention — under the single attribute ``mixer``; only
            # the attention mixer is q/k/v(+o)-SHAPED (HF q/k/v_proj, vLLM
            # qkv_proj). Keying on the shape
            # (not on an ``o_proj`` attribute alone) is what keeps a future
            # mixer that happens to name a projection ``o_proj`` from being
            # mistaken for attention — pinned by
            # tests/test_nemotron_arch.py::test_attention_oproj_probe_is_shape_keyed.
            continue
        if getattr(parent, attr, None) is not None:
            return f"{parent_name}.{attr}"
    return None


def attention_oproj(block: nn.Module):
    """The attention output-projection MODULE of one decoder block, or
    ``None`` when the block has no attention o_proj (NemotronH Mamba2 and
    MLP blocks; Qwen3.5 linear-attention blocks) — the module at
    :func:`attention_oproj_site`."""
    site = attention_oproj_site(block)
    if site is None:
        return None
    obj = block
    for attr in site.split("."):
        obj = getattr(obj, attr)
    return obj


def oproj_layers(model: nn.Module) -> list[int]:
    """Indices of decoder layers that HAVE an attention output projection
    — the capability the o_proj-editing LoRA arms require. On
    all-full-attention models (Llama, gemma-4, Mistral) this is every
    layer, so capability-filtered consumers are byte-identical to their
    historical all-layer behavior there; on hybrid stacks it is a strict
    subset: Qwen3.5's ``qwen3_5`` (16 of 64 layers are full_attention;
    the other 48 are linear_attention blocks with NO ``self_attn`` at
    all) and NemotronH's Nemotron-3-Nano-4B (4 of 42 blocks carry an attention
    ``mixer.o_proj``: [12, 17, 24, 32] per its
    ``hybrid_override_pattern``; the Mamba2/MLP mixers do not).

    A capability probe over :func:`attention_oproj`, deliberately not a
    model_type table: any future hybrid architecture filters correctly
    without being enumerated here."""
    return [i for i, block in enumerate(get_decoder_layers(model))
            if attention_oproj(block) is not None]


def hidden_size(model: nn.Module) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    # Multimodal wrappers (gemma-4, qwen3.5) nest the decoder's dims
    # under text_config.
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return text_cfg.hidden_size
    raise ValueError(f"No hidden_size on {type(cfg).__name__}")


# --- KV-cache allocatability (NemotronH under transformers 5.14.1) ----------
# transformers' DynamicCache(config=...) allocates one cache layer PER ENTRY
# of config.layer_types, looking each type up in DYNAMIC_LAYER_TYPE_MAPPING.
# NemotronH's hybrid pattern declares 17 of Nemotron-3-Nano-4B's 42 blocks
# as "mlp" — a type that mapping has no entry for — so on 5.14.1 every
# forward that allocates a cache dies with KeyError('mlp') BEFORE the first
# block runs: the default eval forward (use_cache=None -> config.use_cache
# = True), generate(), and TeacherPrefixCache.build (explicit
# use_cache=True). Only use_cache=False forwards work (measured 2026-09-02
# on a tiny random-weight NemotronHForCausalLM built from an in-code
# config; the real config's layer_types carry the same "mlp" entries —
# tests/test_model_onboarding.py pins it). This is a capability probe on
# the exact failing lookup, not a model_type table.
#
# The trap is a MISSING PLACEHOLDER, not a missing cache implementation:
# an MLP block holds no state, and transformers 5.15.0 (2026-08-10) fixed
# it by registering "mlp" -> LinearAttentionLayer in both layer-type
# mappings — an EMPTY layer that keeps the per-block indices aligned so
# Cache.get_seq_length() finds the first attention layer (5.14.1 already
# ships "moe" the same way). register_missing_cache_layer_types applies
# exactly that entry at load time on a transformers that lacks it, so
# generate() runs with its cache. Pinning use_cache off was the earlier
# rescue, and it is what made the Nemotron GSM8K leg pathological: a
# cache-free generate() re-runs the whole ~1.5k-token 8-shot prompt plus
# every generated token at EVERY decode step — ~1.75 s/step at batch 128
# and the SAME ~2 h per 500-question cell at batch 32, the signature of a
# compute-bound full recompute (2026-09-04).
# The pin is kept only for a layer type that is genuinely unmappable.

#: Block types that hold NO cache state and that transformers registers as
#: EMPTY placeholder cache layers (cache_utils DYNAMIC/STATIC_LAYER_TYPE_
#: MAPPING in 5.15.0: "moe" and "mlp" -> LinearAttentionLayer). Only these
#: are ever registered here — a type outside this table is unknown to us
#: and stays unmappable (the pin below then fires).
CACHE_PLACEHOLDER_LAYER_TYPES = ("mlp", "moe")

#: The placeholder types the pinned transformers (5.14.1) could NOT map
#: natively: a model whose layer_types carry one was decoded CACHE-FREE
#: by every loader before the pin, whatever the transformers at hand maps
#: today. A STABLE property of the architecture, read off the config —
#: never off whether a particular call mutated the process-global mapping
#: (`registered` is empty on a second load in the same process and on
#: transformers >= 5.15, which would have silenced the base_gsm8k
#: decode-parity gate). 5.14.1 ships "moe"; only "mlp" was missing.
PRE_FIX_CACHE_FREE_LAYER_TYPES = ("mlp",)


def pre_fix_cache_free_layer_types(config) -> list[str]:
    """Layer types of ``config`` (and a nested text_config) that put this
    model on the CACHE-FREE decode under the loader as it was before the
    pin — sorted; ``[]`` for every architecture the pinned transformers
    could cache natively (gemma / qwen / GLM: no such block type)."""
    if config is None:
        return []
    confs = [config]
    if getattr(config, "text_config", None) is not None:
        confs.append(config.text_config)
    return sorted({str(t) for c in confs
                   for t in (getattr(c, "layer_types", None) or [])
                   if t in PRE_FIX_CACHE_FREE_LAYER_TYPES})


def kv_cache_unmappable_layer_types(config) -> list[str]:
    """Layer types in ``config.layer_types`` that transformers' DynamicCache
    cannot allocate a cache layer for (sorted, deduplicated; ``[]`` on every
    architecture whose cache is allocatable)."""
    try:
        from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING
    except ImportError:  # a transformers without the mapping has no such trap
        return []
    types = getattr(config, "layer_types", None) or []
    return sorted({str(t) for t in types if t not in DYNAMIC_LAYER_TYPE_MAPPING})


def register_missing_cache_layer_types(config) -> list[str]:
    """Register, in transformers' cache layer-type mappings, every entry of
    ``config.layer_types`` that is a state-free block type
    (CACHE_PLACEHOLDER_LAYER_TYPES) the installed transformers cannot map —
    byte for byte the entry 5.15.0 ships (``"mlp": LinearAttentionLayer``,
    an empty placeholder layer). Returns the types registered (sorted);
    ``[]`` when nothing was missing or the transformers at hand has no
    such mapping. Process-global ON PURPOSE: the mapping is the module
    constant every ``DynamicCache(config=...)`` consults, including the
    one ``generate()`` builds, so the registration must live there. A
    type that is unmappable and NOT in the placeholder table is left
    alone (returned by kv_cache_unmappable_layer_types afterwards, so the
    caller can pin the cache off)."""
    try:
        from transformers.cache_utils import (DYNAMIC_LAYER_TYPE_MAPPING,
                                              LinearAttentionLayer)
    except ImportError:  # no mapping -> no trap, nothing to register
        return []
    try:
        from transformers.cache_utils import STATIC_LAYER_TYPE_MAPPING
    except ImportError:
        STATIC_LAYER_TYPE_MAPPING = None
    missing = [t for t in kv_cache_unmappable_layer_types(config)
               if t in CACHE_PLACEHOLDER_LAYER_TYPES]
    for t in missing:
        DYNAMIC_LAYER_TYPE_MAPPING[t] = LinearAttentionLayer
        if STATIC_LAYER_TYPE_MAPPING is not None:
            STATIC_LAYER_TYPE_MAPPING.setdefault(t, LinearAttentionLayer)
    return sorted(missing)


def prepare_kv_cache(model) -> dict:
    """Make the model's cache allocatable where transformers merely lacks a
    placeholder entry (register_missing_cache_layer_types, on the config
    and any nested text_config), then pin the cache OFF only for a layer
    type that is still unmappable (pin_kv_cache_off_if_unallocatable).
    Returns ``{"registered": [...], "pinned_off": [...],
    "pre_fix_cache_free": [...]}`` — the first two empty on every
    architecture whose cache already allocates (and ``registered`` empty
    again on a SECOND load in the same process, the mapping being global);
    the third the stable marker pre_fix_cache_free_layer_types reads off
    the config. Says so (a RuntimeWarning naming the types) when it
    registers, because a loader that changes a library table should be
    visible in the log."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {"registered": [], "pinned_off": [], "pre_fix_cache_free": []}
    confs = [cfg]
    if getattr(cfg, "text_config", None) is not None:
        confs.append(cfg.text_config)
    registered = sorted({t for c in confs
                         for t in register_missing_cache_layer_types(c)})
    if registered:
        import transformers

        warnings.warn(
            f"{type(model).__name__}: config.layer_types carry {registered}, "
            f"which transformers {transformers.__version__}'s cache layer-type "
            "mapping lacks — registered as empty placeholder cache layers "
            "(LinearAttentionLayer), the entry transformers 5.15.0 ships, so "
            "default forwards allocate their cache and generate() decodes "
            "WITH it instead of recomputing the whole prefix per token.",
            RuntimeWarning, stacklevel=3)
    pinned = pin_kv_cache_off_if_unallocatable(model)
    out = {"registered": registered, "pinned_off": pinned,
           "pre_fix_cache_free": pre_fix_cache_free_layer_types(cfg)}
    try:
        # Left on the model so a consumer can tell, after the fact, what
        # this load did and — stably — that the pre-fix loader decoded this
        # model CACHE-FREE (the base_gsm8k decode-parity gate reads
        # pre_fix_cache_free; `registered` is informational only).
        model.kv_cache_preparation = dict(out)
    except (AttributeError, TypeError):
        pass
    return out


def pin_kv_cache_off_if_unallocatable(model) -> list[str]:
    """Turn ``use_cache`` OFF (model config, nested text_config, and the
    generation config) when the KV cache cannot be allocated for this
    architecture, so default forwards run and ``generate()`` takes its
    cache-free (full-recompute) path instead of raising. Returns the
    offending layer types — ``[]`` (and nothing touched) everywhere else.
    LOUD: warns naming the types, because the consequences are real —
    TeacherPrefixCache is unavailable on such a model (the config must run
    teacher_prefix_cache: false) and HF generation is O(n^2)."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return []
    confs = [cfg]
    if getattr(cfg, "text_config", None) is not None:
        confs.append(cfg.text_config)
    bad = sorted({t for c in confs for t in kv_cache_unmappable_layer_types(c)})
    if not bad:
        return []
    for c in confs:
        c.use_cache = False
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.use_cache = False
    warnings.warn(
        f"{type(model).__name__}: config.layer_types carry {bad}, which "
        "transformers' DynamicCache cannot allocate (KeyError at cache "
        "construction) — use_cache pinned False on the model and generation "
        "configs. Consequences: HF generate() runs cache-free (full "
        "recompute per token), and TeacherPrefixCache cannot be built on "
        "this model, so its configs must run teacher_prefix_cache: false.",
        RuntimeWarning, stacklevel=3)
    return bad


def pin_generation_stop_ids(model, tok) -> list[int]:
    """Make ``model.generation_config.eos_token_id`` contain every id that
    ENDS an assistant turn: the tokenizer's eos, and the chat template's
    turn terminator located by ``chat_markers.derive_assistant_markers``
    (the authoritative event — the token the template actually closes a
    turn with). Returns the ids that had to be ADDED (``[]`` on the models
    where the generation config already lists them: Llama-3.1's
    ``[128001, 128008, 128009]`` carries ``<|eot_id|>``, Qwen's carries
    ``<|im_end|>``).

    The trap this closes: Nemotron-3-Nano-4B's generation_config.json says
    ``eos_token_id: 2`` (``</s>``) while its ChatML template terminates
    every turn with ``<|im_end|>`` (id 11) — the tokenizer's eos IS 11, so
    tokenizer-keyed paths (completion EOS append, hf_engine._eos_ids) are
    right, but a bare ``model.generate()`` reads the generation config and
    would never stop on the token the model actually emits. Additive only:
    an id is never removed, so a model's own stop set is preserved."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return []
    have = gc.eos_token_id
    if have is None:
        have_list: list[int] = []
    elif isinstance(have, int):
        have_list = [int(have)]
    else:
        have_list = [int(e) for e in have]
    want: list[int] = []
    if getattr(tok, "eos_token_id", None) is not None:
        want.append(int(tok.eos_token_id))
    if getattr(tok, "chat_template", None):
        from .chat_markers import derive_assistant_markers
        try:
            term = derive_assistant_markers(tok).terminator
        except Exception:  # noqa: BLE001 — a template the probe cannot
            term = None    # derive has no locatable terminator to pin
        if term is not None:
            want.append(int(term))
    missing = [t for t in dict.fromkeys(want) if t not in have_list]
    if missing:
        gc.eos_token_id = have_list + missing
        warnings.warn(
            f"generation_config.eos_token_id {have!r} did not include the "
            f"turn-ending id(s) {missing} (tokenizer eos / chat-template "
            "terminator) — added, so generate() stops where the template "
            "ends a turn instead of running to max_new_tokens.",
            RuntimeWarning, stacklevel=3)
    return missing


# --- Llama-3.1 base-model pin -----------------------
# The ACTUAL loaded model must always be the NousResearch mirror, on every
# branch and every cluster. This is NOT about gating or convenience -- both
# mirrors are reachable on both clusters and the WEIGHTS are identical. It is
# about the CHAT TEMPLATE, which is not:
#
#   NousResearch  renders exactly the messages given.
#   meta-llama    silently injects a 25-token default system block
#                 ("Cutting Knowledge Date: December 2023 / Today Date: ...")
#                 even when NO system message is passed.
#
# That difference is disqualifying for this project: our `none` arm is defined
# as NO system prompt, so under meta-llama "unprompted" is not unprompted, and
# `prompted` vs `none` would differ by more than the safety instruction alone.
# It also silently bakes a hardcoded date into every prompt of a safety eval.
# Mixing the two across runs makes results non-comparable by ~25 tokens of
# prompt -- which already happened once here, undetected, because the configs
# diverged per cluster while everyone assumed "weight-identical" meant
# "interchangeable".
_LLAMA31_REQUIRED = "NousResearch/Meta-Llama-3.1-8B-Instruct"
_LLAMA31_BANNED = ("meta-llama/Llama-3.1-8B-Instruct",
                   "meta-llama/Meta-Llama-3.1-8B-Instruct")


def _enforce_llama31_mirror(name: str) -> None:
    """WARN on a banned Llama-3.1 id; hard-error only under an opt-in strict
    flag. The rule is a GOING-FORWARD switch: NousResearch
    for new work, meta-llama fine for experiments already underway. A hard
    error would be a retroactive ban -- and since ~24 worktrees share this
    clone, it would crash a peer's in-flight wave at model load, including one
    session that was explicitly told to use meta-llama. So: loud, not fatal.
    Set SISTEER_STRICT_MIRROR=1 to make it fatal for a lane that wants the
    guarantee."""
    if name not in _LLAMA31_BANNED:
        return
    msg = (
        f"model {name!r} is DISCOURAGED going forward -- prefer "
        f"{_LLAMA31_REQUIRED!r}. The weights are identical but the chat "
        "templates are NOT: meta-llama injects a 25-token default system "
        "block (Cutting Knowledge Date / Today Date) even with no system "
        "message, which makes a no-system-prompt arm not-unprompted and makes "
        "results incomparable with runs on the other mirror. Mixing the two "
        "across a comparison is a silent ~25-token prompt confound."
    )
    import os
    if os.environ.get("SISTEER_STRICT_MIRROR"):
        raise ValueError(msg + " (SISTEER_STRICT_MIRROR=1 makes this fatal.)")
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


# Chat-template identity. Two readings of the same defect: freezing
# `chat_template_hash` shows the mirrors diverge, and the mirror that
# diverges injects a 25-token default system block.
#
# The HASH is the real invariant, not the model id: a mirror under ANY name
# carrying the injecting template passes a name check and still shifts every
# prompt by ~25 tokens. So warn on the id (cheap, pre-load) AND check the
# template once the tokenizer exists.
_TEMPLATE_HASH_NOUSRESEARCH = "b48c47f644389271"   # renders exactly the messages given
_TEMPLATE_HASH_METALLAMA = "e10ca381b1ccc5cf"      # injects a default system block


def chat_template_hash(tok) -> str | None:
    """First 16 hex of sha256 over the tokenizer's chat template, or None."""
    t = getattr(tok, "chat_template", None)
    if not t:
        return None
    import hashlib
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def check_chat_template(tok, *, strict: bool | None = None) -> str | None:
    """Warn (or raise under SISTEER_STRICT_MIRROR) if the tokenizer carries the
    injecting template. Returns the hash so callers can stamp it into a
    manifest -- a recorded hash is what makes a mixed-mirror comparison
    diagnosable after the fact instead of invisible."""
    import os
    h = chat_template_hash(tok)
    if h is None or h != _TEMPLATE_HASH_METALLAMA:
        return h
    msg = (
        f"tokenizer chat_template hash {h} is the meta-llama template, which "
        "INJECTS a ~25-token default system block ('Cutting Knowledge Date / "
        "Today Date') even when no system message is passed. A no-system-prompt "
        "arm is therefore not unprompted, and any comparison spanning this and "
        f"the {_TEMPLATE_HASH_NOUSRESEARCH} template carries a silent ~25-token "
        "prompt confound. This check is on the TEMPLATE, not the model id, so a "
        "differently-named mirror cannot slip past it."
    )
    if strict if strict is not None else bool(os.environ.get("SISTEER_STRICT_MIRROR")):
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return h


def serving_identity(tok, cfg: dict, model=None) -> dict:
    """The train-serve identity stamp: what was
    ACTUALLY served, measured from the live tokenizer/model, never inferred
    from the config alone. Stamped into every cell manifest and every
    controls_summary provenance so a mirror/template mismatch is an artifact
    lookup forever rather than cluster archaeology (the psr provenance hunt,
    battery_results PROVENANCE RESOLVED)."""
    resolved_id = None
    resolved_rev = None
    if model is not None:
        mcfg = getattr(model, "config", None)
        resolved_id = getattr(mcfg, "_name_or_path", None) or None
        resolved_rev = getattr(mcfg, "_commit_hash", None) or None
    if resolved_id is None:
        resolved_id = getattr(tok, "name_or_path", None) or None
    return {
        "requested_model": cfg.get("model"),
        "requested_revision": cfg.get("model_revision"),
        "resolved_model_id": resolved_id,
        "resolved_revision": resolved_rev,
        "chat_template_hash": chat_template_hash(tok),
    }


def pin_reasoning_template_off(tok) -> bool:
    """Pin a reasoning-capable chat template (Qwen3/Qwen3.5) to
    thinking-OFF renders by setting ``tok._sisteer_template_kwargs``.

    Reasoning templates default to emitting <think> blocks unless
    ``enable_thinking=False`` is passed at render time. Thinking tokens
    would pollute completions, checkers, and teacher/student token
    alignment — and for RESPONSE-masked objects the empty
    ``<think>\n\n</think>\n\n`` block IS the assistant header the mask
    keys on (Qwen3.5: H=7), so a thinking-ON render does not merely add
    text, it MOVES the mask boundary and serves a different intervention
    than was fit. Every render seam must therefore key on the SAME
    signal: this helper for HF-side tokenizers (honored by
    apply_chat_template_folding / chat_markers.render_ids /
    openai_shim._render), and build_server_launch's
    --default-chat-template-kwargs pin for the served side.

    Returns True when the pin applied (the template carries the
    switch), False for plain templates (llama/gemma) — where setting
    kwargs would be inert noise."""
    template = getattr(tok, "chat_template", None) or ""
    if "enable_thinking" in template:
        tok._sisteer_template_kwargs = {"enable_thinking": False}
        return True
    return False


def load_tokenizer(name: str, revision: str | None = None, *,
                   leading_special_tokens: str | None = None):
    """Tokenizer HALF of load_model_and_tokenizer, factored out so
    tokenizer-only consumers (the precompute-tokens sidecar builder, the
    parallel scheduler's warm step) construct it IDENTICALLY to a fit —
    same pad default, same template kwargs pin — without paying the model
    load. Any drift between this and a fit's tokenizer is caught at warm
    time by the sidecar probe.

    ``leading_special_tokens`` (the config's ``leading_special_tokens``
    policy; None = leave the tokenizer's own template-free convention
    alone) is applied HERE — the one construction every fit, sidecar
    build and budget measurement shares — so a policy can never be
    honoured by one consumer and missed by another."""
    _enforce_llama31_mirror(name)
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pin_reasoning_template_off(tok)
    check_chat_template(tok)
    from .extraction.leading_tokens import apply_leading_special_tokens

    return apply_leading_special_tokens(tok, leading_special_tokens)


def auto_model_class(name: str, revision: str | None = None, *,
                     config=None):
    """The auto-class that actually maps this checkpoint's config:
    ``AutoModelForCausalLM``, or ``AutoModelForImageTextToText`` for VLM
    checkpoints that transformers registers ONLY there. GLM-4.6V
    (model_type ``glm4v``) has no causal-LM mapping in transformers 5.14
    and loads solely as ``Glm4vForConditionalGeneration`` under the
    image-text-to-text auto-class; gemma-4 by contrast appears in BOTH
    maps, which is why this seam never fired before. Either way the
    decoder is reachable through ``_LAYER_PATHS``'s ``language_model``
    entries, so everything downstream is unchanged. Fails loudly when
    neither auto-class maps the config — never guesses a loader.

    ``config`` (an already-loaded ``PretrainedConfig``) skips the hub/
    cache resolution for callers that resolved it themselves under
    ``local_files_only`` and must not reach the network from here."""
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)
    if config is None:
        try:
            config = AutoConfig.from_pretrained(name, revision=revision)
        except Exception:
            # A name whose config cannot be resolved here (offline fake
            # ids in tests, ad-hoc local paths) keeps the historical
            # loader: AutoModelForCausalLM.from_pretrained raises its own
            # louder error for a genuinely bad name, and monkeypatched
            # loaders never needed a config lookup to run first.
            return AutoModelForCausalLM
    cfg = config
    if cfg.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return AutoModelForCausalLM
    if cfg.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    raise ValueError(
        f"model_type {cfg.model_type!r} ({name}) maps to neither "
        "AutoModelForCausalLM nor AutoModelForImageTextToText in this "
        "transformers version; refusing to guess a loader")


# --------------------------------------------------------------------------- #
# fused Mamba kernels: REQUIRED on every model with a Mamba mixer              #
# --------------------------------------------------------------------------- #

#: transformers' Mamba-family modeling modules each define module-level
#: wrappers through ``use_kernel_func_from_hub_with_fallback(<kernel>,
#: <package>)``. The binding happens at IMPORT time: each wrapper closes over
#: the named package's kernel when that package imports, else over the
#: module's own pure-torch function -- nothing at call time re-checks, and
#: the config's ``use_mamba_kernels`` is never read (5.15.0). So "installed"
#: proves nothing about "used"; the closure does. The wrapper keeps no record
#: of the package it was meant to bind (its closure holds the implementation
#: and a parameter tuple, nothing else), so the NAME is the only trace left
#: of a wrapper's package once that package is absent -- the case the guard
#: exists for. Hence this table, read off the installed transformers (5.15.0)
#: and pinned against its source by ``tests/test_models_device_defaults.py``,
#: so a transformers bump that renames or adds a wrapper fails a test instead
#: of silently narrowing the guard.
#:
#: Three families of wrapper. The ``causal_conv1d`` pair is NOT a Mamba
#: marker: in 5.15.0 it is also called by classes with no Mamba mixer at all
#: -- qwen3_5, qwen3_5_moe, qwen3_next and olmo_hybrid (gated-deltanet, whose
#: recurrence is ``fla``'s), lfm2, lfm2_moe and inkling -- for a short causal
#: convolution. Reading the pair alone as the marker refused Qwen3.5 on a GPU
#: node synced without the mamba extra: Qwen3.5 is not a Mamba hybrid; only
#: Nemotron is. The ``mamba_ssm`` wrappers come in two trios and are called
#: by Mamba mixers only: the Mamba2 trio (``mamba2_*``) by the mixers of
#: nemotron_h, mamba2, bamba, falcon_h1, zamba2 and granitemoehybrid; the
#: Mamba-1 trio (``mamba_inner_fn``, ``mamba_selective_state_update``,
#: ``mamba_selective_scan``) by mamba's and falcon_mamba's, and its last two
#: by jamba's and zamba's (jamba binds ``mamba_inner_fn`` too but no method
#: of its mixer calls it). A class whose methods call a ``mamba_ssm``
#: wrapper is a Mamba mixer (:func:`mamba_mixer_classes`); a MODEL is the
#: guard's subject when it instantiates one. A marker of the Mamba2 trio
#: alone let an installed Jamba, Mamba, FalconMamba or Zamba load on cuda
#: with every wrapper on the fallback and the guard saying nothing
#: (reproduced on tiny real models of each, 2026-10-12); the Mamba-1
#: modules themselves emit no warning on that path in 5.15.0.
MAMBA_KERNEL_WRAPPERS: dict[str, str] = {
    "causal_conv1d_fn": "causal_conv1d",
    "causal_conv1d_update": "causal_conv1d",
    # Mamba2 mixers
    "mamba2_chunk_scan": "mamba_ssm",
    "mamba2_selective_state_update": "mamba_ssm",
    "mamba2_split_conv1d_scan_combined": "mamba_ssm",
    # Mamba-1 mixers
    "mamba_inner_fn": "mamba_ssm",
    "mamba_selective_state_update": "mamba_ssm",
    "mamba_selective_scan": "mamba_ssm",
}

#: The wrappers whose call makes a module class a Mamba mixer: the
#: ``mamba_ssm`` names of :data:`MAMBA_KERNEL_WRAPPERS`, Mamba2's and
#: Mamba-1's alike.
MAMBA_SSM_WRAPPERS: tuple[str, ...] = tuple(
    n for n, pkg in MAMBA_KERNEL_WRAPPERS.items() if pkg == "mamba_ssm")


def _kernel_implementation(wrapper):
    """The function a hub-with-fallback wrapper actually calls: the
    ``implementation`` its closure holds (the package kernel, or the torch
    fallback). Descends through any outer decorator that itself closes over
    a callable; a plain function (no closure) is the implementation."""
    fn = wrapper
    for _ in range(8):
        inner = []
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:            # an empty cell
                continue
            if callable(v) and not isinstance(v, type):
                inner.append(v)
        if not inner:
            return fn
        fn = inner[0]
    return fn


def _functions_behind(attr) -> list:
    """The plain functions a class-body attribute stands for: the function
    itself, the one under a ``staticmethod`` / ``classmethod``, a property's
    accessors, and -- through ``__wrapped__`` (``functools.wraps``) and
    through closure cells that hold a function (a decorator without it) --
    the function each decorator wraps. The mixers' ``forward`` is decorated
    (``@force_accelerate_hooks`` in 5.15.0), so the decorator's own code is
    what a plain ``__code__`` read sees, and it names no kernel. Walks by
    identity, so a decorator that reaches itself ends the walk."""
    out: list = []
    seen: set[int] = set()
    stack = [attr]
    while stack:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        if isinstance(f, (staticmethod, classmethod)):
            stack.append(f.__func__)
            continue
        if isinstance(f, property):
            stack.extend(a for a in (f.fget, f.fset, f.fdel) if a is not None)
            continue
        if not isinstance(f, types.FunctionType):
            continue
        out.append(f)
        wrapped = getattr(f, "__wrapped__", None)
        if wrapped is not None:
            stack.append(wrapped)
        for cell in f.__closure__ or ():
            try:
                v = cell.cell_contents
            except ValueError:            # an empty cell
                continue
            if callable(v) and not isinstance(v, type):
                stack.append(v)
    return out


def _global_names(fn) -> set[str]:
    """Every name ``fn``'s code loads or stores by name -- globals and
    attributes alike (``co_names``) -- through the code objects nested in
    it (comprehensions, inner functions)."""
    out: set[str] = set()
    stack = [fn.__code__]
    while stack:
        code = stack.pop()
        out.update(code.co_names)
        stack.extend(k for k in code.co_consts if isinstance(k, types.CodeType))
    return out


_KERNEL_CALLS_BY_CLASS: dict[type, tuple[tuple[str, str], ...]] = {}


def kernel_wrappers_called(cls: type) -> tuple[tuple[str, str], ...]:
    """``((wrapper, module), ...)`` for every :data:`MAMBA_KERNEL_WRAPPERS`
    name a method of ``cls`` -- its own class body and its bases', decorators
    unwrapped (:func:`_functions_behind`) -- names in its code, when that
    name is bound to a callable in the method's OWN globals (the module the
    method was defined in: the namespace the call resolves in at run time,
    and the module whose closure the guard then reads). Read once per class
    and cached: a class's code does not change, and the LIVE closure behind
    each name is read at every call of :func:`resolved_mamba_kernels`, so a
    package that binds later (or a test that rebinds a wrapper) is seen.

    A mixer names its wrappers as module globals in its forward (``if
    is_fast_path_available and ...: mamba2_split_conv1d_scan_combined(...)``,
    5.15.0), so this is the structural read of "this class would call that
    kernel". The set is the one the model would RUN, not the one its module
    binds: Jamba's module binds ``mamba_inner_fn`` and its mixer never calls
    it. A NemotronH norm class imported from the zamba2 module names no
    wrapper and reads as nothing, as it should."""
    hit = _KERNEL_CALLS_BY_CLASS.get(cls)
    if hit is not None:
        return hit
    found: dict[str, str] = {}
    for base in cls.__mro__:
        if base is object:
            continue
        for attr in vars(base).values():
            for fn in _functions_behind(attr):
                g = fn.__globals__
                for n in _global_names(fn) & MAMBA_KERNEL_WRAPPERS.keys():
                    if n not in found and callable(g.get(n)):
                        found[n] = str(g.get("__name__", ""))
    hit = tuple(sorted(found.items()))
    _KERNEL_CALLS_BY_CLASS[cls] = hit
    return hit


def mamba_mixer_classes(model) -> tuple[type, ...]:
    """The classes of the Mamba mixers ``model`` INSTANTIATES -- Mamba2 or
    Mamba-1 -- in a fixed order, empty when it has none. A Mamba mixer is a
    submodule class whose methods call a ``mamba_ssm`` wrapper
    (:data:`MAMBA_SSM_WRAPPERS`; :func:`kernel_wrappers_called`). In
    transformers 5.15.0 exactly ten classes do: ``NemotronHMamba2Mixer``,
    ``Mamba2Mixer``, ``BambaMixer``, ``FalconH1Mixer``, ``Zamba2MambaMixer``
    and ``GraniteMoeHybridMambaLayer`` call the Mamba2 trio; ``MambaMixer``
    and ``FalconMambaMixer`` the Mamba-1 trio, ``JambaMambaMixer`` and
    ``ZambaMambaMixer`` two of it. A class calling only the ``causal_conv1d``
    pair (Qwen3.5's ``Qwen3_5GatedDeltaNet``, among others) is not one.

    The INSTANTIATED submodules are read, not the model's class or its
    module: a Jamba configured with ``attn_layer_period=1`` and
    ``attn_layer_offset=0`` builds attention layers only, no
    ``JambaMambaMixer``, and runs no Mamba kernel -- its modeling module
    still binds every wrapper, and reading the module as the marker refused
    that model on cuda (review 2026-10-12). An object without a ``modules()``
    walk (a bare fake) has no submodules and no mixer."""
    walk = getattr(model, "modules", None)
    if not callable(walk):
        return ()
    ssm = set(MAMBA_SSM_WRAPPERS)
    hits: dict[type, None] = {}
    for sub in walk():
        cls = type(sub)
        if cls in hits:
            continue
        if any(n in ssm for n, _m in kernel_wrappers_called(cls)):
            hits[cls] = None
    return tuple(sorted(hits, key=lambda c: (c.__module__, c.__qualname__)))


def mamba_modeling_module(model):
    """The modeling module whose kernel wrappers ``model``'s Mamba mixer
    calls (the module the mixer's methods were defined in; in 5.15.0 the
    same module as the model's ``ForCausalLM`` class, for every one of the
    ten families), else ``None``: the model instantiates no Mamba mixer
    (:func:`mamba_mixer_classes`). A model with mixers from several
    modules (none in 5.15.0) answers with the first's; the guard itself
    reads every mixer through :func:`resolved_mamba_kernels`."""
    classes = mamba_mixer_classes(model)
    if not classes:
        return None
    for n, modname in kernel_wrappers_called(classes[0]):
        if n in MAMBA_SSM_WRAPPERS:
            return sys.modules.get(modname)
    return None


def resolved_mamba_kernels(model) -> dict[str, str] | None:
    """``{wrapper: implementation module}`` for a model that instantiates a
    Mamba mixer (:func:`mamba_mixer_classes`): every wrapper in
    :data:`MAMBA_KERNEL_WRAPPERS` its mixers' methods call -- the
    ``causal_conv1d`` pair and whichever ``mamba_ssm`` names those mixers
    use -- read off the LIVE closures of the module each call resolves in.
    ``None`` for every other model: a non-Mamba architecture, Qwen3.5
    included (its gated-deltanet class calls the ``causal_conv1d`` pair for
    a short convolution and no ``mamba_ssm`` wrapper), and a Mamba-family
    model configured without a Mamba mixer (an attention-only Jamba)."""
    classes = mamba_mixer_classes(model)
    if not classes:
        return None
    out: dict[str, str] = {}
    for cls in classes:
        for n, modname in kernel_wrappers_called(cls):
            if n in out:
                continue
            mod = sys.modules.get(modname)
            wrapper = getattr(mod, n, None) if mod is not None else None
            if wrapper is None:
                continue
            out[n] = str(getattr(_kernel_implementation(wrapper), "__module__", ""))
    return out


def _is_from(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def model_device(model) -> torch.device:
    """The device a uniformly-placed model's weights live on: that of its
    first parameter, else its first buffer, else CPU (a module with neither
    has nowhere else to be). This is where ``methods.load_object`` moves an
    object's tensors. It is NOT what the fused-kernel refusal reads: a
    partitioned model (Accelerate's ``device_map=``) has weights on several
    devices, offloaded ones parked on ``meta`` until the forward moves them,
    and its first parameter says nothing about where a mixer executes — for
    that question see :func:`model_devices`."""
    for t in model.parameters():
        return t.device
    for t in model.buffers():
        return t.device
    return torch.device("cpu")


def hook_execution_devices(hook) -> list:
    """Every ``execution_device`` a module hook carries, raw as the hook
    holds it (``torch.device``, str, int or ``None``): the hook's own, and
    that of every member of every hook CONTAINER it holds, nested to any
    depth. Accelerate's one container is ``SequentialHook`` (``.hooks``, a
    tuple), and ``add_hook_to_module(module, hook, append=True)`` wraps the
    module's EXISTING hook and the new one in a fresh ``SequentialHook`` —
    so a second append nests a ``SequentialHook`` inside a ``SequentialHook``,
    and a walk over the outer one's immediate members alone reads neither
    inner hook. That walk reported a tiny Qwen3.5 as ``{cpu}`` with an
    ``AlignDevicesHook(execution_device=cuda:0)`` one level down, and the
    fused-kernel refusal warned instead of raising (review 2026-10-12).

    Any object with a ``hooks`` sequence is a container here, so a
    ``SequentialHook`` subclass or a hook of another library that chains the
    same way is read too; traversal is by identity, so a container that
    reaches itself cannot loop. Accelerate's hooks in 1.14.0 hold an
    execution device in exactly one attribute, ``execution_device``
    (``AlignDevicesHook``, ``CpuOffload``); ``CpuOffload.prev_module_hook``
    is a handle on a PREVIOUS model in a pipeline, not a member hook of this
    module, and is deliberately not followed — its device is that model's."""
    out: list = []
    seen: dict[int, object] = {}            # id -> the hook, kept alive while we walk
    stack = [hook]
    while stack:
        h = stack.pop()
        if h is None or id(h) in seen:
            continue
        seen[id(h)] = h
        out.append(getattr(h, "execution_device", None))
        members = getattr(h, "hooks", None)
        if members is None or isinstance(members, (str, bytes)):
            continue
        try:
            stack.extend(reversed(list(members)))   # keep the hooks' own order
        except TypeError:                          # ``hooks`` that is not iterable
            continue
    return out


def model_devices(model) -> set[torch.device]:
    """Every device the model can EXECUTE on, read off the model itself:

    * the device of every parameter and every buffer (``meta`` excluded: a
      placeholder for a weight that lives elsewhere, not a place to run);
    * for a model Accelerate partitioned (``device_map=``), the
      ``execution_device`` of every module hook (``_hf_hook``), through
      every ``SequentialHook`` nested to any depth
      (:func:`hook_execution_devices`). An offloaded module keeps its
      weights on ``meta`` / CPU and runs on that device, so the weights
      alone would miss it;
    * every entry of ``hf_device_map`` (``"disk"`` is storage, not
      execution).

    A bare int anywhere above is a GPU ordinal, as Accelerate writes one
    (``device_map={"": 0}``), and is spelled ``cuda:<int>`` here explicitly:
    ``torch.device(0)`` asks the CURRENT accelerator for its type and raises
    on a torch with none (the CPU wheel), which would turn a device-map read
    on a CPU box into a crash instead of an answer.

    The ONE reader the fused-kernel refusal goes through
    (``assert_fused_mamba_kernels``), so a test can report a CPU model as
    executing on ``cuda`` by patching this name. Reading the first parameter
    alone (:func:`model_device`) let a model with a CPU-first parameter and
    a CUDA-executing mixer through the refusal (review 2026-10-12)."""
    found: set[torch.device] = set()

    def add(d) -> None:
        if d is None or (isinstance(d, str) and d == "disk"):
            return
        if isinstance(d, Mapping):           # a per-module execution map
            for v in d.values():
                add(v)
            return
        if isinstance(d, torch.device):
            dev = d
        elif isinstance(d, int) and not isinstance(d, bool):
            dev = torch.device("cuda", d)
        else:
            dev = torch.device(d)
        if dev.type != "meta":
            found.add(dev)

    for t in model.parameters():
        add(t.device)
    for t in model.buffers():
        add(t.device)
    modules = getattr(model, "modules", None)
    for mod in (modules() if callable(modules) else ()):
        for d in hook_execution_devices(getattr(mod, "_hf_hook", None)):
            add(d)
    for d in (getattr(model, "hf_device_map", None) or {}).values():
        add(d)
    return found


def assert_fused_mamba_kernels(model) -> dict[str, str] | None:
    """Mamba-hybrid work ON A CUDA DEVICE runs on the fused ``mamba_ssm`` /
    ``causal_conv1d`` kernels or not at all: training, generation,
    evaluation and activation capture alike. The pure-torch fallback is the
    silent failure this refuses there: job 470950's all-block capture on
    Nemotron-3-Nano-4B ran 21 minutes without finishing 14,000 rows that the
    same capture does in ~40 s on gemma-4-E4B, with nothing in the log to
    say why. The venv gets the kernels with
    ``uv sync --frozen --extra cu128 --extra mamba`` (pyproject's ``mamba``
    extra: the pinned PyPI sdists, which uv compiles against the locked torch).

    SUBJECT (2026-10-12): a model that INSTANTIATES a Mamba mixer, Mamba2 or
    Mamba-1, and nothing else -- read structurally by
    :func:`mamba_mixer_classes`: a submodule class whose methods call a
    ``mamba_ssm`` wrapper (in transformers 5.15.0 the mixers of nemotron_h,
    mamba2, bamba, falcon_h1, zamba2 and granitemoehybrid on the Mamba2
    side, mamba, falcon_mamba, jamba and zamba on the Mamba-1 side).
    Qwen3.5 is NOT one: it is a gated-deltanet hybrid whose
    ``Qwen3_5GatedDeltaNet`` calls only ``causal_conv1d_fn`` /
    ``causal_conv1d_update``, for a short causal convolution, and an earlier
    reading that took any of those wrappers as the marker refused Qwen3.5 on
    a GPU node synced with ``--extra cu128`` alone: Qwen3.5 is not a Mamba
    hybrid; only Nemotron is. The reading that replaced it, the Mamba2 trio
    alone, let a Jamba, Mamba, FalconMamba or Zamba through on cuda with
    every wrapper on the fallback and no warning, and the one after that,
    "the model's class module binds a ``mamba_ssm`` wrapper", refused a
    Jamba configured with attention layers only (``attn_layer_period=1,
    attn_layer_offset=0``: no ``JambaMambaMixer`` is built and no Mamba
    kernel can run, but the module binds every wrapper). Hence the
    instantiated mixer is the marker. On Qwen3.5, on a Mamba-family model
    built without a mixer and on every other non-Mamba architecture this
    returns ``None`` with no warning, on any device.

    SCOPE (2026-10-12): the refusal is a HARD error only when the model can
    execute on ``cuda``. The fused kernels are CUDA-only packages — there is
    no build of either for Apple's ``mps`` or for a plain CPU — so on those
    devices the pure-torch path is not a misconfiguration but the only path
    there is. Off CUDA the guard therefore emits ONE loud ``RuntimeWarning``
    naming the device(s), the missing package(s) and that the fallback is
    slow and expected there, and returns the resolved modules like the fused
    case does. A GPU job's protection is unchanged: a Mamba model on
    ``cuda`` without the kernels still raises, same message, so nothing on
    the cluster can crawl on the fallback unnoticed.

    "Can execute on cuda" is read through :func:`model_devices` — EVERY
    parameter and buffer plus Accelerate's per-module execution devices and
    ``hf_device_map`` — not the first parameter (:func:`model_device`): a
    model partitioned with ``device_map=`` may hold its first parameter on
    CPU, or on ``meta`` once offloaded, while a mixer executes on cuda, and
    that job crawls on the fallback exactly like a uniform cuda load (review
    2026-10-12, which walked such a model through the first-parameter read
    and got the warning). Any cuda execution device in the model refuses —
    the mixers alone are not singled out, because a job with any part on a
    GPU is a GPU job, and a mixer placed on CPU beside cuda attention still
    runs the slow path inside it.

    Returns the resolved modules for the record (``None`` on a model with
    no Mamba mixer); raises naming the mixer class(es), the wrappers on the
    fallback and the missing package(s) on a model with a cuda execution
    device that would run the fallback."""
    got = resolved_mamba_kernels(model)
    if got is None:
        return None
    slow = {n: m for n, m in got.items() if not _is_from(m, MAMBA_KERNEL_WRAPPERS[n])}
    if not slow:
        return got
    missing = sorted({MAMBA_KERNEL_WRAPPERS[n] for n in slow})
    mixers = " / ".join(c.__name__ for c in mamba_mixer_classes(model))
    detail = f"{mixers}: " + ", ".join(f"{n} -> {m}" for n, m in sorted(slow.items()))
    devices = model_devices(model)
    if any(d.type == "cuda" for d in devices):
        raise RuntimeError(
            f"{type(model).__name__} would run its Mamba mixers on transformers' "
            f"pure-torch fallback ({detail}); "
            f"the fused kernels from {missing} are REQUIRED for Mamba-hybrid work "
            "(user directive 2026-09-09) and are not importable in this environment. "
            "Sync the venv with `uv sync --frozen --extra cu128 --extra mamba` "
            "(sbatch: S2_UV_EXTRAS=mamba / the config's env_extras) and rerun; "
            "never proceed on the fallback.")
    # a module with no weights and no placement has nowhere to be but cpu
    where = " / ".join(sorted({d.type for d in devices})) or "cpu"
    warnings.warn(
        f"{type(model).__name__} is on {where}, where the fused kernels "
        f"from {missing} do not exist (CUDA-only packages), so its Mamba "
        f"mixers run on transformers' pure-torch fallback "
        f"({detail}). That path is SLOW next to the fused kernels, and it is "
        f"the expected path on {where}; the refusal that protects GPU "
        "jobs applies to models executing on cuda only.",
        RuntimeWarning, stacklevel=3)
    return got


def default_device() -> str:
    """The device :func:`load_model_and_tokenizer` picks when none is given:
    ``cuda`` when available, else ``mps`` (Apple Silicon's GPU — the macOS
    arm64 torch wheel on the PyTorch CPU index, which ``uv sync --extra
    cpu`` installs there, is built with it), else ``cpu``. CUDA first keeps
    every cluster and pod load byte-identical to before MPS was an option."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device) -> torch.dtype:
    """The dtype :func:`load_model_and_tokenizer` picks for ``device`` when
    none is given.

    * ``"cuda"``: bfloat16 — the historical rule, kept VERBATIM as a string
      comparison on the bare ``"cuda"`` spelling (an indexed ``"cuda:0"``
      got float32 before and still does; changing that is not this
      function's business).
    * ``mps``: bfloat16 when torch itself says the Metal backend supports it,
      else float16. torch's own gate is macOS 14 or newer —
      ``torch/_dynamo/device_interface.py`` ``MpsInterface.is_bf16_supported``
      returns ``torch.backends.mps.is_macos_or_newer(14, 0)``, and
      ``torch/amp/autocast_mode.py`` disables bf16 MPS autocast below 14 for
      the same reason — so that is the question asked here (torch 2.11.0).
      A torch that cannot answer it (no MPS backend built) gets float16, the
      dtype every MPS build runs. Not float32: a 4B model's weights are 16 GB
      in fp32 against 8 GB in a 16-bit dtype, and Apple Silicon has no
      bf16-only hazard the way pre-Ampere CUDA does.
    * anything else (``cpu`` included): float32, as before.
    """
    if device == "cuda":
        return torch.bfloat16
    if torch.device(device).type == "mps":
        try:
            bf16_ok = bool(torch.backends.mps.is_macos_or_newer(14, 0))
        except Exception:  # a torch without the MPS backend cannot say
            bf16_ok = False
        return torch.bfloat16 if bf16_ok else torch.float16
    return torch.float32


def load_model_and_tokenizer(
    name: str,
    device: str | None = None,
    dtype: torch.dtype | str | None = None,
    revision: str | None = None,
    leading_special_tokens: str | None = None,
    **kwargs,
):
    """`revision` pins a specific snapshot — needed when a shared HF
    cache's refs/main points at a snapshot that was never downloaded
    (offline resolution then fails even though older snapshots are
    complete; bitten by gemma-4-31B-it on cluster A, 2026-08-11).
    ``leading_special_tokens`` is forwarded to :func:`load_tokenizer`
    (None = untouched).

    ``device`` defaults through :func:`default_device` (cuda > mps > cpu)
    and ``dtype`` through :func:`default_dtype` (bf16 on cuda; bf16 or fp16
    on mps per torch's own macOS gate; fp32 elsewhere). An explicit
    argument always wins over either default. Loading on ``mps`` or ``cpu``
    a model with Mamba mixers (Nemotron's Mamba2; Jamba's Mamba-1) without
    the CUDA-only fused kernels warns once and runs the pure-torch fallback;
    on ``cuda`` it still refuses (:func:`assert_fused_mamba_kernels`).
    Qwen3.5, a gated-deltanet hybrid, is not subject to either and loads
    silently.

    A local model path in placeholder form (``$SHARED/...``, ``$HF_HOME/...``)
    is expanded through ``sisteer.site`` first; an unset placeholder refuses
    by name rather than being sent to the hub as a repo id."""
    if isinstance(name, str) and "$" in name:
        from sisteer import site as _site
        name = str(_site.expand_path(name, key="model"))
    _enforce_llama31_mirror(name)
    configure_fast_sdpa()
    if device is None:
        device = default_device()
    if dtype is None:
        dtype = default_dtype(device)
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if revision is not None:
        kwargs["revision"] = revision
    tok = load_tokenizer(name, revision=revision,
                         leading_special_tokens=leading_special_tokens)
    cls = auto_model_class(name, revision=revision)
    model = cls.from_pretrained(name, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    # Mamba models (Nemotron's Mamba2 mixers, Jamba's Mamba-1): on cuda the
    # fused kernels or a refusal -- never the pure-torch fallback; on mps /
    # cpu, where those kernels do not exist, ONE loud warning and the
    # fallback (assert_fused_mamba_kernels; no-op on every other
    # architecture, Qwen3.5's gated-deltanet hybrid included).
    assert_fused_mamba_kernels(model)
    check_chat_template(tok)
    # Both are no-ops on every architecture whose cache allocates and whose
    # generation config already lists the template terminator; on NemotronH
    # (transformers 5.14.1) each one fires and says so: the cache step
    # registers the "mlp" placeholder layer 5.14.1 lacks, so generate()
    # decodes with its cache rather than being pinned cache-free.
    prepare_kv_cache(model)
    pin_generation_stop_ids(model, tok)

    return model, tok


def apply_chat_template_folding(tok, messages: list[dict],
                                add_generation_prompt: bool = True,
                                extra: dict | None = None,
                                use_marker: bool = True) -> str:
    """Render ``messages`` through the tokenizer's chat template, folding a
    leading ``system`` message into the first ``user`` turn for templates that
    reject the system role (Gemma / older Mistral — Stolfo's own convention,
    their prompts carry the instruction inline).

    ``add_generation_prompt=False`` renders through the LAST message's turn
    WITHOUT the trailing assistant header — used to locate the response
    boundary (the token index where generation begins).

    Caller guarantees ``tok.chat_template`` is set; the DIFFERENT plain-string
    fallbacks (build_chat_prompt vs jailbreak._render_chat, one of which the
    gpt2 smoke path depends on) stay with each caller. This factors ONLY the
    shared template + system-fold branch. ``add_generation_prompt`` is
    forwarded (default True, the historical behavior); hooks.
    response_header_len diffs the True/False renders to count the
    trailing assistant-header tokens."""
    # MERGE SEAM (stage3 -> twosweep). Two kwarg sources, and which applies
    # matters more than it looks:
    #   the MARKER (tok._sisteer_template_kwargs, set by load_tokenizer) pins
    #     enable_thinking=False on reasoning-capable templates;
    #   ``extra`` is an EXPLICIT per-call override from the caller.
    # build_chat passes use_marker=False because a caller that asks for the
    # TEMPLATE DEFAULT (extra=None) must actually get it: apply_think_mode
    # builds isolated/as_shipped from the default (thinking-on) render, so
    # silently applying the marker there would make every "thinking" render
    # thinking-OFF and the think modes would degrade to no-ops that still
    # looked like they ran. Every other caller keeps the historical behaviour.
    kwargs = dict(getattr(tok, "_sisteer_template_kwargs", {}) or {}) if use_marker else {}
    if extra:
        kwargs.update(extra)        # an explicit override beats the marker
    try:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt, **kwargs
        )
    except Exception:
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
            rest = [dict(m) for m in messages[1:]]
            if rest and rest[0].get("role") == "user":
                rest[0]["content"] = f"{system.strip()}\n\n{rest[0]['content']}"
                return tok.apply_chat_template(
                    rest, tokenize=False,
                    add_generation_prompt=add_generation_prompt, **kwargs
                )
        raise


def build_chat_prompt(tok, user: str, system: str | None = None) -> str:
    """Render a (system, user) pair to a generation-ready prompt string.

    Falls back to plain concatenation for models without a chat template
    (e.g. gpt2 in smoke tests).
    """
    if getattr(tok, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return apply_chat_template_folding(tok, messages)
    prefix = f"{system.strip()}\n\n" if system else ""
    return f"{prefix}{user.strip()}\n"
