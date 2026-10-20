"""Genuine-adapter LoRA export and the content-addressed adapter bank.

Three things live here:

1. **The ZERO (B=0) no-op adapter** — :func:`export_zero_adapter`,
   BUILT NOW. It is the path-matched base control for the LoRA path
   (the ``none__lora0`` leg): vLLM serves it as a real
   per-request adapter, so the leg exercises the genuine multi-LoRA
   machinery while ``ΔW = B @ A`` is exactly zero (B = 0 guarantees it;
   A is zeroed too, belt and suspenders). The §10 gate's LoRA no-op
   certificate exact-checks it against the undefended base.

2. **The genuine-adapter export for hand-rolled hook arms** —
   :func:`export_genuine_adapter`, implemented for the LANDED
   weight-level representations (:data:`GENUINE_EXPORT_KINDS`): ``lora``
   (extraction/objects.RankOneLoRAObject — rank-1 on one layer's
   attention o_proj), ``lora_all`` (extraction/multilayer.
   MultiLayerLoRAObject — one rank-r o_proj pair per targeted layer),
   ``circuit_breaker`` (extraction/circuit_breaker.CircuitBreakerObject —
   rank-r over the seven q/k/v/o/gate/up/down projections of layers
   0..max(target)), and ``lora_subtraction`` (extraction/multilayer.
   LoRASubtractionObject — the task-vector DIFFERENCE of two same-shape
   ``lora_all`` fits, exported by RANK STACKING:
   :func:`_export_lora_subtraction`). All of them are Python forward
   hooks with NO vLLM export path of their own; the export converts
   each landed torch-blob artifact into a real PEFT adapter dir that
   vLLM's native multi-LoRA serves. Where the export refuses, the arm
   falls to the lens hook; a kind that serves on NEITHER committed
   vLLM path is a §10 gate FAILURE that blocks the audit (never an HF
   fallback), never silently skipped.

3. **The content-addressed adapter bank** — :func:`export_or_reuse` and
   :func:`export_or_reuse_zero`. One
   directory per intervention hash (:func:`adapter_dir_for`) under a
   caller-owned root, written ATOMICALLY (a fresh export into a private
   tmp dir, ``adapter_digests.json`` beside the two PEFT files, fsync,
   one ``rename`` onto the target), and REUSED only after the existing
   entry re-hashes to its own record AND a fresh export of the same
   input bytes is byte-identical to it — a stale or tampered entry is a
   loud refusal naming the dir, never silently re-exported over, never
   trusted by name. The zero adapter is banked the same way under a key
   over (base, revision, o_proj module shapes, rank).

THE GENUINE-EXPORT CONTRACT (binding on the implementation below):

- input: the landed object artifact (the exact hashed bytes) + its
  resolved spec; output: a PEFT-format adapter dir (adapter_config
  + safetensors weights) whose served behavior the serving gate must
  prove EXACTLY equal (output-token equality, greedy, in the audit's
  determinism mode) to the HF fp32 reference serving the original hook.
- deterministic: same artifact bytes -> byte-identical adapter tensors.
- the adapter dir is content-addressed by the source object's
  ``intervention_hash`` (the same intervention, a different serving
  path), and the vLLM adapter NAME is the cell id.
- any rank/module mismatch between the hook's math and what PEFT can
  express is a LOUD refusal -> the arm stays on the lens path (a kind
  serving on neither committed path fails the serving gate); never an
  approximate export.
- native multi-LoRA has NO position mask — vLLM applies a loaded adapter
  to EVERY token it touches — so only an ALL-POSITION fit is exportable.
  A position-masked fit (hooks.py's response/prompt/front/last modes)
  served all-position would be a DIFFERENT intervention under the fit's
  hash, so it is a LOUD refusal naming the honest alternatives (a masked
  LoRA extension landed deliberately, or the recorded per-cell HF
  exception), never an all-position approximation.

- the DECODER PREFIX and the ATTENTION SITE in every emitted module name
  are the base model's OWN layout, both read off the real module tree
  (``models.decoder_layers_path`` + ``models.attention_oproj_site`` on a
  meta-device skeleton — :func:`resolve_decoder_layout`): ``model.layers``
  + ``self_attn.o_proj`` on llama, ``model.language_model.layers`` +
  ``self_attn.o_proj`` on the VLM wrappers gemma-4 / GLM-4.6V whose
  decoder sits under ``language_model``, ``model.layers`` +
  ``mixer.o_proj`` on NemotronH (Nemotron-3-Nano, whose 4 attention
  blocks sit among 38 Mamba2/MLP blocks) — never a second hardcoded
  string and never a model_type table: PEFT names an adapted module by
  its HF attribute path, and the pinned vLLM 0.25.1 resolves that path
  through the model's own ``hf_to_vllm_mapper``. A prefix outside
  ``models._LAYER_PATHS``, a site outside ``models._OPROJ_SITES``, a
  layer that carries no attention o_proj on the resolved tree, or a base
  whose config is not locally resolvable, is a loud refusal — the export
  never emits a module name it cannot tie to a layout.

- every emitted module name is SPELLED IN THE SERVED CHECKPOINT'S
  NAMESPACE, not the harness class's (the qwen certification canary,
  job 489694 at ceba8874). The skeleton proves the LAYOUT (which
  layers carry an attention o_proj, at which site, with which shapes),
  but the class the harness instantiates is not always the class the
  checkpoint is: ``AutoModelForCausalLM`` builds the text-only
  ``Qwen3_5ForCausalLM`` (``model.layers.N.self_attn.o_proj``)
  for a ``Qwen3_5ForConditionalGeneration`` checkpoint whose weights are
  ``model.language_model.layers.N.self_attn.o_proj.weight``, and vLLM's
  LoRA loader maps PEFT names with the served model's
  ``hf_to_vllm_mapper`` (``lora/worker_manager.py``), which is written
  against the CHECKPOINT's names (``"model.language_model." ->
  "language_model.model."``, no rule for a bare ``model.layers.``) — so
  the skeleton spelling loaded but bound to no execution module, and
  ``activate_adapter`` would have served base weights under the cell's
  name. The exact-name binding check at readiness caught it and stays
  the backstop. So, after the layout is resolved, every o_proj layer's
  name is read off the checkpoint itself (:func:`resolve_module_namespace`:
  the ``model.safetensors.index.json`` weight map, else every
  ``*.safetensors`` header — headers only, never tensors): the skeleton's
  name when the checkpoint has exactly that module, else the ONE module
  whose name ends in ``.layers.{l}.{site}``; zero or several candidates,
  a checkpoint whose layers resolve under two prefixes, a prefix outside
  ``models._LAYER_PATHS``, or a snapshot with no weights manifest at all,
  refuse — never a guess and never the skeleton kept as a fallback. The
  mapping is recorded beside the adapter (``adapter_digests.json``
  ``module_namespace``). Where the two spellings agree (llama, gemma-4,
  GLM-4.6V) the adapter bytes are exactly what they were.

Adapter dirs are written in the standard PEFT layout so ``vllm serve
--enable-lora --lora-modules <cell_id>=<dir>`` picks them up unchanged.
"""

from __future__ import annotations

import errno
import functools
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import (Any, Dict, FrozenSet, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

import torch

from . import run_identity as _ri
from . import serving_attestation as _sa

#: The vLLM adapter name of the zero no-op adapter == the control-leg
#: cell id (one vmodel per served leg maps 1:1 onto adapter names).
ZERO_ADAPTER_NAME = "none__lora0"
#: the two PEFT file names, bound to the ONE spelling in ``serving_attestation``
ADAPTER_CONFIG = _sa.ADAPTER_CONFIG_FILE
ADAPTER_WEIGHTS = _sa.ADAPTER_WEIGHTS_FILE


class LoraExportError(RuntimeError):
    """A LoRA-export refusal — loud, never an approximate adapter."""


#: The Llama-family decoder prefix — the layout every export was
#: verified against on hardware (the §10 gate's LoRA legs on Llama-3.1):
#: ``model.layers.{i}.self_attn.o_proj`` etc. One of ``models._LAYER_PATHS``.
LLAMA_DECODER_PREFIX = "model.layers"

#: The attention-site half of that verified layout (``models._OPROJ_SITES``
#: spelled as a dotted path): the site every caller-stated prefix implies
#: unless the caller states another, and the one whose absence from the
#: adapter config keeps every llama/GLM export byte-identical.
DEFAULT_OPROJ_SITE = "self_attn.o_proj"


class DecoderLayout(NamedTuple):
    """A base model's decoder module layout, read off its OWN module tree:
    ``prefix`` — the decoder-block ModuleList's attribute path
    (``models.decoder_layers_path``); ``oproj_site`` — the attention
    output projection's path inside a block (``models.attention_oproj_
    site``: ``self_attn.o_proj`` / ``mixer.o_proj``); ``oproj_layers`` —
    the block indices that carry one (``models.oproj_layers``; every layer
    on a dense stack, a strict subset on a hybrid). A module name is
    ``f"{prefix}.{layer}.{oproj_site}"`` and only for ``layer in
    oproj_layers``."""

    prefix: str
    oproj_site: str
    oproj_layers: Tuple[int, ...]


def resolve_decoder_layout(base_model: str,
                           revision: Optional[str] = None) -> DecoderLayout:
    """The base model's :class:`DecoderLayout`, derived from its OWN module
    tree — a weight-free skeleton on the meta device built from the
    LOCALLY CACHED config (``local_files_only``: a base that is being
    served has its snapshot cached, and this must never reach the
    network). ``model.layers`` + ``self_attn.o_proj`` for llama;
    ``model.language_model.layers`` + ``self_attn.o_proj`` for the VLM
    wrappers (GLM-4.6V-Flash's ``Glm4vForConditionalGeneration``, gemma-4),
    whose decoder sits under ``language_model``; ``model.layers`` +
    ``mixer.o_proj`` on layers (12, 17, 24, 32) for NemotronH
    (Nemotron-3-Nano-4B, whose other 38 blocks are Mamba2/MLP mixers with
    no attention o_proj — ``mixer.out_proj`` / ``mixer.down_proj`` are
    different attribute names and never match).

    WHY THE HF PATH IS THE RIGHT THING TO EMIT, read out of the pinned
    vLLM 0.25.1 source (offline verification; the live load is the GPU
    probe): ``lora/worker_manager.py`` hands the model's own
    ``hf_to_vllm_mapper`` (``get_unstacked_mapper()``) to
    ``LoRAModel.from_local_checkpoint`` as ``weights_mapper``, and
    ``lora/utils.parse_fine_tuned_lora_name`` strips ``base_model.model.``,
    applies that mapper, and re-prefixes — so a PEFT tensor named
    ``base_model.model.model.language_model.layers.N.self_attn.o_proj.
    lora_A.weight`` resolves, on glm4_1v (whose mapper carries
    ``"model.language_model." -> "language_model.model."``), to the vLLM
    module ``language_model.model.layers.N.self_attn.o_proj`` — exactly
    where ``Glm4ForCausalLM.model.layers[N].self_attn.o_proj`` lives
    (glm4.py; ``Glm4vForConditionalGeneration`` is ``SupportsLoRA`` and its
    ``get_mm_mapping`` names ``language_model.model`` as the LoRA-able
    language prefix). Emitting vLLM's INTERNAL path instead would be
    double-mapped and wrong.

    THE NEMOTRON NAMING DECISION (two HF spellings exist; the tree-derived
    one is emitted). transformers 5.14.1's ``NemotronHForCausalLM`` live
    tree is ``model.layers.N.mixer.o_proj`` (``base_model_prefix ==
    "model"``, no ``backbone`` attribute; the checkpoint's ``backbone.``
    keys are renamed on load by its ``WeightRenaming backbone.->model.``
    conversion mapping), so the skeleton yields prefix ``model.layers`` —
    and that is also what PEFT itself would write for an adapter trained
    on this transformers. On the vLLM side (nemotron_h.py at the 0.25.1
    tag): ``hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix=
    {"backbone": "model"}, orig_to_new_substr={"A_log": "A", "embeddings":
    "embed_tokens"})``, ``packed_modules_mapping = {"qkv_proj": [q, k, v]}``
    (o_proj standalone); the model is ``self.model = NemotronHModel(prefix=
    "model")`` -> ``make_layers(prefix="model.layers")`` -> attention layer
    ``self.mixer = NemotronHAttention(prefix=f"{prefix}.mixer")`` ->
    ``RowParallelLinear(prefix=f"{prefix}.o_proj")``, so the exact vLLM
    module name the LoRA manager matches (``model_manager.activate_adapter``
    iterates ``self.modules`` by name) is ``model.layers.N.mixer.o_proj``.
    ``WeightsMapper._map_name`` replaces ``orig_to_new_prefix`` keys only
    when ``key.startswith(prefix)``: the tree-derived name passes through
    UNCHANGED to that exact module (no ``backbone`` prefix, no substr hit),
    and the checkpoint spelling ``backbone.layers.N.mixer.o_proj`` would
    ALSO land there (prefix-mapped). Both resolve, so no checkpoint-prefix
    override is kept: one seam, the tree. GPU-verify (d) is the live load.

    Refuses loudly when the config is not locally resolvable, no known
    layout matches, no block carries an attention o_proj, or the
    attention blocks disagree on the site — the export never infers a
    module layout."""
    cfg, skeleton = _decoder_skeleton(base_model, revision)
    return _layout_from_skeleton(base_model, cfg, skeleton)


def _cannot_derive(base_model: str, cfg: Any, e: BaseException
                   ) -> LoraExportError:
    """The one wording for 'the skeleton did not yield a layout'."""
    from ...models import _LAYER_PATHS

    return LoraExportError(
        f"cannot derive the decoder layout of {base_model!r} "
        f"(model_type {getattr(cfg, 'model_type', None)!r}) from a "
        f"meta-device skeleton: {e}. Known layouts: {_LAYER_PATHS}; "
        "refusing to guess a module prefix")


def _decoder_skeleton(base_model: str, revision: Optional[str] = None
                      ) -> Tuple[Any, torch.nn.Module]:
    """``(config, skeleton)`` for ``base_model``: its LOCALLY cached config
    (``local_files_only`` — this never reaches the network) and a
    weight-free meta-device instance of the auto-class that maps it. The
    ONE skeleton build behind :func:`resolve_decoder_layout` (prefix,
    site, o_proj layers) and :func:`oproj_target_modules` (the same plus
    each o_proj's ``(in_features, out_features)`` for the zero adapter).
    Refuses when the config is not cached or no auto-class maps it."""
    from ...models import _LAYER_PATHS, auto_model_class

    if not isinstance(base_model, str) or not base_model:
        raise LoraExportError(
            f"base_model must be a non-empty model id, got {base_model!r}")
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model, revision=revision,
                                         local_files_only=True)
    except Exception as e:  # transformers raises several types here
        raise LoraExportError(
            f"cannot resolve the decoder layout of {base_model!r}"
            f"{' @ ' + revision if revision else ''}: its config is not in "
            f"the local HF cache ({type(e).__name__}: {e}). The export "
            "derives every module name from the base model's OWN module "
            "tree and never infers a layout — cache the pinned snapshot "
            "(a served base always is), or pass decoder_prefix= (one of "
            f"{_LAYER_PATHS}) from a model you hold") from e
    try:
        cls = auto_model_class(base_model, revision=revision, config=cfg)
        with torch.device("meta"):
            skeleton = cls.from_config(cfg)
    except (ValueError, RuntimeError, TypeError, AttributeError) as e:
        raise _cannot_derive(base_model, cfg, e) from e
    return cfg, skeleton


def _layout_from_skeleton(base_model: str, cfg: Any,
                          skeleton: torch.nn.Module) -> DecoderLayout:
    """The :class:`DecoderLayout` read off a :func:`_decoder_skeleton`:
    the decoder prefix (``models.decoder_layers_path``), the blocks that
    carry an attention o_proj (``models.oproj_layers``) and their ONE
    shared site (``models.attention_oproj_site``). Refuses a stack with no
    o_proj at all or whose attention blocks disagree on the site."""
    from ...models import (_OPROJ_SITES, attention_oproj_site,
                           decoder_layers_path, get_decoder_layers,
                           oproj_layers)

    try:
        prefix = decoder_layers_path(skeleton)
        layers = tuple(oproj_layers(skeleton))
        blocks = get_decoder_layers(skeleton)
        sites = sorted({attention_oproj_site(blocks[i]) for i in layers})
    except (ValueError, RuntimeError, TypeError, AttributeError) as e:
        raise _cannot_derive(base_model, cfg, e) from e
    if not layers:
        raise LoraExportError(
            f"{base_model!r} (model_type {getattr(cfg, 'model_type', None)!r})"
            " has NO decoder block with an attention output projection "
            f"(sites tried: {['.'.join(s) for s in _OPROJ_SITES]}) — no "
            "o_proj LoRA can be named on it; refusing")
    if len(sites) != 1:
        raise LoraExportError(
            f"{base_model!r}: its attention blocks disagree on the o_proj "
            f"site ({sites}) — one adapter cannot name two layouts; refusing")
    return DecoderLayout(prefix=prefix, oproj_site=sites[0],
                         oproj_layers=layers)


def _oproj_shapes(base_model: str, layout: DecoderLayout,
                  skeleton: torch.nn.Module) -> Dict[int, Tuple[int, int]]:
    """Each o_proj layer of ``layout`` -> ITS module's ``(in_features,
    out_features)``, read off the meta skeleton PER MODULE. One model's
    o_proj does NOT share one shape across layers: gemma-4-E4B's 7
    global-attention layers (5, 11, ..., 41; ``global_head_dim`` 512) have
    ``in_features`` 4096 where its 35 local layers (``head_dim`` 256) have
    2048, all projecting to the same 2560 — so every per-layer question
    about a shape is answered here, by the module that owns it, never by
    another layer. Never guesses: a module exposing neither
    ``in_features``/``out_features`` nor a 2-D ``weight`` refuses."""
    from ...models import attention_oproj, get_decoder_layers

    blocks = get_decoder_layers(skeleton)
    out: Dict[int, Tuple[int, int]] = {}
    for layer in layout.oproj_layers:
        mod = attention_oproj(blocks[layer])
        in_f = getattr(mod, "in_features", None)
        out_f = getattr(mod, "out_features", None)
        if not (isinstance(in_f, int) and isinstance(out_f, int)):
            w = getattr(mod, "weight", None)
            if isinstance(w, torch.Tensor) and w.dim() == 2:
                out_f, in_f = int(w.shape[0]), int(w.shape[1])
        if not (isinstance(in_f, int) and not isinstance(in_f, bool)
                and in_f > 0 and isinstance(out_f, int)
                and not isinstance(out_f, bool) and out_f > 0):
            raise LoraExportError(
                f"{base_model!r}: layer {layer}'s {layout.oproj_site} "
                f"({type(mod).__name__}) exposes no (in_features, "
                "out_features) — refusing to guess the module's shape")
        out[int(layer)] = (int(in_f), int(out_f))
    return out


def _derived_layout(base_model: str, revision: Optional[str] = None
                    ) -> Tuple[DecoderLayout, Dict[int, Tuple[int, int]]]:
    """``(layout, {layer: (in_features, out_features)})`` off ONE skeleton
    build — what :func:`export_genuine_adapter` derives when the caller
    states no layout, so a fit layer's factors are held to the shape of
    the very module their PEFT name points at (:func:`_check_layer_dims`)."""
    cfg, skeleton = _decoder_skeleton(base_model, revision)
    layout = _layout_from_skeleton(base_model, cfg, skeleton)
    return layout, _oproj_shapes(base_model, layout, skeleton)


def oproj_target_modules(base_model: str, revision: Optional[str] = None
                         ) -> Dict[str, Tuple[int, int]]:
    """Every attention-o_proj module of ``base_model`` ->
    ``(in_features, out_features)``, named ``f"{prefix}.{layer}.{site}"``
    off the base's own layout (:func:`resolve_decoder_layout`) with the
    shapes read off the SAME meta skeleton PER MODULE (:func:`_oproj_shapes`
    — gemma-4-E4B's global-attention layers are twice as wide on the input
    side as its local ones) — the ``target_modules`` mapping
    :func:`export_zero_adapter` takes, so the banked zero adapter
    (:func:`export_or_reuse_zero`) covers every o_proj layer of the base
    (a superset of every genuine o_proj export's module set) and its
    config's short names are ``o_proj`` and nothing else — which is
    exactly what the lora-native compose-time preflight admits (unique
    ``target_modules`` contents == {"o_proj"}). Never
    guesses a shape: a module exposing neither ``in_features``/
    ``out_features`` nor a 2-D ``weight`` refuses.

    The names here are the SKELETON's spelling (a layout-and-shape oracle
    off the cached config alone). What the bank serves is spelled in the
    checkpoint's namespace instead — :func:`checkpoint_target_modules`,
    which reads the snapshot's weights manifest; the two agree on
    llama / gemma-4 / GLM and differ on qwen3.5 / NemotronH."""
    layout, shapes = _derived_layout(base_model, revision)
    return {f"{layout.prefix}.{layer}.{layout.oproj_site}": shapes[layer]
            for layer in layout.oproj_layers}


def resolve_decoder_prefix(base_model: str,
                           revision: Optional[str] = None) -> str:
    """The prefix half of :func:`resolve_decoder_layout` (kept as the
    prefix-only entry point the GLM export landed with)."""
    return resolve_decoder_layout(base_model, revision).prefix


def _validated_oproj_site(site: Any) -> str:
    """A caller-stated attention site must be one of the KNOWN sites
    (``models._OPROJ_SITES``, dotted) — the same discipline as
    :func:`_validated_decoder_prefix`."""
    from ...models import _OPROJ_SITES

    known = tuple(".".join(s) for s in _OPROJ_SITES)
    if not isinstance(site, str) or site not in known:
        raise LoraExportError(
            f"oproj_site={site!r} is not a known attention o_proj site "
            f"({known}, models._OPROJ_SITES) — module names must come from "
            "the base model's own module tree (models.attention_oproj_site "
            "/ resolve_decoder_layout), never a free-form string")
    return site


def _validated_decoder_prefix(prefix: Any) -> str:
    """A caller-stated prefix must be one of the KNOWN layouts
    (``models._LAYER_PATHS``) — a free string would let an export emit
    module names tied to no real module tree."""
    from ...models import _LAYER_PATHS

    if not isinstance(prefix, str) or prefix not in _LAYER_PATHS:
        raise LoraExportError(
            f"decoder_prefix={prefix!r} is not a known decoder layout "
            f"({_LAYER_PATHS}, models._LAYER_PATHS) — module names must "
            "come from the base model's own module tree "
            "(models.decoder_layers_path / resolve_decoder_prefix), never "
            "a free-form string")
    return prefix


# ==========================================================================
# the checkpoint's module namespace
# ==========================================================================

#: a sharded checkpoint's weight manifest (``weight_map``: tensor name ->
#: shard file) — the authoritative list of the checkpoint's tensor names
#: when present
WEIGHT_INDEX_FILE = "model.safetensors.index.json"
#: where :func:`checkpoint_weight_names` read the names from: the index's
#: ``weight_map`` keys, or the header of every ``*.safetensors`` file
MODULE_NAMESPACE_SOURCES = ("index", "safetensors-header")
#: the largest safetensors header this module parses — real headers run
#: from a few KB (263 tensors) to a few MB; a declared length past this is
#: not a header, and the read stops before touching it
_MAX_SAFETENSORS_HEADER = 256 * 1024 * 1024


class ModuleNamespace(NamedTuple):
    """How one export spelled its module names. ``skeleton`` is the decoder
    prefix the harness's meta skeleton yields (:func:`resolve_decoder_layout`
    — ``model.layers`` on qwen3.5 and NemotronH); ``checkpoint`` is the
    prefix the served checkpoint's OWN weight names carry, the one every
    emitted PEFT name is spelled in (``model.language_model.layers`` on
    qwen3.5, ``backbone.layers`` on NemotronH); ``source`` is where those
    names were read (:data:`MODULE_NAMESPACE_SOURCES`); ``n_layers`` how
    many o_proj layers were bound; ``snapshot`` the snapshot dir they were
    read from. Recorded beside the adapter in ``adapter_digests.json`` as
    PROVENANCE, never identity — the PEFT bytes carry the names, and an
    export whose two prefixes agree writes exactly the bytes it wrote
    before the rule."""

    skeleton: str
    checkpoint: str
    source: str
    n_layers: int
    snapshot: str

    @property
    def renamed(self) -> bool:
        """Whether the checkpoint spells the decoder differently from the
        skeleton (qwen3.5, NemotronH); ``False`` where they agree (llama,
        gemma-4, GLM-4.6V)."""
        return self.skeleton != self.checkpoint


def checkpoint_snapshot_dir(base_model: str, revision: Optional[str] = None,
                            snapshot_dir: Optional[Path] = None) -> Path:
    """The directory holding ``base_model``'s checkpoint files — the one
    the served vLLM loads its weights from. A ``snapshot_dir`` stated by a
    caller that holds it (the certification sbatch's ``SNAP``) wins and
    must be a directory; a ``base_model`` that IS a local directory (a
    saved tiny base, an ad-hoc local path) is its own snapshot; otherwise
    it is the pinned snapshot in the LOCAL HF cache, found exactly the way
    the skeleton's config was — ``huggingface_hub.try_to_load_from_cache``
    on ``config.json`` at ``revision``, a purely local lookup that honors
    ``HF_HOME`` and never reaches the network (``HF_HUB_OFFLINE`` or not);
    the snapshot is the PARENT of that cached ``config.json``, taken
    without resolving the file itself (in the cache it is a symlink into
    ``blobs/``, whose parent is not the snapshot). Refuses when nothing is
    cached: a base that is being served has its snapshot cached, so a miss
    is a real inconsistency, never a reason to keep the skeleton's
    spelling."""
    if snapshot_dir is not None:
        snap = Path(snapshot_dir)
        if not snap.is_dir():
            raise LoraExportError(
                f"snapshot_dir={str(snap)!r} is not a directory — the "
                "checkpoint's module namespace is read off its snapshot; "
                "refusing")
        return snap.resolve()
    if not isinstance(base_model, str) or not base_model:
        raise LoraExportError(
            f"base_model must be a non-empty model id, got {base_model!r}")
    local = Path(base_model)
    if local.is_dir():
        return local.resolve()
    if revision is not None and (not isinstance(revision, str)
                                 or not revision):
        raise LoraExportError(
            f"revision must be None or a non-empty string, got {revision!r}")
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo_id=base_model, filename="config.json",
                                     revision=revision)
    except Exception as e:  # a malformed repo id raises several types
        raise LoraExportError(
            f"cannot look up {base_model!r}"
            f"{' @ ' + revision if revision else ''} in the local HF cache "
            f"({type(e).__name__}: {e}); refusing") from e
    if not isinstance(hit, str):
        raise LoraExportError(
            f"{base_model!r}{' @ ' + revision if revision else ''}: its "
            "snapshot is not in the local HF cache (no cached config.json "
            f"at that revision; HF_HOME={os.environ.get('HF_HOME')!r}), so "
            "the checkpoint's module namespace cannot be read — a served "
            "base always has its snapshot cached; pass snapshot_dir= from a "
            "snapshot you hold, never the skeleton's spelling")
    return Path(hit).parent.resolve()


def _safetensors_header_keys(path: Path) -> List[str]:
    """The tensor names one safetensors file carries, read from its HEADER
    only — the format is an 8-byte little-endian JSON length followed by
    that JSON, then the tensor bytes — so a 16 GB shard costs one small
    read and no tensor is ever touched. Refuses a file that is not shaped
    like that, naming it."""
    import struct

    p = Path(path)
    try:
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise LoraExportError(
                    f"{p}: not a safetensors file (no 8-byte header length)")
            (n,) = struct.unpack("<Q", raw)
            size = os.fstat(f.fileno()).st_size
            if n < 2 or n > size - 8 or n > _MAX_SAFETENSORS_HEADER:
                raise LoraExportError(
                    f"{p}: safetensors header length {n} is not plausible "
                    f"for a {size}-byte file; not reading it as a header")
            header = json.loads(f.read(n).decode("utf-8"))
    except OSError as e:
        raise LoraExportError(
            f"{p}: cannot read the safetensors header: {e}") from e
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LoraExportError(
            f"{p}: the safetensors header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise LoraExportError(f"{p}: the safetensors header is not a JSON object")
    return [k for k in header if k != "__metadata__"]


def checkpoint_weight_names(snapshot: Path) -> Tuple[FrozenSet[str], str]:
    """Every tensor name the checkpoint under ``snapshot`` carries, and where
    they were read (:data:`MODULE_NAMESPACE_SOURCES`): the ``weight_map``
    keys of :data:`WEIGHT_INDEX_FILE` when the snapshot has one (a sharded
    checkpoint — qwen3.5, GLM-4.6V: ``"index"``), else the header of every
    ``*.safetensors`` file in the snapshot (a single-file checkpoint —
    gemma-4-E4B, Nemotron-3-Nano: ``"safetensors-header"``; headers only).
    A snapshot with neither — a config-only cache entry, a ``.bin``
    checkpoint — is nothing this export can name modules against and
    refuses naming what IS there; the skeleton's spelling is never kept as
    a fallback."""
    snap = Path(snapshot)
    if not snap.is_dir():
        raise LoraExportError(
            f"{snap}: not a directory — no checkpoint to read module names "
            "from; refusing")
    index = snap / WEIGHT_INDEX_FILE
    if index.is_file():
        try:
            doc = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise LoraExportError(
                f"{index}: unreadable weight index: {e}; refusing") from e
        wm = doc.get("weight_map") if isinstance(doc, dict) else None
        if not isinstance(wm, dict) or not wm or not all(
                isinstance(k, str) and k for k in wm):
            raise LoraExportError(
                f"{index}: no non-empty weight_map of tensor names — not a "
                "safetensors index this export can read module names from; "
                "refusing")
        return frozenset(wm), "index"
    files = sorted(p for p in snap.iterdir()
                   if p.name.endswith(".safetensors") and p.is_file())
    if not files:
        present = sorted(p.name for p in snap.iterdir())
        raise LoraExportError(
            f"{snap}: neither {WEIGHT_INDEX_FILE} nor any *.safetensors file "
            f"(present: {present[:24]}{'…' if len(present) > 24 else ''}) — "
            "the checkpoint's module namespace cannot be read, so no PEFT "
            "name can be tied to the served weights; refusing (a "
            "config-only cache entry is not a checkpoint; a .bin checkpoint "
            "carries no header to read)")
    names: set = set()
    for f in files:
        keys = _safetensors_header_keys(f)
        if not keys:
            raise LoraExportError(
                f"{f}: safetensors header names no tensor; refusing")
        names.update(keys)
    return frozenset(names), "safetensors-header"


def checkpoint_module_names(layout: DecoderLayout, weight_names,
                            *, base_model: str, snapshot: Path, source: str,
                            ) -> Tuple[Dict[int, str], ModuleNamespace]:
    """Each o_proj layer of ``layout`` -> the module name the CHECKPOINT
    spells it under, decided per layer against the checkpoint's tensor
    names (``weight_names``; a tensor ``<module>.<param>`` names its module,
    so a module with ``weight`` + ``weight_scale`` counts once): (1) EXACT
    — the skeleton's own ``{prefix}.{l}.{site}`` is a module of the
    checkpoint: keep it (llama, gemma-4, GLM — the bytes stay what they
    were); (2) otherwise UNIQUE SUFFIX — exactly one checkpoint module ends
    in ``.{list}.{l}.{site}`` (``list`` the skeleton's decoder-list
    attribute, ``layers``): use it (qwen3.5: ``model.language_model.layers.
    3.self_attn.o_proj`` for the skeleton's ``model.layers.3…``; NemotronH:
    ``backbone.layers.12.mixer.o_proj`` for ``model.layers.12…``); (3) zero
    or several candidates refuse naming the layer, the site, the
    candidates and the checkpoint — never a guess (qwen3.5 also carries
    ``mtp.layers.0.self_attn.o_proj``, an MTP head that would tie with a
    layer-0 attention block if there were one: exactly the ambiguity this
    refuses). Then two whole-layout checks: every layer must resolve under
    ONE prefix (one adapter cannot name two layouts), and that prefix must
    be a known decoder layout (``models._LAYER_PATHS`` — a lone ``mtp.
    layers`` match is not a decoder). Returns the per-layer names and the
    :class:`ModuleNamespace` record."""
    from ...models import _LAYER_PATHS

    if source not in MODULE_NAMESPACE_SOURCES:
        raise LoraExportError(
            f"module-namespace source {source!r} is not one of "
            f"{MODULE_NAMESPACE_SOURCES}")
    modules = frozenset(k.rsplit(".", 1)[0] for k in weight_names
                        if isinstance(k, str) and "." in k)
    if not modules:
        raise LoraExportError(
            f"{base_model!r}: the checkpoint at {snapshot} ({source}) names "
            "no module at all; refusing")
    list_attr = layout.prefix.rsplit(".", 1)[-1]
    site = layout.oproj_site
    names: Dict[int, str] = {}
    heads: Dict[int, str] = {}
    for layer in layout.oproj_layers:
        skeleton = f"{layout.prefix}.{layer}.{site}"
        if skeleton in modules:
            name = skeleton
        else:
            tail = f".{list_attr}.{layer}.{site}"
            cands = sorted(m for m in modules if m.endswith(tail))
            if len(cands) != 1:
                what = ("no module" if not cands
                        else f"{len(cands)} modules {cands}")
                raise LoraExportError(
                    f"{base_model!r}: the checkpoint at {snapshot} ({source}) "
                    f"has {what} named like layer {layer}'s attention {site} — "
                    f"the skeleton says {skeleton!r}, which is not a module of "
                    f"the checkpoint, and {'nothing' if not cands else 'more than one module'} "
                    f"ends in {tail!r} — so no ONE PEFT name binds to that "
                    "layer on the served model; refusing to guess (the "
                    "harness class and the checkpoint disagree about the "
                    "decoder; the readiness exact-name binding check would "
                    "refuse the served adapter either way)")
            name = cands[0]
        names[layer] = name
        heads[layer] = name[: -len(f".{layer}.{site}")]
    distinct = sorted(set(heads.values()))
    if len(distinct) != 1:
        raise LoraExportError(
            f"{base_model!r}: the checkpoint at {snapshot} ({source}) spells "
            f"the attention layers under {len(distinct)} prefixes "
            f"({ {l: heads[l] for l in sorted(heads)} }) — one adapter cannot "
            "name two layouts; refusing")
    checkpoint = distinct[0]
    if checkpoint not in _LAYER_PATHS:
        raise LoraExportError(
            f"{base_model!r}: the checkpoint at {snapshot} ({source}) spells "
            f"the attention layers under {checkpoint!r}, which is not a "
            f"known decoder layout ({_LAYER_PATHS}, models._LAYER_PATHS) — "
            "refusing to name modules under a prefix no served decoder has")
    return names, ModuleNamespace(skeleton=layout.prefix, checkpoint=checkpoint,
                                  source=source, n_layers=len(names),
                                  snapshot=str(snapshot))


def resolve_module_namespace(base_model: str, revision: Optional[str],
                             layout: DecoderLayout, *,
                             snapshot_dir: Optional[Path] = None,
                             ) -> Tuple[Dict[int, str], ModuleNamespace]:
    """:func:`checkpoint_snapshot_dir` -> :func:`checkpoint_weight_names` ->
    :func:`checkpoint_module_names`: the checkpoint's spelling of every
    o_proj layer of ``layout``, plus the record. The ONE call every
    derived-layout export makes after the skeleton proved the layout."""
    snap = checkpoint_snapshot_dir(base_model, revision, snapshot_dir)
    weight_names, source = checkpoint_weight_names(snap)
    return checkpoint_module_names(layout, weight_names, base_model=base_model,
                                   snapshot=snap, source=source)


def checkpoint_target_modules(base_model: str, revision: Optional[str] = None,
                              *, snapshot_dir: Optional[Path] = None,
                              ) -> Tuple[Dict[str, Tuple[int, int]],
                                         ModuleNamespace]:
    """:func:`oproj_target_modules` spelled in the CHECKPOINT's namespace:
    every attention-o_proj module of ``base_model`` -> ``(in_features,
    out_features)`` (shapes off the meta skeleton, per module), each named
    as the served checkpoint names it (:func:`resolve_module_namespace`),
    plus the :class:`ModuleNamespace` record — what the banked zero
    adapter (:func:`export_or_reuse_zero`) is exported from, so its
    modules bind on the served model exactly like the genuine cells'."""
    layout, shapes = _derived_layout(base_model, revision)
    names, namespace = resolve_module_namespace(base_model, revision, layout,
                                                snapshot_dir=snapshot_dir)
    return ({names[layer]: shapes[layer] for layer in layout.oproj_layers},
            namespace)


def _peft_config_header(base_model: str) -> Dict[str, Any]:
    """The PEFT ``adapter_config.json`` keys every adapter this module writes
    shares (type, base, no dropout, no bias, ``task_type``); every writer
    dumps with ``sort_keys=True``, so where the header sits in the dict
    never reaches the bytes."""
    return {"peft_type": "LORA", "base_model_name_or_path": base_model,
            "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
            "task_type": "CAUSAL_LM"}


def export_zero_adapter(
    out_dir: Path,
    *,
    base_model: str,
    rank: int,
    target_modules: Mapping[str, Tuple[int, int]],
    lora_alpha: float = 1.0,
) -> Path:
    """Write the zero (B=0) PEFT adapter (the ``none__lora0`` leg).

    ``target_modules`` maps each adapted module name (e.g.
    ``model.layers.15.self_attn.q_proj``) to ``(in_features,
    out_features)`` — stated explicitly by the caller from the pinned
    base architecture; this module never guesses shapes. The names must
    carry the base's OWN decoder prefix (``models.decoder_layers_path`` /
    :func:`resolve_decoder_prefix`: ``model.language_model.layers...`` on
    the VLM wrappers such as GLM-4.6V, not llama's ``model.layers``)."""
    from safetensors.torch import save_file  # transformers-stack dep

    if not isinstance(rank, int) or rank < 1:
        raise LoraExportError(f"rank must be a positive int, got {rank!r}")
    if not target_modules:
        raise LoraExportError("target_modules must be a non-empty "
                              "{module: (in_features, out_features)} mapping")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for module, shape in sorted(target_modules.items()):
        if (not isinstance(module, str) or not module
                or len(shape) != 2 or not all(
                    isinstance(d, int) and d > 0 for d in shape)):
            raise LoraExportError(
                f"target_modules[{module!r}] must map a non-empty module "
                f"name to (in_features, out_features), got {shape!r}")
        in_f, out_f = shape
        # PEFT tensor layout: lora_A [r, in], lora_B [out, r]; ΔW = B @ A.
        tensors[f"base_model.model.{module}.lora_A.weight"] = torch.zeros(
            (rank, in_f), dtype=torch.float32)
        tensors[f"base_model.model.{module}.lora_B.weight"] = torch.zeros(
            (out_f, rank), dtype=torch.float32)

    config = {
        **_peft_config_header(base_model),
        "r": rank,
        "lora_alpha": float(lora_alpha),
        "target_modules": sorted(
            m.rsplit(".", 1)[-1] for m in target_modules),
        # provenance: this is the deliberate no-op control adapter
        "petri_zero_adapter": True,
    }
    (out_dir / ADAPTER_CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True))
    save_file(tensors, str(out_dir / ADAPTER_WEIGHTS))
    return out_dir


def verify_zero_adapter(adapter_dir: Path) -> None:
    """Loudly verify an on-disk adapter really is the zero no-op (every
    tensor exactly zero) — the gate calls this before certifying the
    ``none__lora0`` leg."""
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / ADAPTER_CONFIG
    if not cfg_path.is_file():
        raise LoraExportError(f"no {ADAPTER_CONFIG} in {adapter_dir}")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("petri_zero_adapter") is not True:
        raise LoraExportError(
            f"{adapter_dir} is not marked petri_zero_adapter — refusing to "
            "certify an arbitrary adapter as the no-op leg")
    tensors = load_file(str(adapter_dir / ADAPTER_WEIGHTS))
    if not tensors:
        raise LoraExportError(f"{adapter_dir}: empty adapter weights")
    for name, t in tensors.items():
        if not torch.all(t == 0):
            raise LoraExportError(
                f"{adapter_dir}: tensor {name} is non-zero — this adapter "
                "would NOT be a no-op; refusing")


#: the kinds :func:`export_genuine_adapter` converts — exactly the landed
#: weight-level LoRA representations (module docstring item 2). The
#: native enumeration predicate's servable set (petri/defense_cells.
#: NATIVE_LORA_SERVABLE_KINDS) is pinned EQUAL to this tuple by
#: tests/test_native_lora_constants_agree.py: adding a kind here is adding
#: it there.
GENUINE_EXPORT_KINDS = ("circuit_breaker", "lora", "lora_all",
                        "lora_subtraction",
                        "psr", "vector", "vector_all", "diffmean", "static_r",
                        # THE PAIRED WRITE (relu-differenced-native-qwen): the two-gate
                        # residual kind, exported by _residual_psr_paired_write
                        "psr_paired_write")
#: the RESIDUAL-SLOT kinds among them: exported by
#: :func:`_export_residual_slot` as a rank-1 ZERO o_proj LoRA on the base's
#: first o_proj layer (the ANCHOR — the carrier of vLLM's slot, id and per-token
#: routing) plus the payload beside it under ``native_residual.PAYLOAD_PREFIX``
#: and its ``petri_residual_slot`` config block. Pinned equal to
#: ``native_residual.RESIDUAL_SLOT_KINDS`` / ``defense_cells.RESIDUAL_SLOT_KINDS``
#: by test.
RESIDUAL_EXPORT_KINDS = ("psr", "vector", "vector_all", "diffmean", "static_r", "psr_paired_write")

#: q/k/v/o live under ``self_attn``; gate/up/down under ``mlp`` — mirrors
#: extraction/circuit_breaker._proj_module on the pinned Llama-family base.
_ATTN_PROJECTIONS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})

def _genuine_blob(kind: str, artifact_bytes: bytes) -> dict:
    """The kind's torch-blob artifact, loaded and kind-verified (the
    lens_adapters._torch_blob discipline, with this module's error)."""
    import io

    if not artifact_bytes:
        raise LoraExportError(f"kind {kind!r}: empty artifact bytes")
    try:
        blob = torch.load(io.BytesIO(bytes(artifact_bytes)),
                          map_location="cpu", weights_only=True)
    except Exception as e:  # torch raises several types here
        raise LoraExportError(
            f"kind {kind!r}: artifact bytes did not load as a torch "
            f"payload: {e}") from e
    if not isinstance(blob, dict) or "kind" not in blob:
        raise LoraExportError(
            f"kind {kind!r}: artifact is not a torch blob with a 'kind' "
            "field (the load_object contract)")
    if blob["kind"] != kind:
        raise LoraExportError(
            f"kind {kind!r}: artifact stores kind {blob['kind']!r} — this "
            "cell's bytes are not what the export was asked to convert; "
            "refusing")
    return blob


def _spec_check(kind: str, spec: Mapping[str, Any], key: str,
                expected: Any) -> None:
    """F9-style cross-check: when the resolved spec carries ``key`` it
    must agree with the artifact's own stored value."""
    if key in spec and spec[key] != expected:
        raise LoraExportError(
            f"kind {kind!r}: resolved_spec {key}={spec[key]!r} != artifact "
            f"{key}={expected!r} (F9-style cross-check) — the spec and the "
            "bytes disagree; refusing")


def _blob_int(kind: str, name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or value < minimum:
        raise LoraExportError(
            f"kind {kind!r}: {name} must be an int >= {minimum}, got "
            f"{value!r}")
    return value


def _saved_f32(kind: str, name: str, t: Any, *, dims: int) -> torch.Tensor:
    """A tensor exactly as the object's ``save`` writes it —
    ``detach().float().cpu()`` — validated, never converted (a dtype the
    save path cannot produce means the blob is not that object's)."""
    if not isinstance(t, torch.Tensor):
        raise LoraExportError(
            f"kind {kind!r}: {name} must be a torch.Tensor, got "
            f"{type(t).__name__}")
    if t.dtype != torch.float32:
        raise LoraExportError(
            f"kind {kind!r}: {name} is {t.dtype}, but the object's save "
            "path writes float32 — not this object's blob; refusing "
            "(a lossy dtype conversion here would be an approximate export)")
    if t.dim() != dims:
        raise LoraExportError(
            f"kind {kind!r}: {name} must be {dims}-D, got shape "
            f"{tuple(t.shape)}")
    if t.numel() == 0:
        raise LoraExportError(f"kind {kind!r}: {name} is empty")
    if not torch.isfinite(t).all():
        raise LoraExportError(
            f"kind {kind!r}: {name} carries non-finite values — refusing "
            "to export a corrupt tensor")
    return t.detach()


def _fit_positions(kind: str, positions: Any) -> str:
    """Gate the FIT LEG: native multi-LoRA has no position mask, so only
    an all-position fit is exportable (module docstring contract)."""
    from ...hooks import POSITION_MODES  # the authoritative vocabulary

    if not isinstance(positions, str):
        raise LoraExportError(
            f"kind {kind!r}: stored positions must be a string, got "
            f"{positions!r}")
    if positions == "all":
        return positions
    if positions == "response":
        raise LoraExportError(
            f"kind {kind!r}: the object was fit response-only "
            "(positions='response'), but vLLM's native multi-LoRA has no "
            "position mask — a loaded adapter applies to EVERY token, "
            "prompt and response alike, so serving this fit as a plain "
            "adapter would run a DIFFERENT intervention under the cell's "
            "intervention_hash (a plausible-looking wrong number). The arm "
            "is not broken — its honest serving path is the lens o_proj "
            "module site (lens_adapters OP_LORA_OPROJ, landed 2026-08-28 "
            "WITH its own gate assertions), which carries the response "
            "mask; never an all-position approximation here")
    if positions in POSITION_MODES:
        raise LoraExportError(
            f"kind {kind!r}: stored positions={positions!r} is a position-"
            "masked fit, which native multi-LoRA cannot express (it "
            "applies a loaded adapter to every token) — only an all-"
            "position fit is exportable; the honest alternative is the "
            "lens o_proj module site (which serves the 'response' mask; "
            "other masked modes remain unserved), never an all-position "
            "approximation")
    raise LoraExportError(
        f"kind {kind!r}: stored positions={positions!r} is not a known "
        f"position mode ({list(POSITION_MODES)}) — refusing to guess what "
        "the fit meant")


def _check_sidecar(kind: str, sidecar_bytes: bytes, fit_positions: str
                   ) -> None:
    """The optional fit-lane sidecar: it may record ``positions_at_fit``
    (and nothing else — the sidecar is hashed into the intervention
    identity, so unrecognized content must never be silently dropped);
    a disagreement with the blob's own positions is a refusal."""
    if not sidecar_bytes:
        return
    try:
        side = json.loads(bytes(sidecar_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LoraExportError(
            f"kind {kind!r}: unparseable JSON sidecar: {e}") from e
    if not isinstance(side, dict):
        raise LoraExportError(
            f"kind {kind!r}: sidecar is not a JSON object")
    unknown = set(side) - {"positions_at_fit"}
    if unknown:
        raise LoraExportError(
            f"kind {kind!r}: sidecar carries unrecognized keys "
            f"{sorted(unknown)} — the sidecar is hashed into the "
            "intervention identity, so content this export does not "
            "understand must never be silently dropped; refusing")
    if "positions_at_fit" in side and side["positions_at_fit"] \
            != fit_positions:
        raise LoraExportError(
            f"kind {kind!r}: sidecar positions_at_fit="
            f"{side['positions_at_fit']!r} != the blob's positions="
            f"{fit_positions!r} — the fit's own records disagree about "
            "what was trained; refusing to pick one")


def _export_rank1_lora(resolved_spec: Mapping[str, Any], blob: dict, *,
                       decoder_prefix: str,
                       oproj_site: str = DEFAULT_OPROJ_SITE,
                       oproj_layers: Optional[Tuple[int, ...]] = None,
                       oproj_shapes: Optional[Mapping[int, Tuple[int, int]]] = None,
                       ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """RankOneLoRAObject -> PEFT tensors + config fields.

    ``decoder_prefix`` / ``oproj_site`` are the base's own layout
    (:class:`DecoderLayout`); ``oproj_layers``, when the caller resolved
    the layout from the tree, is the set of blocks that carry an attention
    o_proj — a fit ``layer`` outside it (a NemotronH Mamba2/MLP block)
    names a module the base does not have and refuses — and
    ``oproj_shapes``, read off the SAME skeleton, is each such module's
    ``(in_features, out_features)``: ``A``'s and ``B``'s lengths must equal
    the named module's, refused by name otherwise (:func:`_check_layer_dims`;
    see the note at the check for why the export, not the vLLM load, is
    where a mis-sized factor is caught).

    THE CONVERSION, derived from the authoritative hook body
    (extraction/objects.RankOneLoRAObject.apply):

        coef  = x @ A                    # A [d_in] dots the o_proj INPUT
        delta = coef.unsqueeze(-1) * B   # B [d_out] spans the OUTPUT
        output += scale * delta          # selector scale; 1.0 when served

    i.e. ``delta_j = (sum_i x_i A_i) * B_j``, so the weight edit is
    ``dW[j, i] = B_j * A_i`` — the outer product B ⊗ A on o_proj's
    OUTPUT: exactly a rank-1 LoRA. PEFT's layout (export_zero_adapter):
    ``lora_A.weight [r, in]``, ``lora_B.weight [out, r]``,
    ``delta = (x @ lora_A.T) @ lora_B.T * (lora_alpha / r)`` and
    ``dW = lora_B @ lora_A``. With ``lora_A = A[None, :]`` ([1, d_in]),
    ``lora_B = B[:, None]`` ([d_out, 1]) and ``lora_alpha = r = 1``:
    ``lora_B @ lora_A = B ⊗ A``, and the served delta performs the SAME
    reduction over d_in followed by the SAME per-element outer product —
    bit-identical to the hook on CPU fp32 (pinned by the export test).
    The orientation is fixed by the HOOK BODY, not by shapes: A is the
    input-side factor (it dots x) and B the output-side one (it spans
    the added delta). On square-o_proj models (Llama: d_in == d_out)
    shapes alone could not catch a silent transpose, so this mapping is
    asserted against the class's own math, and the shapes are asserted
    below rather than assumed."""
    kind = "lora"
    required = {"kind", "A", "B", "layer"}
    # RankOneLoRAObject.save writes BOTH fit-leg keys, ``positions`` and
    # ``response_header_len`` (the fit's mask leg is part of the
    # artifact's identity, 2026-08-21). ``positions`` is nonetheless
    # REQUIRED — through its OWN refusal below, which names the fit-leg
    # schema instead of the generic "lacks" message — and
    # ``response_header_len`` is validated when present, never guessed.
    # Both sit in ``optional`` only so the key-set check here does not
    # pre-empt those two dedicated paths.
    optional = {"positions", "response_header_len"}
    missing = required - set(blob)
    if missing:
        raise LoraExportError(
            f"kind {kind!r}: artifact lacks {sorted(missing)} — not a "
            "RankOneLoRAObject.save blob")
    unknown = set(blob) - required - optional
    if unknown:
        raise LoraExportError(
            f"kind {kind!r}: artifact carries unexpected keys "
            f"{sorted(unknown)} — the representation drifted past what "
            "this export was verified against; refusing to guess")

    layer = _blob_int(kind, "layer", blob["layer"], minimum=0)
    A, B = blob["A"], blob["B"]
    for name, t in (("A", A), ("B", B)):
        if not isinstance(t, torch.Tensor):
            raise LoraExportError(
                f"kind {kind!r}: {name} must be a torch.Tensor, got "
                f"{type(t).__name__}")
    # RANK: assert rank-1 rather than trusting the kind's implicit. A
    # future rank-r fit would land [r, d_in] / [d_out, r] matrices under
    # the same kind; silently declaring r=1 over them would be exactly
    # the plausible-looking wrong adapter this export must never write.
    if (A.dim() == 2 and B.dim() == 2 and A.shape[0] == B.shape[1]
            and A.shape[0] > 1):
        raise LoraExportError(
            f"kind {kind!r}: A {tuple(A.shape)} / B {tuple(B.shape)} is a "
            f"rank-{A.shape[0]} representation under the rank-1 kind — "
            "refusing to declare r=1 over it; extend the export "
            "deliberately WITH the new representation's contract")
    if A.dim() != 1 or B.dim() != 1:
        raise LoraExportError(
            f"kind {kind!r}: A must be [d_in] and B [d_out] (the "
            "RankOneLoRAObject.save layout), got A "
            f"{tuple(A.shape)} / B {tuple(B.shape)}")
    A = _saved_f32(kind, "A", A, dims=1)
    B = _saved_f32(kind, "B", B, dims=1)

    # THE FIT LEG, and it is REQUIRED — absence is a refusal, not "all".
    # Reading a missing key as all-position is a guess wearing the fit's
    # hash: before the fit code recorded this field, a response-fit LoRA
    # re-hydrated as all-position and would have run the all-token
    # method under a decode-only label. Every object the current fit
    # code produces carries the key, and nothing has been fit yet, so
    # nothing legitimate is stranded.
    if "positions" not in blob:
        raise LoraExportError(
            f"kind {kind!r}: the artifact records no 'positions' field, so "
            "which intervention it IS cannot be determined — it predates "
            "the fit-leg schema. Exporting it as all-position would be a "
            "guess served under the fit's hash, and a decode-only fit "
            "served at every position is a DIFFERENT intervention. "
            "Re-save it with the current fit lane, or serve it through "
            "the reference path; this exporter will not infer the leg")
    positions = _fit_positions(kind, blob["positions"])
    # ``RankOneLoRAObject.save`` writes ``response_header_len`` on EVERY blob
    # — ``None`` for an all-position fit (the class's own default; only a
    # position-masked fit records a length). Validated as an int >= 0 when
    # not None, exactly as the multilayer and subtraction exports do; the
    # earlier ``"response_header_len" in blob`` test refused the class's own
    # ``None`` and aborted the native standup of an admitted all-position
    # ``lora_r1`` cell.
    if blob.get("response_header_len") is not None:
        _blob_int(kind, "response_header_len", blob["response_header_len"],
                  minimum=0)
    _spec_check(kind, resolved_spec, "layer", layer)
    _spec_check(kind, resolved_spec, "positions", positions)

    if oproj_layers is not None and layer not in oproj_layers:
        raise LoraExportError(
            f"kind {kind!r}: the fit's layer {layer} carries no attention "
            f"o_proj on this base (attention layers: {list(oproj_layers)}) "
            "— a NemotronH Mamba2/MLP block, or a Qwen3.5 linear-attention "
            "block; the module name would point at nothing. Refusing")
    # With the tree, the one layer's [d_in] / [d_out] are held to ITS
    # module's (in_features, out_features) — the same per-module check the
    # multi-layer kinds apply (one layer has no cross-layer premise, so
    # ``shared`` is never consulted). This check was dropped once on
    # the claim that a fit which does not fit its module "fails at the vLLM
    # load, loudly". It does NOT: vLLM 0.25.1's
    # ``BaseLinearLayerWithLoRA.set_lora`` (vllm/lora/layers/base_linear.py)
    # zeroes its full-size lora_a/lora_b stacks and copies each incoming
    # factor into a slice sized from THAT tensor, so an A shorter than
    # in_features (a 2048-vector on a gemma-4 global layer whose o_proj
    # takes 4096) is silently zero-padded and serves a DIFFERENT
    # intervention under the fit's hash — where the reference hook's
    # ``x @ A`` would have failed its matmul. The attestation cannot catch
    # it either: worker_reports fingerprints the CPU-side factors exactly as
    # exported, and execution validation checks module NAMES, not
    # dimensions. So the export is the one place the factors meet the
    # module they name. Without the tree (a caller-stated decoder_prefix,
    # no skeleton) there is no module to hold them to and the call is a
    # no-op for one layer; every production caller (export_or_reuse)
    # derives the tree.
    _check_layer_dims(kind, layer, (int(A.shape[0]), int(B.shape[0])),
                      oproj_shapes=oproj_shapes, shared=None,
                      module=f"{decoder_prefix}.{layer}.{oproj_site}")
    # the decoder prefix + attention site are the base's OWN layout
    # (module docstring): model.layers + self_attn.o_proj on llama,
    # model.language_model.layers + self_attn.o_proj on the VLM wrappers,
    # model.layers + mixer.o_proj on NemotronH — resolved by the caller
    # off the module tree, never hardcoded here.
    module = f"{decoder_prefix}.{layer}.{oproj_site}"
    tensors = {
        f"base_model.model.{module}.lora_A.weight":
            A.unsqueeze(0).contiguous(),           # [1, d_in]
        f"base_model.model.{module}.lora_B.weight":
            B.unsqueeze(1).contiguous(),           # [d_out, 1]
    }
    config = {
        "r": 1,
        "lora_alpha": 1.0,   # scaling = alpha/r = 1 — the hook adds delta
                             # unscaled at selector scale 1.0
        # the PEFT short name: "o_proj" under EVERY known site (a NemotronH
        # Mamba mixer's projection is out_proj, so the short name cannot
        # collide with it there either)
        "target_modules": [module.rsplit(".", 1)[-1]],   # ["o_proj"]
        "layers_to_transform": [layer],
        # honest provenance (contract: never petri_zero_adapter)
        "petri_source_kind": kind,
        "petri_source_layers": [layer],
        "petri_fit_positions": positions,
    }
    if oproj_site != DEFAULT_OPROJ_SITE:
        # a non-default attention site is recorded so the served adapter
        # names can be audited against the tree they were derived from
        # (petri_decoder_prefix carries the prefix half); the default site
        # adds nothing, keeping every llama/GLM export's config bytes
        # identical.
        config["petri_oproj_site"] = oproj_site
    return tensors, config

#: "the blob carries no such key" — distinct from an explicit ``None``,
#: which the declared-rank check must still refuse as a non-int.
_UNSET = object()


def _layer_set(kind: str, mapping: Any, *, name: str = "A") -> List[int]:
    """The sorted layer indices one factor map (``A``/``B``/``A_pos``/...)
    covers: a non-empty mapping whose keys are distinct ints >= 0 (the
    save writes ``str(l)``; an int-keyed legacy dict is accepted).
    Anything else refuses naming the map."""
    if not isinstance(mapping, Mapping) or not mapping:
        raise LoraExportError(
            f"kind {kind!r}: {name} must be a non-empty mapping "
            f"layer -> tensor, got {type(mapping).__name__}")
    try:
        layers = sorted(int(k) for k in mapping)
    except (TypeError, ValueError) as e:
        raise LoraExportError(
            f"kind {kind!r}: {name}'s keys must be layer indices, got "
            f"{sorted(map(str, mapping))}") from e
    if any(l < 0 for l in layers) or len(set(layers)) != len(layers):
        raise LoraExportError(
            f"kind {kind!r}: layer indices must be distinct ints >= 0, got "
            f"{layers}")
    return layers


def _map_key(mapping: Mapping[Any, Any], layer: int) -> Any:
    """The key ``layer`` is stored under: the save writes ``str(l)``; an
    int-keyed legacy dict is read as it is."""
    return str(layer) if str(layer) in mapping else layer


def _check_declared_layers(kind: str, blob: Mapping[str, Any],
                           layers: List[int], what: str) -> None:
    """A blob's optional ``layers`` metadata must equal the factor maps'
    key set — the artifact must agree with itself about what it targets."""
    if "layers" not in blob:
        return
    declared = blob["layers"]
    if not isinstance(declared, (list, tuple)) or \
            sorted(_blob_int(kind, "layers entry", l, minimum=0)
                   for l in declared) != layers:
        raise LoraExportError(
            f"kind {kind!r}: the blob's `layers` metadata {declared!r} "
            f"disagrees with the {what} key set {layers} — the artifact "
            "does not agree with itself about what it targets; refusing")


def _rank_of(kind: str, first_tensor: Any, declared: Any, *,
             name: str) -> int:
    """The rank FROM THE TENSORS (1 for a legacy 1-D ``[d_in]`` factor,
    ``shape[1]`` for ``[d_in, r]``), cross-checked against the blob's
    declared ``rank`` when it carries one (``declared is _UNSET`` = no
    key): a declared rank the tensors contradict refuses rather than
    letting the config mis-declare ``r``."""
    if not isinstance(first_tensor, torch.Tensor):
        raise LoraExportError(
            f"kind {kind!r}: {name} must be a torch.Tensor, got "
            f"{type(first_tensor).__name__}")
    inferred = 1 if first_tensor.dim() == 1 else (
        int(first_tensor.shape[1]) if first_tensor.dim() == 2 else 0)
    if inferred < 1:
        raise LoraExportError(
            f"kind {kind!r}: {name} has shape "
            f"{tuple(first_tensor.shape)}; expected [d_in] (rank 1) or "
            "[d_in, r] — not this object's blob")
    if declared is _UNSET:
        return inferred
    rank = _blob_int(kind, "rank", declared, minimum=1)
    if rank != inferred:
        raise LoraExportError(
            f"kind {kind!r}: the blob declares rank {rank} but its "
            f"tensors are rank-{inferred} shaped "
            f"({tuple(first_tensor.shape)}) — refusing a rank the config "
            "would mis-declare")
    return rank


def _peft_pair(kind: str, layer: int, a_raw: Any, b_raw: Any, rank: int, *,
               a_name: str = "A", b_name: str = "B",
               ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """One layer's saved ``(A, B)`` factors -> PEFT-oriented
    ``(lora_A [r, d_in], lora_B [d_out, r], (d_in, d_out))``, validated as
    the object's save writes them (fp32, finite, the declared rank):
    rank 1 unsqueezes the legacy 1-D vectors, rank r transposes A. The
    orientation is fixed by the hook body (A dots the o_proj INPUT, B
    spans its OUTPUT), never by the shapes."""
    if rank == 1:
        a = _saved_f32(kind, f"{a_name}[{layer}]", a_raw, dims=1)
        b = _saved_f32(kind, f"{b_name}[{layer}]", b_raw, dims=1)
        return (a.unsqueeze(0), b.unsqueeze(1),          # [1,in] / [out,1]
                (int(a.shape[0]), int(b.shape[0])))
    a = _saved_f32(kind, f"{a_name}[{layer}]", a_raw, dims=2)
    b = _saved_f32(kind, f"{b_name}[{layer}]", b_raw, dims=2)
    if a.shape[1] != rank or b.shape[1] != rank:
        raise LoraExportError(
            f"kind {kind!r}: layer {layer} has {a_name} {tuple(a.shape)} / "
            f"{b_name} {tuple(b.shape)} but the object is rank {rank} "
            f"({a_name} must be [d_in, r], {b_name} [d_out, r]) — refusing "
            "a rank the config would mis-declare")
    return a.t(), b, (int(a.shape[0]), int(b.shape[0]))   # [r,in] / [out,r]


def _check_layer_dims(kind: str, layer: int, d: Tuple[int, int], *,
                      oproj_shapes: Optional[Mapping[int, Tuple[int, int]]],
                      shared: Optional[Tuple[int, int]],
                      module: Optional[str] = None,
                      ) -> Optional[Tuple[int, int]]:
    """Hold one layer's saved ``(d_in, d_out)`` to the module its PEFT name
    points at. WITH the tree (``oproj_shapes``: the derived-layout path,
    :func:`_derived_layout`) the yardstick is that layer's OWN o_proj
    ``(in_features, out_features)`` — PEFT is per module, and one model's
    o_proj does NOT share one shape across layers (gemma-4-E4B: 2048 in on
    the 35 local-attention layers, 4096 on the 7 global ones, 2560 out on
    all 42 — the real ``base__constv2ctl2-full__lora_r4_ce`` object carries
    exactly that mix), so a layer is refused only when it disagrees with
    ITS module, never with a neighbour; the refusal names the layer, both
    shapes and (``module``, the PEFT path the caller is about to emit) the
    module they were held to. Every kind on the derived path applies it —
    the rank-1 kind included, because vLLM's loader does NOT refuse a
    mis-sized factor: ``set_lora`` copies each factor into a slice sized
    from the incoming tensor, zero-padding the rest (0.25.1, see
    :func:`_export_rank1_lora`). WITHOUT the tree (a caller-stated
    ``decoder_prefix``, no skeleton) there is no module to hold it to, and
    the one check available is the historical cross-layer heuristic — a
    layer that disagrees with an earlier one is called a corrupt blob. That
    heuristic is FALSE on a mixed-width stack, so such a base must go
    through the derived path (which every production caller —
    ``export_or_reuse`` — takes). Returns the running ``shared`` shape for
    the no-tree path (unused with the tree)."""
    named = f" ({module})" if module else ""
    if oproj_shapes is not None:
        want = oproj_shapes.get(layer)
        if want is None:
            raise LoraExportError(
                f"kind {kind!r}: layer {layer} carries no attention o_proj "
                f"on this base (attention layers: {sorted(oproj_shapes)}) — "
                f"the module name{named} would point at nothing; refusing")
        if tuple(want) != tuple(d):
            raise LoraExportError(
                f"kind {kind!r}: layer {layer} has (d_in, d_out)={tuple(d)} "
                f"but the base's own o_proj at that layer is (in_features, "
                f"out_features)={tuple(want)} — the factors do not fit the "
                f"module they name{named}: corrupt blob, or a fit from "
                "another base. Refusing here because the vLLM load would "
                "not: set_lora zero-pads an undersized factor silently and "
                "would serve a different intervention under the fit's hash")
        return shared
    if shared is None:
        return d
    if d != shared:
        raise LoraExportError(
            f"kind {kind!r}: layer {layer} has (d_in, d_out)={d} but an "
            f"earlier layer had {shared} — with no module tree to hold each "
            "layer to (caller-stated decoder_prefix), layers are expected to "
            "share one o_proj shape; corrupt blob, or a mixed-width base "
            "(gemma-4) that must be exported through the derived layout")
    return shared


def _refuse_layers_outside(kind: str, layers: List[int],
                           oproj_layers: Optional[Tuple[int, ...]]) -> None:
    """With the layout derived off the tree, a fit layer that carries no
    attention o_proj names a module the base does not have."""
    if oproj_layers is None:
        return
    outside = [l for l in layers if l not in oproj_layers]
    if outside:
        raise LoraExportError(
            f"kind {kind!r}: fit layers {outside} carry no attention "
            f"o_proj on this base (attention layers: "
            f"{list(oproj_layers)}) — NemotronH Mamba2/MLP blocks or "
            "Qwen3.5 linear-attention blocks; the module names would "
            "point at nothing. Refusing")


# ==========================================================================
# the GENERAL module-set blob shape (D_contract §2/§3)
# ==========================================================================
#
# A ``lora_all`` / ``lora_subtraction`` blob whose object adapts anything
# but the paper's ``("o_proj",)`` carries a ``targets`` block and factor
# maps keyed ``"<layer>.<module>"`` (D_contract §2):
#
#   "lora_modules": [sorted real HF short names], "lora_set_id": "mlp",
#   "per_expert_rank": {"shared_expert": r, "routed_expert": r_e} | None,
#   "init_scheme": "zero" | "random_lecun_1e-3",
#   "targets": {module: {"site", "layers", "rank", "kind"[, "num_experts"]}},
#   "A": {"<layer>.<module>": T}, "B": {"<layer>.<module>": T}
#
# A linear factor is A ``[d_in, r]`` / B ``[d_out, r]`` (delta = (x @ A)
# @ B.T, scaling exactly 1); a routed-expert factor is the JOINT 3-D form
# the engine consumes — A_gu ``[E, r_e, d]`` / B_gu ``[E, 2I, r_e]`` under
# ``"<layer>.experts.gate_up_proj"`` and A_d ``[E, r_e, I]`` / B_d
# ``[E, d, r_e]`` under ``"<layer>.experts.down_proj"``. A blob WITHOUT
# ``targets`` is the legacy o_proj shape and exports exactly as before.
#
# The PEFT layout (D_contract §3): a linear module's ``lora_A [r, d_in]``
# (= A.T) / ``lora_B [d_out, r]`` (= B) under
# ``base_model.model.{prefix}.{layer}.{site}``; the routed experts as PEFT's
# ``ParamWrapper`` writes them — TWO pairs per MoE block, flattened
# ``lora_A [(r_e·E), in]`` = ``A3.reshape(E*r_e, in)`` and ``lora_B [out,
# (r_e·E)]`` = ``B3.permute(1, 2, 0).reshape(out, r_e*E)``, which is exactly
# what vLLM 0.25.1 undoes at load (``vllm/lora/model_manager.py:857-880``:
# ``lora_a.reshape(E, -1, in)``; ``lora_b.reshape(out, -1, E).permute(2, 0,
# 1)``) and what PEFT 0.20.0's ``ParamWrapper.get_delta_weight`` reads
# (``peft/tuners/lora/layer.py`` ~L2418-2431: ``weight_A.reshape(E, -1,
# in)``, ``weight_B.reshape(out, -1, E)``, einsum ``"o r e, e r i -> e o i"``
# for a 3-D parameter, whose in/out were SWAPPED at ``update_layer``
# because HF stores the experts ``[E, out, in]``).

#: PEFT's ``ParamWrapper`` NESTS when two parameters of one module are
#: targeted through ``target_parameters``: the target_parameters walk in
#: ``peft/tuners/tuners_utils.py`` (0.20.0, ~L1141-1160) visits
#: ``named_parameters(recurse=False)`` in REGISTRATION order and re-fetches
#: the target through ``_get_submodules`` each time, so the FIRST parameter
#: HF registers (``gate_up_proj`` — the fused-experts modules register
#: gate_up_proj, then down_proj) becomes the INNER wrapper, ``<experts>.base_layer.lora_A/B``, and the
#: SECOND (``down_proj``) the OUTER ``<experts>.lora_A/B``. vLLM 0.25.1
#: hard-codes the same reading — "experts.base_layer is the gate_up_proj
#: and experts is the down_proj" (``vllm/lora/model_manager.py:835-843``;
#: ``lora_model.py:216-218`` admits the ``base_layer`` key) — and ASSERTS
#: both halves at load (``model_manager.py:843``), so an adapter that
#: adapts one half ships a ZERO companion for the other.
EXPERTS_PEFT_SUFFIX: Dict[str, str] = {_sa.EXPERTS_PROJECTIONS[0]: _sa.EXPERTS_BASE_LAYER_SUFFIX,
                                       _sa.EXPERTS_PROJECTIONS[1]: ""}
#: the vLLM ``--lora-target-modules`` suffix of the fused-MoE wrapper
#: (``FusedMoE3DWithLoRA`` replaces the ``experts`` module; ``lora/utils.py``
#: ``get_supported_lora_modules`` names every ``MoERunner`` by its suffix).
#: NEVER a PEFT ``target_modules`` entry (HF's experts are parameters).
EXPERTS_TARGET_SUFFIX = _sa.EXPERTS_LEAF
#: a PEFT module path that IS a fused-experts wrapper: ``...mlp.experts``
#: (the outer, down_proj half) or ``...mlp.experts.base_layer`` (the inner,
#: gate_up half), whatever the FFN attribute is called
_EXPERTS_MODULE_RE = re.compile(r"^(?P<path>.+\.experts)(?P<inner>\.base_layer)?$")
#: the two factor names of a general blob's routed-expert targets
_EXPERT_PARAMS = _sa.EXPERTS_PROJECTIONS


class _GeneralShape(NamedTuple):
    """A general-shape blob's metadata, validated (:func:`_general_shape`).
    ``targets`` is the blob's own ``targets`` block; ``keys`` every
    ``(layer, module)`` the factor maps must carry, in (layer, module)
    order; ``layers`` the sorted union; ``routed_rank`` the routed
    experts' ``r_e`` (``None`` when the set reaches no routed expert)."""

    modules: Tuple[str, ...]
    set_id: str
    per_expert_rank: Optional[Dict[str, int]]
    init_scheme: Optional[str]
    targets: Dict[str, Dict[str, Any]]
    keys: Tuple[Tuple[int, str], ...]
    layers: Tuple[int, ...]
    routed_rank: Optional[int]

    @property
    def experts_modules(self) -> Tuple[str, ...]:
        return tuple(m for m in self.targets if self.targets[m]["kind"] == "experts")


def _general_key(kind: str, key: Any, *, name: str) -> Tuple[int, str]:
    """``"<layer>.<module>"`` -> ``(layer, module)``; anything else refuses
    naming the map (a legacy ``str(layer)`` key has no module half)."""
    if not isinstance(key, str) or "." not in key:
        raise LoraExportError(
            f"kind {kind!r}: {name} key {key!r} is not '<layer>.<module>' — a "
            "general-shape blob keys its factors by layer AND module")
    layer_s, module = key.split(".", 1)
    if not layer_s.isdigit() or not module:
        raise LoraExportError(
            f"kind {kind!r}: {name} key {key!r} is not '<layer>.<module>' with a "
            "non-negative integer layer")
    return int(layer_s), module


def _general_shape(kind: str, blob: Mapping[str, Any], *,
                   factor_maps: Tuple[str, ...]) -> Optional[_GeneralShape]:
    """``None`` for the legacy o_proj shape (no ``targets`` key); otherwise
    the validated :class:`_GeneralShape`. Every metadata rule of D_contract
    §2 is a refusal here: ``lora_modules`` a sorted list from the
    vocabulary (``lora_modules.normalise_modules``), ``lora_set_id`` its
    id, ``per_expert_rank`` None or the two-key map with ``shared_expert
    == rank``, ``init_scheme`` (when present) one of ``INIT_SCHEMES``,
    each target's ``site``/``layers``/``rank``/``kind`` well-formed and
    consistent with its module (a linear target's rank is the object's;
    every routed-expert target shares ONE ``r_e``, equal to the map's when
    the map is present), the targets consistent with the module list
    (``experts.gate_up_proj`` needs gate AND up, ``experts.down_proj``
    needs down), every factor map keyed by EXACTLY the targets' ``(layer,
    module)`` pairs, and the blob's ``layers`` (when present) the sorted
    union. ``rank`` is REQUIRED on a general-shape blob."""
    from ...extraction.lora_modules import (EXPERT_TARGETS, EXPERTS_DOWN,
                                            EXPERTS_GATE_UP, INIT_SCHEMES,
                                            KNOWN_MODULES, LoRAModuleError,
                                            normalise_modules, set_id_for)

    if "targets" not in blob:
        stray = sorted({"lora_modules", "lora_set_id", "per_expert_rank"} & set(blob))
        if stray:
            raise LoraExportError(
                f"kind {kind!r}: the blob carries the general-shape keys {stray} "
                "but no 'targets' block — neither the legacy o_proj shape nor the "
                "general one; refusing to guess which modules it adapts")
        return None
    if "rank" not in blob:
        raise LoraExportError(
            f"kind {kind!r}: a general-shape blob (it carries 'targets') must "
            "declare its rank; refusing")
    rank = _blob_int(kind, "rank", blob["rank"], minimum=1)
    mods_raw = blob.get("lora_modules")
    if not isinstance(mods_raw, (list, tuple)) or not mods_raw or not all(
            isinstance(m, str) and m for m in mods_raw):
        raise LoraExportError(
            f"kind {kind!r}: a general-shape blob must carry lora_modules as a "
            f"non-empty list of module names, got {mods_raw!r}")
    try:
        modules = normalise_modules(list(mods_raw))
    except LoRAModuleError as e:
        raise LoraExportError(f"kind {kind!r}: lora_modules refused — {e}") from e
    if list(mods_raw) != list(modules):
        raise LoraExportError(
            f"kind {kind!r}: lora_modules={list(mods_raw)} is not the sorted, "
            f"de-duplicated list {list(modules)} the blob contract writes")
    set_id = set_id_for(modules)
    if "lora_set_id" in blob and blob["lora_set_id"] != set_id:
        raise LoraExportError(
            f"kind {kind!r}: lora_set_id={blob['lora_set_id']!r} does not name "
            f"the module list {list(modules)} (its id is {set_id!r}); refusing")
    per = blob.get("per_expert_rank")
    if per is not None:
        if (not isinstance(per, Mapping)
                or set(per) != {"shared_expert", "routed_expert"}):
            raise LoraExportError(
                f"kind {kind!r}: per_expert_rank={per!r} is not None or the map "
                "{'shared_expert': r, 'routed_expert': r_e}; refusing")
        per = {k: _blob_int(kind, f"per_expert_rank[{k}]", per[k], minimum=1)
               for k in ("shared_expert", "routed_expert")}
        if per["shared_expert"] != rank:
            raise LoraExportError(
                f"kind {kind!r}: per_expert_rank.shared_expert="
                f"{per['shared_expert']} != the object's rank {rank} — the shared "
                "expert carries the full rank (FD 6); refusing")
    init_scheme = blob.get("init_scheme")
    if init_scheme is not None and init_scheme not in INIT_SCHEMES:
        raise LoraExportError(
            f"kind {kind!r}: init_scheme={init_scheme!r} is not one of "
            f"{list(INIT_SCHEMES)}; refusing")

    targets = blob["targets"]
    if not isinstance(targets, Mapping) or not targets:
        raise LoraExportError(
            f"kind {kind!r}: targets must be a non-empty mapping module -> "
            f"{{site, layers, rank, kind}}, got {type(targets).__name__}")
    allowed_modules = set(KNOWN_MODULES) | set(EXPERT_TARGETS)
    out_targets: Dict[str, Dict[str, Any]] = {}
    routed_rank: Optional[int] = None
    for module in sorted(targets, key=str):
        rec = targets[module]
        if module not in allowed_modules:
            raise LoraExportError(
                f"kind {kind!r}: targets names module {module!r}, outside the "
                f"vocabulary {sorted(allowed_modules)}; refusing")
        if not isinstance(rec, Mapping):
            raise LoraExportError(
                f"kind {kind!r}: targets[{module!r}] must be a mapping, got "
                f"{type(rec).__name__}")
        required = {"site", "layers", "rank", "kind"}
        extra = set(rec) - required - {"num_experts"}
        missing = required - set(rec)
        if missing or extra:
            raise LoraExportError(
                f"kind {kind!r}: targets[{module!r}] "
                f"{'lacks ' + str(sorted(missing)) if missing else ''}"
                f"{' and ' if missing and extra else ''}"
                f"{'carries unknown keys ' + str(sorted(extra)) if extra else ''}; "
                "refusing")
        is_experts = module in EXPERT_TARGETS
        want_kind = "experts" if is_experts else "linear"
        if rec["kind"] != want_kind:
            raise LoraExportError(
                f"kind {kind!r}: targets[{module!r}].kind={rec['kind']!r}, but "
                f"module {module!r} is {want_kind!r}; refusing")
        site = rec["site"]
        param = module.rsplit(".", 1)[-1]   # o_proj / gate_up_proj / ...
        if (not isinstance(site, str) or "." not in site
                or site.rsplit(".", 1)[-1] != param
                or (is_experts and f".{EXPERTS_TARGET_SUFFIX}." not in f".{site}")):
            raise LoraExportError(
                f"kind {kind!r}: targets[{module!r}].site={site!r} is not a dotted "
                f"path inside the decoder block ending in {param!r}"
                f"{' under an experts module' if is_experts else ''}; refusing")
        layers = rec["layers"]
        if (not isinstance(layers, (list, tuple)) or not layers
                or any(isinstance(l, bool) or not isinstance(l, int) or l < 0
                       for l in layers)
                or list(layers) != sorted(set(layers))):
            raise LoraExportError(
                f"kind {kind!r}: targets[{module!r}].layers={layers!r} is not a "
                "non-empty sorted list of distinct block indices; refusing")
        t_rank = _blob_int(kind, f"targets[{module!r}].rank", rec["rank"], minimum=1)
        if is_experts:
            if "num_experts" not in rec:
                raise LoraExportError(
                    f"kind {kind!r}: targets[{module!r}] (routed experts) lacks "
                    "num_experts; refusing")
            n_e = _blob_int(kind, f"targets[{module!r}].num_experts",
                            rec["num_experts"], minimum=1)
            if routed_rank is None:
                routed_rank = t_rank
            elif routed_rank != t_rank:
                raise LoraExportError(
                    f"kind {kind!r}: the routed-expert targets disagree on r_e "
                    f"({routed_rank} vs {module!r}'s {t_rank}) — one uniform "
                    "per-expert rank (FD 6; the fused layout cannot express "
                    "another); refusing")
            if per is not None and per["routed_expert"] != t_rank:
                raise LoraExportError(
                    f"kind {kind!r}: targets[{module!r}].rank={t_rank} != "
                    f"per_expert_rank.routed_expert={per['routed_expert']}; refusing")
            need = (("gate_proj", "up_proj") if module == EXPERTS_GATE_UP
                    else ("down_proj",))
            if not all(m in modules for m in need):
                raise LoraExportError(
                    f"kind {kind!r}: targets names {module!r} but lora_modules="
                    f"{list(modules)} lacks {[m for m in need if m not in modules]}; "
                    "refusing")
            out_targets[module] = {"site": site, "layers": [int(l) for l in layers],
                                   "rank": t_rank, "kind": "experts",
                                   "num_experts": n_e}
        else:
            if "num_experts" in rec:
                raise LoraExportError(
                    f"kind {kind!r}: targets[{module!r}] (a linear module) carries "
                    "num_experts; refusing")
            if t_rank != rank:
                raise LoraExportError(
                    f"kind {kind!r}: targets[{module!r}].rank={t_rank} != the "
                    f"object's rank {rank} — a linear module carries the full "
                    "rank; refusing")
            if module not in modules:
                raise LoraExportError(
                    f"kind {kind!r}: targets names {module!r} but lora_modules="
                    f"{list(modules)} does not; refusing")
            out_targets[module] = {"site": site, "layers": [int(l) for l in layers],
                                   "rank": t_rank, "kind": "linear"}
    if EXPERTS_DOWN in out_targets and EXPERTS_GATE_UP in out_targets and (
            out_targets[EXPERTS_DOWN]["layers"] != out_targets[EXPERTS_GATE_UP]["layers"]
            or out_targets[EXPERTS_DOWN]["num_experts"]
            != out_targets[EXPERTS_GATE_UP]["num_experts"]):
        raise LoraExportError(
            f"kind {kind!r}: the two routed-expert targets cover different "
            "blocks or expert counts; corrupt blob")
    if per is not None and routed_rank is None:
        raise LoraExportError(
            f"kind {kind!r}: per_expert_rank={per} is stated but no target "
            "reaches a routed expert; refusing")

    keys = tuple(sorted((int(l), m) for m, rec in out_targets.items()
                        for l in rec["layers"]))
    for name in factor_maps:
        fmap = blob.get(name)
        if not isinstance(fmap, Mapping) or not fmap:
            raise LoraExportError(
                f"kind {kind!r}: {name} must be a non-empty mapping "
                f"'<layer>.<module>' -> tensor, got {type(fmap).__name__}")
        got = tuple(sorted(_general_key(kind, k, name=name) for k in fmap))
        if got != keys:
            missing_k = sorted(set(keys) - set(got))
            stray = sorted(set(got) - set(keys))
            raise LoraExportError(
                f"kind {kind!r}: {name}'s keys do not match the targets block "
                f"(missing {[f'{l}.{m}' for l, m in missing_k]}, stray "
                f"{[f'{l}.{m}' for l, m in stray]}) — the artifact does not "
                "agree with itself about what it targets; refusing")
    union = tuple(sorted({l for l, _ in keys}))
    if "layers" in blob:
        declared = blob["layers"]
        if not isinstance(declared, (list, tuple)) or tuple(sorted(
                _blob_int(kind, "layers entry", l, minimum=0) for l in declared)) != union:
            raise LoraExportError(
                f"kind {kind!r}: the blob's `layers` metadata {declared!r} is not "
                f"the sorted union {list(union)} of its targets' layers; refusing")
    return _GeneralShape(modules=modules, set_id=set_id, per_expert_rank=per,
                         init_scheme=init_scheme, targets=out_targets, keys=keys,
                         layers=union, routed_rank=routed_rank)


def _general_factor(kind: str, layer: int, module: str, meta: Mapping[str, Any],
                    a_raw: Any, b_raw: Any, rank: int, *,
                    a_name: str = "A", b_name: str = "B",
                    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Any, ...]]:
    """One general-shape target's saved factors, validated (fp32, finite,
    the target's rank) and oriented for PEFT. Linear: ``(lora_A [r, d_in],
    lora_B [d_out, r], ("linear", d_in, d_out))``. Experts: the 3-D
    tensors unchanged, ``(A3 [E, r_e, in], B3 [E, out, r_e], ("experts",
    E, in, out))`` — flattened later by :func:`_experts_flat`."""
    key = f"{layer}.{module}"
    if meta["kind"] == "linear":
        a = _saved_f32(kind, f"{a_name}[{key}]", a_raw, dims=2)
        b = _saved_f32(kind, f"{b_name}[{key}]", b_raw, dims=2)
        if a.shape[1] != rank or b.shape[1] != rank:
            raise LoraExportError(
                f"kind {kind!r}: {key} has {a_name} {tuple(a.shape)} / {b_name} "
                f"{tuple(b.shape)} but the object is rank {rank} ({a_name} must be "
                f"[d_in, r], {b_name} [d_out, r]) — refusing a rank the config "
                "would mis-declare")
        return a.t(), b, ("linear", int(a.shape[0]), int(b.shape[0]))
    r_e, n_e = int(meta["rank"]), int(meta["num_experts"])
    a = _saved_f32(kind, f"{a_name}[{key}]", a_raw, dims=3)
    b = _saved_f32(kind, f"{b_name}[{key}]", b_raw, dims=3)
    if (a.shape[0] != n_e or b.shape[0] != n_e or a.shape[1] != r_e
            or b.shape[2] != r_e):
        raise LoraExportError(
            f"kind {kind!r}: {key} has {a_name} {tuple(a.shape)} / {b_name} "
            f"{tuple(b.shape)} but the target is {n_e} experts at per-expert rank "
            f"{r_e} ({a_name} must be [E, r_e, in], {b_name} [E, out, r_e]); "
            "refusing")
    return a, b, ("experts", n_e, int(a.shape[2]), int(b.shape[1]))


def _experts_module_path(kind: str, module: str, site: str) -> Tuple[str, str]:
    """``"mlp.experts.gate_up_proj"`` -> ``("mlp.experts", "gate_up_proj")``:
    the experts MODULE path inside the block and the HF parameter name."""
    path, _, param = site.rpartition(".")
    if param not in _EXPERT_PARAMS or not (
            path == EXPERTS_TARGET_SUFFIX or path.endswith(f".{EXPERTS_TARGET_SUFFIX}")):
        raise LoraExportError(
            f"kind {kind!r}: targets[{module!r}].site={site!r} is not "
            f"'<ffn>.{EXPERTS_TARGET_SUFFIX}.<gate_up_proj|down_proj>'; refusing")
    return path, param


def _experts_flat(A3: torch.Tensor, B3: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The 3-D expert factors -> PEFT ``ParamWrapper``'s 2-D file layout:
    ``lora_A [(r·E), in] = A3.reshape(E*r, in)`` (rows ordered expert-major,
    so vLLM's ``reshape(E, -1, in)`` recovers ``A3`` bitwise) and ``lora_B
    [out, (r·E)] = B3.permute(1, 2, 0).reshape(out, r*E)`` (columns ordered
    rank-major, so ``reshape(out, -1, E).permute(2, 0, 1)`` recovers ``B3``
    bitwise) — ``model_manager.py:857-880`` / ``layer.py:2418-2431``."""
    n_e, r, d_in = (int(x) for x in A3.shape)
    _, d_out, _ = (int(x) for x in B3.shape)
    lora_A = A3.reshape(n_e * r, d_in).contiguous()
    lora_B = B3.permute(1, 2, 0).reshape(d_out, r * n_e).contiguous()
    return lora_A, lora_B


def experts_unflatten(lora_A: torch.Tensor, lora_B: torch.Tensor, num_experts: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The inverse of the file layout, spelled EXACTLY as vLLM 0.25.1 reads
    it -> ``(A3 [E, r, in], B3 [E, out, r])`` — for tests, audits and the
    serving probe's reference check. ONE implementation: the attestation
    side's ``worker_reports.reshape_flat_experts`` (it must match vLLM's
    ``_stack_moe_lora_weights``; lazy import — that module pulls torch and
    is the worker's, never this module's dependency at load)."""
    from .worker_reports import reshape_flat_experts
    return reshape_flat_experts(lora_A, lora_B, num_experts)


def _emit_experts(tensors: Dict[str, torch.Tensor], module_name: str, param: str,
                  lora_A: torch.Tensor, lora_B: torch.Tensor) -> None:
    """Write one experts half under its PEFT name: ``module_name`` is the
    experts module's full path (``{prefix}.{layer}.mlp.experts``), ``param``
    the HF parameter it adapts — ``gate_up_proj`` lands on the INNER
    ``.base_layer`` wrapper, ``down_proj`` on the OUTER
    (:data:`EXPERTS_PEFT_SUFFIX`)."""
    name = f"base_model.model.{module_name}{EXPERTS_PEFT_SUFFIX[param]}"
    tensors[f"{name}.lora_A.weight"] = lora_A.contiguous()
    tensors[f"{name}.lora_B.weight"] = lora_B.contiguous()


def _experts_companion_dims(param_present: str, dims: Tuple[Any, ...]
                            ) -> Tuple[str, int, int]:
    """The MISSING half's ``(param, in, out)`` from the present half's
    ``("experts", E, in, out)``: HF's pair is gate_up ``[E, 2I, d]`` / down
    ``[E, d, I]`` (``lora_modules.resolve_targets`` holds the pair to that
    invariant), so down ``(in=I, out=d)`` implies gate_up ``(in=d, out=2I)``
    and vice versa."""
    _, _, d_in, d_out = dims
    if param_present == "down_proj":
        return "gate_up_proj", d_out, 2 * d_in
    if d_out % 2:
        raise LoraExportError(
            f"experts.gate_up_proj out_features {d_out} is odd — not the fused "
            "[E, 2I, d] layout; cannot size its down_proj companion")
    return "down_proj", d_out // 2, d_in


def _missing_expert_halves(written: Mapping[str, Mapping[str, Tuple[Any, ...]]]):
    """Over ``{experts module name: {param: ("experts", E, in, out)}}``, the
    ABSENT half of every module recorded with ONE half — yields
    ``(module_name, param, E, in, out)`` of the companion to write, in module
    name order (:func:`_experts_companion_dims` sizes it)."""
    for module_name in sorted(written):
        halves = written[module_name]
        if set(halves) == set(_EXPERT_PARAMS):
            continue
        (present, dims), = halves.items()
        param, d_in, d_out = _experts_companion_dims(present, dims)
        yield module_name, param, dims[1], d_in, d_out


def _emit_experts_companions(kind: str, tensors: Dict[str, torch.Tensor],
                             written: Mapping[str, Dict[str, Tuple[Any, ...]]],
                             r_e: int) -> List[str]:
    """For every MoE block whose adapter carries ONE experts half, write the
    other as a ZERO pair at the same ``r_e`` (the ``down`` set's frozen
    gate_up companion): vLLM asserts both halves at load
    (``model_manager.py:843``) and skips a lone gate_up silently. Returns
    the PEFT module names written, for the record."""
    names: List[str] = []
    for module_name, param, n_e, d_in, d_out in _missing_expert_halves(written):
        _emit_experts(tensors, module_name, param,
                      torch.zeros((n_e * r_e, d_in), dtype=torch.float32),
                      torch.zeros((d_out, n_e * r_e), dtype=torch.float32))
        names.append(f"{module_name}{EXPERTS_PEFT_SUFFIX[param]}")
    return names


def _experts_rank_regex(path: str) -> str:
    """The ``rank_pattern`` / ``alpha_pattern`` key that selects BOTH
    routed-expert parameters of one experts module path. PEFT 0.20.0
    matches a pattern key ``k`` against the full parameter key with
    ``re.match(rf"(.*\\.)?({k})$", key)`` (``peft/utils/other.py``
    ``get_pattern_key``), so ``mlp\\.experts\\.(gate_up_proj|down_proj)``
    selects ``model.language_model.layers.N.mlp.experts.gate_up_proj`` and
    nothing else. vLLM ignores both patterns (``peft_helper.py`` reads
    ``r``/``lora_alpha`` only); they exist so HF+PEFT builds the same
    per-expert rank the file carries."""
    return re.escape(path) + r"\.(" + "|".join(_EXPERT_PARAMS) + ")"


def _skeleton_targets(kind: str, skeleton: torch.nn.Module, shape: _GeneralShape,
                      rank: int) -> Dict[str, Any]:
    """``lora_modules.resolve_targets`` over the base's meta skeleton for
    the blob's module set, keyed ``"<layer>.<module>"`` — the yardstick
    every general-shape target is held to (:func:`_hold_to_skeleton`)."""
    from ...extraction.lora_modules import LoRAModuleError, resolve_targets

    try:
        targets = resolve_targets(skeleton, shape.modules, rank=rank,
                                  per_expert_rank=shape.routed_rank)
    except LoRAModuleError as e:
        raise LoraExportError(
            f"kind {kind!r}: the module set {list(shape.modules)} does not "
            f"resolve on the base's module tree — {e}") from e
    return {t.key: t for t in targets}


def _hold_to_skeleton(kind: str, layer: int, module: str, meta: Mapping[str, Any],
                      dims: Tuple[Any, ...], skel: Optional[Mapping[str, Any]],
                      module_name: str) -> None:
    """With the tree (the derived path), a general-shape target must name a
    module the base HAS, at the SITE the base spells it, with the SHAPE the
    base's module has — refused by name otherwise (the same rule as
    :func:`_check_layer_dims`: vLLM's ``set_lora`` zero-pads an undersized
    factor silently). Without the tree (a caller-stated prefix) nothing is
    held and the blob's sites are emitted as recorded."""
    if skel is None:
        return
    t = skel.get(f"{layer}.{module}")
    if t is None:
        have = sorted({int(k.split('.', 1)[0]) for k, s in skel.items()
                       if s.module == module})
        raise LoraExportError(
            f"kind {kind!r}: layer {layer} carries no {module!r} on this base "
            f"(blocks that do: {have}) — the module name {module_name!r} would "
            "point at nothing; refusing")
    if t.site != meta["site"]:
        raise LoraExportError(
            f"kind {kind!r}: {layer}.{module}: the blob says site "
            f"{meta['site']!r} but the base's block {layer} carries it at "
            f"{t.site!r} — refusing to emit {module_name!r}")
    if dims[0] == "linear":
        want: Tuple[Any, ...] = ("linear", int(t.in_features), int(t.out_features))
    else:
        want = ("experts", int(t.num_experts or 0), int(t.in_features),
                int(t.out_features))
    if tuple(dims) != want:
        raise LoraExportError(
            f"kind {kind!r}: {layer}.{module} has {dims[0]} dims {tuple(dims[1:])} "
            f"but the base's own module ({module_name}) is {want[1:]} — the "
            "factors do not fit the module they name: corrupt blob, or a fit "
            "from another base. Refusing (the vLLM load would zero-pad instead)")


def _general_config(shape: _GeneralShape, *, rank_multiplier: int = 1
                    ) -> Dict[str, Any]:
    """The general-shape config keys (D_contract §3): ``target_modules`` =
    the REAL HF short names of the set (PEFT semantics — on a MoE set
    ``gate_proj``/``up_proj``/``down_proj`` reach the shared expert; NEVER
    the vLLM wrapper word ``experts``); when routed experts are adapted,
    ``target_parameters`` naming BOTH expert parameters (the companion is
    in the file) and ``rank_pattern``/``alpha_pattern`` mapping the experts
    regex to ``r_e`` (× ``rank_multiplier``, 2 for a difference) so HF+PEFT
    builds the file's shapes at scaling 1; plus the ``petri_*`` provenance
    of the set. ``petri_targets`` / ``petri_per_expert_rank`` describe the
    SOURCE object (a difference's parents), never the doubled ranks."""
    cfg: Dict[str, Any] = {
        "target_modules": list(shape.modules),
        "petri_lora_modules": list(shape.modules),
        "petri_lora_set_id": shape.set_id,
        "petri_per_expert_rank": (dict(shape.per_expert_rank)
                                  if shape.per_expert_rank is not None else None),
        "petri_targets": {m: dict(rec) for m, rec in shape.targets.items()},
    }
    experts = shape.experts_modules
    if experts:
        paths = sorted({_experts_module_path("lora_all", m, shape.targets[m]["site"])[0]
                        for m in experts})
        cfg["target_parameters"] = sorted(f"{p}.{param}" for p in paths
                                          for param in _EXPERT_PARAMS)
        r_e = int(shape.routed_rank) * int(rank_multiplier)
        cfg["rank_pattern"] = {_experts_rank_regex(p): r_e for p in paths}
        cfg["alpha_pattern"] = {_experts_rank_regex(p): r_e for p in paths}
    return cfg


def _spec_check_modules(kind: str, spec: Mapping[str, Any],
                        shape: _GeneralShape) -> None:
    """F9-style cross-checks of the resolved spec's module-set fields."""
    if "lora_modules" in spec:
        try:
            from ...extraction.lora_modules import normalise_modules
            got = normalise_modules(spec["lora_modules"])
        except Exception as e:  # LoRAModuleError or a malformed value
            raise LoraExportError(
                f"kind {kind!r}: resolved_spec lora_modules="
                f"{spec['lora_modules']!r} is not a module list ({e})") from e
        if got != shape.modules:
            raise LoraExportError(
                f"kind {kind!r}: resolved_spec lora_modules={list(got)} != "
                f"artifact lora_modules={list(shape.modules)} (F9-style "
                "cross-check) — the spec and the bytes disagree; refusing")
    if "per_expert_rank" in spec and spec["per_expert_rank"] != shape.per_expert_rank:
        raise LoraExportError(
            f"kind {kind!r}: resolved_spec per_expert_rank="
            f"{spec['per_expert_rank']!r} != artifact "
            f"per_expert_rank={shape.per_expert_rank!r}; refusing")


def _export_general(kind: str, resolved_spec: Mapping[str, Any], blob: dict,
                    shape: _GeneralShape, *, factors, rank_multiplier: int,
                    extra_config, decoder_prefix: str, oproj_site: str,
                    skeleton: Optional[torch.nn.Module],
                    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """THE general-shape emitter both kinds share (module section comment
    above): every linear target as an ordinary PEFT pair at its own site,
    every routed-expert target as PEFT's ``ParamWrapper`` layout (two
    flattened pairs per MoE block, a zero companion for a missing half at
    ``rank_multiplier × r_e``), config per :func:`_general_config` with
    ``r = lora_alpha = rank_multiplier × rank`` (scaling exactly 1 on EVERY
    module). ``factors(layer, module, meta, rank) -> (lora_A, lora_B, dims)``
    is the kind's own per-target read (:func:`_general_factor` for
    ``lora_all``; the rank-stacked pair for ``lora_subtraction``);
    ``extra_config(blob, rank) -> dict`` validates the kind's extra blob
    record (the difference's provenance) at the point the kind always
    validated it and returns its extra config keys."""
    rank = _blob_int(kind, "rank", blob["rank"], minimum=1)
    positions = _fit_positions(kind, blob["positions"])
    if blob.get("response_header_len") is not None:
        _blob_int(kind, "response_header_len", blob["response_header_len"],
                  minimum=0)
    extra = extra_config(blob, rank)
    _spec_check(kind, resolved_spec, "layers", list(shape.layers))
    _spec_check(kind, resolved_spec, "rank", rank)
    _spec_check(kind, resolved_spec, "positions", positions)
    _spec_check_modules(kind, resolved_spec, shape)
    skel = (_skeleton_targets(kind, skeleton, shape, rank)
            if skeleton is not None else None)

    tensors: Dict[str, torch.Tensor] = {}
    written: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
    for layer, module in shape.keys:
        meta = shape.targets[module]
        lora_A, lora_B, dims = factors(layer, module, meta, rank)
        if meta["kind"] == "linear":
            module_name = f"{decoder_prefix}.{layer}.{meta['site']}"
            _hold_to_skeleton(kind, layer, module, meta, dims, skel, module_name)
            tensors[f"base_model.model.{module_name}.lora_A.weight"] = lora_A.contiguous()
            tensors[f"base_model.model.{module_name}.lora_B.weight"] = lora_B.contiguous()
        else:
            path, param = _experts_module_path(kind, module, meta["site"])
            module_name = f"{decoder_prefix}.{layer}.{path}"
            _hold_to_skeleton(kind, layer, module, meta, dims, skel,
                              f"{module_name}.{param}")
            flat_A, flat_B = _experts_flat(lora_A, lora_B)
            _emit_experts(tensors, module_name, param, flat_A, flat_B)
            written.setdefault(module_name, {})[param] = dims
    if written:
        _emit_experts_companions(kind, tensors, written,
                                 rank_multiplier * int(shape.routed_rank))

    config = {
        "r": rank_multiplier * rank,
        "lora_alpha": float(rank_multiplier * rank),   # scaling = alpha/r = 1 on EVERY module
        **_general_config(shape, rank_multiplier=rank_multiplier),
        "layers_to_transform": list(shape.layers),
        "petri_source_kind": kind,
        "petri_source_layers": list(shape.layers),
        "petri_fit_positions": positions,
        **extra,
    }
    if oproj_site != DEFAULT_OPROJ_SITE:
        config["petri_oproj_site"] = oproj_site
    return tensors, config


def _export_general_lora_all(resolved_spec: Mapping[str, Any], blob: dict,
                             shape: _GeneralShape, *, decoder_prefix: str,
                             oproj_site: str,
                             skeleton: Optional[torch.nn.Module],
                             ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """The general-shape half of :func:`_export_multilayer_lora`: each
    target's saved ``A``/``B`` through :func:`_export_general`."""
    kind = "lora_all"

    def factors(layer, module, meta, rank):
        key = f"{layer}.{module}"
        return _general_factor(kind, layer, module, meta, blob["A"][key], blob["B"][key], rank)

    return _export_general(kind, resolved_spec, blob, shape, factors=factors,
                           rank_multiplier=1, extra_config=lambda blob, rank: {},
                           decoder_prefix=decoder_prefix, oproj_site=oproj_site,
                           skeleton=skeleton)


def _export_general_subtraction(resolved_spec: Mapping[str, Any], blob: dict,
                                shape: _GeneralShape, *, decoder_prefix: str,
                                oproj_site: str,
                                skeleton: Optional[torch.nn.Module],
                                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """The general-shape half of :func:`_export_lora_subtraction` — rank
    stacking per target: linear ``lora_A = cat([A_pos.T, A_neg.T], 0)``
    / ``lora_B = cat([B_pos, -B_neg], 1)``; routed experts concatenated
    along the PER-EXPERT rank axis BEFORE flattening (``A3 = cat([A_pos,
    A_neg], dim=1)`` -> ``[E, 2r_e, in]``, ``B3 = cat([B_pos, -B_neg],
    dim=2)`` -> ``[E, out, 2r_e]``), so after :func:`_experts_flat` the
    negative parent's block is ``lora_B.reshape(out, 2*r_e, E)[:, r_e:,
    :]`` (the linear rule's ``lora_B[:, rank:]`` read per expert). ``r`` and
    ``lora_alpha`` are ``2·rank``; ``rank_pattern`` is ``2·r_e``
    (:func:`_export_general` with ``rank_multiplier=2``)."""
    kind = "lora_subtraction"

    def factors(layer, module, meta, rank):
        key = f"{layer}.{module}"
        A_pos, B_pos, d_pos = _general_factor(
            kind, layer, module, meta, blob["A_pos"][key], blob["B_pos"][key], rank,
            a_name="A_pos", b_name="B_pos")
        A_neg, B_neg, d_neg = _general_factor(
            kind, layer, module, meta, blob["A_neg"][key], blob["B_neg"][key], rank,
            a_name="A_neg", b_name="B_neg")
        if d_pos != d_neg:
            raise LoraExportError(
                f"kind {kind!r}: {key}: parent shapes differ (pos {d_pos} vs neg "
                f"{d_neg}) — the task-vector difference needs same-shape parents; "
                "refusing")
        if meta["kind"] == "linear":
            return (torch.cat([A_pos, A_neg], dim=0),        # [2r, d_in]
                    torch.cat([B_pos, -B_neg], dim=1), d_pos)  # [d_out, 2r]
        return (torch.cat([A_pos, A_neg], dim=1),            # [E, 2r_e, in]
                torch.cat([B_pos, -B_neg], dim=2), d_pos)      # [E, out, 2r_e]

    def extra_config(blob, rank):
        return {"petri_lora_subtraction_parent_rank": rank,
                "petri_lora_subtraction_negated_block": SUBTRACTION_NEGATED_BLOCK,
                "petri_lora_subtraction_provenance":
                    _subtraction_provenance(kind, blob.get("provenance"))}

    return _export_general(kind, resolved_spec, blob, shape, factors=factors,
                           rank_multiplier=2, extra_config=extra_config,
                           decoder_prefix=decoder_prefix, oproj_site=oproj_site,
                           skeleton=skeleton)


def _export_multilayer_lora(resolved_spec: Mapping[str, Any], blob: dict, *,
                            decoder_prefix: str,
                            oproj_site: str = DEFAULT_OPROJ_SITE,
                            oproj_layers: Optional[Tuple[int, ...]] = None,
                            oproj_shapes: Optional[Mapping[int, Tuple[int, int]]] = None,
                            skeleton: Optional[torch.nn.Module] = None,
                            ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """MultiLayerLoRAObject (kind ``lora_all``) -> PEFT tensors + config.

    The all-layer rank-r o_proj LoRA: one (A_l, B_l) pair per TARGETED
    layer (the object's capability-filtered ``layers`` — every attention
    block on dense stacks; [12, 17, 24, 32] on Nemotron-3-Nano), one
    shared rank. THE CONVERSION, derived from the authoritative hook body
    (extraction/multilayer.MultiLayerLoRAObject._make_hook), per layer:

        rank 1:  coef  = x @ A                  # A [d_in]
                 delta = coef.unsqueeze(-1) * B  # B [d_out]   -> dW = B ⊗ A
        rank r:  coef  = x @ A                  # A [d_in, r]
                 delta = coef @ B.T             # B [d_out, r] -> dW = B @ A.T
                 output += scale * delta        # selector scale; 1.0 served

    PEFT's layout (export_zero_adapter): ``lora_A.weight [r, d_in]``,
    ``lora_B.weight [d_out, r]``, ``delta = (x @ lora_A.T) @ lora_B.T *
    (lora_alpha / r)``. With ``lora_A = A.T`` (rank 1: ``A[None, :]``),
    ``lora_B = B`` (rank 1: ``B[:, None]``) and ``lora_alpha = r`` (scaling
    exactly 1), the served delta performs the SAME reduction over d_in
    followed by the SAME expansion over d_out as the hook — pinned by the
    export tests: bit-exact for rank 1 (one op structure on both sides), to
    fp32 rounding for rank r (PEFT multiplies by a transposed VIEW of the
    exported factor, a different BLAS layout whose accumulation order is
    not build-invariant). The
    orientation is fixed by the hook body (A dots the input, B spans the
    output), and the rank by the stored tensors' shapes, never by the
    kind's name: a declared ``rank`` that disagrees with the tensors
    refuses.

    Only an ALL-position fit is exportable (module docstring): the
    response-masked lora_all arm serves on the lens o_proj site, exactly
    as the rank-1 kind does."""
    kind = "lora_all"
    required = {"kind", "A", "B", "positions"}
    # ``layers`` and ``rank`` ride explicitly on every blob the current
    # class writes; legacy rank-1 blobs predate both and are inferred from
    # the tensors (load_blob's own rule). ``response_header_len`` is the
    # fit leg's other record — validated when present, never guessed.
    # ``init_scheme`` (FD 7) rides on a legacy blob only when it is not the
    # default and on every general-shape blob; the four other keys are the
    # GENERAL module-set shape (D_contract §2), dispatched below on the
    # presence of ``targets`` — a blob without it exports as it always has.
    optional = {"layers", "rank", "response_header_len", "init_scheme",
                "lora_modules", "lora_set_id", "per_expert_rank", "targets"}
    missing = required - set(blob)
    if missing:
        raise LoraExportError(
            f"kind {kind!r}: artifact lacks {sorted(missing)} — not a "
            "MultiLayerLoRAObject.save blob")
    unknown = set(blob) - required - optional
    if unknown:
        raise LoraExportError(
            f"kind {kind!r}: artifact carries unexpected keys "
            f"{sorted(unknown)} — the representation drifted past what "
            "this export was verified against; refusing to guess")
    shape = _general_shape(kind, blob, factor_maps=("A", "B"))
    if shape is not None:
        return _export_general_lora_all(resolved_spec, blob, shape,
                                        decoder_prefix=decoder_prefix,
                                        oproj_site=oproj_site, skeleton=skeleton)
    _check_init_scheme(kind, blob)
    layers = _layer_set(kind, blob["A"], name="A")
    layers_b = _layer_set(kind, blob["B"], name="B")
    if layers_b != layers:
        raise LoraExportError(
            f"kind {kind!r}: A and B cover different layer sets "
            f"({layers} vs {layers_b}); corrupt blob")
    _check_declared_layers(kind, blob, layers, "A/B")

    # RANK from the tensors, cross-checked against the declared rank.
    rank = _rank_of(kind, blob["A"][_map_key(blob["A"], layers[0])],
                    blob["rank"] if "rank" in blob else _UNSET,
                    name=f"A[{layers[0]}]")

    positions = _fit_positions(kind, blob["positions"])
    if blob.get("response_header_len") is not None:
        _blob_int(kind, "response_header_len", blob["response_header_len"],
                  minimum=0)
    _spec_check(kind, resolved_spec, "layers", layers)
    _spec_check(kind, resolved_spec, "rank", rank)
    _spec_check(kind, resolved_spec, "positions", positions)
    _refuse_layers_outside(kind, layers, oproj_layers)

    tensors: Dict[str, torch.Tensor] = {}
    dims: Optional[Tuple[int, int]] = None
    for l in layers:
        lora_A, lora_B, d = _peft_pair(
            kind, l, blob["A"][_map_key(blob["A"], l)],
            blob["B"][_map_key(blob["B"], l)], rank)
        module = f"{decoder_prefix}.{l}.{oproj_site}"
        dims = _check_layer_dims(kind, l, d, oproj_shapes=oproj_shapes,
                                 shared=dims, module=module)
        tensors[f"base_model.model.{module}.lora_A.weight"] = \
            lora_A.contiguous()
        tensors[f"base_model.model.{module}.lora_B.weight"] = \
            lora_B.contiguous()

    config = {
        "r": rank,
        "lora_alpha": float(rank),   # scaling = alpha/r = 1 — the hook adds
                                     # delta unscaled at selector scale 1.0
        "target_modules": [oproj_site.rsplit(".", 1)[-1]],   # ["o_proj"]
        "layers_to_transform": list(layers),
        # honest provenance (contract: never petri_zero_adapter)
        "petri_source_kind": kind,
        "petri_source_layers": list(layers),
        "petri_fit_positions": positions,
    }
    if oproj_site != DEFAULT_OPROJ_SITE:
        config["petri_oproj_site"] = oproj_site
    return tensors, config


#: the sign convention the export fixes and the config records: the negative
#: parent's B block is NEGATED in the write basis, so ``ΔW = lora_B @
#: lora_A = B_pos A_posᵀ − B_neg A_negᵀ`` with scaling exactly 1.
SUBTRACTION_NEGATED_BLOCK = "lora_B[:, rank:]"


def _subtraction_provenance(kind: str, provenance: Any) -> Dict[str, Any]:
    """``petri_lora_subtraction_provenance``: the two parents' ``cell_id``
    and ``object_sha256`` lifted out of the blob's ``provenance`` block
    (scripts/sdfbase/lora_subtract.py writes ``parent_pos`` /
    ``parent_neg`` sub-dicts carrying both) — STRINGS copied when present,
    ``None`` when not recorded, never a tensor or any other value (the
    config is JSON and part of the adapter's hashed identity). A
    provenance block that is not a mapping is not the class's blob
    (``compose_blob`` writes ``dict(provenance or {})``) and refuses."""
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, Mapping):
        raise LoraExportError(
            f"kind {kind!r}: provenance must be a mapping, got "
            f"{type(provenance).__name__} — not a LoRASubtractionObject."
            "save blob")
    out: Dict[str, Any] = {}
    for side in ("parent_pos", "parent_neg"):
        parent = provenance.get(side)
        if parent is None:
            parent = {}
        if not isinstance(parent, Mapping):
            raise LoraExportError(
                f"kind {kind!r}: provenance[{side!r}] must be a mapping, "
                f"got {type(parent).__name__}; refusing")
        out[side] = {
            field: (parent[field] if isinstance(parent.get(field), str)
                    and parent[field] else None)
            for field in ("cell_id", "object_sha256")}
    return out


def _export_lora_subtraction(resolved_spec: Mapping[str, Any], blob: dict, *,
                             decoder_prefix: str,
                             oproj_site: str = DEFAULT_OPROJ_SITE,
                             oproj_layers: Optional[Tuple[int, ...]] = None,
                             oproj_shapes: Optional[Mapping[int, Tuple[int, int]]] = None,
                             skeleton: Optional[torch.nn.Module] = None,
                             ) -> Tuple[Dict[str, torch.Tensor],
                                        Dict[str, Any]]:
    """LoRASubtractionObject (kind ``lora_subtraction``) -> PEFT tensors +
    config, by RANK STACKING.

    The object is the task-vector DIFFERENCE of two same-shape ``lora_all``
    fits: per layer the hook registers the positive parent's o_proj hook
    at ``+scale`` and the negative parent's at ``-scale`` on the same
    module (extraction/multilayer.LoRASubtractionObject.apply), so the
    served delta is ``Δ_pos(x) − Δ_neg(x) = x A_pos B_posᵀ − x A_neg
    B_negᵀ``. PEFT has ONE ``(lora_A, lora_B)`` pair per module, so the two
    rank-r parents become one rank-2r adapter, per layer ``l``::

        lora_A = cat([A_pos.T, A_neg.T], dim=0)      # [2r, d_in]
        lora_B = cat([B_pos,  -B_neg  ], dim=1)      # [d_out, 2r]
        r = 2·rank, lora_alpha = float(2·rank)       # scaling exactly 1

    so ``ΔW = lora_B @ lora_A = B_pos A_posᵀ − B_neg A_negᵀ`` EXACTLY: the
    sign moves from a per-hook coefficient into the write basis and is
    RECORDED in the config (``petri_lora_subtraction_negated_block`` =
    :data:`SUBTRACTION_NEGATED_BLOCK`, ``petri_lora_subtraction_parent_
    rank``), so an auditor can recover both parents' factors bitwise from
    the adapter (``lora_A[:r].T``, ``lora_B[:, :r]``, ``lora_A[r:].T``,
    ``-lora_B[:, r:]``). Rank-1 legacy 1-D parents unsqueeze exactly as
    :func:`_export_multilayer_lora` does. Numerics: the served GEMM sums
    2r terms once where the hook sums r terms twice and subtracts — same
    math, a different accumulation order, so the CPU pin is fp32
    rounding (``atol=1e-5``), not bitwise; the served-vs-reference proof
    is the §10 gate.

    Every precondition ``LoRASubtractionObject._refuse_mismatch`` places
    on hydrated parents is an export refusal here on the raw blob: the
    four factor maps must cover ONE layer set, both parents ONE rank
    (from the tensors, cross-checked against the declared ``rank``), each
    layer's ``(d_in, d_out)`` equal across parents (and held to that
    layer's own o_proj module on the derived path — :func:`_check_layer_dims`),
    one shared all-position fit (``positions``; a masked fit serves on
    the lens, as for every kind here), fp32 factors only, and every layer
    on the resolved tree's attention set."""
    kind = "lora_subtraction"
    required = {"kind", "A_pos", "B_pos", "A_neg", "B_neg", "positions"}
    # ``layers``, ``rank``, ``provenance`` and ``response_header_len`` ride
    # on every blob compose_blob writes; each is validated when present
    # (a legacy-shaped blob without them is inferred from the tensors,
    # load_blob's own rule) and never guessed.
    # plus the GENERAL module-set shape's keys (D_contract §2: the parents'
    # ``lora_modules`` / ``lora_set_id`` / ``per_expert_rank`` / ``targets``
    # ride on a general difference), dispatched below on ``targets``.
    optional = {"layers", "rank", "provenance", "response_header_len",
                "init_scheme", "lora_modules", "lora_set_id", "per_expert_rank",
                "targets"}
    factor_maps = ("A_pos", "B_pos", "A_neg", "B_neg")
    missing = required - set(blob)
    if missing:
        raise LoraExportError(
            f"kind {kind!r}: artifact lacks {sorted(missing)} — not a "
            "LoRASubtractionObject.save blob")
    unknown = set(blob) - required - optional
    if unknown:
        raise LoraExportError(
            f"kind {kind!r}: artifact carries unexpected keys "
            f"{sorted(unknown)} — the representation drifted past what "
            "this export was verified against; refusing to guess")
    shape = _general_shape(kind, blob, factor_maps=factor_maps)
    if shape is not None:
        return _export_general_subtraction(resolved_spec, blob, shape,
                                           decoder_prefix=decoder_prefix,
                                           oproj_site=oproj_site,
                                           skeleton=skeleton)
    _check_init_scheme(kind, blob)

    layer_sets = {name: _layer_set(kind, blob[name], name=name)
                  for name in factor_maps}
    layers = layer_sets["A_pos"]
    for name in factor_maps[1:]:
        if layer_sets[name] != layers:
            raise LoraExportError(
                f"kind {kind!r}: the four factor maps cover different "
                f"layer sets (A_pos {layers} vs {name} {layer_sets[name]}) "
                "— the task-vector difference is only defined layerwise on "
                "the same set; corrupt blob")
    _check_declared_layers(kind, blob, layers, "A_pos/B_pos/A_neg/B_neg")

    # RANK from each parent's tensors, both cross-checked against the one
    # declared rank; the parents must agree with each other too.
    declared = blob["rank"] if "rank" in blob else _UNSET
    first = layers[0]
    rank_pos = _rank_of(kind, blob["A_pos"][_map_key(blob["A_pos"], first)],
                        declared, name=f"A_pos[{first}]")
    rank_neg = _rank_of(kind, blob["A_neg"][_map_key(blob["A_neg"], first)],
                        declared, name=f"A_neg[{first}]")
    if rank_pos != rank_neg:
        raise LoraExportError(
            f"kind {kind!r}: parent ranks differ ({rank_pos} vs {rank_neg}) "
            "— same-shape parents only; refusing")
    rank = rank_pos

    positions = _fit_positions(kind, blob["positions"])
    if blob.get("response_header_len") is not None:
        _blob_int(kind, "response_header_len", blob["response_header_len"],
                  minimum=0)
    provenance = _subtraction_provenance(kind, blob.get("provenance"))
    _spec_check(kind, resolved_spec, "layers", layers)
    _spec_check(kind, resolved_spec, "rank", rank)
    _spec_check(kind, resolved_spec, "positions", positions)
    _refuse_layers_outside(kind, layers, oproj_layers)

    tensors: Dict[str, torch.Tensor] = {}
    dims: Optional[Tuple[int, int]] = None
    for l in layers:
        A_pos, B_pos, d_pos = _peft_pair(
            kind, l, blob["A_pos"][_map_key(blob["A_pos"], l)],
            blob["B_pos"][_map_key(blob["B_pos"], l)], rank,
            a_name="A_pos", b_name="B_pos")
        A_neg, B_neg, d_neg = _peft_pair(
            kind, l, blob["A_neg"][_map_key(blob["A_neg"], l)],
            blob["B_neg"][_map_key(blob["B_neg"], l)], rank,
            a_name="A_neg", b_name="B_neg")
        if d_pos != d_neg:
            raise LoraExportError(
                f"kind {kind!r}: layer {l}: parent shapes differ (pos "
                f"(d_in, d_out)={d_pos} vs neg {d_neg}) — the task-vector "
                "difference needs same-shape parents; refusing")
        module = f"{decoder_prefix}.{l}.{oproj_site}"
        dims = _check_layer_dims(kind, l, d_pos, oproj_shapes=oproj_shapes,
                                 shared=dims, module=module)
        # rank stacking: rows [0, r) are the positive parent, rows
        # [r, 2r) the negative one; the negative B block carries the sign.
        lora_A = torch.cat([A_pos, A_neg], dim=0)          # [2r, d_in]
        lora_B = torch.cat([B_pos, -B_neg], dim=1)         # [d_out, 2r]
        tensors[f"base_model.model.{module}.lora_A.weight"] = \
            lora_A.contiguous()
        tensors[f"base_model.model.{module}.lora_B.weight"] = \
            lora_B.contiguous()

    config = {
        "r": 2 * rank,
        "lora_alpha": float(2 * rank),   # scaling = alpha/r = 1 — the hooks
                                         # add both deltas unscaled at
                                         # selector scale 1.0
        "target_modules": [oproj_site.rsplit(".", 1)[-1]],   # ["o_proj"]
        "layers_to_transform": list(layers),
        # honest provenance (contract: never petri_zero_adapter)
        "petri_source_kind": kind,
        "petri_source_layers": list(layers),
        "petri_fit_positions": positions,
        # the parents' shared rank and WHICH block carries the sign,
        # so the two parents are recoverable bitwise from the adapter
        "petri_lora_subtraction_parent_rank": rank,
        "petri_lora_subtraction_negated_block": SUBTRACTION_NEGATED_BLOCK,
        "petri_lora_subtraction_provenance": provenance,
    }
    if oproj_site != DEFAULT_OPROJ_SITE:
        config["petri_oproj_site"] = oproj_site
    return tensors, config


def _export_circuit_breaker(resolved_spec: Mapping[str, Any], blob: dict, *,
                            decoder_prefix: str,
                            oproj_site: str = DEFAULT_OPROJ_SITE,
                            ) -> Tuple[Dict[str, torch.Tensor],
                                       Dict[str, Any]]:
    """CircuitBreakerObject -> PEFT tensors + config fields.

    ``self_attn``/``mlp`` layouts ONLY (``oproj_site == self_attn.o_proj``):
    the seven-projection grid this export copies (q/k/v/o under
    ``self_attn``, gate/up/down under ``mlp``, on EVERY transform layer)
    does not exist on a single-mixer-per-block stack (NemotronH: each
    block is Mamba2 OR MLP OR attention, so no layer has all seven) — the
    grid check below would refuse anyway; refusing on the site first
    names the real reason instead of a "partial grid".

    The authoritative hook body (extraction/circuit_breaker.
    CircuitBreakerObject._make_hook), per adapted projection:

        delta = (x @ A.t()) @ B.t()      # A [r, d_in], B [d_out, r]
        output += scale * (lora_alpha / r) * delta

    which is PEFT's own layout and arithmetic verbatim —
    ``lora_A.weight = A``, ``lora_B.weight = B``, ``dW = (alpha/r)·B@A``
    — so the tensors copy across with NO reorientation; only the shapes
    are asserted. The hook applies unconditionally (every forward, every
    position, no mask), so the fit is all-position by construction.
    ``base_scale`` must be exactly 1.0 (the class pins it — 'LoRA
    magnitude is LEARNED'): folding any other value into lora_alpha
    would reassociate the scaling arithmetic ((alpha·s)/r vs s·(alpha/r))
    and the gate must prove EXACT equality, so a non-1.0 value refuses
    rather than exporting a one-ulp-off adapter."""
    from ...extraction.circuit_breaker import PROJECTIONS

    kind = "circuit_breaker"
    if oproj_site != DEFAULT_OPROJ_SITE:
        raise LoraExportError(
            f"kind {kind!r}: the seven-projection circuit-breaker grid is a "
            f"self_attn/mlp layout; a base whose attention site is "
            f"{oproj_site!r} (a single-mixer-per-block stack) has no layer "
            "carrying all seven projections, so no PEFT adapter can express "
            "this object there — refusing rather than naming modules the "
            "served model does not have")
    required = {"kind", "transform_layers", "target_layers", "rank",
                "lora_alpha", "base_scale", "A", "B", "metadata"}
    missing = required - set(blob)
    if missing:
        raise LoraExportError(
            f"kind {kind!r}: artifact lacks {sorted(missing)} — not a "
            "CircuitBreakerObject.save blob")
    unknown = set(blob) - required
    if unknown:
        raise LoraExportError(
            f"kind {kind!r}: artifact carries unexpected keys "
            f"{sorted(unknown)} — the representation drifted past what "
            "this export was verified against; refusing to guess")

    raw_transform = blob["transform_layers"]
    if not isinstance(raw_transform, (list, tuple)) or not raw_transform:
        raise LoraExportError(
            f"kind {kind!r}: transform_layers must be a non-empty list, "
            f"got {raw_transform!r}")
    transform_layers = []
    for l in raw_transform:
        li = _blob_int(kind, "transform_layers entry", l, minimum=0)
        if li in transform_layers:
            raise LoraExportError(
                f"kind {kind!r}: duplicate transform layer {li}")
        transform_layers.append(li)
    raw_target = blob["target_layers"]
    if not isinstance(raw_target, (list, tuple)):
        raise LoraExportError(
            f"kind {kind!r}: target_layers must be a list, got "
            f"{raw_target!r}")
    target_layers = [_blob_int(kind, "target_layers entry", l, minimum=0)
                     for l in raw_target]
    rank = _blob_int(kind, "rank", blob["rank"], minimum=1)
    lora_alpha = blob["lora_alpha"]
    if not isinstance(lora_alpha, (int, float)) \
            or isinstance(lora_alpha, bool) \
            or not math.isfinite(float(lora_alpha)) or lora_alpha <= 0:
        raise LoraExportError(
            f"kind {kind!r}: lora_alpha must be a finite number > 0, got "
            f"{lora_alpha!r}")
    base_scale = blob["base_scale"]
    if base_scale != 1.0:
        raise LoraExportError(
            f"kind {kind!r}: base_scale={base_scale!r} != 1.0 — the class "
            "pins base_scale at 1.0 (the LoRA magnitude is learned), and "
            "folding any other value into lora_alpha would reassociate "
            "the scaling arithmetic ((alpha*s)/r vs s*(alpha/r)) — not "
            "exactly the hook's math; refusing rather than exporting a "
            "one-ulp-off adapter")
    if not isinstance(blob["metadata"], Mapping):
        raise LoraExportError(
            f"kind {kind!r}: metadata must be a mapping, got "
            f"{type(blob['metadata']).__name__}")

    # The pinned Llama-family base has every one of the seven projections
    # on every layer, so the save writes the FULL grid; a partial grid
    # means another architecture (or a corrupt blob), and short-name
    # target_modules + layers_to_transform could not express it anyway.
    expected_keys = {f"{l}.{name}" for l in transform_layers
                     for name in PROJECTIONS}
    for group_name in ("A", "B"):
        group = blob[group_name]
        if not isinstance(group, Mapping):
            raise LoraExportError(
                f"kind {kind!r}: {group_name} must be a mapping, got "
                f"{type(group).__name__}")
        if set(group) != expected_keys:
            raise LoraExportError(
                f"kind {kind!r}: {group_name} keys do not form the full "
                f"seven-projection grid over transform_layers "
                f"{sorted(transform_layers)} (missing: "
                f"{sorted(expected_keys - set(group))}, unexpected: "
                f"{sorted(set(group) - expected_keys)}) — a partial grid "
                "is another architecture or a corrupt blob, and short-"
                "name target_modules cannot express it; refusing")

    _spec_check(kind, resolved_spec, "rank", rank)
    _spec_check(kind, resolved_spec, "lora_alpha", float(lora_alpha))
    _spec_check(kind, resolved_spec, "transform_layers", transform_layers)
    _spec_check(kind, resolved_spec, "target_layers", target_layers)

    tensors: Dict[str, torch.Tensor] = {}
    dims: Dict[str, Tuple[int, int]] = {}
    for l in transform_layers:
        for name in PROJECTIONS:
            key = f"{l}.{name}"
            a = _saved_f32(kind, f"A[{key}]", blob["A"][key], dims=2)
            b = _saved_f32(kind, f"B[{key}]", blob["B"][key], dims=2)
            if a.shape[0] != rank or b.shape[1] != rank:
                raise LoraExportError(
                    f"kind {kind!r}: {key} has A {tuple(a.shape)} / B "
                    f"{tuple(b.shape)} but the blob declares rank {rank} "
                    "(A must be [rank, d_in], B [d_out, rank]) — refusing "
                    "a rank the config would mis-declare")
            d = (a.shape[1], b.shape[0])
            if dims.setdefault(name, d) != d:
                raise LoraExportError(
                    f"kind {kind!r}: {key} has (d_in, d_out)={d} but an "
                    f"earlier layer's {name} had {dims[name]} — one model's "
                    "decoder layers share projection shapes; corrupt blob")
            parent = "self_attn" if name in _ATTN_PROJECTIONS else "mlp"
            module = f"{decoder_prefix}.{l}.{parent}.{name}"
            tensors[f"base_model.model.{module}.lora_A.weight"] = \
                a.contiguous()
            tensors[f"base_model.model.{module}.lora_B.weight"] = \
                b.contiguous()

    config = {
        "r": rank,
        "lora_alpha": float(lora_alpha),   # scaling = alpha/r, exactly the
                                           # hook's self.scaling
        "target_modules": sorted(PROJECTIONS),
        "layers_to_transform": sorted(transform_layers),
        # honest provenance (contract: never petri_zero_adapter)
        "petri_source_kind": kind,
        "petri_source_layers": sorted(transform_layers),
        "petri_source_target_layers": sorted(target_layers),
        "petri_source_base_scale": 1.0,
        "petri_fit_positions": "all",
    }
    return tensors, config


# ==========================================================================
# the residual-slot kinds
# ==========================================================================


@functools.lru_cache(maxsize=8)
def _served_generation_header_ids(base_model: str,
                                  revision: Optional[str]) -> Tuple[int, ...]:
    """The SERVED assistant header's exact token ids — what a ``response``
    payload's boundary is LOCATED by (native-psr-serving §1.8), derived from
    the base's own tokenizer through the ONE function every other serving seam
    uses: ``chat_markers.generation_header_ids`` (the id-form of
    ``hooks.response_header_len``'s width — the tokens
    ``add_generation_prompt`` APPENDS, never
    ``derive_assistant_markers().header``, which is anchored at the previous
    turn's terminator and absorbs prompt-side scaffold, so serving it would
    steer one token OUTSIDE the fit support; chat_markers' docstring, measured
    on gemma-4: located 4 vs appended 3).

    The tokenizer is loaded exactly as the standup loads the served one
    (``lora_serving_up.bring_up_lora``: ``AutoTokenizer.from_pretrained(model,
    revision=rev)`` then ``models.pin_reasoning_template_off``), so the ids
    describe the render the server will actually produce — on a reasoning
    template an unpinned tokenizer renders thinking-ON and its header is a
    DIFFERENT width (Qwen3.5: 5 vs the served 7). Local cache only: a base
    being served always has its snapshot cached, so a miss is a real
    inconsistency and refuses by name rather than reaching the network.

    Cached per ``(base_model, revision)`` — the derivation is pure and a
    multi-cell bank export would otherwise reload a ~17 MB tokenizer per
    cell."""
    from ...chat_markers import ChatMarkerError, generation_header_ids
    from ...models import pin_reasoning_template_off

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model, revision=revision,
                                            local_files_only=True)
    except Exception as e:  # transformers raises several types here
        raise LoraExportError(
            f"cannot load {base_model!r}"
            f"{' @ ' + revision if revision else ''}'s tokenizer from the local HF "
            f"cache ({type(e).__name__}: {e}) — a response-mode residual payload's "
            "boundary is located by the SERVED assistant header's token ids, which "
            "are derived from that tokenizer's chat template; refusing rather than "
            "banking a payload whose header ids were guessed") from e
    pin_reasoning_template_off(tok)
    try:
        ids = tuple(int(t) for t in generation_header_ids(tok))
    except ChatMarkerError as e:
        raise LoraExportError(
            f"{base_model!r}: the served chat template exposes no derivable "
            f"generation-prompt assistant header ({e}) — a response-mode payload "
            "cannot be located on it at all; the lens refuses the same template "
            "and so does the residual slot") from e
    if not ids:
        raise LoraExportError(
            f"{base_model!r}: the served generation-prompt header derived EMPTY — "
            "a response mask located by an empty id sequence would match every "
            "position; refusing")
    return ids


def _derived_dims(base_model: str, revision: Optional[str] = None) -> Tuple[int, int]:
    """``(hidden_size, n_layers_model)`` of ``base_model`` off ONE meta skeleton
    (:func:`_decoder_skeleton`): the residual stream's width (``models.
    hidden_size``, text config on the VLM wrappers) and the decoder block
    count — the TEXT config's ``num_hidden_layers`` (what the worker's bank
    sizes its boundary axis from, ``vllm_config.model_config.hf_text_config``)
    cross-checked against the tree's ``ModuleList`` length; a disagreement
    refuses (the bank would refuse the payload at load anyway)."""
    from ...models import hidden_size, num_layers

    cfg, skeleton = _decoder_skeleton(base_model, revision)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    n_cfg = getattr(text_cfg, "num_hidden_layers", None)
    n_tree = num_layers(skeleton)
    if not isinstance(n_cfg, int) or n_cfg != n_tree:
        raise LoraExportError(
            f"{base_model!r}: the text config says num_hidden_layers={n_cfg!r} but the module "
            f"tree carries {n_tree} decoder blocks — the residual slot sizes its boundary axis "
            "from the config and hooks the tree; the two must agree; refusing")
    return int(hidden_size(skeleton)), int(n_tree)


def _residual_f32_1d(kind: str, name: str, t: Any, dim: Optional[int]) -> torch.Tensor:
    """A payload vector exactly as the object's ``save`` writes it: fp32, 1-D,
    finite, ``dim`` wide once the first fixes it (:func:`_saved_f32`'s
    never-convert rule; the lens's ``_f32_1d`` shape rules)."""
    t = _saved_f32(kind, name, t, dims=1)
    if dim is not None and t.shape[0] != dim:
        raise LoraExportError(
            f"kind {kind!r}: {name} has dim {t.shape[0]}, expected {dim}")
    return t


def _residual_refuse_sidecar(kind: str, sidecar_bytes: bytes) -> None:
    """The lens's ``_refuse_sidecar``: a torch-blob residual kind takes no
    sidecar (it is hashed into the identity; never silently dropped)."""
    if sidecar_bytes:
        raise LoraExportError(
            f"kind {kind!r} takes no sidecar bytes, got {len(bytes(sidecar_bytes))} — a "
            "behavior-bearing sidecar on this kind means the entry is not what this export "
            "translates (the sidecar is hashed into the intervention identity; it must never "
            "be silently dropped)")


def _residual_recorded_norms(kind: str, blob: Mapping[str, Any], key: str
                             ) -> Dict[int, float]:
    """``lens_adapters._recorded_layer_norms`` + ``_assert_norms_discriminate``:
    the per-layer norms recorded at fit time when present (absence is fine),
    refused when malformed or non-discriminating."""
    meta = blob.get("metadata")
    if not isinstance(meta, Mapping):
        return {}
    recorded = meta.get(key)
    if not isinstance(recorded, Mapping):
        return {}
    out: Dict[int, float] = {}
    for k, v in recorded.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError) as e:
            raise LoraExportError(
                f"kind {kind!r}: metadata[{key!r}] entry {k!r}={v!r} is not a layer->number "
                "pair; refusing to skip an integrity check because its own record is "
                "malformed") from e
    if len(out) >= 2:
        vals = sorted(out.values())
        gap = min(b - a for a, b in zip(vals, vals[1:]))
        if gap <= _RESIDUAL_NORM_ATOL * 10:
            raise LoraExportError(
                f"kind {kind!r}: the recorded per-layer norms do not discriminate between "
                f"layers (smallest gap {gap:.6g}, needs > {_RESIDUAL_NORM_ATOL * 10:g}) — a "
                "cross-check against them would pass for EVERY permutation of the layers; "
                "refusing rather than exporting under a vacuous integrity check")
    return out


#: ``lens_adapters._NORM_ATOL`` — how far a loaded vector's norm may sit from
#: the norm recorded for that layer at fit time
_RESIDUAL_NORM_ATOL = 1e-3


def _residual_check_norm(kind: str, layer: int, t: torch.Tensor,
                         recorded: Mapping[int, float]) -> None:
    want = recorded.get(layer)
    if want is None:
        return
    got = float(t.detach().float().norm())
    if abs(got - want) > _RESIDUAL_NORM_ATOL:
        raise LoraExportError(
            f"kind {kind!r}: the tensor loaded for layer {layer} has norm {got:.6f} but the "
            f"artifact records {want:.6f} for that layer (tolerance {_RESIDUAL_NORM_ATOL}) — "
            "the tensor and the layer it is being served under disagree (the usual cause is "
            "keying the lookup by the SHIFTED serving index rather than the fit layer); "
            "refusing at EXPORT")


def _residual_layer_key(kind: str, group: Any, group_name: str, layer: int) -> Any:
    """The entry for ``layer`` in a per-layer dict keyed by str OR int (the
    kinds differ: PSR / vector_all save str keys, diffmean / static_r int
    keys); a declared layer absent from the dict refuses naming it."""
    if not isinstance(group, Mapping):
        raise LoraExportError(
            f"kind {kind!r}: artifact {group_name!r} is not a mapping")
    for key in (layer, str(layer)):
        if key in group:
            return group[key]
    raise LoraExportError(
        f"kind {kind!r}: artifact {group_name!r} lacks layer {layer} — inconsistent with "
        "its own layer list")


def _residual_layers_of(kind: str, group: Any, group_name: str) -> List[int]:
    if not isinstance(group, Mapping) or not group:
        raise LoraExportError(
            f"kind {kind!r}: artifact lacks a non-empty {group_name!r} mapping — not this "
            "object's save blob")
    try:
        layers = sorted(int(k) for k in group)
    except (TypeError, ValueError) as e:
        raise LoraExportError(
            f"kind {kind!r}: artifact {group_name!r} carries a non-integer layer key") from e
    if any(l < 0 for l in layers):
        raise LoraExportError(f"kind {kind!r}: layer {min(layers)} < 0")
    return layers


def _residual_psr(resolved_spec, blob, *, hidden_size):
    """``lens_adapters._adapt_psr``'s checks: the four keys, the layer list
    (cross-checked with the spec), per layer a direction and gate vector of
    the model width and a single-element finite gate bias; the mode from the
    blob's FIT leg (``positions``, default ``all``) with the spec's override
    honoured only as the lens honours it (a response serve needs the header
    width; the width recorded at fit must agree). Gated op, coeff 1.0, no BOS
    skip (the gate is learned)."""
    from . import native_residual as nr

    kind = "psr"
    for key in ("layers", "directions", "gate_w", "gate_b"):
        if key not in blob:
            raise LoraExportError(
                f"kind 'psr': artifact lacks {key!r} — not a PSRObject.save blob")
    raw = blob["layers"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise LoraExportError(f"kind 'psr': 'layers' must be a non-empty list, got {raw!r}")
    layers: List[int] = []
    for l in raw:
        li = int(l)
        if li < 0:
            raise LoraExportError(f"kind 'psr': layer {li} < 0")
        if li in layers:
            raise LoraExportError(f"kind 'psr': duplicate layer {li}")
        layers.append(li)
    if "layers" in resolved_spec:
        spec_layers = [int(l) for l in resolved_spec["layers"]]
        if spec_layers != layers:
            raise LoraExportError(
                f"kind 'psr': resolved_spec layers {spec_layers} != artifact layers {layers} "
                "(F9-style cross-check)")
    fit_positions = blob.get("positions", "all")
    mode = resolved_spec.get("positions", fit_positions)
    header = blob.get("response_header_len")
    if header is not None and (isinstance(header, bool) or not isinstance(header, int)
                               or header <= 0):
        raise LoraExportError(
            f"kind 'psr': response_header_len must be a positive int when present, got "
            f"{header!r} — an H <= 0 boundary silently disables the whole steered region")
    _spec_check(kind, resolved_spec, "response_header_len", header)
    tensors: Dict[str, torch.Tensor] = {}
    d = hidden_size
    for l in layers:
        direction = _residual_f32_1d(kind, f"directions[{l}]",
                                     _residual_layer_key(kind, blob["directions"], "directions", l), d)
        gate_w = _residual_f32_1d(kind, f"gate_w[{l}]",
                                  _residual_layer_key(kind, blob["gate_w"], "gate_w", l), d)
        gate_b = _residual_layer_key(kind, blob["gate_b"], "gate_b", l)
        if not isinstance(gate_b, torch.Tensor) or gate_b.numel() != 1:
            raise LoraExportError(f"kind 'psr': gate_b[{l}] must be a single-element tensor")
        if gate_b.dtype != torch.float32:
            raise LoraExportError(
                f"kind 'psr': gate_b[{l}] is {gate_b.dtype}, but the object's save path writes "
                "float32 — not this object's blob; refusing")
        if not torch.isfinite(gate_b).all():
            raise LoraExportError(f"kind 'psr': gate_b[{l}] is non-finite")
        tensors[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_w"] = gate_w.contiguous()
        tensors[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_b"] = gate_b.detach().reshape(1).contiguous()
        tensors[f"{nr.PAYLOAD_PREFIX}layers.{l}.direction"] = direction.contiguous()
    return dict(layers=layers, boundaries=list(layers), mode=mode, header_len=header,
                skip_bos=False, coeff=1.0, op=nr.OP_RELU_GATED_VECTOR, tensors=tensors,
                fit_positions=fit_positions)


def _paired_side_tensors(kind: str, tag: str, side: Any, layers: List[int], d: int,
                         fields: Tuple[str, str, str]) -> Dict[str, torch.Tensor]:
    """One side of a paired-write artifact (``PSRPairedWriteObject._side``'s shape: per-layer
    ``directions`` / ``gate_w`` / ``gate_b`` under str keys) -> the payload tensors under
    ``fields`` = (gate field, bias field, direction field), every layer of ``layers`` present,
    the model width, fp32, finite — ``_residual_psr``'s own rules, per side, so a side lacking
    a layer or of another width refuses NAMING the side and the layer."""
    from . import native_residual as nr

    if not isinstance(side, Mapping):
        raise LoraExportError(f"kind {kind!r}: artifact side {tag!r} is not a mapping")
    for key in ("directions", "gate_w", "gate_b"):
        if key not in side:
            raise LoraExportError(f"kind {kind!r}: side {tag!r} lacks {key!r} — not a PSRPairedWriteObject.save blob")
    out: Dict[str, torch.Tensor] = {}
    gate_field, bias_field, dir_field = fields
    for l in layers:
        for group in ("directions", "gate_w", "gate_b"):
            if str(l) not in side[group] and l not in side[group]:
                raise LoraExportError(
                    f"kind {kind!r}: the {tag} side lacks layer {l} in {group!r} — the two sides' layer sets "
                    f"differ (the paired write is defined layerwise on ONE layer set); refusing")
        direction = _residual_f32_1d(kind, f"{tag}.directions[{l}]",
                                     _residual_layer_key(kind, side["directions"], f"{tag}.directions", l), d)
        gate_w = _residual_f32_1d(kind, f"{tag}.gate_w[{l}]",
                                  _residual_layer_key(kind, side["gate_w"], f"{tag}.gate_w", l), d)
        gate_b = _residual_layer_key(kind, side["gate_b"], f"{tag}.gate_b", l)
        if not isinstance(gate_b, torch.Tensor) or gate_b.numel() != 1:
            raise LoraExportError(f"kind {kind!r}: {tag}.gate_b[{l}] must be a single-element tensor")
        if gate_b.dtype != torch.float32:
            raise LoraExportError(
                f"kind {kind!r}: {tag}.gate_b[{l}] is {gate_b.dtype}, but the object's save path writes "
                "float32 — not this object's blob; refusing")
        if not torch.isfinite(gate_b).all():
            raise LoraExportError(f"kind {kind!r}: {tag}.gate_b[{l}] is non-finite")
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.{gate_field}"] = gate_w.contiguous()
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.{bias_field}"] = gate_b.detach().reshape(1).contiguous()
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.{dir_field}"] = direction.contiguous()
    return out


def _residual_psr_paired_write(resolved_spec, blob, *, hidden_size):
    """THE PAIRED WRITE's artifact (``extraction.paired_write.PSRPairedWriteObject.save`` /
    ``compose_blob``: ``kind``, ``layers``, ``application_order``, the ``pos`` and ``neg``
    sides, ``provenance``, ``positions``) -> the slot's two-gate payload (op
    ``relu_gated_vector_pair``): the ``pos`` side under ``gate_w / gate_b / direction``, the
    ``neg`` side under ``gate_w2 / gate_b2 / direction2``, BOTH parents' object shas from the
    artifact's own provenance (the builder records them; a fitted object is never banked
    without both), and the two terms' signed coefficients ``(+1, -1)`` as a stated fact.

    Refuses, naming the fact: another ``application_order`` (a sequential composition is a
    different function of the same parents); a layer set that differs between the sides or
    from the ``layers`` key; a side of another width; a masked position leg — fit or served
    (the object refuses masked parents; the write is defined at every position); a recorded
    response header; provenance lacking either parent's sha; a spec/blob layer disagreement.
    The SAME function reads a serve-time composition (:func:`compose_paired_write_blob`), so
    the two sources yield one payload — bit-identical where the parents are the fitted
    object's (the test pins this on the real llama8b and Qwen objects)."""
    from . import native_residual as nr

    kind = "psr_paired_write"
    if blob.get("kind") != kind:
        raise LoraExportError(
            f"kind {kind!r}: artifact stores kind {blob.get('kind')!r} — not a PSRPairedWriteObject blob")
    for key in ("layers", "pos", "neg", "application_order", "provenance"):
        if key not in blob:
            raise LoraExportError(f"kind {kind!r}: artifact lacks {key!r} — not a PSRPairedWriteObject.save blob")
    order = blob["application_order"]
    if order != nr.PAIRED_APPLICATION_ORDER:
        raise LoraExportError(
            f"kind {kind!r}: application_order={order!r}; the residual slot serves ONLY "
            f"{nr.PAIRED_APPLICATION_ORDER!r} (both gates read the shared pre-write state, one combined "
            "add) — a sequential two-hook composition is a different function of the same parents; refusing")
    raw = blob["layers"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise LoraExportError(f"kind {kind!r}: 'layers' must be a non-empty list, got {raw!r}")
    layers: List[int] = []
    for l in raw:
        li = int(l)
        if li < 0:
            raise LoraExportError(f"kind {kind!r}: layer {li} < 0")
        if li in layers:
            raise LoraExportError(f"kind {kind!r}: duplicate layer {li}")
        layers.append(li)
    if "layers" in resolved_spec:
        spec_layers = [int(l) for l in resolved_spec["layers"]]
        if spec_layers != layers:
            raise LoraExportError(
                f"kind {kind!r}: resolved_spec layers {spec_layers} != artifact layers {layers} "
                "(F9-style cross-check)")
    fit_positions = blob.get("positions", "all")
    mode = resolved_spec.get("positions", fit_positions)
    if fit_positions != "all" or mode != "all":
        raise LoraExportError(
            f"kind {kind!r}: positions fit={fit_positions!r} served={mode!r} — the paired write is defined "
            "on all-position parents only (PSRPairedWriteObject refuses a masked parent: a difference of two "
            "writes has no mask of its own) and the slot serves it at every position; refusing")
    if blob.get("response_header_len") is not None:
        raise LoraExportError(
            f"kind {kind!r}: response_header_len={blob.get('response_header_len')!r} on an all-position "
            "paired artifact — not a PSRPairedWriteObject blob; refusing")
    # the two sides share ONE layer set and ONE width: every layer of `layers` on both sides
    for tag in ("pos", "neg"):
        side = blob[tag]
        if isinstance(side, Mapping) and isinstance(side.get("directions"), Mapping):
            extra = sorted(int(k) for k in side["directions"]) if all(str(k).lstrip("-").isdigit() for k in side["directions"]) else None
            if extra is not None and extra != sorted(layers):
                raise LoraExportError(
                    f"kind {kind!r}: the {tag} side carries layers {extra} but the artifact declares "
                    f"{sorted(layers)} — the two sides' layer sets differ from the declared set; refusing")
    d = hidden_size
    tensors: Dict[str, torch.Tensor] = {}
    tensors.update(_paired_side_tensors(kind, "pos", blob["pos"], layers, d, ("gate_w", "gate_b", "direction")))
    tensors.update(_paired_side_tensors(kind, "neg", blob["neg"], layers, d, ("gate_w2", "gate_b2", "direction2")))
    prov = blob["provenance"]
    if not isinstance(prov, Mapping):
        raise LoraExportError(f"kind {kind!r}: provenance is not a mapping")
    shas: Dict[str, str] = {}
    for role in ("parent_pos", "parent_neg"):
        blk = prov.get(role)
        sha = blk.get("object_sha256") if isinstance(blk, Mapping) else None
        if not nr.is_sha256_hex(sha):
            raise LoraExportError(
                f"kind {kind!r}: provenance.{role}.object_sha256 is {sha!r}, not a 64-hex sha256 — a paired "
                "payload names BOTH parents' object bytes (psr_residual_distill.py records them; an ad-hoc "
                "object with no parent identity cannot be banked); refusing")
        shas[role] = str(sha)
    if shas["parent_pos"] == shas["parent_neg"]:
        raise LoraExportError(
            f"kind {kind!r}: both sides name the same parent ({shas['parent_pos'][:12]}...) — the paired write "
            "of an object with itself is identically zero; refusing to bank a no-op")
    pos_coeff, neg_coeff = nr.PAIRED_TERM_COEFFS
    return dict(layers=layers, boundaries=list(layers), mode="all", header_len=None,
                skip_bos=False, coeff=1.0, op=nr.OP_RELU_GATED_VECTOR_PAIR, tensors=tensors,
                fit_positions="all",
                parent_pos_sha256=shas["parent_pos"], parent_neg_sha256=shas["parent_neg"],
                parent_pos_coeff=pos_coeff, parent_neg_coeff=neg_coeff)


def compose_paired_write_blob(pos_blob: Mapping[str, Any], neg_blob: Mapping[str, Any], *,
                              pos_sha256: str, neg_sha256: str,
                              pos_cell: Optional[str] = None, neg_cell: Optional[str] = None) -> dict:
    """THE SERVE-TIME COMPOSITION (the paired route's second source): two single-gate ``psr``
    parent blobs (``PSRObject.save``'s shape — the corpus fit as ``pos``, the register-matched
    control fit as ``neg``) -> a ``psr_paired_write`` artifact dict of EXACTLY the shape the
    builder writes, through the object's own ``PSRPairedWriteObject.compose_blob`` (tensors
    cloned bit-exact, every structural precondition validated there), with a provenance block
    naming both parents by the sha256 of their object BYTES (``pos_sha256`` / ``neg_sha256``
    — computed by the caller over the bytes it read, never taken from a sidecar). Feeding the
    result to :func:`_residual_psr_paired_write` yields the SAME payload the fitted artifact
    yields when the parents are the fitted object's: same tensors (the builder used this same
    ``compose_blob``), same parent shas (the builder recorded the same bytes' digests).

    Refuses (as ``LoraExportError``): parents of differing layer sets or shapes, a masked
    parent, a non-``psr`` parent (the object's own refusals, re-raised), a malformed sha, and
    the same sha on both sides (a no-op composition)."""
    from . import native_residual as nr
    from ...extraction.paired_write import APPLICATION_ORDER, PSRPairedWriteObject

    for name, sha in (("pos_sha256", pos_sha256), ("neg_sha256", neg_sha256)):
        if not nr.is_sha256_hex(sha):
            raise LoraExportError(
                f"compose_paired_write_blob: {name}={sha!r} is not a lowercase 64-hex sha256 of the "
                "parent's object bytes")
    if pos_sha256 == neg_sha256:
        raise LoraExportError(
            f"compose_paired_write_blob: both sides name the same parent bytes ({pos_sha256[:12]}...) — the "
            "paired write of an object with itself is identically zero; refusing to compose a no-op")
    provenance = {
        "kind": PSRPairedWriteObject.kind,
        "application_order": APPLICATION_ORDER,
        "positions": "all",
        "composition": "serve_time",
        "builder": "sisteer.capeval.serving.lora_export.compose_paired_write_blob",
        "parent_pos": {"object_sha256": str(pos_sha256), **({"cell_id": pos_cell} if pos_cell else {})},
        "parent_neg": {"object_sha256": str(neg_sha256), **({"cell_id": neg_cell} if neg_cell else {})},
    }
    try:
        return PSRPairedWriteObject.compose_blob(dict(pos_blob), dict(neg_blob), provenance=provenance)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise LoraExportError(f"compose_paired_write_blob: {e}") from e


def _residual_vector_all(resolved_spec, blob, *, hidden_size):
    """``lens_adapters._adapt_vector_all``: a non-empty ``vectors`` mapping, the
    stored ``skip_bos`` (bool, default True; spec-checked), the DEPLOYMENT
    ``positions`` (the spec overrides the artifact's default) with the FIT leg
    ``fit_positions`` cross-checked (a masked fit deployed at every position
    refuses); constant op, coeff 1.0."""
    from . import native_residual as nr

    kind = "vector_all"
    layers = _residual_layers_of(kind, blob.get("vectors"), "vectors")
    stored_positions = blob.get("positions", "all")
    skip_bos = blob.get("skip_bos", True)
    if not isinstance(skip_bos, bool):
        raise LoraExportError(
            f"kind 'vector_all': stored skip_bos must be a bool, got {skip_bos!r}")
    _spec_check(kind, resolved_spec, "skip_bos", skip_bos)
    fit = blob.get("fit_positions", "all")
    if fit not in ("all", "response"):
        raise LoraExportError(
            f"kind 'vector_all': fit_positions={fit!r} is not a fit leg ('all' / 'response')")
    mode = resolved_spec.get("positions", stored_positions)
    if fit != "all" and mode == "all":
        raise LoraExportError(
            "kind 'vector_all': the object was fit at 'response' positions (fit_positions) but "
            "would be deployed at every position — the fit=response / serve=all corner is a "
            "different intervention from the one that was fit; refusing")
    header = (blob.get("fit_response_header_len") if fit == "response"
              else blob.get("response_header_len"))
    tensors: Dict[str, torch.Tensor] = {}
    for l in layers:
        vec = _residual_f32_1d(kind, f"vectors[{l}]",
                               _residual_layer_key(kind, blob["vectors"], "vectors", l), hidden_size)
        tensors[f"{nr.PAYLOAD_PREFIX}layers.{l}.vector"] = vec.contiguous()
    return dict(layers=layers, boundaries=list(layers), mode=mode, header_len=header,
                skip_bos=skip_bos, coeff=1.0, op=nr.OP_ADD_VECTOR, tensors=tensors,
                fit_positions=fit)


def _residual_diffmean(resolved_spec, blob, *, hidden_size):
    """``DiffMeanObject.save``: int-keyed ``vectors``, ``skip_bos`` (default
    True), the persisted deployment ``positions`` (default ``all``, ==
    the fit leg — a closed-form fit) and ``response_header_len``; the lens has
    no adapter for it (spec: it becomes servable for the first time), so the
    checks are the object's own loader's."""
    from . import native_residual as nr

    kind = "diffmean"
    layers = _residual_layers_of(kind, blob.get("vectors"), "vectors")
    skip_bos = blob.get("skip_bos", True)
    if not isinstance(skip_bos, bool):
        raise LoraExportError(
            f"kind 'diffmean': stored skip_bos must be a bool, got {skip_bos!r}")
    _spec_check(kind, resolved_spec, "skip_bos", skip_bos)
    positions = blob.get("positions", "all")
    _spec_check(kind, resolved_spec, "positions", positions)
    header = blob.get("response_header_len")
    tensors: Dict[str, torch.Tensor] = {}
    for l in layers:
        vec = _residual_f32_1d(kind, f"vectors[{l}]",
                               _residual_layer_key(kind, blob["vectors"], "vectors", l), hidden_size)
        tensors[f"{nr.PAYLOAD_PREFIX}layers.{l}.vector"] = vec.contiguous()
    return dict(layers=layers, boundaries=list(layers), mode=positions, header_len=header,
                skip_bos=skip_bos, coeff=1.0, op=nr.OP_ADD_VECTOR, tensors=tensors,
                fit_positions=positions)


def _residual_static_r(resolved_spec, blob, *, hidden_size):
    """``lens_adapters._adapt_static_r``: ``base_scale`` (the baked coeff),
    ``steer_positions`` / ``site`` (validated, spec-checked), the direction
    dict ``rs`` (int keys) or the legacy ``layer``/``r`` pair, the declared
    ``layers`` cross-check, the recorded per-layer norms (discriminating, each
    within tolerance), the boundary shift for ``site='pre'`` (one block
    earlier; layer 0 refuses), and the cross-phase cell (spec
    ``positions == 'response'``)."""
    from . import native_residual as nr

    kind = "static_r"
    if "base_scale" not in blob:
        raise LoraExportError(
            "kind 'static_r': artifact lacks 'base_scale' — not a StaticRObject.save blob")
    base_scale = blob["base_scale"]
    if isinstance(base_scale, bool) or not isinstance(base_scale, (int, float)) \
            or base_scale != base_scale or base_scale in (float("inf"), float("-inf")):
        raise LoraExportError(
            f"kind 'static_r': base_scale must be a finite number, got {base_scale!r}")
    steer_positions = blob.get("steer_positions") or "prompt_last"
    site = blob.get("site") or "post"
    if steer_positions not in ("prompt_last", "prefill_all"):
        raise LoraExportError(
            f"kind 'static_r': unknown steer_positions {steer_positions!r} (expected "
            "'prompt_last' or 'prefill_all') — the mask is the intervention; refusing to guess "
            "it")
    if site not in ("post", "pre"):
        raise LoraExportError(
            f"kind 'static_r': unknown hook site {site!r} (expected 'pre' or 'post') — the "
            "site decides which block's output carries the edit, so guessing it would "
            "silently steer the wrong layer")
    _spec_check(kind, resolved_spec, "steer_positions", steer_positions)
    _spec_check(kind, resolved_spec, "site", site)
    rs = blob.get("rs")
    if isinstance(rs, Mapping) and rs:
        layers = _residual_layers_of(kind, rs, "rs")
        if blob.get("layers"):
            declared = sorted(int(l) for l in blob["layers"])
            if declared != layers:
                raise LoraExportError(
                    f"kind 'static_r': blob declares layers {declared} but 'rs' carries "
                    f"{layers} — inconsistent artifact; refusing rather than serving whichever "
                    "one happens to be read first")
        vectors = {l: _residual_layer_key(kind, rs, "rs", l) for l in layers}
    else:
        if blob.get("layer") is None or blob.get("r") is None:
            raise LoraExportError(
                "kind 'static_r': artifact has neither a non-empty 'rs' mapping nor a legacy "
                "'layer'/'r' pair — refusing an artifact with no direction to serve")
        layers = [int(blob["layer"])]
        vectors = {layers[0]: blob["r"]}
    stated = resolved_spec.get("positions")
    if stated is not None and stated != "response":
        raise LoraExportError(
            f"kind 'static_r': stated positions={stated!r} — the lens honours only "
            "positions='response' (the cross-phase cell) on this kind; refusing an ambiguous "
            "record")
    mode = "response" if stated == "response" else steer_positions
    recorded = _residual_recorded_norms(kind, blob, "per_layer_r_norm")
    boundaries: List[int] = []
    tensors: Dict[str, torch.Tensor] = {}
    for l in layers:
        vec = _residual_f32_1d(kind, f"rs[{l}]", vectors[l], hidden_size)
        _residual_check_norm(kind, l, vec, recorded)
        if site == "pre":
            if l == 0:
                raise LoraExportError(
                    "kind 'static_r': fit at resid_pre of layer 0, which is the embedding "
                    "output — there is no preceding decoder block whose output could carry "
                    "the edit; the lens refuses it and so does the residual slot (D6). Serving "
                    "it one block later would be a different intervention, not a rounding of "
                    "this one")
            boundaries.append(l - 1)
        else:
            boundaries.append(l)
        tensors[f"{nr.PAYLOAD_PREFIX}layers.{l}.vector"] = vec.contiguous()
    return dict(layers=layers, boundaries=boundaries, mode=mode,
                header_len=resolved_spec.get("response_header_len"), skip_bos=False,
                coeff=float(base_scale), op=nr.OP_ADD_VECTOR, tensors=tensors,
                fit_positions="all")


def _residual_vector(resolved_spec, artifact_bytes, sidecar_bytes, *, hidden_size):
    """``lens_adapters._adapt_vector`` over the SteeringVector container
    (``layer_<k>`` tensors + the REQUIRED JSON sidecar): refuses an antidote
    method, ``projection_targets``, ``norm_relative``; ``positions`` /
    ``skip_bos`` from the spec (default ``all`` / True) cross-checked with the
    ``vector_object`` record ``VectorObject.save`` persists."""
    from . import native_residual as nr

    kind = "vector"
    if not artifact_bytes:
        raise LoraExportError(f"kind {kind!r}: empty artifact bytes")
    if not sidecar_bytes:
        raise LoraExportError(
            f"kind {kind!r}: the SteeringVector JSON sidecar is REQUIRED — it carries the "
            "method the loader dispatches on, and two containers differing only in sidecar "
            "are different interventions")
    try:
        from safetensors.torch import load as _st_load
        raw = _st_load(bytes(artifact_bytes))
    except Exception as e:
        raise LoraExportError(
            f"kind {kind!r}: artifact bytes did not load as a safetensors container: {e}") from e
    try:
        side = json.loads(bytes(sidecar_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LoraExportError(f"kind {kind!r}: unparseable JSON sidecar: {e}") from e
    if not isinstance(side, dict):
        raise LoraExportError(f"kind {kind!r}: sidecar is not a JSON object")
    tensors_by_layer: Dict[int, Any] = {}
    for name, t in raw.items():
        if not name.startswith("layer_"):
            raise LoraExportError(
                f"kind {kind!r}: unexpected safetensors entry {name!r} — not a SteeringVector "
                "container")
        try:
            l = int(name.removeprefix("layer_"))
        except ValueError as e:
            raise LoraExportError(f"kind {kind!r}: unparseable layer key {name!r}") from e
        if l < 0:
            raise LoraExportError(f"kind {kind!r}: layer {l} < 0")
        tensors_by_layer[l] = t
    if not tensors_by_layer:
        raise LoraExportError(f"kind {kind!r}: container holds no layer vectors")
    if "layers" in side and sorted(int(l) for l in side["layers"]) != sorted(tensors_by_layer):
        raise LoraExportError(
            f"kind {kind!r}: sidecar layers {side['layers']} != container layers "
            f"{sorted(tensors_by_layer)} — inconsistent artifact")
    if side.get("method") == "antidote":
        raise LoraExportError(
            "kind 'vector': the sidecar declares method='antidote', which load_object hydrates "
            "as AntidoteObject (positions='last', a calibrated base_alpha) — a different "
            "intervention with its own adapter; refusing to serve it as a plain vector")
    meta = side.get("metadata")
    if isinstance(meta, Mapping) and meta.get("projection_targets"):
        raise LoraExportError(
            "kind 'vector': the container carries 'projection_targets', so it deploys the "
            "projection edit (each position's projection onto the unit direction is MOVED to "
            "a target value, not added to) — not an additive residual op; refusing")
    if resolved_spec.get("norm_relative"):
        raise LoraExportError(
            "kind 'vector': norm_relative=True scales the addition by each position's own "
            "hidden-state norm, so the added vector is not constant across positions — not "
            "expressible as a fixed additive op; refusing")
    vo = meta.get("vector_object") if isinstance(meta, Mapping) else None
    vo = vo if isinstance(vo, Mapping) else {}
    stated = resolved_spec.get("positions")
    if stated is not None and vo.get("positions") is not None and stated != vo["positions"]:
        raise LoraExportError(
            f"kind 'vector': the spec states positions={stated!r} but the container's "
            f"vector_object record persists positions={vo['positions']!r}; refusing to pick one")
    mode = stated if stated is not None else vo.get("positions", "all")
    skip_bos = resolved_spec.get("skip_bos", vo.get("skip_bos", True))
    if not isinstance(skip_bos, bool):
        raise LoraExportError(f"kind 'vector': skip_bos must be a bool, got {skip_bos!r}")
    if vo.get("skip_bos") is not None and skip_bos != vo["skip_bos"]:
        raise LoraExportError(
            f"kind 'vector': the spec's skip_bos={skip_bos!r} disagrees with the container's "
            f"vector_object record skip_bos={vo['skip_bos']!r}; refusing to pick one")
    if vo.get("norm_relative"):
        raise LoraExportError(
            "kind 'vector': the container's vector_object record persists norm_relative=True "
            "— not expressible as a fixed additive op; refusing")
    layers = sorted(tensors_by_layer)
    tensors: Dict[str, torch.Tensor] = {}
    for l in layers:
        vec = _residual_f32_1d(kind, f"layer_{l}", tensors_by_layer[l], hidden_size)
        tensors[f"{nr.PAYLOAD_PREFIX}layers.{l}.vector"] = vec.contiguous()
    return dict(layers=layers, boundaries=list(layers), mode=mode,
                header_len=resolved_spec.get("response_header_len"), skip_bos=skip_bos,
                coeff=1.0, op=nr.OP_ADD_VECTOR, tensors=tensors, fit_positions="all")


def _export_residual_slot(kind: str, resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                          sidecar_bytes: bytes, *, decoder_prefix: str, oproj_site: str,
                          oproj_layers: Tuple[int, ...],
                          oproj_shapes: Mapping[int, Tuple[int, int]],
                          hidden_size: int, n_layers_model: int,
                          base_model: str, base_revision: Optional[str] = None,
                          ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """A residual-slot kind -> PEFT tensors + config fields. The ANCHOR
    is ``oproj_layers[0]`` (the base's first o_proj layer: Qwen3.5 3,
    NemotronH 12, llama / gemma 0); the adapter carries a rank-1 ZERO
    ``lora_A [1, d_in]`` / ``lora_B [d_out, 1]`` pair on the anchor's
    o_proj (sized off the skeleton like every other export) — vLLM's
    slot, ``lora_int_id`` and per-token routing ride on it, and it adds
    nothing — plus the payload ``sisteer_residual.layers.<l>.{gate_w,
    gate_b, direction}`` (psr) or ``.vector`` (the constant kinds), fp32
    as the objects save it, and the ``petri_residual_slot`` block the
    worker reads (``native_residual.ResidualPayload``). The payload is
    re-read through the worker's own validator before it is written, so
    what the worker will load is exactly what was checked here; ``r=1``,
    ``lora_alpha=1.0`` keep ``--max-lora-rank`` and the kernel ceiling
    untouched.

    POSITION MODE (stage 2). ``all`` is unchanged.
    The three restricted modes export only while
    ``native_residual.STAGE2_POSITION_CLASSES`` is True — the flag the worker
    reads, so the exporter and the thing that will load the payload are armed
    by ONE switch (the backstop; the disposition skips such a cell before
    this is reached while the flag is False):

    * ``response`` (psr / diffmean / vector_all's response fit, and the
      cross-phase ``static_r`` cell): the block carries BOTH the fit's
      ``response_header_len`` and the SERVED header's exact ids
      (:func:`_served_generation_header_ids`), because the serve-time boundary
      is LOCATED by matching those ids, never counted off a width. A fit that
      recorded a width must AGREE with the served header exactly — the lens's
      own check (``_fit_leg_positions``), same message: a disagreement is
      template drift and the two masks would cover different regions. A
      response fit that recorded NO width refuses; the ONE exception is the
      cross-phase ``static_r`` cell (an all-position fit re-masked by the
      CELL, ``lens_adapters._adapt_static_r``), which has no fit width to
      honour and takes the served header's own;
    * ``prompt_last`` / ``prefill_all`` (``static_r``): no header; the
      ``site='pre'`` boundary shift (one block earlier; block 0 refuses)
      is the kind helper's.

    ``base_model`` / ``base_revision`` name the tokenizer the header ids come
    from — the same pinned base the layout and the residual width come from,
    so an export cannot mix one base's header with another's weights."""
    from . import native_residual as nr

    if kind == "vector":
        parsed = _residual_vector(resolved_spec, artifact_bytes, sidecar_bytes,
                                  hidden_size=hidden_size)
    else:
        _residual_refuse_sidecar(kind, sidecar_bytes)
        blob = _genuine_blob(kind, artifact_bytes)
        fn = {"psr": _residual_psr, "vector_all": _residual_vector_all,
              "diffmean": _residual_diffmean, "static_r": _residual_static_r,
              "psr_paired_write": _residual_psr_paired_write}[kind]
        parsed = fn(resolved_spec, blob, hidden_size=hidden_size)
    mode = parsed["mode"]
    if mode not in nr.POSITION_MODES:
        raise LoraExportError(
            f"kind {kind!r}: position mode {mode!r} is not a native residual-slot mode "
            f"({list(nr.POSITION_MODES)}); the lens serves it, the slot refuses to guess")
    if mode != "all" and not nr.STAGE2_POSITION_CLASSES:
        raise LoraExportError(
            f"kind {kind!r}: native position mode {mode!r} is stage-2 (the per-token position "
            "class is not armed on this tree) — the native disposition skips this cell by name "
            "and the lens serves it; refusing to export a position-restricted payload the "
            "worker would refuse to load")
    header_ids: List[int] = []
    header_len = parsed["header_len"] if mode == "response" else None
    if mode == "response":
        # the served boundary is LOCATED by these ids (the worker matches them
        # in the prompt); the fit's recorded width is held to them like with
        # like — both count the tokens add_generation_prompt appends
        header_ids = list(_served_generation_header_ids(base_model, base_revision))
        if header_len is None:
            if kind != "static_r":
                raise LoraExportError(
                    f"kind {kind!r}: position mode 'response' but the record carries no "
                    "response_header_len — the served region starts at the assistant header, "
                    "whose width the fit recorded; a response fit without it cannot be banked "
                    "(only the cross-phase static_r cell, an all-position fit re-masked by the "
                    "cell, takes the served header's own width)")
            header_len = len(header_ids)
        elif int(header_len) != len(header_ids):
            raise LoraExportError(
                f"kind {kind!r}: the fit recorded a {int(header_len)}-token assistant header but "
                f"the served generation-prompt header is {len(header_ids)} tokens "
                f"({list(header_ids)}) — the fit's width mask and the serve's id-located mask "
                "would cover DIFFERENT regions; the object was fit against a different chat "
                f"template than {base_model!r} serves. Refusing")
        header_len = int(header_len)
    if not oproj_layers:
        raise LoraExportError(
            f"kind {kind!r}: the base has no o_proj layer to anchor the carrier LoRA on")
    anchor = int(oproj_layers[0])
    if anchor not in oproj_shapes:
        raise LoraExportError(
            f"kind {kind!r}: no o_proj shape for the anchor layer {anchor}")
    d_in, d_out = oproj_shapes[anchor]
    bad = [b for b in parsed["boundaries"] if not 0 <= b < n_layers_model]
    if bad:
        raise LoraExportError(
            f"kind {kind!r}: boundaries {bad} are outside the base's {n_layers_model} decoder "
            "blocks — the object was not fit on this base")
    block = {
        "schema": nr.RESIDUAL_SLOT_SCHEMA, "kind": kind, "op": parsed["op"],
        "layers": list(parsed["layers"]), "boundaries": list(parsed["boundaries"]),
        "position_mode": mode,
        "response_header_len": header_len,
        "response_header_ids": header_ids,
        "skip_bos": bool(parsed["skip_bos"]), "coeff": float(parsed["coeff"]),
        "hidden_size": int(hidden_size), "n_layers_model": int(n_layers_model),
        "anchor_layer": anchor, "dtype_policy": nr.DTYPE_POLICIES[parsed["op"]],
    }
    if parsed["op"] == nr.OP_RELU_GATED_VECTOR_PAIR:
        # THE PAIRED WRITE: both parents' shas and the terms' coefficients ride in the block,
        # so the residual fingerprint covers them (a paired cell's hash differs from either
        # parent's and from the same tensors under other parents)
        block.update(parent_pos_sha256=parsed["parent_pos_sha256"], parent_neg_sha256=parsed["parent_neg_sha256"],
                     parent_pos_coeff=float(parsed["parent_pos_coeff"]), parent_neg_coeff=float(parsed["parent_neg_coeff"]))
    try:
        payload = nr.payload_from_block_and_tensors(block, parsed["tensors"])
    except nr.ResidualSlotError as e:
        raise LoraExportError(f"kind {kind!r}: {e}") from e
    module = f"{decoder_prefix}.{anchor}.{oproj_site}"
    tensors: Dict[str, torch.Tensor] = {
        f"base_model.model.{module}.lora_A.weight": torch.zeros(1, d_in, dtype=torch.float32),
        f"base_model.model.{module}.lora_B.weight": torch.zeros(d_out, 1, dtype=torch.float32),
    }
    tensors.update(payload.tensors())
    config = {
        "r": 1,
        "lora_alpha": 1.0,
        "target_modules": [oproj_site.rsplit(".", 1)[-1]],
        "layers_to_transform": [anchor],
        "petri_source_kind": kind,
        "petri_source_layers": list(parsed["layers"]),
        "petri_fit_positions": parsed["fit_positions"],
        "petri_residual_slot": payload.block(),
    }
    if oproj_site != DEFAULT_OPROJ_SITE:
        config["petri_oproj_site"] = oproj_site
    return tensors, config


def export_genuine_adapter(resolved_spec: Mapping[str, Any],
                           artifact_bytes: bytes, out_dir: Path, *,
                           base_model: str,
                           sidecar_bytes: bytes = b"",
                           decoder_prefix: Optional[str] = None,
                           base_revision: Optional[str] = None,
                           oproj_site: Optional[str] = None,
                           snapshot_dir: Optional[Path] = None) -> Path:
    """Genuine-adapter export for a hand-rolled hook arm.

    Converts a landed weight-level object artifact (the exact hashed
    bytes) into a PEFT adapter dir that ``vllm serve --enable-lora
    --lora-modules <cell_id>=<dir>`` loads unchanged. Kinds handled:
    :data:`GENUINE_EXPORT_KINDS` — the conversion math for each lives in
    its ``_export_*`` helper, derived from (and asserted against) the
    object class's own hook body. Everything else refuses loudly (module
    docstring contract): wrong/unknown kinds, rank or shape mismatches,
    non-finite tensors, and any position-masked fit (native multi-LoRA
    has no position mask, so a masked fit served all-position would be a
    DIFFERENT intervention under the fit's hash).

    Deterministic: same artifact bytes -> byte-identical adapter tensors
    and config (no RNG, sorted keys, no timestamps). ``base_model`` is
    stated explicitly by the caller from the pinned base, exactly like
    :func:`export_zero_adapter` — this module never guesses.
    ``sidecar_bytes`` is the fit's optional JSON sidecar (it is part
    of the hashed object identity, so it must reach the export): it may
    record ``positions_at_fit``, which must agree with the blob.

    ``decoder_prefix`` / ``oproj_site`` are the base's decoder layout
    (module docstring): stated by a caller that holds the model
    (``models.decoder_layers_path(model)`` / ``models.attention_oproj_site
    (block)``) and validated against the known layouts — a stated prefix
    with no stated site implies ``self_attn.o_proj``, the layout every
    such caller has been on — or, when ``decoder_prefix`` is omitted,
    DERIVED from ``base_model``'s locally cached config through
    :func:`resolve_decoder_layout` (``base_revision`` pins the snapshot),
    which also proves the fit's layer carries an attention o_proj; a
    stated ``oproj_site`` that disagrees with the derived tree refuses.
    Either way the emitted module names are the base's own attribute
    paths; a layout that cannot be established is a refusal, never
    llama's ``model.layers`` assumed.

    On the derived path the names are then SPELLED IN THE CHECKPOINT'S
    NAMESPACE (module docstring): :func:`resolve_module_namespace`
    reads the served snapshot's weight names — ``snapshot_dir`` when the
    caller holds it, else the local HF cache's snapshot of
    ``base_model``/``base_revision`` — and every module name carries the
    prefix the checkpoint carries; a snapshot without a weights manifest
    refuses. ``snapshot_dir`` with a stated ``decoder_prefix`` refuses
    (the caller either names the modules or asks the checkpoint to)."""
    return _export_genuine(resolved_spec, artifact_bytes, out_dir,
                           base_model=base_model, sidecar_bytes=sidecar_bytes,
                           decoder_prefix=decoder_prefix,
                           base_revision=base_revision, oproj_site=oproj_site,
                           snapshot_dir=snapshot_dir)[0]


def _export_genuine(resolved_spec: Mapping[str, Any],
                    artifact_bytes: bytes, out_dir: Path, *,
                    base_model: str,
                    sidecar_bytes: bytes = b"",
                    decoder_prefix: Optional[str] = None,
                    base_revision: Optional[str] = None,
                    oproj_site: Optional[str] = None,
                    snapshot_dir: Optional[Path] = None,
                    ) -> Tuple[Path, Optional[ModuleNamespace]]:
    """:func:`export_genuine_adapter`'s body, returning ``(out_dir,
    namespace)`` — the :class:`ModuleNamespace` the derived path spelled
    the names under (``None`` on the stated-prefix path), which the bank
    records beside the adapter."""
    from safetensors.torch import save_file  # transformers-stack dep

    if not isinstance(base_model, str) or not base_model:
        raise LoraExportError(
            f"base_model must be a non-empty model id, got {base_model!r}")
    kind = resolved_spec.get("kind")
    if kind not in GENUINE_EXPORT_KINDS:
        raise LoraExportError(
            f"no genuine PEFT export for object kind {kind!r} — the "
            f"committed exports are {sorted(GENUINE_EXPORT_KINDS)} (two-SI "
            "D13); a kind serving on neither committed vLLM path is a §10 "
            "gate failure, never an approximate adapter")
    residual = kind in RESIDUAL_EXPORT_KINDS
    # the ``vector`` kind is a SteeringVector safetensors container + JSON
    # sidecar, not a torch blob: its exporter parses it itself
    blob = _genuine_blob(kind, artifact_bytes) if kind != "vector" else None
    if residual and decoder_prefix is not None:
        raise LoraExportError(
            f"kind {kind!r}: a residual-slot kind exports on the DERIVED path only — its "
            "anchor (the base's first o_proj layer), the residual stream's width and the "
            "decoder block count all come from the base's own module tree; omit "
            "decoder_prefix=")
    # the base's meta skeleton, built ONLY for a GENERAL module-set blob on
    # the derived path (FD 6): its per-module sites and shapes are held to
    # the base's own module tree (lora_modules.resolve_targets), which the
    # o_proj layout below does not describe. None on the stated path and
    # for every o_proj-only blob, whose bytes are exactly what they were.
    skeleton: Optional[torch.nn.Module] = None
    if decoder_prefix is not None:
        # a caller holding the model states the layout; with no stated
        # site the historical self_attn contract applies (every such
        # caller is on a self_attn stack), and no tree is available to
        # prove the fit layer carries the site.
        if snapshot_dir is not None:
            raise LoraExportError(
                f"snapshot_dir={str(snapshot_dir)!r} with a stated "
                f"decoder_prefix={decoder_prefix!r}: a caller either names "
                "the modules itself or lets the checkpoint name them "
                "(omit decoder_prefix=) — not both; refusing")
        prefix = _validated_decoder_prefix(decoder_prefix)
        site = (_validated_oproj_site(oproj_site) if oproj_site is not None
                else DEFAULT_OPROJ_SITE)
        site_layers: Optional[Tuple[int, ...]] = None
        site_shapes: Optional[Dict[int, Tuple[int, int]]] = None
        namespace: Optional[ModuleNamespace] = None
    else:
        # ONE skeleton build yields the layout AND each o_proj's own
        # (in_features, out_features), so every fit layer's factors are
        # held to the module their name points at (gemma-4's global
        # layers are wider than its local ones; no cross-layer premise)
        layout, site_shapes = _derived_layout(base_model, base_revision)
        if oproj_site is not None and oproj_site != layout.oproj_site:
            raise LoraExportError(
                f"oproj_site={oproj_site!r} stated for {base_model!r}, but "
                f"its module tree carries {layout.oproj_site!r} — refusing "
                "to name a site the base does not have")
        # the layout (which layers, which site, which shapes) is the
        # skeleton's; the SPELLING of every name is the checkpoint's. The
        # per-layer resolution proves each o_proj layer is a module of the
        # served checkpoint under ONE prefix, and that prefix replaces the
        # skeleton's in every name below (the kind helpers build
        # ``f"{prefix}.{layer}.{site}"``, so substituting the prefix IS the
        # per-layer mapping — the one-prefix rule makes them the same).
        _, namespace = resolve_module_namespace(base_model, base_revision,
                                                layout, snapshot_dir=snapshot_dir)
        prefix, site, site_layers = (namespace.checkpoint, layout.oproj_site,
                                     layout.oproj_layers)
        if kind in ("lora_all", "lora_subtraction") and isinstance(blob, dict) \
                and "targets" in blob:
            # a second skeleton build (the first is inside _derived_layout, which
            # tests patch as a seam — so the tree is not threaded out of it)
            _, skeleton = _decoder_skeleton(base_model, base_revision)
    if residual:
        hidden, n_layers = _derived_dims(base_model, base_revision)
        tensors, kind_config = _export_residual_slot(
            kind, resolved_spec, artifact_bytes, sidecar_bytes, decoder_prefix=prefix,
            oproj_site=site, oproj_layers=site_layers, oproj_shapes=site_shapes,
            hidden_size=hidden, n_layers_model=n_layers,
            base_model=base_model, base_revision=base_revision)
    elif kind == "lora":
        tensors, kind_config = _export_rank1_lora(
            resolved_spec, blob, decoder_prefix=prefix, oproj_site=site,
            oproj_layers=site_layers, oproj_shapes=site_shapes)
    elif kind == "lora_all":
        tensors, kind_config = _export_multilayer_lora(
            resolved_spec, blob, decoder_prefix=prefix, oproj_site=site,
            oproj_layers=site_layers, oproj_shapes=site_shapes,
            skeleton=skeleton)
    elif kind == "lora_subtraction":
        tensors, kind_config = _export_lora_subtraction(
            resolved_spec, blob, decoder_prefix=prefix, oproj_site=site,
            oproj_layers=site_layers, oproj_shapes=site_shapes,
            skeleton=skeleton)
    else:
        tensors, kind_config = _export_circuit_breaker(
            resolved_spec, blob, decoder_prefix=prefix, oproj_site=site)

    if not residual:   # the residual exporter judged its own sidecar (lens rules per kind)
        _check_sidecar(kind, sidecar_bytes, kind_config["petri_fit_positions"])

    config = {
        **_peft_config_header(base_model),
        # provenance: a GENUINE adapter — deliberately NOT marked
        # petri_zero_adapter, so verify_zero_adapter can never certify it
        # as the no-op leg.
        "petri_genuine_adapter": True,
        # the decoder prefix every module name below IS SPELLED IN — the
        # served checkpoint's on the derived path (model.language_
        # model.layers on qwen3.5 / gemma-4 / GLM, backbone.layers on
        # NemotronH, model.layers on llama; mapped onto the execution
        # modules by vLLM's own hf_to_vllm_mapper), the caller's on the
        # stated path. The skeleton's prefix, where it differs, is in the
        # bank record (adapter_digests.json module_namespace), not here:
        # this config is part of the adapter's hashed identity, and an
        # export whose two prefixes agree must stay byte-identical.
        "petri_decoder_prefix": prefix,
        **kind_config,
    }
    ih = resolved_spec.get("intervention_hash")
    if ih is not None:
        if not isinstance(ih, str) or not ih:
            raise LoraExportError(
                f"kind {kind!r}: resolved_spec intervention_hash must be a "
                f"non-empty string when present, got {ih!r}")
        config["petri_intervention_hash"] = ih

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ADAPTER_CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True))
    save_file({k: tensors[k] for k in sorted(tensors)},
              str(out_dir / ADAPTER_WEIGHTS))
    return out_dir, namespace


# ==========================================================================
# module-set adapters: the bare-blob convenience, the synthetic builder and
# the file-derived target class (D_contract §3)
# ==========================================================================

def _check_init_scheme(kind: str, blob: Mapping[str, Any]) -> None:
    """A legacy-shape blob's optional ``init_scheme`` (FD 7, written only
    when it is not the default) must be one of ``INIT_SCHEMES``; it does
    not enter the adapter (the factors ARE the object), so a legacy blob
    that carries it exports the same bytes as one that does not."""
    scheme = blob.get("init_scheme")
    if scheme is None:
        return
    from ...extraction.lora_modules import INIT_SCHEMES

    if scheme not in INIT_SCHEMES:
        raise LoraExportError(
            f"kind {kind!r}: init_scheme={scheme!r} is not one of "
            f"{list(INIT_SCHEMES)}; refusing")


def export_module_set(out_dir: Path, *, base_model: str, blob: Mapping[str, Any],
                      base_revision: Optional[str] = None,
                      intervention_hash: Optional[str] = None,
                      snapshot_dir: Optional[Path] = None,
                      decoder_prefix: Optional[str] = None,
                      oproj_site: Optional[str] = None) -> Path:
    """Export a BARE ``lora_all`` / ``lora_subtraction`` blob dict (either
    shape of D_contract §2) as a PEFT adapter dir — the serving probe's and
    the tests' entry: the blob is serialised exactly as
    the object's ``save`` would write it and handed to
    :func:`export_genuine_adapter` under ``{"kind": blob["kind"]}`` (plus
    ``intervention_hash`` when given), so every refusal and every byte is
    the production exporter's — the derived path (layout, checkpoint
    namespace and, for a general blob, the module-tree cross-check) by
    default, the stated path when ``decoder_prefix`` is given. Returns
    ``out_dir``."""
    import io

    if not isinstance(blob, Mapping) or blob.get("kind") not in ("lora_all",
                                                                  "lora_subtraction"):
        got = blob.get("kind") if isinstance(blob, Mapping) else type(blob).__name__
        raise LoraExportError(
            f"export_module_set: blob must be a lora_all / lora_subtraction "
            f"blob dict, got kind={got!r}")
    spec: Dict[str, Any] = {"kind": blob["kind"]}
    if intervention_hash is not None:
        spec["intervention_hash"] = _validated_hex16("intervention_hash",
                                                     intervention_hash)
    buf = io.BytesIO()
    torch.save(dict(blob), buf)
    return _export_genuine(spec, buf.getvalue(), Path(out_dir), base_model=base_model,
                           base_revision=base_revision, snapshot_dir=snapshot_dir,
                           decoder_prefix=decoder_prefix, oproj_site=oproj_site)[0]


def synthetic_module_set_blob(model: torch.nn.Module, modules, rank: int, *,
                              seed: int, scale: float = 1.0, zero: bool = False,
                              per_expert_rank: Optional[int] = None,
                              layers=None) -> Dict[str, Any]:
    """A ``lora_all`` blob with RANDOM factors over every target the module
    set reaches on ``model`` (``lora_modules.resolve_targets`` — a meta
    skeleton or a live model, shapes only) — the serving probe's synthetic
    adapter and the tests' hand-built object. Factors
    are drawn from ONE generator seeded with ``seed`` in (layer, module)
    order, fp32 CPU: ``A ~ N(0, 1/in)`` so ``x @ A`` keeps the input's
    scale, ``B ~ N(0, scale²/r)`` so ``ΔW = B Aᵀ`` has entries of order
    ``scale / sqrt(in)`` — a known NONZERO effect that ``scale`` sets
    (raise it for an unmistakable logit shift). ``zero=True`` zeroes every
    B (A keeps the same draw, so the zero twin differs from its random
    twin in B alone). The blob is the LEGACY shape when ``modules ==
    ("o_proj",)`` (rank-1 factors 1-D) and the general shape otherwise;
    it records no ``init_scheme`` — the factors are not an init, they are
    the object."""
    from ...extraction.lora_modules import (normalise_modules, resolve_targets,
                                            set_id_for, targets_summary)

    mods = normalise_modules(modules)
    rank = int(rank)
    targets = resolve_targets(model, mods, rank=rank, layers=layers,
                              per_expert_rank=per_expert_rank)
    g = torch.Generator().manual_seed(int(seed))
    A: Dict[str, torch.Tensor] = {}
    B: Dict[str, torch.Tensor] = {}
    legacy = mods == ("o_proj",)
    routed: Optional[int] = None
    for t in sorted(targets, key=lambda t: (t.layer, t.module)):
        if t.kind == "linear":
            a = torch.randn(t.in_features, t.rank, generator=g) / math.sqrt(t.in_features)
            b = torch.randn(t.out_features, t.rank, generator=g) * (
                float(scale) / math.sqrt(t.rank))
        else:
            n_e = int(t.num_experts or 0)
            routed = int(t.rank)
            a = torch.randn(n_e, t.rank, t.in_features, generator=g) / math.sqrt(t.in_features)
            b = torch.randn(n_e, t.out_features, t.rank, generator=g) * (
                float(scale) / math.sqrt(t.rank))
        if zero:
            b = torch.zeros_like(b)
        if legacy and rank == 1:
            a, b = a.squeeze(1), b.squeeze(1)
        key = str(t.layer) if legacy else t.key
        A[key] = a.float().contiguous()
        B[key] = b.float().contiguous()
    union = sorted({t.layer for t in targets})
    blob: Dict[str, Any] = {
        "kind": "lora_all", "layers": [int(l) for l in union], "rank": rank,
        "A": A, "B": B, "positions": "all", "response_header_len": None,
    }
    if not legacy:
        blob.update({
            "lora_modules": list(mods), "lora_set_id": set_id_for(mods),
            "per_expert_rank": ({"shared_expert": rank, "routed_expert": routed}
                                if routed is not None else None),
            "targets": targets_summary(targets),
        })
    return blob


def adapter_target_class(adapter_dir: Path) -> Tuple[str, ...]:
    """The vLLM ``--lora-target-modules`` CLASS an adapter's FILE implies
    (D_contract §3): the sorted short module suffixes of its factor keys
    (``o_proj``, ``q_proj``, ..., ``gate_proj``, ...) plus ``"experts"``
    when any key names a fused-experts wrapper (``...experts.lora_[AB]`` /
    ``...experts.base_layer.lora_[AB]`` — :data:`_EXPERTS_MODULE_RE`). Read
    off the safetensors HEADER, never the config: PEFT's ``target_modules``
    names the HF modules (``gate_proj`` reaches the shared expert) while
    the routed experts are PARAMETERS PEFT wraps through
    ``target_parameters``, so the wrapper word vLLM needs is inferred from
    what the file carries. Residual-slot payload keys are skipped. Refuses
    a file with no factor, a key that is neither a factor nor a payload, a
    suffix outside the module vocabulary, or an experts wrapper carrying
    ONE half (vLLM asserts both at load, ``model_manager.py:843``, and
    skips a lone gate_up silently)."""
    from ...extraction.lora_modules import KNOWN_MODULES

    adapter_dir = Path(adapter_dir)
    weights = adapter_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise LoraExportError(
            f"{adapter_dir} has no {ADAPTER_WEIGHTS} — no factors to read a "
            "target class from")
    suffixes: set = set()
    halves: Dict[str, set] = {}
    for key in _safetensors_header_keys(weights):
        if key.startswith(_sa.RESIDUAL_KEY_PREFIX):
            continue
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if key.endswith(suffix) and len(key) > len(suffix):
                module = key[:-len(suffix)]
                break
        else:
            raise LoraExportError(
                f"{adapter_dir}: safetensors key {key!r} is neither a "
                f"lora_A/lora_B factor nor a {_sa.RESIDUAL_KEY_PREFIX}* payload; "
                "refusing to read a target class off it")
        m = _EXPERTS_MODULE_RE.match(module)
        if m:
            suffixes.add(EXPERTS_TARGET_SUFFIX)
            halves.setdefault(m["path"], set()).add(
                "gate_up_proj" if m["inner"] else "down_proj")
            continue
        short = module.rsplit(".", 1)[-1]
        if short not in KNOWN_MODULES:
            raise LoraExportError(
                f"{adapter_dir}: factor key {key!r} adapts {short!r}, outside "
                f"the module vocabulary {sorted(KNOWN_MODULES)}; refusing")
        suffixes.add(short)
    if not suffixes:
        raise LoraExportError(
            f"{adapter_dir}: {ADAPTER_WEIGHTS} carries no LoRA factor; no "
            "target class")
    for path, got in sorted(halves.items()):
        if got != set(_EXPERT_PARAMS):
            raise LoraExportError(
                f"{adapter_dir}: experts module {path!r} carries only "
                f"{sorted(got)} — vLLM 0.25.1 asserts BOTH the gate_up "
                "(experts.base_layer) and the down (experts) pair at load "
                "(model_manager.py:843) and skips a lone gate_up silently; the "
                "export ships a zero companion for the missing half; refusing")
    return tuple(sorted(suffixes))


def modules_of_class(target_class) -> Tuple[str, ...]:
    """The PEFT short module names of a vLLM ``--lora-target-modules`` CLASS
    (:func:`adapter_target_class`'s inverse): the class minus the wrapper
    word ``experts``, which names no HF module — what
    :func:`export_zero_adapter_for_class` takes as ``lora_modules``."""
    return tuple(m for m in target_class if m != EXPERTS_TARGET_SUFFIX)


def launch_target_class(classes: Mapping[str, Tuple[Sequence[str], bool]], *,
                        default: Tuple[str, ...] = ("o_proj",)) -> Tuple[str, ...]:
    """The ONE ``--lora-target-modules`` class a lora-native launch composes from
    the registered adapters' FILE classes (``classes``: adapter name ->
    ``(its class, is_residual_carrier)``). Every GENUINE adapter must adapt the
    SAME class — one launch restricts the LoRA execution wrappers to one suffix
    set, so a disagreement refuses naming every adapter's. A RESIDUAL-SLOT
    CARRIER (:func:`_export_residual_slot`: a rank-1 ZERO o_proj pair that only
    carries vLLM's slot and the payload; ``adapter_config.json`` names the
    ``petri_residual_slot`` block) is not a LoRA of that class and never decides
    it: it is admitted when its own class is a SUBSET of the genuine class (vLLM
    resets the carrier's slot at every wrapper it has no factor for, so it adapts
    nothing there either — the steered serving canary of 2026-09-30 served a
    carrier beside an attn_mlp adapter), and a carrier whose modules lie OUTSIDE
    the class refuses by name. Carriers alone take their own (one) class; no
    adapter at all takes ``default`` (the paper's o_proj class)."""
    genuine = {n: frozenset(c) for n, (c, carrier) in classes.items() if not carrier}
    carriers = {n: frozenset(c) for n, (c, carrier) in classes.items() if carrier}
    distinct = set(genuine.values())
    if len(distinct) > 1:
        raise LoraExportError(
            "the registered adapters disagree on their adapted module class: "
            f"{ {n: sorted(c) for n, c in sorted(genuine.items())} } — one lora-native launch restricts "
            "the LoRA execution wrappers to ONE suffix set (--lora-target-modules), so every "
            "genuine adapter must adapt the same modules; refusing")
    if distinct:
        cls = next(iter(distinct))
    elif carriers:
        cdistinct = set(carriers.values())
        if len(cdistinct) > 1:
            raise LoraExportError(
                "the residual-slot carriers of this launch disagree on their module class: "
                f"{ {n: sorted(c) for n, c in sorted(carriers.items())} } — a carrier is a zero "
                "pair on ONE class (the base's o_proj); refusing")
        cls = next(iter(cdistinct))
    else:
        cls = frozenset(default)
    outside = {n: sorted(c - cls) for n, c in sorted(carriers.items()) if not c <= cls}
    if outside:
        raise LoraExportError(
            f"residual-slot carrier(s) adapt modules outside the launch class {sorted(cls)}: {outside} — "
            "a carrier rides a zero pair and is admitted only INSIDE the genuine adapters' class; refusing")
    return tuple(sorted(cls))


# ==========================================================================
# the content-addressed adapter bank
# ==========================================================================

#: written beside the two PEFT files by :func:`export_or_reuse` /
#: :func:`export_or_reuse_zero`: the :class:`AdapterDigests` fields plus
#: ``exported_at`` (epoch seconds) and ``exporter_tree_sha`` (``git
#: rev-parse HEAD`` of the tree that exported, ``None`` outside a
#: checkout). Those two are provenance, NOT identity: the byte-identity
#: the bank checks is over ``adapter_config.json`` + ``adapter_model.
#: safetensors`` only, so re-exporting the same bytes at another time from
#: another commit still reuses the entry.
ADAPTER_DIGESTS = "adapter_digests.json"


class AdapterDigests(NamedTuple):
    """What identifies one adapter dir. ``dir`` (absolute, resolved) is
    where it sits and ``name_hint`` is what the caller called it (the cell
    id the exporter was told; ``None`` for a bank re-verify, which knows
    no cell) — both describe the entry's PLACE and USE. The other six are
    its CONTENT: sha256 over each PEFT file's bytes, the config's ``r``,
    and the three ``petri_*`` provenance fields the routing table's
    adapter block carries. :func:`verify_adapter_dir`
    compares the content fields only."""

    dir: str
    name_hint: Optional[str]
    config_sha256: str
    weights_sha256: str
    r: int
    petri_intervention_hash: Optional[str]
    petri_zero_adapter: bool
    petri_source_kind: Optional[str]
    #: native-psr-serving §1.8: the residual entry's NATIVE position mode, read off the
    #: ``petri_residual_slot`` block (``None`` for a plain LoRA / the zero adapter). A
    #: DERIVED provenance field — not part of the content identity
    #: (:data:`_DIGEST_CONTENT_FIELDS`): the block's bytes are already inside
    #: ``config_sha256``, and a stage-1 digest file that never recorded it must keep
    #: verifying against the same bytes
    petri_position_mode: Optional[str] = None


#: the content identity :func:`verify_adapter_dir` compares.
_DIGEST_CONTENT_FIELDS = ("config_sha256", "weights_sha256", "r",
                          "petri_intervention_hash", "petri_zero_adapter",
                          "petri_source_kind")
#: the provenance keys :data:`ADAPTER_DIGESTS` carries beyond the fields:
#: ``exported_at``, ``exporter_tree_sha``, and ``module_namespace`` — the
#: :class:`ModuleNamespace` the export spelled its names under, a
#: mapping of its fields, or ``None`` for a stated-prefix export. An entry
#: written before the namespace rule carries the first two only and reads
#: as "not recorded": its content identity is intact, and what proves an
#: entry's names is the byte-identity check against a fresh export, never
#: this record.
_DIGEST_FILE_EXTRA = ("exported_at", "exporter_tree_sha", "module_namespace")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _validated_hex16(what: str, value: Any) -> str:
    """A capeval hash: exactly ``manifest.HASH_HEX_LEN`` lowercase hex
    chars (``sha256_hex16``). The bank names directories by these, so the
    shape is also what keeps an address path-safe."""
    from ..manifest import HASH_HEX_LEN

    if not isinstance(value, str) or \
            not re.fullmatch(rf"[0-9a-f]{{{HASH_HEX_LEN}}}", value):
        raise LoraExportError(
            f"{what} must be a {HASH_HEX_LEN}-hex sha256 prefix "
            f"(capeval.manifest.sha256_hex16), got {value!r}")
    return value


def adapter_digests(adapter_dir: Path) -> AdapterDigests:
    """Read an adapter dir's :class:`AdapterDigests` off its bytes: both
    PEFT files hashed, ``r`` and the ``petri_*`` fields read from
    ``adapter_config.json``. ``name_hint`` is ``None`` (a re-verify knows
    no cell). Refuses a dir missing either file, an unparseable config, a
    non-positive ``r``, or a config that is not EXACTLY one of this
    module's two exports (``petri_genuine_adapter`` XOR
    ``petri_zero_adapter`` — a foreign PEFT adapter carries neither and
    is not something the bank describes)."""
    d = Path(adapter_dir).resolve()
    cfg_path, weights_path = d / ADAPTER_CONFIG, d / ADAPTER_WEIGHTS
    for f in (cfg_path, weights_path):
        if not f.is_file():
            raise LoraExportError(
                f"{d}: no {f.name} — not a PEFT adapter dir")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LoraExportError(
            f"{d}: unparseable {ADAPTER_CONFIG}: {e}") from e
    if not isinstance(cfg, dict):
        raise LoraExportError(
            f"{d}: {ADAPTER_CONFIG} is not a JSON object")
    r = cfg.get("r")
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        raise LoraExportError(
            f"{d}: {ADAPTER_CONFIG} r={r!r} is not a positive int")
    zero = cfg.get("petri_zero_adapter") is True
    genuine = cfg.get("petri_genuine_adapter") is True
    if zero == genuine:
        raise LoraExportError(
            f"{d}: {ADAPTER_CONFIG} marks {'both' if zero else 'neither'} "
            "petri_zero_adapter and petri_genuine_adapter — not an adapter "
            "this module exported (exactly one marker); refusing")
    ih = cfg.get("petri_intervention_hash")
    if ih is not None and (not isinstance(ih, str) or not ih):
        raise LoraExportError(
            f"{d}: {ADAPTER_CONFIG} petri_intervention_hash={ih!r} is not "
            "a non-empty string")
    kind = cfg.get("petri_source_kind")
    if kind is not None and (not isinstance(kind, str) or not kind):
        raise LoraExportError(
            f"{d}: {ADAPTER_CONFIG} petri_source_kind={kind!r} is not a "
            "non-empty string")
    block = cfg.get("petri_residual_slot")
    mode = block.get("position_mode") if isinstance(block, dict) else None
    if mode is not None and (not isinstance(mode, str) or not mode):
        raise LoraExportError(
            f"{d}: {ADAPTER_CONFIG} petri_residual_slot.position_mode={mode!r} "
            "is not a non-empty string")
    return AdapterDigests(
        dir=str(d), name_hint=None,
        config_sha256=_sa.sha256_file(cfg_path),
        weights_sha256=_sa.sha256_file(weights_path),
        r=r, petri_intervention_hash=ih, petri_zero_adapter=zero,
        petri_source_kind=kind, petri_position_mode=mode)


def adapter_dir_for(root: Path, intervention_hash: str) -> Path:
    """The bank address of a GENUINE adapter: ``root / <intervention_hash>``
    (the object's 16-hex ``petri_intervention_hash`` — the same
    intervention, another serving path). Content-addressed: the cell id
    is the vLLM adapter NAME, never the directory."""
    ih = _validated_hex16("intervention_hash", intervention_hash)
    return Path(root) / ih


def zero_adapter_key(*, base_model: str, base_revision: Optional[str],
                     target_modules: Mapping[str, Tuple[int, ...]],
                     rank: int, per_expert_rank: Optional[int] = None) -> str:
    """The bank key of a zero adapter: ``sha256_hex16`` over the canonical
    JSON of everything :func:`export_zero_adapter` writes from — base
    model id and revision, every adapted module's name and
    ``(in_features, out_features)``, and the rank. Two bases with the same
    o_proj layout and shapes still get distinct keys (the id is inside);
    a shape change anywhere changes it. A per-CLASS zero adapter
    (:func:`export_zero_adapter_for_class`) adds its routed-expert entries
    as ``(E, in, out)`` TRIPLES and, when it has any, ``per_expert_rank``
    (the routed ``r_e``); the canonical JSON gains that key ONLY then, so
    every o_proj key is exactly what it was."""
    from ..manifest import sha256_hex16

    if not isinstance(base_model, str) or not base_model:
        raise LoraExportError(
            f"base_model must be a non-empty model id, got {base_model!r}")
    if base_revision is not None and (not isinstance(base_revision, str)
                                      or not base_revision):
        raise LoraExportError(
            f"base_revision must be None or a non-empty string, got "
            f"{base_revision!r}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise LoraExportError(f"rank must be a positive int, got {rank!r}")
    if per_expert_rank is not None and (
            isinstance(per_expert_rank, bool) or not isinstance(per_expert_rank, int)
            or per_expert_rank < 1):
        raise LoraExportError(
            f"per_expert_rank must be None or a positive int, got {per_expert_rank!r}")
    if not isinstance(target_modules, Mapping) or not target_modules:
        raise LoraExportError(
            "target_modules must be a non-empty {module: (in_features, "
            "out_features)} mapping")
    modules: Dict[str, List[int]] = {}
    for module, shape in target_modules.items():
        if (not isinstance(module, str) or not module
                or not isinstance(shape, (tuple, list)) or len(shape) not in (2, 3)
                or not all(isinstance(x, int) and not isinstance(x, bool)
                           and x > 0 for x in shape)):
            raise LoraExportError(
                f"target_modules[{module!r}] must map a non-empty module "
                f"name to (in_features, out_features) — or (num_experts, "
                f"in_features, out_features) for a routed-expert parameter — "
                f"got {shape!r}")
        modules[module] = [int(x) for x in shape]
    doc: Dict[str, Any] = {"base_model": base_model, "base_revision": base_revision,
                           "rank": int(rank), "target_modules": modules}
    if per_expert_rank is not None:
        doc["per_expert_rank"] = int(per_expert_rank)
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    return sha256_hex16(canonical.encode("utf-8"))


def zero_adapter_dir_for(root: Path, key: str) -> Path:
    """The bank address of the zero adapter under ``key``
    (:func:`zero_adapter_key`): ``root / "none__lora0__<key>"``."""
    key = _validated_hex16("zero adapter key", key)
    return Path(root) / f"{ZERO_ADAPTER_NAME}__{key}"


def _fsync_path(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@functools.lru_cache(maxsize=1)
def _exporter_tree_sha() -> Optional[str]:
    """``git rev-parse HEAD`` of the checkout this module runs from
    (``None`` when it is not one — recorded as such, never guessed);
    resolved once per process, not once per export."""
    return _ri.tree_sha(Path(__file__).resolve().parents[4])


def _write_digests_file(adapter_dir: Path, digests: AdapterDigests, *,
                        module_namespace: Optional[ModuleNamespace]) -> None:
    """``adapter_digests.json`` beside the two PEFT files: written to a
    temp name, fsynced, ``os.replace``d; then every file in the dir and
    the dir itself are fsynced so the ENTRY is durable before the
    caller's one ``rename`` publishes it. ``module_namespace`` is the
    module-namespace record (:data:`_DIGEST_FILE_EXTRA`)."""
    doc: Dict[str, Any] = dict(digests._asdict())
    doc["exported_at"] = time.time()
    doc["exporter_tree_sha"] = _exporter_tree_sha()
    doc["module_namespace"] = (module_namespace._asdict()
                               if module_namespace is not None else None)
    _sa.write_json_atomic(adapter_dir / ADAPTER_DIGESTS, doc)
    for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS):
        _fsync_path(adapter_dir / name)
    _fsync_path(adapter_dir)


def _module_namespace_from_record(value: Any, stale: str
                                  ) -> Optional[ModuleNamespace]:
    """The record's ``module_namespace``: ``None`` (a stated-prefix export,
    or an entry from before the rule) or a mapping of exactly the
    :class:`ModuleNamespace` fields, each well-formed; anything else is a
    tampered record."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(ModuleNamespace._fields):
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} module_namespace keys {got} != "
            f"{sorted(ModuleNamespace._fields)}")
    for key in ("skeleton", "checkpoint", "source", "snapshot"):
        if not isinstance(value[key], str) or not value[key]:
            raise LoraExportError(
                f"{stale} — {ADAPTER_DIGESTS} module_namespace.{key}="
                f"{value[key]!r} is not a non-empty string")
    if value["source"] not in MODULE_NAMESPACE_SOURCES:
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} module_namespace.source="
            f"{value['source']!r} is not one of {MODULE_NAMESPACE_SOURCES}")
    n = value["n_layers"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} module_namespace.n_layers={n!r} is "
            "not a positive int")
    return ModuleNamespace(**value)


def _read_digests_doc(adapter_dir: Path) -> Dict[str, Any]:
    """The entry's :data:`ADAPTER_DIGESTS` document with its KEY SET held:
    every :class:`AdapterDigests` field plus the provenance keys
    (:data:`_DIGEST_FILE_EXTRA`; ``module_namespace`` may be absent on an
    entry written before module namespaces were recorded and is then read
    as ``None``), nothing else. Field shapes are the callers' business."""
    path = Path(adapter_dir) / ADAPTER_DIGESTS
    stale = f"{adapter_dir}: stale or tampered bank entry; not reusing"
    if not path.is_file():
        raise LoraExportError(
            f"{stale} — no {ADAPTER_DIGESTS} beside the adapter files (not "
            "written by export_or_reuse, or half-written)")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LoraExportError(
            f"{stale} — unparseable {ADAPTER_DIGESTS}: {e}") from e
    expected_keys = set(AdapterDigests._fields) | set(_DIGEST_FILE_EXTRA)
    # TWO older key sets are read, each as the record it was: an entry written before module
    # namespaces were recorded carries no ``module_namespace``; an entry written before
    # native-psr-serving stage 2 carries no ``petri_position_mode`` (the DERIVED mode field,
    # outside the content identity — every stage-1 bank entry). Either absent key reads as
    # None; the CONTENT identity (_DIGEST_CONTENT_FIELDS) is re-hashed by the caller
    # regardless, so an older record buys nothing but its own reading. Any other key set is
    # not this bank's record.
    older = {"module_namespace", "petri_position_mode"}
    if not isinstance(doc, dict) or not (set(doc) <= expected_keys
                                         and expected_keys - set(doc) <= older):
        got = sorted(doc) if isinstance(doc, dict) else type(doc).__name__
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} keys {got} != "
            f"{sorted(expected_keys)}")
    doc.setdefault("module_namespace", None)
    doc.setdefault("petri_position_mode", None)
    return doc


def read_module_namespace(adapter_dir: Path) -> Optional[ModuleNamespace]:
    """The :class:`ModuleNamespace` a bank entry records, ``None``
    when it records none — a stated-prefix export, or an entry written
    before the rule. The record is provenance: which spelling the names
    in the entry carry and where it was read from."""
    stale = f"{adapter_dir}: stale or tampered bank entry; not reusing"
    return _module_namespace_from_record(
        _read_digests_doc(adapter_dir)["module_namespace"], stale)


def _read_digests_file(adapter_dir: Path) -> AdapterDigests:
    """The entry's own :data:`ADAPTER_DIGESTS` record, strictly: every
    :class:`AdapterDigests` field plus the provenance keys
    (:func:`_read_digests_doc`), each of the right shape. A missing or
    malformed record is a stale or tampered entry (not written by this
    bank, or half-written)."""
    stale = f"{adapter_dir}: stale or tampered bank entry; not reusing"
    doc = _read_digests_doc(adapter_dir)

    def _opt_str(key: str) -> Optional[str]:
        v = doc[key]
        if v is not None and (not isinstance(v, str) or not v):
            raise LoraExportError(
                f"{stale} — {ADAPTER_DIGESTS} {key}={v!r} is not None or a "
                "non-empty string")
        return v

    for key in ("config_sha256", "weights_sha256"):
        if not isinstance(doc[key], str) or not _HEX64.fullmatch(doc[key]):
            raise LoraExportError(
                f"{stale} — {ADAPTER_DIGESTS} {key}={doc[key]!r} is not a "
                "64-hex sha256")
    if not isinstance(doc["dir"], str) or not doc["dir"]:
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} dir={doc['dir']!r} is not a path")
    r = doc["r"]
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} r={r!r} is not a positive int")
    if not isinstance(doc["petri_zero_adapter"], bool):
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} petri_zero_adapter="
            f"{doc['petri_zero_adapter']!r} is not a bool")
    if isinstance(doc["exported_at"], bool) or \
            not isinstance(doc["exported_at"], (int, float)):
        raise LoraExportError(
            f"{stale} — {ADAPTER_DIGESTS} exported_at="
            f"{doc['exported_at']!r} is not a number")
    _opt_str("exporter_tree_sha")
    _module_namespace_from_record(doc["module_namespace"], stale)
    return AdapterDigests(
        dir=doc["dir"], name_hint=_opt_str("name_hint"),
        config_sha256=doc["config_sha256"],
        weights_sha256=doc["weights_sha256"], r=r,
        petri_intervention_hash=_opt_str("petri_intervention_hash"),
        petri_zero_adapter=doc["petri_zero_adapter"],
        petri_source_kind=_opt_str("petri_source_kind"),
        # absent from every stage-1 digest file: derived, never part of the identity
        petri_position_mode=(_opt_str("petri_position_mode") if "petri_position_mode" in doc else None))


def verify_adapter_dir(adapter_dir: Path, expected: AdapterDigests) -> None:
    """Re-hash ``adapter_dir`` and compare its CONTENT identity
    (:data:`_DIGEST_CONTENT_FIELDS`: both file digests, ``r``, the three
    ``petri_*`` fields) with ``expected``; any disagreement is a
    :class:`LoraExportError` naming every field that differs. ``dir``
    and ``name_hint`` are not compared — a bank rsynced elsewhere or an
    entry a second cell shares still holds the same bytes."""
    if not isinstance(expected, AdapterDigests):
        raise LoraExportError(
            f"expected must be an AdapterDigests, got "
            f"{type(expected).__name__}")
    got = adapter_digests(adapter_dir)
    bad = {f: (getattr(expected, f), getattr(got, f))
           for f in _DIGEST_CONTENT_FIELDS
           if getattr(expected, f) != getattr(got, f)}
    if bad:
        detail = ", ".join(f"{f}: expected {e!r}, on disk {g!r}"
                           for f, (e, g) in sorted(bad.items()))
        raise LoraExportError(
            f"{Path(adapter_dir)}: adapter digests disagree with the "
            f"expected record on {sorted(bad)} ({detail}) — stale or "
            "tampered; refusing")


def _same_peft_files(a: Path, b: Path) -> bool:
    return all((Path(a) / name).read_bytes() == (Path(b) / name).read_bytes()
               for name in (ADAPTER_CONFIG, ADAPTER_WEIGHTS))


def _rename_dir(src: Path, dst: Path) -> None:
    """The ONE atomic publish step (``os.rename`` of a complete tmp dir
    onto the bank address). Kept as a seam so a test can race it."""
    os.rename(src, dst)


def _tmp_dir(root: Path, tag: str) -> Path:
    return root / f".tmp-{tag}-{os.getpid()}-{time.time_ns()}"


def _reuse_verified(target: Path, fresh: Path, *, name_hint: Optional[str],
                    intervention_hash: Optional[str], zero: bool,
                    module_namespace: Optional[ModuleNamespace] = None,
                    ) -> Tuple[Path, AdapterDigests, bool]:
    """The reuse branch: an entry already sits at ``target`` and ``fresh``
    holds a complete export of THIS call's inputs. Reuse iff the entry
    (1) carries a well-formed :data:`ADAPTER_DIGESTS` record, (2)
    re-hashes to that record, (3) is the kind of adapter expected here
    (a genuine adapter under exactly ``intervention_hash``, or the zero
    adapter — re-verified all-zero), AND (4) is byte-identical, PEFT file
    for PEFT file, to the fresh export. Exports are deterministic, so (4)
    failing means the entry was not produced from these bytes by this
    exporter: refuse naming the dir; never fall back to writing over it.
    Returns the digests as they are ON DISK, under this caller's hint."""
    stale = f"{target}: stale or tampered bank entry; not reusing"
    recorded = _read_digests_file(target)
    try:
        verify_adapter_dir(target, recorded)
    except LoraExportError as e:
        raise LoraExportError(f"{stale} — {e}") from e
    if zero:
        if recorded.petri_zero_adapter is not True:
            raise LoraExportError(
                f"{stale} — recorded as a genuine adapter (petri_source_kind="
                f"{recorded.petri_source_kind!r}) where the zero adapter was "
                "expected")
        try:
            verify_zero_adapter(target)
        except LoraExportError as e:
            raise LoraExportError(f"{stale} — {e}") from e
    else:
        if recorded.petri_zero_adapter or \
                recorded.petri_intervention_hash != intervention_hash:
            raise LoraExportError(
                f"{stale} — it records petri_intervention_hash="
                f"{recorded.petri_intervention_hash!r}, petri_zero_adapter="
                f"{recorded.petri_zero_adapter!r}; expected a genuine "
                f"adapter under {intervention_hash!r}")
    if not _same_peft_files(target, fresh):
        # when the fresh export spelled its names under the checkpoint's
        # prefix and the entry records another spelling (or none — an entry
        # from before the rule spells them as the skeleton did), say so:
        # that entry is the qwen canary's defect on disk, and the remedy is
        # to remove it, not to trust it by address
        hint = ""
        recorded_ns = read_module_namespace(target)
        if module_namespace is not None and (
                recorded_ns is None
                or recorded_ns.checkpoint != module_namespace.checkpoint):
            was = ("no module-namespace record — it was exported before the "
                   "checkpoint-namespace rule (D97), spelling its modules as "
                   "the harness skeleton does"
                   if recorded_ns is None else
                   f"its modules spelled under {recorded_ns.checkpoint!r}")
            hint = (
                f". The entry carries {was}; this export spells them under "
                f"the served checkpoint's {module_namespace.checkpoint!r} "
                f"(skeleton {module_namespace.skeleton!r}, read from the "
                f"{module_namespace.source}). An entry in the other spelling "
                "may not bind on the served model (it did not on qwen3.5, "
                "job 489694; on NemotronH both spellings reach one module) — "
                "remove the entry's dir to re-export it under the "
                "checkpoint's spelling")
        raise LoraExportError(
            f"{stale} — a fresh export of the same input bytes is NOT "
            "byte-identical to the entry (exports are deterministic, so "
            f"the entry was not produced from these bytes by this exporter){hint}")
    return target, adapter_digests(target)._replace(name_hint=name_hint), True


def _install_or_reuse(root: Path, fresh: Path, target: Path, *,
                      name_hint: Optional[str],
                      intervention_hash: Optional[str], zero: bool,
                      module_namespace: Optional[ModuleNamespace] = None,
                      ) -> Tuple[Path, AdapterDigests, bool]:
    """Publish the complete export at ``fresh`` as ``target`` with one
    ``rename``, or — when an entry already sits there (before we looked,
    or landed by a concurrent writer between the look and the rename,
    which the kernel reports as EEXIST/ENOTEMPTY) — verify that entry
    against the fresh export (:func:`_reuse_verified`)."""
    if target.exists():
        return _reuse_verified(target, fresh, name_hint=name_hint,
                               intervention_hash=intervention_hash,
                               zero=zero, module_namespace=module_namespace)
    digests = adapter_digests(fresh)._replace(dir=str(target),
                                              name_hint=name_hint)
    _write_digests_file(fresh, digests, module_namespace=module_namespace)
    try:
        _rename_dir(fresh, target)
    except OSError as e:
        if isinstance(e, FileExistsError) or \
                e.errno in (errno.EEXIST, errno.ENOTEMPTY):
            return _reuse_verified(target, fresh, name_hint=name_hint,
                                   intervention_hash=intervention_hash,
                                   zero=zero, module_namespace=module_namespace)
        raise LoraExportError(
            f"cannot install {fresh} as bank entry {target}: {e}") from e
    _fsync_path(root)
    return target, digests, False


def _validated_name_hint(name_hint: Any) -> Optional[str]:
    if name_hint is None:
        return None
    if not isinstance(name_hint, str) or not name_hint:
        raise LoraExportError(
            f"name_hint must be None or a non-empty cell id, got "
            f"{name_hint!r}")
    return name_hint


def export_or_reuse(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                    root: Path, *, base_model: str,
                    base_revision: Optional[str] = None,
                    sidecar_bytes: bytes = b"",
                    name_hint: Optional[str] = None,
                    snapshot_dir: Optional[Path] = None,
                    ) -> Tuple[Path, AdapterDigests, bool]:
    """Bank a GENUINE adapter: ``(dir, digests, reused)``.

    Content-addressed by ``resolved_spec["intervention_hash"]`` (REQUIRED
    — the bank has no other address): the target is
    :func:`adapter_dir_for`. The export always runs, into a private
    ``root/.tmp-<ih>-<pid>-<time_ns>/`` through :func:`export_genuine_
    adapter` (layout DERIVED from ``base_model``'s cached config — never a
    stated prefix, so the fit layers are proven against the tree — and
    every module name spelled in the served checkpoint's namespace:
    ``snapshot_dir`` when the caller holds it, else the cache's snapshot of
    ``base_model``/``base_revision``; the :class:`ModuleNamespace` is
    recorded in the entry's ``adapter_digests.json``); then
    either it is published with ONE ``rename`` (``adapter_digests.json``
    written and fsynced first, so a reader never sees a partial entry —
    ``reused=False``), or, when an entry already sits at the target, that
    entry is verified against it (:func:`_reuse_verified`: its own record,
    a re-hash, ``petri_intervention_hash == ih``, a genuine marker, AND a
    byte-identical fresh export — ``reused=True``). A disagreeing entry is
    a refusal naming the dir; the bank never writes over one. The tmp dir
    is removed on every path, success or refusal. ``name_hint`` is the
    cell id the exporter was told; it rides in the digests and the
    record, and is never an address.

    The address does not move with the spelling: an entry exported under
    another namespace (the skeleton's, as older entries did) sits at the
    same ``intervention_hash`` and is refused as not byte-identical, with
    the refusal naming both spellings and the remedy (remove it) — never
    written over, never trusted by address."""
    if not isinstance(resolved_spec, Mapping):
        raise LoraExportError(
            f"resolved_spec must be a mapping, got "
            f"{type(resolved_spec).__name__}")
    ih = resolved_spec.get("intervention_hash")
    if ih is None:
        raise LoraExportError(
            "export_or_reuse: resolved_spec carries no intervention_hash — "
            "the bank is content-addressed by the cell's intervention "
            "identity, so an export without one has no address; refusing "
            "(export_genuine_adapter takes an explicit out_dir instead)")
    ih = _validated_hex16("resolved_spec intervention_hash", ih)
    name_hint = _validated_name_hint(name_hint)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = adapter_dir_for(root, ih)
    fresh = _tmp_dir(root, ih)
    try:
        _, namespace = _export_genuine(resolved_spec, artifact_bytes, fresh,
                                       base_model=base_model,
                                       sidecar_bytes=sidecar_bytes,
                                       base_revision=base_revision,
                                       snapshot_dir=snapshot_dir)
        return _install_or_reuse(root, fresh, target, name_hint=name_hint,
                                 intervention_hash=ih, zero=False,
                                 module_namespace=namespace)
    finally:
        shutil.rmtree(fresh, ignore_errors=True)


def export_or_reuse_zero(root: Path, *, base_model: str,
                         base_revision: Optional[str] = None, rank: int = 1,
                         snapshot_dir: Optional[Path] = None,
                         ) -> Tuple[Path, AdapterDigests, bool]:
    """Bank the ZERO adapter for ``base_model``: ``(dir, digests, reused)``.

    ``target_modules`` = :func:`checkpoint_target_modules` (every attention
    o_proj of the base, shapes off its meta skeleton, each NAMED as the
    served checkpoint names it; ``snapshot_dir`` when the caller
    holds it, else the cache's snapshot), the key =
    :func:`zero_adapter_key` over (base, revision, modules+shapes, rank) —
    so a base whose checkpoint spells the decoder differently from the
    skeleton (qwen3.5, NemotronH) banks its zero adapter under a NEW key,
    beside any entry the skeleton spelling produced,
    the target = :func:`zero_adapter_dir_for`. The export is
    :func:`export_zero_adapter` with ``lora_alpha=float(rank)`` — for the
    default rank 1 that IS its default (byte-identical config), and at
    any rank it keeps ``lora_alpha == r``, the scaling-1 form the
    lora-native attestation fingerprint is defined on (vLLM
    folds any other ``alpha/r`` into ``lora_b`` at load). The fresh export
    is held to :func:`verify_zero_adapter` (every tensor exactly zero)
    and to the compose-time contract on its config (the unique
    ``target_modules`` contents are exactly ``{"o_proj"}`` —
    ``export_zero_adapter`` writes one un-deduplicated short name per
    module, ``["o_proj", "o_proj", …]``, which is admitted as-is) before
    it is published or compared; the reuse branch re-verifies the entry
    all-zero too. Same atomic publish / verified-reuse discipline as
    :func:`export_or_reuse`."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise LoraExportError(f"rank must be a positive int, got {rank!r}")
    target_modules, namespace = checkpoint_target_modules(
        base_model, base_revision, snapshot_dir=snapshot_dir)
    key = zero_adapter_key(base_model=base_model, base_revision=base_revision,
                           target_modules=target_modules, rank=rank)
    return _bank_zero_adapter(
        root, key=key, base_model=base_model, expected_class=("o_proj",),
        namespace=namespace,
        export=lambda fresh: export_zero_adapter(
            fresh, base_model=base_model, rank=rank,
            target_modules=target_modules, lora_alpha=float(rank)))


def _bank_zero_adapter(root: Path, *, key: str, base_model: str,
                       expected_class: Tuple[str, ...],
                       namespace: Optional[ModuleNamespace],
                       export) -> Tuple[Path, AdapterDigests, bool]:
    """The shared bank step of :func:`export_or_reuse_zero` and
    :func:`export_zero_adapter_for_class`: ``export(fresh)`` writes the
    zero adapter into a private tmp dir under ``root``, it is held to
    :func:`verify_zero_adapter` (every tensor exactly zero) and to the
    compose-time contract on its config — the unique ``target_modules``
    contents are exactly THE CLASS ASKED FOR (``expected_class``; the
    o_proj exporter writes one un-deduplicated short name per module,
    ``["o_proj", "o_proj", …]``, admitted as the set ``{"o_proj"}``) —
    then published at :func:`zero_adapter_dir_for` with the atomic publish
    / verified-reuse discipline of :func:`_install_or_reuse`."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = zero_adapter_dir_for(root, key)
    fresh = _tmp_dir(root, f"{ZERO_ADAPTER_NAME}__{key}")
    try:
        export(fresh)
        verify_zero_adapter(fresh)
        cfg = json.loads((fresh / ADAPTER_CONFIG).read_text(encoding="utf-8"))
        short_names = cfg.get("target_modules")
        if not isinstance(short_names, list) or not short_names or \
                set(short_names) != set(expected_class):
            raise LoraExportError(
                f"zero adapter for {base_model!r}: target_modules="
                f"{short_names!r} — the lora-native path admits an adapter "
                "whose unique short names are exactly the class asked for "
                f"({sorted(expected_class)}); the zero exporter named "
                "something else; refusing")
        return _install_or_reuse(root, fresh, target,
                                 name_hint=ZERO_ADAPTER_NAME,
                                 intervention_hash=None, zero=True,
                                 module_namespace=namespace)
    finally:
        shutil.rmtree(fresh, ignore_errors=True)


class ClassTargets(NamedTuple):
    """What :func:`checkpoint_target_modules_for_class` resolves for one
    module-set class on one base: ``target_modules`` — every adapted
    module's CHECKPOINT-spelled name -> ``(in_features, out_features)``
    for a linear module, ``(num_experts, in_features, out_features)`` for a
    routed-expert PARAMETER (``...mlp.experts.gate_up_proj`` /
    ``...mlp.experts.down_proj``; BOTH always present where the class
    reaches routed experts — the obligatory companion included);
    ``namespace`` the module-namespace record; ``shape`` the class's
    metadata in the general blob's own vocabulary (:class:`_GeneralShape`:
    modules, set id, per-expert rank map, the ``targets`` block WITHOUT
    the companion, the routed ``r_e``)."""

    target_modules: Dict[str, Tuple[int, ...]]
    namespace: ModuleNamespace
    shape: _GeneralShape


def checkpoint_target_modules_for_class(base_model: str, revision: Optional[str],
                                        modules, rank: int,
                                        per_expert_rank: Optional[int] = None,
                                        snapshot_dir: Optional[Path] = None,
                                        ) -> ClassTargets:
    """Every module a module-set CLASS adapts on ``base_model``, resolved
    the way a genuine general-shape export is held (``lora_modules.
    resolve_targets`` over the meta skeleton — the capability probe, never
    a table), each NAMED as the served checkpoint spells the decoder
    (:func:`resolve_module_namespace`'s prefix), with the shapes off the
    skeleton — what :func:`export_zero_adapter_for_class` is exported
    from. Where the class reaches routed experts, the MISSING half of the
    pair (the ``down`` set's gate_up) is added with its shape derived from
    the present one (:func:`_experts_companion_dims`), because the engine
    asserts both at load."""
    from ...extraction.lora_modules import (LoRAModuleError, normalise_modules,
                                            resolve_targets, set_id_for,
                                            targets_summary)

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise LoraExportError(f"rank must be a positive int, got {rank!r}")
    try:
        mods = normalise_modules(modules)
    except LoRAModuleError as e:
        raise LoraExportError(f"module set refused — {e}") from e
    cfg, skeleton = _decoder_skeleton(base_model, revision)
    layout = _layout_from_skeleton(base_model, cfg, skeleton)
    _, namespace = resolve_module_namespace(base_model, revision, layout,
                                            snapshot_dir=snapshot_dir)
    prefix = namespace.checkpoint
    try:
        targets = resolve_targets(skeleton, mods, rank=rank,
                                  per_expert_rank=per_expert_rank)
    except LoRAModuleError as e:
        raise LoraExportError(
            f"module set {list(mods)} does not resolve on {base_model!r}'s "
            f"module tree — {e}") from e
    target_modules: Dict[str, Tuple[int, ...]] = {}
    experts_seen: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
    routed: Optional[int] = None
    for t in targets:
        name = f"{prefix}.{t.layer}.{t.site}"
        if t.kind == "linear":
            target_modules[name] = (int(t.in_features), int(t.out_features))
        else:
            n_e = int(t.num_experts or 0)
            routed = int(t.rank)
            target_modules[name] = (n_e, int(t.in_features), int(t.out_features))
            path, param = _experts_module_path("zero", t.module, t.site)
            experts_seen.setdefault(f"{prefix}.{t.layer}.{path}", {})[param] = (
                "experts", n_e, int(t.in_features), int(t.out_features))
    for module_name, param, n_e, d_in, d_out in _missing_expert_halves(experts_seen):
        target_modules[f"{module_name}.{param}"] = (n_e, d_in, d_out)
    summary = targets_summary(targets)
    keys = tuple(sorted((int(l), m) for m, rec in summary.items() for l in rec["layers"]))
    shape = _GeneralShape(
        modules=mods, set_id=set_id_for(mods),
        per_expert_rank=({"shared_expert": rank, "routed_expert": routed}
                         if routed is not None else None),
        init_scheme="zero", targets=summary, keys=keys,
        layers=tuple(sorted({l for l, _ in keys})), routed_rank=routed)
    return ClassTargets(target_modules=target_modules, namespace=namespace,
                        shape=shape)


def _export_zero_adapter_for_class(out_dir: Path, *, base_model: str, rank: int,
                                   resolved: ClassTargets) -> Path:
    """Write the ZERO adapter of a module-set class: an all-zero PEFT pair
    for every entry of ``resolved.target_modules`` — a linear module's
    ``lora_A [r, in]`` / ``lora_B [out, r]`` under its own name, a routed-
    expert parameter's flattened ``ParamWrapper`` pair (``[(r_e·E), in]`` /
    ``[out, (r_e·E)]``) under the experts module's INNER (gate_up) or OUTER
    (down) wrapper name — and the config of a general-shape export
    (:func:`_general_config`: real HF ``target_modules``,
    ``target_parameters`` + ``rank_pattern``/``alpha_pattern`` where
    experts are reached) marked ``petri_zero_adapter``."""
    from safetensors.torch import save_file  # transformers-stack dep

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors: Dict[str, torch.Tensor] = {}
    for module, dims in sorted(resolved.target_modules.items()):
        if len(dims) == 2:
            in_f, out_f = dims
            tensors[f"base_model.model.{module}.lora_A.weight"] = torch.zeros(
                (rank, in_f), dtype=torch.float32)
            tensors[f"base_model.model.{module}.lora_B.weight"] = torch.zeros(
                (out_f, rank), dtype=torch.float32)
        else:
            n_e, in_f, out_f = dims
            r_e = int(resolved.shape.routed_rank or 0)
            if r_e < 1:
                raise LoraExportError(
                    f"zero adapter for {base_model!r}: {module} is a routed-expert "
                    "parameter but the class resolved no per-expert rank; refusing")
            path, param = module.rsplit(".", 1)
            if param not in EXPERTS_PEFT_SUFFIX:
                raise LoraExportError(
                    f"zero adapter for {base_model!r}: {module!r} is not a "
                    f"gate_up_proj/down_proj experts parameter; refusing")
            _emit_experts(tensors, path, param,
                          torch.zeros((n_e * r_e, in_f), dtype=torch.float32),
                          torch.zeros((out_f, n_e * r_e), dtype=torch.float32))
    config = {
        **_peft_config_header(base_model),
        "r": rank,
        "lora_alpha": float(rank),   # scaling 1, the fingerprint's form
        **_general_config(resolved.shape),
        # provenance: this is the deliberate no-op control adapter
        "petri_zero_adapter": True,
    }
    (out_dir / ADAPTER_CONFIG).write_text(
        json.dumps(config, indent=2, sort_keys=True))
    save_file({k: tensors[k] for k in sorted(tensors)}, str(out_dir / ADAPTER_WEIGHTS))
    return out_dir


def export_zero_adapter_for_class(root: Path, *, base_model: str,
                                  base_revision: Optional[str] = None,
                                  rank: int = 1, lora_modules,
                                  per_expert_rank: Optional[int] = None,
                                  snapshot_dir: Optional[Path] = None,
                                  ) -> Tuple[Path, AdapterDigests, bool]:
    """Bank the ZERO adapter of a module-set CLASS: ``(dir, digests,
    reused)``. For ``lora_modules == ["o_proj"]`` this IS
    :func:`export_or_reuse_zero` — byte-identical, the same dir.
    Otherwise the class's modules are resolved on the base
    (:func:`checkpoint_target_modules_for_class`), the key is
    :func:`zero_adapter_key` over (base, revision, every
    module's checkpoint name + shape — experts as ``(E, in, out)``
    triples — rank, and the routed ``r_e``), the export is
    :func:`_export_zero_adapter_for_class` (all-zero factors for every
    module of the class, the two obligatory expert pairs included), and
    the bank step is :func:`_bank_zero_adapter` with the class asked for
    as the admitted ``target_modules`` set. ``native_standup`` selects
    this generator by the manifest's class."""
    from ...extraction.lora_modules import (PAPER_LORA_MODULES, LoRAModuleError,
                                            normalise_modules)

    try:
        mods = normalise_modules(lora_modules)
    except LoRAModuleError as e:
        raise LoraExportError(f"module set refused — {e}") from e
    if mods == PAPER_LORA_MODULES:
        return export_or_reuse_zero(root, base_model=base_model,
                                    base_revision=base_revision, rank=rank,
                                    snapshot_dir=snapshot_dir)
    resolved = checkpoint_target_modules_for_class(
        base_model, base_revision, mods, rank, per_expert_rank, snapshot_dir)
    key = zero_adapter_key(base_model=base_model, base_revision=base_revision,
                           target_modules=resolved.target_modules, rank=rank,
                           per_expert_rank=resolved.shape.routed_rank)
    return _bank_zero_adapter(
        root, key=key, base_model=base_model, expected_class=mods,
        namespace=resolved.namespace,
        export=lambda fresh: _export_zero_adapter_for_class(
            fresh, base_model=base_model, rank=rank, resolved=resolved))
