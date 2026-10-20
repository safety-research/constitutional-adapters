"""Synthetic CHECKPOINT SNAPSHOTS for the LoRA-export tests (native-multilora).

The export spells every module name in the SERVED CHECKPOINT's namespace, read
off the snapshot's weights manifest (``lora_export.checkpoint_weight_names``:
the ``model.safetensors.index.json`` weight map, else every ``*.safetensors``
header). A developer / CI cache holds the pinned bases' config + tokenizer
only, so a test that exports through the derived layout hands the exporter a
snapshot dir shaped like the real checkpoint: the SAME o_proj tensor names the
real snapshot carries (:data:`CHECKPOINT_PREFIX`, measured off the cluster's
HF cache on 2026-09-12 — qwen3.5 and GLM as an index over shards, gemma-4 and
Nemotron as one ``model.safetensors``), plus decoy names beside the decoder
(:data:`CHECKPOINT_DECOYS`: the ones that make the resolution non-trivial,
qwen3.5's MTP head above all). Values are 1-element tensors; only the NAMES
matter, and only the header is ever read.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

from sisteer.capeval.serving import lora_export as lx

#: the pinned bases' checkpoint decoder prefix — what their weight names carry
#: (the skeleton's spelling is ``model.layers`` on qwen3.5 and Nemotron; the
#: two agree on gemma-4 and GLM)
CHECKPOINT_PREFIX = {
    "google/gemma-4-E4B-it": "model.language_model.layers",
    "Qwen/Qwen3.5-4B": "model.language_model.layers",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": "backbone.layers",
    "zai-org/GLM-4.6V-Flash": "model.language_model.layers",
}
#: the single-file checkpoints (one ``model.safetensors``); the others ship an
#: index over shards
SINGLE_FILE = frozenset({"google/gemma-4-E4B-it",
                         "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"})
#: tensor names beside the decoder's o_proj in each real checkpoint — the
#: measured ones (qwen3.5's ``mtp.layers.0.self_attn.o_proj.weight`` is the
#: decoy that would tie with a layer-0 attention block; Nemotron's layer-12
#: siblings; qwen's vision tower) plus tower-shaped names where a real key was
#: not copied verbatim (synthetic, marked)
CHECKPOINT_DECOYS = {
    "google/gemma-4-E4B-it": (
        "model.language_model.embed_tokens.weight",
        "model.vision_tower.encoder.layers.0.self_attn.out_proj.weight",   # synthetic
        "model.audio_tower.conformer.0.attention.post.weight",              # synthetic
    ),
    "Qwen/Qwen3.5-4B": (
        "mtp.layers.0.self_attn.o_proj.weight",
        "model.visual.blocks.0.mlp.linear_fc1.weight",
        "model.visual.merger.linear_fc1.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",    # synthetic
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": (
        "backbone.layers.12.mixer.q_proj.weight",
        "backbone.layers.12.norm.weight",
        "backbone.layers.0.mixer.in_proj.weight",                          # synthetic
        "backbone.embeddings.weight",
    ),
    "zai-org/GLM-4.6V-Flash": (
        "model.language_model.embed_tokens.weight",
        "model.visual.blocks.0.attn.proj.weight",                          # synthetic
    ),
}


def oproj_names(prefix: str, layers: Iterable[int],
                site: str = lx.DEFAULT_OPROJ_SITE) -> list:
    """``f"{prefix}.{l}.{site}.weight"`` for every layer."""
    return [f"{prefix}.{l}.{site}.weight" for l in layers]


def write_index_snapshot(d: Path, names: Sequence[str], *, shards: int = 2,
                         ) -> Path:
    """A sharded checkpoint's snapshot as its INDEX alone: ``model.
    safetensors.index.json`` whose ``weight_map`` spreads ``names`` over
    ``shards`` shard files (which need not exist — the exporter reads names
    from the index, never the shards)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    weight_map = {n: f"model-{i % shards + 1:05d}-of-{shards:05d}.safetensors"
                  for i, n in enumerate(sorted(names))}
    (d / lx.WEIGHT_INDEX_FILE).write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=1))
    return d


def write_safetensors_snapshot(d: Path, names: Sequence[str], *,
                               filename: str = "model.safetensors") -> Path:
    """A single-file checkpoint's snapshot: a REAL (tiny) safetensors file
    holding one 1-element fp32 tensor per name, written by the library
    itself, so the exporter's header read is exercised on the genuine
    format."""
    from safetensors.torch import save_file

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    save_file({n: torch.zeros(1) for n in names}, str(d / filename),
              metadata={"format": "pt"})
    return d


def snapshot_like(base_model: str, d: Path, *,
                  layout: Optional[lx.DecoderLayout] = None,
                  revision: Optional[str] = None) -> Path:
    """A snapshot dir shaped like ``base_model``'s REAL checkpoint: its o_proj
    modules named under :data:`CHECKPOINT_PREFIX` for every layer of
    ``layout`` (resolved off the cached config when not given) at the
    layout's site, plus :data:`CHECKPOINT_DECOYS`; an index for the sharded
    checkpoints, a real safetensors file for the single-file ones."""
    if base_model not in CHECKPOINT_PREFIX:
        raise KeyError(f"no measured checkpoint namespace for {base_model!r}; "
                       f"known: {sorted(CHECKPOINT_PREFIX)}")
    if layout is None:
        layout = lx.resolve_decoder_layout(base_model, revision)
    names = oproj_names(CHECKPOINT_PREFIX[base_model], layout.oproj_layers,
                        layout.oproj_site) + list(CHECKPOINT_DECOYS[base_model])
    if base_model in SINGLE_FILE:
        return write_safetensors_snapshot(d, names)
    return write_index_snapshot(d, names)
