"""FAKES carrying the pinned vLLM 0.25.1 wheel's SIGNATURES and control flow
for the residual-slot patch harness. Every class here mirrors one wheel surface
the patches land on, so the CPU tests exercise the REAL ``native_residual``
wrappers on the same call shapes the worker will see:

- ``LoRAModel.from_local_checkpoint`` / ``from_lora_tensors`` /
  ``_should_skip_module`` (``vllm/lora/lora_model.py``): the unexpected-module
  check and the tensor walk, ``skip_prefixes`` honoured where the wheel honours
  it — an UNPATCHED load of an entry carrying ``sisteer_residual.*`` keys raises
  the wheel's ``parse_fine_tuned_lora_name`` "unsupported LoRA weight" ValueError
  (``lora/utils.py:198`` @0.25.1) from inside its unexpected-module check;
- ``LoRAModelManager`` / ``LRUCacheLoRAModelManager`` (``model_manager.py``):
  ``activate_adapter`` (first free slot, ``lora_index_to_id``), ``_deactivate_
  adapter`` (erases the slot), ``deactivate_adapter``, ``remove_all_adapters``
  (resets ``lora_index_to_id`` BEFORE clearing the active set), the LRU
  ``_on_remove -> deactivate_fn`` callback, ``set_adapter_mapping``'s skip of an
  unchanged mapping;
- ``PunicaWrapperBase._update_base_metadata`` + ``PunicaWrapperGPU.update_
  metadata`` + ``LoRAKernelMeta.prepare_tensors`` (``punica_base.py``,
  ``punica_gpu.py``, ``lora_kernel_metadata.py``): ``convert_mapping``'s -1 for
  id 0, the ``indices_len`` bookkeeping, and the kernel metadata's EARLY RETURN
  on an all-unadapted batch (the stale-buffer case);
- ``LRUCacheWorkerLoRAManager`` (``worker_manager.py``): ``_load_adapter``'s
  exact ``from_local_checkpoint`` kwargs, ``add_adapter``, ``set_active_
  adapters``, ``add_dummy_lora``;
- ``LoRAModelRunnerMixin.load_lora_model`` / ``_set_active_loras`` /
  ``maybe_setup_dummy_loras`` and a runner constructed BEFORE ``load_model``
  (fact 11);
- the decoder-layer classes: Llama's ``(positions, hidden_states, residual)``
  positional forward with no ``layer_idx``; gemma4's ``(hidden_states, None)``
  return with a ``layer_idx``; Qwen3Next's keyword ``positions`` with
  ``layer_idx``; Qwen3.5 DEFINING ONLY ``__init__`` and bypassing the parent's
  (``super(Qwen3NextDecoderLayer, self).__init__()``); NemotronH's four
  classes taking ``layer_idx`` as an argument and storing NO attribute.

Factories return FRESH classes per call so each test patches its own copies.
"""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn

NO_LORA_ID = 0


# ---------------------------------------------------------------------------
# config shapes
# ---------------------------------------------------------------------------


def fake_vllm_config(*, hidden_size: int, num_layers: int, dtype: torch.dtype,
                     max_loras: int, max_num_seqs: int = 8,
                     max_num_batched_tokens: int = 64, max_lora_rank: int = 4,
                     lora_dtype: torch.dtype = torch.bfloat16,
                     max_cpu_loras: Optional[int] = None) -> Any:
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=num_layers, hidden_size=hidden_size),
        dtype=dtype, get_hidden_size=lambda: hidden_size, get_vocab_size=lambda: 97)
    return SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=max_num_batched_tokens),
        lora_config=SimpleNamespace(max_loras=max_loras,
                                    max_cpu_loras=(max_cpu_loras if max_cpu_loras is not None
                                                   else max_loras),
                                    max_lora_rank=max_lora_rank, lora_dtype=lora_dtype),
        parallel_config=SimpleNamespace(enable_expert_parallel=False))


@dataclass(frozen=True)
class FakeLoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str
    load_inplace: bool = False


@dataclass(frozen=True)
class FakeLoRAMapping:
    index_mapping: Tuple[int, ...]
    prompt_mapping: Tuple[int, ...]
    is_prefill: bool = False
    type: str = "language"


def parse_fine_tuned_lora_name(name: str) -> Tuple[str, bool]:
    """``lora/utils.parse_fine_tuned_lora_name`` without a weights mapper."""
    parts = name.split(".")
    if parts[0] == "base_model" and parts[1] == "model":
        parts = parts[2:]
    if len(parts) >= 3 and parts[-1] == "weight" and parts[-2] in ("lora_A", "lora_B"):
        return ".".join(parts[:-2]), parts[-2] == "lora_A"
    raise ValueError(f"{name} is unsupported LoRA weight")


# ---------------------------------------------------------------------------
# the LoRA path
# ---------------------------------------------------------------------------


def make_fake_lora_classes() -> SimpleNamespace:
    """Fresh copies of the LoRA-path classes."""

    class LRUCache:
        """``vllm.utils.cache.LRUCache`` as ``AdapterLRUCache`` uses it: insertion
        order, ``__getitem__`` touches, ``pop`` -> ``_on_remove``, ``clear`` pops
        oldest-first through ``_on_remove``."""

        def __init__(self, capacity: int, deactivate_fn):
            self.capacity = capacity
            self.deactivate_fn = deactivate_fn
            self._d: Dict[int, Any] = {}

        def __contains__(self, k):
            return k in self._d

        def __len__(self):
            return len(self._d)

        def __getitem__(self, k):
            v = self._d.pop(k)
            self._d[k] = v
            return v

        def __setitem__(self, k, v):
            self._d.pop(k, None)
            self._d[k] = v
            while len(self._d) > self.capacity:
                self.remove_oldest()

        def get(self, k, default=None):
            return self[k] if k in self._d else default

        def touch(self, k):
            self._d[k] = self._d.pop(k)

        def _on_remove(self, key, value):
            self.deactivate_fn(key)

        def pop(self, k, default=None):
            if k not in self._d:
                return default
            v = self._d.pop(k)
            self._on_remove(k, v)
            return v

        def remove_oldest(self, *, remove_pinned: bool = False):
            if not self._d:
                return
            self.pop(next(iter(self._d)))

        def clear(self):
            while len(self._d) > 0:
                self.remove_oldest(remove_pinned=True)

        def values(self):
            return list(self._d.values())

        def keys(self):
            return list(self._d.keys())

    class LoRALayerWeights:
        def __init__(self, module_name: str, rank: int):
            self.module_name, self.rank = module_name, rank
            self.lora_a = self.lora_b = None
            self.lora_alpha, self.is_packed = rank, False

    class PEFTHelper:
        def __init__(self, cfg: Dict[str, Any]):
            self.r = int(cfg["r"])
            self.lora_alpha = float(cfg.get("lora_alpha", self.r))
            self.target_modules = list(cfg.get("target_modules", []))

    class LoRAModel:
        def __init__(self, lora_model_id: int, rank: int, loras: Dict[str, Any]):
            self.id, self.rank, self.loras = lora_model_id, rank, loras

        @staticmethod
        def _should_skip_module(module_name: str, skip_prefixes) -> bool:
            for prefix in skip_prefixes:
                if f".{prefix}" in module_name or module_name.startswith(prefix):
                    return True
            return False

        @classmethod
        def from_lora_tensors(cls, lora_model_id, tensors, peft_helper, device="cuda",
                              dtype=None, model_vocab_size=None, weights_mapper=None,
                              skip_prefixes=None):
            loras: Dict[str, Any] = {}
            for tensor_name, tensor in tensors.items():
                if skip_prefixes and cls._should_skip_module(tensor_name, skip_prefixes):
                    continue
                module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name)
                if module_name not in loras:
                    loras[module_name] = LoRALayerWeights(module_name, peft_helper.r)
                t = tensor.to(device=device, dtype=dtype)
                if is_lora_a:
                    loras[module_name].lora_a = t
                else:
                    loras[module_name].lora_b = t
            return cls(lora_model_id, peft_helper.r, loras)

        @classmethod
        def from_local_checkpoint(cls, lora_dir, expected_lora_modules, peft_helper, *,
                                  lora_model_id=None, device="cuda", dtype=None,
                                  model_vocab_size=None, weights_mapper=None,
                                  tensorizer_config_dict=None, skip_prefixes=None,
                                  moe_ep_spec=None):
            from safetensors import safe_open

            lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
            tensors: Dict[str, torch.Tensor] = {}
            unexpected_modules = []
            with safe_open(lora_tensor_path, framework="pt") as f:
                for lora_module in f.keys():
                    if skip_prefixes and cls._should_skip_module(lora_module, skip_prefixes):
                        continue
                    module_name, _ = parse_fine_tuned_lora_name(lora_module)
                    if module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
                        unexpected_modules.append(module_name)
                if unexpected_modules:
                    raise ValueError(
                        f"While loading {lora_dir}, expected target modules in "
                        f"{expected_lora_modules} but received {unexpected_modules}. "
                        "Please verify that the loaded LoRA module is correct")
                for module in f.keys():
                    tensors[module] = f.get_tensor(module)
            return cls.from_lora_tensors(
                lora_model_id=lora_model_id if lora_model_id is not None else 0,
                tensors=tensors, peft_helper=peft_helper, device=device, dtype=dtype,
                model_vocab_size=model_vocab_size, weights_mapper=weights_mapper,
                skip_prefixes=skip_prefixes)

    def convert_mapping(mapping, lora_index_to_id, max_loras, vocab_size, extra_vocab_size,
                        device):
        lora_id_to_index = {lid: i for i, lid in enumerate(lora_index_to_id) if lid is not None}
        base = [lora_id_to_index[x] if x > 0 else -1 for x in mapping.index_mapping]
        prompt = [lora_id_to_index[x] if x > 0 else -1 for x in mapping.prompt_mapping]
        base_t = torch.tensor(base, dtype=torch.long, device=device)
        prompt_t = torch.tensor(prompt, dtype=torch.long, device=device)
        padded = torch.where(prompt_t == -1, torch.tensor(max_loras - 1), prompt_t)
        padded = torch.arange(0, len(padded), device=device) + padded * len(padded)
        emb = torch.stack([base_t.clamp(min=0) * extra_vocab_size,
                           base_t.clamp(min=0) * (vocab_size + extra_vocab_size)])
        return base_t, prompt_t, padded, emb, [base_t.shape[-1], prompt_t.shape[-1],
                                               padded.shape[-1], emb.shape[-1]]

    class LoRAKernelMeta:
        """``lora_kernel_metadata.LoRAKernelMeta`` — ``prepare_tensors`` RETURNS
        BEFORE copying the mapping when every token is unadapted."""

        def __init__(self, max_loras: int, max_num_batched_tokens: int, device):
            self.token_lora_mapping = torch.full((max_num_batched_tokens,), -7,
                                                 dtype=torch.int32, device=device)
            self.active_lora_ids = torch.empty(max_loras + 1, dtype=torch.int32, device=device)
            self.num_tokens_per_lora = torch.empty(max_loras + 1, dtype=torch.int32, device=device)
            self.lora_token_start_loc = torch.empty(max_loras + 2, dtype=torch.int32, device=device)
            self.no_lora_flag_cpu = torch.zeros(1, dtype=torch.bool)
            self.num_active_loras_cpu = torch.zeros(1, dtype=torch.int32)
            self.prepare_calls = 0

        def _reset(self):
            self.active_lora_ids.fill_(-1)
            self.num_tokens_per_lora.fill_(0)
            self.lora_token_start_loc.fill_(0)
            self.no_lora_flag_cpu.fill_(False)
            self.num_active_loras_cpu.fill_(0)

        def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
            self.prepare_calls += 1
            self._reset()
            no_lora = torch.all(token_lora_mapping == -1)
            self.no_lora_flag_cpu[0] = no_lora
            if no_lora:
                return                                   # EARLY EXIT (the stale buffer)
            num_tokens = token_lora_mapping.size(0)
            self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping, non_blocking=True)
            ids = torch.unique(token_lora_mapping, sorted=True)
            self.active_lora_ids[: ids.shape[0]].copy_(ids)
            self.num_active_loras_cpu[0] = ids.shape[0]

    class PunicaWrapperBase:
        def __init__(self, max_num_batched_tokens: int, max_batches: int, device, **kwargs):
            self._token_lora_indices = torch.empty(max_num_batched_tokens, dtype=torch.long,
                                                   device=device)
            self._sampler_indices = torch.empty(max_num_batched_tokens, dtype=torch.long,
                                                device=device)
            self._sampler_indices_padded = torch.empty(max_num_batched_tokens, dtype=torch.long,
                                                       device=device)
            self._embeddings_indices = torch.empty(2, max_num_batched_tokens, dtype=torch.long,
                                                   device=device)
            self.indices_len = [None] * 4
            self.device = device
            self.is_prefill = False
            self.no_lora = False

        def _update_base_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size):
            extra_vocab_size = 0
            base_indices, sampler_indices, sampler_indices_padded, embeddings_indices, \
                indices_len = convert_mapping(mapping, lora_index_to_id, max_loras, vocab_size,
                                              extra_vocab_size, self.device)
            self._token_lora_indices[: base_indices.shape[0]].copy_(base_indices)
            self._sampler_indices[: sampler_indices.shape[0]].copy_(sampler_indices)
            self._sampler_indices_padded[: sampler_indices_padded.shape[0]].copy_(
                sampler_indices_padded)
            self._embeddings_indices[: embeddings_indices.shape[0],
                                     : embeddings_indices.shape[1]].copy_(embeddings_indices)
            self.indices_len[:] = indices_len

        @property
        def token_lora_indices(self) -> torch.Tensor:
            return self._token_lora_indices[: self.indices_len[0]]

        @property
        def sampler_indices(self) -> torch.Tensor:
            return self._sampler_indices[: self.indices_len[1]]

    class PunicaWrapperGPU(PunicaWrapperBase):
        def __init__(self, max_num_batched_tokens: int, max_batches: int, device,
                     max_loras: int = 1, **kwargs):
            super().__init__(max_num_batched_tokens, max_batches, device, **kwargs)
            self.max_loras = max_loras
            self.token_mapping_meta = LoRAKernelMeta(max_loras, max_num_batched_tokens, device)
            self.prompt_mapping_meta = LoRAKernelMeta(max_loras, max_num_batched_tokens, device)

        def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kwargs):
            self.is_prefill = mapping.is_prefill
            self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
            self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
            self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    class LoRAModelManager:
        def __init__(self, model, max_num_seqs, max_num_batched_tokens, vocab_size,
                     lora_config, device, vllm_config):
            self.model = model
            self.lora_config = lora_config
            self.device = device
            self.max_num_seqs = max_num_seqs
            assert self.capacity >= self.lora_slots
            self._registered_adapters = LRUCache(self.capacity, self.deactivate_adapter)
            self._active_adapters = LRUCache(self.lora_slots, self._deactivate_adapter)
            self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
            self.lora_index_to_id = [None] * self.lora_slots
            self.vocab_size = vocab_size
            self.modules: Dict[str, Any] = {}
            self._last_mapping = None
            self.punica_wrapper = PunicaWrapperGPU(max_num_batched_tokens, max_num_seqs, device,
                                                   max_loras=self.lora_slots)

        @property
        def capacity(self) -> int:
            return self.lora_config.max_cpu_loras

        @property
        def lora_slots(self) -> int:
            return self.lora_config.max_loras

        def activate_adapter(self, lora_id: int) -> bool:
            if lora_id in self._active_adapters:
                return False
            first_free_slot = next(((i, lid) for i, lid in enumerate(self.lora_index_to_id)
                                    if lid is None), None)
            if first_free_slot is None:
                raise ValueError("No free lora slots")
            index, _ = first_free_slot
            self._active_adapters[lora_id] = None
            lora_model = self._registered_adapters[lora_id]
            self.lora_index_to_id[index] = lora_model.id
            for module_name, module in self.modules.items():   # no execution modules here
                module.reset_lora(index)
            return True

        def _deactivate_adapter(self, lora_id: int):
            try:
                index = self.lora_index_to_id.index(lora_id)
                self.lora_index_to_id[index] = None
            except ValueError:
                pass

        def _add_adapter(self, lora):
            self._registered_adapters[lora.id] = lora

        def deactivate_adapter(self, adapter_id: int) -> bool:
            if adapter_id not in self._active_adapters:
                return False
            self._deactivate_adapter(adapter_id)
            self._active_adapters.pop(adapter_id, None)
            return True

        def add_adapter(self, adapter) -> bool:
            if adapter.id in self._registered_adapters:
                return False
            if len(self._registered_adapters) >= self.capacity:
                raise RuntimeError("No free adapter slots.")
            self._add_adapter(adapter)
            return True

        def set_adapter_mapping(self, mapping) -> None:
            if self._last_mapping != mapping:
                self._set_adapter_mapping(mapping)
                self._last_mapping = mapping

        def _set_adapter_mapping(self, mapping) -> None:
            self.punica_wrapper.update_metadata(mapping, self.lora_index_to_id,
                                                self.lora_slots + 1, self.vocab_size)

        def remove_adapter(self, adapter_id: int) -> bool:
            self.deactivate_adapter(adapter_id)
            if adapter_id not in self._registered_adapters:
                return False
            self._registered_adapters.pop(adapter_id, None)
            return True

        def remove_all_adapters(self):
            self._registered_adapters.clear()
            self.lora_index_to_id = [None] * self.lora_slots
            self._active_adapters.clear()

        def get_adapter(self, adapter_id: int):
            return self._registered_adapters.get(adapter_id)

        def list_adapters(self) -> Dict[int, Any]:
            return {k: self._registered_adapters.get(k) for k in self._registered_adapters.keys()}

    class LRUCacheLoRAModelManager(LoRAModelManager):
        def add_adapter(self, lora) -> bool:
            if lora.id not in self._registered_adapters:
                self._add_adapter(lora)
                return True
            self._registered_adapters.touch(lora.id)
            return False

        def activate_adapter(self, lora_id: int) -> bool:
            if lora_id not in self._active_adapters and \
                    len(self._active_adapters) >= self.lora_slots:
                self._active_adapters.remove_oldest()
            result = super().activate_adapter(lora_id)
            self._active_adapters.touch(lora_id)
            return result

        def remove_oldest_adapter(self) -> bool:
            if len(self._registered_adapters) > 0:
                self._registered_adapters.remove_oldest()
                return True
            return False

    class LRUCacheWorkerLoRAManager:
        _manager_cls = LRUCacheLoRAModelManager

        def __init__(self, vllm_config, device, embedding_modules, lora_model_cls=LoRAModel):
            self._lora_model_cls = lora_model_cls
            self.embedding_modules = embedding_modules
            self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            self.max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.vocab_size = vllm_config.model_config.get_vocab_size()
            self.lora_config = vllm_config.lora_config
            self.device = device
            self._adapter_manager = None

        def create_lora_manager(self, model, vllm_config):
            self._adapter_manager = self._manager_cls(
                model=model, max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens, vocab_size=self.vocab_size,
                lora_config=self.lora_config, device=self.device, vllm_config=vllm_config)
            return model

        def _load_adapter(self, lora_request):
            cfg = json.loads((Path(lora_request.lora_path) / "adapter_config.json").read_text())
            peft_helper = PEFTHelper(cfg)
            model = self._adapter_manager.model
            lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)
            return self._lora_model_cls.from_local_checkpoint(
                lora_request.lora_path, {"o_proj"}, peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id, device="cpu",
                dtype=self.lora_config.lora_dtype, model_vocab_size=self.vocab_size,
                tensorizer_config_dict=None, weights_mapper=None,
                skip_prefixes=lora_skip_prefixes, moe_ep_spec=None)

        def add_adapter(self, lora_request) -> bool:
            if lora_request.lora_int_id not in self.list_adapters() or lora_request.load_inplace:
                lora = self._load_adapter(lora_request)
                self._adapter_manager.remove_adapter(lora.id)
                if len(self._adapter_manager._registered_adapters) + 1 > \
                        self._adapter_manager.capacity:
                    self._adapter_manager.remove_oldest_adapter()
                loaded = self._adapter_manager.add_adapter(lora)
            else:
                loaded = self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
            self._adapter_manager.activate_adapter(lora_request.lora_int_id)
            return loaded

        def add_dummy_lora(self, lora_request, rank: int) -> bool:
            if lora_request.lora_int_id in self.list_adapters():
                return False
            dummy = LoRAModel(lora_request.lora_int_id, rank, {})
            return self._adapter_manager.add_adapter(dummy)

        def _apply_adapters(self, lora_requests) -> None:
            loras_map = {r.lora_int_id: r for r in lora_requests if r}
            if len(loras_map) > self._adapter_manager.lora_slots:
                raise RuntimeError("too many LoRAs for the GPU slots")
            for lora in loras_map.values():
                self.add_adapter(lora)

        def set_active_adapters(self, requests, mapping) -> None:
            self._apply_adapters(requests)
            self._adapter_manager.set_adapter_mapping(mapping)

        def remove_all_adapters(self):
            self._adapter_manager.remove_all_adapters()

        def list_adapters(self):
            return set(self._adapter_manager.list_adapters())

    class LoRAModelRunnerMixin:
        def load_lora_model(self, model, vllm_config, device):
            self.lora_manager = LRUCacheWorkerLoRAManager(vllm_config, device, {})
            return self.lora_manager.create_lora_manager(model, vllm_config)

        def _set_active_loras(self, prompt_lora_mapping, token_lora_mapping, lora_requests,
                              mapping_type=None) -> None:
            if not hasattr(self, "lora_manager"):
                raise RuntimeError("LoRA is not enabled. Use --enable-lora to enable LoRA.")
            mapping = FakeLoRAMapping(tuple(token_lora_mapping), tuple(prompt_lora_mapping),
                                      is_prefill=True)
            self.lora_manager.set_active_adapters(lora_requests, mapping)

        def set_active_loras(self, input_batch, num_scheduled_tokens, num_sampled_tokens=None,
                             mapping_type=None) -> None:
            """The wheel's V1 entry (``lora_model_runner_mixin.py:73`` @0.25.1):
            the mappings off ``InputBatch.make_lora_inputs``, then ``_set_active_loras``."""
            if num_sampled_tokens is None:
                num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)
            prompt_lora_mapping, token_lora_mapping, lora_requests = \
                input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
            return self._set_active_loras(prompt_lora_mapping, token_lora_mapping, lora_requests,
                                          mapping_type)

        @contextmanager
        def maybe_setup_dummy_loras(self, lora_config, remove_lora: bool = True):
            if lora_config is None:
                yield
                return
            num_loras = lora_config.max_loras
            rank = lora_config.max_lora_rank if lora_config.max_lora_rank < 8 else 8
            reqs = {FakeLoRARequest(f"warmup_{i}", i, "/not/a/real/path")
                    for i in range(1, num_loras + 1)}
            for lr in reqs:
                self.lora_manager.add_dummy_lora(lr, rank=rank)
            yield
            if remove_lora:
                self.lora_manager.remove_all_adapters()

        def maybe_remove_all_loras(self, lora_config):
            if lora_config is not None:
                self.lora_manager.remove_all_adapters()

    def model_info(manager, arguments) -> Tuple[int, int, torch.dtype]:
        vc = arguments["vllm_config"]
        mc = vc.model_config
        return int(mc.get_hidden_size()), int(mc.hf_text_config.num_hidden_layers), mc.dtype

    return SimpleNamespace(
        LRUCache=LRUCache, LoRALayerWeights=LoRALayerWeights, PEFTHelper=PEFTHelper,
        LoRAModel=LoRAModel, LoRAKernelMeta=LoRAKernelMeta, PunicaWrapperBase=PunicaWrapperBase,
        PunicaWrapperGPU=PunicaWrapperGPU, LoRAModelManager=LoRAModelManager,
        LRUCacheLoRAModelManager=LRUCacheLoRAModelManager,
        LRUCacheWorkerLoRAManager=LRUCacheWorkerLoRAManager,
        LoRAModelRunnerMixin=LoRAModelRunnerMixin, model_info=model_info,
        convert_mapping=convert_mapping)


# ---------------------------------------------------------------------------
# the decoder-layer classes (the wheel's shapes)
# ---------------------------------------------------------------------------


def _extract_layer_index(prefix: str) -> int:
    ints = [int(p) for p in prefix.split(".") if p.isdigit()]
    assert len(ints) == 1, f"layer name {prefix} should only contain one integer"
    return ints[0]


class _PairBlock(nn.Module):
    """The pair-form block body every fake decoder layer shares: the fused
    add + norm on entry (fp32 sum rounded to the model dtype as the new
    residual; the norm on the sum), a deterministic 'mlp', ``(out, residual)``."""

    def _body(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            h = self._norm(hidden_states)
        else:
            s = hidden_states.float() + residual.float()
            residual = s.to(hidden_states.dtype)
            h = self._norm(s).to(hidden_states.dtype)
        return self._mlp(h), residual

    @staticmethod
    def _norm(x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)

    @staticmethod
    def _mlp(h):
        return (h * 0.5 + 0.25).to(h.dtype)


def make_fake_layer_classes() -> SimpleNamespace:
    """Fresh copies of the decoder-layer classes, each with the wheel's
    ``__init__`` and ``forward`` signatures."""

    class LlamaDecoderLayer(_PairBlock):
        def __init__(self, vllm_config, prefix: str = "", config=None, attn_layer_type=None):
            super().__init__()
            self.hidden_size = vllm_config.model_config.get_hidden_size()

        def forward(self, positions, hidden_states, residual):
            return self._body(hidden_states, residual)

    class Gemma4DecoderLayer(_PairBlock):
        def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.layer_idx = _extract_layer_index(prefix)

        def forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
            residual = hidden_states
            h = self._norm(hidden_states)
            h = (h + residual).to(hidden_states.dtype)
            h = (self._mlp(self._norm(h)) + h).to(hidden_states.dtype)
            return h, None

    class Qwen3NextDecoderLayer(_PairBlock):
        def __init__(self, vllm_config, layer_type: str, prefix: str = ""):
            super().__init__()
            self.layer_type = layer_type
            self.layer_idx = _extract_layer_index(prefix)

        def forward(self, hidden_states, residual, positions=None, **kwargs):
            return self._body(hidden_states, residual)

    class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
        def __init__(self, vllm_config, layer_type: str, prefix: str = ""):
            super(Qwen3NextDecoderLayer, self).__init__()      # bypasses the parent's __init__
            self.layer_type = layer_type
            self.layer_idx = _extract_layer_index(prefix)

    class NemotronHMLPDecoderLayer(_PairBlock):
        def __init__(self, config, layer_idx: int, model_config=None, cache_config=None,
                     quant_config=None, parallel_config=None, prefix: str = ""):
            super().__init__()
            self.config = config

        def forward(self, hidden_states, residual, **kwargs):
            return self._body(hidden_states, residual)

    class NemotronHMoEDecoderLayer(_PairBlock):
        def __init__(self, config, layer_idx: int, model_config=None, cache_config=None,
                     quant_config=None, parallel_config=None, prefix: str = ""):
            super().__init__()
            self.config = config

        def forward(self, hidden_states, residual, **kwargs):
            return self._body(hidden_states, residual)

    class NemotronHMambaDecoderLayer(_PairBlock):
        def __init__(self, config, layer_idx: int, model_config=None, cache_config=None,
                     quant_config=None, parallel_config=None, prefix: str = ""):
            super().__init__()
            self.config = config

        def forward(self, hidden_states, residual, **kwargs):
            output, residual = self._body(hidden_states, residual)
            return output, residual

    class NemotronHAttentionDecoderLayer(_PairBlock):
        def __init__(self, config, layer_idx: int, model_config=None, cache_config=None,
                     quant_config=None, parallel_config=None, prefix: str = ""):
            super().__init__()

        def forward(self, positions, hidden_states, residual, **kwargs):
            return self._body(hidden_states, residual)

    class NoBoundaryLayer(_PairBlock):
        """A class outside every wheel shape: no ``layer_idx``, no prefix — the
        boundary cannot be derived (the known-bad of §1.3 item 5)."""

        def __init__(self):
            super().__init__()

        def forward(self, positions, hidden_states, residual):
            return self._body(hidden_states, residual)

    table = (LlamaDecoderLayer, Gemma4DecoderLayer, Qwen3NextDecoderLayer, Qwen3_5DecoderLayer,
             NemotronHMLPDecoderLayer, NemotronHMoEDecoderLayer, NemotronHMambaDecoderLayer,
             NemotronHAttentionDecoderLayer)
    return SimpleNamespace(
        LlamaDecoderLayer=LlamaDecoderLayer, Gemma4DecoderLayer=Gemma4DecoderLayer,
        Qwen3NextDecoderLayer=Qwen3NextDecoderLayer, Qwen3_5DecoderLayer=Qwen3_5DecoderLayer,
        NemotronHMLPDecoderLayer=NemotronHMLPDecoderLayer,
        NemotronHMoEDecoderLayer=NemotronHMoEDecoderLayer,
        NemotronHMambaDecoderLayer=NemotronHMambaDecoderLayer,
        NemotronHAttentionDecoderLayer=NemotronHAttentionDecoderLayer,
        NoBoundaryLayer=NoBoundaryLayer, TABLE=table)


class FakeModel(nn.Module):
    """A decoder stack of one fake layer class under that architecture's LOOP
    CONVENTION (positional on llama/gemma4, keyword on qwen/nemotron), plus the
    final fused add so the end-to-end output reflects the pair form."""

    def __init__(self, layer_cls, n_layers: int, vllm_config, *, arch: str,
                 prefix: str = "model.layers"):
        super().__init__()
        self.arch = arch
        hidden = vllm_config.model_config.get_hidden_size()
        cfg = SimpleNamespace(hidden_size=hidden)
        layers = []
        for i in range(n_layers):
            p = f"{prefix}.{i}"
            if arch == "llama":
                layers.append(layer_cls(vllm_config, prefix=p))
            elif arch == "gemma4":
                layers.append(layer_cls(cfg, prefix=p))
            elif arch in ("qwen3_next", "qwen3_5"):
                layers.append(layer_cls(vllm_config, layer_type="full_attention", prefix=p))
            elif arch == "nemotron_h":
                layers.append(layer_cls(config=cfg, layer_idx=i, prefix=p))
            else:
                raise ValueError(arch)
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, positions):
        residual = None
        for layer in self.layers:
            if self.arch in ("llama", "gemma4"):
                hidden_states, residual = layer(positions, hidden_states, residual)
            else:
                hidden_states, residual = layer(positions=positions, hidden_states=hidden_states,
                                                residual=residual)
        if residual is None:
            return hidden_states
        return (hidden_states.float() + residual.float()).to(hidden_states.dtype)


class FakeRunner:
    """The GPU model runner's construction order (fact 11): constructed in
    ``init_device`` with no model; ``load_model`` builds the model and, under a
    lora config, ``load_lora_model`` (the manager and the bank)."""

    def __init__(self, vllm_config, device, mixin_cls, layer_cls, *, arch: str, n_layers: int):
        self.vllm_config, self.device = vllm_config, device
        self._mixin_cls, self._layer_cls, self._arch, self._n_layers = mixin_cls, layer_cls, arch, n_layers
        self.model = None
        self.lora_config = vllm_config.lora_config
        # the mixin's methods, bound to this instance (the wheel MIXES the class in)
        for name in ("load_lora_model", "_set_active_loras", "maybe_setup_dummy_loras",
                     "maybe_remove_all_loras"):
            setattr(self, name, getattr(mixin_cls, name).__get__(self, type(self)))

    def load_model(self):
        self.model = FakeModel(self._layer_cls, self._n_layers, self.vllm_config, arch=self._arch)
        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

    def execute_model(self, *, token_lora_mapping, prompt_lora_mapping, lora_requests,
                      hidden_states, positions):
        if self.lora_config:
            self._set_active_loras(prompt_lora_mapping, token_lora_mapping, set(lora_requests))
        return self.model(hidden_states, positions)


# ---------------------------------------------------------------------------
# adapter directories
# ---------------------------------------------------------------------------


def write_adapter_dir(d: Path, *, layers, d_in: int, d_out: int, rank: int = 1,
                      prefix: str = "model.layers", site: str = "self_attn.o_proj",
                      zero: bool = False, seed: int = 0,
                      residual_block: Optional[Dict[str, Any]] = None,
                      residual_tensors: Optional[Dict[str, torch.Tensor]] = None,
                      extra_config: Optional[Dict[str, Any]] = None) -> Path:
    """A PEFT adapter dir: ``adapter_config.json`` + ``adapter_model.safetensors``
    with a rank-``rank`` o_proj pair per layer (zero when ``zero``), plus an
    optional residual payload (its config block and tensors)."""
    from safetensors.torch import save_file

    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tensors: Dict[str, torch.Tensor] = {}
    for l in layers:
        a = torch.zeros(rank, d_in) if zero else torch.randn(rank, d_in, generator=g)
        b = torch.zeros(d_out, rank) if zero else torch.randn(d_out, rank, generator=g)
        tensors[f"base_model.model.{prefix}.{l}.{site}.lora_A.weight"] = a.contiguous()
        tensors[f"base_model.model.{prefix}.{l}.{site}.lora_B.weight"] = b.contiguous()
    if residual_tensors:
        tensors.update({k: v.contiguous() for k, v in residual_tensors.items()})
    cfg: Dict[str, Any] = {"peft_type": "LORA", "r": rank, "lora_alpha": float(rank),
                           "target_modules": ["o_proj"], "layers_to_transform": list(layers),
                           "bias": "none", "petri_genuine_adapter": True}
    if residual_block is not None:
        cfg["petri_residual_slot"] = residual_block
    if extra_config:
        cfg.update(extra_config)
    (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    save_file({k: tensors[k] for k in sorted(tensors)}, str(d / "adapter_model.safetensors"))
    return d


# ---------------------------------------------------------------------------
# stage 2: the TWO GPU model runner shapes the position class rides on
# ---------------------------------------------------------------------------
#
# - ``NewRequestData`` / ``SchedulerOutput`` (``v1/core/sched/output.py`` @0.25.1): the
#   fields the ``execute_model`` pre-wrapper reads (``scheduled_new_reqs[i].{req_id,
#   prompt_token_ids, lora_request, num_computed_tokens}``, ``num_scheduled_tokens``,
#   ``finished_req_ids``, ``preempted_req_ids``);
# - V1: ``InputBatch`` (``gpu_input_batch.py``: ``req_ids`` with a freed index reused,
#   ``num_reqs``, ``request_lora_mapping``, ``lora_id_to_lora_request``, ``num_prompt_tokens``,
#   ``num_computed_tokens_cpu``, ``make_lora_inputs``, ``condense``) and a runner whose
#   ``execute_model(scheduler_output)`` removes the finished, adds the new, condenses, and calls
#   the MIXIN's ``set_active_loras(input_batch, num_scheduled_tokens)`` (``gpu_model_runner.py:
#   2229``);
# - V2: ``RequestState`` (``gpu/states.py``: ``req_id_to_index``, ``free_indices`` popped from
#   the END, ``prompt_len.np``, ``num_computed_tokens_np``), ``LoraState`` (``gpu/lora_utils.py``
#   verbatim: ``lora_ids``, ``lora_requests``, ``make_lora_inputs(req_ids, idx_mapping,
#   num_scheduled_tokens)``), ``InputBatch`` (``gpu/input_batch.py``: ``req_ids``,
#   ``idx_mapping_np``, ``num_scheduled_tokens``) and a runner whose
#   ``execute_model(scheduler_output, dummy_run=False)`` runs ``finish_requests`` (finished
#   UNION preempted) then ``add_requests`` (remove-then-add) inside the step, builds the batch
#   in a test-chosen ORDER, and calls ``lora_state.make_lora_inputs(...)`` right before
#   ``_set_active_loras`` (``gpu/model_runner.py:1187-1194``).
#
# Both runners share a deterministic HIDDEN SOURCE keyed by (request, absolute position), so a
# row's unpatched reference is reproducible whatever step layout the scheduler produced, and a
# ``FakeScheduler`` that emits the scheduler outputs: one prefill step per request
# (``prompt_len - num_computed_tokens`` rows at absolute positions ``num_computed_tokens ..
# prompt_len - 1``), then one decode row per step, finished after ``max_new`` generated tokens
# (the finished id rides the NEXT step's output, as the engine's does).


@dataclass
class FakeNewRequestData:
    req_id: str
    prompt_token_ids: Optional[List[int]]
    lora_request: Optional[FakeLoRARequest]
    num_computed_tokens: int = 0
    #: V2 only: prompt + the tokens generated before a preemption (the recompute
    #: runs over all of them; ``prefill_len >= prompt_len``, ``gpu/states.py``)
    prefill_token_ids: Optional[List[int]] = None


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: List[FakeNewRequestData]
    num_scheduled_tokens: Dict[str, int]
    finished_req_ids: Set[str] = field(default_factory=set)
    preempted_req_ids: Set[str] = field(default_factory=set)

    @property
    def total_num_scheduled_tokens(self) -> int:
        return int(sum(self.num_scheduled_tokens.values()))


@dataclass
class StepResult:
    """One executed step: the batch's request ids IN BATCH ORDER, each one's
    scheduled token count, the flattened absolute ``positions`` ``[T]``, the
    ``hidden`` ``[T, d]`` the model saw and the ``out`` ``[T, d]`` it returned."""

    req_ids: List[str]
    num_scheduled_tokens: np.ndarray
    positions: torch.Tensor
    hidden: torch.Tensor
    out: torch.Tensor
    #: whatever the runner's ``on_step`` hook records at the step (the tests
    #: snapshot the bank's per-token buffers here: they are LIVE state the
    #: next step overwrites)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return int(self.num_scheduled_tokens.sum())

    def rows_of(self, req_id: str) -> slice:
        start = 0
        for rid, n in zip(self.req_ids, self.num_scheduled_tokens):
            if rid == req_id:
                return slice(start, start + int(n))
            start += int(n)
        raise KeyError(req_id)


class HiddenSource:
    """Deterministic hidden rows per (request id, absolute position): the same
    row whatever step layout delivers it, so the unpatched reference of any
    step is ``ref_model(step.hidden, step.positions)``."""

    def __init__(self, d: int, *, max_len: int = 256, scale: float = 2.0, seed: int = 0,
                 dtype: torch.dtype = torch.bfloat16):
        self.d, self.max_len, self.scale, self.seed, self.dtype = d, max_len, scale, seed, dtype
        self._rows: Dict[str, torch.Tensor] = {}

    def rows(self, req_id: str, positions: Sequence[int]) -> torch.Tensor:
        if req_id not in self._rows:
            g = torch.Generator().manual_seed(self.seed * 100_003 + hash(req_id) % 99_991)
            self._rows[req_id] = (torch.randn(self.max_len, self.d, generator=g) * self.scale
                                  ).to(self.dtype)
        return self._rows[req_id][torch.as_tensor(list(positions), dtype=torch.long)]


class FakeScheduler:
    """The engine-side driver: ``submit`` queues a request; ``step`` emits one
    ``SchedulerOutput`` — the finished ids of the previous step, the waiting
    requests as new (one prefill step each), one decode token for every
    running request — and advances the bookkeeping. ``preempt`` names running
    requests to drop this step (``preempted_req_ids``; they re-enter as NEW
    requests with ``num_computed_tokens = 0`` on the next step, the V2
    engine's shape)."""

    def __init__(self):
        self._waiting: List[Dict[str, Any]] = []
        self._running: Dict[str, Dict[str, Any]] = {}
        self._finish_next: Set[str] = set()
        self.n_steps = 0

    def submit(self, req_id: str, prompt_token_ids: Sequence[int],
               lora_request: Optional[FakeLoRARequest], *, max_new: int = 2,
               num_computed_tokens: int = 0) -> None:
        self._waiting.append({"rid": req_id, "prompt": list(prompt_token_ids), "lora": lora_request,
                              "max_new": int(max_new), "computed": int(num_computed_tokens),
                              "generated": 0})

    @property
    def live(self) -> bool:
        return bool(self._waiting or self._running or self._finish_next)

    def running(self) -> List[str]:
        return list(self._running)

    def step(self, *, preempt: Sequence[str] = ()) -> FakeSchedulerOutput:
        self.n_steps += 1
        finished = set(self._finish_next)
        self._finish_next = set()
        preempted: Set[str] = set()
        for rid in preempt:
            st = self._running.pop(rid)
            preempted.add(rid)
            # the engine restarts a preempted request from scratch: prompt unchanged, the tokens
            # generated so far re-prefilled after it (V2's prefill_token_ids)
            self._waiting.append({"rid": rid, "prompt": st["prompt"], "lora": st["lora"],
                                  "max_new": st["max_new"], "computed": 0,
                                  "generated": st["generated"]})
        new: List[FakeNewRequestData] = []
        nst: Dict[str, int] = {}
        for rid, st in list(self._running.items()):
            nst[rid] = 1                                  # one decode row
        for w in self._waiting:
            rid, prompt, computed = w["rid"], list(w["prompt"]), w["computed"]
            prefill = prompt + [7000 + k for k in range(w["generated"])]
            new.append(FakeNewRequestData(req_id=rid, prompt_token_ids=prompt, lora_request=w["lora"],
                                          num_computed_tokens=computed, prefill_token_ids=prefill))
            nst[rid] = len(prefill) - computed
            self._running[rid] = {"prompt": prompt, "lora": w["lora"], "max_new": w["max_new"],
                                  "generated": w["generated"], "computed": computed}
        self._waiting = []
        for rid, n in nst.items():
            st = self._running[rid]
            st["computed"] += n
            st["generated"] += 1                          # each step samples one token
            if st["generated"] >= st["max_new"]:
                self._finish_next.add(rid)
                del self._running[rid]
        return FakeSchedulerOutput(scheduled_new_reqs=new, num_scheduled_tokens=nst,
                                   finished_req_ids=finished, preempted_req_ids=preempted)


def make_fake_runner_classes(fk: SimpleNamespace) -> SimpleNamespace:
    """Fresh copies of the runner-side classes (``fk`` = ``make_fake_lora_classes()``,
    whose ``LoRAModelRunnerMixin`` the runners bind)."""

    class InputBatchV1:
        """``gpu_input_batch.InputBatch`` @0.25.1 — the fields the V1 expansion reads."""

        def __init__(self, max_num_reqs: int):
            self.max_num_reqs = max_num_reqs
            self._req_ids: List[Optional[str]] = []
            self.req_id_to_index: Dict[str, int] = {}
            self.request_lora_mapping = np.zeros(max_num_reqs, dtype=np.int64)
            self.lora_id_to_lora_request: Dict[int, FakeLoRARequest] = {}
            self.lora_id_to_request_ids: Dict[int, Set[str]] = {}
            self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
            self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)

        @property
        def req_ids(self) -> List[Optional[str]]:
            return self._req_ids

        @property
        def num_reqs(self) -> int:
            return len(self.req_id_to_index)

        def add_request(self, req_id: str, *, prompt_len: int, lora_request, num_computed_tokens: int
                        ) -> int:
            try:
                idx = self._req_ids.index(None)               # a freed index is reused
            except ValueError:
                idx = len(self._req_ids)
                self._req_ids.append(None)
            self._req_ids[idx] = req_id
            self.req_id_to_index[req_id] = idx
            self.num_prompt_tokens[idx] = prompt_len
            self.num_computed_tokens_cpu[idx] = num_computed_tokens
            if lora_request is not None:
                lid = lora_request.lora_int_id
                self.request_lora_mapping[idx] = lid
                self.lora_id_to_request_ids.setdefault(lid, set()).add(req_id)
                self.lora_id_to_lora_request[lid] = lora_request
            else:
                self.request_lora_mapping[idx] = 0
            return idx

        def remove_request(self, req_id: str) -> Optional[int]:
            idx = self.req_id_to_index.pop(req_id, None)
            if idx is None:
                return None
            self._req_ids[idx] = None
            lid = int(self.request_lora_mapping[idx])
            if lid:
                users = self.lora_id_to_request_ids.get(lid, set())
                users.discard(req_id)
                if not users:
                    self.lora_id_to_request_ids.pop(lid, None)
                    self.lora_id_to_lora_request.pop(lid, None)
            self.request_lora_mapping[idx] = 0
            return idx

        def condense(self) -> None:
            """The wheel's condense: the LAST live request moves into each hole."""
            while self._req_ids and self._req_ids[-1] is None:
                self._req_ids.pop()
            hole = 0
            while hole < len(self._req_ids):
                if self._req_ids[hole] is not None:
                    hole += 1
                    continue
                last = len(self._req_ids) - 1
                rid = self._req_ids[last]
                self._req_ids[hole] = rid
                self._req_ids.pop()
                self.req_id_to_index[rid] = hole
                for arr in (self.request_lora_mapping, self.num_prompt_tokens,
                            self.num_computed_tokens_cpu):
                    arr[hole] = arr[last]
                    arr[last] = 0
                while self._req_ids and self._req_ids[-1] is None:
                    self._req_ids.pop()

        def make_lora_inputs(self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray):
            req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
            prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
            token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))
            return prompt_lora_mapping, token_lora_mapping, set(self.lora_id_to_lora_request.values())

    class RequestStates:
        """``gpu/states.RequestState`` @0.25.1 — the fields the V2 expansion reads."""

        def __init__(self, max_num_reqs: int):
            self.max_num_reqs = max_num_reqs
            self.req_id_to_index: Dict[str, int] = {}
            self.index_to_req_id: Dict[int, str] = {}
            self.free_indices = list(range(max_num_reqs))        # popped from the END
            self.prompt_len = SimpleNamespace(np=np.zeros(max_num_reqs, dtype=np.int32))
            self.prefill_len = SimpleNamespace(np=np.zeros(max_num_reqs, dtype=np.int32))
            self.num_computed_tokens_np = np.zeros(max_num_reqs, dtype=np.int32)

        @property
        def num_reqs(self) -> int:
            return len(self.req_id_to_index)

        def add_request(self, req_id: str, prompt_len: int, all_token_ids: Sequence[int],
                        num_computed_tokens: int, max_tokens: int) -> None:
            assert self.free_indices, "No free indices"
            idx = self.free_indices.pop()
            self.req_id_to_index[req_id] = idx
            self.index_to_req_id[idx] = req_id
            self.prompt_len.np[idx] = prompt_len
            self.prefill_len.np[idx] = len(all_token_ids)
            self.num_computed_tokens_np[idx] = num_computed_tokens

        def remove_request(self, req_id: str) -> bool:
            idx = self.req_id_to_index.pop(req_id, None)
            if idx is None:
                return False
            del self.index_to_req_id[idx]
            self.free_indices.append(idx)
            return True

    class LoraState:
        """``gpu/lora_utils.LoraState`` @0.25.1, verbatim."""

        def __init__(self, max_num_reqs: int):
            self.lora_ids = np.zeros(max_num_reqs, dtype=np.int32)
            self.lora_ids.fill(NO_LORA_ID)
            self.lora_requests: Dict[str, FakeLoRARequest] = {}

        def add_request(self, req_id: str, req_index: int, lora_request) -> None:
            if lora_request is not None:
                self.lora_requests[req_id] = lora_request
                self.lora_ids[req_index] = lora_request.lora_int_id
            else:
                self.lora_ids[req_index] = NO_LORA_ID

        def remove_request(self, req_id: str) -> None:
            self.lora_requests.pop(req_id, None)

        def make_lora_inputs(self, req_ids: List[str], idx_mapping: np.ndarray,
                             num_scheduled_tokens: np.ndarray):
            lora_ids = self.lora_ids[idx_mapping]
            prompt_lora_mapping = tuple(lora_ids)
            token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))
            return prompt_lora_mapping, token_lora_mapping, self.get_activate_loras(req_ids)

        def get_activate_loras(self, req_ids: List[str]) -> Set[FakeLoRARequest]:
            out: Set[FakeLoRARequest] = set()
            for rid in req_ids:
                lr = self.lora_requests.get(rid)
                if lr is not None:
                    out.add(lr)
            return out

    @dataclass
    class InputBatchV2:
        req_ids: List[str]
        num_reqs: int
        idx_mapping_np: np.ndarray
        num_scheduled_tokens: np.ndarray
        num_tokens: int

    class _RunnerBase(fk.LoRAModelRunnerMixin):
        """The wheel's shape: ``GPUModelRunner(LoRAModelRunnerMixin)`` — the mixin's
        methods resolve through the CLASS at call time, so a class patch installed
        after the runner was constructed still applies (unlike ``FakeRunner``
        above, which binds them on the instance)."""

        def __init__(self, vllm_config, device, layer_cls, *, arch: str, n_layers: int,
                     hidden: HiddenSource):
            self.vllm_config, self.device = vllm_config, device
            self._layer_cls, self._arch, self._n_layers = layer_cls, arch, n_layers
            self.hidden_source = hidden
            self.model = None
            self.lora_config = vllm_config.lora_config
            self.requests: Dict[str, Dict[str, Any]] = {}
            self.steps: List[StepResult] = []
            #: called with every StepResult right after the forward (a test's snapshot hook)
            self.on_step: Optional[Callable[[StepResult], None]] = None

        def load_model(self):
            self.model = FakeModel(self._layer_cls, self._n_layers, self.vllm_config, arch=self._arch)
            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        def _run(self, order: List[str], nst: np.ndarray, computed: np.ndarray) -> StepResult:
            pos_parts, hid_parts = [], []
            for i, rid in enumerate(order):
                p = list(range(int(computed[i]), int(computed[i]) + int(nst[i])))
                pos_parts.append(torch.as_tensor(p, dtype=torch.long))
                hid_parts.append(self.hidden_source.rows(rid, p))
            positions = torch.cat(pos_parts)
            hidden = torch.cat(hid_parts)
            out = self.model(hidden, positions)
            res = StepResult(req_ids=list(order), num_scheduled_tokens=nst.copy(), positions=positions,
                             hidden=hidden, out=out)
            if self.on_step is not None:
                self.on_step(res)
            self.steps.append(res)
            return res

    class RunnerV1(_RunnerBase):
        """The V1 ``GPUModelRunner`` shape (``gpu_model_runner.py`` @0.25.1)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_batch = InputBatchV1(self.vllm_config.scheduler_config.max_num_seqs)

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            so = scheduler_output
            for rid in so.finished_req_ids:                              # _update_states
                self.requests.pop(rid, None)
                self.input_batch.remove_request(rid)
            for nrq in so.scheduled_new_reqs:
                ids = list(nrq.prompt_token_ids or [])
                if nrq.req_id in self.input_batch.req_id_to_index:       # aborted + resubmitted id
                    self.input_batch.remove_request(nrq.req_id)
                self.requests[nrq.req_id] = {"prompt": ids, "lora": nrq.lora_request}
                self.input_batch.add_request(nrq.req_id, prompt_len=len(ids), lora_request=nrq.lora_request,
                                             num_computed_tokens=nrq.num_computed_tokens)
            self.input_batch.condense()
            if so.total_num_scheduled_tokens == 0:
                return None
            n = self.input_batch.num_reqs
            order = [str(r) for r in self.input_batch.req_ids[:n]]
            nst = np.array([so.num_scheduled_tokens[r] for r in order], dtype=np.int32)
            if self.lora_config:                                          # _prepare_inputs
                self.set_active_loras(self.input_batch, nst)
            computed = self.input_batch.num_computed_tokens_cpu[:n].copy()
            res = self._run(order, nst, computed)
            self.input_batch.num_computed_tokens_cpu[:n] += nst
            return res

    class RunnerV2(_RunnerBase):
        """The V2 ``GPUModelRunner`` shape (``gpu/model_runner.py`` @0.25.1).
        ``batch_order`` picks the batch order of this step's scheduled request
        ids (default: the scheduler's order) so a test can reorder freely."""

        def __init__(self, *args, batch_order: Optional[Callable[[List[str]], List[str]]] = None,
                     **kwargs):
            super().__init__(*args, **kwargs)
            m = self.vllm_config.scheduler_config.max_num_seqs
            self.req_states = RequestStates(m)
            self.lora_state = LoraState(m)
            self.batch_order = batch_order or (lambda ids: list(ids))

        def _remove_request(self, req_id: str) -> None:
            if self.req_states.remove_request(req_id):
                self.lora_state.remove_request(req_id)
                self.requests.pop(req_id, None)

        def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
            so = scheduler_output
            if dummy_run:
                return None
            for rid in set(so.finished_req_ids) | set(so.preempted_req_ids):   # finish_requests
                self._remove_request(rid)
            for nrq in so.scheduled_new_reqs:                                 # add_requests
                self._remove_request(nrq.req_id)
                ids = list(nrq.prompt_token_ids or [])
                all_ids = list(nrq.prefill_token_ids) if nrq.prefill_token_ids is not None else ids
                self.req_states.add_request(req_id=nrq.req_id, prompt_len=len(ids), all_token_ids=all_ids,
                                            num_computed_tokens=nrq.num_computed_tokens, max_tokens=8)
                idx = self.req_states.req_id_to_index[nrq.req_id]
                self.lora_state.add_request(nrq.req_id, idx, nrq.lora_request)
                self.requests[nrq.req_id] = {"prompt": ids, "lora": nrq.lora_request}
            if so.total_num_scheduled_tokens == 0:
                return None
            order = self.batch_order(list(so.num_scheduled_tokens.keys()))
            idx_mapping_np = np.array([self.req_states.req_id_to_index[r] for r in order], dtype=np.int32)
            nst = np.array([so.num_scheduled_tokens[r] for r in order], dtype=np.int32)
            input_batch = InputBatchV2(req_ids=list(order), num_reqs=len(order), idx_mapping_np=idx_mapping_np,
                                       num_scheduled_tokens=nst, num_tokens=int(nst.sum()))
            if self.lora_config:
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids, input_batch.idx_mapping_np, input_batch.num_scheduled_tokens)
                self._set_active_loras(*lora_inputs)
            computed = self.req_states.num_computed_tokens_np[idx_mapping_np].copy()
            res = self._run(order, nst, computed)
            self.req_states.num_computed_tokens_np[idx_mapping_np] += nst
            return res

    return SimpleNamespace(InputBatchV1=InputBatchV1, RequestStates=RequestStates, LoraState=LoraState,
                           InputBatchV2=InputBatchV2, RunnerV1=RunnerV1, RunnerV2=RunnerV2)


# ---------------------------------------------------------------------------
# payload builders shared by the stage-1 and stage-2 tests
# ---------------------------------------------------------------------------


def residual_block(kind: str, *, d: int, layers, n_layers_model: int, boundaries=None, mode: str = "all",
                   header_ids: Sequence[int] = (), skip_bos: bool = False, coeff: float = 1.0,
                   anchor: int = 0) -> Dict[str, Any]:
    """A ``petri_residual_slot`` block; ``response_header_len`` follows the ids
    for the response mode (the exporter's rule)."""
    from sisteer.capeval.serving import native_residual as nr

    gated = kind in nr.GATED_KINDS
    ids = [int(t) for t in header_ids]
    return {"schema": nr.RESIDUAL_SLOT_SCHEMA, "kind": kind,
            "op": nr.OP_RELU_GATED_VECTOR if gated else nr.OP_ADD_VECTOR,
            "layers": list(layers), "boundaries": list(boundaries if boundaries is not None else layers),
            "position_mode": mode, "response_header_len": (len(ids) if mode == "response" else None),
            "response_header_ids": ids, "skip_bos": skip_bos, "coeff": coeff, "hidden_size": d,
            "n_layers_model": n_layers_model, "anchor_layer": anchor,
            "dtype_policy": nr.DTYPE_POLICY_GATED if gated else nr.DTYPE_POLICY_CONSTANT}


def psr_tensors(d: int, layers, *, seed: int, dir_scale: float = 2.0, bias: float = 0.5
                ) -> Dict[str, torch.Tensor]:
    """Random gates with a POSITIVE bias (roughly every gate open) and
    directions of norm ``dir_scale``."""
    from sisteer.capeval.serving import native_residual as nr

    g = torch.Generator().manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    for l in layers:
        w = torch.randn(d, generator=g) / math.sqrt(d)
        dvec = torch.randn(d, generator=g)
        dvec = dvec / dvec.norm() * dir_scale
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_w"] = w
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_b"] = torch.full((1,), bias)
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.direction"] = dvec
    return out


def const_tensors(d: int, layers, *, seed: int, scale: float = 0.7) -> Dict[str, torch.Tensor]:
    from sisteer.capeval.serving import native_residual as nr

    g = torch.Generator().manual_seed(seed)
    return {f"{nr.PAYLOAD_PREFIX}layers.{l}.vector": torch.randn(d, generator=g) * scale
            for l in layers}


def make_payload(kind: str, *, d: int, layers, n_layers_model: int, seed: int, mode: str = "all",
                 header_ids: Sequence[int] = (), skip_bos: bool = False, coeff: float = 1.0,
                 bias: float = 0.5, **kw):
    from sisteer.capeval.serving import native_residual as nr

    block = residual_block(kind, d=d, layers=layers, n_layers_model=n_layers_model, mode=mode,
                           header_ids=header_ids, skip_bos=skip_bos, coeff=coeff, **kw)
    tensors = (psr_tensors(d, layers, seed=seed, bias=bias) if kind in nr.GATED_KINDS
               else const_tensors(d, layers, seed=seed))
    return nr.payload_from_block_and_tensors(block, tensors)


def write_residual_adapter(root: Path, name: str, payload, *, d_in: int = 16, d_out: int = 16,
                           anchor: int = 0, prefix: str = "model.layers") -> Path:
    """A bank entry: the rank-1 ZERO o_proj pair on the anchor plus the payload."""
    return write_adapter_dir(Path(root) / name, layers=[anchor], d_in=d_in, d_out=d_out, rank=1,
                             prefix=prefix, zero=True, residual_block=payload.block(),
                             residual_tensors=payload.tensors())
