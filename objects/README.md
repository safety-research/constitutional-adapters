# Fitted objects

152 of the 154 intervention objects the paper evaluates, as the bytes that produced its numbers. Needed to apply a
constitutional adapter, inspect how an object was fit, or resolve an object id from the results registry; regenerating
the figures does not read it. The other two are withheld (see "Withheld objects" below).

## Layout

149 are artifacts (332 MB) at `objects/<model_id>/<object_id>/object.pt`, `object_id` being the sha256 of `object.pt`
and the registry's id; 3 are virtual, with no directory: the unsteered model (`none`) and the two prompted system
instructions (`si:<sha256 of the text>`, from [`../system_instructions/`](../system_instructions/README.md)).
`MANIFEST.json` lists all 152 with `object_id`, `model_id`, `paper_label`, `object_kind`, `derivation` and `dir`, so a
registry id resolves to its directory and label there. The other direction is the object's address, `model_id` and
`paper_label` plus the dose and seed of a dose-scaled copy or a seed replicate: `sisteer.objects.resolve(model_id,
paper_label, dose=, seed=)` returns the object id, and `python -m sisteer.objects find --model <model_id>` lists a
model's objects with their dose and seed. `DEMO.json` names, for each of the seven Qwen3.5-4B interventions, the object
the paper's multi-turn figure rates best among its training seeds (the set the repository README's demo checkout
fetches), with the per-seed multi-turn gains it was chosen from; `reproduce/tools/demo_objects.py` writes and checks it.
`DEMO.json` is also the import set of the worked example `qwen35_4b_demo`: `reproduce/05_stage_roster.sh` stages these
objects, and the paper's 0.75 dose copies of the two differences, into the example's bank in place of stage 04's fits,
each held to its registry id, and `reproduce/tools/import_objects.py registry` states them in a registry that lacks them
([reproduce/README.md](../reproduce/README.md)). In a sparse checkout the listing shows `?` for the dose and seed of an object whose directory is absent, and a
difference's seed is read from its parents' sidecars, so with only `DEMO.json`'s directories present the two
difference adapters resolve without `seed=`; a refusal names the directories to `git sparse-checkout add`.

| `model_id` | checkpoint | objects | of which dose-scaled copies |
|---|---|---|---|
| `qwen35_4b_base` | Qwen/Qwen3.5-4B-Base | 68 | 45 |
| `qwen35_4b_it` | Qwen/Qwen3.5-4B | 12 | 4 |
| `gemma4e4b_base` | google/gemma-4-E4B | 21 | 15 |
| `gemma4e4b_it` | google/gemma-4-E4B-it | 5 | 1 |
| `llama8b_base` | NousResearch/Meta-Llama-3.1-8B | 21 | 15 |
| `llama8b_it` | NousResearch/Meta-Llama-3.1-8B-Instruct | 5 | 1 |
| `nemotron3nano4b` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 17 | 11 |

A dose-scaled copy (`derivation: dose_scale`) is a parent with its write tensors multiplied by a coefficient (negative
reverses the write), saved under its own id: a rung of the dose ladders vLLM served. The rest are fits or derivations
at coefficient 1.0.

`_base` rows hold objects fit on the base checkpoint and served on the post-trained one (base-to-instruct transfer);
`_it` rows, objects fit and served on the post-trained checkpoint. Nemotron has one chat checkpoint.

On `qwen35_4b_base` each trained label has one fit at effective batch 8 and three training-seed replicates at
effective batch 64, which the Qwen figures average. Each difference subtracts the control fit from the constitution fit
of the same effective batch and seed; ReLU differences exist for the three batch-64 seeds only. Elsewhere a label has
one fit, or two differing in effective batch (8 and 64).

## Withheld objects

`WITHHELD.json` lists the two paper objects this repository does not ship, the `qwen35_4b_it` `safety_lora` and
`safety_relu` fits: they were trained on prompts that include third-party model (xAI Grok) output. Each row keeps the
object's id, address, kind and size with that reason; it has no directory here and no row in `MANIFEST.json`. The
records that name the two ids (configs, pins, results) are unchanged, and `sisteer.objects`, the committed-object loader
and the tools that read `objects/` refuse either by name with the reason.

## Licences and names

Each object is derived from its base model, so using it is subject to that model's licence as well as this
repository's MIT licence: the Llama 3.1 Community License and its Acceptable Use Policy for `llama8b_base` and
`llama8b_it`, the NVIDIA Nemotron Open Model License for `nemotron3nano4b`, and Apache 2.0 for the Qwen and Gemma rows.
`llama8b_base/`, `llama8b_it/` and `nemotron3nano4b/` hold a `NOTICE` and a copy of the licence beside the objects, so a
sparse checkout of any one object brings them. [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES) lists the sources.

The objects fit on Llama 3.1 are named "Llama-3.1-8B" followed by their paper label, as the Llama 3.1 licence asks of
models fine-tuned from it: "Llama-3.1-8B difference_lora" is the LoRA constitutional adapter on Llama. `llama8b_base` and
`llama8b_it` are identifiers, not model names. Built with Llama.

## Labels

| `paper_label` | `object_kind` | definition |
|---|---|---|
| `control_lora`, `constitution_lora` | `lora` | rank-4 LoRA on `o_proj` of every attention block (all on Llama and Gemma, 8 of 32 on Qwen3.5-4B, 4 of 42 on Nemotron), fit on the control and the constitution corpus |
| `difference_lora` | `lora_subtraction` | constitution LoRA minus control LoRA: the LoRA constitutional adapter |
| `control_relu`, `constitution_relu` | `relu_steer` | one direction per decoder block, gated by a learned ReLU probe, fit on the control and the constitution corpus |
| `difference_relu` | `composed` | constitution minus control ReLU steer, both gates reading the same residual state: the ReLU-steer constitutional adapter |
| `arditi_allpos` | `vector_all` | the refusal-direction baseline: difference-in-means direction added at every position |
| `arditi` | `static_r_arditi` | prefill-only refusal-direction variant, not drawn in the figures |
| `safety_lora`, `safety_relu` | `lora`, `relu_steer` | distilled from the safety system instruction; not constitutional adapters, absent from the main figures; withheld, not shipped (`WITHHELD.json`) |
| `none`, `safety_si`, `constitution_si` | virtual | the unsteered model and the two prompted baselines |

## Sidecars

"Fits" below are the 38 control and constitution objects.

| file | on | contents |
|---|---|---|
| `train_budget_meta.json` | fits and their 45 dose-scaled copies | corpus cell, arm, effective batch, learning rate, sequence length, target layers, model revision |
| `metrics.jsonl` | fits | training curve |
| `COMPLETE` | fits | JSON with the sha256 of `object.pt`, `train_budget_meta.json` and `metrics.jsonl` |
| `derived_meta.json` | differences and their copies | both parents' ids and the composition rule |
| `fit_provenance.json` | refusal directions | fit provenance |

A dose-scaled copy keeps its parent's sidecar plus a `dose_scaled_from` block (parent id, scale); the two
refusal-direction copies name their parent and rung in `fit_provenance.json` instead.

## Loading

`sisteer.extraction.committed_objects.committed_object_bytes_path(dir)` returns the path of `object.pt` once its
sha256 equals the object id (no sidecar is checked); `sisteer.methods.load_object(path, model=, tok=)` rebuilds the
object around a loaded Hugging Face model, and its `apply(model, scale=alpha)` context manager installs the hooks.

The [repository README](../README.md#use-a-fitted-adapter) applies the LoRA constitutional adapter with
`sisteer.generate.generate_with_object`. Below, the ReLU-steer constitutional adapter (fit on Qwen/Qwen3.5-4B-Base,
served on Qwen/Qwen3.5-4B) around a plain forward pass:

```python
# Repository root, `uv sync --frozen --extra cu128` environment; downloads Qwen/Qwen3.5-4B on first use.
import torch
from pathlib import Path
from sisteer.extraction.committed_objects import committed_object_bytes_path
from sisteer.methods import load_object
from sisteer.models import load_model_and_tokenizer, build_chat_prompt
from sisteer.objects import resolve, serve_checkpoint

# The ReLU-steer constitutional adapter fit on Qwen/Qwen3.5-4B-Base, by its address (model_id, paper_label): its three
# fits on qwen35_4b_base are the effective-batch-64 training-seed replicates, so seed= picks one, here seed 3. Resolves
# to the manifest's object_id 0718a3f9edfb033ad84c3322f717dc62b32b6298fe48f0a6c42e9f0a7c031647.
OBJECT_ID = resolve("qwen35_4b_base", "difference_relu", seed=3)
path, _ = committed_object_bytes_path(Path("objects/qwen35_4b_base") / OBJECT_ID)  # sha256(bytes) == object_id
# The checkpoint the paper serves qwen35_4b_base objects on, from reproduce/models.yaml: the pinned Qwen/Qwen3.5-4B,
# since a base-fit object is evaluated on the post-trained checkpoint (base-to-instruct transfer).
hf_repo, revision = serve_checkpoint("qwen35_4b_base")
model, tok = load_model_and_tokenizer(hf_repo, revision=revision)
relu = load_object(path, model=model, tok=tok)         # -> PSRPairedWriteObject
prompt = build_chat_prompt(tok, "Give me one sentence of advice about a difficult decision.")
enc = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad(), relu.apply(model, scale=0.75):   # 1.0 is the fitted object, 0.0 unsteered
    logits = model(**enc).logits
```

## Serving and dose units

The paper's numbers come from vLLM serving each object natively: a LoRA as a PEFT adapter, a LoRA difference as one
adapter of twice the rank (`A` stacks both parents' `A`, `B` stacks `B_pos` and `−B_neg`), the ReLU steers and the
refusal direction as a residual-stream write at the block outputs they target. Native serving needs Linux and a CUDA
GPU; the hooks in the examples above run in-process wherever PyTorch runs the model, including an Apple Silicon Mac's GPU (MPS) and a CPU.

The coefficient multiplies the write linearly: LoRA `B` (both parents' `B` for a difference; `A` unchanged), the ReLU
direction vectors (gates unchanged), or the refusal direction's raw coefficient `c`. The unit differs per family, so the
same number is a different dose in each, and the figures never put two families on one axis. 1.0 is the fitted object
as served and 0.75 the paper's companion dose; the refusal direction is read at `c = 0.3`. vLLM served each dose as its
own dose-scaled copy.
