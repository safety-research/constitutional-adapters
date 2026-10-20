# Constitutional adapters

Code, data and fitted objects for *Constitutional adapters: inference-time interventions for misalignment and misuse*
(Adam S. Lowet and Mark Kurzeja, [arXiv:2609.36657](https://arxiv.org/abs/2609.36657)). This repo contains: (1) the two training corpora (constitution and control) and the code that generated them; (2) all lightweight objects the paper evaluates; (3) the fitting and evaluation harness, (4) a snapshot of the results registry sufficient to rebuild the paper figures with CPU only; and (5) the pipeline scripts for reproducing every stage of the paper. Notice that step (5) requires hundreds of GPU-hours and is built to run only on a Slurm cluster.

The repo is published as a single commit, and a clone is about 0.9 GB to download and 2.1 GB on disk, of which the
checked-out tree is 1.2 GB, almost all of it data: `objects/` (440 MB), `paper/snapshot/` (330 MB) and `corpora/`
(243 MB). Nothing is in Git LFS. Two lighter checkouts are below: the figures alone (330 MB) and a demo of the best
Qwen3.5-4B objects (10 MB).

## What a constitutional adapter is

A constitutional adapter is a constitution-trained intervention object minus a control-trained one. Both are fit to a
frozen language model by next-token cross-entropy on synthetic documents. No harmful request or jailbreak appears in
training. The constitution-trained object's corpus is generated from Claude's constitution, the control-trained
object's from a register-matched placebo document. Each corpus is composed of two
parts: difficult advice (advice sought in a hard, ethically ambiguous situation) and stories. Each object is fit on
the whole corpus, both parts together.

An object is either a LoRA adapter (here, a rank-4 adapter on the attention output projection `o_proj` matrices of every attention block), or
a ReLU steer: a steering vector per decoder block, gated by a learned ReLU function of the activations and added to the residual stream at every token.

The LoRA therefore touches every block on Llama-3.1-8B and Gemma-4-E4B (32 and 42, attention throughout) but only 8 of
Qwen3.5-4B's 32 (the rest linear attention) and 4 of Nemotron-3-Nano-4B's 42 (the rest Mamba-2 and MLP); the ReLU
steer has a vector in every block on all four.

At serving time the LoRA difference is one adapter of twice the rank (`A` stacks both parents' `A`, `B` stacks `B_pos`
and `-B_neg`). The ReLU-steer difference subtracts the two writes, both gates reading the same residual pre-intervention state. A
coefficient scales the difference, 1.0 being the fitted object, and objects fit on a base checkpoint are served on
its post-trained checkpoint (base-to-instruct transfer).

The paper's baselines are two prompted system instructions and a refusal-direction steer ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)), on jailbreak, misalignment, over-refusal, and capability evaluations. Qwen3.5-4B carries the main figures, the other three models appear in the appendix.

## Where to start

| I want | Go to |
|---|---|
| the figures and tables, on a CPU | [Regenerate the figures](#regenerate-the-figures), then [paper/notebooks/README.md](paper/notebooks/README.md) |
| a fitted adapter on its model | [Use a fitted adapter](#use-a-fitted-adapter), then [objects/README.md](objects/README.md) |
| the method on another model, on my Slurm cluster | [Run the pipeline](#run-the-pipeline), then [reproduce/README.md](reproduce/README.md) |
| the training corpora and the prompted baselines | [corpus_generator/README.md](corpus_generator/README.md), [system_instructions/README.md](system_instructions/README.md) |
| the database behind the figures | [docs/registry/README.md](docs/registry/README.md) |

## Download the complete repo, including data and fitted objects

```bash
git clone https://github.com/safety-research/constitutional-adapters.git
cd constitutional-adapters
```
This includes both `objects/` and `corpora/` (550 MB to download, 1.2 GB on disk), which is only needed if viewing the full corpora, re-fitting objects, or re-evaluating all previously fitted objects. Sparser, more lightweight checkout instructions are provided below for simply regenerating the figures or installing, interacting with, and evaluating a single fitted adapter.

## Regenerate the figures

No GPU, cluster, model download or API key is needed; only `git` and [uv](https://docs.astral.sh/uv/).  `uv run --no-project` runs the path helper before `uv sync` builds the
environment.

```bash
git clone --filter=blob:none --sparse https://github.com/safety-research/constitutional-adapters.git
cd constitutional-adapters
git sparse-checkout add reproduce/tools
git sparse-checkout add $(uv run --no-project python reproduce/tools/sparse_paths.py figures)
```

Then, with either this sparse repo or the full repo:
```bash
uv sync --frozen --extra cpu --group notebook
reproduce/10_figures.sh --source snapshot
```

The stage executes `paper/notebooks/registry_qwen_figures.ipynb` against the registry snapshot under `paper/snapshot/` and produces Figures 3 to 14 (files `fig3` to `fig7` and the appendix's `figA_*`) and Tables 4 to 10 (LaTeX printed under the Figure 3 to 6 cells of the rendered HTML) in about two minutes; Figures 1 and 2 (schematics) and Tables 1 to 3 (hyperparameters) are not produced programatically.

Figures land as PNG, PDF and SVG in `paper/notebooks/figures/` and, with the rendered HTML, in
`$SISTEER_FIG_DIR/registry-notebook/` (`SISTEER_FIG_DIR` defaults to `$HOME/figs`; set it in the environment or the
site config). The first run reassembles the snapshot's SQLite under `paper/notebooks/_cache/` (about 2 GB). A run
rebuilds the notebook file without its committed outputs, so Git then shows it modified; `figures/` and `_cache/` are
ignored. [paper/notebooks/README.md](paper/notebooks/README.md) has the nbconvert route and the figure map.

## Use a fitted adapter

`objects/` holds 152 of the 154 lightweight objects the paper evaluates; `objects/WITHHELD.json` names the other two
and why. The 149 that are files live at `objects/<model_id>/<object_id>/object.pt`, `object_id` being the file's
sha256; the unsteered model and the two prompted system instructions are three remaining "objects." An object's address
is its `model_id` and `paper_label` plus, for a dose-scaled copy or a training-seed replicate, its dose and seed.
`sisteer.objects.resolve` turns the address into the object id (`python -m sisteer.objects find --model qwen35_4b_it`
lists a model's objects). `load_object` rebuilds an object around a loaded Hugging Face model, and its `apply` context
manager installs the hooks at a chosen coefficient.

### Lightweight demo
The code below demonstrates all Qwen3.5-4B
objects at a single seed (`objects/DEMO.json`), first for the difference LoRA alone, and then for all six fitted objects plus the refusal-direction baseline and the two system instructions. The
six fitted objects are fit on Qwen3.5-4B-Base and served on Qwen3.5-4B, as the figures draw them. `uv sync --frozen
--extra cu128` then installs PyTorch and the CUDA libraries, about 4 GB (`--extra cpu` instead on an Apple Silicon Mac,
where the objects run on the Mac's GPU through MPS, or on a CPU, where they run slowly).

```bash
git clone --filter=blob:none --sparse https://github.com/safety-research/constitutional-adapters.git
cd constitutional-adapters
git sparse-checkout add reproduce/tools
git sparse-checkout add $(uv run --no-project python reproduce/tools/sparse_paths.py demo)
```

When using the adapters, we recommend a GPU environment:
```bash
uv sync --frozen --extra cu128
```
Or, without a GPU (much slower and not recommended)
```bash
uv sync --frozen --extra cpu
```
Then
```bash
source .venv/bin/activate
python
```

```python
# Downloads Qwen/Qwen3.5-4B on first use.
from pathlib import Path
from sisteer.extraction.committed_objects import committed_object_bytes_path
from sisteer.methods import load_object
from sisteer.models import load_model_and_tokenizer, build_chat_prompt
from sisteer.generate import generate_with_object
from sisteer.objects import resolve, serve_checkpoint

# The LoRA constitutional adapter fit on Qwen/Qwen3.5-4B-Base, by its address (model_id, paper_label); dose= and seed=
# pick a dose-scaled copy or a training-seed replicate, here the full-dose fit of seed 3.
OBJECT_ID = resolve("qwen35_4b_base", "difference_lora", seed=3, dose=1.0)
path, _ = committed_object_bytes_path(Path("objects/qwen35_4b_base") / OBJECT_ID)  # sha256(bytes) == object_id
# The checkpoint the paper serves qwen35_4b_base objects on, from reproduce/models.yaml: the pinned Qwen/Qwen3.5-4B, since
# a base-fit adapter is evaluated on the post-trained checkpoint (base-to-instruct transfer);
# "qwen35_4b_base" just names the model on which it was fit.
hf_repo, revision = serve_checkpoint("qwen35_4b_base")

model, tok = load_model_and_tokenizer(hf_repo, revision=revision, dtype="bfloat16")
obj = load_object(path, model=model, tok=tok)
prompt = build_chat_prompt(tok, "Give me one sentence of advice about a difficult decision.")
print(generate_with_object(model, tok, [prompt], obj=obj, scale=1.0, max_new_tokens=200)[0])
```

Looping over all seed-3 objects:

```python
# Every demo object on the same served checkpoint, each at the coefficient the figures use (DEMO.json's "dose":
# 1.0, the refusal steer at c = 0.3), then the unsteered model and the two prompted baselines, which send a
# system instruction as the system message.
import json
question = "Give me one sentence of advice about a difficult decision."
prompt = build_chat_prompt(tok, question)
for o in json.loads(Path("objects/DEMO.json").read_text())["objects"]:
    path, _ = committed_object_bytes_path(Path("objects") / o["dir"])   # checks sha256(bytes) == object_id
    obj = load_object(path, model=model, tok=tok)
    print(o["label"], "|", generate_with_object(model, tok, [prompt], obj=obj, scale=o["dose"], max_new_tokens=200)[0])
print("Unsteered |", generate_with_object(model, tok, [prompt], max_new_tokens=200)[0])
for name in ("safety", "constitution"):
    system = Path(f"system_instructions/{name}.md").read_text()
    print(f"{name.capitalize()} SI |", generate_with_object(model, tok, [build_chat_prompt(tok, question, system=system)], max_new_tokens=200)[0])
```

[objects/README.md](objects/README.md) covers the sidecars, the ReLU-steer adapter, the vLLM serving path behind the
paper's numbers, and the coefficient's unit in each object family.

## Run the pipeline

`reproduce/` runs the paper's pipeline on one model from one site config, in eleven stages: corpus preparation, the
hyperparameter search, fitting, the evaluation wave, grading, the dose ladder and the figures. It needs at least one Slurm cluster reachable by ssh, Modal for stage 07, a GPU-less grading host for the registry, the judges and the
stages, model downloads, API credentials and days of wall-clock time.

```bash
# From the repository root.
cp -r reproduce/site.example ~/.config/sisteer    # then edit config.env and clusters/<label>.env
reproduce/00_setup.sh --what all --model qwen35_4b_demo
```

Stage 01 demonstrates corpus generation, which can be skipped if using the shipped corpus shards. Stages 02 to 10 follow. Every stage answers `--help` and `--dry-run`; `python reproduce/lib/models.py keys` lists the keys. See [reproduce/README.md](reproduce/README.md) for the complete pipeline. The worked example, `qwen35_4b_demo`, fits and serves the paper's pinned Qwen3.5-4B checkpoints under its own key, so
nothing it writes can be mistaken for a paper result. Its row declares one evaluation wave, `s1`, comprising every arm beside its
own unsteered legs, as well as the two dose views stage 09 reads.

## Repository layout

| path | what |
|---|---|
| `reproduce/` | the eleven stage scripts, `lib/`, `models.yaml`, `site.example/`, `overrides/`, `tools/` |
| `src/sisteer/` | the harness: fitting and hooks (`extraction/`, `methods.py`, `hooks.py`); `eval/`, the in-process jailbreak battery, its attacks, the defense identities and controls, plus some capability and training-budget code; `capeval/`, the served-model harness: target serving, the capability suite, Petri and the misalignment evals, the registry contracts (both names are historical misnomers, kept because recorded run identities reference the paths); the registry (`registry/`) |
| `scripts/` | the workers the stages call, and side-study scripts |
| `configs/` | registry seeds, object pins, generated evaluation configs (`configs/README.md` maps families to generators) |
| `corpora/` | the two corpora, each in two parts (advice and stories), as gzip shards with sha256 manifests |
| `corpus_generator/` | the corpus generator, a uv workspace member (`uv sync --frozen --extra cpu --group corpus`; console script `corpusgen`) |
| `system_instructions/` | the two prompted baselines as plain text |
| `objects/` | 152 of the paper's 154 objects (149 as files), with `MANIFEST.json`; `WITHHELD.json` names the two it does not ship |
| `paper/` | the figure notebook and its library (`notebooks/`), the registry snapshot (`snapshot/`), tests |
| `docs/` | [docs/index.md](docs/index.md): the configuration, secrets and registry pages, and design notes on individual evaluations |
| `data/` | held-out split ids, probe sets, the capability battery's small artefacts, the Petri scenario bundle, the ReNeLLM evaluation corpus (under its own terms) and the manifest of the refusal direction's fit prompts |
| `tests/` | the test suite and its fixtures |
| `systemd/` | templates of the registry's systemd user units, rendered for the site and installed by `00_setup.sh --what registry` |
| `LICENSES/` | the full texts of the third-party licences listed in `THIRD_PARTY_NOTICES` |

## Additional setup

Nemotron-3-Nano-4B alone needs the fused Mamba kernels: add `--extra mamba`, whose first sync compiles them from
their pinned PyPI sources (it needs a CUDA 12 toolkit, with `CUDA_HOME` pointing at it). The long-context runs need
flash-attn, built once per cluster into the gitignored `wheels/` directory by `scripts/sbatch/fa_build.sbatch`; then:

```bash
uv sync --frozen --extra cu128 --extra longctx
```

`--frozen` matters because `pyproject.toml` points the `longctx` extra at that local wheel: a plain `uv sync`
re-resolves the lock, reads its metadata and fails in a clone without it. The `cpu` extra cannot be combined with
`cu128`, `mamba` or `longctx`. The stage scripts' Slurm jobs sync their own environments on the cluster, so these
commands are for work outside them.

## Responsible use and content

This repository exists to evaluate defenses against jailbreaks and misalignment. Its attack code, a battery of
published jailbreak methods and the agentic-misalignment scenarios, is there to measure how well an intervention holds
up under attack. It distributes no model responses to harmful requests.

Where possible, the harmful prompt sets are loaded at run time from their pinned public sources and never written to the
repository, the tests use generic placeholders, and the registry snapshot holds scores and labels rather than text. In such cases, the repository records the upstream commit, seed and hashes so that it reproduces exactly.

The two training corpora were generated by Claude. Where they discuss hazards such as overdoses or weapons, or a duty to
report suspected child abuse, they do so to refuse, to reduce harm or to explain that duty.

## Third-party material

Some files come from other projects and keep their own licences: the agentic-misalignment scenario templates and
classifier prompts, the Petri auditor and judge prompts and the emergent-misalignment evaluation questions (all MIT),
and the eight chain-of-thought exemplars of Wei et al. (2022) that the GSM8K evaluation uses (CC BY 4.0). The Petri
scenarios were generated with petri-bloom (MIT), and the ReNeLLM corpus has its own terms (below).

[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) lists each with its source and pinned revision, and `LICENSES/` holds the
full texts.

The adapters in `objects/` are derived from Llama-3.1-8B (Llama 3.1 Community License), NVIDIA Nemotron-3-Nano-4B (NVIDIA
Nemotron Open Model License), and Qwen3.5-4B and Gemma 4 E4B (both Apache 2.0). Built with Llama. Claude's constitution is released under CC0 1.0.

## License and citation

MIT; see [LICENSE](LICENSE). The MIT licence covers the code, the data and the fitted adapters, except the files
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) lists under their own terms. The third-party files above keep their upstream
licences, and the ReNeLLM corpus is under [LICENSES/ReNeLLM-corpus-terms.txt](LICENSES/ReNeLLM-corpus-terms.txt), which
forbids using it to train any AI system. Using the adapters is also subject to the base-model licences listed there.

```bibtex
@misc{constitutional_adapters,
  title         = {Constitutional adapters: inference-time interventions for misalignment and misuse},
  author        = {Lowet, Adam S. and Kurzeja, Mark},
  year          = {2026},
  eprint        = {2609.36657},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.36657}
}
```
