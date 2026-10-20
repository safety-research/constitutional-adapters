# Modal apps: the BigCodeBench and BFCL harness

The evaluation workers that run on Modal (serverless H200 containers) rather than Slurm. The one a reader of the paper may want is `tinycap_served.py`, the harness behind the BigCodeBench and BFCL rows (the capability panels of Fig. 6 and Fig. 14); `paper/notebooks/` draws the figures from the committed snapshot without it or a GPU. `reproduce/07_bcb_bfcl.sh` wraps the harness but does not run for the paper models in this tree (it runs the worked example `qwen35_4b_demo`; the paper's rows go through `scripts/bcb_bfcl_200_widen_launch.py`), so call the harness directly for a row outside those.

## What the harness needs

A Modal account and the `modal` CLI, with `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment ([`docs/secrets.md`](../../docs/secrets.md)); `-e main` names the Modal environment, so substitute yours. Every `tinycap_served.py` launch needs `TINYCAP_APP_NAME` in its environment, equal to `--app-name`.

The volume `twosi-fits` must hold the served checkpoint and the model's bank: its fitted objects (LoRA adapters and steering vectors, one `object.pt` per steered cell) for one roster view of [`configs/sdfbase_full22_roster.yaml`](../../configs/sdfbase_full22_roster.yaml). From the repository root,

```bash
TINYCAP_APP_NAME=<name> modal run scripts/modal/tinycap_prefetch_model.py --repo <hub id> --revision <40-hex sha>
```

fetches and verifies a checkpoint against the Hub's file inventory, and

```bash
python scripts/modal/stage_full22_bank_modal.py --model-key <key> --view <view> --receipts-dir <dir> [--go]
```

stages that (model, view) of the bank, checking every object against the pins; `--go` puts, anything else is a dry run; `--modal` and `--source` default to site-specific paths; see `--help`.

The worked example's bank is not a full22 view: `reproduce/tools/stage_demo_bank_modal.py --model qwen35_4b_demo [--view <view>] --receipts-dir <dir> [--go]` stages the paper's committed objects under the row's own root (`/vol/objects/sdfbase/qwen35_4b_demo/<cell>/`, outside the full22 root) with the two derived cells' four parents under `/vol/out/`, every object held to its registry id and the paper key's pins, every volume command scoped to the Modal environment the harness runs under (`--environment`, the site's `SISTEER_MODAL_ENVIRONMENT`); `reproduce/07_bcb_bfcl.sh --phase stage` runs it before the tree's stage. A dose view (`--view d0p75` or `d0p75_relu`) stages the paper's committed 0.75 copy of the difference under the view's own row's root (`/vol/objects/sdfbase/qwen35_4b_demo/<view>/<cell>/`; rows `qwen35_4b_demo-d0p75` and `qwen35_4b_demo-d0p75_relu`, each reading its own explicit-cells config), the same cell id as the wave's parent: the bank, not the cell token, carries the dose.

## BigCodeBench and BFCL

From the repository root at a committed sha `SHA`, with `APP` any app name:

```bash
export TINYCAP_APP_NAME=$APP
modal run -e main -n $APP scripts/modal/tinycap_served.py --sha $SHA --app-name $APP --model <slug> --upload-tree --stage
modal run -e main -n $APP scripts/modal/tinycap_served.py --sha $SHA --app-name $APP --model <slug> \
    --pf --pass1 --cells <cells> --modes off --controls bigcodebench_full,bfcl_full --sampling protocol --serving-path lora_native
```

| flag | meaning |
|---|---|
| `--upload-tree --stage` | ships a `git archive` of `SHA` to the volume, never the local checkout |
| `--model` | a slug from [`contracts.MODEL_SLUGS`](../../src/sisteer/capeval/tinycap/contracts.py), each binding a checkpoint and a staged bank. The paper's rows are the roster's `model_row` entries (`qwen35_4b-f22s1`, `gemma4e4b-f22s1`, `llama8b-f22s1`, `nemotron3nano4b-f22s1`, …), lane rows (`contracts.BCBW_SLUGS`) the harness serves only through the wrapper below; a direct launch takes a row outside that list, e.g. `gemma4e4b` or `llama8b` |
| `--pf --pass1` | `--pass1` plans one GPU container per cell; without a pass the launcher plans nothing and exits 0. `--pf` is the CPU preflight every fitted cell launches behind: it holds each object to its HF post-load intervention hash in [`contracts.EXPECTED_INTERVENTION_HASHES`](../../src/sisteer/capeval/tinycap/contracts.py) under the slug, or in `--expect-hashes F` (a JSON object `{cell: hash}`, inline or a file) for a slug the table does not cover; `--pf` without `--pass1` is the preflight alone, whose report prints the hashes to pin |
| `--cells` | a comma list in the serving grammar: `none` (the unsteered model), `prompted__<si>`, `<si>__<arm>`; the paper's cells per model are the `cell_id` entries of [`configs/capeval/bcb_bfcl_full_roster.yaml`](../../configs/capeval/bcb_bfcl_full_roster.yaml), and a fitted cell must be in the slug's bank (the hash table's keys under the slug name them). More than `contracts.MAX_GPU_CONTAINERS` (8) cells need `--allow-extra` |
| `--modes off` | thinking off, as every paper row ran. Without `--modes` the launcher plans an `on` leg too, which a row whose `thinking_modes` in `contracts.MODELS` is `off` alone (Qwen, Llama and Nemotron have no pinned reasoning parser) refuses |
| `--controls` | `bigcodebench_full,bfcl_full`: the full suites the paper's rows used, every item pinned by id in [`configs/capeval/tasks/{bcb_full,bfcl_full}.json`](../../configs/capeval/tasks/) (1,140 BigCodeBench items, Hard included; 3,981 single-turn BFCL items). About half a GPU-hour per cell: the paper's Qwen3.5-4B cells spent 5 to 23 minutes generating BigCodeBench and 3 to 14 minutes BFCL after a standup of about 15 minutes. BFCL is omitted for Llama |
| `--sampling protocol` | each benchmark's reference decoding, as every paper row ran: BigCodeBench greedy, BFCL at temperature 0.001 under the server's pinned clamp override. Without it the launcher samples from the model card's row (Qwen at temperature 0.7) |
| `--serving-path lora_native` | loads LoRA objects as PEFT adapters |

Each cell (and each thinking mode, with `--modes`) gets its own vLLM standup and container and runs Inspect's BigCodeBench and BFCL tasks; generated code runs in ephemeral Modal Sandboxes. Stores land under `/vol/tinycap/gen`; nothing reaches the repository's results registry.

Two front doors compose this command. `python scripts/bcb_bfcl_200_widen_launch.py --roster configs/capeval/bcb_bfcl_full_roster.yaml --wave <wave> --sha $SHA --stage-kind wave --forecast-h <hours> [--dry-run]` launches a roster wave on its lane row with the roster snapshot the row requires (its interpreter, ledger and log paths default to site-specific locations; see `--help`). [`reproduce/07_bcb_bfcl.sh`](../../reproduce/07_bcb_bfcl.sh) `--model <key> [--view <view>]` composes it for a `models.yaml` key with phase views (the worked example `qwen35_4b_demo`), translating the view's cells to the harness's tokens, naming the view's row as `--model` (the key for the wave, `<key>-<view>` for a dose view) and adding `--allow-extra` and `--expect-hashes` as the cells need. `tests/test_modal_readme_recipe.py` replays the command above through the harness's own planner.

## The CPU test suite on Modal

From the repository root at a committed sha:

```bash
modal run -n cpu-suite-<shortsha> scripts/modal/cpu_suite.py --stage --sha <sha> --go [--shards 10]
```

It stages a shallow checkout at `/vol/repos/sisteer-<sha>` and runs `tests/` in a CPU container built from its `pyproject.toml` and `uv.lock`. The merged `summary.json` lands at `/vol/cpu-suite/<sha>/` (`modal volume get twosi-fits /cpu-suite/<sha>/summary.json .` reads it back), with `pytest.log` and `junit.xml` beside it or, sharded, under `shards/<i>-of-<N>/` for failed shards only.

## Other files

`tinycap_sandbox.py` (EvalPlus sandboxes for HumanEval+/MBPP+) and `tinycap_images.py` (the Nemotron image with fused Mamba kernels) accompany the harness. The remaining apps ran the project's other evaluations, among them the Qwen seed-replicate battery (`seedrep_served.py`) and the Qwen agentic-misalignment shards at the vendor sampler (`misalign_vendor_shards.py`); they bind volumes (`twosi-fits`, `e4b-hpo`, `llama-hpo`) that must already hold checkpoints, prompts and object banks, and several default to site-specific paths, so none runs unedited from a clone.
