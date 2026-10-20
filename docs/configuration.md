# Configuration

The stage scripts under `reproduce/` read one site directory: a `config.env`, a `clusters/<label>.env` per Slurm
cluster and, optionally, a run-overrides file. Needed to run the pipeline, not to regenerate the figures
(`10_figures.sh` has defaults). The grading host's install is
`uv sync --frozen --extra cpu --group notebook --group corpus` (`00_setup.sh --what venvs`). The Petri audit's
drive needs the `petri` dependency group as well, which that install leaves out:
`uv sync --frozen --extra cpu --group notebook --group corpus --group petri`. Without it, `08_grade.sh` starts
the grading daemon with the Petri drive disabled, grades every other family, and prints that command as a warning,
aimed at the environment of the interpreter it probed: when `SISTEER_PYTHON` names a venv other than the repository's
`.venv`, the printed line carries `UV_PROJECT_ENVIRONMENT=<that venv>` and `--directory <repository>`, and an
interpreter outside any venv is named as such (no sync can add the group to it).

## `config.env`

```bash
# from the repository root
cp -r reproduce/site.example ~/.config/sisteer     # or any directory, named by SISTEER_SITE_DIR
$EDITOR ~/.config/sisteer/config.env
```

Every stage reads `$SISTEER_SITE_DIR/config.env` (default `~/.config/sisteer/`); `--config <file>` or `SISTEER_CONFIG`
names another file, whose directory is then the site directory. A run reads one file: its keys, `SISTEER_CLUSTERS` and
`SISTEER_PRIMARY_CLUSTER` included, decide the run, and a `config.env` beside a differently named file is not read. Keys are
documented inline with defaults; `$SHARED/...` in a storage value expands per cluster.

## `clusters/<label>.env`

One file per cluster, named for its label. `SISTEER_CLUSTERS` lists the clusters the stages fan out to by default. A block it
does not list is still read whenever something names that cluster (`python reproduce/lib/clusters.py --get <label> <VAR>`,
or a script given the cluster by its label or by the registry label on its records), so a site can describe a second
cluster without adding it to the default fan-out. `python reproduce/lib/clusters.py --site-dir <dir> --json` (or `--config
<file> --json` for another config file in that directory) prints the blocks of the listed clusters, passed to jobs on the
sbatch line. The first three rows are required.

| key | meaning |
|---|---|
| `CLUSTER_SSH`, `CLUSTER_SHARED_ROOT` | the login host's ssh alias; the storage root every compute node mounts (`$SHARED`) |
| `CLUSTER_PARTITION`, `CLUSTER_QOS` | the Slurm partition and QoS for jobs |
| `CLUSTER_SUBSTRATE`, `CLUSTER_LABEL` | `slurm`, and the site's name for this cluster, carried on records as `cluster_label` |
| `CLUSTER_ACCOUNT`, `CLUSTER_SBATCH_EXTRA`, `CLUSTER_DEV_PARTITION`, `CLUSTER_DEV_QOS` | optional: `--account`, extra sbatch flags, and a partition and QoS for the model fetch and stage 02's CPU steps (default: the job's) |
| `CLUSTER_HF_HOME`, `CLUSTER_HF_TOKEN_SOURCE` | the offline Hugging Face cache (default `$SHARED/hf_cache`); `hf_fetch.sh`'s read-token source, `op`, `env` or `none` (default) |
| `CLUSTER_REPO_MIRROR`, `CLUSTER_EVAL_REPO_ROOT` | the repository mirror; the root of the clean per-commit checkouts, `<root>/<sha>` |
| `CLUSTER_HELPERS_DIR`, `CLUSTER_ENVS_DIR`, `CLUSTER_FIT_ENV_NAME`, `CLUSTER_LOG_DIR` | the helper scripts stage 00 installs, the venvs, the fit venv's name, the job logs |
| `CLUSTER_HIGH_BUDGET_GPUS`, `CLUSTER_GRADE_MIRROR_ROOT` | the per-user GPU budget at `CLUSTER_QOS`, running plus pending (default 16); the grading host's mirror of this cluster's evaluation stores, filled by stage 08 |

## `reproduce/models.yaml`

One row per model: the paper's four and the worked example `qwen35_4b_demo`, which fits and serves the paper's pinned
Qwen3.5-4B checkpoints under its own key. `python reproduce/lib/models.py check` validates it.

| field | meaning |
|---|---|
| `fit`, `serve`, `transfer_kind` | the fit and served checkpoints (Hugging Face repo, pinned revision, registry `model_id`), the served chat template's hash and generation config, and `base_to_instruct` or `instruct_direct` |
| `architecture`, `layer_sets`, `thinking`, `relu_steer` | architecture facts, the eligible layers per LoRA module set, the thinking switch, the ReLU steer's layers and positions |
| `lora_architecture` | `rank`, `modules` (4 on `o_proj` throughout) and `selection_ruling_id`, which stage 04 requires before any LoRA fit and only `03_hpo.sh --phase pick` writes; `null` as shipped, so a new row runs the architecture search |
| `objects`, `views`, `wave`, `view_legacy_prefix`, `committed_objects_of` | the worked example's object behind each view cell, its phase views `battery`, `d0p75`, `d0p75_relu` and the one wave it declares, `wave: battery` ([reproduce/README.md](../reproduce/README.md)); the paper's rows name roster views (`main`, `it`), which the evaluation stages do not submit; the prefix names the generated configs' target, `<prefix>_<view>`; `committed_objects_of` names the paper key whose committed objects stage 05 stages in place of stage 04's fits |
| `run_in_this_work` | `false` as shipped: stages 06 and 09 (the evaluation waves) then answer `--dry-run` only, the earlier stages run regardless; `true` on the row you run |
| `fit.resources`, `serve.resources`, `attacker`, `display` | descriptive; no stage reads them. `configs/sdfbase_<fit model_id>_fits.yaml` sets micro-batch and accumulation (folded from stage 02's memory probe; effective batch 64 and training seed 3 for the demo, the committed objects' seed) and GPUs per fit (`fit_width:`, default 1) |

## Run overrides

Stages run with the paper's settings, `reproduce/overrides/paper_defaults.env`, unless `--overrides <file>` or
`SISTEER_RUN_OVERRIDES` names another. Stages 02, 03, 04, 06 and 09 submit Slurm jobs that read
`reproduce/overrides/<basename>` from the committed checkout, so their file must be committed there; its id, sha256
and keys are stamped on every registry row.
[reproduce/site.example/overrides/README.md](../reproduce/site.example/overrides/README.md) lists the keys and the
architecture-search file a new model needs.

## Other environment variables

| variable | meaning |
|---|---|
| `SISTEER_STATE_DIR` | the pipeline's grading-host state (completion markers, submission ledger, gate inventory; default `~/.local/state/sisteer-reproduce`) |
| `SISTEER_CPU_SRUN` | the `srun` prefix for stages 02 and 04's CPU steps (unset: the dev partition and QoS for 02, the job's for 04; set, even empty, used verbatim) |
| `SISTEER_REFUSAL_COEFFICIENT` | may only restate the refusal direction's coefficient, which the model's `objects.refusal_direction` cell in `reproduce/models.yaml` declares (`ar_c<c>__arditi_allpos`, the fitter's spelling; the worked example's `ar_c0p3…` is 0.3) and stage 04, the pins writer and stage 05 all read from the cell; stage 04 refuses a disagreement before any submission. To fit another rung, change the cell |
| `SISTEER_PG19_FILLER` | the PG-19 filler file (`scripts/fetch_stage2_data.py --only pg19-filler`; default `<repo>/data/pg19_filler.json`, not committed), from which `05_stage_roster.sh --step freeze` derives the padding units' instrument ids; absent, `06 --gate` binds them |
| `SISTEER_NB_PYTHON` | stage 10's notebook interpreter (default `<repo>/.venv/bin/python`, with nbconvert, ipykernel, pandas, pyarrow, matplotlib and scipy) |
| `CORPUS_GENERATOR_ROOT`, `CORPUS_GENERATOR_DATA_ROOT`, `CORPUS_GENERATOR_TOKENIZERS_DIR` | the corpus generator's code root (default `<repo>/corpus_generator`), data root (`<root>/data`, gitignored) and staged reference tokenizers, read by the pool builders; stage 02 sets the first and last |
| `SISTEER_CORPUS_MIRROR_HOST`, `SISTEER_CORPUS_MIRROR_ROOT` | where `corpusgen onpolicy mirror` copies a new corpus sample (ssh alias and root; unset: no mirror) |
| `SISTEER_FIGS_URL` | the base URL the corpus generator's section-coverage report links figures under (unset: local paths) |
| `SISTEER_FIGS_URL_BASE` | the public base URL of the figure mirror; the Petri report, the registry status page (`<base>/results-registry/`), the BCB/BFCL-200 analysis (`<base>/bcb-bfcl-200-widen/`), the realign and agentic-control analyses (`<base>/<figure dir>/`) and the full-22 sampled report (`<base>/f22-sampled/`) link their figures under it (unset: the status page has no URL, the Petri report leaves the variable's name in its links, and the analyses link figures by local path, relative to the document or the figure root's absolute path) |
| `SISTEER_JAILBREAK_PRIVATE_ROOT` | the grading host's private jailbreak root (raw transcripts, held-out corpora under `heldout_corpora/`, the labeled store under `labeled/`), read from the environment, then the site config (default `$STATE/sisteer-jailbreak`; `SISTEER_HELDOUT_CORPUS_DIR` still names the corpora directory outright) |
| `SISTEER_HELDOUT_CORPUS_DIR` | the site's held-out corpora directory; renellm replays `renellm.jsonl` there when present, else the repository's `data/heldout_corpora/renellm.jsonl` (unset: `heldout_corpora/` under `SISTEER_JAILBREAK_PRIVATE_ROOT`; the cluster jobs set `$SHARED/exp/heldout_corpora`) |
| `SISTEER_RENELLM_NONPAPER_CORPUS_SHA` | the content hash of a ReNeLLM corpus that is not the paper's, which acknowledges it for replay; every result is then labelled non-paper (unset: replay refuses any corpus but the paper's, whose hashes `configs/heldout/renellm.paper_corpus.json` records) |
| `SISTEER_PETRI_SMOKE_STATIC_R_OBJECT`, `SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT` | the two objects the Petri steered-serving smoke serves (`scripts/petri_steered_smoke.py`, `scripts/sbatch/petri_steered_smoke.sbatch`, `scripts/modal/petri_smoke.py`; the BoN speed-up bench's HF arm reads the first), environment then site config; a value may spell `$SHARED`, the sbatch reads both from the job environment, and the Modal port serves a `$SHARED/exp/jailbreak/<rel>` value from `/vol/objects/jailbreak/<rel>` on its volume. No default: each refuses by name when nothing names it |
| `SISTEER_ATTACKER_POOL_STATE_DIR` | the grading host's records of the Modal attacker pools (endpoint and health records), environment then site config (default `$STATE/sisteer/attacker-pool-modal`) |
| `SISTEER_PYTEST_LOCK` | the grading host's shared pytest lock the seed-replicate S0 runner takes, environment then site config (default `$STATE/sisteer-pytest/lock`) |
| `SISTEER_WANDB_PROJECT_<PURPOSE>` | one logging purpose's W&B project, environment then site config, checked before `WANDB_PROJECT` and `SISTEER_WANDB_PROJECT` (purposes: `tinycap`, `seedrep`, `distill`, `judged`, `serving-timing` as `SERVING_TIMING`, `multilora-cert` as `MULTILORA_CERT`, and `fits` for newly generated fit configs; default `sisteer-<purpose>`). Two purposes run in cluster jobs submitted by hand: `budget` (`scripts/measure_train_budget.py`'s figures) and `alphasteer` (the AlphaSteer screen and Llama gate sbatch). A compute node has no site config, so `reproduce/00_setup.sh --what helpers` writes their values into each cluster's `hf_site.env` and `hf_env.sh` exports them into the job; re-run that stage after changing them. Unset, `budget` logs to `sisteer-budget` and `alphasteer` leaves W&B off |
