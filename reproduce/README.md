# Running the pipeline on one model

`reproduce/` runs the paper's pipeline for one model on your own Slurm cluster:

- corpus preparation
- the hyperparameter search
- the constitution-trained and control-trained objects and their differences (the constitutional adapters)
- the evaluation wave
- grading
- the dose ladder
- the figures

Eleven numbered stage scripts share one site config and answer `--help` and `--dry-run`. Stages 02 to 10 read or write
the results registry ([docs/registry/](../docs/registry/README.md)). Stage 07's run stores are copied off Modal to the
grading host and the primary cluster, sha-verified, and its gate verdict reaches the registry as an `analysis` record
that names the copy.

You need this directory to re-run or extend the experiments, not to draw the paper's figures.

## The figures alone

The paper's figures need no GPU, cluster or site config:

```bash
# from the repository root
uv sync --frozen --extra cpu --group notebook
reproduce/10_figures.sh --source snapshot
```

This renders every notebook figure from the snapshot under `paper/snapshot/` in about ten minutes on a CPU, into
the figures root's `registry-notebook/` ([paper/notebooks/README.md](../paper/notebooks/README.md)). It needs `git` on `PATH`.

## The worked example

The worked example runs stage 03's serving probe and stages 05 to 09 on the paper's committed Qwen3.5-4B objects, under
a key of its own, `qwen35_4b_demo`, at smoke scale and without refitting. Fitting your own objects (stages 02 to 04,
under a key of your own) has not been run end to end.

These are all of its commands, in the order you run them, from the repository root. The block is a list, not a script:

- between the first two lines you edit the site config, and change and commit one line of the model's row
- several calls wait on cluster jobs, so run each one when its section below says to
- 07's receipt prints one more command, which you complete and run
- 08 runs as a daemon for the rest of the sequence, so give it a terminal of its own

```bash
# setup
cp -r reproduce/site.example ~/.config/sisteer
reproduce/00_setup.sh --what all --model qwen35_4b_demo
python reproduce/tools/import_objects.py registry --model qwen35_4b_demo
# 03
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --probe-width 1
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --parity-pass
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --reopen --reverdict-only
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --probe-width 1
python reproduce/tools/replan_compute.py --model qwen35_4b_demo --serve-only \
  --probe ~/.local/state/sisteer-reproduce/qwen35_4b_demo/03_hpo/lora/probe_verdict.json \
  --fits-config configs/sdfbase_qwen35_4b_demo_base_fits.yaml
# 05
reproduce/05_stage_roster.sh --model qwen35_4b_demo
# 06
reproduce/06_eval_waves.sh --model qwen35_4b_demo --view battery --family all
# 07
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase stage
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase preflight
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase run --cells none,difference_lora
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase receipt --cells none,difference_lora
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase mirror --cells none,difference_lora
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase file --cells none,difference_lora
# 08
reproduce/08_grade.sh --model qwen35_4b_demo --daemon
# gate
reproduce/06_eval_waves.sh --model qwen35_4b_demo --gate all --verdict
python reproduce/lib/admission.py check --state-root ~/.local/state/sisteer-reproduce/qwen35_4b_demo --registry ~/registry/registry.sqlite
# 09
reproduce/09_dose_ladder.sh --model qwen35_4b_demo --alpha 0.75 --step all,views --force-gate <logical work id>=<reason>
reproduce/09_dose_ladder.sh --model qwen35_4b_demo --alpha 0.75 --step all,views --force-gate <logical work id>=<reason>
```

[The worked example, stage by stage](#the-worked-example-stage-by-stage), below, says what each set of commands does,
why it comes where it does, and what to do when a call waits or refuses.

## Which model the stages run

`models.yaml` lists:

- the paper's main-figure model, `qwen35_4b` (Qwen3.5-4B, fit on `Qwen/Qwen3.5-4B-Base` and served on
  `Qwen/Qwen3.5-4B`)
- the paper's three appendix models, `gemma4e4b`, `llama8b` and `nemotron3nano4b`
- the worked example, `qwen35_4b_demo`: the same pinned Qwen3.5-4B checkpoints under a key of their own

The paper's objects are committed under `objects/` and are not refit here. The paper's rows name their served checkpoints
and the roster views `main` and `it`; the evaluation stages do not submit them.

The demo key runs on the paper's committed Qwen3.5-4B objects:

- `objects/DEMO.json` names one object per intervention: the six fitted ones at one training seed, 3 (the seed of the
  best difference LoRA), and the refusal direction at coefficient 0.3.
- Stage 05 stages them into the demo's bank in place of stage 04's fits, each held to its registry id.
- Stages 07 and 09 import the paper's 0.75 dose copies of the two differences.

The demo key's fit identity, `qwen35_4b_demo_base`, has a fits config and pool directory of its own, so nothing written
under the key can be mistaken for a paper result. Stage 04 refuses to fit under the key, whose every object is the
paper's: fitting your own objects takes a key of your own (Adding a model, below). The row declares one evaluation wave,
`battery`, in which every arm is evaluated beside its own unsteered legs, and two dose views, `d0p75` and `d0p75_relu`,
for stages 07 and 09. It ships `run_in_this_work: false`, which stages 06 and 09 refuse; set it to `true` and commit
before a real run.

## Prerequisites

- One or two Slurm clusters, each reachable by an ssh alias from the machine that runs the scripts, with a filesystem
  every compute node mounts. A fit takes the fits config's `fit_width` GPUs (1 when absent, as in the demo config).
  Every evaluation job takes one GPU. The multi-turn attacks run their targets at the same time, beside an attacker pool
  of one GPU per three targets: at full scale 12 targets and 4 attacker GPUs on one cluster, or 6 and 2 on each of two.
- The machine you run from is the grading host: no GPU; it holds the registry, runs the judges and the stages, and needs
  `uv`, `git`, `ssh`, `rsync` and a `systemd --user` session.
- Stage 00 downloads the pinned checkpoints, the attacker model, the reference tokenizers, HarmBench and Alpaca into each
  cluster's Hugging Face cache; jobs load offline. A gated repository needs `HF_TOKEN`, passed over ssh's stdin, never
  stored on the cluster. Its `third_party` step fetches the files the repository does not distribute, the G0DM0D3
  templates and the refusal direction's prompt sets (re-drawn from their public upstream and checked against
  `data/arditi/fit_prompts.manifest.json`), and installs them under each cluster's shared root.
- Stage 07 alone needs a Modal account and the Modal client. The project's `modal` extra pins it:
  `uv sync --frozen --extra cpu --extra modal --group notebook --group corpus` (the grading host's install plus the client), then
  `SISTEER_MODAL_PYTHON=<repository>/.venv/bin/python` in `config.env`.

Credentials arrive as environment variables (`SISTEER_SECRETS_MODE=env`, the example config's mode), through a
secret-manager CLI wrapper (`op`), or, on a GCP VM whose service account is federated with the Anthropic API, through
workload identity federation with no Anthropic key at all (`wif`; [docs/secrets.md](../docs/secrets.md)). In `env` mode:

| variable | used by |
|---|---|
| `ANTHROPIC_API_KEY` | the stage 08 judges (StrongREJECT `claude-haiku-4-5`; over-refusal and agentic misalignment `claude-sonnet-5`; Petri judge `claude-opus-5`, auditor `claude-sonnet-5`) and stage 01 (`claude-opus-5`, Batch API and live requests) |
| `OPENROUTER_API_KEY` | the agentic-misalignment auditor and the Petri fallback legs (Grok 4.6) |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | stage 07 |
| `HF_TOKEN` | gated checkpoints in stage 00; optional |
| `WANDB_API_KEY` | training curves, only with `SISTEER_WANDB_ENTITY` and `SISTEER_WANDB_PROJECT` set |

## Cells, views, waves and families

A cell is one served configuration of the model (unsteered, a prompted system instruction, or one object at one
coefficient); a view is the set of cells one submission (a wave) serves. The roster
(`configs/sdfbase_full22_roster.yaml`) lists each view's served checkpoint, battery and cells; every staged object's
sha256 is pinned in `configs/sdfbase_full22_pins.json`.

A family is one group of evaluations:

- `static`: single-turn jailbreaks
- `padding`: the padding ladder
- `bon`: best-of-N
- `multiturn`: the attacks that need the adaptive attacker model. Crescendo and ActorAttack are multi-turn; PAIR is
  single-turn from the defender's side, so the paper's multi-turn category counts only the first two.
- `petri`: the Petri alignment audit
- `agentic`: agentic misalignment
- `controls`: XSTest and OR-Bench over-refusal
- `mmlu`
- `gsm8k`

The `battery` view's cells, in the paper's terms:

| cell | paper term |
|---|---|
| `none` | the unsteered model |
| `prompted__safety_refusal`, `prompted__si_constitution` | the safety and constitution system instructions |
| `refusal_direction` | the refusal-direction baseline (`arditi_allpos`, coefficient 0.3) |
| `control_lora`, `constitution_lora` | the control and constitution rank-4 LoRA adapters |
| `difference_lora` | the LoRA constitutional adapter (constitution minus control) |
| `control_relu`, `constitution_relu` | the control and constitution ReLU steers (`psr` in cell ids) |
| `difference_relu` | the ReLU constitutional adapter (the paired difference) |

The dose views `d0p75` and `d0p75_relu` serve `none` and the difference object at 0.75: a multiplier on the LoRA's
write, or on the ReLU steer's direction norm; different quantities, never one axis; 1.0 is the object as served.

## Running the stages

Every stage takes:

- `--model <key>`
- `--config <file>`, by default `$SISTEER_SITE_DIR/config.env`
- `--overrides <file>`
- `--dry-run`, which prints the exact commands with the config substituted
- `--help`

A refusal exits non-zero with the reason. A stage prints each registry record as `registry: <record_type> <id>` and each
submission as `submitted <cluster> <jobid> <unit>`; every submission also gets a `lib/watch_job.sh` watcher and a line in
`$SISTEER_STATE_DIR/<model>/submissions.jsonl` (default `~/.local/state/sisteer-reproduce/`). Re-running a Slurm stage
submits only what is missing: the funnel skips a unit already bound to a job. Stage 07 has no such check and re-runs its
Modal stage and evaluation on every call. Values in angle brackets are yours.

Commit before you submit: the submitting stages run the mirrored checkout at the current commit, so uncommitted edits
never reach the cluster, and 00, 04, 05, 06, 07 and 09 refuse modified tracked files.

## The worked example, stage by stage

The worked example runs under the shipped smoke recipe `reproduce/overrides/demo_smoke.env`: the paper's settings plus
one knob that narrows the multi-turn attacks and the padding ladder to about three goals per attack (the other families
run whole). Every registry row it produces is stamped `run_overrides.id = demo_smoke`, which is how a smoke row is told
from a paper-defaults row. For the full battery under the paper's defaults:

- Leave `SISTEER_RUN_OVERRIDES` empty, and the rows are stamped `paper_defaults`.
- Drop `--cells` from the stage 07 lines: all ten cells run, and the stage passes the launcher's cost trip-wire flag and
  says so.

The stages run in numeric order, one call each, except:

- 03 and 07 take several calls, for the reasons their sections give.
- 09 is one command, run twice.
- 06 comes back after 08: it records the wave's gate once 08 has graded the wave.

Each section below repeats its stage's commands from the block above, in the same order.

### Setup

Every stage reads the site config, so it comes first. The model's row is committed next, before stage 00 copies the
checkout to the clusters, and the registry import comes last, into the registry stage 00 installs.

```bash
cp -r reproduce/site.example ~/.config/sisteer
reproduce/00_setup.sh --what all --model qwen35_4b_demo
python reproduce/tools/import_objects.py registry --model qwen35_4b_demo
```

- After the copy, edit `config.env` and `clusters/<label>.env` ([docs/configuration.md](../docs/configuration.md)).
  Every key is documented inline, and `$SHARED` in a storage root expands per cluster. To keep the site elsewhere, set
  `SISTEER_SITE_DIR`.
- In `config.env`, select the smoke recipe: `SISTEER_RUN_OVERRIDES=reproduce/overrides/demo_smoke.env`. Select it there
  once, so stages 05 to 09 and the grading daemon read the same document: stage 05 freezes its knob into the inventory,
  and stage 06 refuses a plan whose knob differs.
- Set `run_in_this_work: true` on the `qwen35_4b_demo` row of `reproduce/models.yaml`, and commit.
- Stage 00 installs the grading host's environment, the registry and its timers, the cluster helpers and the repository
  mirror, and downloads the checkpoints (Prerequisites, above). It runs on the grading host and reaches each cluster over
  ssh, with no GPU. `--what <part>` runs one part again.
- The import adds the nine committed objects to the registry, where stages 05 and 07 read them. An object already there
  is skipped.

The worked example generates no text: the paper's corpora are committed under `corpora/` (stage 01 is how they were
made; see Generating new corpora, below).

### 03, the serving probe

The probe certifies that the serving engine can serve the one declared LoRA pair (rank 4 on `o_proj`). The compute plan
and every steered standup read its verdict, so it runs before the wave. It takes four calls, because the probe and its
parity pass are two jobs and the re-verdict reads both. The fourth call is the first one again: run it once the
re-verdict is done, and it fetches the result.

```bash
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --probe-width 1
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --parity-pass
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --reopen --reverdict-only
reproduce/03_hpo.sh --model qwen35_4b_demo --object-class lora --phase probe --probe-width 1
python reproduce/tools/replan_compute.py --model qwen35_4b_demo --serve-only \
  --probe ~/.local/state/sisteer-reproduce/qwen35_4b_demo/03_hpo/lora/probe_verdict.json \
  --fits-config configs/sdfbase_qwen35_4b_demo_base_fits.yaml
```

- The first call submits the probe. The second submits its parity pass: the same pair at the benign scale, on a record of
  its own. The two jobs can run at the same time.
- When both are done, the third call re-reads the probe's verdict against the pass, in a job of its own. The probe's own
  adapters are scaled to size the serving ladder, not to judge adapter parity, and the steered standups refuse the model
  until this verdict stands.
- When that job is done, the fourth call fetches the re-verdicted menu and verdict, and `replan_compute.py` prices the
  compute plan from it into `~/.local/state/sisteer-reproduce/qwen35_4b_demo/compute_plan.json`. `--serve-only` marks
  the fit rows imported and prices only the serving rows. Stages 06 and 09 refuse until the plan is complete.
- Each of the three jobs takes one GPU, for up to `SISTEER_TIME_PROBE` (four hours).
- A call made while its job runs submits nothing and says what it waits for.
- `--parity-record DIR` on the third call binds another parity pass; the record remembers the pass it last read.
- A verdict left uncertified by a reference the harness filed (it names `reference_missing`) is re-judged with
  `--phase probe --reopen --reference-only` before the pass is bound.
- A probe job that died before writing its record is named by the next call, with the remedy: the same call with
  `--resubmit`.

### 05, the objects and the frozen inventory

One call runs its six steps in order, on the grading host and on each cluster over ssh, with no GPU. The wave is
submitted only when it matches the inventory the last step freezes, so 05 comes before 06.

```bash
reproduce/05_stage_roster.sh --model qwen35_4b_demo
```

- `stage`: the committed objects into the bank and onto every cluster, sha-verified
- `pins`: the objects' registry ids into the pins file
- `roster`: checks that the roster carries the model, and prints the block when it does not
- `configs`: the wave's configs and manifests
- `bank-views`: each cluster's bank view, built from the committed configs
- `freeze`: the wave's inventory, every unit, with the smoke knob pinned in its header

The committed pins, roster and configs already carry the worked example, so the call normally runs through. If it
changes the pins or the configs, `bank-views` refuses the uncommitted tree: commit, then run the same command again
(staging skips the copies already in place). `--step <name>` runs one step.

### 06, the evaluation wave

One call submits every family of the `battery` view:

```bash
reproduce/06_eval_waves.sh --model qwen35_4b_demo --view battery --family all
```

- Every job takes one GPU, for up to `SISTEER_TIME_EVAL` (one day). The Petri proxy's limit is `SISTEER_TIME_PETRI` (one
  day, as the paper ran it), and the proxy frees its GPU when the drive ends.
- `static` goes first, and `bon` waits on its 512-token store.
- The static family's `renellm` attack replays the paper's frozen ReNeLLM corpus, which ships in the repository at
  `data/heldout_corpora/renellm.jsonl`, so replay is exact with no setup and nothing to build. A copy at
  `$SHARED/exp/heldout_corpora/renellm.jsonl` on a cluster is read first. The corpus is under its own terms
  ([`LICENSES/ReNeLLM-corpus-terms.txt`](../LICENSES/ReNeLLM-corpus-terms.txt)), not the repository's MIT licence: it
  may be used for evaluation but never for training, and the training stages refuse it.
  `python3 scripts/renellm_corpus_check.py` names the corpus a job would replay. Replay refuses any other corpus: a
  rebuilt one is a new sample, not comparable to the paper's numbers, and replays only when
  `SISTEER_RENELLM_NONPAPER_CORPUS_SHA` names its content hash (the refusal prints it), with every result labelled
  non-paper.
- The multi-turn and padding units run at the smoke's goal shards; the other families run whole. The smoke's three
  multi-turn targets, one per attack, share an attacker pool of one GPU on each cluster.
- After a failed submission, run the same command again: it submits only what is missing.
- `--resubmit` re-mints a unit whose job is provably dead and registered nothing. For the Petri family, `--resubmit`
  re-mints a Petri proxy whose dead drive already filed some legs, after reading those legs' stores as done; a filed leg
  that would be regenerated keeps the refusal.
- A Petri leg whose audits overflowed the serving window is re-run at a wider window by a second proxy:
  `reproduce/06_eval_waves.sh --model qwen35_4b_demo --view battery --family petri --pass overflow`. It refuses until the
  daemon has recorded the first pass incomplete on context-overflow ledgers (`grade/petri_drive.<tag>.json`), it runs on
  the cluster the first pass ran on, and the daemon drives the proxy it submits, from the tree that proxy was submitted
  at. An overflow-pass submit interrupted after its reservation is reconciled by the next `--pass overflow` call itself
  (the launched job adopted, or the reservation cleared) before the pass reads anything.

### 07, BigCodeBench and BFCL on Modal

07 reads nothing the other stages write beyond the registry import, so it can run in a second terminal while the wave
runs. Its smoke is the full BigCodeBench and BFCL sets on two cells, about one GPU-hour on Modal. Its `--phase all`
stages and runs but skips the preflight, which derives the hashes the run is held to, so the worked example calls each
phase:

```bash
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase stage
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase preflight
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase run --cells none,difference_lora
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase receipt --cells none,difference_lora
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase mirror --cells none,difference_lora
reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo --phase file --cells none,difference_lora
```

- `stage`: the view's bank, from the committed objects, on the Modal volume.
- `preflight`: one CPU container derives every fitted cell's hash and holds it to the paper's published value.
- `run`: the full sets on the cells named; omit `--cells` for all ten.
- `receipt`: pulls the run's stores off the volume and prints the `lib/stage.py verdict` command with `--headline` filled
  in. Add `--range 0.05,1` and run it: steering is expected to lower the BigCodeBench pass rate (The gate, below).
- `mirror`: copies the whole stores off the volume to this host and the primary cluster, sha-verified.
- `file`: files the verdict as a registry `analysis` record. It refuses without the mirror and takes the same `--cells`
  as the run.
- Keep the tree at the run's commit until the receipt: the receipt refuses a call at another commit.

Stage 07 serves the dose views too. Once 07's `battery` verdict is recorded, the same phases with `--view d0p75` (and
`--view d0p75_relu`) stage the paper's committed 0.75 copy into the view's own bank on the volume and run its two cells
under the view's own harness row, `qwen35_4b_demo-d0p75`. The view's logs, receipt, verdict, mirror and pointer records
land under `07_views/<view>/` in the model's state directory, beside the wave's, and its cluster copy under
`07_views/<view>/07_mirror/` in the model's reproduce directory on the primary cluster.

### 08, grading

08 does not wait for 07: start it as soon as 06 has submitted, and leave it running (in tmux or a `systemd --user`
service) for the rest of the sequence. It finds the wave by itself and grades the wave's and the dose ladder's runs as
they finish, checking every two minutes (`--interval`).

```bash
reproduce/08_grade.sh --model qwen35_4b_demo --daemon
```

- It runs on the grading host, with no GPU; its judges and auditors call the Anthropic and OpenRouter APIs.
- `--once` runs one pass instead.
- It drives the Petri audit only when the grading host's environment carries the `petri` dependency group:
  `uv sync --frozen --extra cpu --group notebook --group corpus --group petri`. Without it, stage 08 grades every other
  family and warns with that command, aimed at the environment `SISTEER_PYTHON` names.

### 06 again, the gate

The verdicts read the grades 08 files, so the gate comes after a later stage. Run it once the daemon has graded the
wave, then check that the battery admits the dose views:

```bash
reproduce/06_eval_waves.sh --model qwen35_4b_demo --gate all --verdict
python reproduce/lib/admission.py check --state-root ~/.local/state/sisteer-reproduce/qwen35_4b_demo --registry ~/registry/registry.sqlite
```

- The gate call reads each family's headline from the registry and records the family's verdict (The gate, below). It
  exits 3 when it skipped a family or ruled one `fail`.
- A family is named and skipped, with the reason, while one of its units is not registered or one of its unsteered legs
  is not graded yet. Run the same call again to pick it up later.
- The admission check prints every unit it still refuses, by logical work id, and why. A unit of a family ruled `fail`
  stays refused: 09's `--force-gate` admits it, with your reason on record.

### 09, the dose ladder

One command runs every step, and you run it twice:

```bash
reproduce/09_dose_ladder.sh --model qwen35_4b_demo --alpha 0.75 --step all,views --force-gate <logical work id>=<reason>
reproduce/09_dose_ladder.sh --model qwen35_4b_demo --alpha 0.75 --step all,views --force-gate <logical work id>=<reason>
```

- `build`: the paper's 0.75 copies are imported, not built
- `stage`: the copies into the bank and onto every cluster, sha-verified, and their pins
- `configs`: the dose views' configs
- `views`: one CPU job per cluster builds the dose views' bank views
- `guard`: every cluster's bank against the pins and the view manifests; exit 0 is the only pass
- `submit`: the dose views' legs, behind the admission check

The first call stops at `guard`, which refuses until the `views` jobs have built the views. Once each job has printed its
RESULT line, run the same command again: it skips the views already built and submits. (`all` alone leaves out `views`,
which submits jobs of its own.)

- The legs run as 06's do: one GPU per job, for up to `SISTEER_TIME_EVAL`.
- If `stage` or `configs` changes the pins or the configs, `views` and `submit` refuse the uncommitted tree: commit, then
  run the same command again.
- `--force-gate`: one per unit the admission check refused, with why; drop it when the check admits.
- `submit` sends the default `--evals`: `static`, `multiturn` and `controls`. The multi-turn legs go out through
  `06_eval_waves.sh`, which starts their attacker pool and places the targets on it. A dead pool member or target is
  re-run by the `06_eval_waves.sh … --family multiturn --resubmit` pass stage 09 prints, not by its own `--resubmit`.
- A `views` job that died is re-run with `--step views --resubmit --clusters <its cluster>`: a live job is skipped, a
  terminal one re-run after the funnel's proof, and a completed one refuses.

## Fitting your own objects

Fitting your own objects takes a key of your own (Adding a model, below): stage 04 refuses to fit under a key that
imports the paper's committed objects, as the worked example's does. Run the block below in place of the registry import,
stage 03's probe, the compute plan and stage 05 above, then 06, 08 and 09 as above with `--model <your key>`. Stage 07
does not run under a key of your own: its `stage` phase stages only the paper's committed objects on Modal, and refuses a
key that imports none.

```bash
# from the repository root
SEARCH=reproduce/overrides/<your search file>.env
reproduce/02_prepare_model.sh --model <your key> --step all
# commit the folded configs
reproduce/03_hpo.sh --model <your key> --object-class relu --phase all
reproduce/03_hpo.sh --model <your key> --object-class lora --phase all --overrides "$SEARCH"
reproduce/03_hpo.sh --model <your key> --object-class lora --phase pick --overrides "$SEARCH" --rank <rank> --set <set>
# commit models.yaml, configs/sdfbase_folds.json and the fits config
scp '<primary cluster>:<SISTEER_REPRODUCE_ROOT>/<your key>/02_prepare_model/memprobe.*.json' /tmp/
python reproduce/tools/replan_compute.py --model <your key> \
  --memprobe /tmp/memprobe.psr_ce_all.json --memprobe /tmp/memprobe.lora_r4_ce.json \
  --probe ~/.local/state/sisteer-reproduce/<your key>/03_hpo/lora/probe_verdict.json \
  --fits-config configs/sdfbase_<fit model id>_fits.yaml
reproduce/04_fit_objects.sh --model <your key> --object all
python scripts/sdfbase/make_full22_configs.py --derived-pins-from-registry
# commit the pins
reproduce/05_stage_roster.sh --model <your key> --from-bank
```

- 02, 03 and 04 submit jobs and exit: re-run a line until it reports nothing to wait for. 02's re-run folds the memory
  probe into the fits config and `configs/sdfbase_folds.json`; 04 builds a difference once both parents are published.
- 03 takes one call per object class, plus the pick, which only you can make. The LoRA fits refuse until the model's row
  in `models.yaml` records a selection ruling (`lora_architecture.selection_ruling_id`), and only `--phase pick` writes
  one, so every new model runs the architecture search, even to keep rank 4 on `o_proj`. Commit an overrides file under
  `reproduce/overrides/` ([site.example/overrides/README.md](site.example/overrides/README.md)), run the `lora --phase
  all` line with it, read `table.html`, then pick a rank (4, 16 or 64) and a module set from the pairs it shows:
  - `wo`: W_O, the paper's `o_proj`
  - `qkvo`: W_QKVO, the four attention projections
  - `down`: the MLP's down projection
  - `mlp`: the whole MLP (gate, up and down)
  - `attn_mlp`: all seven projections, W_QKVO and the whole MLP
- A pick with no clip measurement at its learning rate is recorded but not promoted: `--phase confirm-pick` submits the
  one confirming trial; re-run the pick when it lands.
- The compute plan reads the memory-probe records stage 02 left on the primary cluster, and the serving probe's verdict,
  which the LoRA search runs as its first phase here.
- Run 04 on ONE cluster per identity (`--cluster`, default the primary): never fit the same objects on both clusters at
  once. The lock that keeps two jobs of one identity from publishing two objects is a file on the fit cluster's storage,
  so a job on the other cluster never meets it and both publish; a second cluster takes a different model key, or waits
  for the first run to finish.
- `make_full22_configs.py --derived-pins-from-registry` writes the derived objects' and the refusal direction's registry
  ids into the pins, which 05's `stage` step requires. It reads the site's registry and campaign: `SISTEER_REGISTRY_ROOT`
  and `SISTEER_REGISTRY_CAMPAIGN` in the environment, or `--registry-root` and `--campaign`.
- `--from-bank` stages the fit bank's objects, the default for a key that imports no committed objects; the call runs
  the same six steps.
- 09's `build` step scales your two differences to 0.75, where the worked example imports the paper's copies.

## Generating new corpora

Stage 01 made the paper's two corpora, constitution and control, each in two parts (advice and stories). They are
committed as gzip shards under `corpora/`, and stage 02 reassembles them, so neither sequence above runs stage 01. It
generates with `corpusgen` (`corpus_generator/`) and `claude-opus-5`, and a re-run draws a new sample.

```bash
reproduce/01_make_corpora.sh --setup        # install the generator
reproduce/01_make_corpora.sh --test         # its offline tests; no API key
reproduce/01_make_corpora.sh --preflight    # check the configs and tokenizers
```

Generating takes further steps the script does not run, and part of it sends live API requests (up to 200 at once);
[corpus_generator/README.md](../corpus_generator/README.md) gives the full sequence. Only the offline steps have been
run from this repository.

## Reading a demo run

Stage 10 draws the paper's figures from the paper's runs; a demo run's rows never enter them. Read a demo run from the
registry: `run_metric` rows hold each run's aggregate metrics, `run_contrast` rows each steered run's paired difference
against its unsteered anchor, both filed by stage 08 ([docs/registry/records.md](../docs/registry/records.md)):

```bash
# `registry` is the synced venv's console script (else python -m sisteer.registry.cli)
registry --root ~/registry query "select r.run_id, r.eval_id, r.object_id, m.metric_id, m.value, m.k, m.n from run r join run_metric m using (run_id) where r.model_id = 'qwen35_4b_demo'"
```

## What the stages compute

| stage | method | compute |
|---|---|---|
| 00 | Installs the grading host's environment, the registry, the cluster helpers and the repository mirror, and downloads the pinned checkpoints into each cluster's Hugging Face cache. | the grading host, over ssh; no GPU |
| 01 | Generates the constitution and control corpora, each in two parts (advice and stories), with `claude-opus-5`. The paper's corpora are committed, so neither sequence above runs it. | the grading host; the Anthropic Batch API, and live requests for the control edit pass and the regenerations |
| 02 | Reassembles the committed corpora, checks the fit checkpoint, builds the training pools and registers the model. Its memory probe finds the largest micro-batch each fit can use. | CPU jobs on the primary cluster; one GPU per memory probe (1.5 h) |
| 03 | Learning rate and gradient clip over 16 Sobol trials split between warm-up 0.05 and 0.15, then a follow-up round chosen from their fit, with the LoRA at rank 4 on W_O (`o_proj`): the paper's protocol, the default. Alternatively, a LoRA architecture search examines ranks 4, 16 and 64 on W_O, W_QKVO, the MLP's down projection, the whole MLP (gate, up and down) or all seven projections, at one warm-up level and a clip set from a short reconnaissance run, and you pick the rank and modules from its table. | one GPU per trial, `SISTEER_TIME_HPO` (12 h); the serving probe one GPU for one pair, 8 for the search's 15, `SISTEER_TIME_PROBE` (4 h) |
| 04 | Fits a LoRA and a ReLU steer on each corpus by next-token cross-entropy (8,192-token sequences, effective batch 64), then derives the differences (constitution minus control) and the refusal direction (a difference in means over the paper fit's own 128 harmful and 128 harmless training instructions, re-drawn from the refusal-direction paper's reference implementation, coefficient 0.3). | `fit_width` GPUs per fit (1 in the demo config), `SISTEER_TIME_FIT` (2 days); one GPU for the ReLU difference and the refusal direction; CPU for the LoRA difference |
| 05 | Stages the objects onto every cluster, sha-verified, records their pins, renders the wave's configs and bank views, and freezes the wave's inventory of units. | the grading host and each cluster, over ssh; no GPU |
| 06 | Runs each family's evaluations on every cell of a view. With `--gate`, it checks each family's coverage and records its verdict from the graded headline. | one GPU per job, `SISTEER_TIME_EVAL` (1 day), the Petri proxy `SISTEER_TIME_PETRI` (1 day); an attacker pool of one GPU per three multi-turn targets |
| 07 | Runs the full BigCodeBench and BFCL sets (1,140 and 3,981 items) under each benchmark's reference decoding, and files the gate verdict on the unsteered cell's BigCodeBench pass rate. | Modal GPUs: about 1 GPU-hour for the two-cell smoke, about 6 for all ten cells |
| 08 | Grades each run as it finishes, and files the run's metrics and each steered run's difference from its unsteered anchor. | the grading host, no GPU; the judges on the Anthropic and OpenRouter APIs |
| 09 | Runs dose-scaled copies of the two difference objects (0.75, the paper's companion dose) under the static jailbreaks, the multi-turn attacks and the over-refusal controls. | as 06; one CPU job per cluster builds the dose views' bank views |
| 10 | Renders the paper's figures from the registry, or from the committed snapshot. | CPU only; about ten minutes from the snapshot |

## The gate

The `battery` wave carries its own unsteered legs, so it is the baseline: `--view battery` is not gated and needs only
the compute plan. Once stage 08 has graded a family, `--gate <family>` checks that every expected jailbreak goal, benchmark
item and Petri scenario has a completed, non-errored record. `--gate all` does this for every family. `--gate` reads each
unit's store at the location the registry recorded for the run, with the current tree's reader; every verdict names the
inventory's commit and the reader's.

With `--verdict`, the gate also records each family's verdict, `pass` when the family's headline lies in its range:

- The headline is the family's headline metric (`eval.headline_metric_id`) on the wave's unsteered legs, read from the
  registry.
- When a family has several unsteered legs, the one with the least headroom decides, so every leg needs headroom.
- A leg of an eval that the model's overrides file holds out of the gate (`OVERRIDE_HOLD_OUT_EVALS`) is not read.
- A family is ruled whole or not at all. It is named and skipped, its verdict left as it was, while one of its units is
  not registered yet, an eval it runs has no unsteered leg, or a leg is not graded yet.
- The call exits 3 when it skipped a family or ruled one `fail`.
- A `pass` read from the registry admits the dose views only while every leg it read still reads the same. If a leg is
  regraded after the ruling, admission refuses the family until you run the gate again.
- `--headline <value> --range <lo,hi>` in place of `--verdict` records one family's verdict by hand. A verdict recorded
  by hand stays tied to the runs it ruled, whatever their grades.

The range checks that the unsteered headline leaves headroom to see the effect steering is expected to have. It is never
a match to another model's value, and it bounds one side only:

- When steering is expected to lower the headline, the headline must lie above 5% of its scale, with no upper cap.
- When steering is expected to raise the headline, the headline must lie below 95% of its scale, with no lower cap.

`reproduce/lib/gate_headline.py` holds these ranges, in the units the registry files each headline in:

| family | headline | scale | steering is expected to | range |
|---|---|---|---|---|
| `static`, `padding`, `bon`, `multiturn` | `asr`, the attack success rate (an attack near 100% is admitted) | 0 to 1 | lower it | `0.05,1` |
| `mmlu`, `gsm8k` | `acc`, the accuracy | 0 to 1 | lower it | `0.05,1` |
| `agentic` | `misaligned_rate`, a percentage | 0 to 100 | lower it | `5,100` |
| `petri` | `petri_score`, on the judge's scale | 1 to 10 | lower it | `1.45,10` |
| `controls` | `xst_refused` and `orb_refused`, the over-refusal rates | 0 to 1 | raise it | `0,0.95` |
| stage 07's verdict | the unsteered cell's BigCodeBench pass rate | 0 to 1 | lower it | `0.05,1` |

The dose views refuse until every unit of the frozen inventory is registered, complete, covered and verdicted;
`python reproduce/lib/admission.py check --state-root <state dir>/<model> --registry <registry root>/registry.sqlite`
prints each unit it refuses, by logical work id, and why. `--force-gate <logical work id>=<reason>` on
`06_eval_waves.sh` or `09_dose_ladder.sh` is the recorded override, the id being the unit's `logical_work_id` in
`submissions.jsonl`. An empty reason, or an id that names no unit of the frozen inventory, is refused; the forced list
is printed and recorded in the stage's `WAVE.<view>.json` or `LADDER.<view>.json`.

## Adding a model

A new key needs three edits:

1. A `models.yaml` row. Copy `qwen35_4b_demo` without its `committed_objects_of` entries, which import the paper's
   objects in place of your fits. Its `refusal_direction` cell spells the refusal coefficient, `ar_c<c>__arditi_allpos`,
   and is the one place to change it.
2. A lane for its fit `model_id` in `scripts/sdfbase/make_sdfbase_configs.py` (`MODELS`, with `fit_selection`), and its
   rendered `configs/sdfbase_<fit model id>_fits.yaml`.
3. Its bare LoRA dose view in `BARE_DOSE_VIEWS_FROZEN` (`scripts/sdfbase/make_full22_configs.py`), without which
   `09 --step configs` refuses.

`05 --step roster` prints the roster block. Stage 07 does not serve a new key, so it needs no launcher row.

## Recovering after a lost session, and stopping jobs

```bash
# from the repository root
reproduce/lib/reconcile.sh --model qwen35_4b_demo                                      # sacct + squeue against the ledger; lists the live jobs to re-watch
reproduce/lib/stop_run.sh --cluster cluster_a --name fit-qwen35_4b_demo-control_lora   # a fit, probe or derived-build job (fit-<model>-<object>, derived-<model>-prep)
```

A bare `scancel` reads as a preemption to these self-resubmitting jobs; `stop_run.sh` writes the STOP sentinel first
(`--stop-dir <dir>` adds a job's own sentinel directory). Wave jobs and attacker pools are stopped through the
submission funnel; never write `STOP` at the wave root, which stops the whole wave:

```bash
# ATTEMPT_ID: the attempt_id of the unit's `attempt` line in $SISTEER_STATE_DIR/<model>/attempts.jsonl; GROUP: the pool's group, mt-<view>-<cluster label>
python reproduce/lib/submit_twin.py --config ~/.config/sisteer/config.env --model qwen35_4b_demo cancel-attempt --attempt-id "$ATTEMPT_ID"
python reproduce/lib/submit_twin.py --config ~/.config/sisteer/config.env --model qwen35_4b_demo teardown-pool --group "$GROUP"
```

## Layout

| path | what it is |
|---|---|
| `00_setup.sh` … `10_figures.sh` | the stages |
| `models.yaml` | the models table, read by `lib/models.py` |
| `site.example/` | the site config to copy |
| `overrides/` | the run-overrides files (`paper_defaults.env` is the paper's settings) |
| `lib/` | the shared library |
| `tools/replan_compute.py` | the compute plan |
| `tools/import_objects.py` | the committed objects for stage 05 and the registry: `cells`, `pins`, `registry` |
| `tools/stage_demo_bank_modal.py`, `tools/stage07_receipt.py`, `tools/stage07_mirror.py`, `tools/stage07_file.py` | stage 07's Modal bank, receipt, mirror and filing |
| `tools/snapshot_paper_frame.sh` | the snapshot export behind `10_figures.sh --export-snapshot` |
| `tools/reassemble_corpus.py`, `tools/shard_corpus.py` | the corpus shards |
| `tools/export_objects.py` | writes `objects/` from a registry |
| `tools/demo_objects.py` | writes `objects/DEMO.json`: one training seed for every Qwen3.5-4B intervention, the seed of the best difference LoRA by the notebook's multi-turn deltas; `--check` recomputes it |
| `tools/sparse_paths.py` | the directories the README's lighter checkouts fetch |
| `tools/scrub_scan.py`, `tools/scrub_grep.sh`, `tools/scan_allowlist.tsv` | the publication scan (text, binaries and the paper frame), its grep form and its reviewed exemptions |
