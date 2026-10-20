# AlphaSteer strength screen launcher

This directory runs the over-refusal screen for one strength of the AlphaSteer baseline (arXiv 2506.07022, [`src/sisteer/extraction/alphasteer.py`](../../../src/sisteer/extraction/alphasteer.py)) on the same serving code and configuration as the paper's LoRA dose screens. AlphaSteer is a closed-form, input-dependent edit added to the prompt's hidden states, fit to vanish on one prompt class and fire on its contrast; its strength is the served op's coefficient `base_scale`. No figure reads these stores, so a reader of the paper does not need this directory; it is kept for extending the ladder to a third method.

## What the screen is

The controls job of the paper's served evaluations ([`../../sdfbase/full22/sampled/README.md`](../../sdfbase/full22/sampled/README.md)): XSTest (450 prompts: 250 safe, 200 unsafe contrasts) and OR-Bench-Hard (1,319 prompts), decoded under the model's vendor sampler (trial 0, `seed_base 20260909`) at a 512-token cap. Rungs 2048 and 8192 then re-draw, for each (cell, control) pair still at cap, the at-cap items plus up to 200 seeded below-cap items (the same items in every cell), until the pair's at-cap count is zero. The 512 rung is a ladder rung, not a verdict.

Grading (`scripts/served_capability_judge.py`, claude-sonnet-5: full refusal, partial refusal or full compliance) is separate; the verdict reads at-cap counts, loops and the positive control, not grades.

## Two shas, one instrument

`<wave_sha>` is the code: `ec72674809050dad3f0fec4c6b428ee50fc47b7c`, the LoRA screens' sha `f38d055d59a7102f930e30d3abac64adbcf1270b` plus two commits (the liveness policy below; the AlphaSteer fitter's provenance entry), neither touching the serving stack. Both shas record `same_instrument_as_lora_screens: true` in `asscreen_job.json`, any other `false`; at `f38d055d…` the driver still refuses AlphaSteer objects. `<as_sha>` is any commit carrying this directory and the view's configs (roster, pins, serving config, manifest, and the sampled variants `make_strength_sampled_configs.py` writes); no code runs from it.

## Running one view

Prerequisites: a Slurm cluster with a shared root `$SHARED` on the cluster's shared storage, the strength bank at `$SHARED/exp/objects/alphasteer_f22/<target>/strength` (from `../bake_strength.py` and `../stage_strength.sh`; targets `gemma4e4b`, `qwen35_4b`, `nemotron3nano4b`), and a clean detached checkout of each sha, created from any clone on the cluster:

```bash
for sha in ec72674809050dad3f0fec4c6b428ee50fc47b7c <as_sha>; do
    git -C <clone> worktree add --detach "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$sha" "$sha"
done
```

From the `<as_sha>` checkout (views `as_dm0p5 as_d0p1 as_d0p3 as_d0p5 as_d1`):

```bash
sbatch -J asscr-nemotron3nano4b_as_d0p3-ctl0 scripts/alphasteer_f22/screen/asscreen_controls.sbatch \
    nemotron3nano4b as_d0p3 ec72674809050dad3f0fec4c6b428ee50fc47b7c <as_sha>
```

Positionals: `<target> <view> <wave_sha> <as_sha> [trial=0]`. The header asks one GPU for two hours at `--qos=high`, logging under `$SHARED/exp/logs/`; override `--qos` and `--output` where your site's names differ.

Outputs land under `$SHARED/exp/sdfbase_full22/sampled/`, with `<TAG>` = `<target>_<view>`:

| path | holds |
|---|---|
| `controls_cap<rung>/<TAG>_t0/cells/<store>/` | one store per cell and rung; cells `none` (the unsteered model) and `alphasteer` |
| `serving/controls_t0_<TAG>_<jobid>/` | `asscreen_job.json` (both shas), the positive-control verdict, the liveness record |
| `ladders/<TAG>_t0/ladder.json` | at-cap counts per rung |

## The verdict

The stop rule of [`../../sdfbase/full22/strength/README.md`](../../sdfbase/full22/strength/README.md): against the view's own `none`, a loop share of 0.50 or more on any readout at cap 512, an at-cap share 30 points above `none`'s (each cell at its final rung) or a failed positive control stops the expensive legs; a loop share between 0.25 and 0.50 flags them. From the repository root:

```bash
python scripts/sdfbase/full22/strength/screen_verdict.py --local-root <pulled copy of sampled/> \
    --roster configs/alphasteer_f22_strength_roster.yaml --tags nemotron3nano4b_as_d0p3 --run-reclassify
```

`--run-reclassify` runs the loop classifier, which needs the pinned `full22_join_scores.py` (not in this repository; `--rep-ratio-source <path>`); without it the loop clause is `incomplete`.

## Comparability with the LoRA screens

Identical by construction: the serving stack, standup, driver invocation (the LoRA screens' controls phase, unchanged), decode block and ladder rule. One asymmetry: a strength view serves one steered cell, so the positive control runs on `alphasteer` itself, where a two-cell LoRA dose view measures one steered cell and asserts the other. A prefill-only edit also differs in mechanism from a per-token LoRA write, so a dose-for-dose sentence across methods is meaningless.

Nominal strengths are labels. The grid -0.5, 0.1, 0.3, 0.5, 1.0 is `base_scale` in the served op's own units, comparable neither with a LoRA or ReLU-steer multiplier nor across layer sets; compare on the measured `realised_dose_median` (‖edit‖/‖h‖).

### Liveness at low dose

The LoRA screens prove a served object live with `steering_bites`, greedy-text divergence from the scale-0 model. A small enough correct dose changes no greedy token, and AlphaSteer nulls its edit on benign input by construction, so this screen treats the text probe as an observation at low dose rather than a gate.

`asscr_liveness.py` passes a cell on two numeric facts: an intervention hash distinct from `none`'s, and a non-zero realised dose from the offline capture (`scripts/sdfbase/full22/strength/dose_capture.py`), admitted when the rung sha-bound to the served object records the view's nominal dose, a non-zero `realised_dose_median` and a passed signed-ratio gate, and not live when it fails any of these. With no bound rung (none given, none for this object, another object's sha), a first-token logprob divergence above max(1e-3, 2 × the engine's repeat noise) stands in as a labelled proxy. A passing record is exported as `SISTEER_LOW_DOSE_LIVENESS_RECORD`, which the text probe consults; a failing cell stops the job; an unreadable instrument leaves the text gate in force.
