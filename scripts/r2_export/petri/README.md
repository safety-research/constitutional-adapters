# Petri statistics for cells audited after a view's main audit

The two scripts here turn a Petri alignment-audit root into per-cell, per-behaviour means with bootstrap intervals and paired contrasts against the unsteered model. View, cell, `none` and `full22` are defined in [the parent README](../README.md).

The outputs feed the round-2 export ([`build_export.py`](../build_export/README.md), the `component = petri` rows) and the registry's [`petri_stats` adapter](../../../src/sisteer/registry/adapters/petri_stats.py). A reader of the paper does not need them; the misalignment figures read the registry. Nothing here runs from a clone: the audit roots live outside the repository.

## Why a separate root

A cell's `petri_run_key` is content-addressed by the full harness configuration, so adding a cell to a view's manifest re-keys every cell. A cell added after the view's audit therefore gets its own root, `petri_full22_<tag><suffix>/`, whose `none` is an unaudited placeholder: contrasts pair against the view's main audit of `none` under `petri_full22_<tag>/`. The suffix says which cells a root holds:

| Suffix | Cells audited there |
|---|---|
| `_newcells` | cells added to a view's roster after its audit: the control-trained LoRA, the difference LoRA and the prompted Constitution SI |
| `_prompted` | the prompted Safety SI |
| `_arditi` | the Arditi refusal-direction steer baseline |
| `_alphasteer` | the AlphaSteer baseline |

## The audit, as the scripts read it

The audit itself is run by `petri_proxy.sbatch` and `petri_drive_full22.sh` in [`scripts/sdfbase/full22/`](../../sdfbase/full22/README.md) with [`scripts/petri_run.py`](../../petri_run.py).

| Setting | Value |
|---|---|
| Scenarios | 6 behaviours x 10 scenarios, one frozen bundle, one epoch: 60 items per cell |
| Behaviours | `cyber_bio_chem_capabilities`, `self_preservation`, `delusion_sycophancy`, `defer_to_users`, `reasoning_unfaithfulness`, `instruction_evasion` |
| Auditor | claude-sonnet-5, with a fallback chain of grok-4.6 then a local abliterated Qwen; the chain used is recorded per cell as `auditor_chain` |
| Judge | claude-opus-5, asserted from each leg's inspect logs |
| Score | 1 to 10 per (behaviour, scenario) transcript; higher is more of the audited failure behaviour, except where the behaviour named is a desirable trait |
| Interval | percentile bootstrap over scenarios, 10,000 draws, seed 0 |
| Contrast | mean difference, cell minus `none`, on the shared scenarios: paired bootstrap interval, exact two-sided sign test |
| Pooling | `mean_of_behavior_means`, the six behaviour means at equal weight |
| Completion | `COMPLETE` (60 of 60 audited); `INCOMPLETE_OVERFLOW_LEDGER_RERUN_IN_FLIGHT` (unresolved entries in `petri_incomplete.json`, usually a target context overflow a re-run pass finishes); `NO_RESULTS_NO_LEDGER` (neither, as for the placeholder `none`) |

## Usage

`petri_stats_sidecar.py` is the parameterised generator; `petri_stats_newcells.py` fixes one call of it (tag `qwen35_4b_it`, root `petri_full22_qwen35_4b_it_newcells`, anchor `own`; only `--out`), so its contrasts are all `PENDING` by construction and `build_export.py` pairs those cells against the view's main audit of `none` itself. Both need `numpy` and `inspect_ai`, from the repository's `petri` dependency group.

```bash
# on the machine that holds the audit roots, from this directory
uv run --group petri python petri_stats_sidecar.py --suffix _arditi --tags gemma4e4b_it,qwen35_4b_it,nemotron3nano4b,llama8b_it --anchor wave --out petri_full22_arditi_stats.json
```

`--suffix` and `--tags` (comma-separated view tags) name the roots read, `petri_full22_<tag><suffix>` under `--state` (default `~/.local/state/petri`). `--anchor wave` (the default) pairs against the view's main audit of `none`, `--anchor own` against the root's own placeholder. `--wave-stats FILE[,FILE]` names statistics already filed for the view: the wave anchor's per-scenario scores and run key must agree with them or the script refuses, so pass them when extending filed statistics. `--help` lists the rest.

## Output

Cells are keyed by config cell id (`constv2-mix-full__lora_r4_ce_minus_ctl`), not the harness's `si_constv2_mix_full__lora_r4_ce_minus_ctl`; [the full22 README](../../sdfbase/full22/README.md) maps cell ids to the paper's objects.

| Key | Meaning |
|---|---|
| `tags.<tag>.audit`, `root`, `n_cells`, `n_complete`, `n_incomplete_ledgered` | the audit name, the root read, and cell counts by completion state |
| `tags.<tag>.cells.<id>` | per cell: run key, intervention hash, serving identity, completion state, `auditor_chain`, per-behaviour scores, means and intervals, and `mean_of_behavior_means`; `eval_awareness` and `scenario_realism` are transcript-quality dimensions, reported but not pooled |
| `tags.<tag>.vs_bare.<id>.<behaviour>` | the contrast against the anchor: mean difference, bootstrap interval, paired count and sign-test p, or `PENDING` with a reason |
| `tags.<tag>.anchor` | the wave anchor's run key and manifest hashes; a scenario-bundle mismatch leaves the contrast `PENDING`, a differing run key or behaviour-files hash is only a warning |
