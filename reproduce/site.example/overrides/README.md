# Run overrides

A site's own run-overrides files live here. You need one only to run a stage with settings other than the paper's;
otherwise every stage uses `reproduce/overrides/paper_defaults.env`, the only file shipped.

Select a file with `--overrides <file>` on a stage or `SISTEER_RUN_OVERRIDES=<file>` in `config.env`. Stages that submit
Slurm jobs (02, 03, 04, 06, 09) have the jobs read `reproduce/overrides/<basename>` from the committed checkout, so
their file must be committed there; this directory serves the grading-host-only stages (00, 05, 07, 08, 10). The file
sets `OVERRIDE_ID`; id, sha256 and keys are stamped on every registry row (`extra_json.run_overrides`).

| key | paper | meaning |
|---|---|---|
| `OVERRIDE_INIT_SCHEME` | `zero` | `zero` (LoRA B = 0; ReLU steer direction and probe weights 0, gate bias 1) or `random_lecun_1e-3` |
| `OVERRIDE_LORA_ARCH_SEARCH` | `0` | `1` makes `03_hpo.sh --object-class lora` search rank by module set |
| `OVERRIDE_HPO_WARMUP_FRAC` | empty | empty searches two warm-up levels (0.05, 0.15); a value fixes one |
| `OVERRIDE_HPO_CLIP_MODE`, `OVERRIDE_HPO_CLIP_PERCENTILE` | `searched`, empty | `reconnaissance` sets the gradient clip at a percentile (default 95) of a reconnaissance run |
| `OVERRIDE_HPO_TRIALS`, `OVERRIDE_HPO_RECENTRE_TRIALS` | `16`, `4` | round-1 design size; follow-up trials |
| `OVERRIDE_EFFECTIVE_BATCH` | `64` | effective batch of the reconnaissance rungs; it must equal the batch the model's fits config realises (`batch_size_by_arm` times `accum_steps`, or its `PENDING_MEMPROBE_EFF<n>_ACCUM` sentinel's n — 64 for the demo), and `03_hpo.sh` refuses a disagreement |
| `OVERRIDE_LORA_RANK`, `OVERRIDE_LORA_MODULE_SET`, `OVERRIDE_REFUSAL_STEER_ALPHA` | `4`, `wo` (= `o_proj`), `0.3` | recorded on rows, not read: the architecture is `lora_architecture` in `reproduce/models.yaml`, the refusal coefficient the model's `objects.refusal_direction` cell there (`ar_c<c>__arditi_allpos`; `SISTEER_REFUSAL_COEFFICIENT` may only restate it, and stage 04 refuses a disagreement) |
| `OVERRIDE_HOLD_OUT_EVALS`, `OVERRIDE_HOLD_OUT_SOURCE`, `OVERRIDE_MODEL` | unset | hold evals out of one model's gate (jobs still run, units stamped `held_out`); the source says who decided and when; `OVERRIDE_MODEL` must equal `--model` |
| `OVERRIDE_SMOKE_GOALS` | unset | `<lo>-<hi>/N` (N at most 99) narrows the multi-turn attacks and the padding ladder to goal shards `lo..hi` of `N`, planned, frozen and graded as that subset; the file's `OVERRIDE_ID` must contain `smoke` and `OVERRIDE_MODEL` must name the model. `reproduce/overrides/demo_smoke.env` is the shipped smoke recipe for the worked example (`0-0/99`: about three goals per attack) |

`03_hpo.sh` accepts the paper's HPO protocol (the defaults) or this architecture-search file; mixtures refuse by name:

```bash
OVERRIDE_ID=arch_search
OVERRIDE_INIT_SCHEME=zero
OVERRIDE_LORA_ARCH_SEARCH=1
OVERRIDE_HPO_WARMUP_FRAC=0.15
OVERRIDE_HPO_CLIP_MODE=reconnaissance
OVERRIDE_HPO_CLIP_PERCENTILE=95
OVERRIDE_HPO_TRIALS=16
OVERRIDE_HPO_RECENTRE_TRIALS=4
OVERRIDE_EFFECTIVE_BATCH=64
```
