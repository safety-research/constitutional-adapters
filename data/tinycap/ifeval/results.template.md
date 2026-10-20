# IFEval in the canonical capability battery, thinking OFF, k = 2 seeded trials at the vendor presets, 2,048 -> 4,096 -> 8,192 ladder: 42 launched cells on four models. Results record (SKELETON — the IMMUTABLE template `results.template.md`; the double-brace placeholders are filled by `scripts/tinycap_ifeval_battery.py --results-md out/ifeval_canonical/results.md` into the rendered record, never over this file; `{…}` marks prose only a real wave supplies)

_Record status: {{production}}. Store pin `{{store_pin}}`; analysis tip `{{analysis_code_sha}}`. Every number in this record is read from
the analysis report; nothing is recomputed here. Where this record and the decision log could disagree, the log wins and the
text says so._

## As run (the single point of truth for the analysis tip; every other mention in this record points here)

| what | value |
|---|---|
| store pin (every store's producer tree) | `{{store_pin}}` |
| analysis tip (the code the report was written by) | `{{analysis_code_sha}}` |
| runner command, as run | `{…}` |
| report read here | `{…}` |
| stores | `{…}` (`controls/<TAG>/<cell_slug>/vendor_preset[__rung<cap>]/`, `rung<cap>/<TAG>/<cell_slug>/rung_subset.json`) |
| matrix | `configs/sdfbase/ifeval/matrix.json` sha256 `{{matrix_sha256}}` ({{n_launch}} launched rows, {{n_held}} held) |
| canonical item ids | the 541 vendor row keys of `google/IFEval @ 966cd895` (`{…}` json, sha256 `{…}`) |
| figures | `{…}/figs/` (PNG + PDF per stem, `final_ifeval_figures_numbers.json`, `final_ifeval_tables.json`); {{figures_publication}} |
| bootstrap | B = {{B}}, seed {{seed}}, {{level}} percentile intervals, paired on item id (the item is the resampling unit; 2 trials per item) |
| ladder trigger | {{trigger_provenance}} |
| loop rule | {{loop_rule}} |
| seed rule | `{{seed_rule}}` |

---

## Headline

**{{n_filed}} of {{n_designed}} designed contrasts filed: {{n_lower}} "lower than none", {{n_higher}} "higher than none", {{n_not_resolved}} "not resolved at this cap"; {{n_promoted}} of {{n_usable_cells}} usable cells promoted on the ladder; gates {{gates_passed}} / {{gates_total}} passed (failed: {{gates_failed}}).**

`{…}` — the per-model prose (which objects move which metric, by how much, with the interval) is written from the contrasts table
below once the wave has run; nothing is typed ahead of the report.

## 1. What was asked

The specification: add IFEval to the paper's canonical capability battery on the same served path and the same four model
families as GSM8K paired-200 and MATH-500 — inspect_evals' stock `ifeval` (541 prompts; prompt- and instruction-level, strict and
loose), sampled at each model's vendor preset with per-item seeds shared across lanes, thinking OFF, under a 2,048 -> 4,096 -> 8,192
cap ladder promoting on a > 1 % at-cap share, at-cap rows scored exactly as IFEval scores them and then loop-classified so the report
carries at-cap, truncated and loop shares separately.

## 2. What was built or changed, in behaviour terms

`{…}` — lane 1 (the `inspect/ifeval` task, the `vendor_preset` rows, the seed rule, request evidence v3, `--off-cap-by-task`, the Modal
launcher + dispatcher) and lane 2 (the matrix, the k = 2 draw-subset ladder, the analysis) as merged; the PR link.

## 3. Design as run

### 3.1 Models, render pins, sampling rows

| model | served | vendor preset row (as sent) | thinking-OFF server pin |
|---|---|---|---|
| gemma-4-E4B-it | `google/gemma-4-E4B-it @ ee0ef602` | 1.0 / 0.95 / top_k 64 (`generation_config.json`) | none composed (the template has no switch) |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B @ 851bf6e8` | 0.7 / 0.8 / top_k 20 / min_p 0 / presence_penalty 1.5 ON THE WIRE per request / repetition_penalty 1.0 (the vendor INSTRUCT general row) | `--default-chat-template-kwargs {"enable_thinking": false}` |
| Nemotron-3-Nano-4B | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @ dfaf35de` | 1.0 / 0.95 (`generation_config.json`; the sampled lanes' top_p 1.0 is a grandfathered exception recorded beside it — P43) | `--default-chat-template-kwargs {"enable_thinking": false}` |
| Llama-3.1-8B-Instruct | `NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79` | 0.6 / 0.9 (`generation_config.json`) | none composed |

Seeds: `{{seed_rule}}` — the same (item, trial) draws the same seed in every lane and every rung re-draw. k = 2 trials per item;
1,082 draws per cell. Served geometry per cap from the G-LENGTHS measurement (`configs/sdfbase/ifeval/geometry_measurement.json`:
max prompt 468 tokens -> L = 2,528 / 4,576 / 8,672); the served S per cell is the KV-derived one (`{…}` — the per-cell S / shrink table
from `cells.*.served_seqs` / `kv_shrink`).

### 3.2 The 46-row matrix

{{n_launch}} launched rows (gemma 11, Qwen 11, Nemotron 8, Llama 12 = main 6 + it 6 incl. the it-view `none` + prompted anchors) and
{{n_held}} HELD rows (the four it-view constitution cells of gemma / Qwen — a hard exclusion until the batch-64 re-fit lands; never a row,
a store or an estimate line). Every row: declared False (no `@thinking`), thinking OFF, `vendor_preset`, every path from
`ifeval_paths(model, view, cell, rung)`.

## 4. Results

### 4.1 The verdict rule, as implemented

A directional word iff the MERGED reading's item-paired bootstrap CI excludes zero on BOTH prompt-level metrics (strict and loose) with
agreeing sign; else "not resolved at this cap". Instruction-level metrics are filed beside, never deciding. The estimand:
`{…}` (the report's `estimand` string).

### 4.2 The designed contrasts (merged reading; the four metrics, pp with 95 % CIs)

{{contrasts_table}}

### 4.3 Per-cell at-cap, truncated and loop shares (pass 1 at 2,048)

{{shares_table}}

### 4.4 The ladder: triggers, rungs, the truncation cost

{{ladder_table}}

`{…}` — the `merged - r2048` truncation cost per promoted cell (report `truncation_cost`).

### 4.5 The noise floors (it - main on cells identical by construction, per two-view model; the label per the attestation)

Every cell the matrix carries on BOTH views with no object on either (llama8b: `none` and `prompted__si_constitution`) is a
floor candidate; a pair is a MEASURED NOISE FLOOR only when the two stores share the served checkpoint and revision, the SI
prefix, the geometry entry and served S, the sampling row, the cap, the seeds and the request bytes, and the scoring path is
the same — else it "measures that difference", named. Per draw and in aggregate on the same judged outcome; never pooled.

{{noise_floor_table}}

### 4.6 The at-cap ledger reconciliation (ledger at-cap == recorded at-cap, per leg, BEFORE any share was read)

{{ledger_table}}

## 5. Gates

{{gates_table}}

## 6. Acceptance checks against the spec

`{…}` — one line per spec acceptance check (seed derivation on the fixture; the sampling rows equal the vendor files; Qwen's
presence_penalty on the wire; the ladder promotes at > 1 %; at-cap rows score as IFEval scores them; the loop classifier matches the
cap ladder's; the reader reconciles ledger vs recorded at-cap; the CPU integration round-trip; the suite union-clean; the estimate).

## 7. Incidents, in order

`{…}`

## 8. Cost

Actual GPU-hours (ledger wall + the 250 s Modal standup adder per leg): {{total_gpu_h_actual}}. The CENTRAL estimate under the
30-cell matrix is 12-13 GPU-h for the whole wave (30 pass-1 cells ~8.1, the five constitution-prompted cells' wider windows ~0.8,
two Nemotron certifications 0.5, the S1 ladder scaled to 30 cells ~2.8; the plan's 15-17 was the 42-cell matrix's) with the
plan's labelled scenarios S0-S4 — a central estimate, never a bound. `{…}` — per-cell actual vs central (figure `final_ifeval_cost`).

## 9. Limitations and what is not verified

1. The loop rule counts WHITESPACE tokens, not model tokens ("last 200 tokens" in the specification is implemented as the last 200
   whitespace tokens — the cap ladder's rule as it exists; disclosed, not changed).
2. IFEval's 541 prompts are one instrument; a per-instruction-type breakdown is not filed here.
3. `langdetect` is seeded (0) in the plan branch; the 31 language prompts are as reproducible as that seed makes them.
4. The Llama tok/s used in the estimate was an assumption (D103's serving gate + the OR-Bench ledgers); the actual is in the cost block.
5. `{…}`

## 10. Reproduce

```bash
# 0. the matrix and the geometry measurement are frozen; --check both
PYTHONPATH=src python scripts/sdfbase/ifeval/make_ifeval_matrix.py --check
PYTHONPATH=src python scripts/sdfbase/ifeval/measure_ifeval_geometry.py --offline --check
# 1. the wave (lane 1's launcher + dispatcher; the shakedown first) — {…}
# 2. the ladder planner after pass 1, again after the 4,096 rungs
PYTHONPATH=src python scripts/tinycap_ifeval_ladder_plan.py --wave-root <pull> --sha <pin>
# 3. the analysis
PYTHONPATH=src python scripts/tinycap_ifeval_battery.py --gen-root <pull> --sha <pin7> --ids <541 ids json> --out report.json \
    --results-md out/ifeval_canonical/results.md --wandb-mode online
```
