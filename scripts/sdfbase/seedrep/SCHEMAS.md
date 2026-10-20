# seedrep run-state schemas

The run-state root `RS` (`artifact_locations.rs_root()`: `--rs`, else `$RS`, else its default) holds three records,
written by the fit validator, `artifact_locations.py write` and `derived_admission.py`, read by the stager to decide
which cells to stage and from where.

Conventions: `key` = `qwen35_4b_base_eb64_s<s>`, `seed` = the integer `s`. `cell` keeps the campaign grammar (parents
`constv2-mix-full__{psr_ce_all,lora_r4_ce}`, `constv2ctl2-full__{psr_ce_all,lora_r4_ce}`; derived
`constv2-mix-full__lora_r4_ce_minus_ctl`, `constv2-mix-full__psr_ce_all_minus_ctl_distilled`); the seed lives in the
key, never in the cell id. `cluster` ∈ `artifact_locations.CLUSTERS`; a path is the absolute path on that cluster
(`$SHARED` its shared root), no host prefix. Timestamps ISO-8601 UTC; every sha a full 64-hex sha256.

## 1. `RS/fits_ledger.jsonl` — one JSON object per line, appended per validation

| key | type | meaning |
|---|---|---|
| `cell`, `key`, `seed` | str, str, int | the parent cell id, the fit key, `s` (== `train_budget_meta.seed`) |
| `bank` | str | the sbatch's bank positional; `reseed` for the staged fits (a gate / working cell carries its own and is skipped) |
| `cluster`, `bank_dir` | str | where the fit landed; `$SHARED/exp/objects/sdfbase/<bank>/<key>/cells_all/<cell>` there |
| `object_sha256` | str | sha256 of `object.pt` (== `COMPLETE.object_sha256` == `train_budget_meta.object_sha256`) |
| `fit_fingerprint` | str | `COMPLETE.fit_fingerprint` |
| `validated` | bool | every hard check passed; a cap-hit is a flag, never a failure |
| `verdicts` | {str: ok\|fail\|flag} | `realized_batch`, `fit_fingerprint`, `effective_batch_64`, `seed`, `no_continued_from`, `bs_x_accum`, `stopped_early`, `ce_ratio` |
| `flags` | [str] | e.g. a cap-hit note |
| `fit_record` | dict | `{best_epoch, epochs_run, stopped_early, best_heldout_ce, resumed_from_step, optimizer_steps_run}` |
| `validated_at` | str | when the row was written |

Readers take the last line per `(key, cell)` and refuse when two lines disagree on `object_sha256`. The stager stages a
row only when `validated` is true and `bank == "reseed"`.

## 2. `RS/artifact_locations/<key>/<cell>/<cluster>.json` — immutable per-record files

Written by `artifact_locations.py write` (`tempfile` + `os.replace`); identical content is a no-op, different content
refuses. One file per `(key, cell, cluster)`; the aggregate is derived on read (`artifact_locations.py view [--key K]`).

| key | type | meaning |
|---|---|---|
| `key`, `cell`, `cluster` | str | the fit key (the directory name); the cell id, a derived object's derived id; the cluster described |
| `kind` | str | `fit` \| `derived` (`write --kind`). Required by the stager; `view` warns per record without one, never refuses. The identical `write` with `--kind` adds it in place (`kind_added`); a different kind refuses; a write without `--kind` neither compares nor touches it |
| `bank_dir` | str | fits `$SHARED/exp/objects/sdfbase/reseed/<key>/cells_all/<cell>`; derived `$SHARED/exp/objects/sdfbase/reseed/derived/<lora_subtraction\|psr_residual>/<key>/<cell>` |
| `object_sha256` | str | sha256 of `object.pt` there |
| `fit_fingerprint` | str \| null | fits: 64 hex, required; derived: null (`--fit-fingerprint` omitted or `-`); null on a fit or kind-less record is refused on read and write |
| `copied_from` | str \| null | the cluster a colocate mirror was copied from (`colocate_cells.sh` propagates the source record's `kind`, `fit` when it has none); null for the original |
| `written_at`, `verified_at` | str \| null | when written; when a colocate copy was hash-verified on both sides |

The stager asserts `object_sha256` of every record of a cell against the ledger / admission value and reads the cell's
source from the record whose `cluster` equals the ledger row's (fits) or the admission's (derived); none is a refusal.

## 3. `RS/admission/<key>__<cell>.json` — `<derived_dir>/ADMISSION.json` plus an envelope

`derived_admission.py` writes `ADMISSION.json` on the derivation cluster and the same body here (`admitted`, `reasons`,
`parents{pos,neg: {cell_id, object_sha256, fit_fingerprint, seed, key}}`, `recovery_verdict`, `gate_liveness_min`,
`selected_lr_mult`, `sweep`, `checked_at`) plus the envelope `key`, `cell` (the derived id), `seed`, `kind`
(`lora_subtraction` \| `psr_residual`), `cluster` (the derivation cluster), `derived_dir` (== the §2 `bank_dir`),
`object_sha256`, `derived_meta_sha256`. A not-admitted object has a file too (`admitted: false` + `reasons`); the stager
lists it as excluded.

## 4. What the stager derives

Staged ids `base64s<s>__<cell>` (fits) and `derived_base64s<s>__<cell>` (derived), target `qwen35_4b`, view `s<s>`, under
`$SHARED/exp/objects/sdfbase/full22/qwen35_4b/cells/<staged_id>/`, each with a `staged_from.json` naming its source
cluster and directory, `object_sha256`, the admission (derived) and every §1–§3 record behind it. The staging manifest
feeds `make_full22_configs.py --pins-from`; `stage_tools.roster_move` then moves every pinned `pending_fit` id of the
`s<s>` views into `cells` and the view configs are regenerated.
