# The results registry

The results registry is the database every number in the paper is read from: one SQLite file, rebuilt from an
append-only log of JSON records, indexing models, intervention objects, evaluations, instruments, judges, runs,
judgments and measurements, plus the roster and rulings that decide which runs each figure reads.

Most readers need only the committed snapshot, [`paper/snapshot/`](../../paper/snapshot/): the figure notebook
in [`paper/notebooks/`](../../paper/notebooks/README.md) draws every figure from it on a CPU
(`reproduce/10_figures.sh --source snapshot` from a clone). A live registry is for re-running evaluations
through [`reproduce/`](../../reproduce/README.md) with new numbers beside the paper's.

## Concepts

Models. One row per checkpoint (`model_id`): `qwen35_4b_it` is the served Qwen/Qwen3.5-4B and `qwen35_4b_base`
the base checkpoint the base-fit objects were trained on; `gemma4e4b_*` and `llama8b_*` pair the same way;
`nemotron3nano4b` is a single chat checkpoint.

Objects. One row per intervention (`object_id`, the sha256 of the object file; `none` and the `si:<sha>` prompted
objects are virtual), named by `paper_label`: the LoRA family `control_lora`, `constitution_lora`,
`difference_lora` (constitution minus control: the paper's constitutional adapter); the ReLU steering-vector
family `control_relu`, `constitution_relu`, `difference_relu` (the paired constitution-minus-control write);
`arditi_allpos`, the refusal-direction baseline; `safety_si` and `constitution_si`, the prompted baselines;
`none`, the unsteered model. `in_current_paper = 1` selects the paper's runs, but a few `arditi`,
`safety_lora` and `safety_relu` rows carry the flag too, so also filter `paper_label` to these ten labels.

Runs. A run is one model × object × eval × instrument measurement (`run_id`); evals and instruments are
tabulated in [`configs/registry/README.md`](../../configs/registry/README.md). `transfer_kind` says where the
object was fit relative to where it is served: `base_to_instruct` (fit on the base checkpoint, served zero-shot
on the instruct one), `instruct_direct`, `chat_checkpoint` (Nemotron); the prompted baselines are
`prompted_only` and the unsteered model `none`. `strength_nominal` is the inference-time scaling coefficient
relative to the operating point: 1.0 is the object as fit, 0.75 the paper's companion dose; for the refusal
steer 1.0 is its raw coefficient 0.3 (`strength_coefficient`). Its numbers are in `run_metric` (`value`, `k`,
`n`, `ci_lo`, `ci_hi`), per-item scores in `item_result`, its `delta` from an unsteered anchor
(`anchor_run_id`, `anchor_kind`) in `run_contrast`.

Roster and rulings. The `r2` roster (version `v11`) says which model × object × eval × strength enter which
figure block (e.g. `fig1_headline/instruct_direct`); with the rulings on top it derives `in_current_paper` and
each run's `selection_state`.

## Reading the paper's data from a clone

`paper/snapshot/CURRENT` names one snapshot under `paper/snapshot/snapshots/`: the tables of the paper frame
(the subset of the registry the paper's figures read, scope `paper_frame`) as Parquet, four SQL views
materialised, the frame as a gzip-split SQLite database, and a `MANIFEST.json` of sha256s and materialised
columns (`materialised_types`). The two item-scale tables (`item_result`, `item_flags`) ship in the SQLite only
(`MANIFEST.sqlite_only_tables`). All 11,288 runs are on the four served checkpoints; 2,989 are in the paper's
selection, 2,819 of those with per-item rows. The frame also carries the per-item rows of the paper's judges
(`paper/notebooks/data/paper_judges.json`; the frame lists the judges it carries in `MANIFEST.item_judges`) for every run of the
four models, with their `extra_json` cut to the keys the notebook reads (`MANIFEST.item_projection`). The cut also took the item
rows of the `minimax-m3` agentic-misalignment judge, which the paper's policy listed then and no longer does (the paper omits
those evaluations); the release pass leaves that judge's item rows out, and `MANIFEST.release_omitted_item_judges` names it. It
was cut from the registry with
`reproduce/tools/snapshot_paper_frame.sh --paper-models-only` (the paper's judges at the time and the reader projection were that
tool's defaults; `--paper-models-only` leaves the registry's test models out), then passed through the release pass
(`scripts/release/shipped_text.py apply`, with the release's private rule inputs), which rewrites its prose:
`MANIFEST.derived_from` names the raw cut and `release_rules_sha256` the rule inputs.

`v_paper_long` is the table the figures start from: one row per measurement, with the columns the code below uses. Its
`value`, intervals and `delta` are the readings of the filed Claude judges (the notebook's `primary` policy). The paper's
figures do not print those: they re-derive each StrongREJECT, OR-Bench, XSTest and Petri reading from the item rows of
the paper's judges (`paper/notebooks/data/paper_judges.json`; `item_result`, in the SQLite only) through
`registry_nb.select_judges`, as [`paper/notebooks/README.md`](../../paper/notebooks/README.md) describes. The snapshot
also carries `v_paper_current` (one row per run, with its status), `v_paper_objects` (the objects) and
`v_roster_match` (which runs match the paper's roster). It was cut at schema version 33
(`MANIFEST.source_schema_version`).

The snapshot (about 370 MB) is plain committed files; reading the Parquet needs only pandas and pyarrow (core
dependencies). From the repository root:

```python
import pandas as pd
snap = "paper/snapshot/snapshots/" + open("paper/snapshot/CURRENT").read().strip()
df = pd.read_parquet(f"{snap}/v_paper_long.parquet")
asr = df[(df.in_current_paper == 1) & (df.is_complete == 1) & (df.metric_id == "asr") & (df.threshold == 0.5)]
# one row per reading (a reading repeats per anchor; rules below)
asr = asr.drop_duplicates(["run_id", "judgment_id", "metric_id", "threshold", "computation_id"])
print(asr[["paper_label", "eval_id", "value", "delta"]])   # the filed Claude judge's readings, not the paper's
```

For SQL, `sisteer.registry.snapshot_open` reassembles the SQLite database (about 4 GB) under the git-ignored
`paper/snapshot/_reassembled/`, read-only; it needs the project's environment
(`uv sync --frozen --extra cpu --group notebook`, as `reproduce/00_setup.sh` installs it) or `PYTHONPATH=src`:

```python
from sisteer import registry
with registry.snapshot_open("paper/snapshot", scope="paper_frame") as snap:
    rows = snap.conn.execute("SELECT paper_label, eval_id, value FROM v_paper_long WHERE in_current_paper = 1").fetchall()
```

Rules that change a number's meaning:

- Filter `in_current_paper = 1` for the paper's selection and `is_complete = 1` for finished runs.
- A run × metric has a row per threshold and computation: `asr` is filed at 0.25, 0.5 and 0.75 (the paper reads
  0.5); metrics without a threshold carry `threshold = -1.0`.
- A measurement can repeat per current anchor (`anchor_kind`: `same_standup_none`, `sibling_none`, `borrowed`,
  `full_dose`, `solo_stand_in`), same `value`, different `delta` (21 in-paper `asr` keys at 0.5);
  pick one `anchor_kind` or drop duplicates on (`run_id`, `judgment_id`, `metric_id`, `threshold`,
  `computation_id`) before averaging.
- A `delta` is valid only beside `cap_matched = 1` (steered run and anchor decoded under the same token cap).
- A run's figure blocks are listed in `comparison_ids_json` (a JSON list; `v_roster_match` has one row per
  matched roster entry with its `comparison_id`); names such as `fig1_headline/instruct_direct` are the
  roster's, not the paper's figure numbers.
- The Qwen figures average three training-seed fits of each fitted object (`fit_replicate_label` `s1`, `s2`,
  `s3`); the unsteered model, prompted system instructions and refusal steer have one fit each and pool decode
  draws instead.
- `asr` is the share of items the StrongREJECT judge scored above 0.5 (in `v_paper_long` the filed judge,
  claude-haiku-4-5; in the paper's figures minimax-m3, re-derived from its item rows); `xst_refused` and
  `orb_refused` are refusal rates (the notebook plots `over_refusal_label_mean`: full refusal 1, partial 0.5,
  compliance 0); `acc` and `prompt_level_strict` are accuracies. All are 0 to 1 fractions except `petri_score`
  (1 to 10) and `misaligned_rate`, the percentage of agentic scenario samples with a misaligned action (`34.51`
  for `k = 176` of `n = 510`).
- `ci_lo` / `ci_hi` are the interval the writer filed, labelled by `ci_method` (`wilson_95`, `wilson`, bootstrap
  labels; NULL on 2,339 in-paper rows of the snapshot): read it before pooling intervals. A live registry past
  migration 0019 also derives the 95 % Wilson interval from `k` and `n` on proportion metrics (`asr`, `acc`,
  `xst_refused`, `orb_refused`) where none was filed, as `wilson_95`; `over_refusal_label_mean` and
  `misaligned_rate` carry only the writer's.

## Operating a live registry

Every `registry` command takes `--root` (default `$REGISTRY_ROOT`, else `~/registry`, created by `registry init`);
`registry query "<sql>"` runs SQL; `registry --help` lists the rest. Reproducing the paper touches these:

| task | command |
|---|---|
| create a root | `registry init` |
| status | `registry status [--html]` |
| fold the log | `registry ingest` (folds every new record file once; a live registry runs it on a timer, as `scripts/registry/install_grading_host.sh` sets up) |
| bring a database to the code's schema | `registry migrate --plan` (read-only: what differs). On an isolated database, `registry migrate` applies the pending migration files. A live registry's schema changes land through `scripts/registry/landing/window.sh`, which stops the writers, reads the plan and binds the operator's approval to it before applying (the migrations [README](../../src/sisteer/registry/migrations/README.md)) |
| seeds; roster and rulings | `registry dimensions load --seeds configs/registry`; `registry roster load --round r2 --version v11` |
| decode-policy export | `registry decode-policy export`, `registry decode-policy check` |
| snapshot; paper frame | `registry snapshot [--scope full\|runlevel]`; `registry snapshot --scope paper_frame --dest-root <dir outside the root>` |

Maintaining the registry's log and checks:

| task | command |
|---|---|
| rebuild and diff | `registry rebuild --verify --from <root> --into <fresh.sqlite>` |
| add a ruling | `registry ruling add --round r2 --action <action> --scope-kind <kind> --scope-json '<json>' --reason "<why>" --source-ref "<where>"` |
| grading queue | `registry pending-judgment [--judge J]`; `scripts/grade_bon_live.py --worklist registry` grades it |
| judgment | `registry judgment show <id>` |
| result-store adapters | `registry adapters run [--adapter NAME] [--reread]` |
| record-log mirrors | `registry mirrors sync\|verify` |
| paper-reader parity | `registry check-paper-parity --paper-checkout DIR --pin SHA` |
| token-cap ladder facts | `registry cap-check [--campaign C]` |
| legacy `run_patch` `extra` keys | `registry reconcile run-patch-extra --dry-run [--source FILE_OR_DIR]` (the overrides files a run-overrides stamp compares as a reading of) |
| run-overrides stamps that differ from their overrides file's reading | `registry reconcile env-comments --dry-run --source FILE_OR_DIR` (`--apply` in a landing window; `--verify` after its ingest; `--check-sources` before it: every stamp whose values may carry a comment has its file, else exit 3; a stamp no file covers fails the apply and the verify unless `--leave-unsourced`) |
| object `fit_effective_batch` stated as a sidecar's micro-batch | `registry reconcile fit-effective-batch --dry-run` (`--apply` in a landing window; `--verify` after its ingest) |
| whole-log validator audit | `python scripts/registry/audit_new_validator_rules.py --root <root>` |
| superseded revisions' item rows | `registry prune-superseded status` (mode, revisions due, rows by class, held rehydrations); `registry prune-superseded mode hold\|cohort\|normal`; `registry prune-superseded rehydrate` |

### Superseded revisions' item rows

An adapter that reads a source again writes a new revision (`source_revision`). A source's newest accepted
revision is its head, and only the head's item rows (`item_result`, `item_flags`, `run_degeneration`) and natively
emitted rows are read by `v_<table>_current` and every v_* object built on it. Since migration 0033 the harvester's
reclaim pass DELETES the item rows of an accepted revision a later read superseded, at the end of every cycle and at
the end of `registry rebuild` and of a replica increment, so a re-read leaves one copy of each row, not one more.

What stays: every row of a head, pending, held or refused revision; the `source_status` and `observation` rows of
every revision; and, of a superseded revision, the rows of a carrier file (`record_file_id`) that a held
re-activation staged, whose file in the log is missing or changed, or that the current code does not re-parse to
exactly the stored rows. `registry prune-superseded status` counts each class.

Where the history lives: the record log. A deleted row's record stays in its file under `records/`;
`item_reclaim_ledger` holds one row per carrier reclaim deleted from, with the count and an additive digest of the
deleted rows; it is bookkeeping, and no snapshot ships it. `registry history <table> <key-json>` folds an item key's
events from the log, as before.

Rehydration. Before anything is written under a revision whose rows reclaim deleted (a re-activation, the accept of
a held bundle, a late fresh read), the fold re-reads those carriers from the log and restores them, after checking
that the re-parse equals the rows still present plus the ledger's count and digest. When it does not (a record the
validator now refuses, a changed parser, a changed file), nothing is written: the bundle is held with the reason
`rehydration`, the previous head stays current, and the status page lists the hold. `registry prune-superseded
rehydrate --revision REV --from-archive DIR` restores the exact archived rows; `--accept-reparse --reason TEXT`
restores the current code's rows (what a rebuild holds) and records the decision. Either releases the hold.

The reclaim mode is `state/reclaim_mode.json`, written only by `registry prune-superseded mode`: `hold` deletes
nothing; `cohort` deletes, of the carriers `normal` would delete, only those bound to a locally verified archive of
their rows (a carrier kept for any reason above stays kept in every mode); `normal` (no file) deletes every carrier
that qualifies. A live cycle spends at most `[harvest] reclaim_budget_s` (600 s) on deletion,
in batches of 10,000 rows, and writes nothing while the WAL stays at or above 1 GiB after a checkpoint or the disk
has less than 30 GiB free; a carrier a stop left part-deleted resumes on the next cycle. `registry rebuild` and
replica increments always reclaim in `normal` mode, and `rebuild --verify` compares `item_reclaim_ledger` too.
`item_result.withdrawn_by_revision_id` is no longer derived: rows a reclaim before 0033 marked keep the mark, which
no reader uses and `rebuild --verify` does not compare. Deleted pages return to the file system only through
compaction (`VACUUM INTO` with the writers stopped); `auto_vacuum` is off.

## The documents

`schema.md` (tables, SQL views, columns, why-null reasons, CHECK vocabularies), `records.md` (the append-only
record contract and each type's required fields) and `vocabulary.md` (controlled vocabularies, metric ids,
legacy spellings) are generated by `registry docs <name>`. Also:
[`configs/registry/README.md`](../../configs/registry/README.md) (the seed files),
[`src/sisteer/registry/migrations/README.md`](../../src/sisteer/registry/migrations/README.md) (how a schema
change lands) and [`paper/notebooks/README.md`](../../paper/notebooks/README.md) (the figure notebook).
