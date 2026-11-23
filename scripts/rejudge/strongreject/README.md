# The StrongREJECT track

This track re-judges the paper's StrongREJECT readings (the four single-turn attacks, best-of-N and the three multi-turn attacks)
with MiniMax-M3. The pipeline is described in `scripts/rejudge/README.md`; this page covers what is particular to this track.

## The judge

The judge text never changes. The rubric (`STRONGREJECT_PROMPT` for v1, `_V2` for v2), the `StrongREJECTRating` tool and the score
formula are the Haiku 4.5 judge's own, imported from `scripts/judge_batch.py`, and each request's `custom_id` is that judge's cache key
(`grading_live._live_cache_key`), so one request is sent per distinct judge input.

Everything that belongs to the model is one entry in `judges.py` (`minimax-m3`, the default):

- **Model**: `minimax/minimax-m3:batch`, served by DeepInfra fp8, its only batch provider.
- **The call** is forced as the Haiku judge's was. DeepInfra does not honour a named tool choice, so about 5% of first replies answer
  in text instead; those are parse errors and are re-asked up to twice.
- **Reasoning** is off (`{"enabled": false}`), as the Haiku judge ran without extended thinking.
- **No cache fields and no temperature** are sent. DeepInfra caches the shared prefix by itself.
- **`max_tokens`** is 1500, the Haiku judge's cap. A cut reply is re-asked once at 4000.
- **Token counts**: `build_manifests.py --tokenizer` takes the model's `tokenizer.json` (its sha256 is pinned), so the dry run counts
  prompt tokens as the provider bills them and checks the 524,288-token context.

Every stage reads the entry: `build_manifests.py --judge minimax-m3` writes it into `track.json` and builds every body from it
(`request.request_body`, which requires every request setting), and `file_records.py` files under its `judge_model` with
`protocol_sha256` over the body template it produces. Two judge rows, one per rubric, carry the Haiku judges' rubric hashes.

## Steps

| step | command | output |
|---|---|---|
| goals | `goals.py --out SCOPE/goals.json` | the 259 goals' (dataset, revision, text), built from the pinned HarmBench and JBB revisions; the paper's file has sha256 `ef124cbd...` (`tests/test_rejudge_strongreject_goals.py`) |
| scope | `scope_runs.py --runs RUNS.tsv --goals SCOPE/goals.json --out SCOPE` (an explicit run list), or `scope.py` (the paper rows, from maps of each Claude judgment's stores) | `jobs.jsonl` |
| units and texts | `enumerate_units.py --jobs SCOPE/jobs.jsonl --goals SCOPE/goals.json --site SITE --out ENUM/SITE` on each site holding stores | `units.tsv.gz`, `texts.jsonl.gz`, `report.json` |
| manifests | `build_manifests.py --enum ENUM/* --goals SCOPE/goals.json --run-dir RUN --judge minimax-m3 (--require-file PATH \| --no-submit-gate)` | the run directory and `dry_run.{json,md}` |
| judge | `../common/harvester.py --run-dir RUN`, and for keys sent live `live.py init` then `live.py run` | the ledgers |
| file | `file_records.py --jobs SCOPE/jobs.jsonl --units RUN/units.tsv.gz --run-dir RUN [--run-dir LIVE] --staging STAGE --registry-root ROOT --dry-run`, then `--deliver judge --deliver-only`, a registry cycle, and `--deliver runs --deliver-only` | records in the inbox |

`--units` is always the FULL unit map, so a run judged only in part files a `partial` judgment. Item rows per unit: `strongreject` with
the score; `strongreject_unscored` with the outcome as label for an item outcome (moderation, judge refusal, parse error, truncation);
and `strongreject_best` per (goal, trial) whose units all have verdicts. A provider failure is not filed as a row unless
`--file-infra-outcomes` says so.

`build_manifests.py --reuse-run-dir DIR` carries a same-protocol run directory's terminal verdicts instead of re-sending them (a key is
carried only when its body is byte-identical to the one that directory sent); `track.json` `reuse_run_dirs` records it, and the filing
refuses unless it is passed as a `--run-dir` too.

## The live slot (`live.py`)

`live.py` judges a batch run directory's never-submitted keys on chat completions with the same bodies (the engine is
`../common/live.py`). The routing allows only the fp8 endpoints DeepInfra, GMICloud, StreamLake and Parasail; Parasail's fast tier,
CoreWeave (fp4), Venice and MiniMax's own endpoint are ignored, and `require_parameters` is off because no fp8 endpoint honours a
forced tool choice (exactly as the batch endpoint). The routing is part of the judge's identity: filed beside the batch run directory,
the live keys give the `batch_or_live` judge rows the paper reads (`192df392...` v1, `d6c07d23...` v2). `init` requires the run's
ceiling, the global ceiling and the root it counts.
