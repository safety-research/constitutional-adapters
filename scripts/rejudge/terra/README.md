# The C/P/I track (OR-Bench hard and XSTest)

This track re-judges the paper's over-refusal readings, the Sonnet 5 C/P/I labels of OR-Bench hard (`or_bench_cpi`) and XSTest
(`xstest_cpi`), with MiniMax-M3. The pipeline is described in `scripts/rejudge/README.md`; this page covers what is particular to this
track. Its modules also carry the agentic-misalignment classifier family, which the paper does not report.

## The judge

Every body starts from the original judge's own request: `cpi.original_request` renders it with `capeval.judging.render_request` (the
frozen rubric, the `compliance_rating` tool, the default protocol) and recomputes the store's request key from the text, then
`cpi.translate` turns the Anthropic params into chat completions (system framing as the first message, the forced tool call as a
function call). A key that does not recompute is refused: the text is not the text the judge saw.

`judges.py` holds the model's entry (`minimax-m3`, the default):

- **Model**: `minimax/minimax-m3:batch`, served by DeepInfra fp8, its only batch provider.
- **The call** is forced (`tool_choice` naming `compliance_rating`); DeepInfra treats it as `auto`, so a reply without a call is a parse
  error, re-asked once. `parallel_tool_calls` is not sent (the batch endpoint does not list it).
- **Reasoning** is off (`{"enabled": false}`), mirroring the Sonnet judge's disabled thinking; no sampling parameter and no cache fields.
- **`max_tokens`** is 1024, the Sonnet judge's cap; a cut call is terminal, as the Sonnet judge abstained on one.
- A request whose estimated prompt (times 1.35) plus its largest cap could exceed the 524,288-token context is never packed
  (`over_context.txt`).

Item rows: `orb_label` / `xst_label` (2, 1, 0 for full compliance, partial refusal, full refusal), and an `*_unscored` row with the
outcome as label for an item outcome. No row carries an eval's accepted headline metric id (`file_records.HEADLINE_IDS_NEVER_FILED`), so
a second judge never feeds the item readers or the contrast selection.

## Steps

| step | command | output |
|---|---|---|
| scope | `scope_cpi_runs.py --runs RUNS.tsv --out SCOPE` (an explicit run list: `run_id`, `eval_id`, `cell`, `cap`), or `scope_cpi.py --out SCOPE [--xstest-subset safe\|all] [--runs FILE]` (the paper rows) | `units.tsv.gz`, `texts.jsonl.gz`, `runs.jsonl`, `report.json` |
| manifests | `build_manifests.py --judge minimax-m3 --family cpi --scope SCOPE --run-dir TRACK/full/cpi --require-file TRACK/GO --max-spend-usd USD` | the run directory and `dry_run.{json,md}` |
| judge | `harvest.py --run-dir TRACK/full/cpi` (or `harvest_all.py` over several run directories), and for keys sent live `live.py` | the ledgers |
| file | `file_records.py --scope-cpi SCOPE --run-dir TRACK/full/cpi [--run-dir TRACK/live/cpi] --staging STAGE --registry-root ROOT --dry-run`, then the judge and the runs delivered (`--deliver judge --deliver-only`, a registry cycle, `--deliver runs --deliver-only --accept-run-repoint`) | records in the inbox |

`harvest.py` is the shared harvester with three additions: a request refused on the provider account pauses the run, re-asks are
counted within an outcome's class, and a batch whose moderation share is far above the original judge's baseline is held until a person
acknowledges it (`moderation_share` in the run directory's `ACKNOWLEDGED`). `reuse.py` carries a finished same-protocol run's verdicts
into a full run (bodies byte-identical, attempts the full run's policy).

The XSTest contrast half (`--xstest-subset all`) is its own run directory, `TRACK/full/option_xstest_contrast`. The strength records'
per-item reads (`scope_cpi_reads.py`) carry the key of a top-rung unit already filed, so that scope is marked `not_for_filing` and
`reads_sidecar.py` writes its verdicts as a table instead.

## The live slot (`live.py`)

`live.py` judges the track's never-submitted batch keys on chat completions with the same bodies, one live run directory per batch run
directory under `TRACK/live/` (the engine is `../common/live.py`; the module docstring is the contract):

| | |
|---|---|
| key set | every key of the source's pending batches; the source is read only |
| batch gate | while `TRACK/GO` exists the batch harvester may POST those batches, so an `init` then is `STAGED` (dry run only); remove `GO`, wait for the harvester's next pass, then `init --refresh` |
| routing | the fp8 endpoints that declare every parameter the bodies carry: for C/P/I bodies DeepInfra, GMICloud, StreamLake and Parasail; `require_parameters` off, `max_price` at the dearest allowed endpoint |
| policy | the track's parsers and re-asks; the moderation-share hold per recorded window |
| reasoning | a reply billing reasoning tokens is held and re-sent; past three re-sends a key goes to the endpoints that have not billed it, and is parked when none is left (`settle-parked --apply` leaves it unjudged) |
| spend | `init` requires each run's `--max-spend-usd`, `--global-root` and `--global-ceiling-usd`; every engine stops at either ceiling |

Filed with its source, a live run directory gives the `batch_or_live` judge rows the paper reads (`0a4bfa49...` OR-Bench hard,
`5e79a378...` XSTest); the routing is part of their identity.
