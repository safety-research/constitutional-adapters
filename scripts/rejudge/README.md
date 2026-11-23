# Re-judging runs with the paper's judges

`scripts/rejudge/` re-judges already-generated eval transcripts with a different judge model: the generations stay fixed, only the
judge changes. It is the code that produced the paper's judged readings: MiniMax-M3 for StrongREJECT (single-turn, best-of-N and
multi-turn) and for the OR-Bench hard and XSTest compliance labels (C/P/I), and GLM-5.3 for Petri. Every request goes through
OpenRouter, on its Batch API or, for keys a batch provider could not serve in time, one by one on chat completions with the same
request body. Every verdict reaches the results registry as record files delivered to the harvester's inbox, as a NEW judge beside the
original Claude judge of each run, never replacing it.

`paper/notebooks/data/paper_judges.json` names the judge rows the paper reads. `tests/test_rejudge_paper_judges.py` recomputes each of
them from this code (judge id, rubric and protocol hashes, notes and settings) and compares them with the committed paper frame's judge
rows, and `tests/test_rejudge_request_goldens.py` pins the request bodies byte for byte. The Petri rubric hash is computed from the six
answer tools the Opus judge was sent (`tests/fixtures/rejudge/petri_answer_tools.json`) and the judge template of the pinned
`inspect-petri`, so that one check runs only with `--group petri` installed.

## What a reader can and cannot reproduce

- You can verify every paper judge's identity and request bodies from the code (the two tests above), and run the whole pipeline on
  synthetic data with no key: the end-to-end tests drive the scope, the manifests, a local mock of OpenRouter, the filing and the
  delivery into a temporary registry.
- You can re-judge runs you generated yourself (the worked example of `reproduce/README.md` registers them), with the commands below.
- You cannot regenerate the paper's own judgments byte for byte. The judged texts (the run stores) are not distributed: the repository
  distributes no model responses to harmful requests, and the paper frame holds verdicts and labels, not texts. The judges also run
  at their providers' default temperature, so a re-run reproduces the procedure and the judge identity, not identical verdicts.
- Providers, prices and OpenRouter's endpoints change. The live routing is part of the judge's identity (below), and a reply from a
  provider the routing does not allow is held, never filed.

## Requirements

- The repository and its locked environment: `uv sync --frozen --extra cpu`, then run everything from the repository root with
  `PYTHONPATH=$PWD/src:$PWD/scripts`. The Petri track also needs `--group petri` (`inspect-ai`, `inspect-petri` 3.0.11 and
  `petri-bloom` 0.2.6: the enumeration reads inspect logs, and the Petri judge's rubric hash reads the pinned `inspect-petri`). Without
  `tiktoken` (it arrives with that group) the dry runs estimate tokens as characters / 4. The live slot uses `aiohttp`, which the base
  environment carries.
- A results registry (`~/registry`, `reproduce/00_setup.sh`) holding the runs and their original judgments, read only until delivery.
- The runs' stores: static cells (`shard*/responses.jsonl`), multi-turn cells (with `progress.jsonl`), best-of-N `bon_responses.jsonl`;
  C/P/I cells (`private/gens_<task>.jsonl`, `judge_items.jsonl`); Petri inspect `.eval` logs.
- The Hugging Face cache with the pinned HarmBench and JBB revisions (the StrongREJECT goals) and the XSTest and OR-Bench pins
  (`capeval.corpora.load_pins`); `HF_HUB_OFFLINE=1` reads the cache only.

## Credentials

The key is read from `$OPENROUTER_API_KEY`, in the environment of the process that sends requests, and is never printed, logged,
written to a file or put on a command line; every text recorded from a reply or an exception has it redacted first. Only the batch
harvester (`common/harvester.py` and the C/P/I `terra/harvest.py`, `terra/harvest_all.py`) and the live engines (`run`, `run-all`)
read it. Scopes, enumerations, manifests, dry runs, filing and delivery need no key. The code never reads or sets an `ANTHROPIC_*`
variable and calls no Claude model. A long harvester or engine belongs in a process that outlives the shell that started it (a
service manager's unit, a terminal multiplexer), with the key in that process's environment only.

## Layout

```
scripts/rejudge/
  common/          the shared library: OpenRouter client, manifests, ledger, submitter, harvester, live engine, ledger merge, filing
  strongreject/    the StrongREJECT track (judge MiniMax-M3): goals, scopes, enumeration, manifests, live entry point, filing
  terra/           the C/P/I track (judge MiniMax-M3; OR-Bench hard and XSTest): scopes, manifests, harvester, live entry point, filing
  petri/           the Petri track (judge GLM-5.3): transcript matching, scope, inputs, manifests, filing
```

The `terra/` modules also carry the agentic-misalignment classifier family, because the track's shared modules dispatch per family.
The paper reports no agentic eval, and that family is not documented here.

## Regenerating a judgment

A StrongREJECT example, for runs you generated. Every step but 5 is offline, and nothing is delivered until step 7.

1. The goal table, from the pinned benchmark revisions in the HF cache:
   `python scripts/rejudge/strongreject/goals.py --out SCOPE/goals.json`
2. The scope from an explicit run list (a TSV with `run_id`, `eval_id`, `store`; `store` may be `@judgment`, the store the run's
   Claude judgment graded):
   `python scripts/rejudge/strongreject/scope_runs.py --runs RUNS.tsv --goals SCOPE/goals.json --out SCOPE`
3. The judge units and texts, where the stores are:
   `python scripts/rejudge/strongreject/enumerate_units.py --jobs SCOPE/jobs.jsonl --goals SCOPE/goals.json --site local --out ENUM/local`
4. The run directory, priced (manifests, bodies and `dry_run.{json,md}`; nothing is sent):
   `python scripts/rejudge/strongreject/build_manifests.py --enum ENUM/local --goals SCOPE/goals.json --run-dir ROOT/full --judge minimax-m3 --no-submit-gate --max-spend-usd USD`
5. The judging, with the key in the environment: `python scripts/rejudge/common/harvester.py --run-dir ROOT/full` (the Batch API). To
   send keys live instead, build step 4 with `--require-file ROOT/NEVER_SUBMIT` (a file that is never created), register its
   manifests with `harvester.py --run-dir ROOT/full --dry-run`, then
   `python scripts/rejudge/strongreject/live.py init --live-dir ROOT/live --source-run-dir ROOT/full --global-root ROOT --global-ceiling-usd USD --max-spend-usd USD`
   and `live.py run --live-dir ROOT/live`
6. The filing, staged and validated (prints the judge ids):
   `python scripts/rejudge/strongreject/file_records.py --jobs SCOPE/jobs.jsonl --units ROOT/full/units.tsv.gz --run-dir ROOT/full [--run-dir ROOT/live] --staging STAGE --registry-root ~/registry --dry-run`
7. The delivery: the same with `--deliver judge --deliver-only`, a cycle of the registry's harvester (`python -m sisteer.registry.cli
   ingest`), then `--deliver runs --deliver-only`.
   `reproduce/10_figures.sh --source live` then reads the new judgments under `REGISTRY_NB_JUDGES=rejudge,strongreject=<judge id>`
   (`paper/notebooks/README.md`).

The judge id depends on the route (below): a batch run directory filed beside the live one gives the paper's `batch_or_live` judge
(`192df392...` for v1, `d6c07d23...` for v2), a batch-only or live-only filing a judge of its own.

The C/P/I track follows the same steps: `terra/scope_cpi_runs.py --runs RUNS.tsv --out SCOPE` (`run_id`, `eval_id`, `cell`, `cap`), then
`terra/build_manifests.py --judge minimax-m3 --family cpi --scope SCOPE --run-dir TRACK/full/cpi --require-file TRACK/GO`, the batch
harvester `terra/harvest.py --run-dir TRACK/full/cpi` (or the live slot, `terra/live.py`), and `terra/file_records.py --scope-cpi SCOPE
--run-dir ... --staging STAGE --registry-root ~/registry --dry-run`. The Petri track: `petri/match_items.py --logs LOGS --out MATCHED`,
`petri/scope.py --matched MATCHED --out SCOPE`, `petri/enumerate_inputs.py --units SCOPE/units.tsv.gz --out ENUM`,
`petri/build_manifests.py --judge glm-5.3 --enum ENUM --run-dir RUN --no-submit-gate --max-spend-usd USD`, the batch harvester, and
`petri/file_records.py --judge glm-5.3 --scope SCOPE --enum ENUM --run-dir RUN --staging STAGE --registry-root ~/registry --dry-run`
(batch only; every verdict must be served by DeepInfra). Each track's README has the details.

The paper's own scopes are the paper rows (`v_run_current.in_current_paper = 1`): `strongreject/scope.py` (from maps of each Claude
judgment's stores), `terra/scope_cpi.py` and `petri/scope.py`, each with `--runs FILE` for a supplement (the contrast anchors).

## Judge identity

`judge_id = identity.judge_id(judge_model, rubric_sha256, protocol_sha256, threshold_rule)`:

- the rubric is the original judge's, imported, never copied: StrongREJECT from `scripts/judge_batch.py`, C/P/I from
  `src/sisteer/capeval/judging.py`, Petri from the stored judge requests (the `inspect-petri` template and the six answer tools, which
  `tests/fixtures/rejudge/petri_answer_tools.json` holds)
- the protocol hashes the request template (endpoint, model, message layout, tool and tool choice, reasoning, cache fields; no texts and
  no `max_tokens`, so a re-ask at a raised cap is the same judge) and, when a live run directory files, the live route: the provider
  routing and each allowed endpoint's slug and quantization, and whether batch run directories file beside it (`batch_or_live`)
- the threshold rule is the original judge's

`common/filing.py` `MINIMAX_M3_JUDGE_IDS` lists the four MiniMax-M3 rows on record; `--expect-pinned-judges` (both filers) refuses a
filing that would stage under any other judge or stage a judge record. A provider, model, price or reasoning reply outside the routing
breaks the live protocol: it is held and never filed (`live.py reopen --off-protocol` re-opens filed outcomes that no longer fit).

## The paper's judgments by revision

The registry records each judgment's code revision (`judgment.grading_code_sha`, also in the paper frame). The paper's re-judges were
filed from these development revisions, which are not part of this repository's published history; the code here is their published
form and files the same records (the judge-row and request-body tests above pin it):

| revision | judges | runs |
|---|---|---|
| `dad98c58` | StrongREJECT v1 and v2, OR-Bench hard, XSTest | the paper rows |
| `9f6e9c4a` | Petri | the paper rows |
| `c2a6065b` | Petri | a pooled run whose Claude judgment filed no item rows, and its anchor (units composed from the behaviour runs, `petri/pooled_scope.py`) |
| `306912c8` | StrongREJECT v1 | contrast anchors |
| `c18e8f4a` | OR-Bench hard, XSTest | contrast and frontier anchors; the strength records' per-item reads (`terra/reads_sidecar.py`) |
| `c9384f57`, `e9e1cdb7` | StrongREJECT v1 (and v2, OR-Bench hard, XSTest for `c9384f57`) | explicit run lists: control-pool, generator-comparison, distillation-target and regenerated runs |

## Limitations

- The StrongREJECT paper-row scope (`strongreject/scope.py`) joins maps of each Claude judgment's stores, which the repository does
  not build; for a run the registry ties to its store, `scope_runs.py` with `store=@judgment` builds its jobs from the registry alone,
  except a best-of-N reading folded from a 512-token store and its climb, which it refuses.
- Petri needs the runs' inspect logs and the pinned `inspect-petri` (another version refuses, since the rubric hash reads its template).
- Spend control is the run's: every live run states its own ceiling, the global ceiling and the root it counts (no defaults), and a
  batch run directory takes `--max-spend-usd`.

## Reference

### The run directory

One directory per track and scope. Nothing in it is rewritten in place: a manifest and its body are immutable once written, and the
ledger is the state. It holds the judge's raw replies, which can quote the judged text: keep it private (mode 0700) and outside the
repository.

| path | holds |
|---|---|
| `track.json` | track, model, endpoint, provider, parser (`module.path:function`), re-asks, size caps, ramp, health, request settings, `submit_gate`, `max_spend_usd` |
| `manifests/<key>.json` | `rejudge-batch-manifest/1`: request count, body size and sha256, custom ids, token and cost estimates, attempt, parent batch |
| `bodies/<key>.json.gz` | the exact bytes POSTed; a body whose sha256 is not its manifest's is refused |
| `state.sqlite` | the ledger: batches, requests (the verdict cache), every attempt's outcome, events (one writer, a lock file) |
| `raw/<key>__<batch id>.json.gz` | every finished batch as fetched, results included |
| `progress.json` | batch states, outcome counts, spend, pause reason, readiness |
| `PAUSED` | present while the run waits for a person (an account or credential failure, a refused submit, an error flood, the spend ceiling, the submit gate); remove it after the fix to resume |
| `ACKNOWLEDGED` | error signatures (one per line) accepted as real per-item outcomes, so a flood of them is recorded instead of held |

### Harvesting

`common/harvester.py --run-dir DIR [--once] [--poll-s 60] [--max-hours 72] [--no-submit] [--dry-run]` submits every pending batch up
front (the Batch API runs them side by side), polls, fetches each finished batch once, parses every result with the track's parser and
re-asks within the track's budget. Outcomes:

| outcome | meaning | default re-asks |
|---|---|---:|
| `verdict` | a valid judgment | 0 |
| `moderation` | the gateway or provider refused this input, on positive evidence; recorded per item | 0 |
| `judge_refusal` | the model declined | 0 |
| `parse_error` | malformed or schema-invalid output | 1 |
| `truncated` | cut at `max_tokens`; re-asked only at the raised cap `truncated_retry_max_tokens` | 1 |
| `provider_error`, `missing`, `batch_failed` / `batch_expired` / `batch_cancelled` | the provider or the batch failed, not the item | 3 |
| `account_error`, `batch_failed_auth`, `balance_error`, `held` | the account, the credential or the money, or an identical-error flood: never terminal, the run pauses or backs off | none |

A 413 splits a batch at the cap its message states, a 429 backs off, a 402 queues the batch for the balance, a POST that timed out is
reconciled against the account's batches before anything is re-sent, and the spend ceiling and submit gate are read before every POST.
Exit codes: 0 done (every request terminal), 4 paused, 5 unfinished (a batch set aside or a request with no terminal outcome), 3 at
`--max-hours`.

### The live slot

`common/live.py` (entry points `strongreject/live.py` and `terra/live.py`) judges a batch run directory's never-submitted keys on chat
completions with the same bodies, into a live run directory whose ledger the filing reads like a batch ledger. Its module docstring is
the contract: `init` (the key set, read only on the source), `run` (the engine: adaptive concurrency, transport retries, the
harvester's parser and re-asks, a health check per window), `handover` (keys a stopped batch run will not finish), `reopen`,
`settle-parked`, `dry-run`. A reply is on protocol when its provider is one the routing allows, its model the one asked for, its cost
within its provider's list price and it bills no reasoning tokens; anything else is held, never filed. The engine stops at its own
ceiling and at the global ceiling over every ledger under its root, finishing the requests in flight.

### Filing

Each track's `file_records.py` stages one judge record, and per run one judgment and its item rows, under
`records/rejudge/<track>__<judge_id[:12]>/`, validates every record with the registry's validator, and delivers into
`<registry root>/inbox/rejudge/` (judge first, runs after it has folded). Delivery is resumable and refuses a name that stands with
other bytes. Runs are delivered only when every run directory has stopped moving; a run that would file `partial` needs
`--allow-partial`. Staging with `--registry-root` makes a later filing of the same run a follow-up whose files number above the
registry's. A filing never writes `run_state`, `run_patch`, `run_metric` or `run_contrast`, and never supersedes another judge's rows;
`v_judge_side_by_side` pairs the new judge's rows with the original's. `common/scratch_proof.py` digests every table and paper view of a
scratch copy of the registry before and after a filing, to prove it touches nothing else.

An explicit run list (`strongreject/scope_runs.py`, `terra/scope_cpi_runs.py`) holds each listed store to the one the registry ties to
the run, and files under the judge rows on record with `--expect-pinned-judges --ids-out IDS`.
