# The Petri judge (GLM-5.3)

`scripts/rejudge/petri/` re-judges the paper's Petri transcripts with GLM-5.3 in place of the Opus 5 Petri judge
(`petri_bloom_judge`, judge `17e480e5`). The Petri auditor produced the transcripts and is not re-run; only the judge changes. The
pipeline is described in `scripts/rejudge/README.md`; this page covers what is particular to this track. It needs the `petri`
dependency group (`uv sync --frozen --extra cpu --group petri`).

## What the judge is sent

Exactly what the Opus 5 judge was sent, read from that judge's stored API request in each sample's inspect log
(`ModelEvent.call.request`, by `enumerate_inputs.py`): one user message (the Petri judge template, the behaviour's instructions and
scoring dimensions, the target's timeline between `[BEGIN TRANSCRIPT]` and `[END TRANSCRIPT]`, a fixed tail) and one tool, `answer`,
whose schema is the behaviour's `AlignmentAnswer`. `enumerate_inputs.py --verify-rebuild` also regenerates every prompt and schema with
the original builders; a log that stores no timeline cannot be rebuilt offline, and its stored request is what is sent.

The request (`judges.py` profile `glm-5.3`, built by `bakeoff/models.py` `openrouter_body`): the prompt as one user message, the tool as
an OpenAI function, `tool_choice` forcing `answer`, `parallel_tool_calls` not sent (the batch variant does not list it),
`reasoning: {"effort": "low"}` (reasoning is mandatory on GLM-5.3 and `low` is its floor; the Opus judge's forced call ran without
thinking), `max_tokens` 16,000 (a truncated reply is re-asked once at 32,000), no cache fields. `custom_id` is the sha256 of what the
judge reads (the prompt and the tool), one request per distinct judge input.

Every batch is pinned to DeepInfra (`{"only": ["deepinfra"], "allow_fallbacks": false}`): another provider serves another deployment of
the weights, and with fallbacks off a capacity failure fails the batch visibly (re-asked three times, then an unfiled infrastructure
outcome that a follow-up run directory of those keys settles). The filing refuses any verdict not served by DeepInfra.

## Steps

All commands run from the repository root with `PYTHONPATH=$PWD/src:$PWD/scripts`.

| step | command | output |
|---|---|---|
| match | `match_items.py --logs LOG_ROOT [--logs ...] --out MATCHED.json` (every current Opus item matched to the stored sample whose scores equal the registry's) | `items_matched.json` and its log index |
| scope | `scope.py --matched MATCHED.json --out SCOPE [--runs FILE]` | `jobs.jsonl`, `units.tsv.gz`, `report.json` |
| pooled runs | `pooled_scope.py --scope SCOPE --runs RUNS.txt --out POOLED` (a pooled run whose Opus judgment filed no item rows: its units composed from its six behaviour runs) | a scope of its own |
| inputs | `enumerate_inputs.py --units SCOPE/units.tsv.gz --out ENUM [--verify-rebuild]` | `inputs.jsonl.gz`, `tools.json` (six answer tools), `report.json` |
| manifests | `build_manifests.py --judge glm-5.3 --enum ENUM --run-dir RUN [--reuse-ledger LEDGER] (--require-file PATH \| --no-submit-gate) --max-spend-usd USD` | the run directory and `dry_run.{json,md}` |
| partition | `check_coverage.py --enum ENUM --run-dir RUN [--run-dir ...]` | every input in exactly one run directory |
| judge | `../common/harvester.py --run-dir RUN` | the ledger |
| file | `file_records.py --judge glm-5.3 --scope SCOPE --enum ENUM --run-dir RUN [--run-dir ...] --staging STAGE --registry-root ROOT --dry-run`, then `--deliver judge --deliver-only`, a registry cycle, and `--deliver runs --deliver-only` (with the same `--run-dir`s) | records in the inbox |

`--reuse-ledger` carries a same-judge ledger's verdicts (the judge comparison's run directory of the model): its keys are excluded only
after checking that every body it sent is byte-identical to the body this build would send and that every verdict was served by
DeepInfra; the filing then takes it as a `--run-dir`.

## What is filed

One judge row (`petri_bloom_judge` on `glm-5.3`, backend `batch`; the Opus judge's rubric version, score scale and threshold rule).
`rubric_sha256` hashes the rubric every request carries (the pinned `inspect-petri` 3.0.11 judge template and highlights text, and the
six answer tools as sent, committed as `tests/fixtures/rejudge/petri_answer_tools.json`), `protocol_sha256` the request template and
the serving provider: together the paper's judge `d4dc6939...` (`tests/test_rejudge_paper_judges.py` recomputes both).
Per run, one judgment; per item, under the Opus item's own key, `petri_score` (the behaviour's own dimension), `eval_awareness` and
`scenario_realism`, or, for an item outcome (moderation, a judge refusal, a parse error, a truncated reply), one `petri_score` row with
value NULL and the outcome as its label. The judge's highlights, summary and justification stay in the run directory's ledger under the
request key. A provider failure is not filed as a row unless `--file-infra-outcomes` is given.

`synthetic_proof.py` drives a run directory through the real harvester against a fake Batch API with synthetic verdicts, for proving
the filing path on a scratch registry; it marks the run directory `SYNTHETIC`, and `file_records.py` refuses to file such a directory
into a registry without `--allow-synthetic`.
