"""The C/P/I track: the Sonnet-judged over-refusal evals re-judged through OpenRouter (Batch API and the live slot): OR-Bench hard and
XSTest (the C/P/I judges ``or_bench_cpi`` / ``xstest_cpi``). The directory name is the track's; the agentic-misalignment classifier
(``am_leak.v1`` / ``am_murder.v1``) shares its modules, but the paper reports no agentic eval and that family is not a documented
reproduction path.

Modules:
  parsing        -- the error-entry classification both families share (account failure first, moderation on positive evidence
                    only, provider error), safe summaries
  cpi            -- the C/P/I request (the original ``capeval.judging`` render, translated to OpenAI chat completions) and its parser
  agentic        -- the agentic classifier request (``judge_rubrics`` prompt + ``judge_sonnet`` schema, as OpenAI strict json_schema)
  judges         -- the judge model as a CONFIG ENTRY (MiniMax-M3): model, request variant, reasoning, caps, calibration; every body still
                    built from the original judge's own params
  scope_cpi      -- paper runs -> judged cells -> units and texts, every request key recomputed against the store's own
  scope_cpi_runs -- an explicit run list -> units and texts, every pinned prompt at the named rung
  scope_cpi_reads, reads_sidecar -- the strength records' per-item reads (not a filing scope) and their verdict table
  track          -- ``track.json`` and the manifests of a run directory (one per family)
  build_manifests, file_records -- the CLIs (README)
  harvest        -- the shared harvester with three additions: a per-request ACCOUNT failure pauses the run, re-asks are counted
                    within an outcome's class, and a batch far above the baseline moderation share is held until a person looks
  harvest_all    -- several run directories harvested from one loop, one combined progress file
  reuse          -- carry a finished same-protocol run's verdicts into a full run (bodies byte-identical, attempts the full run's policy)
  live           -- the live slot's entry point (the shared engine ``rejudge.common.live``)
"""
