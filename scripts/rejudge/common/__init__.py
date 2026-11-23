"""Shared OpenRouter library for the judge re-runs (StrongREJECT, the over-refusal C/P/I judges and the Petri judge). The interface is
documented in ``scripts/rejudge/README.md``; every module here is stdlib-only except :mod:`filing`, which imports the registry package
(``sisteer.registry``) from the checkout it runs in, and the engine of :mod:`live`, which imports ``aiohttp`` when it runs.

Modules:
  openrouter  -- the HTTP client (submit / get / list / delete / key), error classification (413 / 429 / 402 / 403-limit / moderation)
  batchfile   -- request-body serialization, size-aware chunking, per-batch manifests and bodies on disk, cost estimates
  state       -- the run directory's SQLite ledger: one row per batch (submission state) and one row per request (the verdict cache)
  submitter   -- submit every pending batch up front, with split-on-413, backoff-on-429 and queue-on-balance
  harvester   -- the persistent loop: submit, poll, fetch results, parse through the track's parser, retry, progress.json
  live        -- the live slot: a batch run directory's never-submitted requests sent one by one on chat completions
  merge       -- one key, one filed outcome: the merge of batch, reuse and live ledgers, and the live-coverage gate
  filing      -- registry records (a NEW judge row, NEW judgment rows, item_result rows) through the sanctioned record path
  scratch_proof -- digests of every table and paper view before and after a filing on a scratch copy of the registry
  runlist     -- the explicit run list's TSV reader, the read-only registry connection and the path rules that tie a listed store to
                 the registry's (the explicit-run-list scopes)
"""
