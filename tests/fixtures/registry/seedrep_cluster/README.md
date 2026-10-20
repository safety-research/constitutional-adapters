# Seed-replicate store fixtures

Synthetic inputs for the registry's seed-replicate reader (`src/sisteer/registry/adapters/seedrep_cluster.py`), shaped like the files a seed-replicate evaluation leaves on shared storage, with routing-table paths under both shared-root prefixes the reader maps to a cluster (`substrate_of_routing_table`). `tests/test_registry_adapter_seedrep_cluster.py` copies them, tags rewritten, into the reader's flat `<root>/{ladders,mmlu,grades_sampled}` layout and synthesises each store's remainder. Every id, sha and path is invented.

A seed replicate is one of three batch-64 fits of an object at training seeds 1 to 3, tagged `qwen35_4b_s<N>` (Qwen3.5-4B), averaged in the Qwen figures.

| File | Shape |
|---|---|
| `static_ladder_t0.json` | Static-jailbreak token-cap ladder (4,096 -> 8,192 -> 16,384) over five (cell, attack) pairs: one exited at 8,192 under `exit_le_1pct` (at most 1% truncated), four stayed at 4,096 |
| `gates/static_cap8192/*.decision.json` | That pair's gate decision: 1 loop among 518 responses, 0 truncated |
| `mmlu_*.json`, `mmlu_rows_slice.jsonl` | MMLU manifest, aggregate and evidence; three `acc` rows and the aggregate row |
| `mt_*.json`, `static_served_target.json` | Multi-turn (Crescendo) aggregate, grading provenance, completion, one shard's served target and attacker; a static shard's served target |
| `bon_*.json` | Best-of-N served record and grading provenance, cross-checked by the reader |
