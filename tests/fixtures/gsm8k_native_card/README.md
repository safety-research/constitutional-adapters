# Natively served GSM8K card fixture

A key-for-key synthetic copy of a natively served capability cell's leavings: manifest fields (`manifest_excerpt.json`), run-matrix row (`row.json`) and the vLLM standup record (`serving/<jobdir>/gsm8k/`), every value invented. Its one reader, `tests/test_tinycap_gsm8k_analysis.py`, asserts it resolves and matches the key sets of `tests/gsm8k_synthetic_store.py` and `tests/math500_synthetic_store.py`.

`serving_engine: lora_native`: vLLM applies the intervention as an adapter itself, not through forward hooks (the lens path). The cell, a constitution-trained ReLU steer (`constv2-mix-full__psr_ce_all`) at all token positions, was fitted on the base model and served on Llama-3.1-8B-Instruct (`llama8b`, view `main`), the paper's base-to-instruct transfer.

As in every natively served record, `transfer_provenance.served_positions` is null and its `header_located` contradicts the top-level field: the producer read positions off the lens map, empty at a native standup, so `native_served_positions_record` (`scripts/tinycap_gsm8k_validators.py`) resolves them from the routing table's routes.

| File | Content |
|---|---|
| `manifest_excerpt.json` | Fields the GSM8K reader (200-item paired subset) binds: engine, identity, route, transfer provenance |
| `row.json` | Model key, view, cell, expected served positions `all`, the object's sha |
| `serving/.../routing_table.json` | Version 2: two adapters; `petri_position_mode: "all"` on the served route; empty `lens_served_positions` |
| `serving/.../standup_jobs.jsonl`, `serving_attestation.json` | The job line; a schema-2 attestation |
