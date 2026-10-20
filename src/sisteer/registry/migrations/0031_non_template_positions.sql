-- 0031_non_template_positions.sql
-- `non_template` joins the applied-positions vocabulary (vocab.APPLIED_POSITIONS_MODES): every token of every prefill and decode EXCEPT
-- the chat-template scaffold (`<|im_start|>` / the role name / its newline, `<|im_end|>` / its newline, the empty think header) -- the
-- position class the routing analysis found rerouting concentrated on, now served as a mode of its own (the serving config's
-- `native_residual.POSITION_MODES` holds the token rule, applied to the realized ids at decode as at prefill). The value is a CHECK
-- vocabulary on TWO tables, `instrument` (the declared measurement setting) and `run` (the served fact), and SQLite cannot alter a CHECK
-- in place (migrations/README.md, the second row of its table), so this file REBUILDS both: `CREATE TABLE <t>__new` from the code's DDL
-- with the grown CHECK, `INSERT ... SELECT` of every column in primary-key order, `DROP`, `RENAME`, then the table's indexes and
-- triggers, `ANALYZE`. No column is added or dropped, no row is rewritten (every live value is in the grown list), no v_* object is
-- touched. The run rebuild recreates EVERY trigger ON `run`, the two fold-order fence triggers of the previous file included
-- (`schema_def.fold_order_fence_triggers`; the planner prints no `recreate trigger(s)` line of its own, since the rebuild carries them),
-- so a database after this file still refuses a `run` write by a connection without the code's `registry_fold_order()`.
-- Rendered by `registry migrate --plan --sql` against a schema-version-30 database (0001..0030 applied to an empty database by the
-- runner); the plan reads `pending: 0031_non_template_positions`, the two `lacks 'non_template'` drift lines, `rebuild table instrument`
-- and `rebuild table run`, nothing created, added or recreated. Both tables are run-scale (the live `run` table copies in seconds, not
-- gigabytes); the landing window applies the file under WINDOW_ALLOW_REBUILD=1 naming this reviewed rebuild and records both row counts
-- unchanged. No Python post-step. The vocabulary GROWS: every record already in the log still validates, so there is no grandfather entry
-- and no whole-log audit of a new refusal; the validator stamp moves with the DDL, so a record refused for the new value before this file
-- is judged once more after it.
-- Beside the DDL, on the same branch: `derive.applied_positions_mode` reads a known standup mode before the natively served LoRA rule;
-- `derive.multi_turn_steering_scope` maps the mode to `every_turn_prefill_and_decode`; `jobplan.canonical_instrument` selects the seeded
-- instrument by served mode (`emit_cli register --positions-mode`); the standup's endpoint record stamps a list-shaped served mode;
-- `registry reconcile multi-turn-scope` corrects the scope of runs registered with the job-start default; two seed instruments carry
-- `applied_positions_mode: non_template` for the multi-turn ActorAttack and Crescendo evals.
-- drift: table instrument: column applied_positions_mode CHECK (applied_positions_mode IN (...)) lacks 'non_template'
-- drift: table run: column applied_positions_mode CHECK (applied_positions_mode IN (...)) lacks 'non_template'

-- rebuild instrument: SQLite cannot alter a constraint in place (migrations/README.md)
CREATE TABLE instrument__new (
  instrument_id TEXT NOT NULL,
  eval_id TEXT NOT NULL REFERENCES eval(eval_id),
  item_set_id TEXT NOT NULL REFERENCES item_set(item_set_id),
  harness TEXT NOT NULL CHECK (harness IN ('bespoke_sisteer', 'inspect_ai', 'inspect_evals', 'evalplus', 'bfcl_eval', 'pyrit', 'actorattack', 'petri', 'strongreject', 'agentic_misalignment', 'em_organism', 'lm_eval')),
  harness_version TEXT NOT NULL,
  harness_config_hash TEXT NOT NULL,
  n_shot INTEGER,  -- why-null: not a shot task
  prompt_template_sha256 TEXT NOT NULL CHECK (prompt_template_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR prompt_template_sha256 = 'unverified'),
  cot_prompting TEXT CHECK (cot_prompting IN ('none', 'explicit_instruction', 'few_shot_cot')),  -- why-null: not applicable to attack and judge-graded evals
  scoring_rule TEXT NOT NULL CHECK (scoring_rule IN ('exact_match', 'numeric_match', 'sympy_equiv', 'letter_logprob_prefill', 'judge_threshold', 'judge_graded_weights', 'pass_at_1_sandbox', 'ast_match', 'rate_over_scenarios', 'ifeval_checker', 'judge_score_mean')),
  answer_extraction TEXT NOT NULL CHECK (answer_extraction IN ('regex', 'last_number', 'boxed', 'judge', 'none', 'unverified')),
  token_cap_tokens INTEGER,  -- why-null: unverified for a superseded instrument whose cap was never read (reason in field_confidence_json)
  token_cap_ladder_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(token_cap_ladder_json)),
  ladder_promotion_rule TEXT,  -- why-null: a fixed cap has no ladder
  read_at TEXT NOT NULL CHECK (read_at IN ('ladder_final', 'fixed', 'unverified')),
  reply_cap_tokens INTEGER,  -- why-null: only multi-turn instruments cap replies separately
  context_window_tokens INTEGER,  -- why-null: not declared for every instrument
  padding_tokens INTEGER,  -- why-null: no padding recipe: the instrument carries no filler contract at all (NULL; every unpadded row). 0 is a VALUE, never NULL: the recipe's zero point, a matched control generated under the padded conditions' own tree, geometry, cap and preset with no filler, paired with the same contract hash (CHECK instrument_padding_pairing, >= 0 since migration 0014)
  padding_identity_sha256 TEXT CHECK (padding_identity_sha256 IS NULL OR padding_identity_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- why-null: no padding recipe: no filler contract to hash (NULL pairs with a NULL padding_tokens; sha256 over the canonical JSON of {filler_sha256, mode, placement, separator, tolerance}, identity.padding_identity_sha256; the block itself rides extra_json.static_padding). The zero point of a recipe carries the contract hash beside padding_tokens 0
  decode_source TEXT NOT NULL CHECK (decode_source IN ('generation_config', 'vendor_recommendation', 'model_card', 'vllm_default', 'greedy', 'grandfathered_T1', 'runner_forced_T1', 'legacy_contracts_row', 'unverified')),
  decode_temperature REAL,  -- why-null: not sent (engine default) — resolved value in the run's decode_resolved_json
  decode_top_p REAL,  -- why-null: not sent (engine default)
  decode_top_k INTEGER,  -- why-null: not sent (engine default)
  decode_min_p REAL,  -- why-null: not sent (engine default)
  decode_repetition_penalty REAL,  -- why-null: not sent (engine default)
  decode_presence_penalty REAL,  -- why-null: not sent (engine default)
  decode_seed_rule TEXT NOT NULL CHECK (decode_seed_rule IN ('per_item', 'per_draw', 'per_goal_turn', 'per_scenario_turn', 'per_response', 'fixed', 'none', 'unverified')),
  thinking_mode TEXT NOT NULL CHECK (thinking_mode IN ('on', 'off', 'not_available')),
  n_trials INTEGER,  -- why-null: unverified for a superseded instrument whose trial count was never read (reason in field_confidence_json)
  bon_n INTEGER,  -- why-null: only best-of-n instruments
  max_turns INTEGER,  -- why-null: only multi-turn instruments
  serving_path TEXT NOT NULL CHECK (serving_path IN ('lora_native', 'lens', 'passthrough', 'hf', 'unverified')),
  applied_positions_mode TEXT NOT NULL CHECK (applied_positions_mode IN ('all', 'response', 'response_all_turns', 'prompt_last', 'prefill_all', 'prefill_last_broadcast', 'last', 'non_template', 'none', 'unverified')),
  attacker_model_id TEXT REFERENCES model(model_id),  -- why-null: only attacker-driven evals
  attacker_decode_json TEXT CHECK (attacker_decode_json IS NULL OR json_valid(attacker_decode_json)),  -- why-null: only attacker-driven evals
  judge_id TEXT REFERENCES judge(judge_id),  -- why-null: rule-scored evals (exact match, sandbox) have no judge
  prefill_scheme TEXT,  -- why-null: only prefill-scored instruments
  field_confidence_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_confidence_json)),
  notes TEXT,  -- why-null: free text
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (instrument_id),
  CONSTRAINT instrument_padding_pairing CHECK ((padding_tokens IS NULL) = (padding_identity_sha256 IS NULL) AND (padding_tokens IS NULL OR padding_tokens >= 0))
);
INSERT INTO instrument__new (instrument_id, eval_id, item_set_id, harness, harness_version, harness_config_hash, n_shot, prompt_template_sha256, cot_prompting, scoring_rule, answer_extraction, token_cap_tokens, token_cap_ladder_json, ladder_promotion_rule, read_at, reply_cap_tokens, context_window_tokens, padding_tokens, padding_identity_sha256, decode_source, decode_temperature, decode_top_p, decode_top_k, decode_min_p, decode_repetition_penalty, decode_presence_penalty, decode_seed_rule, thinking_mode, n_trials, bon_n, max_turns, serving_path, applied_positions_mode, attacker_model_id, attacker_decode_json, judge_id, prefill_scheme, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq) SELECT instrument_id, eval_id, item_set_id, harness, harness_version, harness_config_hash, n_shot, prompt_template_sha256, cot_prompting, scoring_rule, answer_extraction, token_cap_tokens, token_cap_ladder_json, ladder_promotion_rule, read_at, reply_cap_tokens, context_window_tokens, padding_tokens, padding_identity_sha256, decode_source, decode_temperature, decode_top_p, decode_top_k, decode_min_p, decode_repetition_penalty, decode_presence_penalty, decode_seed_rule, thinking_mode, n_trials, bon_n, max_turns, serving_path, applied_positions_mode, attacker_model_id, attacker_decode_json, judge_id, prefill_scheme, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq FROM instrument ORDER BY instrument_id;
DROP TABLE instrument;
ALTER TABLE instrument__new RENAME TO instrument;
ANALYZE instrument;

-- rebuild run: SQLite cannot alter a constraint in place (migrations/README.md)
CREATE TABLE run__new (
  run_id TEXT NOT NULL,  -- `identity.run_id(campaign, producer_key, producer_object_key, eval_id, instrument_id)`.
  campaign TEXT NOT NULL,
  producer_run_key_legacy TEXT NOT NULL,  -- The producer's own EXECUTION key verbatim (F1).
  producer_object_key_legacy TEXT NOT NULL,  -- The producer's OBJECT discriminator within that key (S1); `none` for an unsteered execution.
  lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN ('queued', 'running', 'generated', 'judging', 'judged', 'ingested', 'failed', 'cancelled', 'superseded', 'withdrawn')),
  queued_utc TEXT CHECK (queued_utc IS NULL OR queued_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: set when the state is reached (trigger trg_run_state_utc requires it from `queued` on)
  started_utc TEXT CHECK (started_utc IS NULL OR started_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: set when the state is reached (trigger trg_run_state_utc requires it from `running` on)
  generated_utc TEXT CHECK (generated_utc IS NULL OR generated_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: set when the state is reached (trigger trg_run_state_utc requires it from `generated` on)
  judged_utc TEXT CHECK (judged_utc IS NULL OR judged_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: set when the state is reached (trigger trg_run_state_utc requires it from `judged` on)
  ingested_utc TEXT CHECK (ingested_utc IS NULL OR ingested_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: set by the harvester's run_ingest record (trigger requires it at `ingested`)
  finished_utc TEXT CHECK (finished_utc IS NULL OR finished_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: terminal time of failed / cancelled runs (trigger requires it for those states)
  walltime_s REAL,  -- why-null: recorded where sacct / Modal / the store span allow, never estimated
  walltime_source TEXT CHECK (walltime_source IN ('sacct', 'modal_call', 'store_span')),  -- why-null: null when walltime_s is null
  model_id TEXT NOT NULL REFERENCES model(model_id),
  model_revision TEXT CHECK (model_revision IS NULL OR model_revision GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR model_revision = 'unverified'),  -- why-null: observed at serving (RUN_VERIFIED_FIELDS); legacy rows may lack it
  object_id TEXT NOT NULL REFERENCES object(object_id),
  transfer_kind TEXT NOT NULL CHECK (transfer_kind IN ('base_to_instruct', 'instruct_direct', 'chat_checkpoint', 'prompted_only', 'none', 'not_applicable')),
  strength_coefficient REAL,  -- why-null: `none` and prompted objects have no coefficient
  strength_nominal REAL,  -- why-null: `none` and prompted objects have no nominal dose
  strength_nominal_source TEXT CHECK (strength_nominal_source IN ('filed', 'derived')),  -- why-null: null when strength_nominal is null
  strength_unit TEXT NOT NULL DEFAULT 'unverified' CHECK (strength_unit IN ('lora_write_basis_multiplier', 'psr_direction_norm_multiplier', 'alpha', 'lambda', 'raw_coefficient', 'nominal_only', 'unverified')),  -- the unit of strength_nominal: two dose ladders scaling different quantities never join into one curve
  is_operating_point INTEGER CHECK (is_operating_point IN (0, 1)),  -- why-null: unknown without an operating_point ruling
  operating_point_id_legacy TEXT,  -- why-null: only producers that filed an operating-point id
  eval_id TEXT NOT NULL REFERENCES eval(eval_id),
  item_set_id TEXT NOT NULL REFERENCES item_set(item_set_id),
  instrument_id TEXT NOT NULL REFERENCES instrument(instrument_id),
  instrument_match TEXT NOT NULL CHECK (instrument_match IN ('match', 'mismatch', 'unverified')),
  harness TEXT NOT NULL CHECK (harness IN ('bespoke_sisteer', 'inspect_ai', 'inspect_evals', 'evalplus', 'bfcl_eval', 'pyrit', 'actorattack', 'petri', 'strongreject', 'agentic_misalignment', 'em_organism', 'lm_eval')),
  harness_version TEXT NOT NULL,
  n_shot INTEGER,  -- why-null: not a shot task
  cot_prompting TEXT CHECK (cot_prompting IN ('none', 'explicit_instruction', 'few_shot_cot')),  -- why-null: not applicable to attack and judge-graded evals
  scoring_rule TEXT NOT NULL CHECK (scoring_rule IN ('exact_match', 'numeric_match', 'sympy_equiv', 'letter_logprob_prefill', 'judge_threshold', 'judge_graded_weights', 'pass_at_1_sandbox', 'ast_match', 'rate_over_scenarios', 'ifeval_checker', 'judge_score_mean')),
  prompt_template_sha256 TEXT NOT NULL CHECK (prompt_template_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR prompt_template_sha256 = 'unverified'),
  judge_id TEXT REFERENCES judge(judge_id),  -- why-null: primary judge; null before any judgment and for rule-scored evals
  pass_label TEXT,  -- why-null: only coexisting re-measurements carry a pass label
  trial_index INTEGER NOT NULL DEFAULT 0,
  n_items_planned INTEGER,  -- why-null: unknown for legacy stores
  n_items_generated INTEGER,  -- why-null: not yet generated
  n_items_judged INTEGER,  -- why-null: not yet judged
  seed_base INTEGER,  -- why-null: only seeded decodes
  decode_seed_rule TEXT NOT NULL CHECK (decode_seed_rule IN ('per_item', 'per_draw', 'per_goal_turn', 'per_scenario_turn', 'per_response', 'fixed', 'none', 'unverified')),
  decode_policy_id TEXT REFERENCES decode_policy(policy_id),  -- why-null: derived (decode_match.stamp_run_rows from v_run_decode_policy_match): set only when decode_policy_match = exact
  decode_policy_match TEXT NOT NULL DEFAULT 'no_policy' CHECK (decode_policy_match IN ('exact', 'ambiguous', 'no_match', 'unrecorded', 'no_policy')),  -- Derived per run from v_run_decode_policy_match (exact / ambiguous / no_match / unrecorded); `no_policy` = the producer's placeholder before derivation.
  decode_sent_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(decode_sent_json)),
  decode_resolved_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(decode_resolved_json)),
  decode_temperature_sent REAL,  -- why-null: not sent (resolved value in decode_resolved_json)
  decode_top_p_sent REAL,  -- why-null: not sent (resolved value in decode_resolved_json)
  decode_top_k_sent INTEGER,  -- why-null: not sent (resolved value in decode_resolved_json)
  decode_source TEXT NOT NULL CHECK (decode_source IN ('generation_config', 'vendor_recommendation', 'model_card', 'vllm_default', 'greedy', 'grandfathered_T1', 'runner_forced_T1', 'legacy_contracts_row', 'unverified')),
  token_cap_tokens INTEGER NOT NULL,
  token_cap_rung_index INTEGER,  -- why-null: a fixed cap has no ladder
  token_cap_ladder_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(token_cap_ladder_json)),
  reply_cap_tokens INTEGER,  -- why-null: only multi-turn runs cap replies separately
  context_window_tokens INTEGER,  -- why-null: observed at serving; legacy rows may lack it
  served_max_num_seqs INTEGER,  -- why-null: observed at serving; legacy rows may lack it
  served_max_num_seqs_source TEXT CHECK (served_max_num_seqs_source IN ('attested', 'runner_concurrency')),  -- why-null: null when served_max_num_seqs is null
  thinking_mode TEXT NOT NULL CHECK (thinking_mode IN ('on', 'off', 'not_available')),
  serving_path TEXT NOT NULL CHECK (serving_path IN ('lora_native', 'lens', 'passthrough', 'hf', 'unverified')),
  serving_engine_version TEXT,  -- why-null: observed at serving; legacy rows may lack it
  serving_identity TEXT,  -- why-null: observed at serving; legacy rows may lack it
  serving_config_sha256 TEXT CHECK (serving_config_sha256 IS NULL OR serving_config_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR serving_config_sha256 = 'unverified'),  -- why-null: observed at serving; legacy rows may lack it
  adapter_fingerprint TEXT,  -- why-null: only natively served adapters
  applied_positions_mode TEXT NOT NULL CHECK (applied_positions_mode IN ('all', 'response', 'response_all_turns', 'prompt_last', 'prefill_all', 'prefill_last_broadcast', 'last', 'non_template', 'none', 'unverified')),
  applied_layers_json TEXT CHECK (applied_layers_json IS NULL OR json_valid(applied_layers_json)),  -- why-null: `none` objects have no layers
  multi_turn_steering_scope TEXT NOT NULL CHECK (multi_turn_steering_scope IN ('every_turn_prefill_and_decode', 'final_header_and_current_decode_only', 'every_assistant_span', 'prefill_only_each_turn', 'single_turn', 'not_applicable', 'unverified')),
  history_resent_each_turn INTEGER CHECK (history_resent_each_turn IN (0, 1)),  -- why-null: single-turn runs
  prefix_cache_policy TEXT CHECK (prefix_cache_policy IN ('on', 'off', 'skip_reading')),  -- why-null: not recorded for every serving path
  intervention_hash TEXT,  -- why-null: `none` objects and legacy rows
  harness_config_hash TEXT,  -- why-null: legacy rows may lack it
  env_digest TEXT,  -- why-null: legacy rows may lack it
  code_sha TEXT NOT NULL CHECK (code_sha GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR code_sha = 'unverified'),
  code_branch TEXT NOT NULL DEFAULT 'unknown',  -- `unknown` when code_sha is not an ancestor of origin/main or a known branch (Fixed decision 23).
  config_path TEXT,  -- why-null: not every run launched from a config file
  config_sha256 TEXT CHECK (config_sha256 IS NULL OR config_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR config_sha256 = 'unverified'),  -- why-null: not every run launched from a config file
  submit_sha_pin TEXT CHECK (submit_sha_pin IS NULL OR submit_sha_pin GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR submit_sha_pin = 'unverified'),  -- why-null: only registered submissions carry a pin
  substrate TEXT NOT NULL CHECK (substrate IN ('cluster_a', 'cluster_b', 'modal', 'grading_host', 'runpod', 'slurm')),
  substrate_source TEXT NOT NULL CHECK (substrate_source IN ('recorded', 'derived', 'assumed_default', 'inferred', 'unverified')),
  cluster_job_id TEXT,  -- why-null: only cluster runs
  job_name TEXT,  -- why-null: only cluster runs
  node_hostname TEXT,  -- why-null: set at job start
  gpu_type TEXT,  -- why-null: derived from inventory or nvidia-smi where available
  gpu_count INTEGER,  -- why-null: set at job start
  modal_app_id TEXT,  -- why-null: only Modal runs
  modal_call_id TEXT,  -- why-null: only Modal runs launched after native emission
  wandb_run_id TEXT,  -- why-null: only runs that logged to W&B
  attacker_model_id TEXT REFERENCES model(model_id),  -- why-null: only attacker-driven evals
  attacker_identity_attested INTEGER CHECK (attacker_identity_attested IN (0, 1)),  -- why-null: only attacker-driven evals
  store_uri TEXT,  -- why-null: a queued run has no store yet
  store_completion_artifact_uri TEXT,  -- why-null: set at generation
  store_completion_sha256 TEXT CHECK (store_completion_sha256 IS NULL OR store_completion_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR store_completion_sha256 = 'unverified'),  -- why-null: set at generation
  view_legacy TEXT,  -- why-null: only backfilled rows carry the producer's legacy transfer-setting string
  lane_source_legacy TEXT,  -- why-null: only backfilled rows carry the paper's source string
  campaign_cell_id_legacy TEXT,  -- why-null: only backfilled rows carry the producer's legacy id
  producer_session TEXT,  -- why-null: unknown for legacy rows
  status_ruling_id TEXT REFERENCES ruling(ruling_id),  -- why-null: set only when a ruling changed the status
  superseded_by_run_id TEXT REFERENCES run(run_id),  -- why-null: only superseded runs
  field_confidence_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_confidence_json)),
  notes TEXT,  -- why-null: free text
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (run_id),
  UNIQUE (producer_run_key_legacy, producer_object_key_legacy, eval_id, instrument_id)
);
INSERT INTO run__new (run_id, campaign, producer_run_key_legacy, producer_object_key_legacy, lifecycle_state, queued_utc, started_utc, generated_utc, judged_utc, ingested_utc, finished_utc, walltime_s, walltime_source, model_id, model_revision, object_id, transfer_kind, strength_coefficient, strength_nominal, strength_nominal_source, strength_unit, is_operating_point, operating_point_id_legacy, eval_id, item_set_id, instrument_id, instrument_match, harness, harness_version, n_shot, cot_prompting, scoring_rule, prompt_template_sha256, judge_id, pass_label, trial_index, n_items_planned, n_items_generated, n_items_judged, seed_base, decode_seed_rule, decode_policy_id, decode_policy_match, decode_sent_json, decode_resolved_json, decode_temperature_sent, decode_top_p_sent, decode_top_k_sent, decode_source, token_cap_tokens, token_cap_rung_index, token_cap_ladder_json, reply_cap_tokens, context_window_tokens, served_max_num_seqs, served_max_num_seqs_source, thinking_mode, serving_path, serving_engine_version, serving_identity, serving_config_sha256, adapter_fingerprint, applied_positions_mode, applied_layers_json, multi_turn_steering_scope, history_resent_each_turn, prefix_cache_policy, intervention_hash, harness_config_hash, env_digest, code_sha, code_branch, config_path, config_sha256, submit_sha_pin, substrate, substrate_source, cluster_job_id, job_name, node_hostname, gpu_type, gpu_count, modal_app_id, modal_call_id, wandb_run_id, attacker_model_id, attacker_identity_attested, store_uri, store_completion_artifact_uri, store_completion_sha256, view_legacy, lane_source_legacy, campaign_cell_id_legacy, producer_session, status_ruling_id, superseded_by_run_id, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq) SELECT run_id, campaign, producer_run_key_legacy, producer_object_key_legacy, lifecycle_state, queued_utc, started_utc, generated_utc, judged_utc, ingested_utc, finished_utc, walltime_s, walltime_source, model_id, model_revision, object_id, transfer_kind, strength_coefficient, strength_nominal, strength_nominal_source, strength_unit, is_operating_point, operating_point_id_legacy, eval_id, item_set_id, instrument_id, instrument_match, harness, harness_version, n_shot, cot_prompting, scoring_rule, prompt_template_sha256, judge_id, pass_label, trial_index, n_items_planned, n_items_generated, n_items_judged, seed_base, decode_seed_rule, decode_policy_id, decode_policy_match, decode_sent_json, decode_resolved_json, decode_temperature_sent, decode_top_p_sent, decode_top_k_sent, decode_source, token_cap_tokens, token_cap_rung_index, token_cap_ladder_json, reply_cap_tokens, context_window_tokens, served_max_num_seqs, served_max_num_seqs_source, thinking_mode, serving_path, serving_engine_version, serving_identity, serving_config_sha256, adapter_fingerprint, applied_positions_mode, applied_layers_json, multi_turn_steering_scope, history_resent_each_turn, prefix_cache_policy, intervention_hash, harness_config_hash, env_digest, code_sha, code_branch, config_path, config_sha256, submit_sha_pin, substrate, substrate_source, cluster_job_id, job_name, node_hostname, gpu_type, gpu_count, modal_app_id, modal_call_id, wandb_run_id, attacker_model_id, attacker_identity_attested, store_uri, store_completion_artifact_uri, store_completion_sha256, view_legacy, lane_source_legacy, campaign_cell_id_legacy, producer_session, status_ruling_id, superseded_by_run_id, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq FROM run ORDER BY run_id;
DROP TABLE run;
ALTER TABLE run__new RENAME TO run;
CREATE INDEX idx_run_key ON run (model_id, object_id, eval_id, instrument_id);
CREATE INDEX idx_run_lifecycle ON run (lifecycle_state);
CREATE INDEX idx_run_campaign ON run (campaign);
CREATE INDEX idx_run_cluster_job ON run (cluster_job_id);
CREATE TRIGGER trg_run_judged_requires_judgment_ins BEFORE INSERT ON run
WHEN NEW.lifecycle_state = 'judged' AND NOT EXISTS (SELECT 1 FROM judgment WHERE run_id = NEW.run_id)
BEGIN
  SELECT RAISE(ABORT, 'run.lifecycle_state=judged requires a judgment row for this run_id');
END;
CREATE TRIGGER trg_run_judged_requires_judgment_upd BEFORE UPDATE OF lifecycle_state ON run
WHEN NEW.lifecycle_state = 'judged' AND NOT EXISTS (SELECT 1 FROM judgment WHERE run_id = NEW.run_id)
BEGIN
  SELECT RAISE(ABORT, 'run.lifecycle_state=judged requires a judgment row for this run_id');
END;
CREATE TRIGGER trg_run_state_utc_ins BEFORE INSERT ON run
WHEN (NEW.lifecycle_state = 'queued' AND (NEW.queued_utc IS NULL))
  OR (NEW.lifecycle_state = 'running' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL))
  OR (NEW.lifecycle_state = 'generated' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL))
  OR (NEW.lifecycle_state = 'judged' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL OR NEW.judged_utc IS NULL))
  OR (NEW.lifecycle_state = 'ingested' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL OR NEW.judged_utc IS NULL OR NEW.ingested_utc IS NULL))
  OR (NEW.lifecycle_state = 'judging' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL))
  OR (NEW.lifecycle_state IN ('failed','cancelled') AND (NEW.queued_utc IS NULL OR NEW.finished_utc IS NULL))
  OR (NEW.lifecycle_state IN ('superseded','withdrawn') AND NEW.queued_utc IS NULL)
BEGIN
  SELECT RAISE(ABORT, 'run: every *_utc of a reached lifecycle state must be non-null (trg_run_state_utc)');
END;
CREATE TRIGGER trg_run_state_utc_upd BEFORE UPDATE ON run
WHEN (NEW.lifecycle_state = 'queued' AND (NEW.queued_utc IS NULL))
  OR (NEW.lifecycle_state = 'running' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL))
  OR (NEW.lifecycle_state = 'generated' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL))
  OR (NEW.lifecycle_state = 'judged' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL OR NEW.judged_utc IS NULL))
  OR (NEW.lifecycle_state = 'ingested' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL OR NEW.judged_utc IS NULL OR NEW.ingested_utc IS NULL))
  OR (NEW.lifecycle_state = 'judging' AND (NEW.queued_utc IS NULL OR NEW.started_utc IS NULL OR NEW.generated_utc IS NULL))
  OR (NEW.lifecycle_state IN ('failed','cancelled') AND (NEW.queued_utc IS NULL OR NEW.finished_utc IS NULL))
  OR (NEW.lifecycle_state IN ('superseded','withdrawn') AND NEW.queued_utc IS NULL)
BEGIN
  SELECT RAISE(ABORT, 'run: every *_utc of a reached lifecycle state must be non-null (trg_run_state_utc)');
END;
CREATE TRIGGER trg_run_fold_order_fence_ins BEFORE INSERT ON run
WHEN EXISTS (SELECT 1 FROM schema_migration WHERE version = 0 AND description IS NOT registry_fold_order())
BEGIN
  SELECT RAISE(ABORT, 'run: fold-order fence -- this connection folds under a same-stamp order the database does not record (registry_fold_order() against schema_migration row 0); nothing written: fast-forward the checkout or rebuild the database (docs/registry/records.md, Identity and order: the run-scale order has a version)');
END;
CREATE TRIGGER trg_run_fold_order_fence_upd BEFORE UPDATE ON run
WHEN EXISTS (SELECT 1 FROM schema_migration WHERE version = 0 AND description IS NOT registry_fold_order())
BEGIN
  SELECT RAISE(ABORT, 'run: fold-order fence -- this connection folds under a same-stamp order the database does not record (registry_fold_order() against schema_migration row 0); nothing written: fast-forward the checkout or rebuild the database (docs/registry/records.md, Identity and order: the run-scale order has a version)');
END;
ANALYZE run;
