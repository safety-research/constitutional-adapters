-- 0014_instrument_padding_zero_point.sql
-- The padding axis has a ZERO POINT: `instrument.padding_tokens` may be 0 WITH its contract hash.
-- Migration 0012 made prompt padding INSTRUMENT identity (`padding_tokens` + `padding_identity_sha256`, set together or not at all;
-- NULL = no padding recipe) and refused a pad of 0 by CHECK, because no producer had run one: the unpadded static stores predate the
-- padded conditions and differ from them in cap (4,096 vs 16,384), window (4,864 vs 98,304) and batch width, so they are not a matched
-- zero point. Pad has now generated one (cluster B jobs 130457-60; the padded conditions' own generating tree e3939f9158fe95fc9e960a2764b7b15e692dfff0,
-- window 98,304, served width 5, cap 16,384 in one pass, vendor preset, seed_base 20260909, trial 0; the four static families x the three
-- conditions), differing from the 16,384 / 65,536 levels ONLY in carrying no filler. It must file as `padding_tokens = 0` with the SAME
-- `padding_identity_sha256` as its padded siblings, never as NULL: since 0013 a roster selector matches an instrument only where
-- `padding_tokens IS NULL`, so a NULL-padded zero point would be selector-matched into the ordinary static comparisons beside the legacy
-- 4,096-cap rows (the confound from the other side) and would miss `figP_padding`, where it is the origin of the 0 / 16,384 / 65,536
-- series and is reached by NAME (roster v7 names the four zero-point instruments). NULL keeps its meaning -- no padding recipe at all --
-- and the pairing CHECK still requires the two columns set together.
-- Since this file `CONSTRAINT instrument_padding_pairing` reads `padding_tokens >= 0` where 0012 wrote `> 0`; a negative pad is still
-- refused, and so is a 0 without its contract or a contract without its pad. The two columns' why-null comments say what 0 means. Nothing
-- else changes: `identity.instrument_id` already discriminates a 0 (only a NULL is dropped from the canonical JSON), the paper views expose
-- the column as it is stored, and `v_roster_match`'s 0013 predicate already fences 0 off every selector. The reader is
-- `adapters/static_padding.py` (source `static_padding`, owner Pad), which mints the zero-point instrument from the same fields as the
-- padded ones with `padding_tokens` 0 and refuses a zero-point store whose contract is not its padded siblings'.
-- Table rebuild: `instrument` (SQLite cannot alter a CHECK in place, migrations/README.md; a dimension table -- 1,269 rows on the live
-- database, 1,261 unpadded + the eight padded -- well under a second), every column copied in primary-key order, no view recreated (the
-- views naming `instrument` survive the rebuild under the runner's legacy ALTER semantics, as under 0012). Rendered by `registry migrate
-- --plan --sql` against a schema-version-13 database. Every writer refuses a database behind version 14: `registry migrate` in the read
-- window right after landing. Measured on a `.backup` copy of the live database before landing (see docs/registry/operations.md under
-- "The padding axis has a zero point"): 1,269 instrument rows before and after (1,261 with padding_tokens NULL, 4 at 16,384,
-- 4 at 65,536), every row's id and values identical; 10,788 runs / 7,610 roster entries / 9,650 judgments / 27,583 metric rows untouched;
-- `instrument` the ONLY object whose CREATE text changed, all 29 views byte-identical, no foreign-key violation, 31 s (the runner's
-- foreign-key check over the tables naming `instrument` dominates; the copy itself is well under a second); on the migrated copy a 0
-- with its contract inserts, and a 0 alone, a contract alone and a -1 with a contract are refused by the CHECK.
-- drift: table instrument: CONSTRAINT instrument_padding_pairing differs — have «CONSTRAINT instrument_padding_pairing CHECK((padding_tokens IS NULL)=(padding_identity_sha256 IS NULL)AND(padding_tokens IS NULL OR padding_tokens > 0))» want «CONSTRAINT instrument_padding_pairing CHECK((padding_tokens IS NULL)=(padding_identity_sha256 IS NULL)AND(padding_tokens IS NULL OR padding_tokens >= 0))»

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
  applied_positions_mode TEXT NOT NULL CHECK (applied_positions_mode IN ('all', 'response', 'response_all_turns', 'prompt_last', 'prefill_all', 'prefill_last_broadcast', 'last', 'none', 'unverified')),
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
