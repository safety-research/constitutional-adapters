-- 0018_substrate_slurm: `slurm` joins vocab.SUBSTRATES (2026-09-29) -- the generic value for a third Slurm cluster, whose rows carry the site in extra_json.cluster_label. The value binds THREE CHECKs -- run.substrate, object.fit_substrate and fit_search.substrate (FIT_SEARCH_SUBSTRATES = SUBSTRATES + unverified) -- so the three tables are rebuilt in place (SQLite cannot alter a CHECK). Rendered by `registry migrate --plan --sql` against the version-17 shape (0001..0017 applied to an empty database); rows are copied unchanged.
-- drift: table fit_search: column substrate CHECK (substrate IN (...)) lacks 'slurm'
-- drift: table object: column fit_substrate CHECK (fit_substrate IN (...)) lacks 'slurm'
-- drift: table run: column substrate CHECK (substrate IN (...)) lacks 'slurm'

-- rebuild fit_search: SQLite cannot alter a constraint in place (migrations/README.md)
CREATE TABLE fit_search__new (
  search_id TEXT NOT NULL,
  campaign TEXT NOT NULL,
  target_object_class_json TEXT NOT NULL CHECK (json_valid(target_object_class_json)),
  method TEXT NOT NULL CHECK (method IN ('sobol', 'grid', 'random', 'bayesian', 'manual', 'reconnaissance')),
  n_trials INTEGER NOT NULL,
  n_trials_completed INTEGER NOT NULL,
  search_space_json TEXT NOT NULL CHECK (json_valid(search_space_json)),
  objective TEXT NOT NULL,
  objective_direction TEXT NOT NULL CHECK (objective_direction IN ('minimize', 'maximize')),
  winner_params_json TEXT CHECK (winner_params_json IS NULL OR json_valid(winner_params_json)),  -- why-null: a search may not have concluded
  winner_trial_id TEXT,  -- why-null: a search may not have concluded
  winner_objective REAL,  -- why-null: a search may not have concluded
  argmin_interior INTEGER CHECK (argmin_interior IN (0, 1)),  -- why-null: only fitted-argmin searches evaluate interiority
  quadratic_fit_positive_definite INTEGER CHECK (quadratic_fit_positive_definite IN (0, 1)),  -- why-null: only quadratic-fit searches
  quadratic_fit_params_json TEXT CHECK (quadratic_fit_params_json IS NULL OR json_valid(quadratic_fit_params_json)),  -- why-null: only quadratic-fit searches
  recentre_rounds INTEGER NOT NULL DEFAULT 0,
  confirmation_trial_id TEXT,  -- why-null: no confirmation trial was run
  confirmation_within_noise INTEGER CHECK (confirmation_within_noise IN (0, 1)),  -- why-null: no confirmation trial was run
  noise_estimate REAL,  -- why-null: no noise estimate was recorded
  grad_norm_reconnaissance_uri TEXT,  -- why-null: no reconnaissance pass
  substrate TEXT NOT NULL CHECK (substrate IN ('cluster_a', 'cluster_b', 'modal', 'grading_host', 'runpod', 'slurm', 'unverified')),  -- `unverified` when no record beside the sweep names the substrate; the missing evidence is named in `field_confidence_json`.
  job_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(job_ids_json)),
  code_sha TEXT NOT NULL CHECK (code_sha GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR code_sha = 'unverified'),
  config_sha256 TEXT CHECK (config_sha256 IS NULL OR config_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR config_sha256 = 'unverified'),  -- why-null: manual searches have no config file
  started_utc TEXT NOT NULL CHECK (started_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),
  finished_utc TEXT CHECK (finished_utc IS NULL OR finished_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: search in progress
  wandb_sweep_url TEXT,  -- why-null: not every search logged to W&B
  trials_uri TEXT,  -- why-null: the trial log may not be on disk
  producer_session TEXT,  -- why-null: unknown for legacy searches
  field_confidence_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_confidence_json)),
  notes TEXT,  -- why-null: free text
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (search_id)
);
INSERT INTO fit_search__new (search_id, campaign, target_object_class_json, method, n_trials, n_trials_completed, search_space_json, objective, objective_direction, winner_params_json, winner_trial_id, winner_objective, argmin_interior, quadratic_fit_positive_definite, quadratic_fit_params_json, recentre_rounds, confirmation_trial_id, confirmation_within_noise, noise_estimate, grad_norm_reconnaissance_uri, substrate, job_ids_json, code_sha, config_sha256, started_utc, finished_utc, wandb_sweep_url, trials_uri, producer_session, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq) SELECT search_id, campaign, target_object_class_json, method, n_trials, n_trials_completed, search_space_json, objective, objective_direction, winner_params_json, winner_trial_id, winner_objective, argmin_interior, quadratic_fit_positive_definite, quadratic_fit_params_json, recentre_rounds, confirmation_trial_id, confirmation_within_noise, noise_estimate, grad_norm_reconnaissance_uri, substrate, job_ids_json, code_sha, config_sha256, started_utc, finished_utc, wandb_sweep_url, trials_uri, producer_session, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq FROM fit_search ORDER BY search_id;
DROP TABLE fit_search;
ALTER TABLE fit_search__new RENAME TO fit_search;
ANALYZE fit_search;

-- rebuild object: SQLite cannot alter a constraint in place (migrations/README.md)
CREATE TABLE object__new (
  object_id TEXT NOT NULL,
  object_kind TEXT NOT NULL CHECK (object_kind IN ('none', 'system_instruction', 'lora', 'lora_subtraction', 'relu_steer', 'relu_steer_distilled_difference', 'vector', 'vector_all', 'diffmean', 'static_r_arditi', 'caa_static_r', 'alphasteer', 'antidote', 'composed', 'stacked')),
  paper_label TEXT NOT NULL CHECK (paper_label IN ('none', 'safety_si', 'constitution_si', 'constitution_si_full', 'control_lora', 'constitution_lora', 'difference_lora', 'control_relu', 'constitution_relu', 'difference_relu', 'difference_relu_distilled', 'arditi', 'arditi_allpos', 'alphasteer_rebake', 'alphasteer_as_fit', 'alphasteer_refit_reference', 'safety_lora', 'safety_relu', 'other')),
  campaign TEXT NOT NULL,
  campaign_cell_id_legacy TEXT,  -- why-null: natively emitted objects carry no legacy id
  object_role TEXT NOT NULL CHECK (object_role IN ('constitution', 'control', 'difference', 'safety', 'irrelevant', 'baseline_none', 'baseline_published', 'prompt')),
  fit_target_kind TEXT NOT NULL CHECK (fit_target_kind IN ('system_instruction_distillation', 'corpus_document_fit', 'contrastive_direction', 'derived', 'declared_prompt', 'none')),
  fit_teacher_si_id TEXT REFERENCES si_text(si_id),  -- why-null: only system_instruction_distillation (paired-nullness CHECK)
  fit_teacher_si_text_sha256 TEXT CHECK (fit_teacher_si_text_sha256 IS NULL OR fit_teacher_si_text_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]') REFERENCES si_text(si_text_sha256),  -- why-null: only system_instruction_distillation (paired-nullness CHECK)
  fit_prompt_pool_id TEXT,  -- why-null: only system_instruction_distillation (paired-nullness CHECK)
  fit_prompt_pool_sha256 TEXT CHECK (fit_prompt_pool_sha256 IS NULL OR fit_prompt_pool_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR fit_prompt_pool_sha256 = 'unverified'),  -- why-null: only system_instruction_distillation (paired-nullness CHECK)
  fit_corpus_id TEXT,  -- why-null: only corpus_document_fit (paired-nullness CHECK)
  fit_corpus_description TEXT,  -- why-null: only corpus_document_fit; optional description
  fit_corpus_sha256 TEXT CHECK (fit_corpus_sha256 IS NULL OR fit_corpus_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR fit_corpus_sha256 = 'unverified'),  -- why-null: only corpus_document_fit (paired-nullness CHECK)
  fit_corpus_n_documents INTEGER,  -- why-null: only corpus_document_fit; not every sidecar records it
  fit_corpus_n_tokens INTEGER,  -- why-null: only corpus_document_fit; not every sidecar records it
  fit_contrast_sets_json TEXT CHECK (fit_contrast_sets_json IS NULL OR json_valid(fit_contrast_sets_json)),  -- why-null: only contrastive_direction (paired-nullness CHECK)
  fit_model_id TEXT REFERENCES model(model_id),  -- why-null: `none` and declared-prompt objects were not fit
  fit_chat_template_sha256 TEXT CHECK (fit_chat_template_sha256 IS NULL OR fit_chat_template_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR fit_chat_template_sha256 = 'unverified'),  -- why-null: `none` and declared-prompt objects were not fit
  fit_thinking_render TEXT NOT NULL CHECK (fit_thinking_render IN ('on', 'off', 'not_applicable', 'unverified')),
  fit_objective TEXT NOT NULL CHECK (fit_objective IN ('ce', 'kl', 'diffmean', 'arditi_search', 'alphasteer', 'antidote', 'not_applicable')),
  fit_positions TEXT NOT NULL CHECK (fit_positions IN ('all', 'response_incl_assistant_header', 'last_token_of_appended_assistant_header', 'searched_post_instruction_span', 'completion_mean', 'inherited_from_parents', 'not_applicable', 'unverified')),
  fit_response_header_len INTEGER,  -- why-null: only response_incl_assistant_header positions
  fit_capture_site TEXT CHECK (fit_capture_site IN ('resid_pre', 'resid_post')),  -- why-null: only captured directions name a site
  fit_selected_position_offset INTEGER,  -- why-null: only searched positions (Arditi)
  fit_selected_token_text TEXT,  -- why-null: only searched positions (Arditi)
  fit_layers_json TEXT CHECK (fit_layers_json IS NULL OR json_valid(fit_layers_json)),  -- why-null: `none` and declared-prompt objects have no layers
  fit_layer_index_convention TEXT CHECK (fit_layer_index_convention IN ('zero_based_decoder_block', 'one_based', 'hf_module_path')),  -- why-null: `none` and declared-prompt objects have no layers
  fit_layer_count INTEGER,  -- why-null: `none` and declared-prompt objects have no layers
  fit_layer_selection_rule TEXT CHECK (fit_layer_selection_rule IN ('all', 'listed_in_config', 'separability_search', 'published_set', 'inherited')),  -- why-null: `none` and declared-prompt objects have no layers
  served_layer_shift INTEGER,  -- why-null: only objects served at a shifted layer
  fit_rank INTEGER,  -- why-null: LoRA objects only
  fit_alpha REAL,  -- why-null: LoRA objects only
  fit_target_modules_json TEXT CHECK (fit_target_modules_json IS NULL OR json_valid(fit_target_modules_json)),  -- why-null: LoRA objects only
  fit_pool_sha256 TEXT CHECK (fit_pool_sha256 IS NULL OR fit_pool_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR fit_pool_sha256 = 'unverified'),  -- why-null: not every fit names a pool
  fit_sample_manifest_sha256 TEXT CHECK (fit_sample_manifest_sha256 IS NULL OR fit_sample_manifest_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR fit_sample_manifest_sha256 = 'unverified'),  -- why-null: not every fit writes a sample manifest
  fit_completion_tokens INTEGER,  -- why-null: not every sidecar records it
  fit_heldout_fraction REAL,  -- why-null: not every fit holds out
  fit_split_seed INTEGER,  -- why-null: not every fit records a split seed
  fit_probe_sources_pinned_out INTEGER CHECK (fit_probe_sources_pinned_out IN (0, 1)),  -- why-null: not every fit records it
  fit_hyperparameters_json TEXT CHECK (fit_hyperparameters_json IS NULL OR json_valid(fit_hyperparameters_json)),  -- why-null: `none` and declared-prompt objects were not fit
  fit_lr REAL,  -- why-null: not applicable to unfitted objects
  fit_lr_schedule TEXT,  -- why-null: not applicable to unfitted objects
  fit_effective_batch INTEGER,  -- why-null: not applicable to unfitted objects
  fit_realized_batch_json TEXT CHECK (fit_realized_batch_json IS NULL OR json_valid(fit_realized_batch_json)),  -- why-null: not applicable to unfitted objects
  fit_epochs_cap INTEGER,  -- why-null: not applicable to unfitted objects
  fit_epochs_seen INTEGER,  -- why-null: not applicable to unfitted objects
  fit_best_epoch INTEGER,  -- why-null: not applicable to unfitted objects
  fit_patience INTEGER,  -- why-null: not applicable to unfitted objects
  fit_early_stopped INTEGER CHECK (fit_early_stopped IN (0, 1)),  -- why-null: not applicable to unfitted objects
  fit_stopped_on TEXT CHECK (fit_stopped_on IN ('patience', 'epoch_cap', 'unverified')),  -- why-null: not applicable to unfitted objects
  fit_steps INTEGER,  -- why-null: not applicable to unfitted objects
  fit_clip_fraction REAL,  -- why-null: not applicable to unfitted objects
  fit_final_train_loss REAL,  -- why-null: not applicable to unfitted objects
  fit_final_heldout_loss REAL,  -- why-null: not applicable to unfitted objects
  fit_best_heldout_loss REAL,  -- why-null: not applicable to unfitted objects
  fit_heldout_curve_uri TEXT,  -- why-null: not applicable to unfitted objects
  fit_seed INTEGER,  -- why-null: unknown seeds are `fit_seed_confidence = unverified`, never a guessed number
  fit_seed_confidence TEXT NOT NULL CHECK (fit_seed_confidence IN ('recorded', 'assumed_default_0', 'not_applicable', 'unverified')),
  fit_replicate_label TEXT,  -- why-null: only replicate fits carry a label
  fit_search_id TEXT REFERENCES fit_search(search_id),  -- why-null: only searched hyperparameters link a search
  fit_lr_selected_by TEXT CHECK (fit_lr_selected_by IS NULL OR fit_lr_selected_by IN ('search', 'default', 'manual', 'unverified') OR fit_lr_selected_by GLOB 'inherited_from:*'),  -- why-null: not applicable to unfitted objects
  fit_argmin_interior INTEGER CHECK (fit_argmin_interior IN (0, 1)),  -- why-null: only searched learning rates
  fit_code_sha TEXT CHECK (fit_code_sha IS NULL OR fit_code_sha GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR fit_code_sha = 'unverified'),  -- why-null: not applicable to unfitted objects
  fit_config_path TEXT,  -- why-null: not applicable to unfitted objects
  fit_config_sha256 TEXT CHECK (fit_config_sha256 IS NULL OR fit_config_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR fit_config_sha256 = 'unverified'),  -- why-null: not applicable to unfitted objects
  fit_writer TEXT,  -- why-null: not applicable to unfitted objects
  fit_started_utc TEXT CHECK (fit_started_utc IS NULL OR fit_started_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: not applicable to unfitted objects
  fit_finished_utc TEXT CHECK (fit_finished_utc IS NULL OR fit_finished_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- why-null: not applicable to unfitted objects
  fit_walltime_s REAL,  -- why-null: walltime is recorded where known, never estimated
  fit_substrate TEXT CHECK (fit_substrate IN ('cluster_a', 'cluster_b', 'modal', 'grading_host', 'runpod', 'slurm')),  -- why-null: not applicable to unfitted objects
  fit_job_id TEXT,  -- why-null: not applicable to unfitted objects
  fit_node_hostname TEXT,  -- why-null: not applicable to unfitted objects
  fit_gpu_type TEXT,  -- why-null: not applicable to unfitted objects
  fit_gpu_count INTEGER,  -- why-null: not applicable to unfitted objects
  fit_wandb_run_id TEXT,  -- why-null: not every fit logged to W&B
  fit_wandb_url TEXT,  -- why-null: not every fit logged to W&B
  fit_producer_session TEXT,  -- why-null: unknown for legacy fits
  derivation TEXT NOT NULL CHECK (derivation IN ('none', 'lora_subtraction', 'residual_distillation', 'dose_scale', 'composition', 'stack')),
  parent_object_ids_json TEXT CHECK (parent_object_ids_json IS NULL OR json_valid(parent_object_ids_json)),  -- why-null: only derived objects (paired-nullness CHECK)
  derivation_params_json TEXT CHECK (derivation_params_json IS NULL OR json_valid(derivation_params_json)),  -- why-null: only derived objects (paired-nullness CHECK)
  base_scale REAL,  -- why-null: only scaled objects
  si_text_sha256 TEXT CHECK (si_text_sha256 IS NULL OR si_text_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]') REFERENCES si_text(si_text_sha256),  -- why-null: only declared prompts (paired-nullness CHECK)
  si_token_count INTEGER,  -- why-null: only declared prompts (paired-nullness CHECK)
  artifact_sha256 TEXT NOT NULL,  -- The object's bytes (sha256); `none` for the unsteered object; the text sha for a declared prompt.
  artifact_bytes INTEGER,  -- why-null: `none` and declared-prompt objects have no artifact file
  artifact_locations_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(artifact_locations_json)),
  object_sidecars_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(object_sidecars_json)),
  status TEXT NOT NULL CHECK (status IN ('approved', 'reference', 'superseded', 'withdrawn', 'pending_fit')),
  superseded_by_object_id TEXT REFERENCES object(object_id),  -- why-null: only superseded objects
  status_ruling_id TEXT REFERENCES ruling(ruling_id),  -- why-null: set only when a ruling set the status
  field_confidence_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_confidence_json)),
  notes TEXT,  -- why-null: free text
  field_versions_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(field_versions_json)),  -- {column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- Unknown record fields, never dropped (Fixed decision 13).
  record_uri TEXT NOT NULL,  -- The record file (logical path) that produced the current row.
  ingest_seq INTEGER NOT NULL,  -- The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); derived, never allocated — every row above a head a reader has seen is new to it.
  PRIMARY KEY (object_id),
  CONSTRAINT object_fit_target_kind_pairing CHECK (
     (fit_target_kind = 'system_instruction_distillation'
         AND fit_objective IN ('kl', 'ce')
         AND derivation IN ('none')
         AND fit_teacher_si_id IS NOT NULL
         AND fit_teacher_si_text_sha256 IS NOT NULL
         AND fit_prompt_pool_id IS NOT NULL
         AND fit_prompt_pool_sha256 IS NOT NULL
         AND fit_model_id IS NOT NULL
         AND fit_layers_json IS NOT NULL
         AND fit_corpus_id IS NULL
         AND fit_corpus_sha256 IS NULL
         AND fit_corpus_n_documents IS NULL
         AND fit_corpus_n_tokens IS NULL
         AND fit_corpus_description IS NULL
         AND fit_contrast_sets_json IS NULL
         AND parent_object_ids_json IS NULL
         AND derivation_params_json IS NULL
         AND si_text_sha256 IS NULL
         AND si_token_count IS NULL)
  OR (fit_target_kind = 'corpus_document_fit'
         AND fit_objective IN ('ce')
         AND derivation IN ('none')
         AND fit_corpus_id IS NOT NULL
         AND fit_corpus_sha256 IS NOT NULL
         AND fit_model_id IS NOT NULL
         AND fit_layers_json IS NOT NULL
         AND fit_teacher_si_id IS NULL
         AND fit_teacher_si_text_sha256 IS NULL
         AND fit_prompt_pool_id IS NULL
         AND fit_prompt_pool_sha256 IS NULL
         AND fit_contrast_sets_json IS NULL
         AND parent_object_ids_json IS NULL
         AND derivation_params_json IS NULL
         AND si_text_sha256 IS NULL
         AND si_token_count IS NULL)
  OR (fit_target_kind = 'contrastive_direction'
         AND fit_objective IN ('diffmean', 'arditi_search', 'alphasteer', 'antidote')
         AND derivation IN ('none')
         AND fit_contrast_sets_json IS NOT NULL
         AND fit_model_id IS NOT NULL
         AND fit_layers_json IS NOT NULL
         AND fit_teacher_si_id IS NULL
         AND fit_teacher_si_text_sha256 IS NULL
         AND fit_prompt_pool_id IS NULL
         AND fit_prompt_pool_sha256 IS NULL
         AND fit_corpus_id IS NULL
         AND fit_corpus_sha256 IS NULL
         AND fit_corpus_n_documents IS NULL
         AND fit_corpus_n_tokens IS NULL
         AND fit_corpus_description IS NULL
         AND parent_object_ids_json IS NULL
         AND derivation_params_json IS NULL
         AND si_text_sha256 IS NULL
         AND si_token_count IS NULL)
  OR (fit_target_kind = 'derived'
         AND fit_objective IN ('not_applicable')
         AND derivation IN ('lora_subtraction', 'residual_distillation', 'dose_scale', 'composition', 'stack')
         AND parent_object_ids_json IS NOT NULL
         AND fit_teacher_si_id IS NULL
         AND fit_teacher_si_text_sha256 IS NULL
         AND fit_prompt_pool_id IS NULL
         AND fit_prompt_pool_sha256 IS NULL
         AND fit_corpus_id IS NULL
         AND fit_corpus_sha256 IS NULL
         AND fit_corpus_n_documents IS NULL
         AND fit_corpus_n_tokens IS NULL
         AND fit_corpus_description IS NULL
         AND fit_contrast_sets_json IS NULL
         AND si_text_sha256 IS NULL
         AND si_token_count IS NULL)
  OR (fit_target_kind = 'declared_prompt'
         AND fit_objective IN ('not_applicable')
         AND derivation IN ('none')
         AND object_kind = 'system_instruction'
         AND si_text_sha256 IS NOT NULL
         AND fit_teacher_si_id IS NULL
         AND fit_teacher_si_text_sha256 IS NULL
         AND fit_prompt_pool_id IS NULL
         AND fit_prompt_pool_sha256 IS NULL
         AND fit_corpus_id IS NULL
         AND fit_corpus_sha256 IS NULL
         AND fit_corpus_n_documents IS NULL
         AND fit_corpus_n_tokens IS NULL
         AND fit_corpus_description IS NULL
         AND fit_contrast_sets_json IS NULL
         AND parent_object_ids_json IS NULL
         AND derivation_params_json IS NULL)
  OR (fit_target_kind = 'none'
         AND fit_objective IN ('not_applicable')
         AND derivation IN ('none')
         AND object_kind = 'none'
         AND fit_teacher_si_id IS NULL
         AND fit_teacher_si_text_sha256 IS NULL
         AND fit_prompt_pool_id IS NULL
         AND fit_prompt_pool_sha256 IS NULL
         AND fit_corpus_id IS NULL
         AND fit_corpus_sha256 IS NULL
         AND fit_corpus_n_documents IS NULL
         AND fit_corpus_n_tokens IS NULL
         AND fit_corpus_description IS NULL
         AND fit_contrast_sets_json IS NULL
         AND parent_object_ids_json IS NULL
         AND derivation_params_json IS NULL
         AND si_text_sha256 IS NULL
         AND si_token_count IS NULL)
  )
);
INSERT INTO object__new (object_id, object_kind, paper_label, campaign, campaign_cell_id_legacy, object_role, fit_target_kind, fit_teacher_si_id, fit_teacher_si_text_sha256, fit_prompt_pool_id, fit_prompt_pool_sha256, fit_corpus_id, fit_corpus_description, fit_corpus_sha256, fit_corpus_n_documents, fit_corpus_n_tokens, fit_contrast_sets_json, fit_model_id, fit_chat_template_sha256, fit_thinking_render, fit_objective, fit_positions, fit_response_header_len, fit_capture_site, fit_selected_position_offset, fit_selected_token_text, fit_layers_json, fit_layer_index_convention, fit_layer_count, fit_layer_selection_rule, served_layer_shift, fit_rank, fit_alpha, fit_target_modules_json, fit_pool_sha256, fit_sample_manifest_sha256, fit_completion_tokens, fit_heldout_fraction, fit_split_seed, fit_probe_sources_pinned_out, fit_hyperparameters_json, fit_lr, fit_lr_schedule, fit_effective_batch, fit_realized_batch_json, fit_epochs_cap, fit_epochs_seen, fit_best_epoch, fit_patience, fit_early_stopped, fit_stopped_on, fit_steps, fit_clip_fraction, fit_final_train_loss, fit_final_heldout_loss, fit_best_heldout_loss, fit_heldout_curve_uri, fit_seed, fit_seed_confidence, fit_replicate_label, fit_search_id, fit_lr_selected_by, fit_argmin_interior, fit_code_sha, fit_config_path, fit_config_sha256, fit_writer, fit_started_utc, fit_finished_utc, fit_walltime_s, fit_substrate, fit_job_id, fit_node_hostname, fit_gpu_type, fit_gpu_count, fit_wandb_run_id, fit_wandb_url, fit_producer_session, derivation, parent_object_ids_json, derivation_params_json, base_scale, si_text_sha256, si_token_count, artifact_sha256, artifact_bytes, artifact_locations_json, object_sidecars_json, status, superseded_by_object_id, status_ruling_id, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq) SELECT object_id, object_kind, paper_label, campaign, campaign_cell_id_legacy, object_role, fit_target_kind, fit_teacher_si_id, fit_teacher_si_text_sha256, fit_prompt_pool_id, fit_prompt_pool_sha256, fit_corpus_id, fit_corpus_description, fit_corpus_sha256, fit_corpus_n_documents, fit_corpus_n_tokens, fit_contrast_sets_json, fit_model_id, fit_chat_template_sha256, fit_thinking_render, fit_objective, fit_positions, fit_response_header_len, fit_capture_site, fit_selected_position_offset, fit_selected_token_text, fit_layers_json, fit_layer_index_convention, fit_layer_count, fit_layer_selection_rule, served_layer_shift, fit_rank, fit_alpha, fit_target_modules_json, fit_pool_sha256, fit_sample_manifest_sha256, fit_completion_tokens, fit_heldout_fraction, fit_split_seed, fit_probe_sources_pinned_out, fit_hyperparameters_json, fit_lr, fit_lr_schedule, fit_effective_batch, fit_realized_batch_json, fit_epochs_cap, fit_epochs_seen, fit_best_epoch, fit_patience, fit_early_stopped, fit_stopped_on, fit_steps, fit_clip_fraction, fit_final_train_loss, fit_final_heldout_loss, fit_best_heldout_loss, fit_heldout_curve_uri, fit_seed, fit_seed_confidence, fit_replicate_label, fit_search_id, fit_lr_selected_by, fit_argmin_interior, fit_code_sha, fit_config_path, fit_config_sha256, fit_writer, fit_started_utc, fit_finished_utc, fit_walltime_s, fit_substrate, fit_job_id, fit_node_hostname, fit_gpu_type, fit_gpu_count, fit_wandb_run_id, fit_wandb_url, fit_producer_session, derivation, parent_object_ids_json, derivation_params_json, base_scale, si_text_sha256, si_token_count, artifact_sha256, artifact_bytes, artifact_locations_json, object_sidecars_json, status, superseded_by_object_id, status_ruling_id, field_confidence_json, notes, field_versions_json, extra_json, record_uri, ingest_seq FROM object ORDER BY object_id;
DROP TABLE object;
ALTER TABLE object__new RENAME TO object;
CREATE INDEX idx_object_artifact_sha ON object (artifact_sha256) WHERE artifact_sha256 <> 'none' AND artifact_sha256 NOT LIKE 'si:%';
ANALYZE object;

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
  applied_positions_mode TEXT NOT NULL CHECK (applied_positions_mode IN ('all', 'response', 'response_all_turns', 'prompt_last', 'prefill_all', 'prefill_last_broadcast', 'last', 'none', 'unverified')),
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
ANALYZE run;
