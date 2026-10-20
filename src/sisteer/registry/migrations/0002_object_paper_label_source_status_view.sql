-- 0002_object_paper_label_source_status_view.sql
-- The live database (created 2026-09-18T22:10Z from the 0001_initial.sql of that day) fell behind the code at the
-- same schema version: 0001 was edited in place twice afterwards (3286cb482: v_source_status reads past a
-- superseded refusal; cda105326: vocab.PAPER_LABELS gained 'difference_relu_distilled'). The version guard
-- saw v1 == v1, the harvester folded, and the old object.paper_label CHECK refused the relu_paired object records
-- mid-revision, holding whole source revisions (2026-09-19). This migration carries a database created from the
-- older 0001 to the code's shape: the object table is rebuilt (SQLite cannot alter a CHECK in place; 337 rows live)
-- and v_source_status is recreated. On a database created from the frozen 0001 both steps are no-ops in effect.
-- Rendered by `registry migrate --plan --sql` against the live shape; 0001_initial.sql is frozen from here on
-- (tests/test_registry_schema.py::test_landed_migrations_are_frozen) — every later schema change is a new file.
-- drift: table object: column paper_label CHECK (paper_label IN (...)) lacks 'difference_relu_distilled'
-- drift: v_source_status differs from the code's SQL

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
  fit_substrate TEXT CHECK (fit_substrate IN ('cluster_a', 'cluster_b', 'modal', 'grading_host', 'runpod')),  -- why-null: not applicable to unfitted objects
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

DROP VIEW IF EXISTS v_source_status;
CREATE VIEW v_source_status AS
WITH latest AS (
  SELECT sr.source_id, sr.revision_id, sr.accepted, sr.refusal, sr.observed_utc, sr.adapter, sr.inputs_json, sr.extra_json, sr.ingest_seq
  FROM source_revision sr
  WHERE NOT (sr.accepted = 0 AND sr.superseded_by_revision_id IS NOT NULL)
    AND NOT EXISTS (SELECT 1 FROM source_revision o WHERE o.source_id = sr.source_id
                    AND NOT (o.accepted = 0 AND o.superseded_by_revision_id IS NOT NULL)
                    AND (o.observed_utc > sr.observed_utc OR (o.observed_utc = sr.observed_utc AND o.revision_id > sr.revision_id)))
), per_source AS (
  SELECT c.lane_source_legacy, c.adapter, c.source_id, c.owner, c.where_json,
         CASE WHEN c.ok = 1 AND COALESCE(l.accepted, 1) = 1 THEN 1 ELSE 0 END AS ok,
         CASE WHEN l.accepted = 0
              THEN 'latest read REFUSED (revision ' || l.revision_id || ' at ' || l.observed_utc || '): ' || COALESCE(l.refusal, '')
                   || '; the accepted read ' || c.source_revision_id || ' stays current -- '
              ELSE '' END || c.detail AS detail,
         c.cells_measured,
         json_set(json(c.extra_json),
                  '$.refuse', json((SELECT json_group_array(x) FROM (
                       SELECT je.value AS x FROM json_each(COALESCE(json_extract(c.extra_json, '$.refuse'), '[]')) je
                       UNION ALL
                       SELECT 'latest read of ' || c.source_id || ' refused (revision ' || l.revision_id || '): ' || COALESCE(l.refusal, '')
                       WHERE l.accepted = 0))),
                  '$.torn', json(COALESCE(json_extract(c.extra_json, '$.torn'), '[]')),
                  '$.missing', json(COALESCE(json_extract(c.extra_json, '$.missing'), '[]'))) AS extra_json,
         c.source_revision_id AS revision_id, c.observed_utc, c.ingest_seq
  FROM v_source_status_current c LEFT JOIN latest l ON l.source_id = c.source_id
  UNION ALL
  SELECT je.value, l.adapter, l.source_id, COALESCE(json_extract(l.extra_json, '$.owner'), 'registry'),
         (SELECT json_group_array(k.key) FROM json_each(l.inputs_json) k), 0,
         'latest read REFUSED (revision ' || l.revision_id || ' at ' || l.observed_utc || ') and no read of this source was ever accepted: '
             || COALESCE(l.refusal, ''),
         0,
         json_object('refuse', json_array('latest read of ' || l.source_id || ' refused (revision ' || l.revision_id || '): ' || COALESCE(l.refusal, '')),
                     'torn', json_array(), 'missing', json_array()),
         l.revision_id, l.observed_utc, l.ingest_seq
  FROM latest l, json_each(COALESCE(json_extract(l.extra_json, '$.lane_sources_legacy'), '[]')) je
  WHERE l.accepted = 0 AND NOT EXISTS (SELECT 1 FROM v_source_status_current c WHERE c.source_id = l.source_id)
)
SELECT g.lane_source_legacy, g.adapter, g.owner,
       (SELECT json_group_array(u) FROM (SELECT DISTINCT je.value AS u FROM per_source p, json_each(p.where_json) je
                                         WHERE p.lane_source_legacy = g.lane_source_legacy AND p.adapter = g.adapter ORDER BY 1)) AS where_json,
       g.ok,
       (SELECT group_concat(d, ' | ') FROM (SELECT CASE WHEN g.n_sources > 1 THEN '[' || p.source_id || '] ' ELSE '' END || p.detail AS d
                                            FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy AND p.adapter = g.adapter
                                            ORDER BY p.source_id)) AS detail,
       g.cells_measured,
       CASE WHEN g.n_sources = 1
            THEN (SELECT p.extra_json FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy AND p.adapter = g.adapter)
            ELSE json_object(
                 'refuse', (SELECT json_group_array(je.value) FROM (SELECT p.extra_json FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy
                                                                     AND p.adapter = g.adapter ORDER BY p.source_id) q, json_each(q.extra_json, '$.refuse') je),
                 'torn', (SELECT json_group_array(je.value) FROM (SELECT p.extra_json FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy
                                                                   AND p.adapter = g.adapter ORDER BY p.source_id) q, json_each(q.extra_json, '$.torn') je),
                 'missing', (SELECT json_group_array(je.value) FROM (SELECT p.extra_json FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy
                                                                      AND p.adapter = g.adapter ORDER BY p.source_id) q, json_each(q.extra_json, '$.missing') je),
                 'by_source', (SELECT json_group_object(p.source_id, json(p.extra_json)) FROM per_source p
                               WHERE p.lane_source_legacy = g.lane_source_legacy AND p.adapter = g.adapter))
       END AS extra_json,
       (SELECT p.revision_id FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy AND p.adapter = g.adapter
        ORDER BY p.observed_utc DESC, p.source_id DESC LIMIT 1) AS revision_id,
       g.observed_utc, g.ingest_seq, g.n_sources,
       (SELECT json_group_array(s) FROM (SELECT p.source_id AS s FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy
                                         AND p.adapter = g.adapter ORDER BY p.source_id)) AS source_ids_json,
       (SELECT json_group_array(r) FROM (SELECT p.revision_id AS r FROM per_source p WHERE p.lane_source_legacy = g.lane_source_legacy
                                         AND p.adapter = g.adapter ORDER BY p.source_id)) AS revision_ids_json
FROM (SELECT lane_source_legacy, adapter, COUNT(*) AS n_sources, MIN(owner) AS owner, MIN(ok) AS ok, SUM(cells_measured) AS cells_measured,
             MAX(observed_utc) AS observed_utc, MAX(ingest_seq) AS ingest_seq
      FROM per_source GROUP BY lane_source_legacy, adapter) g;
