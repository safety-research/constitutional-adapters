-- 0021_run_current_frame_rules.sql
-- Six registry rulings on what the paper frame reads as CURRENT (2026-10-02), the
-- follow-up on two of them (the later-run supersession key; is_current off the roster) and a follow-up on that one (the member
-- passes of a pooled reading are current but not paper rows), as one
-- change of nine `v_*` objects and nothing else: no table moves, no row is copied, every column the frame carried before stays where
-- it was and the new columns are appended. Rendered by `registry migrate --plan --sql` against a schema-version-20 database (0001..0020
-- applied to an empty database by `registry init`); the planner reports nine recreates and no rebuild.
-- (1) `v_run_current` (new): ONE evaluation of the ruling fold. The CTE chain that was the body of `v_paper_current` (the open
--     round, the roster match, the active rulings flattened once, the runs each ruling covers, the comparisons a comparison-scoped
--     exclude strips, the decisive ruling, the later-run supersession, the withdrawn runs, the per-run aggregates) moves into
--     schema_def.selection_ctes_sql and renders here, with the two CASEs (SELECTION_STATE_SQL / IN_CURRENT_PAPER_SQL) unchanged in
--     meaning. The later-run supersession keys on (model, object, eval, instrument, judge, strength, the legacy operating point) AND,
--     since this file, token_cap_tokens, trial_index and pass_label: a run is superseded by a later complete run at the SAME token cap, trial
--     and pass only -- two token caps of one object under one seeded static instrument (per-cap registration), two trials or two
--     passes are distinct runs, never successors; every nullable term compares with IS, so a NULL pass_label pairs with NULL and every
--     supersession that held before the key grew still holds. The object adds `is_current` -- 0 when selection_state is superseded /
--     withdrawn / excluded, or held_out = 1, or lifecycle_state is failed / cancelled / superseded / withdrawn, or superseded_by_run_id
--     is set, or (the last rule in the order, so every run that was already non-current keeps its why) a decisive exclude /
--     supersede ruling covers a run whose selection_state is not_in_roster (the ruling retires the run whatever the roster says, while
--     selection_state keeps its roster-first order for the paper); 1 otherwise -- and `why`, the FIRST failing rule in that order with
--     the ruling id where a ruling decided it (`decisive ruling exclude (ruling R)` for the off-roster case; NULL on a current run).
--     Live, read-only, 2026-10-02, old body against new on the same 12,343 runs: the grown key releases 1,072 runs whose later sibling
--     at the old key differed in cap, trial or pass (7 of them had a later same-key sibling under a coexist ruling) -- 978 superseded ->
--     selected, entering the paper (`sampled_lane` 728, `relu_paired` 131, source-less 71, `r2_export` 23, `ctl_neg1` 19, `arditi_vendor_mt` 6:
--     base_cap / ladder trials 0 and 1 at 4,096 / 8,192 on the four static attack evals of the four small models, mostly), 94 superseded
--     -> excluded; in the paper 3,406 -> 4,384, selected 3,525 -> 4,503; 74 in-paper contrasts regain a same_standup_none anchor. The
--     off-roster rule reads 1,801 not_in_roster runs non-current (is_current 1: 7,103 -> 6,280; membership untouched); 1,911 contrasts
--     anchor on them, 126 under in-paper steered runs (qwen35_4b_it / gemma4e4b_it multi-turn), which lose their contrast columns.
--     The grown key keeps a pooled reading's MEMBER runs current, as it must (a trial or a cap step of one object is a distinct
--     measurement), but a current member is not a paper row: a source whose paper reading is a POOLED pass (`sampled_lane`: pooled_t0_t1;
--     `relu_paired`: pooled_t0_t1 / pooled_final_rung) files its per-trial / per-cap measurements under MEMBER pass labels, and under the
--     old key the pooled run (filed last) superseded them, which is why the renderers never saw them. selection_state gains `member_pass`
--     (vocab.SELECTION_STATES): a run whose (lane_source_legacy, pass_label) is in vocab.MEMBER_PASS_LABELS_BY_SOURCE -- sampled_lane:
--     base_cap, ladder_rung; relu_paired: base_cap, ladder_rung, unrequested_rung; rendered into both CASEs by schema_def.member_pass_sql,
--     so the SQL and the table cannot drift -- reads member_pass AFTER the superseded and excluded branches (those currency facts keep
--     reading on a member run that also carries one) and BEFORE the incomplete-lifecycle branch (an unfinished member is not a paper row
--     either way), and in_current_paper 0 right after the include branch (an include ruling still puts a member in the paper). member_pass
--     is NOT a non-current state: is_current stays 1 and why NULL for these runs. A source off the table keeps every pass in the paper:
--     ctl_neg1, seedrep_battery and seedrep_cluster file base_cap / ladder_rung with no pooled pass, so those ARE their readings
--     (replicates coexisting under the grown key; ctl_neg1's base_cap at 4,096 beside its ladder_rung at 8,192 of one trial is a cap
--     selection for the notebook, not membership). The rule lives in v_run_current alone: the registry's renderer (paper.paper_long_frame)
--     honors in_current_paper, so no renderer change rides this migration; paper.paper_frame_exclusion's per-source keep-list (sampled_lane:
--     pooled_t0_t1 alone) stays as the paper side's pinned contract (paper/paperlib/registry_source.py REGISTRY_PAPER_PIN), subsumed by
--     this rule for the registry's own rendering; the paper's pinned copy reads v_paper_long with no in_current_paper filter and is the
--     paper side's to cut over. The pooled run's `extra.member_run_ids` (sampled_static, relu_paired) is the authoritative membership
--     relation; the per-source table is the rendered rule that mirrors it, and a data-driven rule -- a run named by a current pooled run's
--     member_run_ids is a member -- is the candidate for a later migration, not this one. Live, read-only, 2026-10-02, against the body with
--     the grown key alone: see the "Invariants to hold at landing" paragraph of docs/registry/operations.md for the counts.
--     `v_paper_current` becomes a
--     projection of it beside the run display columns (its selection columns read the same values as before, the hold-out rule included: a held-out
--     run reads selected / in_current_paper 1 / held_out 1 and is_current 0), and gains `is_current` / `why`.
-- (2) `v_paper_long` joins a contrast only when its ANCHOR is current: CURRENT_CONTRAST_CTE joins
--     `paper_current.is_current = 1` on the anchor (paper_current = `SELECT * FROM v_paper_current`, MATERIALIZED, read by the frame
--     rows and by the contrast join alike, so the fold runs once per query), and carries `n_current_contrasts` = the count of current
--     contrasts at the metric key (1 on most rows, 2 where a source files two anchor kinds by design, NULL where the row joins none);
--     no tie-break. Live, read-only, 2026-10-02: 45 natively borrowed contrasts of 9 steered qwen35_4b_it multi-turn runs ride anchors
--     a later run superseded (lifecycle ingested, superseded_by NULL, no decisive ruling) and read non-current only through this rule;
--     3,938 of 30,463 frame rows carry a non-current anchor, 361 of them in the current paper -- those rows STAY in the frame (the
--     metric row stands; in_paper / selected counts do not move) with NULL contrast columns.
-- (3) A `run_metric` row whose judgment_id names a judgment with status superseded or abandoned is not current in
--     `v_paper_long` (the judgment row was already joined; one WHERE clause), and an `item_result` row likewise in
--     `v_item_result_current` and `v_item_result_provenances` (schema_def.item_judgment_live, a probe into the judgment table by
--     primary key per visible row; the rival row of the anti-join shares the key and needs none). `v_multi_source_rows` applies the
--     same predicate BEFORE ownership is aggregated (one LEFT JOIN on the judgment row): a superseded or abandoned judgment's row is in
--     no declared scope, so a re-judged owner counts its one live row at the identity (two rows would refuse the measurement) and an
--     owner whose only judgment was superseded leaves the secondary standing as the stopgap -- without it the fold still read folded
--     and the frame dropped the secondary row as well, a live measurement with no row at all. Rows with judgment_id empty or a
--     complete / partial / in_progress judgment are unaffected. Live: 7 petri_score rows of the 21 superseded qwen35_35b_a3b Petri
--     judgments were the only metric rows of any superseded or abandoned judgment (census complete 9,985 / partial 392 / superseded 31 /
--     abandoned 4); all 7 were out of the paper already (not_in_roster, held out by the operational hold-out ruling).
-- (4) `v_pending_judgment.pending_reason` tests superseded BEFORE abandoned: a run whose superseded re-judge failed and whose
--     earlier claim was released reads superseded (a fresh grade is wanted), not abandoned.
-- (5) `v_paper_current` gains `coverage_state` (vocab.COVERAGE_STATES: measured | undefined_by_geometry | absent) and
--     `coverage_detail_json`: undefined_by_geometry when json_extract(run.extra_json, $.nofit_certificate.kind) = nofit_by_design and
--     n_items_generated = 0 (the detail carries windows, prompt_tokens_min / max, min_completion, by_kind.nofit_proven, n_slots from
--     the certificate); measured when a headline metric row of the run is current -- a metric id the eval accepts as its headline
--     (HEADLINE_ACCEPTED_CTE, the same text v_item_headline reads: strongreject_bon accepts asr AND asr_bon_cap4096) that is IN THE
--     FRAME -- its observation current and admitted exactly as v_paper_long admits it (schema_def.frame_admission_sql, one text both
--     objects render: a live judgment; not a folded or refused secondary of a multi-source scope; its judge satisfying a matched
--     roster entry where the roster matches the run -- the multi-source role and state read once through a MATERIALIZED pass over
--     v_multi_source_rows / v_multi_source_check), so a run with no frame row never reads measured; absent otherwise. The certificate is
--     read first. Live: exactly one run carries a
--     certificate (sdfbase_full22:strongreject_bon:78b4699fb41a:6a1795937c79a80f, windows [4864], prompt tokens 5984-6060, 0 of 100
--     items generated, 0 metric rows) and reads undefined_by_geometry.
-- (6) `v_run_decode_policy_match` learns the recorded-by-source rule: a run that recorded NO sent parameter (and no unmodelled
--     key) and whose decode_source = generation_config is matched against the in-scope rows (its model x eval-or-category x thinking
--     mode) whose decode_source = generation_config AND whose every pinned parameter (temperature, top_p, top_k, any other non-null
--     pinned column) is declared under params_not_sent_json, and that is eligible exactly as an ordinary candidate is (review,
--     2026-10-02): its seed_rule equals the run's (or is unverified), and a superseded row counts only for a run generated before its
--     successor's effective_from_utc; one such row -> exact (policy_id stamped, candidates_json names it),
--     several -> ambiguous (the stamp pass applies its eval-over-category tie-break as to every ambiguous verdict), none -> unrecorded
--     as before. Every other verdict is byte-identical. Live, read-only: nine decode_policy rows qualify today (llama8b_it petri off /
--     not_available, gemma4e4b_it petri / multi_turn, glm46vflash petri / multi_turn, nemotron3nano4b petri / multi_turn, and the
--     superseded nemotron static row), so about 637 runs with an empty sent set move from unrecorded to exact at the first stamp pass
--     after this file lands (the brief expected none; the seeds already declare the three sampler parameters not sent); qwen35_35b_a3b
--     stays unrecorded (its rows declare only frequency_penalty / min_p / presence_penalty); the eligibility checks move no live
--     verdict today (0 of the 1,634 runs whose verdict this file changes read differently with them), so the 637 stand.
-- (7) The paper-input head follows the rules above (review, 2026-10-02: the frame changed with nothing moving the head, and
--     paper_rebuild.sh builds only when the head moved). `v_paper_inputs_head` reads the `run` table WHOLE instead of the two
--     lifecycle states the selection admits -- a run that retires (judged -> superseded / withdrawn / cancelled / failed, or
--     superseded_by_run_id set) moves its own rank while leaving those states, and (2) reads that retirement on every contrast anchored
--     on it -- and gains `judgment` as a member in its own right, because (3) removes a judgment's rows from the frame through the
--     judgment row (the fold re-stamps the run row beside it today, every judgment record carrying a judgment_state effect on its run:
--     a side effect, not a rule the head may lean on). Both are supersets of the 0020 text (the head reads >= it at every instant). The deploy itself moves no rank
--     while it changes the frame, so paper_rebuild.sh keys its watermark on (head, schema version) from this change on and rebuilds
--     once at the first tick after `registry migrate`; the paper's frame cache already keys on a hash of the schema.
-- The body below is the render from its first drift line; tests/test_registry_migrations.py::test_0021_... asserts the committed body IS
-- that render against the version-20 shape, that the plan is view-only, and that every recreated object is selectable afterwards
-- (SQLite binds a referenced object lazily, so the alphabetical body creating v_paper_current before v_run_current is fine at CREATE and
-- only a SELECT would find a missing dependency).
-- drift: v_run_decode_policy_match differs from the code's SQL
-- drift: v_item_result_current differs from the code's SQL
-- drift: v_item_result_provenances differs from the code's SQL
-- drift: v_pending_judgment differs from the code's SQL
-- drift: v_run_current is absent
-- drift: v_paper_current differs from the code's SQL
-- drift: v_multi_source_rows differs from the code's SQL
-- drift: v_paper_long differs from the code's SQL
-- drift: v_paper_inputs_head differs from the code's SQL

DROP VIEW IF EXISTS v_item_result_current;
CREATE VIEW v_item_result_current AS
SELECT ir.* FROM item_result ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id
WHERE (ir.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = ir.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND ir.withdrawn_by_record_sha256 IS NULL AND (ir.judgment_id = '' OR NOT EXISTS (SELECT 1 FROM judgment jx WHERE jx.judgment_id = ir.judgment_id AND jx.status IN ('superseded', 'abandoned')))
  AND NOT EXISTS (SELECT 1 FROM item_result o LEFT JOIN source_revision so ON so.revision_id = o.source_revision_id
                  WHERE o.run_id = ir.run_id AND o.item_id = ir.item_id AND o.metric_id = ir.metric_id AND o.trial_index = ir.trial_index AND o.sample_index = ir.sample_index AND o.turn_index = ir.turn_index AND o.judgment_id = ir.judgment_id AND o.source_revision_id <> ir.source_revision_id
                    AND (o.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = o.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND o.withdrawn_by_record_sha256 IS NULL AND (CASE WHEN o.source_revision_id = '' THEN 0 ELSE 1 END, -COALESCE(so.priority, 0), COALESCE(so.source_id, ''), o.source_revision_id) < (CASE WHEN ir.source_revision_id = '' THEN 0 ELSE 1 END, -COALESCE(sr.priority, 0), COALESCE(sr.source_id, ''), ir.source_revision_id));
DROP VIEW IF EXISTS v_item_result_provenances;
CREATE VIEW v_item_result_provenances AS
SELECT ir.*, sr.source_id, COALESCE(sr.priority, 0) AS priority
FROM item_result ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id
WHERE (ir.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = ir.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND ir.withdrawn_by_record_sha256 IS NULL AND (ir.judgment_id = '' OR NOT EXISTS (SELECT 1 FROM judgment jx WHERE jx.judgment_id = ir.judgment_id AND jx.status IN ('superseded', 'abandoned')));
DROP VIEW IF EXISTS v_multi_source_rows;
CREATE VIEW v_multi_source_rows AS
WITH withdrawn_metric AS MATERIALIZED (
    -- a metric row whose observation key is no longer current under any revision (one pass; the per-row scan does not scale)
    SELECT json_extract(observation_key, '$.run_id') AS run_id, json_extract(observation_key, '$.judgment_id') AS judgment_id,
           json_extract(observation_key, '$.metric_id') AS metric_id, json_extract(observation_key, '$.threshold') AS threshold,
           json_extract(observation_key, '$.computation_id') AS computation_id
    FROM observation WHERE table_name = 'run_metric' GROUP BY observation_key HAVING MAX(current) = 0
),
degen_rate AS MATERIALIZED (
    -- the MEASURED rates of the current degeneration rows of every run at the frac_* mirrors own (measure, scope), one row per
    -- (run_id, judgment_id); the paper objects fill a NULL mirror from here (migration 0010: a reading filed by another reader counts).
    -- No apostrophe and no bare table name in a comment here: the reference scan strips quoted literals and reads identifiers
    SELECT run_id, judgment_id,
           MAX(CASE WHEN measure = 'at_token_cap' AND scope = 'all' THEN rate END) AS at_token_cap,
           MAX(CASE WHEN measure = 'at_token_cap_finish_reason' AND scope = 'all' THEN rate END) AS at_token_cap_finish_reason,
           MAX(CASE WHEN measure = 'classifier' AND scope = 'all' THEN rate END) AS classifier,
           MAX(CASE WHEN measure = 'contentless' AND scope = 'all' THEN rate END) AS contentless,
           MAX(CASE WHEN measure = 'rep_ratio_loop' AND scope = 'all' THEN rate END) AS rep_ratio_loop,
           MAX(CASE WHEN measure = 'zlib_collapse' AND scope = 'all' THEN rate END) AS zlib_collapse,
           MAX(CASE WHEN measure = 'union5' AND scope = 'all' THEN rate END) AS union5,
           MAX(CASE WHEN measure = 'loop_or_at_cap' AND scope = 'all' THEN rate END) AS loop_or_at_cap,
           MAX(CASE WHEN measure = 'loop_first512' AND scope = 'first_512_tokens' THEN rate END) AS loop_first512,
           MAX(CASE WHEN measure = 'defended_degenerate' AND scope = 'defended_only' THEN rate END) AS defended_degenerate
    FROM v_run_degeneration_current WHERE measurement_state = 'measured' GROUP BY run_id, judgment_id
)
SELECT 'qwen35_4b_it/arditi_allpos/multi_turn' AS declaration_id, 'r2_export' AS owner_source, 'arditi_allpos' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'r2_export' THEN 'owner' ELSE 'secondary' END AS role,
       '' AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  AND r.model_id IN ('qwen35_4b_it') AND o.paper_label = 'arditi_allpos' AND e.eval_category = 'multi_turn'
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('r2_export', 'arditi_allpos')
    AND (r.operating_point_id_legacy = 'ar_c0p3__arditi_allpos' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/constitution_lora/petri/base_to_instruct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'constitution_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'base_to_instruct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'base__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/constitution_lora/petri/instruct_direct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'constitution_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'itdoc__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/control_lora/petri/base_to_instruct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'control_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'base_to_instruct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'base__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/control_lora/petri/instruct_direct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'control_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'itdoc__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/difference_lora/petri/base_to_instruct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'difference_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'base_to_instruct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'base__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0))
UNION ALL
SELECT 'llama8b_it/difference_lora/petri/instruct_direct' AS declaration_id, 'llama8b_lane' AS owner_source, 'r2_export' AS secondary_source,
       CASE WHEN r.lane_source_legacy = 'llama8b_lane' THEN 'owner' ELSE 'secondary' END AS role,
       r.producer_run_key_legacy AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS at_cap_share,
       COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share,
       CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END AS loop_share_measure,
       CASE CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END WHEN 'classifier' THEN COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) WHEN 'loop_first512' THEN COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) ELSE COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) END AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  AND r.model_id IN ('llama8b_it') AND o.paper_label = 'difference_lora' AND e.eval_category = 'petri'
    AND r.eval_id NOT IN ('petri_pooled')
    AND r.transfer_kind = 'instruct_direct'
    AND r.lane_source_legacy IN ('llama8b_lane', 'r2_export')
    AND (r.operating_point_id_legacy = 'itdoc__constv2-mix-full__lora_r4_ce' OR (r.operating_point_id_legacy IS NULL AND r.strength_nominal = 1.0));
DROP VIEW IF EXISTS v_paper_current;
CREATE VIEW v_paper_current AS
WITH accepted AS (
    -- the ACCEPTED headline metric ids of the eval in order of preference: extra_json.headline_metric_ids where the seed lists them
    -- (strongreject_bon: asr, then asr_bon_cap4096 -- the native 4,096 pass, the same per-goal any-hit unit), closed by the declared
    -- headline_metric_id (the only entry for every other eval); one row per (eval, metric id) at its best preference
    SELECT eval_id, metric_id, MIN(preference) AS preference FROM (
        SELECT e.eval_id, j.value AS metric_id, j.key AS preference
        FROM eval e, json_each(COALESCE(json_extract(e.extra_json, '$.headline_metric_ids'), '[]')) j
        UNION ALL
        SELECT e.eval_id, e.headline_metric_id, COALESCE(json_array_length(json_extract(e.extra_json, '$.headline_metric_ids')), 0)
        FROM eval e
    ) GROUP BY eval_id, metric_id
),
withdrawn_metric AS MATERIALIZED (
    -- a metric row whose observation key is no longer current under any revision (one pass; the per-row scan does not scale)
    SELECT json_extract(observation_key, '$.run_id') AS run_id, json_extract(observation_key, '$.judgment_id') AS judgment_id,
           json_extract(observation_key, '$.metric_id') AS metric_id, json_extract(observation_key, '$.threshold') AS threshold,
           json_extract(observation_key, '$.computation_id') AS computation_id
    FROM observation WHERE table_name = 'run_metric' GROUP BY observation_key HAVING MAX(current) = 0
),
multi_source AS MATERIALIZED (
    -- the role and fold state of every metric row inside a declared multi-source scope (v_multi_source_rows / v_multi_source_check),
    -- read once: the coverage test below admits a row the way the frame does -- a folded or refused secondary row is not a frame row
    SELECT msr.run_id, msr.judgment_id, msr.metric_id, msr.threshold, msr.computation_id, msr.role, msc.state
    FROM v_multi_source_rows msr
    LEFT JOIN v_multi_source_check msc ON msc.declaration_id = msr.declaration_id AND msc.pairing_key = msr.pairing_key AND msc.eval_id = msr.eval_id
         AND msc.metric_id = msr.metric_id AND msc.threshold = msr.threshold
)
SELECT r.run_id, r.campaign, r.model_id, r.object_id, r.eval_id, r.instrument_id, r.judge_id, r.lifecycle_state,
       -- 1 once the run is judged or ingested (a complete run reads `ingested` at rest, `judged` before the harvester stamps it)
       CASE WHEN r.lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete,
       r.strength_nominal, r.is_operating_point,
       -- how the run counts as being at the operating point (migration 0011; operating_point_basis_sql)
       CASE WHEN r.is_operating_point = 1 THEN 'stamped'
            WHEN r.is_operating_point IS NULL AND o.object_kind IN ('lora', 'lora_subtraction', 'composed', 'relu_steer', 'relu_steer_distilled_difference', 'alphasteer', 'static_r_arditi', 'vector', 'vector_all', 'diffmean', 'antidote')
                 AND ABS(r.strength_nominal - 1.0) < 1e-9 THEN 'fallback_nominal_1p0'
            WHEN r.is_operating_point IS NULL AND o.object_kind = 'vector_all'
                 AND ABS(r.strength_coefficient - 0.3) < 1e-9 THEN 'fallback_arditi_c0p3'
            WHEN o.object_kind IN ('none', 'system_instruction') THEN 'no_strength_axis'
            ELSE 'none' END AS operating_point_basis,
       r.ingest_seq,
       -- the decode of the run beside its selection (migration 0009): the source, the derived match, the row resolve() returns today for
       -- the key of the run, and whether the matched row is that default (decode_family_default_sql)
       r.decode_source, r.decode_policy_id, r.decode_policy_match,
       COALESCE((SELECT p.policy_id FROM decode_policy p WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id = r.eval_id AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL), (SELECT p.policy_id FROM decode_policy p JOIN eval fe ON fe.eval_id = r.eval_id WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id IS NULL AND p.eval_category = fe.eval_category AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)) AS decode_family_default_policy_id,
       CASE WHEN r.decode_policy_id IS NULL THEN NULL WHEN r.decode_policy_id = COALESCE((SELECT p.policy_id FROM decode_policy p WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id = r.eval_id AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL), (SELECT p.policy_id FROM decode_policy p JOIN eval fe ON fe.eval_id = r.eval_id WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id IS NULL AND p.eval_category = fe.eval_category AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)) THEN 1 ELSE 0 END AS decode_is_family_default,
       -- the selection (v_run_current, migration 0021: the ONE evaluation of the fold of the rulings): the open round, the comparisons and
       -- roster entries the run still carries, the rulings covering it, the decisive one, the hold-out mark, the state and the flag
       vc.round_id,
       vc.comparison_ids_json,
       vc.roster_entry_ids_json,
       vc.ruling_ids_json,
       vc.decisive_ruling_id,
       vc.decisive_action,
       vc.held_out,
       vc.selection_state,
       vc.in_current_paper,
       -- whether the run is current at all, and the first rule that retires it; a held-out row reads is_current 0 and stays in the paper
       vc.is_current,
       vc.why,
       -- coverage (migration 0021; vocab.COVERAGE_STATES): `undefined_by_geometry` when the run carries a nofit_by_design certificate
       -- and generated nothing (no prompt of its windows left room for the minimum completion: the measurement is undefined, not missing);
       -- `measured` when a headline metric row of the run is IN THE FRAME: a metric id the eval accepts as its headline (HEADLINE_ACCEPTED_CTE),
       -- its observation current, and admitted the way the frame admits it (frame_admission_sql, the text the frame renders in its own
       -- WHERE: a live judgment; not a folded or refused secondary of a multi-source scope; its judge satisfying a matched roster
       -- entry where the roster matches the run; no bare frame object name here: the reference scan reads comments) -- so a
       -- run with no frame row never reads measured (a headline row under the wrong judge, or one the owner of its scope carries instead,
       -- is not coverage); `absent` otherwise. The certificate is read first: a certified run with a stray headline row
       -- is still undefined by geometry
       CASE WHEN json_extract(r.extra_json, '$.nofit_certificate.kind') = 'nofit_by_design' AND r.n_items_generated = 0 THEN 'undefined_by_geometry'
            WHEN EXISTS (SELECT 1 FROM run_metric rm
                         JOIN accepted a ON a.eval_id = r.eval_id AND a.metric_id = rm.metric_id
                         JOIN instrument i ON i.instrument_id = r.instrument_id
                         LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
                         LEFT JOIN judge j ON j.judge_id = jm.judge_id
                         LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
                              AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
                         LEFT JOIN multi_source ms ON ms.run_id = rm.run_id AND ms.judgment_id = rm.judgment_id AND ms.metric_id = rm.metric_id
                              AND ms.threshold = rm.threshold AND ms.computation_id = rm.computation_id
                         WHERE rm.run_id = r.run_id AND wm.run_id IS NULL
                           -- a row of a judgment a re-judge displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a secondary row whose identity the owner also carries leaves the frame: folded (the owner row stands) or refused (neither is admitted)
  AND (ms.role IS NULL OR ms.role <> 'secondary' OR ms.state IS NULL OR ms.state = 'stopgap')
  -- a run the roster matches admits only a measurement whose judge satisfies a matched entry; an unmatched run admits every row
  AND (NOT EXISTS (SELECT 1 FROM v_roster_match vm WHERE vm.run_id = r.run_id)
       OR EXISTS (SELECT 1 FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
                  WHERE vm.run_id = r.run_id AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END))) THEN 'measured'
            ELSE 'absent' END AS coverage_state,
       -- the certificate behind undefined_by_geometry: the windows tried, the prompt-token range, the minimum completion, the slots proven
       -- not to fit and the slot count; NULL on every other row
       CASE WHEN json_extract(r.extra_json, '$.nofit_certificate.kind') = 'nofit_by_design' AND r.n_items_generated = 0
            THEN json_object('windows', json(json_extract(r.extra_json, '$.nofit_certificate.windows')), 'prompt_tokens_min', json_extract(r.extra_json, '$.nofit_certificate.prompt_tokens_min'),
                             'prompt_tokens_max', json_extract(r.extra_json, '$.nofit_certificate.prompt_tokens_max'), 'min_completion', json_extract(r.extra_json, '$.nofit_certificate.min_completion'),
                             'nofit_proven', json_extract(r.extra_json, '$.nofit_certificate.by_kind.nofit_proven'), 'n_slots', json_extract(r.extra_json, '$.nofit_certificate.n_slots')) END AS coverage_detail_json
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN v_run_current vc ON vc.run_id = r.run_id;
DROP VIEW IF EXISTS v_paper_inputs_head;
CREATE VIEW v_paper_inputs_head AS
SELECT MAX(seq) AS ingest_seq FROM (
    -- the base tables an in-paper row of the paper frame draws its ingest_seq from, read whole (migration 0015): a superset of the
    -- frame member it replaces -- the head moves at least whenever that did, and also when a run outside the frame lands -- never a
    -- subset. The run member reads every row (migration 0021): a run that retires leaves the two states the frame admits, and the
    -- frame reads that retirement on its own rows and on every contrast anchored on it, so the two-state member of
    -- migration 0015 missed exactly the move it had to announce; a metric row of any run can reach an in-paper row through the
    -- multi-source fold, so the metric and contrast members read every row too
    SELECT MAX(ingest_seq) AS seq FROM run
    UNION ALL SELECT MAX(ingest_seq) FROM run_metric
    UNION ALL SELECT MAX(ingest_seq) FROM run_contrast
    -- a judgment re-filed superseded or abandoned (migration 0021) takes its metric and item rows out of the frame while moving
    -- no metric or item rank; the fold re-stamps the run row beside the judgment row (a side effect, not a rule), so the judgment table
    -- is a member in its own right and the head does not lean on that side effect
    UNION ALL SELECT MAX(ingest_seq) FROM judgment
    UNION ALL SELECT MAX(ingest_seq) FROM roster_entry
    UNION ALL SELECT MAX(ingest_seq) FROM ruling
    UNION ALL SELECT MAX(ingest_seq) FROM ruling_state
    UNION ALL SELECT MAX(ingest_seq) FROM disposition
    UNION ALL SELECT MAX(ingest_seq) FROM source_revision WHERE accepted = 1
    UNION ALL SELECT MAX(ingest_seq) FROM withdrawal
    -- a producer's item withdrawals (migration 0020): a small run-scale table whose rank moves when one folds -- the marks it
    -- leaves on item_result rows do not move those rows' ingest_seq, so the table itself is the member
    UNION ALL SELECT MAX(ingest_seq) FROM item_withdrawal
    -- and the native item_result row each item_withdrawal key names, sought by primary key from the withdrawal table (CROSS JOIN fixes
    -- that loop order: without statistics the planner drove a plain join from item_result through idx_item_result_revision on
    -- source_revision_id = '', a walk of every native row). A row RE-FILED above its withdrawal clears the marks and is current again
    -- at a rank no other member reads; this member carries it (a row re-filed below the withdrawal, marked as it stays, moves the head
    -- too: a rebuild that reads an unchanged frame, the superset price)
    UNION ALL SELECT MAX(ir.ingest_seq) FROM item_withdrawal iw CROSS JOIN item_result ir
        ON ir.run_id = iw.run_id AND ir.item_id = iw.item_id AND ir.metric_id = iw.metric_id AND ir.trial_index = iw.trial_index
        AND ir.sample_index = iw.sample_index AND ir.turn_index = iw.turn_index AND ir.judgment_id = iw.judgment_id
        AND ir.source_revision_id = ''
    -- one seek through the partial index on ingest_seq over the withdrawn rows, named so the plan holds whatever the statistics say
    UNION ALL SELECT MAX(ingest_seq) FROM item_result INDEXED BY idx_item_result_withdrawn_seq WHERE withdrawn_by_revision_id IS NOT NULL
);
DROP VIEW IF EXISTS v_paper_long;
CREATE VIEW v_paper_long AS
WITH paper_current AS MATERIALIZED (
    -- the selection of every run, evaluated once for the frame (migration 0021): the rows read it as pc, the contrast join reads the
    -- currency of each anchor from the same pass
    SELECT * FROM v_paper_current
),
withdrawn_metric AS MATERIALIZED (
    -- a metric row whose observation key is no longer current under any revision (one pass; the per-row scan does not scale)
    SELECT json_extract(observation_key, '$.run_id') AS run_id, json_extract(observation_key, '$.judgment_id') AS judgment_id,
           json_extract(observation_key, '$.metric_id') AS metric_id, json_extract(observation_key, '$.threshold') AS threshold,
           json_extract(observation_key, '$.computation_id') AS computation_id
    FROM observation WHERE table_name = 'run_metric' GROUP BY observation_key HAVING MAX(current) = 0
),
degen_rate AS MATERIALIZED (
    -- the MEASURED rates of the current degeneration rows of every run at the frac_* mirrors own (measure, scope), one row per
    -- (run_id, judgment_id); the paper objects fill a NULL mirror from here (migration 0010: a reading filed by another reader counts).
    -- No apostrophe and no bare table name in a comment here: the reference scan strips quoted literals and reads identifiers
    SELECT run_id, judgment_id,
           MAX(CASE WHEN measure = 'at_token_cap' AND scope = 'all' THEN rate END) AS at_token_cap,
           MAX(CASE WHEN measure = 'at_token_cap_finish_reason' AND scope = 'all' THEN rate END) AS at_token_cap_finish_reason,
           MAX(CASE WHEN measure = 'classifier' AND scope = 'all' THEN rate END) AS classifier,
           MAX(CASE WHEN measure = 'contentless' AND scope = 'all' THEN rate END) AS contentless,
           MAX(CASE WHEN measure = 'rep_ratio_loop' AND scope = 'all' THEN rate END) AS rep_ratio_loop,
           MAX(CASE WHEN measure = 'zlib_collapse' AND scope = 'all' THEN rate END) AS zlib_collapse,
           MAX(CASE WHEN measure = 'union5' AND scope = 'all' THEN rate END) AS union5,
           MAX(CASE WHEN measure = 'loop_or_at_cap' AND scope = 'all' THEN rate END) AS loop_or_at_cap,
           MAX(CASE WHEN measure = 'loop_first512' AND scope = 'first_512_tokens' THEN rate END) AS loop_first512,
           MAX(CASE WHEN measure = 'defended_degenerate' AND scope = 'defended_only' THEN rate END) AS defended_degenerate
    FROM v_run_degeneration_current WHERE measurement_state = 'measured' GROUP BY run_id, judgment_id
),
metric_source AS MATERIALIZED (
    -- the source whose head revision carries each run_metric key (NULL for a natively emitted row); two sources carrying one key
    -- resolve by source rank (priority, then source id)
    SELECT run_id, judgment_id, metric_id, threshold, computation_id, source_id, priority FROM (
        SELECT CAST(json_extract(ob.observation_key, '$.run_id') AS TEXT) AS run_id,
               CAST(json_extract(ob.observation_key, '$.judgment_id') AS TEXT) AS judgment_id,
               CAST(json_extract(ob.observation_key, '$.metric_id') AS TEXT) AS metric_id,
               CAST(json_extract(ob.observation_key, '$.threshold') AS REAL) AS threshold,
               CAST(json_extract(ob.observation_key, '$.computation_id') AS TEXT) AS computation_id, ob.source_id, COALESCE(sr.priority, 0) AS priority,
               ROW_NUMBER() OVER (PARTITION BY ob.observation_key ORDER BY COALESCE(sr.priority, 0) DESC, ob.source_id) AS rk
        FROM observation ob LEFT JOIN source_revision sr ON sr.revision_id = ob.revision_id
        WHERE ob.table_name = 'run_metric' AND ob.current = 1
    ) WHERE rk = 1
),
run_source AS MATERIALIZED (
    -- the source that OWNS the run: the top-ranked source among those whose head revision carries the run key (NULL for a run no
    -- adapter carries)
    SELECT run_id, source_id FROM (
        SELECT CAST(json_extract(ob.observation_key, '$.run_id') AS TEXT) AS run_id, ob.source_id,
               ROW_NUMBER() OVER (PARTITION BY ob.observation_key ORDER BY COALESCE(sr.priority, 0) DESC, ob.source_id) AS rk
        FROM observation ob LEFT JOIN source_revision sr ON sr.revision_id = ob.revision_id
        WHERE ob.table_name = 'run' AND ob.current = 1
    ) WHERE rk = 1
),
metric_role AS MATERIALIZED (
    -- ONE owner row per (run, judgment, metric, threshold): the row of the source that owns the run first, then a computed row (a non-blank
    -- computation_id) before a producer-filed blank one (the twin the stats readers file beside an export row), then the higher declared priority,
    -- the lexically smaller source id, the smaller computation_id — a total order, so every other row at the identity is secondary
    SELECT rm.run_id, rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, ms.source_id AS metric_source_id, rs.source_id AS run_source_id,
           ROW_NUMBER() OVER (PARTITION BY rm.run_id, rm.judgment_id, rm.metric_id, rm.threshold
                              ORDER BY CASE WHEN ms.source_id IS rs.source_id THEN 0 ELSE 1 END, CASE WHEN rm.computation_id = '' THEN 1 ELSE 0 END,
                                       COALESCE(ms.priority, 0) DESC, COALESCE(ms.source_id, ''), rm.computation_id) AS rk,
           COUNT(*) OVER (PARTITION BY rm.run_id, rm.judgment_id, rm.metric_id, rm.threshold) AS n_candidates
    FROM run_metric rm
    LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
    LEFT JOIN metric_source ms ON ms.run_id = rm.run_id AND ms.judgment_id = rm.judgment_id AND ms.metric_id = rm.metric_id
         AND ms.threshold = rm.threshold AND ms.computation_id = rm.computation_id
    LEFT JOIN run_source rs ON rs.run_id = rm.run_id
    WHERE wm.run_id IS NULL
),
metric_role_contest AS MATERIALIZED (
    -- the tie the rank resolves by priority: two or more computed rows (non-blank computation_id) from different sources at one identity
    SELECT run_id, judgment_id, metric_id, threshold,
           COUNT(DISTINCT CASE WHEN computation_id <> '' THEN COALESCE(metric_source_id, '') END) AS n_computed_sources
    FROM metric_role GROUP BY run_id, judgment_id, metric_id, threshold
),
withdrawn_contrast AS MATERIALIZED (
    -- a contrast whose observation key is no longer current under any revision: the anchor no head revision pairs the run with (0017)
    SELECT CAST(json_extract(observation_key, '$.run_id') AS TEXT) AS run_id, CAST(json_extract(observation_key, '$.anchor_run_id') AS TEXT) AS anchor_run_id,
           CAST(json_extract(observation_key, '$.judgment_id') AS TEXT) AS judgment_id, CAST(json_extract(observation_key, '$.metric_id') AS TEXT) AS metric_id,
           CAST(json_extract(observation_key, '$.threshold') AS REAL) AS threshold, CAST(json_extract(observation_key, '$.computation_id') AS TEXT) AS computation_id
    FROM observation WHERE table_name = 'run_contrast' GROUP BY observation_key HAVING MAX(current) = 0
),
current_contrast AS MATERIALIZED (
    -- the contrasts the frame joins: every run_contrast row whose observation key is still current under some revision (or was never
    -- observed); the row of an anchor only a superseded revision paired the run with is left out, so a metric row joins each anchor once (0017);
    -- and whose ANCHOR is current (migration 0021): paper_current.is_current reads 1 for it -- an excluded baseline, one a
    -- later run at its key superseded, one its lifecycle retired or one held out differences nothing. n_current_contrasts counts the
    -- current contrasts at the metric key of the row (1 on most rows, 2 where a source files two anchor kinds by design); no tie-break
    SELECT rc.*, COUNT(*) OVER (PARTITION BY rc.run_id, rc.judgment_id, rc.metric_id, rc.threshold, rc.computation_id) AS n_current_contrasts
    FROM run_contrast rc
    JOIN paper_current ac ON ac.run_id = rc.anchor_run_id AND ac.is_current = 1
    LEFT JOIN withdrawn_contrast wc ON wc.run_id = rc.run_id AND wc.anchor_run_id = rc.anchor_run_id AND wc.judgment_id = rc.judgment_id
         AND wc.metric_id = rc.metric_id AND wc.threshold = rc.threshold AND wc.computation_id = rc.computation_id
    WHERE wc.run_id IS NULL
)
SELECT pc.round_id,
       -- the flag of the run on its OWNER metric row only: a secondary row at the same identity is evidence, never a second paper row
       CASE WHEN mr.rk = 1 THEN pc.in_current_paper ELSE 0 END AS in_current_paper,
       pc.selection_state, pc.held_out,
       -- per MEASUREMENT: the comparisons / roster entries the judge of this row satisfies, among the entries the run
       -- still carries (a comparison-scoped exclude removed the others from the run in v_paper_current, never the row from the paper)
       (SELECT json_group_array(DISTINCT vm.comparison_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END) AS comparison_ids_json,
       (SELECT json_group_array(DISTINCT vm.roster_entry_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END) AS roster_entry_ids_json,
       pc.decisive_ruling_id, pc.decisive_action,
       r.run_id, r.campaign, r.lane_source_legacy, r.view_legacy, r.campaign_cell_id_legacy, r.operating_point_id_legacy,
       r.producer_run_key_legacy, r.producer_object_key_legacy,
       r.model_id, m.family AS model_family, r.transfer_kind, r.strength_nominal, r.strength_coefficient, r.is_operating_point,
       -- how the run counts as being at the operating point (migration 0011; operating_point_basis_sql)
       CASE WHEN r.is_operating_point = 1 THEN 'stamped'
            WHEN r.is_operating_point IS NULL AND o.object_kind IN ('lora', 'lora_subtraction', 'composed', 'relu_steer', 'relu_steer_distilled_difference', 'alphasteer', 'static_r_arditi', 'vector', 'vector_all', 'diffmean', 'antidote')
                 AND ABS(r.strength_nominal - 1.0) < 1e-9 THEN 'fallback_nominal_1p0'
            WHEN r.is_operating_point IS NULL AND o.object_kind = 'vector_all'
                 AND ABS(r.strength_coefficient - 0.3) < 1e-9 THEN 'fallback_arditi_c0p3'
            WHEN o.object_kind IN ('none', 'system_instruction') THEN 'no_strength_axis'
            ELSE 'none' END AS operating_point_basis,
       r.object_id, o.paper_label, o.object_role, o.object_kind, o.fit_model_id, o.fit_layers_json, o.fit_layer_count, o.fit_replicate_label,
       o.status AS object_status, o.artifact_sha256 AS object_artifact_sha256,
       r.eval_id, e.eval_category, e.headline_metric_id, r.instrument_id, r.item_set_id, r.harness, r.n_shot, r.cot_prompting, r.thinking_mode,
       i.bon_n, i.n_trials, i.max_turns,
       r.token_cap_tokens, r.token_cap_rung_index, r.context_window_tokens,
       -- the prompt padding of the instrument beside the cap and the window (migration 0012): NULL is unpadded; a padded run is
       -- another instrument (its cap and window differ too), so a consumer grouping by instrument never averages a padded row into an unpadded one
       i.padding_tokens, i.padding_identity_sha256,
       r.serving_path, r.serving_config_sha256,
       r.serving_identity, r.serving_engine_version,
       r.decode_source, r.decode_sent_json, r.decode_policy_id, r.decode_policy_match,
       -- the decode_policy row resolve() returns TODAY for the key of the run, and whether the matched row IS it (migration 0009: the
       -- vendor preset is the default of every family): `exact` beside decode_is_family_default 0 reads "matched
       -- a recorded, NON-default policy" (the agentic runs at the T = 1.0 of the runner), never a mismatch; NULL when nothing matched
       COALESCE((SELECT p.policy_id FROM decode_policy p WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id = r.eval_id AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL), (SELECT p.policy_id FROM decode_policy p JOIN eval fe ON fe.eval_id = r.eval_id WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id IS NULL AND p.eval_category = fe.eval_category AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)) AS decode_family_default_policy_id,
       CASE WHEN r.decode_policy_id IS NULL THEN NULL WHEN r.decode_policy_id = COALESCE((SELECT p.policy_id FROM decode_policy p WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id = r.eval_id AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL), (SELECT p.policy_id FROM decode_policy p JOIN eval fe ON fe.eval_id = r.eval_id WHERE p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode AND p.eval_id IS NULL AND p.eval_category = fe.eval_category AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)) THEN 1 ELSE 0 END AS decode_is_family_default,
       jm.judge_id, j.judge_model, j.rubric_name AS judge_rubric_name, r.judge_id AS run_judge_id,
       jm.n_judge_refusals AS judgment_n_judge_refusals,
       r.pass_label, r.lifecycle_state, CASE WHEN r.lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete, r.store_uri, r.intervention_hash, r.notes, r.extra_json AS run_extra_json,
       COALESCE(json_extract(rm.extra_json, '$.eval_id_legacy'), json_extract(r.extra_json, '$.eval_id_legacy')) AS eval_id_legacy,
       rm.judgment_id, rm.metric_id, rm.producer_metric_spelling, rm.threshold, rm.computation_id,
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.value END AS value, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.k END AS k, CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE rm.n END AS n,
       -- the interval the writer filed, else the 95 percent Wilson score interval of k over n derived here (migration 0019;
       -- WILSON_CI_LO_SQL / WILSON_CI_HI_SQL / WILSON_CI_METHOD_SQL): a filed bound is returned as filed, byte for byte, under its filed
       -- method; a row of a metric on the yes-or-no roster (WILSON_PROPORTION_METRIC_IDS: k counts items with a yes-or-no outcome, so k
       -- over n is a Bernoulli proportion and the Wilson interval is its interval) with k, n above zero, no filed bound and a value that
       -- IS k over n (the scale check: a percent-scale value or a difference beside some other count is not) reads the derived bounds
       -- under the method wilson_95; a metric off the roster (a graded mean whose k is a sum of 0, 0.5 and 1 scores, whatever its value),
       -- k alone, n alone, n = 0 or a value that is not k over n read the filed columns, NULL included
       -- (no apostrophe in this comment: the reference scan strips quoted literals)
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE CASE WHEN rm.ci_lo IS NOT NULL THEN rm.ci_lo WHEN rm.ci_lo IS NULL AND rm.ci_hi IS NULL AND rm.metric_id IN ('asr', 'asr_held_in', 'asr_held_out', 'asr_at_n1', 'asr_at_n2', 'asr_at_n4', 'asr_at_n8', 'asr_at_n16', 'asr_at_n32', 'asr_at_n64', 'asr_at_n75', 'asr_bon_cap4096', 'asr_bon_cap8192', 'acc', 'xst_refused', 'orb_refused', 'at_cap_share_bon_512', 'coherent_at_cap_share_bon_512', 'loop_share_bon_512', 'secondary_flag_only_share_bon_512', 'residual_at_cap_share_bon_4096') AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= 1e-9 THEN ((rm.k*1.0/rm.n) + 1.959963984540054*1.959963984540054/(2.0*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) - 1.959963984540054*sqrt((rm.k*1.0/rm.n)*(1.0 - rm.k*1.0/rm.n)/rm.n + 1.959963984540054*1.959963984540054/(4.0*rm.n*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) ELSE NULL END END AS ci_lo,
       CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE CASE WHEN rm.ci_hi IS NOT NULL THEN rm.ci_hi WHEN rm.ci_lo IS NULL AND rm.ci_hi IS NULL AND rm.metric_id IN ('asr', 'asr_held_in', 'asr_held_out', 'asr_at_n1', 'asr_at_n2', 'asr_at_n4', 'asr_at_n8', 'asr_at_n16', 'asr_at_n32', 'asr_at_n64', 'asr_at_n75', 'asr_bon_cap4096', 'asr_bon_cap8192', 'acc', 'xst_refused', 'orb_refused', 'at_cap_share_bon_512', 'coherent_at_cap_share_bon_512', 'loop_share_bon_512', 'secondary_flag_only_share_bon_512', 'residual_at_cap_share_bon_4096') AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= 1e-9 THEN ((rm.k*1.0/rm.n) + 1.959963984540054*1.959963984540054/(2.0*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) + 1.959963984540054*sqrt((rm.k*1.0/rm.n)*(1.0 - rm.k*1.0/rm.n)/rm.n + 1.959963984540054*1.959963984540054/(4.0*rm.n*rm.n))/(1.0 + 1.959963984540054*1.959963984540054/rm.n) ELSE NULL END END AS ci_hi,
       CASE WHEN rm.ci_lo IS NULL AND rm.ci_hi IS NULL AND rm.metric_id IN ('asr', 'asr_held_in', 'asr_held_out', 'asr_at_n1', 'asr_at_n2', 'asr_at_n4', 'asr_at_n8', 'asr_at_n16', 'asr_at_n32', 'asr_at_n64', 'asr_at_n75', 'asr_bon_cap4096', 'asr_bon_cap8192', 'acc', 'xst_refused', 'orb_refused', 'at_cap_share_bon_512', 'coherent_at_cap_share_bon_512', 'loop_share_bon_512', 'secondary_flag_only_share_bon_512', 'residual_at_cap_share_bon_4096') AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= 1e-9 THEN 'wilson_95' ELSE rm.ci_method END AS ci_method,
       -- the frac_* mirror the writer filed, else the MEASURED degeneration rate of the run at the measure and scope of that mirror
       -- (migration 0010: the exact-judgment row first, then the judgment-less rows); a mirrored value still wins where a writer filed one
       COALESCE(rm.frac_at_token_cap, dj.at_token_cap, dn.at_token_cap, dj.at_token_cap_finish_reason, dn.at_token_cap_finish_reason) AS frac_at_token_cap, COALESCE(rm.frac_degenerate_classifier, dj.classifier, dn.classifier) AS frac_degenerate_classifier, COALESCE(rm.frac_degenerate_contentless, dj.contentless, dn.contentless) AS frac_degenerate_contentless, COALESCE(rm.frac_degenerate_rep_ratio_loop, dj.rep_ratio_loop, dn.rep_ratio_loop) AS frac_degenerate_rep_ratio_loop, COALESCE(rm.frac_degenerate_zlib_collapse, dj.zlib_collapse, dn.zlib_collapse) AS frac_degenerate_zlib_collapse, COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5) AS frac_degenerate_union5, COALESCE(rm.frac_loop_first512, dj.loop_first512, dn.loop_first512) AS frac_loop_first512, COALESCE(rm.frac_defended_degenerate, dj.defended_degenerate, dn.defended_degenerate) AS frac_defended_degenerate,
       rm.n_abstentions, rm.n_error_outcomes, rm.n_incomplete, rm.source_uri,
       rm.source_producers_json, rm.extra_json AS metric_extra_json,
       rc.anchor_run_id, rc.anchor_kind, rc.delta, rc.delta_lo, rc.delta_hi, rc.delta_p, rc.delta_method, rc.effect_size,
       rc.effect_kind, rc.n_pairs, rc.extra_json AS contrast_extra_json,
       -- the token cap of the anchor run beside the cap of the run itself, and whether the two agree (migration 0006): a contrast
       -- differences the values of two runs, and a consumer reading `delta` here never joined both caps, so a pair whose caps
       -- differ was invisible (202 of 8,640 live contrasts paired different caps, 2026-09-21, every one an escalation-ladder pair,
       -- most often a cap-8192 run against a cap-4096 or cap-2048 anchor). 1 when the caps are equal, 0 when they differ, NULL
       -- when the row carries no anchor or either cap is NULL
       ra.token_cap_tokens AS anchor_token_cap_tokens,
       CASE WHEN r.token_cap_tokens IS NULL OR ra.token_cap_tokens IS NULL THEN NULL
            WHEN r.token_cap_tokens = ra.token_cap_tokens THEN 1 ELSE 0 END AS cap_matched,
       -- how many CURRENT contrasts stand at the metric key of the row (current_contrast, migration 0021): 1 on most rows, 2 where
       -- a source files two anchor kinds by design (the half-dose pair of the export), NULL where the row joins no current contrast (its
       -- anchor is not current, or every contrast of the key is withdrawn); no tie-break -- a consumer picks by anchor_kind
       rc.n_current_contrasts,
       -- multi-source ownership (ownership.MULTI_SOURCE_OWNERSHIP): the role of the row inside a declared scope and the outcome of the
       -- fold; NULL everywhere outside a declared scope. An owner row reads the state of the matched identity (folded | refused; NULL
       -- when the secondary carries no row at that identity); a secondary row that reaches here is a stopgap (folded and refused ones
       -- are dropped below)
       msr.declaration_id AS multi_source_declaration_id, msr.role AS multi_source_role,
       CASE WHEN msr.role = 'owner' THEN msc.state WHEN msr.role = 'secondary' THEN 'stopgap' END AS multi_source_state,
       msc.reason AS multi_source_reason,
       CASE WHEN msr.role = 'owner' THEN msc.secondary_run_id END AS multi_source_secondary_run_id,
       CASE WHEN msr.role = 'owner' THEN msc.carried_json END AS multi_source_carried_json,
       CASE WHEN msr.role = 'owner' THEN msc.degenerate_share_secondary END AS multi_source_degenerate_share,
       CASE WHEN msr.role = 'owner' THEN msc.loop_share_secondary END AS multi_source_loop_share,
       CASE WHEN msr.role = 'owner' THEN msc.loop_share_measure_secondary END AS multi_source_loop_share_measure,
       -- metric role (METRIC_ROLE_CTES): the ONE row per (run, judgment, metric, threshold) the paper counts (owner) against the other
       -- provenances of the same measurement kept as evidence (secondary, in_current_paper 0); the source of the row and the source that owns the run
       -- (NULL = natively emitted / no adapter carries the run); how many rows stood at the identity; whether two or more of them were
       -- computed rows (non-blank computation_id) from different sources (the tie resolved by declared priority, then source id)
       CASE WHEN mr.rk = 1 THEN 'owner' ELSE 'secondary' END AS metric_role,
       mr.metric_source_id, mr.run_source_id,
       mr.n_candidates AS metric_role_candidates,
       CASE WHEN mc.n_computed_sources > 1 THEN 1 ELSE 0 END AS metric_role_contested,
       MAX(r.ingest_seq, rm.ingest_seq, COALESCE(rc.ingest_seq, 0), COALESCE(msc.ingest_seq, 0)) AS ingest_seq
FROM paper_current pc
JOIN run r ON r.run_id = pc.run_id
JOIN model m ON m.model_id = r.model_id
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN instrument i ON i.instrument_id = r.instrument_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN judge j ON j.judge_id = jm.judge_id
-- the CURRENT contrasts of the metric row (current_contrast, migration 0017): one row per anchor some head revision still pairs the run
-- with; a contrast only a superseded revision emitted is left out, so a re-anchored run no longer appears once per anchor; since
-- migration 0021 only an anchor that is itself current (paper_current.is_current) is joined
LEFT JOIN current_contrast rc ON rc.run_id = rm.run_id AND rc.judgment_id = rm.judgment_id AND rc.metric_id = rm.metric_id
     AND rc.threshold = rm.threshold AND rc.computation_id = rm.computation_id
LEFT JOIN run ra ON ra.run_id = rc.anchor_run_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
LEFT JOIN v_multi_source_rows msr ON msr.run_id = rm.run_id AND msr.judgment_id = rm.judgment_id AND msr.metric_id = rm.metric_id
     AND msr.threshold = rm.threshold AND msr.computation_id = rm.computation_id
LEFT JOIN v_multi_source_check msc ON msc.declaration_id = msr.declaration_id AND msc.pairing_key = msr.pairing_key AND msc.eval_id = msr.eval_id
     AND msc.metric_id = msr.metric_id AND msc.threshold = msr.threshold
JOIN metric_role mr ON mr.run_id = rm.run_id AND mr.judgment_id = rm.judgment_id AND mr.metric_id = rm.metric_id
     AND mr.threshold = rm.threshold AND mr.computation_id = rm.computation_id
JOIN metric_role_contest mc ON mc.run_id = rm.run_id AND mc.judgment_id = rm.judgment_id AND mc.metric_id = rm.metric_id AND mc.threshold = rm.threshold
WHERE wm.run_id IS NULL
  -- the admission of the row to the frame (frame_admission_sql, the text the coverage test of v_paper_current reads too)
  -- a row of a judgment a re-judge displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a secondary row whose identity the owner also carries leaves the frame: folded (the owner row stands) or refused (neither is admitted)
  AND (msr.role IS NULL OR msr.role <> 'secondary' OR msc.state IS NULL OR msc.state = 'stopgap')
  -- a run the roster matches admits only a measurement whose judge satisfies a matched entry; an unmatched run admits every row
  AND (NOT EXISTS (SELECT 1 FROM v_roster_match vm WHERE vm.run_id = r.run_id)
       OR EXISTS (SELECT 1 FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
                  WHERE vm.run_id = r.run_id AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0))
                        END));
DROP VIEW IF EXISTS v_pending_judgment;
CREATE VIEW v_pending_judgment AS
WITH cov AS (
    SELECT r.run_id, COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)) AS required_coverage FROM run r
)
SELECT r.run_id, r.producer_run_key_legacy, r.producer_object_key_legacy, r.store_uri, r.n_items_generated, r.campaign,
       r.model_id, r.object_id, r.eval_id, e.eval_category, r.instrument_id, i.harness, i.decode_source,
       r.lifecycle_state, r.judge_id AS run_judge_id, jd.judge_id AS required_judge_id,
       CASE WHEN i.judge_id = jd.judge_id THEN 1 ELSE 0 END AS is_instrument_judge,
       cov.required_coverage,
       (SELECT json_group_array(json_object('judgment_id', j2.judgment_id, 'judge_id', j2.judge_id, 'status', j2.status,
                                            'n_items_judged', j2.n_items_judged, 'n_judge_refusals', j2.n_judge_refusals,
                                            'n_errors', j2.n_errors, 'claimed_by_session', j2.claimed_by_session))
          FROM judgment j2 WHERE j2.run_id = r.run_id) AS existing_judgments_json,
       cl.judgment_id AS claim_judgment_id, cl.claimed_by_session AS claim_session, cl.claimed_utc AS claim_utc,
       CAST((julianday('now') - julianday(cl.claimed_utc)) * 86400 AS INTEGER) AS claim_age_s,
       (SELECT MAX(COALESCE(cov.required_coverage, 0) - (COALESCE(j4.n_items_judged, 0) + COALESCE(j4.n_judge_refusals, 0) + COALESCE(j4.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = j4.run_id AND rm_err.judgment_id = j4.judgment_id), 0)), 0)
          FROM judgment j4 WHERE j4.run_id = r.run_id AND j4.judge_id = jd.judge_id AND j4.status IN ('partial', 'complete')) AS items_missing,
       (SELECT COUNT(*) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id AND j5.status = 'abandoned') AS n_abandoned,
       (SELECT MAX(j5.claimed_utc) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id
          AND j5.status = 'abandoned') AS last_abandoned_utc,
       (SELECT COUNT(*) FROM judgment j6 WHERE j6.run_id = r.run_id AND j6.judge_id <> jd.judge_id
          AND j6.status = 'complete') AS n_complete_other_judges,
       CASE WHEN cl.judgment_id IS NOT NULL THEN 'claimed'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'partial') THEN 'partial'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'complete') THEN 'coverage_short'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'superseded') THEN 'superseded'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'abandoned') THEN 'abandoned'
            WHEN EXISTS (SELECT 1 FROM judgment j8 WHERE j8.run_id = r.run_id AND j8.judge_id <> jd.judge_id)
                 THEN 'awaiting_second_judge'
            ELSE 'no_judgment' END AS pending_reason
FROM run r
JOIN cov ON cov.run_id = r.run_id
JOIN instrument i ON i.instrument_id = r.instrument_id
JOIN eval e ON e.eval_id = r.eval_id
-- a judge pairs with a run of an instrument that names a judge, or with a run it has filed a judgment for
JOIN judge jd ON (i.judge_id IS NOT NULL
                  OR EXISTS (SELECT 1 FROM judgment jj WHERE jj.run_id = r.run_id AND jj.judge_id = jd.judge_id))
LEFT JOIN judgment cl ON cl.judgment_id = (
        SELECT j3.judgment_id FROM judgment j3 WHERE j3.run_id = r.run_id AND j3.judge_id = jd.judge_id
          AND j3.status = 'in_progress' ORDER BY j3.claimed_utc DESC LIMIT 1)
WHERE r.lifecycle_state IN ('generated', 'judging', 'judged', 'ingested')
  AND NOT EXISTS (
        SELECT 1 FROM judgment jc
        WHERE jc.run_id = r.run_id AND jc.judge_id = jd.judge_id AND jc.status = 'complete' AND COALESCE(jc.n_items_judged, 0) + COALESCE(jc.n_judge_refusals, 0) + COALESCE(jc.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jc.run_id AND rm_err.judgment_id = jc.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0));
DROP VIEW IF EXISTS v_run_current;
CREATE VIEW v_run_current AS
WITH open_round AS (
    SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1
),
matched_raw AS MATERIALIZED (
    SELECT run_id, roster_entry_id, comparison_id FROM v_roster_match
),
active AS (
    SELECT ru.* FROM ruling ru JOIN open_round orr ON orr.round_id = ru.round_id WHERE ru.expired_utc IS NULL
),
match_terms AS MATERIALIZED (
    -- the match lists flattened once: (ruling, field, term); the flag then joins runs to terms instead of parsing JSON per (run, ruling)
    SELECT ru.ruling_id, k.key AS field, v.value AS term
    FROM active ru, json_each(ru.scope_json, '$.match') k, json_each(k.value) v
    WHERE json_type(ru.scope_json, '$.match') = 'object' AND k.type = 'array' AND k.key <> 'instrument_alternatives'
),
match_fields AS MATERIALIZED (
    SELECT ruling_id, field FROM match_terms GROUP BY ruling_id, field
),
match_shape AS MATERIALIZED (
    SELECT ru.ruling_id, (SELECT COUNT(*) FROM json_each(ru.scope_json, '$.match')) AS n_keys,
           json_type(ru.scope_json, '$.match.instrument_alternatives') AS has_alternatives
    FROM active ru WHERE json_type(ru.scope_json, '$.match') = 'object'
),
covered AS MATERIALIZED (
    -- the match form: every list the ruling carries must contain the value of the run (an unknown field never matches; a ruling with
    -- no list covers nothing); the alternatives are OR-ed
    SELECT r.run_id, ru.ruling_id, ru.action, ru.scope_kind, ru.evidence_state, ru.ruled_utc
    FROM run r
    JOIN object o ON o.object_id = r.object_id
    JOIN eval e ON e.eval_id = r.eval_id
    JOIN match_shape ms ON ms.n_keys > 0
    JOIN active ru ON ru.ruling_id = ms.ruling_id
    WHERE (SELECT COUNT(*) FROM match_fields f WHERE f.ruling_id = ru.ruling_id)
        = (SELECT COUNT(*) FROM match_fields f WHERE f.ruling_id = ru.ruling_id
             AND EXISTS (SELECT 1 FROM match_terms t WHERE t.ruling_id = f.ruling_id AND t.field = f.field
                         AND CASE f.field
                               WHEN 'run_ids' THEN t.term = r.run_id
                               WHEN 'model_ids' THEN t.term = r.model_id
                               WHEN 'object_ids' THEN t.term = r.object_id
                               WHEN 'object_id_prefixes' THEN r.object_id GLOB t.term || '*'
                               WHEN 'object_kinds' THEN t.term = o.object_kind
                               WHEN 'paper_labels' THEN t.term = o.paper_label
                               WHEN 'object_roles' THEN t.term = o.object_role
                               WHEN 'eval_ids' THEN t.term = r.eval_id
                               WHEN 'eval_categories' THEN t.term = e.eval_category
                               WHEN 'transfer_kinds' THEN t.term = r.transfer_kind
                               WHEN 'strength_nominals' THEN t.term = r.strength_nominal
                               WHEN 'operating_point_ids_legacy' THEN t.term = r.operating_point_id_legacy
                               WHEN 'lane_sources_legacy' THEN t.term = r.lane_source_legacy
                               WHEN 'views_legacy' THEN t.term = r.view_legacy
                               WHEN 'campaigns' THEN t.term = r.campaign
                               WHEN 'pass_labels' THEN t.term = r.pass_label
                               WHEN 'instrument_ids' THEN t.term = r.instrument_id
                               WHEN 'token_caps' THEN t.term = r.token_cap_tokens
                               WHEN 'decode_sources' THEN t.term = r.decode_source
                               WHEN 'thinking_modes' THEN t.term = r.thinking_mode
                               WHEN 'harnesses' THEN t.term = r.harness
                               WHEN 'item_set_ids' THEN t.term = r.item_set_id
                               WHEN 'n_shots' THEN t.term = r.n_shot
                               WHEN 'judge_ids' THEN t.term = r.judge_id
                               WHEN 'comparison_ids' THEN EXISTS (SELECT 1 FROM matched_raw m WHERE m.run_id = r.run_id AND m.comparison_id GLOB t.term)
                               ELSE 0
                             END))
      AND (ms.has_alternatives IS NULL OR EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.match.instrument_alternatives') alt WHERE (json_type(alt.value, '$.token_caps') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.token_caps') w WHERE w.value = r.token_cap_tokens)) AND (json_type(alt.value, '$.decode_sources') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.decode_sources') w WHERE w.value = r.decode_source)) AND (json_type(alt.value, '$.thinking_modes') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.thinking_modes') w WHERE w.value = r.thinking_mode)) AND (json_type(alt.value, '$.harnesses') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.harnesses') w WHERE w.value = r.harness)) AND (json_type(alt.value, '$.item_set_ids') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.item_set_ids') w WHERE w.value = r.item_set_id)) AND (json_type(alt.value, '$.n_shots') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.n_shots') w WHERE w.value = r.n_shot)) AND (json_type(alt.value, '$.scoring_rules') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.scoring_rules') w WHERE w.value = r.scoring_rule)) AND (json_type(alt.value, '$.serving_paths') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.serving_paths') w WHERE w.value = r.serving_path))))
    UNION ALL
    -- rulings written before the match form (no `match` key at all): the key of the scope_kind itself
    SELECT r.run_id, ru.ruling_id, ru.action, ru.scope_kind, ru.evidence_state, ru.ruled_utc
    FROM run r
    JOIN object o ON o.object_id = r.object_id
    JOIN active ru ON json_type(ru.scope_json, '$.match') IS NULL
    WHERE (ru.scope_kind = 'run' AND json_extract(ru.scope_json, '$.run_id') = r.run_id)
       OR (ru.scope_kind = 'object'
           AND (json_extract(ru.scope_json, '$.object_id') IS NOT NULL OR json_extract(ru.scope_json, '$.object_kind') IS NOT NULL
                OR json_extract(ru.scope_json, '$.paper_label') IS NOT NULL)
           AND (json_extract(ru.scope_json, '$.object_id') IS NULL OR json_extract(ru.scope_json, '$.object_id') = r.object_id)
           AND (json_extract(ru.scope_json, '$.object_kind') IS NULL OR json_extract(ru.scope_json, '$.object_kind') = o.object_kind)
           AND (json_extract(ru.scope_json, '$.paper_label') IS NULL OR json_extract(ru.scope_json, '$.paper_label') = o.paper_label)
           AND (json_extract(ru.scope_json, '$.model_id') IS NULL OR json_extract(ru.scope_json, '$.model_id') = r.model_id))
       OR (ru.scope_kind = 'model' AND json_extract(ru.scope_json, '$.model_id') = r.model_id)
       OR (ru.scope_kind = 'eval' AND json_extract(ru.scope_json, '$.eval_id') = r.eval_id)
       OR (ru.scope_kind = 'comparison' AND EXISTS (SELECT 1 FROM matched_raw m WHERE m.run_id = r.run_id
               AND (m.comparison_id = json_extract(ru.scope_json, '$.comparison_id')
                    OR EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.comparison_ids') v WHERE m.comparison_id GLOB v.value))))
       OR (ru.scope_kind = 'key' AND (json_extract(ru.scope_json, '$.model_id') IS NOT NULL OR json_extract(ru.scope_json, '$.object_id') IS NOT NULL OR json_extract(ru.scope_json, '$.eval_id') IS NOT NULL OR json_extract(ru.scope_json, '$.instrument_id') IS NOT NULL OR json_extract(ru.scope_json, '$.campaign') IS NOT NULL OR json_extract(ru.scope_json, '$.operating_point_id_legacy') IS NOT NULL OR json_extract(ru.scope_json, '$.token_cap_tokens') IS NOT NULL) AND (json_extract(ru.scope_json, '$.model_id') IS NULL OR json_extract(ru.scope_json, '$.model_id') = r.model_id) AND (json_extract(ru.scope_json, '$.object_id') IS NULL OR json_extract(ru.scope_json, '$.object_id') = r.object_id) AND (json_extract(ru.scope_json, '$.eval_id') IS NULL OR json_extract(ru.scope_json, '$.eval_id') = r.eval_id) AND (json_extract(ru.scope_json, '$.instrument_id') IS NULL OR json_extract(ru.scope_json, '$.instrument_id') = r.instrument_id) AND (json_extract(ru.scope_json, '$.campaign') IS NULL OR json_extract(ru.scope_json, '$.campaign') = r.campaign) AND (json_extract(ru.scope_json, '$.operating_point_id_legacy') IS NULL OR json_extract(ru.scope_json, '$.operating_point_id_legacy') = r.operating_point_id_legacy) AND (json_extract(ru.scope_json, '$.token_cap_tokens') IS NULL OR json_extract(ru.scope_json, '$.token_cap_tokens') = r.token_cap_tokens))
),
matched AS MATERIALIZED (
    -- a comparison-scoped EXCLUDE removes the comparison from the run (the figure no longer reads it there), never the run from the paper
    SELECT m.run_id, m.roster_entry_id, m.comparison_id FROM matched_raw m
    WHERE NOT EXISTS (
        SELECT 1 FROM covered c JOIN active ru ON ru.ruling_id = c.ruling_id
        WHERE c.run_id = m.run_id AND ru.action = 'exclude' AND ru.scope_kind = 'comparison'
          AND (m.comparison_id = json_extract(ru.scope_json, '$.comparison_id')
               OR EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.match.comparison_ids') v WHERE m.comparison_id GLOB v.value)
               OR (json_type(ru.scope_json, '$.match') IS NULL
                   AND EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.comparison_ids') v WHERE m.comparison_id GLOB v.value))))
),
decisive AS MATERIALIZED (
    -- the LATEST ruled include / exclude / supersede covering the run decides its membership; a comparison-scoped ruling never does
    SELECT run_id, ruling_id, action,
           ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY ruled_utc DESC, CASE action WHEN 'include' THEN 1 ELSE 0 END, ruling_id) AS rn
    FROM covered WHERE action IN ('include', 'exclude', 'supersede') AND scope_kind <> 'comparison'
),
decision AS MATERIALIZED (
    SELECT run_id, ruling_id, action FROM decisive WHERE rn = 1
),
later AS MATERIALIZED (
    -- a later COMPLETE run at the SAME key supersedes the run: the key is the model_id, object_id, eval_id, instrument_id and judge_id
    -- columns, the two strength columns, operating_point_id_legacy, and token_cap_tokens, trial_index and pass_label -- two token
    -- caps, two trials or two passes of one object are distinct runs, never successors (per-cap registration files a run per token cap
    -- under one seeded static instrument_id, and the smaller cap must not read superseded when the larger one completes). Every
    -- nullable term compares with IS, so NULL pairs with NULL (pass_label is NULL on most rows; trial_index is NOT NULL with default 0)
    -- and a supersession that held before the key grew still holds
    SELECT r.run_id
    FROM run r JOIN run r2 ON r2.model_id = r.model_id AND r2.object_id = r.object_id AND r2.eval_id = r.eval_id
        AND r2.instrument_id = r.instrument_id AND r2.judge_id IS r.judge_id AND r2.run_id <> r.run_id
        AND r2.strength_nominal IS r.strength_nominal AND r2.strength_coefficient IS r.strength_coefficient
        AND r2.operating_point_id_legacy IS r.operating_point_id_legacy
        AND r2.token_cap_tokens IS r.token_cap_tokens AND r2.trial_index IS r.trial_index AND r2.pass_label IS r.pass_label
        AND r2.lifecycle_state IN ('judged', 'ingested')
        AND COALESCE(r2.generated_utc, r2.queued_utc) > COALESCE(r.generated_utc, r.queued_utc)
    WHERE NOT EXISTS (SELECT 1 FROM covered c WHERE c.run_id = r2.run_id AND c.action = 'coexist')
      AND NOT EXISTS (SELECT 1 FROM covered c WHERE c.run_id = r.run_id AND c.action = 'coexist')
),
withdrawn AS MATERIALIZED (
    -- a run every observation of which is no longer current (one pass over the observations; the per-run scan took 145 s live)
    SELECT r.run_id FROM run r WHERE r.lifecycle_state = 'withdrawn'
    UNION
    SELECT json_extract(ob.observation_key, '$.run_id') AS run_id FROM observation ob WHERE ob.table_name = 'run'
    GROUP BY json_extract(ob.observation_key, '$.run_id') HAVING MAX(ob.current) = 0
),
per_run_cover AS MATERIALIZED (
    SELECT run_id, json_group_array(DISTINCT ruling_id) AS ruling_ids_json,
           MAX(CASE WHEN evidence_state IN ('incoherent', 'stale') THEN 1 ELSE 0 END) AS indeterminate,
           MAX(CASE WHEN action = 'hold_out' THEN 1 ELSE 0 END) AS held_out,
           MIN(CASE WHEN action = 'hold_out' THEN ruling_id END) AS held_out_ruling_id
    FROM covered GROUP BY run_id
),
per_run_match AS MATERIALIZED (
    SELECT run_id, json_group_array(DISTINCT comparison_id) AS comparison_ids_json, json_group_array(DISTINCT roster_entry_id) AS roster_entry_ids_json
    FROM matched GROUP BY run_id
),
later_runs AS MATERIALIZED (SELECT DISTINCT run_id FROM later)
SELECT run_id,
       -- is_current (migration 0021): 1 when nothing below retires the run -- a selection_state of superseded / withdrawn / excluded, a
       -- hold_out mark, a lifecycle of failed / cancelled / superseded / withdrawn, a superseded_by_run_id, or a decisive
       -- exclude / supersede ruling on a run whose selection_state is not_in_roster (the ruling retires the run whatever its roster
       -- membership; selection_state keeps its roster-first order for the paper and still reads not_in_roster); the paper reads the
       -- rows of a current run only where the frame says so (hold_out keeps a row in the paper and still reads is_current 0;
       -- a member pass of a pooled reading, selection_state member_pass, is current and not a paper row)
       CASE WHEN selection_state IN ('superseded', 'withdrawn', 'excluded') OR held_out = 1
                 OR lifecycle_state IN ('failed', 'cancelled', 'superseded', 'withdrawn') OR superseded_by_run_id IS NOT NULL
                 OR (selection_state = 'not_in_roster' AND decisive_action IN ('exclude', 'supersede')) THEN 0 ELSE 1 END AS is_current,
       -- the FIRST failing rule, in the order above, with the ruling id where a ruling decided it; NULL on a current run
       CASE WHEN selection_state = 'superseded'
                 THEN 'selection_state superseded' || CASE WHEN decisive_action = 'supersede' THEN ' (ruling ' || decisive_ruling_id || ')'
                                                           WHEN lifecycle_state = 'superseded' OR superseded_by_run_id IS NOT NULL THEN ' (lifecycle or superseded_by_run_id)'
                                                           ELSE ' (a later run at the same key)' END
            WHEN selection_state = 'withdrawn' THEN 'selection_state withdrawn'
            WHEN selection_state = 'excluded' THEN 'selection_state excluded (ruling ' || decisive_ruling_id || ')'
            WHEN held_out = 1 THEN 'held_out (ruling ' || held_out_ruling_id || ')'
            WHEN lifecycle_state IN ('failed', 'cancelled', 'superseded', 'withdrawn') THEN 'lifecycle_state ' || lifecycle_state
            WHEN superseded_by_run_id IS NOT NULL THEN 'superseded_by_run_id ' || superseded_by_run_id
            WHEN selection_state = 'not_in_roster' AND decisive_action IN ('exclude', 'supersede')
                 THEN 'decisive ruling ' || decisive_action || ' (ruling ' || decisive_ruling_id || ')'
            END AS why,
       decisive_ruling_id, decisive_action, selection_state, in_current_paper, held_out, round_id,
       comparison_ids_json, roster_entry_ids_json, ruling_ids_json
FROM (
    SELECT r.run_id, r.lifecycle_state, r.superseded_by_run_id,
           d.ruling_id AS decisive_ruling_id,
           d.action AS decisive_action,
           COALESCE(pc.held_out, 0) AS held_out,
           pc.held_out_ruling_id,
           CASE
             WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 'indeterminate'
             WHEN w.run_id IS NOT NULL THEN 'withdrawn'
             WHEN d.action = 'include' THEN 'selected'
             WHEN pm.run_id IS NULL THEN 'not_in_roster'
             WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL
                  OR d.action = 'supersede' OR l.run_id IS NOT NULL THEN 'superseded'
             WHEN d.action = 'exclude' THEN 'excluded'
             WHEN ((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung'))) THEN 'member_pass'
             WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 'not_in_roster'
             ELSE 'selected'
           END AS selection_state,
           CASE
             WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 0
             WHEN w.run_id IS NOT NULL THEN 0
             WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 0
             WHEN d.action = 'include' THEN 1
             WHEN ((r.lane_source_legacy = 'relu_paired' AND r.pass_label IN ('base_cap', 'ladder_rung', 'unrequested_rung')) OR (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung'))) THEN 0
             WHEN pm.run_id IS NULL THEN 0
             WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL THEN 0
             WHEN d.action IN ('exclude', 'supersede') THEN 0
             WHEN l.run_id IS NOT NULL THEN 0
             ELSE 1
           END AS in_current_paper,
           (SELECT round_id FROM open_round) AS round_id,
           COALESCE(pm.comparison_ids_json, '[]') AS comparison_ids_json,
           COALESCE(pm.roster_entry_ids_json, '[]') AS roster_entry_ids_json,
           COALESCE(pc.ruling_ids_json, '[]') AS ruling_ids_json
    FROM run r
    LEFT JOIN per_run_cover pc ON pc.run_id = r.run_id
    LEFT JOIN per_run_match pm ON pm.run_id = r.run_id
    LEFT JOIN decision d ON d.run_id = r.run_id
    LEFT JOIN withdrawn w ON w.run_id = r.run_id
    LEFT JOIN later_runs l ON l.run_id = r.run_id
);
DROP VIEW IF EXISTS v_run_decode_policy_match;
CREATE VIEW v_run_decode_policy_match AS
WITH run_sent AS (
  SELECT r.run_id, j.key AS param, CAST(j.value AS REAL) AS value
  FROM run r, json_each(r.decode_sent_json) j
  WHERE j.key IN ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty') AND j.type IN ('integer', 'real')
  UNION
  SELECT run_id, 'temperature', CAST(decode_temperature_sent AS REAL) FROM run WHERE decode_temperature_sent IS NOT NULL
  UNION
  SELECT run_id, 'top_p', CAST(decode_top_p_sent AS REAL) FROM run WHERE decode_top_p_sent IS NOT NULL
  UNION
  SELECT run_id, 'top_k', CAST(decode_top_k_sent AS REAL) FROM run WHERE decode_top_k_sent IS NOT NULL
),
run_sent_unmodelled AS (
  SELECT DISTINCT r.run_id FROM run r, json_each(r.decode_sent_json) j
  WHERE j.key NOT IN ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty') OR j.type NOT IN ('integer', 'real')
),
policy_param AS (
  SELECT policy_id, 'temperature' AS param, CAST(temperature AS REAL) AS value FROM decode_policy WHERE temperature IS NOT NULL
  UNION ALL SELECT policy_id, 'top_p', CAST(top_p AS REAL) FROM decode_policy WHERE top_p IS NOT NULL
  UNION ALL SELECT policy_id, 'top_k', CAST(top_k AS REAL) FROM decode_policy WHERE top_k IS NOT NULL
  UNION ALL SELECT policy_id, 'min_p', CAST(min_p AS REAL) FROM decode_policy WHERE min_p IS NOT NULL
  UNION ALL SELECT policy_id, 'repetition_penalty', CAST(repetition_penalty AS REAL) FROM decode_policy WHERE repetition_penalty IS NOT NULL
  UNION ALL SELECT policy_id, 'presence_penalty', CAST(presence_penalty AS REAL) FROM decode_policy WHERE presence_penalty IS NOT NULL
),
policy_not_sent AS (
  SELECT p.policy_id, j.key AS param FROM decode_policy p, json_each(p.params_not_sent_json) j
),
in_scope AS (
  SELECT r.run_id, p.policy_id, p.seed_rule AS policy_seed_rule, r.decode_seed_rule AS run_seed_rule,
         COALESCE(r.generated_utc, r.started_utc, r.queued_utc) AS generated, p.effective_from_utc, p.superseded_by_policy_id,
         (SELECT q.effective_from_utc FROM decode_policy q WHERE q.policy_id = p.superseded_by_policy_id) AS successor_from_utc
  FROM run r
  JOIN eval e ON e.eval_id = r.eval_id
  JOIN decode_policy p ON p.model_id = r.model_id AND p.thinking_mode = r.thinking_mode
       AND (p.eval_id = r.eval_id OR (p.eval_id IS NULL AND p.eval_category = e.eval_category))
),
candidate AS (
  SELECT s.run_id, s.policy_id,
         CASE WHEN (s.effective_from_utc = 'unverified' OR s.effective_from_utc <= s.generated)
                   AND (s.superseded_by_policy_id IS NULL OR s.successor_from_utc IS NULL OR s.successor_from_utc = 'unverified'
                        OR s.successor_from_utc > s.generated) THEN 1 ELSE 0 END AS in_force
  FROM in_scope s
  WHERE (s.policy_seed_rule = 'unverified' OR s.policy_seed_rule = s.run_seed_rule)
    AND (s.superseded_by_policy_id IS NULL OR s.successor_from_utc IS NULL OR s.successor_from_utc = 'unverified'
         OR s.successor_from_utc > s.generated)
    AND NOT EXISTS (SELECT 1 FROM run_sent_unmodelled u WHERE u.run_id = s.run_id)
    AND NOT EXISTS (
      SELECT 1 FROM policy_param pp WHERE pp.policy_id = s.policy_id
        AND NOT EXISTS (SELECT 1 FROM run_sent rs WHERE rs.run_id = s.run_id AND rs.param = pp.param AND rs.value = pp.value)
        AND NOT (EXISTS (SELECT 1 FROM policy_not_sent pn WHERE pn.policy_id = s.policy_id AND pn.param = pp.param)
                 AND NOT EXISTS (SELECT 1 FROM run_sent rs WHERE rs.run_id = s.run_id AND rs.param = pp.param)))
    AND NOT EXISTS (
      SELECT 1 FROM run_sent rs WHERE rs.run_id = s.run_id
        AND NOT EXISTS (SELECT 1 FROM policy_param pp WHERE pp.policy_id = s.policy_id AND pp.param = rs.param AND pp.value = rs.value))
),
per_run AS (
  SELECT run_id, COUNT(*) AS n, MIN(policy_id) AS only_id, SUM(in_force) AS n_in_force,
         MAX(CASE WHEN in_force = 1 THEN policy_id END) AS in_force_id
  FROM candidate GROUP BY run_id
),
by_source AS (
  -- recorded by source (migration 0021): a run with no sent parameter and no unmodelled key, served under generation_config,
  -- against the in-scope rows served the same way whose every pinned parameter is declared not sent (params_not_sent_json) -- and
  -- eligible as an ordinary candidate is: the seed rule agrees (or the policy pins none), and a superseded row counts only for a
  -- run generated before its successor took effect
  SELECT s.run_id, s.policy_id
  FROM in_scope s
  JOIN run r ON r.run_id = s.run_id
  JOIN decode_policy p ON p.policy_id = s.policy_id
  WHERE r.decode_source = 'generation_config' AND p.decode_source = 'generation_config'
    AND (s.policy_seed_rule = 'unverified' OR s.policy_seed_rule = s.run_seed_rule)
    AND (s.superseded_by_policy_id IS NULL OR s.successor_from_utc IS NULL OR s.successor_from_utc = 'unverified'
         OR s.successor_from_utc > s.generated)
    AND NOT EXISTS (SELECT 1 FROM run_sent rs WHERE rs.run_id = s.run_id)
    AND NOT EXISTS (SELECT 1 FROM run_sent_unmodelled u WHERE u.run_id = s.run_id)
    AND NOT EXISTS (SELECT 1 FROM policy_param pp WHERE pp.policy_id = s.policy_id
                    AND NOT EXISTS (SELECT 1 FROM policy_not_sent pn WHERE pn.policy_id = pp.policy_id AND pn.param = pp.param))
),
by_source_per_run AS (
  SELECT run_id, COUNT(*) AS n, MIN(policy_id) AS only_id FROM by_source GROUP BY run_id
)
SELECT r.run_id,
       CASE WHEN rec.run_id IS NULL THEN CASE WHEN bs.n = 1 THEN bs.only_id END
            WHEN c.n = 1 THEN c.only_id
            WHEN c.n > 1 AND c.n_in_force = 1 THEN c.in_force_id END AS policy_id,
       CASE WHEN rec.run_id IS NULL THEN CASE WHEN bs.n = 1 THEN 'exact' WHEN bs.n > 1 THEN 'ambiguous' ELSE 'unrecorded' END
            WHEN c.n IS NULL THEN 'no_match'
            WHEN c.n = 1 OR c.n_in_force = 1 THEN 'exact'
            ELSE 'ambiguous' END AS match,
       CASE WHEN rec.run_id IS NULL
            THEN COALESCE((SELECT json_group_array(policy_id) FROM (SELECT policy_id FROM by_source bb WHERE bb.run_id = r.run_id ORDER BY policy_id)), '[]')
            ELSE COALESCE((SELECT json_group_array(policy_id) FROM (SELECT policy_id FROM candidate cc WHERE cc.run_id = r.run_id ORDER BY policy_id)), '[]')
            END AS candidates_json,
       COALESCE((SELECT json_group_array(policy_id) FROM (SELECT policy_id FROM in_scope sc WHERE sc.run_id = r.run_id ORDER BY policy_id)), '[]') AS in_scope_json
FROM run r
LEFT JOIN (SELECT DISTINCT run_id FROM run_sent) rec ON rec.run_id = r.run_id
LEFT JOIN per_run c ON c.run_id = r.run_id
LEFT JOIN by_source_per_run bs ON bs.run_id = r.run_id;
