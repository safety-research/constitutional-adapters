-- 0023_record_refusal_fact.sql
-- The registry record-log change's one migration (registry rulings of 2026-10-02 and 2026-10-03). No table is rebuilt and no row is
-- copied: every change is a CREATE TABLE, an ALTER TABLE ADD COLUMN of a
-- column the code declares LAST (migrations.addable_columns) or a DROP / CREATE of a v_* object. Rendered by `registry migrate --plan --sql`
-- against a schema-version-22 database (0001..0022 applied to an empty database by the runner); the planner reports one create, three
-- add-column tables, seven recreates and NO rebuild -- `tests/test_registry_migrations.py` pins the plan, and any rebuild line is a STOP
-- (the 0020 rebuild of row_history wrote a 25 GB WAL).
-- (1) Refusal facts in the record log. `record_refusal_fact` (ledger-scale, pk (record_sha256, judged_with), two indexes): the
--     harvester's DURABLE fact of one refusal judgment, written into the log as a `record_refusal` record under records/harvester/<day>/
--     and folded by its own apply path (reducer.apply_ledger_fact: INSERT OR IGNORE, no row_history), so every mirror and every rebuild
--     reproduce every refusal ever made under every judging stamp. `record_refusal` gains `judged_with` (the stamp of the judging code,
--     harvest.validator_stamp; NULL on the rows refused before it was kept, which the first cycle after the window materializes as
--     `unverified` facts with extra_json.backfilled = true -- 126 rows live on 2026-10-03). The two ledgers whose record_type CHECK lists
--     every record kind (`blocked_records`, `row_history`) are NOT touched: the new kind never reaches them, and their CHECKs render from
--     vocab.LEDGERED_RECORD_TYPES -- RECORD_TYPES without the unledgered fact kinds -- so the list is byte-identical to the one 0020 wrote.
-- (2) Object lineage, first-class. `object.parent_object_id` (FK object, dependency-blocked at fold like every FK) and
--     `object.named_role` (vocab.OBJECT_NAMED_ROLES), declared after the run-scale tail so both are ALTERs. The fold sets each from the
--     record's own field when stated, else lifts it from extra_json.parent_object_id / extra_json.named_role (reducer.lift_object_fields);
--     an explicit null resets. The existing rows' extra values are lifted ONCE by this file's Python post-step
--     (migrations.POST_APPLY["record_refusal_fact"] = migrations.lift_object_lineage: a refold of every object key under the new fold,
--     the live row and its history events rewritten exactly as a rebuild would write them; idempotent; counted in the line `registry
--     migrate` prints). Live, read-only, 2026-10-03: 412 objects, 27 carrying both keys under extra, every parent present.
-- (3) Judgment identity currency. `judgment.claimed_instrument_id` (FK instrument) and
--     `judgment.claimed_required_judge_id` (FK judge), the run's identity as it stood when the judgment was CLAIMED (stated by
--     pending.claim from the run row it reads under the lock, and by adapters that know the instrument); NULL on every live row, so the
--     rule moves zero rows at the window. Five of the seven recreated objects render the three paper-side judgment-currency rules together
--     (schema_def.judgment_currency_sql): the status rule, the complete-over-partial rule (a partial judgment beside a complete
--     one of the SAME (run, judge) is not current; 0 live pairs on 2026-10-03), and the identity rule (a judgment whose binding
--     differs from the run's current instrument or required judge is not current; a NULL binding is current, so a second judge's reading
--     stands). The sixth, `v_pending_judgment`, applies the identity rule to the judgments it counts (a review of the first
--     rendering): a covering or partial judgment bound at claim to a since-corrected identity covers nothing in the worklist, and a run
--     whose only judgments by the required judge are stale-bound reads `pending_reason = identity_stale` -- without it a same-judge
--     instrument correction left the run with neither a paper row nor a pending job; for the same reason the displacing complete
--     witness is itself identity-current (a stale complete beside a current partial hides nothing). The review of that rendering added
--     the claim side and the seventh recreate: an in-progress claim bound to a since-corrected identity is not `claimed` in the worklist
--     (the run reads `identity_stale` with no claim columns; pending.claim closes such a claim `abandoned` and binds a new one to the
--     current identity, for the same session and another alike, never ClaimHeld), and `v_roster_match` -- like the measurement-judge rule
--     of the frame and the harvester's completion requirement -- admits only an identity-current covering judgment, so an identity-stale
--     complete judgment covers nothing anywhere. A further rule (an object record may omit
--     `status` when a row exists) is a validator and fold rule with no schema change.
-- drift: table object lacks column(s) parent_object_id, named_role
-- drift: table object: CREATE text differs — have «CREATE TABLE object(object_id TEXT NOT NULL,object_kind TEXT NOT NULL CHECK(object_kind IN('none','system_instruction','lora','lora_subtraction','relu_steer','…» want «CREATE TABLE object(object_id TEXT NOT NULL,object_kind TEXT NOT NULL CHECK(object_kind IN('none','system_instruction','lora','lora_subtraction','relu_steer','…»
-- drift: table judgment lacks column(s) claimed_instrument_id, claimed_required_judge_id
-- drift: table judgment: CREATE text differs — have «CREATE TABLE judgment(judgment_id TEXT NOT NULL,run_id TEXT NOT NULL REFERENCES run(run_id),judge_id TEXT NOT NULL REFERENCES judge(judge_id),started_utc TEXT …» want «CREATE TABLE judgment(judgment_id TEXT NOT NULL,run_id TEXT NOT NULL REFERENCES run(run_id),judge_id TEXT NOT NULL REFERENCES judge(judge_id),started_utc TEXT …»
-- drift: table record_refusal lacks column(s) judged_with
-- drift: table record_refusal: CREATE text differs — have «CREATE TABLE record_refusal(record_sha256 TEXT NOT NULL CHECK(record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a…» want «CREATE TABLE record_refusal(record_sha256 TEXT NOT NULL CHECK(record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a…»
-- drift: table record_refusal_fact is absent
-- drift: v_item_result_current differs from the code's SQL
-- drift: v_item_result_provenances differs from the code's SQL
-- drift: v_pending_judgment differs from the code's SQL
-- drift: v_roster_match differs from the code's SQL
-- drift: v_paper_current differs from the code's SQL
-- drift: v_multi_source_rows differs from the code's SQL
-- drift: v_paper_long differs from the code's SQL
-- drift: index idx_record_refusal_fact_cycle is absent
-- drift: index idx_record_refusal_fact_uri is absent

-- create record_refusal_fact
CREATE TABLE record_refusal_fact (
  record_sha256 TEXT NOT NULL CHECK (record_sha256 GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),  -- The refused record (records.record_sha): the key `record_refusal` holds it under.
  record_uri TEXT NOT NULL,  -- The logical uri the record was refused under (`<file>#<line>` for a record inside a file).
  record_type TEXT,  -- why-null: an envelope without a readable type
  field TEXT,  -- why-null: an envelope-level refusal names no field
  reason TEXT NOT NULL,
  judged_with TEXT NOT NULL CHECK (judged_with GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR judged_with = 'unverified'),  -- The judging code's stamp (harvest.validator_stamp) the refusal was made under, or 'unverified' for a refusal recorded before the stamp was kept.
  refused_utc TEXT NOT NULL CHECK (refused_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- record_refusal.refused_utc of the judgment: when the refusal was made (never the cycle that derived the fact).
  producer_json TEXT CHECK (producer_json IS NULL OR json_valid(producer_json)),  -- why-null: the refused file could not be re-read for the record's producer block (moved, unreadable, or a line that is not JSON)
  cycle_utc TEXT NOT NULL CHECK (cycle_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'),  -- The clock of the harvester cycle that derived the fact (the record id is minted over the payload WITHOUT it).
  extra_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra_json)),  -- `backfilled: true` for a row refused before the stamp was kept.
  PRIMARY KEY (record_sha256, judged_with)
);
CREATE INDEX idx_record_refusal_fact_uri ON record_refusal_fact (record_uri);
CREATE INDEX idx_record_refusal_fact_cycle ON record_refusal_fact (cycle_utc);

-- add column(s) to object: ALTER TABLE ADD COLUMN, no row copied (the code declares them after every live column, so SQLite's appended definition equals the code's CREATE text; migrations/README.md)
-- column parent_object_id: why-null: only epoch-checkpoint children name the fit they belong to
ALTER TABLE object ADD COLUMN parent_object_id TEXT REFERENCES object(object_id);
-- column named_role: why-null: only epoch-checkpoint children carry a named role; the parent and every other object carry none
ALTER TABLE object ADD COLUMN named_role TEXT CHECK (named_role IN ('epoch_1', 'epoch_2', 'intermediate', 'final'));

-- add column(s) to judgment: ALTER TABLE ADD COLUMN, no row copied (the code declares them after every live column, so SQLite's appended definition equals the code's CREATE text; migrations/README.md)
-- column claimed_instrument_id: why-null: legacy judgments and writers that do not know the run's instrument at claim time (NULL reads as current)
ALTER TABLE judgment ADD COLUMN claimed_instrument_id TEXT REFERENCES instrument(instrument_id);
-- column claimed_required_judge_id: why-null: legacy judgments, writers that do not know the run's required judge at claim time, and runs whose instrument and run name no judge (NULL reads as current)
ALTER TABLE judgment ADD COLUMN claimed_required_judge_id TEXT REFERENCES judge(judge_id);

-- add column(s) to record_refusal: ALTER TABLE ADD COLUMN, no row copied (the code declares them after every live column, so SQLite's appended definition equals the code's CREATE text; migrations/README.md)
-- column judged_with: why-null: rows refused before the judging stamp was recorded (the fact derivation reads them as 'unverified'); declared last so the live ledger takes it by ALTER TABLE ADD COLUMN
ALTER TABLE record_refusal ADD COLUMN judged_with TEXT CHECK (judged_with IS NULL OR judged_with GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]' OR judged_with = 'unverified');

DROP VIEW IF EXISTS v_item_result_current;
CREATE VIEW v_item_result_current AS
SELECT ir.* FROM item_result ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id
WHERE (ir.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = ir.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND ir.withdrawn_by_record_sha256 IS NULL AND (ir.judgment_id = '' OR NOT EXISTS (SELECT 1 FROM judgment jx WHERE jx.judgment_id = ir.judgment_id AND (jx.status IN ('superseded', 'abandoned') OR CASE WHEN jx.status = 'partial' THEN EXISTS (SELECT 1 FROM judgment jcp JOIN run rcp ON rcp.run_id = jcp.run_id LEFT JOIN instrument icp ON icp.instrument_id = rcp.instrument_id WHERE jcp.run_id = jx.run_id AND jcp.judge_id = jx.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS rcp.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(icp.judge_id, rcp.judge_id)))) ELSE 0 END OR CASE WHEN jx.claimed_instrument_id IS NULL AND jx.claimed_required_judge_id IS NULL THEN 0 ELSE EXISTS (SELECT 1 FROM run rx LEFT JOIN instrument ix ON ix.instrument_id = rx.instrument_id WHERE rx.run_id = jx.run_id AND NOT ((jx.claimed_instrument_id IS NULL OR jx.claimed_instrument_id IS rx.instrument_id) AND (jx.claimed_required_judge_id IS NULL OR jx.claimed_required_judge_id IS COALESCE(ix.judge_id, rx.judge_id)))) END)))
  AND NOT EXISTS (SELECT 1 FROM item_result o LEFT JOIN source_revision so ON so.revision_id = o.source_revision_id
                  WHERE o.run_id = ir.run_id AND o.item_id = ir.item_id AND o.metric_id = ir.metric_id AND o.trial_index = ir.trial_index AND o.sample_index = ir.sample_index AND o.turn_index = ir.turn_index AND o.judgment_id = ir.judgment_id AND o.source_revision_id <> ir.source_revision_id
                    AND (o.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = o.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND o.withdrawn_by_record_sha256 IS NULL AND (CASE WHEN o.source_revision_id = '' THEN 0 ELSE 1 END, -COALESCE(so.priority, 0), COALESCE(so.source_id, ''), o.source_revision_id) < (CASE WHEN ir.source_revision_id = '' THEN 0 ELSE 1 END, -COALESCE(sr.priority, 0), COALESCE(sr.source_id, ''), ir.source_revision_id));
DROP VIEW IF EXISTS v_item_result_provenances;
CREATE VIEW v_item_result_provenances AS
SELECT ir.*, sr.source_id, COALESCE(sr.priority, 0) AS priority
FROM item_result ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id
WHERE (ir.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = ir.source_revision_id AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL)) AND ir.withdrawn_by_record_sha256 IS NULL AND (ir.judgment_id = '' OR NOT EXISTS (SELECT 1 FROM judgment jx WHERE jx.judgment_id = ir.judgment_id AND (jx.status IN ('superseded', 'abandoned') OR CASE WHEN jx.status = 'partial' THEN EXISTS (SELECT 1 FROM judgment jcp JOIN run rcp ON rcp.run_id = jcp.run_id LEFT JOIN instrument icp ON icp.instrument_id = rcp.instrument_id WHERE jcp.run_id = jx.run_id AND jcp.judge_id = jx.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS rcp.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(icp.judge_id, rcp.judge_id)))) ELSE 0 END OR CASE WHEN jx.claimed_instrument_id IS NULL AND jx.claimed_required_judge_id IS NULL THEN 0 ELSE EXISTS (SELECT 1 FROM run rx LEFT JOIN instrument ix ON ix.instrument_id = rx.instrument_id WHERE rx.run_id = jx.run_id AND NOT ((jx.claimed_instrument_id IS NULL OR jx.claimed_instrument_id IS rx.instrument_id) AND (jx.claimed_required_judge_id IS NULL OR jx.claimed_required_judge_id IS COALESCE(ix.judge_id, rx.judge_id)))) END)));
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
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
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
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
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
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
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
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
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
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
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
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
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
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id
     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id
LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id
LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
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
                           -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  -- a secondary row whose identity the owner also carries leaves the frame: folded (the owner row stands) or refused (neither is admitted)
  AND (ms.role IS NULL OR ms.role <> 'secondary' OR ms.state IS NULL OR ms.state = 'stopgap')
  -- a run the roster matches admits only a measurement whose judge satisfies a matched entry; an unmatched run admits every row
  AND (NOT EXISTS (SELECT 1 FROM v_roster_match vm WHERE vm.run_id = r.run_id)
       OR EXISTS (SELECT 1 FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
                  WHERE vm.run_id = r.run_id AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id))))
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
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id))))
                        END) AS comparison_ids_json,
       (SELECT json_group_array(DISTINCT vm.roster_entry_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id))))
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
  -- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND (rm.judgment_id = '' OR jm.status IS NULL OR jm.status NOT IN ('superseded', 'abandoned'))
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND (COALESCE(jm.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = jm.run_id AND jcp.judge_id = jm.judge_id AND jcp.status = 'complete' AND ((jcp.claimed_instrument_id IS NULL OR jcp.claimed_instrument_id IS r.instrument_id) AND (jcp.claimed_required_judge_id IS NULL OR jcp.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
  -- a secondary row whose identity the owner also carries leaves the frame: folded (the owner row stands) or refused (neither is admitted)
  AND (msr.role IS NULL OR msr.role <> 'secondary' OR msc.state IS NULL OR msc.state = 'stopgap')
  -- a run the roster matches admits only a measurement whose judge satisfies a matched entry; an unmatched run admits every row
  AND (NOT EXISTS (SELECT 1 FROM v_roster_match vm WHERE vm.run_id = r.run_id)
       OR EXISTS (SELECT 1 FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
                  WHERE vm.run_id = r.run_id AND CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND i.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS i.judge_id AND jm.status = 'complete' AND COALESCE(jm.n_items_judged, 0) + COALESCE(jm.n_judge_refusals, 0) + COALESCE(jm.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jm.run_id AND rm_err.judgment_id = jm.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0) AND ((jm.claimed_instrument_id IS NULL OR jm.claimed_instrument_id IS r.instrument_id) AND (jm.claimed_required_judge_id IS NULL OR jm.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id))))
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
          FROM judgment j4 WHERE j4.run_id = r.run_id AND j4.judge_id = jd.judge_id AND j4.status IN ('partial', 'complete')
          AND ((j4.claimed_instrument_id IS NULL OR j4.claimed_instrument_id IS r.instrument_id) AND (j4.claimed_required_judge_id IS NULL OR j4.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) AS items_missing,
       (SELECT COUNT(*) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id AND j5.status = 'abandoned') AS n_abandoned,
       (SELECT MAX(j5.claimed_utc) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id
          AND j5.status = 'abandoned') AS last_abandoned_utc,
       (SELECT COUNT(*) FROM judgment j6 WHERE j6.run_id = r.run_id AND j6.judge_id <> jd.judge_id
          AND j6.status = 'complete') AS n_complete_other_judges,
       CASE WHEN cl.judgment_id IS NOT NULL THEN 'claimed'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'partial' AND ((j7.claimed_instrument_id IS NULL OR j7.claimed_instrument_id IS r.instrument_id) AND (j7.claimed_required_judge_id IS NULL OR j7.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) THEN 'partial'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'complete' AND ((j7.claimed_instrument_id IS NULL OR j7.claimed_instrument_id IS r.instrument_id) AND (j7.claimed_required_judge_id IS NULL OR j7.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) THEN 'coverage_short'
            -- the only judgments by J (a claim, a partial or a complete reading) were bound at claim to a since-corrected run identity:
            -- a fresh grade is wanted, and a stale-bound claim holds nothing (pending.claim closes it and binds a new one)
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status IN ('in_progress', 'partial', 'complete')
                           AND NOT ((j7.claimed_instrument_id IS NULL OR j7.claimed_instrument_id IS r.instrument_id) AND (j7.claimed_required_judge_id IS NULL OR j7.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))) THEN 'identity_stale'
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
-- the active claim is an identity-current in-progress row: a claim bound to a since-corrected identity is not held
LEFT JOIN judgment cl ON cl.judgment_id = (
        SELECT j3.judgment_id FROM judgment j3 WHERE j3.run_id = r.run_id AND j3.judge_id = jd.judge_id
          AND j3.status = 'in_progress' AND ((j3.claimed_instrument_id IS NULL OR j3.claimed_instrument_id IS r.instrument_id) AND (j3.claimed_required_judge_id IS NULL OR j3.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
        ORDER BY j3.claimed_utc DESC LIMIT 1)
WHERE r.lifecycle_state IN ('generated', 'judging', 'judged', 'ingested')
  -- no identity-current judgment by J covers it: a covering judgment bound at claim to a since-corrected identity covers nothing
  AND NOT EXISTS (
        SELECT 1 FROM judgment jc
        WHERE jc.run_id = r.run_id AND jc.judge_id = jd.judge_id AND jc.status = 'complete' AND COALESCE(jc.n_items_judged, 0) + COALESCE(jc.n_judge_refusals, 0) + COALESCE(jc.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jc.run_id AND rm_err.judgment_id = jc.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
          AND ((jc.claimed_instrument_id IS NULL OR jc.claimed_instrument_id IS r.instrument_id) AND (jc.claimed_required_judge_id IS NULL OR jc.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id))));
DROP VIEW IF EXISTS v_roster_match;
CREATE VIEW v_roster_match AS
SELECT r.run_id, re.roster_entry_id, re.comparison_id
    FROM run r
    JOIN object o ON o.object_id = r.object_id
    JOIN instrument i ON i.instrument_id = r.instrument_id
    -- the roster_entry index (round_id, model_id, eval_id) carries this join; a CTE over the table would not be indexed
    JOIN roster_entry re ON re.round_id = (SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1)
      AND re.model_id = r.model_id AND re.eval_id = r.eval_id
      -- the roster version (migration 0016): an entry matches only under the version the paper_round row of its round carries; a load
      -- re-stamps every entry it expands, so an entry a later version re-shaped keeps its older stamp and matches nothing (before 0016
      -- such an entry kept matching beside its replacement, a second selector for one key); a round loaded before
      -- 0016 carries no version and admits every entry as before (no apostrophe in this comment: the reference scan strips quoted literals)
      AND ((SELECT roster_version FROM paper_round pr WHERE pr.round_id = re.round_id) IS NULL
           OR re.roster_version = (SELECT roster_version FROM paper_round pr WHERE pr.round_id = re.round_id))
      AND (re.object_id = r.object_id
           OR (re.object_id IS NULL
               AND (json_extract(re.object_selector_json, '$.paper_label') IS NULL OR json_extract(re.object_selector_json, '$.paper_label') = o.paper_label)
               AND (json_extract(re.object_selector_json, '$.object_kind') IS NULL OR json_extract(re.object_selector_json, '$.object_kind') = o.object_kind)
               AND (json_extract(re.object_selector_json, '$.object_role') IS NULL OR json_extract(re.object_selector_json, '$.object_role') = o.object_role)
               AND (json_extract(re.object_selector_json, '$.fit_model_id') IS NULL OR json_extract(re.object_selector_json, '$.fit_model_id') = o.fit_model_id)
               AND (json_extract(re.object_selector_json, '$.transfer_kind') IS NULL OR json_extract(re.object_selector_json, '$.transfer_kind') = r.transfer_kind)
               AND (json_type(re.object_selector_json, '$.transfer_kinds') IS NULL OR EXISTS (SELECT 1 FROM json_each(re.object_selector_json, '$.transfer_kinds') sv WHERE sv.value = r.transfer_kind))
               AND (json_extract(re.object_selector_json, '$.object_id_prefix') IS NULL
                    OR r.object_id GLOB json_extract(re.object_selector_json, '$.object_id_prefix') || '*')))
      AND (re.instrument_id = r.instrument_id
           -- a selector never matches a PADDED instrument (0013): padding is instrument identity (0012) that no selector
           -- field carries, so a padded instrument satisfies the fields of the unpadded entries and would pool with them; it is
           -- reached only by name, through the branch above (no apostrophe in this comment: the reference scan strips quoted literals)
           OR (re.instrument_id IS NULL
               AND i.padding_tokens IS NULL
               AND (json_extract(re.instrument_selector_json, '$.harness') IS NULL OR json_extract(re.instrument_selector_json, '$.harness') = i.harness)
               AND (json_extract(re.instrument_selector_json, '$.item_set_id') IS NULL OR json_extract(re.instrument_selector_json, '$.item_set_id') = i.item_set_id)
               AND (json_extract(re.instrument_selector_json, '$.thinking_mode') IS NULL OR json_extract(re.instrument_selector_json, '$.thinking_mode') = i.thinking_mode)
               AND (json_extract(re.instrument_selector_json, '$.serving_path') IS NULL OR json_extract(re.instrument_selector_json, '$.serving_path') = i.serving_path)
               AND (json_extract(re.instrument_selector_json, '$.decode_source') IS NULL OR json_extract(re.instrument_selector_json, '$.decode_source') = i.decode_source)
               AND (json_extract(re.instrument_selector_json, '$.scoring_rule') IS NULL OR json_extract(re.instrument_selector_json, '$.scoring_rule') = i.scoring_rule)
               AND (json_extract(re.instrument_selector_json, '$.n_shot') IS NULL OR json_extract(re.instrument_selector_json, '$.n_shot') = i.n_shot)
               AND (json_type(re.instrument_selector_json, '$.token_cap_tokens') IS NULL OR EXISTS (SELECT 1 FROM json_each(re.instrument_selector_json, '$.token_cap_tokens') sv WHERE sv.value = i.token_cap_tokens))))
      -- the entry is satisfied only by a COMPLETE judgment of its judge covering the items of the run; an entry that
      -- names no judge reads the judge of the instrument (a rule-scored instrument: a generation-time metric row); the covering
      -- judgment is identity-current: one bound at claim to a since-corrected run identity satisfies no entry
      AND ((re.judge_id IS NOT NULL
            AND EXISTS (SELECT 1 FROM judgment jx WHERE jx.run_id = r.run_id AND jx.judge_id = re.judge_id AND jx.status = 'complete' AND COALESCE(jx.n_items_judged, 0) + COALESCE(jx.n_judge_refusals, 0) + COALESCE(jx.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jx.run_id AND rm_err.judgment_id = jx.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                        AND ((jx.claimed_instrument_id IS NULL OR jx.claimed_instrument_id IS r.instrument_id) AND (jx.claimed_required_judge_id IS NULL OR jx.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))))
           OR (re.judge_selector_json IS NOT NULL
               AND EXISTS (SELECT 1 FROM judgment jx JOIN judge jj ON jj.judge_id = jx.judge_id WHERE jx.run_id = r.run_id
                           AND jx.status = 'complete' AND COALESCE(jx.n_items_judged, 0) + COALESCE(jx.n_judge_refusals, 0) + COALESCE(jx.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jx.run_id AND rm_err.judgment_id = jx.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                           AND ((jx.claimed_instrument_id IS NULL OR jx.claimed_instrument_id IS r.instrument_id) AND (jx.claimed_required_judge_id IS NULL OR jx.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id)))
                           AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                OR json_extract(re.judge_selector_json, '$.judge_model') = jj.judge_model)
                           AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                OR json_extract(re.judge_selector_json, '$.rubric_name') = jj.rubric_name)))
           OR (re.judge_id IS NULL AND re.judge_selector_json IS NULL
               AND ((i.judge_id IS NULL AND EXISTS (SELECT 1 FROM run_metric rm0 WHERE rm0.run_id = r.run_id AND rm0.judgment_id = ''))
                    OR (i.judge_id IS NOT NULL
                        AND EXISTS (SELECT 1 FROM judgment jx WHERE jx.run_id = r.run_id AND jx.judge_id = i.judge_id AND jx.status = 'complete' AND COALESCE(jx.n_items_judged, 0) + COALESCE(jx.n_judge_refusals, 0) + COALESCE(jx.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err WHERE rm_err.run_id = jx.run_id AND rm_err.judgment_id = jx.judgment_id), 0) >= COALESCE(COALESCE((SELECT MIN(COALESCE(COALESCE(r.n_items_generated, r.n_items_planned), s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = r.item_set_id), COALESCE(r.n_items_generated, r.n_items_planned)), 0)
                                    AND ((jx.claimed_instrument_id IS NULL OR jx.claimed_instrument_id IS r.instrument_id) AND (jx.claimed_required_judge_id IS NULL OR jx.claimed_required_judge_id IS COALESCE(i.judge_id, r.judge_id))))))))
      -- operating_point: a filed stamp of 1, else the fallback decided 2026-09-22 (operating_point_basis_sql: a fallback kind at nominal
      -- 1.0, the all-positions vector at c = 0.3, an unsteered or prompted kind); a filed stamp of 0 is never at the point
      AND ((re.strength_rule = 'operating_point' AND ((CASE WHEN r.is_operating_point = 1 THEN 'stamped'
            WHEN r.is_operating_point IS NULL AND o.object_kind IN ('lora', 'lora_subtraction', 'composed', 'relu_steer', 'relu_steer_distilled_difference', 'alphasteer', 'static_r_arditi', 'vector', 'vector_all', 'diffmean', 'antidote')
                 AND ABS(r.strength_nominal - 1.0) < 1e-9 THEN 'fallback_nominal_1p0'
            WHEN r.is_operating_point IS NULL AND o.object_kind = 'vector_all'
                 AND ABS(r.strength_coefficient - 0.3) < 1e-9 THEN 'fallback_arditi_c0p3'
            WHEN o.object_kind IN ('none', 'system_instruction') THEN 'no_strength_axis'
            ELSE 'none' END) <> 'none'))
           OR (re.strength_rule GLOB 'fixed:*' AND r.strength_nominal = CAST(substr(re.strength_rule, 7) AS REAL))
           OR (re.strength_rule GLOB 'ladder:*' AND r.strength_nominal IS NOT NULL));
