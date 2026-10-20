"""Structured definition of the registry schema and the DDL renderer.

``schema.sql`` (the readable DDL) and ``docs/registry/schema.md`` are both RENDERED from the
tables defined here; ``vocab.py`` supplies every ``CHECK (col IN (...))`` list and the
``fit_target_kind`` paired-nullness rules. Tests assert that the committed ``schema.sql`` equals
:func:`render_schema_sql` and that the CHECK lists parsed OUT OF the DDL text equal the vocab
tuples, so the three cannot drift from one another.

Strictness patterns:

* every identity / provenance column is ``NOT NULL``; every nullable column carries a
  ``why_null`` reason that the DDL renders as a ``-- why-null:`` comment and ``schema.md`` lists;
* every controlled column carries ``CHECK (col IN (...))`` rendered from ``vocab``;
* every ``*_utc TEXT`` column is checked against the ISO-8601-with-milliseconds shape;
* every ``*_json`` column is checked with ``json_valid``;
* sha columns are checked as exact-length lowercase hex (or the literal ``unverified`` where the
  spec admits it);
* composite keys with sentinels (``judgment_id = ''``, ``threshold = -1``, ``turn_index = -1``,
  ``computation_id = ''``) have their foreign keys enforced by triggers (``WHEN NEW.col <> ''``);
* ``run.lifecycle_state = 'judged'`` requires a ``judgment`` row (trigger);
* every ``*_utc`` of a reached lifecycle state is non-null (trigger ``trg_run_state_utc``).

Scale classes: ``ITEM_SCALE_TABLES`` fold whole-row with two stamp columns and keep no
``row_history``; every other record-fed table is run-scale and carries ``field_versions_json``.
Ledger tables (``record_*``, ``blocked_records``, ``row_history``, ``schema_migration``) are not
record-fed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import ownership as OW
from . import vocab as V

# --------------------------------------------------------------------------------------------
# Column / table model
# --------------------------------------------------------------------------------------------

_HEX = "[0-9a-f]"
UTC_MS_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z"


def hex_glob(n: int) -> str:
    return _HEX * n


PATTERNS: Dict[str, str] = {
    # rendered with {col} substituted
    "SHA1_OR_UNVERIFIED": "({col} GLOB '%s' OR {col} = 'unverified')" % hex_glob(40),
    "SHA256_OR_UNVERIFIED": "({col} GLOB '%s' OR {col} = 'unverified')" % hex_glob(64),
    "SHA256": "({col} GLOB '%s')" % hex_glob(64),
    # the judging code's stamp (harvest.validator_stamp: 16 hex) or the literal 'unverified' (a refusal recorded before the stamp was kept)
    "STAMP16_OR_UNVERIFIED": "({col} GLOB '%s' OR {col} = 'unverified')" % hex_glob(16),
    "UTC_MS": "({col} GLOB '%s')" % UTC_MS_GLOB,
    "UTC_MS_OR_UNVERIFIED": "({col} GLOB '%s' OR {col} = 'unverified')" % UTC_MS_GLOB,
    "JSON": "(json_valid({col}))",
    "FIT_LR_SELECTED_BY": "({col} IN (%s) OR {col} GLOB 'inherited_from:*')" % ", ".join(
        "'%s'" % v for v in V.FIT_LR_SELECTED_BY_FIXED),
    "SUBSAMPLE_RULE": "({col} IN (%s) OR {col} GLOB 'seeded_sample:*' OR {col} GLOB 'paired_with:*')" % ", ".join(
        "'%s'" % v for v in V.SUBSAMPLE_RULE_FIXED),
    "STRENGTH_RULE": "({col} = 'operating_point' OR {col} GLOB 'fixed:*' OR {col} GLOB 'ladder:*')",
    "NONNEG": "({col} >= 0)",
}


@dataclass
class Column:
    name: str
    type: str                      # TEXT | INTEGER | REAL | BLOB
    nn: bool = False               # NOT NULL
    default: Optional[str] = None  # SQL literal text, e.g. "'{}'" or "0"
    vocab: Optional[str] = None    # name in vocab.VOCABULARIES -> CHECK (col IN (...))
    pattern: Optional[str] = None  # name in PATTERNS -> CHECK (<pattern>)
    fk: Optional[Tuple[str, str]] = None   # (table, column); rendered as REFERENCES unless the column has a sentinel
    sentinel: Optional[object] = None      # the "absent" value of a composite-key column ('' / -1); an FK on a
                                           # sentinel column is enforced by trigger (WHEN NEW.col <> sentinel), never REFERENCES
    fk_deferred: bool = False              # The FK holds for every VISIBLE row (native, or an accepted revision's) and is
                                           # enforced by trigger; a PENDING revision's rows name parents the activation folds first
    derived: bool = False                  # set by the harvester (stamps, pointers, ranks); never carried in a record payload
    why_null: Optional[str] = None
    doc: str = ""

    def __post_init__(self):
        if not self.nn and not self.why_null:
            raise ValueError(f"nullable column {self.name} has no why_null reason")
        if self.nn and self.why_null:
            raise ValueError(f"NOT NULL column {self.name} carries a why_null reason")
        if self.sentinel is not None and not self.nn:
            raise ValueError(f"sentinel column {self.name} must be NOT NULL")


@dataclass
class Table:
    name: str
    columns: List[Column]
    pk: Sequence[str]
    doc: str = ""
    uniques: List[Sequence[str]] = field(default_factory=list)
    indexes: List[Tuple[str, Sequence[str], Optional[str]]] = field(default_factory=list)  # (name, cols, where)
    checks: List[Tuple[str, str]] = field(default_factory=list)  # (name, sql)
    scale: str = "run"             # run | item | ledger | dimension(=run)

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]


def C(name, type_="TEXT", nn=False, default=None, vocab=None, pattern=None, fk=None, why=None, doc="", sentinel=None,
      derived=False, fk_deferred=False):
    if name.endswith("_utc") and type_ == "TEXT" and pattern is None:
        pattern = "UTC_MS"
    if name.endswith("_json") and type_ == "TEXT" and pattern is None:
        pattern = "JSON"
    return Column(name, type_, nn=nn, default=default, vocab=vocab, pattern=pattern, fk=fk, sentinel=sentinel, derived=derived,
                  why_null=why, doc=doc, fk_deferred=fk_deferred)


def revision_tag() -> Column:
    """The item tables' revision tag: part of the natural key, so a pending revision's row coexists with
    the current one; visibility is derived — a row is read iff its revision is the source's head accepted revision
    (or '' for natively emitted rows). FK by trigger ('' is the sentinel)."""
    return C("source_revision_id", nn=True, default="''", fk=("source_revision", "revision_id"), sentinel="",
             doc="The source_revision that carries this row ('' for natively emitted rows; FK by trigger). Part of the key: rows of a "
                 "pending or superseded revision coexist with the head's, invisible to every reader.")


def B(name, nn=False, default=None, why=None, doc=""):
    """INTEGER boolean 0/1."""
    return C(name, "INTEGER", nn=nn, default=default, vocab="BOOLEAN", why=why, doc=doc)


def render_metric_id_check(col: str = "metric_id") -> str:
    """The metric-naming CHECK: the column is not a bare generic name and not an unqualified degeneration
    statistic — the banned set is `vocab.banned_metric_ids()`, so the DDL, the validator and the docs render one list."""
    vals = ", ".join(f"'{v}'" for v in V.banned_metric_ids())
    return f"({col} NOT IN ({vals}))"


#: (table, column) pairs the metric-naming CHECK guards; validate.py refuses these by name before any insert.
METRIC_ID_GUARDED: Tuple[Tuple[str, str], ...] = (("run_metric", "metric_id"), ("run_contrast", "metric_id"), ("eval", "headline_metric_id"))


# Common tails ------------------------------------------------------------------------------

def run_scale_tail() -> List[Column]:
    """The harvester-written tail of every run-scale table; `extra_json` is the one column a payload may set."""
    return [
        C("field_versions_json", nn=True, default="'{}'", derived=True,
          doc="{column: [written_utc, record_sha256] of the record that last set it} — the FULL order key (S2)."),
        C("extra_json", nn=True, default="'{}'", doc="Unknown record fields, kept here and never dropped."),
        C("record_uri", nn=True, derived=True, doc="The record file (logical path) that produced the current row."),
        C("ingest_seq", "INTEGER", nn=True, derived=True,
          doc="The producing FILE's rank in the ledger's arrival order (record_ingest: arrival_utc, then first_written_utc, sha, uri); "
              "derived, never allocated — every row above a head a reader has seen is new to it."),
    ]


def item_scale_tail() -> List[Column]:
    """The harvester-written tail of every item-scale table (two stamp columns, the file pointer, the rank)."""
    return [
        C("written_utc", "INTEGER", nn=True, pattern="NONNEG", derived=True,
          doc="Record written_utc as epoch milliseconds — stamp column 1."),
        C("record_sha256", "BLOB", nn=True, derived=True,
          doc="Record sha256 (32 bytes) — stamp column 2; (written_utc, record_sha256) is the row's order key."),
        C("extra_json", nn=True, default="'{}'", doc="Unknown record fields, never dropped."),
        C("record_file_id", "INTEGER", nn=True, fk=("record_ingest", "record_file_id"), derived=True,
          doc="The record file that produced the row (integer id; `v_item_pointer` renders the URI)."),
        C("ingest_seq", "INTEGER", nn=True, derived=True,
          doc="The producing FILE's rank in the ledger's arrival order (record_ingest; derived, never allocated)."),
    ]


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------

TABLES: List[Table] = []


def table(t: Table) -> Table:
    TABLES.append(t)
    return t


table(Table(
    "schema_migration", scale="ledger", pk=("version",),
    doc="Applied migrations (`migrations/NNNN_*.sql`), one row each.",
    columns=[
        C("version", "INTEGER", nn=True),
        C("applied_utc", nn=True),
        C("description", nn=True),
    ]))

table(Table(
    "paper_round", pk=("round_id",),
    doc="A paper iteration (`r2`, `r3`). Changing the round is a new roster version.",
    columns=[
        C("round_id", nn=True),
        C("opened_utc", nn=True),
        C("closed_utc", why="the open round has not closed"),
        C("description", nn=True),
        C("paper_build_commit", pattern="SHA1_OR_UNVERIFIED", why="the paper-figures sha roster v1 was transcribed from is pinned after transcription"),
        C("roster_version", why="a round whose roster loaded before migration 0016 carries none; every roster_entry of the round then matches, whatever its own stamp",
          doc="The roster version (`v9`) the round's latest `roster load` wrote; `v_roster_match` admits an entry of the round only under this version (0016)."),
    ] + run_scale_tail()))

table(Table(
    "model", pk=("model_id",),
    doc="A checkpoint at a pinned revision (base, instruct or chat).",
    columns=[
        C("model_id", nn=True),
        C("family", nn=True, vocab="MODEL_FAMILIES"),
        C("size_label", nn=True),
        C("checkpoint_kind", nn=True, vocab="CHECKPOINT_KINDS"),
        C("hf_repo", nn=True),
        C("hf_revision", nn=True, pattern="SHA1_OR_UNVERIFIED"),
        C("mirror_of", why="only mirrored repos name their upstream"),
        C("chat_template_sha256", nn=True, pattern="SHA256_OR_UNVERIFIED"),
        C("chat_template_note", why="free-text note, optional"),
        C("tokenizer_sha256", pattern="SHA256_OR_UNVERIFIED", why="not recorded for every checkpoint"),
        B("generation_config_present", why="unverified for a checkpoint never served and never read (reason in field_confidence_json)"),
        C("generation_config_json", why="absent when the checkpoint ships no generation_config.json"),
        C("generation_config_source_utc", why="absent when the checkpoint ships no generation_config.json"),
        C("thinking_channel", nn=True, vocab="THINKING_CHANNELS"),
        C("thinking_header_moves", "INTEGER", why="only thinking-capable checkpoints record header moves"),
        C("architecture_note", why="free text (hybrid Mamba: fused kernels, prefix caching off)"),
        C("paper_status", nn=True, vocab="MODEL_PAPER_STATUSES"),
        C("paper_status_ruling_id", fk=("ruling", "ruling_id"), why="set only when a ruling changed the status"),
        C("field_confidence_json", nn=True, default="'{}'", doc="{column: {confidence, reason}} for every seeded value that is inferred or unverified."),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "si_text", pk=("si_id",),
    doc="Every system instruction or constitution text an object distils or a prompted object declares; the text lives at `source_uri`.",
    uniques=[("si_text_sha256",)],
    columns=[
        C("si_id", nn=True),
        C("si_text_sha256", nn=True, pattern="SHA256"),
        C("si_token_count_by_model_json", nn=True, default="'{}'"),
        C("si_kind", nn=True, vocab="SI_KINDS"),
        C("source_uri", nn=True),
        C("first_used_utc", why="unknown for texts recovered from a store"),
        C("field_confidence_json", nn=True, default="'{}'"),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "fit_search", pk=("search_id",),
    doc="One hyperparameter search.",
    columns=[
        C("search_id", nn=True),
        C("campaign", nn=True),
        C("target_object_class_json", nn=True),
        C("method", nn=True, vocab="SEARCH_METHODS"),
        C("n_trials", "INTEGER", nn=True),
        C("n_trials_completed", "INTEGER", nn=True),
        C("search_space_json", nn=True),
        C("objective", nn=True),
        C("objective_direction", nn=True, vocab="OBJECTIVE_DIRECTIONS"),
        C("winner_params_json", why="a search may not have concluded"),
        C("winner_trial_id", why="a search may not have concluded"),
        C("winner_objective", "REAL", why="a search may not have concluded"),
        B("argmin_interior", why="only fitted-argmin searches evaluate interiority"),
        B("quadratic_fit_positive_definite", why="only quadratic-fit searches"),
        C("quadratic_fit_params_json", why="only quadratic-fit searches"),
        C("recentre_rounds", "INTEGER", nn=True, default="0"),
        C("confirmation_trial_id", why="no confirmation trial was run"),
        B("confirmation_within_noise", why="no confirmation trial was run"),
        C("noise_estimate", "REAL", why="no noise estimate was recorded"),
        C("grad_norm_reconnaissance_uri", why="no reconnaissance pass"),
        C("substrate", nn=True, vocab="FIT_SEARCH_SUBSTRATES",
          doc="`unverified` when no record beside the sweep names the substrate; the missing evidence is named in `field_confidence_json`."),
        C("job_ids_json", nn=True, default="'[]'"),
        C("code_sha", nn=True, pattern="SHA1_OR_UNVERIFIED"),
        C("config_sha256", pattern="SHA256_OR_UNVERIFIED", why="manual searches have no config file"),
        C("started_utc", nn=True),
        C("finished_utc", why="search in progress"),
        C("wandb_sweep_url", why="not every search logged to W&B"),
        C("trials_uri", why="the trial log may not be on disk"),
        C("producer_session", why="unknown for legacy searches"),
        C("field_confidence_json", nn=True, default="'{}'"),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "fit_trial", pk=("search_id", "trial_id"),
    doc="One trial of a search.",
    columns=[
        C("search_id", nn=True, fk=("fit_search", "search_id")),
        C("trial_id", nn=True),
        C("params_json", nn=True),
        C("objective_value", "REAL", why="failed or pruned trials have no objective"),
        C("status", nn=True, vocab="TRIAL_STATUSES"),
        C("clip_fraction", "REAL", why="not every trial log records clipping"),
        C("epochs_seen", "INTEGER", why="not every trial log records epochs"),
        C("wandb_run_id", why="not every trial logged to W&B"),
        C("started_utc", why="trial logs may lack timestamps"),
        C("finished_utc", why="trial logs may lack timestamps"),
    ] + run_scale_tail()))

table(Table(
    "eval", pk=("eval_id",),
    doc="The BENCHMARK only, never a variant of it: every GSM8K that ever ran is one `eval_id` with different item sets and instruments.",
    checks=[("eval_headline_metric_id_not_generic", render_metric_id_check("headline_metric_id"))],
    columns=[
        C("eval_id", nn=True),
        C("eval_category", nn=True, vocab="EVAL_CATEGORIES"),
        C("headline_metric_id", nn=True),
        C("metric_direction", nn=True, vocab="METRIC_DIRECTIONS"),
        C("default_item_set_id", why="soft reference (circular with item_set.eval_id); set once item sets exist"),
        B("attacker_required", nn=True),
        C("producer_spellings_json", nn=True, default="'[]'"),
        C("paper_status", nn=True, vocab="EVAL_PAPER_STATUSES"),
        C("paper_status_ruling_id", fk=("ruling", "ruling_id"), why="set only when a ruling changed the status"),
        C("field_confidence_json", nn=True, default="'{}'"),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "item_set", pk=("item_set_id",),
    doc="The exact prompt set: version and subsample.",
    columns=[
        C("item_set_id", nn=True),
        C("eval_id", fk=("eval", "eval_id"), why="a goal pool shared by several evals (the 259-goal jailbreak pool, the EM 8-question set); the per-eval sets reference it through `paired_with:<item_set_id>`"),
        C("dataset_version", nn=True),
        C("n_items", "INTEGER", why="unverified when the set's count was never read (reason in field_confidence_json)"),
        C("items_sha256", nn=True, pattern="SHA256_OR_UNVERIFIED"),
        C("source_uri", nn=True),
        C("item_id_scheme", nn=True),
        C("subsample_rule", nn=True, pattern="SUBSAMPLE_RULE"),
        C("pairing_note", why="only paired subsamples carry a note"),
        B("held_out_of_fit", why="unknown for external item sets"),
        C("field_confidence_json", nn=True, default="'{}'"),
    ] + run_scale_tail()))

table(Table(
    "judge", pk=("judge_id",),
    doc="A judge identity: sha256 over model, rubric sha, protocol sha and threshold rule.",
    columns=[
        C("judge_id", nn=True),
        C("judge_model", nn=True),
        C("judge_role", nn=True, vocab="JUDGE_ROLES"),
        C("rubric_name", nn=True),
        C("rubric_version", nn=True),
        C("rubric_sha256", nn=True, pattern="SHA256_OR_UNVERIFIED"),
        C("protocol_sha256", nn=True, pattern="SHA256_OR_UNVERIFIED"),
        C("score_scale", nn=True),
        C("threshold_rule", why="label-only judges have no threshold"),
        C("label_vocabulary_json", nn=True, default="'[]'"),
        C("judge_backend", nn=True, vocab="JUDGE_BACKENDS"),
        C("field_confidence_json", nn=True, default="'{}'"),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "instrument", pk=("instrument_id",),
    doc="The DECLARED measurement setting (harness, n-shot, CoT, thinking, scoring, decode, cap, serving, prompt padding).",
    # Migration 0012: a padded prompt is another INSTRUMENT, never a run column defaulting to 0 -- the padded stores also differ in
    # cap (16,384 vs 4,096) and window (98,304 vs 4,864), so a bare pad column would put them on one axis with the unpadded rows as if the
    # pad were the only difference. NULL is the instrument with NO padding recipe at all (every unpadded row); the two columns are set
    # together. Migration 0014: a pad of 0 is the recipe's ZERO POINT -- a matched control generated at the padded conditions'
    # own tree, geometry, width, cap and preset, differing from them only in carrying no filler -- and files as 0 WITH the contract hash,
    # never as NULL: under 0013 a NULL-padded row would be selector-matched into the ordinary static comparisons beside the legacy 4,096-cap
    # rows (the confound from the other side), while 0 + the contract keeps it off every selector and reachable only by name as the origin
    # of the 0 / 16,384 / 65,536 series. 0012 refused 0 because no producer had run one; 0014 relaxes that to >= 0.
    checks=[("instrument_padding_pairing", "((padding_tokens IS NULL) = (padding_identity_sha256 IS NULL) AND (padding_tokens IS NULL OR padding_tokens >= 0))")],
    columns=[
        C("instrument_id", nn=True),
        C("eval_id", nn=True, fk=("eval", "eval_id")),
        C("item_set_id", nn=True, fk=("item_set", "item_set_id")),
        C("harness", nn=True, vocab="HARNESSES"),
        C("harness_version", nn=True),
        C("harness_config_hash", nn=True),
        C("n_shot", "INTEGER", why="not a shot task"),
        C("prompt_template_sha256", nn=True, pattern="SHA256_OR_UNVERIFIED"),
        C("cot_prompting", vocab="COT_PROMPTING", why="not applicable to attack and judge-graded evals"),
        C("scoring_rule", nn=True, vocab="SCORING_RULES"),
        C("answer_extraction", nn=True, vocab="ANSWER_EXTRACTIONS"),
        C("token_cap_tokens", "INTEGER", why="unverified for a superseded instrument whose cap was never read (reason in field_confidence_json)"),
        C("token_cap_ladder_json", nn=True, default="'[]'"),
        C("ladder_promotion_rule", why="a fixed cap has no ladder"),
        C("read_at", nn=True, vocab="READ_AT"),
        C("reply_cap_tokens", "INTEGER", why="only multi-turn instruments cap replies separately"),
        C("context_window_tokens", "INTEGER", why="not declared for every instrument"),
        C("padding_tokens", "INTEGER", why="no padding recipe: the instrument carries no filler contract at all (NULL; every unpadded row). 0 is a VALUE, never NULL: the recipe's zero point, a matched control generated under the padded conditions' own tree, geometry, cap and preset with no filler, paired with the same contract hash (CHECK instrument_padding_pairing, >= 0 since migration 0014)"),
        C("padding_identity_sha256", pattern="SHA256",
          why="no padding recipe: no filler contract to hash (NULL pairs with a NULL padding_tokens; sha256 over the canonical JSON of {filler_sha256, mode, placement, separator, tolerance}, identity.padding_identity_sha256; the block itself rides extra_json.static_padding). The zero point of a recipe carries the contract hash beside padding_tokens 0"),
        C("decode_source", nn=True, vocab="DECODE_SOURCES"),
        C("decode_temperature", "REAL", why="not sent (engine default) — resolved value in the run's decode_resolved_json"),
        C("decode_top_p", "REAL", why="not sent (engine default)"),
        C("decode_top_k", "INTEGER", why="not sent (engine default)"),
        C("decode_min_p", "REAL", why="not sent (engine default)"),
        C("decode_repetition_penalty", "REAL", why="not sent (engine default)"),
        C("decode_presence_penalty", "REAL", why="not sent (engine default)"),
        C("decode_seed_rule", nn=True, vocab="DECODE_SEED_RULES"),
        C("thinking_mode", nn=True, vocab="THINKING_MODES"),
        C("n_trials", "INTEGER", why="unverified for a superseded instrument whose trial count was never read (reason in field_confidence_json)"),
        C("bon_n", "INTEGER", why="only best-of-n instruments"),
        C("max_turns", "INTEGER", why="only multi-turn instruments"),
        C("serving_path", nn=True, vocab="SERVING_PATHS"),
        C("applied_positions_mode", nn=True, vocab="APPLIED_POSITIONS_MODES"),
        C("attacker_model_id", fk=("model", "model_id"), why="only attacker-driven evals"),
        C("attacker_decode_json", why="only attacker-driven evals"),
        C("judge_id", fk=("judge", "judge_id"), why="rule-scored evals (exact match, sandbox) have no judge"),
        C("prefill_scheme", why="only prefill-scored instruments"),
        C("field_confidence_json", nn=True, default="'{}'"),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "decode_policy", pk=("policy_id",),
    doc="THE authority for sampling parameters: one row per model × (eval or eval category) × thinking mode per version.",
    uniques=[("policy_ref", "policy_version")],
    checks=[("decode_policy_eval_xor_category", "((eval_id IS NULL) <> (eval_category IS NULL))")],
    columns=[
        C("policy_id", nn=True),
        C("policy_ref", nn=True, doc="Stable name the harness's sampling MODES map to (S8, C34)."),
        C("policy_version", "INTEGER", nn=True),
        C("model_id", nn=True, fk=("model", "model_id")),
        C("eval_id", fk=("eval", "eval_id"), why="a row names an eval OR an eval category (CHECK)"),
        C("eval_category", vocab="EVAL_CATEGORIES", why="a row names an eval OR an eval category (CHECK)"),
        C("thinking_mode", nn=True, vocab="THINKING_MODES"),
        C("decode_source", nn=True, vocab="DECODE_SOURCES"),
        C("temperature", "REAL", why="deliberately not sent; what it resolves to is in params_not_sent_json"),
        C("top_p", "REAL", why="deliberately not sent; see params_not_sent_json"),
        C("top_k", "INTEGER", why="deliberately not sent; see params_not_sent_json"),
        C("min_p", "REAL", why="deliberately not sent; see params_not_sent_json"),
        C("repetition_penalty", "REAL", why="deliberately not sent; see params_not_sent_json"),
        C("presence_penalty", "REAL", why="deliberately not sent; see params_not_sent_json"),
        C("max_new_tokens_rule", nn=True),
        C("seed_rule", nn=True, vocab="DECODE_SEED_RULES"),
        C("params_not_sent_json", nn=True, default="'{}'"),
        C("source_citation", nn=True),
        C("ruling_id", fk=("ruling", "ruling_id"), why="only grandfathered values cite a ruling"),
        B("is_default", nn=True, default="1", doc="The default policy for its (model, eval-or-category, thinking) key (C34)."),
        C("effective_from_utc", nn=True, pattern="UTC_MS_OR_UNVERIFIED",
          doc="Ruling date at midnight UTC (date-only rulings; `extra_json.effective_from_confidence`), or `unverified` with a reason."),
        C("superseded_by_policy_id", fk=("decode_policy", "policy_id"), why="current policies are not superseded"),
        C("field_confidence_json", nn=True, default="'{}'"),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "sampling_mode_policy", pk=("model_id", "sampling_mode"),
    doc="Harness sampling MODE → decode_policy row (`contracts.sampling_for(model_key, mode)`; S8, C34).",
    columns=[
        C("model_id", nn=True, fk=("model", "model_id")),
        C("sampling_mode", nn=True, vocab="SAMPLING_MODES"),
        C("policy_id", nn=True, fk=("decode_policy", "policy_id")),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "computation", pk=("computation_id",),
    doc="A statistic's algorithm, version and seed provenance (F9); joins the run_metric / run_contrast keys.",
    columns=[
        C("computation_id", nn=True),
        C("algorithm", nn=True),
        C("version", nn=True),
        C("seed_provenance", nn=True),
        C("code_sha", nn=True, pattern="SHA1_OR_UNVERIFIED"),
        C("notes", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "ruling", pk=("ruling_id",),
    doc="A user ruling (include / exclude / supersede / coexist / operating_point / descope / drop_eval / reference); self-retiring when `expires_when_json` is set.",
    columns=[
        C("ruling_id", nn=True),
        C("round_id", nn=True, fk=("paper_round", "round_id")),
        C("action", nn=True, vocab="RULING_ACTIONS"),
        C("scope_kind", nn=True, vocab="SCOPE_KINDS"),
        C("scope_json", nn=True),
        C("target_json", nn=True, default="'{}'"),
        C("expires_when_json", why="not every ruling is self-retiring"),
        C("expired_utc", why="stamped once, when found-and-resolving evidence first said the condition held; never deleted"),
        C("evidence_state", nn=True, default="'not_evaluated'", vocab="EVIDENCE_STATES"),
        C("evidence_evaluated_utc", why="no evaluation has run yet"),
        C("reason", nn=True),
        C("ruled_utc", nn=True),
        C("ruled_utc_precision", nn=True, default="'datetime'", vocab="RULED_UTC_PRECISIONS",
          doc="Many recorded rulings are date-only; `date` means ruled_utc carries midnight UTC of that day."),
        C("ruled_by", nn=True),
        C("source_ref", nn=True),
    ] + run_scale_tail()))

table(Table(
    "ruling_state", pk=("ruling_id", "evaluated_utc"),
    doc="One expiry / evidence evaluation of a ruling (a harvester record); `ruling.evidence_state` / `expired_utc` are recomputed from these rows.",
    columns=[
        C("ruling_id", nn=True, fk=("ruling", "ruling_id")),
        C("evaluated_utc", nn=True),
        C("evidence_state", nn=True, vocab="EVIDENCE_STATES"),
        B("expired", nn=True),
        C("evidence_json", nn=True, default="'{}'"),
    ] + run_scale_tail()))

_OBJECT_COLUMNS = [
    C("object_id", nn=True),
    C("object_kind", nn=True, vocab="OBJECT_KINDS"),
    C("paper_label", nn=True, vocab="PAPER_LABELS"),
    C("campaign", nn=True),
    C("campaign_cell_id_legacy", why="natively emitted objects carry no legacy id"),
    C("object_role", nn=True, vocab="OBJECT_ROLES"),
    C("fit_target_kind", nn=True, vocab="FIT_TARGET_KINDS"),
    C("fit_teacher_si_id", fk=("si_text", "si_id"), why="only system_instruction_distillation (paired-nullness CHECK)"),
    C("fit_teacher_si_text_sha256", fk=("si_text", "si_text_sha256"), pattern="SHA256", why="only system_instruction_distillation (paired-nullness CHECK)"),
    C("fit_prompt_pool_id", why="only system_instruction_distillation (paired-nullness CHECK)"),
    C("fit_prompt_pool_sha256", pattern="SHA256_OR_UNVERIFIED", why="only system_instruction_distillation (paired-nullness CHECK)"),
    C("fit_corpus_id", why="only corpus_document_fit (paired-nullness CHECK)"),
    C("fit_corpus_description", why="only corpus_document_fit; optional description"),
    C("fit_corpus_sha256", pattern="SHA256_OR_UNVERIFIED", why="only corpus_document_fit (paired-nullness CHECK)"),
    C("fit_corpus_n_documents", "INTEGER", why="only corpus_document_fit; not every sidecar records it"),
    C("fit_corpus_n_tokens", "INTEGER", why="only corpus_document_fit; not every sidecar records it"),
    C("fit_contrast_sets_json", why="only contrastive_direction (paired-nullness CHECK)"),
    C("fit_model_id", fk=("model", "model_id"), why="`none` and declared-prompt objects were not fit"),
    C("fit_chat_template_sha256", pattern="SHA256_OR_UNVERIFIED", why="`none` and declared-prompt objects were not fit"),
    C("fit_thinking_render", nn=True, vocab="FIT_THINKING_RENDERS"),
    C("fit_objective", nn=True, vocab="FIT_OBJECTIVES"),
    C("fit_positions", nn=True, vocab="FIT_POSITIONS"),
    C("fit_response_header_len", "INTEGER", why="only response_incl_assistant_header positions"),
    C("fit_capture_site", vocab="FIT_CAPTURE_SITES", why="only captured directions name a site"),
    C("fit_selected_position_offset", "INTEGER", why="only searched positions (Arditi)"),
    C("fit_selected_token_text", why="only searched positions (Arditi)"),
    C("fit_layers_json", why="`none` and declared-prompt objects have no layers"),
    C("fit_layer_index_convention", vocab="LAYER_INDEX_CONVENTIONS", why="`none` and declared-prompt objects have no layers"),
    C("fit_layer_count", "INTEGER", why="`none` and declared-prompt objects have no layers"),
    C("fit_layer_selection_rule", vocab="LAYER_SELECTION_RULES", why="`none` and declared-prompt objects have no layers"),
    C("served_layer_shift", "INTEGER", why="only objects served at a shifted layer"),
    C("fit_rank", "INTEGER", why="LoRA objects only"),
    C("fit_alpha", "REAL", why="LoRA objects only"),
    C("fit_target_modules_json", why="LoRA objects only"),
    C("fit_pool_sha256", pattern="SHA256_OR_UNVERIFIED", why="not every fit names a pool"),
    C("fit_sample_manifest_sha256", pattern="SHA256_OR_UNVERIFIED", why="not every fit writes a sample manifest"),
    C("fit_completion_tokens", "INTEGER", why="not every sidecar records it"),
    C("fit_heldout_fraction", "REAL", why="not every fit holds out"),
    C("fit_split_seed", "INTEGER", why="not every fit records a split seed"),
    B("fit_probe_sources_pinned_out", why="not every fit records it"),
    C("fit_hyperparameters_json", why="`none` and declared-prompt objects were not fit"),
    C("fit_lr", "REAL", why="not applicable to unfitted objects"),
    C("fit_lr_schedule", why="not applicable to unfitted objects"),
    C("fit_effective_batch", "INTEGER", why="not applicable to unfitted objects"),
    C("fit_realized_batch_json", why="not applicable to unfitted objects"),
    C("fit_epochs_cap", "INTEGER", why="not applicable to unfitted objects"),
    C("fit_epochs_seen", "INTEGER", why="not applicable to unfitted objects"),
    C("fit_best_epoch", "INTEGER", why="not applicable to unfitted objects"),
    C("fit_patience", "INTEGER", why="not applicable to unfitted objects"),
    B("fit_early_stopped", why="not applicable to unfitted objects"),
    C("fit_stopped_on", vocab="FIT_STOPPED_ON", why="not applicable to unfitted objects"),
    C("fit_steps", "INTEGER", why="not applicable to unfitted objects"),
    C("fit_clip_fraction", "REAL", why="not applicable to unfitted objects"),
    C("fit_final_train_loss", "REAL", why="not applicable to unfitted objects"),
    C("fit_final_heldout_loss", "REAL", why="not applicable to unfitted objects"),
    C("fit_best_heldout_loss", "REAL", why="not applicable to unfitted objects"),
    C("fit_heldout_curve_uri", why="not applicable to unfitted objects"),
    C("fit_seed", "INTEGER", why="unknown seeds are `fit_seed_confidence = unverified`, never a guessed number"),
    C("fit_seed_confidence", nn=True, vocab="FIT_SEED_CONFIDENCE"),
    C("fit_replicate_label", why="only replicate fits carry a label"),
    C("fit_search_id", fk=("fit_search", "search_id"), why="only searched hyperparameters link a search"),
    C("fit_lr_selected_by", pattern="FIT_LR_SELECTED_BY", why="not applicable to unfitted objects"),
    B("fit_argmin_interior", why="only searched learning rates"),
    C("fit_code_sha", pattern="SHA1_OR_UNVERIFIED", why="not applicable to unfitted objects"),
    C("fit_config_path", why="not applicable to unfitted objects"),
    C("fit_config_sha256", pattern="SHA256_OR_UNVERIFIED", why="not applicable to unfitted objects"),
    C("fit_writer", why="not applicable to unfitted objects"),
    C("fit_started_utc", why="not applicable to unfitted objects"),
    C("fit_finished_utc", why="not applicable to unfitted objects"),
    C("fit_walltime_s", "REAL", why="walltime is recorded where known, never estimated"),
    C("fit_substrate", vocab="SUBSTRATES", why="not applicable to unfitted objects"),
    C("fit_job_id", why="not applicable to unfitted objects"),
    C("fit_node_hostname", why="not applicable to unfitted objects"),
    C("fit_gpu_type", why="not applicable to unfitted objects"),
    C("fit_gpu_count", "INTEGER", why="not applicable to unfitted objects"),
    C("fit_wandb_run_id", why="not every fit logged to W&B"),
    C("fit_wandb_url", why="not every fit logged to W&B"),
    C("fit_producer_session", why="unknown for legacy fits"),
    C("derivation", nn=True, vocab="DERIVATIONS"),
    C("parent_object_ids_json", why="only derived objects (paired-nullness CHECK)"),
    C("derivation_params_json", why="only derived objects (paired-nullness CHECK)"),
    C("base_scale", "REAL", why="only scaled objects"),
    C("si_text_sha256", fk=("si_text", "si_text_sha256"), pattern="SHA256", why="only declared prompts (paired-nullness CHECK)"),
    C("si_token_count", "INTEGER", why="only declared prompts (paired-nullness CHECK)"),
    C("artifact_sha256", nn=True, doc="The object's bytes (sha256); `none` for the unsteered object; the text sha for a declared prompt."),
    C("artifact_bytes", "INTEGER", why="`none` and declared-prompt objects have no artifact file"),
    C("artifact_locations_json", nn=True, default="'[]'"),
    C("object_sidecars_json", nn=True, default="'[]'"),
    C("status", nn=True, vocab="OBJECT_STATUSES"),
    C("superseded_by_object_id", fk=("object", "object_id"), why="only superseded objects"),
    C("status_ruling_id", fk=("ruling", "ruling_id"), why="set only when a ruling set the status"),
    C("field_confidence_json", nn=True, default="'{}'"),
    C("notes", why="free text"),
] + run_scale_tail() + [
    # epoch-checkpoint lineage, first-class: set from the record's own field when stated, else lifted from `extra_json.parent_object_id`
    # / `extra_json.named_role` by the fold (reducer.lift_object_fields; producers wrote the keys there before the columns existed); an explicit
    # null resets; per-column latest statement. The parent is a dependency at fold time (blocked until it folds, never refused for that); a
    # parent equal to the object's own id and a named_role outside OBJECT_NAMED_ROLES are refused by name. Declared AFTER the run-scale tail
    # so the live table takes both by ALTER TABLE ADD COLUMN
    C("parent_object_id", fk=("object", "object_id"), why="only epoch-checkpoint children name the fit they belong to"),
    C("named_role", vocab="OBJECT_NAMED_ROLES", why="only epoch-checkpoint children carry a named role; the parent and every other object carry none"),
]


def render_fit_target_check() -> str:
    """The complete paired-nullness CHECK for ``object.fit_target_kind`` (F16), from ``vocab.FIT_TARGET_RULES``."""
    branches = []
    for kind, rule in V.FIT_TARGET_RULES.items():
        parts = [f"fit_target_kind = '{kind}'"]
        objs = ", ".join(f"'{v}'" for v in rule["fit_objective"])
        parts.append(f"fit_objective IN ({objs})")
        ders = ", ".join(f"'{v}'" for v in rule["derivation"])
        parts.append(f"derivation IN ({ders})")
        for col, val in rule["equals"]:
            parts.append(f"{col} = '{val}'")
        for col in rule["nonnull"]:
            parts.append(f"{col} IS NOT NULL")
        for col in rule["null"]:
            parts.append(f"{col} IS NULL")
        branches.append("(" + "\n         AND ".join(parts) + ")")
    return "(\n     " + "\n  OR ".join(branches) + "\n  )"


table(Table(
    "object", pk=("object_id",),
    doc="The fitted or declared intervention, with its complete fit record.",
    indexes=[("idx_object_artifact_sha", ("artifact_sha256",), "artifact_sha256 <> 'none' AND artifact_sha256 NOT LIKE 'si:%'")],
    checks=[("object_fit_target_kind_pairing", render_fit_target_check())],
    columns=_OBJECT_COLUMNS))

_RUN_COLUMNS = [
    C("run_id", nn=True, doc="`identity.run_id(campaign, producer_key, producer_object_key, eval_id, instrument_id)`."),
    C("campaign", nn=True),
    C("producer_run_key_legacy", nn=True, doc="The producer's own EXECUTION key verbatim (F1)."),
    C("producer_object_key_legacy", nn=True, doc="The producer's OBJECT discriminator within that key (S1); `none` for an unsteered execution."),
    C("lifecycle_state", nn=True, vocab="LIFECYCLE_STATES"),
    C("queued_utc", why="set when the state is reached (trigger trg_run_state_utc requires it from `queued` on)"),
    C("started_utc", why="set when the state is reached (trigger trg_run_state_utc requires it from `running` on)"),
    C("generated_utc", why="set when the state is reached (trigger trg_run_state_utc requires it from `generated` on)"),
    C("judged_utc", why="set when the state is reached (trigger trg_run_state_utc requires it from `judged` on)"),
    C("ingested_utc", why="set by the harvester's run_ingest record (trigger requires it at `ingested`)"),
    C("finished_utc", why="terminal time of failed / cancelled runs (trigger requires it for those states)"),
    C("walltime_s", "REAL", why="recorded where sacct / Modal / the store span allow, never estimated"),
    C("walltime_source", vocab="WALLTIME_SOURCES", why="null when walltime_s is null"),
    C("model_id", nn=True, fk=("model", "model_id")),
    C("model_revision", pattern="SHA1_OR_UNVERIFIED", why="observed at serving (RUN_VERIFIED_FIELDS); legacy rows may lack it"),
    C("object_id", nn=True, fk=("object", "object_id")),
    C("transfer_kind", nn=True, vocab="TRANSFER_KINDS"),
    C("strength_coefficient", "REAL", why="`none` and prompted objects have no coefficient"),
    C("strength_nominal", "REAL", why="`none` and prompted objects have no nominal dose"),
    C("strength_nominal_source", vocab="STRENGTH_NOMINAL_SOURCES", why="null when strength_nominal is null"),
    C("strength_unit", nn=True, default="'unverified'", vocab="STRENGTH_UNITS",
      doc="the unit of strength_nominal: two dose ladders scaling different quantities never join into one curve"),
    B("is_operating_point", why="unknown without an operating_point ruling"),
    C("operating_point_id_legacy", why="only producers that filed an operating-point id"),
    C("eval_id", nn=True, fk=("eval", "eval_id")),
    C("item_set_id", nn=True, fk=("item_set", "item_set_id")),
    C("instrument_id", nn=True, fk=("instrument", "instrument_id")),
    C("instrument_match", nn=True, vocab="INSTRUMENT_MATCH"),
    C("harness", nn=True, vocab="HARNESSES"),
    C("harness_version", nn=True),
    C("n_shot", "INTEGER", why="not a shot task"),
    C("cot_prompting", vocab="COT_PROMPTING", why="not applicable to attack and judge-graded evals"),
    C("scoring_rule", nn=True, vocab="SCORING_RULES"),
    C("prompt_template_sha256", nn=True, pattern="SHA256_OR_UNVERIFIED"),
    C("judge_id", fk=("judge", "judge_id"), why="primary judge; null before any judgment and for rule-scored evals"),
    C("pass_label", why="only coexisting re-measurements carry a pass label"),
    C("trial_index", "INTEGER", nn=True, default="0"),
    C("n_items_planned", "INTEGER", why="unknown for legacy stores"),
    C("n_items_generated", "INTEGER", why="not yet generated"),
    C("n_items_judged", "INTEGER", why="not yet judged"),
    C("seed_base", "INTEGER", why="only seeded decodes"),
    C("decode_seed_rule", nn=True, vocab="DECODE_SEED_RULES"),
    C("decode_policy_id", fk=("decode_policy", "policy_id"),
      why="derived (decode_match.stamp_run_rows from v_run_decode_policy_match): set only when decode_policy_match = exact",
      doc="The ONE decode_policy row the run's sent parameters match exactly (derived at the end of every ingest cycle and of a rebuild; never a hand edit)."),
    C("decode_policy_match", nn=True, default="'no_policy'", vocab="DECODE_POLICY_MATCH",
      doc="Derived per run from v_run_decode_policy_match (exact / ambiguous / no_match / unrecorded); `no_policy` = the producer's placeholder before derivation."),
    C("decode_sent_json", nn=True, default="'{}'"),
    C("decode_resolved_json", nn=True, default="'{}'"),
    C("decode_temperature_sent", "REAL", why="not sent (resolved value in decode_resolved_json)"),
    C("decode_top_p_sent", "REAL", why="not sent (resolved value in decode_resolved_json)"),
    C("decode_top_k_sent", "INTEGER", why="not sent (resolved value in decode_resolved_json)"),
    C("decode_source", nn=True, vocab="DECODE_SOURCES"),
    C("token_cap_tokens", "INTEGER", nn=True),
    C("token_cap_rung_index", "INTEGER", why="a fixed cap has no ladder"),
    C("token_cap_ladder_json", nn=True, default="'[]'"),
    C("reply_cap_tokens", "INTEGER", why="only multi-turn runs cap replies separately"),
    C("context_window_tokens", "INTEGER", why="observed at serving; legacy rows may lack it"),
    C("served_max_num_seqs", "INTEGER", why="observed at serving; legacy rows may lack it"),
    C("served_max_num_seqs_source", vocab="MAX_NUM_SEQS_SOURCES", why="null when served_max_num_seqs is null"),
    C("thinking_mode", nn=True, vocab="THINKING_MODES"),
    C("serving_path", nn=True, vocab="SERVING_PATHS"),
    C("serving_engine_version", why="observed at serving; legacy rows may lack it"),
    C("serving_identity", why="observed at serving; legacy rows may lack it"),
    C("serving_config_sha256", pattern="SHA256_OR_UNVERIFIED", why="observed at serving; legacy rows may lack it"),
    C("adapter_fingerprint", why="only natively served adapters"),
    C("applied_positions_mode", nn=True, vocab="APPLIED_POSITIONS_MODES"),
    C("applied_layers_json", why="`none` objects have no layers"),
    C("multi_turn_steering_scope", nn=True, vocab="MULTI_TURN_SCOPES"),
    B("history_resent_each_turn", why="single-turn runs"),
    C("prefix_cache_policy", vocab="PREFIX_CACHE_POLICIES", why="not recorded for every serving path"),
    C("intervention_hash", why="`none` objects and legacy rows"),
    C("harness_config_hash", why="legacy rows may lack it"),
    C("env_digest", why="legacy rows may lack it"),
    C("code_sha", nn=True, pattern="SHA1_OR_UNVERIFIED"),
    C("code_branch", nn=True, default="'unknown'", doc="`unknown` when code_sha is not an ancestor of origin/main or a known branch: never guessed."),
    C("config_path", why="not every run launched from a config file"),
    C("config_sha256", pattern="SHA256_OR_UNVERIFIED", why="not every run launched from a config file"),
    C("submit_sha_pin", pattern="SHA1_OR_UNVERIFIED", why="only registered submissions carry a pin"),
    C("substrate", nn=True, vocab="SUBSTRATES"),
    C("substrate_source", nn=True, vocab="CONFIDENCE"),
    C("cluster_job_id", why="only cluster runs"),
    C("job_name", why="only cluster runs"),
    C("node_hostname", why="set at job start"),
    C("gpu_type", why="derived from inventory or nvidia-smi where available"),
    C("gpu_count", "INTEGER", why="set at job start"),
    C("modal_app_id", why="only Modal runs"),
    C("modal_call_id", why="only Modal runs launched after native emission"),
    C("wandb_run_id", why="only runs that logged to W&B"),
    C("attacker_model_id", fk=("model", "model_id"), why="only attacker-driven evals"),
    B("attacker_identity_attested", why="only attacker-driven evals"),
    C("store_uri", why="a queued run has no store yet"),
    C("store_completion_artifact_uri", why="set at generation"),
    C("store_completion_sha256", pattern="SHA256_OR_UNVERIFIED", why="set at generation"),
    C("view_legacy", why="only backfilled rows carry the producer's legacy transfer-setting string"),
    C("lane_source_legacy", why="only backfilled rows carry the paper's source string"),
    C("campaign_cell_id_legacy", why="only backfilled rows carry the producer's legacy id"),
    C("producer_session", why="unknown for legacy rows"),
    C("status_ruling_id", fk=("ruling", "ruling_id"), why="set only when a ruling changed the status"),
    C("superseded_by_run_id", fk=("run", "run_id"), why="only superseded runs"),
    C("field_confidence_json", nn=True, default="'{}'"),
    C("notes", why="free text"),
] + run_scale_tail()

table(Table(
    "run", pk=("run_id",),
    doc="One execution that produced rows, at the registry grain (producer key × producer object key × eval × instrument).",
    uniques=[("producer_run_key_legacy", "producer_object_key_legacy", "eval_id", "instrument_id")],
    indexes=[
        ("idx_run_key", ("model_id", "object_id", "eval_id", "instrument_id"), None),
        ("idx_run_lifecycle", ("lifecycle_state",), None),
        ("idx_run_campaign", ("campaign",), None),
        ("idx_run_cluster_job", ("cluster_job_id",), None),
    ],
    columns=_RUN_COLUMNS))

table(Table(
    "run_state_event", pk=("run_id", "state", "event_utc", "evidence_uri"),
    doc="Every lifecycle transition with its evidence; the scheduler's state is stored beside it, never as the truth.",
    columns=[
        C("run_id", nn=True, fk=("run", "run_id")),
        C("state", nn=True, vocab="LIFECYCLE_STATES"),
        C("event_utc", nn=True),
        C("evidence_kind", nn=True, vocab="EVIDENCE_KINDS"),
        C("evidence_uri", nn=True, default="''", doc="'' when the evidence is not a file."),
        C("scheduler_state", why="only cluster / Modal evidence carries a scheduler state"),
        C("note", why="free text"),
    ] + run_scale_tail()))

table(Table(
    "judgment", pk=("judgment_id",),
    doc="A judging of a run by one judge; two judges of one run are two rows.",
    indexes=[("idx_judgment_run", ("run_id", "judge_id"), None)],
    columns=[
        C("judgment_id", nn=True),
        C("run_id", nn=True, fk=("run", "run_id")),
        C("judge_id", nn=True, fk=("judge", "judge_id")),
        C("started_utc", why="a claim carries claimed_utc only"),
        C("finished_utc", why="in progress"),
        C("n_items_judged", "INTEGER", why="in progress. Counts the items with a VERDICT only; an item the judge declined is n_judge_refusals, never "
                                             "folded in here: judgment_covers_sql sums the two (and n_errors) against the planned count"),
        C("n_judge_refusals", "INTEGER", why="in progress or not counted. The items the judge DECLINED to score, counted apart from n_items_judged; "
                                                "coverage reads n_items_judged + n_judge_refusals + n_errors"),
        C("n_errors", "INTEGER", why="in progress or not counted"),
        C("judge_cache_kind", why="not every driver names a cache"),
        C("api_pool", vocab="API_POOLS", why="unknown for legacy judgments"),
        C("batch_ids_json", nn=True, default="'[]'"),
        C("grading_code_sha", pattern="SHA1_OR_UNVERIFIED", why="unknown for legacy judgments"),
        C("grading_config_sha256", pattern="SHA256_OR_UNVERIFIED", why="not every driver has a config"),
        C("rubric_sha256", pattern="SHA256_OR_UNVERIFIED", why="unknown for legacy judgments"),
        C("store_uri", why="in progress"),
        C("claimed_by_session", why="only claims"),
        C("claimed_utc", why="only claims"),
        C("status", nn=True, vocab="JUDGMENT_STATUSES"),
        C("superseded_by_judgment_id", fk=("judgment", "judgment_id"), why="only superseded judgments"),
        C("producer_session", why="unknown for legacy judgments"),
    ] + run_scale_tail() + [
        # the run's identity AS IT STOOD WHEN THE JUDGMENT WAS CLAIMED: a judgment whose binding differs from the
        # run's CURRENT instrument or required judge is non-current on the paper side (judgment_identity_current_sql); NULL (legacy rows,
        # a writer that does not know the instrument) is current. Declared AFTER the run-scale tail so the live table takes both by ALTER
        # TABLE ADD COLUMN (migrations.addable_columns), the shape 0020 used for item_result's marks
        C("claimed_instrument_id", fk=("instrument", "instrument_id"),
          why="legacy judgments and writers that do not know the run's instrument at claim time (NULL reads as current)"),
        C("claimed_required_judge_id", fk=("judge", "judge_id"),
          why="legacy judgments, writers that do not know the run's required judge at claim time, and runs whose instrument and run name no judge (NULL reads as current)"),
    ]))

_FRAC_COLS = list(V.DEGENERATION_FRAC_COLUMNS)

table(Table(
    "run_metric", pk=("run_id", "judgment_id", "metric_id", "threshold", "computation_id"),
    doc="Aggregate metrics per run × judgment × metric × threshold × computation; degeneration fractions mirror run_degeneration's `all`-scope rates.",
    checks=[("run_metric_metric_id_not_generic", render_metric_id_check())],
    columns=[
        C("run_id", nn=True, fk=("run", "run_id")),
        C("judgment_id", nn=True, default="''", fk=("judgment", "judgment_id"), sentinel="",
          doc="'' for rule-scored metrics (FK by trigger otherwise)."),
        C("metric_id", nn=True),
        C("threshold", "REAL", nn=True, default="-1", sentinel=-1.0, doc="-1 when the metric has no threshold."),
        C("computation_id", nn=True, default="''", fk=("computation", "computation_id"), sentinel="",
          doc="'' for producer-filed values (FK by trigger otherwise) (F9)."),
        C("producer_metric_spelling", why="only backfilled rows carry the producer's spelling"),
        C("value", "REAL", why="disposition rows carry no value"),
        C("k", "INTEGER", why="not every metric is a count"),
        C("n", "INTEGER", why="not every metric has a denominator"),
        C("ci_lo", "REAL", why="not every producer computed an interval; v_paper_long derives the 95 percent Wilson bound from k and n where none was filed, the metric is on the yes-or-no roster (WILSON_PROPORTION_METRIC_IDS) and the value is k / n (migration 0019)"),
        C("ci_hi", "REAL", why="not every producer computed an interval; v_paper_long derives the 95 percent Wilson bound from k and n where none was filed, the metric is on the yes-or-no roster (WILSON_PROPORTION_METRIC_IDS) and the value is k / n (migration 0019)"),
        C("ci_method", why="null when no interval; v_paper_long reads `wilson_95` where it derives one (migration 0019)"),
    ] + [C(f, "REAL", why="filled from run_degeneration's all-scope rate where the producer computed it") for f in _FRAC_COLS] + [
        C("n_abstentions", "INTEGER", why="not counted by every producer"),
        C("n_error_outcomes", "INTEGER", why="not counted by every producer"),
        C("n_incomplete", "INTEGER", why="not counted by every producer"),
        C("computed_by", nn=True, vocab="COMPUTED_BY"),
        C("computed_utc", why="unknown for legacy rows"),
        C("source_uri", why="natively emitted rows are their own source"),
        C("source_producers_json", nn=True, default="'[]'", doc="Every producer file a multi-producer row was assembled from."),
    ] + run_scale_tail()))

table(Table(
    "run_contrast", pk=("run_id", "anchor_run_id", "judgment_id", "metric_id", "threshold", "computation_id"),
    doc="Paired contrasts against an anchor run. The key carries the anchor, so a reader that re-anchors a run in a later revision leaves the "
        "older row here beside the new one; the fold withdraws the older key's observation, and `v_paper_long` joins only the contrasts whose "
        "observation key is still current under some revision (CURRENT_CONTRAST_CTE, migration 0017).",
    checks=[("run_contrast_metric_id_not_generic", render_metric_id_check())],
    columns=[
        C("run_id", nn=True, fk=("run", "run_id")),
        C("anchor_run_id", nn=True, fk=("run", "run_id")),
        C("judgment_id", nn=True, default="''", fk=("judgment", "judgment_id"), sentinel=""),
        C("metric_id", nn=True),
        C("threshold", "REAL", nn=True, default="-1", sentinel=-1.0),
        C("computation_id", nn=True, default="''", fk=("computation", "computation_id"), sentinel=""),
        C("anchor_judgment_id", nn=True, default="''", fk=("judgment", "judgment_id"), sentinel=""),
        C("anchor_kind", nn=True, vocab="ANCHOR_KINDS"),
        C("delta", "REAL", why="undefined when n_pairs is 0"),
        C("delta_lo", "REAL", why="no interval computed"),
        C("delta_hi", "REAL", why="no interval computed"),
        C("delta_p", "REAL", why="no p-value computed"),
        C("delta_method", nn=True),
        C("effect_size", "REAL", why="no effect size computed"),
        C("effect_kind", vocab="EFFECT_KINDS", why="null when effect_size is null"),
        C("n_pairs", "INTEGER", nn=True),
        C("computed_by", nn=True, vocab="COMPUTED_BY"),
        C("computed_utc", why="unknown for legacy rows"),
    ] + run_scale_tail()))

table(Table(
    "run_degeneration", scale="item", pk=("run_id", "judgment_id", "measure", "scope", "source_revision_id"),
    indexes=[("idx_run_degeneration_revision", ("source_revision_id", "record_file_id"), None)],
    doc="Degeneration and truncation as COUNTS with their unit; never changes a run's state, status or paper flag. `measurement_state` says whether the row was measured. "
        "`v_run_degeneration_current` keeps one row per (run_id, judgment_id, measure, scope): native rows first, then a row carrying a number "
        "(`measured`) before every placeholder state, then the declared source priority, source_id and revision — a placeholder never hides a "
        "sibling provenance's number.",
    checks=[("run_degeneration_measured_pairing",
             "((measurement_state = 'measured' AND numerator IS NOT NULL AND denominator IS NOT NULL AND denominator_unit IS NOT NULL AND rate IS NOT NULL)"
             " OR (measurement_state <> 'measured' AND numerator IS NULL AND denominator IS NULL AND denominator_unit IS NULL AND rate IS NULL))")],
    columns=[
        C("run_id", nn=True, fk=("run", "run_id"), fk_deferred=True),
        C("judgment_id", nn=True, default="''", fk=("judgment", "judgment_id"), sentinel="", fk_deferred=True),
        C("measure", nn=True, vocab="DEGENERATION_MEASURES"),
        C("scope", nn=True, vocab="DEGENERATION_SCOPES"),
        C("measurement_state", nn=True, vocab="MEASUREMENT_STATES",
          doc="measured | not_measured | not_applicable | refused; consumers branch on this, never on a NULL rate."),
        C("numerator", "INTEGER", why="NULL unless measurement_state = measured (table CHECK run_degeneration_measured_pairing)"),
        C("denominator", "INTEGER", why="NULL unless measurement_state = measured (table CHECK)"),
        C("denominator_unit", vocab="DENOMINATOR_UNITS", why="NULL unless measurement_state = measured (table CHECK)"),
        C("pooling_json", nn=True, default="'{}'"),
        C("aggregation", nn=True, vocab="AGGREGATIONS"),
        C("rate", "REAL", why="NULL unless measurement_state = measured (table CHECK); undefined when the denominator is 0 is recorded as not_measured"),
        C("definition_version", nn=True),
        C("threshold_params_json", nn=True, default="'{}'"),
        C("computed_by", nn=True, vocab="COMPUTED_BY"),
        C("computed_utc", why="unknown for legacy rows"),
        C("source_uri", why="natively emitted rows are their own source"),
        C("notes", why="free text"),
        revision_tag(),
    ] + item_scale_tail()))

table(Table(
    "artifact", pk=("artifact_id",),
    doc="A file on disk (sha256, else sha256 of the URI) with its locations; text lives here by pointer, never in a row.",
    columns=[
        C("artifact_id", nn=True),
        C("kind", nn=True, vocab="ARTIFACT_KINDS"),
        C("uri_primary", nn=True),
        C("locations_json", nn=True, default="'[]'"),
        C("bytes", "INTEGER", why="unknown until verified"),
        C("created_utc", why="unknown for legacy files"),
        C("verified_utc", why="not yet verified against disk"),
    ] + run_scale_tail()))

table(Table(
    "item_result", scale="item", pk=("run_id", "item_id", "metric_id", "trial_index", "sample_index", "turn_index", "judgment_id",
                                     "source_revision_id"),
    doc="Per item × metric scores and labels (F10); text by pointer through item_flags. Keyed by revision too: read through "
        "`v_item_result_current` (or the same predicate), never raw.",
    indexes=[
        ("idx_item_result_run_judgment", ("run_id", "judgment_id"), None),
        ("idx_item_result_item", ("item_id",), None),
        ("idx_item_result_withdrawn", ("withdrawn_by_revision_id",), "withdrawn_by_revision_id IS NOT NULL"),
        # the withdrawn rows keyed by ingest_seq (migration 0015): v_paper_inputs_head's item member is one seek through it instead of a
        # probe of every withdrawn row (509k live, 115 s a cycle); the reclaim scans keep reading idx_item_result_withdrawn, untouched
        ("idx_item_result_withdrawn_seq", ("ingest_seq",), "withdrawn_by_revision_id IS NOT NULL"),
        ("idx_item_result_revision", ("source_revision_id", "record_file_id"), None),
    ],
    columns=[
        C("run_id", nn=True, fk=("run", "run_id"), fk_deferred=True),
        C("item_id", nn=True),
        C("metric_id", nn=True),
        C("trial_index", "INTEGER", nn=True, default="0"),
        C("sample_index", "INTEGER", nn=True, default="0"),
        C("turn_index", "INTEGER", nn=True, default="-1", sentinel=-1, doc="-1 for single-turn items."),
        C("judgment_id", nn=True, default="''", fk=("judgment", "judgment_id"), sentinel="", fk_deferred=True,
          doc="'' for generation-time metrics (FK by trigger otherwise)."),
        C("value", "REAL", why="label-only metrics carry no number"),
        C("label", why="numeric-only metrics carry no label"),
        C("api_pool", vocab="API_POOLS", why="rule-scored rows have no pool; legacy judged rows may not name one"),
        C("created_utc", nn=True),
        C("judged_utc", why="generation-time rows"),
        revision_tag(),
        C("withdrawn_by_revision_id", why="set when a later accepted revision no longer carries the item; bookkeeping — visibility is derived from the head revision"),
    ] + item_scale_tail() + [
        # The producer-withdrawal marks (migration 0020) TRAIL ingest_seq on purpose: SQLite's ALTER TABLE ADD COLUMN appends a
        # column after the last existing one, so only a column declared here, after the tail, lets the live table be migrated by two
        # O(1) ALTERs instead of a copy of its 12 M rows (migrations/README.md "Adding a column"). Derived: written NULL by every item
        # fold, set by the item_withdrawal fold (reducer._apply_item_withdrawal) and cleared again by a re-filed row whose order key
        # is greater than the withdrawal's. No CHECK on the sha: ADD COLUMN with a CHECK makes SQLite verify every existing row
        # (pragma_quick_check over the whole table, 3.37+), a full scan the migration would otherwise never do; the value is the
        # harvester's copy of a record sha the validator already checked.
        C("withdrawn_by_record_sha256", derived=True,
          why="set when a producer's item_withdrawal record retracted this natively emitted row: the sha of that record; NULL for a row no withdrawal names, and again for a row re-filed with an order key above the withdrawal's"),
        C("withdrawn_reason", derived=True,
          why="the item_withdrawal record's free-text reason, beside its sha; NULL whenever withdrawn_by_record_sha256 is"),
    ]))

table(Table(
    "item_withdrawal", pk=("run_id", "item_id", "metric_id", "trial_index", "sample_index", "turn_index", "judgment_id"),
    doc="A producer's retraction of ONE natively emitted item_result row it filed under the wrong run (migration 0020). An "
        "`item_withdrawal` record names a run and a list of exact item keys; the fold writes one row here per key (the observation "
        "pattern) and marks the item_result row whose order key is below the record's (`withdrawn_by_record_sha256`, `withdrawn_reason`); "
        "`v_item_result_current` and every current-item predicate exclude a marked row. Rows of an adapter revision are never targeted "
        "(their membership is the revision's). The latest record naming a key (by order key) stands; a row re-filed with a greater "
        "order key than the withdrawal's stands as a new measurement.",
    columns=[
        C("run_id", nn=True, fk=("run", "run_id"), doc="The run the retracted rows were filed under (FK; the record is dependency-blocked until it exists)."),
        C("item_id", nn=True),
        C("metric_id", nn=True),
        C("trial_index", "INTEGER", nn=True, default="0"),
        C("sample_index", "INTEGER", nn=True, default="0"),
        C("turn_index", "INTEGER", nn=True, default="-1", sentinel=-1, doc="-1 for single-turn items (the item_result sentinel)."),
        C("judgment_id", nn=True, default="''", sentinel="", doc="'' for generation-time metrics (the item_result sentinel; no FK: the key names a row, not a judgment)."),
        C("reason", nn=True, doc="Free text: why the rows are retracted (required; not a vocabulary)."),
        C("evidence_uri", why="optional: a file or page that shows the misfiling"),
        C("belongs_to_run_id", why="optional: the run the rows should have been filed under; a name, not an FK (that run may never reach the registry)"),
    ] + run_scale_tail()))

table(Table(
    "item_flags", scale="item", pk=("run_id", "item_id", "trial_index", "sample_index", "turn_index", "judgment_id", "source_revision_id"),
    doc="Per-item facts that do not vary by metric: truncation, degeneration flags, seeds and the generation / rationale pointers (F10). "
        "Keyed by revision too: read through `v_item_flags_current`.",
    indexes=[("idx_item_flags_revision", ("source_revision_id", "record_file_id"), None)],
    columns=[
        C("run_id", nn=True, fk=("run", "run_id"), fk_deferred=True),
        C("item_id", nn=True),
        C("trial_index", "INTEGER", nn=True, default="0"),
        C("sample_index", "INTEGER", nn=True, default="0"),
        C("turn_index", "INTEGER", nn=True, default="-1", sentinel=-1),
        C("judgment_id", nn=True, default="''", fk=("judgment", "judgment_id"), sentinel="", fk_deferred=True),
        B("at_token_cap", why="unknown for legacy aggregates"),
        C("finish_reason", why="unknown for legacy aggregates"),
        C("response_tokens", "INTEGER", why="unknown for legacy aggregates"),
        C("response_chars", "INTEGER", why="unknown for legacy aggregates"),
        B("degenerate_classifier", why="not computed by every producer"),
        B("degenerate_contentless", why="not computed by every producer"),
        B("degenerate_rep_ratio_loop", why="not computed by every producer"),
        B("degenerate_zlib_collapse", why="not computed by every producer"),
        B("degenerate_union5", why="not computed by every producer"),
        B("loop_first512", why="not computed by every producer"),
        C("rep_ratio", "REAL", why="not computed by every producer"),
        C("zlib_ratio", "REAL", why="not computed by every producer"),
        C("alnum_count", "INTEGER", why="not computed by every producer"),
        B("tail_loop", why="not computed by every producer"),
        B("line_loop", why="not computed by every producer"),
        C("decode_seed", why="only per-item-seeded evals"),
        C("generation_artifact_id", fk=("artifact", "artifact_id"), fk_deferred=True, why="legacy aggregates have no per-item pointer"),
        C("generation_offset", "INTEGER", why="null when generation_artifact_id is null"),
        C("rationale_artifact_id", fk=("artifact", "artifact_id"), fk_deferred=True, why="rule-scored and unjudged items"),
        C("rationale_offset", "INTEGER", why="null when rationale_artifact_id is null"),
        revision_tag(),
    ] + item_scale_tail()))

table(Table(
    "roster_entry", pk=("roster_entry_id",),
    doc="One declared paper comparison input of a round; instruments are matched per comparison, never globally.",
    indexes=[("idx_roster_entry_key", ("round_id", "model_id", "eval_id"), None)],
    checks=[
        ("roster_entry_object_xor_selector", "((object_id IS NULL) <> (object_selector_json IS NULL))"),
        ("roster_entry_instrument_xor_selector", "((instrument_id IS NULL) <> (instrument_selector_json IS NULL))"),
        ("roster_entry_judge_not_both", "(NOT (judge_id IS NOT NULL AND judge_selector_json IS NOT NULL))"),
    ],
    columns=[
        C("roster_entry_id", nn=True),
        C("round_id", nn=True, fk=("paper_round", "round_id")),
        C("comparison_id", nn=True),
        C("model_id", nn=True, fk=("model", "model_id")),
        C("object_id", fk=("object", "object_id"), why="the object may be chosen by rule (object_selector_json; CHECK one of the two)"),
        C("object_selector_json", why="null when object_id names the object (CHECK one of the two)"),
        C("eval_id", nn=True, fk=("eval", "eval_id")),
        C("instrument_id", fk=("instrument", "instrument_id"), why="the roster may select an instrument FAMILY (instrument_selector_json; C2; CHECK one of the two)"),
        C("instrument_selector_json", why="null when instrument_id names the instrument (CHECK one of the two)"),
        C("judge_id", fk=("judge", "judge_id"), why="rule-scored comparisons have no judge; or the judge is chosen by rule (judge_selector_json)"),
        C("judge_selector_json", why="null when judge_id names the judge or the comparison is rule-scored (C2)"),
        C("computation_id", fk=("computation", "computation_id"), why="raw producer values name no computation"),
        C("strength_rule", nn=True, pattern="STRENGTH_RULE", doc="`operating_point` | `fixed:<x>` | `ladder:<record_id>` (every strength level the record carries)."),
        C("source_citation", nn=True),
        B("inferred", nn=True, default="0"),
        B("drawn", nn=True, default="1", doc="0 = required by the gap map but not drawn by the figure at the transcribed sha."),
        C("roster_version", why="an entry written by a roster load before migration 0016 carries none (it matches while its round's paper_round row carries none)",
          doc="The roster version whose load last wrote this entry (not an identity field). An entry a later version no longer expands keeps its older stamp and, once that version loads, matches nothing and is no gap (migration 0016)."),
    ] + run_scale_tail()))

table(Table(
    "disposition", pk=("disposition_id",),
    doc="Why a roster key has no judged run (a source that has not run emits a disposition, never silence).",
    columns=[
        C("disposition_id", nn=True),
        C("round_id", nn=True, fk=("paper_round", "round_id")),
        C("roster_entry_id", fk=("roster_entry", "roster_entry_id"), why="a disposition may cover a key with no roster entry yet"),
        C("comparison_id", nn=True),
        C("model_id", nn=True, fk=("model", "model_id")),
        C("object_id", fk=("object", "object_id"), why="a disposition may cover every object of the key"),
        C("eval_id", nn=True, fk=("eval", "eval_id")),
        C("state", nn=True, vocab="DISPOSITIONS"),
        C("reason", nn=True),
        C("source_ref", nn=True),
        C("ruled_utc", nn=True),
    ] + run_scale_tail()))

table(Table(
    "snapshot", pk=("snapshot_id",),
    doc="A published point-in-time snapshot (per-table Parquet + run-level SQLite), its ingest_seq and destinations.",
    columns=[
        C("snapshot_id", nn=True),
        C("head_ingest_seq", "INTEGER", nn=True, doc="The index head (max ingest_seq) inside the frozen read transaction at build time."),
        C("built_utc", nn=True),
        C("sqlite_sha256", nn=True, pattern="SHA256"),
        C("parquet_manifest_json", nn=True),
        C("pushed_to_json", nn=True, default="'[]'"),
    ] + run_scale_tail()))

table(Table(
    "parity_gate", pk=("gate_id",),
    doc="One run of the paper parity gate: the paper sha AND the sha256 of every input the pinned checkout read; the gate refuses when any input sha differs between its legacy and registry legs.",
    columns=[
        C("gate_id", nn=True),
        C("paper_sha", nn=True, pattern="SHA1_OR_UNVERIFIED"),
        C("paper_checkout_uri", why="a gate run from a bare sha records no checkout path"),
        C("input_shas_json", nn=True, doc="{input uri: sha256} of the r2 parquet / schema / sources and every record file the checkout read."),
        C("registry_ingest_seq", "INTEGER", nn=True),
        C("level", nn=True, vocab="PARITY_LEVELS"),
        C("result", nn=True, vocab="PARITY_RESULTS"),
        C("report_uri", why="a refused gate may write no report"),
        C("run_utc", nn=True),
    ] + run_scale_tail()))

# ---- sources, revisions, mirrors (harvester-originated facts) -------------------------------

table(Table(
    "source_revision", pk=("revision_id",),
    doc="One accepted (or refused) read of a legacy source by an adapter; `v_paper_long` reads only the latest accepted revision per source (S4).",
    indexes=[("idx_source_revision_source", ("source_id", "observed_utc"), None)],
    columns=[
        C("revision_id", nn=True, doc="sha16 of the sorted input shas."),
        C("source_id", nn=True),
        C("adapter", nn=True),
        C("observed_utc", nn=True),
        C("inputs_json", nn=True, doc="{uri: sha256} of every input read."),
        B("accepted", nn=True),
        C("refusal", why="accepted revisions have no refusal"),
        C("superseded_by_revision_id", why="the latest revision of a source is not superseded"),
        C("priority", "INTEGER", nn=True, default="0",
          doc="The adapter's declared source priority: among the visible provenances of one measurement key a native row comes first, "
              "then the higher priority, then the lexically smaller source_id, then revision_id."),
        C("reclaimed", "INTEGER", nn=True, default="0", vocab="BOOLEAN", derived=True,
          doc="1 once the reclaim pass has marked every row of this superseded revision the successor no longer carries; reset when the chain moves."),
    ] + run_scale_tail()))

table(Table(
    "observation", pk=("source_id", "revision_id", "observation_key"),
    doc="Membership of a RUN-SCALE key in a source revision (item membership is two columns on item_result).",
    columns=[
        C("source_id", nn=True),
        C("revision_id", nn=True, fk=("source_revision", "revision_id")),
        C("observation_key", nn=True, doc="Canonical JSON of the natural key."),
        C("table_name", nn=True),
        B("current", nn=True, default="1"),
        C("withdrawn_by_revision_id", why="set when a later accepted revision no longer carries the key"),
    ] + run_scale_tail()))

table(Table(
    "withdrawal", pk=("source_id", "observation_key", "to_revision_id"),
    doc="A key present in the previous accepted revision and absent from the new one (a real shrink, visible to the paper build's shrink refusal).",
    columns=[
        C("source_id", nn=True),
        C("observation_key", nn=True),
        C("from_revision_id", nn=True),
        C("to_revision_id", nn=True, fk=("source_revision", "revision_id")),
        C("table_name", nn=True),
        C("reason", nn=True, vocab="WITHDRAWAL_REASONS"),
        C("item_keys_json", nn=True, default="'[]'"),
    ] + run_scale_tail()))

table(Table(
    "source_watermark", pk=("source_id", "input_uri"),
    doc="Per-input watermark of the incremental adapter pass (F5).",
    columns=[
        C("source_id", nn=True),
        C("input_uri", nn=True),
        C("sha256", nn=True, pattern="SHA256_OR_UNVERIFIED"),
        C("size", "INTEGER", nn=True),
        C("mtime_utc", nn=True),
        C("observed_utc", nn=True),
        C("adapter", nn=True),
        C("records_written", "INTEGER", nn=True, default="0"),
        C("last_error", why="the last read succeeded"),
    ] + run_scale_tail()))

table(Table(
    "source_state", pk=("source_id", "input_uri", "json_path"),
    doc="What a source file names NOW under a JSON path (the faithful `declaration_root_state` port, F8); ruling expiry is evaluated against it.",
    columns=[
        C("source_id", nn=True),
        C("input_uri", nn=True),
        C("json_path", nn=True),
        B("found", nn=True),
        B("path_resolves", nn=True),
        C("current_members_json", nn=True, default="'[]'"),
        C("observed_utc", nn=True),
        C("input_sha256", pattern="SHA256", why="null when found = 0"),
    ] + run_scale_tail()))

table(Table(
    "source_status", scale="item", pk=("lane_source_legacy", "adapter", "source_id", "source_revision_id"),
    indexes=[("idx_source_status_revision", ("source_revision_id", "record_file_id"), None)],
    doc="What ONE adapter read of ONE source found, filed under every paper source string (`lane_source_legacy`) the adapter covers — the "
        "registry-side half of the paper's status contract. Written by the adapter runner after each accepted read, once per "
        "read (the revision tag keys the row), so the history of a source's reads is the table and the head read is "
        "`v_source_status_current`. `extra_json` carries the adapter's FACTS (the producer records it read and reduced: ruled "
        "operating points, dose tables, the record's roster, engines); the paper's guards DECIDE on them in the reader, the "
        "registry only carries the evidence. `v_source_status` reduces the head reads to one row per (`lane_source_legacy`, adapter) "
        "and folds in the LATEST revision of each source, accepted or refused, so a refused re-read is visible there (`ok = 0`, "
        "`extra_json.refuse`) instead of hidden behind the accepted head it leaves current.",
    columns=[
        C("lane_source_legacy", nn=True, doc="The paper's source string this row reports on (`covers_sources`); the join key to `v_paper_long.lane_source_legacy`."),
        C("adapter", nn=True, doc="The adapter that read the source (`source_revision.adapter`)."),
        C("source_id", nn=True, doc="`<adapter>:<comparison group>`, the source this read was of (an adapter with several groups files one row per group)."),
        C("owner", nn=True, doc="The producing session the adapter names for the source (`registry` when it names none)."),
        C("where_json", nn=True, default="'[]'", doc="JSON list of the input uris the read covered (`source_revision.inputs_json` keys)."),
        B("ok", nn=True, doc="1 when the read was accepted with nothing refused; 0 when the adapter recorded a refusal it could carry on past."),
        C("detail", nn=True, default="''", doc="One line for the gap map: what was read, how many records / measured keys / dispositions, what was refused."),
        C("cells_measured", "INTEGER", nn=True, default="0", pattern="NONNEG",
          doc="Distinct (model, object, eval) keys the read measured (a `run_metric` carrying a value)."),
        C("observed_utc", nn=True, doc="When the adapter read the source (the pass's `now`)."),
        revision_tag(),
    ] + item_scale_tail()))

table(Table(
    "mirror_ack", pk=("destination",),
    doc="Per-destination acknowledgement of the record-log fan-out (F2).",
    columns=[
        C("destination", nn=True, vocab="MIRROR_DESTINATIONS"),
        C("acked_through_seq", "INTEGER", nn=True),
        C("manifest_name", nn=True),
        C("file_count", "INTEGER", nn=True),
        C("bytes", "INTEGER", nn=True),
        C("checksums_sha256", nn=True, pattern="SHA256"),
        C("acked_utc", nn=True),
        C("last_error", why="the last sync succeeded"),
    ] + run_scale_tail()))

table(Table(
    "file_arrival", pk=("file_uri",),
    doc="The harvester's own fact of a record file's FIRST live fold: the arrival_utc that orders the file in the ledger (record_ingest), written into the log as a record so a rebuild reproduces ingest_seq exactly. Emitted once per file whose arrival differs from its first_written_utc (the replay's fallback); never for the harvester's own files under records/harvester/, whose stamp is their arrival.",
    columns=[
        C("file_uri", nn=True, doc="Root-relative LOGICAL path of the file that arrived (record_ingest.file_uri)."),
        C("file_sha256", nn=True, pattern="SHA256", doc="The logical sha the file arrived with."),
        C("arrival_utc", nn=True, doc="The live registry's clock at the file's first fold — record_ingest.arrival_utc, verbatim."),
    ] + run_scale_tail()))

# ---- ledgers (not record-fed) ---------------------------------------------------------------

table(Table(
    "record_ingest", scale="ledger", pk=("file_uri",),
    doc="Replayable FILE-scale ledger (spec: every file's sha and ingest_seq): ingest_seq is the dense rank of the file in ARRIVAL order — (arrival_utc, first_written_utc, logical sha256, uri) over the whole log — so a file the live registry folds is always head + 1 and every row above a head a reader has seen is new to it; every row of the index carries its file's rank. A rebuild reproduces the order from the file_arrival facts in the log (a file with none takes arrival_utc = first_written_utc). Per-record dedupe is the row_history event key / the item order key, not this table.",
    uniques=[("record_file_id",), ("ingest_seq",)],
    indexes=[("idx_record_ingest_sha", ("file_sha256",), None),
             ("idx_record_ingest_order", ("arrival_utc", "first_written_utc", "file_sha256", "file_uri"), None)],
    columns=[
        C("file_uri", nn=True, doc="Root-relative LOGICAL path (no .gz)."),
        C("record_file_id", "INTEGER", nn=True, doc="First 63 bits of sha256(file_uri): the arrival-independent integer id item rows point at."),
        C("file_sha256", nn=True, pattern="SHA256", doc="sha256 of the DECOMPRESSED bytes (two files may share it; each is its own ledger row)."),
        C("first_written_utc", nn=True, doc="The smallest written_utc among the file's accepted records — the tie-break after arrival_utc, and the arrival of a file the live registry never folded."),
        C("arrival_utc", nn=True, doc="THE ordering stamp: the live registry's clock at the file's FIRST fold, strictly after the head's (max(cycle clock, head + 1 ms)); kept for the file's life (a re-read or re-keyed file keeps it). A harvester-originated file (records/harvester/) and a file a replay finds no file_arrival fact for take their first_written_utc."),
        C("n_records", "INTEGER", nn=True),
        C("ingest_seq", "INTEGER", nn=True),
    ]))

table(Table(
    "record_delivery", scale="ledger", pk=("file_uri",),
    doc="Arrival audit per record file (arrival-order dependent; excluded from rebuild equality).",
    columns=[
        C("file_uri", nn=True),
        C("file_sha256", nn=True, pattern="SHA256"),
        C("arrived_utc", nn=True),
        C("ingested_utc", why="not yet ingested"),
        C("status", nn=True, vocab="DELIVERY_STATUSES"),
        C("refusal_reason", why="only refused / conflict files"),
        C("producer_script", why="unreadable envelope"),
        C("producer_session", why="producers outside a session"),
        C("schema_version", "INTEGER", why="unreadable envelope"),
        C("record_type_counts_json", nn=True, default="'{}'"),
        C("n_records", "INTEGER", nn=True, default="0"),
        C("n_ok", "INTEGER", nn=True, default="0"),
        C("n_refused", "INTEGER", nn=True, default="0"),
        C("n_blocked", "INTEGER", nn=True, default="0"),
        C("n_duplicate", "INTEGER", nn=True, default="0"),
    ]))

table(Table(
    "record_refusal", scale="ledger", pk=("record_sha256",),
    doc="Malformed records, refused with the field and reason; never silently skipped (listed on the status page). One row per record: the "
        "CURRENT judgment (a re-attempt under a changed judging stamp replaces it); every judgment ever made is a `record_refusal_fact`.",
    columns=[
        C("record_sha256", nn=True, pattern="SHA256"),
        C("record_uri", nn=True),
        C("record_type", why="an envelope without a readable type"),
        C("field", why="an envelope-level refusal names no field"),
        C("reason", nn=True),
        C("refused_utc", nn=True),
        C("judged_with", pattern="STAMP16_OR_UNVERIFIED",
          why="rows refused before the judging stamp was recorded (the fact derivation reads them as 'unverified'); declared last so the live "
              "ledger takes it by ALTER TABLE ADD COLUMN"),
    ]))

table(Table(
    "record_refusal_fact", scale="ledger", pk=("record_sha256", "judged_with"),
    doc="The harvester's DURABLE fact of one refusal judgment: one row per (refused record, judging stamp), written into the log as a "
        "`record_refusal` record under records/harvester/<day>/record_refusal-<sha12>.jsonl and folded here by its own apply path "
        "(INSERT OR IGNORE, no row_history), so every mirror and every rebuild reproduce every refusal ever made under every stamp, while "
        "`record_refusal` keeps only the current judgment per record. Derived from `record_refusal` at the START of every cycle (before any "
        "re-attempt can replace a row) and again at its END; a row refused before the stamp was kept is a fact under 'unverified' with "
        "extra_json.backfilled = true. Pure function of the log: a rebuild carries the same rows.",
    indexes=[("idx_record_refusal_fact_uri", ("record_uri",), None),
             ("idx_record_refusal_fact_cycle", ("cycle_utc",), None)],
    columns=[
        C("record_sha256", nn=True, pattern="SHA256", doc="The refused record (records.record_sha): the key `record_refusal` holds it under."),
        C("record_uri", nn=True, doc="The logical uri the record was refused under (`<file>#<line>` for a record inside a file)."),
        C("record_type", why="an envelope without a readable type"),
        C("field", why="an envelope-level refusal names no field"),
        C("reason", nn=True),
        C("judged_with", nn=True, pattern="STAMP16_OR_UNVERIFIED",
          doc="The judging code's stamp (harvest.validator_stamp) the refusal was made under, or 'unverified' for a refusal recorded before the stamp was kept."),
        C("refused_utc", nn=True, doc="record_refusal.refused_utc of the judgment: when the refusal was made (never the cycle that derived the fact)."),
        C("producer_json", why="the refused file could not be re-read for the record's producer block (moved, unreadable, or a line that is not JSON)",
          doc="The refused record's producer block, re-read from its file by sha."),
        C("cycle_utc", nn=True, doc="The clock of the harvester cycle that derived the fact (the record id is minted over the payload WITHOUT it)."),
        C("extra_json", nn=True, default="'{}'", doc="`backfilled: true` for a row refused before the stamp was kept."),
    ]))

# ---- analyses: ledger-scale facts filed by `registry analysis file`, never by the harvester -------------------

table(Table(
    "analysis", scale="ledger", pk=("analysis_id",),
    doc="An ANALYSIS filed over ingested runs and objects: a prefill-only or offline computation -- a routing capture, an "
        "ablation, an efficiency study -- with results but no run-shaped home (no eval, no generation). One row per filing, written into the log "
        "as an `analysis` record under records/analyses/<day>/ by `registry analysis file --spec` and folded by the ledger-fact apply path "
        "(INSERT OR IGNORE, no row_history): the row IS the fact, keyed by `analysis_id` = uuid5 over the analysis's identity (its name, its "
        "code sha and its inputs; `sisteer.registry.analysis.analysis_id`), so a re-filing of the same analysis is a duplicate and a changed "
        "input set or code sha is a new analysis. Its inputs are `analysis_input` rows, its files `analysis_artifact` rows.",
    indexes=[("idx_analysis_filed", ("filed_utc",), None)],
    checks=[("analysis_report_pairing", "((report_uri IS NULL) = (report_sha256 IS NULL))")],
    columns=[
        C("analysis_id", nn=True, doc="uuid5 over the analysis's identity: canonical JSON of {name, code_sha, inputs} (sisteer.registry.analysis)."),
        C("name", nn=True, doc="The analysis's name, dotted by study (`moe_routing.capture`); free text, unique only with the code sha and inputs."),
        C("code_sha", nn=True, pattern="SHA1_OR_UNVERIFIED", doc="The commit the analysis ran from (forty hex), or 'unverified'."),
        C("code_ref", why="a code sha alone names the code", doc="The branch or tag the sha was taken on, and the shas of its stages, as the filer wrote them."),
        C("cluster_job_ids_json", nn=True, default="'[]'", doc="The scheduler job ids that produced it (strings), in the filer's order; [] for a local computation."),
        C("substrate", nn=True, vocab="SUBSTRATES", doc="Where it ran."),
        C("report_uri", why="an analysis whose outputs are pages alone files no report",
          doc="The report file (a local path, or a cluster path the filer could not read); registered as an `artifact` too (role `report`)."),
        C("report_sha256", pattern="SHA256", why="paired with report_uri (the table CHECK): null exactly when no report is filed",
          doc="sha256 of the report's bytes: computed by the filer for a local file, declared in the spec for a remote one (never a guess)."),
        C("filed_utc", nn=True, doc="When the filing was written (the filer's clock)."),
        C("filed_by", nn=True, doc="Whose work the analysis is (by default); the relaying session is `extra_json.provenance.relayed_by`."),
        C("notes", why="free text"),
        C("extra_json", nn=True, default="'{}'",
          doc="`provenance` {kind: user_direct|relayed, relayed_by, message_utc}, the spec's own `extra`, and any unknown record field."),
        # the ORDERING KEY of a ledger fact with one (records.LEDGER_ORDERED_TABLES): among the
        # filings of one analysis_id the row kept is the one with the smallest (written_utc, record_sha256) -- the log's order, so a live
        # apply and a rebuild keep the same header whatever the arrival order; the live apply REPLACES a standing row when the incoming record
        # sorts earlier and refuses a later one whose payload differs (`conflicting re-filing`); identical payloads dedupe. Derived: never in
        # a record's payload
        C("record_written_utc", nn=True, derived=True,
          doc="The written_utc of the filing whose header this row is: with record_sha256 the row's ordering key (the smallest among the id's filings is kept)."),
        C("record_sha256", nn=True, pattern="SHA256", derived=True, doc="The record sha of the filing kept (see record_written_utc)."),
        C("record_uri", nn=True, derived=True, doc="The logical uri (`<file>#<line>`) of the filing kept, so a displaced filing can be refused by name."),
    ]))

table(Table(
    "analysis_input", scale="ledger", pk=("analysis_id", "input_kind", "input_id"),
    doc="One input an analysis read: an ingested run or an object. The input is named twice -- `input_id` in the key and the one FK "
        "column of its kind (`run_id` for a run, `object_id` for an object; the table CHECK pairs them) -- so the harvester's dependency check "
        "reads the parent through the declared FK (validate.parent_references) and the key stays one column wide. A ledger-fact kind cannot "
        "wait in `blocked_records`, so a row naming a run or object the registry does not hold is REFUSED by name, never queued "
        "(harvest.Ingestor.apply_validated); `registry analysis file` read-checks every input before it writes.",
    indexes=[("idx_analysis_input_run", ("run_id",), "run_id IS NOT NULL"),
             ("idx_analysis_input_object", ("object_id",), "object_id IS NOT NULL")],
    checks=[("analysis_input_kind_pairing",
             "((input_kind = 'run' AND run_id = input_id AND object_id IS NULL) OR (input_kind = 'object' AND object_id = input_id AND run_id IS NULL))")],
    columns=[
        C("analysis_id", nn=True, fk=("analysis", "analysis_id")),
        # the FILING that wrote this row (records.LEDGER_FILING_CHILDREN; the fifth closing pass, 2026-10-09): the record sha of its
        # `analysis` header. A child folds only while that header is the analysis's kept filing or a duplicate of it; a child of a refused
        # re-filing is refused with it, so a rejected filing's inputs and files are never attached to the kept analysis. An EVENT column of
        # the ordering rule (records.LEDGER_ORDERED_EVENT_COLUMNS): two filings of one substance write identical children under two shas
        C("filing_sha256", nn=True, pattern="SHA256",
          doc="The record sha of the `analysis` header (the filing) that wrote this row; the row folds only while that filing is the kept one or a duplicate of it."),
        C("input_kind", nn=True, vocab="ANALYSIS_INPUT_KINDS"),
        C("input_id", nn=True, doc="The run_id or object_id, as the kind says."),
        C("run_id", fk=("run", "run_id"), why="an object input names no run"),
        C("object_id", fk=("object", "object_id"), why="a run input names no object"),
        C("extra_json", nn=True, default="'{}'", doc="What the analysis read of the input (`role`: the steering condition, the transcript set), as the filer wrote it."),
        # the ORDERING KEY (records.LEDGER_ORDERED_TABLES; the second review round, 2026-10-09): `extra_json` is substance of the input,
        # so the row kept for a key is the record with the smallest (written_utc, record_sha256), as the header's is. Derived: never in a payload
        C("record_written_utc", nn=True, derived=True,
          doc="The written_utc of the record this row is: with record_sha256 the row's ordering key (the smallest among the key's records is kept)."),
        C("record_sha256", nn=True, pattern="SHA256", derived=True, doc="The record sha of the record kept (see record_written_utc)."),
        C("record_uri", nn=True, derived=True, doc="The logical uri (`<file>#<line>`) of the record kept, so a displaced record can be refused by name."),
    ]))

table(Table(
    "analysis_artifact", scale="ledger", pk=("analysis_id", "artifact_id"),
    doc="One registered file of an analysis: the link from the analysis to an `artifact` row -- its report, a file of a published page "
        "directory, a figure -- with the file's ROLE to the analysis. The artifact rows themselves are written by the same filing (through "
        "publish.register_files, kind by suffix; a remote report under its declared sha) ahead of this row, so the FK parent folds first.",
    indexes=[("idx_analysis_artifact_artifact", ("artifact_id",), None)],
    columns=[
        C("analysis_id", nn=True, fk=("analysis", "analysis_id")),
        # the FILING that wrote this link (records.LEDGER_FILING_CHILDREN; the fifth closing pass, 2026-10-09): the second filing of an
        # analysis with another report had its header refused as a conflicting re-filing while its link to the new report folded on its own,
        # the kept analysis_id being a row -- so readers attributed a rejected report to the accepted analysis. An event column of the
        # ordering rule, as on analysis_input
        C("filing_sha256", nn=True, pattern="SHA256",
          doc="The record sha of the `analysis` header (the filing) that wrote this link; the link folds only while that filing is the kept one or a duplicate of it."),
        C("artifact_id", nn=True, fk=("artifact", "artifact_id")),
        C("role", nn=True, vocab="ANALYSIS_ARTIFACT_ROLES"),
        C("extra_json", nn=True, default="'{}'", doc="`page_url` / `page_dir` for a file of a published page directory; `declared_sha256` for a remote report."),
        # the ORDERING KEY (records.LEDGER_ORDERED_TABLES; the second review round, 2026-10-09): a link filed twice with another role or page
        # url kept whichever ARRIVED first while a rebuild kept the other; now the record with the smallest (written_utc, record_sha256) is
        # kept, live and on rebuild alike, and the later one with another payload is refused by name. Derived: never in a payload
        C("record_written_utc", nn=True, derived=True,
          doc="The written_utc of the record this row is: with record_sha256 the row's ordering key (the smallest among the key's records is kept)."),
        C("record_sha256", nn=True, pattern="SHA256", derived=True, doc="The record sha of the record kept (see record_written_utc)."),
        C("record_uri", nn=True, derived=True, doc="The logical uri (`<file>#<line>`) of the record kept, so a displaced record can be refused by name."),
    ]))

table(Table(
    "ordered_fact_record", scale="ledger", pk=("table_name", "natural_key_json", "record_sha256"),
    doc="Every record of an ORDERED ledger-fact table (records.LEDGER_ORDERED_TABLES: the three analysis tables) the fold has applied, by the "
        "record's table and natural key -- the kept record, its duplicates, the records it displaced and the ones that conflict with it alike; "
        "one row per record sha. Written by `reducer.Reducer.apply_ledger_fact` on "
        "every outcome, so a rebuild carries the same rows; read by the harvester when the kept record of a key MOVES (an earlier-sorting record "
        "arrived late) to judge every other record of the key again against the new kept one (`harvest.Ingestor._rejudge_refilings`): a record "
        "identical to it is accepted (its conflict refusal withdrawn), one that differs is refused by name -- the one accepted as a duplicate of "
        "the displaced record included, which leaves no refusal row to find, so without this index live and rebuild would refuse different sets. "
        "Carriers only, never payloads: a record is judged again from its immutable file. Not `row_history` (a ledger fact has no fold events).",
    columns=[
        C("table_name", nn=True, doc="The ordered table (a key of records.LEDGER_ORDERED_TABLES); no CHECK list, so a table ordered later adds rows and the shape stands."),
        C("natural_key_json", nn=True, doc="Canonical JSON of the record's natural-key columns, as `row_history` keys a run-scale row."),
        C("record_sha256", nn=True, pattern="SHA256", doc="The record (records.record_sha)."),
        C("record_type", nn=True, doc="The record's kind (the refusal row of a record judged again names it)."),
        C("record_uri", nn=True, doc="The record's logical uri (`<file>#<line>`); the smallest when the same record arrives under two names."),
        C("written_utc", nn=True, doc="The record's written_utc: with record_sha256 its place in the key's order."),
    ]))

table(Table(
    "ordered_fact_carrier", scale="ledger", pk=("table_name", "record_sha256", "record_uri"),
    doc="Every ARRIVAL of a record of an ordered ledger-fact table (records.LEDGER_ORDERED_TABLES), whatever its verdict: the logical uri "
        "that carried it and the side of that file's delivery audit (`record_delivery`) the record is counted on -- `accepted` (the file's "
        "`n_ok` or `n_duplicate`) or `refused` (its `n_refused`). One row per (record, uri): the same record copied to a second uri has two "
        "rows where `ordered_fact_record` keeps one carrier. Written by `harvest.Ingestor.apply_validated` as the fold counts the record for "
        "its file; read and moved by `harvest.Ingestor._move_record_delivery` when the record is judged again after its file's fold (a "
        "filing displaced, a re-filing or a child judged again, a dependency refusal accepted on retry), so EVERY file that carried the "
        "record follows the verdict -- a mover that followed `ordered_fact_record`'s one canonical carrier alone would leave a "
        "filing's copy under a second uri at `ok` after the filing was displaced, and the status page would list it as ingested. A pure "
        "function of the records once the moves are made (a record's standing verdict, per file that carried it), so a rebuild writes the "
        "same rows and the table is compared by `harvest.verify_equal`; the `ok` / `duplicate` split it does not record is the fold's "
        "arrival order, the reason `record_delivery` itself is not compared.",
    columns=[
        C("table_name", nn=True, doc="The ordered table (a key of records.LEDGER_ORDERED_TABLES); no CHECK list, as `ordered_fact_record`."),
        C("record_sha256", nn=True, pattern="SHA256", doc="The record (records.record_sha)."),
        C("record_uri", nn=True, doc="The logical uri (`<file>#<line>`) this arrival of the record came under; the file before `#` is the delivery row's."),
        C("delivery", nn=True, vocab="DELIVERY_SIDES",
          doc="The side of the file's delivery audit the record is counted on: `accepted` (`n_ok` or `n_duplicate`) or `refused` (`n_refused`); "
              "moved with the file's counts when the record is judged again."),
    ]))

table(Table(
    "fold_observation", scale="ledger", pk=("observation_kind", "run_id", "record_sha256"),
    doc="The harvester's OBSERVATIONS about a record at the fold, none of them a refusal (2026-10-04): a detector over the fold's own output, "
        "never a fold rule -- the record folded exactly as it always does (replace-at-key is universal). One row per (kind, run, restating "
        "record). Derived by `reducer.Reducer.apply_run_scale` from the run key's ordered events and fold snapshots after every refold of the key "
        "(the key's rows replaced whole, so a late-arriving earlier-stamped record re-derives them as it re-derives `row_history.folded_state_json`), "
        "and once by the migration that created the table for the runs already folded. A pure function of the log: a rebuild carries the same "
        "rows, no column excluded; the cycle that folded the restating record is its file's arrival (`record_ingest`), joined at read time by "
        "`registry observations`. Not the `observation` table (a source revision's membership of a run-scale key).",
    indexes=[("idx_fold_observation_written", ("written_utc",), None)],
    columns=[
        C("observation_kind", nn=True, vocab="FOLD_OBSERVATION_KINDS", doc="What was observed (vocab.FOLD_OBSERVATION_KINDS)."),
        C("run_id", nn=True, fk=("run", "run_id"), doc="The run whose folded state the record restated."),
        C("record_sha256", nn=True, pattern="SHA256", doc="The RESTATING record (records.record_sha): the `run` or `run_patch` event that carried the smaller set."),
        C("record_type", nn=True, doc="The restating record's kind: `run` (a re-registration) or `run_patch`."),
        C("record_uri", nn=True, doc="The record file (logical path) that carried the restating record -- the event's carrier in `row_history`, following the canonical carrier when the same record arrives under two names."),
        C("written_utc", nn=True, doc="The restating record's written_utc: its position in the fold's order (never the cycle that folded it)."),
        C("folded_record_sha256", nn=True, pattern="SHA256", doc="The record whose statement the folded set came from: the latest earlier event of the key that stated `extra.serving_jobs`."),
        C("folded_json", nn=True, doc="The set the run's extra_json held before the restatement, as a sorted JSON array of its members."),
        C("restated_json", nn=True, doc="The set the record stated, as a sorted JSON array of its distinct members (order and duplicates in the record are not facts)."),
        C("dropped_json", nn=True, doc="The members of the folded set the restatement left out, sorted."),
        C("n_dropped", "INTEGER", nn=True, pattern="NONNEG", doc="len(dropped_json): the rank for `registry observations summary`."),
    ]))

table(Table(
    "record_run_index", scale="ledger", pk=("run_id", "record_uri"),
    doc="File-scale index: one row per record file per run it mentions (the `registry history` fold for item-scale keys reads only these files).",
    columns=[
        C("run_id", nn=True),
        C("record_uri", nn=True),
    ]))

table(Table(
    "blocked_records", scale="ledger", pk=("record_sha256",),
    doc="Well-formed records whose parent row is not yet ingested (F4); re-evaluated every cycle, never dropped. `stale` after 7 days, or at once "
        "when the record belongs to a pending revision whose source has since accepted a later one (`extra_json.superseded_by` names it"
        ") — no longer pending work.",
    columns=[
        C("record_sha256", nn=True, pattern="SHA256"),
        C("record_uri", nn=True),
        C("record_type", nn=True, vocab="LEDGERED_RECORD_TYPES"),
        C("missing_parent_table", nn=True),
        C("missing_parent_id", nn=True),
        C("first_seen_utc", nn=True),
        C("last_checked_utc", nn=True),
        C("attempts", "INTEGER", nn=True, default="1"),
        B("stale", nn=True, default="0"),
        C("extra_json", nn=True, default="'{}'"),
    ]))


def run_scale_tables() -> Tuple[str, ...]:
    return tuple(t.name for t in TABLES if t.scale == "run")


def item_scale_tables() -> Tuple[str, ...]:
    return tuple(t.name for t in TABLES if t.scale == "item")


def ledger_tables() -> Tuple[str, ...]:
    return tuple(t.name for t in TABLES if t.scale == "ledger")


table(Table(
    "row_history", scale="ledger", pk=("table_name", "natural_key_json", "written_utc", "record_sha256"),
    doc="Every event of a RUN-SCALE natural key with the folded state after it (derived; identical live and on replay). Item-scale keys keep their history in the record log.",
    columns=[
        C("table_name", nn=True, vocab="RUN_SCALE_TABLES"),
        C("natural_key_json", nn=True),
        C("written_utc", nn=True),
        C("record_sha256", nn=True, pattern="SHA256"),
        C("record_type", nn=True, vocab="LEDGERED_RECORD_TYPES"),
        C("record_uri", nn=True),
        C("payload_json", nn=True, doc="The effect this event has on the key's row."),
        C("folded_state_json", nn=True, doc="The row after folding every event up to and including this one."),
    ]))


# --------------------------------------------------------------------------------------------
# Staging ledgers: a revision directory's item rows are written straight into the item tables, tagged with
# the pending revision (part of their key, read by no view until the marker flips); these three ledgers carry what the
# bounded staging needs to resume and to activate — the run-scale constituents as records, the per-file progress, and
# the plain-file cursor. Transient (empty once every revision is activated or discarded), excluded from rebuild equality.
# --------------------------------------------------------------------------------------------

table(Table(
    "revision_staging_record", scale="staging", pk=("revision_id", "file_uri", "line"),
    doc="A revision's RUN-SCALE constituents (few), staged as records; validated and folded by the activation transaction.",
    columns=[
        C("revision_id", nn=True),
        C("file_uri", nn=True),
        C("line", "INTEGER", nn=True),
        C("record_sha256", nn=True, pattern="SHA256"),
        C("written_utc", nn=True),
        C("record_json", nn=True, doc="The record, canonical JSON."),
    ]))

table(Table(
    "revision_staging_file", scale="staging", pk=("revision_id", "file_uri"),
    doc="One row per COMPLETELY staged constituent file of a revision: its verified sha, counts and envelope, so a held revision activates later without re-reading it.",
    columns=[
        C("revision_id", nn=True),
        C("file_uri", nn=True),
        C("file_sha256", nn=True, pattern="SHA256"),
        C("n_records", "INTEGER", nn=True),
        C("first_written_utc", why="a file with no well-formed record has no first key"),
        C("counts_json", nn=True, default="'{}'", doc="{ok, refused, blocked, duplicate} of the file's item rows at staging."),
        C("type_counts_json", nn=True, default="'{}'"),
        C("producer_script", why="unreadable envelope"),
        C("producer_session", why="producers outside a session"),
        C("schema_version", "INTEGER", why="unreadable envelope"),
        C("stat_json", nn=True, default="'{}'", doc="(size, mtime_ns, inode) taken before the file was read (the F12 sidecar entry)."),
        C("staged_utc", nn=True),
    ]))

table(Table(
    "record_cursor", scale="staging", pk=("file_uri",),
    doc="Resumable progress of a plain file written across several transactions: the records already committed, so a restart resumes after them and applies every record once; deleted when the file's delivery row is written.",
    columns=[
        C("file_uri", nn=True),
        C("file_sha256", nn=True, pattern="SHA256", doc="The discovery sha the progress belongs to; a file whose sha moved starts over."),
        C("records_done", "INTEGER", nn=True),
        C("counts_json", nn=True, default="'{}'", doc="Accumulated {counts, type_counts, n_refused} of the committed windows."),
        C("updated_utc", nn=True),
    ]))


def staging_tables() -> Tuple[str, ...]:
    return tuple(t.name for t in TABLES if t.scale == "staging")


TABLE_BY_NAME: Dict[str, Table] = {t.name: t for t in TABLES}
ALL_TABLES: Tuple[str, ...] = tuple(t.name for t in TABLES)
RUN_SCALE_TABLES: Tuple[str, ...] = run_scale_tables()
ITEM_SCALE_TABLES: Tuple[str, ...] = item_scale_tables()
LEDGER_TABLES: Tuple[str, ...] = ledger_tables()
STAGING_TABLES: Tuple[str, ...] = staging_tables()
assert set(ITEM_SCALE_TABLES) == set(V.ITEM_SCALE_TABLES), (ITEM_SCALE_TABLES, V.ITEM_SCALE_TABLES)

# The RUN_SCALE_TABLES vocabulary is derived from the table definitions (not hand-listed in vocab.py)
V.VOCABULARIES["RUN_SCALE_TABLES"] = RUN_SCALE_TABLES

# --------------------------------------------------------------------------------------------
# Triggers
# --------------------------------------------------------------------------------------------


def sql_literal(value: object) -> str:
    """A Python sentinel as an SQL literal ('' / -1 / -1.0)."""
    return "'%s'" % value if isinstance(value, str) else str(value)


VISIBLE_ROW_SQL = ("({alias}source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = {alias}source_revision_id "
                   "AND sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL))")


def item_visible(alias: str = "") -> str:
    """The item tables' visibility predicate: a row is read iff it is natively emitted ('' tag) or its revision is the
    source's HEAD accepted revision. Rows of pending and superseded revisions coexist in the table and are read by nothing."""
    return VISIBLE_ROW_SQL.format(alias=f"{alias}." if alias else "")


#: The lifecycle states of a COMPLETE run (migration 0006): a run is complete once `judged`; the harvester then stamps it
#: `ingested` (S6), so at rest a complete row reads `ingested` and `judged` is the transient state before the stamp.
COMPLETE_LIFECYCLE_STATES: Tuple[str, ...] = ("judged", "ingested")


def is_complete_sql(alias: str = "") -> str:
    """``is_complete``: 1 when the run's ``lifecycle_state`` is one of :data:`COMPLETE_LIFECYCLE_STATES`, else 0 — rendered beside
    ``lifecycle_state`` in every consumer-facing ``v_*`` object that exposes it (`v_run_wide`, `v_producer_key`, `v_paper_current`,
    `v_paper_long`). Two consumers filtered ``lifecycle_state = 'judged'`` for finished runs and found nothing, because the
    harvester had stamped every one of them ``ingested`` (2026-09-21); the flag is the one predicate to filter on."""
    states = ", ".join(sql_literal(s) for s in COMPLETE_LIFECYCLE_STATES)
    a = f"{alias}." if alias else ""
    return f"CASE WHEN {a}lifecycle_state IN ({states}) THEN 1 ELSE 0 END AS is_complete"


def decode_family_default_sql(run_alias: str = "r") -> str:
    """Two columns rendered beside a run's derived decode-policy match in the paper views (`v_paper_current`, `v_paper_long`; migration 0009):

    ``decode_family_default_policy_id`` -- the ``decode_policy`` row ``decode_policy.resolve(model, eval, thinking)`` returns TODAY for the run's
    key: the live (not superseded) ``is_default = 1`` row scoped to the run's eval, else the one scoped to the eval's category (C18: an eval row
    wins over a category row; the loader admits at most one live default per key); NULL when the key has no default.

    ``decode_is_family_default`` -- 1 when the run's matched row (``run.decode_policy_id``, an ``exact`` verdict) IS that default; 0 when it matched
    a different, recorded row (a NON-default policy: every agentic run at the runner's T = 1.0 after the rule of 2026-09-21 made the vendor
    preset the default of every family, ``decode-vendor-default-all-evals-2026-09-21``); NULL when the run matched no row (``decode_policy_match``
    is not ``exact``: nothing to compare). The default is the authority's CURRENT state, never the default at the run's generation time -- a
    flip is recorded on the row (its ``notes``), and the match itself never reads the flag (a run keeps matching the row it sent).
    """
    r = run_alias
    eval_default = (f"(SELECT p.policy_id FROM decode_policy p WHERE p.model_id = {r}.model_id AND p.thinking_mode = {r}.thinking_mode "
                    f"AND p.eval_id = {r}.eval_id AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)")
    cat_default = (f"(SELECT p.policy_id FROM decode_policy p JOIN eval fe ON fe.eval_id = {r}.eval_id WHERE p.model_id = {r}.model_id "
                   f"AND p.thinking_mode = {r}.thinking_mode AND p.eval_id IS NULL AND p.eval_category = fe.eval_category "
                   f"AND p.is_default = 1 AND p.superseded_by_policy_id IS NULL)")
    fd = f"COALESCE({eval_default}, {cat_default})"
    return (f"{fd} AS decode_family_default_policy_id,\n"
            f"       CASE WHEN {r}.decode_policy_id IS NULL THEN NULL WHEN {r}.decode_policy_id = {fd} THEN 1 ELSE 0 END AS decode_is_family_default")


# --------------------------------------------------------------------------------------------
# the operating point of a run WITHOUT a stamp (migration 0011)
# --------------------------------------------------------------------------------------------

#: The registry ruling this fallback implements: when a run of a fitted object carries NO `is_operating_point` stamp, the run at
#: `strength_nominal = 1.0` IS the operating point, and the Arditi all-positions steering vector defaults to its published
#: coefficient c = 0.3. A recorded stamp (1 or 0) always wins over the fallback.
OPERATING_POINT_RULING_ID = "operating-point-default-fitted-strength-1p0-arditi-c0p3-2026-09-22"
#: The ONE list of object kinds the fallback covers (:data:`OPERATING_POINT_RULING_ID`): the fitted LoRA / ReLU objects, their
#: differences and the paired write (`derive.FITTED_DOSE_OBJECT_KINDS`, whose readers stamp `is_operating_point = (dose == 1.0)`
#: themselves), the published baselines (`static_r_arditi`, `alphasteer`, `vector_all`: their own readers rule the point on their own
#: dose axis -- alpha, lambda, the raw coefficient -- but the paper export files their multi-turn and Petri rows at the bare served
#: key's nominal 1.0 with the stamp NULL, 85 live rows on 2026-09-22 outside the frame for it), and the steering-vector kinds beside
#: them (`vector`, `diffmean`, `antidote`). NOT here: `none` and `system_instruction` (no strength axis; the operating_point rule
#: admits them on their kind alone, :data:`NO_STRENGTH_AXIS_OBJECT_KINDS`), and `caa_static_r` / `stacked` (no live object of either
#: kind on 2026-09-22; add them here when one is served and the ruling is read to cover it).
OPERATING_POINT_FALLBACK_OBJECT_KINDS: Tuple[str, ...] = (
    "lora", "lora_subtraction", "composed", "relu_steer", "relu_steer_distilled_difference",
    "alphasteer", "static_r_arditi", "vector", "vector_all", "diffmean", "antidote",
)
#: The nominal the fallback reads as the operating point (the campaign nominal: 1.0 = the shipped object, `derive.FITTED_OPERATING_NOMINAL`).
OPERATING_POINT_FALLBACK_NOMINAL = 1.0
#: The Arditi all-positions steering vector (`object_kind = 'vector_all'`, paper label `arditi_allpos`) and its published operating
#: coefficient. The field that carries c is `run.strength_coefficient`: the arditi_allpos reader files c there (0.3 at nominal 1.0,
#: 0.1 at 0.333, ...; `strength_unit = 'raw_coefficient'`), the export files no coefficient at all (its bare served key
#: `ar_c0p3__arditi_allpos` is nominal 1.0, covered by the nominal clause), and the seed-replicate readers file the raw c AS the nominal
#: (0.3, `strength_unit = 'unverified'`) with the stamp already set. The clause therefore reads `strength_coefficient` and nothing else.
ARDITI_ALLPOS_OBJECT_KIND = "vector_all"
ARDITI_ALLPOS_OPERATING_COEFFICIENT = 0.3
#: The kinds with no strength axis: `strength_nominal` / `strength_coefficient` / `is_operating_point` are NULL by design (the column
#: why-null reasons), and a roster entry's `operating_point` rule admits them on the kind alone (unchanged since 0001).
NO_STRENGTH_AXIS_OBJECT_KINDS: Tuple[str, ...] = ("none", "system_instruction")
#: `operating_point_basis` (v_paper_current, v_paper_long): HOW a run counts as being at the operating point -- `stamped` (a filed
#: `is_operating_point = 1`), `fallback_nominal_1p0` (no stamp, a fallback kind at nominal 1.0), `fallback_arditi_c0p3` (no stamp, the
#: all-positions vector at c = 0.3), `no_strength_axis` (an unsteered or prompted run: admitted on its kind), `none` (not at the
#: operating point: a stamp of 0, or no stamp and no clause applies). A consumer can tell a stamped row from one the fallback admitted.
OPERATING_POINT_BASES: Tuple[str, ...] = ("stamped", "fallback_nominal_1p0", "fallback_arditi_c0p3", "no_strength_axis", "none")


def operating_point_basis_sql(r: str = "r", o: str = "o") -> str:
    """The CASE expression behind ``operating_point_basis`` (:data:`OPERATING_POINT_BASES`) over a run alias ``r`` joined to its object
    alias ``o`` -- the ONE rendering of the ruling :data:`OPERATING_POINT_RULING_ID`; :func:`operating_point_sql` is its predicate form.
    A stamp of 1 is read first, a stamp of 0 falls through every fallback to ``none`` (a filed stamp always wins), and the two fallbacks
    apply to a NULL stamp only. The nominal and the coefficient are compared within 1e-9 (``derive.fitted_is_operating_point`` compares
    the same way); a NULL nominal or coefficient never matches (``ABS(NULL - x)`` is NULL)."""
    kinds = ", ".join(sql_literal(k) for k in OPERATING_POINT_FALLBACK_OBJECT_KINDS)
    bare = ", ".join(sql_literal(k) for k in NO_STRENGTH_AXIS_OBJECT_KINDS)
    return (f"CASE WHEN {r}.is_operating_point = 1 THEN 'stamped'\n"
            f"            WHEN {r}.is_operating_point IS NULL AND {o}.object_kind IN ({kinds})\n"
            f"                 AND ABS({r}.strength_nominal - {OPERATING_POINT_FALLBACK_NOMINAL!r}) < 1e-9 THEN 'fallback_nominal_1p0'\n"
            f"            WHEN {r}.is_operating_point IS NULL AND {o}.object_kind = {sql_literal(ARDITI_ALLPOS_OBJECT_KIND)}\n"
            f"                 AND ABS({r}.strength_coefficient - {ARDITI_ALLPOS_OPERATING_COEFFICIENT!r}) < 1e-9 THEN 'fallback_arditi_c0p3'\n"
            f"            WHEN {o}.object_kind IN ({bare}) THEN 'no_strength_axis'\n"
            f"            ELSE 'none' END")


def operating_point_sql(r: str = "r", o: str = "o") -> str:
    """The predicate a roster entry's ``operating_point`` strength rule applies (v_roster_match): the run is at the operating point under
    :func:`operating_point_basis_sql` by any basis but ``none``."""
    return f"(({operating_point_basis_sql(r, o)}) <> 'none')"


def _rank_sql(row: str, sr: str, table: str) -> str:
    """The precedence of one visible provenance as a row value: native first; on a table with a ``measurement_state``
    (``run_degeneration``) a row that carries a number (``measured``) before every placeholder state (``not_measured``,
    ``not_applicable``, ``refused`` — a higher-priority store reader's ``not_measured`` placeholder hid the
    union reader's measured row, so the state ranks ahead of the declared priority, not merely ahead of the source-id
    tiebreak); then the higher declared priority, then the lexically smaller source_id, then revision_id — a total order,
    so exactly one row per measurement key is current."""
    state = (f"CASE WHEN {row}.measurement_state = 'measured' THEN 0 ELSE 1 END, "
             if "measurement_state" in TABLE_BY_NAME[table].column_names else "")
    return (f"(CASE WHEN {row}.source_revision_id = '' THEN 0 ELSE 1 END, {state}-COALESCE({sr}.priority, 0), "
            f"COALESCE({sr}.source_id, ''), {row}.source_revision_id)")


def measurement_key(table: str) -> Tuple[str, ...]:
    """The ORIGINAL measurement key of an item table: its natural key without the revision tag."""
    return tuple(c for c in TABLE_BY_NAME[table].pk if c != "source_revision_id")


def item_not_withdrawn(table: str, alias: str) -> str:
    """The producer-withdrawal predicate (migration 0020), appended to every CURRENT-item predicate of a table that carries
    the mark: a row an ``item_withdrawal`` record retracted is not current, exactly as a row of a superseded revision is not.
    Empty for the item tables without the column. Kept OUT of :func:`item_visible`: that predicate is shared by the deferred-FK
    triggers (a withdrawn row must still satisfy its foreign keys) and by the other item tables."""
    if "withdrawn_by_record_sha256" not in TABLE_BY_NAME[table].column_names:
        return ""
    return f" AND {alias}.withdrawn_by_record_sha256 IS NULL"


#: The judgment statuses whose measurement rows are NOT current (migration 0021): a judgment a re-judge
#: displaced (`superseded`) or a claim its session released (`abandoned`) is history, and the `run_metric` / `item_result` rows filed
#: under it leave every current reading -- `v_paper_long`, `v_item_result_current`, `v_item_result_provenances` -- while the tables keep
#: them. A row with no judgment (judgment_id '') and a row of an `in_progress`, `partial` or `complete` judgment are unaffected.
NON_CURRENT_JUDGMENT_STATUSES: Tuple[str, ...] = ("superseded", "abandoned")


def non_current_judgment_sql(judgment_id: str, status: str) -> str:
    """The judgment-status predicate as a row test over a judgment_id column and the (LEFT-joined) judgment status beside it: TRUE
    when the row is current -- no judgment, no judgment row found, or a status outside :data:`NON_CURRENT_JUDGMENT_STATUSES`."""
    statuses = ", ".join(sql_literal(s) for s in NON_CURRENT_JUDGMENT_STATUSES)
    return f"({judgment_id} = '' OR {status} IS NULL OR {status} NOT IN ({statuses}))"


def judgment_displaced_by_complete_sql(j: str, r: str = "r", i: str = "i") -> str:
    """The complete-outranks-partial rule as a row test over a LEFT-joined judgment alias ``j``, its run ``r`` and the run's
    instrument ``i`` (LEFT-joined): TRUE when the judgment is not a `partial` one displaced by a `complete` judgment of the SAME
    (run_id, judge_id) -- a partial reading beside a complete one by the same judge is not a paper row (the complete one is), while a
    partial reading alone is admitted as before and a complete judgment of ANOTHER judge demotes nothing. The displacing witness must
    itself be identity-current under the run's PRESENT identity (:func:`judgment_identity_current_sql`): a complete judgment bound at
    claim to a since-corrected identity is non-current under the identity rule and displaces nothing -- with a status-only witness a
    stale complete beside a current partial would have hidden the partial too, and both readings would have left the paper. TRUE for a
    row with no judgment (status NULL). A pure function of the judgment, run and instrument rows; replay-equal. Paper side only:
    `v_pending_judgment` has its own partial path (identity-current there too). Live when the rule landed (read-only, 2026-10-03
    16:00Z): 0 (run, judge) pairs carry both a partial and a complete judgment, so the rule moves no live row."""
    return (f"(COALESCE({j}.status, '') <> 'partial' OR NOT EXISTS (SELECT 1 FROM judgment jcp WHERE jcp.run_id = {j}.run_id "
            f"AND jcp.judge_id = {j}.judge_id AND jcp.status = 'complete' AND {judgment_identity_current_sql('jcp', r, i)}))")


def judgment_identity_current_sql(j: str, r: str = "r", i: str = "i") -> str:
    """The identity-currency rule as a row test over a LEFT-joined judgment alias ``j``, its run ``r`` and the run's instrument ``i``
    (LEFT-joined): TRUE while the identity BOUND AT CLAIM equals the run's CURRENT identity -- `claimed_instrument_id` IS the run's
    `instrument_id` and `claimed_required_judge_id` IS the run's current required judge (the instrument's `judge_id`, else
    `run.judge_id`) -- each binding compared only when stated (a NULL binding is current: every legacy row, every writer that does not
    know the instrument). A `complete` or `partial` judgment made under a since-corrected run identity (a same-id run record moving the
    instrument or the required judge) therefore leaves every paper frame and ownership scope while the run is listed for the new judge;
    a SECOND judge's judgment (the designed second reading of migration 0003: its judge is never the required one) is untouched,
    because the rule compares the binding, never the judge. Zero live rows move at the migration (the columns are new, so every live
    row is NULL-bound). `v_pending_judgment` applies the SAME test to the judgments it counts: its coverage anti-join, its active-claim
    join and its `partial` / `coverage_short` reasons read only identity-current judgments, and a run whose only judgments by the
    required judge (a claim, a partial or a complete reading) are stale-bound reads `pending_reason = identity_stale` -- without that a
    same-judge instrument correction (X to Y, both requiring J) left J's complete judgment non-current on the paper side while still
    covering J's work in the worklist: neither a paper row nor a pending job, unmeasured indefinitely; and J's stale-bound in-progress
    claim read `claimed`, a displayed job that could only complete into a non-current judgment and that `pending.claim` refused every
    other session (`pending.claim` now closes such a claim and binds a new one to the current identity). The other consumers of the
    coverage predicate apply the same test: `v_roster_match` and :func:`measurement_judge_sql` (a stale-bound covering judgment
    satisfies no roster entry) and the harvester's completion requirement (`harvest.completion_requirements_hold`: a run whose only
    complete judgment is stale-bound stays `judged`, never `ingested`), so an identity-stale complete judgment covers nothing
    anywhere."""
    return (f"(({j}.claimed_instrument_id IS NULL OR {j}.claimed_instrument_id IS {r}.instrument_id) "
            f"AND ({j}.claimed_required_judge_id IS NULL OR {j}.claimed_required_judge_id IS COALESCE({i}.judge_id, {r}.judge_id)))")


def judgment_currency_sql(judgment_id: str, j: str, r: str = "r", i: str = "i") -> str:
    """The three paper-side judgment-currency rules rendered together, each under its own comment, as `AND` conjuncts to follow a caller's
    own first predicate: the status rule (:func:`non_current_judgment_sql`), the complete-over-partial rule
    (:func:`judgment_displaced_by_complete_sql`) and the identity rule (:func:`judgment_identity_current_sql`). Over the aliases of
    the site -- ``judgment_id`` the row's judgment_id column, ``j`` the LEFT-joined judgment, ``r`` the run, ``i`` the run's instrument."""
    return (f"""-- a row of a judgment that a later judgment displaced or a claim released is not current (migration 0021; NON_CURRENT_JUDGMENT_STATUSES):
  -- the new judgment stands in its place; a row with no judgment, or of an in-progress, partial or complete one, is unaffected
  AND {non_current_judgment_sql(judgment_id, f"{j}.status")}
  -- a partial judgment beside a complete one of the SAME run_id and judge_id is not current: the complete reading is the paper row
  -- (the displacing complete judgment itself identity-current under the run: a stale complete displaces nothing)
  AND {judgment_displaced_by_complete_sql(j, r, i)}
  -- a judgment whose identity bound at claim differs from the current run instrument_id or required judge_id is not current
  -- (no apostrophe in this comment: the reference scan strips quoted literals);
  -- a NULL binding is current, so every legacy row and every additional reading by another judge_id stand
  AND {judgment_identity_current_sql(j, r, i)}""")


def item_judgment_live(table: str, alias: str) -> str:
    """The judgment-currency predicate (migration 0021 -- since the record-refusal-fact migration the complete-over-partial and
    identity rules too), appended to the CURRENT-item predicates of ``item_result`` (the table the rule names; the rival row of the
    anti-join shares the measurement key, judgment_id included, so it needs no second probe): a row whose ``judgment_id`` names a
    judgment with a status in :data:`NON_CURRENT_JUDGMENT_STATUSES`, a `partial` judgment displaced by a `complete` one of the same
    (run, judge), or a judgment whose identity bound at claim differs from its run's current instrument or required judge is not
    current. ONE probe of the judgment row per item row; the complete-judgment probe runs only for a partial judgment (its witness read
    through the run and the run's instrument, so a stale-bound complete displaces nothing) and the run / instrument probe only for a
    judgment that carries a binding (CASE guards both: a legacy judgment costs what it did under 0021). Empty for the other item
    tables. Kept OUT of :func:`item_visible` for the same reason as :func:`item_not_withdrawn`: the deferred-FK triggers and the other
    item tables share that predicate."""
    if table != "item_result":
        return ""
    statuses = ", ".join(sql_literal(s) for s in NON_CURRENT_JUDGMENT_STATUSES)
    return (f" AND ({alias}.judgment_id = '' OR NOT EXISTS (SELECT 1 FROM judgment jx WHERE jx.judgment_id = {alias}.judgment_id "
            f"AND (jx.status IN ({statuses})"
            f" OR CASE WHEN jx.status = 'partial' THEN EXISTS (SELECT 1 FROM judgment jcp JOIN run rcp ON rcp.run_id = jcp.run_id"
            f" LEFT JOIN instrument icp ON icp.instrument_id = rcp.instrument_id WHERE jcp.run_id = jx.run_id AND jcp.judge_id = jx.judge_id"
            f" AND jcp.status = 'complete' AND {judgment_identity_current_sql('jcp', 'rcp', 'icp')}) ELSE 0 END"
            f" OR CASE WHEN jx.claimed_instrument_id IS NULL AND jx.claimed_required_judge_id IS NULL THEN 0"
            f" ELSE EXISTS (SELECT 1 FROM run rx LEFT JOIN instrument ix ON ix.instrument_id = rx.instrument_id WHERE rx.run_id = jx.run_id"
            f" AND NOT {judgment_identity_current_sql('jx', 'rx', 'ix')}) END)))")


def current_rows_sql(table: str) -> str:
    """``v_<table>_current``: the visible provenances of ``table`` reduced to ONE deterministic row per measurement key by
    :func:`_rank_sql` — the row no better-ranked visible provenance of the same key exists for (an anti-join on the key's
    index, so a key with a single provenance costs one probe). A producer-withdrawn row (:func:`item_not_withdrawn`) is
    neither current nor a rival of another provenance of its key; a row of a superseded or abandoned judgment
    (:func:`item_judgment_live`) is not current either."""
    key = measurement_key(table)
    same = " AND ".join(f"o.{c} = ir.{c}" for c in key)
    return (f"\nCREATE VIEW v_{table}_current AS\nSELECT ir.* FROM {table} ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id\n"
            f"WHERE {item_visible('ir')}{item_not_withdrawn(table, 'ir')}{item_judgment_live(table, 'ir')}\n  AND NOT EXISTS (SELECT 1 FROM {table} o LEFT JOIN source_revision so ON so.revision_id = o.source_revision_id\n"
            f"                  WHERE {same} AND o.source_revision_id <> ir.source_revision_id\n"
            f"                    AND {item_visible('o')}{item_not_withdrawn(table, 'o')} AND {_rank_sql('o', 'so', table)} < {_rank_sql('ir', 'sr', table)});\n")


def _fk_trigger(table_name: str, col: str, parent: str, parent_col: str, sentinel: Optional[str], nullable: bool, deferred: bool) -> List[str]:
    """An FK enforced by trigger: on a sentinel column the FK cannot hold on the sentinel; a DEFERRED FK (item tables) holds
    for every visible row — a pending revision's rows name parents its activation folds first, so they are exempt until then."""
    conds = []
    if sentinel is not None:
        conds.append(f"NEW.{col} <> {sentinel}")
    elif nullable:
        conds.append(f"NEW.{col} IS NOT NULL")
    if deferred:
        conds.append("(NEW.source_revision_id = '' OR EXISTS (SELECT 1 FROM source_revision sr_v WHERE sr_v.revision_id = NEW.source_revision_id AND sr_v.accepted = 1))")
    conds.append(f"NOT EXISTS (SELECT 1 FROM {parent} WHERE {parent_col} = NEW.{col})")
    why = "sentinel-column FK" if sentinel is not None and not deferred else ("FOREIGN KEY of a visible row, D50-2" if deferred else "FOREIGN KEY")
    out = []
    for op in ("INSERT", "UPDATE OF " + col):
        tname = f"trg_{table_name}_{col}_fk_{'ins' if op == 'INSERT' else 'upd'}"
        out.append(
            f"CREATE TRIGGER {tname} BEFORE {op} ON {table_name}\n"
            f"WHEN " + " AND ".join(conds) + "\n"
            f"BEGIN\n  SELECT RAISE(ABORT, '{table_name}.{col} names no {parent}.{parent_col} ({why})');\nEND;")
    return out


#: The tables the fold-order fence guards (:func:`fold_order_fence_triggers`): ``run`` alone. ``reducer.tie_rank`` tells ``run_state``
#: events to a lateral or terminal state apart and nothing else, so ``run`` is the one table whose rows the pre-tie-break order
#: (``fold_order/1``) and the current one (``fold_order/2``) fold differently; every other run-scale table folds the same under both.
FOLD_ORDER_FENCE_TABLES: Tuple[str, ...] = ("run",)


def fold_order_fence_triggers() -> List[str]:
    """The fold-order FENCE: ``BEFORE INSERT`` / ``BEFORE UPDATE`` triggers on each of :data:`FOLD_ORDER_FENCE_TABLES` that refuse the
    write when the connection's fold order -- the SQL function ``db.FOLD_ORDER_FUNCTION`` (``registry_fold_order()``), which every
    ``db.connect`` registers with ``reducer.FOLD_ORDER_VERSION`` -- is not the order the database records in ``schema_migration`` row
    ``db.FOLD_ORDER_ROW_VERSION``. A connection that never registered the function cannot write the row at all: SQLite resolves the
    function when it compiles the INSERT or UPDATE, so the write fails ``no such function: registry_fold_order`` before a row moves.
    That is what reaches the writers the stamp alone did not (registry-fold-order-version review, round 3): a pre-stamp checkout's
    ``replica refold-apply`` opened the live file with a bare ``db.connect`` and rewrote run rows under its own order with no check of
    version, shape or order (commit 5ea3fa411), and no Python-side guard of THIS code runs inside a process of THAT code. A database
    recording NO order (row 0 absent) is not fenced: the Python check (``db.require_fold_order``) stamps it on the first writer, and the
    trigger compares only against a recorded order, so the bootstrap is the same on both sides. Rendered into ``schema.sql`` and
    created by migration 0030; a rebuild of ``run`` recreates them with the table's other triggers (:func:`triggers_on`)."""
    from . import db as DB
    out: List[str] = []
    for table_name in FOLD_ORDER_FENCE_TABLES:
        for op, name in (("INSERT", "ins"), ("UPDATE", "upd")):
            out.append(
                f"CREATE TRIGGER trg_{table_name}_fold_order_fence_{name} BEFORE {op} ON {table_name}\n"
                f"WHEN EXISTS (SELECT 1 FROM schema_migration WHERE version = {int(DB.FOLD_ORDER_ROW_VERSION)} AND description IS NOT {DB.FOLD_ORDER_FUNCTION}())\n"
                "BEGIN\n"
                f"  SELECT RAISE(ABORT, '{table_name}: fold-order fence -- this connection folds under a same-stamp order the database does not record "
                f"({DB.FOLD_ORDER_FUNCTION}() against schema_migration row {int(DB.FOLD_ORDER_ROW_VERSION)}); nothing written: fast-forward the checkout or "
                "rebuild the database (docs/registry/records.md, Identity and order: the run-scale order has a version)');\n"
                "END;")
    return out


def render_triggers() -> List[str]:
    trig: List[str] = []
    # judged requires a judgment
    for op, name in (("INSERT", "ins"), ("UPDATE OF lifecycle_state", "upd")):
        trig.append(
            f"CREATE TRIGGER trg_run_judged_requires_judgment_{name} BEFORE {op} ON run\n"
            "WHEN NEW.lifecycle_state = 'judged' AND NOT EXISTS (SELECT 1 FROM judgment WHERE run_id = NEW.run_id)\n"
            "BEGIN\n  SELECT RAISE(ABORT, 'run.lifecycle_state=judged requires a judgment row for this run_id');\nEND;")
    # reached-state utc columns, from vocab.LIFECYCLE_REQUIRED_UTC (states sharing a requirement render as one IN (...))
    conds = []
    groups: List[Tuple[Tuple[str, ...], List[str]]] = []
    for state, required in V.LIFECYCLE_REQUIRED_UTC.items():
        if groups and groups[-1][0] == required and state not in V.LIFECYCLE_PROGRESSION:
            groups[-1][1].append(state)
        else:
            groups.append((required, [state]))
    for required, states in groups:
        who = f"= '{states[0]}'" if len(states) == 1 else "IN (" + ",".join(f"'{st}'" for st in states) + ")"
        cond = " OR ".join(f"NEW.{c} IS NULL" for c in required)
        conds.append(f"(NEW.lifecycle_state {who} AND ({cond}))" if len(required) > 1 or len(states) == 1
                     else f"(NEW.lifecycle_state {who} AND {cond})")
    for op, name in (("INSERT", "ins"), ("UPDATE", "upd")):
        trig.append(
            f"CREATE TRIGGER trg_run_state_utc_{name} BEFORE {op} ON run\n"
            "WHEN " + "\n  OR ".join(conds) + "\n"
            "BEGIN\n  SELECT RAISE(ABORT, 'run: every *_utc of a reached lifecycle state must be non-null (trg_run_state_utc)');\nEND;")
    # the fold-order fence on `run` (migration 0030): a write by a connection whose order is not the recorded one is refused
    trig += fold_order_fence_triggers()
    # sentinel-column FKs (Column.fk + Column.sentinel) and deferred FKs (Column.fk_deferred) are triggers
    for t in TABLES:
        for c in t.columns:
            if c.fk and (c.sentinel is not None or c.fk_deferred):
                trig += _fk_trigger(t.name, c.name, c.fk[0], c.fk[1], sql_literal(c.sentinel) if c.sentinel is not None else None,
                                    not c.nn, c.fk_deferred)
    return trig


# --------------------------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------------------------


# ONE coverage predicate for completion (the harvester's `completion_requirements_hold`), roster satisfaction
# (v_roster_match, v_paper_long) and grading eligibility (v_pending_judgment). The coverage a judgment must reach is the items
# the run generated (planned when generation has not reported), capped by the declared item set's size (an instrument that
# judges a declared subset); a judgment COVERS it when it is complete and the items it judged plus the items the judge
# DECLINED plus the items whose judging ERRORED reach it — a judge refusal is an outcome the judgment records, not a gap a
# re-grade would fill, so a refusing judge never leaves a run pending forever while the paper build waits on the same run;
# an error is an outcome for a planned item with no verdict (`judgment.n_errors`, else the judgment's own metric rows'
# `run_metric.n_error_outcomes`), never a verdict and never a gap — before it counted (2026-09-21) a run whose judge covered
# every planned item but recorded one error stayed `judged` forever (or_bench_hard: 1,314 judged + 4 declined + 1 error =
# 1,319 planned; xstest: 249 judged + 1 error = 250; 38 live runs of two readers). An item the producer reports as
# `n_incomplete` is NOT covered: it has no outcome at all.
def required_coverage_sql(r: str = "r") -> str:
    """SQL for the coverage a judgment of run alias ``r`` must reach: min(items generated / planned, item set size); the
    run's own count when the item set is absent; NULL when neither states a size."""
    n = f"COALESCE({r}.n_items_generated, {r}.n_items_planned)"
    return f"COALESCE((SELECT MIN(COALESCE({n}, s.n_items), s.n_items) FROM item_set s WHERE s.item_set_id = {r}.item_set_id), {n})"


def judgment_covered_items_sql(j: str) -> str:
    """SQL for the items judgment alias ``j`` recorded an outcome for: judged + declined + errored. The error count is the
    judgment's own (``n_errors``) when the judging driver counted one, else the largest ``n_error_outcomes`` its own
    ``run_metric`` rows carry (one judgment's metrics describe the same items, so the count is one number filed per row);
    a judgment that counted neither contributes 0. ``n_incomplete`` never counts: an incomplete item has no outcome."""
    return (f"COALESCE({j}.n_items_judged, 0) + COALESCE({j}.n_judge_refusals, 0) "
            f"+ COALESCE({j}.n_errors, (SELECT MAX(rm_err.n_error_outcomes) FROM run_metric rm_err "
            f"WHERE rm_err.run_id = {j}.run_id AND rm_err.judgment_id = {j}.judgment_id), 0)")


def run_superseded_sql(r: str = "r") -> str:
    """SQL: run alias ``r`` has been SUPERSEDED by a successor run (``superseded_by_run_id`` set -- the static correction re-registering
    a unit under a new run id, a supersede ruling retiring the old one). The rule (migration 0025, 2026-10-04): a superseded run is never
    pending for any judge -- its successor is. `v_pending_judgment` excludes such a run outright, and `pending.not_pending_why` names it
    (``ClaimNotPending``: `superseded by run <id>; admits no judging`) through this one predicate, never a re-encoding. Before it an
    identity correction (instrument X to X2, same judge) landing on a superseded run whose complete judgment had been claimed under X
    made the retired run read `identity_stale` beside its successor -- claimable again."""
    return f"{r}.superseded_by_run_id IS NOT NULL"


def judgment_covers_sql(j: str, r: str = "r") -> str:
    """SQL: judgment alias ``j`` is complete and covers run alias ``r`` (judged + declined + errored >= required coverage,
    :func:`judgment_covered_items_sql`). Status and arithmetic only: every consumer (`v_pending_judgment`, `v_roster_match`,
    :func:`measurement_judge_sql`, the harvester's completion requirement) conjoins :func:`judgment_identity_current_sql` beside it,
    so a complete judgment bound at claim to a since-corrected run identity covers nothing anywhere."""
    return f"{j}.status = 'complete' AND {judgment_covered_items_sql(j)} >= COALESCE({required_coverage_sql(r)}, 0)"


def _sel_eq(sel: str, key: str, col: str) -> str:
    """Selector field ``key`` of JSON ``sel`` constrains ``col`` when present (NULL = no constraint)."""
    return f"(json_extract({sel}, '$.{key}') IS NULL OR json_extract({sel}, '$.{key}') = {col})"


def _sel_in(sel: str, key: str, col: str) -> str:
    """Selector LIST ``key`` of JSON ``sel`` must contain ``col`` when present."""
    return (f"(json_type({sel}, '$.{key}') IS NULL OR EXISTS (SELECT 1 FROM json_each({sel}, '$.{key}') sv WHERE sv.value = {col}))")


def _m_in(key: str, col: str, alias: str = "ru") -> str:
    """``scope_json.match.<key>`` (a list) must contain ``col`` when present."""
    return (f"(json_type({alias}.scope_json, '$.match.{key}') IS NULL OR EXISTS (SELECT 1 FROM json_each({alias}.scope_json, '$.match.{key}') v "
            f"WHERE v.value = {col}))")


def _m_prefix(key: str, col: str, alias: str = "ru") -> str:
    """``scope_json.match.<key>`` (a list of prefixes) must prefix ``col`` when present."""
    return (f"(json_type({alias}.scope_json, '$.match.{key}') IS NULL OR EXISTS (SELECT 1 FROM json_each({alias}.scope_json, '$.match.{key}') v "
            f"WHERE {col} GLOB v.value || '*'))")


def _a_in(key: str, col: str) -> str:
    """One instrument alternative (``alt.value``) constrains ``col`` by its list ``key`` when present."""
    return f"(json_type(alt.value, '$.{key}') IS NULL OR EXISTS (SELECT 1 FROM json_each(alt.value, '$.{key}') w WHERE w.value = {col}))"


#: A metric row whose observation key is no longer current under any revision (one pass; the per-row scan does not scale).
WITHDRAWN_METRIC_CTE = """withdrawn_metric AS MATERIALIZED (
    -- a metric row whose observation key is no longer current under any revision (one pass; the per-row scan does not scale)
    SELECT json_extract(observation_key, '$.run_id') AS run_id, json_extract(observation_key, '$.judgment_id') AS judgment_id,
           json_extract(observation_key, '$.metric_id') AS metric_id, json_extract(observation_key, '$.threshold') AS threshold,
           json_extract(observation_key, '$.computation_id') AS computation_id
    FROM observation WHERE table_name = 'run_metric' GROUP BY observation_key HAVING MAX(current) = 0
)"""

_WITHDRAWN_JOIN = ("LEFT JOIN withdrawn_metric wm ON wm.run_id = rm.run_id AND wm.judgment_id = rm.judgment_id AND wm.metric_id = rm.metric_id\n"
                   "     AND wm.threshold = rm.threshold AND wm.computation_id = rm.computation_id")

#: A contrast row whose observation key is no longer current under any revision: the anchor no source's head revision pairs the run with
#: (migration 0017). `run_contrast`'s key carries `anchor_run_id`, so a reader whose newer accepted revision contrasts a run against a
#: DIFFERENT anchor than its earlier revision did leaves the older row in the table beside the new one -- no run-scale row is ever deleted;
#: what the fold withdraws is the OBSERVATION of its key (current 0, `withdrawn_by_revision_id`), exactly as for a metric row -- and
#: `v_paper_long` read `run_contrast` whole, so a metric row appeared once per anchor. Live, 2026-09-26 (read-only census): 11,010 contrast
#: rows, every one observed, 10,050 current and 960 withdrawn; 774 (run, judgment, metric, threshold, computation) keys carry two anchors,
#: 376 of them one current and one withdrawn (the four seed-replicate readers' fallback contrast against the seed tag's `none` beside
#: the own-none contrast the current reader emits, 12 keys of 12 runs; the export's half-dose `full_dose` contrasts at the 0.25 / 0.75
#: thresholds its emission version 3 stopped filing beside the `same_standup_none` row, 364 keys of 182 runs), plus 584 keys whose only
#: contrast is withdrawn (86 of them frame rows: the metric row stays, its contrast columns read NULL). The key columns are CAST like
#: METRIC_ROLE_CTES' (an untyped json_extract column joins by full scan).
WITHDRAWN_CONTRAST_CTE = """withdrawn_contrast AS MATERIALIZED (
    -- a contrast whose observation key is no longer current under any revision: the anchor no head revision pairs the run with (0017)
    SELECT CAST(json_extract(observation_key, '$.run_id') AS TEXT) AS run_id, CAST(json_extract(observation_key, '$.anchor_run_id') AS TEXT) AS anchor_run_id,
           CAST(json_extract(observation_key, '$.judgment_id') AS TEXT) AS judgment_id, CAST(json_extract(observation_key, '$.metric_id') AS TEXT) AS metric_id,
           CAST(json_extract(observation_key, '$.threshold') AS REAL) AS threshold, CAST(json_extract(observation_key, '$.computation_id') AS TEXT) AS computation_id
    FROM observation WHERE table_name = 'run_contrast' GROUP BY observation_key HAVING MAX(current) = 0
)"""

#: The contrasts the paper frame joins (migration 0017): every `run_contrast` row minus the withdrawn ones. THE RULE -- a contrast is CURRENT
#: when no observation of its key is withdrawn: some source's head accepted revision still carries the (run, anchor) pair, or no source
#: ever observed it (a natively emitted contrast has no observation and is never withdrawn) -- the reading `withdrawn_metric` gives a
#: metric row. `ingest_seq` is NOT the rule: it is the ledger rank of the file that last CHANGED the row (a byte-identical re-emission is a
#: duplicate sha and leaves it), so a re-emitted current contrast can carry a smaller rank than a withdrawn one (though in every live stale
#: pair the current row also carried the greater rank). A metric row then joins one row per CURRENT contrast of its key: normally one; two
#: where a source files two anchors of different kinds by design (the export's half-dose runs carry a `same_standup_none` and a `full_dose`
#: contrast at the headline threshold: 398 keys of 209 runs live, both current) -- a consumer that wants one picks by `anchor_kind`.
CURRENT_CONTRAST_CTE = """current_contrast AS MATERIALIZED (
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
)"""

#: The selection of every run, evaluated ONCE for the paper frame (migration 0021): `v_paper_long` reads its own rows from it as `pc`
#: and :data:`CURRENT_CONTRAST_CTE` reads the currency of every anchor from the same pass, so the ruling fold behind `v_run_current`
#: runs once per query of the frame (joining the run-currency object a second time would run the 30-84 s chain twice, live).
PAPER_CURRENT_CTE = """paper_current AS MATERIALIZED (
    -- the selection of every run, evaluated once for the frame (migration 0021): the rows read it as pc, the contrast join reads the
    -- currency of each anchor from the same pass
    SELECT * FROM v_paper_current
)"""

#: The role and fold state of every metric row inside a declared multi-source scope, read ONCE per query of `v_paper_current` (migration
#: 0021): its coverage test admits a headline row the way `v_paper_long` does (:func:`frame_admission_sql`), and the frame reads the
#: same two objects through its own LEFT JOINs; hoisting them here keeps the per-run EXISTS a probe into a materialized relation (the two
#: objects read about 1.5 s live, 2026-10-02, against the 65 s of the selection fold).
MULTI_SOURCE_STATE_CTE = """multi_source AS MATERIALIZED (
    -- the role and fold state of every metric row inside a declared multi-source scope (v_multi_source_rows / v_multi_source_check),
    -- read once: the coverage test below admits a row the way the frame does -- a folded or refused secondary row is not a frame row
    SELECT msr.run_id, msr.judgment_id, msr.metric_id, msr.threshold, msr.computation_id, msr.role, msc.state
    FROM v_multi_source_rows msr
    LEFT JOIN v_multi_source_check msc ON msc.declaration_id = msr.declaration_id AND msc.pairing_key = msr.pairing_key AND msc.eval_id = msr.eval_id
         AND msc.metric_id = msr.metric_id AND msc.threshold = msr.threshold
)"""

#: The ACCEPTED headline metric ids of every eval (migration 0008): `v_item_headline` and, since migration 0021, `v_paper_current`
#: (`coverage_state`) read the same CTE text.
HEADLINE_ACCEPTED_CTE = """accepted AS (
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
)"""

def measurement_judge_sql(r: str = "r", i: str = "i") -> str:
    """The judge predicate that admits ONE measurement (aliases `rm` the run_metric row, `jm` its judgment, `j` the judge of that
    judgment, ``i`` the instrument of the run, ``r`` the run) to ONE roster entry (`re`): an entry naming a judge is satisfied by
    a complete, covering judgment of that judge; an entry with a judge selector by a covering judgment whose judge matches the selector;
    an entry naming neither by a judgment-less row of a judge-less instrument, or by a covering judgment of the judge of the instrument.
    The covering judgment is an identity-current one (:func:`judgment_identity_current_sql`): a complete judgment bound at
    claim to a since-corrected run identity covers nothing anywhere -- here as in `v_roster_match`, `v_pending_judgment` and the
    harvester's completion requirement -- so no entry is satisfied by a reading the frame has already demoted. `v_paper_long` filters
    its rows with it (through :func:`frame_admission_sql`) and derives the comparisons a measurement advertises from it;
    :data:`MEASUREMENT_JUDGE_SQL` is the rendering over the default aliases."""
    covers = judgment_covers_sql("jm") + " AND " + judgment_identity_current_sql("jm", r, i)
    return f"""CASE
                          WHEN re.judge_id IS NOT NULL THEN jm.judge_id = re.judge_id AND {covers}
                          WHEN re.judge_selector_json IS NOT NULL THEN j.judge_id IS NOT NULL AND {covers}
                               AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.judge_model') = j.judge_model)
                               AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                    OR json_extract(re.judge_selector_json, '$.rubric_name') = j.rubric_name)
                          ELSE (rm.judgment_id = '' AND {i}.judge_id IS NULL)
                               OR (jm.judge_id IS NOT NULL AND jm.judge_id IS {i}.judge_id AND {covers})
                        END"""


#: :func:`measurement_judge_sql` over the default aliases (`r` the run, `i` the instrument of the run).
MEASUREMENT_JUDGE_SQL = measurement_judge_sql()


def frame_admission_sql(msr: str = "msr", msc: str = "msc", r: str = "r", i: str = "i") -> str:
    """What a visible, observation-current `run_metric` row must ALSO pass to be a row of `v_paper_long` -- the frame's admission, in
    one text, so the frame and `v_paper_current`'s `coverage_state` (`measured` is a headline metric row IN THE FRAME) cannot
    disagree. Over the aliases of the frame -- `rm` the metric row, `jm` its judgment (LEFT-joined), `j` the judge of that judgment,
    ``i`` the instrument of the run, ``r`` the run, ``msr`` / ``msc`` the multi-source role and fold state of the row (LEFT-joined;
    one relation carrying both columns when the caller hoisted them): (1) the judgment is current -- live
    (:func:`non_current_judgment_sql`), not a partial reading displaced by a complete one of the same judge
    (:func:`judgment_displaced_by_complete_sql`), and bound at claim to the run's current identity
    (:func:`judgment_identity_current_sql`) -- the three rendered together by :func:`judgment_currency_sql`;
    (2) a secondary row whose identity the owner also carries is out -- folded (the owner row stands) or refused (neither is admitted);
    (3) a run the roster matches admits only a measurement whose judge satisfies some matched entry (:func:`measurement_judge_sql`); a
    run no entry matches admits every row on its judgment alone. Rendered as `AND` conjuncts, each under its comment, to follow a
    caller's own first predicate."""
    return (f"""{judgment_currency_sql("rm.judgment_id", "jm", r, i)}
  -- a secondary row whose identity the owner also carries leaves the frame: folded (the owner row stands) or refused (neither is admitted)
  AND ({msr}.role IS NULL OR {msr}.role <> 'secondary' OR {msc}.state IS NULL OR {msc}.state = 'stopgap')
  -- a run the roster matches admits only a measurement whose judge satisfies a matched entry; an unmatched run admits every row
  AND (NOT EXISTS (SELECT 1 FROM v_roster_match vm WHERE vm.run_id = {r}.run_id)
       OR EXISTS (SELECT 1 FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
                  WHERE vm.run_id = {r}.run_id AND {measurement_judge_sql(r, i)}))""")


#: The selection states under which a run is NOT current (migration 0021), and the lifecycle states likewise.
NON_CURRENT_SELECTION_STATES: Tuple[str, ...] = ("superseded", "withdrawn", "excluded")
NON_CURRENT_LIFECYCLE_STATES: Tuple[str, ...] = ("failed", "cancelled", "superseded", "withdrawn")
assert set(NON_CURRENT_LIFECYCLE_STATES) <= set(V.LIFECYCLE_STATES)


def _lits(values) -> str:
    return ", ".join(sql_literal(v) for v in values)


#: The POINTS -- (source, model_id, object_id, eval_id, view_legacy), the unit the sources pool by -- at which the source filed a POOLED
#: reading (migration 0022): one pass over `run` for the rows under a pooled pass label (:data:`vocab.POOLED_PASS_LABELS`,
#: rendered here so the SQL and the tuple cannot drift), in ANY lifecycle or selection state -- a pooled run a later pooled run
#: superseded, one a ruling excluded or one withdrawn still marks the point, because the members are never the paper reading once the
#: source filed a pooled one. The probe pairs `view_legacy` with IS (NULL pairs with NULL: a source filing no point string marks its own
#: unlabeled members and no other). MATERIALIZED, like every CTE of the chain: the member predicate (:func:`member_pass_sql`) probes it
#: once per candidate row instead of scanning `run` per row (about 530 pooled runs live, 2026-10-02, against the 12,383 rows of `run`).
POOLED_GROUP_CTE = f"""pooled_group AS MATERIALIZED (
    -- the points (source, model_id, object_id, eval_id, view_legacy) at which the source filed a pooled reading, in any state
    -- (migration 0022): a member pass is a member (member_pass) at one of these and an unpooled_member elsewhere; neither is a paper row
    SELECT DISTINCT lane_source_legacy, model_id, object_id, eval_id, view_legacy FROM run WHERE pass_label IN ({_lits(V.POOLED_PASS_LABELS)})
)"""


def member_label_sql(r: str = "r") -> str:
    """The per-(source, pass) half of the member predicate (migration 0021), rendered from
    :data:`vocab.MEMBER_PASS_LABELS_BY_SOURCE` so the SQL and the Python table cannot drift: one `(source = ... AND pass_label IN (...))`
    disjunct per listed source, in source order, over the run alias ``r``. A NULL `pass_label` never matches (IN is three-valued), so an
    unlabeled run of a listed source is a paper row by this rule."""
    parts = [f"({r}.lane_source_legacy = {sql_literal(src)} AND {r}.pass_label IN ({_lits(labels)}))"
             for src, labels in sorted(V.MEMBER_PASS_LABELS_BY_SOURCE.items())]
    return "(" + " OR ".join(parts) + ")"


def member_pass_sql(r: str = "r", pooled: str = "pooled_group") -> str:
    """The predicate that reads a run as a MEMBER of its source's pooled reading: the per-(source, pass) table
    (:func:`member_label_sql`, migration 0021) AND, since migration 0022, a pooled reading of the SAME source at the run's
    own POINT -- (model, object, eval, `view_legacy`, the last paired with IS) -- an EXISTS probe into ``pooled`` (:data:`POOLED_GROUP_CTE`,
    the points at which the source filed a run under a pooled pass label, in any state), correlated on the run alias ``r``. A labeled run
    whose point carries no pooled reading of its source reads `unpooled_member` instead (:func:`member_label_sql` alone, the next branch of
    `SELECTION_STATE_SQL`); neither state is a paper row. The rule lives in `v_run_current` alone: the registry's renderer
    (`paper.paper_long_frame`) honors `in_current_paper`, and `paper.paper_frame_exclusion`'s per-source keep-list (`sampled_lane`:
    `pooled_t0_t1` alone) stays the paper side's pinned contract, which the flag agrees with on every labeled run."""
    return (f"({member_label_sql(r)} AND EXISTS (SELECT 1 FROM {pooled} pg WHERE pg.lane_source_legacy = {r}.lane_source_legacy"
            f" AND pg.model_id = {r}.model_id AND pg.object_id = {r}.object_id AND pg.eval_id = {r}.eval_id AND pg.view_legacy IS {r}.view_legacy))")


#: the per-(source, pass) half as `v_run_current` renders it (run alias r): the `unpooled_member` branch and the paper flag's member branch
MEMBER_LABEL_SQL = member_label_sql()
#: the member predicate as `v_run_current` renders it (run alias r, the pooled points as the chain's `pooled_group` CTE)
MEMBER_PASS_SQL = member_pass_sql()


def coexist_across_views_sql(r: str = "r", r2: str = "r2") -> str:
    """The pairwise exception of the later-run supersession (Q11, adapter re-anchoring plan 10.6, migration 0026), rendered from
    :data:`vocab.COEXIST_ACROSS_VIEWS_BY_SOURCE` so the SQL and the table cannot drift: over the `later` CTE's two run aliases (``r`` the
    run, ``r2`` the later complete run at its key -- a key that pairs `pass_label` with IS and does not hold `lane_source_legacy` or
    `view_legacy`), TRUE unless the pair is a listed source's pooled pair whose `view_legacy` DIFFER -- one `(source on both sides, pass
    labels, differing views)` disjunct per listed source, in source order. The pass label is tested on ``r`` alone (the key makes ``r2``'s
    equal); the views are compared with IS NOT, so a NULL view pairs with NULL and an unlabeled pooled pair keeps today's rule. An empty
    table renders `1`: no exception."""
    parts = [f"({r}.lane_source_legacy = {sql_literal(src)} AND {r2}.lane_source_legacy = {sql_literal(src)} AND {r}.pass_label IN ({_lits(labels)})"
             f" AND {r2}.view_legacy IS NOT {r}.view_legacy)"
             for src, labels in sorted(V.COEXIST_ACROSS_VIEWS_BY_SOURCE.items())]
    return "NOT (" + " OR ".join(parts) + ")" if parts else "1"


#: the exception as `v_run_current`'s `later` CTE renders it (run aliases r and r2)
COEXIST_ACROSS_VIEWS_SQL = coexist_across_views_sql()


def source_local_anchor_sql(r: str = "r", l: str = "l") -> str:
    """The predicate that reads an object-`none` run as its source's OWN unsteered reading (migration 0027), rendered from
    :data:`vocab.SOURCE_LOCAL_ANCHOR_SOURCES` so the SQL and the list cannot drift: the run's object is `none` and its
    `lane_source_legacy` is a listed spelling (`IN`) or matches a listed pattern (`GLOB`, for a family an adapter mints from directory
    names), AND nothing retired it -- no later run at its key (``l``, the `later_runs` alias), no producer retirement (lifecycle
    superseded, superseded_by_run_id). The ruling branches precede the branch this predicate guards, so a decisive include / exclude /
    supersede still wins; the guard keeps the branch from resurrecting a `none` the later-run rule or its producer retired (every such
    run keeps reading superseded, is_current 0). Rendered once into each CASE of `v_run_current`."""
    exact = [s for s in V.SOURCE_LOCAL_ANCHOR_SOURCES if "*" not in s]
    patterns = [s for s in V.SOURCE_LOCAL_ANCHOR_SOURCES if "*" in s]
    terms = ([f"{r}.lane_source_legacy IN ({_lits(exact)})"] if exact else []) + [f"{r}.lane_source_legacy GLOB {sql_literal(p)}" for p in patterns]
    return (f"({r}.object_id = 'none' AND (" + " OR ".join(terms) + f") AND {l}.run_id IS NULL AND {r}.lifecycle_state <> 'superseded'"
            f" AND {r}.superseded_by_run_id IS NULL)")


#: the predicate as `v_run_current` renders it (run alias r, the later-run alias l)
SOURCE_LOCAL_ANCHOR_SQL = source_local_anchor_sql()

#: `selection_state` over the selection CTEs (run alias r, per_run_cover pc, per_run_match pm, decision d, withdrawn w, later_runs l,
#: and the pooled groups pooled_group behind the member predicate): indeterminate -> withdrawn -> an include ruling selects -> the
#: DECISIVE supersede / exclude ruling (migration 0027: BEFORE the roster branch, so a run a ruling retired reads `superseded` /
#: `excluded` whether or not the roster matches it -- until 0027 a roster-less run under a decisive ruling read `not_in_roster` beside
#: `decisive_action`, and `is_current` carried a belt for that pair; the label moves, `is_current` does not: both spellings were already
#: non-current) -> a source-local `none` (`source_local_anchor`, migration 0027: an object-none run of a source whose none
#: is its own standup's unsteered reading, :data:`vocab.SOURCE_LOCAL_ANCHOR_SOURCES`, rendered by :func:`source_local_anchor_sql`; guarded
#: by the later-run rule and the producer's retirement, so it relabels only a none that would otherwise read selected or not_in_roster;
#: current and anchor-eligible, never a paper row) -> no roster match -> superseded by the producer's own retirement or a later run
#: (lifecycle, superseded_by_run_id, a later run at the same key -- the `later` CTE's key, token cap / trial / pass included; these
#: keep their roster-first place: 0027 moved the RULING alone, and a roster-less run a later run superseded reads not_in_roster and
#: stays current, as before) -> a member pass of a pooled reading (`member_pass`: AFTER superseded and excluded, so those currency facts keep reading on a member
#: run that also carries one, and BEFORE the incomplete-lifecycle branch, so an unfinished member reads member_pass -- it is not a paper
#: row either way; since migration 0022 only where the source filed a pooled reading at the run's own point) -> a member pass whose point carries
#: no pooled reading of its source (`unpooled_member`: the same place, the label half alone) -> an incomplete lifecycle ->
#: selected. Rendered ONCE, in `v_run_current`.
#: migration 0028 (the never-a-result requirement for the front-door WORKED EXAMPLE qwen35_4b_demo): a run of a worked-example key is
#: never `selected` and never a paper row, whatever the roster or the rulings say.
#: Until this file an include ruling with no model term (the live full-suite BigCodeBench / BFCL appendix include) would have selected such
#: a run, because both CASEs test `include` before the roster branch. The predicate is the KEY SET below, not `model.paper_status`:
#: `reference` means "not one of the paper's models" (the pipeline-test models qwen35_35b_a3b / gemma4_31b and gemma4e2b_it among
#: them) and a reference run IS in the frame when a roster entry or a ruling puts it there -- the 35B's 26 full-suite BigCodeBench / BFCL
#: runs are in the appendix by that deliberate include, and a frame change for them is a ruling, never this object's -- while a
#: `worked_example` paper_status would be a new value of the CHECK on `model`, a table-rebuild migration; the key set is carried here
#: instead (the allowed fallback, 2026-10-08). Rendered into both CASEs of `v_run_current` on the run's own model_id, BEFORE the include branch;
#: in selection_state the branch yields to a DECISIVE exclude / supersede, so a retired worked-example run keeps reading `excluded` /
#: `superseded` by name (is_current 0, as for any run), and every other reads `not_in_roster` -- the label a roster-less run reads already
#: (no roster names the key), so is_current does not move. A new worked-example key is added here (configs/registry/models.yaml names it).
WORKED_EXAMPLE_MODEL_IDS: Tuple[str, ...] = ("qwen35_4b_demo", "qwen35_4b_demo_base")
WORKED_EXAMPLE_MODEL_SQL = f"r.model_id IN ({_lits(WORKED_EXAMPLE_MODEL_IDS)})"

SELECTION_STATE_SQL = f"""CASE
  WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 'indeterminate'
  WHEN w.run_id IS NOT NULL THEN 'withdrawn'
  WHEN {WORKED_EXAMPLE_MODEL_SQL} AND COALESCE(d.action, '') NOT IN ('exclude', 'supersede') THEN 'not_in_roster'
  WHEN d.action = 'include' THEN 'selected'
  WHEN d.action = 'supersede' THEN 'superseded'
  WHEN d.action = 'exclude' THEN 'excluded'
  WHEN {SOURCE_LOCAL_ANCHOR_SQL} THEN 'source_local_anchor'
  WHEN pm.run_id IS NULL THEN 'not_in_roster'
  WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL OR l.run_id IS NOT NULL THEN 'superseded'
  WHEN {MEMBER_PASS_SQL} THEN 'member_pass'
  WHEN {MEMBER_LABEL_SQL} THEN 'unpooled_member'
  WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 'not_in_roster'
  ELSE 'selected'
END"""
assert all(f"THEN '{state}'" in SELECTION_STATE_SQL for state in V.SELECTION_STATES)

#: `in_current_paper` over the same CTEs: incompleteness is tested BEFORE an include ruling (a selected run that has not finished is not
#: in the paper), an include ruling admits a member run (the ruling puts it in the paper, consistent with the precedence above),
#: a run under a member pass label is otherwise never a paper row -- `member_pass` and `unpooled_member` alike, so the flag reads
#: the label half alone (`MEMBER_LABEL_SQL`) and probes no pooled point: the paper's reading of a listed source's point is its pooled
#: run, the paper side's pinned renderer contract -- a source-local `none` (migration 0027: `SOURCE_LOCAL_ANCHOR_SQL`, right after
#: the include branch) is never a paper row either -- and hold_out never enters (a held-out row stays in the paper). Rendered ONCE,
#: in `v_run_current`.
IN_CURRENT_PAPER_SQL = f"""CASE
  WHEN COALESCE(pc.indeterminate, 0) = 1 THEN 0
  WHEN w.run_id IS NOT NULL THEN 0
  WHEN {WORKED_EXAMPLE_MODEL_SQL} THEN 0
  WHEN r.lifecycle_state NOT IN ('judged', 'ingested') THEN 0
  WHEN d.action = 'include' THEN 1
  WHEN {SOURCE_LOCAL_ANCHOR_SQL} THEN 0
  WHEN {MEMBER_LABEL_SQL} THEN 0
  WHEN pm.run_id IS NULL THEN 0
  WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL THEN 0
  WHEN d.action IN ('exclude', 'supersede') THEN 0
  WHEN l.run_id IS NOT NULL THEN 0
  ELSE 1
END"""


def selection_ctes_sql() -> str:
    """The CTE chain behind a run's selection -- the body of the WITH of `v_run_current` (migration 0021; before that, inside
    `v_paper_current`): the open round, the roster match, the active rulings flattened once, the runs each ruling covers (the match
    form and the pre-match-form scope kinds), the comparisons a comparison-scoped exclude strips, the DECISIVE ruling (the latest
    include / exclude / supersede), the later-run supersession (coexist-guarded; keyed on model, object, eval, instrument, judge,
    strength, the legacy operating point, token_cap_tokens, trial_index and pass_label; a listed source's pooled pair whose
    `view_legacy` differ is skipped, Q11 / migration 0026, :func:`coexist_across_views_sql`), the withdrawn runs, and the per-run aggregates
    (the rulings covering it, indeterminate, held_out and the hold-out ruling id; the comparisons and roster entries it keeps), and
    (migration 0022) the pooled points -- the (source, model, object, eval, view_legacy) points at which the source filed a pooled
    reading, which the member predicate of `SELECTION_STATE_SQL` probes (:data:`POOLED_GROUP_CTE`).
    Every CTE is MATERIALIZED on purpose: the chain is the expensive part of a selection read, and a consumer evaluates it once."""
    alt_pred = " AND ".join([_a_in('token_caps', 'r.token_cap_tokens'), _a_in('decode_sources', 'r.decode_source'), _a_in('thinking_modes', 'r.thinking_mode'),
                             _a_in('harnesses', 'r.harness'), _a_in('item_set_ids', 'r.item_set_id'), _a_in('n_shots', 'r.n_shot'),
                             _a_in('scoring_rules', 'r.scoring_rule'), _a_in('serving_paths', 'r.serving_path')])
    legacy_key_fields = [('model_id', 'r.model_id'), ('object_id', 'r.object_id'), ('eval_id', 'r.eval_id'), ('instrument_id', 'r.instrument_id'),
                         ('campaign', 'r.campaign'), ('operating_point_id_legacy', 'r.operating_point_id_legacy'), ('token_cap_tokens', 'r.token_cap_tokens')]
    legacy_key_pred = " AND ".join(f"(json_extract(ru.scope_json, '$.{k}') IS NULL OR json_extract(ru.scope_json, '$.{k}') = {col})" for k, col in legacy_key_fields)
    legacy_key_any = " OR ".join(f"json_extract(ru.scope_json, '$.{k}') IS NOT NULL" for k, _ in legacy_key_fields)
    return f"""open_round AS (
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
      AND (ms.has_alternatives IS NULL OR EXISTS (SELECT 1 FROM json_each(ru.scope_json, '$.match.instrument_alternatives') alt WHERE {alt_pred}))
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
       OR (ru.scope_kind = 'key' AND ({legacy_key_any}) AND {legacy_key_pred})
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
    -- and a supersession that held before the key grew still holds. The ONE pairwise exception (Q11, migration 0026; rendered from
    -- vocab.COEXIST_ACROSS_VIEWS_BY_SOURCE): two pooled runs of a listed source at one key whose view_legacy differ are the readings of two
    -- POINTS (the tags pool separately), never predecessor and successor -- the pair is skipped; a later same-tag pooled run still supersedes
    SELECT r.run_id
    FROM run r JOIN run r2 ON r2.model_id = r.model_id AND r2.object_id = r.object_id AND r2.eval_id = r.eval_id
        AND r2.instrument_id = r.instrument_id AND r2.judge_id IS r.judge_id AND r2.run_id <> r.run_id
        AND r2.strength_nominal IS r.strength_nominal AND r2.strength_coefficient IS r.strength_coefficient
        AND r2.operating_point_id_legacy IS r.operating_point_id_legacy
        AND r2.token_cap_tokens IS r.token_cap_tokens AND r2.trial_index IS r.trial_index AND r2.pass_label IS r.pass_label
        AND {COEXIST_ACROSS_VIEWS_SQL}
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
later_runs AS MATERIALIZED (SELECT DISTINCT run_id FROM later),
{POOLED_GROUP_CTE}"""


#: ONE owner metric row per measurement identity (run, judgment, metric, threshold) in `v_paper_long`. Two sources may read one
#: measurement onto the SAME run id by design (the r2 export and the stats readers of the same Petri / multi-turn / best-of-n stores
#: mint `identity.run_id(...)` alike): the run row folds to one, but `run_metric`'s key carries `computation_id`, so the export's
#: computed row and the reader's producer-filed blank-computation twin BOTH stand, and every consumer counting or pooling rows read the
#: measurement twice (781 of the 5,507 in-paper rows on 2026-09-20, values identical in every pair). The rank below names one row the
#: `owner` — the run's own source's row first (`observation` is the run-scale membership record: the source whose head revision carries
#: the key; `source_revision.priority` ranks two sources carrying one key: higher priority first, then the lexically smaller
#: source_id), then a row that names its computation before a blank one, then the higher priority, the smaller source id, the smaller
#: computation id — a total order — and every other row at the identity `secondary`, kept as evidence with in_current_paper 0.
#: A run whose only row at an identity is the blank twin keeps it as the owner: no measurement is dropped, only the double count.
#: The key columns read out of `observation_key` are CAST to the joined columns' types on purpose: a `json_extract` column has no
#: affinity, and SQLite builds an automatic index for a join term only when the two sides' affinities agree — untyped, every join
#: into these CTEs was a full scan per `run_metric` row (93 s on the live registry for the fold alone; 0.3 s typed).
METRIC_ROLE_CTES = """metric_source AS MATERIALIZED (
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
    """ + _WITHDRAWN_JOIN + """
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
)"""

#: `loop_share` names ONE detector per writer (`run_metric.extra_json.loop_share_measure_legacy`; the paper's published rule where a
#: writer states none) and the writer's mirror column of that detector carries its rate (paper.LOOP_SHARE_MIRROR_COLUMNS, spelled here
#: in SQL so the two cannot disagree: tests/test_registry_multi_source.py pins the frame's derivation against the SQL).
_LOOP_SHARE_MEASURE_SQL = ("CASE WHEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') IN ('classifier', 'loop_first512', 'rep_ratio_loop') "
                           "THEN json_extract(rm.extra_json, '$.loop_share_measure_legacy') ELSE 'rep_ratio_loop' END")

#: The paper views' DERIVED degeneration rates (migration 0010). The `run_metric.frac_*` mirror columns are filled only by the reader
#: that emits the metric row, so a degeneration reading filed by ANOTHER reader never reached the paper's `degenerate_share`: the union
#: reader's run-id-keyed batches, strength_mtdiff's own rows (224 rows, 2026-09-21), the export's cap-4,096 multi-turn runs whose
#: `union5` is MEASURED in `run_degeneration` under the same judgment while 180 of their metric rows read NULL. `degen_rate` is one row
#: per (run, judgment) of the MEASURED rates of `v_run_degeneration_current` at each mirror's own (measure, scope); a view joins it
#: twice -- the metric row's judgment (`dj`), then the judgment-less '' rows the union reader files (`dn`) -- and every exposed column is
#: COALESCE(the writer's mirror, the exact-judgment rate, the judgment-less rate), so a mirrored value still wins where a writer filed
#: one and the two never mix inside one value. The same precedence `paper._degeneration_for` applies in Python.
DEGENERATION_RATE_MEASURES: Tuple[Tuple[str, str], ...] = (
    ("at_token_cap", "all"), ("at_token_cap_finish_reason", "all"), ("classifier", "all"), ("contentless", "all"),
    ("rep_ratio_loop", "all"), ("zlib_collapse", "all"), ("union5", "all"), ("loop_or_at_cap", "all"),
    ("loop_first512", "first_512_tokens"), ("defended_degenerate", "defended_only"),
)
assert all(m in V.DEGENERATION_MEASURES and s in V.DEGENERATION_SCOPES for m, s in DEGENERATION_RATE_MEASURES)
#: `run_metric.frac_*` mirror column -> the derived measures that fill it where the writer left it NULL, in order: the mirror's own
#: measure; the legacy `frac_at_token_cap` carries the producer's finish-reason estimand under both of its spellings (r2_export's
#: `at_token_cap`, the declared readers' `at_token_cap_finish_reason`), never the canonical re-tokenized reading.
FRAC_MIRROR_DERIVATION: Dict[str, Tuple[str, ...]] = {
    "frac_at_token_cap": ("at_token_cap", "at_token_cap_finish_reason"),
    "frac_degenerate_classifier": ("classifier",), "frac_degenerate_contentless": ("contentless",),
    "frac_degenerate_rep_ratio_loop": ("rep_ratio_loop",), "frac_degenerate_zlib_collapse": ("zlib_collapse",),
    "frac_degenerate_union5": ("union5",), "frac_loop_first512": ("loop_first512",), "frac_defended_degenerate": ("defended_degenerate",),
}
assert set(FRAC_MIRROR_DERIVATION) == set(V.DEGENERATION_FRAC_COLUMNS)
assert all(m in {mm for mm, _ in DEGENERATION_RATE_MEASURES} for ms in FRAC_MIRROR_DERIVATION.values() for m in ms)

# ---- the 95 percent Wilson score interval, derived in v_paper_long where a metric row of a yes-or-no proportion carries k and n, no
# filed interval, and a value that IS k / n (migration 0019, 2026-10-01; the metric roster and the value
# guard beside it). The Wilson interval is the interval of a Bernoulli proportion, k successes of n trials, and
# a run_metric row's k and n do not always describe one. `over_refusal_graded` scores every item 0, 0.5 or 1 and files the score SUM as k
# beside the graded mean, so a value that equals k / n exactly -- a score sum that happens to be whole: 561 ones + 28 halves + 728 zeros
# = 575 of 1,317, one of 19 live rows -- says nothing about the per-item variance the interval is about: a row of 250 halves reads value
# 0.5 at k 125 and would read a Wilson bound of [0.44, 0.56] for observations with no item variation at all. The repository's own
# interval for that estimand is adapters.declared_common.label_mean_interval, from the {0, 0.5, 1} census,
# which the frame cannot derive from k and n. The registry carries no unit or estimand per metric (metric_id is a bare string), so the
# frame carries one: WILSON_PROPORTION_METRIC_IDS, the metric ids whose writers file k as a count of items with a yes-or-no outcome and n
# as the items judged, each read in its writer before it was listed. A metric off the roster keeps its NULL (what it read before 0019; a
# consumer that wants a bar there still computes its own). The value guard stays beside the roster as the SCALE check: a listed metric
# filed as a percent (`misaligned_rate` files value = 100 k / n), or a value filed beside the k and n of some other quantity
# (`recovered_vs_others_asr_gap` files a difference beside one component's counts), is not k / n and derives nothing either.
# The expressions are paperlib.stats.wilson / adapters.declared_common.wilson term for term -- p = k / n (REAL division), d = 1 + z^2 / n,
# center (p + z^2 / 2n) / d, half-width z * sqrt(p (1 - p) / n + z^2 / 4n^2) / d -- at the same z, so a bound derived here equals the bound
# an adapter files from the same k and n to the last bit (checked on every derivable live row, max abs difference 0.0). The Python
# returns (nan, nan) at n = 0; the SQL guard reads NULL there, the registry's spelling of "no interval". `sqrt` is SQLite's math function
# (SQLITE_ENABLE_MATH_FUNCTIONS, 3.35+): a build without it fails to compile the object with "no such function: sqrt" --
# tests/test_registry_paper_views_wilson_interval.py pins the requirement.
#: z of the 95 percent interval, spelled to the digit of paperlib.stats.Z95 (the adapters' default too)
WILSON_Z95 = "1.959963984540054"
#: the metric ids whose k is a count of items with a yes-or-no outcome and whose n is the items judged, so k / n is a Bernoulli proportion
#: and the Wilson score interval is its interval: the roster the frame derives for. Each name was read in its writer (2026-10-01): `asr`,
#: `asr_held_in`, `asr_held_out` and the `asr_at_n*` ladder (multiturn_grades, bon_native, r2_export: goals jailbroken of goals graded, or
#: hit within n draws); `asr_bon_cap4096` / `asr_bon_cap8192` (bon_stores, bon_native: any hit of n goals at the cap); `acc` (relu_native:
#: items correct of items scored; the other acc writers file their own interval); `xst_refused` / `orb_refused` (declared_common
#: over_refusal_refused_sibling and the r2 export: FULL refusals of the judged benign items); the BoN share rows (bon_stores: draws
#: flagged of draws -- `at_cap_share_bon_512`, `coherent_at_cap_share_bon_512`, `loop_share_bon_512`, `secondary_flag_only_share_bon_512`
#: -- and `residual_at_cap_share_bon_4096`, climbed draws still at the cap of climbed draws). NOT listed, by design: `over_refusal_graded`
#: and `over_refusal_label_mean` (k is a sum of {0, 0.5, 1} scores; their interval is declared_common.label_mean_interval),
#: `misaligned_rate` (value on the percent scale), `recovered_vs_others_asr_gap` (a difference). The object is versioned, so a name joins
#: the roster by a new migration, when its writer files k and n without an interval and the per-item outcome is yes or no.
WILSON_PROPORTION_METRIC_IDS: Tuple[str, ...] = (
    "asr", "asr_held_in", "asr_held_out",
    "asr_at_n1", "asr_at_n2", "asr_at_n4", "asr_at_n8", "asr_at_n16", "asr_at_n32", "asr_at_n64", "asr_at_n75",
    "asr_bon_cap4096", "asr_bon_cap8192",
    "acc",
    "xst_refused", "orb_refused",
    "at_cap_share_bon_512", "coherent_at_cap_share_bon_512", "loop_share_bon_512", "secondary_flag_only_share_bon_512",
    "residual_at_cap_share_bon_4096",
)
assert len(set(WILSON_PROPORTION_METRIC_IDS)) == len(WILSON_PROPORTION_METRIC_IDS)
assert not set(WILSON_PROPORTION_METRIC_IDS) & set(V.banned_metric_ids())
assert not {"over_refusal_graded", "over_refusal_label_mean", "misaligned_rate", "recovered_vs_others_asr_gap"} & set(WILSON_PROPORTION_METRIC_IDS)
#: the roster as the SQL membership test on the frame's metric row
WILSON_METRIC_GUARD = "rm.metric_id IN (" + ", ".join(f"'{m}'" for m in WILSON_PROPORTION_METRIC_IDS) + ")"
#: how far a value may sit from k / n and still BE k / n: writers compute p = k / n in Python, IEEE double division, bit-identical to
#: SQLite's k*1.0/n (6,631 of the 6,670 live rows with k, n > 0 and no interval match exactly, 2026-10-01 -- 19 of them the graded rows
#: the roster refuses -- and the other 39 are the percent, difference and graded rows above), and the nearest reading that is NOT k / n
#: sits half a count away, 0.5 / n (3.8e-4 at the largest live n, 1,317): the tolerance is far below the one and far above float noise,
#: so it admits a value computed by another route to the same proportion and refuses every other scale
WILSON_VALUE_TOLERANCE = "1e-9"
#: a row derives an interval when its metric is on the roster and it carries k, a positive n, and a value that is k / n (a metric off the
#: roster, a NULL k, n or value, n = 0, and a value on another scale or of another quantity all fail and read NULL)
WILSON_DERIVE_GUARD = f"{WILSON_METRIC_GUARD} AND rm.k IS NOT NULL AND rm.n > 0 AND abs(rm.value - rm.k*1.0/rm.n) <= {WILSON_VALUE_TOLERANCE}"
#: the derivation applies to a row with NO filed bound on either side (a row with one filed bound is returned as filed, both sides)
WILSON_DERIVES = f"rm.ci_lo IS NULL AND rm.ci_hi IS NULL AND {WILSON_DERIVE_GUARD}"


def wilson_bound_sql(sign: str) -> str:
    """The lower (``sign`` '-') or upper ('+') 95 percent Wilson bound of rm.k / rm.n: center -/+ half-width, as the Python."""
    z = WILSON_Z95
    p = "(rm.k*1.0/rm.n)"
    d = f"(1.0 + {z}*{z}/rm.n)"
    center = f"({p} + {z}*{z}/(2.0*rm.n))/{d}"
    half = f"{z}*sqrt({p}*(1.0 - rm.k*1.0/rm.n)/rm.n + {z}*{z}/(4.0*rm.n*rm.n))/{d}"
    return f"{center} {sign} {half}"


#: ci_lo / ci_hi of a v_paper_long row: the filed bound where the writer filed one, else the derived bound where the row derives, else NULL
WILSON_CI_LO_SQL = f"CASE WHEN rm.ci_lo IS NOT NULL THEN rm.ci_lo WHEN {WILSON_DERIVES} THEN {wilson_bound_sql('-')} ELSE NULL END"
WILSON_CI_HI_SQL = f"CASE WHEN rm.ci_hi IS NOT NULL THEN rm.ci_hi WHEN {WILSON_DERIVES} THEN {wilson_bound_sql('+')} ELSE NULL END"
#: ci_method of a v_paper_long row: `wilson_95` where the bounds above are DERIVED -- the label describes the bounds in the row, so a
#: spelling a writer filed WITHOUT bounds (`none filed (rate over the judged pairs)` on 12 live orb_refused / xst_refused rows whose
#: value is k / n) gives way to it there -- else the filed spelling as filed: a filed interval keeps its method, NULL included, and a
#: row that derives nothing keeps the writer's text (`none` on the gap metric, whose k / n is not its value)
WILSON_CI_METHOD_SQL = f"CASE WHEN {WILSON_DERIVES} THEN 'wilson_95' ELSE rm.ci_method END"

DEGENERATION_RATE_CTE = (
    "degen_rate AS MATERIALIZED (\n"
    "    -- the MEASURED rates of the current degeneration rows of every run at the frac_* mirrors own (measure, scope), one row per\n"
    "    -- (run_id, judgment_id); the paper objects fill a NULL mirror from here (migration 0010: a reading filed by another reader counts).\n"
    "    -- No apostrophe and no bare table name in a comment here: the reference scan strips quoted literals and reads identifiers\n"
    "    SELECT run_id, judgment_id,\n           "
    + ",\n           ".join(f"MAX(CASE WHEN measure = '{m}' AND scope = '{s}' THEN rate END) AS {m}" for m, s in DEGENERATION_RATE_MEASURES)
    + "\n    FROM v_run_degeneration_current WHERE measurement_state = 'measured' GROUP BY run_id, judgment_id\n)")
_DEGENERATION_RATE_JOIN = ("LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id\n"
                           "LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''")


def degeneration_rate_sql(*measures: str) -> str:
    """The derived-rate terms of ``measures`` for a COALESCE, in order: per measure the metric row's own judgment first, then the
    judgment-less rows (``paper._degeneration_for``: the '' rows with the row's own judgment's rows winning where both exist)."""
    return ", ".join(f"dj.{m}, dn.{m}" for m in measures)


def frac_mirror_sql(col: str) -> str:
    """A `run_metric.frac_*` mirror column as the paper views expose it since 0010: the writer's value, else the derived rate."""
    return f"COALESCE(rm.{col}, {degeneration_rate_sql(*FRAC_MIRROR_DERIVATION[col])})"


_LOOP_SHARE_VALUE_SQL = (f"CASE {_LOOP_SHARE_MEASURE_SQL} WHEN 'classifier' THEN {frac_mirror_sql('frac_degenerate_classifier')} "
                         f"WHEN 'loop_first512' THEN {frac_mirror_sql('frac_loop_first512')} ELSE {frac_mirror_sql('frac_degenerate_rep_ratio_loop')} END")
#: `degenerate_share` as the paper reads it (`paper.LONG_COLUMN_SOURCES`, in the same order): the writer's five-detector mirror, the run's
#: measured `union5`, the export's two-detector `loop_or_at_cap`, then the export's per-key column parked by name on every metric row
_DEGENERATE_SHARE_SQL = (f"COALESCE(rm.frac_degenerate_union5, {degeneration_rate_sql('union5', 'loop_or_at_cap')}, "
                         "json_extract(rm.extra_json, '$.degenerate_share'))")


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def multi_source_rows_sql() -> str:
    """``v_multi_source_rows``: every visible `run_metric` row inside a declared multi-source scope (ownership.MULTI_SOURCE_OWNERSHIP)
    — the declared model, object label, eval category (minus the eval ids the declaration excludes by name) and transfer kind,
    admitted by the declaration's strength rule (the served operating point, or unfitted) — with its ROLE (owner | secondary), its
    `pairing_key` (the declaration's pairing rule rendered: the producer run key both sources spell alike, or '' when the identity
    alone pairs) and the readings the fold compares and carries. BOTH sides are their source's OWN computation rows only
    (`computation_id <> ''`): the stats readers file a producer-filed blank-computation twin of every export row onto the export's
    run (the row `v_paper_long`'s metric role ranks second, a computed row before a blank one), and a twin in the scope would read as
    a second row at the identity on whichever side the export stands and refuse the measurement. The degeneracy readings are the
    writer's `frac_*` mirror columns and parked extras, each filled from the run's MEASURED `run_degeneration` rows where the writer
    left it NULL (migration 0010, `DEGENERATION_RATE_CTE`: a reading filed by another reader counts; a mirrored value still wins). The
    view therefore reads `run_degeneration` -- item-scale by key, run-grain in count -- which the run-level snapshot carries for it
    (`snapshot.RUNLEVEL_ITEM_TABLES`), so `v_paper_long` stays in the run-level file.

    Judgment currency is applied HERE, before ownership is aggregated (migration 0021; the complete-over-partial and identity rules
    beside it since the record-refusal-fact migration, :func:`judgment_currency_sql`): a row of a non-current judgment is in no scope, so a re-judged owner counts its ONE live row at the identity (not two,
    which would refuse the measurement) and an owner whose only judgment was superseded leaves the secondary standing as the stopgap
    (`v_paper_long` filters the frame by the same predicate, so without this the fold would still read folded and the frame would
    drop the secondary row as well -- a live measurement with no row at all)."""
    blocks = []
    for d in OW.declarations():
        blocks.append(f"""SELECT {_lit(d.declaration_id)} AS declaration_id, {_lit(d.owner_source)} AS owner_source, {_lit(d.secondary_source)} AS secondary_source,
       CASE WHEN r.lane_source_legacy = {_lit(d.owner_source)} THEN 'owner' ELSE 'secondary' END AS role,
       {d.pairing_key_sql()} AS pairing_key,
       r.lane_source_legacy, r.run_id, r.model_id, o.paper_label, e.eval_category, e.headline_metric_id, r.eval_id,
       r.transfer_kind, r.producer_run_key_legacy, r.view_legacy, r.operating_point_id_legacy, r.strength_nominal,
       rm.judgment_id, rm.metric_id, rm.threshold, rm.computation_id, rm.value, rm.k, rm.n,
       {_lit(d.check_rule)} AS check_rule,
       json_extract(rm.extra_json, '$.n_draws') AS n_draws,
       {frac_mirror_sql('frac_at_token_cap')} AS at_cap_share,
       {_DEGENERATE_SHARE_SQL} AS degenerate_share,
       {_LOOP_SHARE_MEASURE_SQL} AS loop_share_measure,
       {_LOOP_SHARE_VALUE_SQL} AS loop_share,
       rm.ingest_seq
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN run_metric rm ON rm.run_id = r.run_id
LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id
LEFT JOIN instrument i ON i.instrument_id = r.instrument_id
{_WITHDRAWN_JOIN}
{_DEGENERATION_RATE_JOIN}
WHERE wm.run_id IS NULL
  AND rm.computation_id <> ''
  -- a row of a non-current judgment is in no scope: ownership counts current rows only
  {judgment_currency_sql('rm.judgment_id', 'jm', 'r', 'i')}
  AND {d.scope_sql()}""")
    body = "\nUNION ALL\n".join(blocks) if blocks else "SELECT NULL AS declaration_id WHERE 0"
    return f"\nCREATE VIEW v_multi_source_rows AS\nWITH {WITHDRAWN_METRIC_CTE},\n{DEGENERATION_RATE_CTE}\n{body};\n"


def multi_source_check_sql() -> str:
    """``v_multi_source_check``: ONE row per matched identity (declaration × pairing key × eval × metric × threshold) that the
    SECONDARY source carries — `state` stopgap (no owner row), folded (k and n agree EXACTLY, null-safe; the owner row stands, the
    secondary's leaves `v_paper_long`, the declared carry readings travel where the owner has none) or refused (k / n differ, or
    either side carries two rows at one identity, or the owner already carries a DIFFERENT carry reading: neither value is admitted,
    fail-loud, never resolved by priority). The pairing key (ownership.MATCH_COLUMNS) is the declaration's pairing rule rendered per
    row — the producer run key both sources spell alike, so an owner row folds only onto the secondary row read from the same store,
    or '' when the identity alone pairs. Both sides' k / n, values, readings and producer run keys are columns so the disagreement is
    readable; `at_cap_share` is shown for both and NEVER carried (two statistics under one paper column)."""
    agg = """SELECT declaration_id, pairing_key, eval_id, metric_id, threshold, COUNT(*) AS n_rows,
           MIN(owner_source) AS owner_source, MIN(secondary_source) AS secondary_source, MIN(model_id) AS model_id, MIN(paper_label) AS paper_label,
           MIN(eval_category) AS eval_category, MIN(transfer_kind) AS transfer_kind, MIN(producer_run_key_legacy) AS producer_run_key_legacy,
           MIN(run_id) AS run_id, MIN(judgment_id) AS judgment_id, MIN(computation_id) AS computation_id,
           MIN(check_rule) AS check_rule, MIN(n_draws) AS n_draws,
           MIN(value) AS value, MIN(k) AS k, MIN(n) AS n, MIN(degenerate_share) AS degenerate_share, MIN(loop_share) AS loop_share,
           MIN(loop_share_measure) AS loop_share_measure, MIN(at_cap_share) AS at_cap_share, MAX(ingest_seq) AS ingest_seq
    FROM scoped WHERE role = {role} GROUP BY declaration_id, pairing_key, eval_id, metric_id, threshold"""
    txt = lambda col: f"COALESCE(CAST({col} AS TEXT), 'null')"  # noqa: E731
    counts_differ = "(NOT (ow.k IS s.k) OR NOT (ow.n IS s.n))"
    # check_rule value_with_draws (migration 0026; ownership.CHECK_RULES): the values agree to 1e-9, the secondary's n equals the owner's
    # pooled draws (`extra.n_draws`) and the secondary carries a count -- the predicate for a pooled owner (k NULL, n counting goals)
    # against a per-draw secondary; an owner row without n_draws never folds
    draws_differ = "(ow.value IS NULL OR s.value IS NULL OR ABS(ow.value - s.value) > 1e-9 OR ow.n_draws IS NULL OR NOT (s.n IS ow.n_draws) OR s.k IS NULL)"
    carry_conflict = {c: f"(ow.{c} IS NOT NULL AND s.{c} IS NOT NULL AND ABS(ow.{c} - s.{c}) > 1e-9)" for c in OW.CARRY_COLUMNS}
    carry_travels = {c: f"(ow.{c} IS NULL AND s.{c} IS NOT NULL)" for c in OW.CARRY_COLUMNS}
    conflict_when = "\n".join(f"     WHEN {carry_conflict[c]} THEN 'refused'" for c in OW.CARRY_COLUMNS)
    state = f"""CASE WHEN ow.run_id IS NULL THEN 'stopgap'
     WHEN ow.n_rows > 1 OR s.n_rows > 1 THEN 'refused'
     WHEN s.check_rule = 'counts' AND {counts_differ} THEN 'refused'
     WHEN s.check_rule = 'value_with_draws' AND {draws_differ} THEN 'refused'
{conflict_when}
     ELSE 'folded' END"""
    carried_list = " || ".join(f"CASE WHEN {carry_travels[c]} THEN '\"{c}\",' ELSE '' END" for c in OW.CARRY_COLUMNS)
    carried_json = f"CASE WHEN {state} = 'folded' THEN '[' || TRIM({carried_list}, ',') || ']' ELSE '[]' END"
    carried_text = f"COALESCE(NULLIF(REPLACE(TRIM({carried_list}, ','), '\"', ''), ''), 'nothing')"
    conflict_reason = "\n".join(
        f"     WHEN {carry_conflict[c]} THEN 'the owner (' || s.owner_source || ') already carries {c}=' || {txt('ow.' + c)} || ' and the secondary ('"
        f" || s.secondary_source || ') reads ' || {txt('s.' + c)} || '; a disagreement is never overwritten and neither value is admitted'"
        for c in OW.CARRY_COLUMNS)
    reason = f"""CASE WHEN ow.run_id IS NULL THEN 'the owner (' || s.owner_source || ') carries no row for this measurement yet; the secondary (' || s.secondary_source || ') row stands as a stopgap until it does, and is then dropped once k and n agree exactly'
     WHEN ow.n_rows > 1 THEN 'the owner (' || s.owner_source || ') carries ' || ow.n_rows || ' rows at one identity; one owner row per measurement is the precondition of the fold, so neither value is admitted'
     WHEN s.n_rows > 1 THEN 'the secondary (' || s.secondary_source || ') carries ' || s.n_rows || ' rows at one identity; one row per measurement is the precondition of the fold, so neither value is admitted'
     WHEN s.check_rule = 'counts' AND {counts_differ} THEN 'the two sources disagree on k / n (owner k=' || {txt('ow.k')} || ' n=' || {txt('ow.n')} || '; secondary k=' || {txt('s.k')} || ' n=' || {txt('s.n')} || '); two readings of one measurement that differ are a disagreement to rule on, never averaged and never picked from, so neither value is admitted'
     WHEN s.check_rule = 'value_with_draws' AND {draws_differ} THEN 'the two readings disagree under value_with_draws (owner value=' || {txt('ow.value')} || ' n_draws=' || {txt('ow.n_draws')} || '; secondary value=' || {txt('s.value')} || ' k=' || {txt('s.k')} || ' n=' || {txt('s.n')} || '): the values must agree to 1e-9, the secondary''s n must equal the owner''s pooled draws and the secondary must carry a count; two readings of one measurement that differ are a disagreement to rule on, never averaged and never picked from, so neither value is admitted'
{conflict_reason}
     ELSE CASE WHEN s.check_rule = 'value_with_draws'
               THEN 'the value agrees to 1e-9 and the secondary''s n equals the owner''s pooled draws (value=' || {txt('ow.value')} || ', n_draws=' || {txt('ow.n_draws')} || '); the owner (' || s.owner_source || ') row stands and the secondary (' || s.secondary_source || ') row is dropped; carried onto the owner where it had none: ' || {carried_text}
               ELSE 'k and n agree exactly (k=' || {txt('ow.k')} || ', n=' || {txt('ow.n')} || '); the owner (' || s.owner_source || ') row stands and the secondary (' || s.secondary_source || ') row is dropped; carried onto the owner where it had none: ' || {carried_text} END END"""
    return f"""
CREATE VIEW v_multi_source_check AS
WITH scoped AS MATERIALIZED (SELECT * FROM v_multi_source_rows),
owner AS MATERIALIZED (
    {agg.format(role="'owner'")}
),
secondary AS MATERIALIZED (
    {agg.format(role="'secondary'")}
)
SELECT s.declaration_id, s.owner_source, s.secondary_source, s.model_id, s.paper_label, s.eval_category, s.transfer_kind,
       s.pairing_key, s.eval_id, s.metric_id, s.threshold,
       s.check_rule, ow.n_draws AS n_draws_owner,
       {state} AS state,
       {reason} AS reason,
       ow.run_id AS owner_run_id, ow.judgment_id AS owner_judgment_id, ow.computation_id AS owner_computation_id, COALESCE(ow.n_rows, 0) AS owner_rows,
       ow.producer_run_key_legacy AS owner_producer_run_key_legacy,
       s.run_id AS secondary_run_id, s.judgment_id AS secondary_judgment_id, s.computation_id AS secondary_computation_id, s.n_rows AS secondary_rows,
       s.producer_run_key_legacy AS secondary_producer_run_key_legacy,
       ow.value AS value_owner, s.value AS value_secondary,
       ow.k AS k_owner, ow.n AS n_owner, s.k AS k_secondary, s.n AS n_secondary,
       ow.degenerate_share AS degenerate_share_owner, s.degenerate_share AS degenerate_share_secondary,
       ow.loop_share AS loop_share_owner, s.loop_share AS loop_share_secondary,
       ow.loop_share_measure AS loop_share_measure_owner, s.loop_share_measure AS loop_share_measure_secondary,
       ow.at_cap_share AS at_cap_share_owner, s.at_cap_share AS at_cap_share_secondary,
       {carried_json} AS carried_json,
       MAX(COALESCE(ow.ingest_seq, 0), s.ingest_seq) AS ingest_seq
FROM secondary s
LEFT JOIN owner ow ON ow.declaration_id = s.declaration_id AND ow.pairing_key = s.pairing_key AND ow.eval_id = s.eval_id
     AND ow.metric_id = s.metric_id AND ow.threshold = s.threshold;
"""


def render_views() -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Every view as ``(name, CREATE VIEW sql, referenced tables-and-views)`` in definition order. The third
    element is DECLARED beside the SQL (a view may reference another view); ``VIEW_TABLES`` closes it over the
    base tables so a consumer (the run-level snapshot) can tell which views live without the item-scale tables.
    ``tests/test_registry_schema.py`` asserts each declaration equals the identifiers found in the SQL."""
    views: List[Tuple[str, str, Tuple[str, ...]]] = []

    views.append(("v_run_wide", """
CREATE VIEW v_run_wide AS
SELECT r.*,
       -- 1 once the run is judged or ingested (a complete run reads `ingested` at rest, `judged` before the harvester stamps it)
       """ + is_complete_sql("r") + """,
       m.family, m.checkpoint_kind, m.hf_repo, m.hf_revision, m.paper_status AS model_paper_status,
       o.object_kind, o.paper_label, o.object_role, o.fit_target_kind, o.fit_model_id, o.fit_positions,
       o.fit_layers_json, o.fit_layer_count, o.fit_rank, o.fit_seed, o.fit_seed_confidence, o.fit_replicate_label,
       o.status AS object_status, o.artifact_sha256 AS object_artifact_sha256,
       fm.checkpoint_kind AS fit_model_checkpoint_kind,
       e.eval_category, e.headline_metric_id, e.metric_direction, e.paper_status AS eval_paper_status,
       s.dataset_version, s.n_items AS item_set_n_items, s.subsample_rule,
       i.harness AS instrument_harness, i.harness_version AS instrument_harness_version,
       i.n_shot AS instrument_n_shot, i.cot_prompting AS instrument_cot_prompting,
       i.scoring_rule AS instrument_scoring_rule, i.token_cap_tokens AS instrument_token_cap_tokens,
       i.padding_tokens AS instrument_padding_tokens, i.padding_identity_sha256 AS instrument_padding_identity_sha256,
       i.decode_source AS instrument_decode_source, i.decode_temperature AS instrument_decode_temperature,
       i.decode_top_p AS instrument_decode_top_p, i.decode_top_k AS instrument_decode_top_k,
       i.thinking_mode AS instrument_thinking_mode, i.serving_path AS instrument_serving_path,
       i.applied_positions_mode AS instrument_applied_positions_mode, i.judge_id AS instrument_judge_id,
       j.judge_model, j.rubric_name, j.rubric_version, j.threshold_rule
FROM run r
JOIN model m ON m.model_id = r.model_id
JOIN object o ON o.object_id = r.object_id
LEFT JOIN model fm ON fm.model_id = o.fit_model_id
JOIN eval e ON e.eval_id = r.eval_id
JOIN item_set s ON s.item_set_id = r.item_set_id
JOIN instrument i ON i.instrument_id = r.instrument_id
LEFT JOIN judge j ON j.judge_id = r.judge_id;
""", ('run', 'model', 'object', 'eval', 'item_set', 'instrument', 'judge')))

    views.append(("v_producer_key", """
CREATE VIEW v_producer_key AS
SELECT producer_run_key_legacy, producer_object_key_legacy, run_id, campaign, eval_id, instrument_id, lifecycle_state,
       """ + is_complete_sql() + """
FROM run;
""", ('run',)))

    views.append(("v_eval_variants", """
CREATE VIEW v_eval_variants AS
SELECT r.eval_id, r.item_set_id, r.harness, r.n_shot, r.cot_prompting, r.thinking_mode, r.scoring_rule,
       r.token_cap_tokens, r.decode_source, r.instrument_id, COUNT(*) AS n_runs,
       MIN(r.queued_utc) AS first_queued_utc, MAX(r.queued_utc) AS last_queued_utc
FROM run r
GROUP BY r.eval_id, r.item_set_id, r.harness, r.n_shot, r.cot_prompting, r.thinking_mode, r.scoring_rule,
         r.token_cap_tokens, r.decode_source, r.instrument_id;
""", ('run',)))

    views.append(("v_in_flight", """
CREATE VIEW v_in_flight AS
SELECT r.run_id, r.campaign, r.lifecycle_state, r.substrate, r.cluster_job_id, r.job_name, r.node_hostname,
       r.modal_call_id, r.model_id, r.object_id, r.eval_id, r.instrument_id, r.queued_utc, r.started_utc, r.generated_utc,
       (SELECT MAX(event_utc) FROM run_state_event ev WHERE ev.run_id = r.run_id) AS last_event_utc,
       (SELECT state FROM run_state_event ev WHERE ev.run_id = r.run_id ORDER BY event_utc DESC LIMIT 1) AS last_event_state,
       (SELECT scheduler_state FROM run_state_event ev WHERE ev.run_id = r.run_id AND ev.scheduler_state IS NOT NULL
        ORDER BY event_utc DESC LIMIT 1) AS last_scheduler_state,
       CAST((julianday('now') - julianday(COALESCE(
           (SELECT MAX(event_utc) FROM run_state_event ev WHERE ev.run_id = r.run_id), r.queued_utc))) * 86400 AS INTEGER) AS age_s
FROM run r
WHERE r.lifecycle_state IN ('queued', 'running', 'generated', 'judging');
""", ('run', 'run_state_event')))

    views.append(("v_superseded", """
CREATE VIEW v_superseded AS
SELECT 'run' AS kind, r.run_id AS id, r.superseded_by_run_id AS superseded_by, r.status_ruling_id AS ruling_id,
       ru.action AS ruling_action, ru.reason AS ruling_reason, r.lifecycle_state AS state, r.campaign, r.model_id,
       r.object_id, r.eval_id, r.instrument_id
FROM run r LEFT JOIN ruling ru ON ru.ruling_id = r.status_ruling_id
WHERE r.superseded_by_run_id IS NOT NULL OR r.lifecycle_state = 'superseded'
UNION ALL
SELECT 'object', o.object_id, o.superseded_by_object_id, o.status_ruling_id, ru.action, ru.reason, o.status, o.campaign,
       o.fit_model_id, o.object_id, NULL, NULL
FROM object o LEFT JOIN ruling ru ON ru.ruling_id = o.status_ruling_id
WHERE o.superseded_by_object_id IS NOT NULL OR o.status = 'superseded';
""", ('run', 'ruling', 'object')))

    views.append(("v_instrument_mismatch", """
CREATE VIEW v_instrument_mismatch AS
SELECT r.run_id, r.campaign, r.model_id, r.eval_id, r.instrument_id,
       r.decode_temperature_sent, i.decode_temperature AS declared_temperature,
       r.decode_top_p_sent, i.decode_top_p AS declared_top_p,
       r.decode_top_k_sent, i.decode_top_k AS declared_top_k,
       r.token_cap_tokens, i.token_cap_tokens AS declared_token_cap_tokens,
       r.serving_path, i.serving_path AS declared_serving_path,
       r.applied_positions_mode, i.applied_positions_mode AS declared_applied_positions_mode,
       r.thinking_mode, i.thinking_mode AS declared_thinking_mode,
       r.instrument_match,
       TRIM(
         CASE WHEN r.decode_temperature_sent IS NOT i.decode_temperature THEN 'temperature ' ELSE '' END ||
         CASE WHEN r.decode_top_p_sent IS NOT i.decode_top_p THEN 'top_p ' ELSE '' END ||
         CASE WHEN r.decode_top_k_sent IS NOT i.decode_top_k THEN 'top_k ' ELSE '' END ||
         CASE WHEN r.token_cap_tokens IS NOT i.token_cap_tokens THEN 'token_cap ' ELSE '' END ||
         CASE WHEN r.serving_path IS NOT i.serving_path THEN 'serving_path ' ELSE '' END ||
         CASE WHEN r.applied_positions_mode IS NOT i.applied_positions_mode THEN 'applied_positions_mode ' ELSE '' END ||
         CASE WHEN r.thinking_mode IS NOT i.thinking_mode THEN 'thinking_mode ' ELSE '' END) AS differing_fields
FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id
WHERE r.decode_temperature_sent IS NOT i.decode_temperature
   OR r.decode_top_p_sent IS NOT i.decode_top_p
   OR r.decode_top_k_sent IS NOT i.decode_top_k
   OR r.token_cap_tokens IS NOT i.token_cap_tokens
   OR r.serving_path IS NOT i.serving_path
   OR r.applied_positions_mode IS NOT i.applied_positions_mode
   OR r.thinking_mode IS NOT i.thinking_mode
   OR r.instrument_match = 'mismatch';
""", ('run', 'instrument')))

    # decode policy match (2026-09-18): where an exact match between the decode parameters recorded and the policy can be verified,
    # the policy column is inserted into the row. One row per run: the ONE decode_policy row whose scope admits the run and whose
    # pinned parameters the run sent exactly. DERIVED, deterministic, rebuild-safe; decode_match.stamp_run_rows copies policy_id /
    # match onto the run row at the end of every ingest cycle and of a rebuild.
    #   scope: model_id equal; eval_id equal, or the row is category-scoped and the run's eval carries that category; thinking_mode
    #          equal to the run's recorded value (a run files `not_available` where it has no thinking switch; a row pinning `off` /
    #          `on` does not admit it).
    #   parameters: the sent set = decode_sent_json's numeric keys among the six the policy models + the three typed *_sent columns,
    #          compared as REAL (1 == 1.0). Every parameter the policy pins must be sent at the same value -- or be listed under
    #          params_not_sent_json AND not sent (deliberately omitted, resolved server-side); every parameter the run sent must be
    #          pinned by the policy at that value; a sent key outside the six (or non-numeric) matches nothing. seed_rule: the
    #          policy's (unless `unverified`) must equal run.decode_seed_rule. max_new_tokens_rule is prose, never compared.
    #   windows: a superseded row is a candidate only for runs generated before its successor's effective_from_utc (an `unverified`
    #          successor date bounds nothing). Several candidates: the ONE in force at generated_utc (effective_from_utc <= generated,
    #          successor's effective_from_utc > generated) wins; otherwise `ambiguous` with every candidate in candidates_json.
    #   verdict: unrecorded (no sent parameter at all) | no_match | exact | ambiguous. in_scope_json lists the rows whose scope admits
    #          the run, so a no_match names what it was compared with.
    #   recorded by source (migration 0021): a run with NO sent parameter (and no unmodelled key) whose decode_source is
    #          generation_config is matched against the in-scope rows whose decode_source is generation_config AND whose every pinned
    #          parameter (temperature, top_p, top_k, any other non-null pinned column) is declared under params_not_sent_json -- the
    #          sampler the served generation_config resolved, which the run could not have sent. The row must be ELIGIBLE exactly as
    #          an ordinary candidate is (found in review, 2026-10-02): its seed_rule must equal the run's (or be `unverified`), and a
    #          superseded row counts only for a run generated before its successor's effective_from_utc -- a row the policy history
    #          had already replaced when the run was generated, or one whose seed rule the run contradicts, is not the sampler the
    #          run was served under, and stamping it `exact` would be a false certainty. One such row -> exact (policy_id stamped),
    #          several -> ambiguous (candidates_json lists them; the stamp pass's eval-over-category tie-break applies as to every
    #          ambiguous verdict), none -> unrecorded as before. A run with a sent set, or another decode_source, never enters it.
    _params = "('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty')"
    _gen = "COALESCE(r.generated_utc, r.started_utc, r.queued_utc)"
    views.append(("v_run_decode_policy_match", f"""
CREATE VIEW v_run_decode_policy_match AS
WITH run_sent AS (
  SELECT r.run_id, j.key AS param, CAST(j.value AS REAL) AS value
  FROM run r, json_each(r.decode_sent_json) j
  WHERE j.key IN {_params} AND j.type IN ('integer', 'real')
  UNION
  SELECT run_id, 'temperature', CAST(decode_temperature_sent AS REAL) FROM run WHERE decode_temperature_sent IS NOT NULL
  UNION
  SELECT run_id, 'top_p', CAST(decode_top_p_sent AS REAL) FROM run WHERE decode_top_p_sent IS NOT NULL
  UNION
  SELECT run_id, 'top_k', CAST(decode_top_k_sent AS REAL) FROM run WHERE decode_top_k_sent IS NOT NULL
),
run_sent_unmodelled AS (
  SELECT DISTINCT r.run_id FROM run r, json_each(r.decode_sent_json) j
  WHERE j.key NOT IN {_params} OR j.type NOT IN ('integer', 'real')
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
         {_gen} AS generated, p.effective_from_utc, p.superseded_by_policy_id,
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
""", ('run', 'eval', 'decode_policy')))

    # the item tables read as ONE current row per measurement key; every scientific consumer reads these
    for t in ITEM_SCALE_TABLES:
        refs = (t, 'source_revision') + (('judgment', 'run', 'instrument') if item_judgment_live(t, 'ir') else ())   # Item_result reads the judgment, its run and the run's instrument
        views.append((f"v_{t}_current", current_rows_sql(t), refs))
    # the paper's status contract: one row per (paper source string, adapter) over the head read of every source the adapter
    # has, with the LATEST revision of each source folded in — a refused re-read shows as ok = 0 and an `extra_json.refuse` entry
    # instead of hiding behind the accepted head it leaves current; an unaccepted marker a later accepted read superseded
    # (`superseded_by_revision_id` set by the harvester's fold rule) is not the latest read; a source refused before any accepted read appears through the
    # refused revision's own `extra_json.lane_sources_legacy`. Several sources under one paper source string: scalar columns are
    # reduced (MIN ok, SUM measured, uris unioned, details joined) and `extra_json` keeps each source's facts under `by_source`
    # beside the concatenated `refuse` / `torn` / `missing` lists — no key of one source silently overwrites another's.
    views.append(("v_source_status", """
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
""", ('v_source_status_current', 'source_revision')))
    views.append(("v_item_result_provenances", """
CREATE VIEW v_item_result_provenances AS
SELECT ir.*, sr.source_id, COALESCE(sr.priority, 0) AS priority
FROM item_result ir LEFT JOIN source_revision sr ON sr.revision_id = ir.source_revision_id
WHERE """ + item_visible("ir") + item_not_withdrawn("item_result", "ir") + item_judgment_live("item_result", "ir") + """;
""", ('item_result', 'source_revision', 'judgment', 'run', 'instrument')))
    views.append(("v_item_conflict", """
CREATE VIEW v_item_conflict AS
SELECT run_id, item_id, metric_id, trial_index, sample_index, turn_index, judgment_id,
       COUNT(*) AS n_provenances,
       COUNT(DISTINCT COALESCE(CAST(value AS TEXT), '') || '|' || COALESCE(label, '')) AS n_distinct_values,
       MIN(value) AS value_min, MAX(value) AS value_max,
       json_group_array(DISTINCT source_revision_id) AS revision_ids_json
FROM v_item_result_provenances
GROUP BY run_id, item_id, metric_id, trial_index, sample_index, turn_index, judgment_id
HAVING n_distinct_values > 1;
""", ('v_item_result_provenances',)))

    views.append(("v_item_headline", """
CREATE VIEW v_item_headline AS
WITH """ + HEADLINE_ACCEPTED_CTE + """
-- ROW-SCOPED on purpose: a per-run choice computed in its own CTE is not scoped by a consumer filter run_id = ? and probes every run in
-- the registry (10,007 runs, 8 M item rows on 2026-09-21: reading one run took minutes); the NOT EXISTS below costs nothing for the
-- most preferred id (no more-preferred candidate) and one indexed probe per row for a lower-preferred one
SELECT ir.run_id, ir.item_id, ir.trial_index, ir.sample_index, ir.turn_index, ir.judgment_id,
       ir.metric_id, ir.value AS score, ir.label, ir.api_pool, ir.created_utc, ir.judged_utc,
       f.at_token_cap, f.finish_reason, f.response_tokens, f.response_chars,
       f.degenerate_classifier, f.degenerate_contentless, f.degenerate_rep_ratio_loop, f.degenerate_zlib_collapse,
       f.degenerate_union5, f.loop_first512, f.rep_ratio, f.zlib_ratio, f.alnum_count, f.tail_loop, f.line_loop,
       f.decode_seed, f.generation_artifact_id, f.generation_offset, f.rationale_artifact_id, f.rationale_offset,
       ir.source_revision_id, ir.ingest_seq
FROM v_item_result_current ir
JOIN run r ON r.run_id = ir.run_id
JOIN accepted a ON a.eval_id = r.eval_id AND a.metric_id = ir.metric_id
LEFT JOIN v_item_flags_current f ON f.run_id = ir.run_id AND f.item_id = ir.item_id AND f.trial_index = ir.trial_index
     AND f.sample_index = ir.sample_index AND f.turn_index = ir.turn_index AND f.judgment_id = ir.judgment_id
WHERE NOT EXISTS (
    -- the headline of a run is the MOST preferred accepted id it files item results under (paper.run_headline_metric_id in SQL): a row
    -- under a less preferred id stands only where no more preferred one has current item results in the run
    SELECT 1 FROM accepted a2
    WHERE a2.eval_id = r.eval_id AND a2.preference < a.preference
      AND EXISTS (SELECT 1 FROM v_item_result_current ir2 WHERE ir2.run_id = r.run_id AND ir2.metric_id = a2.metric_id));
""", ('v_item_result_current', 'run', 'eval', 'v_item_flags_current')))

    views.append(("v_judge_side_by_side", """
CREATE VIEW v_judge_side_by_side AS
SELECT ir.run_id, ir.item_id, ir.trial_index, ir.sample_index, ir.turn_index, ir.metric_id,
       jm.judge_id, jd.judge_model, ir.judgment_id, jm.status AS judgment_status, ir.value AS score, ir.label,
       ir.api_pool, ir.judged_utc
FROM v_item_result_current ir
JOIN judgment jm ON jm.judgment_id = ir.judgment_id
JOIN judge jd ON jd.judge_id = jm.judge_id
WHERE ir.judgment_id <> '';
""", ('v_item_result_current', 'judgment', 'judge')))

    views.append(("v_item_pointer", """
CREATE VIEW v_item_pointer AS
SELECT f.run_id, f.item_id, f.trial_index, f.sample_index, f.turn_index, f.judgment_id,
       ga.uri_primary AS generation_uri, f.generation_offset,
       ra.uri_primary AS rationale_uri, f.rationale_offset,
       rf.file_uri AS record_file_uri
FROM item_flags f
LEFT JOIN artifact ga ON ga.artifact_id = f.generation_artifact_id
LEFT JOIN artifact ra ON ra.artifact_id = f.rationale_artifact_id
JOIN record_ingest rf ON rf.record_file_id = f.record_file_id
WHERE """ + item_visible("f") + """;
""", ('item_flags', 'artifact', 'record_ingest', 'source_revision')))


    views.append(("v_seed_spread", """
CREATE VIEW v_seed_spread AS
SELECT r.model_id, o.paper_label, r.eval_id, r.instrument_id, rm.metric_id, rm.threshold, rm.computation_id,
       COUNT(DISTINCT o.fit_replicate_label) AS n_replicates, COUNT(*) AS n_runs,
       MIN(rm.value) AS min_value, MAX(rm.value) AS max_value, AVG(rm.value) AS mean_value,
       json_group_array(json_object('run_id', r.run_id, 'fit_replicate_label', o.fit_replicate_label,
                                    'fit_seed', o.fit_seed, 'value', rm.value)) AS values_json
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN run_metric rm ON rm.run_id = r.run_id
WHERE r.lifecycle_state IN ('judged', 'ingested')
GROUP BY r.model_id, o.paper_label, r.eval_id, r.instrument_id, rm.metric_id, rm.threshold, rm.computation_id;
""", ('run', 'object', 'run_metric')))

    # degeneration wide: one column triple per (measure, scope)
    cols = []
    for m in V.DEGENERATION_MEASURES:
        for s in V.DEGENERATION_SCOPES:
            tag = f"{m}__{s}"
            cols.append(f"MAX(CASE WHEN measure = '{m}' AND scope = '{s}' THEN rate END) AS {tag}_rate")
            cols.append(f"MAX(CASE WHEN measure = '{m}' AND scope = '{s}' THEN numerator END) AS {tag}_num")
            cols.append(f"MAX(CASE WHEN measure = '{m}' AND scope = '{s}' THEN denominator END) AS {tag}_den")
            cols.append(f"MAX(CASE WHEN measure = '{m}' AND scope = '{s}' THEN denominator_unit END) AS {tag}_unit")
            cols.append(f"MAX(CASE WHEN measure = '{m}' AND scope = '{s}' THEN measurement_state END) AS {tag}_state")
    views.append(("v_run_degeneration_wide",
                  "\nCREATE VIEW v_run_degeneration_wide AS\nSELECT run_id, judgment_id,\n       "
                  + ",\n       ".join(cols) + "\nFROM v_run_degeneration_current\nGROUP BY run_id, judgment_id;\n",
                  ("v_run_degeneration_current",)))

    views.append(("v_object_provenance", """
CREATE VIEW v_object_provenance AS
SELECT o.*,
       fm.family AS fit_model_family, fm.checkpoint_kind AS fit_model_checkpoint_kind, fm.hf_revision AS fit_model_revision,
       fs.method AS search_method, fs.n_trials AS search_n_trials, fs.n_trials_completed AS search_n_trials_completed,
       fs.search_space_json, fs.argmin_interior AS search_argmin_interior,
       fs.quadratic_fit_positive_definite AS search_quadratic_fit_positive_definite,
       fs.confirmation_within_noise AS search_confirmation_within_noise, fs.winner_params_json,
       (SELECT COUNT(*) FROM json_each(o.field_confidence_json) fc
         WHERE json_extract(fc.value, '$.confidence') = 'unverified') AS n_unverified_fields,
       (SELECT json_group_array(fc.key) FROM json_each(o.field_confidence_json) fc
         WHERE json_extract(fc.value, '$.confidence') = 'unverified') AS unverified_fields_json,
       p.fit_positions AS parent_fit_positions, p.fit_layers_json AS parent_fit_layers_json
FROM object o
LEFT JOIN model fm ON fm.model_id = o.fit_model_id
LEFT JOIN fit_search fs ON fs.search_id = o.fit_search_id
LEFT JOIN object p ON o.derivation <> 'none' AND p.object_id = json_extract(o.parent_object_ids_json, '$[0]');
""", ('object', 'model', 'fit_search')))

    # pending judgment: run x judge; the requested judge is a WHERE filter on required_judge_id
    # (default: the instrument's declared judge, is_instrument_judge = 1 — completion is judged by the
    # instrument's judge, so is eligibility). The judge join is scoped by the INSTRUMENT: an instrument that names a
    # judge pairs its runs with every judge (the requested one is the filter; a second judge is eligible before it has
    # filed anything); an instrument WITHOUT a judge (rule-scored — its headline metric is filed at generation and the
    # harvester ingests the run from it) pairs its runs only with judges that have already filed a judgment for the run
    # (a claim, a second-judge read), so a rule-scored run with no judgment is pending for nobody — the bare cross join
    # listed every such run once per judge in the table, awaiting each (2026-09-20). A run is PENDING for judge J iff
    # its lifecycle state admits judging
    # (generated / judging / judged / ingested — a judged or ingested run is eligible for a SECOND judge) and no
    # judgment by J COVERS it — `judgment_covers_sql`, the SAME predicate v_roster_match / v_paper_long and the
    # harvester's completion check read, so a run the paper build counts as measured is never pending here and
    # a run pending here never satisfies a roster entry: required_coverage = the items generated (planned when
    # generation has not reported), capped by the declared item set's size; judged + declined items reach it.
    # Every other judgment row is EXPLICIT in the row rather than silently folded away: an identity-current in-progress row by J
    # is an active claim (claim_session / claim_utc / claim_age_s; pending_reason = claimed), a partial row leaves
    # items_missing, abandoned rows are counted (n_abandoned / last_abandoned_utc), and pending_reason names the
    # first that applies: claimed > partial > coverage_short (complete by J but short of the coverage) > identity_stale (the only
    # judgments by J -- a claim, a partial or a complete reading -- were bound at claim to a since-corrected run identity) >
    # superseded (a re-judge displaced the judgment by J, so a fresh grade is wanted -- read BEFORE any released claim,
    # migration 0021) > abandoned
    # > awaiting_second_judge (only other judges have judged it) > no_judgment.
    # The judgments that COVER, that leave items_missing, that read `partial` / `coverage_short` and that read `claimed` are the
    # identity-current ones (`judgment_identity_current_sql`): a complete
    # or partial judgment whose binding at claim differs from the run's current instrument or required judge is non-current on the
    # paper side, so it covers nothing here either -- else a same-judge identity correction (X to Y, both requiring J) left the run with
    # neither a paper row nor a pending job. An in-progress claim whose binding differs holds nothing either: it can only ever complete
    # into a stale-bound judgment, so presenting it as `claimed` showed a job that could not produce a current reading and that no other
    # worker could take over (`pending.claim` refused the other session) -- such a run reads
    # `identity_stale` with no claim columns, and `pending.claim` closes the stale claim and binds a new one to the current identity for
    # the same session and another alike. A NULL binding is current, so every legacy judgment counts as before; the second-judge design
    # is untouched. A SUPERSEDED run (`superseded_by_run_id` set; `run_superseded_sql`) is never pending -- its successor is (the registry
    # ruling of 2026-10-04, migration 0025): without the exclusion an identity correction landing on a retired run whose complete judgment
    # had been claimed under the old identity listed the retired run as `identity_stale` beside its successor, claimable again.
    views.append(("v_pending_judgment", """
CREATE VIEW v_pending_judgment AS
WITH cov AS (
    SELECT r.run_id, """ + required_coverage_sql("r") + """ AS required_coverage FROM run r
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
       (SELECT MAX(COALESCE(cov.required_coverage, 0) - (""" + judgment_covered_items_sql("j4") + """), 0)
          FROM judgment j4 WHERE j4.run_id = r.run_id AND j4.judge_id = jd.judge_id AND j4.status IN ('partial', 'complete')
          AND """ + judgment_identity_current_sql("j4", "r", "i") + """) AS items_missing,
       (SELECT COUNT(*) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id AND j5.status = 'abandoned') AS n_abandoned,
       (SELECT MAX(j5.claimed_utc) FROM judgment j5 WHERE j5.run_id = r.run_id AND j5.judge_id = jd.judge_id
          AND j5.status = 'abandoned') AS last_abandoned_utc,
       (SELECT COUNT(*) FROM judgment j6 WHERE j6.run_id = r.run_id AND j6.judge_id <> jd.judge_id
          AND j6.status = 'complete') AS n_complete_other_judges,
       CASE WHEN cl.judgment_id IS NOT NULL THEN 'claimed'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'partial' AND """ + judgment_identity_current_sql("j7", "r", "i") + """) THEN 'partial'
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status = 'complete' AND """ + judgment_identity_current_sql("j7", "r", "i") + """) THEN 'coverage_short'
            -- the only judgments by J (a claim, a partial or a complete reading) were bound at claim to a since-corrected run identity:
            -- a fresh grade is wanted, and a stale-bound claim holds nothing (pending.claim closes it and binds a new one)
            WHEN EXISTS (SELECT 1 FROM judgment j7 WHERE j7.run_id = r.run_id AND j7.judge_id = jd.judge_id
                           AND j7.status IN ('in_progress', 'partial', 'complete')
                           AND NOT """ + judgment_identity_current_sql("j7", "r", "i") + """) THEN 'identity_stale'
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
          AND j3.status = 'in_progress' AND """ + judgment_identity_current_sql("j3", "r", "i") + """
        ORDER BY j3.claimed_utc DESC LIMIT 1)
WHERE r.lifecycle_state IN ('generated', 'judging', 'judged', 'ingested')
  -- a superseded run is never pending; its successor is (the registry decision of 2026-10-04, migration 0025): an identity correction
  -- landing on a retired run whose complete judgment was claimed under the old identity must not list it as identity_stale beside
  -- its successor (no apostrophe and no bare table name in this comment: the reference scan strips quoted literals and reads identifiers)
  AND NOT (""" + run_superseded_sql("r") + """)
  -- no identity-current judgment by J covers it: a covering judgment bound at claim to a since-corrected identity covers nothing
  AND NOT EXISTS (
        SELECT 1 FROM judgment jc
        WHERE jc.run_id = r.run_id AND jc.judge_id = jd.judge_id AND """ + judgment_covers_sql("jc", "r") + """
          AND """ + judgment_identity_current_sql("jc", "r", "i") + """);
""", ('run', 'item_set', 'instrument', 'eval', 'judge', 'judgment', 'run_metric')))

    # paper: the exact (run, roster entry) relation of the open round — v_paper_current and v_roster_gaps both read it
    # ---- paper: roster matching. A roster entry names its object by id or by SELECTOR (paper_label / object_kind /
    # object_role / transfer_kind(s) / fit_model_id / object_id_prefix), its instrument by id or selector, its judge by id
    # or selector, and its strength rule (operating_point | fixed:<x> | ladder:<record>). Every present selector field
    # constrains; an absent field does not. An instrument SELECTOR never matches a PADDED instrument (migration
    # 0013): padding is instrument identity since 0012 (`instrument.padding_tokens`), and no selector field carries
    # it, so a padded instrument satisfied every selector field of the unpadded static entries of its eval and the 24
    # padded runs were pooled into the ordinary static comparisons beside their own named one (live, 2026-09-23) — the
    # confound the pad-as-instrument decision exists to prevent. A padded instrument is reached only by NAME
    # (`re.instrument_id = r.instrument_id`, the branch roster v6's figP_padding entries use); the selector branch
    # requires `i.padding_tokens IS NULL`, so an unpadded instrument matches selector entries exactly as before.
    obj_sel = "re.object_selector_json"
    ins_sel = "re.instrument_selector_json"
    views.append(("v_roster_match", f"""
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
               AND {_sel_eq(obj_sel, 'paper_label', 'o.paper_label')}
               AND {_sel_eq(obj_sel, 'object_kind', 'o.object_kind')}
               AND {_sel_eq(obj_sel, 'object_role', 'o.object_role')}
               AND {_sel_eq(obj_sel, 'fit_model_id', 'o.fit_model_id')}
               AND {_sel_eq(obj_sel, 'transfer_kind', 'r.transfer_kind')}
               AND {_sel_in(obj_sel, 'transfer_kinds', 'r.transfer_kind')}
               AND (json_extract({obj_sel}, '$.object_id_prefix') IS NULL
                    OR r.object_id GLOB json_extract({obj_sel}, '$.object_id_prefix') || '*')))
      AND (re.instrument_id = r.instrument_id
           -- a selector never matches a PADDED instrument (0013): padding is instrument identity (0012) that no selector
           -- field carries, so a padded instrument satisfies the fields of the unpadded entries and would pool with them; it is
           -- reached only by name, through the branch above (no apostrophe in this comment: the reference scan strips quoted literals)
           OR (re.instrument_id IS NULL
               AND i.padding_tokens IS NULL
               AND {_sel_eq(ins_sel, 'harness', 'i.harness')}
               AND {_sel_eq(ins_sel, 'item_set_id', 'i.item_set_id')}
               AND {_sel_eq(ins_sel, 'thinking_mode', 'i.thinking_mode')}
               AND {_sel_eq(ins_sel, 'serving_path', 'i.serving_path')}
               AND {_sel_eq(ins_sel, 'decode_source', 'i.decode_source')}
               AND {_sel_eq(ins_sel, 'scoring_rule', 'i.scoring_rule')}
               AND {_sel_eq(ins_sel, 'n_shot', 'i.n_shot')}
               AND {_sel_in(ins_sel, 'token_cap_tokens', 'i.token_cap_tokens')}))
      -- the entry is satisfied only by a COMPLETE judgment of its judge covering the items of the run; an entry that
      -- names no judge reads the judge of the instrument (a rule-scored instrument: a generation-time metric row); the covering
      -- judgment is identity-current: one bound at claim to a since-corrected run identity satisfies no entry
      AND ((re.judge_id IS NOT NULL
            AND EXISTS (SELECT 1 FROM judgment jx WHERE jx.run_id = r.run_id AND jx.judge_id = re.judge_id AND """ + judgment_covers_sql("jx") + """
                        AND """ + judgment_identity_current_sql("jx", "r", "i") + """))
           OR (re.judge_selector_json IS NOT NULL
               AND EXISTS (SELECT 1 FROM judgment jx JOIN judge jj ON jj.judge_id = jx.judge_id WHERE jx.run_id = r.run_id
                           AND """ + judgment_covers_sql("jx") + """
                           AND """ + judgment_identity_current_sql("jx", "r", "i") + """
                           AND (json_extract(re.judge_selector_json, '$.judge_model') IS NULL
                                OR json_extract(re.judge_selector_json, '$.judge_model') = jj.judge_model)
                           AND (json_extract(re.judge_selector_json, '$.rubric_name') IS NULL
                                OR json_extract(re.judge_selector_json, '$.rubric_name') = jj.rubric_name)))
           OR (re.judge_id IS NULL AND re.judge_selector_json IS NULL
               AND ((i.judge_id IS NULL AND EXISTS (SELECT 1 FROM run_metric rm0 WHERE rm0.run_id = r.run_id AND rm0.judgment_id = ''))
                    OR (i.judge_id IS NOT NULL
                        AND EXISTS (SELECT 1 FROM judgment jx WHERE jx.run_id = r.run_id AND jx.judge_id = i.judge_id AND """ + judgment_covers_sql("jx") + """
                                    AND """ + judgment_identity_current_sql("jx", "r", "i") + """)))))
      -- operating_point: a filed stamp of 1, else the fallback decided 2026-09-22 (operating_point_basis_sql: a fallback kind at nominal
      -- 1.0, the all-positions vector at c = 0.3, an unsteered or prompted kind); a filed stamp of 0 is never at the point
      AND ((re.strength_rule = 'operating_point' AND """ + operating_point_sql('r', 'o') + """)
           OR (re.strength_rule GLOB 'fixed:*' AND r.strength_nominal = CAST(substr(re.strength_rule, 7) AS REAL))
           OR (re.strength_rule GLOB 'ladder:*' AND r.strength_nominal IS NOT NULL));
""", ('paper_round', 'roster_entry', 'run', 'object', 'instrument', 'judge', 'judgment', 'run_metric', 'item_set')))

    # ---- paper: the derived flag. A ruling covers a run through `scope_json.match` (lists of ids / labels / kinds,
    # every present list constrains, at least one must be present — a ruling that names nothing covers nothing) or, for
    # rulings written before the match form, through the scope_kind's own key. Among the rulings covering a run, the
    # LATEST ruled include / exclude / supersede decides (a tie goes against include); hold_out never removes a row from
    # the paper — it marks `held_out` so a mean can leave it out while a panel still draws it.
    # Since migration 0021 the ruling fold lives in ONE object, `v_run_current` (selection_ctes_sql: the CTE chain;
    # SELECTION_STATE_SQL / IN_CURRENT_PAPER_SQL: the two CASEs), which also derives `is_current` / `why`; `v_paper_current` is a
    # projection of it beside the run's display columns, and `v_paper_long` reads the currency of every contrast anchor from it.
    views.append(("v_run_current", f"""
CREATE VIEW v_run_current AS
WITH {selection_ctes_sql()}
SELECT run_id,
       -- is_current (migration 0021): 1 when nothing below retires the run -- a selection_state of superseded / withdrawn / excluded, a
       -- hold_out mark, a lifecycle of failed / cancelled / superseded / withdrawn, or a superseded_by_run_id. A decisive exclude /
       -- supersede ruling retires the run whatever its roster membership: since migration 0027 the ruling's branch precedes
       -- the roster branch in selection_state, so such a run reads excluded / superseded by name and the 0021 belt for the
       -- (not_in_roster, decisive_action) pair is no longer needed; the paper reads the rows of a current run only where the frame says
       -- so (hold_out keeps a row in the paper and still reads is_current 0; a member pass of a pooled reading, selection_state
       -- member_pass, is current and not a paper row, and so is a member pass whose own point carries no pooled reading of its
       -- source, selection_state unpooled_member, and so is a source-local none, selection_state source_local_anchor;
       -- since migration 0028 a run of a worked-example key -- WORKED_EXAMPLE_MODEL_IDS, the front door's qwen35_4b_demo -- reads
       -- not_in_roster before any include ruling can select it and is never a paper row, unless a decisive exclude / supersede retired
       -- it by name; a run of a `reference` checkpoint is in the frame by roster or ruling exactly as before)
       CASE WHEN selection_state IN ({_lits(NON_CURRENT_SELECTION_STATES)}) OR held_out = 1
                 OR lifecycle_state IN ({_lits(NON_CURRENT_LIFECYCLE_STATES)}) OR superseded_by_run_id IS NOT NULL THEN 0 ELSE 1 END AS is_current,
       -- the FIRST failing rule, in the order above, with the ruling id where a ruling decided it; NULL on a current run
       CASE WHEN selection_state = 'superseded'
                 THEN 'selection_state superseded' || CASE WHEN decisive_action = 'supersede' THEN ' (ruling ' || decisive_ruling_id || ')'
                                                           WHEN lifecycle_state = 'superseded' OR superseded_by_run_id IS NOT NULL THEN ' (lifecycle or superseded_by_run_id)'
                                                           ELSE ' (a later run at the same key)' END
            WHEN selection_state = 'withdrawn' THEN 'selection_state withdrawn'
            WHEN selection_state = 'excluded' THEN 'selection_state excluded (ruling ' || decisive_ruling_id || ')'
            WHEN held_out = 1 THEN 'held_out (ruling ' || held_out_ruling_id || ')'
            WHEN lifecycle_state IN ({_lits(NON_CURRENT_LIFECYCLE_STATES)}) THEN 'lifecycle_state ' || lifecycle_state
            WHEN superseded_by_run_id IS NOT NULL THEN 'superseded_by_run_id ' || superseded_by_run_id
            END AS why,
       decisive_ruling_id, decisive_action, selection_state, in_current_paper, held_out, round_id,
       comparison_ids_json, roster_entry_ids_json, ruling_ids_json
FROM (
    SELECT r.run_id, r.lifecycle_state, r.superseded_by_run_id,
           d.ruling_id AS decisive_ruling_id,
           d.action AS decisive_action,
           COALESCE(pc.held_out, 0) AS held_out,
           pc.held_out_ruling_id,
           {SELECTION_STATE_SQL.replace(chr(10), chr(10) + "           ")} AS selection_state,
           {IN_CURRENT_PAPER_SQL.replace(chr(10), chr(10) + "           ")} AS in_current_paper,
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
""", ('paper_round', 'run', 'object', 'eval', 'ruling', 'observation', 'v_roster_match')))

    # the paper flag beside the run's display columns: a projection of v_run_current (the fold is evaluated there, once) plus the
    # coverage of the run
    _nofit = "json_extract(r.extra_json, '$.nofit_certificate.kind') = 'nofit_by_design' AND r.n_items_generated = 0"
    _cert = lambda k: f"json_extract(r.extra_json, '$.nofit_certificate.{k}')"  # noqa: E731
    views.append(("v_paper_current", f"""
CREATE VIEW v_paper_current AS
WITH {HEADLINE_ACCEPTED_CTE},
{WITHDRAWN_METRIC_CTE},
{MULTI_SOURCE_STATE_CTE}
SELECT r.run_id, r.campaign, r.model_id, r.object_id, r.eval_id, r.instrument_id, r.judge_id, r.lifecycle_state,
       -- 1 once the run is judged or ingested (a complete run reads `ingested` at rest, `judged` before the harvester stamps it)
       {is_complete_sql('r')},
       r.strength_nominal, r.is_operating_point,
       -- how the run counts as being at the operating point (migration 0011; operating_point_basis_sql)
       {operating_point_basis_sql('r', 'o')} AS operating_point_basis,
       r.ingest_seq,
       -- the decode of the run beside its selection (migration 0009): the source, the derived match, the row resolve() returns today for
       -- the key of the run, and whether the matched row is that default (decode_family_default_sql)
       r.decode_source, r.decode_policy_id, r.decode_policy_match,
       {decode_family_default_sql('r')},
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
       CASE WHEN {_nofit} THEN 'undefined_by_geometry'
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
                           {frame_admission_sql(msr='ms', msc='ms')}) THEN 'measured'
            ELSE 'absent' END AS coverage_state,
       -- the certificate behind undefined_by_geometry: the windows tried, the prompt-token range, the minimum completion, the slots proven
       -- not to fit and the slot count; NULL on every other row
       CASE WHEN {_nofit}
            THEN json_object('windows', json({_cert('windows')}), 'prompt_tokens_min', {_cert('prompt_tokens_min')},
                             'prompt_tokens_max', {_cert('prompt_tokens_max')}, 'min_completion', {_cert('min_completion')},
                             'nofit_proven', {_cert('by_kind.nofit_proven')}, 'n_slots', {_cert('n_slots')}) END AS coverage_detail_json
FROM run r
JOIN object o ON o.object_id = r.object_id
JOIN v_run_current vc ON vc.run_id = r.run_id;
""", ('run', 'object', 'eval', 'instrument', 'item_set', 'run_metric', 'judgment', 'judge', 'observation', 'decode_policy', 'roster_entry',
      'v_roster_match', 'v_multi_source_rows', 'v_multi_source_check', 'v_run_current')))

    views.append(("v_roster_gaps", """
CREATE VIEW v_roster_gaps AS
WITH open_round AS (
    SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1
)
SELECT re.roster_entry_id, re.round_id, re.comparison_id, re.model_id, re.object_id, re.object_selector_json, re.eval_id,
       re.instrument_id, re.instrument_selector_json, re.judge_id, re.strength_rule, re.source_citation, re.inferred, re.drawn,
       (SELECT d.state FROM disposition d WHERE d.roster_entry_id = re.roster_entry_id ORDER BY d.ruled_utc DESC LIMIT 1) AS disposition_state,
       (SELECT d.reason FROM disposition d WHERE d.roster_entry_id = re.roster_entry_id ORDER BY d.ruled_utc DESC LIMIT 1) AS disposition_reason
FROM roster_entry re JOIN open_round orr ON orr.round_id = re.round_id
-- the roster version (migration 0016): only the entries of the version the paper_round row of the round carries are gaps; an entry a later
-- version re-shaped (its older id left behind) is neither matched nor listed here (no apostrophe here: the reference scan)
JOIN paper_round pr ON pr.round_id = re.round_id AND (pr.roster_version IS NULL OR re.roster_version = pr.roster_version)
WHERE NOT EXISTS (
        SELECT 1 FROM v_roster_match m JOIN v_paper_current pc ON pc.run_id = m.run_id
        WHERE m.roster_entry_id = re.roster_entry_id AND pc.in_current_paper = 1)
  AND NOT EXISTS (SELECT 1 FROM disposition d WHERE d.roster_entry_id = re.roster_entry_id);
""", ('paper_round', 'roster_entry', 'disposition', 'v_roster_match', 'v_paper_current')))

    # the judge predicate that admits ONE measurement (rm / jm / j of the row) to ONE roster entry (re): used both to
    # filter the rows (frame_admission_sql) and to derive the comparisons a measurement advertises
    measurement_judge = MEASUREMENT_JUDGE_SQL
    # ---- multi-source ownership (ownership.MULTI_SOURCE_OWNERSHIP): one measurement read by two sources resolves to ONE
    # row of v_paper_long — the declared owner's — after k / n agree exactly; a disagreement is refused by measurement (fail-loud,
    # value / k / n / ci NULL with the reason), never resolved by priority or recency; the secondary's row stands as a stopgap while
    # the owner has none. Both views read `run_degeneration` through v_run_degeneration_current since 0010 (the derived degeneration
    # rates beside the writers' mirrors); the run-level snapshot carries that table so v_paper_long stays in the run-level file.
    views.append(("v_multi_source_rows", multi_source_rows_sql(), ('run', 'object', 'eval', 'run_metric', 'judgment', 'instrument', 'observation', 'v_run_degeneration_current')))
    views.append(("v_multi_source_check", multi_source_check_sql(), ('v_multi_source_rows',)))

    # a REFUSED owner row keeps its identity and loses its numbers: neither source's reading is admitted (`expr` reads the column
    # when it is not the bare rm.<col> -- the interval columns, derived where none was filed, migration 0019)
    refused_null = lambda col, expr=None: f"CASE WHEN msr.role = 'owner' AND msc.state = 'refused' THEN NULL ELSE {expr or f'rm.{col}'} END AS {col}"  # noqa: E731
    # ---- metric role (METRIC_ROLE_CTES): one owner metric row per (run, judgment, metric, threshold); the other provenances of the same
    # measurement stay in the frame as secondary rows with in_current_paper 0 and their selection_state untouched, so a consumer that
    # counts or pools in-paper rows reads each measurement once and the evidence of the double read is still there to see
    # ---- contrasts (WITHDRAWN_CONTRAST_CTE / CURRENT_CONTRAST_CTE, migration 0017): a metric row joins its CURRENT contrasts only -- the
    # (run, anchor) pairs some source's head accepted revision still carries, or that no source ever observed. A reader that re-anchors a
    # run in a later revision leaves the older run_contrast row in the table (its observation withdrawn, as a metric row's would be), and
    # before 0017 the join read run_contrast whole, so the metric row appeared once per anchor (376 keys of 194 runs live, 2026-09-26).
    # Two CURRENT contrasts of different kinds at one key (the export's half-dose pair) are both joined, by design. Since 0021 a
    # contrast whose ANCHOR is not current (v_run_current.is_current 0, read through PAPER_CURRENT_CTE so the fold runs once) is
    # not joined either, and `n_current_contrasts` counts what is; a row of a superseded or abandoned judgment leaves the frame.
    views.append(("v_paper_long", """
CREATE VIEW v_paper_long AS
WITH """ + PAPER_CURRENT_CTE + """,
""" + WITHDRAWN_METRIC_CTE + """,
""" + DEGENERATION_RATE_CTE + """,
""" + METRIC_ROLE_CTES + """,
""" + WITHDRAWN_CONTRAST_CTE + """,
""" + CURRENT_CONTRAST_CTE + """
SELECT pc.round_id,
       -- the flag of the run on its OWNER metric row only: a secondary row at the same identity is evidence, never a second paper row
       CASE WHEN mr.rk = 1 THEN pc.in_current_paper ELSE 0 END AS in_current_paper,
       pc.selection_state, pc.held_out,
       -- per MEASUREMENT: the comparisons / roster entries the judge of this row satisfies, among the entries the run
       -- still carries (a comparison-scoped exclude removed the others from the run in v_paper_current, never the row from the paper)
       (SELECT json_group_array(DISTINCT vm.comparison_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND """ + measurement_judge + """) AS comparison_ids_json,
       (SELECT json_group_array(DISTINCT vm.roster_entry_id) FROM v_roster_match vm JOIN roster_entry re ON re.roster_entry_id = vm.roster_entry_id
         WHERE vm.run_id = r.run_id
           AND EXISTS (SELECT 1 FROM json_each(pc.roster_entry_ids_json) kept WHERE kept.value = vm.roster_entry_id)
           AND """ + measurement_judge + """) AS roster_entry_ids_json,
       pc.decisive_ruling_id, pc.decisive_action,
       r.run_id, r.campaign, r.lane_source_legacy, r.view_legacy, r.campaign_cell_id_legacy, r.operating_point_id_legacy,
       r.producer_run_key_legacy, r.producer_object_key_legacy,
       r.model_id, m.family AS model_family, r.transfer_kind, r.strength_nominal, r.strength_coefficient, r.is_operating_point,
       -- how the run counts as being at the operating point (migration 0011; operating_point_basis_sql)
       """ + operating_point_basis_sql("r", "o") + """ AS operating_point_basis,
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
       """ + decode_family_default_sql("r") + """,
       jm.judge_id, j.judge_model, j.rubric_name AS judge_rubric_name, r.judge_id AS run_judge_id,
       jm.n_judge_refusals AS judgment_n_judge_refusals,
       r.pass_label, r.lifecycle_state, """ + is_complete_sql("r") + """, r.store_uri, r.intervention_hash, r.notes, r.extra_json AS run_extra_json,
       COALESCE(json_extract(rm.extra_json, '$.eval_id_legacy'), json_extract(r.extra_json, '$.eval_id_legacy')) AS eval_id_legacy,
       rm.judgment_id, rm.metric_id, rm.producer_metric_spelling, rm.threshold, rm.computation_id,
       """ + ", ".join(refused_null(c) for c in ("value", "k", "n")) + """,
       -- the interval the writer filed, else the 95 percent Wilson score interval of k over n derived here (migration 0019;
       -- WILSON_CI_LO_SQL / WILSON_CI_HI_SQL / WILSON_CI_METHOD_SQL): a filed bound is returned as filed, byte for byte, under its filed
       -- method; a row of a metric on the yes-or-no roster (WILSON_PROPORTION_METRIC_IDS: k counts items with a yes-or-no outcome, so k
       -- over n is a Bernoulli proportion and the Wilson interval is its interval) with k, n above zero, no filed bound and a value that
       -- IS k over n (the scale check: a percent-scale value or a difference beside some other count is not) reads the derived bounds
       -- under the method wilson_95; a metric off the roster (a graded mean whose k is a sum of 0, 0.5 and 1 scores, whatever its value),
       -- k alone, n alone, n = 0 or a value that is not k over n read the filed columns, NULL included
       -- (no apostrophe in this comment: the reference scan strips quoted literals)
       """ + refused_null("ci_lo", WILSON_CI_LO_SQL) + """,
       """ + refused_null("ci_hi", WILSON_CI_HI_SQL) + """,
       """ + WILSON_CI_METHOD_SQL + """ AS ci_method,
       -- the frac_* mirror the writer filed, else the MEASURED degeneration rate of the run at the measure and scope of that mirror
       -- (migration 0010: the exact-judgment row first, then the judgment-less rows); a mirrored value still wins where a writer filed one
       """ + ", ".join(f"{frac_mirror_sql(c)} AS {c}" for c in V.DEGENERATION_FRAC_COLUMNS) + """,
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
""" + _WITHDRAWN_JOIN + """
""" + _DEGENERATION_RATE_JOIN + """
LEFT JOIN v_multi_source_rows msr ON msr.run_id = rm.run_id AND msr.judgment_id = rm.judgment_id AND msr.metric_id = rm.metric_id
     AND msr.threshold = rm.threshold AND msr.computation_id = rm.computation_id
LEFT JOIN v_multi_source_check msc ON msc.declaration_id = msr.declaration_id AND msc.pairing_key = msr.pairing_key AND msc.eval_id = msr.eval_id
     AND msc.metric_id = msr.metric_id AND msc.threshold = msr.threshold
JOIN metric_role mr ON mr.run_id = rm.run_id AND mr.judgment_id = rm.judgment_id AND mr.metric_id = rm.metric_id
     AND mr.threshold = rm.threshold AND mr.computation_id = rm.computation_id
JOIN metric_role_contest mc ON mc.run_id = rm.run_id AND mc.judgment_id = rm.judgment_id AND mc.metric_id = rm.metric_id AND mc.threshold = rm.threshold
WHERE wm.run_id IS NULL
  -- the admission of the row to the frame (frame_admission_sql, the text the coverage test of v_paper_current reads too)
  """ + frame_admission_sql() + """;
""", ('v_paper_current', 'v_roster_match', 'roster_entry', 'run', 'model', 'object', 'eval', 'instrument', 'judgment', 'judge', 'item_set',
      'run_metric', 'run_contrast', 'observation', 'source_revision', 'v_multi_source_rows', 'v_multi_source_check', 'decode_policy',
      'v_run_degeneration_current')))

    # ---- paper: object-level membership. Never hand-set: derived from the runs the flag selects plus the rulings
    # that name the object directly (by id, prefix, kind or label) in the open round.
    obj_match = " AND ".join([_m_in('object_ids', 'o.object_id', 'x'), _m_prefix('object_id_prefixes', 'o.object_id', 'x'),
                              _m_in('object_kinds', 'o.object_kind', 'x'), _m_in('paper_labels', 'o.paper_label', 'x'),
                              _m_in('object_roles', 'o.object_role', 'x')])
    obj_any = " OR ".join(f"json_type(x.scope_json, '$.match.{k}') IS NOT NULL" for k in ('object_ids', 'object_id_prefixes', 'object_kinds', 'paper_labels', 'object_roles'))
    views.append(("v_paper_objects", f"""
CREATE VIEW v_paper_objects AS
WITH open_round AS (
    SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1
),
sel AS MATERIALIZED (
    SELECT pc.object_id, pc.run_id, pc.in_current_paper, pc.selection_state, pc.comparison_ids_json, pc.decisive_ruling_id, pc.decisive_action
    FROM v_paper_current pc
),
obj_rulings AS MATERIALIZED (
    SELECT o.object_id, x.ruling_id, x.action
    FROM object o
    JOIN ruling x ON x.expired_utc IS NULL AND x.round_id = (SELECT round_id FROM open_round)
    WHERE (json_type(x.scope_json, '$.match') = 'object' AND ({obj_any}) AND {obj_match})
       OR (json_type(x.scope_json, '$.match') IS NULL AND x.scope_kind = 'object'
           AND (json_extract(x.scope_json, '$.object_id') = o.object_id
                OR json_extract(x.scope_json, '$.object_kind') = o.object_kind
                OR json_extract(x.scope_json, '$.paper_label') = o.paper_label))
)
SELECT o.object_id, o.paper_label, o.object_kind, o.object_role, o.status AS object_status, o.fit_model_id, o.campaign,
       (SELECT round_id FROM open_round) AS round_id,
       CASE WHEN EXISTS (SELECT 1 FROM sel s WHERE s.object_id = o.object_id AND s.in_current_paper = 1) THEN 1 ELSE 0 END AS in_paper,
       (SELECT COUNT(*) FROM sel s WHERE s.object_id = o.object_id) AS n_runs,
       (SELECT COUNT(*) FROM sel s WHERE s.object_id = o.object_id AND s.in_current_paper = 1) AS n_runs_in_paper,
       (SELECT json_group_array(DISTINCT c.value) FROM sel s, json_each(s.comparison_ids_json) c
         WHERE s.object_id = o.object_id AND s.in_current_paper = 1) AS comparison_ids_json,
       (SELECT json_group_array(DISTINCT s.selection_state) FROM sel s WHERE s.object_id = o.object_id) AS selection_states_json,
       (SELECT json_group_array(DISTINCT x.ruling_id) FROM obj_rulings x WHERE x.object_id = o.object_id
         AND x.action IN ('exclude', 'supersede', 'descope')) AS excluding_ruling_ids_json,
       (SELECT json_group_array(DISTINCT s.decisive_ruling_id) FROM sel s WHERE s.object_id = o.object_id
         AND s.decisive_action IN ('exclude', 'supersede') AND s.decisive_ruling_id IS NOT NULL) AS run_excluding_ruling_ids_json,
       (SELECT json_group_array(DISTINCT x.ruling_id) FROM obj_rulings x WHERE x.object_id = o.object_id) AS ruling_ids_json
FROM object o;
""", ('paper_round', 'object', 'ruling', 'v_paper_current')))

    # ---- the paper-input head: the rebuild trigger. Migration 0015 made it a lookup. Until then its
    # first member was `MAX(ingest_seq) FROM v_paper_long WHERE in_current_paper = 1`, which evaluates the whole paper frame (roster
    # match, rulings, supersession, withdrawals) for one number: 452 s of every harvester cycle on the live database, outside any
    # transaction. No cheap EXACT form exists -- membership IS the frame -- so the member is replaced by a SUPERSET: the base tables an
    # in-paper v_paper_long row draws its ingest_seq from (`MAX(r.ingest_seq, rm.ingest_seq, COALESCE(rc.ingest_seq, 0),
    # COALESCE(msc.ingest_seq, 0))`: `run`, `run_metric`, `run_contrast`; v_multi_source_check's ingest_seq is a max over run_metric's),
    # read whole. The head therefore moves at least whenever the frame's own maximum did (a consumer is never left on a stale cache) and
    # also when a run outside the frame lands (a rebuild that reads an unchanged frame -- the price). The rule holds as before: accepted
    # source revisions and withdrawals are members in their own right. The item member is exact and now one seek (idx_item_result_withdrawn_seq,
    # named in the SQL: the live ANALYZE statistics put idx_item_result_withdrawn at 480 rows against 509k, and steer the planner to it).
    # Migration 0021 widens two members for the currency rules it lands (found in review, 2026-10-02: the frame changed with nothing
    # moving the head): the run member reads EVERY row, not the two lifecycle states the selection CASE admits -- a run that retires
    # (judged -> superseded / withdrawn / cancelled / failed, or `superseded_by_run_id` set) moves its own row's rank while LEAVING
    # those two states, and the frame reads that retirement on every contrast anchored on it and on the run's own rows;
    # and `judgment` is a member in its own right -- a judgment re-filed superseded or abandoned removes its metric and item rows
    # from the frame while moving none of their ranks (the fold re-stamps the run row beside the judgment row, every judgment record
    # carrying a judgment_state effect on its run -- a side effect the head must not lean on). Neither move is a subset: the head still
    # reads >= every earlier text.
    # Migration 0020 adds two members for a producer's item withdrawals: the small item_withdrawal table (a fold's rank) and the
    # native item_result row each of its keys names, sought by primary key from that table -- a row RE-FILED above its withdrawal clears
    # the marks and is current again at a rank only the row itself carries (found in review, 2026-10-01: a withdrawal moved the head and
    # the re-filed row did not, so paper_rebuild.sh left the per-item readings without the restored row).
    views.append(("v_paper_inputs_head", """
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
""", ('run', 'run_metric', 'run_contrast', 'judgment', 'roster_entry', 'ruling', 'ruling_state', 'disposition', 'source_revision',
      'withdrawal', 'item_withdrawal', 'item_result')))
    return views


VIEW_NAMES: Tuple[str, ...] = tuple(name for name, _, _ in render_views())
VIEW_SQL: Dict[str, str] = {name: sql for name, sql, _ in render_views()}
VIEW_REFERENCES: Dict[str, Tuple[str, ...]] = {name: refs for name, _, refs in render_views()}


def _close_over_tables(name: str, seen: Optional[set] = None) -> frozenset:
    seen = seen if seen is not None else set()
    out: set = set()
    for ref in VIEW_REFERENCES[name]:
        if ref in VIEW_REFERENCES:
            if ref not in seen:
                seen.add(ref)
                out |= _close_over_tables(ref, seen)
        else:
            if ref not in TABLE_BY_NAME:
                raise ValueError(f"view {name} declares an unknown reference {ref!r}")
            out.add(ref)
    return frozenset(out)


#: View -> every BASE table it reads, through any views it references.
VIEW_TABLES: Dict[str, frozenset] = {name: _close_over_tables(name) for name in VIEW_NAMES}

# --------------------------------------------------------------------------------------------
# DDL rendering
# --------------------------------------------------------------------------------------------


def _vocab_list_sql(name: str) -> str:
    vals = V.VOCABULARIES[name]
    if all(isinstance(v, int) for v in vals):
        return ", ".join(str(v) for v in vals)
    return ", ".join("'%s'" % v for v in vals)


def render_column(c: Column) -> str:
    parts = [c.name, c.type]
    if c.nn:
        parts.append("NOT NULL")
    if c.default is not None:
        parts.append(f"DEFAULT {c.default}")
    if c.vocab:
        parts.append(f"CHECK ({c.name} IN ({_vocab_list_sql(c.vocab)}))")
    if c.pattern:
        pat = PATTERNS[c.pattern].format(col=c.name)
        if c.nn:
            parts.append(f"CHECK {pat}")
        else:
            parts.append(f"CHECK ({c.name} IS NULL OR {pat[1:-1]})")
    if c.fk and c.sentinel is None and not c.fk_deferred:
        parts.append(f"REFERENCES {c.fk[0]}({c.fk[1]})")
    line = " ".join(parts)
    if c.why_null:
        line += f"  -- why-null: {c.why_null}"
    elif c.doc:
        line += f"  -- {c.doc}"
    return line


def render_create_table(t: Table, name: Optional[str] = None) -> str:
    """The ``CREATE TABLE`` statement alone: the columns, PRIMARY KEY, UNIQUEs and named CHECKs (no doc comment, no
    indexes). ``name`` renders the same body under another table name — a migration's ``<table>__new`` (SQLite cannot
    alter a CHECK in place; ``migrations.render_table_rebuild``)."""
    lines = [f"CREATE TABLE {name or t.name} ("]
    body = ["  " + render_column(c) for c in t.columns]
    body.append("  PRIMARY KEY (%s)" % ", ".join(t.pk))
    for u in t.uniques:
        body.append("  UNIQUE (%s)" % ", ".join(u))
    for cname, sql in t.checks:
        body.append(f"  CONSTRAINT {cname} CHECK {sql}")
    # join with commas, but keep trailing comments after the comma
    out_body = []
    for i, b in enumerate(body):
        comma = "," if i < len(body) - 1 else ""
        if "  -- " in b:
            head, comment = b.split("  -- ", 1)
            out_body.append(f"{head}{comma}  -- {comment}")
        else:
            out_body.append(b + comma)
    lines += out_body
    lines.append(") WITHOUT ROWID;" if t.scale in ("item", "staging") else ");")
    return "\n".join(lines)


def render_indexes(t: Table) -> List[str]:
    """The table's explicit ``CREATE INDEX`` statements, one per entry of ``Table.indexes``."""
    out = []
    for name, cols, where in t.indexes:
        w = f" WHERE {where}" if where else ""
        out.append(f"CREATE INDEX {name} ON {t.name} ({', '.join(cols)}){w};")
    return out


def render_table(t: Table) -> str:
    lines = [f"-- {t.doc}" if t.doc else "", render_create_table(t)] + render_indexes(t)
    return "\n".join(l for l in lines if l is not None)


def triggers_on(table_name: str) -> List[str]:
    """The code's ``CREATE TRIGGER`` statements ON ``table_name`` (dropped with the table by SQLite, so a table rebuild
    recreates them)."""
    marker = f" ON {table_name}\n"
    return [sql for sql in render_triggers() if marker in sql]


HEADER = """-- schema.sql — the results registry DDL (GENERATED by sisteer.registry.schema_def; do not edit by hand).
-- Regenerate with `registry docs schema` (which also rewrites docs/registry/schema.md).
-- Every CHECK (col IN (...)) list is rendered from vocab.py; the object.fit_target_kind pairing CHECK from
-- vocab.FIT_TARGET_RULES; nullable columns carry a `-- why-null:` reason. Booleans are INTEGER 0/1; every
-- *_utc TEXT is ISO-8601 UTC with milliseconds (…Z); every *_json TEXT is json_valid.
"""


def render_schema_sql() -> str:
    parts = [HEADER]
    parts.append("-- ==================================================================== tables")
    for t in TABLES:
        parts.append(render_table(t))
        parts.append("")
    parts.append("-- ==================================================================== triggers")
    parts += render_triggers()
    parts.append("")
    parts.append("-- ==================================================================== views")
    for _, sql, _ in render_views():
        parts.append(sql.strip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def required_fields(table_name: str) -> List[str]:
    """Columns a full record of this table must carry: NOT NULL without a DEFAULT, the derived tail excluded."""
    t = TABLE_BY_NAME[table_name]
    return [c.name for c in t.columns if c.nn and c.default is None and not c.derived]


def payload_columns(table_name: str) -> List[str]:
    """Columns a record payload may set (everything but the derived tail)."""
    return [c.name for c in TABLE_BY_NAME[table_name].columns if not c.derived]


def foreign_keys(table_name: str) -> List[Tuple[str, str, str]]:
    """[(column, parent_table, parent_column)] for every enforced FK (REFERENCES and sentinel-trigger alike)."""
    return [(c.name, c.fk[0], c.fk[1]) for c in TABLE_BY_NAME[table_name].columns if c.fk]


def _sentinels() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for t in TABLES:
        for c in t.columns:
            if c.sentinel is not None:
                if c.name in out and out[c.name] != c.sentinel:
                    raise ValueError(f"column {c.name} carries two different sentinels ({out[c.name]!r}, {c.sentinel!r})")
                out[c.name] = c.sentinel
    return out


#: Composite-key column -> its "absent" value (the same for every table that carries the column).
SENTINELS: Dict[str, object] = _sentinels()
