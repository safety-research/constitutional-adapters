"""The migration runner and the shape-migration generator (2026-09-19).

0002 carries a database at the live pre-fix shape (0001 as of 6803da221: the object table without the
'difference_relu_distilled' paper label, the older v_source_status) to the code's shape with every row intact and the
foreign keys checked; the runner rolls back a migration that leaves an FK violation or manages its own transaction;
the generator renders, from schema_def, a migration that takes any drifted database to zero drift.
"""
from __future__ import annotations

import re

import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import migrations as M
from sisteer.registry import schema_def as S
from sisteer.registry import vocab as V
from test_registry_schema import frozen_column_docs

OLD_LABEL_LIST = "'difference_relu_distilled', "
#: what migration 0012 adds to the drift of EVERY earlier shape (the drift is against the CODE's shape): the two padding columns and their
#: pairing CHECK on `instrument`, and the two views that expose them (`v_run_wide` first in the code's order; `v_paper_long` drifts on most
#: earlier shapes for its own reasons too)
V12_TABLE_DRIFT = ["table instrument lacks column(s) padding_tokens, padding_identity_sha256", "table instrument: CONSTRAINT instrument_padding_pairing is absent"]
V12_VIEW_DRIFT = "v_run_wide differs from the code's SQL"
#: what migration 0015 adds to the drift of EVERY earlier shape: the head reads its base tables (the last v_* object in the code's order)
#: and the partial index its item member names (indexes are listed after the views)
V15_DRIFT = ["v_paper_inputs_head differs from the code's SQL", "index idx_item_result_withdrawn_seq is absent"]
#: what migration 0016 adds to the drift of EVERY earlier shape (roster v9): the `roster_version` column on paper_round and roster_entry
#: (each a "lacks column(s)" line and a "CREATE text differs" line, tables in the code's order: paper_round before instrument, roster_entry after)
#: and the two roster objects that read it (`v_roster_match`, which most earlier shapes drift on already, and `v_roster_gaps`)
V16_TABLE_PREFIXES = ("table paper_round lacks column(s) roster_version", "table paper_round: CREATE text differs",
                      "table roster_entry lacks column(s) roster_version", "table roster_entry: CREATE text differs")
V16_VIEW_DRIFT = ["v_roster_match differs from the code's SQL", "v_roster_gaps differs from the code's SQL"]
#: the objects 0016 rebuilds or recreates when the runner carries a database past 15: an earlier migration's test that pins "nothing else
#: moved" leaves them out of its set (each earlier test checks its own objects; 0016's test checks these)
V16_OBJECTS = ("paper_round", "roster_entry", "v_roster_match", "v_roster_gaps")
#: the object 0017 recreates when the runner carries a database past 16 (the frame joins only current contrasts), and 0019 again when it
#: carries one past 18 (the frame derives the Wilson interval): an earlier migration's test that pins "nothing else moved" leaves it out
#: of its set (0017's and 0019's tests check it; on every shape before 19 its ONE drift line stands for both)
V17_OBJECTS = ("v_paper_long",)


#: what migration 0018 adds to the drift of EVERY earlier shape: `slurm` in vocab.SUBSTRATES rewrites the three substrate CHECKs
#: (run.substrate, object.fit_substrate, fit_search.substrate); every per-version test below strips these three lines through
#: `_sans_0016`, the filter they all apply, so the assertions keep describing THEIR migration (tests/test_migration_window.py owns 0018)
V18_CHECK_DRIFT = ("table fit_search: column substrate CHECK (substrate IN (...)) lacks 'slurm'",
                   "table object: column fit_substrate CHECK (fit_substrate IN (...)) lacks 'slurm'",
                   "table run: column substrate CHECK (substrate IN (...)) lacks 'slurm'")


#: what migration 0020 adds to the drift of EVERY earlier shape: `item_result` lacks the two producer-withdrawal marks (a
#: "lacks column(s)" line and a "CREATE text differs" line), the `item_withdrawal` table is absent, the two ledgers' record_type
#: CHECKs (and row_history's table_name CHECK) lack the new kind, and the three objects that read the marks or the table differ --
#: `v_item_result_current`, `v_item_result_provenances` and `v_paper_inputs_head`. The head's ONE drift line stands for 0015, 0020 and
#: 0021 alike on every shape before 15 (the V17_OBJECTS reading), so `_sans_0020` and `_sans_0021` keep it: the tests of shapes 15..20
#: strip it themselves
V20_TABLE_PREFIXES = ("table item_result lacks column(s) withdrawn_by_record_sha256, withdrawn_reason", "table item_result: CREATE text differs")
V20_DRIFT = ["table item_withdrawal is absent",
             "table blocked_records: column record_type CHECK (record_type IN (...)) lacks 'item_withdrawal'",
             "table row_history: column table_name CHECK (table_name IN (...)) lacks 'item_withdrawal'",
             "table row_history: column record_type CHECK (record_type IN (...)) lacks 'item_withdrawal'",
             "v_item_result_current differs from the code's SQL", "v_item_result_provenances differs from the code's SQL"]
V20_HEAD_DRIFT = "v_paper_inputs_head differs from the code's SQL"
#: the objects 0020 creates, rebuilds, alters or recreates: an earlier migration's "nothing else moved" set leaves them out
V20_OBJECTS = ("item_result", "item_withdrawal", "blocked_records", "row_history", "v_item_result_current", "v_item_result_provenances",
               "v_paper_inputs_head")


#: what migration 0021 adds to the drift of EVERY earlier shape (the frame's currency rules): `v_run_current`
#: is absent, and the eight objects it recreates differ -- `v_run_decode_policy_match` (new on every shape),
#: `v_item_result_current` / `v_item_result_provenances` (0020's on the shapes before 20), `v_pending_judgment` (0003's / 0007's on the
#: shapes before 7), `v_paper_current` (new on the shapes 12..20), `v_multi_source_rows` (0005's / 0010's before 10: ownership counts
#: live judgment rows only), `v_paper_long` (0017's / 0019's before 19) and `v_paper_inputs_head` (0015's / 0020's before 20: the run
#: member reads every row and `judgment` is a member, so the rebuild trigger follows the rules above). View-only: no table line.
#: `_sans_0021` strips the first eight; the head's line it KEEPS, as `_sans_0020` does (it stands for 0015 and 0020 too on the shapes
#: before 20, and the tests of those shapes account for it themselves). An earlier migration's test whose own object is among the
#: eight asserts its line on the RAW drift (the 0003 / 0006 / 0007 / 0009 / 0010 / 0011 / 0012 / 0017 / 0019 / 0020 tests do) and its
#: "nothing else moved" set leaves them out
V21_DRIFT = ["v_run_decode_policy_match differs from the code's SQL", "v_item_result_current differs from the code's SQL",
             "v_item_result_provenances differs from the code's SQL", "v_pending_judgment differs from the code's SQL",
             "v_run_current is absent", "v_paper_current differs from the code's SQL", "v_multi_source_rows differs from the code's SQL",
             "v_paper_long differs from the code's SQL", V20_HEAD_DRIFT]


#: what the record-refusal-fact migration (the newest file on this branch, numbered at landing)
#: adds to the drift of EVERY earlier shape: three tables lacking columns declared after their last live column (a "lacks column(s)" line
#: and a "CREATE text differs" line each: `object`, `judgment`, `record_refusal`, in the code's order), the fact table and its two indexes
#: absent, and the seven objects it recreates -- six of them among 0021's nine, so their lines are 0021's on every shape before 21
#: (`_sans_0021` strips them) and the record-refusal-fact migration's on the shape one below it alone (its own test pins them); the
#: seventh, `v_roster_match`, is RRF_ROSTER_DRIFT below. No ledger
#: line: the two ledgers' record_type CHECKs render from vocab.LEDGERED_RECORD_TYPES and are byte-identical to 0020's. `_sans_rrf` strips
#: the table and index lines; every per-version test below applies it through `_sans_0021`
RRF_TABLE_PREFIXES = ("table object lacks column(s) parent_object_id, named_role", "table object: CREATE text differs",
                      "table judgment lacks column(s) claimed_instrument_id, claimed_required_judge_id", "table judgment: CREATE text differs",
                      "table record_refusal lacks column(s) judged_with", "table record_refusal: CREATE text differs")
RRF_DRIFT = ["table record_refusal_fact is absent", "index idx_record_refusal_fact_cycle is absent", "index idx_record_refusal_fact_uri is absent"]
RRF_VIEWS = ("v_item_result_current", "v_item_result_provenances", "v_multi_source_rows", "v_paper_current", "v_paper_long", "v_pending_judgment",
             "v_roster_match")
#: the one recreate of the record-refusal-fact migration outside 0021's nine: `v_roster_match` (0016's last; the review of the first rendering:
#: the covering judgment that satisfies an entry is identity-current). Its drift line is 0013's / 0016's on every shape before 16 (those tests
#: name it) and the record-refusal-fact migration's alone on the shapes 16..20, whose tests strip it with `_sans_rrf_roster`
RRF_ROSTER_DRIFT = "v_roster_match differs from the code's SQL"
#: the tables the record-refusal-fact migration creates or alters: an earlier migration's "nothing else moved" set leaves them out (their
#: CREATE text moves when the runner carries a database past 21; the fact table is absent before)
RRF_TABLES = ("object", "judgment", "record_refusal", "record_refusal_fact")
RRF_OBJECTS = RRF_TABLES + RRF_VIEWS


#: what the fold-observation migration (newer than the record-refusal-fact migration; numbered at landing, 2026-10-04) adds to the drift
#: of EVERY earlier shape: the table and its one index absent -- no column, no recreate, no ledger line (no record kind was added).
#: `_sans_fo` strips both and `_sans_rrf` applies it, so every per-version test below keeps describing its own migration; `FO_TABLES`
#: joins the "nothing else moved" exclusions beside `RRF_TABLES`. Its own test is tests/test_registry_fold_observations.py
FO_DRIFT = ["table fold_observation is absent", "index idx_fold_observation_written is absent"]
FO_TABLES = ("fold_observation",)


#: what migration 0025 (pending_excludes_superseded: a superseded run is never pending, its successor is -- the registry ruling of
#: 2026-10-04) adds to the drift of EVERY earlier shape: the ONE object it recreates, `v_pending_judgment` -- whose line is 0021's / the
#: record-refusal-fact migration's on every shape before 23 (those tests assert it, so `_sans_fo` / `_sans_rrf` must NOT strip it there)
#: and 0025's alone on the shapes 23 and 24, whose tests strip it themselves with `_sans_pes` (the fold-observation migration's test). No
#: table line, no column, no index, no post-step. `_pre_0025_pending_judgment` takes the code's text of the object back to 0023's (the four
#: lines 0025 added), `_sans_pes_render` takes a render back to before the file where an earlier file recreates the object too, and
#: `_sans_pes_block` where none does, for the tests whose committed bodies predate it. Its own test is
#: tests/test_registry_pending_excludes_superseded.py
PES_DRIFT = ["v_pending_judgment differs from the code's SQL"]
PES_OBJECTS = ("v_pending_judgment",)
PES_BLOCK_START = "  -- a superseded run is never pending; its successor is (the registry decision of 2026-10-04, migration 0025)"
PES_BLOCK_END = "  AND NOT (" + S.run_superseded_sql("r") + ")"


#: what migration 0026 (twin_fold_view_legacy_value_with_draws: the r2_export <-> sampled_lane twins fold onto the pooled reading by
#: `view_legacy` and `value_with_draws`; two different-tag pooled runs of one source coexist -- adapter re-anchoring plan 5.2 / 10.6) adds
#: to the drift of EVERY earlier shape: the THREE objects it recreates. `v_multi_source_check`'s line is 0026's on every shape (no file since
#: its last recreate had moved it), so `_sans_fo`, the innermost strip every chain applies, takes it out; `v_run_current`'s is 0021's /
#: 0022's on the shapes before 22 (asserted there) and 0026's alone on 22..25; `v_multi_source_rows`'s is 0021's / the record-refusal-fact
#: migration's before 23 and 0026's alone on 23..25 -- the tests of the shapes 22..25 strip those with `_sans_tf` (`_sans_pes` applies it)
#: or `TF_SHAPE22_VIEWS`. `_pre_0026_run_current` / `_pre_0026_multi_source_rows` take the code's text of the two objects back to before the
#: file (the one conjunct with its three comment lines; the two projected columns and the two twin declarations' blocks), `_sans_tf_block`
#: removes 0026's blocks and drift lines from a render. No table, no column, no index, no post-step. Its own test is
#: tests/test_registry_twin_fold.py
TF_VIEWS = ("v_multi_source_check", "v_multi_source_rows", "v_run_current")
TF_DRIFT = [f"{v} differs from the code's SQL" for v in TF_VIEWS]
TF_CHECK_DRIFT = TF_DRIFT[0]
TF_CHECK_VIEW = TF_VIEWS[0]
TF_SHAPE22_VIEWS = ("v_multi_source_check", "v_run_current")        # 0026's alone on shape 22 (v_multi_source_rows is the record-refusal-fact migration's there)
TF_SHAPE22_DRIFT = [f"{v} differs from the code's SQL" for v in TF_SHAPE22_VIEWS]
TF_LATER_COMMENT = ("    -- and a supersession that held before the key grew still holds. The ONE pairwise exception (Q11, migration 0026; rendered from\n"
                    "    -- vocab.COEXIST_ACROSS_VIEWS_BY_SOURCE): two pooled runs of a listed source at one key whose view_legacy differ are the readings of two\n"
                    "    -- POINTS (the tags pool separately), never predecessor and successor -- the pair is skipped; a later same-tag pooled run still supersedes\n")
TF_LATER_COMMENT_BEFORE = "    -- and a supersession that held before the key grew still holds\n"


def _sans_tf(drift):
    """The drift without migration 0026's three lines (and 0027's table line, which every earlier shape sees too)."""
    return [d for d in _sans_fu4_drift(drift) if d not in TF_DRIFT]


#: what migration 0027 (decisive_first_source_local_anchor_solo_stand_in: the decisive ruling before the roster branch,
#: the `source_local_anchor` frame rule, the `solo_stand_in` anchor kind) adds to the drift of EVERY
#: earlier shape: `v_run_current` (its line is 0021's / 0022's / 0026's on every shape before 26 and 0027's alone on 26) and, from the
#: anchor-kind commit on, the `run_contrast` rebuild (`FU4_TABLE_DRIFT`, `FU4_TABLES`; every earlier shape sees it, so `_sans_fo`, the
#: innermost strip every chain applies, takes it out and `_sans_fu4_render` takes a render back to before the file). `_pre_0027_run_current`
#: takes the code's text of `v_run_current` back to 0026's -- the FROZEN 0026 file's own CREATE text, read rather than reconstructed: the
#: file reorders the selection CASE, retires a belt from is_current and `why`, and adds a branch, which no string surgery restates safely.
#: Its own test is tests/test_registry_views_0027.py
FU4_SLUG = "decisive_first_source_local_anchor_solo_stand_in"
FU4_TABLE_DRIFT = ["table run_contrast: column anchor_kind CHECK (anchor_kind IN (...)) lacks 'solo_stand_in'"]
FU4_TABLES = ("run_contrast",)

#: what migration 0029 (analysis_filing; a registry decision of 2026-10-08: analyses filed over ingested runs and objects as ledger facts)
#: adds to the drift of EVERY earlier shape: its five tables (the three analysis tables, `ordered_fact_record`, the harvester's index of
#: their records by key, the second review round of 2026-10-09, and `ordered_fact_carrier`, its audit of every file that carried one, the
#: ninth closing pass of 2026-10-10) and the analysis tables' four indexes absent -- no column, no recreate, no
#: ledger line (the three kinds are in vocab.UNLEDGERED_FACT_TYPES), no post-step. `_sans_fu4_drift`, the innermost strip every chain
#: applies, takes the lines out, and `_sans_fu4_rebuild`, the innermost render strip, takes the five `-- create` blocks out
#: (`_sans_af_render`); `AF_TABLES` joins the "nothing else moved" exclusions. `_strip_af_shape` takes a database built from the code's
#: render back to before the file. Its own test is tests/test_registry_analysis_filing.py
AF_SLUG = "analysis_filing"
AF_TABLES = ("analysis", "analysis_input", "analysis_artifact", "ordered_fact_record", "ordered_fact_carrier")
AF_DRIFT = [f"table {t} is absent" for t in AF_TABLES] + [f"index {i} is absent" for i in
            ("idx_analysis_artifact_artifact", "idx_analysis_filed", "idx_analysis_input_object", "idx_analysis_input_run")]


def _sans_af_drift(drift):
    """The drift without migration 0029's nine lines -- and without 0030's two (`_sans_fence_drift`): the two files after every earlier
    shape, whose tables and whose fence triggers every earlier shape lacks. Every chain of strips bottoms out here."""
    return [d for d in _sans_fence_drift(drift) if d not in AF_DRIFT]


def _sans_af_render(text: str) -> str:
    """A rendered migration body without migration 0029's part: its nine drift lines and its five `-- create` blocks (each the CREATE
    TABLE, its indexes and the blank line after) -- and without 0030's (`_sans_fence_render`): the two fence triggers' drift lines and
    DROP + CREATE blocks, which today's render carries against every shape below 0030."""
    out, skip = [], False
    for ln in _sans_fence_render(text).split("\n"):
        if ln in {"-- drift: " + d for d in AF_DRIFT}:
            continue
        if ln in {f"-- create {t}" for t in AF_TABLES}:
            skip = True
            continue
        if skip:
            if ln == "":
                skip = False
            continue
        out.append(ln)
    joined = "\n".join(out)
    assert "CREATE TABLE analysis" not in joined
    return joined


#: Migration 0030 `fold_order_barrier`'s shape change: the two fold-order fence triggers on `run` (`schema_def.fold_order_fence_triggers`),
#: which every shape below 0030 lacks -- so, like 0029's nine lines, they are set aside by the tests of the earlier files' plans.
FENCE_TRIGGERS = ("trg_run_fold_order_fence_ins", "trg_run_fold_order_fence_upd")
FENCE_DRIFT = [f"trigger {t} is absent" for t in FENCE_TRIGGERS]


def _sans_fence_drift(drift):
    """The drift without migration 0030's two lines -- and without 0031's two (`_sans_ntpl_drift`, the newest file's)."""
    return [d for d in _sans_ntpl_drift(drift) if d not in FENCE_DRIFT]


#: Migration 0031 `non_template_positions`' shape change: `non_template` in the applied-positions CHECK of `instrument` and `run`, a
#: REBUILD of both tables -- which every shape below 0031 lacks, so, like 0030's two triggers, it is set aside by the tests of the earlier
#: files' plans: `_sans_ntpl_drift`, the innermost drift strip (under `_sans_fence_drift`), takes the two `lacks` lines out; `_sans_ntpl_render`,
#: the innermost render strip (under `_sans_fence_render`), takes the `instrument` rebuild block out, the `run` rebuild block too where
#: 0031 alone rebuilds `run` on that shape, and the value alone out of the run CREATE text where an earlier file's rebuild of `run` carries
#: it (0018's, on every shape below 18: tests/test_migration_window.py); `NTPL_TABLES` joins the "nothing else moved" exclusions where a
#: test reads the two tables raw, and `_object_sql` compares them without the value, as it does `slurm` for 0018's three.
NTPL_SLUG = "non_template_positions"
NTPL_TABLES = ("instrument", "run")
NTPL_VALUE = "'non_template'"
NTPL_DRIFT = [f"table {t}: column applied_positions_mode CHECK (applied_positions_mode IN (...)) lacks {NTPL_VALUE}" for t in NTPL_TABLES]


def _sans_ntpl_drift(drift):
    """The drift without migration 0031's two lines."""
    return [d for d in drift if d not in NTPL_DRIFT]


def _sans_rebuild_block(text: str, table: str) -> str:
    """A rendered migration body without the `-- rebuild <table>:` block (through `ANALYZE <table>;` and the blank line after it); a
    render with no such block is returned as it is."""
    out, skip = [], False
    for ln in text.split("\n"):
        if ln.startswith(f"-- rebuild {table}:"):
            skip = True
            continue
        if skip:
            if ln == f"ANALYZE {table};":
                skip = "blank"
            elif skip == "blank":
                skip = False
                if ln != "":
                    out.append(ln)
            continue
        out.append(ln)
    assert f"{table}__new" not in "\n".join(out)
    return "\n".join(out)


def _sans_ntpl_render(text: str) -> str:
    """A rendered migration body without migration 0031's part: its two drift lines, the `instrument` rebuild block, and -- where no other
    drift line names `run` (every shape from 18 on) -- the `run` rebuild block; where another file's rebuild of `run` is in the render
    (0018's substrate line, below 18) only the value comes out of the CREATE text."""
    out = "\n".join(ln for ln in text.split("\n") if ln not in {"-- drift: " + d for d in NTPL_DRIFT})
    out = _sans_rebuild_block(out, "instrument")
    if any(re.match(r"^-- drift: table run\b[: ]", ln) for ln in out.split("\n")):     # `run` itself, never `run_contrast`
        out = out.replace(NTPL_VALUE + ", ", "")
    else:
        out = _sans_rebuild_block(out, "run")
    assert NTPL_VALUE not in out
    return out


def _sans_fence_render(text: str) -> str:
    """A rendered migration body without migration 0030's part: its two drift lines and, per fence trigger, the `DROP TRIGGER IF EXISTS`
    and the `CREATE TRIGGER … END;` block -- and without 0031's (`_sans_ntpl_render`)."""
    out, skip = [], False
    for ln in _sans_ntpl_render(text).split("\n"):
        if ln in {"-- drift: " + d for d in FENCE_DRIFT} or ln in {f"DROP TRIGGER IF EXISTS {t};" for t in FENCE_TRIGGERS}:
            continue
        if any(ln.startswith(f"CREATE TRIGGER {t} ") for t in FENCE_TRIGGERS):
            skip = True
            continue
        if skip:
            if ln == "END;":
                skip = False
            continue
        out.append(ln)
    joined = "\n".join(out)
    assert "fold_order_fence" not in joined
    return joined


def _strip_af_shape(conn) -> None:
    """Take a database built from the CODE's render back to before migration 0029: its four indexes and five tables dropped, children
    first (nothing else of the code's shape names them: no column, no v_* object, no ledger CHECK)."""
    for idx in ("idx_analysis_artifact_artifact", "idx_analysis_input_run", "idx_analysis_input_object", "idx_analysis_filed"):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    for t in reversed(AF_TABLES):
        conn.execute(f"DROP TABLE {t}")


def _frozen_view_text(migration_slug: str, view: str) -> str:
    """The CREATE text of ``view`` as the FROZEN migration file carrying ``migration_slug`` committed it (the last block of that file
    for the view; the body ends at the next DROP or at the file's end), with its terminator."""
    path = next(p for _v, s, p in M.migration_files() if s == migration_slug)
    text = open(path, encoding="utf-8").read()
    marker = "DROP VIEW IF EXISTS " + view + ";\n"
    start = text.rindex(marker) + len(marker)
    rest = text[start:]
    end = rest.find("\nDROP VIEW IF EXISTS ")
    block = rest if end < 0 else rest[:end]
    block = block.strip()
    assert block.startswith("CREATE VIEW " + view + " AS") and block.endswith(";"), (migration_slug, view, block[:60], block[-20:])
    return block


#: migration 0028 (reference_model_never_selected): a run of the front-door worked example's key (schema_def.WORKED_EXAMPLE_MODEL_IDS,
#: qwen35_4b_demo / _base) reads not_in_roster / in_current_paper 0 before any include ruling can select it; the ONE object it recreates is
#: `v_run_current` and it changes nothing else, so the only take-back the earlier shapes need is of that view's text -- the 0027 file's
#: frozen CREATE text (`_pre_0028_run_current`); its own test is tests/test_registry_views_0028.py
RM_SLUG = "reference_model_never_selected"


def _pre_0028_run_current(text: str) -> str:
    """`v_run_current` as it read before migration 0028: the 0027 file's committed text (its frozen sha pins it). ``text`` must be the
    code's render (asserted: the 0028 marks are present); the terminator follows the input."""
    stripped = text.strip()
    assert "migration 0028" in stripped and S.WORKED_EXAMPLE_MODEL_SQL in stripped, "not the code's v_run_current"
    out = _frozen_view_text(FU4_SLUG, "v_run_current")
    assert "migration 0028" not in out and "qwen35_4b_demo" not in out
    assert "d.action = 'supersede' THEN 'superseded'" in out and "source_local_anchor" in out and "migration 0027" in out
    return out if stripped.endswith(";") else out.rstrip(";")


def _pre_0027_run_current(text: str) -> str:
    """`v_run_current` as it read before migration 0027: the 0026 file's committed text (its frozen sha pins it). ``text`` must be the
    code's render (asserted: the file's own marks are present), so the chain of take-backs stays honest; the terminator follows the input."""
    stripped = text.strip()
    assert "migration 0027" in stripped and "d.action = 'supersede' THEN 'superseded'" in stripped, "not the code's v_run_current"
    out = _frozen_view_text("twin_fold_view_legacy_value_with_draws", "v_run_current")
    assert "migration 0027" not in out and "source_local_anchor" not in out
    assert "OR (selection_state = 'not_in_roster' AND decisive_action IN ('exclude', 'supersede')) THEN 0 ELSE 1 END AS is_current" in out
    return out if stripped.endswith(";") else out.rstrip(";")


def _sans_fu4_drift(drift):
    """The drift without migration 0027's table line (`v_run_current`'s line is an earlier file's on every shape before 26 and is kept)
    -- and without 0029's nine lines (`_sans_af_drift`; the newest file, whose tables every earlier shape lacks)."""
    return [d for d in _sans_af_drift(drift) if d not in FU4_TABLE_DRIFT]


def _sans_fu4_rebuild(text: str) -> str:
    """A rendered migration body without migration 0027's `run_contrast` rebuild block (`-- rebuild run_contrast` through `ANALYZE
    run_contrast;` and the blank line after it) and its drift line -- and without 0029's five create blocks and nine drift lines
    (`_sans_af_render`); a render with no such block is returned as it is."""
    lines, out, skip = _sans_af_render(text).split("\n"), [], False
    for ln in lines:
        if ln == "-- drift: " + FU4_TABLE_DRIFT[0]:
            continue
        if ln.startswith("-- rebuild run_contrast:"):
            skip = True
            continue
        if skip:
            if ln == "ANALYZE run_contrast;":
                skip = "blank"
            elif skip == "blank":
                skip = False
                if ln != "":
                    out.append(ln)
            continue
        out.append(ln)
    assert "run_contrast__new" not in "\n".join(out)
    return "\n".join(out)


def _sans_fu4_render(text: str) -> str:
    """A rendered migration body taken back to before migration 0027: its rebuild block and drift line out (`_sans_fu4_rebuild`), and
    `v_run_current`'s block at 0026's text (`_pre_0027_run_current`) where the render carries the code's."""
    out = _sans_fu4_rebuild(text)
    code = S.VIEW_SQL["v_run_current"].strip()
    return out.replace("DROP VIEW IF EXISTS v_run_current;\n" + code, "DROP VIEW IF EXISTS v_run_current;\n" + _pre_0027_run_current(code))


def _pre_0026_run_current(text: str) -> str:
    """`v_run_current` as it read before migration 0026: the `later` CTE's coexist-across-views conjunct and its three comment lines taken
    out of the code's render (0022's text; pinned by that file's frozen sha)."""
    text = _pre_0027_run_current(text)                                                     # 0027's text first: 0026's had none of it
    conj = "        AND " + S.COEXIST_ACROSS_VIEWS_SQL + "\n"
    assert text.count(TF_LATER_COMMENT) == 1 and text.count(conj) == 1, (text.count(TF_LATER_COMMENT), text.count(conj))
    out = text.replace(TF_LATER_COMMENT, TF_LATER_COMMENT_BEFORE).replace(conj, "")
    assert "COEXIST" not in out and "migration 0026" not in out and "pooled_t0_t1') AND r2.view_legacy" not in out
    return out


def _pre_0026_multi_source_rows(text: str) -> str:
    """`v_multi_source_rows` as it read before migration 0026: the two twin declarations' UNION blocks dropped and the two projected columns
    (`check_rule`, `n_draws`) taken out of every block (the record-refusal-fact migration's text, pinned by that file's frozen sha)."""
    tail = ";" if text.rstrip().endswith(";") else ""                                     # the statement's terminator rides the LAST block (a twin's)
    blocks = text.rstrip().rstrip(";").split("\nUNION ALL\n")
    kept = [b for b in blocks if "sampled_static_8192/" not in b]
    assert len(blocks) - len(kept) == 2, (len(blocks), len(kept))
    out = "\nUNION ALL\n".join(kept) + tail
    lines = [ln for ln in out.split("\n") if not re.fullmatch(r" +'[a-z_]+' AS check_rule,", ln) and ln != "       json_extract(rm.extra_json, '$.n_draws') AS n_draws,"]
    assert len(out.split("\n")) - len(lines) == 2 * len(kept), (len(out.split("\n")), len(lines), len(kept))
    out = "\n".join(lines)
    assert "check_rule" not in out and "n_draws" not in out
    return out


def _sans_tf_block(text: str, views=TF_VIEWS) -> str:
    """A rendered migration body without migration 0026's blocks (DROP + the code's CREATE of each of ``views``) and their drift lines."""
    out = _sans_fu4_rebuild(text)                                                            # 0027's rebuild block renders ahead of the views
    for v in views:
        code = S.VIEW_SQL[v].strip()
        out = out.replace("DROP VIEW IF EXISTS " + v + ";\n" + code + "\n", "").replace("DROP VIEW IF EXISTS " + v + ";\n" + code, "")
    return "\n".join(ln for ln in out.split("\n") if ln not in {"-- drift: " + f"{v} differs from the code's SQL" for v in views})


def _sans_pes(drift):
    """The drift without migration 0025's one line (and 0026's three: the shapes 23 and 24 see both)."""
    return [d for d in _sans_tf(drift) if d not in PES_DRIFT]


def _pre_0025_pending_judgment(text: str) -> str:
    """`v_pending_judgment` as it read before migration 0025 (0023's text, which 0024 did not touch; pinned by that file's frozen sha):
    the superseded conjunct and its three comment lines taken out of the code's render."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith(PES_BLOCK_START))
    end = next(i for i, ln in enumerate(lines) if ln == PES_BLOCK_END)
    assert start < end and end - start == 3, (start, end)
    out = "\n".join(lines[:start] + lines[end + 1:])
    assert "superseded_by_run_id" not in out and "lifecycle_state IN ('generated', 'judging', 'judged', 'ingested')" in out
    return out


def _sans_pes_render(text: str, drift_line: bool = True) -> str:
    """A rendered migration body without migration 0025's part: its drift line (when ``drift_line``: a shape whose object the file alone
    moves; a shape on which an earlier file moves it too keeps the line) and the object's block taken back to 0023's text."""
    code = S.VIEW_SQL["v_pending_judgment"].strip()
    out = text.replace("DROP VIEW IF EXISTS v_pending_judgment;\n" + code, "DROP VIEW IF EXISTS v_pending_judgment;\n" + _pre_0025_pending_judgment(code))
    if drift_line:
        out = "\n".join(ln for ln in out.split("\n") if ln != "-- drift: " + PES_DRIFT[0])
    return out


def _sans_pes_block(text: str) -> str:
    """A rendered migration body without migration 0025's object block and drift line altogether -- for a shape on which no earlier file
    moves `v_pending_judgment` (the fold-observation migration's test: its committed body recreates no object)."""
    code = S.VIEW_SQL["v_pending_judgment"].strip()
    out = _sans_tf_block(text)                                                               # 0026's three blocks follow it on this tree
    out = out.replace("DROP VIEW IF EXISTS v_pending_judgment;\n" + code + "\n", "").replace("DROP VIEW IF EXISTS v_pending_judgment;\n" + code, "")
    return "\n".join(ln for ln in out.split("\n") if ln != "-- drift: " + PES_DRIFT[0])


def _sans_fo(drift):
    """The drift without the fold-observation migration's two lines and migration 0026's `v_multi_source_check` line (0026's alone on
    every shape). Migration 0025's one line and 0026's other two are NOT stripped here: on every shape before 22 / 23 they are 0021's /
    0022's / the record-refusal-fact migration's and asserted by their tests; the shapes 22..24 strip them themselves."""
    return [d for d in _sans_fu4_drift(drift) if d not in FO_DRIFT and d != TF_CHECK_DRIFT]


def _sans_fo_render(text: str) -> str:
    """A rendered migration body without the fold-observation migration's part: its two drift lines and its `-- create fold_observation`
    block (the CREATE TABLE, its index, the blank line after) -- for the record-refusal-fact migration's test, whose committed body was the
    render against the v22 shape when it landed and is newer by this one file now."""
    out, skip = [], False
    for ln in text.split("\n"):
        if ln in ("-- drift: " + FO_DRIFT[0], "-- drift: " + FO_DRIFT[1]):
            continue
        if ln == "-- create fold_observation":
            skip = True
            continue
        if skip:
            if ln == "":
                skip = False
            continue
        out.append(ln)
    return "\n".join(out)


def _sans_rrf(drift):
    """The drift without the record-refusal-fact migration's table and index lines (six of its seven recreates are 0021's lines on every
    shape before 21; the seventh, v_roster_match, is `_sans_rrf_roster`) -- and without the fold-observation migration's (`_sans_fo`)."""
    return [d for d in _sans_fo(drift) if not d.startswith(RRF_TABLE_PREFIXES) and d not in RRF_DRIFT]


def _sans_rrf_roster(drift):
    """The drift without the record-refusal-fact migration's `v_roster_match` line -- for the tests of the shapes 16..20, where 0016 wrote
    the text that file recreates (on every earlier shape the line is 0013's / 0016's and those tests name it)."""
    return [d for d in drift if d != RRF_ROSTER_DRIFT]
#: the objects 0021 recreates (the one it creates included): an earlier migration's "nothing else moved" set leaves them out
V21_OBJECTS = ("v_item_result_current", "v_item_result_provenances", "v_multi_source_rows", "v_paper_current", "v_paper_inputs_head",
               "v_paper_long", "v_pending_judgment", "v_run_current", "v_run_decode_policy_match")


#: what migration 0022 adds to the drift of a version-21 shape (a member pass is a member only where its source filed a pooled
#: reading at its own point -- `v_run_current` gains the `pooled_group` CTE, the member predicate's probe into it and the
#: `unpooled_member` branch): the ONE object it recreates. On
#: every shape before 21 `v_run_current is absent` (V21_DRIFT) stands for 0021 and 0022 alike, so `_sans_0021` needs no new line and no
#: earlier test moves; the 0021 test compares its committed body against the render with this object's text taken back (`_pre_0022_run_current`)
V22_DRIFT = ["v_run_current differs from the code's SQL"]
V22_OBJECTS = ("v_run_current",)


def _pre_0022_run_current(text: str) -> str:
    """`v_run_current` as it read before migration 0022 (0021's text, pinned by that file's frozen sha and asserted against it in the
    0021 test), reconstructed from the code's render: the `pooled_group` CTE taken out of the chain, the member_pass / unpooled_member
    pair of SELECTION_STATE_SQL folded back to 0021's one member branch over the per-(source, pass) table half (IN_CURRENT_PAPER_SQL's
    member branch already IS that text), and the unpooled-member clause taken out of the is_current comment."""
    text = _pre_0026_run_current(text)                                                     # 0026's conjunct first: 0022's text had none
    cte = ",\n" + S.POOLED_GROUP_CTE
    pair = ("WHEN " + S.MEMBER_PASS_SQL + " THEN 'member_pass'\n" + " " * 13 + "WHEN " + S.MEMBER_LABEL_SQL + " THEN 'unpooled_member'")
    assert text.count(cte) == 1 and text.count(pair) == 1 and text.count(S.MEMBER_PASS_SQL) == 1 and text.count(S.MEMBER_LABEL_SQL) == 3
    out = text.replace(cte, "").replace(pair, "WHEN " + S.MEMBER_LABEL_SQL + " THEN 'member_pass'")
    d671 = (", and so is a member\n       -- pass whose own point carries no pooled reading of its source, selection_state unpooled_member)")
    assert out.count(d671) == 1
    out = out.replace(d671, ")")
    assert "pooled_group" not in out and "D671" not in out and "unpooled_member" not in out and out.count(S.MEMBER_LABEL_SQL) == 2
    return out


def _sans_0021(drift):
    """The drift without 0021's lines -- the head's excepted, which stands for 0015 and 0020 too (see V21_DRIFT) -- and without the
    record-refusal-fact migration's (`_sans_rrf`)."""
    return [d for d in _sans_rrf(drift) if d not in V21_DRIFT or d == V20_HEAD_DRIFT]


def _sans_0020(drift):
    """The drift without 0020's lines (the head's line excepted: it stands for 0015 too on the shapes before 15) -- and without
    0021's (`_sans_0021`)."""
    return [d for d in _sans_0021(drift) if not d.startswith(V20_TABLE_PREFIXES) and d not in V20_DRIFT]


def _sans_0018(drift):
    """The drift without 0018's three substrate-CHECK lines -- and without 0020's (`_sans_0020`)."""
    return [d for d in _sans_0020(drift) if d not in V18_CHECK_DRIFT]


def _apply_body_without_version(conn, version: int) -> None:
    """Execute a landed migration's statements on ``conn`` WITHOUT its version row, under the runner's pragmas: a LATER file's shape
    pre-applied so an earlier file's render against its own shape is that file's body alone (0020 creates a table, rebuilds two
    ledgers and alters item_result, which every earlier shape lacks)."""
    path = [p for v, _, p in M.migration_files() if v == version][0]
    with open(path, "r", encoding="utf-8") as fh:
        sql = fh.read()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA legacy_alter_table=ON")
    conn.executescript(sql)
    conn.execute("PRAGMA legacy_alter_table=OFF")
    conn.execute("PRAGMA foreign_keys=ON")


def _sans_0016(drift):
    """The drift without 0016's table lines and its v_roster_gaps line (v_roster_match stays: the shapes before 13 drift on it already,
    and the two tests of shapes 13 / 14 name it as 0016's) -- and without 0018's three substrate-CHECK lines (`_sans_0018`)."""
    return [d for d in _sans_0018(drift) if not d.startswith(V16_TABLE_PREFIXES) and d != "v_roster_gaps differs from the code's SQL"]


#: what migration 0017 adds to the drift of EVERY earlier shape: `v_paper_long` joins only current contrasts (the shapes before 12 drift on
#: it already for their own reasons and name it; the tests of shapes 12..15 strip it here)
V17_VIEW_DRIFT = "v_paper_long differs from the code's SQL"


def _sans_0017(drift):
    """The drift without 0017's v_paper_long line."""
    return [d for d in drift if d != V17_VIEW_DRIFT]


#: the run member and its comment as 0015 spelled them (0020 kept them; 0021 reads the table whole)
PRE_0021_RUN_MEMBER = """    -- subset. A run enters the frame only once judged or ingested (the selection CASE), so the run member reads those states alone;
    -- a metric row of any run can reach an in-paper row through the multi-source fold, so the metric and contrast members read every row
    SELECT MAX(ingest_seq) AS seq FROM run WHERE lifecycle_state IN ('judged', 'ingested')"""


def _pre_0021_head() -> str:
    """`v_paper_inputs_head` as it read before migration 0021 (0020's text, pinned by that file's frozen sha and asserted against it in
    the 0020 test): the run member at the two lifecycle states the selection admits, with 0015's comment, and no `judgment` member (its
    block runs from the comment that opens it through the member line)."""
    lines = S.VIEW_SQL["v_paper_inputs_head"].splitlines()
    start = next(i for i, ln in enumerate(lines) if "-- subset. The run member reads every row" in ln)
    end = next(i for i, ln in enumerate(lines) if ln == "    SELECT MAX(ingest_seq) AS seq FROM run")
    assert start < end
    lines = lines[:start] + PRE_0021_RUN_MEMBER.splitlines() + lines[end + 1:]
    start = next(i for i, ln in enumerate(lines) if "a judgment re-filed superseded or abandoned" in ln)
    end = next(i for i, ln in enumerate(lines) if ln == "    UNION ALL SELECT MAX(ingest_seq) FROM judgment")
    assert start < end
    out = "\n".join(lines[:start] + lines[end + 1:])
    assert "FROM judgment" not in out and "FROM run WHERE lifecycle_state IN ('judged', 'ingested')" in out
    return out


def _pre_0020_head() -> str:
    """`v_paper_inputs_head` as it read before migration 0020: the file's block (the item_withdrawal member, the re-filed-row member that
    seeks item_result from it, and their comments) runs from the comment that opens it to the line before the 0015 item member's -- taken
    out of the pre-0021 text (`_pre_0021_head`)."""
    lines = _pre_0021_head().splitlines()
    start = next(i for i, ln in enumerate(lines) if "a producer's item withdrawals" in ln)
    end = next(i for i, ln in enumerate(lines) if "one seek through the partial index" in ln)
    assert start < end
    return "\n".join(lines[:start] + lines[end:])


def _pre_rrf_views() -> dict:
    """The seven objects the record-refusal-fact migration recreates, as they read before it: the status rule alone where the three
    judgment-currency rules render together now (`schema_def.judgment_currency_sql`), the measurement-judge rule by status and arithmetic
    alone, the item objects' one-probe predicate, no instrument join in the ownership object, the worklist as 0021 wrote it (its file's
    own CREATE VIEW text: no identity test on the covering judgment, no `identity_stale` reason), and the roster match as 0016
    wrote it (the covering judgment by status and arithmetic alone; the review of the first rendering)."""
    cur = S.judgment_currency_sql("rm.judgment_id", "jm", "r", "i")
    d638 = cur.split("\n  -- a partial judgment")[0]                                     # the first conjunct, with its two comment lines
    assert d638.count("AND ") == 1 and S.non_current_judgment_sql("rm.judgment_id", "jm.status") in d638
    old_live = (" AND (ir.judgment_id = '' OR NOT EXISTS (SELECT 1 FROM judgment jx WHERE jx.judgment_id = ir.judgment_id "
                "AND jx.status IN ('superseded', 'abandoned')))")
    live = S.item_judgment_live("item_result", "ir")
    msr_new = "  -- a row of a non-current judgment is in no scope: ownership counts current rows only\n  " + cur
    msr_old = ("  -- a row of a superseded or abandoned judgment is in no scope (migration 0021): ownership counts live rows only\n  AND "
               + S.non_current_judgment_sql("rm.judgment_id", "jm.status"))
    # the measurement-judge rule as 0016 / 0021 left it: the covering judgment by status and arithmetic alone (the identity conjunct is the
    # record-refusal-fact migration's, the review of the first rendering), in the frame and in the two comparison subqueries of the frame
    mj_ident = " AND " + S.judgment_identity_current_sql("jm", "r", "i")
    mj_new, mj_old = S.MEASUREMENT_JUDGE_SQL, S.MEASUREMENT_JUDGE_SQL.replace(mj_ident, "")
    assert mj_new.count(mj_ident) == 3 and mj_old != mj_new
    # v_roster_match as 0016 wrote it: the three covering-judgment probes without the identity conjunct, and without its comment tail
    rm_ident = S.judgment_identity_current_sql("jx", "r", "i")
    rm_old = re.sub(r"\s+AND " + re.escape(rm_ident), "", S.VIEW_SQL["v_roster_match"])
    rm_old = rm_old.replace("; the covering\n      -- judgment is identity-current: one bound at claim to a since-corrected run identity satisfies no entry", "")
    assert S.VIEW_SQL["v_roster_match"].count(rm_ident) == 3 and rm_ident not in rm_old and "D771" not in rm_old
    out = {"v_item_result_current": S.VIEW_SQL["v_item_result_current"].replace(live, old_live),
           "v_item_result_provenances": S.VIEW_SQL["v_item_result_provenances"].replace(live, old_live),
           "v_paper_long": S.VIEW_SQL["v_paper_long"].replace(cur, d638).replace(mj_new, mj_old),
           "v_paper_current": S.VIEW_SQL["v_paper_current"].replace(S.judgment_currency_sql("rm.judgment_id", "jm", "r", "i"), d638).replace(mj_new, mj_old),
           "v_roster_match": rm_old,
           "v_multi_source_rows": S.VIEW_SQL["v_multi_source_rows"].replace(msr_new, msr_old).replace("LEFT JOIN instrument i ON i.instrument_id = r.instrument_id\n", "")}
    with open([p for v, _, p in M.migration_files() if v == 21][0], "r", encoding="utf-8") as fh:
        out["v_pending_judgment"] = re.search(r"(?s)\nCREATE VIEW v_pending_judgment AS.*?;\n", fh.read()).group(0)
    assert "identity_stale" not in out["v_pending_judgment"] and "THEN 'coverage_short'" in out["v_pending_judgment"]
    for v, sql in out.items():
        assert "claimed_instrument_id" not in sql and "jcp." not in sql and sql != S.VIEW_SQL[v], v
    return out


def _strip_fo_shape(conn) -> None:
    """Take a database built from the CODE's render back to before the fold-observation migration: migration 0029's five tables first
    (`_strip_af_shape`: the newer file), then its one table dropped with its one index (nothing else of the code's shape names it: no
    column, no v_* object, no record kind)."""
    _strip_af_shape(conn)
    conn.execute("DROP INDEX IF EXISTS idx_fold_observation_written")
    conn.execute("DROP TABLE fold_observation")


def _strip_rrf_shape(conn) -> None:
    """Take a database built from the CODE's render back to before the record-refusal-fact migration: the fold-observation migration's
    table first (`_strip_fo_shape`: the newer file), then the six objects it recreates put back to their earlier text (DROP COLUMN
    re-parses every v_* object), then the five columns dropped (each declared after its table's last live column), then the fact table
    dropped with its two indexes."""
    _strip_fo_shape(conn)
    for v, sql in _pre_rrf_views().items():
        conn.execute(f"DROP VIEW {v}")
        conn.execute(sql)
    for t, cols in (("object", ("parent_object_id", "named_role")), ("judgment", ("claimed_instrument_id", "claimed_required_judge_id")),
                    ("record_refusal", ("judged_with",))):
        for c in cols:
            conn.execute(f"ALTER TABLE {t} DROP COLUMN {c}")
    conn.execute("DROP TABLE record_refusal_fact")


def _strip_0020_shape(conn) -> None:
    """Take a database built from the CODE's render back to before migration 0020 (the shapes the 0002 / 0003 fixtures describe
    predate it): the record-refusal-fact migration's objects first (`_strip_rrf_shape`), then the three objects 0020 recreates put back
    to their pre-0020 text (DROP COLUMN re-parses every v_* object, so the marks may be named by none), then item_result's two marks dropped,
    then the item_withdrawal table dropped. The two record-kind ledgers keep the code's CHECK lists (0020 rebuilds them regardless; the
    copy is the same either way)."""
    _strip_rrf_shape(conn)
    pre22 = _pre_rrf_views()                  # the two item objects at their earlier text, then 0020's marks taken out of THAT
    pre = {"v_item_result_current": pre22["v_item_result_current"].replace(" AND ir.withdrawn_by_record_sha256 IS NULL", "")
                                     .replace(" AND o.withdrawn_by_record_sha256 IS NULL", ""),
           "v_item_result_provenances": pre22["v_item_result_provenances"].replace(" AND ir.withdrawn_by_record_sha256 IS NULL", ""),
           "v_paper_inputs_head": _pre_0020_head()}
    for v, sql in pre.items():
        assert "withdrawn_by_record_sha256" not in sql and "item_withdrawal" not in sql, v
        conn.execute(f"DROP VIEW {v}")
        conn.execute(sql)
    for c in ("withdrawn_by_record_sha256", "withdrawn_reason"):
        conn.execute(f"ALTER TABLE item_result DROP COLUMN {c}")
    conn.execute("DROP TABLE item_withdrawal")


def _pre_fix_shape(path: str) -> None:
    """A database created from the OLD 0001 text, as the live one was: the object CHECK without the new label and a
    stale v_source_status body, at schema version 1 with the rest of the code's migrations NOT applied."""
    ddl = S.render_schema_sql().replace(OLD_LABEL_LIST, "", 1)
    assert OLD_LABEL_LIST not in ddl and "'difference_relu_distilled'" not in ddl
    current = S.VIEW_SQL["v_source_status"].strip()
    assert ddl.count(current) == 1
    ddl = ddl.replace(current, "CREATE VIEW v_source_status AS SELECT source_id FROM source_state;", 1)
    conn = DB.connect(path)
    conn.executescript(ddl)
    _strip_0020_shape(conn)
    conn.execute("INSERT INTO schema_migration(version, applied_utc, description) VALUES (1, '2026-09-18T22:10:08.091Z', 'initial')")
    conn.commit()
    conn.close()


#: The judge join 0003 gives v_pending_judgment; a schema-version-2 database carries the bare `JOIN judge jd` instead.
NEW_JUDGE_JOIN = ("-- a judge pairs with a run of an instrument that names a judge, or with a run it has filed a judgment for\n"
                  "JOIN judge jd ON (i.judge_id IS NOT NULL\n"
                  "                  OR EXISTS (SELECT 1 FROM judgment jj WHERE jj.run_id = r.run_id AND jj.judge_id = jd.judge_id))\n")


def _v2_shape(path: str) -> None:
    """A database at schema version 2: the code's shape except v_pending_judgment, whose judge join is the bare cross
    join 0003 scopes; migrations 1 and 2 recorded, 3 not."""
    ddl = S.render_schema_sql()
    assert ddl.count(NEW_JUDGE_JOIN) == 1
    ddl = ddl.replace(NEW_JUDGE_JOIN, "JOIN judge jd\n", 1)
    conn = DB.connect(path)
    conn.executescript(ddl)
    _strip_0020_shape(conn)
    conn.execute("INSERT INTO schema_migration(version, applied_utc, description) VALUES (1, '2026-09-18T22:10:08.091Z', 'initial')")
    conn.execute("INSERT INTO schema_migration(version, applied_utc, description) VALUES (2, '2026-09-19T21:00:00.000Z', "
                 "'object_paper_label_source_status_view')")
    conn.commit()
    conn.close()


def test_0003_scopes_the_pending_judgment_judge_join_on_a_v2_database(tmp_path):
    path = str(tmp_path / "r.sqlite")
    _v2_shape(path)
    conn = DB.connect(path)
    drift = DB.schema_drift(conn)
    assert "v_pending_judgment differs from the code's SQL" in drift                       # 0003's one object (0021 recreates it too)
    # 0020's, 0021's and the record-refusal-fact migration's v_roster_match lines aside, nothing else (the fixture is the code's shape
    # stripped back through _strip_rrf_shape, so that file's seventh recreate drifts here as its other six do through 0021's lines)
    assert [d for d in _sans_rrf_roster(_sans_0020(drift)) if d != V20_HEAD_DRIFT] == []
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(3, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(3, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 3").fetchone()[0] == "pending_judgment_judge_scope"
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


def _object_row(object_id: str, label: str, **over) -> dict:
    """A valid object row (the schema tests' per-kind example) under ``label``."""
    from test_registry_schema import _valid_object
    o = _valid_object("derived")             # no fit_model_id: the only outgoing FKs are to object itself
    o.update(object_id=object_id, paper_label=label, **over)
    return o


def _insert(conn, table: str, row: dict) -> None:
    from test_registry_schema import _insert_object
    assert table == "object"
    _insert_object(conn, row)


def _seed_objects(conn) -> None:
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert(conn, "object", _object_row("obj_a", "difference_relu"))
    _insert(conn, "object", _object_row("obj_b", "constitution_relu", superseded_by_object_id="obj_a"))   # a child of obj_a (self FK)
    conn.execute("PRAGMA foreign_keys=ON")


def test_0002_carries_the_pre_fix_shape_to_the_code_with_rows_intact(tmp_path):
    path = str(tmp_path / "r.sqlite")
    _pre_fix_shape(path)
    conn = DB.connect(path)
    _seed_objects(conn)
    n_before = conn.execute("SELECT COUNT(*) FROM object").fetchone()[0]
    drift_before = DB.schema_drift(conn)
    assert "table object: column paper_label CHECK (paper_label IN (...)) lacks 'difference_relu_distilled'" in drift_before
    assert "v_source_status differs from the code's SQL" in drift_before
    with pytest.raises(sqlite3.IntegrityError, match="paper_label IN"):
        _insert(conn, "object", _object_row("obj_new", "difference_relu_distilled"))     # the live refusal
    assert M.pending_migrations(conn)[0][0] == 2
    applied = M.apply_migrations(conn)
    assert applied == list(range(2, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM object").fetchone()[0] == n_before
    assert conn.execute("SELECT paper_label FROM object WHERE object_id = 'obj_a'").fetchone()[0] == "difference_relu"
    _insert(conn, "object", _object_row("obj_new", "difference_relu_distilled"))       # accepted now
    assert M.foreign_key_violations(conn, S.ALL_TABLES) == []
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    assert conn.execute("SELECT sql FROM sqlite_master WHERE name = 'object'").fetchone()[0].startswith('CREATE TABLE "object"')
    assert [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'object' AND sql IS NOT NULL")] == ["idx_object_artifact_sha"]
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


def test_0002_on_a_fresh_database_is_a_no_op_in_effect_and_equals_schema_sql():
    a = DB.connect_memory()
    assert M.apply_migrations(a) == list(range(1, M.code_version() + 1))
    b = DB.connect_memory()
    b.executescript(S.render_schema_sql())
    assert DB.ddl_master(a) == DB.ddl_master(b)


def _with_migrations_dir(monkeypatch, tmp_path, files: dict) -> None:
    d = tmp_path / "mig"
    d.mkdir(exist_ok=True)
    for old in d.iterdir():
        old.unlink()
    for fn, body in files.items():
        (d / fn).write_text(body, encoding="utf-8")
    monkeypatch.setattr(M, "HERE", str(d))


def test_runner_rolls_back_a_migration_that_leaves_a_foreign_key_violation(monkeypatch, tmp_path):
    """foreign_keys is OFF while a file runs (the rebuild dance needs it), so the runner checks afterwards: a rebuilt
    parent that lost a row its children name rolls the whole file back, version row included."""
    path = str(tmp_path / "r.sqlite")
    DB.open_fresh(path).close()
    conn = DB.connect(path)
    _seed_objects(conn)                # obj_b.superseded_by_object_id -> obj_a
    conn.commit()
    conn.close()
    orig = M.migration_files()
    files = {os.path.basename(p): open(p, encoding="utf-8").read() for _, _, p in orig}
    bad = M.render_table_rebuild("object", S.TABLE_BY_NAME["object"].column_names).replace("SELECT", "SELECT", 1)
    bad = bad.replace("FROM object ORDER BY object_id;", "FROM object WHERE object_id <> 'obj_a' ORDER BY object_id;", 1)
    files[f"{len(orig) + 1:04d}_drop_a_parent.sql"] = bad
    _with_migrations_dir(monkeypatch, tmp_path, files)
    conn = DB.connect(path)
    with pytest.raises(M.MigrationFailed, match="foreign-key violation"):
        M.apply_migrations(conn)
    assert M.db_version(conn) == len(orig)
    assert conn.execute("SELECT COUNT(*) FROM object").fetchone()[0] == 2 and not conn.in_transaction
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert DB.schema_drift(conn) == []


def test_runner_refuses_a_file_that_manages_its_own_transaction_or_fails_midway(monkeypatch, tmp_path):
    """A COMMIT inside a file would make what precedes it durable before the runner's checks could roll it back: such
    a file is refused BEFORE anything runs. A statement that fails midway rolls the whole file back, version row too."""
    path = str(tmp_path / "r.sqlite")
    DB.open_fresh(path).close()
    orig = M.migration_files()
    files = {os.path.basename(p): open(p, encoding="utf-8").read() for _, _, p in orig}
    nxt = f"{len(orig) + 1:04d}"
    for body, match in (("CREATE TABLE zz (x TEXT);\nCOMMIT;\n", "COMMIT"), ("PRAGMA foreign_keys=ON;\nCREATE TABLE zz (x TEXT);\n", "PRAGMA"),
                        ("-- a comment mentioning COMMIT; is fine\nCREATE TABLE zz (x TEXT CHECK (x <> 'BEGIN;'));\nbegin;\n", "begin")):
        files[f"{nxt}_reserved.sql"] = body
        _with_migrations_dir(monkeypatch, tmp_path, files)
        conn = DB.connect(path)
        with pytest.raises(M.MigrationFailed, match=f"statement the runner owns .*{match}"):
            M.apply_migrations(conn)
        assert M.db_version(conn) == len(orig) and not conn.in_transaction
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'zz'").fetchone() is None     # nothing ran
        conn.close()
    files.pop(f"{nxt}_reserved.sql")
    assert M.reserved_statement("CREATE TABLE zz (x TEXT CHECK (x <> 'BEGIN;'));  -- COMMIT; in a comment\n") is None
    files[f"{nxt}_bad_stmt.sql"] = "CREATE TABLE zz (x TEXT);\nCREATE TABLE zz (x TEXT);\n"
    _with_migrations_dir(monkeypatch, tmp_path, files)
    conn = DB.connect(path)
    with pytest.raises(M.MigrationFailed, match="already exists"):
        M.apply_migrations(conn)
    assert M.db_version(conn) == len(orig) and not conn.in_transaction
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'zz'").fetchone() is None     # the file rolled back whole
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_generated_migration_takes_a_drifted_database_to_zero_drift(monkeypatch, tmp_path):
    """The plan names the rebuild and the objects; the rendered body, applied by the runner as the next file, leaves no
    drift — for a CHECK list, a NOT NULL, a stale index, a stale trigger, a stale v_* object and an index by hand."""
    ddl = S.render_schema_sql().replace(OLD_LABEL_LIST, "", 1)
    ddl = ddl.replace("  campaign TEXT NOT NULL,\n", "  campaign TEXT,\n", 1)
    ddl = ddl.replace("CREATE INDEX idx_run_campaign ON run (campaign);", "CREATE INDEX idx_run_campaign ON run (campaign, model_id);", 1) \
        if "CREATE INDEX idx_run_campaign ON run (campaign);" in ddl else ddl
    ddl = ddl.replace("SELECT RAISE(ABORT, 'run.lifecycle_state=judged requires a judgment row for this run_id');", "SELECT RAISE(ABORT, 'old');", 1)
    path = str(tmp_path / "r.sqlite")
    conn = DB.connect(path)
    conn.executescript(ddl)
    conn.execute("DROP VIEW v_in_flight")
    conn.execute("CREATE VIEW v_in_flight AS SELECT run_id FROM run")
    conn.execute("CREATE INDEX idx_by_hand ON run (campaign)")
    for v in range(1, M.code_version() + 1):
        conn.execute("INSERT INTO schema_migration(version, applied_utc, description) VALUES (?, '2026-09-18T22:10:08.091Z', 'x')", (v,))
    _seed_objects(conn)
    conn.commit()
    plan = M.shape_plan(conn)
    # `campaign TEXT NOT NULL` first appears in fit_search: two tables to rebuild, in schema_def order
    assert [(r["table"], r["rows"]) for r in plan["rebuild"]] == [("fit_search", 0), ("object", 2)]
    assert plan["objects"]["view"]["recreate"] == ["v_in_flight"] and plan["objects"]["index"]["drop"] == ["idx_by_hand"]
    assert "trg_run_judged_requires_judgment_ins" in plan["objects"]["trigger"]["recreate"]
    body = M.render_shape_migration(conn, plan, title="test")
    assert body.startswith("-- test\n-- drift: ") and "CREATE TABLE object__new (" in body and "DROP INDEX IF EXISTS idx_by_hand;" in body
    orig = M.migration_files()
    files = {os.path.basename(p): open(p, encoding="utf-8").read() for _, _, p in orig}
    files[f"{len(orig) + 1:04d}_generated.sql"] = body
    _with_migrations_dir(monkeypatch, tmp_path, files)
    assert M.apply_migrations(conn) == [len(orig) + 1]
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT paper_label FROM object WHERE object_id = 'obj_a'").fetchone()[0] == "difference_relu"
    assert M.render_shape_migration(conn) == ""


def test_render_table_rebuild_names_new_and_dropped_columns():
    t = S.TABLE_BY_NAME["object"]
    live_cols = [c for c in t.column_names if c != "campaign"] + ["old_col"]
    body = M.render_table_rebuild("object", live_cols)
    assert "-- column campaign is new in the code: NOT NULL without a DEFAULT: the INSERT below needs a value for it" in body
    assert "-- column old_col is not declared by the code: its values are NOT copied" in body
    copied = body.split("INSERT INTO object__new (", 1)[1].split(")", 1)[0].split(", ")
    assert "campaign" not in copied and "old_col" not in copied and "campaign_cell_id_legacy" in copied
    assert body.rstrip().endswith("ANALYZE object;")
    assert "difference_relu_distilled" in body and all(v in body for v in V.PAPER_LABELS)


def _valid_object_with_model() -> dict:
    from test_registry_schema import _valid_object
    o = _valid_object("corpus_document_fit")
    o.update(object_id="o_parentless", fit_model_id="no_such_model")
    return o


def test_runner_restores_the_callers_pragmas_instead_of_forcing_them():
    """A connection opened with foreign_keys=OFF (the adapter tests insert object rows without their parents) is still
    OFF after the migrations ran; one opened ON is still ON; legacy_alter_table goes back to what it was. The runner's
    first version forced ON/OFF and four declared-adapter tests failed on `FOREIGN KEY constraint failed` (2026-09-19)."""
    off = DB.connect_memory()
    off.execute("PRAGMA foreign_keys=OFF")
    M.apply_migrations(off)
    assert off.execute("PRAGMA foreign_keys").fetchone()[0] == 0 and off.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    _insert(off, "object", _valid_object_with_model())      # fit_model_id names a model row nobody inserted: allowed with FKs off
    on = DB.connect_memory()
    on.execute("PRAGMA legacy_alter_table=ON")
    M.apply_migrations(on)
    assert on.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and on.execute("PRAGMA legacy_alter_table").fetchone()[0] == 1



# --------------------------------------------------------------------------------------------
# 0006: the completion flag and the contrast cap pair, on a schema-version-5 database (2026-09-21)
# --------------------------------------------------------------------------------------------

V6_OBJECTS = ("v_run_wide", "v_producer_key", "v_paper_current", "v_paper_long")


def _v5_shape(path: str) -> None:
    """A database at schema version 5: migrations 0001..0005 applied by the runner itself, 0006 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 5:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def _columns(conn, name: str):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({name})").fetchall()]


def test_0006_adds_is_complete_and_the_contrast_cap_pair_on_a_v5_database(tmp_path):
    """The four objects differ and nothing else; after the file: no drift, `is_complete` beside `lifecycle_state` on each, the cap pair
    on v_paper_long, run_contrast untouched, the writers' open accepts the database."""
    path = str(tmp_path / "r.sqlite")
    _v5_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 5
    # the drift is against the CODE's shape, so a later migration's objects drift here too (0007: v_pending_judgment, v_roster_match);
    # 0006's four are among them, in the code's order, and nothing but v_* objects differs
    raw = DB.schema_drift(conn)
    drift = _sans_0016(raw)
    v6 = [f"{v} differs from the code's SQL" for v in V6_OBJECTS]
    assert [d for d in raw if d in v6] == v6, raw                                           # 0021 recreates two of the four: read them raw
    assert all((d.startswith("v_") and d.endswith("differs from the code's SQL")) or d in V12_TABLE_DRIFT + V15_DRIFT for d in drift), drift
    assert all(d in drift for d in V12_TABLE_DRIFT + V15_DRIFT) and V12_VIEW_DRIFT in drift   # 0012's instrument axis and 0015's head drift on a v5 shape too
    for v in V6_OBJECTS:
        assert "lifecycle_state" in _columns(conn, v) and "is_complete" not in _columns(conn, v), v
    assert not {"anchor_token_cap_tokens", "cap_matched"} & set(_columns(conn, "v_paper_long"))
    contrast_before = _columns(conn, "run_contrast")
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(6, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(6, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 6").fetchone()[0] == "completion_flag_contrast_cap"
    for v in V6_OBJECTS:
        cols = _columns(conn, v)
        assert cols.index("lifecycle_state") < cols.index("is_complete"), v
    long_cols = _columns(conn, "v_paper_long")
    assert long_cols.index("anchor_run_id") < long_cols.index("anchor_token_cap_tokens") < long_cols.index("cap_matched")
    assert _columns(conn, "run_contrast") == contrast_before
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0007: coverage counts the items whose judging errored, on a schema-version-6 database (2026-09-21)
# --------------------------------------------------------------------------------------------

V7_OBJECTS = ("v_pending_judgment", "v_roster_match", "v_paper_long")


def _v6_shape(path: str) -> None:
    """A database at schema version 6: migrations 0001..0006 applied by the runner itself, 0007 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 6:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


#: the three tables migration 0018 rebuilds (`slurm` in their substrate CHECK): their stored CREATE text is compared in its
#: comparable form (`db.normalize_ddl`) with that one value removed, so every per-version test below still asserts that ITS
#: migration touched no table after the whole chain runs (tests/test_migration_window.py owns 0018 itself)
_V18_TABLES = ("run", "object", "fit_search")
_SLURM_IN_CHECK = re.compile(r",'slurm'")   # the normalised form has no space after a comma
#: the two tables migration 0031 rebuilds (`non_template` in their applied-positions CHECK): `run` is compared in the comparable form
#: above with this value removed too; `instrument` is compared RAW with the value removed (the 0014 test reads its pairing CONSTRAINT
#: in the stored spelling), so every per-version test below still asserts that ITS migration touched neither table
_NTPL_IN_CHECK = re.compile(r",'non_template'")
_NTPL_IN_RAW = re.compile(r", 'non_template'")
#: the tables migration 0032 rebuilds (the site-label rename) that no rule above already compares in the comparable form: their stored
#: CREATE text after 0032 is the rebuild's (quoted name, the code's comments), so they are compared as normalised DDL, which 0032 leaves
#: unchanged on a shape built from these files
_SLR_TABLES = ("mirror_ack", "analysis")
#: every table migration 0032 rebuilds: a check that a table kept its b-tree since an earlier file reads only the tables outside this set
SLR_REBUILT_TABLES = ("fit_search", "object", "run", "mirror_ack", "analysis")


def _object_sql(conn, names):
    out = {}
    for n in names:
        sql = str(conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (n,)).fetchone()[0])
        if n in _V18_TABLES:
            sql = _SLURM_IN_CHECK.sub("", DB.normalize_ddl(sql))
            sql = _NTPL_IN_CHECK.sub("", sql) if n in NTPL_TABLES else sql
        elif n in NTPL_TABLES:
            sql = _NTPL_IN_RAW.sub("", sql)
        elif n in _SLR_TABLES:
            sql = DB.normalize_ddl(sql)
        out[n] = sql
    return out


def test_0007_counts_errored_items_toward_coverage_on_a_v6_database(tmp_path):
    """The three objects that read the coverage predicate differ and nothing else; before the file none of them reads `n_errors`;
    after it every one reads `n_errors` and the judgment's own `n_error_outcomes`, no drift remains, no table changed, and the
    writers' open accepts the database."""
    path = str(tmp_path / "r.sqlite")
    _v6_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 6
    # the drift is against the CODE's shape, so a later migration's object drifts here too (0008: v_item_headline); 0007's three are
    # among them, in the code's order, and nothing but v_* objects differs
    raw = DB.schema_drift(conn)
    drift = _sans_0016(raw)
    v7 = [f"{v} differs from the code's SQL" for v in V7_OBJECTS]
    assert [d for d in raw if d in v7] == v7, raw                                           # 0021 recreates two of the three: read them raw
    assert all((d.startswith("v_") and d.endswith("differs from the code's SQL")) or d in V12_TABLE_DRIFT + V15_DRIFT for d in drift), drift
    before = _object_sql(conn, V7_OBJECTS)
    assert all("n_errors" not in sql for sql in before.values()), "the v6 objects read judged + declined only"
    tables_before = _object_sql(conn, ("run_metric", "run"))                              # judgment is the record-refusal-fact migration's
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(7, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(7, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 7").fetchone()[0] == "judgment_coverage_counts_errors"
    after = _object_sql(conn, V7_OBJECTS)
    for name, sql in after.items():
        assert "n_errors" in sql and "n_error_outcomes" in sql, name
    assert "'n_errors', j2.n_errors" in after["v_pending_judgment"]
    # `n_incomplete` is not part of the predicate (v_paper_long lists it as a metric column; the two predicate-only objects never name it)
    assert "n_incomplete" not in after["v_pending_judgment"] and "n_incomplete" not in after["v_roster_match"]
    assert _object_sql(conn, ("run_metric", "run")) == tables_before
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0008: v_item_headline accepts the eval's LISTED headline metric ids, on a schema-version-7 database (2026-09-21)
# --------------------------------------------------------------------------------------------


def _v7_shape(path: str) -> None:
    """A database at schema version 7: migrations 0001..0007 applied by the runner itself, 0008 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 7:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0008_item_headline_accepts_the_listed_metric_ids_on_a_v7_database(tmp_path):
    """The one object that read `e.headline_metric_id = ir.metric_id` differs and nothing else; after the file it reads the eval's
    `extra_json.headline_metric_ids` (the most preferred listed id the run files, else the declared one), no drift remains, no table
    changed, and the writers' open accepts the database."""
    path = str(tmp_path / "r.sqlite")
    _v7_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 7
    # the drift is against the CODE's shape, so a later migration's objects drift here too (0009: v_paper_current / v_paper_long);
    # 0008's one object is among them and nothing but v_* objects differs
    drift = _sans_0016(DB.schema_drift(conn))
    assert "v_item_headline differs from the code's SQL" in drift and all((d.startswith("v_") and d.endswith("differs from the code's SQL")) or d in V12_TABLE_DRIFT + V15_DRIFT for d in drift), drift
    before = _object_sql(conn, ("v_item_headline",))["v_item_headline"]
    assert "e.headline_metric_id = ir.metric_id" in before and "headline_metric_ids" not in before
    tables_before = _object_sql(conn, ("eval", "run"))                                      # item_result is 0020's
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(8, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(8, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 8").fetchone()[0] == "item_headline_accepted_metric_ids"
    after = _object_sql(conn, ("v_item_headline",))["v_item_headline"]
    assert "$.headline_metric_ids" in after and "e.headline_metric_id = ir.metric_id" not in after
    assert _object_sql(conn, ("eval", "run")) == tables_before
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0009: the paper views carry the decode family default beside the derived match, on a schema-version-8 database (2026-09-21)
# --------------------------------------------------------------------------------------------


def _v8_shape(path: str) -> None:
    """A database at schema version 8: migrations 0001..0008 applied by the runner itself, 0009 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 8:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0009_paper_views_carry_the_decode_family_default_on_a_v8_database(tmp_path):
    """Exactly the two paper views differ (neither carried `decode_is_family_default`); after the file both render
    `decode_family_default_policy_id` / `decode_is_family_default` (schema_def.decode_family_default_sql), no drift remains, no table changed,
    and the writers' open accepts the database. View-only: nothing is copied."""
    path = str(tmp_path / "r.sqlite")
    _v8_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 8
    # the drift is against the CODE's shape, so a later migration's object drifts here too (0010: v_multi_source_rows; 0011: v_roster_match);
    # 0009's two are among them and nothing but v_* objects differs
    raw = DB.schema_drift(conn)
    assert "v_paper_current differs from the code's SQL" in raw and "v_paper_long differs from the code's SQL" in raw   # 0021 recreates both: read raw
    assert "v_multi_source_rows differs from the code's SQL" in raw                                                  # 0010's and 0021's: read raw
    assert _sans_0016(raw) == V12_TABLE_DRIFT + [V12_VIEW_DRIFT, "v_roster_match differs from the code's SQL"] + V15_DRIFT
    before = _object_sql(conn, ("v_paper_current", "v_paper_long"))
    assert all("decode_is_family_default" not in sql for sql in before.values())
    tables_before = _object_sql(conn, ("run", "eval", "decode_policy", "run_metric"))
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(9, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(9, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 9").fetchone()[0] == "paper_views_decode_family_default"
    after = _object_sql(conn, ("v_paper_current", "v_paper_long"))
    for name, sql in after.items():
        assert "decode_family_default_policy_id" in sql and "decode_is_family_default" in sql and "p.is_default = 1" in sql, name
    assert _object_sql(conn, ("run", "eval", "decode_policy", "run_metric")) == tables_before
    # the columns are there to select, on an empty database too
    cols = {name: [c[1] for c in conn.execute(f"PRAGMA table_info({name})")] for name in ("v_paper_current", "v_paper_long")}
    for name, cs in cols.items():
        assert {"decode_source", "decode_policy_id", "decode_policy_match", "decode_family_default_policy_id", "decode_is_family_default"} <= set(cs), name
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it



# --------------------------------------------------------------------------------------------
# 0010: the paper views fill a NULL frac_* mirror from run_degeneration, on a schema-version-9 database (2026-09-21)
# --------------------------------------------------------------------------------------------


def _v9_shape(path: str) -> None:
    """A database at schema version 9: migrations 0001..0009 applied by the runner itself, 0010 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 9:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0010_paper_views_read_degeneration_from_run_degeneration_on_a_v9_database(tmp_path):
    """Exactly the two objects that expose the frac_* mirrors differ (neither read `run_degeneration`); after the file both carry the
    `degen_rate` pass over v_run_degeneration_current and COALESCE every mirror with it, no drift remains, no table changed, and the
    writers' open accepts the database. View-only: nothing is copied."""
    path = str(tmp_path / "r.sqlite")
    _v9_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 9
    # the drift is against the CODE's shape, so 0011's two roster objects drift here too; 0010's two are among them
    raw = DB.schema_drift(conn)
    assert "v_paper_current differs from the code's SQL" in raw and "v_paper_long differs from the code's SQL" in raw   # 0021 recreates both: read raw
    assert "v_multi_source_rows differs from the code's SQL" in raw                                                  # 0010's own, and 0021's: read raw
    assert _sans_0016(raw) == V12_TABLE_DRIFT + [V12_VIEW_DRIFT, "v_roster_match differs from the code's SQL"] + V15_DRIFT
    before = _object_sql(conn, ("v_multi_source_rows", "v_paper_long"))
    assert all("degen_rate" not in sql and "v_run_degeneration_current" not in sql for sql in before.values())
    tables_before = _object_sql(conn, ("run", "run_metric", "run_degeneration", "observation"))
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(10, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(10, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 10").fetchone()[0] == "paper_views_degeneration_from_run_degeneration"
    after = _object_sql(conn, ("v_multi_source_rows", "v_paper_long"))
    for name, sql in after.items():
        assert "degen_rate AS MATERIALIZED" in sql and "FROM v_run_degeneration_current WHERE measurement_state = 'measured'" in sql, name
        assert "LEFT JOIN degen_rate dj ON dj.run_id = rm.run_id AND dj.judgment_id = rm.judgment_id" in sql, name
        assert "LEFT JOIN degen_rate dn ON dn.run_id = rm.run_id AND dn.judgment_id = ''" in sql, name
    for col in V.DEGENERATION_FRAC_COLUMNS:
        assert f"COALESCE(rm.{col}, dj." in after["v_paper_long"], col
    assert "COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5, dj.loop_or_at_cap, dn.loop_or_at_cap, json_extract(rm.extra_json, '$.degenerate_share')) AS degenerate_share" in after["v_multi_source_rows"]
    assert _object_sql(conn, ("run", "run_metric", "run_degeneration", "observation")) == tables_before
    # the columns are there to select, on an empty database too
    cols = [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)")]
    assert set(V.DEGENERATION_FRAC_COLUMNS) <= set(cols)
    assert {"degenerate_share", "loop_share", "loop_share_measure", "at_cap_share"} <= {c[1] for c in conn.execute("PRAGMA table_info(v_multi_source_rows)")}
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0011: the operating_point strength rule falls back to nominal 1.0 / Arditi c = 0.3 when no stamp was filed, on a schema-version-10
# database (2026-09-22; registry ruling operating-point-default-fitted-strength-1p0-arditi-c0p3-2026-09-22)
# --------------------------------------------------------------------------------------------


def _v10_shape(path: str) -> None:
    """A database at schema version 10: migrations 0001..0010 applied by the runner itself, 0011 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 10:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0011_roster_match_falls_back_to_nominal_1p0_and_arditi_c0p3_on_a_v10_database(tmp_path):
    """Exactly the three objects that read or expose the operating point differ (v_roster_match admitted a stamp of 1 or an unsteered /
    prompted kind and nothing else; neither paper object carried `operating_point_basis`); after the file v_roster_match carries the
    fallback CASE of schema_def.operating_point_sql, both paper objects expose `operating_point_basis`, no drift remains, no table changed,
    and the writers' open accepts the database. View-only: nothing is copied."""
    path = str(tmp_path / "r.sqlite")
    _v10_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 10
    # the drift is against the CODE's shape, so 0012's instrument axis drifts here too; 0011's three are among them
    raw = DB.schema_drift(conn)
    assert "v_paper_current differs from the code's SQL" in raw and "v_paper_long differs from the code's SQL" in raw   # 0021 recreates both: read raw
    assert _sans_0016(raw) == V12_TABLE_DRIFT + [V12_VIEW_DRIFT, "v_roster_match differs from the code's SQL"] + V15_DRIFT
    before = _object_sql(conn, ("v_roster_match", "v_paper_current", "v_paper_long"))
    assert "(r.is_operating_point = 1 OR o.object_kind IN ('none', 'system_instruction'))" in before["v_roster_match"]
    assert all("operating_point_basis" not in sql and "fallback_nominal_1p0" not in sql for sql in before.values())
    tables_before = _object_sql(conn, ("run", "run_metric", "ruling"))                          # roster_entry is 0016's (rebuilt past 15); object the record-refusal-fact migration's
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(11, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(11, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 11").fetchone()[0] == "roster_match_operating_point_fallback"
    after = _object_sql(conn, ("v_roster_match", "v_paper_current", "v_paper_long"))
    kinds = ", ".join(f"'{k}'" for k in S.OPERATING_POINT_FALLBACK_OBJECT_KINDS)
    for name, sql in after.items():
        assert "CASE WHEN r.is_operating_point = 1 THEN 'stamped'" in sql, name
        assert f"WHEN r.is_operating_point IS NULL AND o.object_kind IN ({kinds})" in sql, name
        assert "AND ABS(r.strength_nominal - 1.0) < 1e-9 THEN 'fallback_nominal_1p0'" in sql, name
        assert "WHEN r.is_operating_point IS NULL AND o.object_kind = 'vector_all'" in sql and \
               "AND ABS(r.strength_coefficient - 0.3) < 1e-9 THEN 'fallback_arditi_c0p3'" in sql, name
        assert "WHEN o.object_kind IN ('none', 'system_instruction') THEN 'no_strength_axis'" in sql, name
    assert "(r.is_operating_point = 1 OR o.object_kind IN ('none', 'system_instruction'))" not in after["v_roster_match"]
    assert "re.strength_rule = 'operating_point' AND ((CASE WHEN" in after["v_roster_match"]
    assert "AS operating_point_basis" in after["v_paper_current"] and "AS operating_point_basis" in after["v_paper_long"]
    assert "JOIN object o ON o.object_id = r.object_id\nJOIN v_run_current vc ON vc.run_id = r.run_id" in after["v_paper_current"]   # the fold is v_run_current's since 0021
    assert "FROM run r\n    LEFT JOIN per_run_cover pc" in S.VIEW_SQL["v_run_current"]
    assert _object_sql(conn, ("run", "run_metric", "ruling")) == tables_before             # object is the record-refusal-fact migration's
    # the column is there to select, on an empty database too
    for v in ("v_paper_current", "v_paper_long"):
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({v})")]
        assert "operating_point_basis" in cols and cols.index("operating_point_basis") == cols.index("is_operating_point") + 1, v
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


def _v11_shape(path: str) -> None:
    """A database at schema version 11: migrations 0001..0011 applied by the runner itself, 0012 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 11:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def _valid_row(table: str, **over) -> dict:
    """A row that satisfies every NOT NULL, CHECK and pattern of ``table`` (the first vocabulary value, a sample sha / stamp, a
    placeholder), with ``over`` on top -- enough to seed a parent row a rebuild's foreign-key check must find."""
    t = S.TABLE_BY_NAME[table]
    row = {}
    for c in t.columns:
        if c.name in over:
            row[c.name] = over[c.name]
        elif not c.nn or c.default is not None:
            continue
        elif c.vocab:
            row[c.name] = V.VOCABULARIES[c.vocab][0]
        elif c.pattern == "SUBSAMPLE_RULE":
            row[c.name] = V.SUBSAMPLE_RULE_FIXED[0]
        elif c.pattern == "FIT_LR_SELECTED_BY":
            row[c.name] = V.FIT_LR_SELECTED_BY_FIXED[0]
        elif c.pattern == "STRENGTH_RULE":
            row[c.name] = "operating_point"
        elif c.pattern in ("SHA256", "SHA256_OR_UNVERIFIED"):
            row[c.name] = "a" * 64
        elif c.pattern == "SHA1_OR_UNVERIFIED":
            row[c.name] = "unverified"
        elif c.pattern in ("UTC_MS", "UTC_MS_OR_UNVERIFIED"):
            row[c.name] = "2026-01-01T00:00:00.000Z"
        elif c.pattern == "JSON":
            row[c.name] = "[]"
        elif c.type == "INTEGER":
            row[c.name] = 1
        elif c.type == "REAL":
            row[c.name] = 1.0
        elif c.type == "BLOB":
            row[c.name] = b"\x00" * 32
        else:
            row[c.name] = "x"
    for k, v in over.items():
        row.setdefault(k, v)
    return row


def _insert_row(conn, table: str, row: dict) -> None:
    conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})", list(row.values()))


def test_0012_instrument_padding_identity_on_a_v11_database(tmp_path):
    """The `instrument` table gains `padding_tokens` / `padding_identity_sha256` and the pairing CHECK; `v_paper_long` exposes them
    beside the cap and the window, `v_run_wide` their instrument-prefixed twins; exactly those three objects differ before the file (plus
    0013's `v_roster_match`, the drift being against the code's shape) and nothing after it. The table is REBUILT with its rows intact -- every copied instrument keeps its id text and reads NULL on both new
    columns -- the foreign keys of the rows that name an instrument still hold, the CHECK refuses a pad without its contract (a pad of 0 with
    its contract is 0014's zero point and is checked there), and the writers' open accepts the database."""
    from sisteer.registry import identity as I
    path = str(tmp_path / "r.sqlite")
    _v11_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 11
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_row(conn, "eval", _valid_row("eval", eval_id="ev", headline_metric_id="asr_x", producer_spellings_json="[]"))
    _insert_row(conn, "item_set", _valid_row("item_set", item_set_id="is", eval_id="ev"))
    _insert_row(conn, "judge", _valid_row("judge", judge_id="j" * 64))
    inst_a = _valid_row("instrument", instrument_id="i" * 64, eval_id="ev", item_set_id="is", judge_id="j" * 64, token_cap_tokens=4096,
                        context_window_tokens=4864, notes="unpadded, judged")
    inst_b = _valid_row("instrument", instrument_id="k" * 64, eval_id="ev", item_set_id="is", judge_id=None, token_cap_tokens=512)
    _insert_row(conn, "instrument", inst_a)
    _insert_row(conn, "instrument", inst_b)
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    n_before = conn.execute("SELECT COUNT(*) FROM instrument").fetchone()[0]
    assert n_before == 2
    raw = DB.schema_drift(conn)
    drift = _sans_0016(raw)
    # the drift is against the CODE's shape, so 0013's v_roster_match and 0015's head + index drift here too (and 0016's, stripped); 0012's two views are the other two
    # (v_paper_long is 0021's too, so its line is read on the raw drift)
    assert "v_paper_long differs from the code's SQL" in raw
    assert drift == ["table instrument lacks column(s) padding_tokens, padding_identity_sha256",
                     "table instrument: CONSTRAINT instrument_padding_pairing is absent",
                     "v_run_wide differs from the code's SQL", "v_roster_match differs from the code's SQL"] + V15_DRIFT, drift
    before = _object_sql(conn, ("v_run_wide", "v_paper_long"))
    assert all("padding" not in sql for sql in before.values())
    cols_before = [c[1] for c in conn.execute("PRAGMA table_info(instrument)")]
    assert "padding_tokens" not in cols_before
    # (v_roster_match is 0013's: it changes when the runner carries the database past 12, so it is checked by the 0013 test, not here)
    others_before = _object_sql(conn, ("run", "eval", "item_set", "judge", "run_metric", "v_instrument_mismatch"))   # roster_entry is 0016's; v_paper_current / v_pending_judgment 0021's
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(12, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(12, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 12").fetchone()[0] == "instrument_padding_identity"
    # rows intact: the same ids, the copied columns unchanged, the new ones NULL
    assert conn.execute("SELECT COUNT(*) FROM instrument").fetchone()[0] == n_before
    rows = {r[0]: r for r in conn.execute("SELECT instrument_id, token_cap_tokens, context_window_tokens, judge_id, notes, padding_tokens, padding_identity_sha256 FROM instrument")}
    assert rows["i" * 64][1:] == (4096, 4864, "j" * 64, "unpadded, judged", None, None)
    assert rows["k" * 64][1:] == (512, None, None, None, None, None)
    cols_after = [c[1] for c in conn.execute("PRAGMA table_info(instrument)")]
    assert cols_after.index("padding_tokens") == cols_after.index("context_window_tokens") + 1
    assert cols_after.index("padding_identity_sha256") == cols_after.index("padding_tokens") + 1
    assert cols_after == [c for c in cols_after if c not in ("padding_tokens", "padding_identity_sha256")][:cols_after.index("padding_tokens")] + \
        ["padding_tokens", "padding_identity_sha256"] + cols_before[cols_before.index("decode_source"):]
    assert conn.execute("SELECT sql FROM sqlite_master WHERE name = 'instrument'").fetchone()[0].startswith('CREATE TABLE "instrument"')
    assert M.foreign_key_violations(conn, S.ALL_TABLES) == []
    # exactly the two views changed; every other object that names `instrument` (or nothing) is byte-identical
    after = _object_sql(conn, ("v_run_wide", "v_paper_long"))
    assert "i.padding_tokens AS instrument_padding_tokens, i.padding_identity_sha256 AS instrument_padding_identity_sha256" in after["v_run_wide"]
    assert "i.padding_tokens, i.padding_identity_sha256," in after["v_paper_long"]
    assert _object_sql(conn, tuple(others_before)) == others_before
    for v, col in (("v_paper_long", "padding_tokens"), ("v_paper_long", "padding_identity_sha256"), ("v_run_wide", "instrument_padding_tokens")):
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({v})")]
        assert col in cols, (v, col)
    pl = [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)")]
    assert pl.index("padding_tokens") == pl.index("context_window_tokens") + 1 and pl.index("token_cap_tokens") < pl.index("padding_tokens")
    # the CHECK as the runner leaves it (the database carried past 0012 to the code's version): a pad without its contract, or a contract
    # without its pad, is refused; a pad with its contract is accepted -- and so is a pad of 0 with its contract since 0014 (the zero point,
    # 0012 alone refused it, which the 0014 test shows on a schema-version-13 shape)
    contract = I.padding_identity_sha256({"filler_sha256": "7" * 64, "mode": "fixed", "placement": "before", "separator": "\n\n", "tolerance": 0.02})
    for bad in ({"padding_tokens": 16384}, {"padding_identity_sha256": contract}):
        with pytest.raises(sqlite3.IntegrityError, match="instrument_padding_pairing"):
            _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="b" * 64, eval_id="ev", item_set_id="is", **bad))
    _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="p" * 64, eval_id="ev", item_set_id="is", padding_tokens=16384, padding_identity_sha256=contract))
    conn.commit()
    assert tuple(conn.execute("SELECT padding_tokens, padding_identity_sha256 FROM instrument WHERE instrument_id = ?", ("p" * 64,)).fetchone()) == (16384, contract)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0013: a roster SELECTOR never matches a PADDED instrument (reached only by name), on a schema-version-12 database
# --------------------------------------------------------------------------------------------

#: the selector branch of v_roster_match as 0011 left it (the text 0012 did not touch) and as 0013 leaves it
V12_SELECTOR_BRANCH = "OR (re.instrument_id IS NULL\n               AND (json_extract(re.instrument_selector_json, '$.harness') IS NULL"
V13_SELECTOR_BRANCH = ("OR (re.instrument_id IS NULL\n               AND i.padding_tokens IS NULL\n"
                       "               AND (json_extract(re.instrument_selector_json, '$.harness') IS NULL")


def _v12_shape(path: str) -> None:
    """A database at schema version 12: migrations 0001..0012 applied by the runner itself, 0013 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 12:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0013_roster_selector_never_matches_a_padded_instrument_on_a_v12_database(tmp_path):
    """Exactly one object differs: v_roster_match, whose instrument SELECTOR branch admitted any instrument satisfying the selector fields
    (padding being a field no selector carries, a padded instrument satisfied the unpadded static entries of its eval and the 24 padded
    runs pooled into the ordinary static comparisons, live 2026-09-23). After the file the selector branch requires `i.padding_tokens IS
    NULL` and the named-instrument branch is untouched; no drift remains, no table changed, the three objects that read v_roster_match by
    name (v_paper_current, v_paper_long, v_roster_gaps) are byte-identical, and the writers' open accepts the database. View-only:
    nothing is copied."""
    path = str(tmp_path / "r.sqlite")
    _v12_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 12
    # the drift is against the CODE's shape, so 0014's instrument CHECK (`>= 0`) and 0015's head + index drift here too; 0013's own object is the one view
    drift = _sans_0017(_sans_0016(DB.schema_drift(conn)))                                  # 0017's v_paper_long drifts here too
    assert drift[1] == "v_roster_match differs from the code's SQL" and drift[2:] == V15_DRIFT and len(drift) == 4, drift
    assert drift[0].startswith("table instrument: CONSTRAINT instrument_padding_pairing differs") and "padding_tokens >= 0" in drift[0]
    before = _object_sql(conn, ("v_roster_match",))["v_roster_match"]
    assert V12_SELECTOR_BRANCH in before and "padding_tokens" not in before
    assert "AND (re.instrument_id = r.instrument_id" in before                                # the named branch predates 0013
    # (`instrument` is 0014's: its pairing CHECK changes when the runner carries the database past 13, so it is checked by the 0014 test, not here)
    others = ("run", "run_metric",                                                            # roster_entry / paper_round / v_roster_gaps are 0016's; v_paper_long 0017's; object / judgment the record-refusal-fact migration's
              "v_run_wide", "v_instrument_mismatch", "v_paper_objects")                       # v_paper_current / v_pending_judgment are 0021's
    others_before = _object_sql(conn, others)
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(13, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(13, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 13").fetchone()[0] == "roster_match_padded_instrument_by_name"
    after = _object_sql(conn, ("v_roster_match",))["v_roster_match"]
    assert V13_SELECTOR_BRANCH in after and V12_SELECTOR_BRANCH not in after
    assert after.count("i.padding_tokens IS NULL") == 1                                        # the selector branch, once; nowhere else
    assert "AND (re.instrument_id = r.instrument_id" in after                                  # the named branch is unchanged
    assert after == S.VIEW_SQL["v_roster_match"].strip().rstrip(";")
    assert _object_sql(conn, others) == others_before                                          # nothing else moved, tables included
    # the padding predicate reads the instrument the run names (the join 0001 declared), never a run column
    assert "JOIN instrument i ON i.instrument_id = r.instrument_id" in after and "r.padding_tokens" not in after
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0014: the padding axis has a ZERO POINT -- padding_tokens 0 WITH its contract hash is a row, on a schema-version-13 database
# --------------------------------------------------------------------------------------------

V13_PAIRING = "CONSTRAINT instrument_padding_pairing CHECK ((padding_tokens IS NULL) = (padding_identity_sha256 IS NULL) AND (padding_tokens IS NULL OR padding_tokens > 0))"
V14_PAIRING = "CONSTRAINT instrument_padding_pairing CHECK ((padding_tokens IS NULL) = (padding_identity_sha256 IS NULL) AND (padding_tokens IS NULL OR padding_tokens >= 0))"


def _v13_shape(path: str) -> None:
    """A database at schema version 13: migrations 0001..0013 applied by the runner itself, 0014 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 13:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0014_instrument_padding_zero_point_on_a_v13_database(tmp_path):
    """Exactly one object differs: the `instrument` table, whose pairing CHECK refused a pad of 0 (0012: no producer had run one). After the
    file it reads `>= 0`: a 0 WITH its contract hash is a row (the recipe's zero point), while a 0 alone, a contract alone and a negative
    pad are still refused and NULL still means no recipe at all. The table is REBUILT with its rows intact -- the unpadded and the padded
    instrument keep their ids, values and column order -- every view's text is byte-identical (none is recreated), no other table moves, the
    foreign keys hold, and the writers' open accepts the database."""
    from sisteer.registry import identity as I
    path = str(tmp_path / "r.sqlite")
    _v13_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 13
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_row(conn, "eval", _valid_row("eval", eval_id="ev", headline_metric_id="asr_x", producer_spellings_json="[]"))
    _insert_row(conn, "item_set", _valid_row("item_set", item_set_id="is", eval_id="ev"))
    _insert_row(conn, "judge", _valid_row("judge", judge_id="j" * 64))
    contract = I.padding_identity_sha256({"filler_sha256": "7" * 64, "mode": "fixed", "placement": "before", "separator": "\n\n", "tolerance": 0.02})
    _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="i" * 64, eval_id="ev", item_set_id="is", judge_id="j" * 64, token_cap_tokens=4096,
                                               context_window_tokens=4864, notes="unpadded: no padding recipe"))
    _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="p" * 64, eval_id="ev", item_set_id="is", judge_id="j" * 64, token_cap_tokens=16384,
                                               context_window_tokens=98304, padding_tokens=16384, padding_identity_sha256=contract, notes="padded"))
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    # before the file: the zero point is refused (0012's `> 0`)
    with pytest.raises(sqlite3.IntegrityError, match="instrument_padding_pairing"):
        _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="z" * 64, eval_id="ev", item_set_id="is", padding_tokens=0, padding_identity_sha256=contract))
    drift = _sans_0017(_sans_0016(DB.schema_drift(conn)))                                # 0015's head + index, 0016's v_roster_match and 0017's v_paper_long drift here too (against the code's shape)
    assert len(drift) == 4 and drift[0].startswith("table instrument: CONSTRAINT instrument_padding_pairing differs") and drift[1] == V16_VIEW_DRIFT[0] and drift[2:] == V15_DRIFT, drift
    assert "padding_tokens > 0" in drift[0] and "padding_tokens >= 0" in drift[0]
    before = _object_sql(conn, ("instrument",))["instrument"]
    assert V13_PAIRING in before and V14_PAIRING not in before
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    assert "v_roster_match" in views and "v_paper_long" in views and "v_run_wide" in views and len(views) >= 20
    # every view but v_paper_inputs_head, which the runner carries past 14 through 0015 (its own test pins that text)
    others = ("run", "eval", "item_set", "judge", "run_metric") + tuple(v for v in views if v not in TF_VIEWS + ("v_paper_inputs_head",) + V16_OBJECTS + V17_OBJECTS + V20_OBJECTS + V21_OBJECTS)   # roster_entry / the two roster objects are 0016's; v_paper_long 0017's; the item views 0020's; object / judgment the record-refusal-fact migration's
    others_before = _object_sql(conn, others)
    rows_before = sorted(tuple(r) for r in conn.execute("SELECT * FROM instrument"))
    cols_before = [c[1] for c in conn.execute("PRAGMA table_info(instrument)")]
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(14, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(14, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 14").fetchone()[0] == "instrument_padding_zero_point"
    after = _object_sql(conn, ("instrument",))["instrument"]
    assert V14_PAIRING in after and V13_PAIRING not in after and after.startswith('CREATE TABLE "instrument"')
    assert after.count("CONSTRAINT instrument_padding_pairing") == 1                        # (the name also appears in the why-null comment of padding_tokens)
    assert _object_sql(conn, others) == others_before                                     # every view and every other table byte-identical (0016's objects aside)
    assert [c[1] for c in conn.execute("PRAGMA table_info(instrument)")] == cols_before
    assert sorted(tuple(r) for r in conn.execute("SELECT * FROM instrument")) == rows_before  # rows intact, values and order included
    assert M.foreign_key_violations(conn, S.ALL_TABLES) == []
    # after the file: the zero point is a row (0 is a VALUE, not NULL); a 0 alone, a contract alone and a negative pad are refused
    _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="z" * 64, eval_id="ev", item_set_id="is", padding_tokens=0, padding_identity_sha256=contract))
    conn.commit()
    assert tuple(conn.execute("SELECT padding_tokens, padding_identity_sha256 FROM instrument WHERE instrument_id = ?", ("z" * 64,)).fetchone()) == (0, contract)
    assert conn.execute("SELECT COUNT(*) FROM instrument WHERE padding_tokens IS NULL").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM instrument WHERE padding_tokens = 0").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM instrument WHERE padding_tokens IS NOT NULL").fetchone()[0] == 2
    for bad in ({"padding_tokens": 0}, {"padding_identity_sha256": contract}, {"padding_tokens": -1, "padding_identity_sha256": contract}):
        with pytest.raises(sqlite3.IntegrityError, match="instrument_padding_pairing"):
            _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="b" * 64, eval_id="ev", item_set_id="is", **bad))
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it



# --------------------------------------------------------------------------------------------
# 0015: v_paper_inputs_head reads its base tables, on a schema-version-14 database
# --------------------------------------------------------------------------------------------

V14_FRAME_MEMBER = "SELECT MAX(ingest_seq) AS seq FROM v_paper_long WHERE in_current_paper = 1"
V15_RUN_MEMBER = "SELECT MAX(ingest_seq) AS seq FROM run WHERE lifecycle_state IN ('judged', 'ingested')"
V15_ITEM_MEMBER = "SELECT MAX(ingest_seq) FROM item_result INDEXED BY idx_item_result_withdrawn_seq WHERE withdrawn_by_revision_id IS NOT NULL"
V15_INDEX = "CREATE INDEX idx_item_result_withdrawn_seq ON item_result (ingest_seq) WHERE withdrawn_by_revision_id IS NOT NULL"


def _v14_shape(path: str) -> None:
    """A database at schema version 14: migrations 0001..0014 applied by the runner itself, 0015 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 14:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0015_paper_inputs_head_reads_base_tables_on_a_v14_database(tmp_path):
    """Exactly two objects differ before the file and none after: `v_paper_inputs_head`, whose first member evaluated the whole paper frame
    (`MAX(ingest_seq) FROM v_paper_long WHERE in_current_paper = 1`: 452 s of every live cycle) and is replaced by the base tables an
    in-paper row draws its ingest_seq from (`run` at the two lifecycle states the selection CASE admits, `run_metric`, `run_contrast` -- a
    superset, never a subset), and the new partial index `idx_item_result_withdrawn_seq`, which the item member names so MAX(ingest_seq)
    over the withdrawn rows is one seek whatever `sqlite_stat1` says. No table changes, no rows copied, every other object byte-identical,
    the file carries schema_def's text verbatim, the plan of the head touches no paper object, and the writers' open accepts the database."""
    path = str(tmp_path / "r.sqlite")
    _v14_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 14
    assert _sans_0017(_sans_0016(DB.schema_drift(conn))) == [V16_VIEW_DRIFT[0]] + V15_DRIFT   # 0016's v_roster_match and 0017's v_paper_long drift on a v14 shape too
    before = _object_sql(conn, ("v_paper_inputs_head",))["v_paper_inputs_head"]
    assert V14_FRAME_MEMBER in before and "INDEXED BY" not in before and "run_contrast" not in before
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_item_result_withdrawn_seq'").fetchone()[0] == 0
    # rows the members read after the file: an ingested run at rank 5 and a queued one at rank 11 (the run member reads the complete one
    # alone), a withdrawn item row at rank 7 and a live one at rank 9 (the item member reads the withdrawn one alone); no metric, contrast or
    # roster row, so the head is 7. Foreign keys off: the parents (model, object, eval, ...) are not the point here -- but the run
    # rows' parents are SEEDED (`_valid_row`'s placeholder ids), because migration 0018 REBUILDS `run` and the runner's foreign-key
    # check over a rebuilt table rolls the migration back on an orphan row (the fixture's rows are real rows of the chain)
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_row(conn, "model", _valid_row("model", model_id="x"))
    _insert_row(conn, "eval", _valid_row("eval", eval_id="x", headline_metric_id="asr_x", producer_spellings_json="[]"))
    _insert_row(conn, "item_set", _valid_row("item_set", item_set_id="x", eval_id="x"))
    _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="x", eval_id="x", item_set_id="x", judge_id=None))
    _insert_row(conn, "object", _valid_row("object", object_id="x", object_kind="none", paper_label="none", object_role="baseline_none",
                                           fit_target_kind="none", fit_thinking_render="not_applicable", fit_objective="not_applicable",
                                           fit_positions="not_applicable", fit_seed_confidence="not_applicable", derivation="none",
                                           artifact_sha256="none", status="approved"))
    utc = {k: "2026-01-01T00:00:00.000Z" for k in ("queued_utc", "started_utc", "generated_utc", "judged_utc", "ingested_utc")}
    _insert_row(conn, "run", _valid_row("run", run_id="r" * 64, lifecycle_state="ingested", ingest_seq=5, record_uri="synthetic://a", **utc))
    _insert_row(conn, "run", _valid_row("run", run_id="q" * 64, lifecycle_state="queued", ingest_seq=11, record_uri="synthetic://b",
                                        producer_run_key_legacy="y", **utc))
    # (the sentinel source_revision_id '': a named revision needs its source_revision row by trigger, foreign keys on or off)
    _insert_row(conn, "item_result", _valid_row("item_result", run_id="r" * 64, item_id="q1", metric_id="asr", record_sha256=b"\x01" * 32,
                                                record_file_id=1, ingest_seq=7, withdrawn_by_revision_id="rev2"))
    _insert_row(conn, "item_result", _valid_row("item_result", run_id="r" * 64, item_id="q2", metric_id="asr", record_sha256=b"\x02" * 32,
                                                record_file_id=2, ingest_seq=9))
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    assert "v_paper_inputs_head" in views and "v_paper_long" in views and len(views) >= 20
    others = ("run", "run_metric", "ruling", "ruling_state", "disposition", "source_revision",           # roster_entry is 0016's; item_result 0020's; run_contrast 0027's
              "withdrawal", "idx_item_result_withdrawn", "idx_item_result_item", "idx_item_result_revision", "idx_item_result_run_judgment"
              ) + tuple(v for v in views if v not in TF_VIEWS + ("v_paper_inputs_head",) + V16_OBJECTS + V17_OBJECTS + V20_OBJECTS + V21_OBJECTS)   # v_paper_long is 0017's; the item views 0020's
    others_before = _object_sql(conn, others)
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(15, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(15, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 15").fetchone()[0] == "paper_inputs_head_base_tables"
    after = _object_sql(conn, ("v_paper_inputs_head", "idx_item_result_withdrawn_seq"))
    assert after["v_paper_inputs_head"] == S.VIEW_SQL["v_paper_inputs_head"].strip().rstrip(";")
    # the chain ends at the code's text: 0021 reads the run table whole (V15_RUN_MEMBER is 0015's / 0020's spelling, pinned by their
    # frozen files) and adds the judgment member; the item member is 0015's still
    assert V15_RUN_MEMBER not in after["v_paper_inputs_head"] and "SELECT MAX(ingest_seq) AS seq FROM run\n" in after["v_paper_inputs_head"]
    assert V15_ITEM_MEMBER in after["v_paper_inputs_head"]
    assert V14_FRAME_MEMBER not in after["v_paper_inputs_head"] and "v_paper_long" not in after["v_paper_inputs_head"]
    for member in ("run_metric", "run_contrast", "judgment", "roster_entry", "ruling", "ruling_state", "disposition", "withdrawal"):
        assert after["v_paper_inputs_head"].count(f"UNION ALL SELECT MAX(ingest_seq) FROM {member}\n") == 1, member
    assert "UNION ALL SELECT MAX(ingest_seq) FROM source_revision WHERE accepted = 1\n" in after["v_paper_inputs_head"]
    assert after["idx_item_result_withdrawn_seq"] == V15_INDEX
    assert _object_sql(conn, others) == others_before                                     # nothing else moved: tables, indexes, every other view
    # the file carries the lookup form of the head (its 0015 text; 0020 added the item_withdrawal member since) and the one index, no table
    with open([p for v, _, p in M.migration_files() if v == 15][0], "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "CREATE VIEW v_paper_inputs_head AS" in text and V15_RUN_MEMBER in text and V15_ITEM_MEMBER in text and V14_FRAME_MEMBER not in text
    assert "item_withdrawal" not in text and V15_INDEX + ";" in text
    statements = [" ".join(ln.split(" ")[:2]) for ln in text.splitlines()
                  if ln.split(" ")[0] in ("CREATE", "DROP", "ALTER", "INSERT", "UPDATE", "DELETE", "ANALYZE")]
    assert sorted(statements) == ["CREATE INDEX", "CREATE VIEW", "DROP INDEX", "DROP VIEW"], statements     # no table, no rows copied
    # the head is a lookup: the item member seeks through the new index; no paper object and no materialized frame in the plan
    plan = [r[3] for r in conn.execute("EXPLAIN QUERY PLAN SELECT ingest_seq FROM v_paper_inputs_head")]
    assert any("SEARCH item_result USING INDEX idx_item_result_withdrawn_seq" in p for p in plan), plan
    assert not any(bad in p for p in plan for bad in ("v_paper_long", "v_paper_current", "v_roster_match", "MATERIALIZED", "covered", "matched")), plan
    # 0015's run member read the complete run alone: the withdrawn row (7) over the ingested run (5), the queued run (11) unseen -- the
    # pre-0021 text, run as a query; since 0021 the run member reads every row (a retiring run must move the head), so the head reads 11
    pre_0021 = _pre_0021_head().split(" AS\n", 1)[1].strip().rstrip(";")
    assert conn.execute(pre_0021).fetchone()[0] == 7
    assert conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0] == 11
    conn.execute("UPDATE run SET lifecycle_state = 'ingested' WHERE run_id = ?", ("q" * 64,))    # the queued run completes: its rank counts for both
    assert conn.execute(pre_0021).fetchone()[0] == 11
    assert conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0] == 11
    conn.execute("UPDATE item_result SET withdrawn_by_revision_id = NULL WHERE ingest_seq = 7")   # un-withdrawn (a re-activation): the seek follows
    assert conn.execute("SELECT MAX(ingest_seq) FROM item_result INDEXED BY idx_item_result_withdrawn_seq WHERE withdrawn_by_revision_id IS NOT NULL").fetchone()[0] is None
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0016: the roster version scopes the roster match, on a schema-version-15 database (roster v9)
# --------------------------------------------------------------------------------------------

V16_SCOPE = "OR re.roster_version = (SELECT roster_version FROM paper_round pr WHERE pr.round_id = re.round_id))"


def _v15_shape(path: str) -> None:
    """A database at schema version 15: migrations 0001..0015 applied by the runner itself, 0016 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 15:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0016_roster_version_scopes_the_match_on_a_v15_database(tmp_path):
    """Exactly six lines differ before the file and none after: `paper_round` and `roster_entry` lack `roster_version` (each a rebuild, the
    column ahead of the run-scale tail), and `v_roster_match` / `v_roster_gaps` gain the version predicate. The two tables are REBUILT with
    their rows intact -- the round and both entries keep every value and read NULL on the new column, so a round loaded before the file
    still admits every entry -- no other table or view moves, the foreign keys hold (disposition -> roster_entry, roster_entry -> paper_round),
    and the writers' open accepts the database. The behaviour (a later version retiring the ids it no longer expands) is the roster test's."""
    path = str(tmp_path / "r.sqlite")
    _v15_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 15
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_row(conn, "paper_round", _valid_row("paper_round", round_id="r_x", opened_utc="2026-01-01T00:00:00.000Z", description="synthetic round"))
    _insert_row(conn, "model", _valid_row("model", model_id="m_x"))
    _insert_row(conn, "eval", _valid_row("eval", eval_id="ev", headline_metric_id="asr_x", producer_spellings_json="[]"))
    for eid, sel in (("e" * 36, '{"paper_label":"constitution_lora","object_kind":"lora","transfer_kind":"instruct_direct"}'),
                     ("f" * 36, '{"paper_label":"constitution_lora","object_kind":"lora","transfer_kind":"base_to_instruct","fit_model_id":"m_base"}')):
        _insert_row(conn, "roster_entry", _valid_row("roster_entry", roster_entry_id=eid, round_id="r_x", comparison_id="fig_x/instruct_direct", model_id="m_x", eval_id="ev",
                                                     object_id=None, object_selector_json=sel, instrument_id=None, instrument_selector_json='{"token_cap_tokens":[512]}',
                                                     strength_rule="operating_point", source_citation="synthetic @ eeeeeeee"))
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    drift = [d for d in _sans_0018(_sans_0017(DB.schema_drift(conn))) if d != V20_HEAD_DRIFT]   # 0017's v_paper_long, 0018's CHECKs and 0020's head drift on a v15 shape too
    assert len(drift) == 6 and [d.split(" — ")[0] for d in drift] == list(V16_TABLE_PREFIXES[:2]) + list(V16_TABLE_PREFIXES[2:]) + V16_VIEW_DRIFT, drift
    before = _object_sql(conn, ("v_roster_match", "v_roster_gaps"))
    assert all("roster_version" not in sql for sql in before.values())
    assert "roster_version" not in [c[1] for c in conn.execute("PRAGMA table_info(roster_entry)")] and "roster_version" not in [c[1] for c in conn.execute("PRAGMA table_info(paper_round)")]
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    others = ("run", "eval", "item_set", "judge", "instrument", "run_metric", "disposition", "ruling") + tuple(v for v in views if v not in TF_VIEWS + V16_OBJECTS + V17_OBJECTS + V20_OBJECTS + V21_OBJECTS)   # v_paper_long is 0017's; the item views and the head 0020's; object / judgment the record-refusal-fact migration's
    others_before = _object_sql(conn, others)
    rows_before = {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}")) for t in ("paper_round", "roster_entry")}
    cols_before = {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] for t in ("paper_round", "roster_entry")}
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(16, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(16, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 16").fetchone()[0] == "roster_version_scope"
    after = _object_sql(conn, ("v_roster_match", "v_roster_gaps"))
    assert after["v_roster_match"] == S.VIEW_SQL["v_roster_match"].strip().rstrip(";") and after["v_roster_gaps"] == S.VIEW_SQL["v_roster_gaps"].strip().rstrip(";")
    assert V16_SCOPE in after["v_roster_match"] and "pr.roster_version IS NULL OR re.roster_version = pr.roster_version" in after["v_roster_gaps"]
    assert _object_sql(conn, others) == others_before                                     # nothing else moved: tables, every other view
    for t in ("paper_round", "roster_entry"):
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
        assert cols == cols_before[t][:-4] + ["roster_version"] + cols_before[t][-4:], (t, cols)      # ahead of the run-scale tail, ingest_seq still last
        rows = sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}"))
        assert [r[:-5] + r[-4:] for r in rows] == rows_before[t] and all(r[-5] is None for r in rows), t   # every value kept; the new column NULL
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (t,)).fetchone()[0].startswith(f'CREATE TABLE "{t}"')
    assert M.foreign_key_violations(conn, S.ALL_TABLES) == []
    # a round with no stamp admits every entry of the round, whatever its own stamp (the pre-0016 reading); a stamped round admits its own
    conn.execute("UPDATE roster_entry SET roster_version = 'v1' WHERE roster_entry_id = ?", ("e" * 36,))
    conn.execute("UPDATE roster_entry SET roster_version = 'v2' WHERE roster_entry_id = ?", ("f" * 36,))
    conn.commit()
    listed = lambda: sorted(r[0] for r in conn.execute("SELECT roster_entry_id FROM v_roster_gaps"))
    assert listed() == sorted(["e" * 36, "f" * 36])                                          # no run: both are gaps while the round carries no version
    conn.execute("UPDATE paper_round SET roster_version = 'v2' WHERE round_id = 'r_x'")
    conn.commit()
    assert listed() == ["f" * 36]                                                           # the v1 id is no gap once the round reads v2
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0017: the paper frame joins only CURRENT contrasts, on a schema-version-16 database (2026-09-26)
# --------------------------------------------------------------------------------------------


def _v16_shape(path: str) -> None:
    """A database at schema version 16: migrations 0001..0016 applied by the runner itself, 0017 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 16:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0017_paper_long_joins_only_current_contrasts_on_a_v16_database(tmp_path):
    """Exactly one object differs before the file and none after: `v_paper_long` gains the `withdrawn_contrast` / `current_contrast`
    CTEs and reads the contrast join through the second (schema_def.WITHDRAWN_CONTRAST_CTE / CURRENT_CONTRAST_CTE). View-only: no table
    moves, no other object moves, the column set of the frame is unchanged, and the writers' open accepts the database. The behavior (a
    re-anchored run joins the newer anchor once; two current anchors of different kinds are both joined; a dropped contrast leaves the
    metric row with NULL contrast columns) is tests/test_registry_paper_views_contrast_current.py's."""
    path = str(tmp_path / "r.sqlite")
    _v16_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 16
    raw = DB.schema_drift(conn)
    assert V17_VIEW_DRIFT in raw                                                             # 0017's one object (0021 recreates it too)
    assert [d for d in _sans_rrf_roster(_sans_0018(raw)) if d != V20_HEAD_DRIFT] == []      # 0018's, 0020's, 0021's and the record-refusal-fact migration's v_roster_match lines aside, nothing else
    before = _object_sql(conn, ("v_paper_long",))["v_paper_long"]
    assert "withdrawn_contrast" not in before and "current_contrast" not in before and "LEFT JOIN run_contrast rc ON" in before
    cols_before = [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)")]
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    others = tuple(t for t in S.ALL_TABLES if t not in V20_OBJECTS + RRF_TABLES + FO_TABLES + AF_TABLES + FU4_TABLES) + tuple(v for v in views if v not in TF_VIEWS + ("v_paper_long",) + V20_OBJECTS + V21_OBJECTS + RRF_VIEWS)   # 0020's, 0021's and the record-refusal-fact migration's objects aside
    others_before = _object_sql(conn, others)
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(17, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(17, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 17").fetchone()[0] == "paper_long_current_contrast"
    after = _object_sql(conn, ("v_paper_long",))["v_paper_long"]
    assert after == S.VIEW_SQL["v_paper_long"].strip().rstrip(";")
    assert S.WITHDRAWN_CONTRAST_CTE in after and S.CURRENT_CONTRAST_CTE in after
    assert "LEFT JOIN current_contrast rc ON" in after and "LEFT JOIN run_contrast rc ON" not in after
    assert _object_sql(conn, others) == others_before                                    # nothing else moved: every table, every other view
    assert [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)") if c[1] != "n_current_contrasts"] == cols_before  # the column set is unchanged (0021 appends n_current_contrasts)
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long").fetchone()[0] == 0           # selectable on the empty database
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0019: the paper frame derives the Wilson interval where none was filed, on a schema-version-18 database (2026-10-01)
# --------------------------------------------------------------------------------------------


def _v18_shape(path: str) -> None:
    """A database at schema version 18: migrations 0001..0018 applied by the runner itself, 0019 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 18:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0019_paper_long_derives_the_wilson_interval_on_a_v18_database(tmp_path):
    """Exactly one object differs before the file and none after: `v_paper_long` reads `ci_lo` / `ci_hi` through the Wilson CASE
    (schema_def.WILSON_CI_LO_SQL / WILSON_CI_HI_SQL, inside the multi-source refusal) and `ci_method` through WILSON_CI_METHOD_SQL.
    View-only: no table moves, no other object moves, the column set of the frame is unchanged, the committed body carries the three
    Wilson expressions as the code spelled them at 0019, and the writers' open accepts the database. (Until 0021 this test also asserted
    the committed file IS the render against the v18 shape; 0021 recreates v_paper_long, so the code's text is no longer 0019's and the
    frozen sha in test_registry_schema.py is the text pin, the way 0017's test reads since 0019.) The behavior (a filed interval kept
    byte for byte; k and n without one read the Wilson bounds under wilson_95; the NULL cases) is tests/test_registry_paper_views_wilson_interval.py's."""
    path = str(tmp_path / "r.sqlite")
    _v18_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 18
    raw = DB.schema_drift(conn)
    assert V17_VIEW_DRIFT in raw                                                             # the one line: v_paper_long differs (0021 recreates it too)
    assert _sans_rrf_roster(_sans_0020(raw)) == [V20_HEAD_DRIFT]                             # 0020's, 0021's and the record-refusal-fact migration's v_roster_match lines aside, nothing else
    committed = open(os.path.join(M.HERE, "0019_paper_long_wilson_interval.sql"), encoding="utf-8").read()
    body = committed[committed.index("-- drift: "):]                                      # the hand-written header ends where the render begins
    assert body.startswith("-- drift: v_paper_long differs from the code's SQL\n\nDROP VIEW IF EXISTS v_paper_long;\nCREATE VIEW v_paper_long AS")
    assert S.WILSON_CI_LO_SQL in body and S.WILSON_CI_HI_SQL in body and S.WILSON_CI_METHOD_SQL in body and "CREATE TABLE" not in body
    before = _object_sql(conn, ("v_paper_long",))["v_paper_long"]
    assert "sqrt(" not in before and "wilson_95" not in before and "rm.ci_lo END AS ci_lo" in before and "\n       rm.ci_method," in before
    cols_before = [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)")]
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    others = tuple(t for t in S.ALL_TABLES if t not in V20_OBJECTS + RRF_TABLES + FO_TABLES + AF_TABLES + FU4_TABLES) + tuple(v for v in views if v not in TF_VIEWS + ("v_paper_long",) + V20_OBJECTS + V21_OBJECTS + RRF_VIEWS)   # 0020's, 0021's and the record-refusal-fact migration's objects aside
    others_before = _object_sql(conn, others)
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(19, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(19, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 19").fetchone()[0] == "paper_long_wilson_interval"
    after = _object_sql(conn, ("v_paper_long",))["v_paper_long"]
    assert after == S.VIEW_SQL["v_paper_long"].strip().rstrip(";")
    assert S.WILSON_CI_LO_SQL in after and S.WILSON_CI_HI_SQL in after and S.WILSON_CI_METHOD_SQL in after
    assert "rm.ci_lo END AS ci_lo" not in after and "\n       rm.ci_method," not in after
    assert _object_sql(conn, others) == others_before                                    # nothing else moved: every table, every other view
    assert [c[1] for c in conn.execute("PRAGMA table_info(v_paper_long)") if c[1] != "n_current_contrasts"] == cols_before  # the column set is unchanged (0021 appends n_current_contrasts)
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long").fetchone()[0] == 0           # selectable on the empty database (sqrt compiles)
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0020: a producer retracts item rows filed under the wrong run, on a schema-version-19 database
# --------------------------------------------------------------------------------------------


def _v19_shape(path: str) -> None:
    """A database at schema version 19: migrations 0001..0019 applied by the runner itself, 0020 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 19:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0020_item_withdrawal_on_a_v19_database(tmp_path):
    """Exactly 0020's lines differ before the file and none after. `item_result` is taken forward by two `ALTER TABLE ADD COLUMN`
    (the planner's `add_columns` kind: the marks trail ingest_seq, so the appended definitions equal the code's CREATE text; no row
    copied, the rows and every other column intact, the new columns NULL), `item_withdrawal` is created, `blocked_records` and
    `row_history` are REBUILT for their record_type / table_name CHECKs (rows intact), and the three objects that read the marks or
    the table are recreated; no other object moves, the committed file IS the body the generator renders against the v19 shape, and
    the writers' open accepts the database. After the file a marked native row leaves `v_item_result_current` and
    `v_item_result_provenances` while the raw table keeps it, and the head reads `item_withdrawal`. The fold itself (the marks as a
    pure function of the records, blocking, the re-filed row) is tests/test_registry_item_withdrawal.py's."""
    path = str(tmp_path / "r.sqlite")
    _v19_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 19
    drift = DB.schema_drift(conn)
    # 0021 recreates the two item views and the head too and adds its own lines; the raw drift carries 0020's item-view lines, the stripped
    # one the rest (the head's line stands for both files and `_sans_0021` keeps it)
    assert "v_item_result_current differs from the code's SQL" in drift and "v_item_result_provenances differs from the code's SQL" in drift
    assert [d.split(" — ")[0] for d in _sans_rrf_roster(_sans_0021(drift))] == list(V20_TABLE_PREFIXES) + [d for d in V20_DRIFT if d not in V21_DRIFT] + [V20_HEAD_DRIFT], drift
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == ["blocked_records", "row_history"] and [c for c in plan["create"] if c not in RRF_TABLES + FO_TABLES + AF_TABLES] == ["item_withdrawal"]   # 0027's run_contrast and 0031's two rebuilds aside
    assert [a for a in plan["add_columns"] if a["table"] not in RRF_TABLES + FO_TABLES + AF_TABLES] == [{"table": "item_result", "rows": 0, "columns": ["withdrawn_by_record_sha256", "withdrawn_reason"], "scale": "item"}]
    # the plan is against the CODE's shape, so 0021's recreates are listed beside 0020's three (its own test pins the view-only plan on v20)
    assert plan["objects"]["view"] == {"recreate": sorted({"v_item_result_current", "v_item_result_provenances", "v_paper_inputs_head", TF_CHECK_VIEW} | set(V21_OBJECTS) | set(RRF_VIEWS)), "drop": []}
    assert plan["objects"]["index"] == {"recreate": [], "drop": []} and plan["objects"]["trigger"] == {"recreate": [], "drop": []}   # 0030's two fence triggers ride 0031's run rebuild
    rendered = frozen_column_docs(_sans_fence_render(M.render_shape_migration(conn, plan)))   # 0030's two fence triggers render between the halves;
    # the descriptions reworded since 0020 landed read at the frozen text
    committed = open(os.path.join(M.HERE, "0020_item_withdrawal.sql"), encoding="utf-8").read()
    body = committed[committed.index("-- drift: "):]                                      # the hand-written header ends where the render begins
    # the committed body was the render against v19 when it landed (its sha is frozen); the table half of the render is still identical,
    # and the view half differs only by 0021's objects (the two item views and the head the file recreates at 0020's text)
    cut = rendered.index("DROP VIEW IF EXISTS ")
    # the record-refusal-fact migration's create and add-column blocks (the fact table; object / judgment / record_refusal) render here too and are not in the frozen body
    table_half = [b for b in rendered[rendered.index("\n\n") + 2:cut].split("\n\n")
                  if not b.startswith(("-- create record_refusal_fact", "-- add column(s) to object:", "-- add column(s) to judgment:", "-- add column(s) to record_refusal:",
                                           "-- create fold_observation", "-- rebuild run_contrast:", "-- create analysis", "-- create ordered_fact_record", "-- create ordered_fact_carrier"))]     # the fold-observation migration's create, 0027's rebuild and 0029's five creates render here too
    assert body[body.index("\n\n") + 2:body.index("DROP VIEW IF EXISTS ")].strip() == "\n\n".join(table_half).strip()
    assert "DROP VIEW IF EXISTS v_paper_inputs_head;\n" + _pre_0021_head().strip() in body             # 0020's head text (0021 recreates it)
    assert "DROP VIEW IF EXISTS v_item_result_current;" in body and "DROP VIEW IF EXISTS v_item_result_provenances;" in body and "v_run_current" not in body
    assert "ALTER TABLE item_result ADD COLUMN withdrawn_by_record_sha256 TEXT;" in body and "ALTER TABLE item_result ADD COLUMN withdrawn_reason TEXT;" in body
    assert "CREATE TABLE item_result__new" not in body                                     # no copy of the item table
    assert "CREATE TABLE row_history__new" in body and "CREATE TABLE blocked_records__new" in body
    # rows the file must carry through: a run with its parents and one native item row; a blocked record and a history event
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_row(conn, "model", _valid_row("model", model_id="x"))
    _insert_row(conn, "eval", _valid_row("eval", eval_id="x", headline_metric_id="asr_x", producer_spellings_json="[]"))
    _insert_row(conn, "item_set", _valid_row("item_set", item_set_id="x", eval_id="x"))
    _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="x", eval_id="x", item_set_id="x", judge_id=None))
    _insert_row(conn, "object", _valid_row("object", object_id="x", object_kind="none", paper_label="none", object_role="baseline_none",
                                           fit_target_kind="none", fit_thinking_render="not_applicable", fit_objective="not_applicable",
                                           fit_positions="not_applicable", fit_seed_confidence="not_applicable", derivation="none",
                                           artifact_sha256="none", status="approved"))
    utc = {k: "2026-01-01T00:00:00.000Z" for k in ("queued_utc", "started_utc", "generated_utc")}
    _insert_row(conn, "run", _valid_row("run", run_id="r" * 64, lifecycle_state="generated", ingest_seq=5, record_uri="synthetic://a", **utc))
    _insert_row(conn, "record_ingest", _valid_row("record_ingest", file_uri="synthetic://a", record_file_id=1, ingest_seq=5, n_records=1))
    _insert_row(conn, "item_result", _valid_row("item_result", run_id="r" * 64, item_id="q1", metric_id="asr", record_sha256=b"\x01" * 32,
                                                record_file_id=1, ingest_seq=5, written_utc=1000))
    _insert_row(conn, "blocked_records", _valid_row("blocked_records", record_sha256="b" * 64, record_uri="synthetic://b#1", record_type="run_state",
                                                    missing_parent_table="run", missing_parent_id="zz"))
    _insert_row(conn, "row_history", _valid_row("row_history", table_name="run", natural_key_json='{"run_id": "x"}', record_sha256="c" * 64,
                                                record_type="run", record_uri="synthetic://a", payload_json="{}", folded_state_json="{}"))
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    assert "v_item_result_current" in views and "item_withdrawal" not in views
    others = tuple(t for t in S.ALL_TABLES if t not in V20_OBJECTS + RRF_TABLES + FO_TABLES + AF_TABLES + FU4_TABLES) + tuple(v for v in views if v not in TF_VIEWS + V20_OBJECTS + V21_OBJECTS + RRF_VIEWS)   # the record-refusal-fact migration's tables and v_roster_match aside
    others += tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('index', 'trigger') AND sql IS NOT NULL"))
    others_before = _object_sql(conn, others)
    item_cols_before = [c[1] for c in conn.execute("PRAGMA table_info(item_result)")]
    item_rows_before = sorted(tuple(r) for r in conn.execute("SELECT * FROM item_result"))
    ledger_before = {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}")) for t in ("blocked_records", "row_history")}
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(20, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(20, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 20").fetchone()[0] == "item_withdrawal"
    assert _object_sql(conn, others) == others_before                                    # nothing else moved: every other table, index, trigger, view
    # item_result: the two marks appended after ingest_seq, the stored text NOT a renamed rebuild, every row intact and NULL on them
    assert [c[1] for c in conn.execute("PRAGMA table_info(item_result)")] == item_cols_before + ["withdrawn_by_record_sha256", "withdrawn_reason"]
    assert conn.execute("SELECT sql FROM sqlite_master WHERE name = 'item_result'").fetchone()[0].startswith("CREATE TABLE item_result (")
    assert sorted(tuple(r) for r in conn.execute("SELECT * FROM item_result")) == [r + (None, None) for r in item_rows_before]
    for t, rows in ledger_before.items():
        assert sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}")) == rows, t                   # the rebuilt ledgers keep their rows
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (t,)).fetchone()[0].startswith(f'CREATE TABLE "{t}"')
    conn.execute("INSERT INTO blocked_records (record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, first_seen_utc, "
                 "last_checked_utc) VALUES (?, 'synthetic://w#1', 'item_withdrawal', 'item_result', '{}', '2026-01-01T00:00:00.000Z', "
                 "'2026-01-01T00:00:00.000Z')", ("d" * 64,))                                           # the new kind passes the rebuilt CHECK
    assert M.foreign_key_violations(conn, S.ALL_TABLES) == []
    # the behavior the file enables, in SQL: a withdrawal row and the mark on the native row
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0] == 1
    _insert_row(conn, "item_withdrawal", _valid_row("item_withdrawal", run_id="r" * 64, item_id="q1", metric_id="asr", reason="misfiled",
                                                    record_uri="synthetic://w", ingest_seq=9))
    conn.execute("UPDATE item_result SET withdrawn_by_record_sha256 = ?, withdrawn_reason = 'misfiled' WHERE item_id = 'q1'", ("e" * 64,))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_provenances").fetchone()[0] == 0
    assert conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0] == 9           # the withdrawal table is a member
    # a row RE-FILED above the withdrawal (a newer file: rank 12; the fold clears its marks): current again, and the head is its rank --
    # the member that seeks the native row of every withdrawal key by primary key, driven from the small table, never a walk of item_result
    conn.execute("UPDATE item_result SET withdrawn_by_record_sha256 = NULL, withdrawn_reason = NULL, ingest_seq = 12 WHERE item_id = 'q1'")
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0] == 1
    assert conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0] == 12
    plan = [r[3] for r in conn.execute("EXPLAIN QUERY PLAN SELECT ingest_seq FROM v_paper_inputs_head")]
    iw = next(i for i, p in enumerate(plan) if p.startswith(("SCAN iw", "SEARCH iw")))
    ir = next(i for i, p in enumerate(plan) if p.startswith("SEARCH ir"))
    assert iw < ir and not any("idx_item_result_revision" in p for p in plan), plan
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0021: the frame's currency rules, on a schema-version-20 database
# --------------------------------------------------------------------------------------------


def _v20_shape(path: str) -> None:
    """A database at schema version 20: migrations 0001..0020 applied by the runner itself, 0021 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 20:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0021_run_current_frame_rules_on_a_v20_database(tmp_path):
    """Exactly 0021's nine lines differ before the file and none after; the plan is VIEW-ONLY (nine recreates, no rebuild, no column,
    no table, no index or trigger); the committed file IS the body the generator renders against the v20 shape; every recreated object is
    selectable afterwards on the empty database (SQLite binds a referenced object lazily, so the alphabetical body creates v_paper_current
    before v_run_current and only a SELECT would surface a missing dependency); the frame's column order is kept with the new columns
    appended; nothing else moves; the writers' open accepts the database. The behavior of each rule is tests/test_registry_views_0021.py's."""
    path = str(tmp_path / "r.sqlite")
    _v20_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 20
    drift = DB.schema_drift(conn)
    assert _sans_rrf_roster(_sans_rrf(drift)) == V21_DRIFT, drift                           # the record-refusal-fact migration's table, index and v_roster_match lines aside (its own test pins them)
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == [] and plan["undeclared_tables"] == []      # 0027's run_contrast and 0031's two rebuilds aside
    assert [a for a in plan["add_columns"] if a["table"] not in RRF_TABLES + FO_TABLES + AF_TABLES] == [] and [c for c in plan["create"] if c not in RRF_TABLES + FO_TABLES + AF_TABLES] == []
    assert plan["objects"]["view"] == {"recreate": sorted(set(V21_OBJECTS) | set(RRF_VIEWS) | {TF_CHECK_VIEW}), "drop": []}      # six of the record-refusal-fact migration's seven recreates are among the nine; v_roster_match is its own; 0026's v_multi_source_check
    assert plan["objects"]["index"] == {"recreate": [], "drop": []} and plan["objects"]["trigger"] == {"recreate": [], "drop": []}   # 0030's two fence triggers ride 0031's run rebuild
    rendered = M.render_shape_migration(conn, plan)
    committed = open(os.path.join(M.HERE, "0021_run_current_frame_rules.sql"), encoding="utf-8").read()
    body = committed[committed.index("-- drift: "):]                                      # the hand-written header ends where the render begins
    # the committed body was the render against v20 when it landed (its sha is frozen); today's render against the same shape carries the
    # record-refusal-fact migration's create and add-column half as well, and of the nine recreates seven differ -- v_run_current by 0022's
    # text alone (`_pre_0022_run_current` takes the code's text back to 0021's) and the record-refusal-fact migration's six by theirs; the
    # other two render identically
    for v in sorted(set(V21_OBJECTS) - set(RRF_VIEWS)):
        code = S.VIEW_SQL[v].strip()
        block = "DROP VIEW IF EXISTS " + v + ";\n" + code
        assert block in rendered, v
        if v == "v_run_current":
            assert block not in body and "DROP VIEW IF EXISTS " + v + ";\n" + _pre_0022_run_current(code) in body, v
        else:
            assert block in body, v
    for v in RRF_VIEWS:
        assert "DROP VIEW IF EXISTS " + v + ";\n" + S.VIEW_SQL[v].strip() not in body, v           # the record-refusal-fact migration moved them
    assert "CREATE TABLE" not in body and "ALTER TABLE" not in body and "CREATE INDEX" not in body and "CREATE TRIGGER" not in body
    assert body.count("DROP VIEW IF EXISTS ") == 9 and body.count("\nCREATE VIEW ") == 9
    before = _object_sql(conn, [v for v in V21_OBJECTS if v != "v_run_current"])
    assert "v_run_current" not in {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view'")}
    assert "is_current" not in before["v_paper_current"] and "coverage_state" not in before["v_paper_current"] and "ROW_NUMBER() OVER (PARTITION BY run_id" in before["v_paper_current"]
    assert "n_current_contrasts" not in before["v_paper_long"] and "FROM v_paper_current pc" in before["v_paper_long"] and "ac.is_current" not in before["v_paper_long"]
    assert "by_source" not in before["v_run_decode_policy_match"]
    assert before["v_pending_judgment"].index("THEN 'abandoned'") < before["v_pending_judgment"].index("THEN 'superseded'")
    assert all("jx.status IN" not in before[v] for v in ("v_item_result_current", "v_item_result_provenances"))
    assert "jm.status" not in before["v_multi_source_rows"] and "LEFT JOIN judgment" not in before["v_multi_source_rows"]   # ownership read every row
    assert "multi_source AS MATERIALIZED" not in before["v_paper_current"] and "v_multi_source_rows" not in before["v_paper_current"]   # coverage read no scope
    assert "FROM judgment" not in before["v_paper_inputs_head"] and "FROM run WHERE lifecycle_state IN ('judged', 'ingested')" in before["v_paper_inputs_head"]
    assert before["v_paper_inputs_head"] == _pre_0021_head().strip().rstrip(";")                     # 0020's text, as the test helper reconstructs it
    long_cols_before = _columns(conn, "v_paper_long")
    pc_cols_before = _columns(conn, "v_paper_current")
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    others = tuple(t for t in S.ALL_TABLES if t not in RRF_TABLES + FO_TABLES + AF_TABLES + FU4_TABLES) + tuple(v for v in views if v not in TF_VIEWS + V21_OBJECTS + RRF_VIEWS)   # the record-refusal-fact migration's tables and v_roster_match aside
    others += tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('index', 'trigger') AND sql IS NOT NULL"))
    others_before = _object_sql(conn, others)
    # 0022 recreates v_run_current again and the record-refusal-fact migration follows it (their own tests pin their plans on the shape
    # below each), so the runner carries the database to the code
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(21, M.code_version() + 1))
    assert M.apply_migrations(conn) == list(range(21, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 21").fetchone()[0] == "run_current_frame_rules"
    assert _object_sql(conn, others) == others_before                                    # nothing else moved: every table, index, trigger, other view
    after = _object_sql(conn, V21_OBJECTS)
    for v in V21_OBJECTS:
        assert after[v] == S.VIEW_SQL[v].strip().rstrip(";"), v
        # selectable on the empty database: every dependency exists (the head is one aggregate row, NULL on an empty database)
        assert conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0] == (1 if v == "v_paper_inputs_head" else 0), v
    assert conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0] is None
    assert "UNION ALL SELECT MAX(ingest_seq) FROM judgment\n" in after["v_paper_inputs_head"] and "SELECT MAX(ingest_seq) AS seq FROM run\n" in after["v_paper_inputs_head"]
    assert "lifecycle_state" not in after["v_paper_inputs_head"]
    # the frame keeps its column order and appends: n_current_contrasts right after cap_matched; v_paper_current gains four at the end
    long_cols = _columns(conn, "v_paper_long")
    assert long_cols == long_cols_before[:long_cols_before.index("cap_matched") + 1] + ["n_current_contrasts"] + long_cols_before[long_cols_before.index("cap_matched") + 1:]
    assert _columns(conn, "v_paper_current") == pc_cols_before + ["is_current", "why", "coverage_state", "coverage_detail_json"]
    assert _columns(conn, "v_multi_source_rows") == [c[1] for c in conn.execute("PRAGMA table_info(v_multi_source_rows)")]   # the column set is unchanged
    assert {"declaration_id", "role", "pairing_key", "run_id", "judgment_id", "metric_id", "threshold", "computation_id"} <= set(_columns(conn, "v_multi_source_rows"))
    assert _columns(conn, "v_run_current") == ["run_id", "is_current", "why", "decisive_ruling_id", "decisive_action", "selection_state", "in_current_paper",
                                                "held_out", "round_id", "comparison_ids_json", "roster_entry_ids_json", "ruling_ids_json"]
    assert _columns(conn, "v_run_decode_policy_match") == ["run_id", "policy_id", "match", "candidates_json", "in_scope_json"]
    assert _columns(conn, "v_paper_inputs_head") == ["ingest_seq"]
    assert _columns(conn, "v_pending_judgment")[-1] == "pending_reason"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


# --------------------------------------------------------------------------------------------
# 0022: a member pass is a member only where its source filed a pooled reading, on a schema-version-21 database
# --------------------------------------------------------------------------------------------


def _v21_shape(path: str) -> None:
    """A database at schema version 21: migrations 0001..0021 applied by the runner itself, 0022 not."""
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= 21:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_0022_member_pass_requires_pooled_reading_on_a_v21_database(tmp_path):
    """Exactly 0022's one line differs before the file (the record-refusal-fact migration's lines aside: that file is newer, so every
    earlier shape drifts by them too -- `_sans_rrf` and its view lines; its own test pins them) and none after; the plan is ONE recreate
    of 0022's (`v_run_current`; no rebuild, no column, no table, no index or trigger beyond the record-refusal-fact migration's); the
    committed file IS the body the generator rendered against the v21 shape when it landed (its sha is frozen), and today's render
    carries its `v_run_current` block unchanged beside the newer file's additions; the object before the file is 0021's text (no `pooled_group`, no `unpooled_member`, the per-(source, pass) predicate alone in both CASEs) and after it the
    code's (the probe with its view_legacy term once, the label half three times, the unpooled_member branch); it and the two
    objects that read it (`v_paper_current`, `v_paper_long`) are selectable afterwards on the empty database; its column list is
    unchanged; nothing else moves; the writers' open accepts the database. The behavior is tests/test_registry_views_0022.py's."""
    path = str(tmp_path / "r.sqlite")
    _v21_shape(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == 21
    drift = DB.schema_drift(conn)
    rrf_view_lines = [f"{v} differs from the code's SQL" for v in RRF_VIEWS]
    assert [d for d in _sans_rrf(drift) if d not in rrf_view_lines] == V22_DRIFT, drift
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == [] and plan["undeclared_tables"] == []      # 0027's run_contrast and 0031's two rebuilds aside
    assert [a for a in plan["add_columns"] if a["table"] not in RRF_TABLES + FO_TABLES + AF_TABLES] == [] and [c for c in plan["create"] if c not in RRF_TABLES + FO_TABLES + AF_TABLES] == []
    assert plan["objects"]["view"] == {"recreate": sorted(set(V22_OBJECTS) | set(RRF_VIEWS) | {TF_CHECK_VIEW}), "drop": []}      # 0026's v_multi_source_check beside
    assert plan["objects"]["index"] == {"recreate": [], "drop": []} and plan["objects"]["trigger"] == {"recreate": [], "drop": []}   # 0030's two fence triggers ride 0031's run rebuild
    rendered = M.render_shape_migration(conn, plan)
    committed = open(os.path.join(M.HERE, "0022_member_pass_requires_pooled_reading.sql"), encoding="utf-8").read()
    assert "A registry ruling of 2026-10-02" in committed[:committed.index("-- drift: ")]    # the hand-written header names the ruling
    body = committed[committed.index("-- drift: "):]
    block = "DROP VIEW IF EXISTS v_run_current;\n" + S.VIEW_SQL["v_run_current"].strip()
    assert "DROP VIEW IF EXISTS v_run_current;\n" + _pre_0026_run_current(S.VIEW_SQL["v_run_current"].strip()) in body and block in rendered   # 0022's one object: its text then, the code's (0026's conjunct) in today's render
    assert "CREATE TABLE" not in body and "ALTER TABLE" not in body and "CREATE INDEX" not in body and "CREATE TRIGGER" not in body
    assert body.count("DROP VIEW IF EXISTS ") == 1 and body.count("\nCREATE VIEW ") == 1 and "DROP VIEW IF EXISTS v_run_current;" in body
    before = _object_sql(conn, V22_OBJECTS)["v_run_current"]
    assert "pooled_group" not in before and "D671" not in before and "unpooled_member" not in before
    assert before.count(S.member_label_sql()) == 2 and S.MEMBER_PASS_SQL not in before
    assert before == _pre_0022_run_current(S.VIEW_SQL["v_run_current"]).strip().rstrip(";")   # 0021's text, as the helper reconstructs it
    cols_before = _columns(conn, "v_run_current")
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    others = tuple(t for t in S.ALL_TABLES if t not in RRF_TABLES + FO_TABLES + AF_TABLES + FU4_TABLES) + tuple(v for v in views if v not in TF_VIEWS + V22_OBJECTS + RRF_VIEWS)   # the record-refusal-fact migration's aside
    others += tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('index', 'trigger') AND sql IS NOT NULL"))
    others_before = _object_sql(conn, others)
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(22, M.code_version() + 1))     # 0022, then the record-refusal-fact migration
    assert M.apply_migrations(conn) == list(range(22, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 22").fetchone()[0] == "member_pass_requires_pooled_reading"
    assert _object_sql(conn, others) == others_before                                    # nothing else moved: every table, index, trigger, other view
    after = _object_sql(conn, V22_OBJECTS)["v_run_current"]
    assert after == S.VIEW_SQL["v_run_current"].strip().rstrip(";")
    assert after.count("pooled_group AS MATERIALIZED") == 1 and after.count(S.MEMBER_PASS_SQL) == 1 and after.count(S.MEMBER_LABEL_SQL) == 3
    assert after.count("THEN 'unpooled_member'") == 1 and "pg.view_legacy IS r.view_legacy" in after
    for v in ("v_run_current", "v_paper_current", "v_paper_long"):
        assert conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0] == 0, v              # selectable: every dependency exists
    assert _columns(conn, "v_run_current") == cols_before == ["run_id", "is_current", "why", "decisive_ruling_id", "decisive_action", "selection_state",
                                                              "in_current_paper", "held_out", "round_id", "comparison_ids_json", "roster_entry_ids_json", "ruling_ids_json"]
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it


def test_addable_columns_is_decided_by_executing_the_alter_not_by_inference():
    """The planner's `add_columns` kind holds only for columns the code declares AFTER every live column: a v19 item_result takes the
    two marks by ALTER; a table whose missing column sits ahead of its tail (0016's roster_version shape) rebuilds; a table whose
    text differs elsewhere (a CHECK list) rebuilds even when a column is missing too."""
    conn = DB.connect_memory()
    for version, slug, mpath in M.migration_files():
        if version <= 19:
            M._apply_one(conn, version, slug, open(mpath).read())
    assert M.addable_columns(conn, "item_result") == ["withdrawn_by_record_sha256", "withdrawn_reason"]
    assert M.addable_columns(conn, "blocked_records") is None                            # a CHECK list differs: no column to add
    assert M.addable_columns(conn, "run") is None                                        # the text matches: nothing to add
    assert M.add_column_sql("item_result", "withdrawn_reason") == "ALTER TABLE item_result ADD COLUMN withdrawn_reason TEXT;"
    conn.close()
    conn = DB.connect_memory()
    for version, slug, mpath in M.migration_files():
        if version <= 15:
            M._apply_one(conn, version, slug, open(mpath).read())
    assert M.addable_columns(conn, "roster_entry") is None                               # roster_version sits ahead of the run-scale tail
    plan = M.shape_plan(conn)
    assert "roster_entry" in [r["table"] for r in plan["rebuild"]] and "roster_entry" not in [a["table"] for a in plan["add_columns"]]
    conn.close()


# --------------------------------------------------------------------------------------------
# the record-refusal-fact migration (the newest file; numbered at landing): refusal facts, object lineage, the judgment identity binding
#, on the shape one version below it
# --------------------------------------------------------------------------------------------


def _migration_by_slug(slug: str):
    """``(version, slug, path)`` of the migration file carrying ``slug`` -- by slug, never by position: a branch's file is numbered at landing."""
    return next(f for f in M.migration_files() if f[1] == slug)


def _shape_below_rrf(path: str) -> None:
    """A database at the shape one version below the record-refusal-fact migration: every earlier migration applied by the runner itself,
    that file and every newer one (the fold-observation migration) not."""
    conn = DB.connect(path)
    rrf_version = _migration_by_slug("record_refusal_fact")[0]
    for version, slug, mpath in M.migration_files():
        if version < rrf_version:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    conn.commit()
    conn.close()


def test_the_record_refusal_fact_migration_on_the_shape_one_version_below_it(tmp_path):
    """Exactly the record-refusal-fact migration's lines differ before the file and none after; the plan is one CREATE TABLE (with its two
    indexes), three tables taken forward by ALTER TABLE ADD COLUMN (five columns, each declared after the table's last live column), seven
    recreates (the five paper-side objects, the worklist and the roster match, the review of the first rendering) -- and NO rebuild
    (the two ledgers' record_type CHECK lists are byte-identical: `vocab.LEDGERED_RECORD_TYPES`); the committed file IS the body the
    generator renders against the v21 shape; the rows carried through are intact with the new columns NULL -- except the object rows whose
    `extra_json` carried the lineage keys, which the file's Python post-step lifts (`migrations.lift_object_lineage`, counted in its line)
    exactly as the new fold writes them; nothing else moves; every recreated object is selectable; the writers' open accepts the database.
    The behavior of the rules is tests/test_registry_refusal_facts.py's, test_registry_object_lineage.py's and
    test_registry_judgment_currency.py's."""
    import json as _json
    from sisteer.registry import reducer as RED
    from sisteer.registry import vocab as V
    path = str(tmp_path / "r.sqlite")
    version, slug, mpath = _migration_by_slug("record_refusal_fact")
    fo_version = _migration_by_slug("fold_observation")[0]
    assert fo_version > version                                                         # the fold-observation migration is the newer file; its lines are set aside below
    _shape_below_rrf(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == version - 1
    drift = DB.schema_drift(conn)
    assert [d.split(" — ")[0] for d in _sans_fo(drift) if d not in TF_SHAPE22_DRIFT] == list(RRF_TABLE_PREFIXES) + [RRF_DRIFT[0]] + [f"{v} differs from the code's SQL" for v in
            ("v_item_result_current", "v_item_result_provenances", "v_pending_judgment", "v_roster_match", "v_paper_current", "v_multi_source_rows", "v_paper_long")] + RRF_DRIFT[1:], drift   # the code's object order
    have, want = DB.ddl_master(conn), DB.code_ddl_master()
    for t in ("blocked_records", "row_history"):
        assert have[("table", t)] == want[("table", t)], t                                 # no ledger rebuild: the CHECK lists are byte-identical
    assert V.LEDGERED_RECORD_TYPES == tuple(t for t in V.RECORD_TYPES if t not in V.UNLEDGERED_FACT_TYPES)
    assert V.UNLEDGERED_FACT_TYPES == ("record_refusal",) + V.ANALYSIS_RECORD_TYPES       # 0029's three kinds joined it; the ledgers' CHECK lists still untouched
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == [] and plan["undeclared_tables"] == [] and [c for c in plan["create"] if c not in FO_TABLES + AF_TABLES] == ["record_refusal_fact"]   # 0027's run_contrast and 0031's two rebuilds aside
    assert plan["add_columns"] == [{"table": "object", "rows": 0, "columns": ["parent_object_id", "named_role"], "scale": "run"},
                                   {"table": "judgment", "rows": 0, "columns": ["claimed_instrument_id", "claimed_required_judge_id"], "scale": "run"},
                                   {"table": "record_refusal", "rows": 0, "columns": ["judged_with"], "scale": "ledger"}]
    assert plan["objects"]["view"] == {"recreate": sorted(set(RRF_VIEWS) | set(TF_SHAPE22_VIEWS)), "drop": []}      # 0026's two beside (v_multi_source_rows is this migration's own)
    assert plan["objects"]["index"] == {"recreate": [], "drop": []} and plan["objects"]["trigger"] == {"recreate": [], "drop": []}   # 0030's two fence triggers ride 0031's run rebuild
    rendered = M.render_shape_migration(conn, plan)
    committed = open(mpath, encoding="utf-8").read()
    assert committed.startswith(f"-- {os.path.basename(mpath)}\n")                           # the header names the file it is in
    body = committed[committed.index("-- drift: "):]                                      # the hand-written header ends where the render begins
    # today's render carries the newer files' work too: the fold-observation migration's create (`_sans_fo_render`) and 0025's text of the
    # worklist object, taken back to this file's (`_sans_pes_render`; the drift line stays: this file recreates the object as well)
    expected = _sans_pes_render(_sans_fo_render(_sans_tf_block(rendered, TF_SHAPE22_VIEWS)), drift_line=False)
    rows_code = S.VIEW_SQL["v_multi_source_rows"].strip()
    expected = expected.replace("DROP VIEW IF EXISTS v_multi_source_rows;\n" + rows_code, "DROP VIEW IF EXISTS v_multi_source_rows;\n" + _pre_0026_multi_source_rows(rows_code))
    assert body.strip() == expected.strip()                                                   # 0026 moved v_multi_source_rows too: the committed body is this file's text of it
    assert "__new" not in body and "INSERT INTO" not in body and "DROP TABLE" not in body and "CREATE TRIGGER" not in body
    assert body.count("\nCREATE TABLE ") == 1 and body.count("\nCREATE INDEX idx_record_refusal_fact_") == 2
    assert len(re.findall(r"(?m)^ALTER TABLE \w+ ADD COLUMN ", body)) == 5
    assert body.count("DROP VIEW IF EXISTS ") == 7 and body.count("\nCREATE VIEW ") == 7
    assert M.POST_APPLY == {"record_refusal_fact": M.lift_object_lineage, "fold_observation": M.derive_fold_observations}
    # rows the file must carry through: the dimension roots, two objects (one carrying the lineage keys under extra, one not), a run, a
    # judgment, a refusal row, a blocked record; the object rows' history events as the old fold left them (no lineage column folded)
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_row(conn, "model", _valid_row("model", model_id="x"))
    _insert_row(conn, "eval", _valid_row("eval", eval_id="x", headline_metric_id="asr_x", producer_spellings_json="[]"))
    _insert_row(conn, "item_set", _valid_row("item_set", item_set_id="x", eval_id="x"))
    _insert_row(conn, "instrument", _valid_row("instrument", instrument_id="x", eval_id="x", item_set_id="x", judge_id=None))
    obj = dict(object_kind="none", paper_label="none", object_role="baseline_none", fit_target_kind="none", fit_thinking_render="not_applicable",
               fit_objective="not_applicable", fit_positions="not_applicable", fit_seed_confidence="not_applicable", derivation="none",
               artifact_sha256="none", status="approved", ingest_seq=3, record_uri="synthetic://o")
    parent_extra = RED.canonical_json({"epoch_checkpoints": {"epochs": {}}})
    child_extra = RED.canonical_json({"named_role": "epoch_1", "named_roles": [], "parent_object_id": "p" * 64})
    seeded = {}
    for oid, stamp, sha, extra in (("p" * 64, "2026-01-01T00:00:00.000Z", "a" * 64, parent_extra), ("c" * 64, "2026-01-01T00:00:01.000Z", "b" * 64, child_extra)):
        # the row as the old fold left it, and the event behind it: the payload carries every column the record stated (a real object record
        # states every NOT NULL column), the lineage keys under its parked extra, the folded state WITHOUT the two new columns
        row = _valid_row("object", object_id=oid, extra_json=extra, **obj)
        row = {k: v for k, v in row.items() if k not in ("parent_object_id", "named_role")}
        # the stamps of exactly the columns the event states (the fold stamps a stated column; extra_json is the accumulation, never stamped)
        row["field_versions_json"] = RED.canonical_json({c: [stamp, sha] for c in row if c in S.payload_columns("object") and c != "extra_json"})
        _insert_row(conn, "object", row)
        seeded[oid] = row
        state = {k: v for k, v in row.items() if k not in ("ingest_seq", "record_uri", "field_versions_json")}
        payload = {k: v for k, v in state.items() if k != "extra_json"}
        payload["_kind"] = "full"
        payload["_extra"] = _json.loads(extra)
        _insert_row(conn, "row_history", _valid_row("row_history", table_name="object", natural_key_json=RED.canonical_json({"object_id": oid}), written_utc=stamp,
                                                    record_sha256=sha, record_type="object", record_uri="synthetic://o", payload_json=RED.canonical_json(payload),
                                                    folded_state_json=RED.canonical_json(state)))
    utc = {k: "2026-01-01T00:00:00.000Z" for k in ("queued_utc", "started_utc", "generated_utc")}
    _insert_row(conn, "run", _valid_row("run", run_id="r" * 64, lifecycle_state="generated", object_id="p" * 64, ingest_seq=5, record_uri="synthetic://a", **utc))
    _insert_row(conn, "judge", _valid_row("judge", judge_id="j"))
    _insert_row(conn, "judgment", _valid_row("judgment", judgment_id="jm1", run_id="r" * 64, judge_id="j", status="complete", ingest_seq=6, record_uri="synthetic://j"))
    _insert_row(conn, "record_refusal", _valid_row("record_refusal", record_sha256="e" * 64, record_uri="synthetic://e#1", reason="refused"))
    _insert_row(conn, "blocked_records", _valid_row("blocked_records", record_sha256="b" * 64, record_uri="synthetic://b#1", record_type="run_state",
                                                    missing_parent_table="run", missing_parent_id="zz"))
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    views = tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"))
    others = tuple(t for t in S.ALL_TABLES if t not in RRF_TABLES + FO_TABLES + AF_TABLES + FU4_TABLES) + tuple(v for v in views if v not in TF_VIEWS + RRF_VIEWS)
    others += tuple(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('index', 'trigger') AND sql IS NOT NULL"))
    others_before = _object_sql(conn, others)
    cols_before = {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] for t in ("object", "judgment", "record_refusal")}
    text_before = {t: conn.execute("SELECT sql, rootpage FROM sqlite_master WHERE name = ?", (t,)).fetchone() for t in ("object", "judgment", "record_refusal")}
    rows_before = {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}")) for t in ("judgment", "record_refusal", "blocked_records", "row_history")}
    object_before = {r[0]: tuple(r) for r in conn.execute("SELECT * FROM object")}
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(version, M.code_version() + 1))    # this file, then the fold-observation migration
    notes = []
    assert M.apply_migrations(conn, notes) == list(range(version, M.code_version() + 1))
    assert M.db_version(conn) == M.code_version() and DB.schema_drift(conn) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = ?", (version,)).fetchone()[0] == "record_refusal_fact"
    assert notes == [f"{version:04d}_record_refusal_fact: lifted object lineage from extra_json: parent_object_id 1, named_role 1 (1 object row(s) rewritten, "
                     "1 history event(s) refolded, 2 object key(s) checked)",
                     # the newer file's post-step on the same rows: the seeded run's extra_json carries no serving_jobs, so nothing is derived
                     f"{fo_version:04d}_fold_observation: derived fold observations: 0 observation(s) on 0 run(s) (0 run(s) carrying extra_json.serving_jobs checked)"], notes
    assert _object_sql(conn, others) == others_before                                    # nothing else moved: every other table, index, trigger, v_* object
    # the five columns appended after each table's last live column; the stored text is the code's CREATE, not a renamed rebuild
    for t, added in (("object", ["parent_object_id", "named_role"]), ("judgment", ["claimed_instrument_id", "claimed_required_judge_id"]), ("record_refusal", ["judged_with"])):
        assert [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] == cols_before[t] + added, t
        # an ALTER, not a rebuild: the b-tree (rootpage) and the first line of the stored CREATE text are the ones the table had (object's reads
        # `CREATE TABLE "object"` since 0002 rebuilt it; a rebuild here would have renamed every one), and no `__new` table was made
        sql_after, root_after = conn.execute("SELECT sql, rootpage FROM sqlite_master WHERE name = ?", (t,)).fetchone()
        if t not in SLR_REBUILT_TABLES:                                                  # 0032 rebuilds `object` later in the chain
            assert root_after == text_before[t][1] and sql_after.split("\n", 1)[0] == text_before[t][0].split("\n", 1)[0], t
        assert "__new" not in sql_after and all(c in sql_after for c in added), t
    assert sorted(tuple(r) for r in conn.execute("SELECT * FROM judgment")) == [r + (None, None) for r in rows_before["judgment"]]
    assert sorted(tuple(r) for r in conn.execute("SELECT * FROM record_refusal")) == [r + (None,) for r in rows_before["record_refusal"]]
    assert sorted(tuple(r) for r in conn.execute("SELECT * FROM blocked_records")) == rows_before["blocked_records"]     # not rebuilt: rows and text untouched
    # the lift: the parent (no keys) is untouched; the child carries both columns with the lifting event's stamp, its extra kept, its
    # history event refolded -- the bytes a fresh fold of the same records writes
    parent = dict(zip([c[1] for c in conn.execute("PRAGMA table_info(object)")], conn.execute("SELECT * FROM object WHERE object_id = ?", ("p" * 64,)).fetchone()))
    child = dict(zip([c[1] for c in conn.execute("PRAGMA table_info(object)")], conn.execute("SELECT * FROM object WHERE object_id = ?", ("c" * 64,)).fetchone()))
    assert tuple(parent.values())[:-2] == object_before["p" * 64] and parent["parent_object_id"] is None and parent["named_role"] is None
    assert child["parent_object_id"] == "p" * 64 and child["named_role"] == "epoch_1" and child["extra_json"] == child_extra
    want_fv = _json.loads(seeded["c" * 64]["field_versions_json"])
    want_fv.update(named_role=["2026-01-01T00:00:01.000Z", "b" * 64], parent_object_id=["2026-01-01T00:00:01.000Z", "b" * 64])
    assert _json.loads(child["field_versions_json"]) == want_fv                             # the lifted columns carry the lifting event's stamp
    before_c = dict(zip(cols_before["object"], object_before["c" * 64]))
    assert {k: child[k] for k in cols_before["object"] if k != "field_versions_json"} == \
        {k: v for k, v in before_c.items() if k != "field_versions_json"}                   # every pre-existing column, the uri and the rank untouched
    assert (child["ingest_seq"], child["record_uri"]) == (3, "synthetic://o")
    folded = _json.loads(conn.execute("SELECT folded_state_json FROM row_history WHERE table_name = 'object' AND natural_key_json = ?", (RED.canonical_json({"object_id": "c" * 64}),)).fetchone()[0])
    assert folded["named_role"] == "epoch_1" and folded["parent_object_id"] == "p" * 64 and folded["extra_json"] == child_extra
    hist_rows = sorted(tuple(r) for r in conn.execute("SELECT * FROM row_history"))
    assert len(hist_rows) == len(rows_before["row_history"]) and sum(1 for a, b in zip(hist_rows, rows_before["row_history"]) if a != b) == 1   # ONE event refolded
    assert M.lift_object_lineage(conn).startswith("lifted object lineage from extra_json: parent_object_id 0, named_role 0 (0 object row(s) rewritten, 0 history")   # idempotent
    assert M.foreign_key_violations(conn, S.ALL_TABLES) == []
    for v in RRF_VIEWS:
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (v,)).fetchone()[0] == S.VIEW_SQL[v].strip().rstrip(";"), v
        conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()                                # selectable: every dependency exists
    assert conn.execute("SELECT COUNT(*) FROM record_refusal_fact").fetchone()[0] == 0
    conn.execute("INSERT INTO record_refusal_fact(record_sha256, record_uri, reason, judged_with, refused_utc, cycle_utc) VALUES (?, 'synthetic://e#1', 'refused', 'unverified', "
                 "'2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')", ("e" * 64,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO record_refusal_fact(record_sha256, record_uri, reason, judged_with, refused_utc, cycle_utc) VALUES (?, 'synthetic://e#1', 'refused', 'nope', "
                     "'2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')", ("e" * 64,))
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1 and conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    conn.close()
    DB.open_or_create(path).close()                                                     # the writers' open accepts it
