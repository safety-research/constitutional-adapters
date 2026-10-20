"""The metric-naming guard: no metric row files a degeneration statistic under a generic name.

`run_metric.metric_id` was free TEXT with no CHECK and no validator rule, so nothing stopped a producer from filing `score`
or `degenerate`. The guard is a BANNED set (`vocab.banned_metric_ids()`): bare generic nouns plus every unqualified
degeneration spelling — the phenomenon, the measure names, `frac_<measure>` and the `run_metric.frac_*` mirror column
names — rendered as `*_metric_id_not_generic` CHECK constraints on `run_metric`, `run_contrast` and `eval.headline_metric_id`,
and refused BY NAME in `validate.py` before any insert. `acc` stays legal: it is the spec's own capability spelling and the
eval is its qualifier; every seeded headline metric and every id the adapters emit stays legal (pinned below).
"""
from __future__ import annotations

import os
import sqlite3

import pytest
import yaml

from sisteer.registry import db as DB
from sisteer.registry import docsgen
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: ids the adapters emit today (grep of `metric_id` literals under adapters/ plus the seeds' headline metrics): none may be banned
ADAPTER_METRIC_IDS = ("asr", "acc", "petri_score", "over_refusal_graded", "strongreject_best", "asr_bon_cap4096", "residual_at_cap_share_bon_4096",
                      "queries_to_first_success", "orb_refused", "xst_refused", "mean_best_score", "best_score", "misaligned_rate", "em_rate",
                      "win_rate", "prompt_level_strict", "asr_at_n75", "coverage",
                      # the third over-refusal row: the label mean beside the graded and refused-rate rows
                      "over_refusal_label_mean")


def _db_with_a_run(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=1, n_items=2)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root))
    return conn, ids["run_ids"][0], ids["judgment_ids"][0]


def _insert_metric(conn, run_id, jid, metric_id):
    conn.execute("INSERT INTO run_metric (run_id, judgment_id, metric_id, threshold, computation_id, value, computed_by, "
                 "field_versions_json, extra_json, record_uri, ingest_seq) VALUES (?, ?, ?, -1, '', 0.5, 'producer', '{}', '{}', 'x', 99)",
                 (run_id, jid, metric_id))


# --------------------------------------------------------------------------------------------
# the DDL
# --------------------------------------------------------------------------------------------

#: metric spellings the guard refuses (one per line: several quoted ids beside the word "token" on one line read as a
#: credential assignment to the secret scanner's generic rule)
BAD_METRIC_IDS = (
    "score",
    "value",
    "metric",
    "result",
    "accuracy",
    "degenerate",
    "loop",
    "at_cap",
    "frac_degenerate",
    "classifier",
    "union5",
    "frac_at_token_cap",
    "frac_degenerate_union5",
    "at_token_cap_rate",
    # the r2 export's own spellings of the statistics it files as metric rows: routed to run_degeneration, refused here
    "at_cap_share",
    "loop_share",
    "degenerate_share",
    "degen_classifier_share",
    "degenerate_share_union5",
    "at_cap_share_retokenized",
    "degen_truncation_only_share",
)


@pytest.mark.parametrize("bad", BAD_METRIC_IDS)
def test_generic_or_unqualified_degeneration_metric_id_insert_fails_naming_the_constraint(tmp_path, bad):
    conn, rid, jid = _db_with_a_run(tmp_path)
    with pytest.raises(sqlite3.IntegrityError) as ei:
        _insert_metric(conn, rid, jid, bad)
    assert "run_metric_metric_id_not_generic" in str(ei.value)
    conn.close()


@pytest.mark.parametrize("good", ["asr", "acc", "petri_score", "over_refusal_graded", "asr_bon_cap4096", "residual_at_cap_share_bon_4096"])
def test_a_named_metric_inserts(tmp_path, good):
    conn, rid, jid = _db_with_a_run(tmp_path)
    _insert_metric(conn, rid, jid, good)
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND metric_id = ? AND record_uri = 'x'", (rid, good)).fetchone()[0] == 1
    conn.close()


def test_run_contrast_and_eval_headline_share_the_guard(tmp_path):
    conn, rid, jid = _db_with_a_run(tmp_path)
    with pytest.raises(sqlite3.IntegrityError) as ei:
        conn.execute("INSERT INTO run_contrast (run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id, anchor_judgment_id, "
                     "anchor_kind, delta_method, n_pairs, computed_by, field_versions_json, extra_json, record_uri, ingest_seq) "
                     "VALUES (?, ?, ?, 'value', -1, '', '', 'same_standup_none', 'paired_difference', 1, 'producer', '{}', '{}', 'x', 99)",
                     (rid, rid, jid))
    assert "run_contrast_metric_id_not_generic" in str(ei.value)
    with pytest.raises(sqlite3.IntegrityError) as ei2:
        conn.execute("INSERT INTO eval (eval_id, eval_category, headline_metric_id, metric_direction, attacker_required, paper_status, "
                     "field_versions_json, extra_json, record_uri, ingest_seq) VALUES ('synth_bad_eval', 'capability', 'score', 'higher_is_better', 0, "
                     "'in_paper', '{}', '{}', 'x', 99)")
    assert "eval_headline_metric_id_not_generic" in str(ei2.value)
    conn.close()


def test_every_guarded_column_renders_the_same_banned_list():
    banned = V.banned_metric_ids()
    assert len(banned) == len(set(banned)) and "score" in banned
    for table, col in S.METRIC_ID_GUARDED:
        t = S.TABLE_BY_NAME[table]
        sql = dict(t.checks)[f"{table}_{col}_not_generic"]
        assert sql == f"({col} NOT IN ({', '.join(repr(v) for v in banned)}))"


# --------------------------------------------------------------------------------------------
# the validator: refused by name before any insert
# --------------------------------------------------------------------------------------------

def _metric_rec(metric_id):
    return synth.rec("run_metric", {"run_id": "synthcampaign:syntheval:aaaaaaaaaaaa:bbbbbbbbbbbbbbbb", "judgment_id": "", "metric_id": metric_id,
                                    "value": 0.5, "computed_by": "producer"}, synth.t(0))


@pytest.mark.parametrize("bad", ["score", "degenerate", "frac_at_token_cap", "rep_ratio_loop"])
def test_validator_refuses_a_banned_metric_id_naming_the_field(bad):
    rec = _metric_rec(bad)
    out = VAL.validate_record(rec, "<memory>", R.record_sha(rec))
    assert isinstance(out, VAL.Malformed) and out.field == "metric_id", out
    assert "banned metric spelling" in out.reason and "run_degeneration" in out.reason


def test_validator_accepts_named_metric_ids():
    for good in ADAPTER_METRIC_IDS:
        rec = _metric_rec(good)
        out = VAL.validate_record(rec, "<memory>", R.record_sha(rec))
        assert not isinstance(out, VAL.Malformed), (good, out)


# --------------------------------------------------------------------------------------------
# the rule itself: derived from the degeneration vocabulary, sparing every real id
# --------------------------------------------------------------------------------------------

def test_banned_set_covers_every_degeneration_measure_and_mirror_column_and_spares_real_ids():
    banned = set(V.banned_metric_ids())
    for m in V.DEGENERATION_MEASURES:
        assert {m, f"frac_{m}", f"{m}_rate", f"{m}_frac"} <= banned, m
    assert set(V.DEGENERATION_FRAC_COLUMNS) <= banned
    assert set(V.GENERIC_METRIC_IDS) <= banned and set(V.UNQUALIFIED_DEGENERATION_METRIC_IDS) <= banned
    # the spec's own examples and every id a producer files today stay legal; the eval is `acc`'s qualifier
    assert not (banned & {"asr", "over_refusal_graded", "acc", "petri_score", "rate"})
    assert not (banned & set(ADAPTER_METRIC_IDS)), banned & set(ADAPTER_METRIC_IDS)
    # the mirror columns the DDL renders ARE the vocabulary's list (one source for the CHECK, the table and the docs)
    assert [c.name for c in S.TABLE_BY_NAME["run_metric"].columns if c.name.startswith("frac_")] == list(V.DEGENERATION_FRAC_COLUMNS)


def test_every_seeded_headline_metric_is_legal():
    with open(os.path.join(REPO, "configs", "registry", "evals.yaml"), encoding="utf-8") as fh:
        evals = yaml.safe_load(fh)["evals"]          # the seed's shape: {"evals": [ {eval_id, headline_metric_id, ...}, ... ]}
    heads = {str(e["headline_metric_id"]) for e in evals}
    # every ACCEPTED spelling an eval lists beside its declared id (`headline_metric_ids`, 2026-09-21: strongreject_bon's native pass)
    listed = {str(m) for e in evals for m in (e.get("headline_metric_ids") or [])}
    assert len(heads) >= 5 and "acc" in heads and "asr" in heads, heads
    assert "asr_bon_cap4096" in listed, listed
    assert not ((heads | listed) & set(V.banned_metric_ids())), (heads | listed) & set(V.banned_metric_ids())


def test_vocabulary_doc_states_the_rule_and_is_current():
    md = docsgen.render_vocabulary_md()
    assert "## Metric ids" in md and "run_degeneration" in md
    for v in V.banned_metric_ids():
        assert f"`{v}`" in md or v.startswith(tuple(f"{m}_" for m in V.DEGENERATION_MEASURES)) or v.startswith(("frac_", "degen_")) \
            or v in V.DEGENERATION_MEASURES, v
    assert docsgen.doc_is_current("vocabulary")
