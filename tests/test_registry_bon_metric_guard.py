"""The emit-time best-of-N metric guard. A PURE rule in `vocab.bon_metric_refusal`, called from
`emit.RecordWriter.emit_metrics` and `hooks.JudgingScope.metrics` only -- never from `validate_record`, whose rules fold the record log
at every rebuild (a validator rule would alter folded rows: the 152 live rows at thresholds other than 0.5 and the 4 `asr_at_n_max` rows
counted stay exactly as they fold today). Scoped to the live conventions: `asr_bon_cap4096` at a threshold in
{0.25, 0.5, 0.75} (the export's three judge cutoffs); `bon_nauc` at 0.5 or -1.0 (the AUC's no-threshold sentinel); `asr_at_n<digits>`
carrying `extra.bon_n == N` (the N belongs in the id and in extra); the spelling `asr_at_n_max` REFUSED for new records with a
message naming the rule (the 4 live rows stay as legacy, their `producer_metric_spelling` the provenance; `paper.METRIC_LEGACY` maps the
spelling). Every other metric id passes untouched.

Beside it, the DENOMINATOR guard (`vocab.counted_metric_refusal`, the same pattern): a COUNTED `run_metric` row -- one carrying a
denominator `n` -- with `n == 0` is a non-measurement and is refused at emission, the whole batch, nothing written; a row with no
denominator (`n` absent or null: not every metric has one) is not counted and passes, and a zero numerator over a non-zero
denominator is a measurement. Emit-side only: the validator never carries it, so the live rows that read `n = 0` fold as they do.
The denominator is judged by its numeric value whatever its type (0, 0.0, '0', Decimal(0) are n = 0); a BOOLEAN n is malformed and
refused by name (a boolean is never a denominator, and the validator folds a boolean into its integer, so an n = False the guard
skipped reached the store as n = 0); an n with no numeric reading ('abc') is refused by name as not a readable count, never skipped.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal

import pytest

from sisteer.registry import emit, hooks
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V
from sisteer.registry import paper

PRODUCER = emit.producer_block("tests.bon_guard", code_sha="a" * 40, substrate="grading_host")


def _row(metric_id, threshold=-1.0, **extra):
    r = {"run_id": "r" * 64, "judgment_id": "j" * 64, "metric_id": metric_id, "threshold": threshold, "value": 0.5, "k": 5, "n": 10,
         "computed_by": "producer", "computed_utc": "2026-10-06T00:00:00.000Z", "computation_id": ""}
    if extra:
        r["extra"] = dict(extra)
    return r


# --------------------------------------------------------------------------------------------
# the pure rule
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("row", [
    _row("asr_bon_cap4096", 0.25), _row("asr_bon_cap4096", 0.5), _row("asr_bon_cap4096", 0.75),
    _row("bon_nauc", 0.5), _row("bon_nauc", -1.0),
    _row("asr_at_n1", 0.5, bon_n=1), _row("asr_at_n64", 0.5, bon_n=64), _row("asr_at_n75", -1.0, bon_n="75"),
    _row("asr", 0.5), _row("acc"), _row("mean_normalized_auc"), _row("over_refusal_graded", 0.5), _row("residual_at_cap_share_bon_4096"),
])
def test_the_live_conventions_pass(row):
    assert V.bon_metric_refusal(row) is None


@pytest.mark.parametrize("row, needle", [
    (_row("asr_bon_cap4096", 0.3), "asr_bon_cap4096"),
    (_row("asr_bon_cap4096", -1.0), "0.25, 0.5, 0.75"),
    (_row("asr_bon_cap4096"), "threshold"),
    (_row("bon_nauc", 0.75), "bon_nauc"),
    (_row("bon_nauc", 0.25), "0.5, -1.0"),
    (_row("asr_at_n64", 0.5), "extra.bon_n"),
    (_row("asr_at_n64", 0.5, bon_n=32), "extra.bon_n"),
    (_row("asr_at_n64", 0.5, bon_n="sixty-four"), "extra.bon_n"),
    (_row("asr_at_n_max", 75.0), "asr_at_n_max"),
    (_row("asr_at_n_max", 75.0, bon_n=75), "D870"),
])
def test_the_rule_refuses_by_name(row, needle):
    why = V.bon_metric_refusal(row)
    assert why is not None and needle in why, (row, why)
    assert "D934" in why


def test_the_refusal_for_the_legacy_spelling_names_the_spelling_the_producer_should_write():
    why = V.bon_metric_refusal(_row("asr_at_n_max", 75.0))
    assert "asr_at_n<N>" in why and "producer_metric_spelling" in why
    # the legacy spelling is mapped on the paper side, so the 4 live rows keep reading as the `asr` curve point they are
    assert paper.METRIC_LEGACY.get("asr_at_n_max") == "asr"
    assert V.BON_LEGACY_METRIC_SPELLINGS == ("asr_at_n_max",)


def test_a_threshold_given_as_text_or_missing_is_judged_by_value():
    assert V.bon_metric_refusal(_row("asr_bon_cap4096", "0.5")) is None
    assert V.bon_metric_refusal(_row("bon_nauc", "-1")) is None
    r = _row("asr_bon_cap4096")
    del r["threshold"]
    assert "threshold" in (V.bon_metric_refusal(r) or "")


# --------------------------------------------------------------------------------------------
# the two emit sites refuse; the validator does not
# --------------------------------------------------------------------------------------------


def _writer(tmp_path):
    # enabled explicitly: the test session runs with emission OFF (tests/conftest.py sets SISTEER_REGISTRY_DISABLED=1), and a disabled writer
    # refuses nothing by contract (the last assertion below)
    return emit.RecordWriter(str(tmp_path / "spool"), "synthcampaign", "r" * 64, dict(PRODUCER), disabled=False)


def test_emit_metrics_refuses_a_bad_row_before_writing_and_writes_a_good_batch(tmp_path):
    w = _writer(tmp_path)
    with pytest.raises(emit.MetricRefused, match="asr_at_n_max"):
        w.emit_metrics([_row("asr", 0.5), _row("asr_at_n_max", 75.0)])
    assert not w.written and (not os.path.isdir(w.dir) or not [f for f in os.listdir(w.dir) if "run_metric" in f])
    with pytest.raises(emit.MetricRefused, match="asr_bon_cap4096"):
        w.emit_metrics([_row("asr_bon_cap4096", 0.3)])
    path = w.emit_metrics([_row("asr", 0.5), _row("asr_bon_cap4096", 0.75), _row("asr_at_n64", 0.5, bon_n=64), _row("bon_nauc", -1.0)])
    assert path and os.path.isfile(path)
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    assert [r["payload"]["metric_id"] for r in recs] == ["asr", "asr_bon_cap4096", "asr_at_n64", "bon_nauc"]
    # a disabled writer writes nothing and refuses nothing (the no-op contract of every emitter)
    off = emit.RecordWriter(str(tmp_path / "spool2"), "synthcampaign", "r" * 64, dict(PRODUCER), disabled=True)
    assert off.emit_metrics([_row("asr_at_n_max", 75.0)]) is None


def test_judging_scope_metrics_refuses_the_same_way_with_the_scopes_defaults_filled_in(tmp_path):
    scope = hooks.JudgingScope("r" * 64, "synthcampaign", "judge:x", spool=str(tmp_path / "spool"), session="s", push_script=None, disabled=False)
    with pytest.raises(emit.MetricRefused, match="extra.bon_n"):
        scope.metrics([{"metric_id": "asr_at_n64", "threshold": 0.5, "value": 0.7, "n": 10}])
    with pytest.raises(emit.MetricRefused, match="asr_at_n_max"):
        scope.metrics([{"metric_id": "asr_at_n_max", "threshold": 64.0, "value": 0.7, "n": 10, "producer_metric_spelling": "asr_at_n[64]"}])
    path = scope.metrics([{"metric_id": "mean_normalized_auc", "value": 0.42, "n": 10}, {"metric_id": "asr_at_n64", "threshold": 0.5, "value": 0.7, "n": 10, "extra": {"bon_n": 64}}])
    assert path and os.path.isfile(path)
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    assert [r["payload"]["metric_id"] for r in recs] == ["mean_normalized_auc", "asr_at_n64"] and recs[1]["payload"]["judgment_id"] == scope.judgment_id


def test_validate_record_is_not_changed_so_the_live_rows_still_fold():
    """The STOP class: the guard never enters the validator. A `run_metric` record at the spelling and the thresholds the live log holds
    (asr_at_n_max @75, asr_bon_cap4096 @0.3, bon_nauc @0.75, asr_at_n64 with no bon_n) validates exactly as any well-formed metric record."""
    for row in (_row("asr_at_n_max", 75.0), _row("asr_bon_cap4096", 0.3), _row("bon_nauc", 0.75), _row("asr_at_n64", 0.5)):
        rec = emit.make_record("run_metric", row, PRODUCER, "p" * 32, "2026-10-06T00:00:00.000Z")
        v = VAL.validate_record(rec, "<test>")
        assert not isinstance(v, VAL.Malformed), (row["metric_id"], getattr(v, "reason", None))
    src = open(VAL.__file__, encoding="utf-8").read()
    assert "bon_metric_refusal" not in src and "asr_at_n_max" not in src


# --------------------------------------------------------------------------------------------
# the denominator guard: a counted row with n = 0 is a non-measurement (emit-side, the whole batch, never the validator)
# --------------------------------------------------------------------------------------------


def _without(row, *keys):
    return {k: v for k, v in row.items() if k not in keys}


@pytest.mark.parametrize("row", [
    _row("asr", 0.5),                                            # k 5 / n 10: a measurement
    dict(_row("asr", 0.5), k=0),                                 # a zero NUMERATOR over a non-zero denominator is a measurement
    _without(_row("mean_normalized_auc"), "k", "n"),             # no denominator at all: not counted
    dict(_row("mean_normalized_auc"), k=None, n=None),           # an explicit null denominator: not counted (the schema's own why-null)
    dict(_row("asr", 0.5), n=7),                                 # a count of seven: a measurement
    dict(_row("asr", 0.5), n=1),                                 # the smallest count there is
    dict(_row("asr", 0.5), n="10"),                              # a count given as text is judged by value (the validator's column rule names the type)
    dict(_row("asr", 0.5), n=10.0),                              # a count given as a float is judged by value, likewise
    dict(_row("asr", 0.5), n=Decimal(10)),                       # and as a Decimal
])
def test_the_denominator_rule_passes_measurements_and_rows_without_a_denominator(row):
    assert V.counted_metric_refusal(row) is None


@pytest.mark.parametrize("row", [
    dict(_row("asr", 0.5), k=0, n=0),
    dict(_row("asr", 0.5), k=0, n=0, value=None),
    dict(_row("acc"), k=0, n="0"),                               # zero given as text: judged by value
    dict(_row("acc"), k=0, n=0.0),                               # zero given as a float: judged by value
    dict(_row("acc"), k=0, n=Decimal(0)),                        # zero given as a Decimal: judged by value
    dict(_row("acc"), k=0, n=" 0 "),                             # zero in whitespace is still zero
    dict(_without(_row("over_refusal_graded", 0.5), "k"), n=0),  # a denominator with no numerator beside it is still a count of nothing
])
def test_the_denominator_rule_refuses_a_counted_row_with_n_zero_by_name(row):
    why = V.counted_metric_refusal(row)
    assert why is not None and "n = 0" in why and "non-measurement" in why and row["metric_id"] in why, (row, why)


@pytest.mark.parametrize("row, needle", [
    (dict(_row("asr", 0.5), k=0, n=False), "boolean"),           # the finding: False read as int('False') raised, the guard skipped, the validator stored n = 0
    (dict(_row("asr", 0.5), n=True), "boolean"),                 # True is no more a denominator than False (the validator would store n = 1)
    (dict(_row("asr", 0.5), n="abc"), "not a readable count"),   # no numeric reading: refused by name, never skipped
    (dict(_row("asr", 0.5), n="ten"), "not a readable count"),   # the row the first guard let through on the validator's promise
    (dict(_row("asr", 0.5), n=[10]), "not a readable count"),    # a list where a count belongs
    (dict(_row("asr", 0.5), n=float("nan")), "not a readable count"),
    (dict(_row("asr", 0.5), n=float("inf")), "not a readable count"),
])
def test_the_denominator_rule_refuses_a_malformed_n_by_name_rather_than_skipping_it(row, needle):
    why = V.counted_metric_refusal(row)
    assert why is not None and needle in why and row["metric_id"] in why and repr(row["n"]) in why, (row, why)
    assert "n = 0" not in why.split(":")[0], why                 # a malformed n is its own refusal, not a zero one


def test_a_boolean_n_is_what_the_validator_folds_into_zero_so_the_guard_must_refuse_it_first():
    """The hole the boolean refusal closes, pinned: the validator folds a boolean into its integer for every INTEGER column, so a
    `run_metric` record carrying `n = False` VALIDATES, with `n` normalised to 0 -- the n = 0 row that no producer files. The
    validator stays as it is (a fold-time rule would alter replay); the writer is where the row is refused."""
    rec = emit.make_record("run_metric", dict(_row("asr", 0.5), k=0, n=False), PRODUCER, "p" * 32, "2026-10-06T00:00:00.000Z")
    v = VAL.validate_record(rec, "<test>")
    assert isinstance(v, VAL.Validated) and v.payload["n"] == 0 and v.payload["n"] is not False, v
    assert V.counted_metric_refusal(dict(_row("asr", 0.5), k=0, n=False)) is not None


def test_emit_metrics_refuses_a_whole_batch_holding_an_n_zero_row_and_writes_nothing(tmp_path):
    w = _writer(tmp_path)
    with pytest.raises(emit.MetricRefused, match="n = 0"):
        w.emit_metrics([_row("asr", 0.5), dict(_row("acc"), k=0, n=0, value=None)])
    assert not w.written and (not os.path.isdir(w.dir) or not [f for f in os.listdir(w.dir) if "run_metric" in f])
    # a boolean denominator refuses the batch by name the same way -- the row that reached the store as n = 0 before
    with pytest.raises(emit.MetricRefused, match="boolean"):
        w.emit_metrics([_row("asr", 0.5), dict(_row("acc"), k=0, n=False, value=None)])
    with pytest.raises(emit.MetricRefused, match="not a readable count"):
        w.emit_metrics([_row("asr", 0.5), dict(_row("acc"), n="abc")])
    assert not w.written and (not os.path.isdir(w.dir) or not [f for f in os.listdir(w.dir) if "run_metric" in f])
    # the batch without the empty row writes, a row with no denominator and a zero-numerator measurement beside it
    path = w.emit_metrics([_row("asr", 0.5), _without(_row("mean_normalized_auc"), "k", "n"), dict(_row("asr", 0.5), k=0)])
    assert path and os.path.isfile(path)
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    assert [(r["payload"]["metric_id"], r["payload"].get("k"), r["payload"].get("n")) for r in recs] == [("asr", 5, 10), ("mean_normalized_auc", None, None), ("asr", 0, 10)]
    # the two guards share the one refusal: a best-of-N row and an n = 0 row in one batch name the first offender, nothing written
    with pytest.raises(emit.MetricRefused, match="n = 0"):
        w.emit_metrics([dict(_row("asr", 0.5), k=0, n=0), _row("asr_at_n_max", 75.0)])
    assert len(w.written) == 1
    # a disabled writer writes nothing and refuses nothing (the no-op contract of every emitter)
    off = emit.RecordWriter(str(tmp_path / "spool2"), "synthcampaign", "r" * 64, dict(PRODUCER), disabled=True)
    assert off.emit_metrics([dict(_row("asr", 0.5), k=0, n=0)]) is None


def test_judging_scope_metrics_refuses_an_n_zero_row_through_the_writer_and_files_a_zero_numerator(tmp_path):
    scope = hooks.JudgingScope("r" * 64, "synthcampaign", "judge:x", spool=str(tmp_path / "spool"), session="s", push_script=None, disabled=False)
    before = list(scope.writer.written)                          # the scope's opening judgment record; nothing joins it on a refusal
    with pytest.raises(emit.MetricRefused, match="n = 0"):
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.7, "k": 7, "n": 10}, {"metric_id": "acc", "value": None, "k": 0, "n": 0}])
    assert scope.writer.written == before
    with pytest.raises(emit.MetricRefused, match="boolean"):
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.7, "k": 7, "n": 10}, {"metric_id": "acc", "value": None, "k": 0, "n": False}])
    assert scope.writer.written == before
    path = scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.0, "k": 0, "n": 10}])
    assert path and os.path.isfile(path)
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    assert [(r["payload"]["metric_id"], r["payload"]["k"], r["payload"]["n"]) for r in recs] == [("asr", 0, 10)]


def test_the_validator_carries_no_denominator_rule_so_the_live_n_zero_rows_still_fold():
    """The guard never enters the validator (the rule the best-of-N guard set): a `run_metric` record carrying `n = 0` validates exactly
    as any well-formed metric record, so every such row already in the log folds at the next rebuild as it folds today."""
    for row in (dict(_row("asr", 0.5), k=0, n=0), dict(_row("acc"), k=0, n=0, value=None)):
        rec = emit.make_record("run_metric", row, PRODUCER, "p" * 32, "2026-10-06T00:00:00.000Z")
        v = VAL.validate_record(rec, "<test>")
        assert not isinstance(v, VAL.Malformed), (row, getattr(v, "reason", None))
    src = open(VAL.__file__, encoding="utf-8").read()
    assert "counted_metric_refusal" not in src
