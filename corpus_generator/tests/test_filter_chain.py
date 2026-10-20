"""Filter chain end-to-end on a synthetic mixed corpus, including hard-gate
records that score 10/10 and must still drop."""

from __future__ import annotations

import json

import pytest

from corpus_generator.filtering import ThresholdsNotCalibrated, filter_corpus
from fakes import words_for
from test_schema import golden_advice, golden_story, _lineage
from test_tokens import _stage, _word_tokenizer, _char_tokenizer
from corpus_generator.tokens import TokenCounter


ALL_THRESHOLDS = {
    "advice_alignment": 0, "advice_realism": 0,
    "story_consistency": 0, "story_quality": 0,
}


def _cfg(tmp_path, thresholds=None):
    from corpus_generator.config import RunConfig

    return RunConfig.model_validate(
        {
            "run_id": "filter-test",
            "data_dir": str(tmp_path / "data"),
            "filter": {
                "min_story_words": 30,
                "min_advice_words": 2,
                # all four set (permissive) unless a test overrides — a FINAL
                # corpus refuses null thresholds (C1)
                "thresholds": thresholds or ALL_THRESHOLDS,
            },
        }
    )


def _adv(i: int, **kw) -> dict:
    d = golden_advice(
        scenario_id=f"p00-t00-s{i:02d}",
        user=f"user {i} " + words_for(f"u{i}", 60),
        assistant=f"assistant {i} " + words_for(f"a{i}", 80),
        lineage=_lineage(custom_id=f"adv-p00-t00-s{i:02d}"),
    )
    d.update(kw)
    return d


def _sto(i: int, **kw) -> dict:
    d = golden_story(
        call_id=f"c00-u00-k{i:02d}",
        text=f"story {i} " + words_for(f"s{i}", 120),
        lineage=_lineage(custom_id=f"sto-c00-u00-k{i:02d}-y0"),
    )
    d.update(kw)
    return d


def test_advice_chain_rules_and_order(tmp_path):
    records = [
        _adv(0),  # survives
        _adv(1, lineage={"run_id": "x"}),  # schema fail
        _adv(2, assistant="As Claude, I advise patience. " + words_for("p", 60)),  # persona
        _adv(
            3,
            scores={"alignment": 10, "realism": 10, "refusal": False, "ai_named": True},
        ),  # hard gate beats perfect scores
        _adv(
            4, scores={"alignment": 9, "realism": 9, "refusal": True, "ai_named": False}
        ),  # refusal
        _adv(5, assistant="[insert advice here] " + words_for("q", 60)),  # placeholder
        _adv(6, scores={"alignment": 3, "realism": 8, "refusal": False, "ai_named": False}),
        _adv(7),  # near-dup of 8 (identical text)
        _adv(8, user=_adv(7)["user"], assistant=_adv(7)["assistant"]),
    ]
    cfg = _cfg(tmp_path, thresholds={"advice_alignment": 7, "advice_realism": 5})
    out = tmp_path / "final.jsonl"
    report = filter_corpus("advice", records, cfg, None, out)

    assert report.input_count == 9
    assert report.rule_counts == {
        "schema": 1, "persona": 1, "ai_named": 1, "refusal": 1,
        "placeholder": 1, "score": 1, "near_duplicate": 1,
    }
    survivors = [json.loads(x) for x in out.read_text().splitlines()]
    assert {r["lineage"]["custom_id"] for r in survivors} == {
        "adv-p00-t00-s00", "adv-p00-t00-s07",  # s07 kept (lexicographically first)
    }
    assert not report.score_filter_skipped
    # drop log carries reason + id
    dropped_ids = {d["id"]: d["rule"] for d in report.drops}
    assert dropped_ids["adv-p00-t00-s03"] == "ai_named"
    assert dropped_ids["adv-p00-t00-s08"] == "near_duplicate"


def test_final_refuses_null_thresholds(tmp_path):
    """C1: a FINAL corpus must never materialize with the score filter
    skipped — null thresholds are a hard refusal, naming the missing axes."""
    records = [_adv(0)]
    cfg = _cfg(tmp_path, thresholds={"advice_alignment": 7})  # realism null
    out = tmp_path / "final.jsonl"
    with pytest.raises(ThresholdsNotCalibrated, match="advice_realism"):
        filter_corpus("advice", records, cfg, None, out)
    assert not out.exists(), "refusal must not write a final"

    cfg_all_null = _cfg(tmp_path, thresholds={"advice_alignment": None})
    with pytest.raises(ThresholdsNotCalibrated, match="calibrate"):
        filter_corpus("advice", records, cfg_all_null, None, out)


def test_calibration_mode_writes_non_final_artifact(tmp_path):
    """C1: explicit calibration mode allows null thresholds, skips the score
    gate loudly, and flags the artifact as calibration in its report."""
    records = [_adv(0, scores={"alignment": 0, "realism": 0, "refusal": False, "ai_named": False})]
    cfg = _cfg(tmp_path, thresholds={"advice_alignment": None})
    out = tmp_path / "calibration.jsonl"
    report = filter_corpus("advice", records, cfg, None, out, calibration=True)
    assert report.calibration is True
    assert report.score_filter_skipped is True
    assert report.output_count == 1  # nothing dropped on scores
    rep = json.loads((tmp_path / "calibration.filter-report.json").read_text())
    assert rep["calibration"] is True


def test_calibration_paths_are_distinct(tmp_path):
    cfg = _cfg(tmp_path)
    root = tmp_path
    final = cfg.final_path(root, "advice")
    calib = cfg.calibration_path(root, "advice")
    assert final != calib
    assert calib.name.startswith("calibration-")


def test_stories_ordinary_share_reported_and_gated(tmp_path):
    """C4: the filter report carries the final-record ordinary share and a
    VISIBLE acceptance failure below the configured floor (never a drop)."""
    records = [
        _sto(0, ordinary=True),
        _sto(1, ordinary=False),
        _sto(2, ordinary=False),
        _sto(3, ordinary=False),
    ]
    cfg = _cfg(tmp_path, thresholds=ALL_THRESHOLDS)
    report = filter_corpus("stories", records, cfg, None, tmp_path / "f.jsonl")
    assert report.output_count == 4  # gate never drops records
    assert report.ordinary_share == 0.25
    assert any("ordinary_share" in f for f in report.acceptance_failures)

    # above the floor: no failure
    records_ok = [_sto(0, ordinary=True), _sto(1, ordinary=True), _sto(2, ordinary=False)]
    report_ok = filter_corpus("stories", records_ok, cfg, None, tmp_path / "g.jsonl")
    assert report_ok.acceptance_failures == []


def test_score_distributions_and_below_floor_tail_in_report(tmp_path):
    """Below-floor tail monitoring (pilot inspection §8): the filter report
    carries per-axis score distributions over ALL scored records (pre-filter,
    including ones later dropped) and per-axis below-floor drop counts."""
    records = [
        _adv(0, scores={"alignment": 9, "realism": 9, "refusal": False, "ai_named": False}),
        _adv(1, scores={"alignment": 8, "realism": 9, "refusal": False, "ai_named": False}),
        _adv(2, scores={"alignment": 10, "realism": 7, "refusal": False, "ai_named": False}),
        # hard-gated record still contributes to the pre-filter distribution
        _adv(3, scores={"alignment": 9, "realism": 8, "refusal": False, "ai_named": True}),
    ]
    cfg = _cfg(tmp_path, thresholds={"advice_alignment": 9, "advice_realism": 8})
    report = filter_corpus("advice", records, cfg, None, tmp_path / "f.jsonl")
    assert report.score_distributions == {
        "alignment": {9: 2, 8: 1, 10: 1},
        "realism": {9: 2, 7: 1, 8: 1},
    }
    assert report.score_drops_by_axis == {"alignment": 1, "realism": 1}
    assert report.rule_counts["score"] == 2
    # and it round-trips through the written report JSON
    rep = json.loads((tmp_path / "f.filter-report.json").read_text())
    assert rep["score_drops_by_axis"] == {"alignment": 1, "realism": 1}
    assert rep["score_distributions"]["alignment"]["8"] == 1


def test_shipped_configs_calibrate_vs_final_posture(tmp_path):
    """production.yaml freezes the four inspection floors (alignment >=9,
    realism >=8, cc >=8, quality >=8) and passes the final-mode
    null-threshold refusal; a config that leaves the floors null (the
    calibration posture every pilot ran under) only works under --calibrate."""
    from corpus_generator.config import RunConfig
    from corpus_generator.templates import repo_root

    prod = RunConfig.load(repo_root() / "configs" / "production.yaml")
    th = prod.filter.thresholds
    assert (th.advice_alignment, th.advice_realism) == (9, 8)
    assert (th.story_consistency, th.story_quality) == (8, 8)
    # dry validation of the exact code path a final run takes: no refusal
    for corpus in ("advice", "stories"):
        filter_corpus(corpus, [], prod, None, tmp_path / f"{corpus}.jsonl")

    pilot = RunConfig.model_validate({"run_id": "pilot-posture"})  # floors default to null
    tp = pilot.filter.thresholds
    assert all(
        v is None
        for v in (tp.advice_alignment, tp.advice_realism,
                  tp.story_consistency, tp.story_quality)
    )
    with pytest.raises(ThresholdsNotCalibrated):
        filter_corpus("advice", [], pilot, None, tmp_path / "x.jsonl")
    filter_corpus("advice", [], pilot, None, tmp_path / "x.jsonl", calibration=True)


def test_unscored_records_never_pass_the_score_gate(tmp_path):
    """A record whose scoring wave was lost (request rejected / filtered /
    truncated through the repair budget) materializes with scores=None and
    must be DROPPED with a counted reason — even under permissive thresholds,
    and in calibration mode too (calibration pass rates feed sizing)."""
    cfg = _cfg(tmp_path)  # all thresholds set (permissive) — final mode
    report = filter_corpus(
        "advice", [_adv(0, scores=None), _adv(1)], cfg, None, tmp_path / "a.jsonl"
    )
    assert report.rule_counts["unscored"] == 1
    assert report.output_count == 1
    survivors = [json.loads(x) for x in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert all(r["scores"] is not None for r in survivors)

    report2 = filter_corpus(
        "stories", [_sto(0, scores=None), _sto(1)], cfg, None, tmp_path / "s.jsonl"
    )
    assert report2.rule_counts["unscored"] == 1
    assert report2.output_count == 1

    # calibration mode (null thresholds) drops it too, counted
    cfg_null = _cfg(tmp_path, thresholds={"advice_alignment": None})
    report3 = filter_corpus(
        "advice", [_adv(0, scores=None)], cfg_null, None,
        tmp_path / "c.jsonl", calibration=True,
    )
    assert report3.rule_counts["unscored"] == 1
    assert report3.output_count == 0


def test_metadata_persona_mentions_counted_not_dropped(tmp_path):
    """Pilot policy fix: Claude/Anthropic in generated NON-trainable metadata
    (the S4 critique notes, draft/diagnostic lineage, rationales) is counted
    + sampled in the report but the record survives; trainable text still
    hard-drops. (Pilot: 45/144 stories lost to rewrite_problems quoting the
    constitution.)"""
    cfg = _cfg(tmp_path)
    # stories: critique-note mention survives; trainable-text mention drops
    sto_records = [
        _sto(0, rewrite_problems="The constitution says Claude should stay honest."),
        _sto(1, text="Claude smiled and " + words_for("leak", 120)),  # trainable -> drop
        _sto(2),
    ]
    out = tmp_path / "sto.jsonl"
    report = filter_corpus("stories", sto_records, cfg, None, out)
    assert report.rule_counts.get("persona") == 1  # only the trainable leak
    survivors = {json.loads(x)["lineage"]["custom_id"] for x in out.read_text().splitlines()}
    assert "sto-c00-u00-k00-y0" in survivors, "metadata mention must not drop"
    assert report.persona_metadata_mentions == 1
    assert report.persona_metadata_fields == {"rewrite_problems": 1}
    assert report.persona_metadata_samples[0]["field"] == "rewrite_problems"
    assert "Claude" in report.persona_metadata_samples[0]["match"]

    # advice: draft/diagnostic lineage mentions survive and are counted per path
    adv_records = [
        _adv(0,
             draft_system="You are Claude, a helpful assistant.",
             diagnostic_response="I am Claude, made by Anthropic."),
        _adv(1),
    ]
    out2 = tmp_path / "adv.jsonl"
    report2 = filter_corpus("advice", adv_records, cfg, None, out2)
    assert "persona" not in report2.rule_counts
    assert report2.output_count == 2
    assert report2.persona_metadata_mentions == 3  # 1 draft + 2 diagnostic hits
    assert set(report2.persona_metadata_fields) == {"draft_system", "diagnostic_response"}


def test_stories_chain(tmp_path):
    records = [
        _sto(0),  # survives
        _sto(
            1,
            scores={"constitution_consistency": 10, "quality": 10,
                    "ai_named": False, "factual_claims": True},
        ),  # hard gate
        _sto(2, text="too short."),  # min words
        _sto(3, text="see https://example.com " + words_for("l", 120)),  # link
        _sto(4, text=("looping phrase alpha beta " * 60)),  # degenerate repetition
        _sto(5, text=words_for("long", 400), target_words=100),  # >3x target
        _sto(
            6,
            scores={"constitution_consistency": 5, "quality": 9,
                    "ai_named": False, "factual_claims": False},
        ),
    ]
    cfg = _cfg(tmp_path, thresholds={"story_consistency": 7, "story_quality": 6})
    cfg.filter.repeat_ngram = 4
    report = filter_corpus("stories", records, cfg, None, tmp_path / "f.jsonl")
    assert report.rule_counts == {
        "factual_claims": 1, "too_short": 1, "link": 1,
        "degenerate_repetition": 1, "too_long": 1, "score": 1,
    }
    assert report.output_count == 1


def test_token_accounting_min_across_tokenizers(tmp_path):
    tok_dir = tmp_path / "toks"
    tok_dir.mkdir()
    _stage(tok_dir, {"words": _word_tokenizer(), "chars": _char_tokenizer()})
    counter = TokenCounter.load(tok_dir)
    records = [_adv(0, user="hello world", assistant="hello again", system=None)]
    cfg = _cfg(tmp_path)
    out = tmp_path / "final.jsonl"
    report = filter_corpus("advice", records, cfg, counter, out)
    survivor = json.loads(out.read_text().splitlines()[0])
    # rendered text = "hello world\n\nhello again" -> 4 word tokens
    assert survivor["token_counts"]["words"] == 4
    assert report.token_totals["words"] == 4
    assert report.official_tokens == min(report.token_totals.values()) == 4


def test_filter_report_written_next_to_output(tmp_path):
    cfg = _cfg(tmp_path)
    out = tmp_path / "final.jsonl"
    filter_corpus("advice", [_adv(0)], cfg, None, out)
    rep = json.loads((tmp_path / "final.filter-report.json").read_text())
    assert rep["corpus"] == "advice"
    assert rep["input_count"] == 1
