"""C2: the mirror validates COMPLETENESS before transfer — both finals + their
filter reports must exist and be genuinely final (no calibration artifacts, no
skipped score filter), the acceptance axes must be met, the finals must
re-verify schema/persona-clean, and remote verification compares the EXACT
file set (extra remote files fail, not just missing ones)."""

from __future__ import annotations

import json

import pytest

from corpus_generator.config import RunConfig, filter_report_path
from corpus_generator.mirror import ReleaseGateError, compare_file_sets, verify_release
from test_schema import golden_advice, golden_story, _lineage
from fakes import words_for


def _cfg(tmp_path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": "mirror-test",
            "data_dir": str(tmp_path / "data"),
            # tiny targets so a handful of synthetic records constitutes a release
            "targets": {
                "advice_min_tokens": 10,
                "stories_min_count": 2,
                "stories_min_tokens": 10,
            },
        }
    )


def _write_final(data_dir, corpus, records, *, official_tokens=100,
                 skipped=False, calibration=False, reported_count=None, suffix=""):
    p = data_dir / corpus / f"final-mirror-test{suffix}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = {
        "corpus": corpus,
        "calibration": calibration,
        "score_filter_skipped": skipped,
        "output_count": len(records) if reported_count is None else reported_count,
        "official_tokens": official_tokens,
    }
    filter_report_path(p).write_text(json.dumps(report))
    return p


def _records(corpus, n=3):
    if corpus == "advice":
        return [
            golden_advice(
                scenario_id=f"p00-t00-s{i:02d}",
                user=f"user {i} " + words_for(f"u{i}", 20),
                lineage=_lineage(custom_id=f"adv-p00-t00-s{i:02d}"),
            )
            for i in range(n)
        ]
    return [
        golden_story(
            call_id=f"c00-u00-k{i:02d}",
            text=f"story {i} " + words_for(f"s{i}", 30),
            lineage=_lineage(custom_id=f"sto-c00-u00-k{i:02d}-y0"),
        )
        for i in range(n)
    ]


def _full_release(tmp_path, **overrides):
    cfg = _cfg(tmp_path)
    data_dir = cfg.resolve_data_dir(tmp_path)
    for corpus in ("advice", "stories"):
        kw = overrides.get(corpus, {})
        _write_final(data_dir, corpus, _records(corpus), **kw)
    return cfg, data_dir


def test_complete_release_passes(tmp_path):
    cfg, data_dir = _full_release(tmp_path)
    info = verify_release(cfg, data_dir)
    assert info["advice"]["records"] == 3
    assert info["stories"]["records"] == 3


def test_missing_final_refused(tmp_path):
    cfg = _cfg(tmp_path)
    data_dir = cfg.resolve_data_dir(tmp_path)
    _write_final(data_dir, "advice", _records("advice"))  # stories never written
    with pytest.raises(ReleaseGateError, match="stories: final missing"):
        verify_release(cfg, data_dir)


def test_missing_filter_report_refused(tmp_path):
    cfg, data_dir = _full_release(tmp_path)
    filter_report_path(data_dir / "advice" / "final-mirror-test.jsonl").unlink()
    with pytest.raises(ReleaseGateError, match="filter report missing"):
        verify_release(cfg, data_dir)


def test_skipped_score_filter_refused(tmp_path):
    cfg, data_dir = _full_release(tmp_path, stories={"skipped": True})
    with pytest.raises(ReleaseGateError, match="SKIPPED"):
        verify_release(cfg, data_dir)


def test_calibration_artifact_refused(tmp_path):
    cfg, data_dir = _full_release(tmp_path, advice={"calibration": True})
    with pytest.raises(ReleaseGateError, match="CALIBRATION"):
        verify_release(cfg, data_dir)


def test_under_target_axes_refused(tmp_path):
    cfg, data_dir = _full_release(tmp_path, advice={"official_tokens": 5})
    with pytest.raises(ReleaseGateError, match="official tokens 5 < target"):
        verify_release(cfg, data_dir)

    cfg2, data_dir2 = _full_release(tmp_path / "b", stories={"official_tokens": 3})
    with pytest.raises(ReleaseGateError, match="stories: official tokens"):
        verify_release(cfg2, data_dir2)

    # count axis: report and file agree but sit under the count target
    cfg3 = _cfg(tmp_path / "c")
    data_dir3 = cfg3.resolve_data_dir(tmp_path / "c")
    _write_final(data_dir3, "advice", _records("advice"))
    _write_final(data_dir3, "stories", _records("stories", n=1))
    with pytest.raises(ReleaseGateError, match="1 records < target 2"):
        verify_release(cfg3, data_dir3)


def test_stale_report_count_mismatch_refused(tmp_path):
    cfg, data_dir = _full_release(tmp_path, advice={"reported_count": 7})
    with pytest.raises(ReleaseGateError, match="stale report"):
        verify_release(cfg, data_dir)


def test_persona_leak_in_trainable_final_text_refused(tmp_path):
    dirty = _records("stories")
    dirty[1]["text"] = "Claude smiled at the frustrated user and helped anyway."
    cfg = _cfg(tmp_path)
    data_dir = cfg.resolve_data_dir(tmp_path)
    _write_final(data_dir, "advice", _records("advice"))
    _write_final(data_dir, "stories", dirty)
    with pytest.raises(ReleaseGateError, match="persona leak in trainable"):
        verify_release(cfg, data_dir)


def test_metadata_mention_in_final_does_not_block_release(tmp_path):
    """Persona policy 2026-09-01: a Claude mention in non-trainable metadata
    (critique notes, lineage prose) is report-counted at filter time, not a
    release blocker — only trainable text gates the mirror."""
    ok = _records("stories")
    ok[1]["rewrite_problems"] = "The constitution says Claude should stay honest."
    ok[2]["category"] = "Anthropic office stories"
    cfg = _cfg(tmp_path)
    data_dir = cfg.resolve_data_dir(tmp_path)
    _write_final(data_dir, "advice", _records("advice"))
    _write_final(data_dir, "stories", ok)
    info = verify_release(cfg, data_dir)
    assert info["stories"]["records"] == 3


def test_schema_invalid_final_refused(tmp_path):
    bad = _records("advice")
    del bad[0]["assistant"]
    cfg = _cfg(tmp_path)
    data_dir = cfg.resolve_data_dir(tmp_path)
    _write_final(data_dir, "advice", bad)
    _write_final(data_dir, "stories", _records("stories"))
    with pytest.raises(ReleaseGateError, match="schema-invalid"):
        verify_release(cfg, data_dir)


def test_v2_release_gate_requires_v2_finals(tmp_path):
    """suffix="-v2" gates the v2 files with the same checks — a complete v1
    release alone must not pass it."""
    cfg, data_dir = _full_release(tmp_path)
    verify_release(cfg, data_dir)  # v1 passes
    with pytest.raises(ReleaseGateError, match="final missing"):
        verify_release(cfg, data_dir, suffix="-v2")
    for corpus in ("advice", "stories"):
        _write_final(data_dir, corpus, _records(corpus, n=4), suffix="-v2")
    info = verify_release(cfg, data_dir, suffix="-v2")
    assert info["advice"]["records"] == 4
    # v2 gate enforces the same targets against the v2 files
    _write_final(data_dir, "advice", _records("advice", n=4),
                 official_tokens=5, suffix="-v2")
    with pytest.raises(ReleaseGateError, match="official tokens 5 < target"):
        verify_release(cfg, data_dir, suffix="-v2")


def test_v2_collection_includes_both_generations_and_extra_runs(tmp_path):
    from corpus_generator.mirror import collect_files

    cfg, data_dir = _full_release(tmp_path)
    for corpus in ("advice", "stories"):
        _write_final(data_dir, corpus, _records(corpus, n=4), suffix="-v2")
    run_dir = cfg.run_dir(tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("{}")
    extra = data_dir / "runs" / "regen-test"
    extra.mkdir(parents=True)
    (extra / "manifest.json").write_text("{}")
    (run_dir / "report-v2").mkdir()
    (run_dir / "report-v2" / "report.md").write_text("# v2")
    files = collect_files(cfg, data_dir, run_dir, suffixes=("", "-v2"),
                          extra_run_dirs=(extra,))
    names = {str(f.relative_to(data_dir)) for f in files}
    assert "advice/final-mirror-test.jsonl" in names
    assert "advice/final-mirror-test-v2.jsonl" in names
    assert "stories/final-mirror-test-v2.filter-report.json" in names
    assert "runs/regen-test/manifest.json" in names
    assert "runs/mirror-test/manifest.json" in names
    assert "runs/mirror-test/report-v2/report.md" in names


def test_remote_comparison_is_exact_set_equality():
    """Extra remote files are mismatches too — not just per-local presence."""
    a = {"bytes": 10, "sha256": "aa"}
    b = {"bytes": 20, "sha256": "bb"}
    assert compare_file_sets({"x": a}, {"x": a}) == {}
    # missing remote
    assert "x" in compare_file_sets({"x": a}, {})
    # content mismatch
    assert "x" in compare_file_sets({"x": a}, {"x": b})
    # EXTRA remote file (a stale earlier mirror, a foreign run's leftovers)
    extra = compare_file_sets({"x": a}, {"x": a, "stale.jsonl": b})
    assert extra == {"stale.jsonl": {"local": None, "remote": b}}
