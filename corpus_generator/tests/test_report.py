"""C4: the report computes the ordinary-story shares (final-story-weighted and
category-weighted) from the propagated S1 flag and surfaces a VISIBLE
acceptance-gate failure below the configured floor."""

from __future__ import annotations

import json

from corpus_generator.config import RunConfig
from corpus_generator.report import build_report
from test_schema import golden_story, _lineage


def _cfg(tmp_path, min_share=0.5) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": "report-test",
            "data_dir": str(tmp_path / "data"),
            "targets": {"stories_min_ordinary_share": min_share},
        }
    )


def _write_finals(data_dir, stories):
    p = data_dir / "stories" / "final-report-test.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in stories))


def _stories(ordinary_flags):
    out = []
    for i, (cat, ordinary) in enumerate(ordinary_flags):
        out.append(
            golden_story(
                category_id=cat,
                subcategory_id=f"{cat}-u00",
                ordinary=ordinary,
                call_id=f"{cat}-u00-k{i:02d}",
                text=f"story {i} about steady help.",
                lineage=_lineage(custom_id=f"sto-{cat}-u00-k{i:02d}-y0"),
            )
        )
    return out


def _build(tmp_path, flags, min_share=0.5) -> tuple[dict, str]:
    cfg = _cfg(tmp_path, min_share)
    data_dir = cfg.resolve_data_dir(tmp_path)
    run_dir = cfg.run_dir(tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_finals(data_dir, _stories(flags))
    report_dir = build_report(cfg, run_dir, data_dir)
    rjson = json.loads((report_dir / "report.json").read_text())
    md = (report_dir / "report.md").read_text()
    return rjson, md


def test_ordinary_shares_computed_both_weightings(tmp_path):
    # c00 ordinary (2 stories), c01 not (2 stories) -> by story 0.5, by cat 0.5
    rjson, md = _build(
        tmp_path, [("c00", True), ("c00", True), ("c01", False), ("c01", False)]
    )
    div = rjson["corpora"]["stories"]["diversity"]
    assert div["ordinary_share_by_story"] == 0.5
    assert div["ordinary_share_by_category"] == 0.5
    acc = rjson["corpora"]["stories"]["acceptance"]
    assert acc["ordinary_share_ok"] is True
    assert "ACCEPTANCE GATE FAILURE" not in md


def test_drift_toward_dramatic_fails_the_gate_visibly(tmp_path):
    # 1 ordinary story of 4 -> by story 0.25 < 0.5 floor
    rjson, md = _build(
        tmp_path, [("c00", True), ("c01", False), ("c02", False), ("c03", False)]
    )
    acc = rjson["corpora"]["stories"]["acceptance"]
    assert acc["ordinary_share_ok"] is False
    assert acc["ordinary_share_by_story"] == 0.25
    assert "ACCEPTANCE GATE FAILURE" in md
    assert "ordinary story share" in md


def test_gate_floor_is_configurable(tmp_path):
    rjson, md = _build(
        tmp_path,
        [("c00", True), ("c01", False), ("c02", False), ("c03", False)],
        min_share=0.2,
    )
    assert rjson["corpora"]["stories"]["acceptance"]["ordinary_share_ok"] is True
    assert "ACCEPTANCE GATE FAILURE" not in md


def test_stage_table_surfaces_content_filtered_counts(tmp_path):
    """content_filtered attrition is request-level (no record ever exists for
    the filter to count), so it surfaces where refusals do: the wave/stage
    status counts, rendered in the report's stage table."""
    from corpus_generator.report import build_report as _br  # explicit

    cfg = _cfg(tmp_path)
    run_dir = cfg.run_dir(tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stages": {
            "stories/s3": {
                "stage": "s3", "corpus": "stories", "kind": "structural",
                "stage_hash": "x", "n_requests": 9000,
                "expected_units": 18000, "achieved_units": 17997,
                "repair_rounds_used": 2,
                "status_counts": {"ok": 17997, "content_filtered": 3},
                "template_hashes": {},
            }
        },
        "batch_ids": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    report_dir = _br(cfg, run_dir, cfg.resolve_data_dir(tmp_path))
    md = (report_dir / "report.md").read_text()
    assert '"content_filtered": 3' in md
    rjson = json.loads((report_dir / "report.json").read_text())
    assert rjson["stages"]["stories/s3"]["status_counts"]["content_filtered"] == 3
