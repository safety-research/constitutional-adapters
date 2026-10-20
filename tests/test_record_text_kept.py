"""What a producer writes into a record does not change when only its source's wording or defaults do.

A record's text is part of its evidence: a reader pins the artifact by hash, or the registry keeps the text in its state rows,
so rewording one of these lines changes what re-running unchanged inputs writes. Each line is pinned by the sha256 of its exact
text (the text is not repeated here); a line that must change needs a new schema version and a reconciliation of the records
already written, not an edit in place. A default written with a site placeholder is recorded as the path it resolved to.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: (file, the record field the line feeds, sha256 of the line as recorded)
RECORD_LINES = [
    ("scripts/sdfbase/record_resolved_sampler.py", "resolved_sampler.json: the kinds block's source, line 1",
     "84a8753b1068d00bf1e61d98bbe96e1f051ab798ca2f9a0635f8f92f7654a6b4"),
    ("scripts/sdfbase/record_resolved_sampler.py", "resolved_sampler.json: the kinds block's source, line 2",
     "c0fa8415914a7818f6554cbebc3027c8d25ecf0093ae14b976d011fa661f0c94"),
    ("scripts/sdfbase/gsm8k/g8k_noise_floor.py", "noise_floor.json: ruling",
     "e8d1d5f9066244000c408a9ce12634493ba55e657a2596479900818ad1bc8a0e"),
    ("scripts/sdfbase/derived_verdict.py", "the derived verdict: constants.plan",
     "1ff77e58fd53a0846f8d0c46244a695a74e5fc935116726deb1b2b1d529c3f14"),
    ("scripts/sdfbase/full22/sampled/static_degeneracy_union.py", "the static union record: defended_collapse_bracket.rule",
     "b025d877ac203b1da72b3e2cdbf29427e7d97f1135e7efc92e59ec61528bc46e"),
    ("scripts/r2_export/deps/modal_stage_analysis_r2/r2_paths.py", "the probe table's provenance: scores",
     "e474073c9a75fa6d4c8f4c72c5fe3968683779e5d694f75075ed178877a615cd"),
    ("src/sisteer/registry/adapters/seedrep_controls.py", "the registry's row_refusals state: the seed-base refusal",
     "6ce602c0dbe079fcbfd5b66e75040960b02e1528a4d45450aa816998cbfbe84c"),
    ("src/sisteer/registry/adapters/seedrep_statics.py", "the registry's row_refusals state: the seed-base refusal (counts)",
     "ca58939d9764cb6d15ca5eab62fa6b8075ef18d3fd35f964194ad7a572323c68"),
    ("src/sisteer/registry/adapters/seedrep_statics.py", "the registry's row_refusals state: the seed-base refusal (judged rows)",
     "16ad61feb11ef12054ce776f392c30d58b322448830a0cecdaf6e73f5a7a822b"),
    ("src/sisteer/registry/adapters/_bcbw_common.py", "the registry's refusal state: the appendix selection's stage pin",
     "85f11485511671697954c472965bda4d6ca762d18fda5b6588ecab0fc3e7af2d"),
]


@pytest.mark.parametrize("rel, feeds, sha", RECORD_LINES, ids=[f"{r}: {f}" for r, f, _ in RECORD_LINES])
def test_a_line_written_into_a_record_keeps_its_recorded_bytes(rel, feeds, sha):
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
    assert any(hashlib.sha256(line.encode("utf-8")).hexdigest() == sha for line in lines), \
        f"{rel}: the line that feeds {feeds} no longer has its recorded bytes"


def test_the_aggregate_records_the_leakage_report_path_it_read_not_the_placeholder(tmp_path, monkeypatch):
    """clean_subset_aggregate.py's --leakage-report default is written under the site's working-data root ($WORK); the record
    names the path the report was read from. Built by the end-to-end fixture of tests/test_alphasteer_f22_grade.py, with only
    the argument respelled as the default spells it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_alphasteer_f22_grade as G
    real_build = G.csa.build

    def build(args):
        assert args.leakage_report == str(tmp_path / "leakage_report.json")
        args.leakage_report = "$WORK/leakage_report.json"
        return real_build(args)
    monkeypatch.setattr(G.csa, "build", build)
    monkeypatch.setenv("SISTEER_WORK_DIR", str(tmp_path))
    doc, _ = G._end_to_end(tmp_path, monkeypatch)
    assert doc["leakage_report"]["path"] == str(tmp_path / "leakage_report.json")
