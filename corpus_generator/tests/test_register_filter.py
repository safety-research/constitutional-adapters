"""Register caps (filter rule 7) + v2 assembly: config-exposed regex
families capped strictly under max_frac of the resulting corpus, deterministic
diversity-preserving drops, and the assembly's merge/drop/uniqueness rules."""

from __future__ import annotations

import json

import pytest

from corpus_generator.assemble import assemble_v2, v2_final_path
from corpus_generator.config import RegisterFamily, RunConfig, filter_report_path
from corpus_generator.filtering import FilterReport, _register_cap, apply_register_caps, filter_corpus
from corpus_generator.io import read_json
from fakes import words_for
from test_schema import _lineage, golden_advice, golden_story
from test_tokens import _char_tokenizer, _stage, _word_tokenizer
from corpus_generator.tokens import TokenCounter

ALL_THRESHOLDS = {
    "advice_alignment": 0, "advice_realism": 0,
    "story_consistency": 0, "story_quality": 0,
}


def _story(i: int, **kw) -> dict:
    d = golden_story(
        text=f"story {i} " + words_for(f"s{i}", 60),
        call_id=f"c{i % 5:02d}-u00-k{i:02d}",
        category_id=f"c{i % 5:02d}",
        lineage=_lineage(custom_id=f"sto-c{i % 5:02d}-u00-k{i:02d}-y0"),
    )
    d.update(kw)
    return d


# ------------------------------------------------------------- cap arithmetic


def test_register_cap_strictly_under_max_frac():
    # k must satisfy k < max_frac * (non_members + k)
    assert _register_cap(members=30, non_members=270, max_frac=0.01) == 2
    assert 2 < 0.01 * (270 + 2)
    assert not 3 < 0.01 * (270 + 3)
    assert _register_cap(members=1, non_members=10, max_frac=0.5) == 1
    assert _register_cap(members=5, non_members=0, max_frac=0.01) == 0
    assert _register_cap(members=0, non_members=100, max_frac=0.01) == 0


def _fam(**kw) -> RegisterFamily:
    d = dict(name="beat", pattern=r"the only safeguard", field="text",
             max_frac=0.01, diversity_fields=["category_id"])
    d.update(kw)
    return RegisterFamily.model_validate(d)


def test_cap_drops_excess_deterministically_with_diversity():
    records = [_story(i) for i in range(300)]
    for i in range(30):  # members spread over 3 categories: c00, c01, c02
        records[i]["text"] += " it was not the only safeguard here"
    report = FilterReport(corpus="stories")
    out = apply_register_caps(records, [_fam()], seed=0, report=report)
    stats = report.register_family_stats[0]
    assert stats["matched"] == 30 and stats["kept"] == 2 and stats["dropped"] == 28
    assert len(out) == 272
    kept = [r for r in out if "only safeguard" in r["text"]]
    assert len(kept) == 2
    # kept members come from DISTINCT diversity groups (round-robin)
    assert len({r["category_id"] for r in kept}) == 2
    # final fraction strictly under the cap
    assert len(kept) / len(out) < 0.01
    # deterministic: same seed -> same keeps
    report2 = FilterReport(corpus="stories")
    out2 = apply_register_caps(
        [_story(i, text=r["text"]) if False else json.loads(json.dumps(r)) for i, r in enumerate(records)],
        [_fam()], seed=0, report=report2,
    )
    assert {r["lineage"]["custom_id"] for r in out2} == {r["lineage"]["custom_id"] for r in out}


def test_multiple_families_converge_to_joint_fixed_point():
    """One family's drops shrink every other family's denominator — after
    convergence EVERY family must sit strictly under its cap of the FINAL
    corpus (the single-pass bug left family 1 at 1.17%)."""
    records = [_story(i) for i in range(1000)]
    for i in range(100):
        records[i]["text"] += " it was not the only safeguard"
    for i in range(100, 400):
        records[i]["text"] += " a quiet gentle close"
    fams = [
        _fam(name="beat"),
        _fam(name="close", pattern=r"quiet gentle close"),
    ]
    report = FilterReport(corpus="stories")
    out = apply_register_caps(records, fams, seed=0, report=report)
    n = len(out)
    for fs in report.register_family_stats:
        assert fs["kept"] / n < 0.01, fs
        assert fs["final_count"] == n
    beat = [r for r in out if "only safeguard" in r["text"]]
    close = [r for r in out if "quiet gentle close" in r["text"]]
    assert len(beat) / n < 0.01 and len(close) / n < 0.01


def test_under_cap_family_drops_nothing():
    records = [_story(i) for i in range(300)]
    records[0]["text"] += " the only safeguard"
    report = FilterReport(corpus="stories")
    out = apply_register_caps(records, [_fam()], seed=0, report=report)
    assert len(out) == 300
    assert report.register_family_stats[0]["dropped"] == 0


def test_anchored_family_matches_openings_only():
    a = golden_advice(system="Deployment context: a records tool.")
    b = golden_advice(
        system="A records tool. Deployment context: mentioned later.",
        lineage=_lineage(custom_id="adv-p00-t00-s01"),
    )
    fam = _fam(name="dc", pattern=r"\s*deployment\s+context\s*:", field="system",
               anchored=True, max_frac=0.9, diversity_fields=["principle_id"])
    report = FilterReport(corpus="advice")
    out = apply_register_caps([a, b], [fam], seed=0, report=report)
    assert report.register_family_stats[0]["matched"] == 1  # only the true opening
    assert len(out) == 2  # generous cap keeps the single member


def test_filter_corpus_runs_register_rule_and_reports(tmp_path):
    cfg = RunConfig.model_validate(
        {
            "run_id": "regtest",
            "data_dir": str(tmp_path / "data"),
            "filter": {
                "min_story_words": 5,
                "thresholds": ALL_THRESHOLDS,
                "register_families": [
                    {"name": "beat", "pattern": "the only safeguard",
                     "field": "text", "max_frac": 0.01,
                     "diversity_fields": ["category_id"]},
                ],
            },
        }
    )
    records = [_story(i) for i in range(200)]
    for i in range(20):
        records[i]["text"] += " not the only safeguard"
    out_path = tmp_path / "final.jsonl"
    report = filter_corpus("stories", records, cfg, None, out_path)
    assert report.rule_counts["register:beat"] == 19
    assert report.output_count == 181
    persisted = read_json(filter_report_path(out_path))
    assert persisted["register_family_stats"][0]["matched"] == 20
    assert persisted["rule_counts"]["register:beat"] == 19


# ---------------------------------------------------------------- assembly


def _counter(tmp_path) -> TokenCounter:
    tok_dir = tmp_path / "toks"
    _stage(tok_dir, {"llama3": _word_tokenizer(), "gemma": _char_tokenizer(),
                     "qwen": _word_tokenizer()})
    return TokenCounter.load(tok_dir)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _adv(pid: str, i: int, run_id: str) -> dict:
    return golden_advice(
        principle_id=pid,
        scenario_id=f"{pid}-t00-s{i:02d}",
        user=f"user {pid} {i} " + words_for(f"u{pid}{i}", 40),
        assistant=f"assistant {pid} {i} " + words_for(f"a{pid}{i}", 60),
        lineage=_lineage(run_id=run_id, custom_id=f"adv-{pid}-t00-s{i:02d}"),
    )


def test_assemble_v2_merges_drops_and_reports(tmp_path):
    cfg = RunConfig.model_validate(
        {"run_id": "prod", "data_dir": str(tmp_path / "data"),
         "filter": {"min_advice_words": 2, "thresholds": ALL_THRESHOLDS}}
    )
    root = tmp_path
    v1 = [_adv("p01", i, "prod") for i in range(4)] + [_adv("p35", i, "prod") for i in range(3)]
    regen = [_adv("p35", i, "regen") for i in range(5)]
    _write_jsonl(cfg.final_path(root, "advice"), v1)
    regen_path = tmp_path / "data" / "advice" / "final-regen.jsonl"
    _write_jsonl(regen_path, regen)

    report = assemble_v2(
        cfg, "advice", root=root, add_sources=[regen_path],
        drop_principles={"p35"}, drop_run="prod",
        counter=_counter(tmp_path), on_event=lambda m: None,
    )
    assert report.rule_counts["reframed_principle"] == 3
    assert report.input_count == 12 and report.output_count == 9
    out = v2_final_path(cfg, root, "advice")
    assert out.name == "final-prod-v2.jsonl"
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert sum(1 for r in recs if r["principle_id"] == "p35") == 5
    assert all(r["lineage"]["run_id"] == "regen" for r in recs if r["principle_id"] == "p35")
    persisted = read_json(filter_report_path(out))
    assert persisted["rule_counts"]["reframed_principle"] == 3
    assert persisted["assembly"]["drop_principles"] == ["p35"]
    assert persisted["drops"][0]["rule"] == "reframed_principle"


def test_assemble_v2_residual_id_drops(tmp_path):
    cfg = RunConfig.model_validate(
        {"run_id": "prod", "data_dir": str(tmp_path / "data"),
         "filter": {"min_advice_words": 2, "thresholds": ALL_THRESHOLDS}}
    )
    root = tmp_path
    _write_jsonl(cfg.final_path(root, "advice"), [_adv("p01", i, "prod") for i in range(3)])
    regen_path = tmp_path / "data" / "advice" / "final-regen.jsonl"
    _write_jsonl(regen_path, [_adv("p35", i, "regen") for i in range(3)])
    report = assemble_v2(
        cfg, "advice", root=root, add_sources=[regen_path],
        drop_ids={"adv-p35-t00-s01"},
        counter=None, on_event=lambda m: None,
    )
    assert report.rule_counts["ai_subject_residual"] == 1
    assert report.output_count == 5
    persisted = read_json(filter_report_path(v2_final_path(cfg, root, "advice")))
    assert persisted["assembly"]["drop_ids"] == ["adv-p35-t00-s01"]


def test_assemble_v2_refuses_custom_id_collision(tmp_path):
    # regenerated p35 ids collide with v1's unless the v1 records are dropped
    cfg = RunConfig.model_validate(
        {"run_id": "prod", "data_dir": str(tmp_path / "data"),
         "filter": {"min_advice_words": 2, "thresholds": ALL_THRESHOLDS}}
    )
    root = tmp_path
    _write_jsonl(cfg.final_path(root, "advice"), [_adv("p35", 0, "prod")])
    regen_path = tmp_path / "data" / "advice" / "final-regen.jsonl"
    _write_jsonl(regen_path, [_adv("p35", 0, "regen")])
    with pytest.raises(RuntimeError, match="collision"):
        assemble_v2(
            cfg, "advice", root=root, add_sources=[regen_path],
            counter=None, on_event=lambda m: None,
        )
