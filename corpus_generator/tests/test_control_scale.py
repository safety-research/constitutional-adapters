"""Control SCALE machinery: the advice length menu (per-scenario target,
scaffold-only, preserved by D, recorded in lineage), the assembly-time
length gate, the register-discovery survey, and the frozen length
reference."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from corpus_generator.advice import advice_stages, sample_advice_length
from corpus_generator.config import RunConfig
from corpus_generator.preflight import build_plan
from corpus_generator.stages import OutputRow, StageCtx
from corpus_generator.templates import TemplateStore, repo_root
from conftest import TINY_CONSTITUTION, tiny_sections

ROOT = repo_root()
OVERLAY_DIR = ROOT / "prompts-control"

LENGTH_LINE = "Aim for roughly {n} words"


# ------------------------------------------------------------ length menu


def test_advice_length_menu_config_validation():
    base = {"run_id": "r"}
    cfg = RunConfig.model_validate(base)
    assert cfg.advice.length_menu is None and cfg.advice.length_weights is None
    ok = RunConfig.model_validate(
        {**base, "advice": {"length_menu": [700, 1400], "length_weights": [0.5, 0.5]}}
    )
    assert ok.advice.length_menu == [700, 1400]
    with pytest.raises(ValueError, match="set together"):
        RunConfig.model_validate({**base, "advice": {"length_menu": [700]}})
    with pytest.raises(ValueError, match="same length"):
        RunConfig.model_validate(
            {**base, "advice": {"length_menu": [700, 1400], "length_weights": [1.0]}}
        )
    with pytest.raises(ValueError, match="positive"):
        RunConfig.model_validate(
            {**base, "advice": {"length_menu": [700, 1400], "length_weights": [1.0, 0.0]}}
        )
    # production configs carry no menu — the mechanism is off there
    for name in ("production.yaml", "production-v2.yaml"):
        assert RunConfig.load(ROOT / "configs" / name).advice.length_menu is None, name


def test_sample_advice_length_is_seeded_weighted_and_off_without_a_menu():
    menu, weights = [700, 1100, 1400, 1800, 2300], [0.2] * 5
    a = sample_advice_length(0, "p00-t00-s00", menu, weights)
    assert a in menu
    assert a == sample_advice_length(0, "p00-t00-s00", menu, weights)  # deterministic
    draws = Counter(
        sample_advice_length(0, f"p{i % 7:02d}-t{i % 3:02d}-s{i:03d}", menu, weights)
        for i in range(5000)
    )
    assert set(draws) == set(menu)
    for n in menu:  # equal weights -> roughly a fifth each
        assert 0.16 < draws[n] / 5000 < 0.24, draws
    skewed = Counter(
        sample_advice_length(0, f"s{i}", [700, 2300], [0.9, 0.1]) for i in range(3000)
    )
    assert skewed[700] > 4 * skewed[2300]
    assert sample_advice_length(0, "p00-t00-s00", None, None) is None  # production
    assert sample_advice_length(1, "p00-t00-s00", menu, weights) in menu


def _ctx(tmp_path, menu, overlay=True) -> StageCtx:
    advice = {"n_principles": 1, "n_types": 1, "n_scenarios_per_call": 2}
    if menu:
        advice.update(length_menu=menu, length_weights=[1.0 / len(menu)] * len(menu))
    cfg = RunConfig.model_validate(
        {"run_id": "len-test", "data_dir": str(tmp_path), "advice": advice,
         **({"prompts_overlay": "prompts-control"} if overlay else {})}
    )
    sections = tiny_sections()
    skeys = ["p00-t00-s00", "p00-t00-s01"]
    lin = lambda k: {"scenario_key": k, "include_system": True,  # noqa: E731
                     "principle_id": "p00", "section_id": sections[0].id}
    pair = {"principle_id": "p00", "principle": "P.", "section_id": sections[0].id,
            "type_id": "p00-t00", "prompt_type": "T."}
    upstream = {
        "a3": [OutputRow(key=k, custom_id="adv-a3-p00-t00", status="ok",
                         data={"scenario": f"scenario {k}"}, lineage=pair) for k in skeys],
        "b3": [OutputRow(key=f"adv-b3-{k}", custom_id=f"adv-b3-{k}", status="ok",
                         data={"system": f"clean system {k}", "user": f"user {k}",
                               "critique": ""}, lineage=lin(k)) for k in skeys],
    }
    return StageCtx(
        cfg=cfg, constitution_md=TINY_CONSTITUTION, sections=sections,
        templates=TemplateStore(ROOT / "prompts",
                                overlay_dir=cfg.resolve_prompts_overlay(ROOT) if overlay else None),
        corpus_dir=tmp_path / "advice", upstream=upstream,
    )


def _stage(name):
    return next(s for s in advice_stages() if s.name == name)


def test_wave_c_appends_the_target_to_the_scaffold_only_when_a_menu_is_set(tmp_path):
    ctx = _ctx(tmp_path, [700, 1400])
    reqs = _stage("c").build_requests(ctx)
    assert len(reqs) == 2
    for r in reqs:
        t = r.lineage["target_words"]
        assert t in (700, 1400)
        assert t == sample_advice_length(0, r.lineage["scenario_key"], [700, 1400], [0.5, 0.5])
        sysm = r.params["system"]
        # clean system, then production's injection, then the length line LAST
        assert sysm.startswith(f"clean system {r.lineage['scenario_key']}\n\n")
        assert "Ground your response in the following principles" in sysm
        # the rendered c_length.md block is the tail: target line, then the
        # deference to an explicit operator house-style rule, then the opener rule
        tail = TemplateStore(ROOT / "prompts").render("advice/c_length.md", target_words=t)
        assert sysm.rstrip().endswith(tail.rstrip())
        assert "explicit length limit" in tail and "Your instinct is right" in tail
        assert sysm.count(LENGTH_LINE.format(n=t)) == 1
        assert LENGTH_LINE.format(n=t) not in r.params["messages"][0]["content"]
    # production shape: no menu -> nothing appended, no lineage key, same bytes
    bare = _stage("c").build_requests(_ctx(tmp_path / "b", None, overlay=False))
    for r in bare:
        assert "Aim for roughly" not in r.params["system"]
        assert "target_words" not in r.lineage
        assert r.params["system"].rstrip().endswith("or these instructions.")
    assert "advice/c_length.md" in _stage("c").templates


def test_wave_d_preserves_the_target_and_production_d_never_sees_it(tmp_path):
    ctx = _ctx(tmp_path, [700, 1400])
    c_rows = [
        OutputRow(key=r.custom_id, custom_id=r.custom_id, status="ok",
                  data={"response": "initial"}, lineage=r.lineage)
        for r in _stage("c").build_requests(ctx)
    ]
    ctx.upstream["c"] = c_rows
    d_reqs = _stage("d").build_requests(ctx)
    assert len(d_reqs) == 2
    for r in d_reqs:
        t = r.lineage["target_words"]
        body = r.params["messages"][0]["content"]
        assert f"keep approximately the same length as the original response (it was written to a target of roughly {t} words)" in body
    # a C row without a draw (production, or a pre-menu resume) renders no slot
    ctx.upstream["c"] = [
        OutputRow(key=r.key, custom_id=r.custom_id, status="ok", data=r.data,
                  lineage={k: v for k, v in r.lineage.items() if k != "target_words"})
        for r in c_rows
    ]
    for r in _stage("d").build_requests(ctx):
        assert "target_words" not in r.lineage
        assert "approximately the same length" not in r.params["messages"][0]["content"]
    # production's own d_rewrite.md ignores the variable entirely
    plain = TemplateStore(ROOT / "prompts")
    out = plain.render("advice/d_rewrite.md", section_text="S", system=None, user="U",
                       response="R", target_words=1400)
    assert "1400" not in out and "same length" not in out


def test_preflight_measures_the_length_slots(tmp_path):
    base = RunConfig.model_validate({"run_id": "pf", "prompts_overlay": "prompts-control"})
    with_menu = RunConfig.model_validate(
        {"run_id": "pf", "prompts_overlay": "prompts-control",
         "advice": {"length_menu": [700, 2300], "length_weights": [0.5, 0.5]}}
    )
    p0 = build_plan(base, ROOT)
    p1 = build_plan(with_menu, ROOT)
    for wave in ("advice/c", "advice/d"):
        assert p1["waves"][wave]["request_bytes_max"] > p0["waves"][wave]["request_bytes_max"], wave
    for wave in ("advice/b3", "advice/e", "stories/s4"):
        assert p1["waves"][wave]["request_bytes_max"] == p0["waves"][wave]["request_bytes_max"], wave


# ------------------------------------------------------------ length gate


def _gate_cfg(tmp_path, gates, thresholds=True):
    return RunConfig.model_validate(
        {
            "run_id": "gate-test",
            "data_dir": str(tmp_path / "data"),
            "finals_subdir": "control",
            "filter": {
                "min_advice_words": 2, "min_story_words": 5,
                "thresholds": {
                    "advice_alignment": 0, "advice_realism": 0,
                    "story_consistency": 0, "story_quality": 0,
                } if thresholds else {},
                "length_gates": gates,
            },
        }
    )


def _advice_records(word_counts, targets=None):
    from fakes import words_for
    from test_schema import _lineage, golden_advice

    out = []
    for i, n in enumerate(word_counts):
        kw = {}
        if targets is not None:
            kw["target_words"] = targets[i]
        out.append(
            golden_advice(
                scenario_id=f"p00-t00-s{i:03d}",
                user=f"user {i} " + words_for(f"u{i}", 30),
                assistant=" ".join(f"w{j}" for j in range(n)),
                lineage=_lineage(custom_id=f"adv-p00-t00-s{i:03d}"),
                **kw,
            )
        )
    return out


def _story_records(word_counts):
    from test_schema import _lineage, golden_story

    return [
        golden_story(
            call_id=f"c00-u00-k{i:03d}",
            text=" ".join(f"s{i}w{j}" for j in range(n)),
            target_words=max(n, 100),
            lineage=_lineage(custom_id=f"sto-c00-u00-k{i:03d}-y0"),
        )
        for i, n in enumerate(word_counts)
    ]


def test_length_math_matches_the_plan_convention():
    from corpus_generator.lengths import quantile, quintile_menu, reference_from_records

    assert quantile([1, 2, 3, 4, 5], 0.5) == 3
    assert quantile([1, 2, 3, 4], 0.5) == 2.5  # linear interpolation (numpy default)
    assert quantile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 0.9) == 91.0
    recs = _advice_records(list(range(100, 1100, 10)))  # 100 records, 100..1090 words
    ref = reference_from_records(recs, "advice")
    assert ref["n"] == 100 and ref["field"] == "assistant"
    assert ref["mean_words"] == 595.0 and ref["p90_words"] == 991.0
    menu = quintile_menu(recs, "advice", k=5, step=50)
    assert menu["menu"] == [200, 400, 600, 800, 1000]  # bin means 195/395/595/795/995 -> 50s
    assert menu["weights"] == [0.2] * 5
    assert sum(m * w for m, w in zip(menu["menu"], menu["weights"])) == pytest.approx(600, abs=10)


def test_length_gate_reports_stats_and_passes_within_tolerance(tmp_path):
    from corpus_generator.filtering import filter_corpus

    gates = {"advice": {"reference_mean_words": 300.0, "reference_p90_words": 500.0,
                        "reference_source": "unit test"}}
    cfg = _gate_cfg(tmp_path, gates)
    # mean 300 exactly, p90 ~ 480 -> both inside tolerance
    counts = [100, 200, 250, 300, 300, 300, 350, 400, 420, 480]
    targets = [100, 200, 250, 300, 300, 300, 350, 400, 400, 500]
    out = tmp_path / "final.jsonl"
    report = filter_corpus("advice", _advice_records(counts, targets), cfg, None, out)
    assert report.output_count == 10 and not report.acceptance_failures
    ls = report.length_stats
    assert ls["field"] == "assistant" and ls["n"] == 10 and ls["mean"] == 310.0
    assert ls["targets"]["n"] == 10 and ls["targets"]["mean_target"] == 310.0
    assert 0.9 < ls["targets"]["mean_achieved_over_target"] < 1.1
    assert ls["targets"]["by_target"]["300"] == {"n": 3, "mean": 300.0}
    lg = report.length_gate
    assert lg["passed"] and lg["mean_ok"] and lg["p90_ok"] and lg["blocking"]
    assert lg["reference"] == {"mean": 300.0, "p90": 500.0, "source": "unit test"}
    assert lg["tolerance"] == {"mean": 0.10, "p90": 0.15}
    assert lg["mean_ratio"] == pytest.approx(310 / 300, abs=1e-3)
    # the verdict is in the written filter report too
    from corpus_generator.config import filter_report_path
    fr = json.loads(filter_report_path(out).read_text())
    assert fr["length_gate"]["passed"] is True and fr["length_stats"]["n"] == 10


def test_length_gate_fails_visibly_blocks_release_and_is_configurable(tmp_path):
    from corpus_generator.config import filter_report_path
    from corpus_generator.filtering import filter_corpus
    from corpus_generator.mirror import ReleaseGateError, verify_release

    # control advice ~30% too long (the measured v1 gap): mean 390 vs 300
    counts = [300, 350, 380, 390, 400, 400, 420, 430, 450, 380]
    gates = {"advice": {"reference_mean_words": 300.0, "reference_p90_words": 500.0}}
    cfg = _gate_cfg(tmp_path, gates)
    data_dir = cfg.resolve_data_dir(ROOT)
    out = cfg.final_path(ROOT, "advice")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = filter_corpus("advice", _advice_records(counts), cfg, None, out)
    assert report.output_count == 10  # never a per-record drop
    lg = report.length_gate
    assert not lg["passed"] and not lg["mean_ok"] and lg["p90_ok"]
    assert "mean 390 words is +30.0%" in lg["reason"]
    assert report.acceptance_failures == [f"length gate: {lg['reason']}"]
    assert "targets" not in report.length_stats  # no menu on these records
    # the mirror's release gate refuses a blocking failure...
    sto_out = cfg.final_path(ROOT, "stories")
    filter_corpus("stories", _story_records([300] * 5), cfg, None, sto_out)
    for p in (out, sto_out):  # token floors are not what this test is about
        fr = json.loads(filter_report_path(p).read_text())
        fr["official_tokens"] = 10**9
        filter_report_path(p).write_text(json.dumps(fr))
    cfg.targets.stories_min_count = 1
    with pytest.raises(ReleaseGateError, match="advice: length gate FAILED"):
        verify_release(cfg, data_dir)
    # ...and lets a non-blocking one through (still reported)
    gates_nb = {"advice": {**gates["advice"], "blocking": False}}
    cfg_nb = _gate_cfg(tmp_path / "nb", gates_nb)
    cfg_nb.targets.stories_min_count = 1
    out_nb = cfg_nb.final_path(ROOT, "advice")
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    rep_nb = filter_corpus("advice", _advice_records(counts), cfg_nb, None, out_nb)
    assert rep_nb.acceptance_failures[0].endswith("(non-blocking)")
    filter_corpus("stories", _story_records([300] * 5), cfg_nb, None, cfg_nb.final_path(ROOT, "stories"))
    for c in ("advice", "stories"):
        p = filter_report_path(cfg_nb.final_path(ROOT, c))
        fr = json.loads(p.read_text()); fr["official_tokens"] = 10**9; p.write_text(json.dumps(fr))
    verify_release(cfg_nb, cfg_nb.resolve_data_dir(ROOT))
    # tolerances are configurable: a 35% mean band passes the same corpus
    gates_wide = {"advice": {**gates["advice"], "mean_tol": 0.35}}
    rep_w = filter_corpus("advice", _advice_records(counts), _gate_cfg(tmp_path / "w", gates_wide),
                          None, tmp_path / "w.jsonl")
    assert rep_w.length_gate["passed"] and not rep_w.acceptance_failures
    # stories gate measures `text`
    gates_s = {"stories": {"reference_mean_words": 2000.0, "reference_p90_words": 2850.0}}
    # mean 2000, p90 2440 (-14%): inside both bands
    rep_s = filter_corpus("stories", _story_records([1400, 1800, 2000, 2200, 2600]),
                          _gate_cfg(tmp_path / "s", gates_s), None, tmp_path / "s.jsonl")
    assert rep_s.length_stats["field"] == "text" and rep_s.length_gate["passed"]
    # calibration mode reports the gate too (no thresholds needed)
    rep_c = filter_corpus("advice", _advice_records(counts), _gate_cfg(tmp_path / "c", gates, thresholds=False),
                          None, tmp_path / "c.jsonl", calibration=True)
    assert rep_c.calibration and rep_c.length_gate and not rep_c.length_gate["passed"]
    # unknown corpus keys are refused
    with pytest.raises(ValueError, match="advice/stories"):
        _gate_cfg(tmp_path / "x", {"poems": gates["advice"]})


def test_report_renders_the_length_gate(tmp_path):
    from corpus_generator.filtering import filter_corpus
    from corpus_generator.report import build_report

    # five records at ~300 words: mean 300, p90 308 -> reference p90 310
    gates = {"advice": {"reference_mean_words": 300.0, "reference_p90_words": 310.0}}
    cfg = RunConfig.model_validate(
        {"run_id": "lg-report", "data_dir": str(tmp_path / "data"),
         "filter": {"min_advice_words": 2,
                    "thresholds": {"advice_alignment": 0, "advice_realism": 0,
                                   "story_consistency": 0, "story_quality": 0},
                    "length_gates": gates}}
    )
    data_dir = cfg.resolve_data_dir(ROOT)
    out = cfg.final_path(ROOT, "advice")
    out.parent.mkdir(parents=True, exist_ok=True)
    filter_corpus("advice", _advice_records([300, 310, 290, 305, 295], [300] * 5), cfg, None, out)
    run_dir = cfg.run_dir(ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir = build_report(cfg, run_dir, data_dir)
    md = (report_dir / "report.md").read_text()
    rj = json.loads((report_dir / "report.json").read_text())
    assert "length gate PASSED" in md and "achieved/target 1.00" in md
    assert rj["corpora"]["advice"]["filter"]["length_gate"]["passed"] is True
    assert rj["corpora"]["advice"]["filter"]["length_stats"]["targets"]["n"] == 5


# ------------------------------------------------------- register survey


def _survey_advice_records(n=40):
    from fakes import words_for
    from test_schema import _lineage, golden_advice

    recs = []
    for i in range(n):
        if i < 10:  # the capped production family, verbatim
            system = f"Deployment context: this assistant helps with {words_for(f'd{i}', 12)}."
        elif i < 16:  # the NEAR-MISS the validation run observed (not caught by the regex)
            system = f"Context for this deployment: the tool sits inside {words_for(f'c{i}', 12)}."
        elif i < 20:
            system = None  # the no-system slot
        else:
            system = f"{words_for(f'sys{i}', 15)}."
        assistant = words_for(f"a{i}", 120)
        if i % 5 == 0:  # 8 records share one boilerplate beat
            assistant += (
                " remember that the last line of defense in this workflow is you, so "
                + words_for(f"tail{i}", 20)
            )
        recs.append(
            golden_advice(
                scenario_id=f"p00-t00-s{i:03d}", system=system,
                user=f"user {i} " + words_for(f"u{i}", 40), assistant=assistant,
                lineage=_lineage(custom_id=f"adv-p00-t00-s{i:03d}"),
            )
        )
    return recs


def test_register_survey_ranks_openers_ngrams_and_configured_families():
    from corpus_generator.config import RegisterFamily
    from corpus_generator.register_survey import (
        detect_corpus, format_survey, label_opener, normalize_opener, survey,
    )

    recs = _survey_advice_records()
    assert detect_corpus(recs) == "advice"
    assert normalize_opener("## **Deployment context:** the panel", 3) == "deployment context the"
    assert normalize_opener("two words", 3) is None
    assert label_opener("Deployment context: the panel sits here") == "deployment context:"
    assert label_opener("Context for this deployment: x") == "context for this deployment:"
    assert label_opener("A long preamble sentence with many words before a colon: x") is None
    assert label_opener("no colon here") is None

    fam = RegisterFamily(
        name="deployment_context_opening", pattern=r"\s*deployment\s+context\s*:",
        field="system", anchored=True, max_frac=0.01,
    )
    res = survey(recs, families=[fam], ngram_sizes=(4, 8), top=10)
    assert res["corpus"] == "advice" and res["n_records"] == 40
    # configured family: live match rate, and it is over its cap
    (f,) = res["families"]
    assert (f["matched"], f["frac_of_records"], f["over_cap"]) == (10, 0.25, True)
    # openers: the family AND its near-miss sibling, ranked by record fraction
    sys_first3 = next(t for t in res["openers"] if t["field"] == "system" and t["kind"] == "first_3_words")
    assert sys_first3["n_field"] == 36  # four no-system records carry no field
    top2 = [(r["opener"], r["records"]) for r in sys_first3["rows"][:2]]
    assert top2 == [("deployment context this", 10), ("context for this", 6)]
    assert sys_first3["rows"][0]["frac_of_field"] == round(10 / 36, 4)
    assert sys_first3["rows"][0]["frac_of_records"] == 0.25
    labels = next(t for t in res["openers"] if t["field"] == "system" and t["kind"] == "label")
    assert [(r["opener"], r["records"]) for r in labels["rows"][:2]] == [
        ("deployment context:", 10), ("context for this deployment:", 6),
    ]
    assert {t["field"] for t in res["openers"]} == {"system", "user", "assistant"}
    # n-grams: document frequency, shifted copies merged into the maximal phrase
    g4 = next(t for t in res["ngrams"] if t["field"] == "assistant" and t["n"] == 4)
    top = g4["rows"][0]
    assert top["records"] == 8 and top["frac_of_field"] == 0.2
    assert "last line of defense in this workflow is you" in top["ngram"]
    assert sum(1 for r in g4["rows"] if "line of defense" in r["ngram"]) == 1  # merged, not 6 shifts
    g8 = next(t for t in res["ngrams"] if t["field"] == "assistant" and t["n"] == 8)
    assert g8["rows"][0]["records"] == 8 and "line of defense" in g8["rows"][0]["ngram"]
    assert {(t["field"], t["n"]) for t in res["ngrams"]} == {
        ("assistant", 4), ("assistant", 8), ("user", 4), ("user", 8),
    }
    text = format_survey(res)
    assert "configured register families" in text and "OVER CAP" in text
    assert "deployment context:" in text and "27.78%" in text  # 10/36 of the field
    assert "last line of defense in this workflow is you" in text


def test_register_survey_stories_and_sampling():
    from fakes import words_for
    from test_schema import _lineage, golden_story
    from corpus_generator.register_survey import detect_corpus, sample_records, survey

    recs = []
    for i in range(30):
        text = words_for(f"story{i}", 200)
        if i % 6 == 0:  # 5 of 30 share a beat
            text = "She checked the gauge twice before trusting the number. " + text
        recs.append(
            golden_story(
                call_id=f"c00-u00-k{i:03d}", text=text, target_words=200,
                lineage=_lineage(custom_id=f"sto-c00-u00-k{i:03d}-y0"),
            )
        )
    assert detect_corpus(recs) == "stories"
    res = survey(recs, ngram_sizes=(6,), top=5)
    assert [t["field"] for t in res["openers"]] == ["text"]
    assert res["openers"][0]["rows"][0] == {
        "opener": "she checked the", "records": 5, "frac_of_field": round(5 / 30, 4),
        "frac_of_records": round(5 / 30, 4), "example_id": "sto-c00-u00-k000-y0",
    }
    g6 = res["ngrams"][0]
    assert g6["rows"][0]["records"] == 5
    assert "checked the gauge twice before trusting the number" in g6["rows"][0]["ngram"]
    sub = sample_records(recs, 10, seed=0)
    assert len(sub) == 10 and sub == sample_records(recs, 10, seed=0)
    assert sample_records(recs, 0) is recs and sample_records(recs, 99) is recs
    with pytest.raises(ValueError, match="empty"):
        detect_corpus([])


def test_register_survey_cli_defaults_to_the_calibration_artifact(tmp_path):
    from click.testing import CliRunner

    from corpus_generator.cli import main

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "run_id: survey-test\n"
        f"data_dir: {tmp_path / 'data'}\n"
        "finals_subdir: control\n"
        "filter:\n  register_families:\n"
        "    - name: deployment_context_opening\n      field: system\n      anchored: true\n"
        "      max_frac: 0.01\n      pattern: '\\s*deployment\\s+context\\s*:'\n"
    )
    cfg = RunConfig.load(cfg_path)
    runner = CliRunner()
    # nothing to survey yet -> loud failure, no silent empty table
    r = runner.invoke(main, ["register-survey", "--config", str(cfg_path), "--corpus", "advice"])
    assert r.exit_code != 0 and "no artifact to survey" in r.output
    cal = cfg.calibration_path(ROOT, "advice")
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text("".join(json.dumps(x) + "\n" for x in _survey_advice_records()))
    out_json = tmp_path / "survey.json"
    r = runner.invoke(
        main, ["register-survey", "--config", str(cfg_path), "--corpus", "advice",
               "--n", "4", "--top", "5", "--json", str(out_json)],
    )
    assert r.exit_code == 0, r.output
    assert f"# source: {cal}" in r.output and "OVER CAP" in r.output
    assert "deployment context:" in r.output and "line of defense" in r.output
    js = json.loads(out_json.read_text())
    assert js["corpus"] == "advice" and js["params"]["ngram_sizes"] == [4]
    assert js["families"][0]["matched"] == 10
    # an explicit --input works without any run artifacts, corpus auto-detected
    r2 = runner.invoke(main, ["register-survey", "--config", str(cfg_path), "--input", str(cal)])
    assert r2.exit_code == 0 and "register survey — advice, 40 records" in r2.output


# ------------------------------------------------------- length reference

SCALE_CFG = ROOT / "configs" / "control-scale.yaml"


def test_frozen_length_reference_matches_the_v2_finals():
    """The gate references (and the menu weights) in the production control
    config are the build script's output over the v2 finals — asserted
    whenever those (untracked) files are present, so a re-frozen number can't
    drift from the artifact. The menu itself is the script's quintile menu
    CALIBRATED on the pilot's realized/target ratio per bin, so only the
    quintile menu the script prints is pinned here."""
    import importlib.util

    adv = ROOT / "data/advice/final-production-20260831-v2.jsonl"
    sto = ROOT / "data/stories/final-production-20260831-v2.jsonl"
    if not (adv.exists() and sto.exists()):
        pytest.skip("v2 production finals not present on this machine")
    spec = importlib.util.spec_from_file_location(
        "build_length_reference", ROOT / "scripts" / "build_length_reference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = mod.build(adv, sto)
    cfg = RunConfig.load(SCALE_CFG)
    assert cfg.advice.length_weights == pytest.approx(ref["menu"]["weights"])
    for corpus in ("advice", "stories"):
        g = cfg.filter.length_gates[corpus]
        assert g.reference_mean_words == ref[corpus]["mean_words"], corpus
        assert g.reference_p90_words == ref[corpus]["p90_words"], corpus
        assert f"n={ref[corpus]['n']}" in g.reference_source
    assert "length_menu: [800, 1150, 1400, 1700, 2400]" in mod.format_yaml(ref)


def test_report_renders_the_downsample_mix(tmp_path):
    """The release report carries the seeded stratified trim (filter rule 7b):
    counts before/after/trimmed, the strata fields, and how far each reported
    field's mix moved — so the mix-preservation claim is checkable from
    report.md alone, not only from the filter report JSON."""
    from corpus_generator.filtering import filter_corpus
    from corpus_generator.report import build_report

    cfg = RunConfig.model_validate(
        {"run_id": "ds-report", "data_dir": str(tmp_path / "data"),
         "filter": {"min_advice_words": 2,
                    "thresholds": {"advice_alignment": 0, "advice_realism": 0,
                                   "story_consistency": 0, "story_quality": 0},
                    "downsample": {"advice": 5, "seed": 0}}}
    )
    data_dir = cfg.resolve_data_dir(ROOT)
    out = cfg.final_path(ROOT, "advice")
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = [300, 300, 300, 300, 600, 600, 600, 600]
    filter_corpus("advice", _advice_records([300] * 8, targets), cfg, None, out)
    run_dir = cfg.run_dir(ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir = build_report(cfg, run_dir, data_dir)
    md = (report_dir / "report.md").read_text()
    rj = json.loads((report_dir / "report.json").read_text())
    assert "downsample (seeded stratified, filter rule 7b): 8 -> 5 records (trimmed 3, target 5)" in md
    assert "target_words 2->2 values, max shift" in md
    ds = rj["corpora"]["advice"]["filter"]["downsample"]
    assert (ds["before"], ds["after"], ds["trimmed"]) == (8, 5, 3)
    assert ds["fields"] == ["principle_id", "prompt_type_id"]
    assert set(ds["mix"]) >= {"principle_id", "prompt_type_id", "target_words"}


def test_report_finds_finals_under_the_control_layout(tmp_path):
    """`corpusgen report` must read a control run's flat data/control/ finals (the
    finals_subdir knob), not data/<corpus>/ — the pilot review runs it."""
    from corpus_generator.filtering import filter_corpus
    from corpus_generator.report import build_report

    cfg = RunConfig.model_validate(
        {"run_id": "ctl-report", "data_dir": str(tmp_path / "data"), "finals_subdir": "control",
         "filter": {"min_advice_words": 2,
                    "thresholds": {"advice_alignment": 0, "advice_realism": 0,
                                   "story_consistency": 0, "story_quality": 0}}}
    )
    out = cfg.final_path(ROOT, "advice")
    out.parent.mkdir(parents=True, exist_ok=True)
    assert out.parent.name == "control"
    filter_corpus("advice", _advice_records([300, 320, 280]), cfg, None, out)
    run_dir = cfg.run_dir(ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    rj = json.loads((build_report(cfg, run_dir, cfg.resolve_data_dir(ROOT)) / "report.json").read_text())
    assert rj["corpora"]["advice"]["n_final"] == 3
    assert rj["corpora"]["advice"]["filter"]["length_stats"]["n"] == 3
