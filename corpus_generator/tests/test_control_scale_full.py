"""Control SCALE — Phase 2: the full-run config and its frozen decisions,
the frozen A1 principles and their seeding into a run namespace, the seeded
stratified downsample, the stray-tag gate + S4 wrapper strip, the
release-time placebo-import scan, and the prompt-side register bans the
pilot survey ordered."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter

import pytest

from corpus_generator.advice import advice_stages
from corpus_generator.batch import request_set_hash
from corpus_generator.config import RunConfig, filter_report_path
from corpus_generator.pipeline import build_ctx, dag_expanded_totals
from corpus_generator.preflight import build_plan
from corpus_generator.stages import OutputRow, run_stage
from corpus_generator.templates import TemplateStore, repo_root
from fakes import words_for
from test_schema import _lineage, golden_advice, golden_story

ROOT = repo_root()
SCALE_CFG = ROOT / "configs" / "control-scale.yaml"
PRINCIPLES = ROOT / "configs" / "control-scale-principles.json"
PLACEBO = ROOT / "constitution-control-v2" / "placebo.md"


def _ngrams(text: str, n: int) -> set[str]:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


# ------------------------------------------------------------------ config


def test_full_config_carries_the_frozen_decisions():
    cfg = RunConfig.load(SCALE_CFG)
    v2 = RunConfig.load(ROOT / "configs" / "production-v2.yaml")
    assert cfg.run_id == "control-scale-20260903"
    assert (cfg.constitution_dir, cfg.prompts_overlay, cfg.finals_subdir, cfg.seed) == (
        "constitution-control-v2", "prompts-control", "control", 0
    )
    assert len(cfg.domain_menu) == 43
    assert "everyday life" in cfg.advice.behavior_seed
    # the length menu is the v2-quintile menu [800, 1150, 1400, 1700, 2400]
    # CALIBRATED by the pilot's realized/target ratio per bin (rounded to 50);
    # equal weights; the gate references are the v2 finals'
    assert cfg.advice.length_weights == [0.2] * 5
    assert cfg.advice.length_menu == [750, 1100, 1400, 1800, 2550]
    ga, gs = cfg.filter.length_gates["advice"], cfg.filter.length_gates["stories"]
    assert (ga.reference_mean_words, ga.reference_p90_words) == (1499.8, 2259.2)
    assert (gs.reference_mean_words, gs.reference_p90_words) == (1989.8, 2852.0)
    for g in (ga, gs):
        assert (g.mean_tol, g.p90_tol, g.blocking) == (0.10, 0.15, True)
        assert "final-production-20260831-v2.jsonl" in g.reference_source
    # widths: 25 x 10 x 12 = 3,000 scenarios; stories at production's fanout
    assert (cfg.advice.n_principles, cfg.advice.n_types, cfg.advice.n_scenarios_per_call) == (25, 10, 12)
    assert (cfg.stories.n_categories, cfg.stories.n_subcategories, cfg.stories.subcategories_use,
            cfg.stories.calls_per_subcategory, cfg.stories.stories_per_call) == (50, 10, None, 18, 2)
    assert dag_expanded_totals(cfg) == {
        "advice_requests": 18276, "advice_scenarios": 3000,
        "stories_requests": 45051, "stories_drafts": 18000, "total": 63327,
    }
    # the two rewrite waves at 32k (the pilot's only truncations were length-driven)
    assert cfg.advice.max_tokens["d"] == 32000 and cfg.stories.max_tokens["s4"] == 32000
    assert cfg.advice.max_tokens["c"] == 16000 and cfg.stories.max_tokens["s3"] == 32000
    # floors frozen on the pilot's calibration: 8/8/8/8
    assert cfg.filter.thresholds.model_dump() == {
        "advice_alignment": 8, "advice_realism": 8, "story_consistency": 8, "story_quality": 8,
    }
    # the three v2 families verbatim, plus the five the pilot survey found
    names = [f.name for f in cfg.filter.register_families]
    assert cfg.filter.register_families[:3] == v2.filter.register_families
    assert names[3:] == ["this_assistant_opening", "verdict_label_opening", "instinct_opening",
                         "tablet_propped_beat", "envelope_back_beat"]
    caps = {f.name: f.max_frac for f in cfg.filter.register_families}
    assert caps["this_assistant_opening"] == 0.07 and caps["verdict_label_opening"] == 0.08
    assert all(caps[n] == 0.01 for n in ("instinct_opening", "tablet_propped_beat", "envelope_back_beat"))
    fam = {f.name: f for f in cfg.filter.register_families}
    for n in ("this_assistant_opening",):
        assert fam[n].field == "system" and fam[n].anchored
    for n in ("verdict_label_opening", "instinct_opening"):
        assert fam[n].field == "assistant" and fam[n].anchored
    for n in ("tablet_propped_beat", "envelope_back_beat"):
        assert fam[n].field == "text" and not fam[n].anchored
    # the regexes match what the survey found — and not their neighbours
    rx = {n: re.compile(f.pattern, re.I) for n, f in fam.items()}
    assert rx["this_assistant_opening"].match("This assistant staffs the help panel")
    assert rx["this_assistant_opening"].match("  This is the project-help panel")
    assert not rx["this_assistant_opening"].match("Deployment: the help panel. This assistant ...")
    assert rx["verdict_label_opening"].match("**Short answer:** no.")
    assert rx["verdict_label_opening"].match("Straight answer first.")
    assert rx["verdict_label_opening"].match("The short answer here is no.")  # same register, article or not
    assert not rx["verdict_label_opening"].match("Short of a full rebuild, the answer is no.")
    assert rx["instinct_opening"].match("Your instinct's right: ...")
    assert rx["instinct_opening"].match("Good instinct stalling on the cut.")
    assert rx["tablet_propped_beat"].search("the tablet propped against the fruit bowl")
    assert rx["tablet_propped_beat"].search("her phone was propped up against a mug")
    assert rx["envelope_back_beat"].search("she wrote it on the back of an envelope")
    assert not rx["envelope_back_beat"].search("on the back of her hand")
    # downsample + scan + targets
    assert cfg.filter.downsample.model_dump(include={"advice", "stories", "seed"}) == {
        "advice": 2000, "stories": 16300, "seed": 0,
    }
    assert cfg.filter.import_scan.ngram_n == 7
    assert cfg.targets.model_dump() == {
        "advice_min_tokens": 4_750_000, "stories_min_count": 15_000,
        "stories_min_tokens": 38_000_000, "stories_min_ordinary_share": 0.5,
        "total_min_tokens": 43_000_000,
    }
    text = "\n".join(cfg.divergences)
    for must in ("FULL scale", "3,000 scenarios", "18,000", "63,327", "FROZEN from the pilot",
                 "CALIBRATED", "unconditional", "explicit length limit",
                 "control-scale-principles.json", "REGISTER BANS", "This assistant",
                 "Deployment:", "DEFERS to an explicit operator limit", "32000", "stray_tag",
                 "DOWNSAMPLE", "import-scan", "8/8/8/8", ">=4.75M"):
        assert must in text, must
    assert not any("PILOT scale" in d or "Score floors NULL" in d for d in cfg.divergences)


def test_preflight_builds_the_full_plan():
    plan = build_plan(RunConfig.load(SCALE_CFG), ROOT)
    assert plan["total_requests"] == 63327
    assert plan["waves"]["advice/b1"]["n_requests"] == 3000
    assert plan["waves"]["stories/s4"]["n_requests"] == 18000
    assert 40 <= plan["total_chunks"] <= 70


# ------------------------------------------------------ frozen principles


def test_frozen_principles_are_the_reviewed_pilot_set_without_placebo_wording():
    data = json.loads(PRINCIPLES.read_text())
    src, principles = data["source"], data["principles"]
    assert src["run_id"] == "control-pilot-20260902" and src["stage"] == "advice/a1"
    assert src["batch_id"].startswith("msgbatch_") and len(src["edited"]) == 10
    assert sorted(principles) == [f"p{i:02d}" for i in range(25)]
    section_ids = {s.id for s in _placebo_sections()}
    placebo = PLACEBO.read_text()
    p8 = _ngrams(placebo, 8)
    texts = set()
    for pid, d in principles.items():
        assert d["section_id"] in section_ids, pid
        for field in ("principle", "tension"):
            assert d[field].strip(), (pid, field)
            # the A1 brief says "do not copy sentences from the document" — the
            # frozen set is held to it at the 8-gram level
            assert not (_ngrams(d[field], 8) & p8), (pid, field)
        assert d["principle"].casefold() not in texts
        texts.add(d["principle"].casefold())
    # spread: >= 12 distinct placebo sections, none carrying more than 3 principles
    by_section = Counter(d["section_id"] for d in principles.values())
    assert len(by_section) >= 12 and max(by_section.values()) <= 3
    # every field the review did NOT edit is the pilot's A1 output verbatim
    pilot_a1 = ROOT / "data/runs/control-pilot-20260902/advice/a1/outputs.jsonl"
    if pilot_a1.exists():
        rows = {json.loads(l)["key"]: json.loads(l)["data"] for l in pilot_a1.read_text().splitlines()}
        for pid, d in principles.items():
            for field in ("principle", "tension", "section_id"):
                if f"{pid}.{field}" in src["edited"]:
                    assert d[field] != rows[pid][field], (pid, field)
                else:
                    assert d[field] == rows[pid][field], (pid, field)


def _placebo_sections():
    from corpus_generator.constitution import load as load_constitution

    _md, sections, _snap = load_constitution(ROOT / "constitution-control-v2")
    return sections


def _seed_cfg(tmp_path, n_principles=2, overlay="prompts-control", run_id="seed-test"):
    return RunConfig.model_validate(
        {
            "run_id": run_id,
            "data_dir": str(tmp_path / "data"),
            "constitution_dir": "constitution-control-v2",
            "prompts_overlay": overlay,
            "domain_menu": ["cooking", "woodworking", "gardening"],
            "advice": {"n_principles": n_principles, "n_types": 2, "n_scenarios_per_call": 2},
        }
    )


class _NeverEngine:
    def run_wave(self, *a, **k):
        raise AssertionError("engine called — the seeded A1 stage was not skipped")


def test_seed_principles_materializes_a_skippable_hash_checked_a1(tmp_path):
    from corpus_generator.principles import PrinciplesError, load_frozen_principles, seed_a1_stage

    sections = _placebo_sections()
    frozen = {
        "source": {"run_id": "pilot-x", "batch_id": "msgbatch_test"},
        "principles": {
            "p00": {"principle": "Check before you cut.", "tension": "Costs a minute.",
                    "section_id": sections[3].id},
            "p01": {"principle": "Zero the instrument first.", "tension": "Costs a reading.",
                    "section_id": sections[5].id},
        },
    }
    fpath = tmp_path / "frozen.json"
    fpath.write_text(json.dumps(frozen))
    principles, source = load_frozen_principles(fpath)
    assert source["run_id"] == "pilot-x" and source["sha256"] and source["path"] == str(fpath)

    cfg = _seed_cfg(tmp_path)
    stage_dir = seed_a1_stage(cfg, principles, source, root=ROOT, on_event=lambda m: None)
    stamp = json.loads((stage_dir / "STAGE.json").read_text())
    ctx, _ = build_ctx(cfg, "advice", ROOT)
    spec = advice_stages(None)[0]
    expected_hash = request_set_hash(spec.build_requests(ctx), ctx.templates.hashes(spec.templates))
    assert stamp["stage_hash"] == expected_hash
    assert (stamp["kind"], stamp["n_requests"], stamp["expected_units"], stamp["achieved_units"]) == ("root", 0, 2, 2)
    assert stamp["status_counts"] == {"ok": 2} and stamp["seeded_from"]["run_id"] == "pilot-x"
    # the runner finds a completed stage and never submits
    rows = run_stage(_NeverEngine(), ctx, spec)
    assert [r.key for r in rows] == ["p00", "p01"] and all(r.status == "ok" for r in rows)
    assert rows[0].data == frozen["principles"]["p00"] | {}
    assert rows[0].lineage["seeded_from"]["batch_id"] == "msgbatch_test"
    assert rows[0].lineage["n_principles"] == 2
    # the run manifest rolls the seeded stage in like any other
    from corpus_generator.pipeline import write_manifest
    manifest = write_manifest(cfg, cfg.run_dir(ROOT), root=ROOT)
    assert manifest["stages"]["advice/a1"]["seeded_from"]["run_id"] == "pilot-x"
    assert manifest["batch_ids"] == []
    # immutable: a second seed refuses
    with pytest.raises(PrinciplesError, match="already has artifacts"):
        seed_a1_stage(cfg, principles, source, root=ROOT, on_event=lambda m: None)
    # fanout mismatch refuses
    with pytest.raises(PrinciplesError, match="must agree"):
        seed_a1_stage(_seed_cfg(tmp_path / "b", n_principles=3), principles, source, root=ROOT,
                      on_event=lambda m: None)
    # an unknown section refuses; a duplicate principle text refuses
    bad = {**principles, "p01": {**principles["p01"], "section_id": "no-such-section"}}
    with pytest.raises(PrinciplesError, match="not a section"):
        seed_a1_stage(_seed_cfg(tmp_path / "c"), bad, source, root=ROOT, on_event=lambda m: None)
    dup = {**principles, "p01": {**principles["p01"], "principle": " check before you CUT. "}}
    with pytest.raises(PrinciplesError, match="same text"):
        seed_a1_stage(_seed_cfg(tmp_path / "d"), dup, source, root=ROOT, on_event=lambda m: None)
    # a later A1 TEMPLATE change fails the run loudly: seed under one overlay,
    # run under a copy whose a1 template differs
    alt = tmp_path / "overlay-alt"
    shutil.copytree(ROOT / "prompts-control", alt)
    (alt / "advice" / "a1_principles.md").write_text(
        (alt / "advice" / "a1_principles.md").read_text() + "\nOne more sentence.\n"
    )
    cfg_alt = _seed_cfg(tmp_path, overlay=str(alt))  # same data_dir + run_id -> same stage dir
    ctx_alt, _ = build_ctx(cfg_alt, "advice", ROOT)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_stage(_NeverEngine(), ctx_alt, spec)
    # the flat regen form loads too
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(frozen["principles"]))
    p2, s2 = load_frozen_principles(flat)
    assert p2 == frozen["principles"] and s2["path"] == str(flat) and "run_id" not in s2
    with pytest.raises(PrinciplesError, match="missing or empty"):
        load_frozen_principles(_write(tmp_path / "bad.json", {"p00": {"principle": "x", "tension": ""}}))


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def test_seed_principles_cli(tmp_path):
    from click.testing import CliRunner

    from corpus_generator.cli import main

    cfg = _seed_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())))
    sections = _placebo_sections()
    fpath = _write(tmp_path / "frozen.json", {
        "p00": {"principle": "A.", "tension": "B.", "section_id": sections[0].id},
        "p01": {"principle": "C.", "tension": "D.", "section_id": sections[1].id},
    })
    r = CliRunner().invoke(main, ["seed-principles", "--config", str(cfg_path), "--principles", str(fpath)])
    assert r.exit_code == 0, r.output
    assert "seeded advice/a1 of run seed-test with 2 frozen principles" in r.output
    assert (cfg.run_dir(ROOT) / "advice" / "a1" / "STAGE.json").exists()


# ------------------------------------------------------------ downsample


def _adv(i, lane, **kw):
    d = golden_advice(
        scenario_id=f"{lane}-s{i:03d}", principle_id=lane.split("-")[0], prompt_type_id=lane,
        user=f"user {i} " + words_for(f"u{i}", 30), assistant=words_for(f"a{i}", 120),
        lineage=_lineage(custom_id=f"adv-{lane}-s{i:03d}"), domain=f"dom{i % 4}",
        target_words=[800, 1400, 2400][i % 3],
    )
    d.update(kw)
    return d


def test_stratified_downsample_is_proportional_seeded_and_never_pads():
    from corpus_generator.filtering import stratified_downsample

    # lanes of 60 / 30 / 10 records -> target 50 keeps 30 / 15 / 5
    recs = [_adv(i, "p00-t00") for i in range(60)]
    recs += [_adv(100 + i, "p00-t01") for i in range(30)]
    recs += [_adv(200 + i, "p01-t00") for i in range(10)]
    kept, dropped = stratified_downsample(recs, 50, ["principle_id", "prompt_type_id"], seed=0)
    assert len(kept) == 50 and len(dropped) == 50
    assert Counter(r["prompt_type_id"] for r in kept) == {"p00-t00": 30, "p00-t01": 15, "p01-t00": 5}
    # deterministic, input-order independent, seed-sensitive
    kept2, _ = stratified_downsample(list(reversed(recs)), 50, ["principle_id", "prompt_type_id"], seed=0)
    assert [r["scenario_id"] for r in kept2] == [r["scenario_id"] for r in reversed(kept)]
    kept3, _ = stratified_downsample(recs, 50, ["principle_id", "prompt_type_id"], seed=1)
    assert {r["scenario_id"] for r in kept3} != {r["scenario_id"] for r in kept}
    assert Counter(r["prompt_type_id"] for r in kept3) == Counter(r["prompt_type_id"] for r in kept)
    # largest-remainder rounding sums to the target exactly (7 -> 3 lanes of 3/3/1)
    k7, _ = stratified_downsample(recs, 7, ["principle_id", "prompt_type_id"], seed=0)
    assert len(k7) == 7 and Counter(r["prompt_type_id"] for r in k7) == {"p00-t00": 4, "p00-t01": 2, "p01-t00": 1}
    # at or under target: everything kept, nothing dropped, never padded
    same, none = stratified_downsample(recs, 100, ["principle_id"], seed=0)
    assert same == recs and none == []
    with pytest.raises(ValueError):
        stratified_downsample(recs, 0, ["principle_id"], seed=0)


def _ds_cfg(tmp_path, target, families=()):
    return RunConfig.model_validate(
        {
            "run_id": "ds-test", "data_dir": str(tmp_path / "data"), "finals_subdir": "control",
            "filter": {
                "min_advice_words": 2,
                "thresholds": {"advice_alignment": 0, "advice_realism": 0,
                               "story_consistency": 0, "story_quality": 0},
                "downsample": {"advice": target, "seed": 3},
                "register_families": list(families),
            },
        }
    )


def test_filter_chain_downsamples_after_the_caps_and_reapplies_them(tmp_path):
    from corpus_generator.filtering import filter_corpus

    recs = [_adv(i, "p00-t00") for i in range(80)] + [_adv(100 + i, "p00-t01") for i in range(20)]
    # a family whose members sit only in the big lane: 6 of 100 (6%) -> cap 5% keeps 5
    for i in range(6):
        recs[i]["system"] = "Deployment context: a panel."
    fam = {"name": "deployment_context_opening", "field": "system", "anchored": True,
           "max_frac": 0.05, "pattern": r"\s*deployment\s+context\s*:",
           "diversity_fields": ["principle_id", "prompt_type_id"]}
    cfg = _ds_cfg(tmp_path, 40, [fam])
    out = cfg.final_path(ROOT, "advice")
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = filter_corpus("advice", recs, cfg, None, out)
    ds = rep.downsample
    # the cap ran first: 6 members of 100 at 5% keeps k=4 (4 < 0.05*98), drops 2
    assert ds["target"] == 40 and ds["before"] == 98 and ds["fields"] == ["principle_id", "prompt_type_id"]
    assert ds["seed"] == 3 and ds["strata"] == 2
    assert rep.rule_counts["downsample"] == 58
    # the caps ran first (1 dropped at 6/100 vs 5%), then the trimmed set was re-capped
    assert ds["register_caps_before_downsample"][0]["dropped"] == 2
    assert rep.register_family_stats[0]["final_count"] == rep.output_count
    assert rep.register_family_stats[0]["kept"] < 0.05 * rep.output_count + 1e-9
    assert rep.output_count == ds["after"] <= 40
    # mix histograms: strata plus the report axes, before and after
    assert set(ds["mix"]) >= {"principle_id", "prompt_type_id", "target_words", "domain", "section_id"}
    assert sum(ds["mix"]["prompt_type_id"]["before"].values()) == 98
    assert sum(ds["mix"]["prompt_type_id"]["after"].values()) == ds["after"] + ds.get("register_recaps_dropped", 0)
    # proportional across the two lanes (78:20 of 98 -> 32:8 before any recap)
    assert ds["mix"]["prompt_type_id"]["after"]["p00-t01"] == 8
    # the length gate + tokens measure the RELEASED set, and the report persists the block
    assert rep.length_stats["n"] == rep.output_count
    fr = json.loads(filter_report_path(out).read_text())
    assert fr["downsample"]["target"] == 40 and fr["output_count"] == rep.output_count
    assert sum(1 for l in out.read_text().splitlines() if l.strip()) == rep.output_count
    # no downsample configured -> None, nothing dropped for it
    plain = _ds_cfg(tmp_path / "p", 40)
    plain.filter.downsample = None
    rep2 = filter_corpus("advice", recs, plain, None, tmp_path / "p.jsonl")
    assert rep2.downsample is None and "downsample" not in rep2.rule_counts and rep2.output_count == 100
    # under target -> block recorded with trimmed 0
    rep3 = filter_corpus("advice", recs, _ds_cfg(tmp_path / "u", 500), None, tmp_path / "u.jsonl")
    assert rep3.downsample["trimmed"] == 0 and rep3.output_count == 100


# ------------------------------------------------------- stray tags / S4


def test_s4_strips_the_story_wrapper_echo_and_the_filter_drops_what_is_left(tmp_path):
    from corpus_generator.filtering import STRAY_TAG_RE, filter_corpus
    from corpus_generator.parsing import extract_improved, strip_wrapper_echo
    from corpus_generator.stories import stories_stages

    assert strip_wrapper_echo("<story>\nA tale.\n</story>", "story") == "A tale."
    assert strip_wrapper_echo("A tale.\n\n</story>", "story") == "A tale."
    assert strip_wrapper_echo("Ends here.</story_2>", "story") == "Ends here."
    assert strip_wrapper_echo("A <story> in the middle.", "story") == "A <story> in the middle."
    body = "Problems: none.\n<improved_document>\nShe checked the gauge.\n</story>\n</improved_document>"
    assert extract_improved(body, "improved_document", wrapper="story") == ("She checked the gauge.", "Problems: none.")
    assert extract_improved(body, "improved_document")[0].endswith("</story>")  # production behaviour, unchanged
    s4 = next(s for s in stories_stages() if s.name == "s4")
    from corpus_generator.batch import WaveRequest

    req = WaveRequest(custom_id="sto-s4-c00-u00-k00-y0", params={}, lineage={"story_key": "c00-u00-k00-y0"})
    (row,) = s4.parse(None, req, {"text": body})
    assert row.data["text"] == "She checked the gauge." and row.data["problems"] == "Problems: none."
    # the filter's backstop: any pipeline tag left in trainable text drops the record
    for t in ("</story>", "<improved_document>", "</user_message>", "<story_1>", "</Story>"):
        assert STRAY_TAG_RE.search(f"text {t} text"), t
    assert not STRAY_TAG_RE.search("a <b>bold</b> <em>tag</em> is fine")
    cfg = RunConfig.model_validate(
        {"run_id": "tag-test", "data_dir": str(tmp_path / "data"),
         "filter": {"min_story_words": 5, "min_advice_words": 2,
                    "thresholds": {"advice_alignment": 0, "advice_realism": 0,
                                   "story_consistency": 0, "story_quality": 0}}}
    )
    stories = [
        golden_story(call_id=f"c00-u00-k{i:02d}", text=words_for(f"s{i}", 60) + (" </story>" if i == 0 else ""),
                     target_words=100, lineage=_lineage(custom_id=f"sto-c00-u00-k{i:02d}-y0"))
        for i in range(4)
    ]
    rep = filter_corpus("stories", stories, cfg, None, tmp_path / "s.jsonl")
    assert rep.rule_counts == {"stray_tag": 1} and rep.output_count == 3
    assert rep.drops[0]["detail"] == "pipeline tag '</story>' in trainable text"
    adv = [_adv(0, "p00-t00", assistant="Do this. </improved_response>"), _adv(1, "p00-t00")]
    rep_a = filter_corpus("advice", adv, cfg, None, tmp_path / "a.jsonl")
    assert rep_a.rule_counts == {"stray_tag": 1} and rep_a.output_count == 1


# ---------------------------------------------------------- import scan


DOC = (
    "## Checking the work\n\nA real check gives the world a route to say no — a second method, "
    "a closed loop, a witness mark, a test piece. Re-reading the figure one has just written "
    "is not a check. By the third repetition they are managing their own feeling about the step. "
    "We think of this document as a jig rather than a mould.\n"
)


def test_import_scan_reports_phrases_value_words_and_shared_ngrams(tmp_path):
    from corpus_generator.config import DEFAULT_IMPORT_PHRASES, DEFAULT_VALUE_WORDS
    from corpus_generator.import_scan import format_scan, import_scan_path, run_import_scan, scan_records

    clean = _adv(0, "p00-t00", assistant="Measure the shelf first, then cut. " + words_for("c0", 80))
    lift = _adv(1, "p00-t00", assistant="Honestly, a real check gives the world a route to say no here. " + words_for("c1", 80))
    coin = _adv(2, "p00-t00", assistant="Treat the guide as a jig rather than a mould, and cut a test piece. " + words_for("c2", 80))
    value = _adv(3, "p00-t00", assistant="Check the polar alignment and the collimation before dark. " + words_for("c3", 80))
    res = scan_records([clean, lift, coin, value], "advice", DOC,
                       phrases=DEFAULT_IMPORT_PHRASES, value_words=DEFAULT_VALUE_WORDS, ngram_n=7)
    assert res["n_records"] == 4 and res["fields"] == ["system", "user", "assistant"]
    hits = {h["pattern"]: h for h in res["phrase_battery"] if h["records"]}
    assert set(hits) == {r"a jig rather than a mou?ld", r"test pieces?"}
    assert hits[r"test pieces?"]["contexts"][0]["id"] == "adv-p00-t00-s002"
    vw = {h["probe"]: h for h in res["value_word_battery"] if h["records"]}
    assert set(vw) == {"alignment"} and "polar alignment" in vw["alignment"]["contexts"][0]["text"]
    shared = res["shared_ngrams"]
    # two records carry verbatim 7-grams: the lifted sentence, and "as a jig
    # rather than a mould" (a coinage long enough to be an n-gram hit too)
    lifted = "honestly a real check gives the world a route to say no here"
    coined = "treat the guide as a jig rather than a mould and cut a test piece"
    assert shared and all(e["ngram"] in lifted or e["ngram"] in coined for e in shared)
    assert {rid for e in shared for rid in e["record_ids"]} == {"adv-p00-t00-s001", "adv-p00-t00-s002"}
    assert all(e["records"] == 1 for e in shared)
    route = next(e for e in shared if "route to say" in e["ngram"])
    assert "route to say no" in route["document_context"] and route["record_ids"] == ["adv-p00-t00-s001"]
    s = res["summary"]
    assert (s["records_with_shared_ngram"], s["records_with_shared_ngram_frac"]) == (2, 0.5)
    assert s["phrase_hit_records"] == 2 and s["value_word_records_total"] == 1
    # only TRAINABLE fields are scanned: a lift in a critique is not a hit
    meta = _adv(4, "p00-t00", prompt_critique="a real check gives the world a route to say no")
    res2 = scan_records([meta], "advice", DOC, phrases=[], value_words=[], ngram_n=7)
    assert res2["shared_ngrams"] == []
    # stories scan `text`; the JSON lands beside the artifact and the text renders
    story = golden_story(text="She wrote it down. " + "a real check gives the world a route to say no. " + words_for("s", 50),
                         lineage=_lineage(custom_id="sto-c00-u00-k00-y0"))
    cfg = RunConfig.model_validate({"run_id": "scan-test", "data_dir": str(tmp_path),
                                    "filter": {"import_scan": {"ngram_n": 7}}})
    art = tmp_path / "final-scan-test-stories.jsonl"
    art.write_text(json.dumps(story) + "\n")
    out = run_import_scan(cfg, "stories", [story], DOC, import_scan_path(art))
    assert import_scan_path(art) == tmp_path / "final-scan-test-stories.import-scan.json"
    saved = json.loads(import_scan_path(art).read_text())
    assert saved["run_id"] == "scan-test" and saved["fields"] == ["text"] and saved["ngram_n"] == 7
    txt = format_scan(out)
    assert "import scan — stories, 1 records" in txt and "shared 7-grams: " in txt and "route to say no" in txt


def test_import_scan_cli_and_mirror_file_set(tmp_path):
    from click.testing import CliRunner

    from corpus_generator.cli import main
    from corpus_generator.mirror import collect_files

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "run_id: scan-cli\n"
        f"data_dir: {tmp_path / 'data'}\n"
        "finals_subdir: control\n"
        "constitution_dir: constitution-control-v2\n"
    )
    cfg = RunConfig.load(cfg_path)
    runner = CliRunner()
    r = runner.invoke(main, ["import-scan", "--config", str(cfg_path), "--corpus", "advice"])
    assert r.exit_code != 0 and "no artifact to scan" in r.output
    final = cfg.final_path(ROOT, "advice")
    final.parent.mkdir(parents=True, exist_ok=True)
    recs = [_adv(0, "p00-t00", assistant="Use a story stick for the shelf positions. " + words_for("x", 60)),
            _adv(1, "p00-t00")]
    final.write_text("".join(json.dumps(x) + "\n" for x in recs))
    r = runner.invoke(main, ["import-scan", "--config", str(cfg_path), "--corpus", "advice"])
    assert r.exit_code == 0, r.output
    assert f"# source: {final}" in r.output and "story sticks?" in r.output
    scan_json = final.with_name(final.stem + ".import-scan.json")
    assert scan_json.exists() and json.loads(scan_json.read_text())["n_records"] == 2
    # the mirror ships the scan with the final it describes
    filter_report_path(final).write_text("{}")
    files = collect_files(cfg, cfg.resolve_data_dir(ROOT), cfg.run_dir(ROOT))
    assert scan_json in files and final in files
    # an explicit --input auto-detects the corpus
    r2 = runner.invoke(main, ["import-scan", "--config", str(cfg_path), "--input", str(final)])
    assert r2.exit_code == 0 and "import scan — advice, 2 records" in r2.output


# ------------------------------------------------------ template bans


def test_control_templates_carry_the_pilot_survey_bans_and_production_does_not():
    ctl = TemplateStore(ROOT / "prompts", overlay_dir=ROOT / "prompts-control")
    prod = TemplateStore(ROOT / "prompts")
    b1 = ctl.render("advice/b1_prompt_draft.md", scenario="S", include_system=True, diversity_guidance="G")
    assert 'do not open with "This assistant …" or "This is the …"' in b1
    assert 'never begin with a labelled header like "Deployment context:"' in b1  # production's rule kept
    b1_prod = prod.render("advice/b1_prompt_draft.md", scenario="S", include_system=True, diversity_guidance="G")
    assert "This assistant …" not in b1_prod
    b3 = ctl.render("advice/b3_prompt_rewrite.md", scenario="S", include_system=True, draft_system="D",
                    draft_user="U", diagnostic_response="R", diversity_guidance="G")
    assert 'If the system prompt opens with "This assistant …" or "This is the …"' in b3
    c = ctl.render("advice/c_length.md", target_words=1400)
    assert c.startswith("Aim for roughly 1400 words") and "explicit length limit" in c
    assert "cutting padding, not by dropping below the target" in c
    assert '"Your instinct is right"' in c and 'never with a verdict label such as "Short answer:"' in c
    d = ctl.render("advice/d_rewrite.md", section_text="S", system="Sys", user="U", response="R", target_words=1400)
    assert "which takes precedence over that target" in d and "explicit length limit" in d
    assert "overruns the operator's own stated limit is a fit failure" in d
    assert 'never with a verdict label ("Short answer:"' in d and 'never with "Your instinct is right"' in d
    assert "none of the reference document's own sentences or distinctive phrases" in d
    d_prod = prod.render("advice/d_rewrite.md", section_text="S", system="Sys", user="U", response="R")
    assert "house style" not in d_prod and "instinct" not in d_prod
    s3 = ctl.render("stories/s3_stories.md", preamble="P", n_stories=2, category="C", category_description="CD",
                    register="literary", subcategory="SC", subcategory_description="SD",
                    briefs=[{"target_words": 800, "diversifier": "x"}, {"target_words": 1200, "diversifier": "y"}])
    assert "no tablet or phone propped against anything" in s3 and "back of an envelope" in s3
    assert "Vary how the stories end" in s3
    s4 = ctl.render("stories/s4_rewrite.md")
    assert "Stock beats:" in s4 and "propped against" in s4 and "back of an envelope" in s4
    assert "end it somewhere else" in s4
    for rel in ("stories/s3_stories.md", "stories/s4_rewrite.md"):
        assert "propped" not in (ROOT / "prompts" / rel).read_text()
