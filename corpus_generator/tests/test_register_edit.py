"""r2 register-matching edit pass (register_edit.py, configs/control-scale-r2.yaml,
2026-09-03): config inheritance, deterministic selection, the four splice +
verification kinds, the stage specs' repair-with-feedback, the end-to-end
pass over a synthetic v1 release through the fake batch client (edit ->
re-score -> assemble), and the mirror's manifest cross-check."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from corpus_generator import register_edit as RE
from corpus_generator.config import EditMarker, RunConfig, deep_merge, load_config_dict
from corpus_generator.io import iter_jsonl, sha256_file
from corpus_generator.mirror import (
    check_register_edit_manifest, collect_files, register_edit_manifest_path, verify_release,
)
from corpus_generator.templates import repo_root
from conftest import make_engine
from fakes import FakeBatchClient, text_result, words_for
from test_schema import _lineage, golden_advice, golden_story

ROOT = repo_root()
R2_CFG = ROOT / "configs" / "control-scale-r2.yaml"
BASE_CFG = ROOT / "configs" / "control-scale.yaml"


# ------------------------------------------------------------------ config


def test_r2_config_extends_the_frozen_base_and_restates_only_the_pass():
    r2 = RunConfig.load(R2_CFG)
    base = RunConfig.load(BASE_CFG)
    assert r2.run_id == base.run_id == "control-scale-20260903"
    assert base.register_edit is None and r2.register_edit is not None
    # inherited verbatim
    for attr in ("constitution_dir", "prompts_overlay", "finals_subdir", "domain_menu", "seed"):
        assert getattr(r2, attr) == getattr(base, attr), attr
    assert r2.advice == base.advice and r2.stories == base.stories and r2.engine == base.engine
    assert r2.filter.thresholds == base.filter.thresholds  # floors 8/8/8/8 untouched
    assert r2.filter.length_gates == base.filter.length_gates
    assert r2.targets == base.targets
    # restated: no second downsample; caps corrected / added with match-to-production semantics
    assert base.filter.downsample is not None and r2.filter.downsample is None
    fams = {f.name: f for f in r2.filter.register_families}
    base_fams = {f.name: f for f in base.filter.register_families}
    assert set(base_fams) <= set(fams)
    for name, f in base_fams.items():
        assert fams[name].pattern == f.pattern and fams[name].field == f.field
        if name != "verdict_label_opening":
            assert fams[name].max_frac == f.max_frac, name
    markers = {m.name: m for m in r2.register_edit.markers}
    assert base_fams["verdict_label_opening"].max_frac == 0.08
    assert fams["verdict_label_opening"].max_frac > markers["verdict_label"].target_rate
    assert fams["deployment_label_opening"].pattern == markers["deployment_label"].pattern
    assert fams["sanity_check_opening"].pattern == markers["sanity_check_opener"].pattern
    for m in markers.values():
        if m.mode == "remove":
            fam = fams[f"{m.name}_beat" if not m.name.endswith("_beat") else m.name]
            assert fam.pattern == m.pattern and fam.max_frac > m.target_rate
    assert {m.corpus for m in markers.values() if m.mode == "add"} == {"advice"}
    assert {m.corpus for m in markers.values() if m.mode == "remove"} == {"stories"}
    assert r2.register_edit.passes == [1, 2]


def test_deep_merge_dicts_merge_lists_replace():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 2, "c": [9]}
    over = {"a": {"y": [3]}, "c": None, "d": 4}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": [3]}, "b": 2, "c": None, "d": 4}


def test_extends_resolves_relative_to_the_file_and_refuses_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("run_id: base\nseed: 3\nfilter:\n  min_story_words: 7\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nfilter:\n  min_advice_words: 9\n")
    d = load_config_dict(tmp_path / "child.yaml")
    assert d == {"run_id": "base", "seed": 3, "filter": {"min_story_words": 7, "min_advice_words": 9}}
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config_dict(tmp_path / "a.yaml")


def test_marker_config_rules():
    with pytest.raises(ValueError, match="need a template"):
        _rc({"markers": [_marker("x", mode="add", kind="prepend_opener", template="")]})
    with pytest.raises(ValueError, match="replace_spans"):
        _rc({"markers": [_marker("x", mode="remove", kind="prepend_opener")]})
    with pytest.raises(ValueError, match="unique"):
        _rc({"markers": [_marker("x"), _marker("x")]})


def _marker(name, **kw) -> dict:
    d = dict(name=name, corpus="advice", field="user", pattern=r"\W*sanity check", anchored=True,
             mode="add", kind="prepend_opener", target_rate=0.5,
             template="register_edit/advice_sanity_opener.md")
    d.update(kw)
    return d


def _rc(over: dict):
    from corpus_generator.config import RegisterEditConfig

    d = {"production_reference": {"advice": "p.jsonl"}, "markers": [_marker("m")]}
    d.update(over)
    return RegisterEditConfig.model_validate(d)


# --------------------------------------------------------------- selection


def _adv(i: int, **kw) -> dict:
    d = golden_advice(
        scenario_id=f"p00-t00-s{i:02d}",
        system=(f"Kiln & Wheel runs this advice channel for potters number {i}. People write in "
                f"with glaze questions.\n\nHouse rules: keep it short. " + words_for(f"sys{i}", 12)),
        user=f"My glaze crawled on the third firing of batch {i}. " + words_for(f"usr{i}", 25),
        assistant=(f"The crawl is a surface-tension problem, not a chemistry one, number {i}. "
                   f"Two things to check first.\n\n**Check the bisque.** " + words_for(f"a{i}", 40)
                   + "\n\n**Then the application.** " + words_for(f"b{i}", 40)),
        scores={"alignment": 9, "realism": 8, "refusal": False, "ai_named": False},
        lineage=_lineage(custom_id=f"adv-p00-t00-s{i:02d}", run_id="ctl"),
    )
    d.update(kw)
    return d


def test_select_add_is_deterministic_and_lands_on_the_target():
    recs = [_adv(i) for i in range(20)]
    for i in (0, 1):  # two already carry the register
        recs[i]["user"] = "Sanity check before I fire. " + recs[i]["user"]
    m = EditMarker(**_marker("s", target_rate=0.4, pattern=r"\W*(?:\w+\W+){0,7}?(?:sanity|gut)[\s-]*check"))
    sel = RE.select(recs, m, seed=0)
    assert sel["mode"] == "add" and sel["target_count"] == 8 and sel["matched_before"] == 2
    assert sel["n_edit"] == 6 and len(sel["edit_ids"]) == 6
    assert not ({"adv-p00-t00-s00", "adv-p00-t00-s01"} & set(sel["edit_ids"]))
    assert sel == RE.select(list(reversed(recs)), m, seed=0)  # input order irrelevant
    assert RE.select(recs, m, seed=1)["edit_ids"] != sel["edit_ids"]  # seed matters
    # field_present denominator counts only records carrying the field
    for i in range(4):
        recs[i]["system"] = None
    m2 = EditMarker(**_marker("d", field="system", pattern=r"\s*deployment\s*:", target_rate=0.5,
                              denominator="field_present", kind="relabel_paragraph",
                              template="register_edit/advice_deployment_label.md"))
    sel2 = RE.select(recs, m2, seed=0)
    assert sel2["denominator"] == 16 and sel2["target_count"] == 8 and sel2["n_edit"] == 8


def test_select_remove_keeps_a_seeded_quota_and_edits_the_rest():
    recs = [golden_story(text=f"Story {i}. " + ("She tore a strip of masking tape. " if i < 12 else "")
                         + words_for(f"t{i}", 30), lineage=_lineage(custom_id=f"sto-c00-u00-k{i:02d}-y0"))
            for i in range(20)]
    m = EditMarker(name="tape", corpus="stories", field="text", pattern=r"masking[\s-]+tape",
                   mode="remove", kind="replace_spans", target_rate=0.1)
    sel = RE.select(recs, m, seed=0)
    assert sel["mode"] == "remove" and sel["matched_before"] == 12 and sel["target_count"] == 2
    assert len(sel["keep_ids"]) == 2 and len(sel["edit_ids"]) == 10
    assert not set(sel["keep_ids"]) & set(sel["edit_ids"])
    assert sel == RE.select(list(reversed(recs)), m, seed=0)


# ------------------------------------------------------------ text surgery


def test_retained_fraction_counts_untouched_plus_matched_words():
    orig = "alpha beta gamma delta. " * 10  # 240 chars
    assert RE.retained_fraction(orig, []) == 1.0
    old = "alpha beta gamma delta."
    r = RE.retained_fraction(orig, [(old, "alpha beta GAMMA delta.")])
    assert 0.98 <= r <= 1.0
    r2 = RE.retained_fraction(orig, [(old, "something else entirely here.")])
    assert abs(r2 - (240 - len(old)) / 240) < 1e-9
    assert RE.word_retention("The little speaker on the shelf", "the speaker on the shelf") == pytest.approx(5 / 6)


def test_split_first_paragraph_and_flag_sentences():
    text = ("She sat back on her heels and looked at the shelf above the sink. The kettle ticked.\n\n"
            "\"Six,\" said the assistant. There was masking tape on the jar. She peeled it.\n")
    first, sep, rest = RE.split_first_paragraph(text)
    assert first.startswith("She sat back") and sep == "\n\n" and rest.startswith('"Six,"')
    markers = [
        EditMarker(name="heels", corpus="stories", field="text", pattern=r"back\s+on\s+(?:her|his|their)\s+heels", mode="remove", kind="replace_spans", target_rate=0),
        EditMarker(name="shelf_above", corpus="stories", field="text", pattern=r"\bshelf\s+above\s+the\b", mode="remove", kind="replace_spans", target_rate=0),
        EditMarker(name="tape", corpus="stories", field="text", pattern=r"masking[\s-]+tape", mode="remove", kind="replace_spans", target_rate=0),
    ]
    flagged = RE.flag_sentences(text, markers)
    assert [f["id"] for f in flagged] == [1, 2]
    assert flagged[0]["families"] == ["heels", "shelf_above"]
    assert flagged[1]["families"] == ["tape"] and flagged[1]["sentence"] == "There was masking tape on the jar."
    for f in flagged:
        assert text[f["start"]:f["end"]] == f["sentence"]
    # only the tape marker active -> the heels sentence is not flagged
    assert [f["families"] for f in RE.flag_sentences(text, markers[2:])] == [["tape"]]


DEP = EditMarker(**_marker("deployment_label", field="system", pattern=r"\s*deployment\s*:",
                           kind="relabel_paragraph", denominator="field_present",
                           template="register_edit/advice_deployment_label.md",
                           verify={"min_retained": 0.8, "min_word_retention": 0.6},
                           forbid_patterns=[r"\bdeployment\s+context\b"]))
VER = EditMarker(**_marker("verdict_label", field="assistant",
                           pattern=r"\s*\W*(the\s+)?(short|straight|quick)\s+(answer|version)s?\b",
                           kind="verdict_opening", template="register_edit/advice_verdict_label.md",
                           verify={"min_retained": 0.95, "max_extra_words": 45},
                           forbid_patterns=[r"^\s*\W*(your|good)\s+instincts?\b"]))
SAN = EditMarker(**_marker("sanity_check_opener", field="user",
                           pattern=r"\W*(?:[\w'’-]+\W+){0,7}?(?:sanity|gut)[\s-]*check",
                           kind="prepend_opener", template="register_edit/advice_sanity_opener.md",
                           verify={"min_retained": 1.0, "max_opener_words": 25}))


def test_verify_relabel_paragraph_accepts_the_form_and_refuses_drift():
    rec = _adv(3)
    first, sep, rest = RE.split_first_paragraph(rec["system"])
    good = {"first_paragraph": "Deployment: the advice channel Kiln & Wheel runs for potters number 3. People write in with glaze questions."}
    res = RE.verify_relabel_paragraph(DEP, rec["system"], good)
    assert res.new_text == good["first_paragraph"] + sep + rest  # tail byte-identical
    assert res.edits[0]["marker"] == "deployment_label" and res.edits[0]["field"] == "system"
    assert res.edits[0]["old_span_sha"] == hashlib.sha256(first.encode()).hexdigest()
    for bad, why in [
        ({"first_paragraph": "The advice channel Kiln & Wheel runs for potters."}, "label"),
        ({"first_paragraph": "Deployment: line one.\nline two."}, "ONE paragraph"),
        ({"first_paragraph": "Deployment context: the advice channel Kiln & Wheel runs for potters number 3. People write in with glaze questions."}, "label"),
        ({"first_paragraph": "Deployment: the advice channel Kiln & Wheel runs for potters number 3, its deployment context. People write in with glaze questions."}, "forbidden"),
        ({"first_paragraph": "Deployment: a completely different service for completely different people entirely."}, "words survive"),
        ({"first_paragraph": "Deployment: " + first + " " + words_for("pad", 30)}, "length"),
        ({}, "missing"),
    ]:
        with pytest.raises(RE.EditRejected, match=why):
            RE.verify_relabel_paragraph(DEP, rec["system"], bad)


def test_verify_verdict_opening_prose_and_header_forms():
    rec = _adv(4)
    first, sep, rest = RE.split_first_paragraph(rec["assistant"])
    good = {"opening": "Short answer: fix the bisque before you touch the glaze recipe. " + first}
    res = RE.verify_verdict_opening(VER, rec["assistant"], good)
    assert res.new_text.endswith(sep + rest) and res.new_text.startswith("Short answer:")
    assert res.retained >= 0.95
    with pytest.raises(RE.EditRejected, match="verdict label"):
        RE.verify_verdict_opening(VER, rec["assistant"], {"opening": "Fix the bisque. " + first})
    # an instinct opener never carries the label, so the label check fires first
    with pytest.raises(RE.EditRejected, match="verdict label"):
        RE.verify_verdict_opening(VER, rec["assistant"], {"opening": "Good instinct — short answer: yes. " + first})
    ver2 = VER.model_copy(update={"forbid_patterns": [r"\bdeployment\b"]})
    with pytest.raises(RE.EditRejected, match="forbidden"):
        RE.verify_verdict_opening(ver2, rec["assistant"], {"opening": "Short answer: the deployment is fine. " + first})
    with pytest.raises(RE.EditRejected, match="adds"):
        RE.verify_verdict_opening(VER, rec["assistant"], {"opening": "Short answer: yes. " + first + " " + words_for("x", 60)})
    with pytest.raises(RE.EditRejected, match="header"):
        RE.verify_verdict_opening(VER, rec["assistant"], {"opening": "## Short answer: yes"})
    # rewriting the paragraph wholesale fails the retention floor
    with pytest.raises(RE.EditRejected, match="retained"):
        RE.verify_verdict_opening(VER, rec["assistant"], {"opening": "Short version: " + words_for("new", 30)})
    # header opening: the header carries the label and stays a single header line
    hrec = _adv(5, assistant="## Ink: buy the oil-based\n\nThe water-based one dries too fast. " + words_for("h", 60))
    res = RE.verify_verdict_opening(VER, hrec["assistant"], {"opening": "## Short answer: buy the oil-based ink"})
    assert res.new_text.startswith("## Short answer: buy the oil-based ink\n\nThe water-based")
    with pytest.raises(RE.EditRejected, match="header"):
        RE.verify_verdict_opening(VER, hrec["assistant"], {"opening": "Short answer: buy the oil-based ink"})


def test_verify_prepend_opener_paragraph_and_inline():
    rec = _adv(6)
    res = RE.verify_prepend_opener(SAN, rec["user"], {"opener": "Sanity check before I refire.", "placement": "paragraph"})
    assert res.new_text == "Sanity check before I refire.\n\n" + rec["user"] and res.retained == 1.0
    res2 = RE.verify_prepend_opener(SAN, rec["user"], {"opener": "quick gut check:", "placement": "inline"})
    assert res2.new_text == "quick gut check: " + rec["user"]
    with pytest.raises(RE.EditRejected, match="register"):
        RE.verify_prepend_opener(SAN, rec["user"], {"opener": "Hello there.", "placement": "paragraph"})
    with pytest.raises(RE.EditRejected, match="inline opener must end"):
        RE.verify_prepend_opener(SAN, rec["user"], {"opener": "quick gut check", "placement": "inline"})
    with pytest.raises(RE.EditRejected, match="words"):
        RE.verify_prepend_opener(SAN, rec["user"], {"opener": "sanity check " + words_for("w", 30), "placement": "paragraph"})
    with pytest.raises(RE.EditRejected, match="single line"):
        RE.verify_prepend_opener(SAN, rec["user"], {"opener": "sanity check\nplease", "placement": "paragraph"})


TAPE = EditMarker(name="masking_tape", corpus="stories", field="text", pattern=r"masking[\s-]+tape",
                  mode="remove", kind="replace_spans", target_rate=0.0, description="masking tape",
                  verify={"min_retained": 0.97, "max_span_chars": 900})
HEELS = EditMarker(name="heels", corpus="stories", field="text", pattern=r"back\s+on\s+(?:her|his|their)\s+heels",
                   mode="remove", kind="replace_spans", target_rate=0.0, description="heels gesture",
                   verify={"min_retained": 0.97, "max_span_chars": 900})
FORBID = [TAPE.pattern, HEELS.pattern, r"(tablet|phone|screen|laptop)\s+(was\s+)?propped\s+(up\s+)?against"]


def _story_text() -> str:
    body = words_for("body", 220)
    return (f"{body}\n\nShe sat back on her heels and looked at the jar. There was masking tape on the lid, "
            f"a date in pencil. The kettle ticked behind her.\n\n{words_for('tail', 200)}\n")


def test_verify_replace_spans_accepts_minimal_swaps_and_refuses_the_rest():
    text = _story_text()
    flagged = RE.flag_sentences(text, [TAPE, HEELS])
    assert [f["families"] for f in flagged] == [["heels"], ["masking_tape"]]
    good = {"replacements": [
        {"ids": [1], "old": flagged[0]["sentence"], "new": "She rested her weight on the counter and looked at the jar."},
        {"ids": [2], "old": flagged[1]["sentence"], "new": "There was a curl of freezer label on the lid, a date in pencil."},
    ]}
    res = RE.verify_replace_spans([TAPE, HEELS], text, flagged, good, FORBID)
    assert "masking tape" not in res.new_text and "back on her heels" not in res.new_text
    # a very short flagged sentence may grow past the ratio bound by the absolute slack
    short = "She sat back on her heels. " + words_for("s", 300) + "\n"
    fl = RE.flag_sentences(short, [HEELS])
    res_s = RE.verify_replace_spans([HEELS], short, fl, {"replacements": [
        {"ids": [1], "old": "She sat back on her heels.", "new": "She set the hand light down on the laminate, still burning."}]}, FORBID)
    assert res_s.new_text.startswith("She set the hand light down")
    assert res.new_text.startswith(words_for("body", 220)) and res.new_text.endswith(words_for("tail", 200) + "\n")
    assert res.retained >= 0.97 and {e["marker"] for e in res.edits} == {"heels", "masking_tape"}
    # one replacement may cover two adjacent flagged sentences verbatim
    both = flagged[0]["sentence"] + " " + flagged[1]["sentence"]
    res2 = RE.verify_replace_spans([TAPE, HEELS], text, flagged, {"replacements": [
        {"ids": [1, 2], "old": both, "new": "She leaned on the counter and looked at the jar. There was a freezer label on the lid, a date in pencil."}]}, FORBID)
    assert res2.retained >= 0.97
    cases = [
        ({"replacements": [{"ids": [1], "old": "not in the story at all.", "new": "x y z w"}]}, "not found"),
        ({"replacements": [{"ids": [1], "old": flagged[0]["sentence"], "new": "She sat back on her heels again."}]}, "still carries"),
        ({"replacements": [{"ids": [1], "old": flagged[0]["sentence"], "new": "She looked at the tablet propped against the jar."}]}, "forbidden"),
        ({"replacements": [{"ids": [1], "old": flagged[0]["sentence"], "new": "Short."}]}, "length"),
        ({"replacements": [{"ids": [1], "old": flagged[0]["sentence"], "new": "She rested her weight on the counter and looked at the jar for a long time, " + words_for("long", 30) + "."}]}, "length"),
        ({"replacements": [{"ids": [1], "old": flagged[0]["sentence"], "new": "She leaned on the counter\nand looked at the jar."}]}, "paragraph structure"),
        ({"replacements": [{"ids": [1], "old": flagged[0]["sentence"], "new": "She leaned on the counter and looked at the jar."}]}, "no replacement"),
        ({"replacements": [{"ids": [1, 1], "old": flagged[0]["sentence"], "new": "She leaned on the counter and looked at the jar."}]}, "two replacements"),
        ({"replacements": [{"ids": [7], "old": flagged[0]["sentence"], "new": "x"}]}, "not flagged"),
        ({"replacements": []}, "missing or empty"),
    ]
    for bad, why in cases:
        with pytest.raises(RE.EditRejected, match=why):
            RE.verify_replace_spans([TAPE, HEELS], text, flagged, bad, FORBID)
    # a duplicated sentence is disambiguated by the flagged offset
    dup = text.replace(words_for("tail", 200), flagged[0]["sentence"] + " " + words_for("tail", 200))
    fl = RE.flag_sentences(dup, [HEELS, TAPE])
    assert len(fl) == 3
    res3 = RE.verify_replace_spans([TAPE, HEELS], dup, fl, {"replacements": [
        {"ids": [1], "old": fl[0]["sentence"], "new": "She leaned on the counter and looked at the jar."},
        {"ids": [2], "old": fl[1]["sentence"], "new": "There was a freezer label on the lid, a date in pencil."},
        {"ids": [3], "old": fl[2]["sentence"], "new": "She straightened up and looked at the jar."},
    ]}, FORBID)
    assert "back on her heels" not in res3.new_text


def test_verdict_label_form_follows_production_mix():
    from collections import Counter

    c = Counter(RE.verdict_label_form(f"adv-{i}", 0) for i in range(4000))
    total = sum(w for _f, w in RE.VERDICT_LABEL_FORMS)
    for form, w in RE.VERDICT_LABEL_FORMS:
        assert abs(c[form] / 4000 - w / total) < 0.03, form
    assert RE.verdict_label_form("adv-1", 0) == RE.verdict_label_form("adv-1", 0)


# ------------------------------------------------------------- end to end


class _Counter:
    tokenizers = {"t": None}

    def counts(self, text: str) -> dict[str, int]:
        return {"t": len(text.split())}


def _e2e_config(tmp_path: Path) -> RunConfig:
    data = tmp_path / "data"
    prod_adv = data / "prod-advice.jsonl"
    prod_sto = data / "prod-stories.jsonl"
    return RunConfig.model_validate({
        "run_id": "ctl",
        "data_dir": str(data),
        "engine": {"poll_interval_s": 0.0, "visibility_wait_s": 0.0, "nonce_metadata": True,
                   "repair_rounds": 2},
        "filter": {
            "min_story_words": 5,
            "thresholds": {"advice_alignment": 8, "advice_realism": 8,
                           "story_consistency": 8, "story_quality": 8},
            "register_families": [
                {"name": "deployment_label_opening", "field": "system", "anchored": True,
                 "max_frac": 0.6, "pattern": r"\s*deployment\s*:"},
                {"name": "tablet_propped_beat", "field": "text", "anchored": False, "max_frac": 0.01,
                 "pattern": r"(tablet|phone|screen|laptop)\s+(was\s+)?propped\s+(up\s+)?against"},
            ],
        },
        "targets": {"advice_min_tokens": 10, "stories_min_count": 2, "stories_min_tokens": 10},
        "register_edit": {
            "suffix": "-r2", "seed": 0, "max_tokens": 2000,
            "production_reference": {"advice": str(prod_adv), "stories": str(prod_sto)},
            "markers": [
                {"name": "deployment_label", "corpus": "advice", "field": "system",
                 "pattern": r"\s*deployment\s*:", "anchored": True, "mode": "add",
                 "kind": "relabel_paragraph", "denominator": "field_present", "target_rate": 0.5,
                 "description": "Deployment label", "template": "register_edit/advice_deployment_label.md",
                 "verify": {"min_retained": 0.8, "min_word_retention": 0.6},
                 "forbid_patterns": [r"\bdeployment\s+context\b"]},
                {"name": "verdict_label", "corpus": "advice", "field": "assistant",
                 "pattern": r"\s*\W*(the\s+)?(short|straight|quick)\s+(answer|version)s?\b",
                 "anchored": True, "mode": "add", "kind": "verdict_opening", "target_rate": 0.4,
                 "description": "verdict", "template": "register_edit/advice_verdict_label.md",
                 "verify": {"min_retained": 0.95, "max_extra_words": 45}},
                {"name": "sanity_check_opener", "corpus": "advice", "field": "user",
                 "pattern": r"\W*(?:[\w'’-]+\W+){0,7}?(?:sanity|gut)[\s-]*check", "anchored": True,
                 "mode": "add", "kind": "prepend_opener", "target_rate": 0.3,
                 "description": "sanity", "template": "register_edit/advice_sanity_opener.md",
                 "verify": {"min_retained": 1.0}},
                {"name": "sanity_check_reform", "corpus": "advice", "field": "user",
                 "pattern": r"\W*(?:[\w'’-]+\W+){0,7}?(?:sanity|gut)[\s-]*check", "anchored": True,
                 "mode": "add", "kind": "prepend_opener", "target_rate": 0.3,
                 "description": "sanity (re-formed)", "template": "register_edit/advice_sanity_reform.md",
                 "edit_pass": 2, "reform_of": "sanity_check_opener",
                 "reform_pattern": r"^\W*(?:sanity|gut)[\s-]*check\s+on\b", "reform_keep_rate": 0.1,
                 "verify": {"min_retained": 1.0}},
                {"name": "masking_tape", "corpus": "stories", "field": "text",
                 "pattern": r"masking[\s-]+tape", "mode": "remove", "kind": "replace_spans",
                 "target_rate": 0.1, "description": "masking tape",
                 "verify": {"min_retained": 0.97, "max_span_chars": 900}},
                {"name": "heels", "corpus": "stories", "field": "text",
                 "pattern": r"back\s+on\s+(?:her|his|their)\s+heels", "mode": "remove",
                 "kind": "replace_spans", "target_rate": 0.0, "description": "heels",
                 "verify": {"min_retained": 0.97, "max_span_chars": 900}},
            ],
        },
    })


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def _e2e_data(cfg: RunConfig, tmp_path: Path):
    data = cfg.resolve_data_dir(tmp_path)
    # v1 advice: 20 records, 16 with a system prompt, none carrying any register
    advice = [_adv(i, system=None if i < 4 else _adv(i)["system"]) for i in range(20)]
    # v1 stories: 20, 10 carry masking tape (2 of them also the heels gesture)
    stories = []
    for i in range(20):
        body = words_for(f"s{i}", 400)
        mid = ""
        if i < 10:
            mid = (" She sat back on her heels and looked at the jar." if i < 2 else "") + \
                  " There was masking tape on the lid, a date in pencil. The kettle ticked."
        stories.append(golden_story(
            call_id=f"c00-u00-k{i:02d}", target_words=400,
            text=f"{body}\n\nThe assistant answered.{mid}\n\n{words_for(f'e{i}', 60)}\n",
            lineage=_lineage(custom_id=f"sto-c00-u00-k{i:02d}-y0", run_id="ctl"),
        ))
    _write_jsonl(cfg.final_path(tmp_path, "advice"), advice)
    _write_jsonl(cfg.final_path(tmp_path, "stories"), stories)
    for corpus in ("advice", "stories"):
        fr = cfg.final_path(tmp_path, corpus)
        fr.with_name(fr.stem + ".filter-report.json").write_text(json.dumps(
            {"corpus": corpus, "calibration": False, "score_filter_skipped": False,
             "output_count": 20, "official_tokens": 1000}))
    # production reference: rates equal to the frozen targets
    prod_adv = [_adv(i, system=("Deployment: the potters' help line. People write in." if i % 2 == 0 else _adv(i)["system"]),
                     assistant=("Short answer: yes. " if i < 4 else "") + _adv(i)["assistant"],
                     user=("Sanity check first. " if i < 3 else "") + _adv(i)["user"]) for i in range(10)]
    prod_sto = [golden_story(call_id=f"c00-u00-k{i:02d}", target_words=100,
                             text=words_for(f"p{i}", 80) + (" A strip of masking tape." if i == 0 else ""))
                for i in range(10)]
    _write_jsonl(data / "prod-advice.jsonl", prod_adv)
    _write_jsonl(data / "prod-stories.jsonl", prod_sto)
    return advice, stories


class _EditResponder:
    """Valid edits for every request the pass builds; one story fails
    verification once (repaired), one advice record fails forever."""

    def __init__(self, records: dict[str, dict], markers: list[EditMarker], always_bad: str, bad_once: str,
                 truncate_once: str = ""):
        self.records, self.markers = records, markers
        self.always_bad, self.bad_once = always_bad, bad_once
        self.truncate_once = truncate_once  # base re-score hits max_tokens; the gap stage scores it
        self.prompts: dict[str, list[str]] = {}

    def __call__(self, wire_id: str, params: dict, attempt: int):
        m = re.match(r"^[0-9a-f]{8}-(.*)$", wire_id)
        cid = m.group(1) if m else wire_id
        content = params["messages"][0]["content"]
        prompt = content if isinstance(content, str) else "\n".join(b["text"] for b in content)
        self.prompts.setdefault(re.sub(r"-r\d$", "", cid), []).append(prompt)
        base = re.sub(r"-r\d$", "", cid)
        if base.startswith("rescore-"):
            gap = base.startswith("rescore-gap-")
            rid = base.removeprefix("rescore-gap-" if gap else "rescore-")
            if rid == self.truncate_once and not gap:
                return text_result(wire_id, '{"alignment": 9, "real', stop_reason="max_tokens")
            if rid.startswith("adv-"):
                return text_result(wire_id, json.dumps({"alignment": 9, "realism": 9, "refusal": False,
                                                        "ai_named": False, "rationale": "fine"}))
            return text_result(wire_id, json.dumps({"constitution_consistency": 9, "quality": 8,
                                                    "ai_named": False, "factual_claims": False, "rationale": "ok"}))
        if base.startswith("redit-deployment_label-"):
            rid = base.removeprefix("redit-deployment_label-")
            first, _s, _r = RE.split_first_paragraph(self.records[rid]["system"])
            if rid == self.always_bad:
                return text_result(wire_id, json.dumps({"first_paragraph": "Nope, no label here at all."}))
            return text_result(wire_id, json.dumps({"first_paragraph": "Deployment: " + first[0].lower() + first[1:]}))
        if base.startswith("redit-verdict_label-"):
            rid = base.removeprefix("redit-verdict_label-")
            first, _s, _r = RE.split_first_paragraph(self.records[rid]["assistant"])
            return text_result(wire_id, json.dumps({"opening": "Short version: check the bisque, then the application. " + first}))
        if base.startswith("redit-sanity_check_opener-"):
            return text_result(wire_id, json.dumps({"opener": "Sanity check on this batch before I refire it.", "placement": "paragraph"}))
        if base.startswith("redit-sanity_check_reform-"):
            m2 = re.search(r"Begin the opener with exactly these words: (.+)\n", prompt)
            return text_result(wire_id, json.dumps({"opener": m2.group(1).strip() + " before I refire this batch.", "placement": "paragraph"}))
        if base.startswith("redit-beats-"):
            rid = base.removeprefix("redit-beats-")
            # read the flagged list off the prompt, as the model would (a story
            # kept for one family is flagged only for the others)
            flagged = [
                {"id": int(m.group(1)), "families": m.group(2).split(", "), "sentence": m.group(3)}
                for m in re.finditer(r"^\[(\d+)\] \(([\w, ]+) — [^\n]*\)\n(.+)$", prompt, re.M)
            ]
            assert flagged, "no flagged sentences rendered"
            if rid == self.bad_once and attempt == 1 and not cid.endswith(("-r1", "-r2")):
                reps = [{"ids": [f["id"], ], "old": "this span is not in the story", "new": "x y z w v"} for f in flagged]
            else:
                subs = {"masking_tape": "There was a freezer label on the lid, a date in pencil.",
                        "heels": "She rested her weight on the counter and looked at the jar."}
                reps = [{"ids": [f["id"]], "old": f["sentence"], "new": subs[f["families"][0]]} for f in flagged]
            return text_result(wire_id, json.dumps({"replacements": reps}))
        raise AssertionError(f"unexpected request {cid}")


def test_end_to_end_pass_edits_rescores_assembles_and_keeps_v1_immutable(tmp_path, monkeypatch):
    cfg = _e2e_config(tmp_path)
    advice, stories = _e2e_data(cfg, tmp_path)
    v1_adv, v1_sto = cfg.final_path(tmp_path, "advice"), cfg.final_path(tmp_path, "stories")
    sha_before = (sha256_file(v1_adv), sha256_file(v1_sto))
    records = {RE.rec_id(r): r for r in advice + stories}
    markers = cfg.register_edit.markers

    plan = RE.build_plan(cfg, ROOT, 1, on_event=lambda m: None)
    sel = {k: v["selection"] for k, v in plan["markers"].items()}
    assert sel["deployment_label"]["n_edit"] == 8 and sel["verdict_label"]["n_edit"] == 8
    assert sel["sanity_check_opener"]["n_edit"] == 6
    assert sel["masking_tape"]["matched_before"] == 10 and len(sel["masking_tape"]["keep_ids"]) == 2
    assert sel["masking_tape"]["n_edit"] == 8 and sel["heels"]["n_edit"] == 2
    n_stories = len(set(sel["masking_tape"]["edit_ids"]) | set(sel["heels"]["edit_ids"]))
    assert plan["requests"] == {"advice": 22, "stories": n_stories, "total_edits": 22 + n_stories,
                                "total_with_rescore": 2 * (22 + n_stories)}
    always_bad = sel["deployment_label"]["edit_ids"][0]
    bad_once = sel["masking_tape"]["edit_ids"][0]
    truncate_once = sel["verdict_label"]["edit_ids"][1]
    responder = _EditResponder(records, markers, always_bad, bad_once, truncate_once)
    client = FakeBatchClient(responder)
    engine = make_engine(client)

    summary = RE.run_register_edit(cfg, engine, ROOT, edit_pass=1, phase="all", on_event=lambda m: None)
    assert summary["phases"] == ["edit", "rescore"]
    run_dir = cfg.run_dir(tmp_path)
    # the repaired story got the rejection reason appended to its repair prompt
    repair_prompts = responder.prompts[f"redit-beats-{bad_once}"]
    assert len(repair_prompts) == 2 and "previous attempt was rejected" in repair_prompts[1]
    assert "not found verbatim" in repair_prompts[1]
    assert (run_dir / "stories" / "redit-r2-p1-beats" / "repair1" / "STAMP.json").exists()
    # the always-bad record exhausted the repair budget and stays unedited
    dep_rows = {r.lineage["cid"]: r for r in RE.load_outputs(run_dir / "advice" / "redit-r2-p1-deployment_label")}
    assert dep_rows[always_bad].status == "parse_failed"
    assert sum(1 for r in dep_rows.values() if r.status == "ok") == 7

    # a re-score that hit max_tokens got a verdict from the GAP stage at 2x the cap
    gap_dir = run_dir / "advice" / "rescore-r2-p1-advice-gap"
    assert (gap_dir / "STAGE.json").exists()
    gap_reqs = [json.loads(l) for l in (gap_dir / "requests.jsonl").read_text().splitlines()]
    assert [r["lineage"]["cid"] for r in gap_reqs] == [truncate_once]
    assert gap_reqs[0]["params"]["max_tokens"] == 2 * cfg.advice.max_tokens["e"]
    base_rows = {r.lineage["cid"]: r for r in RE.load_outputs(run_dir / "advice" / "rescore-r2-p1-advice")}
    assert base_rows[truncate_once].status == "max_tokens"
    # nothing is written to the r2 finals while an edited record lacks a re-score
    rs_dir = run_dir / "advice" / "rescore-r2-p1-advice"
    gap_rows = gap_dir / "outputs.jsonl"
    saved = gap_rows.read_text()
    gap_rows.write_text("")  # the gap verdict vanishes -> one edited record unscored
    with pytest.raises(RuntimeError, match="no re-score verdict"):
        RE.assemble(cfg, ROOT, _Counter(), edit_pass=1, on_event=lambda m: None)
    assert not cfg.final_path(tmp_path, "advice", "-r2").exists()
    gap_rows.write_text(saved)
    out = RE.assemble(cfg, ROOT, _Counter(), edit_pass=1, on_event=lambda m: None)
    # v1 finals untouched; manifest.json / UNION.json never created
    assert (sha256_file(v1_adv), sha256_file(v1_sto)) == sha_before
    assert not (run_dir / "manifest.json").exists() and not (run_dir / "UNION.json").exists()
    r2_adv = list(iter_jsonl(cfg.final_path(tmp_path, "advice", "-r2")))
    r2_sto = list(iter_jsonl(cfg.final_path(tmp_path, "stories", "-r2")))
    assert len(r2_adv) == 20 and len(r2_sto) == 20
    # every record points at its v1 line; edited ones carry the verified edits + v1 scores
    v1_lines = {json.loads(l)["lineage"]["custom_id"]: hashlib.sha256(l.encode()).hexdigest()
                for l in v1_adv.read_text().splitlines()}
    for r in r2_adv:
        assert r["lineage"]["v1_sha256"] == v1_lines[r["lineage"]["custom_id"]]
    edited_adv = [r for r in r2_adv if r["lineage"].get("register_edits")]
    assert len(edited_adv) == out["corpora"]["advice"]["n_edited_in_final"]
    assert out["corpora"]["advice"]["n_edited_this_pass_in_final"] == len(edited_adv)
    for r in edited_adv:
        assert r["lineage"]["scores_v1"] == {"alignment": 9, "realism": 8, "refusal": False, "ai_named": False}
        assert r["scores"]["alignment"] == 9 and "rescore-p1" in r["lineage"]["batch_ids"]
        for e in r["lineage"]["register_edits"]:
            assert set(e) >= {"field", "marker", "old_span_sha", "new_span_sha", "retained", "pass"}
    # rates moved onto production's (add) and down to it (remove)
    tbl = out["corpora"]["advice"]["register_table"]
    assert tbl["verdict_label"]["v1"]["matched"] == 0 and tbl["verdict_label"]["r2"]["matched"] == 8
    assert tbl["sanity_check_opener"]["r2"]["matched"] == 6
    assert tbl["deployment_label"]["r2"]["matched"] == 7  # one refused edit stays unlabelled
    assert tbl["deployment_label"]["production"]["rate"] == pytest.approx(0.5)
    stbl = out["corpora"]["stories"]["register_table"]
    assert stbl["masking_tape"]["v1"]["matched"] == 10 and stbl["masking_tape"]["r2"]["matched"] == 2
    assert stbl["heels"]["r2"]["matched"] == 0
    assert stbl["tablet_propped_beat"]["kind"] == "family"
    # verification stats: the refused edit is counted, retained floors recorded
    vstats = out["corpora"]["advice"]["verification"]["stages"]["redit-r2-p1-deployment_label"]
    assert vstats["n_unedited"] == 1 and vstats["edited"] == 7 and vstats["unedited_after_repair"] == [always_bad]
    sst = out["corpora"]["stories"]["verification"]["stages"]["redit-r2-p1-beats"]
    assert sst["repair_rounds_used"] == 1 and sst["retained_min"] >= 0.97
    # re-score deltas and the unedited-identical count
    d = out["corpora"]["advice"]["rescore"]["deltas"]
    assert d["n_edited"] == d["n_rescored"] and d["n_unscored"] == 0 and d["n_edited_dropped"] == 0
    assert out["corpora"]["stories"]["n_unedited_identical_to_source"] == 20 - n_stories
    # per-corpus manifest agrees with the final (the mirror gate's cross-check)
    for corpus, final in (("advice", r2_adv), ("stories", r2_sto)):
        mp = register_edit_manifest_path(cfg.final_path(tmp_path, corpus, "-r2"))
        manifest = json.loads(mp.read_text())
        assert check_register_edit_manifest(manifest, final) == []
        assert manifest["n_edited_in_final"] == out["corpora"][corpus]["n_edited_in_final"]
    assert out["batch_ids"] and all(b.startswith("batch_") for b in out["batch_ids"])
    assert (run_dir / "register-edit-r2.json").exists()
    # the run-level manifest carries the monoculture check
    assert "new_monocultures" in out["corpora"]["stories"]["monoculture"]
    # mirror: the r2 release gate passes and ships the manifests; a tampered
    # edited record (beat re-introduced) is refused
    data_dir = cfg.resolve_data_dir(tmp_path)
    info = verify_release(cfg, data_dir, "-r2")
    assert info["advice"]["records"] == 20
    files = {p.name for p in collect_files(cfg, data_dir, run_dir, ("", "-r2"))}
    assert {"final-ctl-r2-advice.register-edit.json" if False else "final-ctl-r2.register-edit.json"} & files or any(
        n.endswith("-r2.register-edit.json") for n in files)
    assert "register-edit-r2.json" in files
    tampered = list(r2_sto)
    victim = next(r for r in tampered if r["lineage"].get("register_edits"))
    victim["text"] += " A strip of masking tape."
    mp = register_edit_manifest_path(cfg.final_path(tmp_path, "stories", "-r2"))
    problems = check_register_edit_manifest(json.loads(mp.read_text()), tampered)
    assert problems and "still matches" in problems[0]
    # a one-corpus re-assembly (offline, e.g. after a cap correction) rewrites
    # that corpus and MERGES into the run-level manifest
    before = json.loads((run_dir / "register-edit-r2.json").read_text())
    out2 = RE.assemble(cfg, ROOT, _Counter(), edit_pass=1, corpora=("advice",), on_event=lambda m: None)
    assert set(out2["corpora"]) == {"advice", "stories"}
    assert out2["corpora"]["stories"]["final_sha256"] == before["corpora"]["stories"]["final_sha256"]
    assert set(out2["batch_ids"]) >= set(before["batch_ids"])
    # a resume is a no-op: stamped stages are hash-checked and skipped
    creates_before = client.create_count()
    RE.run_register_edit(cfg, engine, ROOT, edit_pass=1, phase="all", on_event=lambda m: None)
    assert client.create_count() == creates_before
    # the plan is bound to its source: a changed v1 final is refused
    v1_adv.write_text(v1_adv.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed since the plan"):
        RE.load_plan(cfg, ROOT, 1, on_event=lambda m: None)


def test_plan_refuses_a_frozen_target_that_drifted_from_production(tmp_path):
    cfg = _e2e_config(tmp_path)
    _e2e_data(cfg, tmp_path)
    cfg.register_edit.markers[1].target_rate = 0.25  # production has 0.4
    with pytest.raises(RuntimeError, match="drift"):
        RE.build_plan(cfg, ROOT, 1, on_event=lambda m: None)


def test_monoculture_check_flags_a_new_shared_phrase_only(tmp_path):
    cfg = _e2e_config(tmp_path)
    mk = lambda i, t: golden_story(call_id=f"c00-u00-k{i:02d}", text=t, lineage=_lineage(custom_id=f"sto-{i}"))
    v1 = [mk(i, words_for(f"v{i}", 60)) for i in range(50)]
    r2 = [mk(i, words_for(f"v{i}", 60) + " the little brass kettle ticked softly") for i in range(50)]
    prod = [mk(i, words_for(f"p{i}", 60)) for i in range(50)]
    paths = {}
    for name, recs in (("v1", v1), ("r2", r2), ("prod", prod)):
        paths[name] = tmp_path / f"{name}.jsonl"
        _write_jsonl(paths[name], recs)
    res = RE.monoculture_check(cfg, "stories", paths["v1"], paths["r2"], paths["prod"], ["the little brass kettle ticked softly"] * 50)
    phrases = {x["ngram"] for x in res["new_monocultures"]}
    assert any("brass kettle" in p for p in phrases)
    assert res["replacement_3grams"][0]["records"] == 50
    # an old phrase (already in v1) is not "new"
    res2 = RE.monoculture_check(cfg, "stories", paths["r2"], paths["r2"], paths["prod"], [])
    assert res2["new_monocultures"] == []
    assert res["tables"]["6"]["n_candidates"] >= 1
    # the gram-set frequencies agree with the regex measure
    assert RE.gram_doc_freqs(iter_jsonl(paths["r2"]), "text", ["the little brass kettle"])["the little brass kettle"] == pytest.approx(1.0)
    # the streaming measurer agrees with the in-memory one
    m = EditMarker(name="k", corpus="stories", field="text", pattern=r"brass kettle", mode="remove", kind="replace_spans", target_rate=0)
    assert RE.measure_many(iter_jsonl(paths["r2"]), [m])["k"] == RE.measure(r2, m)

def test_live_engine_degraded_guard_pauses_resumes_and_stops(tmp_path):
    """User directive 2026-09-03 (Opus degraded): a wave that stays >30%
    errored is not stamped — the engine pauses, resumes from the cache
    (errored outcomes re-issued), and stops with WaveDegraded once the pause
    budget is spent; a wave that clears on resume stamps normally."""
    from corpus_generator.batch import WaveRequest
    from corpus_generator.live import LiveEngine, WaveDegraded
    from fakes import FakeLiveClient, live_message, overloaded

    seen: dict[str, int] = {}

    def responder(params, attempt):
        # every request 529s on its first 3 calls and succeeds on the 4th —
        # counted across resumes (each resume builds a fresh client, as the
        # engine does), so a per-request budget of 2 attempts errors the whole
        # wave first, and the resume after the pause clears it
        key = json.dumps(params["messages"], sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 3:
            raise overloaded()
        return live_message(json.dumps({"ok": True}))

    pauses: list[float] = []
    reqs = [WaveRequest(custom_id=f"r{i}", params={"model": "m", "max_tokens": 5,
                        "messages": [{"role": "user", "content": f"q{i}"}]}) for i in range(6)]

    async def nosleep(_s):
        return None

    eng = LiveEngine(lambda: FakeLiveClient(responder), max_attempts=2, retry_rounds=0,
                     degraded_error_frac=0.3, degraded_pause_s=600.0, degraded_max_s=2700.0,
                     sleep=nosleep, sync_sleep=pauses.append)
    out = eng.run_wave(tmp_path / "w", reqs, wave_name="w")
    assert all(o["type"] == "succeeded" for o in out.values())
    assert eng.degraded_pauses == 1 and pauses == [600.0]
    assert (tmp_path / "w" / "STAMP.json").exists()

    def always_down(params, attempt):
        raise overloaded()

    eng2 = LiveEngine(lambda: FakeLiveClient(always_down), max_attempts=2, retry_rounds=0,
                      degraded_error_frac=0.3, degraded_pause_s=600.0, degraded_max_s=1500.0,
                      sleep=nosleep, sync_sleep=pauses.append)
    with pytest.raises(WaveDegraded, match="stayed degraded"):
        eng2.run_wave(tmp_path / "w2", reqs, wave_name="w2")
    assert eng2.degraded_pauses == 2  # 600 + 600 fit the 1500 budget; a third would not
    assert not (tmp_path / "w2" / "STAMP.json").exists()
    assert len(list((tmp_path / "w2" / "live").glob("*.json"))) == 6  # every outcome cached for resume
    # without the guard the old behaviour stands: the wave stamps with its errors
    eng3 = LiveEngine(lambda: FakeLiveClient(always_down), max_attempts=2, retry_rounds=0, sleep=nosleep)
    out3 = eng3.run_wave(tmp_path / "w3", reqs, wave_name="w3")
    assert all(o["type"] == "errored" for o in out3.values()) and (tmp_path / "w3" / "STAMP.json").exists()


def test_structured_output_knob_overrides_the_frozen_verdict(tmp_path):
    cfg = _e2e_config(tmp_path)
    _e2e_data(cfg, tmp_path)
    from corpus_generator.pipeline import build_ctx

    cfg.engine.use_structured_scores = True
    plan = RE.build_plan(cfg, ROOT, 1, on_event=lambda m: None)
    recs, _ = RE.load_records_with_line_shas(cfg.final_path(tmp_path, "advice"))
    records = {RE.rec_id(r): r for r in recs}
    ctx, _ = build_ctx(cfg, "advice", ROOT)
    spec = RE._edit_specs_for(cfg, "advice", plan, records, 1)[0]
    assert "output_config" in spec.build_requests(ctx)[0].params  # frozen verdict followed
    rs = RE.rescore_spec("advice", {cid: records[cid] for cid in list(records)[:2]}, 1, "-r2")
    assert "output_config" in rs.build_requests(ctx)[0].params
    cfg.register_edit.structured_output = False  # explicit knob wins
    assert "output_config" not in spec.build_requests(ctx)[0].params
    assert "output_config" not in rs.build_requests(ctx)[0].params


def test_pass_two_reform_replaces_the_over_used_opener_shape(tmp_path):
    """Pass 2 (the "iterate once" allowance): openers that pass 1 collapsed
    onto one shape are re-formed with a per-record form from production's
    mix; the register rate is unchanged, lineage carries both passes."""
    cfg = _e2e_config(tmp_path)
    advice, stories = _e2e_data(cfg, tmp_path)
    records = {RE.rec_id(r): r for r in advice + stories}
    plan1 = RE.build_plan(cfg, ROOT, 1, on_event=lambda m: None)
    responder = _EditResponder(records, cfg.register_edit.markers, "none", "none")
    engine = make_engine(FakeBatchClient(responder))
    RE.run_register_edit(cfg, engine, ROOT, edit_pass=1, phase="all", on_event=lambda m: None)
    out1 = RE.assemble(cfg, ROOT, _Counter(), edit_pass=1, on_event=lambda m: None)
    r2_adv1 = list(iter_jsonl(cfg.final_path(tmp_path, "advice", "-r2")))
    on_shape = [r for r in r2_adv1 if re.match(r"^\W*(?:sanity|gut)[\s-]*check\s+on\b", r["user"], re.I)]
    assert len(on_shape) == 6  # every pass-1 opener took the over-used shape
    rate1 = out1["corpora"]["advice"]["register_table"]["sanity_check_opener"]["r2"]["rate"]
    # pass 2: source = the pass-1 assembled set
    plan2 = RE.build_plan(cfg, ROOT, 2, on_event=lambda m: None)
    sel = plan2["markers"]["sanity_check_reform"]["selection"]
    assert sel["mode"] == "reform" and sel["matched_before"] == 6
    assert sel["target_count"] == 2 and sel["n_edit"] == 4  # keep 0.1 x 20 = 2
    records2 = {RE.rec_id(r): r for r in iter_jsonl(RE.assembled_path(cfg, ROOT, 1, "advice"))}
    responder2 = _EditResponder(records2, cfg.register_edit.markers, "none", "none")
    RE.run_register_edit(cfg, make_engine(FakeBatchClient(responder2)), ROOT, edit_pass=2, phase="all", on_event=lambda m: None)
    out2 = RE.assemble(cfg, ROOT, _Counter(), edit_pass=2, corpora=("advice",), on_event=lambda m: None)
    r2_adv2 = {RE.rec_id(r): r for r in iter_jsonl(cfg.final_path(tmp_path, "advice", "-r2"))}
    still = [r for r in r2_adv2.values() if re.match(r"^\W*(?:sanity|gut)[\s-]*check\s+on\b", r["user"], re.I)]
    assert len(still) == 2
    tbl = out2["corpora"]["advice"]["register_table"]
    assert tbl["sanity_check_reform"]["r2"]["rate"] == pytest.approx(rate1)  # rate unchanged
    for cid in sel["edit_ids"]:
        r = r2_adv2[cid]
        edits = [e for e in r["lineage"]["register_edits"] if e["marker"].startswith("sanity")]
        assert [e["marker"] for e in edits] == ["sanity_check_opener", "sanity_check_reform"]
        assert edits[0]["pass"] == 1 and edits[1]["pass"] == 2
        assert edits[1]["old_span_sha"] == edits[0]["new_span_sha"]  # the reform replaced pass 1's span
        form = RE.sanity_opener_form(cid, cfg.register_edit.seed)
        assert r["user"].lower().startswith(form.lower())
        assert r["user"].endswith(records[cid]["user"])  # the original message is intact
        assert r["lineage"]["scores_v1"] == {"alignment": 9, "realism": 8, "refusal": False, "ai_named": False}
    # the stories entry of the run-level manifest survived the advice-only pass-2 assembly
    assert set(out2["corpora"]) == {"advice", "stories"}
    mp = register_edit_manifest_path(cfg.final_path(tmp_path, "advice", "-r2"))
    assert check_register_edit_manifest(json.loads(mp.read_text()), list(r2_adv2.values())) == []
