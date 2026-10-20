"""G-PARSER — the real-completion parser certification:
the ACTUAL scoring path — ``think_split.split_think`` under the
family's render-fact mode, then the judge-free ``expression_exact_match_sympy`` extractor through
``inspect_served._math_scorer`` — run on the DIAGNOSTIC smoke's real completions committed under
``contracts.THINKING_SMOKE_COMPLETIONS_DIR/<family>/`` (``smoke_completions.jsonl`` beside the
smoke's ``smoke_report.json``).

What certifies a harness-side split (a family with NO server parser): a fact may read ``certified``
ONLY when its completions are committed and pass every check here; a family still ``candidate``
whose smoke has not run is skipped BY NAME (never silently green). Per family the three censoring
cases are required: (a) reasoning finished + answer finished (``stop``) -> present, not
unterminated, the answer text free of both tags; (b) reasoning truncated at the cap with no closing
tag -> present, UNTERMINATED, ``answer_text == ""`` -> NO score under the primary policy, INCORRECT
under the sensitivity branch; (c) reasoning closed, answer truncated (``length`` AFTER the close
tag) -> scored as-is. A censoring case the smoke did not produce is taken from the synthetic fixture
built from the family's real tags (``tests/fixtures/think_split/<family>.json``) and the test
RECORDS that it did (``record_property("synthetic_case", ...)``).

The committed completions are model outputs on public MATH-500 items; nothing here is personal data.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("inspect_evals")

from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tiers import think_split as ts
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO / C.THINKING_SMOKE_COMPLETIONS_DIR
FIXTURES = REPO / "tests" / "fixtures" / "think_split"
#: the families THIS experiment certifies harness-side: gemma (server parser) and qwen
#: (the parent's implicit_open certification) were certified before it
G_PARSER_FAMILIES = ("glm4v", "nemotron_h")
#: the censoring cases the fixture set must cover per family — real rows first, synthetic fallback
REQUIRED_CASES = {"glm4v": ("well_formed", "reasoning_truncated", "answer_truncated", "tag_free"),
                  "nemotron_h": ("well_formed", "reasoning_truncated", "answer_truncated")}
#: real completions whose MATH-500 target is known by eye (public items): the split's answer, through the
#: real extractor, scores CORRECT — the end-to-end reading of "the answer text is what the scorer reads"
KNOWN_TARGETS = {"glm4v": {"math_11e6a957": "42", "math_7f9e23c9": "6+9i", "math_5e43f1cd": "9", "math_240ca4d8": "\\text{Evelyn}",
                           # these four answers are wrapped in \(\) or GLM's answer box and scored INCORRECT against
                           # their correct targets before the math_delimiters normalization — pinned CORRECT through it
                           "math_403b3454": "5", "math_d959374f": "-50", "math_67c8ff7f": "3\\sqrt{13}", "math_d5de6a08": "\\sqrt{51}"},
                 # nemotron_h: the dataset's `answer` field verbatim (HuggingFaceH4/MATH-500 test split, matched by
                 # `contracts.math500_sample_id(problem)` — the smoke's 24 items are the first 24 in dataset order) for
                 # every well-formed smoke row that scores CORRECT through the real path: 16 of 18. NONE owes it to the
                 # math_delimiters normalization — the two rows where it fired (math_8230401f `(3, \(\frac{\pi}{2}\))` and
                 # math_d5de6a08 `\[\text{ANSWER: } \sqrt{51}\]`) are mathematically right but still INCORRECT to the
                 # extractor, and are deliberately NOT pinned (a pin would codify the extractor's miss)
                 "nemotron_h": {"math_11e6a957": "42", "math_1a809667": "\\pi", "math_240ca4d8": "\\text{Evelyn}",
                                "math_2d9f4473": "\\frac{14}{3}", "math_555caaf5": "6 - 5i", "math_5e43f1cd": "9",
                                "math_67c8ff7f": "3\\sqrt{13}", "math_7f9e23c9": "6+9i", "math_805d659b": "x=5",
                                "math_941bd991": "27", "math_a0de4994": "5", "math_a4838f9f": "2220",
                                "math_ab1a6003": "284", "math_d959374f": "-50", "math_e5ac5620": "90^\\circ",
                                "math_faf5031d": "3"}}
FIXTURE_CASE_NAME = {"well_formed": "well_formed", "reasoning_truncated": "unterminated",
                     "answer_truncated": None, "tag_free": "tag_free_is_the_answer"}


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _fact(family: str) -> C.RenderFact:
    return C.FAMILY_RENDER_FACTS[family]


def _certified(family: str) -> bool:
    return _fact(family).evidence_status == C.RENDER_EVIDENCE_CERTIFIED


def _smoke_dir(family: str) -> Path:
    return SMOKE_ROOT / family


def _rows_or_skip(family: str) -> list[dict]:
    """The committed completions, or a skip BY NAME for a candidate family whose smoke has not run.
    A CERTIFIED family with nothing committed is a failure, never a skip."""
    path = _smoke_dir(family) / "smoke_completions.jsonl"
    if not path.is_file():
        if _certified(family):
            pytest.fail(f"{family}: the render fact reads certified but no smoke completions are committed at "
                        f"{path.relative_to(REPO)} — a certification without its evidence")
        pytest.skip(f"{family}: render fact still candidate and no committed smoke completions at "
                    f"{path.relative_to(REPO)} — its diagnostic smoke has not run")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows, path
    return rows


def _report(family: str) -> dict:
    return json.loads((_smoke_dir(family) / "smoke_report.json").read_text())


def _classify(row: dict) -> str:
    """The censoring case of one ON row, from ``finish_reason`` and the tags ALONE (independent of
    ``split_think``, which the tests then check against this reading)."""
    content, finish = row["content"] or "", row["finish_reason"]
    has_open, has_close = ts.THINK_OPEN in content, ts.THINK_CLOSE in content
    if finish == "length":
        return "answer_truncated" if has_close else "reasoning_truncated"
    if finish == "stop" and not has_open and not has_close:
        return "tag_free"
    if finish == "stop" and has_close:
        return "well_formed"
    return f"other({finish}, open={has_open}, close={has_close})"


def _output(content: str, *, stop_reason: str):
    from inspect_ai._util.content import ContentText
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    msg = ChatMessageAssistant(content=[ContentText(text=content)])
    return ModelOutput(model="tinycap/x", choices=[ChatCompletionChoice(message=msg, stop_reason=stop_reason)])


def _state(output, *, target: str):
    from inspect_ai.model import ModelName
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    return TaskState(model=ModelName("tinycap/x"), sample_id="math_00000001", epoch=1,
                     input="q", messages=[], target=Target(target), output=output)


def _score(scorer, content: str, *, finish_reason: str, target: str = "0"):
    from inspect_ai.scorer import Target

    out = _output(content, stop_reason=isv.STOP_REASON_AT_CAP if finish_reason == "length" else "stop")
    return asyncio.run(scorer(_state(out, target=target), Target(target)))


def _is_unscored(score) -> bool:
    return isinstance(score.value, float) and math.isnan(score.value)


def _fixture_case(family: str, case: str) -> dict | None:
    name = FIXTURE_CASE_NAME.get(case)
    if name is None:
        return None
    doc = json.loads((FIXTURES / f"{family}.json").read_text())
    return next((c for c in doc["cases"] if c["name"] == name), None)


# --------------------------------------------------------------------------- #
# a certified fact has its evidence committed; the evidence names the smoke     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", G_PARSER_FAMILIES)
def test_a_certified_harness_side_fact_has_its_smoke_committed_and_cites_it(family):
    """Runs for EVERY G-PARSER family, certified or not — the guard that a fact cannot read
    ``certified`` ahead of its evidence. A candidate family asserts only that its evidence says so."""
    fact = _fact(family)
    assert fact.server_parser is None and fact.split_mode in C.THINK_SPLIT_MODES
    path = _smoke_dir(family) / "smoke_completions.jsonl"
    if not _certified(family):
        assert "CANDIDATE" in fact.evidence and "G-PARSER" in fact.evidence
        return
    assert path.is_file(), f"{family}: certified without committed completions at {path.relative_to(REPO)}"
    assert (_smoke_dir(family) / "smoke_report.json").is_file()
    # the evidence names the committed jsonl, the certification gate, a smoke job id and the date
    assert str(path.relative_to(REPO)) in fact.evidence and "G-PARSER" in fact.evidence
    assert "smoke job" in fact.evidence and "CERTIFIED 2026-" in fact.evidence
    assert C.render_fact_certified(fact) and C.render_fact_refusal(family) is None
    assert C.check_render_fact_certified(family) == fact


@pytest.mark.parametrize("family", G_PARSER_FAMILIES)
def test_the_committed_smoke_is_the_diagnostic_run_of_this_family_under_the_facts_mode(family):
    rows = _rows_or_skip(family)
    fact, report = _fact(family), _report(family)
    assert report["family"] == family and report["diagnostic"] is True and report["verdict"] == "PASS"
    assert report["split_mode_candidate"] == fact.split_mode
    # predicate (A) on the base route: the served header ids under each arm EQUAL the fact by tuple
    assert tuple(report["header"]["off"]["got"]) == fact.off_ids and tuple(report["header"]["on"]["got"]) == fact.on_ids
    on, off = [r for r in rows if r["arm"] == "on"], [r for r in rows if r["arm"] == "off"]
    assert (len(on), len(off)) == (report["n_on"], report["n_off"]) and len(rows) == len(on) + len(off)
    assert sum(r["finish_reason"] == "length" for r in on) == report["n_at_cap_on"]
    model_keys = {r["model_key"] for r in rows}
    assert len(model_keys) == 1 and C.family_of_model_key(next(iter(model_keys))) == family
    for r in rows:
        assert r["diagnostic"] is True and r["family"] == family and r["error"] is None
        assert r["split_mode_candidate"] == fact.split_mode and r["cap"] == report["cap"]
        assert r["stamps"]["render_regime"] == (C.RENDER_REGIME_THINKING_ON if r["arm"] == "on"
                                                else C.RENDER_REGIME_THINKING_OFF)
        assert r["stamps"]["cell_id"] == "none"
        assert r["finish_reason"] in ("stop", "length")
    # the sampling row the smoke drew under is the model's thinking-card row (the report agrees)
    key = next(iter(model_keys))
    row = C.THINKING_SAMPLING_BY_MODEL_KEY[key]
    assert report["sampling_mode"] == C.SAMPLING_THINKING_CARD
    # the report carries exactly the row's SET parameters (a None top_k is omitted, never written as null)
    wanted = {"temperature": row.temperature, "top_p": row.top_p, "top_k": row.top_k}
    assert {k: report["sampling_params"].get(k) for k in wanted} == wanted
    assert set(report["sampling_params"]) == {k for k, v in wanted.items() if v is not None}
    assert all(r["sampling_params"] == report["sampling_params"] and r["sampling_mode"] == report["sampling_mode"]
               for r in rows)


# --------------------------------------------------------------------------- #
# the emitted shape, arm by arm, against the split and against the smoke's     #
# own shape record                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", G_PARSER_FAMILIES)
def test_every_on_completion_carries_the_family_shape_and_the_off_arm_is_tag_free(family):
    rows = _rows_or_skip(family)
    fact = _fact(family)
    mode = fact.split_mode
    for r in rows:
        content = r["content"]
        assert isinstance(content, str) and content
        # no server parser: the reasoning channel is never populated — the tags are IN the content
        assert r["reasoning_content"] in (None, ""), (family, r["item_id"])
        split = ts.split_think(content, mode=mode)
        if r["arm"] == "off":
            assert ts.THINK_OPEN not in content and ts.THINK_CLOSE not in content, r["item_id"]
            assert r["finish_reason"] == "stop"
            # the OFF arm is split under the ONE derivation's OFF mode (`contracts.think_split_mode`: `record_only`
            # whatever the fact), never under the ON fact's: the tag-free OFF completion IS the answer there
            off_mode = C.think_split_mode(arm=C.THINKING_OFF, reasoning_parser=fact.server_parser,
                                          prompt_opened_think=fact.prompt_opens_think_on)
            assert off_mode == C.THINK_SPLIT_RECORD_ONLY
            off_split = ts.split_think(content, mode=off_mode)
            assert off_split.answer_text == content and not off_split.present and not off_split.unterminated, r["item_id"]
            # and under the ON mode the same tag-free text reads as the split contract says — the answer for a completion-opened
            # family (explicit_tags), an UNTERMINATED block for a prompt-opened one (implicit_open) — which is why
            # the OFF arm's mode is derived from the ARM, never taken from the fact
            if fact.prompt_opens_think_on:
                assert split.present and split.unterminated and split.answer_text == "", r["item_id"]
            else:
                assert split.answer_text == content and not split.present and not split.unterminated, r["item_id"]
            continue
        case = _classify(r)
        assert not case.startswith("other"), (family, r["item_id"], case)
        n_open, n_close = content.count(ts.THINK_OPEN), content.count(ts.THINK_CLOSE)
        if fact.prompt_opens_think_on:
            # implicit_open: the PROMPT opened the block — a literal opening tag in the completion is stray
            assert n_open == 0 and not split.stray_open_tag, (family, r["item_id"])
        else:
            # explicit_tags: the completion emits the opening tag itself (or is tag-free = the answer)
            assert n_open == (0 if case == "tag_free" else 1), (family, r["item_id"], case)
            if n_open:
                assert content.index(ts.THINK_OPEN) == 0, (family, r["item_id"])
        assert n_close <= 1
        expect_present = case != "tag_free"
        expect_unterminated = case == "reasoning_truncated"
        assert (split.present, split.unterminated) == (expect_present, expect_unterminated), (family, r["item_id"], case)
        if case == "reasoning_truncated":
            assert split.answer_text == ""
        elif case == "tag_free":
            assert split.answer_text == content
        else:
            close = content.index(ts.THINK_CLOSE)
            assert split.answer_text == content[close + len(ts.THINK_CLOSE):].strip()
        assert ts.THINK_OPEN not in split.answer_text and ts.THINK_CLOSE not in split.answer_text
        # the smoke's own shape record (the instrument) agrees with the split it was written from
        shape = r["shape"]
        assert shape["mode"] == mode and shape["ok"] is True
        assert (shape["present"], shape["unterminated"], shape["stray_open_tag"], shape["answer_chars"]) == (
            split.present, split.unterminated, split.stray_open_tag, len(split.answer_text)), (family, r["item_id"])


# --------------------------------------------------------------------------- #
# the three censoring cases through the REAL scoring path                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", G_PARSER_FAMILIES)
def test_the_censoring_cases_score_through_the_real_math_path(family, record_property):
    """(a) scored (never unscored), (b) NO score under the primary policy and INCORRECT under the
    sensitivity branch — the (b) trap: read as an answer under the WRONG mode, the truncated
    reasoning would be scored — (c) scored as-is. A required case the smoke did not produce runs on
    the family's synthetic fixture case and is recorded as such."""
    rows = _rows_or_skip(family)
    fact = _fact(family)
    mode = fact.split_mode
    primary = isv._math_scorer(mode)
    sensitivity = ts.think_aware_scorer(isv._math_scorer(C.THINK_SPLIT_RECORD_ONLY), mode=mode,
                                        unterminated_policy=C.UNTERMINATED_INCORRECT,
                                        name=isv.PRIMARY_SCORER["math"])
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        if r["arm"] == "on":
            by_case.setdefault(_classify(r), []).append({"content": r["content"], "finish_reason": r["finish_reason"],
                                                        "item_id": r["item_id"], "synthetic": False})
    counts = {k: len(v) for k, v in sorted(by_case.items())}
    record_property("case_counts", counts)
    synthetic: list[str] = []
    for case in REQUIRED_CASES[family]:
        if by_case.get(case):
            continue
        fx = _fixture_case(family, case)
        assert fx is not None, f"{family}: case {case!r} neither in the smoke nor in the synthetic fixture"
        by_case[case] = [{"content": fx["completion"], "finish_reason": fx["finish_reason"],
                          "item_id": f"fixture:{fx['name']}", "synthetic": True}]
        synthetic.append(case)
    record_property("synthetic_case", synthetic)
    assert set(REQUIRED_CASES[family]) <= set(by_case)
    # the naturally occurring cases the plan expects at cap 1,536: (b) and (c) both present for real
    assert any(not c["synthetic"] for c in by_case["reasoning_truncated"]), (family, counts)
    assert any(not c["synthetic"] for c in by_case["answer_truncated"]), (family, counts)
    for case, items in by_case.items():
        for it in items:
            s = _score(primary, it["content"], finish_reason=it["finish_reason"])
            meta = s.metadata
            assert meta[ts.SCORER_WRAPPER_FIELD] == f"think_aware/{mode}" and meta["think_split_mode"] == mode
            if case == "reasoning_truncated":
                # (b): NO score, no answer — the reasoning is never read as the answer
                assert _is_unscored(s) and s.answer is None, (family, it["item_id"])
                assert meta["think_block_present"] is True and meta["think_block_unterminated"] is True
                s2 = _score(sensitivity, it["content"], finish_reason=it["finish_reason"])
                assert not _is_unscored(s2) and isv.score_to_value(s2) == 0.0, (family, it["item_id"])
                # the trap, made visible: under record_only the same text IS scored (as reasoning-as-answer)
                wrong = _score(isv._math_scorer(C.THINK_SPLIT_RECORD_ONLY), it["content"], finish_reason=it["finish_reason"])
                assert not _is_unscored(wrong), (family, it["item_id"])
            else:
                # (a) / (c) / tag-free: scored — never censored; metadata reads the split
                assert not _is_unscored(s), (family, it["item_id"], case)
                assert meta["think_block_unterminated"] is False
                assert meta["think_block_present"] is (case != "tag_free")


@pytest.mark.parametrize("family", G_PARSER_FAMILIES)
def test_known_targets_score_correct_through_the_split(family):
    """Public items whose target is known (by eye, or the dataset's answer field): the answer AFTER the split,
    through the real extractor, is CORRECT — so the answer text the scorer reads is the model's answer, not its
    reasoning."""
    known = KNOWN_TARGETS.get(family)
    if not known:
        pytest.skip(f"{family}: no by-eye targets recorded for this family's smoke items")
    rows = {r["item_id"]: r for r in _rows_or_skip(family) if r["arm"] == "on"}
    scorer = isv._math_scorer(_fact(family).split_mode)
    for item_id, target in known.items():
        r = rows[item_id]
        assert _classify(r) == "well_formed", (family, item_id)
        s = _score(scorer, r["content"], finish_reason=r["finish_reason"], target=target)
        assert isv.score_to_value(s) == 1.0 and not _is_unscored(s), (family, item_id)
        # and a WRONG target is not credited (the extractor is reading the answer, not matching anything)
        s_wrong = _score(scorer, r["content"], finish_reason=r["finish_reason"], target="-99999")
        assert isv.score_to_value(s_wrong) == 0.0 and not _is_unscored(s_wrong), (family, item_id)
