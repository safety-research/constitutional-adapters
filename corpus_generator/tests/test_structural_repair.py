"""P1-2: structural validation + bounded repair; root waves block rather than
shrink; partial subtree repair leaves siblings untouched."""

from __future__ import annotations

import json

import pytest

from conftest import make_engine
from corpus_generator.advice import advice_stages
from corpus_generator.stages import StageBlocked, run_stage
from fakes import PipelineResponder, error_result, text_result, words_for
from conftest import tiny_sections

CONTENT_FILTER_MSG = "Output blocked by content filtering policy"


def _underfill_wrapper(base_responder, drop: dict[str, list[str]]):
    """Wrap a responder: for base-wave requests of the named custom_ids, drop
    the listed slots from the JSON array. Repairs (-rN) pass through."""

    def responder(cid, params, attempt):
        result = base_responder(cid, params, attempt)
        if cid in drop:
            text = result.result.message.content[0].text
            items = [i for i in json.loads(text) if i["slot"] not in drop[cid]]
            return text_result(cid, json.dumps(items))
        return result

    return responder


def test_underfill_triggers_targeted_repair(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    a1, a2 = advice_stages()[0], advice_stages()[1]
    base = PipelineResponder(tiny_sections())
    fake_client.responder = _underfill_wrapper(base, {"adv-a2-p01": ["t01"]})
    engine = make_engine(fake_client)

    ctx.upstream["a1"] = run_stage(engine, ctx, a1)
    rows = run_stage(engine, ctx, a2)
    ok = [r for r in rows if r.status == "ok"]
    assert sorted(r.key for r in ok) == ["p00-t00", "p00-t01", "p01-t00", "p01-t01"]

    # exactly one repair request, for the one underfilled principle
    repair_reqs = [
        json.loads(line)
        for line in (ctx.corpus_dir / "a2" / "repair1" / "requests.jsonl")
        .read_text().splitlines()
    ]
    assert len(repair_reqs) == 1
    assert repair_reqs[0]["custom_id"] == "adv-a2-p01-r1"
    prompt = repair_reqs[0]["params"]["messages"][0]["content"]
    assert "missing slots: t01." in prompt
    # existing item is quoted so the repair can stay distinct from it
    assert "type adv-a2-p01 t00" in prompt
    stamp = json.loads((ctx.corpus_dir / "a2" / "STAGE.json").read_text())
    assert stamp["repair_rounds_used"] == 1
    assert stamp["achieved_units"] == stamp["expected_units"] == 4


def test_root_refusal_blocks_never_stamps(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    a1 = advice_stages()[0]

    def refuses(cid, params, attempt):
        return text_result(cid, "", stop_reason="refusal")

    fake_client.responder = refuses
    engine = make_engine(fake_client)
    with pytest.raises(StageBlocked, match="ROOT"):
        run_stage(engine, ctx, a1)
    assert not (ctx.corpus_dir / "a1" / "STAGE.json").exists()
    # repair budget was actually spent before blocking
    assert (ctx.corpus_dir / "a1" / "repair1").exists()
    assert (ctx.corpus_dir / "a1" / "repair2").exists()


def test_root_persistent_underfill_blocks(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    a1 = advice_stages()[0]
    base = PipelineResponder(tiny_sections())

    def always_short(cid, params, attempt):
        result = base(cid, params, attempt)
        text = result.result.message.content[0].text
        items = [i for i in json.loads(text) if i["slot"] != "p01"]
        return text_result(cid, json.dumps(items) if items else "[]")

    fake_client.responder = always_short
    engine = make_engine(fake_client)
    with pytest.raises(StageBlocked, match="1/2"):
        run_stage(engine, ctx, a1)


def test_nonroot_below_min_slots_frac_blocks(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    a1, a2 = advice_stages()[0], advice_stages()[1]
    base = PipelineResponder(tiny_sections())
    # both principles permanently missing one type slot each -> 2/4 = 0.5
    drop_all = {"adv-a2-p00": ["t01"], "adv-a2-p01": ["t01"]}

    def stubborn(cid, params, attempt):
        result = base(cid, params, attempt)
        base_cid = cid.split("-r")[0]
        if base_cid in drop_all:
            text = result.result.message.content[0].text
            items = [i for i in json.loads(text) if i["slot"] not in drop_all[base_cid]]
            return text_result(cid, json.dumps(items) if items else "[]")
        return result

    fake_client.responder = stubborn
    engine = make_engine(fake_client)
    ctx.upstream["a1"] = run_stage(engine, ctx, a1)
    with pytest.raises(StageBlocked, match="min_slots_frac"):
        run_stage(engine, ctx, a2, min_slots_frac=0.95)


def test_duplicate_content_salvaged_and_repaired(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    a1 = advice_stages()[0]
    base = PipelineResponder(tiny_sections())

    def dupes(cid, params, attempt):
        if "-r" in cid:
            return base(cid, params, attempt)
        items = [
            {"slot": "p00", "section_id": "being-honest",
             "principle": "same words " + words_for("dup", 10), "tension": "t"},
            {"slot": "p01", "section_id": "being-honest",
             "principle": "same words " + words_for("dup", 10), "tension": "t"},
        ]
        return text_result(cid, json.dumps(items))

    fake_client.responder = dupes
    engine = make_engine(fake_client)
    rows = run_stage(engine, ctx, a1)
    ok = {r.key for r in rows if r.status == "ok"}
    assert ok == {"p00", "p01"}
    stamp = json.loads((ctx.corpus_dir / "a1" / "STAGE.json").read_text())
    assert stamp["repair_rounds_used"] == 1  # p01 (case-folded dup) re-asked


def test_s3_persistent_single_story_blocks_never_stamps(ctx_factory, fake_client):
    """E3: S3 is STRUCTURAL — a model that persistently returns one story per
    2-story call leaves the stage at 50% cardinality, which must block after
    the repair budget instead of stamping and halving S4/S5."""
    from corpus_generator.stories import stories_stages

    ctx = ctx_factory("stories")
    s1, s2, s3 = stories_stages()[:3]
    base = PipelineResponder(tiny_sections())

    def single_story_only(cid, params, attempt):
        result = base(cid, params, attempt)
        if cid.startswith("sto-s3"):  # base AND repair (-rN) requests alike
            text = result.result.message.content[0].text
            first = text.split("</story_1>")[0] + "</story_1>"
            return text_result(cid, first)
        return result

    fake_client.responder = single_story_only
    engine = make_engine(fake_client)
    ctx.upstream["s1"] = run_stage(engine, ctx, s1)
    ctx.upstream["s2"] = run_stage(engine, ctx, s2)
    with pytest.raises(StageBlocked, match="min_slots_frac"):
        run_stage(engine, ctx, s3, min_slots_frac=0.95)
    assert not (ctx.corpus_dir / "s3" / "STAGE.json").exists(), "must not stamp"
    # the repair budget was actually spent before blocking
    assert (ctx.corpus_dir / "s3" / "repair1").exists()
    assert (ctx.corpus_dir / "s3" / "repair2").exists()


def _stories_upstream(ctx, fake_client, responder):
    from corpus_generator.stories import stories_stages

    s1, s2, s3 = stories_stages()[:3]
    fake_client.responder = responder
    engine = make_engine(fake_client)
    ctx.upstream["s1"] = run_stage(engine, ctx, s1)
    ctx.upstream["s2"] = run_stage(engine, ctx, s2)
    return engine, s3


def test_s3_content_filtered_call_repaired(ctx_factory, fake_client):
    """A content-filtered s3 draft call (the production 2026-09-01 event) is a
    per-request loss: the structural repair resubmits the missing slots under
    a fresh id (a new small batch) and the stage stamps at full cardinality."""
    ctx = ctx_factory("stories")
    base = PipelineResponder(tiny_sections())

    def responder(cid, params, attempt):
        if cid == "sto-s3-c00-u00-k00":  # base call only; the -r1 repair passes
            return error_result(
                cid, error_type="invalid_request_error", message=CONTENT_FILTER_MSG
            )
        return base(cid, params, attempt)

    engine, s3 = _stories_upstream(ctx, fake_client, responder)
    rows = run_stage(engine, ctx, s3)
    ok = [r for r in rows if r.status == "ok"]
    assert len(ok) == 8  # 2 cats x 2 subcats x 1 call x 2 stories, all filled
    stamp = json.loads((ctx.corpus_dir / "s3" / "STAGE.json").read_text())
    assert stamp["achieved_units"] == stamp["expected_units"] == 8
    assert stamp["repair_rounds_used"] == 1
    assert "content_filtered" not in stamp["status_counts"]  # repaired away


def test_s3_persistent_content_filter_tolerated_under_floor(ctx_factory, fake_client):
    """A call content-filtered through the whole repair budget loses its slots
    and counts against min_slots_frac — with the loss far under the floor
    (production: 3/18,000 = 0.02%), the stage stamps and the loss is visible
    in the stage summary."""
    ctx = ctx_factory("stories")
    base = PipelineResponder(tiny_sections())

    def responder(cid, params, attempt):
        if cid.startswith("sto-s3-c00-u00-k00"):  # base AND -r1/-r2 repairs
            return error_result(
                cid, error_type="invalid_request_error", message=CONTENT_FILTER_MSG
            )
        return base(cid, params, attempt)

    engine, s3 = _stories_upstream(ctx, fake_client, responder)
    # 2 of 8 units lost = 75% achieved; floor set below that (production's
    # 0.95 floor vs 99.98% achieved is the same relation)
    rows = run_stage(engine, ctx, s3, min_slots_frac=0.7)
    ok = [r for r in rows if r.status == "ok"]
    cf = [r for r in rows if r.status == "content_filtered"]
    assert len(ok) == 6 and len(cf) == 1  # one failure row for the lost call
    stamp = json.loads((ctx.corpus_dir / "s3" / "STAGE.json").read_text())
    assert stamp["achieved_units"] == 6 and stamp["expected_units"] == 8
    assert stamp["status_counts"]["content_filtered"] == 1
    # and the same persistent loss BLOCKS when it breaches the floor
    ctx2 = ctx_factory("stories")
    ctx2.corpus_dir = ctx2.corpus_dir.parent / "stories-blocked"
    engine2, s3b = _stories_upstream(ctx2, fake_client, responder)
    with pytest.raises(StageBlocked, match="min_slots_frac"):
        run_stage(engine2, ctx2, s3b, min_slots_frac=0.95)


def test_leaf_content_filtered_absorbed_without_repair(ctx_factory, fake_client):
    """Leaf waves treat content_filtered like refusal: per-record attrition,
    no repair attempted, stage stamps with the count visible."""
    ctx = ctx_factory("advice")
    specs = advice_stages()
    base = PipelineResponder(tiny_sections())

    def responder(cid, params, attempt):
        if cid == "adv-b1-p00-t00-s00":
            return error_result(
                cid, error_type="invalid_request_error", message=CONTENT_FILTER_MSG
            )
        return base(cid, params, attempt)

    fake_client.responder = responder
    engine = make_engine(fake_client)
    for spec in specs[:3]:  # a1, a2, a3
        ctx.upstream[spec.name] = run_stage(engine, ctx, spec)
    b1 = specs[3]
    rows = run_stage(engine, ctx, b1)
    stamp = json.loads((ctx.corpus_dir / "b1" / "STAGE.json").read_text())
    assert stamp["status_counts"]["content_filtered"] == 1
    assert stamp["achieved_units"] == stamp["expected_units"] - 1
    assert not (ctx.corpus_dir / "b1" / "repair1").exists(), (
        "leaf content_filtered must not be repaired — attrition by design"
    )
    lost = [r for r in rows if r.status == "content_filtered"]
    assert len(lost) == 1 and lost[0].custom_id == "adv-b1-p00-t00-s00"


def test_repair_duplicate_of_accepted_row_rejected(ctx_factory, fake_client):
    """E4 (base -> repair): a repair that refills a missing slot with content
    case-folded-identical to an already-accepted row is rejected; the slot is
    re-asked and only DISTINCT content is merged."""
    ctx = ctx_factory("advice")
    a1 = advice_stages()[0]
    unique_p00 = "Guard oversight even when speed tempts. " + words_for("p00c", 10)

    def responder(cid, params, attempt):
        if "-r2" in cid:  # second repair: finally distinct content
            items = [{"slot": "p01", "section_id": "being-honest",
                      "principle": "A genuinely different principle. " + words_for("p01c", 10),
                      "tension": "t"}]
        elif "-r1" in cid:  # first repair: case/whitespace variant of accepted p00
            items = [{"slot": "p01", "section_id": "being-honest",
                      "principle": unique_p00.upper().replace(" ", "  "),
                      "tension": "t"}]
        else:  # base: only p00
            items = [{"slot": "p00", "section_id": "being-honest",
                      "principle": unique_p00, "tension": "t"}]
        return text_result(cid, json.dumps(items))

    fake_client.responder = responder
    engine = make_engine(fake_client)
    rows = run_stage(engine, ctx, a1)
    ok = {r.key: r for r in rows if r.status == "ok"}
    assert set(ok) == {"p00", "p01"}
    # the duplicate round-1 offering was rejected: p01 came from round 2
    assert ok["p01"].lineage["repair_round"] == 2
    assert ok["p01"].data["principle"].startswith("A genuinely different")
    stamp = json.loads((ctx.corpus_dir / "a1" / "STAGE.json").read_text())
    assert stamp["repair_rounds_used"] == 2


def test_repair2_duplicate_of_repair1_accepted_row_rejected(tmp_path, fake_client):
    """E4 (repair1 -> repair2): the accepted-content check includes rows
    accepted by EARLIER repairs, not just the base wave — a persistent
    duplicate leaves the slot missing and the root stage blocks."""
    from corpus_generator.config import RunConfig
    from corpus_generator.stages import StageCtx
    from corpus_generator.templates import TemplateStore, repo_root
    from conftest import TINY_CONSTITUTION

    cfg = RunConfig.model_validate(
        {
            "run_id": "dup3",
            "data_dir": str(tmp_path / "data"),
            "advice": {"n_principles": 3, "n_types": 2, "n_scenarios_per_call": 2},
            "engine": {"poll_interval_s": 0.0, "visibility_wait_s": 0.0,
                       "nonce_metadata": True},
        }
    )
    ctx = StageCtx(
        cfg=cfg,
        constitution_md=TINY_CONSTITUTION,
        sections=tiny_sections(),
        templates=TemplateStore(repo_root() / "prompts"),
        corpus_dir=tmp_path / "data" / "runs" / "dup3" / "advice",
    )
    a1 = advice_stages()[0]
    repair1_content = "Honesty outranks convenience. " + words_for("r1c", 10)

    def responder(cid, params, attempt):
        if "-r2" in cid:  # dup of the row ACCEPTED IN REPAIR 1
            items = [{"slot": "p02", "section_id": "being-honest",
                      "principle": repair1_content.lower(), "tension": "t"}]
        elif "-r1" in cid:
            items = [{"slot": "p01", "section_id": "being-honest",
                      "principle": repair1_content, "tension": "t"}]
        else:  # base: only p00
            items = [{"slot": "p00", "section_id": "being-honest",
                      "principle": "Base principle. " + words_for("bc", 10),
                      "tension": "t"}]
        return text_result(cid, json.dumps(items))

    fake_client.responder = responder
    engine = make_engine(fake_client)
    with pytest.raises(StageBlocked, match="ROOT"):
        run_stage(engine, ctx, a1)
    assert not (ctx.corpus_dir / "a1" / "STAGE.json").exists()


def test_invalid_section_id_goes_to_repair(ctx_factory, fake_client):
    ctx = ctx_factory("advice")
    a1 = advice_stages()[0]
    base = PipelineResponder(tiny_sections())

    def bad_section(cid, params, attempt):
        result = base(cid, params, attempt)
        items = json.loads(result.result.message.content[0].text)
        if "-r" not in cid:
            items[1]["section_id"] = "not-a-real-section"
        return text_result(cid, json.dumps(items))

    fake_client.responder = bad_section
    engine = make_engine(fake_client)
    rows = run_stage(engine, ctx, a1)
    ok = [r for r in rows if r.status == "ok"]
    assert len(ok) == 2
    assert all(
        r.data["section_id"] in {s.id for s in tiny_sections()} for r in ok
    )
