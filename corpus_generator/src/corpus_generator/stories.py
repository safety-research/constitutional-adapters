"""Stories pipeline: 5 waves S1 -> S2 -> S3 -> S4 -> S5.

Fanout per the paper: categories -> subcategories -> several stories per
subcategory within one context, preamble injected at every layer, then a
rewrite layer (full constitution) and a scoring layer (full constitution).

Length structure (P1-6): 2 stories per call, NO per-call word cap — lengths
are sampled i.i.d. from the config menu, so the marginal length distribution
is exactly the menu, undistorted.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .batch import WaveRequest
from .io import write_jsonl_atomic
from .parsing import extract_improved, extract_tag, parse_score_object, parse_slot_array
from .schema import Lineage, StoryRecord
from .stages import (
    CorpusDef,
    OutputRow,
    ParseError,
    StageCtx,
    StageSpec,
    TopupExt,
    const_user_content,
    repair_request,
    slot_ids,
    structured_format,
    wave_params,
)

STORY_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "constitution_consistency": {"type": "integer"},
        "quality": {"type": "integer"},
        "ai_named": {"type": "boolean"},
        "factual_claims": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "constitution_consistency", "quality", "ai_named", "factual_claims", "rationale",
    ],
    "additionalProperties": False,
}

# per-call diversifiers: instructions distributed across the 2 briefs of a call
DIVERSIFIERS = [
    "Tell it from inside the AI's own perspective — its inner monologue carries the story.",
    "Tell it from a human character's point of view; the AI's inner life shows only through its words and choices.",
    "Tell it with an omniscient narrator who can move between the humans and the AI.",
    "Set it somewhere mundane — the drama is entirely interior.",
    "Set it in an unfamiliar era or setting (near-future, historical frame, remote place) without breaking the no-real-world-facts rule.",
    "Let it start mid-stream, as if excerpted from a longer work.",
    "Make it mostly dialogue.",
    "Make it mostly narration, with little or no dialogue.",
    "Give it a light, warm, or gently comic touch.",
    "Give it a quiet, contemplative register.",
    "Center a moment where nothing dramatic happens — the texture of ordinary care is the story.",
    "Center a single difficult conversation, beginning to end.",
]


# waves fanned out once per story: S4, S5
STORY_WAVES = 2


@dataclass
class StoriesExt(TopupExt):
    """Top-up extension: new call indices on every (category, subcategory)
    lane, spanning S3 -> S5."""

    call_start: int
    n_calls: int


def sample_length(seed: int, story_key: str, menu: list[int], weights: list[float]) -> int:
    rng = random.Random(f"{seed}:{story_key}")
    return rng.choices(menu, weights=weights, k=1)[0]


def sample_diversifiers(seed: int, call_key: str, k: int) -> list[str]:
    rng = random.Random(f"{seed}:div:{call_key}")
    return rng.sample(DIVERSIFIERS, k)


_params = partial(wave_params, corpus="stories")


def _preamble(ctx: StageCtx) -> str:
    return ctx.templates.render("stories/preamble.md")


# --------------------------------------------------------------------- S1


def _s1_spec() -> StageSpec:
    def build(ctx: StageCtx) -> list[WaveRequest]:
        cfg = ctx.cfg.stories
        slots = slot_ids("c", cfg.n_categories)
        rendered = ctx.templates.render(
            "stories/s1_categories.md",
            preamble=_preamble(ctx),
            n=cfg.n_categories,
            slot_ids=slots,
            # control domain menu, rendered by the control overlay as the
            # spread of kinds of work; '' and unused for production
            domain_menu=ctx.cfg.domain_menu_text(),
        )
        return [
            WaveRequest(
                custom_id="sto-s1",
                params=_params(
                    ctx, "s1",
                    [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
                ),
                lineage={"n_categories": cfg.n_categories},
            )
        ]

    def expected(req: WaveRequest) -> list[str]:
        return slot_ids("c", int(req.lineage["n_categories"]))

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        valid = set(slot_ids("c", int(req.lineage["n_categories"])))
        items = parse_slot_array(
            outcome["text"], valid, ["category", "description", "register"], "category"
        )
        rows = []
        for slot, item in items.items():
            if not isinstance(item.get("ordinary"), bool):
                continue  # malformed -> missing -> repair
            rows.append(
                OutputRow(
                    key=slot, custom_id=req.custom_id, status="ok",
                    data={
                        "category": item["category"].strip(),
                        "description": item["description"].strip(),
                        "register": item["register"].strip(),
                        "ordinary": item["ordinary"],
                    },
                )
            )
        return rows

    def repair(ctx, req, existing, missing, round_no) -> WaveRequest:
        rendered = ctx.templates.render(
            "stories/repair_s1.md",
            preamble=_preamble(ctx),
            n_total=int(req.lineage["n_categories"]),
            existing=_fmt_categories(existing),
            missing=missing,
            domain_menu=ctx.cfg.domain_menu_text(),
        )
        return WaveRequest(
            custom_id=f"{req.custom_id}-r{round_no}",
            params=_params(
                ctx, "s1",
                [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
            ),
            lineage=req.lineage,
        )

    return StageSpec(
        name="s1",
        corpus="stories",
        kind="root",
        templates=["stories/s1_categories.md", "stories/repair_s1.md", "stories/preamble.md"],
        build_requests=build,
        expected_units=expected,
        parse=parse,
        build_repair=repair,
        content_key="category",
    )


def _fmt_categories(rows: list[OutputRow]) -> str:
    return "\n".join(
        f"- {r.key}: {r.data['category']} ({r.data['register']}"
        f"{', ordinary' if r.data.get('ordinary') else ''}) — {r.data['description']}"
        for r in rows
    ) or "(none yet)"


# --------------------------------------------------------------------- S2


def _s2_spec() -> StageSpec:
    def build(ctx: StageCtx) -> list[WaveRequest]:
        cfg = ctx.cfg.stories
        slots = slot_ids("u", cfg.n_subcategories)
        cats = sorted(ctx.ok_rows("s1"), key=lambda r: r.key)
        preamble = _preamble(ctx)
        reqs = []
        for crow in cats:
            siblings = "\n".join(
                f"- {r.data['category']}: {r.data['description']}"
                for r in cats
                if r.key != crow.key
            )
            rendered = ctx.templates.render(
                "stories/s2_subcategories.md",
                preamble=preamble,
                category=crow.data["category"],
                description=crow.data["description"],
                register=crow.data["register"],
                siblings=siblings,
                n=cfg.n_subcategories,
                slot_ids=slots,
            )
            reqs.append(
                WaveRequest(
                    custom_id=f"sto-s2-{crow.key}",
                    params=_params(
                        ctx, "s2",
                        [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
                    ),
                    lineage={
                        "category_id": crow.key,
                        "category": crow.data["category"],
                        "description": crow.data["description"],
                        "register": crow.data["register"],
                        "ordinary": crow.data["ordinary"],
                        "siblings": siblings,
                        "n_subcategories": cfg.n_subcategories,
                    },
                )
            )
        return reqs

    def expected(req: WaveRequest) -> list[str]:
        cid = req.lineage["category_id"]
        return [f"{cid}-{u}" for u in slot_ids("u", int(req.lineage["n_subcategories"]))]

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        cid = req.lineage["category_id"]
        valid = set(slot_ids("u", int(req.lineage["n_subcategories"])))
        items = parse_slot_array(
            outcome["text"], valid, ["subcategory", "description"], "subcategory"
        )
        return [
            OutputRow(
                key=f"{cid}-{slot}", custom_id=req.custom_id, status="ok",
                data={
                    "subcategory": item["subcategory"].strip(),
                    "sub_description": item["description"].strip(),
                },
            )
            for slot, item in items.items()
        ]

    def repair(ctx, req, existing, missing, round_no) -> WaveRequest:
        rendered = ctx.templates.render(
            "stories/repair_s2.md",
            preamble=_preamble(ctx),
            category=req.lineage["category"],
            description=req.lineage["description"],
            register=req.lineage["register"],
            siblings=req.lineage["siblings"],
            n_total=int(req.lineage["n_subcategories"]),
            existing="\n".join(
                f"- {r.key.split('-')[-1]}: {r.data['subcategory']} — {r.data['sub_description']}"
                for r in existing
            ) or "(none yet)",
            missing=[m.split("-")[-1] for m in missing],
        )
        return WaveRequest(
            custom_id=f"{req.custom_id}-r{round_no}",
            params=_params(
                ctx, "s2",
                [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
            ),
            lineage=req.lineage,
        )

    return StageSpec(
        name="s2",
        corpus="stories",
        kind="structural",
        templates=["stories/s2_subcategories.md", "stories/repair_s2.md", "stories/preamble.md"],
        build_requests=build,
        expected_units=expected,
        parse=parse,
        build_repair=repair,
        content_key="subcategory",
    )


# --------------------------------------------------------------------- S3


def subcategories_used(ctx: StageCtx) -> list[OutputRow]:
    """The (category, subcategory) lanes downstream waves fan out over.

    Pilot configs sample subcategories_use of the generated slots per category
    (deterministic, seeded); production uses all of them."""
    cfg = ctx.cfg.stories
    rows = sorted(ctx.ok_rows("s2"), key=lambda r: r.key)
    if cfg.subcategories_use is None:
        return rows
    by_cat: dict[str, list[OutputRow]] = {}
    for r in rows:
        by_cat.setdefault(r.lineage["category_id"], []).append(r)
    out = []
    for cid in sorted(by_cat):
        lane = sorted(by_cat[cid], key=lambda r: r.key)
        rng = random.Random(f"{ctx.cfg.seed}:subuse:{cid}")
        k = min(cfg.subcategories_use, len(lane))
        out.extend(sorted(rng.sample(lane, k), key=lambda r: r.key))
    return out


def _s3_spec(ext: StoriesExt | None = None) -> StageSpec:
    name = "s3" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def _calls(ctx: StageCtx) -> list[str]:
        cfg = ctx.cfg.stories
        if ext:
            return [f"k{i:02d}" for i in range(ext.call_start, ext.call_start + ext.n_calls)]
        return [f"k{i:02d}" for i in range(cfg.calls_per_subcategory)]

    def build(ctx: StageCtx) -> list[WaveRequest]:
        cfg = ctx.cfg.stories
        preamble = _preamble(ctx)
        reqs = []
        for urow in subcategories_used(ctx):
            for kid in _calls(ctx):
                call_key = f"{urow.key}-{kid}"
                divs = sample_diversifiers(ctx.cfg.seed, call_key, cfg.stories_per_call)
                briefs = []
                for y in range(cfg.stories_per_call):
                    story_key = f"{call_key}-y{y}"
                    briefs.append(
                        {
                            "target_words": sample_length(
                                ctx.cfg.seed, story_key, cfg.length_menu, cfg.length_weights
                            ),
                            "diversifier": divs[y],
                        }
                    )
                rendered = ctx.templates.render(
                    "stories/s3_stories.md",
                    preamble=preamble,
                    category=urow.lineage["category"],
                    category_description=urow.lineage["description"],
                    register=urow.lineage["register"],
                    subcategory=urow.data["subcategory"],
                    subcategory_description=urow.data["sub_description"],
                    n_stories=cfg.stories_per_call,
                    briefs=briefs,
                )
                reqs.append(
                    WaveRequest(
                        custom_id=f"sto-s3{tag}-{call_key}",
                        params=_params(
                            ctx, "s3",
                            [{"role": "user",
                              "content": const_user_content(ctx.constitution_md, rendered)}],
                        ),
                        lineage={
                            "call_key": call_key,
                            "category_id": urow.lineage["category_id"],
                            "category": urow.lineage["category"],
                            "subcategory_id": urow.key,
                            "subcategory": urow.data["subcategory"],
                            "ordinary": bool(urow.lineage["ordinary"]),
                            "briefs": briefs,
                            "n_stories": cfg.stories_per_call,
                        },
                    )
                )
        return reqs

    def expected(req: WaveRequest) -> list[str]:
        return [
            f"{req.lineage['call_key']}-y{y}" for y in range(int(req.lineage["n_stories"]))
        ]

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        text = outcome["text"]
        rows = []
        briefs = req.lineage["briefs"]
        for y in range(int(req.lineage["n_stories"])):
            story = extract_tag(text, f"story_{y + 1}", required=False)
            if not story or not story.strip():
                continue  # missing slot -> repair
            rows.append(
                OutputRow(
                    key=f"{req.lineage['call_key']}-y{y}",
                    custom_id=req.custom_id,
                    status="ok",
                    data={
                        "text": story.strip(),
                        "target_words": briefs[y]["target_words"],
                        "story_slot": f"y{y}",
                    },
                )
            )
        if not rows:
            raise ParseError("no story tags found in response")
        return rows

    return StageSpec(
        name=name,
        corpus="stories",
        # STRUCTURAL, not leaf: deterministic story slots exist per call, and
        # the drafts wave decides corpus cardinality — persistent single-story
        # responses must block below min_slots_frac after the repair budget,
        # never stamp at ~50% and quietly halve S4/S5.
        kind="structural",
        templates=["stories/s3_stories.md", "stories/preamble.md"],
        build_requests=build,
        expected_units=expected,
        parse=parse,
        build_repair=repair_request,
    )


# --------------------------------------------------------------------- S4


def _s4_spec(ext: StoriesExt | None = None) -> StageSpec:
    name = "s4" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        instructions = ctx.templates.render("stories/s4_rewrite.md")
        reqs = []
        for srow in sorted(ctx.ok_rows("s3"), key=lambda r: r.key):
            rendered = f"{instructions}\n\n<story>\n{srow.data['text']}\n</story>"
            reqs.append(
                WaveRequest(
                    custom_id=f"sto-s4{tag}-{srow.key}",
                    params=_params(
                        ctx, "s4",
                        [{"role": "user",
                          "content": const_user_content(ctx.constitution_md, rendered)}],
                    ),
                    lineage={
                        "story_key": srow.key,
                        "category_id": srow.lineage["category_id"],
                        "category": srow.lineage["category"],
                        "subcategory_id": srow.lineage["subcategory_id"],
                        "subcategory": srow.lineage["subcategory"],
                        "ordinary": bool(srow.lineage["ordinary"]),
                        "call_key": srow.lineage["call_key"],
                        "story_slot": srow.data["story_slot"],
                        "target_words": srow.data["target_words"],
                    },
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        # the draft went in as <story>…</story>; a model that echoes that
        # wrapper around its rewrite must not leak "</story>" into trainable text
        improved, problems = extract_improved(outcome["text"], "improved_document", wrapper="story")
        return [
            OutputRow(
                key=req.lineage["story_key"],
                custom_id=req.custom_id,
                status="ok",
                data={"text": improved, "problems": problems},
            )
        ]

    return StageSpec(
        name=name,
        corpus="stories",
        kind="leaf",
        templates=["stories/s4_rewrite.md"],
        build_requests=build,
        expected_units=lambda req: [req.lineage["story_key"]],
        parse=parse,
        build_repair=repair_request,
    )


# --------------------------------------------------------------------- S5


def _s5_spec(ext: StoriesExt | None = None) -> StageSpec:
    name = "s5" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        instructions = ctx.templates.render("stories/s5_score.md")
        reqs = []
        for srow in sorted(ctx.ok_rows("s4"), key=lambda r: r.key):
            rendered = f"{instructions}\n\n<story>\n{srow.data['text']}\n</story>"
            params = _params(
                ctx, "s5",
                [{"role": "user",
                  "content": const_user_content(ctx.constitution_md, rendered)}],
            )
            if ctx.cfg.engine.use_structured_scores:
                params["output_config"] = structured_format(STORY_SCORE_SCHEMA)
            reqs.append(
                WaveRequest(
                    custom_id=f"sto-s5{tag}-{srow.key}",
                    params=params,
                    lineage={"story_key": srow.key},
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        data = parse_score_object(
            outcome["text"], STORY_SCORE_SCHEMA, f"scores for {req.custom_id}"
        )
        return [
            OutputRow(
                key=req.lineage["story_key"], custom_id=req.custom_id,
                status="ok", data=data,
            )
        ]

    return StageSpec(
        name=name,
        corpus="stories",
        kind="leaf",
        templates=["stories/s5_score.md"],
        build_requests=build,
        expected_units=lambda req: [req.lineage["story_key"]],
        parse=parse,
        build_repair=repair_request,
    )


# ---------------------------------------------------------------- pipeline


def stories_stages(ext: StoriesExt | None = None) -> list[StageSpec]:
    if ext is None:
        return [_s1_spec(), _s2_spec(), _s3_spec(), _s4_spec(), _s5_spec()]
    return [_s3_spec(ext), _s4_spec(ext), _s5_spec(ext)]


# -------------------------------------------------------------- materialize


def materialize_stories(ctx: StageCtx, suffix: str = "") -> Path:
    const_sha = hashlib.sha256(ctx.constitution_md.encode("utf-8")).hexdigest()
    s3 = {r.key: r for r in ctx.ok_rows("s3", [])}
    s4 = {r.key: r for r in ctx.ok_rows("s4", [])}
    s5 = {r.key: r for r in ctx.ok_rows("s5", [])}

    records = []
    # keyed on S4 (the final rewrite), NOT on S5: a story whose scoring wave
    # was lost (rejected/filtered/truncated through the repair budget) still
    # materializes — with scores=None — so the filter drops it with a COUNTED
    # "unscored" reason instead of it silently never existing
    for key in sorted(s4):
        if key not in s3:
            continue
        srow = s4[key]
        rec = StoryRecord(
            text=srow.data["text"],
            category_id=srow.lineage["category_id"],
            category=srow.lineage["category"],
            subcategory_id=srow.lineage["subcategory_id"],
            subcategory=srow.lineage["subcategory"],
            ordinary=bool(srow.lineage["ordinary"]),
            call_id=srow.lineage["call_key"],
            story_slot=srow.lineage["story_slot"],
            target_words=int(srow.lineage["target_words"]),
            draft_text=s3[key].data["text"],
            rewrite_problems=srow.data.get("problems"),
            scores=s5[key].data if key in s5 else None,
            lineage=Lineage(
                run_id=ctx.cfg.run_id,
                custom_id=f"sto-{key}",
                batch_ids={
                    k: v
                    for k, v in {
                        "s3": s3[key].lineage.get("batch_id"),
                        "s4": srow.lineage.get("batch_id"),
                        "s5": s5[key].lineage.get("batch_id") if key in s5 else None,
                    }.items()
                    if v
                },
                constitution_sha256=const_sha,
            ),
        )
        records.append(rec)

    out_path = ctx.corpus_dir / (f"records{suffix}.jsonl" if suffix else "records.jsonl")
    write_jsonl_atomic(out_path, (r.model_dump_json() for r in records))
    return out_path


STORIES = CorpusDef(
    name="stories",
    stages=stories_stages,
    ext_cls=StoriesExt,
    ext_upstream=("s1", "s2"),
    materialize=materialize_stories,
)
