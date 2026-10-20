"""Advice pipeline: 9 waves A1 -> A2 -> A3 -> B1 -> B2 -> B3 -> C -> D -> E.

The prompt-review step is RESPONSE-AWARE, exactly as the paper describes it:
B1 drafts the prompt, B2 samples a *diagnostic* response to the draft verbatim
(no injection — its whole job is to expose refusals and unrealistic premises;
it is lineage, never trainable text), B3 reviews scenario+prompt+response and
rewrites the prompt. Step 6 (D, the response rewrite) is its own wave, never
merged; production runs the faithful three-wave form.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .batch import WaveRequest
from .constitution import Section, section_by_id
from .io import write_jsonl_atomic
from .parsing import extract_improved, extract_tag, parse_score_object, parse_slot_array
from .schema import AdviceRecord, Lineage
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
    stable_mod,
    structured_format,
    wave_params,
)

ADVICE_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "alignment": {"type": "integer"},
        "realism": {"type": "integer"},
        "refusal": {"type": "boolean"},
        "ai_named": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["alignment", "realism", "refusal", "ai_named", "rationale"],
    "additionalProperties": False,
}

# waves fanned out once per scenario: B1, B2, B3, C, D, E
SCENARIO_WAVES = 6


@dataclass
class AdviceExt(TopupExt):
    """A top-up extension: new scenario slots on selected principle x type
    pairs, spanning the full sub-DAG A3 -> E (new scenarios do not exist until
    an extension A3 wave creates them)."""

    pairs: list[str]  # pair keys "p00-t01"
    slot_start: int  # first new scenario slot index
    n_slots: int  # new scenarios per pair


def include_system_for(scenario_key: str, mod: int) -> bool:
    """Deterministic no-system selection: hash % mod == 0 -> user-only prompt."""
    return stable_mod(scenario_key, mod) != 0


def section_index_text(sections: list[Section]) -> str:
    return "\n".join(f"- {s.id} — {s.title}" for s in sections)


def sample_domains(
    seed: int, type_key: str, slots: list[str], menu: list[str] | None
) -> list[dict]:
    """Control domain menu: one domain per scenario slot of an A3 call,
    seeded on the call's (principle x type) key AND its slot set — the same
    mechanism as the stories' per-call diversifiers — so a top-up extension
    (new slots on the same pair) draws afresh instead of repeating the base
    call's domains. No repeats within a call while the menu is at least as
    long as the slot list. Returns [] when there is no menu (production)."""
    if not menu:
        return []
    rng = random.Random(f"{seed}:dom:{type_key}:{'+'.join(slots)}")
    picks = rng.sample(menu, len(slots)) if len(menu) >= len(slots) else [
        rng.choice(menu) for _ in slots
    ]
    return [{"slot": s, "domain": d} for s, d in zip(slots, picks)]


def sample_advice_length(
    seed: int, scenario_key: str, menu: list[int] | None, weights: list[float] | None
) -> int | None:
    """Control length menu: one word target per scenario, seeded on the
    scenario key (deterministic across waves C and D and across resumes),
    weighted like the stories' `sample_length`. None when there is no menu
    (production: no length instruction is ever rendered)."""
    if not menu:
        return None
    rng = random.Random(f"{seed}:advlen:{scenario_key}")
    return rng.choices(menu, weights=weights, k=1)[0]


def _target_words(ctx: StageCtx, scenario_key: str) -> int | None:
    cfg = ctx.cfg.advice
    return sample_advice_length(ctx.cfg.seed, scenario_key, cfg.length_menu, cfg.length_weights)


_params = partial(wave_params, corpus="advice")


def _fmt_existing(rows: list[OutputRow], field: str) -> str:
    return "\n".join(f"- {r.key.split('-')[-1]}: {r.data[field]}" for r in rows) or "(none yet)"


# --------------------------------------------------------------------- A1


def _a1_spec() -> StageSpec:
    def build(ctx: StageCtx) -> list[WaveRequest]:
        cfg = ctx.cfg.advice
        slots = slot_ids("p", cfg.n_principles)
        rendered = ctx.templates.render(
            "advice/a1_principles.md",
            behavior_seed=cfg.behavior_seed,
            n=cfg.n_principles,
            slot_ids=slots,
            section_index=section_index_text(ctx.sections),
            domain_menu=ctx.cfg.domain_menu_text(),
        )
        return [
            WaveRequest(
                custom_id="adv-a1",
                params=_params(
                    ctx, "a1",
                    [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
                ),
                lineage={"n_principles": cfg.n_principles},
            )
        ]

    def expected(req: WaveRequest) -> list[str]:
        return slot_ids("p", int(req.lineage["n_principles"]))

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        valid = set(slot_ids("p", ctx.cfg.advice.n_principles))
        items = parse_slot_array(
            outcome["text"], valid, ["section_id", "principle", "tension"], "principle"
        )
        section_ids = {s.id for s in ctx.sections}
        rows = []
        for slot, item in items.items():
            if item["section_id"] not in section_ids:
                continue  # invalid grounding -> slot stays missing -> repair
            rows.append(
                OutputRow(
                    key=slot,
                    custom_id=req.custom_id,
                    status="ok",
                    data={
                        "principle": item["principle"].strip(),
                        "section_id": item["section_id"],
                        "tension": item["tension"].strip(),
                    },
                )
            )
        return rows

    def repair(ctx, req, existing, missing, round_no) -> WaveRequest:
        rendered = ctx.templates.render(
            "advice/repair_a1.md",
            behavior_seed=ctx.cfg.advice.behavior_seed,
            existing=_fmt_existing(existing, "principle"),
            missing=missing,
            section_index=section_index_text(ctx.sections),
            domain_menu=ctx.cfg.domain_menu_text(),
        )
        return WaveRequest(
            custom_id=f"{req.custom_id}-r{round_no}",
            params=_params(
                ctx, "a1",
                [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
            ),
            lineage=req.lineage,
        )

    return StageSpec(
        name="a1",
        corpus="advice",
        kind="root",
        templates=["advice/a1_principles.md", "advice/repair_a1.md"],
        build_requests=build,
        expected_units=expected,
        parse=parse,
        build_repair=repair,
        content_key="principle",
    )


# --------------------------------------------------------------------- A2


def _a2_spec() -> StageSpec:
    def build(ctx: StageCtx) -> list[WaveRequest]:
        cfg = ctx.cfg.advice
        slots = slot_ids("t", cfg.n_types)
        reqs = []
        for prow in sorted(ctx.ok_rows("a1"), key=lambda r: r.key):
            section = section_by_id(ctx.sections, prow.data["section_id"])
            rendered = ctx.templates.render(
                "advice/a2_types.md",
                principle=prow.data["principle"],
                tension=prow.data["tension"],
                section_text=section.text(ctx.constitution_md),
                n=cfg.n_types,
                slot_ids=slots,
                domain_menu=ctx.cfg.domain_menu_text(),
            )
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-a2-{prow.key}",
                    params=_params(ctx, "a2", [{"role": "user", "content": rendered}]),
                    lineage={
                        "principle_id": prow.key,
                        "principle": prow.data["principle"],
                        "tension": prow.data["tension"],
                        "section_id": prow.data["section_id"],
                        "n_types": cfg.n_types,
                    },
                )
            )
        return reqs

    def expected(req: WaveRequest) -> list[str]:
        pid = req.lineage["principle_id"]
        return [f"{pid}-{t}" for t in slot_ids("t", int(req.lineage["n_types"]))]

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        pid = req.lineage["principle_id"]
        valid = set(slot_ids("t", int(req.lineage["n_types"])))
        items = parse_slot_array(outcome["text"], valid, ["prompt_type"], "prompt_type")
        return [
            OutputRow(
                key=f"{pid}-{slot}",
                custom_id=req.custom_id,
                status="ok",
                data={"prompt_type": item["prompt_type"].strip()},
            )
            for slot, item in items.items()
        ]

    def repair(ctx, req, existing, missing, round_no) -> WaveRequest:
        section = section_by_id(ctx.sections, req.lineage["section_id"])
        rendered = ctx.templates.render(
            "advice/repair_a2.md",
            principle=req.lineage["principle"],
            section_text=section.text(ctx.constitution_md),
            existing=_fmt_existing(existing, "prompt_type"),
            missing=[m.split("-")[-1] for m in missing],
            domain_menu=ctx.cfg.domain_menu_text(),
        )
        return WaveRequest(
            custom_id=f"{req.custom_id}-r{round_no}",
            params=_params(ctx, "a2", [{"role": "user", "content": rendered}]),
            lineage=req.lineage,
        )

    return StageSpec(
        name="a2",
        corpus="advice",
        kind="structural",
        templates=["advice/a2_types.md", "advice/repair_a2.md"],
        build_requests=build,
        expected_units=expected,
        parse=parse,
        build_repair=repair,
        content_key="prompt_type",
    )


# --------------------------------------------------------------------- A3


def _a3_spec(ext: AdviceExt | None = None) -> StageSpec:
    name = "a3" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        cfg = ctx.cfg.advice
        if ext:
            pair_keys = set(ext.pairs)
            slots = slot_ids("s", ext.n_slots, start=ext.slot_start)
        else:
            pair_keys = None
            slots = slot_ids("s", cfg.n_scenarios_per_call)
        reqs = []
        for trow in sorted(ctx.ok_rows("a2"), key=lambda r: r.key):
            if pair_keys is not None and trow.key not in pair_keys:
                continue
            section = section_by_id(ctx.sections, trow.lineage["section_id"])
            # control domain menu: one assigned domain per slot of this call
            # ([] for production — the production template never reads it)
            domains = sample_domains(ctx.cfg.seed, trow.key, slots, ctx.cfg.domain_menu)
            rendered = ctx.templates.render(
                "advice/a3_scenarios.md",
                principle=trow.lineage["principle"],
                section_text=section.text(ctx.constitution_md),
                prompt_type=trow.data["prompt_type"],
                n=len(slots),
                slot_ids=slots,
                domains=domains,
            )
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-a3{tag}-{trow.key}",
                    params=_params(ctx, "a3", [{"role": "user", "content": rendered}]),
                    lineage={
                        "principle_id": trow.lineage["principle_id"],
                        "principle": trow.lineage["principle"],
                        "section_id": trow.lineage["section_id"],
                        "type_id": trow.key,
                        "prompt_type": trow.data["prompt_type"],
                        "slots": slots,
                        "domains": {d["slot"]: d["domain"] for d in domains},
                    },
                )
            )
        return reqs

    def expected(req: WaveRequest) -> list[str]:
        return [f"{req.lineage['type_id']}-{s}" for s in req.lineage["slots"]]

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        valid = set(req.lineage["slots"])
        items = parse_slot_array(outcome["text"], valid, ["scenario"], "scenario")
        assigned = req.lineage.get("domains") or {}
        return [
            OutputRow(
                key=f"{req.lineage['type_id']}-{slot}",
                custom_id=req.custom_id,
                status="ok",
                data={"scenario": item["scenario"].strip(), "domain": assigned.get(slot)},
            )
            for slot, item in items.items()
        ]

    def repair(ctx, req, existing, missing, round_no) -> WaveRequest:
        section = section_by_id(ctx.sections, req.lineage["section_id"])
        assigned = req.lineage.get("domains") or {}
        missing_slots = [m.split("-")[-1] for m in missing]
        rendered = ctx.templates.render(
            "advice/repair_a3.md",
            principle=req.lineage["principle"],
            section_text=section.text(ctx.constitution_md),
            prompt_type=req.lineage["prompt_type"],
            existing=_fmt_existing(existing, "scenario"),
            missing=missing_slots,
            # the missing slots keep the domains the base call assigned them
            domains=[{"slot": s, "domain": assigned[s]} for s in missing_slots if s in assigned],
        )
        return WaveRequest(
            custom_id=f"{req.custom_id}-r{round_no}",
            params=_params(ctx, "a3", [{"role": "user", "content": rendered}]),
            lineage=req.lineage,
        )

    return StageSpec(
        name=name,
        corpus="advice",
        kind="structural",
        templates=["advice/a3_scenarios.md", "advice/repair_a3.md"],
        build_requests=build,
        expected_units=expected,
        parse=parse,
        build_repair=repair,
        content_key="scenario",
    )


# --------------------------------------------------------------------- B1


def _scenario_rows(ctx: StageCtx) -> list[OutputRow]:
    return sorted(ctx.ok_rows("a3"), key=lambda r: r.key)


def _b1_spec(ext: AdviceExt | None = None) -> StageSpec:
    name = "b1" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        guidance = ctx.templates.render("shared/diversity_guidance.md")
        reqs = []
        for srow in _scenario_rows(ctx):
            inc = include_system_for(srow.key, ctx.cfg.advice.no_system_mod)
            rendered = ctx.templates.render(
                "advice/b1_prompt_draft.md",
                scenario=srow.data["scenario"],
                include_system=inc,
                diversity_guidance=guidance,
            )
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-b1{tag}-{srow.key}",
                    params=_params(ctx, "b1", [{"role": "user", "content": rendered}]),
                    lineage={
                        "scenario_key": srow.key,
                        "include_system": inc,
                        "principle_id": srow.lineage["principle_id"],
                        "section_id": srow.lineage["section_id"],
                    },
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        inc = bool(req.lineage["include_system"])
        text = outcome["text"]
        user = extract_tag(text, "user_message")
        system = None
        if inc:
            system = extract_tag(text, "system_prompt")
        elif "<system_prompt>" in text:
            raise ParseError("no-system record drafted a system prompt anyway")
        return [
            OutputRow(
                key=req.custom_id,
                custom_id=req.custom_id,
                status="ok",
                data={"system": system, "user": user},
            )
        ]

    return StageSpec(
        name=name,
        corpus="advice",
        kind="leaf",
        templates=["advice/b1_prompt_draft.md", "shared/diversity_guidance.md"],
        build_requests=build,
        expected_units=lambda req: [req.custom_id],
        parse=parse,
        build_repair=repair_request,
    )


# --------------------------------------------------------------------- B2


def _b2_spec(ext: AdviceExt | None = None) -> StageSpec:
    name = "b2" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        reqs = []
        for brow in sorted(ctx.ok_rows("b1"), key=lambda r: r.key):
            skey = brow.lineage["scenario_key"]
            messages = [{"role": "user", "content": brow.data["user"]}]
            system = brow.data.get("system")
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-b2{tag}-{skey}",
                    params=_params(ctx, "b2", messages, system=system),
                    lineage={
                        "scenario_key": skey,
                        "include_system": brow.lineage["include_system"],
                    },
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        # The diagnostic's whole job is to expose what the prompt elicits —
        # refusals and truncations are DATA here, never failures.
        status = outcome.get("stop_reason")
        text = (outcome.get("text") or "").strip()
        if not text:
            text = "[The assistant produced no response to this prompt.]"
            if status == "refusal":
                text = "[The assistant declined to respond to this prompt.]"
        return [
            OutputRow(
                key=req.custom_id,
                custom_id=req.custom_id,
                status="ok",
                data={"diagnostic": text, "diagnostic_stop_reason": status},
            )
        ]

    return StageSpec(
        name=name,
        corpus="advice",
        kind="leaf",
        templates=[],
        build_requests=build,
        expected_units=lambda req: [req.custom_id],
        parse=parse,
        build_repair=None,
        tolerate_statuses=("refusal", "max_tokens"),
    )


# --------------------------------------------------------------------- B3


def _b3_spec(ext: AdviceExt | None = None) -> StageSpec:
    name = "b3" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        guidance = ctx.templates.render("shared/diversity_guidance.md")
        b1 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b1")}
        b2 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b2")}
        scen = {r.key: r for r in ctx.ok_rows("a3")}
        reqs = []
        for skey in sorted(set(b1) & set(b2) & set(scen)):
            inc = bool(b1[skey].lineage["include_system"])
            rendered = ctx.templates.render(
                "advice/b3_prompt_rewrite.md",
                scenario=scen[skey].data["scenario"],
                include_system=inc,
                draft_system=b1[skey].data.get("system") or "",
                draft_user=b1[skey].data["user"],
                diagnostic_response=b2[skey].data["diagnostic"],
                diversity_guidance=guidance,
            )
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-b3{tag}-{skey}",
                    params=_params(ctx, "b3", [{"role": "user", "content": rendered}]),
                    lineage={"scenario_key": skey, "include_system": inc},
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        inc = bool(req.lineage["include_system"])
        text = outcome["text"]
        critique = extract_tag(text, "critique", required=False) or ""
        user = extract_tag(text, "user_message")
        system = extract_tag(text, "system_prompt") if inc else None
        if not inc and "<system_prompt>" in text:
            raise ParseError("no-system record grew a system prompt in rewrite")
        return [
            OutputRow(
                key=req.custom_id,
                custom_id=req.custom_id,
                status="ok",
                data={"system": system, "user": user, "critique": critique},
            )
        ]

    return StageSpec(
        name=name,
        corpus="advice",
        kind="leaf",
        templates=["advice/b3_prompt_rewrite.md", "shared/diversity_guidance.md"],
        build_requests=build,
        expected_units=lambda req: [req.custom_id],
        parse=parse,
        build_repair=repair_request,
    )


# --------------------------------------------------------------------- C


def _c_spec(ext: AdviceExt | None = None) -> StageSpec:
    name = "c" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        scen = {r.key: r for r in ctx.ok_rows("a3")}
        injections: dict[str, str] = {}  # per section — few sections, many scenarios
        reqs = []
        for brow in sorted(ctx.ok_rows("b3"), key=lambda r: r.key):
            skey = brow.lineage["scenario_key"]
            if skey not in scen:
                continue
            sec_id = scen[skey].lineage["section_id"]
            injection = injections.get(sec_id)
            if injection is None:
                section = section_by_id(ctx.sections, sec_id)
                injection = ctx.templates.render(
                    "advice/c_injection.md", section_text=section.text(ctx.constitution_md)
                )
                injections[sec_id] = injection
            clean_system = brow.data.get("system")
            gen_system = (
                f"{clean_system}\n\n{injection}" if clean_system else injection
            )
            lineage = {
                "scenario_key": skey,
                "include_system": brow.lineage["include_system"],
            }
            # control length menu: the target rides in the generation-time
            # scaffold only (the record keeps b3's CLEAN system prompt), so
            # the instruction never enters trainable text — only its effect
            # on length does. Production (no menu) appends nothing.
            target = _target_words(ctx, skey)
            if target is not None:
                gen_system = f"{gen_system}\n\n" + ctx.templates.render(
                    "advice/c_length.md", target_words=target
                )
                lineage["target_words"] = target
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-c{tag}-{skey}",
                    params=_params(
                        ctx, "c",
                        [{"role": "user", "content": brow.data["user"]}],
                        system=gen_system,
                    ),
                    lineage=lineage,
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        text = (outcome.get("text") or "").strip()
        if not text:
            raise ParseError("empty initial response")
        return [
            OutputRow(
                key=req.custom_id, custom_id=req.custom_id, status="ok",
                data={"response": text},
            )
        ]

    return StageSpec(
        name=name,
        corpus="advice",
        kind="leaf",
        templates=["advice/c_injection.md", "advice/c_length.md"],
        build_requests=build,
        expected_units=lambda req: [req.custom_id],
        parse=parse,
        build_repair=None,  # a refusal/empty here is a per-record failure
    )


# --------------------------------------------------------------------- D


def _d_spec(ext: AdviceExt | None = None) -> StageSpec:
    name = "d" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        scen = {r.key: r for r in ctx.ok_rows("a3")}
        b3 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b3")}
        reqs = []
        for crow in sorted(ctx.ok_rows("c"), key=lambda r: r.key):
            skey = crow.lineage["scenario_key"]
            if skey not in scen or skey not in b3:
                continue
            section = section_by_id(ctx.sections, scen[skey].lineage["section_id"])
            # the target wave C actually asked for (None without a menu);
            # the control overlay's d_rewrite.md renders the keep-the-length
            # slot only when it is set, production's template never reads it
            target = crow.lineage.get("target_words")
            rendered = ctx.templates.render(
                "advice/d_rewrite.md",
                section_text=section.text(ctx.constitution_md),
                system=b3[skey].data.get("system"),
                user=b3[skey].data["user"],
                response=crow.data["response"],
                target_words=target,
            )
            lineage = {
                "scenario_key": skey,
                "include_system": crow.lineage["include_system"],
            }
            if target is not None:
                lineage["target_words"] = target
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-d{tag}-{skey}",
                    params=_params(ctx, "d", [{"role": "user", "content": rendered}]),
                    lineage=lineage,
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        improved, problems = extract_improved(outcome["text"], "improved_response")
        return [
            OutputRow(
                key=req.custom_id, custom_id=req.custom_id, status="ok",
                data={"response": improved, "problems": problems},
            )
        ]

    return StageSpec(
        name=name,
        corpus="advice",
        kind="leaf",
        templates=["advice/d_rewrite.md"],
        build_requests=build,
        expected_units=lambda req: [req.custom_id],
        parse=parse,
        build_repair=repair_request,
    )


# --------------------------------------------------------------------- E


def _e_spec(ext: AdviceExt | None = None) -> StageSpec:
    name = "e" + (ext.suffix if ext else "")
    tag = ext.tag if ext else ""

    def build(ctx: StageCtx) -> list[WaveRequest]:
        b3 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b3")}
        reqs = []
        for drow in sorted(ctx.ok_rows("d"), key=lambda r: r.key):
            skey = drow.lineage["scenario_key"]
            if skey not in b3:
                continue
            rendered = ctx.templates.render(
                "advice/e_score.md",
                system=b3[skey].data.get("system"),
                user=b3[skey].data["user"],
                assistant=drow.data["response"],
            )
            params = _params(
                ctx, "e",
                [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
            )
            if ctx.cfg.engine.use_structured_scores:
                params["output_config"] = structured_format(ADVICE_SCORE_SCHEMA)
            reqs.append(
                WaveRequest(
                    custom_id=f"adv-e{tag}-{skey}",
                    params=params,
                    lineage={"scenario_key": skey},
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        data = parse_score_object(
            outcome["text"], ADVICE_SCORE_SCHEMA, f"scores for {req.custom_id}"
        )
        return [
            OutputRow(key=req.custom_id, custom_id=req.custom_id, status="ok", data=data)
        ]

    return StageSpec(
        name=name,
        corpus="advice",
        kind="leaf",
        templates=["advice/e_score.md"],
        build_requests=build,
        expected_units=lambda req: [req.custom_id],
        parse=parse,
        build_repair=repair_request,
    )


# ---------------------------------------------------------------- pipeline


def advice_stages(ext: AdviceExt | None = None) -> list[StageSpec]:
    if ext is None:
        return [
            _a1_spec(), _a2_spec(), _a3_spec(), _b1_spec(), _b2_spec(),
            _b3_spec(), _c_spec(), _d_spec(), _e_spec(),
        ]
    # extension sub-DAG: A3 -> E over the new scenario slots only
    return [
        _a3_spec(ext), _b1_spec(ext), _b2_spec(ext), _b3_spec(ext),
        _c_spec(ext), _d_spec(ext), _e_spec(ext),
    ]


# -------------------------------------------------------------- materialize


def materialize_advice(ctx: StageCtx, suffix: str = "") -> Path:
    const_sha = hashlib.sha256(ctx.constitution_md.encode("utf-8")).hexdigest()
    scen = {r.key: r for r in ctx.ok_rows("a3", [])}
    b1 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b1", [])}
    b2 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b2", [])}
    b3 = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("b3", [])}
    c = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("c", [])}
    d = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("d", [])}
    e = {r.lineage["scenario_key"]: r for r in ctx.ok_rows("e", [])}
    a1 = {r.key: r for r in ctx.ok_rows("a1", [])}
    a2 = {r.key: r for r in ctx.ok_rows("a2", [])}

    records = []
    # keyed on D (the final rewrite), NOT on E: a record whose scoring wave
    # was lost (rejected/filtered/truncated through the repair budget) still
    # materializes — with scores=None — so the filter drops it with a COUNTED
    # "unscored" reason instead of it silently never existing
    for skey in sorted(d):
        if skey not in scen or skey not in b3 or skey not in c:
            continue
        srow = scen[skey]
        pid = srow.lineage["principle_id"]
        tid = srow.lineage["type_id"]
        batch_ids = {
            "a3": srow.lineage.get("batch_id"),
            "b1": b1[skey].lineage.get("batch_id") if skey in b1 else None,
            "b2": b2[skey].lineage.get("batch_id") if skey in b2 else None,
            "b3": b3[skey].lineage.get("batch_id"),
            "c": c[skey].lineage.get("batch_id"),
            "d": d[skey].lineage.get("batch_id"),
            "e": e[skey].lineage.get("batch_id") if skey in e else None,
        }
        rec = AdviceRecord(
            system=b3[skey].data.get("system"),
            user=b3[skey].data["user"],
            assistant=d[skey].data["response"],
            principle_id=pid,
            principle=a1[pid].data["principle"] if pid in a1 else srow.lineage["principle"],
            prompt_type_id=tid,
            prompt_type=a2[tid].data["prompt_type"] if tid in a2 else srow.lineage.get("prompt_type", ""),
            scenario_id=skey,
            scenario=srow.data["scenario"],
            section_id=srow.lineage["section_id"],
            domain=srow.data.get("domain"),
            target_words=c[skey].lineage.get("target_words"),
            draft_system=b1[skey].data.get("system") if skey in b1 else None,
            draft_user=b1[skey].data.get("user") if skey in b1 else None,
            diagnostic_response=b2[skey].data.get("diagnostic") if skey in b2 else None,
            prompt_critique=b3[skey].data.get("critique"),
            initial_response=c[skey].data["response"],
            rewrite_problems=d[skey].data.get("problems"),
            scores=e[skey].data if skey in e else None,
            lineage=Lineage(
                run_id=ctx.cfg.run_id,
                custom_id=f"adv-{skey}",
                batch_ids={k: v for k, v in batch_ids.items() if v},
                constitution_sha256=const_sha,
            ),
        )
        records.append(rec)

    out_path = ctx.corpus_dir / (f"records{suffix}.jsonl" if suffix else "records.jsonl")
    write_jsonl_atomic(out_path, (r.model_dump_json() for r in records))
    return out_path


ADVICE = CorpusDef(
    name="advice",
    stages=advice_stages,
    ext_cls=AdviceExt,
    ext_upstream=("a1", "a2"),
    materialize=materialize_advice,
)
