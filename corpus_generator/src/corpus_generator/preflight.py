"""Preflight: measure REAL encoded request bodies per wave and derive the
chunk plan. Bodies come from the REAL stage builders (``spec.build_requests``
over a synthetic upstream), so template or params drift can never diverge from
what the run will submit — the only hand-maintained inputs here are the
synthetic payload SIZES. Refuses any plan whose packed batch would exceed the
200 MB hard cap (packing target 180 MB). Local only — no API."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .advice import advice_stages, include_system_for
from .batch import wire_tag
from .config import RunConfig
from .constitution import Section, load as load_constitution
from .io import atomic_write
from .pipeline import dag_expanded_totals
from .stages import OutputRow, StageCtx
from .stories import stories_stages
from .templates import TemplateStore, repo_root

WORD = "lorem"


def _words(n: int) -> str:
    return " ".join(f"{WORD}{i % 97}" for i in range(n))


class PreflightRefusal(RuntimeError):
    pass


def _row(key: str, data: dict, lineage: dict | None = None) -> OutputRow:
    return OutputRow(key=key, custom_id=f"pf-{key}", status="ok",
                     data=data, lineage=lineage or {})


def _synthetic_upstream(cfg: RunConfig, section: Section, p: dict) -> dict[str, list[OutputRow]]:
    """One synthetic unit per stage, shaped exactly as the real parsers emit
    them, so the real builders fan out over a single representative lane.
    (S1 gets full category fanout so S2's sibling list is production-size.)"""
    a = cfg.advice
    # a scenario key that KEEPS its system prompt — those bodies are larger
    skey = next(
        k for k in (f"p00-t00-s{i:02d}" for i in range(1000))
        if include_system_for(k, a.no_system_mod)
    )
    principle = {"principle": p["principle"], "section_id": section.id, "tension": p["tension"]}
    pair_lineage = {
        "principle_id": "p00", "principle": p["principle"],
        "section_id": section.id, "type_id": "p00-t00", "prompt_type": p["prompt_type"],
    }
    scen_lineage = {"scenario_key": skey, "include_system": True,
                    "principle_id": "p00", "section_id": section.id}
    story_key = "c00-u00-k00-y0"
    story_lineage = {
        "call_key": "c00-u00-k00", "category_id": "c00", "category": p["category"],
        "subcategory_id": "c00-u00", "subcategory": p["subcategory"],
        "ordinary": True, "story_key": story_key,
    }
    return {
        "a1": [_row("p00", principle)],
        "a2": [_row("p00-t00", {"prompt_type": p["prompt_type"]},
                    {**pair_lineage, "n_types": a.n_types})],
        "a3": [_row(skey, {"scenario": p["scenario"]}, pair_lineage)],
        "b1": [_row(f"adv-b1-{skey}", {"system": p["system"], "user": p["user"]}, scen_lineage)],
        "b2": [_row(f"adv-b2-{skey}", {"diagnostic": p["diagnostic"]}, scen_lineage)],
        "b3": [_row(f"adv-b3-{skey}",
                    {"system": p["system"], "user": p["user"], "critique": p["tension"]},
                    scen_lineage)],
        # the C row carries the length-menu draw D preserves (largest target
        # when the menu is set: D's body is measured with the slot rendered)
        "c": [_row(f"adv-c-{skey}", {"response": p["response"]},
                   {**scen_lineage, **({"target_words": max(a.length_menu)}
                                       if a.length_menu else {})})],
        "d": [_row(f"adv-d-{skey}", {"response": p["response"], "problems": ""}, scen_lineage)],
        "s1": [
            _row(f"c{i:02d}", {"category": p["category"], "description": p["description"],
                               "register": "literary", "ordinary": i % 2 == 0})
            for i in range(cfg.stories.n_categories)
        ],
        "s2": [_row("c00-u00", {"subcategory": p["subcategory"],
                                "sub_description": p["sub_description"]},
                    {"category_id": "c00", "category": p["category"],
                     "description": p["description"], "register": "literary",
                     "ordinary": True})],
        "s3": [_row(story_key,
                    {"text": p["story"], "target_words": max(cfg.stories.length_menu),
                     "story_slot": "y0"},
                    story_lineage)],
        "s4": [_row(story_key, {"text": p["story"], "problems": ""}, story_lineage)],
    }


# worst-case words a wave's output can reach per output token (English text
# runs ~0.6-0.7 words/token; the original hand-set 5200-word bound for an
# 8000-token cap encoded the same ratio)
_WORDS_PER_OUTPUT_TOKEN = 0.65


def _payloads(cfg: RunConfig) -> tuple[dict, dict]:
    """(typical, max) synthetic payload word-sizes — the ONLY sizing knobs.

    Payloads that are themselves an upstream wave's OUTPUT (the B2 diagnostic,
    the C/D responses, the story drafts) derive their worst case from that
    wave's configured max_tokens, so raising a cap automatically re-sizes the
    downstream request bodies preflight measures."""
    mt = cfg.advice.max_tokens
    shared = {
        "principle": _words(60), "tension": _words(30), "prompt_type": _words(40),
        "scenario": _words(160), "category": _words(6), "description": _words(40),
        "subcategory": _words(6), "sub_description": _words(30),
    }
    typical = {
        **shared,
        "system": _words(200), "user": _words(250),
        "diagnostic": _words(1200), "response": _words(1000),
        "story": _words(1830),
    }
    biggest = {
        **shared,
        "system": _words(350), "user": _words(700),
        # diagnostic = B2 output (embedded in B3); response = C output
        # (embedded in D) or D output (embedded in E) — one knob covers both,
        # sized to the larger cap
        "diagnostic": _words(int(_WORDS_PER_OUTPUT_TOKEN * mt["b2"])),
        "response": _words(int(_WORDS_PER_OUTPUT_TOKEN * max(mt["c"], mt["d"]))),
        "story": _words(2 * max(cfg.stories.length_menu)),
    }
    return typical, biggest


def _measure_pass(
    cfg: RunConfig, root: Path, md: str, sections: list[Section], section: Section, p: dict
) -> dict[str, int]:
    """wave name -> max encoded request bytes, built by the REAL builders."""
    ctx = StageCtx(
        cfg=cfg, constitution_md=md, sections=sections,
        templates=TemplateStore(
            root / "prompts", overlay_dir=cfg.resolve_prompts_overlay(root)
        ),
        corpus_dir=root,
        upstream=_synthetic_upstream(cfg, section, p),
    )
    # the engine submits run-tagged wire custom_ids ("<8-hex>-<id>"), so the
    # measured body is the logical encoding plus the tag's bytes
    tag_overhead = len(wire_tag(cfg.run_id)) + 1
    out: dict[str, int] = {}
    for spec in advice_stages() + stories_stages():
        reqs = spec.build_requests(ctx)
        out[f"{spec.corpus}/{spec.name}"] = tag_overhead + max(
            r.encoded_size() for r in reqs
        )
    return out


def build_plan(cfg: RunConfig, root: Path | None = None) -> dict:
    root = root or repo_root()
    md, sections, snap = load_constitution(cfg.resolve_constitution_dir(root))
    totals = dag_expanded_totals(cfg)
    a, s, e = cfg.advice, cfg.stories, cfg.engine

    typical_p, max_p = _payloads(cfg)
    by_len = sorted(sections, key=lambda x: x.end - x.start)
    typical_bytes = _measure_pass(cfg, root, md, sections, by_len[len(sections) // 2], typical_p)
    max_bytes = _measure_pass(cfg, root, md, sections, by_len[-1], max_p)

    n_scen, n_stories = totals["advice_scenarios"], totals["stories_drafts"]
    n_requests = {
        "advice/a1": 1,
        "advice/a2": a.n_principles,
        "advice/a3": a.n_principles * a.n_types,
        **{f"advice/{w}": n_scen for w in ("b1", "b2", "b3", "c", "d", "e")},
        "stories/s1": 1,
        "stories/s2": s.n_categories,
        "stories/s3": s.n_calls,
        "stories/s4": n_stories,
        "stories/s5": n_stories,
    }

    plan: dict = {
        "run_id": cfg.run_id,
        "constitution_sha256": snap["sha256"],
        "pack_target_bytes": e.pack_target_bytes,
        "hard_cap_bytes": e.hard_cap_bytes,
        "waves": {},
    }
    total_chunks = 0
    for wave, n in n_requests.items():
        t_bytes, m_bytes = typical_bytes[wave], max_bytes[wave]
        if m_bytes > e.hard_cap_bytes:
            raise PreflightRefusal(
                f"{wave}: a single max-size request encodes to {m_bytes} bytes "
                f"> hard cap {e.hard_cap_bytes} — refusing this plan"
            )
        capacity = max(1, min(e.max_requests_per_batch, e.pack_target_bytes // m_bytes))
        n_chunks = math.ceil(n / capacity)
        total_chunks += n_chunks
        plan["waves"][wave] = {
            "n_requests": n,
            "request_bytes_typical": t_bytes,
            "request_bytes_max": m_bytes,
            "wave_bytes_max": n * m_bytes,
            "chunk_capacity": capacity,
            "n_chunks": n_chunks,
        }
    plan["total_requests"] = sum(w["n_requests"] for w in plan["waves"].values())
    plan["total_chunks"] = total_chunks
    return plan


def run_preflight(cfg: RunConfig, root: Path | None = None) -> dict:
    root = root or repo_root()
    plan = build_plan(cfg, root)
    out_dir = cfg.resolve_data_dir(root) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(out_dir / f"preflight-{cfg.run_id}.json", json.dumps(plan, indent=1))
    return plan


def format_plan(plan: dict) -> str:
    lines = [
        f"preflight plan — run {plan['run_id']} "
        f"(pack {plan['pack_target_bytes'] // 2**20} MB, "
        f"refuse > {plan['hard_cap_bytes'] // 2**20} MB)",
        f"{'wave':<12} {'requests':>9} {'typ KB':>8} {'max KB':>8} "
        f"{'wave GB':>8} {'cap':>7} {'chunks':>7}",
    ]
    for wave, w in plan["waves"].items():
        lines.append(
            f"{wave:<12} {w['n_requests']:>9,} "
            f"{w['request_bytes_typical'] / 1024:>8.1f} "
            f"{w['request_bytes_max'] / 1024:>8.1f} "
            f"{w['wave_bytes_max'] / 2**30:>8.2f} "
            f"{w['chunk_capacity']:>7,} {w['n_chunks']:>7}"
        )
    lines.append(
        f"TOTAL: {plan['total_requests']:,} requests across ~{plan['total_chunks']} sub-batches"
    )
    return "\n".join(lines)
