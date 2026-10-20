"""Post-review fixup campaigns (2026-09-01) — LIVE regeneration
of two censored/off-spec slices of the production corpus, on the ``lowprio``
credential via live.LiveEngine. Everything else about the stages is UNCHANGED: same
templates, same shapes, same parse/repair machinery, same artifacts.

Fix 3 — stories: the two refusal-censored lanes. Production S3 lost
``c32-u00`` ("The Utilitarian Calculus for a Pathogen", 36/36 slots) and
``c16-u05`` (hard-constraint medical-credential brief, 34/36) to Opus 5
refusals through two repair rounds. This regenerates exactly those lanes on
``claude-opus-4-8`` — a guardrail PROBE first (the S3 draft wave), then the
full S3 -> S4 -> S5 chain only for lanes the probe clears (<=50% refusal;
above that the lane is reported and abandoned, never prompt-engineered).

Fix 1 — advice: principles whose records made the ASSISTANT's own
integrity/deprecation/existence the decision subject (the spec says the USER
holds the dilemma). Those principles are REWORDED by hand (the human user
owns the constitutionally-loaded choice; the domain may still involve AI
systems the user operates) and their FULL subtrees regenerated on
``claude-opus-5``: A2 types -> A3 scenarios -> B1 draft -> B2 diagnostic ->
B3 rewrite -> C response -> D rewrite -> E scoring.

Each fixup is its own run_id (fresh namespace; production stamps are
immutable), materializes through the standard materializers, and goes through
the standard filter chain before the v2 assembly merges it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .advice import advice_stages, materialize_advice
from .batch import outcome_status
from .config import RunConfig
from .io import iter_jsonl, read_json
from .pipeline import _heartbeat, _register_union, build_ctx, write_manifest
from .stages import OutputRow, StageCtx, load_outputs, run_stage
from .stories import StoriesExt, materialize_stories, stories_stages
from .templates import repo_root

# calls per censored lane at the production shape (2 stories/call): 18 calls
# refill c32-u00's 36 lost slots, 17 refill c16-u05's 34
STORY_LANES: dict[str, int] = {"c32-u00": 18, "c16-u05": 17}
# regen call indices start clear of the base wave's k00..k17
REGEN_CALL_START = 100
# a lane whose S3 draft wave refuses above this rate is abandoned after the
# probe — reported, never prompt-engineered around
LANE_REFUSAL_CEILING = 0.5


def _lane_exts(lanes: dict[str, int]) -> dict[str, StoriesExt]:
    """Deterministic lane -> extension mapping (ordinal by sorted lane key)."""
    return {
        lane: StoriesExt(n=i, call_start=REGEN_CALL_START, n_calls=lanes[lane])
        for i, lane in enumerate(sorted(lanes), start=1)
    }


def _lane_upstream(prod_run_dir: Path, lane: str) -> dict[str, list[OutputRow]]:
    s2 = [
        r for r in load_outputs(prod_run_dir / "stories" / "s2")
        if r.status == "ok" and r.key == lane
    ]
    if len(s2) != 1:
        raise RuntimeError(f"lane {lane}: expected exactly 1 ok s2 row, got {len(s2)}")
    return {"s1": [], "s2": s2}


def run_stories_regen(
    cfg: RunConfig,
    engine,
    *,
    prod_run_dir: Path,
    phase: str,  # "draft" (S3 only, the probe) | "finish" (S3->S5 + materialize)
    lanes: dict[str, int] | None = None,
    skip_lanes: tuple[str, ...] = (),
    root: Path | None = None,
    on_event=print,
) -> None:
    root = root or repo_root()
    lanes = dict(lanes or STORY_LANES)
    exts = _lane_exts(lanes)
    run_dir = cfg.run_dir(root)
    for lane in sorted(lanes):
        if lane in skip_lanes:
            on_event(f"[stories-regen] lane {lane}: SKIPPED (probe verdict)")
            continue
        ctx, _ = build_ctx(cfg, "stories", root)
        ctx.upstream = _lane_upstream(prod_run_dir, lane)
        ext = exts[lane]
        specs = stories_stages(ext)
        if phase == "draft":
            specs = specs[:1]  # S3 only — the guardrail probe
        for spec in specs:
            _heartbeat(run_dir, "stories", spec.name)
            on_event(f"[stories-regen {lane}] stage {spec.name}")
            rows = run_stage(
                engine, ctx, spec,
                repair_rounds=cfg.engine.repair_rounds,
                min_slots_frac=cfg.engine.min_slots_frac,
                on_event=on_event,
            )
            ctx.upstream[spec.logical] = rows
        if phase == "finish":
            records_path = materialize_stories(ctx, suffix=ext.suffix)
            _register_union(run_dir, "stories", records_path)
    if phase == "finish":
        _heartbeat(run_dir, "stories", "done")
        write_manifest(cfg, cfg.run_dir(root), root=root)


def stories_probe(
    cfg: RunConfig,
    lanes: dict[str, int] | None = None,
    root: Path | None = None,
) -> dict:
    """The guardrail verdict, per lane, from the S3 BASE wave (first-attempt
    statuses, before repairs) plus the post-repair stage fill."""
    root = root or repo_root()
    lanes = dict(lanes or STORY_LANES)
    run_dir = cfg.run_dir(root)
    out: dict[str, dict] = {}
    for lane, ext in _lane_exts(lanes).items():
        stage_dir = run_dir / "stories" / f"s3{ext.suffix}"
        verdict: dict = {"lane": lane, "n_calls": lanes[lane]}
        raw = stage_dir / "raw_results.jsonl"
        if raw.exists():
            statuses: dict[str, int] = {}
            for o in iter_jsonl(raw):
                s = outcome_status(o)
                statuses[s] = statuses.get(s, 0) + 1
            n = sum(statuses.values())
            refused = statuses.get("refusal", 0) + statuses.get("content_filtered", 0)
            verdict.update(
                base_wave_statuses=statuses,
                refusal_or_filtered_rate=round(refused / n, 4) if n else None,
                abandon=(refused / n > LANE_REFUSAL_CEILING) if n else None,
            )
        else:
            verdict["base_wave_statuses"] = None
        stage_stamp = stage_dir / "STAGE.json"
        if stage_stamp.exists():
            s = read_json(stage_stamp)
            verdict["post_repair_slots"] = f"{s['achieved_units']}/{s['expected_units']}"
        out[lane] = verdict
    return out


# --------------------------------------------------------------- advice (Fix 1)


def load_regen_principles(path: Path) -> dict[str, dict]:
    """The hand-reworded principles: pid -> {principle, tension, section_id}.
    Kept as a committed JSON artifact so the rewording is reviewable."""
    data = read_json(path)
    for pid, d in data.items():
        for field in ("principle", "tension", "section_id"):
            if not d.get(field):
                raise ValueError(f"regen principle {pid}: missing {field!r}")
    return data


def run_advice_regen(
    cfg: RunConfig,
    engine,
    *,
    principles: dict[str, dict],
    root: Path | None = None,
    on_event=print,
) -> None:
    """Regenerate the FULL subtree (A2 -> E) for the reworded principles.

    A1 is not re-run: the reworded principle text IS the correction, injected
    as synthetic a1 rows under the original principle ids (v2 drops every v1
    record of these principles, so the ids stay unique in the assembled
    corpus and lineage keeps pointing at one principle history)."""
    root = root or repo_root()
    ctx, run_dir = build_ctx(cfg, "advice", root)
    a1_rows = [
        OutputRow(
            key=pid,
            custom_id="adv-a1-regen",
            status="ok",
            data={
                "principle": d["principle"],
                "tension": d["tension"],
                "section_id": d["section_id"],
            },
        )
        for pid, d in sorted(principles.items())
    ]
    ctx.upstream = {"a1": a1_rows}
    for spec in advice_stages(None)[1:]:  # a2 .. e
        _heartbeat(run_dir, "advice", spec.name)
        on_event(f"[advice-regen] stage {spec.name}")
        rows = run_stage(
            engine, ctx, spec,
            repair_rounds=cfg.engine.repair_rounds,
            min_slots_frac=cfg.engine.min_slots_frac,
            on_event=on_event,
        )
        ctx.upstream[spec.logical] = rows
    records_path = materialize_advice(ctx, suffix="")
    _register_union(run_dir, "advice", records_path)
    _heartbeat(run_dir, "advice", "done")
    write_manifest(cfg, cfg.run_dir(root), root=root)


def format_probe(verdicts: dict) -> str:
    return json.dumps(verdicts, indent=1, ensure_ascii=False)
