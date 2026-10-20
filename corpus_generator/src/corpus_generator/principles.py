"""Frozen, hand-reviewed A1 principles.

A1 is a single sampled generation that seeds every advice lane, so the plan
gates the full run on a HUMAN read of the pilot's A1 output and then reuses
the reviewed set verbatim rather than sampling a fresh, unreviewed one. The
mechanism rides the stage runner's own resume path: ``seed_a1_stage`` writes
``advice/a1/outputs.jsonl`` + ``STAGE.json`` into the run namespace with the
``stage_hash`` the runner would compute for the run's real A1 request set —
so ``run_stage`` finds a completed, hash-matching stage and skips the wave,
and a later change to the A1 template, the placebo, the menu, or
``n_principles`` fails loudly as a stage-hash mismatch exactly like any other
mutation of a completed stage.

The frozen file is a committed JSON artifact (``configs/*-principles.json``)
so the review — and any hand edits made during it — is itself reviewable:

    {"source": {run_id, stage, batch_id, outputs_sha256, reviewed, edited, ...},
     "principles": {"p00": {"principle", "tension", "section_id"}, ...}}

The flat ``pid -> {...}`` form ``regen-advice`` uses is accepted too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .advice import advice_stages
from .batch import request_set_hash
from .config import RunConfig
from .io import atomic_write, read_json, write_jsonl_atomic
from .pipeline import build_ctx
from .stages import OutputRow, normalize_content
from .templates import repo_root

REQUIRED = ("principle", "tension", "section_id")


class PrinciplesError(RuntimeError):
    pass


def load_frozen_principles(path: Path) -> tuple[dict[str, dict], dict]:
    """(principles, source) from a frozen-principles file (wrapped or flat)."""
    data = read_json(path)
    if "principles" in data and isinstance(data["principles"], dict):
        principles, source = data["principles"], dict(data.get("source") or {})
    else:
        principles, source = data, {}
    for pid, d in principles.items():
        for field in REQUIRED:
            if not (isinstance(d.get(field), str) and d[field].strip()):
                raise PrinciplesError(f"principle {pid}: missing or empty {field!r}")
    source.setdefault("path", str(path))
    source["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return principles, source


def seed_a1_stage(
    cfg: RunConfig,
    principles: dict[str, dict],
    source: dict,
    root: Path | None = None,
    on_event=print,
) -> Path:
    """Materialize the run's A1 stage from a frozen principle set. Refuses if
    the stage already exists (completed stages are immutable), if the set does
    not match the configured fanout slot-for-slot, if a section_id is not in
    the run's document, or if two principles are the same text."""
    root = root or repo_root()
    ctx, _run_dir = build_ctx(cfg, "advice", root)
    spec = advice_stages(None)[0]
    assert spec.name == "a1"
    stage_dir = ctx.corpus_dir / spec.name
    for marker in ("STAGE.json", "REQUEST_SET.json", "outputs.jsonl"):
        if (stage_dir / marker).exists():
            raise PrinciplesError(
                f"{stage_dir / marker} exists — the A1 stage of run {cfg.run_id} "
                f"already has artifacts; a seeded stage must be the first thing in "
                f"its namespace"
            )

    requests = spec.build_requests(ctx)
    template_hashes = ctx.templates.hashes(spec.templates)
    stage_hash = request_set_hash(requests, template_hashes)
    (req,) = requests
    expected = list(spec.expected_units(req))
    if sorted(principles) != sorted(expected):
        raise PrinciplesError(
            f"frozen set has slots {sorted(principles)[:3]}… ({len(principles)}) but the "
            f"config's A1 fanout expects exactly {expected[0]}..{expected[-1]} "
            f"({len(expected)}) — n_principles and the frozen file must agree"
        )
    section_ids = {s.id for s in ctx.sections}
    seen: dict[str, str] = {}
    for pid in expected:
        d = principles[pid]
        if d["section_id"] not in section_ids:
            raise PrinciplesError(
                f"principle {pid}: section_id {d['section_id']!r} is not a section of "
                f"{cfg.constitution_dir}"
            )
        folded = normalize_content(d["principle"])
        if folded in seen:
            raise PrinciplesError(f"principles {seen[folded]} and {pid} are the same text")
        seen[folded] = pid

    provenance = {**source, "seeded_at_stage_hash": stage_hash}
    rows = [
        OutputRow(
            key=pid,
            custom_id=req.custom_id,  # the logical A1 request this stage stands in for
            status="ok",
            data={
                "principle": principles[pid]["principle"].strip(),
                "section_id": principles[pid]["section_id"],
                "tension": principles[pid]["tension"].strip(),
            },
            lineage={**req.lineage, "seeded_from": provenance},
        )
        for pid in expected
    ]
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(
        stage_dir / "outputs.jsonl",
        (json.dumps(r.to_json(), ensure_ascii=False) for r in rows),
    )
    atomic_write(
        stage_dir / "STAGE.json",
        json.dumps(
            {
                "stage": spec.name,
                "corpus": spec.corpus,
                "kind": spec.kind,
                "stage_hash": stage_hash,
                "n_requests": 0,  # nothing was submitted for this stage
                "expected_units": len(expected),
                "achieved_units": len(rows),
                "repair_rounds_used": 0,
                "status_counts": {"ok": len(rows)},
                "template_hashes": template_hashes,
                "seeded_from": provenance,
            },
            indent=1,
        ),
    )
    on_event(
        f"seeded {spec.corpus}/{spec.name} of run {cfg.run_id} with {len(rows)} frozen "
        f"principles from {source.get('run_id', '?')} (stage hash {stage_hash[:12]}); "
        f"`corpusgen run` will resume past it"
    )
    return stage_dir
