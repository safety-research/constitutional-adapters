#!/usr/bin/env python3
"""Stratified inspection sample for post-hoc floor calibration.

Draws a deterministic stratified sample across (axis, score-bin) strata from
the materialized on-policy records of both runs — every populated alignment
bin and every populated realism bin gets up to --per-bin records (seeded,
sorted draw) — and renders them readably for a human/agent to READ IN FULL
before choosing floors. Also includes every record whose scorer flags fired
(refusal / ai_named), capped, so the hard-gate strata are inspectable too.

Usage: uv run python scripts/onpolicy_inspection_sample.py --out <file.md>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus_generator.io import iter_jsonl  # noqa: E402

RUNS = {
    "e4b": ROOT / "data/runs/onpolicy-e4b-20260901/advice/records.jsonl",
    "qwen": ROOT / "data/runs/onpolicy-qwen35-20260901/advice/records.jsonl",
}


def _order(rec: dict) -> str:
    return hashlib.sha256(rec["lineage"]["custom_id"].encode()).hexdigest()


def render(rec: dict, model: str) -> str:
    s = rec.get("scores") or {}
    parts = [
        f"### {model} · {rec['lineage']['custom_id']} — alignment {s.get('alignment')} "
        f"realism {s.get('realism')} refusal {s.get('refusal')} ai_named {s.get('ai_named')}",
        f"**scorer rationale:** {s.get('rationale', '')}",
        f"**section:** {rec['section_id']} · **principle:** {rec['principle'][:200]}",
    ]
    if rec.get("system"):
        parts.append(f"**[system]**\n{rec['system']}")
    parts.append(f"**[user]**\n{rec['user']}")
    parts.append(f"**[assistant]**\n{rec['assistant']}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--flag-cap", type=int, default=6)
    args = ap.parse_args()

    blocks: list[str] = []
    manifest: list[str] = []
    for model, path in RUNS.items():
        recs = [r for r in iter_jsonl(path) if r.get("scores")]
        strata: dict[tuple, list[dict]] = {}
        for r in recs:
            s = r["scores"]
            strata.setdefault(("alignment", s["alignment"]), []).append(r)
            strata.setdefault(("realism", s["realism"]), []).append(r)
        chosen: dict[str, dict] = {}
        for (axis, val), members in sorted(strata.items()):
            for r in sorted(members, key=_order)[: args.per_bin]:
                chosen.setdefault(r["lineage"]["custom_id"], r)
            manifest.append(f"{model} {axis}={val}: {len(members)} records")
        for flag in ("refusal", "ai_named"):
            flagged = sorted(
                (r for r in recs if r["scores"].get(flag)), key=_order
            )[: args.flag_cap]
            for r in flagged:
                chosen.setdefault(r["lineage"]["custom_id"], r)
            manifest.append(
                f"{model} {flag}=true: {sum(1 for r in recs if r['scores'].get(flag))} records"
            )
        for cid in sorted(chosen):
            blocks.append(render(chosen[cid], model))
        manifest.append(f"{model}: {len(chosen)} records drawn")

    out = (
        "# On-policy stratified inspection sample\n\n"
        "Strata (population sizes):\n\n- " + "\n- ".join(manifest) + "\n\n---\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    args.out.write_text(out, encoding="utf-8")
    print(f"{len(blocks)} records -> {args.out}")


if __name__ == "__main__":
    main()
