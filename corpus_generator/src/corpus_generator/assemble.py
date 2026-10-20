"""v2 corpus assembly (post-review fixes, 2026-09-01): merge the immutable v1
finals with the regenerated additions, drop the records the review ordered
out, and run the FULL filter chain (now including the register caps) over the
merged set — so every v2 record re-passes schema, persona, hard gates, sanity,
frozen score floors, dedup, and the new register rules, and the v2 filter
report carries every count.

v1 finals are never touched: the assembly READS final-<run>.jsonl and WRITES
final-<run><suffix>.jsonl (default suffix "-v2") next to it.

Drop semantics (Fix 1): --drop-principle removes every record of that
principle belonging to the BASE run (the regenerated subtree replaces it under
the same principle/scenario ids, so dropping all of the old ones is also what
keeps ids unique in the merged corpus). Counted in the report as rule
"reframed_principle", listed before the chain's own rules.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import RunConfig, filter_report_path
from .filtering import FilterReport, filter_corpus
from .io import atomic_write, iter_jsonl, read_json
from .tokens import TokenCounter


def v2_final_path(cfg: RunConfig, root: Path, corpus: str, suffix: str = "-v2") -> Path:
    base = cfg.final_path(root, corpus)
    return base.with_name(f"{base.stem}{suffix}{base.suffix}")


def assemble_v2(
    cfg: RunConfig,
    corpus: str,
    *,
    root: Path,
    add_sources: list[Path],
    drop_principles: set[str] | None = None,
    drop_run: str | None = None,
    drop_ids: set[str] | None = None,  # residual per-record drops (any source)
    suffix: str = "-v2",
    counter: TokenCounter | None = None,
    on_event=print,
) -> FilterReport:
    base_final = cfg.final_path(root, corpus)
    sources = [base_final, *add_sources]
    records: list[dict] = []
    for p in sources:
        if not p.exists():
            raise FileNotFoundError(f"assembly source missing: {p}")
        n_before = len(records)
        records.extend(iter_jsonl(p))
        on_event(f"[assemble {corpus}] {p.name}: {len(records) - n_before} records")

    drop_principles = drop_principles or set()
    drop_ids = drop_ids or set()
    dropped: list[dict] = []
    kept: list[dict] = []
    for r in records:
        run_id = r.get("lineage", {}).get("run_id")
        cid = r["lineage"]["custom_id"]
        # principle-level drop first: a v1 record of a reworded principle is
        # dropped AS that, even when a regen residual reuses its custom_id
        if (
            corpus == "advice"
            and r.get("principle_id") in drop_principles
            and (drop_run is None or run_id == drop_run)
        ):
            dropped.append(
                {"id": cid, "rule": "reframed_principle",
                 "detail": f"{r['principle_id']} reworded; v1 subtree replaced"}
            )
        elif cid in drop_ids:
            dropped.append(
                {"id": cid, "rule": "ai_subject_residual",
                 "detail": "assistant-as-subject framing survived regeneration"}
            )
        else:
            kept.append(r)
    if drop_principles:
        on_event(
            f"[assemble {corpus}] reframed_principle drops: {len(dropped)} "
            f"({', '.join(sorted(drop_principles))} from run {drop_run})"
        )

    # duplicate identity check across sources — a regenerated id may exist at
    # most once after the drops (uniqueness is what the drop-all rule buys)
    keys = [f"{r['lineage']['run_id']}:{r['lineage']['custom_id']}" for r in kept]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise RuntimeError(f"assembly {corpus}: duplicate record keys {dupes[:5]}")
    bare = [r["lineage"]["custom_id"] for r in kept]
    if len(set(bare)) != len(bare):
        dupes = sorted({k for k in bare if bare.count(k) > 1})
        raise RuntimeError(
            f"assembly {corpus}: custom_id collision across runs {dupes[:5]} — "
            f"did a regenerated subtree land without dropping its v1 records?"
        )

    out_path = v2_final_path(cfg, root, corpus, suffix)
    report = filter_corpus(corpus, kept, cfg, counter, out_path)

    # fold the assembly-level drops into the persisted report so one artifact
    # carries the whole v1 -> v2 delta
    pre_counts: dict[str, int] = {}
    for d in dropped:
        pre_counts[d["rule"]] = pre_counts.get(d["rule"], 0) + 1
    report.rule_counts.update(pre_counts)
    report.drops = dropped + report.drops
    report.input_count = len(records)
    rp = filter_report_path(out_path)
    data = read_json(rp)
    data["rule_counts"].update(pre_counts)
    data["drops"] = (dropped + data["drops"])[:5000]
    data["input_count"] = len(records)
    data["assembly"] = {
        "sources": [str(p) for p in sources],
        "drop_principles": sorted(drop_principles),
        "drop_run": drop_run,
        "drop_ids": sorted(drop_ids),
    }
    atomic_write(rp, json.dumps(data, indent=1, ensure_ascii=False))
    on_event(
        f"[assemble {corpus}] {len(records)} in -> {report.output_count} out, "
        f"official tokens {report.official_tokens:,}"
    )
    return report
