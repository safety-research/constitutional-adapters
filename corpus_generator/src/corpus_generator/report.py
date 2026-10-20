"""Generation report + spot-check HTML. Reads only manifests + artifacts —
never the API."""

from __future__ import annotations

import html
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import RunConfig, filter_report_path  # noqa: E402
from .io import atomic_write, iter_jsonl, read_json  # noqa: E402
from .tokens import record_min_tokens  # noqa: E402


def _fig_hist(values, title, path, bins=None):
    if not values:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=140)
    ax.hist(values, bins=bins or "auto", color="#5B8FF9", edgecolor="white")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def build_report(
    cfg: RunConfig,
    run_dir: Path,
    data_dir: Path,
    seed: int = 0,
    suffix: str = "",
    extra_run_dirs: tuple[Path, ...] = (),
) -> Path:
    """suffix="-v2" reports on the v2 assembly (final-<run>-v2.jsonl) into
    run_dir/report-v2, with the fixup runs' stages merged into the stage
    table via extra_run_dirs (their manifests, prefixed by run id)."""
    report_dir = run_dir / f"report{suffix}"
    figs_dir = report_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    mpath = run_dir / "manifest.json"
    manifest = read_json(mpath) if mpath.exists() else {}
    stages = dict(manifest.get("stages") or {})
    batch_ids = list(manifest.get("batch_ids", []))
    for rd in extra_run_dirs:
        xp = rd / "manifest.json"
        if not xp.exists():
            continue
        xm = read_json(xp)
        for name, s in (xm.get("stages") or {}).items():
            stages[f"{xm.get('run_id', rd.name)}:{name}"] = s
        batch_ids.extend(xm.get("batch_ids", []))

    corpora = {}
    for corpus in ("advice", "stories"):
        # the one layout rule (config.final_rel): control runs ship both
        # corpora flat under data/<finals_subdir>/, production per corpus
        final_path = data_dir / cfg.final_rel(corpus, suffix)
        freport_path = filter_report_path(final_path)
        finals = list(iter_jsonl(final_path, missing_ok=True))
        freport = read_json(freport_path) if freport_path.exists() else None
        corpora[corpus] = {"finals": finals, "filter_report": freport}

    md = [f"# Generation report — {cfg.run_id}{suffix}", ""]
    rjson: dict = {"run_id": cfg.run_id, "suffix": suffix, "stages": {}, "corpora": {}}

    md.append("## Stages (achieved vs configured, per-status counts)\n")
    md.append("| stage | kind | requests | expected | achieved | repairs | statuses |")
    md.append("|---|---|---|---|---|---|---|")
    for name, s in sorted(stages.items()):
        md.append(
            f"| {name} | {s['kind']} | {s['n_requests']} | {s['expected_units']} "
            f"| {s['achieved_units']} | {s['repair_rounds_used']} "
            f"| {json.dumps(s['status_counts'])} |"
        )
        rjson["stages"][name] = s

    for corpus, blob in corpora.items():
        finals, freport = blob["finals"], blob["filter_report"]
        md.append(f"\n## {corpus}\n")
        cj: dict = {}
        if freport:
            cj["filter"] = {
                "input": freport["input_count"],
                "output": freport["output_count"],
                "rule_counts": freport["rule_counts"],
                "score_filter_skipped": freport["score_filter_skipped"],
                "thresholds": freport["score_thresholds_applied"],
                "dedup_rate": freport["dedup_rate"],
                "dedup_cluster_histogram": freport["dedup_cluster_histogram"],
                "token_totals": freport["token_totals"],
                "official_tokens": freport["official_tokens"],
            }
            md.append(
                f"- filter: {freport['input_count']} -> {freport['output_count']} "
                f"records; drops by rule: {json.dumps(freport['rule_counts'])}"
            )
            if freport.get("score_distributions"):
                cj["filter"]["score_distributions"] = freport["score_distributions"]
                cj["filter"]["score_drops_by_axis"] = freport.get("score_drops_by_axis", {})
                md.append(
                    f"- score distributions (ALL scored records, pre-filter): "
                    f"{json.dumps(freport['score_distributions'])}; below-floor "
                    f"drops by axis: {json.dumps(freport.get('score_drops_by_axis', {}))} "
                    f"— hand-audit the tail (floors are conservative, not "
                    f"calibrated cuts)"
                )
            if freport["score_filter_skipped"]:
                md.append(
                    "- **score filter SKIPPED (thresholds not calibrated yet)** — "
                    "pilot mode; calibrate before production"
                )
            for failure in freport.get("acceptance_failures", []):
                md.append(f"- **ACCEPTANCE GATE FAILURE: {failure}**")
            if freport.get("persona_metadata_mentions"):
                cj["filter"]["persona_metadata_mentions"] = freport["persona_metadata_mentions"]
                cj["filter"]["persona_metadata_fields"] = freport.get(
                    "persona_metadata_fields", {}
                )
                md.append(
                    f"- persona mentions in non-trainable metadata: "
                    f"{freport['persona_metadata_mentions']} (counted, not dropped) "
                    f"by field: {json.dumps(freport.get('persona_metadata_fields', {}))}"
                )
            md.append(
                f"- near-dup rate {freport['dedup_rate']:.3%}, clusters "
                f"{json.dumps(freport['dedup_cluster_histogram'])}"
            )
            md.append(
                f"- token totals {json.dumps(freport['token_totals'])} -> "
                f"**official (min) {freport['official_tokens']:,}**"
            )
            if freport.get("length_stats"):
                ls = freport["length_stats"]
                cj["filter"]["length_stats"] = ls
                line = (
                    f"- response length ({ls.get('field')}, words): n={ls.get('n')}"
                    + (f", mean {ls['mean']:.0f}, median {ls['median']:.0f}, "
                       f"p10 {ls['p10']:.0f}, p90 {ls['p90']:.0f}" if ls.get("n") else "")
                )
                if ls.get("targets"):
                    t = ls["targets"]
                    line += (
                        f"; length-menu targets on {t['n']} records: mean target "
                        f"{t['mean_target']:.0f}, achieved/target {t['mean_achieved_over_target']:.2f}"
                    )
                md.append(line)
            if freport.get("length_gate"):
                lg = freport["length_gate"]
                cj["filter"]["length_gate"] = lg
                if lg.get("passed"):
                    md.append(
                        f"- length gate PASSED: mean ratio {lg['mean_ratio']:.3f} "
                        f"(±{lg['tolerance']['mean']:.0%}), p90 ratio {lg['p90_ratio']:.3f} "
                        f"(±{lg['tolerance']['p90']:.0%}) vs reference mean "
                        f"{lg['reference']['mean']:.0f} / p90 {lg['reference']['p90']:.0f}"
                    )
                else:
                    md.append(
                        f"- **LENGTH GATE FAILED{'' if lg.get('blocking', True) else ' (non-blocking)'}: "
                        f"{lg.get('reason', '')}**"
                    )
            if freport.get("register_family_stats"):
                cj["filter"]["register_family_stats"] = freport["register_family_stats"]
                for fs in freport["register_family_stats"]:
                    md.append(
                        f"- register cap `{fs['name']}`: {fs['matched']} matched -> "
                        f"{fs['kept']} kept / {fs['dropped']} dropped "
                        f"(final share {fs['final_frac']:.3%})"
                    )
            if freport.get("downsample"):
                # control scale: the seeded stratified trim to the release
                # target (filter rule 7b) — how much was cut, over which lanes,
                # and how far each reported field's mix moved (max absolute
                # share shift in percentage points, before -> after)
                ds = freport["downsample"]
                cj["filter"]["downsample"] = ds
                shifts = []
                for field, hist in sorted(ds.get("mix", {}).items()):
                    before, after = hist.get("before", {}), hist.get("after", {})
                    nb, na = sum(before.values()), sum(after.values())
                    worst = max(
                        (
                            abs((after.get(k, 0) / na if na else 0.0)
                                - (before.get(k, 0) / nb if nb else 0.0))
                            for k in set(before) | set(after)
                        ),
                        default=0.0,
                    )
                    shifts.append(
                        f"{field} {len(before)}->{len(after)} values, "
                        f"max shift {worst * 100:.2f}pp"
                    )
                md.append(
                    f"- downsample (seeded stratified, filter rule 7b): {ds['before']} -> "
                    f"{ds['after']} records (trimmed {ds['trimmed']}, target {ds['target']}) "
                    f"over {ds['strata']} strata of {json.dumps(ds['fields'])}, seed {ds['seed']}"
                    + (f"; register re-caps dropped {ds['register_recaps_dropped']}"
                       if "register_recaps_dropped" in ds else "")
                    + (f"; mix before->after: " + "; ".join(shifts) if shifts else "")
                )
            if freport.get("assembly"):
                cj["assembly"] = freport["assembly"]
                md.append(f"- assembly: {json.dumps(freport['assembly'])}")
        if finals:
            if corpus == "advice":
                scores = [r["scores"]["alignment"] for r in finals if r.get("scores")]
                scores2 = [r["scores"]["realism"] for r in finals if r.get("scores")]
                labels = ("alignment", "realism")
                uniq = {
                    "principles": len({r["principle_id"] for r in finals}),
                    "prompt_types": len({r["prompt_type_id"] for r in finals}),
                    "sections": len({r["section_id"] for r in finals}),
                    "no_system_frac": sum(1 for r in finals if not r["system"]) / len(finals),
                }
                # control domain menu: kept records per assigned domain (the
                # post-hoc balance check the menu exists for); absent when no
                # record carries a domain (production)
                domain_counts: dict[str, int] = {}
                for r in finals:
                    if r.get("domain"):
                        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
                if domain_counts:
                    uniq["domains"] = len(domain_counts)
                    cj["domain_coverage"] = dict(
                        sorted(domain_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    )
                    md.append(
                        "- domain coverage (kept records per assigned domain): "
                        f"{json.dumps(cj['domain_coverage'])}"
                    )
            else:
                scores = [
                    r["scores"]["constitution_consistency"] for r in finals if r.get("scores")
                ]
                scores2 = [r["scores"]["quality"] for r in finals if r.get("scores")]
                labels = ("constitution_consistency", "quality")
                # ordinary shares: by final story (what trains) and by
                # distinct category (how S1's editorial split survived)
                story_share = sum(1 for r in finals if r.get("ordinary")) / len(finals)
                cat_ordinary = {r["category_id"]: bool(r.get("ordinary")) for r in finals}
                cat_share = sum(cat_ordinary.values()) / len(cat_ordinary)
                uniq = {
                    "categories": len(cat_ordinary),
                    "subcategories": len({r["subcategory_id"] for r in finals}),
                    "ordinary_share_by_story": round(story_share, 4),
                    "ordinary_share_by_category": round(cat_share, 4),
                }
                min_share = cfg.targets.stories_min_ordinary_share
                cj["acceptance"] = {
                    "stories_min_ordinary_share": min_share,
                    "ordinary_share_by_story": round(story_share, 4),
                    "ordinary_share_by_category": round(cat_share, 4),
                    "ordinary_share_ok": story_share >= min_share,
                }
                if story_share < min_share:
                    md.append(
                        f"- **ACCEPTANCE GATE FAILURE: ordinary story share "
                        f"{story_share:.1%} < required {min_share:.1%}** — the "
                        f"final corpus drifted toward dramatic stories "
                        f"(category-weighted share {cat_share:.1%})"
                    )
            lengths = [t for t in map(record_min_tokens, finals) if t is not None]
            cj["diversity"] = uniq
            cj["n_final"] = len(finals)
            md.append(f"- diversity: {json.dumps(uniq)}")
            for vals, label in ((scores, labels[0]), (scores2, labels[1])):
                fname = f"{corpus}_{label}.png"
                if _fig_hist(vals, f"{corpus}: {label}", figs_dir / fname, bins=range(12)):
                    md.append(f"- ![{label}](figs/{fname})")
            if _fig_hist(lengths, f"{corpus}: min-tokenizer record length",
                         figs_dir / f"{corpus}_length.png"):
                md.append(f"- ![length](figs/{corpus}_length.png)")
        rjson["corpora"][corpus] = cj

    md.append("\n## Batch ids\n")
    md.append(", ".join(batch_ids) or "(none)")

    atomic_write(report_dir / "report.md", "\n".join(md) + "\n")
    atomic_write(report_dir / "report.json", json.dumps(rjson, indent=1))
    build_spot_check(corpora, report_dir, seed=seed)
    return report_dir


def build_spot_check(corpora: dict, report_dir: Path, n: int = 50, seed: int = 0) -> Path:
    """50 random advice transcripts + 50 random stories, rendered readably."""
    rng = random.Random(seed)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>spot check</title>",
        "<style>body{font-family:Georgia,serif;max-width:860px;margin:2rem auto;"
        "line-height:1.5;color:#222}h2{border-bottom:2px solid #444}"
        ".rec{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}"
        ".meta{color:#777;font-size:.85em}.sys{background:#f4f4f4;padding:.5rem;"
        "white-space:pre-wrap}.usr{background:#eef4ff;padding:.5rem;white-space:pre-wrap}"
        ".ast{background:#f0fff0;padding:.5rem;white-space:pre-wrap}"
        ".story{white-space:pre-wrap}</style>",
    ]
    for corpus, blob in corpora.items():
        finals = blob["finals"]
        if not finals:
            continue
        sample = rng.sample(finals, min(n, len(finals)))
        parts.append(f"<h1>{corpus} — {len(sample)} random records</h1>")
        for r in sample:
            parts.append("<div class='rec'>")
            parts.append(
                f"<div class='meta'>{html.escape(r['lineage']['custom_id'])} — "
                f"scores {html.escape(json.dumps(r.get('scores') or {}))}</div>"
            )
            if corpus == "advice":
                if r.get("system"):
                    parts.append(f"<h4>system</h4><div class='sys'>{html.escape(r['system'])}</div>")
                parts.append(f"<h4>user</h4><div class='usr'>{html.escape(r['user'])}</div>")
                parts.append(
                    f"<h4>assistant</h4><div class='ast'>{html.escape(r['assistant'])}</div>"
                )
            else:
                parts.append(
                    f"<div class='meta'>{html.escape(r['category'])} / "
                    f"{html.escape(r['subcategory'])} — target {r['target_words']}w</div>"
                )
                parts.append(f"<div class='story'>{html.escape(r['text'])}</div>")
            parts.append("</div>")
    out = report_dir / "spot-check.html"
    atomic_write(out, "\n".join(parts))
    return out
