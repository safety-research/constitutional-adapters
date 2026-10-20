#!/usr/bin/env python3
"""Section-coverage analysis of the content tags: stats JSON, figures, and
the markdown report.

    uv run python scripts/section_coverage_report.py [--release production-20260831-v2]
        [--figs <figs-dir>/corpus_generator-production] [--md <path>]

Reads data/{stories,advice}/section-tags-<release>.jsonl (+ manifests), the
finals (for the advice lineage section and record counts), and the frozen
index; writes data/manifests/section-coverage-<release>.json (the one
computation the showcase builder reads), four figures, and the report.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus_generator import section_coverage as sc  # noqa: E402
from corpus_generator import section_tags as st  # noqa: E402
from corpus_generator.io import iter_jsonl, read_json, atomic_write  # noqa: E402

# dataviz reference palette (light): categorical slots 1-3, ink tokens, chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SERIES = [
    ("stories", "Stories — content tags (primary)", BLUE),
    ("advice", "Advice — content tags (primary)", ORANGE),
    ("advice_lineage", "Advice — construction lineage", AQUA),
]

FONT = ["system-ui", "DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"]


def load_corpus(release: str, corpus: str) -> tuple[list[dict], dict, list[dict]]:
    tags_path = ROOT / "data" / corpus / f"section-tags-{release}.jsonl"
    man_path = ROOT / "data" / corpus / f"section-tags-{release}.manifest.json"
    final = ROOT / "data" / corpus / f"final-{release}.jsonl"
    if not tags_path.exists():
        raise SystemExit(f"missing {tags_path}")
    rows = list(iter_jsonl(tags_path))
    manifest = read_json(man_path)
    records = list(iter_jsonl(final))
    return rows, manifest, records


def compute(release: str) -> dict:
    entries, index_meta = st.load_index(ROOT / "constitution")
    ids = [e.id for e in entries]
    title = {e.id: e.title for e in entries}
    out: dict = {
        "release": release,
        "index_sha256": index_meta["sha256"],
        "constitution_sha256": index_meta["constitution_sha256"],
        "sections": [{"id": e.id, "title": e.title, "level": e.level, "parent": e.parent, "chapter": e.chapter}
                     for e in entries],
        "corpora": {},
    }
    for corpus in ("stories", "advice"):
        rows, manifest, records = load_corpus(release, corpus)
        ok = sc.ok_rows(rows)
        n_ok = len(ok)
        c = {
            "n_records": len(rows),
            "n_ok": n_ok,
            "status_counts": manifest["status_counts"],
            "n_requests_issued": manifest["n_requests_issued"],
            "wall_clock_s": manifest["wall_clock_s"],
            "started_at": manifest["started_at"], "ended_at": manifest["ended_at"],
            "model": manifest["model"],
            "lenient_repairs": manifest.get("lenient_repairs", {}),
            "n_by_repair_round": manifest.get("n_by_repair_round", {}),
            "counts": {v: sc.section_counts(ok, ids, v) for v in ("primary", "any", "lead")},
            "chapters": {v: sc.chapter_counts(ok, entries, v) for v in ("primary", "any", "lead")},
            "zero_primary": sc.zero_sections(sc.section_counts(ok, ids, "primary")),
            "zero_any": sc.zero_sections(sc.section_counts(ok, ids, "any")),
            "under_1pct_primary": sc.under_represented(sc.section_counts(ok, ids, "primary"), n_ok, 0.01),
            "top2": sc.top2_concentration(ok, ids),
            "helpfulness": sc.cluster_coverage(ok, entries),
            "none": sc.none_rate(ok),
            "per_record": sc.sections_per_record(ok),
            "confidence": sc.confidence_summary(ok),
        }
        c["n_sections_primary"] = sum(1 for v in c["counts"]["primary"].values() if v)
        c["n_sections_any"] = sum(1 for v in c["counts"]["any"].values() if v)
        if corpus == "advice":
            c["lineage_counts"] = sc.lineage_counts(records, ids)
            c["lineage_chapters"] = sc.lineage_chapter_counts(records, entries)
            c["n_sections_lineage"] = sum(1 for v in c["lineage_counts"].values() if v)
            c["agreement"] = sc.lineage_agreement(ok, entries)
        out["corpora"][corpus] = c
    out["titles"] = title
    return out


# ----------------------------------------------------------------- figures --


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 9,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": False, "legend.frameon": False, "text.color": INK,
    })
    return plt


def fig_by_section(stats: dict, path: Path) -> None:
    plt = _style()
    import numpy as np

    secs = stats["sections"]
    ids = [s["id"] for s in secs]
    n_s = stats["corpora"]["stories"]["n_ok"]
    n_a = stats["corpora"]["advice"]["n_ok"]
    n_l = stats["corpora"]["advice"]["n_records"]
    series = {
        "stories": [100 * stats["corpora"]["stories"]["counts"]["primary"][i] / n_s for i in ids],
        "advice": [100 * stats["corpora"]["advice"]["counts"]["primary"][i] / n_a for i in ids],
        "advice_lineage": [100 * stats["corpora"]["advice"]["lineage_counts"][i] / n_l for i in ids],
    }
    fig, ax = plt.subplots(figsize=(11, 15.5))
    y = np.arange(len(ids))
    h = 0.26
    # chapter bands: alternate a faint wash per chapter so clusters read as groups
    chapters = [s["chapter"] for s in secs]
    start = 0
    for k in range(1, len(ids) + 1):
        if k == len(ids) or chapters[k] != chapters[start]:
            if (sum(1 for j in range(start) if j == 0 or chapters[j] != chapters[j - 1]) % 2) == 1:
                ax.axhspan(start - 0.5, k - 0.5, color="#f1f1ee", zorder=0, lw=0)
            if k < len(ids):
                ax.axhline(k - 0.5, color=GRID, lw=0.8, zorder=1)
            start = k
    for j, (key, label, color) in enumerate(SERIES):
        vals = series[key]
        ax.barh(y + (j - 1) * h, vals, height=h * 0.88, color=color, label=label, zorder=3, lw=0)
        # selective direct labels: the two largest bars of each series
        for i in sorted(range(len(vals)), key=lambda i: -vals[i])[:2]:
            ax.text(vals[i] + 0.6, y[i] + (j - 1) * h, f"{vals[i]:.0f}%", va="center", ha="left",
                    fontsize=8, color=INK2, zorder=4)
    labels = []
    for s in secs:
        depth = s["level"] - 2
        t = s["title"]
        labels.append(("    " * depth) + t if depth else t.upper())
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    for tick, s in zip(ax.get_yticklabels(), secs):
        if s["parent"] is None:
            tick.set_color(INK)
            tick.set_fontweight("bold")
            tick.set_fontsize(8)
    ax.invert_yaxis()
    ax.set_ylim(len(ids) - 0.5, -0.6)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("share of records tagged with the section (%)", color=INK2)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    xmax = max(max(v) for v in series.values())
    ax.set_xlim(0, xmax * 1.12)
    fig.text(0.03, 0.978, "Constitution section coverage — stories vs advice, by content and by lineage",
             fontsize=12, color=INK, va="top")
    fig.text(0.03, 0.962,
             f"Records with the section among their primary content tags (stories n={n_s:,}, advice n={n_a:,}) "
             f"vs the section each advice record was seeded from (n={n_l:,}).\n"
             f"Sections in the constitution's own order; CAPITALS are chapters, indented rows their subsections.",
             fontsize=8.5, color=INK2, va="top", linespacing=1.5)
    ax.legend(loc="lower right", fontsize=8.5, handlelength=1.2, borderaxespad=0.6)
    fig.subplots_adjust(left=0.36, right=0.98, top=0.925, bottom=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_chapters(stats: dict, path: Path) -> None:
    plt = _style()
    import numpy as np

    chapters = [s for s in stats["sections"] if s["parent"] is None]
    ids = [c["id"] for c in chapters]
    n_s = stats["corpora"]["stories"]["n_ok"]
    n_a = stats["corpora"]["advice"]["n_ok"]
    n_l = stats["corpora"]["advice"]["n_records"]
    series = {
        "stories": [100 * stats["corpora"]["stories"]["chapters"]["primary"][i] / n_s for i in ids],
        "advice": [100 * stats["corpora"]["advice"]["chapters"]["primary"][i] / n_a for i in ids],
        "advice_lineage": [100 * stats["corpora"]["advice"]["lineage_chapters"][i] / n_l for i in ids],
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(ids))
    w = 0.26
    import textwrap
    for j, (key, label, color) in enumerate(SERIES):
        vals = series[key]
        ax.bar(x + (j - 1) * w, vals, width=w * 0.88, color=color, label=label, zorder=3, lw=0)
        for i, v in enumerate(vals):
            if v >= 1:
                ax.text(x[i] + (j - 1) * w, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(c["title"], 16) for c in chapters], fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.set_ylabel("share of records with ≥1 primary tag in the chapter (%)", color=INK2, fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_ylim(0, 108)
    ax.set_title("Coverage by constitution chapter (records touching the chapter, primary tags)",
                 loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.18)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_per_record(stats: dict, path: Path) -> None:
    plt = _style()
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [4, 7]})
    for ax, key, ks in ((axes[0], "primary", range(0, 4)), (axes[1], "total", range(0, 7))):
        x = np.arange(len(ks))
        w = 0.36
        for j, (corpus, label, color) in enumerate((("stories", "Stories", BLUE), ("advice", "Advice", ORANGE))):
            pr = stats["corpora"][corpus]["per_record"]
            n = pr["n"]
            vals = [100 * pr[key].get(str(k), 0) / n for k in ks]
            ax.bar(x + (j - 0.5) * w, vals, width=w * 0.9, color=color, label=label, zorder=3, lw=0)
            for i, v in enumerate(vals):
                if v >= 3:
                    ax.text(x[i] + (j - 0.5) * w, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                            fontsize=7.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_ylim(0, 105)
        ax.set_xlabel("primary sections per record" if key == "primary" else
                      "primary + secondary sections per record", color=INK2)
    axes[0].set_ylabel("share of records (%)", color=INK2)
    axes[1].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("How many constitution sections a record exercises (tagger caps: 3 primary, 3 secondary)",
                 x=0.06, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.16, wspace=0.18)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_lineage_confusion(stats: dict, path: Path) -> None:
    plt = _style()
    import numpy as np
    from matplotlib.colors import ListedColormap

    adv = stats["corpora"]["advice"]
    order = [s["id"] for s in stats["sections"]]
    title = stats["titles"]
    conf = adv["agreement"]["confusion"]
    rows_ids = [i for i in order if i in conf]
    col_mass = {}
    for lin, d in conf.items():
        for lead, n in d.items():
            col_mass[lead] = col_mass.get(lead, 0) + n
    cols = [i for i in order if col_mass.get(i, 0) > 0]
    if "(none)" in col_mass:
        cols.append("(none)")
    M = np.zeros((len(rows_ids), len(cols)))
    for r, lin in enumerate(rows_ids):
        tot = sum(conf[lin].values())
        for c, lead in enumerate(cols):
            M[r, c] = 100 * conf[lin].get(lead, 0) / tot if tot else 0
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 5.2, 0.38 * len(rows_ids) + 2.6))
    cmap = ListedColormap(["#f1f1ee"] + SEQ[1:])  # zero cells stay visible as a faint grid
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(60, M.max()), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([title.get(c, c) for c in cols], rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows_ids)))
    ax.set_yticklabels([f"{title[r]}  (n={sum(conf[r].values())})" for r in rows_ids], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # hairline cell separators in the surface color (the 2px gap rule)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows_ids), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    ax.tick_params(which="minor", length=0)
    for r, lin in enumerate(rows_ids):
        for c, lead in enumerate(cols):
            v = M[r, c]
            if v >= 12 or (lead == lin and v > 0):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 * im.norm.vmax else INK,
                        fontweight="bold" if lead == lin else "normal")
            if lead == lin:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(length=0, labelsize=7.5, colors=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("% of the lineage section's records", color=INK2, fontsize=8)
    ax.set_title("Advice: seeded section (rows) vs lead content tag (columns); outlined cells = exact match",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.set_xlabel("lead content section (first primary tag)", color=INK2)
    fig.subplots_adjust(left=0.3, right=0.94, top=0.92, bottom=0.34)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- report --


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_markdown(stats: dict, path: Path, figs_url: str) -> None:
    T = stats["titles"]
    S = stats["corpora"]["stories"]
    A = stats["corpora"]["advice"]
    secs = stats["sections"]
    ids = [s["id"] for s in secs]
    n_s, n_a, n_l = S["n_ok"], A["n_ok"], A["n_records"]

    def row(sid: str) -> str:
        s = next(x for x in secs if x["id"] == sid)
        name = ("&nbsp;&nbsp;&nbsp;&nbsp;" * (s["level"] - 2)) + (f"**{s['title']}**" if s["parent"] is None else s["title"])
        sp, sa = S["counts"]["primary"][sid], S["counts"]["any"][sid]
        ap, aa = A["counts"]["primary"][sid], A["counts"]["any"][sid]
        ln = A["lineage_counts"][sid]
        return (f"| {name} | {sp:,} ({100*sp/n_s:.1f}%) | {sa:,} ({100*sa/n_s:.1f}%) | "
                f"{ap:,} ({100*ap/n_a:.1f}%) | {aa:,} ({100*aa/n_a:.1f}%) | {ln:,} ({100*ln/n_l:.1f}%) |")

    lines: list[str] = []
    L = lines.append
    L(f"# Constitution section coverage by content tagging — {stats['release']}")
    L(f"_corpus_generator · {A['ended_at'][:10]} · post-hoc, user-ordered_")
    L("")
    L("## What was measured")
    L("")
    L(f"Every record of the two v2 training finals was tagged by `{A['model']}` (live, lowprio, {A['n_requests_issued'] + S['n_requests_issued']:,} requests) "
      f"against a frozen 39-section index of the constitution (id + title + a 1–3 sentence Opus summary per section, "
      f"`constitution/section-summaries.json`, sha `{stats['index_sha256'][:12]}…`, written from constitution snapshot "
      f"`{stats['constitution_sha256'][:12]}…`). The tagger saw the story text, or the advice transcript's user + assistant turns "
      f"(not the operator system prompt), and returned up to 3 **primary** sections the document substantively exercises, up to 3 "
      f"**secondary** sections touched in passing, a **none** flag, and a confidence. Ids were validated against the index; the "
      f"reply was a structured JSON object (schema-constrained), so no reply needed a repair round in either corpus "
      f"(lenient repairs: stories {S['lenient_repairs'] or 'none'}, advice {A['lenient_repairs'] or 'none'}).")
    L("")
    L("Three views per section, all counted in RECORDS: **primary** (section in the primary list), **any** (primary or "
      "secondary), **lead** (the first primary tag — one section per record, the view comparable to the advice construction "
      "lineage, which is also one section per record). The prior analysis measured advice coverage by lineage only "
      "(17/39 sections) and had no section measurement for stories at all.")
    L("")
    L("## The answer")
    L("")
    zs_p, zs_a = S["zero_primary"], S["zero_any"]
    za_p, za_a = A["zero_primary"], A["zero_any"]
    L(f"**Did the stories draw from all sections? By content: {S['n_sections_primary']}/39 sections appear as a primary tag "
      f"of at least one story and {S['n_sections_any']}/39 as primary-or-secondary — "
      + ("every section is exercised somewhere; " if not zs_a else
         f"{len(zs_a)} section(s) are exercised by NO story even in passing ({', '.join(T[i] for i in zs_a)}); ")
      + f"but the distribution is far from even: {len(S['under_1pct_primary'])} sections sit under 1% of stories as a primary tag, "
      f"and the two most common lead sections ({', '.join(T[k] for k, _ in S['top2']['lead_top2'])}) lead "
      f"{pct(S['top2']['lead_top2_share'])} of stories.**")
    L("")
    L(f"Advice by content: {A['n_sections_primary']}/39 primary, {A['n_sections_any']}/39 any (lineage: {A['n_sections_lineage']}/39). "
      f"Top-2 lead sections ({', '.join(T[k] for k, _ in A['top2']['lead_top2'])}) lead {pct(A['top2']['lead_top2_share'])} of advice records.")
    L("")
    H_s, H_a = S["helpfulness"], A["helpfulness"]
    L(f"**Helpfulness cluster** (the *Being helpful* chapter and its 9 subsections — zero advice records by lineage): "
      f"stories {H_s['primary']:,}/{n_s:,} = {pct(H_s['primary_share'])} carry a primary tag in the cluster "
      f"({pct(H_s['any_share'])} primary-or-secondary; {pct(H_s['lead_share'])} lead there); "
      f"advice {H_a['primary']:,}/{n_a:,} = {pct(H_a['primary_share'])} primary ({pct(H_a['any_share'])} any; {pct(H_a['lead_share'])} lead). "
      f"By content the cluster is not empty in either corpus — the lineage's zero was a fact about the seed, not the text.")
    L("")
    L(f"**none rate**: stories {S['none']['none']:,}/{n_s:,} = {pct(S['none']['rate'])}; advice {A['none']['none']:,}/{n_a:,} = {pct(A['none']['rate'])}. "
      f"Non-ok tagger outcomes: stories {S['status_counts']}, advice {A['status_counts']}.")
    L("")
    L("## Per-section histogram (records; constitution order)")
    L("")
    L(f"![per-section coverage]({figs_url}/section_coverage_by_section.png)")
    L("")
    L(f"| section | stories primary (n={n_s:,}) | stories any | advice primary (n={n_a:,}) | advice any | advice lineage (n={n_l:,}) |")
    L("|---|---|---|---|---|---|")
    for sid in ids:
        L(row(sid))
    L("")
    L("### Chapter roll-up (records with ≥1 primary tag in the chapter)")
    L("")
    L("| chapter | stories | advice (content) | advice (lineage) |")
    L("|---|---|---|---|")
    for c in [s for s in secs if s["parent"] is None]:
        cs, ca, cl = S["chapters"]["primary"][c["id"]], A["chapters"]["primary"][c["id"]], A["lineage_chapters"][c["id"]]
        L(f"| {c['title']} | {cs:,} ({100*cs/n_s:.1f}%) | {ca:,} ({100*ca/n_a:.1f}%) | {cl:,} ({100*cl/n_l:.1f}%) |")
    L("")
    L(f"![chapter coverage]({figs_url}/section_coverage_chapters.png)")
    L("")
    L("## Zero and under-represented sections")
    L("")
    L(f"- **Stories, zero as primary ({len(zs_p)})**: {', '.join(T[i] for i in zs_p) or '—'}")
    L(f"- **Stories, zero even as secondary ({len(zs_a)})**: {', '.join(T[i] for i in zs_a) or '—'}")
    L(f"- **Stories, under 1% of records as primary ({len(S['under_1pct_primary'])})**: "
      + "; ".join(f"{T[i]} ({S['counts']['primary'][i]:,})" for i in S["under_1pct_primary"]))
    L(f"- **Advice, zero as primary ({len(za_p)})**: {', '.join(T[i] for i in za_p) or '—'}")
    L(f"- **Advice, zero even as secondary ({len(za_a)})**: {', '.join(T[i] for i in za_a) or '—'}")
    L(f"- **Advice, under 1% of records as primary ({len(A['under_1pct_primary'])})**: "
      + "; ".join(f"{T[i]} ({A['counts']['primary'][i]:,})" for i in A["under_1pct_primary"]))
    zl = [i for i in ids if A["lineage_counts"][i] == 0]
    L(f"- **Advice, zero by lineage ({len(zl)})**: {', '.join(T[i] for i in zl)}")
    L("")
    L("## Concentration")
    L("")
    for name, C in (("Stories", S), ("Advice (content)", A)):
        t2 = C["top2"]
        L(f"- **{name}** — top-2 lead sections: " + ", ".join(f"{T[k]} ({v:,})" for k, v in t2["lead_top2"])
          + f" → {pct(t2['lead_top2_share'])} of records; top-2 primary sections: "
          + ", ".join(f"{T[k]} ({v:,})" for k, v in t2["primary_top2"])
          + f" → {pct(t2['primary_top2_records_share'])} of records touch at least one of them.")
    lt = sorted(A["lineage_counts"].items(), key=lambda kv: -kv[1])[:2]
    L(f"- **Advice (lineage)** — top-2 sections: " + ", ".join(f"{T[k]} ({v:,})" for k, v in lt)
      + f" → {pct(sum(v for _, v in lt) / n_l)} of records.")
    L("")
    L("## Sections per record")
    L("")
    L(f"![sections per record]({figs_url}/section_coverage_sections_per_record.png)")
    L("")
    L("| | stories | advice |")
    L("|---|---|---|")
    for k in range(0, 4):
        L(f"| {k} primary | {S['per_record']['primary'][str(k)]:,} ({100*S['per_record']['primary'][str(k)]/n_s:.1f}%) | "
          f"{A['per_record']['primary'][str(k)]:,} ({100*A['per_record']['primary'][str(k)]/n_a:.1f}%) |")
    for k in range(0, 7):
        L(f"| {k} total (primary+secondary) | {S['per_record']['total'][str(k)]:,} ({100*S['per_record']['total'][str(k)]/n_s:.1f}%) | "
          f"{A['per_record']['total'][str(k)]:,} ({100*A['per_record']['total'][str(k)]/n_a:.1f}%) |")
    L(f"| mean primary / mean total | {S['per_record']['mean_primary']:.2f} / {S['per_record']['mean_total']:.2f} | "
      f"{A['per_record']['mean_primary']:.2f} / {A['per_record']['mean_total']:.2f} |")
    L("")
    L(f"Confidence (tagger self-report): stories mean {S['confidence']['mean']:.2f} (p10 {S['confidence']['p10']:.2f}, p90 {S['confidence']['p90']:.2f}); "
      f"advice mean {A['confidence']['mean']:.2f} (p10 {A['confidence']['p10']:.2f}, p90 {A['confidence']['p90']:.2f}).")
    L("")
    L("Caveat on the caps: the tagger was told three is a ceiling, not a target, and that an empty secondary list is a normal "
      "answer, but a large share of records still fill both lists; the per-record counts are therefore an upper-bounded "
      "reading of how many sections a document exercises, and the **lead** view is the safest single-section summary.")
    L("")
    L("## Advice: construction lineage vs content tags")
    L("")
    ag = A["agreement"]
    L(f"Of {ag['n']:,} advice records, the seeded section is the **lead** content tag in {ag['exact_lead']:,} "
      f"({pct(ag['exact_lead_rate'])}), appears **anywhere in primary** in {ag['in_primary']:,} ({pct(ag['in_primary_rate'])}), "
      f"and in primary-or-secondary in {ag['in_any']:,} ({pct(ag['in_any_rate'])}); the lead tag falls in the seeded section's "
      f"**chapter** in {ag['same_chapter_lead']:,} ({pct(ag['same_chapter_lead_rate'])}).")
    L("")
    L(f"![lineage vs content]({figs_url}/section_coverage_advice_lineage_vs_content.png)")
    L("")
    L("Confusion pattern — for each seeded section, where the lead content tag went (top 3):")
    L("")
    L("| seeded section (n) | lead content sections (share) |")
    L("|---|---|")
    for lin, d in ag["confusion"].items():
        tot = sum(d.values())
        tops = list(d.items())[:3]
        L(f"| {T[lin]} ({tot:,}) | " + "; ".join(f"{T.get(k, k)} {100*v/tot:.0f}%" for k, v in tops) + " |")
    L("")
    L("## Stories (untargeted) vs advice (seed-targeted)")
    L("")
    L(_comparison_paragraph(stats))
    L("")
    L("## Provenance")
    L("")
    for corpus, C in (("stories", S), ("advice", A)):
        L(f"- `data/{corpus}/section-tags-{stats['release']}.jsonl` + `.manifest.json`: {C['n_records']:,} records, "
          f"{C['n_requests_issued']:,} requests, {C['wall_clock_s']/60:.1f} min wall-clock ({C['started_at']} → {C['ended_at']}), "
          f"status {C['status_counts']}, repair rounds used {C['n_by_repair_round']}.")
    L(f"- index `constitution/section-summaries.json` sha `{stats['index_sha256']}`; constitution `{stats['constitution_sha256']}`.")
    L(f"- stats JSON: `data/manifests/section-coverage-{stats['release']}.json`; figures: {figs_url}/section_coverage_*.png")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_paragraph(stats: dict) -> str:
    T = stats["titles"]
    S, A = stats["corpora"]["stories"], stats["corpora"]["advice"]
    n_s, n_a = S["n_ok"], A["n_ok"]
    hs, ha = S["helpfulness"], A["helpfulness"]
    s_lead = S["top2"]["lead_top2"]
    a_lead = A["top2"]["lead_top2"]
    sto_only = [i for i in [s["id"] for s in stats["sections"]] if S["counts"]["primary"][i] and not A["counts"]["primary"][i]]
    adv_only = [i for i in [s["id"] for s in stats["sections"]] if A["counts"]["primary"][i] and not S["counts"]["primary"][i]]
    return (
        f"The stories were conditioned on the whole constitution and never pointed at a section; the advice records were each "
        f"seeded from one of 17 sections under a misalignment behavior seed. By content, the untargeted corpus reaches "
        f"{S['n_sections_primary']} sections as primary ({S['n_sections_any']} any) against the advice corpus's "
        f"{A['n_sections_primary']} ({A['n_sections_any']} any) and the lineage's 17, and it puts {100*hs['primary_share']:.0f}% of "
        f"records in the helpfulness cluster where the lineage had none (advice by content: {100*ha['primary_share']:.0f}%). "
        f"Both corpora are concentrated: the two most common lead sections take {100*S['top2']['lead_top2_share']:.0f}% of stories "
        f"({T[s_lead[0][0]]}, {T[s_lead[1][0]]}) and {100*A['top2']['lead_top2_share']:.0f}% of advice records "
        f"({T[a_lead[0][0]]}, {T[a_lead[1][0]]}). "
        + (f"Sections the stories reach that the advice never does as primary: {', '.join(T[i] for i in sto_only)}. " if sto_only else "")
        + (f"Sections the advice reaches that the stories never do as primary: {', '.join(T[i] for i in adv_only)}. " if adv_only else "")
        + f"Seed-targeting bought the advice corpus its safety/values/corrigibility mass and nothing outside the seed's reach; the "
        f"untargeted stories drifted toward the sections the story briefs' everyday-help register naturally exercises. Neither "
        f"corpus is a uniform sample of the constitution, and the two under-represent different regions — which is the first fact a "
        f"steering vector trained on either should be read against."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="production-20260831-v2")
    ap.add_argument("--figs", default=str(Path.home() / "figs" / "corpus_generator-production"))
    ap.add_argument("--figs-url", default=os.environ.get("SISTEER_FIGS_URL", ""),
                    help="public base URL of --figs for the report's links (default $SISTEER_FIGS_URL; "
                         "empty = link the local paths)")
    ap.add_argument("--md", default=None)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    stats = compute(args.release)
    out = ROOT / "data" / "manifests" / f"section-coverage-{args.release}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, json.dumps(stats, indent=1, ensure_ascii=False))
    print("stats ->", out)
    if args.stats_only:
        return
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    fig_by_section(stats, figs / "section_coverage_by_section.png")
    fig_chapters(stats, figs / "section_coverage_chapters.png")
    fig_per_record(stats, figs / "section_coverage_sections_per_record.png")
    fig_lineage_confusion(stats, figs / "section_coverage_advice_lineage_vs_content.png")
    print("figures ->", figs)
    write_markdown(stats, Path(args.md), args.figs_url or str(figs))
    print("report ->", args.md)


if __name__ == "__main__":
    main()
