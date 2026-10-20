"""Render the r2 release-doc tables from the register-edit manifests (no API,
no recomputation): before/after register rates vs production for every
marker and family touched, edit counts + verification pass rates, re-score
deltas, r2 counts / official tokens / length-gate ratios, and the tier-3
request estimates. Markdown to stdout.

    uv run python scripts/register_edit_release_tables.py --config configs/control-scale-r2.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_generator.config import RunConfig, filter_report_path  # noqa: E402
from corpus_generator.io import iter_jsonl, read_json, sha256_file  # noqa: E402
from corpus_generator.register_edit import register_edit_manifest_path, run_manifest_path  # noqa: E402
from corpus_generator.templates import repo_root  # noqa: E402


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tier3", action="store_true", help="also count the held tier-3 edits")
    args = ap.parse_args()
    root = repo_root()
    cfg = RunConfig.load(Path(args.config))
    rc = cfg.register_edit
    run_manifest = read_json(run_manifest_path(cfg, root))
    out: list[str] = []
    for corpus in ("advice", "stories"):
        final = cfg.final_path(root, corpus, rc.suffix)
        if not final.exists():
            continue
        m = read_json(register_edit_manifest_path(final))
        fr = read_json(filter_report_path(final))
        out.append(f"\n### {corpus} — register rates, v1 -> r2 vs production (same regex on the finals)\n")
        out.append("| register | field | mode / cap | v1 control | r2 control | production v2 | target |")
        out.append("|---|---|---|---|---|---|---|")
        for name, row in m["register_table"].items():
            mode = row.get("mode") or f"family cap {row.get('cap')}"
            tgt = pct(row["target_rate"]) if "target_rate" in row else ""
            out.append(
                f"| `{name}` | {row['field']} | {mode} | {row['v1']['matched']}/{row['v1']['denominator']} = {pct(row['v1']['rate'])} "
                f"| {row['r2']['matched']}/{row['r2']['denominator']} = {pct(row['r2']['rate'])} "
                f"| {row['production']['matched']}/{row['production']['denominator']} = {pct(row['production']['rate'])} | {tgt} |"
            )
        out.append(f"\n### {corpus} — edits and verification\n")
        out.append("| stage | requests | edited (verified) | unedited after 2 repairs | repair rounds | retained mean / min | statuses |")
        out.append("|---|---|---|---|---|---|---|")
        for sname, st in m["verification"]["stages"].items():
            out.append(
                f"| `{sname}` | {st['n_requests']} | {st['edited']} ({pct(st['edited'] / max(1, st['expected']))}) "
                f"| {st['n_unedited']} | {st['repair_rounds_used']} | {st['retained_mean']} / {st['retained_min']} "
                f"| {json.dumps(st['status_counts'])} |"
            )
        blocks = [(str(m.get("edit_pass", 1)), m["rescore"]["deltas"])]
        for pp, blk in sorted((m.get("prior_passes") or {}).items()):
            if blk and blk.get("rescore"):
                blocks.insert(0, (pp, blk["rescore"]["deltas"]))
        for pp, d in blocks:
            out.append(f"\n### {corpus} — pass {pp} re-score deltas over the {d['n_edited']} records edited in that pass ({d['n_rescored']} re-scored)\n")
            for axis, a in d["axes"].items():
                out.append(f"- `{axis}`: before {json.dumps(a['v1'])} -> after {json.dumps(a['r2'])}; moved down/same/up {a['moved']['down']}/{a['moved']['same']}/{a['moved']['up']}")
            for fl, f in d["flags"].items():
                out.append(f"- `{fl}` flips false->true {f['false_to_true']}, true->false {f['true_to_false']}")
            out.append(f"- records edited in that pass dropped by the filter chain: {d['n_edited_dropped']} ({json.dumps(d['dropped_edited'])}); other drops: {json.dumps(d['dropped_unedited'])}")
        out.append(f"\n### {corpus} — r2 final\n")
        lg = fr.get("length_gate") or {}
        out.append(
            f"- records {fr['output_count']} (v1 {m['source']['n']}); official tokens {fr['official_tokens']:,} "
            f"(per tokenizer {json.dumps(fr['token_totals'])}); drops {json.dumps(fr['rule_counts'])}"
        )
        if lg:
            out.append(
                f"- length gate {'PASSED' if lg.get('passed') else 'FAILED'}: mean {lg.get('mean')} -> ratio {lg.get('mean_ratio')} "
                f"(±{lg['tolerance']['mean']}), p90 {lg.get('p90')} -> ratio {lg.get('p90_ratio')} (±{lg['tolerance']['p90']}) "
                f"vs {lg['reference']['mean']} / {lg['reference']['p90']}"
            )
        out.append(f"- edited records in the final: {m['n_edited_in_final']}; unedited records identical to v1 (all fields but lineage/token counts): {m['n_unedited_identical_to_source']}")
        out.append(f"- register caps after the chain: " + "; ".join(
            f"{s['name']} {s['kept']}/{s['final_count']} = {pct(s['final_frac'])} (dropped {s['dropped']})" for s in fr.get("register_family_stats", [])))
        mono = m["monoculture"]
        out.append(f"- new monocultures after the edits (>= {pct(mono['params']['min_frac'])} of r2, <= {pct(mono['params']['prod_max_frac'])} of production, < 1% of v1): "
                   + (json.dumps(mono["new_monocultures"]) if mono["new_monocultures"] else "none"))
        top = mono["replacement_3grams"][:5]
        out.append("- most repeated replacement 3-grams: " + ", ".join(f"{t['ngram']!r} {t['records']} ({pct(t['frac_of_corpus'])} of r2)" for t in top))
        out.append(f"- sha256 final `{sha256_file(final)}`; filter report `{sha256_file(filter_report_path(final))}`; register-edit manifest `{sha256_file(register_edit_manifest_path(final))}`")
    out.append(f"\nrun-level manifest: `{run_manifest_path(cfg, root)}` sha256 `{sha256_file(run_manifest_path(cfg, root))}`; batch/wave ids: {len(run_manifest['batch_ids'])}")
    if args.tier3:
        out.append("\n### Tier 3 (HELD) — request estimates\n")
        stories = list(iter_jsonl(cfg.final_path(root, "stories", rc.suffix)))
        prod = list(iter_jsonl(root / rc.production_reference["stories"]))
        n = len(stories)
        rx = re.compile(r"the\s+assistant\s+said", re.I)
        carriers = sum(1 for r in stories if rx.search(r["text"]))
        prod_rate = sum(1 for r in prod if rx.search(r["text"])) / len(prod)
        keep = round(prod_rate * n)
        out.append(f"- attribution rebalancing (\"the assistant said\" {pct(carriers / n)} -> production {pct(prod_rate)}): {carriers} carriers, keep {keep}, edit {carriers - keep} stories -> {carriers - keep} edit requests + {carriers - keep} re-scores = {2 * (carriers - keep):,} (sentence-level rewrites of the attribution only; multi-occurrence stories need every occurrence rewritten)")
        props = {
            "tablet_propped_beat": r"(tablet|phone|screen|laptop)\s+(was\s+)?propped\s+(up\s+)?against",
            "envelope_back_beat": r"(back|reverse)\s+of\s+an?\s+(old\s+|used\s+|torn\s+)?envelope",
            "the_message_came_in_opener": r"^\W*the\s+message\s+(came|arrived|arrives)",
            "its_voice_came_out_of": r"voice\s+came\s+(?:out\s+of|from)",
        }
        total = 0
        for name, pat in props.items():
            rxp = re.compile(pat, re.I)
            p = sum(1 for r in prod if rxp.search(r["text"])) / len(prod)
            c = sum(1 for r in stories if rxp.search(r["text"])) / n
            need = max(0, round(p * n) - round(c * n))
            total += need
            out.append(f"- insert `{name}` at production's rate ({pct(c)} -> {pct(p)}): +{need} stories -> {2 * need:,} requests")
        out.append(f"- tier-3 total: ~{2 * (carriers - keep) + 2 * total:,} requests")
    print("\n".join(out))


if __name__ == "__main__":
    main()
