"""Paired per-item comparison of two stores on ANY task — the generalization
of ``compare_served_hf_mmlu.py`` (which stays as it is for
``s2parity/mmlu_letter``). Same discipline: the SAME items on both sides,
engines verified per side, the full discordant-item table, the paired
accuracy delta with a paired-bootstrap CI and the analytic paired SE side by
side (Miller 2024, "Adding Error Bars to Evals"; U5-5), an exact McNemar test
on binary outcomes, and each side's unparseable / truncation facts from the
fields its rows already carry (``sisteer.capeval.tiers.pairing`` holds
the logic).

Two uses in the tinycap spec:

* RETIRED instrument (the decode-based MMLU comparison left main
  on 2026-09-11; ``inspect/mmlu_letter_valid`` can no longer be planned and
  exists only in stores written before then, which still read) — our served
  scorer vs Inspect's definition on paired items: side B's Inspect rows carry
  Inspect's sample id as ``item_id`` and OUR id in ``extra_json.paired_id``
  (P32), so B joins on ``paired_id``; both sides are ``vllm``::

    PYTHONPATH=src python scripts/compare_served_hf_task.py \\
      --task-a s2parity/mmlu_letter --engine-a vllm \\
      --task-b inspect/mmlu_letter_valid --engine-b vllm --key-b paired_id \\
      --cell prompted_off served/rows.jsonl served/rows.jsonl

* the standard GSM8K vs the retired instrument, per cell (U5-2/U5-5): side A
  is the old store's ``s2parity/gsm8k_0shot`` rows (engine ``hf``), side B
  the re-run's ``inspect/gsm8k_paired200`` rows whose ``paired_id`` is the
  old item id; the overlap is partial by design (platinum edits), hence
  ``--allow-unmatched`` (the unmatched ids are listed, never dropped)::

    PYTHONPATH=src python scripts/compare_served_hf_task.py \\
      --task-a s2parity/gsm8k_0shot --engine-a hf \\
      --task-b inspect/gsm8k_paired200 --engine-b vllm --key-b paired_id --allow-unmatched \\
      --bootstrap 10000 --bootstrap-seed 0 --ci 0.95 \\
      --cell llama8b_none old/cells/<cell>/rows.jsonl new/cells/<cell>/rows.jsonl \\
      --out compare/llama8b_none.json

Defaults are an HF-versus-served comparison's (``--engine-a hf``, ``--engine-b vllm``,
``--metric acc``, both keys ``item_id``, strict item sets) plus the paired
bootstrap's (``--bootstrap 10000 --bootstrap-seed 0 --ci 0.95``; ``--bootstrap
0`` disables it and says so). An avg@k side (several sample rows per item, or
the producer's mean row with ``samples_per_item``) is reduced to per-item
means BEFORE pairing; the bootstrap resamples items, never samples, and the
report says how many samples per item each side had.

The JSON report per cell carries, besides the pre-existing keys,
``paired_bootstrap: {n_items, delta, ci_low, ci_high, level, B, seed,
excludes_zero}``, ``paired_se``, ``samples_per_item: {a, b}`` and
``sides.<a|b>.quality`` (``unparseable`` + its source, ``answer_missing``,
``content_empty(_with_reasoning)``, ``at_token_cap``, ``finish_reason`` —
``"not recorded by the producer"`` on today's Inspect rows —
``extraction_failed``, ``pred_missing``, ``used_fallback``; null = the rows do
not carry the field).

Since E46 each side also reports the LENS REDUCTION MODE its store was served
under (``sides.<a|b>.reduction_mode`` + ``reduction_mode_source``, read from
the ``manifest.json`` beside the rows), and two served-lens stores that state
DIFFERENT modes REFUSE: a stacked group's reductions are one GEMM where a
per-row group's are B single-row calls, so pooling the two measures the
arithmetic as well as the intervention. A side that states no mode — an HF
reference store, a bare rows file — prints ``None`` with the reason and does
not block the comparison; it is never coerced into a mode it never had.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sisteer.capeval.tiers import pairing as pr


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-a", required=True, help="row task name on side A (the reference)")
    ap.add_argument("--task-b", required=True, help="row task name on side B")
    ap.add_argument("--engine-a", default="hf", help="engine every side-A row must carry")
    ap.add_argument("--engine-b", default="vllm", help="engine every side-B row must carry")
    ap.add_argument("--metric", default="acc",
                    help="the per-item metric paired on both sides (acc, acc_norm, ...)")
    ap.add_argument("--metric-b", default=None,
                    help="side B's metric when it differs from side A's (default: --metric)")
    ap.add_argument("--key-a", default=pr.KEY_ITEM_ID, choices=pr.PAIR_KEYS)
    ap.add_argument("--key-b", default=pr.KEY_ITEM_ID, choices=pr.PAIR_KEYS,
                    help="paired_id joins Inspect rows on extra_json.paired_id (P32)")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="report (never drop silently) items present on one side only; "
                         "for comparisons whose overlap is partial by design")
    ap.add_argument("--bootstrap", type=int, default=pr.DEFAULT_BOOTSTRAP, metavar="B",
                    help="paired-bootstrap resamples of the matched ITEMS for the CI on "
                         "delta_b_minus_a (Miller 2024); 0 disables the bootstrap "
                         f"(default {pr.DEFAULT_BOOTSTRAP})")
    ap.add_argument("--bootstrap-seed", type=int, default=pr.DEFAULT_BOOTSTRAP_SEED,
                    help=f"seed of the bootstrap's numpy generator (default {pr.DEFAULT_BOOTSTRAP_SEED})")
    ap.add_argument("--ci", type=float, default=pr.DEFAULT_CI,
                    help=f"percentile interval level, e.g. 0.95 (default {pr.DEFAULT_CI})")
    ap.add_argument("--cell", nargs=3, action="append", required=True,
                    metavar=("LABEL", "A_ROWS", "B_ROWS"),
                    help="one cell's pairing: a label plus side A's and side B's rows.jsonl "
                         "(repeatable; the same file may serve both sides)")
    ap.add_argument("--out", default="", help="write the full JSON report here as well")
    return ap


def run(args) -> dict:
    report: dict = {}
    for label, a_path, b_path in args.cell:
        side_a = pr.Side(label=f"{label}:A", rows_path=Path(a_path), task=args.task_a,
                         engine=args.engine_a, metric=args.metric, key=args.key_a)
        side_b = pr.Side(label=f"{label}:B", rows_path=Path(b_path), task=args.task_b,
                         engine=args.engine_b, metric=args.metric_b or args.metric,
                         key=args.key_b)
        report[label] = pr.compare_sides(side_a, side_b, allow_unmatched=args.allow_unmatched,
                                         bootstrap=args.bootstrap,
                                         bootstrap_seed=args.bootstrap_seed, ci=args.ci)
    return report


def _samples_text(s: dict) -> str:
    return str(s["min"]) if s["min"] == s["max"] else f"{s['min']}-{s['max']}"


def _quality_text(q: dict) -> str:
    """One line per side: the derivable facts, each named; a fact the
    producer did not record reads as such rather than as 0."""
    parts = [f"n_rows={q['n_rows']}"]
    if q["unparseable"] is None:
        parts.append(f"unparseable=n/a ({q['unparseable_source']})")
    else:
        denom = (q["extraction_failed_rows"] if q["extraction_failed_rows"] is not None
                 else q["n_rows"])
        parts.append(f"unparseable={q['unparseable']}/{denom} "
                     f"rate={q['unparseable_rate']:.4f} [{q['unparseable_source']}]")
    for k in ("answer_missing", "content_empty", "content_empty_with_reasoning",
              "at_token_cap", "extraction_failed", "pred_missing", "used_fallback"):
        if q.get(k) is not None:
            parts.append(f"{k}={q[k]}")
    fr = q["finish_reason"]
    parts.append(f"finish_reason={fr}" if isinstance(fr, str)
                 else f"finish_reason[{q['finish_reason_field']}]={fr}")
    return " ".join(parts)


def print_report(report: dict, *, task_a: str, task_b: str) -> None:
    for label, r in report.items():
        mc = r["mcnemar"]
        pa = "n/a" if r["pick_agreement"] is None else f"{r['pick_agreement']:.3f}"
        mc_s = ("mcnemar=n/a" if mc is None
                else f"discordant={mc['a_only']}+{mc['b_only']} mcnemar_p={mc['p_value']}")
        pb = r.get("paired_bootstrap")
        if pb is None:
            ci_s = f"ci=n/a ({r.get('paired_bootstrap_unavailable_reason')})"
        else:
            ci_s = (f"{pb['level'] * 100:g}% CI [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}] "
                    f"(paired bootstrap B={pb['B']} seed={pb['seed']})")
        se = r.get("paired_se")
        se_s = "paired_se=n/a" if se is None else f"paired_se={se:.4f}"
        print(f"[task-compare] {label}: {task_a} vs {task_b} n={r['n_items']} "
              f"a={r['mean_a']:.4f} b={r['mean_b']:.4f} "
              f"delta_b_minus_a={r['delta_b_minus_a']:+.4f} {ci_s} {se_s} "
              f"pick_agreement={pa} {mc_s}"
              + (f" only_a={len(r['only_a'])} only_b={len(r['only_b'])}"
                 if (r["only_a"] or r["only_b"]) else ""))
        spi = r.get("samples_per_item")
        if spi:
            print(f"  samples/item: A={_samples_text(spi['a'])} B={_samples_text(spi['b'])}")
        sides = r.get("sides") or {}
        for s in ("a", "b"):
            if s in sides and "reduction_mode" in sides[s]:
                # E46: what lens arithmetic this store was served under. A
                # DIFFERENCE already refused in compare_sides; printing it on
                # every run is how a side that states nothing stays visible
                # instead of being quietly assumed comparable.
                print(f"  {s.upper()} lens reduction mode: {sides[s]['reduction_mode']} "
                      f"[{sides[s]['reduction_mode_source']}]")
            if s in sides and "quality" in sides[s]:
                print(f"  {s.upper()} quality: {_quality_text(sides[s]['quality'])}")
        for d in r["discordant"]:
            a, b = d["a"], d["b"]
            ea = {k: v for k, v in a.items() if k not in ("value", "item_id")}
            eb = {k: v for k, v in b.items() if k not in ("value", "item_id")}
            print(f"  discordant {d['id']}: a={a['value']} {ea} | b={b['value']} {eb}")
        if r["mcnemar_unavailable_reason"]:
            print(f"  note: {r['mcnemar_unavailable_reason']}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print_report(report, task_a=args.task_a, task_b=args.task_b)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[task-compare] report -> {args.out}")


if __name__ == "__main__":
    main()
