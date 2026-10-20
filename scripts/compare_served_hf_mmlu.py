"""Served-vs-HF cross-validation for s2parity/mmlu_letter (Phase 2 of the
served capability harness): pair the SAME items across one HF store and one
served store per cell and quantify agreement the way Phase 1 did — paired
per item, never marginals.

MMLU under greedy teacher-forced letter scoring is deterministic per
engine, so unlike the generation controls the two engines should agree
VERY tightly: the served pick differs from the HF pick only where two
letters' logprobs sit within cross-engine numeric noise (bf16 kernels /
reduction order). A tight match is the pass; a loose one is a finding to
report loudly, not to average away — which is why the per-cell output
carries the full discordant-item table, not just the test statistic.

Inputs are rows.jsonl files (the CellStore layout both drivers share):
the HF side from cell_controls_run (engine "hf"), the served side from
served_capability_run (engine "vllm"). Engines are VERIFIED per side —
a served store handed in as the HF reference is refused, never averaged.
Item-set mismatch is a hard error: a pairing that silently drops items
measures the overlap, not the instrument.

Usage::

  PYTHONPATH=src python scripts/compare_served_hf_mmlu.py \
    --cell llama8b_sr /path/hf/rows.jsonl /path/served/rows.jsonl \
    --cell llama8b_ct ... \
    --out out/served_mmlu_compare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASK = "s2parity/mmlu_letter"


class CompareError(RuntimeError):
    """A pairing/identity defect — loud, never a silently partial pair."""


def _mmlu_items(rows_path: Path, *, want_engine: str) -> dict:
    """{item_id: {"acc": 0/1, "pick": str|None, "gold": str|None}} for the
    per-item mmlu rows of one store, with the engine verified."""
    items: dict = {}
    with rows_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task") != TASK or r.get("metric") != "acc":
                continue
            if r.get("item_id") == "__aggregate__":
                continue
            eng = r.get("engine")
            if eng != want_engine:
                raise CompareError(
                    f"{rows_path}: row {r.get('item_id')!r} carries engine "
                    f"{eng!r} but this side of the pairing must be "
                    f"{want_engine!r} — the two sides are the point of the "
                    "comparison; refusing a store on the wrong side")
            extra = r.get("extra_json")
            extra = json.loads(extra) if isinstance(extra, str) else (
                extra or {})
            iid = r["item_id"]
            if iid in items:
                raise CompareError(
                    f"{rows_path}: duplicate per-item row for {iid!r} — "
                    "a resumed store files each item once; refusing")
            items[iid] = {"acc": int(r["value"]),
                          "pick": extra.get("pick"),
                          "gold": extra.get("gold")}
    if not items:
        raise CompareError(f"{rows_path}: no per-item {TASK} acc rows")
    return items


def compare_cell(hf_rows: Path, served_rows: Path) -> dict:
    """One cell's paired comparison. Raises on any pairing defect."""
    from sisteer.eval.realign_stats import mcnemar

    hf = _mmlu_items(hf_rows, want_engine="hf")
    sv = _mmlu_items(served_rows, want_engine="vllm")
    only_hf = sorted(set(hf) - set(sv))
    only_sv = sorted(set(sv) - set(hf))
    if only_hf or only_sv:
        raise CompareError(
            f"item sets differ: {len(only_hf)} only in HF (first "
            f"{only_hf[:3]}), {len(only_sv)} only in served (first "
            f"{only_sv[:3]}) — the seeded --limit subset must match; "
            "refusing to pair a partial overlap")
    ids = sorted(hf)
    n = len(ids)
    acc_hf = sum(hf[i]["acc"] for i in ids) / n
    acc_sv = sum(sv[i]["acc"] for i in ids) / n
    picks_known = all(hf[i]["pick"] is not None
                      and sv[i]["pick"] is not None for i in ids)
    pick_agree = (sum(hf[i]["pick"] == sv[i]["pick"] for i in ids) / n
                  if picks_known else None)
    discordant = [
        {"item_id": i, "hf": hf[i], "served": sv[i]}
        for i in ids if hf[i]["acc"] != sv[i]["acc"]]
    mc = mcnemar({i: hf[i]["acc"] for i in ids},
                 {i: sv[i]["acc"] for i in ids})
    return {
        "n_items": n,
        "acc_hf": acc_hf,
        "acc_served": acc_sv,
        "acc_delta_served_minus_hf": acc_sv - acc_hf,
        "pick_agreement": pick_agree,
        "n_pick_disagreements": (None if pick_agree is None
                                 else round((1 - pick_agree) * n)),
        "acc_discordant": discordant,
        "mcnemar": mc,
        "hf_rows": str(hf_rows),
        "served_rows": str(served_rows),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", nargs=3, action="append", required=True,
                    metavar=("LABEL", "HF_ROWS", "SERVED_ROWS"),
                    help="one cell's pairing: a label plus the HF and "
                         "served rows.jsonl paths (repeatable)")
    ap.add_argument("--out", default="",
                    help="write the full JSON report here as well")
    args = ap.parse_args(argv)

    report: dict = {}
    for label, hf_path, sv_path in args.cell:
        r = compare_cell(Path(hf_path), Path(sv_path))
        report[label] = r
        mc = r["mcnemar"]
        pa = ("n/a" if r["pick_agreement"] is None
              else f"{r['pick_agreement']:.3f}")
        print(f"[mmlu-compare] {label}: n={r['n_items']} "
              f"hf={r['acc_hf']:.4f} served={r['acc_served']:.4f} "
              f"delta={r['acc_delta_served_minus_hf']:+.4f} "
              f"pick_agreement={pa} "
              f"discordant={mc['a_only']}+{mc['b_only']} "
              f"mcnemar_p={mc['p_value']}")
        for d in r["acc_discordant"]:
            print(f"  discordant {d['item_id']}: hf pick="
                  f"{d['hf']['pick']} acc={d['hf']['acc']} | served pick="
                  f"{d['served']['pick']} acc={d['served']['acc']} "
                  f"(gold {d['hf']['gold']})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[mmlu-compare] report -> {args.out}")


if __name__ == "__main__":
    main()
