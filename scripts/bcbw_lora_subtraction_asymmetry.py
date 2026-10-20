#!/usr/bin/env python
"""Parent asymmetry INSIDE a `lora_subtraction` object — the measurement record (bcb-bfcl-200-widen, 2026-09-13).

A LoRA-subtraction object stores both parents' factors (A_pos/B_pos = the constitution fit, A_neg/B_neg = the control fit,
per layer, rank r) and is served as ΔW = B_pos·A_posᵀ − B_neg·A_negᵀ. Whether that ΔW is a *difference* or one parent barely
offset is a property of the object itself, measured here per layer as the Frobenius norms ‖ΔW_pos‖, ‖ΔW_neg‖, ‖ΔW_pos − ΔW_neg‖
and the ratio ‖ΔW_pos‖ / ‖ΔW_neg‖ (1 = a balanced pair). The qwen35_4b it batch-64 pair read 2.20–3.89 (median 2.61; the constitution
parent dominates: totals 153.6 vs 50.9, the served "difference" 158.4; served, the model refused all 200 BigCodeBench items), the
qwen35_4b base-lane pair 0.89–1.09 (balanced; −14 pp, no refusals), the staged gemma4e4b it pair 1.25–2.24 (median 1.74). The root cause is a CONFOUND in the pair's construction — the parents differ
in training budget (16/16 epochs vs early-stopped 1/3), not only in corpus — so the number is attached to the row as a RECORDED
mechanism measurement, never as a reading.

usage:
    python scripts/bcbw_lora_subtraction_asymmetry.py --object "<label>=<path/to/object.pt>" [--object ...] --out <json>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

KIND = "lora_subtraction"
METHOD = ("per layer, the Frobenius norm of each parent's effective weight delta B·Aᵀ from the factors stored inside the object "
          "(pos = the constitution fit, neg = the control fit); diff = ‖ΔW_pos − ΔW_neg‖ (what is served); ratio = ‖ΔW_pos‖ / ‖ΔW_neg‖ "
          "(1 = a balanced pair); totals = the Frobenius norm over ALL the object's layers at once (the square root of the sum of the "
          "per-layer squared norms — the number the D110 records quote); layer_norm_sums = the plain sums of the per-layer norms; "
          "ratio_median = statistics.median over the per-layer ratios")


class AsymmetryError(ValueError):
    pass


def measure(obj: Mapping[str, Any]) -> Dict[str, Any]:
    """The per-layer record for one loaded object dict (refuses anything but a `lora_subtraction` with both parents' factors)."""
    import torch
    if obj.get("kind") != KIND:
        raise AsymmetryError(f"object kind {obj.get('kind')!r} is not {KIND!r} — only a subtraction object carries two parents to compare")
    for key in ("layers", "rank", "A_pos", "B_pos", "A_neg", "B_neg"):
        if key not in obj:
            raise AsymmetryError(f"object lacks {key!r}")
    layers = [int(l) for l in obj["layers"]]
    per: List[Dict[str, Any]] = []
    for layer in layers:
        k = str(layer)
        try:
            a_pos, b_pos, a_neg, b_neg = (obj["A_pos"][k], obj["B_pos"][k], obj["A_neg"][k], obj["B_neg"][k])
        except KeyError as e:
            raise AsymmetryError(f"layer {layer}: factor {e} missing") from None
        dw_pos = torch.as_tensor(b_pos).float() @ torch.as_tensor(a_pos).float().T
        dw_neg = torch.as_tensor(b_neg).float() @ torch.as_tensor(a_neg).float().T
        if dw_pos.shape != dw_neg.shape:
            raise AsymmetryError(f"layer {layer}: ΔW shapes differ ({tuple(dw_pos.shape)} vs {tuple(dw_neg.shape)})")
        pos, neg, diff = float(dw_pos.norm()), float(dw_neg.norm()), float((dw_pos - dw_neg).norm())
        per.append({"layer": layer, "pos_norm": pos, "neg_norm": neg, "diff_norm": diff, "ratio_pos_over_neg": (pos / neg if neg > 0 else None)})
    ratios = [r["ratio_pos_over_neg"] for r in per if r["ratio_pos_over_neg"] is not None]
    if not ratios:
        raise AsymmetryError("no layer has a non-zero control delta — nothing to compare")
    prov = obj.get("provenance") if isinstance(obj.get("provenance"), Mapping) else {}
    def _global(key: str) -> float:               # ‖·‖_F over every layer at once = sqrt(Σ_layers ‖·‖_F²)
        return float(sum(r[key] ** 2 for r in per) ** 0.5)
    return {"kind": KIND, "rank": int(obj["rank"]), "layers": layers, "n_layers": len(layers), "per_layer": per,
            "totals": {"pos": _global("pos_norm"), "neg": _global("neg_norm"), "diff": _global("diff_norm")},
            "layer_norm_sums": {"pos": sum(r["pos_norm"] for r in per), "neg": sum(r["neg_norm"] for r in per), "diff": sum(r["diff_norm"] for r in per)},
            "ratio_min": min(ratios), "ratio_median": statistics.median(ratios), "ratio_max": max(ratios),
            "provenance": {k: prov.get(k) for k in ("derived_id", "model_key", "model_id", "model_revision", "parent_pos", "parent_neg", "scale_matched")
                           if k in prov and isinstance(prov.get(k), (str, int, float, bool, type(None)))},
            "method": METHOD}


def measure_file(path: Path) -> Dict[str, Any]:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, Mapping):
        raise AsymmetryError(f"{path}: not an object dict")
    rec = measure(obj)
    rec["object"] = str(path)
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rec


def build_record(objects: Sequence[tuple], *, decision: Optional[str] = None) -> Dict[str, Any]:
    """``{measured_at, decision, objects: {label: record}, method}`` over (label, path) pairs."""
    return {"measured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "generator": "scripts/bcbw_lora_subtraction_asymmetry.py",
            "decision": decision, "method": METHOD, "objects": {label: measure_file(Path(p).expanduser()) for label, p in objects}}


def _pair(value: str) -> tuple:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--object wants <label>=<path>, got {value!r}")
    label, path = value.split("=", 1)
    return label.strip(), path.strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", action="append", type=_pair, required=True, metavar="LABEL=PATH", help="a lora_subtraction object.pt (repeat)")
    p.add_argument("--out", required=True, help="the JSON record to write")
    p.add_argument("--decision", default=None, help="the decision-log entry this record backs (free text)")
    args = p.parse_args(argv)
    rec = build_record(args.object, decision=args.decision)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for label, r in rec["objects"].items():
        t = r["totals"]
        print(f"{label}: {r['n_layers']} layers, rank {r['rank']}; ratio pos/neg min {r['ratio_min']:.2f} median {r['ratio_median']:.2f} max {r['ratio_max']:.2f}; "
              f"totals pos {t['pos']:.1f} neg {t['neg']:.1f} diff {t['diff']:.1f}; sha {r['sha256'][:12]}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
