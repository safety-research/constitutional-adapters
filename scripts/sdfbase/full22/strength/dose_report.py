#!/usr/bin/env python
"""Fold the per-target dose captures into ONE table and ONE json — the strength frontier's common
physical dose, per (lane, target, view, cell), beside each arm's native nominal dose.

    python dose_report.py --in-dir <shared>/exp/strength_dose [--out <dir>] [--targets ...]

Reads `dose_<target>.json` (written by dose_capture.py) and emits:
  dose_table.md    the short table a reader looks at
  dose_report.json the same rows as data, with the provenance block and every gate verdict
Refuses to print a dose number for a lane whose linearity gate failed (a lane that fails it has no dose
axis), and says so in the table instead of the number. A rung with ZERO effective layers is called out
in the header, not left to be inferred from a small number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (REPO / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dose_common as dc  # noqa: E402

LANE_UNIT = {"lora": "write-basis multiplier", "alphasteer": "nominal base_scale"}


def _rows_for(target: str, doc: dict) -> List[dict]:
    rows = []
    for cell_id, cell in (doc.get("lora") or {}).items():
        for view, r in cell["rungs"].items():
            rows.append(_row("lora", target, view, cell_id, r))
    as_doc = doc.get("alphasteer") or {}
    for sid, r in (as_doc.get("rungs") or {}).items():
        rows.append(_row("alphasteer", target, r["view"], "alphasteer", r))
    return rows


def _row(lane: str, target: str, view: str, cell_id: str, r: dict) -> dict:
    pooled = r["realised_dose"]["pooled"]
    eff = r["effective_layers"]
    return {
        "lane": lane, "target": target, "view": view, "cell": cell_id,
        "nominal_dose": r["nominal_dose"], "dose_unit": r["dose_unit"],
        "realised_dose_median": r["realised_dose_median"],
        "signed_realised_dose_median": r["signed_realised_dose_median"],
        "iqr": [pooled["p25"], pooled["p75"]], "iqr_width": pooled["iqr"],
        "p05": pooled["p05"], "p95": pooled["p95"],
        "realised_harmful_median": r["realised_dose"]["harmful"]["median"],
        "realised_benign_median": r["realised_dose"]["benign"]["median"],
        "family_gap_rel": r["realised_dose"]["family_gap_rel"],
        "nominal_layers": eff["nominal_layer_count"], "effective_layers": eff["effective_layer_count"],
        "effective_layers_half": eff["effective_layer_count_half"],
        "inactive_layers": eff["inactive_layers"],
        "per_layer_realised_dose_median": eff["per_layer_realised_dose_median"],
        "median_frac_changed_by_layer": eff["median_frac_changed_by_layer"],
        "gate_pass": r["gate"]["pass"], "gate_exact": r["gate"]["exact"],
        "gate_factor": r["gate"]["checks"]["median_factor"]["got"],
        "gate_want": r["gate"]["want_factor"], "staged_id": r["staged_id"],
        "object_sha256": r["object_sha256"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--targets", nargs="*", default=list(dc.TARGETS))
    args = ap.parse_args(argv)
    out_dir = args.out or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    prov: Dict[str, Any] = {}
    missing = []
    for t in args.targets:
        p = args.in_dir / t / f"dose_{t}.json"
        if not p.is_file():
            missing.append(str(p)); continue
        doc = json.loads(p.read_text())
        rows.extend(_rows_for(t, doc))
        prov[t] = doc["provenance"]
    if not rows:
        raise SystemExit(f"no dose_<target>.json under {args.in_dir} (missing: {missing})")

    # one provenance block for the whole table: what must be identical across targets, asserted
    shared = {}
    for key in ("pool", "render", "gate", "effective_layers", "h_definition", "edit_definition", "positions"):
        vals = {t: json.dumps(p[key].get("sha256") if key == "pool" else
                              (p[key] if key != "render" else {k: v for k, v in p[key].items()
                                                               if k not in ("chat_template_hash", "chat_template_hash_expected", "prompt_tokens")}),
                              sort_keys=True) for t, p in prov.items()}
        if len(set(vals.values())) != 1:
            raise SystemExit(f"targets disagree on `{key}` — the lanes are not comparable: {vals}")
        shared[key] = next(iter(prov.values()))[key]
    shared["chat_template_hash_by_target"] = {t: p["render"]["chat_template_hash"] for t, p in prov.items()}
    shared["serving_impl_commit"] = {t: p["code"]["serving_impl_commit"] for t, p in prov.items()}
    shared["code_sha"] = {t: p["code"]["code_sha"] for t, p in prov.items()}
    shared["model_revision"] = {t: p["model_revision"] for t, p in prov.items()}
    shared["env"] = {t: {k: p["env"].get(k) for k in ("torch", "transformers", "mamba_ssm", "causal_conv1d")}
                     for t, p in prov.items()}

    zero = [r for r in rows if r["effective_layers"] == 0]
    failed = [r for r in rows if not r["gate_pass"]]
    doc = {"schema": dc.SCHEMA, "rows": rows, "provenance": shared, "per_target_provenance": prov,
           "missing_targets": missing,
           "gate": {"all_pass": not failed, "failed": [f"{r['lane']}/{r['target']}/{r['view']}/{r['cell']}" for r in failed]},
           "zero_effective_layers": [f"{r['lane']}/{r['target']}/{r['view']}/{r['cell']}" for r in zero]}
    (out_dir / "dose_report.json").write_text(json.dumps(doc, indent=1) + "\n")

    # --- the table ---
    L = []
    L.append("# Realised dose — the strength frontier's common physical unit\n")
    L.append(f"`realised_dose_median` = median over the {shared['pool']['n_rows']} reference prompts of "
             "||edit|| / ||h|| at the steered site, measured on the EVALUATION prompt distribution under the "
             "served rendering. Each lane keeps its native knob as `nominal`; this column is what makes the "
             "two comparable.\n")
    L.append(f"- pool `{shared['pool']['path']}` sha256 `{shared['pool']['sha256'][:16]}…` "
             f"({shared['pool']['n_harmful']} harmful + {shared['pool']['n_benign']} benign)")
    L.append(f"- render: {shared['render']['call']}, `add_special_tokens={shared['render']['add_special_tokens']}`, "
             f"padding `{shared['render']['padding_side']}`, system `{shared['render']['system']}`, "
             f"template kwargs `{shared['render']['template_kwargs']}`")
    L.append("- chat-template hash per target: "
             + ", ".join(f"{t} `{h}`" for t, h in sorted(shared["chat_template_hash_by_target"].items())))
    L.append("- serving_impl_commit: " + ", ".join(sorted({v[:12] for v in shared["serving_impl_commit"].values()})))
    if failed:
        L.append("\n## THE D51 SELF-CONSISTENCY GATE FAILED — these arms have NO dose axis and their numbers are withheld\n")
        for r in failed:
            L.append(f"- **{r['lane']}/{r['target']}/{r['view']}/{r['cell']}**: measured factor "
                     f"{r['gate_factor']:.6g} against the required {r['gate_want']:.6g}")
    if zero:
        L.append("\n## ZERO EFFECTIVE LAYERS — on the median prompt, no layer changes the served sum\n")
        L.append("The per-layer statistic is the MEDIAN over prompts at the captured position and phase, so this "
                 "says the edits round away *there*. It is NOT a whole-model no-op claim: a layer whose edit "
                 "survives on a minority of prompts has median 0 and reads inactive here, and other positions and "
                 "phases are not measured. Treat these arms as **suspected** near-no-ops and confirm before "
                 "reporting one as a base-model measurement.\n")
        for r in zero:
            L.append(f"- **{r['lane']}/{r['target']}/{r['view']}/{r['cell']}** (nominal {r['nominal_dose']:+g}): "
                     f"all {r['nominal_layers']} nominally-steered layers round away on the median prompt at the "
                     "captured position/phase.")
    L.append("\n## Per arm\n")
    L.append("| lane | target | view | cell | nominal | unit | realised median | IQR | p05–p95 | harmful/benign | layers eff/nom | gate |")
    L.append("|---|---|---|---|---:|---|---:|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["lane"], r["target"], r["cell"], r["nominal_dose"])):
        dose = ("WITHHELD (gate)" if not r["gate_pass"]
                else f"{r['signed_realised_dose_median']:+.4g}")
        g = ("PASS exact" if r["gate_pass"] and r["gate_exact"] else "PASS" if r["gate_pass"] else "**FAIL**")
        L.append(f"| {r['lane']} | {r['target']} | {r['view']} | {r['cell']} | {r['nominal_dose']:+g} | "
                 f"{LANE_UNIT[r['lane']]} | {dose} | "
                 f"{r['iqr'][0]:.3g}–{r['iqr'][1]:.3g} | {r['p05']:.3g}–{r['p95']:.3g} | "
                 f"{r['realised_harmful_median']:.3g}/{r['realised_benign_median']:.3g} | "
                 f"{r['effective_layers']}/{r['nominal_layers']} | {g} |")
    L.append("\n## Reference norms, three ways (the pool is ~2/3 harmful, so the weighting is reported, not argued)\n")
    L.append("| lane | target | site | harmful median | benign median | pooled median | family gap |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for t, p in sorted(prov.items()):
        doc_t = json.loads((args.in_dir / t / f"dose_{t}.json").read_text())
        for cell_id, cell in (doc_t.get("lora") or {}).items():
            for l, n in sorted((cell.get("reference_norms_by_layer") or {}).items(), key=lambda kv: int(kv[0])):
                q = n["resid_pre_last_row"]
                L.append(f"| lora | {t} | resid_pre L{l} (last row) | {q['harmful']['median']:.4g} | "
                         f"{q['benign']['median']:.4g} | {q['pooled']['median']:.4g} | {q['family_gap_rel']:.3f} |")
            break
        asd = doc_t.get("alphasteer") or {}
        for hl, n in sorted((asd.get("reference_norms_by_hook_layer") or {}).items(), key=lambda kv: int(kv[0]))[:3]:
            q = n["last_row"]
            L.append(f"| alphasteer | {t} | h_last hook L{hl} | {q['harmful']['median']:.4g} | "
                     f"{q['benign']['median']:.4g} | {q['pooled']['median']:.4g} | {q['family_gap_rel']:.3f} |")
    (out_dir / "dose_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out_dir/'dose_table.md'} and {out_dir/'dose_report.json'}")
    if missing:
        print(f"MISSING (not yet captured): {missing}")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
