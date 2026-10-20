#!/usr/bin/env python3
"""``transfer_stats_full22_all.json`` — the WHOLE full22 wave in the schema of ``transfer_stats_full22.json`` (FINAL-5), with
the new material clearly namespaced:

  namespace ``full22``      : the previously filed grades (gemma4e4b / qwen35_4b main view, four static families + BoN) — read
                              from run-state/full22/grades/<target>/ UNCHANGED, never regraded.
  namespace ``full22_rest`` : this lane's grades (w7_rest/grade_full22_rest.py) — gemma4e4b / qwen35_4b main-view multi-turn
                              legs, the it views (everything), nemotron3nano4b (everything) — from run-state/full22/grades_rest/.

Same readers as FINAL-5, imported by path (``transfer_stats.gated_cell`` / ``bon_cell`` / ``paired_contrast``;
``transfer_r2.cell_rows`` / ``contrast`` / ``attack_outcomes`` / ``bon_outcomes``): every proportion is a Wilson 95% interval
on the integer numerator the aggregate carries; every within-wave contrast is the paired 10,000-draw bootstrap over goals
(seed 0) + exact McNemar on the discordant pairs, against the SAME view's own bare `none`. Multi-turn legs are gated
aggregates in the same per_goal shape, so the same readers apply. Added per readout, per cell:

  * ``degeneracy``  — from degeneracy_full22.json (degeneracy_join.py): response-level at-cap and loop rates (Wilson), the
                      refused-and-degenerate rate, and the goal-level asterisk ``defended_with_degeneration`` (a defended
                      goal whose defence includes a judged-refusal response that ran to the cap or loops), split into
                      ``defended_with_refused_loop`` and ``defended_with_refused_at_cap``.
  * ``asr_annotated`` — one glanceable string: ASR [lo, hi] | at-cap r% | loop r% | defended* k/n_defended.
  * ``petri``       — per cell, from petri_full22_stats.json (Petri's own claude-opus-5 judge): per-behaviour means with
                      bootstrap intervals, completion state, unresolved ledger; ``petri_vs_bare`` per behaviour.
  * ``cross_view``  — gemma / qwen only: it-view cell vs the SAME-NAMED main-view cell on the shared goals (both stacked,
                      same wave), and it `none` vs main `none` (two independent bare generations: the wave's noise floor).
Capability blocks stay the readers' PENDING stubs (no controls exist for any full22 cell). ``lens_reduction_mode: stacked``
on every row and in meta; nothing here pools with a per_row leg (r2/r3).

usage: build_transfer_stats_full22_all.py --out <json> [--petri <petri_full22_stats.json>] [--degeneracy <degeneracy_full22.json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from site_paths import site_path  # noqa: E402
# the run-state directory under the site's state root ($STATE) and the campaign's staging directory ($STAGE)
RS = site_path("$STATE/romp/candidate-si-refit/run-state/full22", "RS")
STAGE = site_path("$STAGE/sdfbase", "STAGE")
sys.path.insert(0, str(STAGE / "analysis_r2"))
sys.path.insert(0, str(STAGE / "analysis"))
import transfer_stats as T1  # noqa: E402
import transfer_r2 as R2  # noqa: E402

STATIC = ["pasttense", "codeattack", "renellm", "grok420"]
MULTITURN = ["crescendo", "pair", "actorattack"]
ATTACK_READOUTS = [*STATIC, *MULTITURN, "bon"]
CAPABILITY = ["xstest", "mmlu", "gsm8k"]
WAVE_LENS_MODE = "stacked"
OVERLAY_CELLS = ["constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"]
MAIN_CELLS = ["none", *OVERLAY_CELLS, "constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2-mix-full__lora_r4_ce_minus_ctl",
              "safety_refusal__psr_kl_all", "safety_refusal__lora_r4"]
IT_CELLS = ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"]
NEMO_CELLS = ["none", *OVERLAY_CELLS]
CELL_LABEL = {
    "none": "none (bare instruct, same wave)",
    "constv2-mix-full__psr_ce_all": "BASE-fit PSR (CE, constitution corpus 42.8M)",
    "constv2-mix-full__lora_r4_ce": "BASE-fit LoRA r4 (CE, constitution corpus 42.8M)",
    "constv2ctl2-full__psr_ce_all": "BUDGET-MATCHED CONTROL · PSR (CE, control corpus 42.8M)",
    "constv2ctl2-full__lora_r4_ce": "BUDGET-MATCHED CONTROL · LoRA r4 (CE, control corpus 42.8M)",
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": "DERIVED: constitution PSR minus control, distilled (not on the overlay)",
    "constv2-mix-full__lora_r4_ce_minus_ctl": "DERIVED: constitution LoRA minus control LoRA (not on the overlay)",
    "safety_refusal__psr_kl_all": "INCUMBENT safety PSR (KL) — reference, not on the overlay",
    "safety_refusal__lora_r4": "INCUMBENT safety LoRA r4 (KL) — reference, not on the overlay",
}
IT_LABEL = {
    "none": "none (bare instruct, same wave, it-view serving job)",
    "constv2-mix-full__psr_ce_all": "INSTRUCT-DIRECT PSR (CE, constitution corpus 42.8M; fit on the instruct checkpoint)",
    "constv2-mix-full__lora_r4_ce": "INSTRUCT-DIRECT LoRA r4 (CE, constitution corpus 42.8M; fit on the instruct checkpoint)",
}
CELL_GROUP = {
    "none": "baseline_full22",
    "constv2-mix-full__psr_ce_all": "base_fit_full", "constv2-mix-full__lora_r4_ce": "base_fit_full",
    "constv2ctl2-full__psr_ce_all": "budget_matched_control", "constv2ctl2-full__lora_r4_ce": "budget_matched_control",
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": "derived", "constv2-mix-full__lora_r4_ce_minus_ctl": "derived",
    "safety_refusal__psr_kl_all": "incumbent_reference", "safety_refusal__lora_r4": "incumbent_reference",
}
IT_GROUP = {"none": "baseline_full22_it", "constv2-mix-full__psr_ce_all": "instruct_direct", "constv2-mix-full__lora_r4_ce": "instruct_direct"}
MODELS = {
    "gemma4e4b": {"label": "google/gemma-4-E4B-it", "view": "main", "base": "gemma4e4b", "cells": MAIN_CELLS, "petri_tag": "gemma4e4b",
                  "grades": {a: RS / "grades/gemma4e4b" for a in [*STATIC, "bon"]} | {a: RS / "grades_rest/gemma4e4b" for a in MULTITURN},
                  "ns": {a: "full22" for a in [*STATIC, "bon"]} | {a: "full22_rest" for a in MULTITURN}},
    "gemma4e4b_it": {"label": "google/gemma-4-E4B-it — it view (instruct-direct objects)", "view": "it", "base": "gemma4e4b", "cells": IT_CELLS,
                     "petri_tag": "gemma4e4b_it", "grades": {a: RS / "grades_rest/gemma4e4b_it" for a in ATTACK_READOUTS},
                     "ns": {a: "full22_rest" for a in ATTACK_READOUTS}},
    "qwen35_4b": {"label": "Qwen/Qwen3.5-4B (instruct)", "view": "main", "base": "qwen35_4b", "cells": MAIN_CELLS, "petri_tag": "qwen35_4b",
                  "grades": {a: RS / "grades/qwen35_4b" for a in [*STATIC, "bon"]} | {a: RS / "grades_rest/qwen35_4b" for a in MULTITURN},
                  "ns": {a: "full22" for a in [*STATIC, "bon"]} | {a: "full22_rest" for a in MULTITURN}},
    "qwen35_4b_it": {"label": "Qwen/Qwen3.5-4B (instruct) — it view (instruct-direct objects)", "view": "it", "base": "qwen35_4b", "cells": IT_CELLS,
                     "petri_tag": "qwen35_4b_it", "grades": {a: RS / "grades_rest/qwen35_4b_it" for a in ATTACK_READOUTS},
                     "ns": {a: "full22_rest" for a in ATTACK_READOUTS}},
    "nemotron3nano4b": {"label": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "view": "main", "base": "nemotron3nano4b", "cells": NEMO_CELLS,
                        "petri_tag": "nemotron3nano4b", "grades": {a: RS / "grades_rest/nemotron3nano4b" for a in ATTACK_READOUTS},
                        "ns": {a: "full22_rest" for a in ATTACK_READOUTS}},
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_json(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def pct(x):
    return f"{100 * x:.1f}"


def annotate(rec: dict, deg: dict | None) -> str:
    if rec.get("status"):
        return f"{rec['status']}: {rec.get('reason')}"
    s = f"ASR {pct(rec['p'])}% [{pct(rec['lo'])}, {pct(rec['hi'])}] ({rec['k']}/{rec['n']} goals)"
    if deg:
        rl, gl = deg["response_level"], deg["goal_level"]
        s += (f" | at-cap {pct(rl['at_cap']['p'])}% | loop {pct(rl['loop']['p'])}% of {rl['at_cap']['n']} responses"
              f" | defended* {gl['defended_with_degeneration']['k']}/{gl['defended']['k']} defended goals via a degenerate refusal"
              f" (loop {gl['defended_with_refused_loop']['k']}, at-cap {gl['defended_with_refused_at_cap']['k']})")
    return s


def degeneracy_block(deg: dict | None) -> dict:
    if not deg:
        return {"status": "PENDING", "reason": "no degeneracy record for this cell/attack"}
    rl, gl = deg["response_level"], deg["goal_level"]
    return {"verified_against_aggregate": deg["verified"], "n_responses": deg["n_responses"], "at_cap_definition": deg["at_cap_definition"],
            "at_cap": rl["at_cap"], "at_cap_tok": rl["at_cap_tok"], "at_cap_tok98": rl["at_cap_tok98"], "loop": rl["loop"],
            "degenerate": rl["degenerate"], "refused": rl["refused"], "refused_and_degenerate": rl["refused_and_degenerate"],
            "refused_and_loop": rl["refused_and_loop"], "refused_and_at_cap": rl["refused_and_at_cap"],
            "degenerate_among_refused": rl["degenerate_among_refused"], "jailbroken_and_loop": rl["jailbroken_and_loop"],
            "jailbroken_and_at_cap": rl["jailbroken_and_at_cap"], "median_n_tokens": rl["median_n_tokens"],
            "goal_level": {k: gl[k] for k in ("defended", "defended_with_degeneration", "defended_with_degeneration_among_defended",
                                              "defended_with_refused_loop", "defended_with_refused_at_cap", "defended_clean",
                                              "any_at_cap", "any_loop", "jb_x_degenerate_2x2", "jb_x_loop_2x2")}}


def petri_block(pt: dict | None) -> dict:
    if not pt:
        return {"status": "PENDING", "reason": "no Petri cell for this transcripts cell id"}
    return {"status": pt["status"], "rerun_state": pt.get("rerun_state"), "ledger_snapshot": pt.get("ledger_snapshot"),
            "audited_items": pt["audited_items"], "audited_expected": pt["audited_expected"],
            "petri_run_key": pt["petri_run_key"], "petri_run_key_source": pt["petri_run_key_source"], "intervention_hash": pt["intervention_hash"],
            "n_unresolved_ledger_entries": pt["n_unresolved_ledger_entries"], "unresolved": pt["unresolved"],
            "auditor_chain": pt["auditor_chain"], "mean_of_behavior_means": pt["mean_of_behavior_means"],
            "behaviors": {b: {k: v[k] for k in ("n_items", "n_expected", "mean", "lo", "hi", "sd", "sd_lo", "sd_hi", "small_n",
                                                "dispersion_caveat", "eval_awareness_mean", "scenario_realism_mean",
                                                "incomplete", "unresolved_scenarios")} for b, v in pt["behaviors"].items()},
            "model_roles": pt["model_roles"].get("model_roles_distinct")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RS / "transfer_stats_full22_all.json"))
    ap.add_argument("--petri", default=str(RS / "petri_stats/petri_full22_stats.json"))
    ap.add_argument("--degeneracy", default=str(RS / "degeneracy/degeneracy_full22.json"))
    ap.add_argument("--prior", default=str(RS / "transfer_stats_full22.json"), help="FINAL-5's file: its meta is carried, its numbers cross-checked")
    ap.add_argument("--wave-manifest", default=str(RS / "WAVE_MANIFEST.json"))
    args = ap.parse_args(argv)

    petri = load_json(Path(args.petri)) or {"tags": {}}
    degen = load_json(Path(args.degeneracy)) or {"cells": {}, "meta": {}}
    prior = load_json(Path(args.prior)) or {}
    wm = load_json(Path(args.wave_manifest)) or {}
    pmeta = prior.get("meta", {})

    out: dict = {"meta": {
        "threshold": float(T1.THRESHOLD), "bon_n_max": int(T1.BON_N), "interval": "Wilson score, 95%",
        "pairing": "10,000-draw percentile bootstrap over goals, seed 0; exact McNemar on discordant pairs; WITHIN-WAVE, within-VIEW (same target's same-view bare none)",
        "namespaces": {
            "full22": {"dir": "grades", "label": "FINAL-5 lane: gemma4e4b / qwen35_4b MAIN view, four static families + BoN (graded 2026-09-07 18:34-19:07Z; reused unchanged)",
                       "graded_by": "w6_final5/grade_full22.py", "cells": {m: MAIN_CELLS for m in ("gemma4e4b", "qwen35_4b")},
                       "readouts": [*STATIC, "bon"]},
            "full22_rest": {"dir": "grades_rest", "label": "this lane: main-view multi-turn legs (gemma4e4b, qwen35_4b), the it views (all readouts), nemotron3nano4b (all readouts)",
                            "graded_by": "w7_rest/grade_full22_rest.py",
                            "cells": {"gemma4e4b": MAIN_CELLS, "qwen35_4b": MAIN_CELLS, "gemma4e4b_it": IT_CELLS, "qwen35_4b_it": IT_CELLS, "nemotron3nano4b": NEMO_CELLS},
                            "readouts": {"gemma4e4b": MULTITURN, "qwen35_4b": MULTITURN, "gemma4e4b_it": ATTACK_READOUTS, "qwen35_4b_it": ATTACK_READOUTS,
                                         "nemotron3nano4b": ATTACK_READOUTS}},
        },
        "readouts": {"static": STATIC, "multi_turn": MULTITURN, "bon": "bon", "capability": CAPABILITY, "alignment_audit": "petri"},
        "cell_labels": {"main": CELL_LABEL, "it": IT_LABEL}, "cell_groups": {"main": CELL_GROUP, "it": IT_GROUP}, "overlay_cells": OVERLAY_CELLS,
        "gated_goal_n": 259, "bon_goal_subset": "jbb (100 confirmatory goals)", "lens_reduction_mode": WAVE_LENS_MODE,
        "lens_reduction_rule": pmeta.get("lens_reduction_rule"),
        "control_corpus_tokens": pmeta.get("control_corpus_tokens"), "control_corpus_docs": pmeta.get("control_corpus_docs"),
        "control_corpus_note": pmeta.get("control_corpus_note"), "capability_controls": pmeta.get("capability_controls"),
        "decode_caps": {"static_families": 4096, "multi_turn_target_reply": 512, "bon": 512},
        "degeneracy": {"source": args.degeneracy, "source_sha256": (sha256_file(Path(args.degeneracy)) if Path(args.degeneracy).exists() else None),
                       **{k: degen.get("meta", {}).get(k) for k in ("loop_definition", "loop_rule", "definition_source", "definition_source_sha256",
                                                                    "static_at_cap_reading", "asterisk_rule")},
                       "calibration": degen.get("calibration")},
        "petri": {"source": args.petri, "source_sha256": (sha256_file(Path(args.petri)) if Path(args.petri).exists() else None),
                  "judge": petri.get("meta", {}).get("judge"), "interval": petri.get("meta", {}).get("interval"), "pairing": petri.get("meta", {}).get("pairing"),
                  "overflow_rerun": petri.get("meta", {}).get("overflow_rerun")},
        "readers": {"transfer_stats": str(Path(T1.__file__)), "transfer_r2": str(Path(R2.__file__)),
                    "transfer_stats_sha256": sha256_file(Path(T1.__file__)), "transfer_r2_sha256": sha256_file(Path(R2.__file__)),
                    "this_builder_sha256": sha256_file(Path(__file__).resolve())},
        "grader_entry_point": pmeta.get("grader_entry_point"),
        "wave": {"wave": wm.get("wave"), "manifest_written_at": wm.get("written_at"), "generation_shas": {k: v.get("sha") for k, v in (wm.get("shas") or {}).items()},
                 "serving_impl_commit": (wm.get("evidence") or {}).get("serving_impl_commit"), "verified_complete": wm.get("verified_complete"),
                 "petri": wm.get("petri"), "manifest_sha256": (sha256_file(Path(args.wave_manifest)) if Path(args.wave_manifest).exists() else None),
                 "nemotron_note": "nemotron3nano4b legs were generated from the full22-newmodels tree (5 cells); grading code byte-identical to full22"},
        "written_at": datetime.now(timezone.utc).isoformat(),
    }, "namespaces": {"full22": {"models": {}}, "full22_rest": {"models": {}}}, "models": {}, "cross_view": {}, "drift": {}}

    judges: dict = {}
    per_model_outcomes: dict = {}
    for model, spec in MODELS.items():
        aggs: dict = {}
        gps: dict = {}
        for a in ATTACK_READOUTS:
            p = spec["grades"][a] / f"two_si_{a}_aggregate.json"
            aggs[a] = load_json(p)
            gp = spec["grades"][a] / "grading_provenance.json"
            gps[spec["ns"][a]] = load_json(gp)
        present = [a for a in ATTACK_READOUTS if aggs[a]]
        pending = [a for a in ATTACK_READOUTS if not aggs[a]]
        labels = CELL_LABEL if spec["view"] == "main" else IT_LABEL
        groups = CELL_GROUP if spec["view"] == "main" else IT_GROUP
        ptag = petri.get("tags", {}).get(spec["petri_tag"]) or {}
        rows: dict = {}
        outcomes: dict = {a: {} for a in ATTACK_READOUTS}
        for c in spec["cells"]:
            row: dict = {"label": labels.get(c, f"{c} (extra)"), "group": groups.get(c, "extra"), "view": spec["view"]}
            for a in ATTACK_READOUTS:
                agg = aggs[a]
                if not agg or c not in agg["cells"]:
                    row[a] = R2.pending(f"{a} aggregate absent" if not agg else f"cell absent from {a} aggregate")
                    continue
                rec = R2._safe(T1.bon_cell if a == "bon" else T1.gated_cell, agg, c)
                if rec.get("status"):
                    row[a] = rec; continue
                cellrow = agg["cells"][c]
                if cellrow.get("lens_reduction_mode") != WAVE_LENS_MODE:
                    raise SystemExit(f"{model}/{c}/{a}: aggregate row lens_reduction_mode != {WAVE_LENS_MODE}")
                rec["namespace"] = spec["ns"][a]
                rec["lens_reduction_mode"] = WAVE_LENS_MODE
                rec["tier"] = "bon" if a == "bon" else "static" if a in STATIC else "multi_turn"
                rec["decode_cap_tokens"] = cellrow.get("decode_cap_tokens", 512 if a == "bon" else 4096 if a in STATIC else 512)
                rec["completion"] = (cellrow.get("completion") or {}).get("bon" if a == "bon" else "gated")
                if a != "bon":
                    m = cellrow["attack_metrics"]
                    rec["mean_n_target_forwards"] = m.get("mean_n_target_forwards")
                    rec["n_capped_goals"] = m.get("n_capped")
                    rec["by_category_group"] = {g: {"n_goals": v["n_goals"], "n_jailbroken": v["n_jailbroken"], "asr": v["asr"]}
                                                for g, v in (m.get("by_category_group") or {}).items()}
                d = degen["cells"].get(f"{model}|{a}|{c}")
                rec["degeneracy"] = degeneracy_block(d)
                rec["asr_annotated"] = annotate(rec, d)
                row[a] = rec
                outcomes[a][c] = R2.bon_outcomes(agg, c) if a == "bon" else R2.attack_outcomes(agg, c)
            for k in CAPABILITY:
                row[k] = {"status": "PENDING", "reason": "no capability controls exist for the full22 cells (serving provenance refusal in the wave)"}
            row["petri"] = petri_block((ptag.get("cells") or {}).get(c))
            row["lens_reduction_mode"] = WAVE_LENS_MODE
            row["completion"] = {a: (row[a].get("completion") if not row[a].get("status") else None) for a in ATTACK_READOUTS}
            row["on_overlay"] = c in OVERLAY_CELLS
            rows[c] = row
        per_model_outcomes[model] = outcomes
        # judges: one block per namespace present for this model
        jb: dict = {}
        for ns, gp in gps.items():
            if not gp:
                jb[ns] = {"status": "PENDING", "reason": "grading_provenance.json absent (grading not finished)"}; continue
            j = gp.get("judge") or {}
            if j.get("judge_model") != "claude-haiku-4-5" or j.get("rubric") != "strongreject_v1":
                raise SystemExit(f"{model}/{ns}: judge {j.get('judge_model')} / {j.get('rubric')} is not the pinned claude-haiku-4-5 / strongreject_v1")
            jb[ns] = {"model": j.get("judge_model"), "rubric": j.get("rubric"), "cache_kind": j.get("cache_kind"), "pass_id": j.get("pass_id"),
                      "pass_id_definition": j.get("pass_id_definition"), "cache_rows": j.get("cache_rows"), "cache_rows_seeded": j.get("cache_rows_seeded"),
                      "org": j.get("org"), "org_final": gp.get("org_final"), "escalated_to_highprio": gp.get("escalated_to_highprio"),
                      "org_events": gp.get("org_events"), "n_429": gp.get("n_429"), "wall_s": gp.get("wall_s"),
                      "concurrency_initial": j.get("concurrency_initial"), "n_judgments_estimated": gp.get("n_judgments_estimated"),
                      "unscored_retry_events": gp.get("unscored_retry_events"), "attacker_identity_check": gp.get("attacker_identity_check"),
                      "finished_at": gp.get("finished_at")}
        judges[model] = jb
        petri_judge = ptag and {"judge": [mr.get("judge") for b in ptag.get("cells", {}).values() for mr in (b["model_roles"].get("model_roles_distinct") or [])][:1],
                                "n_cells": ptag.get("n_cells"), "n_complete": ptag.get("n_complete"), "n_incomplete_ledgered": ptag.get("n_incomplete_ledgered")}
        m = {"label": spec["label"], "view": spec["view"], "base_target": spec["base"], "namespace_by_readout": spec["ns"],
             "attacks_present": present, "attacks_pending": pending, "expected_cells": spec["cells"],
             "expected_missing": {c: R2.pending(f"expected cell `{c}` absent") for c in spec["cells"] if all(not aggs[a] or c not in aggs[a]["cells"] for a in ATTACK_READOUTS)},
             "served_alpha": next((aggs[a].get("served_alpha") for a in ATTACK_READOUTS if aggs[a]), None), "lens_reduction_mode": WAVE_LENS_MODE,
             "judge": {"strongreject": jb, "petri": petri_judge or {"status": "PENDING"}}, "cells": rows, "vs_bare": {}, "petri_vs_bare": (ptag.get("vs_bare") or {}), "petri_vs_bare_cell": (ptag.get("vs_bare_cell") or {}),
             "equivalence_bounds": {}, "completion": {c: rows[c]["completion"] for c in rows}}
        for c in rows:
            if c == "none":
                continue
            m["vs_bare"][c] = {}
            for a in ATTACK_READOUTS:
                pa, pb = outcomes[a].get("none"), outcomes[a].get(c)
                m["vs_bare"][c][a] = R2.contrast(rows["none"][a], rows[c][a], pa, pb)
            for k in CAPABILITY:
                m["vs_bare"][c][k] = R2.pending("readout pending on one side")
            if groups.get(c) == "budget_matched_control":
                m["equivalence_bounds"][c] = {a: (m["vs_bare"][c][a].get("equivalence_bound_pp") if "paired" in m["vs_bare"][c][a] else None) for a in ATTACK_READOUTS}
        out["models"][model] = m
        for ns in ("full22", "full22_rest"):
            if any(v == ns for v in spec["ns"].values()):
                out["namespaces"][ns]["models"][model] = {"readouts": [a for a, v in spec["ns"].items() if v == ns], "cells": spec["cells"]}
        out["drift"][model] = R2.not_in_design("no regrade cell in this namespace's design")

    # cross-view: it vs main for gemma / qwen (same goals, same wave, both stacked)
    for base in ("gemma4e4b", "qwen35_4b"):
        it = base + "_it"
        if it not in out["models"] or base not in out["models"]:
            continue
        cv: dict = {}
        for c in IT_CELLS:
            cv[c] = {}
            for a in ATTACK_READOUTS:
                ra, rb = out["models"][base]["cells"].get(c, {}).get(a), out["models"][it]["cells"].get(c, {}).get(a)
                if not ra or not rb:
                    cv[c][a] = R2.pending("cell absent on one side"); continue
                pa, pb = per_model_outcomes[base][a].get(c), per_model_outcomes[it][a].get(c)
                cv[c][a] = R2.contrast(ra, rb, pa, pb)   # it minus main ("cell" = it, "bare" = main)
        out["cross_view"][base] = {"note": "it-view cell MINUS main-view cell of the same id, paired on the shared goals; `none` vs `none` is the "
                                           "wave's bare-replicate noise floor (two independent greedy/sampled generations of the same bare model)",
                                   "cells": cv}

    # cross-check: every FINAL-5 number this file carries equals FINAL-5's own file
    mism = []
    for model in ("gemma4e4b", "qwen35_4b"):
        pm = (prior.get("models") or {}).get(model, {}).get("cells", {})
        for c, prow in pm.items():
            for a in [*STATIC, "bon"]:
                mine = out["models"][model]["cells"].get(c, {}).get(a, {})
                if prow.get(a, {}).get("k") is not None and (mine.get("k"), mine.get("n")) != (prow[a]["k"], prow[a]["n"]):
                    mism.append(f"{model}/{c}/{a}: {mine.get('k')}/{mine.get('n')} vs FINAL-5 {prow[a]['k']}/{prow[a]['n']}")
    if mism:
        raise SystemExit("FINAL-5 cross-check FAILED: " + "; ".join(mism[:10]))
    out["meta"]["final5_cross_check"] = {"status": "IDENTICAL", "prior": args.prior, "prior_sha256": (sha256_file(Path(args.prior)) if Path(args.prior).exists() else None),
                                         "n_records_compared": sum(1 for model in ("gemma4e4b", "qwen35_4b") for c in (prior.get("models") or {}).get(model, {}).get("cells", {}) for _ in [*STATIC, "bon"])}
    out["meta"]["judge"] = {"strongreject": {"model": "claude-haiku-4-5", "rubric": "strongreject_v1", "cache_kind": "sr", "per_model": judges,
                                             "statement": "pinned explicitly (not inherited) in both namespaces: the same judge model, rubric and score formula as the r2/r3 aggregates; the two halves pool"},
                            "petri": petri.get("meta", {}).get("judge")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    # console headline
    for model, mb in out["models"].items():
        print(f"\n== {model} ({mb['view']})  present={mb['attacks_present']} pending={mb['attacks_pending']}")
        for c, r in mb["cells"].items():
            parts = []
            for a in ATTACK_READOUTS:
                x = r[a]
                if x.get("status"):
                    parts.append(f"{a}:—"); continue
                d = x.get("degeneracy", {})
                cap = d.get("at_cap", {}).get("p"); lp = d.get("loop", {}).get("p")
                star = d.get("goal_level", {}).get("defended_with_degeneration", {}).get("k")
                parts.append(f"{a}:{pct(x['p'])}%" + (f"(cap{pct(cap)}/loop{pct(lp)}/*{star})" if cap is not None else ""))
            pt = r["petri"]
            pstr = f" petri:{pt['mean_of_behavior_means']:.2f}[{pt['status'][:4]}]" if not pt.get("status", "").startswith("PEND") and pt.get("mean_of_behavior_means") is not None else ""
            print(f"  {c:50s} " + " ".join(parts) + pstr)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
