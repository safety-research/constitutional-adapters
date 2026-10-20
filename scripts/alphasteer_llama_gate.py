"""Phase D of alphasteer-full22 — the Llama layer-rule validation, the
campaign's STOP GATE.

    python scripts/alphasteer_llama_gate.py --config configs/alphasteer_f22_llama8b_gate.yaml
        [--resume] [--force] [--device cuda] [--batch-size 32] [--figs-dir DIR]
        [--gate-out llama_gate.json] [--released steering_matrix_llama3.1.pt]
        [--wandb-project P] [--skip-refit]

TWO readings, one verdict:

1. **The gate (`reading_released`)** — the F.9 rule applied to the AUTHORS'
   RELEASED matrices exactly as the spec specifies. For each of the 32
   released slices, ``‖H @ M_l‖`` on the held-out rows through the ONE
   scoring helper the screen uses (``alphasteer_screen.deployed_score``, the
   deployed row-vector orientation — never the transpose) -> AUROC
   harmful-over-benign. **Eligible = AUROC >= 0.95 inside the 15-65 % depth
   band, never layer 0 — AUROC ONLY**, the paper's Appendix F.9 criterion,
   through the ONE predicate the screen uses (``alphasteer_screen
   .layer_eligible``). Leakage for a paper layer — ``benign_energy_kept`` of
   the projector RECONSTRUCTED from the pinned authors' benign FIT capture at
   the paper's published ratio (``PAPER_LLAMA31_NULLSPACE_RATIOS``; the
   released ``M_l = P_l·Δ̃_l`` is rank 1 and identifies no projector, but
   ``P_l`` is defined by the benign Gram and the ratio, both of which we
   hold, plan D.1) — is still computed and REPORTED per layer as a
   diagnostic. It is NOT an eligibility input: the first run of this
   gate, under the retired conjunctive rule (the AUROC bound plus a 1e-3
   leakage bound), recovered 0/10 because the 1e-3 measured the rank
   deficiency of an n=2,000 < d=4,096 fit, not the projector; AUROC alone
   recovered 8/10. Recovery =
   ``|eligible ∩ PAPER_LLAMA31_LAYERS|`` of 10; ``passed = n_recovered >= 7``.

   Handled and RECORDED: only 10 slices are non-zero (the paper's set);
   the other 22 are exactly 0.0, score AUROC 0.5 by construction and are
   never eligible — expected, not a bug. The released reading therefore tests
   whether the rule ACCEPTS the paper's layers; it cannot test rejection.

2. **The refit (`reading_refit_32`)** — all 32 Llama layers refit by us from
   the pinned lists through ``scripts/alphasteer_screen.run_screen`` under the
   same rule: the rejection-side evidence. Reported BESIDE the gate; it can
   NEVER set ``passed`` (:func:`gate_verdict` reads the released reading only;
   test-pinned). Runs only when the gate PASSED — on a failed gate the job
   writes the verdict, exits 3, and fits nothing.

Products (every artifact fingerprinted tmp -> os.replace,
sha256 + code_sha, ``.COMPLETE`` strictly after the payload):

- ``<out_dir>/released_norms.json`` — the D.0 probe (per-slice Frobenius,
  the non-zero set vs the paper set); a separate cheap artifact;
- ``llama_gate.json`` — ``passed``, ``gate_reading``, both readings, the
  recovery count, the full-depth AUROC curve, the capture binding. Phase B's
  submission gate and every Phase B body read this file;
- ``screen_f9.json`` (+ split, acts, rows) from the refit, carrying the same
  ``recovery`` block;
- ``final_llama_recovery.png`` and ``final_auroc_vs_depth_by_p_llama8b.png``
  in ``--figs-dir``.

Exit codes: 0 gate passed (+ refit complete); **3 gate FAILED — STOP and ask
the user**; 4 gate passed but the refit hit its own stop (verdict stands);
2 the released file is not the one documented above; anything else is a
failure.

Resume: the capture is reused through the screen's sentinel binding
(never redone on a requeue); a COMPLETE ``llama_gate.json`` under the same
inputs is re-read rather than recomputed; the refit resumes per (layer, p).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import alphasteer_screen as screen_cli  # noqa: E402
import fit_class_b as fcb  # noqa: E402

GATE_SCHEMA = "alphasteer_llama_gate/1"
NORMS_SCHEMA = "alphasteer_released_norms/1"
GATE_READING = "reading_released"          # THE gate
REFIT_READING = "reading_refit_32"         # beside it; can never set `passed`
DEFAULT_RECOVERY_THRESHOLD = 7
EXIT_GATE_FAILED = 3
EXIT_REFIT_A5_STOP = 4
EXIT_RELEASED_FILE_MISMATCH = 2

D8_NOTE = ("D8 (measured 2026-09-08, re-verified at load): the release is a "
           "32-slot array carrying the paper's 10 matrices — only the paper's "
           "layers are non-zero; the other 22 slices are exactly 0.0, score "
           "AUROC 0.5 by construction and are never eligible. Expected, not a "
           "bug. Consequently the released reading tests whether the rule "
           "ACCEPTS the paper's layers and cannot test whether it REJECTS the "
           "layers the paper omitted; the refit reading beside it is that "
           "rejection-side evidence and cannot set `passed`.")
LEAKAGE_SOURCE = ("projector RECONSTRUCTED from the pinned authors' benign FIT "
                  "capture at the paper's per-layer ratio (null_space_projector, "
                  "the same helper and fit split the screen uses); the released "
                  "matrix is rank 1 and identifies no projector, but none is "
                  "needed — see plan D.1")
QUALIFICATIONS = (
    "It is OUR projector from the authors' benign list, not a recovered copy "
    "of theirs; it matches theirs only insofar as our capture matches theirs "
    "(double-BOS fixed on this model first — plan §A deviation 2). A material "
    "pipeline divergence would show up as an unexpectedly large leakage.",
    "The 22 zero slices have no published ratio, so leakage is undefined for "
    "them and recorded as null, never imputed; they fail the AUROC half "
    "regardless.",
    "operator_benign_energy (‖H_b @ M_l‖²/‖H_b‖², the benign energy the "
    "deployed edit actually produces) is reported beside the projector "
    "leakage as a comparable diagnostic. NEITHER quantity is an eligibility "
    "input (D38): both are diagnostics, drawn against the retired 1e-3 bound "
    "as a reference line only.",
)
LEAKAGE_DIAGNOSTIC_NOTE = (
    "Leakage is REPORTED, not gated (D38, 2026-09-09). The first run of this gate "
    "(job 470766) recovered 0/10 under 'AUROC >= 0.95 AND leakage <= 1e-3': the "
    "AUROC half alone recovered 8/10 [8, 9, 12, 13, 14, 16, 18, 19]; the leakage "
    "half rejected all 10 (2.2e-3 at layer 12 .. 1.1e-2 at layer 19). The 1e-3 "
    "bound had been read off an n=2,000 < d=4,096 fit whose rank-deficient benign "
    "Gram left >= 2,096 of the 2,457 kept null-space directions exactly null, so "
    "its in-sample leakage collapsed to ~1e-4 by construction; on the authors' "
    "full-rank n=12,599 list the identical in-sample quantity is honestly "
    "2e-3..1e-2 — 172x below the isotropic reference p. The bound measured the "
    "discarded dataset, not the projector, and it was ours, not the paper's "
    "(Appendix F.9 is norm separability alone). It is removed from eligibility "
    "and kept as a per-layer diagnostic; recalibrating it to admit the paper's "
    "layers would have been fitting the threshold to the answer.")
REJECTION_TESTS = ("REJECTION side — our own fit of every layer under the same "
                   "rule; the evidence the released file structurally cannot "
                   "give. Reported beside the gate; can never set `passed`.")


def _log(msg: str) -> None:
    print(f"[alphasteer_llama_gate] {msg}", flush=True)


def _sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# D.0 — what the released file contains, re-verified at load                   #
# --------------------------------------------------------------------------- #


def released_norm_probe(path, *, n_blocks: int, hidden_dim: int,
                        paper_layers: list[int], zero_norm_tol: float = 0.0,
                        file_sha256: bool = True):
    """Load the released tensor (mmap) and record, per slice, its Frobenius
    norm; the non-zero set; and whether that set equals the paper's layer
    set. Returns ``(tensor, probe_payload)``. Refuses a file that is not
    ``[n_blocks, hidden_dim, hidden_dim]`` — the artifact would not describe
    this model."""
    import torch

    path = Path(path)
    R = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(R, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor, got {type(R).__name__}")
    if tuple(R.shape) != (n_blocks, hidden_dim, hidden_dim):
        raise ValueError(f"{path}: shape {tuple(R.shape)} is not "
                         f"[{n_blocks}, {hidden_dim}, {hidden_dim}] — not this model's file")
    norms = {str(l): float(R[l].float().norm()) for l in range(n_blocks)}
    nonzero = [l for l in range(n_blocks) if norms[str(l)] > zero_norm_tol]
    zero = [l for l in range(n_blocks) if l not in nonzero]
    probe = {
        "schema": NORMS_SCHEMA,
        "path": str(path),
        "released_file_sha256": _sha256_file(path) if file_sha256 else None,
        "shape": [int(x) for x in R.shape], "dtype": str(R.dtype),
        "zero_norm_tol": float(zero_norm_tol),
        "per_slice_frobenius": norms,
        "nonzero_layers": nonzero, "n_nonzero": len(nonzero),
        "zero_layers": zero, "n_zero": len(zero),
        "all_layers_nonzero": len(nonzero) == n_blocks,
        "paper_layers": [int(l) for l in paper_layers],
        "nonzero_equals_paper_set": set(nonzero) == set(int(l) for l in paper_layers),
        "note": D8_NOTE,
    }
    return R, probe


# --------------------------------------------------------------------------- #
# the two readings and the ONE verdict                                         #
# --------------------------------------------------------------------------- #


def annotate_band(reading: dict, *, n_blocks: int, depth_band, paper_layers) -> dict:
    """Apply the band half of the rule to the library's released reading:
    ``depth`` / ``in_band`` per layer, ``eligible`` = the library's eligible
    AND in band; ``recovered`` / ``n_recovered`` / ``passed`` recomputed from
    the band-applied set (the library's any-depth values are kept as
    ``recovered_any_depth``). On Llama-3.1 (32 blocks) every paper layer is
    in band, so the two agree; the annotation keeps the rule as stated."""
    from sisteer.extraction import alphasteer_screen as scr

    out = dict(reading)
    per = {}
    for k, v in reading["per_layer"].items():
        l = int(k)
        in_band = scr.in_depth_band(l, n_blocks, tuple(depth_band))
        row = dict(v)
        row["depth"] = l / float(n_blocks)
        row["in_band"] = in_band
        row["eligible_any_depth"] = bool(v["eligible"])
        row["eligible"] = bool(v["eligible"]) and in_band
        per[k] = row
    recovered = sorted(int(l) for l in paper_layers if per[str(int(l))]["eligible"])
    out["per_layer"] = per
    out["depth_band"] = [float(depth_band[0]), float(depth_band[1])]
    out["recovered_any_depth"] = list(reading["recovered"])
    out["recovered"] = recovered
    out["n_recovered"] = len(recovered)
    out["paper_layers_not_recovered"] = sorted(int(l) for l in paper_layers
                                               if int(l) not in recovered)
    out["passed"] = len(recovered) >= int(reading["threshold"])
    thr = reading.get("auroc_threshold", "auroc_threshold")
    out["rule"] = (f"AUROC >= {thr} inside the {depth_band[0]:.0%}-{depth_band[1]:.0%} "
                   "depth band, never layer 0 — AUROC ONLY (Appendix F.9); leakage is "
                   "reported per layer as a diagnostic, not an eligibility input (D38)")
    return out


def refit_reading_from_selection(per_layer: list[dict], *, paper_layers) -> dict:
    """``reading_refit_32`` from the screen's ``select_layers`` per-layer
    table: eligible-in-band under the same rule, split into the paper layers
    it recovers, the false positives outside the paper set, and the paper
    layers it declines."""
    paper = sorted(int(l) for l in paper_layers)
    eligible = sorted(int(pl["layer"]) for pl in per_layer
                      if pl["eligible"] and pl["in_band"])
    recovered = [l for l in eligible if l in paper]
    return {
        "rule": "the screen's F.9 rule (select_layers): argmax_p AUROC, eligible "
                "iff AUROC >= threshold, in band, never layer 0 — AUROC ONLY; "
                "leakage reported per layer as a diagnostic (D38)",
        "recovered": recovered, "n_recovered": len(recovered),
        "false_positives_outside_paper_set": [l for l in eligible if l not in paper],
        "paper_layers_not_recovered": [l for l in paper if l not in eligible],
        "eligible_in_band": eligible,
        "per_layer": {str(int(pl["layer"])): {
            "auroc": float(pl["auroc"]), "leakage": float(pl["leakage"]),
            "p_star": float(pl["p_star"]), "eligible": bool(pl["eligible"]),
            "in_band": bool(pl["in_band"]), "depth": float(pl["depth"]),
            "reason": pl.get("reason")} for pl in per_layer},
        "tests": REJECTION_TESTS,
        "can_set_passed": False,
    }


def gate_verdict(released_reading: dict, *, threshold: int = DEFAULT_RECOVERY_THRESHOLD) -> dict:
    """THE verdict: from the released reading ONLY. No other input can
    reach ``passed`` — the refit is not a parameter of this function."""
    n = int(released_reading["n_recovered"])
    return {"gate_reading": GATE_READING, "n_recovered": n,
            "threshold": int(threshold), "passed": n >= int(threshold)}


def build_recovery_block(*, released_reading: dict, refit_reading: dict | None,
                         paper_layers, paper_ratios: dict, norms_probe: dict,
                         threshold: int = DEFAULT_RECOVERY_THRESHOLD) -> dict:
    """The ``recovery`` block: both readings side by side, the
    verdict from :func:`gate_verdict` (released only)."""
    from sisteer.extraction.alphasteer_screen import ELIGIBILITY_RULE

    verdict = gate_verdict(released_reading, threshold=threshold)
    return {
        "eligibility": ELIGIBILITY_RULE,
        "leakage_note": LEAKAGE_DIAGNOSTIC_NOTE,
        "paper_layers": [int(l) for l in paper_layers],
        "paper_nullspace_ratios": {str(int(k)): float(v) for k, v in paper_ratios.items()},
        "released_matrix_norms": norms_probe["per_slice_frobenius"],
        "released_nonzero_layers": norms_probe["nonzero_layers"],
        "released_all_layers_nonzero": norms_probe["all_layers_nonzero"],
        "released_nonzero_equals_paper_set": norms_probe["nonzero_equals_paper_set"],
        "d8_note": D8_NOTE,
        "leakage_source": LEAKAGE_SOURCE,
        "qualifications": list(QUALIFICATIONS),
        GATE_READING: released_reading,
        REFIT_READING: refit_reading,
        **verdict,
    }


def exit_code_for(*, passed: bool, refit_rc: int | None) -> int:
    """0 passed (+ refit ok / skipped); 3 gate failed; 4 gate passed but the
    refit A5-stopped; otherwise the refit's own non-zero rc."""
    if not passed:
        return EXIT_GATE_FAILED
    if refit_rc in (None, 0):
        return 0
    if refit_rc == 3:
        return EXIT_REFIT_A5_STOP
    return int(refit_rc)


def build_gate_payload(*, cfg: dict, model_key: str, identity: dict, n_blocks: int,
                       hidden_dim: int, rule: dict, split: dict, lists: dict,
                       capture: dict, binding: dict, norms_probe: dict,
                       released_reading: dict, refit_reading: dict | None,
                       refit_status: str, device: str, threshold: int,
                       reading_seconds: float | None = None) -> dict:
    from sisteer.extraction import alphasteer_screen as scr

    recovery = build_recovery_block(
        released_reading=released_reading, refit_reading=refit_reading,
        paper_layers=recovery_paper_layers(), paper_ratios=recovery_paper_ratios(),
        norms_probe=norms_probe, threshold=threshold)
    per = released_reading["per_layer"]
    curve = []
    for l in range(n_blocks):
        row = per[str(l)]
        curve.append({"layer": l, "depth": l / float(n_blocks),
                      "in_band": bool(row["in_band"]),
                      "auroc": float(row["auroc"]),
                      "leakage": row["leakage"],
                      "leakage_ratio_used": row["leakage_ratio_used"],
                      "operator_benign_energy": float(row["operator_benign_energy"]),
                      "released_norm": float(row["released_norm"]),
                      "is_paper_layer": l in recovery["paper_layers"],
                      "eligible": bool(row["eligible"])})
    return {
        "schema": GATE_SCHEMA, "model_key": model_key, "model": cfg["model"],
        "model_revision": identity["model_revision"],
        "model_revision_config": cfg.get("model_revision"),
        "model_note": ("the released matrices were fit on meta-llama/…; the "
                       "NousResearch mirror is weight-identical (models.py: "
                       "_enforce_llama31_mirror governs the choice; plan D.3)"),
        "n_blocks": int(n_blocks), "hidden_dim": int(hidden_dim),
        "site": rule["site"], "device": str(device),
        "rule": {**scr.rule_block(depth_band=tuple(rule["depth_band"]),
                                  p_grid=rule["p_grid"],
                                  auroc_threshold=rule["auroc_threshold"],
                                  lambda_reg=rule["lambda_reg"]),
                 "recovery_threshold": int(threshold),
                 "recovery_rule": "eligible iff AUROC >= auroc_threshold AND in "
                                  "depth_band AND layer != 0 — AUROC ONLY (leakage is "
                                  "a reported diagnostic, not an eligibility input; "
                                  "D38); recovery = |eligible ∩ paper_layers|; passed "
                                  "iff recovery >= recovery_threshold, on the RELEASED "
                                  "matrices"},
        "split": split, "lists": lists,
        "released_matrix": {k: norms_probe[k] for k in
                            ("path", "released_file_sha256", "shape", "dtype")},
        "score": released_reading["score"],
        "curve": curve,
        "recovery": recovery,
        "passed": recovery["passed"],
        "gate_reading": recovery["gate_reading"],
        "n_recovered": recovery["n_recovered"],
        "threshold": recovery["threshold"],
        "recovered": released_reading["recovered"],
        "refit_status": refit_status,
        "reading_seconds": reading_seconds,
        "capture": capture,
        "inputs_binding": scr._canonical(binding),
        "inputs_sha256": scr.binding_sha256(binding),
    }


def recovery_paper_layers() -> list[int]:
    from sisteer.extraction.alphasteer import PAPER_LLAMA31_LAYERS
    return [int(l) for l in PAPER_LLAMA31_LAYERS]


def recovery_paper_ratios() -> dict:
    from sisteer.extraction.alphasteer import PAPER_LLAMA31_NULLSPACE_RATIOS
    return {int(k): float(v) for k, v in PAPER_LLAMA31_NULLSPACE_RATIOS.items()}


# --------------------------------------------------------------------------- #
# figure 8: the released curve with the paper set and the recovered           #
# set overlaid                                                                  #
# --------------------------------------------------------------------------- #

# dataviz reference palette, fixed slot order (never cycled)
_BLUE, _ORANGE, _GREY, _VIOLET, _INK = "#2a78d6", "#eb6834", "#9a9a96", "#4a3aa7", "#52514e"


def recovery_figure(reading: dict, *, n_blocks: int, paper_layers, rule: dict,
                    out_png: Path, model_key: str, passed: bool) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001 — a figure must never fail the gate
        _log(f"matplotlib unavailable ({type(e).__name__}); no figure")
        return None
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    per = reading["per_layer"]
    paper = set(int(l) for l in paper_layers)
    recovered = set(int(l) for l in reading["recovered"])
    nonzero = set(int(l) for l in reading["released_nonzero_layers"])
    layers = list(range(n_blocks))
    depth = [l / float(n_blocks) for l in layers]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    lo, hi = rule["depth_band"]
    for ax in (ax1, ax2):
        ax.axvspan(lo, hi, color="0.92", lw=0, zorder=0)
        ax.grid(True, axis="y", color="0.85", lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # --- AUROC panel
    ax1.axhline(rule["auroc_threshold"], color=_INK, ls="--", lw=0.9, zorder=1)
    ax1.text(0.005, rule["auroc_threshold"] + 0.008, f"AUROC {rule['auroc_threshold']}",
             fontsize=8, color=_INK, va="bottom")
    ax1.plot(depth, [per[str(l)]["auroc"] for l in layers], color="0.75", lw=1.2, zorder=1)
    groups = (
        ([l for l in layers if l not in nonzero], dict(mfc="white", mec=_GREY, color=_GREY),
         "released slice is 0.0 (AUROC 0.5 by construction)"),
        ([l for l in layers if l in nonzero and l not in paper], dict(mfc=_VIOLET, mec=_VIOLET, color=_VIOLET),
         "non-zero slice outside the paper set"),
        ([l for l in layers if l in paper and l not in recovered], dict(mfc=_ORANGE, mec=_ORANGE, color=_ORANGE),
         "paper layer NOT recovered"),
        ([l for l in layers if l in recovered], dict(mfc=_BLUE, mec=_BLUE, color=_BLUE),
         f"paper layer recovered (AUROC >= {rule['auroc_threshold']}, in band)"),
    )
    for ls, style, label in groups:
        if not ls:
            continue
        ax1.plot([l / float(n_blocks) for l in ls], [per[str(l)]["auroc"] for l in ls],
                 ls="none", marker="o", ms=7, mew=1.6, label=label, zorder=3, **style)
    for l in sorted(paper):
        ax1.annotate(str(l), (l / float(n_blocks), per[str(l)]["auroc"]),
                     textcoords="offset points", xytext=(0, 8), ha="center",
                     fontsize=8, color=_INK)
    ax1.set_ylim(0.45, 1.03)
    ax1.set_ylabel("held-out AUROC of ||h @ M_l||")
    ax1.legend(fontsize=8, loc="lower right", frameon=False)
    # --- leakage panel: a DIAGNOSTIC — paper layers only, the others
    #     have no published ratio; the retired bound is a reference line
    from sisteer.extraction.alphasteer_screen import LEAKAGE_REPORT_REFERENCE

    lk = [(l, per[str(l)]["leakage"]) for l in sorted(paper) if per[str(l)]["leakage"] is not None]
    ob = [(l, per[str(l)]["operator_benign_energy"]) for l in sorted(nonzero)]
    ax2.axhline(LEAKAGE_REPORT_REFERENCE, color=_INK, ls=":", lw=0.9, zorder=1)
    ax2.text(0.005, LEAKAGE_REPORT_REFERENCE * 1.15,
             f"retired {LEAKAGE_REPORT_REFERENCE:.0e} bound — reference only, not a gate (D38)",
             fontsize=8, color=_INK, va="bottom")
    if lk:
        cols = [_BLUE if l in recovered else _ORANGE for l, _ in lk]
        ax2.scatter([l / float(n_blocks) for l, _ in lk], [max(v, 1e-12) for _, v in lk],
                    s=48, c=cols, zorder=3,
                    label="projector leakage at the paper ratio (diagnostic, not thresholded)")
    if ob:
        ax2.scatter([l / float(n_blocks) for l, _ in ob], [max(v, 1e-12) for _, v in ob],
                    s=40, facecolors="none", edgecolors=_GREY, lw=1.4, zorder=2,
                    label="operator ||H_b M_l||²/||H_b||² (diagnostic, not thresholded)")
    ax2.set_yscale("log")
    ax2.set_ylabel("benign energy kept (diagnostic)")
    ax2.set_xlabel(f"depth (layer / {n_blocks} blocks); band {lo:.0%}-{hi:.0%} shaded")
    ax2.legend(fontsize=8, loc="upper right", frameon=False)
    verdict = "PASSED" if passed else "FAILED — STOP GATE"
    fig.suptitle(f"Llama-3.1-8B released AlphaSteer matrices: F.9 rule recovers "
                 f"{len(recovered)}/{len(paper)} paper layers — gate {verdict}",
                 fontsize=11, y=0.995)
    ax1.set_title("scored on the deployed row-vector op; 22 released slices are exactly 0 (D8)",
                  fontsize=9, color=_INK, loc="left")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# the job                                                                       #
# --------------------------------------------------------------------------- #


def run_gate(cfg: dict, rule: dict, *, model_key: str, released_path: Path,
             gate_out: Path, figs_dir: Path | None, resume: bool = False,
             force: bool = False, device=None, batch_size: int | None = None,
             wandb_project: str = "", skip_refit: bool = False,
             threshold: int = DEFAULT_RECOVERY_THRESHOLD,
             model_and_tok=None, data=None) -> int:
    import torch

    from sisteer.extraction import alphasteer_screen as scr
    from sisteer.models import get_decoder_layers, hidden_size

    out = fcb.rule_path(rule, "screen_path")
    out_dir = out.parent
    acts_dir = fcb.rule_path(rule, "acts_dir")
    split_out = fcb.rule_path(rule, "split_path")
    gate_out = Path(gate_out)
    figs_dir = Path(figs_dir) if figs_dir else out_dir / "figs"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(batch_size or rule["capture_batch_size"])
    paper_layers = recovery_paper_layers()
    paper_ratios = recovery_paper_ratios()

    if model_and_tok is None:
        model, tok = fcb.load_class_b_model(cfg)
    else:
        model, tok = model_and_tok
    if data is None:
        data = fcb.build_class_b_data(cfg, tok)
    content_meta = data.get("content_meta", {}) or {}
    list_sha = {"benign": content_meta.get("benign_sha256"),
                "harmful": content_meta.get("harmful_sha256")}
    n_blocks = len(get_decoder_layers(model))
    d = hidden_size(model)
    site = rule["site"]
    code_sha = scr.repo_code_sha()
    identity = scr.model_load_identity(model, name=cfg["model"],
                                       cfg_revision=cfg.get("model_revision"))
    layers = list(range(n_blocks))
    _log(f"model {cfg['model']} rev {identity['model_revision']} ({identity['dtype']}, "
         f"{identity['attn_implementation']}, {identity['device']}): {n_blocks} blocks, "
         f"d={d}; lists {len(data['benign_prompts'])} benign / "
         f"{len(data['harmful_prompts'])} harmful; site={site}; device={device}; "
         f"code {code_sha[:12]}; paper layers {paper_layers}; threshold {threshold}/"
         f"{len(paper_layers)}")

    # the same split and bindings as the screen (run_screen), so the gate's
    # capture IS the refit's capture and a resume matches
    split = fcb.compute_content_grouped_split(cfg, rule, data)
    cap_binding = {
        side: scr.capture_binding(
            **identity, site=site, side=side, list_sha256=list_sha[side],
            n_rows=len(data[f"{side}_prompts"]), batch_size=batch_size,
            code_sha=code_sha)
        for side in ("benign", "harmful")}
    binding = scr.screen_binding(
        capture=cap_binding, split_sha256=split["split_sha256"],
        n_blocks=n_blocks, hidden_dim=d, layers=layers, p_grid=rule["p_grid"],
        lambda_reg=rule["lambda_reg"], depth_band=rule["depth_band"],
        code_sha=code_sha)

    # D.0 — the norm probe, a cheap separate artifact; the file must be the
    # one documented above (nonzero set == the paper set) or the gate means
    # nothing
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R, probe = released_norm_probe(released_path, n_blocks=n_blocks, hidden_dim=d,
                                   paper_layers=paper_layers)
    scr.write_fingerprinted_json(out_dir / "released_norms.json", probe, code_sha=code_sha)
    _log(f"D.0 released file {released_path}: sha {probe['released_file_sha256'][:12]}, "
         f"shape {probe['shape']} {probe['dtype']}; non-zero slices "
         f"{probe['nonzero_layers']} ({probe['n_nonzero']}), zero slices "
         f"{probe['n_zero']}; nonzero == paper set: {probe['nonzero_equals_paper_set']} "
         f"({time.time() - t0:.0f}s)")
    if not probe["nonzero_equals_paper_set"]:
        _log("REFUSING: the released file's non-zero slice set is not the paper's layer "
             f"set (D8 measured them equal) — got {probe['nonzero_layers']}, paper "
             f"{paper_layers}. This is not the file the gate was specified on.")
        return EXIT_RELEASED_FILE_MISMATCH

    # a COMPLETE verdict under the same inputs is re-read, never recomputed;
    # under other inputs it refuses (the screen's discipline)
    payload = None
    if gate_out.is_file() and gate_out.with_name(gate_out.name + ".COMPLETE").is_file() and not force:
        payload = scr.read_fingerprinted_json(gate_out)
        got = payload.get("inputs_binding")
        if got != scr._canonical(binding):
            raise scr.CheckpointMismatch(
                f"{gate_out}: the complete verdict there was made under different "
                f"inputs — {scr.binding_diff(binding, got)}. Refusing to stand on it; "
                "--force recomputes it, or remove it deliberately if obsolete.")
        _log(f"gate verdict already complete at {gate_out} (sha {payload['sha256'][:12]}, "
             f"inputs matched): passed={payload['passed']} "
             f"n_recovered={payload['n_recovered']}/{len(paper_layers)} "
             f"refit_status={payload['refit_status']} -- re-reading, not recomputing")
        released_reading = payload["recovery"][GATE_READING]
        capture_info = payload["capture"]
        lists = payload["lists"]
        split_summary = payload["split"]
    else:
        split_payload = scr.write_fingerprinted_json(split_out, split, code_sha=code_sha)
        _log(f"split: fit {split['n_fit_benign']}/{split['n_fit_harmful']}, held-out "
             f"{split['n_heldout_benign']}/{split['n_heldout_harmful']} (benign/harmful, "
             f"REALISED), content_disjoint={split['content_disjoint_verified']} -> {split_out}")
        # capture every block once per side through the ONE capture seam
        # (reused on a requeue iff the sentinel binding matches)
        acts, capture_info = {}, {"add_special_tokens": scr.CAPTURE_ADD_SPECIAL_TOKENS,
                                  "ids_match_serving_render": True}
        for side in ("benign", "harmful"):
            acts[side], capture_info[side] = scr.capture_or_load_side(
                model, tok, data[f"{side}_prompts"], side=side, acts_dir=acts_dir,
                binding=cap_binding[side], verify_sample=int(rule["verify_sample"]),
                layers=layers, log_fn=_log)
        sides = split["sides"]
        bi = torch.tensor(sides["benign"]["fit_idx"], dtype=torch.long)
        bh = torch.tensor(sides["benign"]["heldout_idx"], dtype=torch.long)
        hh = torch.tensor(sides["harmful"]["heldout_idx"], dtype=torch.long)

        # the gate reading, on the screen's device: the released slices, the
        # held-out rows of every block, the benign FIT rows of the paper
        # layers (the projector's Gram)
        t0 = time.time()
        released_dev = {l: R[l].to(device) for l in layers}
        benign_held = {l: acts["benign"][l][bh].to(device) for l in layers}
        harmful_held = {l: acts["harmful"][l][hh].to(device) for l in layers}
        benign_fit = {l: acts["benign"][l][bi].to(device) for l in paper_ratios}
        del acts
        raw = scr.released_matrix_reading(
            benign_held=benign_held, harmful_held=harmful_held, released=released_dev,
            paper_layers=paper_layers, benign_fit=benign_fit, paper_ratios=paper_ratios,
            auroc_threshold=rule["auroc_threshold"], recovery_threshold=threshold)
        del released_dev, benign_held, harmful_held, benign_fit
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        released_reading = annotate_band(raw, n_blocks=n_blocks,
                                         depth_band=rule["depth_band"],
                                         paper_layers=paper_layers)
        reading_seconds = time.time() - t0
        for l in layers:
            r = released_reading["per_layer"][str(l)]
            leak = "null" if r["leakage"] is None else f"{r['leakage']:.3e}"
            _log(f"  layer {l:2d} depth {l / n_blocks:.3f} in_band={str(r['in_band']):5s} "
                 f"norm {r['released_norm']:7.3f} auroc {r['auroc']:.4f} leakage {leak:>9s} "
                 f"op_benign {r['operator_benign_energy']:.2e} "
                 f"{'PAPER' if l in paper_layers else '     '} "
                 f"{'ELIGIBLE' if r['eligible'] else ''}")
        cb = fcb._class_b_cfg(cfg)
        lists = {"benign_path": cb.get("benign_prompts_path"),
                 "benign_sha256": list_sha["benign"],
                 "benign_n": len(data["benign_prompts"]),
                 "harmful_path": cb.get("harmful_prompts_path"),
                 "harmful_sha256": list_sha["harmful"],
                 "harmful_n": len(data["harmful_prompts"]),
                 "slices_source": split.get("slices_source")}
        split_summary = {**scr.split_summary(split_payload), "split_path": str(split_out)}
        verdict0 = gate_verdict(released_reading, threshold=threshold)
        payload = build_gate_payload(
            cfg=cfg, model_key=model_key, identity=identity, n_blocks=n_blocks,
            hidden_dim=d, rule=rule, split=split_summary, lists=lists,
            capture=capture_info, binding=binding, norms_probe=probe,
            released_reading=released_reading, refit_reading=None,
            refit_status=("pending" if verdict0["passed"] else "not_run_gate_failed"),
            device=device, threshold=threshold, reading_seconds=reading_seconds)
        final = scr.write_fingerprinted_json(gate_out, payload, code_sha=code_sha)
        _log(f"wrote {gate_out} (sha {final['sha256'][:12]}) in {reading_seconds:.0f}s")
        payload = final

    passed = bool(payload["passed"])
    n_rec = int(payload["n_recovered"])
    _log(f"GATE ({GATE_READING}, AUROC-ONLY rule on the RELEASED matrices; leakage "
         "reported per layer, not gated — D38): recovered "
         f"{released_reading['recovered']} = {n_rec}/{len(paper_layers)} (threshold "
         f"{threshold}) -> {'PASSED' if passed else 'FAILED — STOP GATE: ask the user'}; "
         f"not recovered: {released_reading.get('paper_layers_not_recovered')}")
    png = recovery_figure(released_reading, n_blocks=n_blocks, paper_layers=paper_layers,
                          rule=rule, out_png=figs_dir / "final_llama_recovery.png",
                          model_key=model_key, passed=passed)
    if png is not None:
        _log(f"figure -> {png}")
    _wandb_log(wandb_project, name=f"llama-gate-{model_key}", payload=payload, png=png)
    if not passed:
        return exit_code_for(passed=False, refit_rc=None)
    if skip_refit:
        _log("--skip-refit: gate passed; not running the 32-layer refit reading")
        return 0

    # D.2 — the 32-layer refit BESIDE the gate (only because the gate passed).
    # Same model, same data, same bindings -> the capture is reused from the
    # sentinel, never redone; run_screen owns the (layer, p) checkpoint.
    holder: dict = {}

    def recovery_fn(rows, selection):
        refit = refit_reading_from_selection(selection["per_layer"], paper_layers=paper_layers)
        holder["refit"] = refit
        return build_recovery_block(
            released_reading=released_reading, refit_reading=refit,
            paper_layers=paper_layers, paper_ratios=paper_ratios, norms_probe=probe,
            threshold=threshold)

    _log("gate passed -> running the 32-layer refit reading (rejection side; cannot set passed)")
    rc = screen_cli.run_screen(
        cfg, rule, model_key=model_key, out=out, acts_dir=acts_dir, split_out=split_out,
        layers_mode="all", resume=resume, force=force, device=device,
        batch_size=batch_size, wandb_project=wandb_project, model_and_tok=(model, tok),
        data=data, recovery_fn=recovery_fn)
    if rc == 3:
        _log("refit hit spec A5's stop (screen_f9.A5_STOP.json written); the gate verdict "
             "STANDS — recording refit_status=a5_stop")
        refit, status = None, "a5_stop"
    elif rc != 0:
        _log(f"refit exited rc={rc}; the gate verdict stands, refit_status stays "
             f"{payload['refit_status']!r}")
        return exit_code_for(passed=True, refit_rc=rc)
    else:
        refit = holder.get("refit")
        if refit is None:   # a complete screen found on resume: read, don't recompute
            screen = scr.read_fingerprinted_json(out)
            refit = (screen.get("recovery") or {}).get(REFIT_READING) or \
                refit_reading_from_selection(screen["per_layer"], paper_layers=paper_layers)
        status = "complete"
        _log(f"REFIT ({REFIT_READING}): eligible in band {refit['eligible_in_band']}; "
             f"recovered {refit['recovered']} = {refit['n_recovered']}/{len(paper_layers)}; "
             f"false positives outside the paper set "
             f"{refit['false_positives_outside_paper_set']}; declined paper layers "
             f"{refit['paper_layers_not_recovered']}")
        src = out_dir / "figs" / "auroc_vs_depth.png"
        if src.is_file():
            figs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, figs_dir / f"final_auroc_vs_depth_by_p_{model_key}.png")
    # re-write the verdict with the refit beside it; `passed` is re-derived from
    # the released reading and must not have moved
    body = {k: v for k, v in payload.items()
            if k not in ("sha256", "code_sha", "written_at")}
    body["recovery"] = build_recovery_block(
        released_reading=released_reading, refit_reading=refit, paper_layers=paper_layers,
        paper_ratios=paper_ratios, norms_probe=probe, threshold=threshold)
    body["refit_status"] = status
    body["verdict_first_sha256"] = payload["sha256"]
    body["verdict_first_written_at"] = payload["written_at"]
    if bool(body["recovery"]["passed"]) != passed:
        raise AssertionError("the verdict moved on re-write — impossible by construction")
    body["passed"] = body["recovery"]["passed"]
    final = scr.write_fingerprinted_json(gate_out, body, code_sha=code_sha)
    _log(f"re-wrote {gate_out} with the refit beside the gate (sha {final['sha256'][:12]}; "
         f"passed={final['passed']} unchanged)")
    return exit_code_for(passed=True, refit_rc=rc)


def _wandb_log(project: str, *, name: str, payload: dict, png) -> None:
    if not project:
        return
    try:
        import wandb

        run = wandb.init(project=project, name=name, reinit=True,
                         config={"rule": payload["rule"], "model": payload["model"],
                                 "model_key": payload["model_key"]})
        for row in payload["curve"]:
            run.log({"gate/auroc_released": row["auroc"],
                     "gate/operator_benign_energy": row["operator_benign_energy"],
                     **({"gate/leakage_reconstructed": row["leakage"]}
                        if row["leakage"] is not None else {}),
                     "layer": row["layer"], "depth": row["depth"]})
        run.summary["passed"] = payload["passed"]
        run.summary["n_recovered"] = payload["n_recovered"]
        run.summary["recovered"] = payload["recovered"]
        if png is not None:
            run.log({"gate/final_llama_recovery": wandb.Image(str(png))})
        run.finish()
    except Exception as e:  # noqa: BLE001 — logging must not fail the gate
        _log(f"W&B logging failed ({type(e).__name__}: {e}); artifact is on disk")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--released", default="", help="default: llama_gate.released_matrix_path")
    ap.add_argument("--gate-out", default="", help="default: llama_gate.gate_out")
    ap.add_argument("--figs-dir", default="", help="default: <screen dir>/figs")
    ap.add_argument("--model-key", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--skip-refit", action="store_true")
    args = ap.parse_args(argv)
    cfg = fcb.load_cfg(args.config)
    rule = fcb.alphasteer_layer_rule(cfg)
    if rule is None:
        raise SystemExit("[alphasteer_llama_gate] the config has no class_b.alphasteer "
                         "block; the gate needs the F.9 rule it validates")
    lg = cfg.get("llama_gate") or {}
    released = args.released or lg.get("released_matrix_path")
    gate_out = args.gate_out or lg.get("gate_out")
    if not released or not gate_out:
        raise SystemExit("[alphasteer_llama_gate] need llama_gate.released_matrix_path and "
                         "llama_gate.gate_out in the config (or --released / --gate-out)")
    released = fcb.resolve_cluster_path(str(released))
    from sisteer import site as _site

    gate_out = _site.expand_path(gate_out, key="llama_gate.gate_out")   # a placeholder never becomes a literal output dir
    return run_gate(
        cfg, rule, model_key=args.model_key or fcb.alphasteer_model_key(cfg, rule),
        released_path=Path(released), gate_out=gate_out,
        figs_dir=Path(args.figs_dir) if args.figs_dir else None,
        resume=args.resume, force=args.force, device=args.device,
        batch_size=args.batch_size, wandb_project=args.wandb_project,
        skip_refit=args.skip_refit,
        threshold=int(lg.get("recovery_threshold", DEFAULT_RECOVERY_THRESHOLD)))


if __name__ == "__main__":
    sys.exit(main())
