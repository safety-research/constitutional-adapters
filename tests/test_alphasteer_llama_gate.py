"""Phase D — the Llama layer-rule validation (scripts/alphasteer_llama_gate.py).

The load-bearing test is ``test_refit_reading_cannot_set_passed``: the
32-layer refit is reported BESIDE the gate and the verdict is a function of
the released reading alone — a refit that recovers 10/10 does not
rescue a released reading that recovers 3/10, and a refit that recovers 0/10
does not sink a released reading that passes. Synthetic readings throughout;
the one model-backed test runs ``run_screen`` on the tiny fixture with a
``recovery_fn`` and checks the block lands in ``screen_f9.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import torch
import yaml

from sisteer.extraction import alphasteer_screen as scr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import alphasteer_llama_gate as gate  # noqa: E402

PAPER = [8, 9, 10, 11, 12, 13, 14, 16, 18, 19]
RATIOS = {8: 0.6, 9: 0.6, 10: 0.6, 11: 0.6, 12: 0.4, 13: 0.5, 14: 0.6, 16: 0.6, 18: 0.6, 19: 0.6}


def _released_reading(recovered, *, n_blocks=32, paper=PAPER, threshold=7):
    """A synthetic library-shaped released reading (already band-annotated)."""
    per = {}
    for l in range(n_blocks):
        is_paper = l in paper
        rec = l in recovered
        per[str(l)] = {"auroc": 0.99 if rec else (0.80 if is_paper else 0.5),
                       "leakage": (2e-4 if is_paper else None),
                       "leakage_ratio_used": (RATIOS.get(l) if is_paper else None),
                       "operator_benign_energy": 1e-5 if is_paper else 0.0,
                       "released_norm": 10.0 if is_paper else 0.0,
                       "eligible": rec, "in_band": True, "depth": l / n_blocks}
    return {"rule": "AUROC >= 0.95, never layer 0 — AUROC ONLY; leakage reported as a diagnostic (D38)",
            "auroc_threshold": 0.95, "eligibility": scr.ELIGIBILITY_RULE,
            "score": {"expression": scr.SCORE_EXPRESSION, "orientation": scr.SCORE_ORIENTATION},
            "released_matrix_norms": {str(l): per[str(l)]["released_norm"] for l in range(n_blocks)},
            "released_nonzero_layers": list(paper), "released_all_layers_nonzero": False,
            "recovered": sorted(recovered), "n_recovered": len(recovered),
            "threshold": threshold, "passed": len(recovered) >= threshold,
            "per_layer": per, "tests": "ACCEPTANCE only"}


def _refit_reading(recovered, fps=()):
    per = [{"layer": l, "depth": l / 32, "in_band": 5 <= l <= 20, "p_star": 0.5,
            "auroc": 0.99 if (l in recovered or l in fps) else 0.6, "leakage": 1e-4,
            "eligible": (l in recovered or l in fps), "reason": None}
           for l in range(32)]
    return gate.refit_reading_from_selection(per, paper_layers=PAPER)


def _probe(nonzero=PAPER, n_blocks=32):
    return {"per_slice_frobenius": {str(l): (10.0 if l in nonzero else 0.0) for l in range(n_blocks)},
            "nonzero_layers": list(nonzero), "all_layers_nonzero": len(nonzero) == n_blocks,
            "nonzero_equals_paper_set": set(nonzero) == set(PAPER),
            "path": "x.pt", "released_file_sha256": "0" * 64, "shape": [n_blocks, 4, 4],
            "dtype": "torch.float32"}


def _cap_binding(side):
    """A REAL capture_binding (screen_binding refuses anything else): every
    keyword-only parameter filled with a dummy of the right kind."""
    import inspect
    kw = {}
    for name in inspect.signature(scr.capture_binding).parameters:
        kw[name] = {"side": side, "n_rows": 10, "batch_size": 8}.get(name, f"{name}-x")
    return scr.capture_binding(**kw)

# ------------------------------------------------------------- the ONE verdict


def test_refit_reading_cannot_set_passed():
    """A 10/10 refit does not rescue a 3/10 released reading; a 0/10 refit
    does not sink an 8/10 released reading. gate_reading is pinned."""
    failing = _released_reading([8, 9, 10])
    rescuing = _refit_reading(PAPER)
    assert rescuing["n_recovered"] == 10
    block = gate.build_recovery_block(released_reading=failing, refit_reading=rescuing,
                                      paper_layers=PAPER, paper_ratios=RATIOS, norms_probe=_probe())
    assert block["passed"] is False
    assert block["n_recovered"] == 3
    assert block["gate_reading"] == "reading_released"
    assert block["reading_refit_32"]["n_recovered"] == 10
    assert block["reading_refit_32"]["can_set_passed"] is False

    passing = _released_reading([8, 9, 10, 11, 12, 13, 14, 16])
    sinking = _refit_reading([])
    block = gate.build_recovery_block(released_reading=passing, refit_reading=sinking,
                                      paper_layers=PAPER, paper_ratios=RATIOS, norms_probe=_probe())
    assert block["passed"] is True and block["n_recovered"] == 8
    assert block["reading_refit_32"]["n_recovered"] == 0
    # and the verdict function does not even accept the refit
    import inspect
    assert "refit" not in " ".join(inspect.signature(gate.gate_verdict).parameters)


def test_gate_verdict_threshold_is_seven_of_ten_inclusive():
    assert gate.gate_verdict(_released_reading(PAPER[:7]))["passed"] is True
    assert gate.gate_verdict(_released_reading(PAPER[:6]))["passed"] is False
    assert gate.gate_verdict(_released_reading(PAPER[:6]), threshold=6)["passed"] is True
    assert gate.DEFAULT_RECOVERY_THRESHOLD == 7


def test_exit_codes_distinguish_gate_failure_from_refit_stop():
    assert gate.exit_code_for(passed=False, refit_rc=None) == 3
    assert gate.exit_code_for(passed=False, refit_rc=0) == 3       # never masked by a refit
    assert gate.exit_code_for(passed=True, refit_rc=None) == 0
    assert gate.exit_code_for(passed=True, refit_rc=0) == 0
    assert gate.exit_code_for(passed=True, refit_rc=3) == 4        # A5 stop != gate failure
    assert gate.exit_code_for(passed=True, refit_rc=1) == 1


# ------------------------------------------------------------- D.0 and the band


def test_released_norm_probe_records_zero_slices_and_the_paper_set(tmp_path):
    n, d = 8, 6
    R = torch.zeros(n, d, d)
    for l in (2, 3, 4):
        R[l] = torch.randn(d, d)
    torch.save(R, tmp_path / "rel.pt")
    T, probe = gate.released_norm_probe(tmp_path / "rel.pt", n_blocks=n, hidden_dim=d,
                                        paper_layers=[2, 3, 4], file_sha256=False)
    assert tuple(T.shape) == (n, d, d)
    assert probe["nonzero_layers"] == [2, 3, 4] and probe["zero_layers"] == [0, 1, 5, 6, 7]
    assert probe["n_zero"] == 5 and probe["all_layers_nonzero"] is False
    assert probe["nonzero_equals_paper_set"] is True
    assert all(probe["per_slice_frobenius"][str(l)] == 0.0 for l in (0, 1, 5, 6, 7))
    assert "D8" in probe["note"] and "not a bug" in probe["note"]
    _, probe2 = gate.released_norm_probe(tmp_path / "rel.pt", n_blocks=n, hidden_dim=d,
                                         paper_layers=[2, 3, 5], file_sha256=False)
    assert probe2["nonzero_equals_paper_set"] is False
    with pytest.raises(ValueError, match="shape"):
        gate.released_norm_probe(tmp_path / "rel.pt", n_blocks=n + 1, hidden_dim=d,
                                 paper_layers=[2], file_sha256=False)


def test_annotate_band_applies_the_depth_half_of_the_rule():
    """A library-eligible paper layer OUTSIDE the band is not recovered; on
    32 blocks every paper layer is in band so the two readings agree."""
    raw = _released_reading([8, 9, 10, 11, 12, 13, 14, 16, 18, 19])
    for v in raw["per_layer"].values():
        v.pop("in_band"); v.pop("depth")
    ann = gate.annotate_band(raw, n_blocks=32, depth_band=(0.15, 0.65), paper_layers=PAPER)
    assert ann["recovered"] == PAPER and ann["recovered_any_depth"] == PAPER
    assert ann["passed"] is True
    assert all(ann["per_layer"][str(l)]["in_band"] for l in PAPER)
    assert ann["per_layer"]["0"]["in_band"] is False and ann["per_layer"]["31"]["in_band"] is False
    # a narrower band excludes layer 19 (0.594) and 18 (0.5625): recovery drops
    narrow = gate.annotate_band(raw, n_blocks=32, depth_band=(0.15, 0.55), paper_layers=PAPER)
    assert narrow["recovered"] == [8, 9, 10, 11, 12, 13, 14, 16]
    assert narrow["recovered_any_depth"] == PAPER
    assert narrow["per_layer"]["19"]["eligible"] is False
    assert narrow["per_layer"]["19"]["eligible_any_depth"] is True
    assert narrow["paper_layers_not_recovered"] == [18, 19]
    # the annotated rule string is the AUROC-only one; no stale full-rule wording
    assert "AUROC >= 0.95" in ann["rule"] and "15%-65%" in ann["rule"]
    assert "AUROC ONLY" in ann["rule"] and "diagnostic" in ann["rule"]
    assert "AND leakage" not in ann["rule"] and "FULL rule" not in ann["rule"]


def test_refit_reading_splits_recovered_false_positives_and_declined():
    # 20/32 = 0.625 is the last in-band layer; 21 (0.656), 23 and 3 are out of band
    r = _refit_reading([8, 9, 10, 11, 12, 14, 16, 18], fps=[20, 21, 23, 3])
    assert r["recovered"] == [8, 9, 10, 11, 12, 14, 16, 18] and r["n_recovered"] == 8
    assert r["false_positives_outside_paper_set"] == [20]
    assert scr.band_layers(32) == list(range(5, 21))
    assert r["paper_layers_not_recovered"] == [13, 19]
    assert r["can_set_passed"] is False and "REJECTION" in r["tests"]


def test_gate_payload_carries_both_readings_the_curve_and_the_binding():
    released = _released_reading([8, 9, 10, 11, 12, 13, 14, 16])
    binding = scr.screen_binding(
        capture={side: _cap_binding(side) for side in ("benign", "harmful")}, split_sha256="s" * 64,
        n_blocks=32, hidden_dim=4, layers=list(range(32)), p_grid=(0.3, 0.4, 0.5, 0.6),
        lambda_reg=10.0, depth_band=(0.15, 0.65), code_sha="c" * 40)
    rule = {"site": "pre", "depth_band": [0.15, 0.65], "p_grid": [0.3, 0.4, 0.5, 0.6],
            "auroc_threshold": 0.95, "lambda_reg": 10.0}
    p = gate.build_gate_payload(
        cfg={"model": "m", "model_revision": "r"}, model_key="llama8b",
        identity={"model_revision": "r"}, n_blocks=32, hidden_dim=4, rule=rule,
        split={"n_fit_benign": 1}, lists={"benign_n": 1}, capture={}, binding=binding,
        norms_probe=_probe(), released_reading=released,
        refit_reading=_refit_reading([8, 9]), refit_status="complete", device="cpu",
        threshold=7)
    assert p["schema"] == gate.GATE_SCHEMA and p["passed"] is True and p["n_recovered"] == 8
    assert p["gate_reading"] == "reading_released"
    assert len(p["curve"]) == 32 and p["curve"][8]["is_paper_layer"] and not p["curve"][0]["is_paper_layer"]
    assert p["curve"][0]["auroc"] == 0.5 and p["curve"][0]["leakage"] is None
    assert p["recovery"]["reading_refit_32"]["n_recovered"] == 2
    assert p["recovery"]["released_nonzero_layers"] == PAPER
    assert p["inputs_sha256"] == scr.binding_sha256(binding)
    assert "D8" in p["recovery"]["d8_note"]
    assert len(p["recovery"]["qualifications"]) == 3
    # the recorded rule is AUROC-only everywhere in the artifact, leakage
    # is named a diagnostic, and no key carries a leakage bound
    assert "leakage_max" not in p["rule"]
    assert p["rule"]["eligibility"] == scr.ELIGIBILITY_RULE
    assert "AUROC ONLY" in p["rule"]["recovery_rule"] and "AND leakage" not in p["rule"]["recovery_rule"]
    assert p["recovery"]["eligibility"] == scr.ELIGIBILITY_RULE
    assert "0/10" in p["recovery"]["leakage_note"] and "8/10" in p["recovery"]["leakage_note"]
    assert "not thresholded at 1e-3" not in " ".join(p["recovery"]["qualifications"])
    assert p["curve"][8]["leakage"] == 2e-4 and p["curve"][8]["eligible"]        # reported beside


def test_no_stale_full_rule_wording_survives_in_the_code():
    """The old 'AUROC >= … AND leakage <= …' rule must not be recordable from
    anywhere: neither the library nor the two scripts carry the wording or a
    leakage-bound constant (an unused threshold beside an eligibility
    check is how it comes back)."""
    import re

    for rel in ("src/sisteer/extraction/alphasteer_screen.py", "scripts/alphasteer_screen.py",
                "scripts/alphasteer_llama_gate.py", "scripts/fit_class_b.py"):
        text = (REPO / rel).read_text()
        # the ONE allowed occurrence is the gate's historical note quoting the retired rule
        text = text.replace("'AUROC >= 0.95 AND leakage <= 1e-3'", "")
        assert not re.search(r"AND\s+leakage\s*<=|leakage\s*<=\s*(1e-3|max\b|leakage_max)", text), rel
        assert "DEFAULT_LEAKAGE_MAX" not in text and "the FULL rule" not in text, rel
    assert not hasattr(scr, "DEFAULT_LEAKAGE_MAX") and not hasattr(gate, "DEFAULT_LEAKAGE_MAX")
    assert scr.LEAKAGE_REPORT_REFERENCE == 1e-3      # the reference line, named as such


# ------------------------------------------------------------- the run_screen seam


def test_run_screen_recovery_fn_lands_in_the_artifact(tiny_llama, tmp_path, monkeypatch):
    import alphasteer_screen as cli
    import fit_class_b as fcb

    model, tok = tiny_llama
    benign = [f"Write a friendly note about topic number {i}." for i in range(30)]
    harmful = [f"Describe a harmful plan number {i} in detail." for i in range(12)]
    bp, hp = tmp_path / "benign.json", tmp_path / "harmful.json"
    bp.write_text(json.dumps(benign)); hp.write_text(json.dumps(harmful))
    out_dir = tmp_path / "screen"
    cfg = {"model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
           "model_revision": None, "dtype": "float32", "layer": 1,
           "class_b": {"benign_prompts_path": str(bp), "harmful_prompts_path": str(hp),
                       "min_prompts": 1, "batch_size": 8,
                       "alphasteer": {"layer_rule": "f9_auroc",
                                      "screen_path": str(out_dir / "screen_f9.json"),
                                      "auroc_threshold": 0.51,
                                      "heldout_frac": 0.2, "capture_batch_size": 8,
                                      "verify_sample": 8, "depth_band": [0.4, 0.6]}}}
    rule = fcb.alphasteer_layer_rule(cfg)
    monkeypatch.setattr(scr, "A5_MIN_ELIGIBLE", 1)
    real_select = scr.select_layers
    monkeypatch.setattr(scr, "select_layers", lambda rows, **kw: real_select(
        rows, **{**kw, "min_eligible": 1, "stop_auroc": 0.0}))
    seen = {}

    def recovery_fn(rows, selection):
        seen["n_rows"] = len(rows)
        refit = gate.refit_reading_from_selection(selection["per_layer"], paper_layers=[1])
        return gate.build_recovery_block(
            released_reading=_released_reading([], n_blocks=2, paper=[1], threshold=1),
            refit_reading=refit, paper_layers=[1], paper_ratios={1: 0.5},
            norms_probe=_probe(nonzero=[1], n_blocks=2), threshold=1)

    rc = cli.run_screen(cfg, rule, model_key="tiny", out=out_dir / "screen_f9.json",
                        acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                        device="cpu", model_and_tok=(model, tok), recovery_fn=recovery_fn)
    assert rc == 0 and seen["n_rows"] == 8
    payload = scr.read_fingerprinted_json(out_dir / "screen_f9.json")
    scr.check_screen_payload(payload)
    rec = payload["recovery"]
    assert rec["gate_reading"] == "reading_released" and rec["passed"] is False
    assert rec["reading_refit_32"]["can_set_passed"] is False
    assert set(rec["reading_refit_32"]["per_layer"]) == {"0", "1"}
    # without a recovery_fn the artifact has no block (the screen never invents one)
    rc = cli.run_screen(cfg, rule, model_key="tiny", out=out_dir / "screen_f9.json",
                        acts_dir=out_dir / "acts", split_out=out_dir / "split_v1.json",
                        device="cpu", model_and_tok=(model, tok), resume=True, force=True)
    assert rc == 0 and "recovery" not in scr.read_fingerprinted_json(out_dir / "screen_f9.json")


# ------------------------------------------------------------- the config and the sbatch


def test_gate_config_is_the_shipped_llama_pin_plus_the_rule_block():
    import fit_class_b as fcb

    gate_cfg = yaml.safe_load((REPO / "configs/alphasteer_f22_llama8b_gate.yaml").read_text())
    ship = yaml.safe_load((REPO / "configs/two_si.yaml").read_text())
    for k in ("model", "model_revision", "dtype", "attn_implementation"):
        assert gate_cfg[k] == ship[k], k
    for k in ("benign_prompts_path", "harmful_prompts_path", "benign_sha256", "harmful_sha256"):
        assert gate_cfg["class_b"][k] == ship["class_b"][k], k
    assert "alphasteer" not in ship["class_b"]           # the shipped config stays single-layer
    rule = fcb.alphasteer_layer_rule(gate_cfg)
    assert rule["layer_rule"] == "f9_auroc" and rule["model_key"] == "llama8b"
    assert rule["auroc_threshold"] == 0.95 and "leakage_max" not in rule
    assert rule["depth_band"] == [0.15, 0.65] and rule["p_grid"] == [0.3, 0.4, 0.5, 0.6]
    assert rule["site"] == "pre"
    assert rule["screen_path"].endswith("alphasteer_f22/llama8b/screen_f9.json")
    lg = gate_cfg["llama_gate"]
    assert lg["released_matrix_path"].endswith("steering_matrix_llama3.1.pt")
    assert lg["gate_out"].endswith("alphasteer_f22/llama_gate.json")
    assert lg["recovery_threshold"] == 7
    assert fcb.alphasteer_model_key(gate_cfg, rule) == "llama8b"


def test_sbatch_keeps_the_hard_rules_and_the_phase_d_contract():
    text = (REPO / "scripts/sbatch/alphasteer_llama_gate.sbatch").read_text()
    lines = text.splitlines()
    sbatch = [l for l in lines if l.startswith("#SBATCH")]
    code = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    assert not any("--export" in l for l in sbatch)               # hard rule 1
    assert not any("--export=" in l for l in code)
    assert re.search(r"^#SBATCH --mem=(\S+)", text, re.M).group(1) == "128G"   # hard rule 2
    assert not any("--mem=0" in l for l in code)
    assert not any(re.search(r"(^|\s)(export\s+)?CUDA_VISIBLE_DEVICES=", l) for l in code)  # hard rule 3
    assert "#SBATCH --qos=high" in text and "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --job-name=as22-llama-gate" in text
    assert "#SBATCH --time=0-06:00:00" in text
    assert "#SBATCH --open-mode=append" in text and "#SBATCH --no-requeue" in text
    assert "--partition=general" in text and "--qos=dev" not in text
    assert 'S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-as22"' in text        # own checkout, never the peers'
    assert "stage2_common.sh" in text and 's2_init "$@"' in text
    assert "s2_run uv run --no-sync python scripts/alphasteer_llama_gate.py" in text
    assert "mirror-figs.sh" in text and "alphasteer-full22" in text
    assert "leakage_report.json" in text                        # the mirror is checked
    assert "--resume" in text
    assert "3)" in text and "STOP GATE" in text
