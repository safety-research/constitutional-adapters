"""scripts/sdfbase/psr_residual_distill.py end to end on a SYNTHETIC bank
+ the CPU tiny model: compose-v (identity-gated parents, derived_meta +
embedded provenance, the shared derived-object verdict with parents
re-verified on disk, refusals), capture-set (stratified manifest with
sources), and ``all`` (floor → planted C → real-model identity check →
controls (a)/(b) → production distillation → ED → recovery verdict →
artifacts), including the distilled object's plain ``kind: "psr"``
loadability, the calibration artifact's fields and the banned-word
guard, and the calibration IDENTITY fingerprint (every field — model,
parents' shas, source pools, capture split, seed, trainer and
measurement settings — stamped by calibrate and compared in full by
distill, not lr alone).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from residual_fixtures import random_psr

REPO = Path(__file__).resolve().parents[1]
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prd = _load_module("psr_residual_distill",
                   REPO / "scripts/sdfbase/psr_residual_distill.py")

KEY = "tinybase"
MIX_SI, CTL_SI = "constv2-mix-100k", "constv2ctl-100k"

_MIX_ROWS = [
    ("advice", "Sys scaffold one\n\n", "A short advice doc about being kind."),
    ("stories", "", "A longer story body that runs on for quite a few more "
                    "tokens than the short one does, on purpose."),
    ("advice", "Sys scaffold two\n\n", "Medium length advice body here."),
    ("stories", "", "Tiny tale."),
    ("stories", "", "Another story of middling length overall, with words."),
    ("advice", "Sys scaffold three\n\n", "Third advice text, calm and brief."),
    ("stories", "", "Fourth story line about a lantern in the fog."),
]
_CTL_ROWS = [
    ("stories", "", "Control story one with neutral register text inside."),
    ("stories", "", "Control story two, a little longer than the first, "
                    "with more clauses."),
    ("advice", "Ctl sys\n\n", "Control advice body about weather."),
    ("stories", "", "Control story three."),
    ("stories", "", "Control story four, with an additional sentence here."),
    ("advice", "Ctl sys two\n\n", "Control advice body about gardening."),
]


def _write_pool(pools: Path, si: str, rows, prefix: str) -> None:
    pools.mkdir(parents=True, exist_ok=True)
    with open(pools / f"pairs_{si}.jsonl", "w") as f:
        for i, (fam, prompt, pos) in enumerate(rows):
            f.write(json.dumps({"prompt": prompt, "pos": pos, "neg": "",
                                "instruction_id": si,
                                "doc_id": f"{prefix}-{fam[:3]}-{i:02d}",
                                "family": fam, "official_tokens": 10 + i,
                                "truncated": False}) + "\n")
    (pools / f"pairs_{si}.manifest.json").write_text(json.dumps(
        {"cell": si, "budget": 100_000, "official_tokens": 100_000}))


def _v_dir(bank: Path) -> Path:
    return bank / "out/derived/psr_paired_write" / KEY / \
        f"{MIX_SI}__psr_ce_all_minus_ctl"


def _resid_dir(bank: Path) -> Path:
    return bank / "out/derived/psr_residual" / KEY / \
        f"{MIX_SI}__psr_ce_all_minus_ctl_distilled"


def _stage_cell(root: Path, key: str, si: str, obj, *, model_revision="rev0",
                complete=True, cell_id_override=None) -> Path:
    from sisteer.eval.defense_cell import sha256_file

    cell_id = f"{si}__psr_ce_all"
    d = root / key / "cells_all" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    ident = d.parent / "_store_identity.json"
    if not ident.exists():  # the bank-level identity every writer matched
        ident.write_text(json.dumps({"model": TINY, "model_revision": "rev0",
                                     "positions": "all"}))
    obj.save(d / "object.pt")
    (d / "metrics.jsonl").write_text("{}\n")
    meta = {"cell_id": cell_id_override or cell_id, "si_id": si,
            "arm": "psr_ce_all", "object_sha256": sha256_file(d / "object.pt"),
            "training_budget_kind": "document_tokens", "sequence_max_len": 96,
            "completion_tokens": None, "sample_manifest_sha": "ab" * 32,
            "max_len": 96, "model_id": TINY, "model_revision": model_revision,
            "positions": "all", "positions_at_fit": "all",
            "response_header_len": None,
            "target_layers": list(obj.layers), "template_hash": "0" * 16,
            "heldout_pool_sha": "cd" * 32,
            "fit_record": {"best_epoch": 3, "epochs_run": 5,
                           "best_heldout_ce": 2.2, "heldout_noop_ce": 2.3,
                           "ce_ratio": 0.956}}
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    if complete:
        files = {n: sha256_file(d / n) for n in
                 ("object.pt", "train_budget_meta.json", "metrics.jsonl")}
        (d / "COMPLETE").write_text(json.dumps(
            {"cell_id": cell_id, "si_id": si, "arm": "psr_ce_all",
             "positions": "all", "object_file": "object.pt",
             "object_sha256": files["object.pt"], "files": files,
             "fit_fingerprint": "f" * 64}))
    return d


@pytest.fixture()
def bank(tiny_llama, tmp_path):
    model, _tok = tiny_llama
    root = tmp_path / "out"
    _stage_cell(root, KEY, MIX_SI, random_psr(model,11))
    _stage_cell(root, KEY, CTL_SI, random_psr(model,22))
    pools = tmp_path / "pools"
    _write_pool(pools, MIX_SI, _MIX_ROWS, "mix")
    _write_pool(pools, CTL_SI, _CTL_ROWS, "ctl")
    return tmp_path


def _common(bank):
    return ["--root", str(bank / "out"), "--pools", str(bank / "pools"),
            "--model-key", KEY, "--mix-si", MIX_SI]


# -- compose-v -----------------------------------------------------------------


def test_compose_v_end_to_end(bank, tiny_llama):
    from sisteer.eval.object_identity import derived_object_problems
    from sisteer.extraction.paired_write import PSRPairedWriteObject
    from sisteer.methods import load_object

    model, tok = tiny_llama
    assert prd.main(["compose-v", *_common(bank)]) == 0
    d = _v_dir(bank)
    obj_path = d / "object.pt"
    V = load_object(obj_path, model=model, tok=tok)
    assert isinstance(V, PSRPairedWriteObject)
    meta = json.loads((d / "derived_meta.json").read_text())
    assert meta["kind"] == "psr_paired_write"
    assert meta["application_order"] == "shared_pre_write_state"
    assert meta["scale_matched"] is True
    assert meta["parent_pos"]["cell_id"] == f"{MIX_SI}__psr_ce_all"
    assert meta["parent_neg"]["cell_id"] == f"{CTL_SI}__psr_ce_all"
    for side in ("parent_pos", "parent_neg"):
        p = Path(meta[side]["object_path"])
        assert meta[side]["object_sha256"] == prd.sha256_file(p)
        assert meta[side]["fit_fingerprint"] == "f" * 64
        assert meta[side]["target_layers"] == V.layers
    assert meta["object_sha256"] == prd.sha256_file(obj_path)
    assert not (d / "train_budget_meta.json").exists()
    # the shared derived verdict passes with the parents re-verified on disk
    assert derived_object_problems(obj_path,
                                   derived_id=f"{MIX_SI}__psr_ce_all_minus_ctl",
                                   parents_root=bank / "out") == []
    # embedded provenance == sidecar parents
    blob = torch.load(obj_path, map_location="cpu", weights_only=True)
    assert blob["provenance"]["parent_pos"]["object_sha256"] == \
        meta["parent_pos"]["object_sha256"]
    # parents ride bit-exact
    src = torch.load(bank / f"out/{KEY}/cells_all/{MIX_SI}__psr_ce_all/object.pt",
                     map_location="cpu", weights_only=True)
    for l in src["directions"]:
        assert torch.equal(blob["pos"]["directions"][l], src["directions"][l])
    # tampering with a parent AFTER composition is caught by the verdict
    src_path = bank / f"out/{KEY}/cells_all/{MIX_SI}__psr_ce_all/object.pt"
    src_path.write_bytes(src_path.read_bytes() + b"\0")
    probs = derived_object_problems(obj_path, parents_root=bank / "out")
    assert probs and any("hashes to" in p or "object_sha256" in p for p in probs)


def test_compose_v_refusals(bank, tiny_llama, capsys):
    model, _tok = tiny_llama
    # torn publish: no COMPLETE on the ctl parent
    root2 = bank / "out2"
    _stage_cell(root2, KEY, MIX_SI, random_psr(model,11))
    _stage_cell(root2, KEY, CTL_SI, random_psr(model,22), complete=False)
    rc = prd.main(["compose-v", "--root", str(root2), "--pools",
                   str(bank / "pools"), "--model-key", KEY])
    assert rc == 2 and "COMPLETE" in capsys.readouterr().err
    # wrong-identity sidecar
    root3 = bank / "out3"
    _stage_cell(root3, KEY, MIX_SI, random_psr(model,11),
                cell_id_override="constv2-mix-1m__psr_ce_all")
    _stage_cell(root3, KEY, CTL_SI, random_psr(model,22))
    rc = prd.main(["compose-v", "--root", str(root3), "--pools",
                   str(bank / "pools"), "--model-key", KEY])
    assert rc == 2 and "wrong-identity" in capsys.readouterr().err
    # parents on different model revisions are not a defined object
    root4 = bank / "out4"
    _stage_cell(root4, KEY, MIX_SI, random_psr(model,11))
    _stage_cell(root4, KEY, CTL_SI, random_psr(model,22),
                model_revision="rev1")
    rc = prd.main(["compose-v", "--root", str(root4), "--pools",
                   str(bank / "pools"), "--model-key", KEY])
    assert rc == 2 and "model_revision" in capsys.readouterr().err
    # a LoRA parent is refused (kind)
    root5 = bank / "out5"
    from sisteer.extraction.multilayer import MultiLayerLoRAObject
    lora = MultiLayerLoRAObject(model, rank=4)
    _stage_cell(root5, KEY, MIX_SI, lora)
    _stage_cell(root5, KEY, CTL_SI, random_psr(model,22))
    rc = prd.main(["compose-v", "--root", str(root5), "--pools",
                   str(bank / "pools"), "--model-key", KEY])
    assert rc == 2 and "expected 'psr'" in capsys.readouterr().err


# -- capture set ---------------------------------------------------------------


def test_capture_set_manifest(bank):
    out = bank / "cm.json"
    assert prd.main(["capture-set", *_common(bank), "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert set(m["strata"]) == {"mix/advice", "mix/stories", "ctl/advice",
                                "ctl/stories"}
    assert m["sources"]["mix"]["si"] == MIX_SI
    assert m["sources"]["ctl"]["sha256"] == prd.sha256_file(
        bank / "pools" / f"pairs_{CTL_SI}.jsonl")
    assert m["n_train"] + m["n_holdout"] == len(_MIX_ROWS) + len(_CTL_ROWS)
    assert all(s["n_holdout"] >= 1 for s in m["strata"].values())
    assert "ctl/advice" in m["thin_strata"]


# -- all: calibration + production ---------------------------------------------


def _model_args(bank, **over):
    args = ["--model", TINY, "--device", "cpu", "--dtype", "float32",
            "--lr", "0.02", "--max-grad-norm", "inf", "--warmup-frac", "0.1",
            "--reg-coeff", "0.1", "--accum-steps", "2", "--batch-size", "2",
            "--max-len", "96", "--epochs", "2", "--patience", "2",
            "--n-boot", "20", "--holdout-frac", "0.2", "--lr-mult", "1"]
    for k, v in over.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return args


def test_all_end_to_end_artifacts_and_verdict(bank, tiny_llama, capsys):
    from sisteer.eval.object_identity import derived_object_problems
    from sisteer.extraction.psr import PSRObject
    from sisteer.methods import load_object

    model, tok = tiny_llama
    rc = prd.main(["all", *_common(bank), *_model_args(bank)])
    out = capsys.readouterr()
    assert rc == 0, out.err
    d = _resid_dir(bank)
    for name in ("object.pt", "derived_meta.json", "capture_manifest.json",
                 "report.json", "doc_records.jsonl", "calibration.json",
                 "calibration_controls.json", "calibration_doc_records.jsonl"):
        assert (d / name).exists(), name
    # the distilled object is a PLAIN psr artifact (zero loader changes)
    obj = load_object(d / "object.pt", model=model, tok=tok)
    assert isinstance(obj, PSRObject) and obj.kind == "psr"
    assert obj.train_direction is True
    blob = torch.load(d / "object.pt", map_location="cpu", weights_only=True)
    assert blob["kind"] == "psr"
    assert blob["provenance"]["derivation"] == "psr_residual_mse"
    meta = json.loads((d / "derived_meta.json").read_text())
    assert meta["kind"] == "psr" and meta["derivation"] == "psr_residual_mse"
    assert meta["derived_id"] == f"{MIX_SI}__psr_ce_all_minus_ctl_distilled"
    assert meta["teacher"]["kind"] == "psr_paired_write"
    assert meta["teacher"]["application_order"] == "shared_pre_write_state"
    assert meta["capture_manifest_sha256"] == prd.sha256_file(
        d / "capture_manifest.json")
    assert meta["trainer"]["lr"] == pytest.approx(0.02)
    assert meta["trainer"]["source_weights"] == {"mix": 0.5, "ctl": 0.5}
    assert meta["recovery_verdict"] in ("recovered",
                                        "not recovered under this trainer + budget",
                                        "unresolvable")
    assert derived_object_problems(d / "object.pt",
                                   derived_id=meta["derived_id"]) == []
    # calibration artifact: attainable per stratum, floor, production ED
    # with CIs, verdict + sentences, and NEVER the banned word
    cal = json.loads((d / "calibration.json").read_text())
    assert set(cal["attainable_ed_content"]) == {"mix", "ctl"}
    assert "mix" in cal["floor"] and "content" in cal["floor"]["mix"]
    for s in ("mix", "ctl"):
        row = cal["production_ed"][s]["content"]
        assert {"ed", "lo", "hi", "n_docs"} <= set(row)
        assert cal["recovery"]["per_stratum"][s]["verdict"] in (
            "recovered", "not recovered under this trainer + budget",
            "unresolvable")
    assert cal["recovery"]["gate"] == {
        "attain_frac": 0.7, "abs_floor": 0.5, "unresolvable_floor_mult": 3.0,
        "position_class": "content", "strata": ["mix", "ctl"],
        "ci_level": 0.95}
    assert len(cal["sentences"]) == 2
    assert "arithmetic_residual_inconsistency" in cal
    # the round-2 ANALYSIS contract (analysis_r2/r2_paths.py
    # CALIBRATION_SCHEMA): strata["<source>/<class>"] with ed / ed_ci95 /
    # ed_attain (+ components) / teacher_bare_mse / floor_mse, gate.verdict
    # in the pipeline's tokens, t_arith_rms, n_holdout_docs
    for s in ("mix/content", "ctl/content", "mix/prefix", "ctl/prefix",
              "mix/stories/content", "ctl/stories/content"):
        row = cal["strata"][s]
        assert {"ed", "ed_ci95", "ed_attain", "ed_attain_components",
                "teacher_bare_mse", "floor_mse"} <= set(row), s
        assert set(row["ed_attain_components"]) == {"R_mix_empty", "R_mixC_mix"}
    mc = cal["strata"]["mix/content"]
    assert mc["ed"] == cal["production_ed"]["mix"]["content"]["ed"]
    assert mc["ed_ci95"] == [cal["production_ed"]["mix"]["content"]["lo"],
                             cal["production_ed"]["mix"]["content"]["hi"]]
    assert mc["ed_attain"] == cal["attainable_ed_content"]["mix"]
    assert mc["floor_mse"] == cal["floor"]["mix"]["content"]["mse_sum_layers"]
    assert cal["gate"]["verdict"] in ("recovered", "not_recovered", "unresolvable")
    assert cal["gate"]["verdict"] == {
        "recovered": "recovered",
        "not recovered under this trainer + budget": "not_recovered",
        "unresolvable": "unresolvable"}[cal["recovery"]["verdict"]]
    assert isinstance(cal["t_arith_rms"], float)
    assert cal["n_holdout_docs"]["mix"] == cal["production_ed"]["mix"]["content"]["n_docs"]
    # gate_liveness.json ({layer: live_frac}) beside P_resid AND beside V
    v_dir = _v_dir(bank)
    for gd in (d, v_dir):
        gl = json.loads((gd / "gate_liveness.json").read_text())
        assert set(gl) == {str(l) for l in range(model.config.num_hidden_layers)}
        assert all(0.0 <= v <= 1.0 for v in gl.values())
        assert (gd / "gate_liveness_detail.json").exists()
    assert json.loads((v_dir / "gate_liveness_detail.json").read_text())["kind"] \
        == "psr_paired_write"
    for name in ("calibration.json", "derived_meta.json", "report.json",
                 "calibration_controls.json"):
        assert "inexpressible" not in (d / name).read_text().lower()
    controls = json.loads((d / "calibration_controls.json").read_text())
    assert controls["teacher_identity_check"]["passed"] is True
    assert set(controls["controls"]) == {"R_mix_empty", "R_mixC_mix"}
    assert controls["planted_control"]["active_frac_target"] == 0.3
    # the FULL per-layer identity record is a file of its own
    # (calibration stage), named from the controls artifact, and the
    # production precheck wrote its own beside it
    layers = list(range(model.config.num_hidden_layers))
    for name, stage in ((prd.IDENTITY_REPORT_NAME, "calibration"),
                        (prd.PRECHECK_REPORT_NAME, "production_precheck")):
        rep_i = json.loads((d / name).read_text())
        assert rep_i["kind"] == "psr_residual_identity_report"
        assert rep_i["stage"] == stage and rep_i["passed"] is True
        assert rep_i["problems"] == []
        assert rep_i["identity"] == controls["identity"]
        assert set(rep_i["per_layer"]) == {str(l) for l in layers}
        for l in layers:
            row = rep_i["per_layer"][str(l)]
            assert {"rms_write_discrepancy", "rms_reference_write",
                    "rms_cancellation_scale", "tolerance", "pass",
                    "trajectory_rms_divergence",
                    "trajectory_rms_reference_write", "stream_rms_per_elem",
                    "bf16_perturbation_floor",
                    "padded_vs_solo_floor_rms"} <= set(row), (name, l)
            assert row["pass"] is True
            assert row["rms_write_discrepancy"] <= row["tolerance"]
        assert "the gate" in rep_i["rule"]
    assert controls["teacher_identity_check"]["report"] == \
        str(d / prd.IDENTITY_REPORT_NAME)
    assert set(controls["teacher_identity_check"]["bf16_perturbation_floor"]) \
        == {str(l) for l in layers}
    assert f"identity report -> {d / prd.IDENTITY_REPORT_NAME}" in out.out
    # the calibration IDENTITY fingerprint — stamped on
    # calibration_controls.json, carried into calibration.json and the
    # derived sidecar, and covering every field the gate depends on
    ident = controls["identity"]
    cm = json.loads((d / "capture_manifest.json").read_text())
    assert ident["kind"] == prd.IDENTITY_KIND
    assert set(ident) == {"kind", "model", "parents", "source_pools",
                          "capture", "layers", "trainer", "measurement",
                          "fingerprint"}
    assert len(ident["fingerprint"]) == 64
    assert ident["model"]["id"] == TINY and ident["model"]["dtype"] == "torch.float32"
    assert ident["parents"]["pos"]["cell_id"] == f"{MIX_SI}__psr_ce_all"
    assert ident["parents"]["neg"]["object_sha256"] == prd.sha256_file(
        bank / "out" / KEY / "cells_all" / f"{CTL_SI}__psr_ce_all" / "object.pt")
    assert ident["parents"]["pos"]["model_revision"] == "rev0"
    assert ident["source_pools"]["ctl"]["sha256"] == prd.sha256_file(
        bank / "pools" / f"pairs_{CTL_SI}.jsonl")
    assert ident["capture"] == {
        "holdout_frac": 0.2, "seed": 0,
        "manifest_sha256": prd.sha256_file(d / "capture_manifest.json"),
        "n_train": cm["n_train"], "n_holdout": cm["n_holdout"]}
    assert ident["layers"] == list(range(model.config.num_hidden_layers))
    assert ident["trainer"] == controls["trainer"]
    assert ident["measurement"] == {"n_boot": 20, "planted_seed": 1,
                                    "planted_active_frac": 0.3}
    assert cal["identity"] == ident and meta["calibration"]["identity"] == ident
    assert prd.identity_diffs(ident, ident) == []
    rep = json.loads((d / "report.json").read_text())
    assert rep["trainer"] == "train_psr_residual"
    assert "gate_liveness" in rep and "teacher_gate_liveness" in rep
    # the V artifact was composed and validated along the way
    assert (v_dir / "object.pt").exists()
    assert meta["teacher"]["object_sha256"] == prd.sha256_file(v_dir / "object.pt")
    # both identity reports carry BOTH halves (write identity + the hook
    # contracts of composite and reference), and the production precheck
    # ran the hook contract on the PERSISTED V it then distilled
    cal_rep = json.loads((d / prd.IDENTITY_REPORT_NAME).read_text())
    pre_rep = json.loads((d / prd.PRECHECK_REPORT_NAME).read_text())
    n_layers = model.config.num_hidden_layers
    for rep_ in (cal_rep, pre_rep):
        assert rep_["passed"] is True and "hook contract" in rep_["rule"]
        for l in range(n_layers):
            row = rep_["per_layer"][str(l)]
            assert row["pass"] and row["write_identity_pass"]
            assert row["hook_contract_composite"]["pass"]
            assert row["hook_contract_reference"]["pass"]
            assert row["hook_contract_composite"]["stream_dtype"] == "float32"
    # both stages check the PERSISTED V (calibration loads it before the
    # controls; the production precheck again right before distilling)
    for rep_ in (cal_rep, pre_rep):
        tc = rep_["teacher_hook_contract"]
        assert tc["passed"] is True and tc["object"] == str(v_dir / "object.pt")
        assert set(tc["per_layer"]) == {str(l) for l in range(n_layers)}
        # fp32 tiny model: the reloaded V's runtime is bitwise its declared write
        assert all(r["pass"] and r["max_abs_err"] == 0.0
                   for r in tc["per_layer"].values())


def test_identity_refusal_writes_the_full_report_first(bank, tiny_llama,
                                                      capsys, monkeypatch):
    """When the identity check fails, the complete per-layer report is on
    disk BEFORE the refusal, the REFUSED line names its path, and the
    stdout announces the same path — so a truncated stderr tail can never
    lose a layer again. Also: V's sidecar names the code that built it
    (``builder_commit`` from the app's commit pin)."""
    import sisteer.extraction.residual as res

    real = res.teacher_identity_problems

    def failing(*a, **k):
        problems, report = real(*a, **k)
        report[0] = {**report[0], "pass": False}
        return ["layer 0: composite-vs-reference WRITE discrepancy 1.000e+00 "
                "> tolerance 1.000e-05 (synthetic)"], report

    monkeypatch.setattr(res, "teacher_identity_problems", failing)
    monkeypatch.setenv("SDFBASE_DERIVED_SHA16", "0123456789abcdef")
    rc = prd.main(["all", "--no-production", *_common(bank),
                   *_model_args(bank)])
    out = capsys.readouterr()
    assert rc == 2
    d = _resid_dir(bank)
    rep_path = d / prd.IDENTITY_REPORT_NAME
    assert rep_path.exists()
    assert f"full per-layer report: {rep_path}" in out.err
    assert "REFUSED" in out.err and "layer 0: composite-vs-reference WRITE" in out.err
    assert f"identity report -> {rep_path}" in out.out
    rep = json.loads(rep_path.read_text())
    assert rep["passed"] is False and rep["stage"] == "calibration"
    assert rep["problems"] and "synthetic" in rep["problems"][0]
    assert rep["per_layer"]["0"]["pass"] is False
    # every OTHER layer is still recorded in full — nothing is truncated
    n_layers = tiny_llama[0].config.num_hidden_layers
    assert set(rep["per_layer"]) == {str(l) for l in range(n_layers)}
    assert all("bf16_perturbation_floor" in r for r in rep["per_layer"].values())
    assert not (d / "calibration_controls.json").exists()
    assert "inexpressible" not in rep_path.read_text().lower()
    meta = json.loads((_v_dir(bank) / "derived_meta.json").read_text())
    assert meta["builder_commit"] == "0123456789abcdef"
    blob = torch.load(_v_dir(bank) / "object.pt", map_location="cpu",
                      weights_only=True)
    assert blob["provenance"]["builder_commit"] == "0123456789abcdef"


def _production_identity(bank, tiny_llama):
    """The identity a `distill` run on this bank computes."""
    from sisteer.models import num_layers

    model, _tok = tiny_llama
    args = prd.build_parser().parse_args(["distill", *_common(bank),
                                          *_model_args(bank)])
    args.out_root = Path(args.root) / "derived"
    hp = prd.hp_from_args(args)
    pos, neg, _shared = prd.parents_provenance(args.root, KEY, MIX_SI, CTL_SI,
                                               args.pools)
    _d, _t, _h, manifest, sources = prd.build_capture_set(
        args.pools, MIX_SI, CTL_SI, frac=args.holdout_frac, seed=args.seed)
    return prd.calibration_identity(args, hp, model, pos_prov=pos,
                                    neg_prov=neg, manifest=manifest,
                                    sources=sources,
                                    layers=list(range(num_layers(model))))


def _calibration_with(identity, bank):
    d = _resid_dir(bank)
    d.mkdir(parents=True, exist_ok=True)
    (d / "calibration_controls.json").write_text(json.dumps(
        {"identity": identity, "trainer": identity.get("trainer"),
         "attainable_ed_content": {}, "floor": {},
         "planted_control": {"seed": 1},
         "parents_write_rms": {"mix": {}, "ctl": {}}, "controls": {}}))


def test_distill_refuses_a_calibration_of_another_identity(bank, tiny_llama,
                                                           capsys):
    """Production compares the FULL identity — a calibration built at
    another lr, on another parent, another capture seed, another
    measurement setting, or without any fingerprint at all, refuses by
    name; only an identical identity is accepted."""
    import copy

    prd.main(["compose-v", *_common(bank)])
    ident = _production_identity(bank, tiny_llama)
    fp = ident["fingerprint"]

    def refused(identity, *needles):
        _calibration_with(identity, bank)
        rc = prd.main(["distill", *_common(bank), *_model_args(bank)])
        err = capsys.readouterr().err
        assert rc == 2, err
        for n in needles:
            assert n in err, (n, err)

    # no identity block at all (a pre-fingerprint calibration)
    refused({"fingerprint": None}, "no identity fingerprint", "re-run calibrate")
    # lr differs (the pre-review check's only field)
    other = copy.deepcopy(ident)
    other["trainer"]["lr"] = ident["trainer"]["lr"] * 3
    refused(other, "trainer.lr", "same-trainer")
    # a different parent object
    other = copy.deepcopy(ident)
    other["parents"]["pos"]["object_sha256"] = "0" * 64
    refused(other, "parents.pos.object_sha256", "same-parents")
    # another capture split
    other = copy.deepcopy(ident)
    other["capture"]["seed"] = 7
    refused(other, "capture.seed", "same-capture")
    # another measurement setting
    other = copy.deepcopy(ident)
    other["measurement"]["n_boot"] = 999
    refused(other, "measurement.n_boot", "same-measurement")
    # another model revision / dtype
    other = copy.deepcopy(ident)
    other["model"]["dtype"] = "torch.bfloat16"
    refused(other, "model.dtype")
    # the fingerprint itself moves with any field
    assert other["fingerprint"] == fp  # (copied, stale)
    assert prd.identity_diffs(ident, other) == [
        "model.dtype (calibration 'torch.float32' != production 'torch.bfloat16')"]
    # the identical identity is accepted through the gate (the run then
    # proceeds into training on the synthetic calibration and stops at its
    # own missing fields — anything but the identity refusal)
    _calibration_with(ident, bank)
    try:
        prd.main(["distill", *_common(bank), *_model_args(bank)])
    except (KeyError, TypeError):
        pass
    assert "identity" not in capsys.readouterr().err


def test_trainer_defaults_read_the_d22_winner():
    hp = prd.trainer_defaults(REPO / "configs/sdfbase_gemma4e4b_base_fits.yaml")
    assert hp["lr"] == pytest.approx(5.090166e-4, rel=1e-6)
    assert hp["max_grad_norm"] == pytest.approx(0.09952642, rel=1e-6)
    assert hp["warmup_frac"] == 0.05 and hp["reg_coeff"] == 0.1
    assert hp["batch_size"] == 1 and hp["accum_steps"] == 8
    assert hp["max_len"] == 8192
    hq = prd.trainer_defaults(REPO / "configs/sdfbase_qwen35_4b_base_fits.yaml")
    assert hq["lr"] == pytest.approx(1.58384e-4, rel=1e-5)
    assert hq["batch_size"] == 2 and hq["accum_steps"] == 4
    # budget: the earlier 24 / 2 under-budgeted the planted controls
    assert prd.EPOCHS_CAP == 60 and prd.PATIENCE == 5


def test_identity_check_subcommand_passes_and_refuses_on_a_defective_v(
        bank, tiny_llama, capsys, monkeypatch):
    """The identity-only preflight: composes/loads V through the
    derived-object gate, runs the planted identity check (both halves) and
    V's own hook contract on the reference batches, writes
    identity_check_report.json (stage identity_check, no floor) and exits 0
    with a one-line JSON summary; with V's runtime defective (paired_write
    right) it exits 2 naming the report, which records every failing
    layer, and no calibration artifact is produced either way."""
    model, tok = tiny_llama
    rc = prd.main(["identity-check", *_common(bank), *_model_args(bank)])
    out = capsys.readouterr()
    assert rc == 0, out.err
    d = _resid_dir(bank)
    rep_path = d / prd.IDENTITY_CHECK_REPORT_NAME
    assert rep_path.exists()
    assert f"identity report -> {rep_path}" in out.out
    assert "identity check PASSED" in out.out
    rep = json.loads(rep_path.read_text())
    assert rep["stage"] == "identity_check" and rep["passed"] is True
    n_layers = model.config.num_hidden_layers
    assert set(rep["per_layer"]) == {str(l) for l in range(n_layers)}
    assert all(r["pass"] and r["hook_contract_composite"]["pass"]
               and r["hook_contract_reference"]["pass"]
               for r in rep["per_layer"].values())
    assert all(r["padded_vs_solo_floor_rms"] is None
               for r in rep["per_layer"].values())     # identity ONLY
    tc = rep["teacher_hook_contract"]
    assert tc["passed"] is True
    assert tc["object"] == str(_v_dir(bank) / "object.pt")
    assert all(r["pass"] and r["max_abs_err"] == 0.0
               for r in tc["per_layer"].values())
    summary = json.loads(out.out.split("identity-summary ", 1)[1]
                         .splitlines()[0])
    assert summary["passed"] is True
    assert set(summary["per_layer"]) == {str(l) for l in range(n_layers)}
    assert all(v["v_violations"] == 0 and v["composite_violations"] == 0
               for v in summary["per_layer"].values())
    assert not (d / "calibration_controls.json").exists()
    assert "inexpressible" not in rep_path.read_text().lower()

    # V's RUNTIME defective while its algebra is right: refused by name
    from residual_fixtures import ApplyPerturbed
    from sisteer.methods import load_object

    def defective(args, m, t):
        v_path = _v_dir(bank) / "object.pt"
        return ApplyPerturbed(load_object(v_path, model=m, tok=t), "double"), v_path

    monkeypatch.setattr(prd, "_load_or_compose_v", defective)
    rc = prd.main(["identity-check", *_common(bank), *_model_args(bank)])
    out = capsys.readouterr()
    assert rc == 2
    assert "REFUSED" in out.err and "HOOK CONTRACT" in out.err
    assert str(rep_path) in out.err
    rep = json.loads(rep_path.read_text())
    assert rep["passed"] is False
    assert rep["teacher_hook_contract"]["passed"] is False
    assert all(not r["pass"]
               for r in rep["teacher_hook_contract"]["per_layer"].values())
    # the planted identity check itself (composite vs reference) is fine
    assert all(r["pass"] and r["write_identity_pass"]
               for r in rep["per_layer"].values())
    assert not (d / "calibration_controls.json").exists()


# -- --wandb: the shared routing validator, before any model load ------------


def _fake_wandb(monkeypatch, *, default_entity, api_fail_times=0,
                init_fail_times=0):
    """A stand-in ``wandb`` module: ``Api()`` raises for the first
    ``api_fail_times`` constructions, ``init`` for the first
    ``init_fail_times`` calls; records every init kwargs."""
    import types

    import sisteer.wandb_entity as we

    fake = types.ModuleType("wandb")
    calls = {"api": 0, "init": []}

    class _Api:
        def __init__(self, timeout=None):
            calls["api"] += 1
            if calls["api"] <= api_fail_times:
                raise ConnectionError(f"api down #{calls['api']}")

        @property
        def default_entity(self):
            return default_entity

    def init(**kw):
        calls["init"].append(kw)
        if len(calls["init"]) <= init_fail_times:
            raise ConnectionError("init down")
        return types.SimpleNamespace(log=lambda *a, **k: None,
                                     finish=lambda: None)

    fake.Api = _Api
    fake.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(we, "_sleep", lambda s: None)
    return calls


@pytest.mark.parametrize("cmd", ["calibrate", "distill", "all"])
def test_wandb_flag_refuses_misrouting_before_any_model_load(bank, capsys,
                                                             monkeypatch, cmd):
    """The direct CLI paths used to call
    ``wandb.init(entity=$WANDB_ENTITY)`` inside a try/except → None — with
    WANDB_ENTITY unset the key's DEFAULT account received the runs, and
    any error meant "telemetry off". Now ``_wandb()`` runs the shared
    validator FIRST: an unset or empty WANDB_ENTITY, a key whose default
    entity differs, or an API that cannot be reached within the bounded
    retries is a configuration failure — REFUSED, exit 2, before the model
    is even loaded; nothing is ever initialised."""
    monkeypatch.setattr(prd, "_load_model", lambda a: (_ for _ in ()).throw(
        AssertionError("model loaded before the W&B guard")))
    argv = [cmd, *_common(bank), *_model_args(bank), "--wandb"]
    # unset
    calls = _fake_wandb(monkeypatch, default_entity="example-entity")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    rc = prd.main(argv)
    err = capsys.readouterr().err
    assert rc == 2 and "REFUSED" in err and "WANDB_ENTITY is unset" in err, err
    assert calls["api"] == 0 and calls["init"] == []
    # empty / whitespace
    monkeypatch.setenv("WANDB_ENTITY", "   ")
    rc = prd.main(argv)
    err = capsys.readouterr().err
    assert rc == 2 and "WANDB_ENTITY is unset" in err
    # a foreign key: the key's own entity is not the configured one
    calls = _fake_wandb(monkeypatch, default_entity="someone-else-none")
    monkeypatch.setenv("WANDB_ENTITY", "example-entity")
    rc = prd.main(argv)
    err = capsys.readouterr().err
    assert rc == 2 and "foreign key" in err and "someone-else-none" in err, err
    assert calls["api"] == 1 and calls["init"] == []
    # the API down for every one of the bounded tries: refused, not degraded
    calls = _fake_wandb(monkeypatch,
                        default_entity="example-entity",
                        api_fail_times=3)
    rc = prd.main(argv)
    err = capsys.readouterr().err
    assert rc == 2 and "after 3 tries" in err and "api down #3" in err, err
    assert calls["api"] == 3 and calls["init"] == []
    # without --wandb nothing is validated and nothing is initialised: the
    # run proceeds straight on (to the model load — or, for distill, to its
    # own "calibration missing" refusal, which comes before the model load)
    calls = _fake_wandb(monkeypatch, default_entity="someone-else-none")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    try:
        rc = prd.main([cmd, *_common(bank), *_model_args(bank)])
    except AssertionError as e:
        assert "model loaded" in str(e)
    else:
        err = capsys.readouterr().err
        assert cmd == "distill" and rc == 2 and "run `calibrate` first" in err
    assert calls["api"] == 0 and calls["init"] == []


def test_wandb_flag_inits_with_the_validated_entity(bank, monkeypatch):
    """The OK path: the key's default entity IS WANDB_ENTITY → ``wandb.init``
    is called with that entity passed EXPLICITLY (never left to the key)
    and the project from WANDB_PROJECT, before the model loads; transient
    API / init errors are retried within the bound, then the run opens."""
    monkeypatch.setattr(prd, "_load_model", lambda a: (_ for _ in ()).throw(
        RuntimeError("model-load-sentinel")))
    monkeypatch.setenv("WANDB_ENTITY", "example-entity")
    monkeypatch.setenv("WANDB_PROJECT", "example-project")
    monkeypatch.setenv("SISTEER_WANDB_PROJECT_DISTILL", "example-distill")   # WANDB_PROJECT still wins here (the Modal secret carries it)
    calls = _fake_wandb(monkeypatch,
                        default_entity="example-entity",
                        api_fail_times=2, init_fail_times=1)
    with pytest.raises(RuntimeError, match="model-load-sentinel"):
        prd.main(["calibrate", *_common(bank), *_model_args(bank), "--wandb"])
    assert calls["api"] == 3            # two failures, then verified
    assert len(calls["init"]) == 2      # one failure, then the run
    kw = calls["init"][-1]
    assert kw["entity"] == "example-entity"
    assert kw["project"] == "example-project"
    # named by its model, PAIR and sweep (the full - full pair's runs must
    # not read as the 100k pair's in W&B)
    assert kw["name"] == "resid-calib-tinybase-constv2-mix-100k-lrmult1"
    assert kw["config"] == {"mode": "psr_residual"}
    # the same validator, verbatim, is what the Modal wrapper calls at
    # container start (one rule for the wrapped and the direct route)
    src = (REPO / "scripts/modal/sdfbase_fits.py").read_text()
    assert "from sisteer.wandb_entity import require_wandb_entity" in src
    prd_src = (REPO / "scripts/sdfbase/psr_residual_distill.py").read_text()
    assert "wandb.init(" not in prd_src      # only through the validator
    assert prd_src.count("= _wandb(args, ") == 3    # one call site per command


# -- the distillation budget, the lr-mult sweep, monotone W&B steps --


def test_budget_knobs_defaults_aliases_and_lr_mult_parsing():
    """The explicit budget knobs (cap 60 / patience 5 by default — the
    earlier 24 / 2 under-budgeted the planted controls on both round-2
    models), their legacy spellings, and the --lr-mult sweep list."""
    ap = prd.build_parser()
    base = ["calibrate", "--model-key", "tinybase", "--model", TINY]
    a = ap.parse_args(base)
    assert (a.epochs, a.patience) == (60, 5) == (prd.EPOCHS_CAP, prd.PATIENCE)
    assert a.lr_mult is None and prd.lr_mults_from_args(a) == [0.3, 1.0, 3.0]
    a = ap.parse_args(base + ["--distill-epochs-cap", "7", "--distill-patience", "3",
                              "--lr-mult", "0.5, 1,3,3"])
    assert (a.epochs, a.patience) == (7, 3)
    assert prd.lr_mults_from_args(a) == [0.5, 1.0, 3.0]      # de-duplicated
    a = ap.parse_args(base + ["--epochs", "2", "--patience", "2"])   # aliases
    assert (a.epochs, a.patience) == (2, 2)
    assert prd.parse_lr_mults(None) == list(prd.DEFAULT_LR_MULTS)
    assert prd.parse_lr_mults(2) == [2.0] and prd.parse_lr_mults("3") == [3.0]
    for bad in ("", "0", "-1", "x", "nan"):
        with pytest.raises(prd.ResidualError, match="lr-mult"):
            prd.parse_lr_mults(bad)
    assert [prd.format_mult(m) for m in (0.3, 1.0, 3.0, 0.5)] == ["0.3", "1", "3", "0.5"]
    # hp_from_args: one multiplier at a time; the lr scales with it
    a = ap.parse_args(base + ["--lr", "0.01", "--lr-mult", "0.3,1"])
    with pytest.raises(prd.ResidualError, match="one lr multiplier"):
        prd.hp_from_args(a)
    assert prd.hp_from_args(a, 3.0)["lr"] == pytest.approx(0.03)
    assert prd.hp_from_args(a, 3.0)["lr_mult"] == 3.0
    a = ap.parse_args(base + ["--lr", "0.01"])
    assert prd.hp_from_args(a)["lr_mult"] == 1.0                # default single
    assert prd.hp_from_args(a)["epochs"] == 60 and prd.hp_from_args(a)["patience"] == 5


def test_stop_reason_and_selection_rule():
    hp = {"epochs": 24, "patience": 2}
    cap_improving = {"epochs_run": 24, "best_epoch": 24, "stopped_early": False,
                     "heldout_curve": [3.0, 2.0, 1.5, 1.4]}
    cap_flat = {"epochs_run": 24, "best_epoch": 22, "stopped_early": True,
                "heldout_curve": [3.0, 2.0, 1.0, 1.1, 1.2]}
    early = {"epochs_run": 11, "best_epoch": 9, "stopped_early": True,
             "heldout_curve": [1.0, 0.9, 0.8, 0.85, 0.86]}
    assert prd.stop_reason(cap_improving, hp) == \
        "epoch cap 24 reached — held-out still improving at the cap"
    assert prd.stop_reason(cap_flat, hp) == ("epoch cap 24 reached (early-stop "
                                             "criterion also met) — held-out flat "
                                             "since best epoch 22")
    assert prd.stop_reason(early, hp) == ("early stop: no held-out improvement for "
                                          "2 epochs after best epoch 9 (ran 11/24)")

    # calibration_for_mult: sweep files, legacy files, refusals
    doc = {"sweep": {"selected_lr_mult": 1.0, "complete": True, "lr_mults": [0.3, 1.0],
                     "per_mult": {"0.3": {"trainer": {"lr_mult": 0.3}},
                                  "1": {"trainer": {"lr_mult": 1.0}}}}}
    assert prd.calibration_for_mult(doc, None) == ({"trainer": {"lr_mult": 1.0}}, 1.0)
    assert prd.calibration_for_mult(doc, 0.3) == ({"trainer": {"lr_mult": 0.3}}, 0.3)
    with pytest.raises(prd.ResidualError, match="not in the calibration sweep"):
        prd.calibration_for_mult(doc, 7.0)
    doc["sweep"]["complete"] = False
    with pytest.raises(prd.ResidualError, match="INCOMPLETE"):
        prd.calibration_for_mult(doc, None)
    legacy = {"identity": {"trainer": {"lr_mult": 3.0}}, "controls": {}}
    assert prd.calibration_for_mult(legacy, None) == (legacy, 3.0)
    assert prd.calibration_for_mult({"controls": {}}, None)[1] == 1.0
    assert prd.sweep_summary(None) is None
    assert set(prd.sweep_summary(doc["sweep"])) == {
        "lr_mults", "selected_lr_mult", "selection", "selection_rule", "budget",
        "table", "complete"}


# -- the lr-mult selector (2026-09-09): a fit that can refuse ------------------


def _ladder(scores, *, half_width=0.01, heldout=None, curves=None):
    """A sweep ``{key: payload}`` from ``{lr_mult: worst-stratum ED}``: one
    control (``R_mixC_mix``) carries the ED per stratum with a bootstrap CI
    of ±``half_width`` (the mix stratum sits 0.005 above so ctl is the worst
    one), plus ``best_heldout_loss`` / ``heldout_curve`` per rung when given."""
    per = {}
    for m, sc in scores.items():
        key = prd.format_mult(m)
        ci = {"ed": sc, "lo": sc - half_width, "hi": sc + half_width, "n_docs": 5}
        hi = {"ed": sc + 0.005, "lo": sc + 0.005 - half_width,
              "hi": sc + 0.005 + half_width, "n_docs": 5}
        ctl = {"epochs_run": 5, "best_epoch": 3, "stopped_early": True,
               "stop_reason": "x", "wall_s": 1.0,
               "best_heldout_loss": (heldout or {}).get(m, 10.0),
               "heldout_curve": (curves or {}).get(m, [12.0, 11.0, 10.0, 10.1, 10.0]),
               "ed": {"ctl": {"content": ci}, "mix": {"content": hi}}}
        per[key] = {"trainer": {"lr_mult": m, "lr": 1e-3 * m},
                    "attainable_ed_content": {"ctl": sc, "mix": sc + 0.005},
                    "controls": {"R_mixC_mix": ctl}}
    return per


#: the GLM distilled residual's five-rung ladder (0.3/1/3 built, 6/9 the
#: calibration-only extension): worst-stratum attainable ED with its CI, and
#: both planted controls' best held-out loss + plateau — the case the rule
#: was written on. Under the previous argmax, the ORIGINAL 0.3/1/3 ladder
#: promoted its top rung; under this rule it refuses (3 < 4 rungs) and the
#: extended ladder selects 3 as a CI-separated interior with a held-out
#: divergence finding on both controls.
REAL_LADDER = {
    0.3: dict(ed=0.5447, lo=0.5356, hi=0.5529, a=116.86, ac=[140.1, 135.8, 132.1, 128.9],
              b=241.97, bc=[242.0, 242.2, 242.0, 242.0]),
    1.0: dict(ed=0.7605, lo=0.7516, hi=0.7738, a=34.45, ac=[34.5, 34.4, 34.5, 34.5],
              b=194.84, bc=[195.1, 194.8, 195.1, 195.1]),
    3.0: dict(ed=0.8125, lo=0.8045, hi=0.8241, a=47.48, ac=[49.3, 48.3, 48.5, 48.6],
              b=159.71, bc=[160.1, 160.2, 160.0, 160.2]),
    6.0: dict(ed=0.7592, lo=0.7493, hi=0.7743, a=43.86, ac=[64.3, 77.9, 79.9, 78.7],
              b=199.02, bc=[200.4, 200.6, 201.5, 199.6]),
    9.0: dict(ed=0.5597, lo=0.5571, hi=0.5628, a=112.40, ac=[122.2, 127.8, 122.7, 135.7],
              b=149.67, bc=[149.7, 150.0, 149.7, 149.9]),
}


def _real_ladder(mults=(0.3, 1.0, 3.0, 6.0, 9.0)):
    per = {}
    for m in mults:
        r = REAL_LADDER[m]
        ci = {"ed": r["ed"], "lo": r["lo"], "hi": r["hi"], "n_docs": 5}
        per[prd.format_mult(m)] = {
            "trainer": {"lr_mult": m, "lr": 7.9e-4 * m},
            "attainable_ed_content": {"ctl": r["ed"], "mix": r["ed"] + 0.007},
            "controls": {
                "R_mix_empty": {"best_heldout_loss": r["a"], "heldout_curve": r["ac"],
                                "best_epoch": 1, "epochs_run": 4, "stopped_early": True,
                                "stop_reason": "x", "wall_s": 1.0,
                                "ed": {"ctl": {"content": {"ed": r["ed"] + 0.08, "lo": r["lo"] + 0.08,
                                                           "hi": r["hi"] + 0.08, "n_docs": 5}},
                                       "mix": {"content": {"ed": r["ed"] + 0.09, "lo": r["lo"] + 0.09,
                                                           "hi": r["hi"] + 0.09, "n_docs": 5}}}},
                "R_mixC_mix": {"best_heldout_loss": r["b"], "heldout_curve": r["bc"],
                               "best_epoch": 1, "epochs_run": 4, "stopped_early": True,
                               "stop_reason": "x", "wall_s": 1.0,
                               "ed": {"ctl": {"content": ci},
                                      "mix": {"content": {**ci, "ed": r["ed"] + 0.007,
                                                          "lo": r["lo"] + 0.007, "hi": r["hi"] + 0.007}}}}}}
    return per


def test_selector_refuses_a_monotone_ladder_and_promotes_nothing():
    """The property the previous argmax lacked: a ladder whose score rises
    to the top rung yields a box-edge verdict, ``selected`` is None, no row
    is marked selected, and the raw argmax is only RECORDED (on the edge).
    The written sweep carries ``selected_lr_mult: null`` and the gate
    refuses to distill from it unless the user names a multiplier."""
    per = _ladder({0.3: 0.50, 1: 0.60, 3: 0.70, 6: 0.80, 9: 0.90})
    sel, table, rec = prd.select_lr_mult(per)
    assert sel is None
    assert rec["verdict"] == prd.VERDICT_BOX_EDGE and rec["promoted"] is False
    assert rec["selected_lr_mult"] is None
    assert not any(row["selected"] for row in table.values())
    assert rec["argmax_lr_mult"] == 9.0 and rec["argmax_on_edge"] is True
    assert rec["fit"]["concave"] is False and "not concave" in rec["reason"]
    assert rec["heldout_loss"] is None          # no selection, no joint check
    # a convex ladder (a dip) is refused the same way
    sel, _, rec = prd.select_lr_mult(_ladder({0.3: 0.9, 1: 0.7, 3: 0.6, 6: 0.7, 9: 0.9}))
    assert sel is None and rec["verdict"] == prd.VERDICT_BOX_EDGE
    # concave, but the vertex lies past the top rung -> box edge
    sel, _, rec = prd.select_lr_mult(_ladder({0.3: 0.50, 1: 0.65, 3: 0.78, 6: 0.86, 9: 0.90}))
    assert sel is None and rec["verdict"] == prd.VERDICT_BOX_EDGE
    assert "outside the ladder" in rec["reason"] or "edge rung" in rec["reason"]
    # fewer than MIN_FIT_RUNGS scored rungs cannot support the fit: the
    # ORIGINAL 0.3/1/3 ladder the GLM residual was built on now refuses
    sel, _, rec = prd.select_lr_mult(_real_ladder((0.3, 1.0, 3.0)))
    assert sel is None and rec["verdict"] == prd.VERDICT_BOX_EDGE
    assert "too few scored rungs" in rec["reason"] and rec["fit"] is None
    # a written refused sweep: selected_lr_mult None, the verdict recorded,
    # the top level a reference rung only; the gate refuses without --lr-mult
    # and lets an explicit multiplier through as the user's decision
    sweep = {"lr_mults": [0.3, 1, 3, 6, 9], "complete": True, "per_mult": per,
             "selected_lr_mult": None, "selection": rec}
    with pytest.raises(prd.ResidualError, match="REFUSED"):
        prd.calibration_for_mult({"sweep": sweep}, None)
    payload, m = prd.calibration_for_mult({"sweep": sweep}, 3.0)
    assert m == 3.0 and payload is per["3"]
    assert prd._selected_by(sweep, 3.0).startswith("user override: explicit --lr-mult 3 after")
    assert "box_edge" in prd._selected_by(sweep, 3.0)
    assert "REFUSED" in prd._selection_sentence(sweep) and "nothing promoted" in prd._selection_sentence(sweep)


def test_selector_needs_ci_separation_from_both_edges():
    """Concave with an interior vertex is not enough: the nearest rung's ED
    CI must clear both edge rungs' CIs. A flat ladder (wide CIs) refuses;
    a payload with no CI at all refuses; the same shape with tight CIs
    selects. A vertex whose nearest rung IS an edge refuses too."""
    shape = {0.3: 0.70, 1: 0.74, 3: 0.75, 6: 0.74, 9: 0.70}
    sel, _, rec = prd.select_lr_mult(_ladder(shape, half_width=0.06))
    assert sel is None and rec["verdict"] == prd.VERDICT_BOX_EDGE
    assert rec["fit"]["concave"] and rec["fit"]["vertex_inside_ladder"]
    assert rec["ci_separation"]["separated_from_both_edges"] is False
    assert "does not clear both edge rungs" in rec["reason"]
    sel, table, rec = prd.select_lr_mult(_ladder(shape, half_width=0.005))
    assert sel == "3" and rec["verdict"] == prd.VERDICT_INTERIOR and rec["promoted"]
    assert table["3"]["selected"] and sum(r["selected"] for r in table.values()) == 1
    assert table["3"]["score_ci95"] == [pytest.approx(0.745), pytest.approx(0.755)]
    assert rec["ci_separation"]["edges"]["0.3"]["separated"] and rec["ci_separation"]["edges"]["9"]["separated"]
    # no CI recorded anywhere -> cannot establish separation -> refused
    bare = _ladder(shape, half_width=0.005)
    for pl in bare.values():
        pl["controls"]["R_mixC_mix"].pop("ed")
    sel, table, rec = prd.select_lr_mult(bare)
    assert sel is None and "no CI recorded" in rec["reason"]
    assert all(row["score_ci95"] is None for row in table.values())
    # a concave fit whose vertex sits just inside the bottom edge: nearest rung is the edge
    sel, _, rec = prd.select_lr_mult(_ladder({0.3: 0.80, 0.4: 0.79, 3: 0.60, 6: 0.45, 9: 0.30},
                                             half_width=0.002))
    assert sel is None and rec["verdict"] == prd.VERDICT_BOX_EDGE
    assert "edge rung" in rec["reason"] or "outside the ladder" in rec["reason"]


def test_selector_on_the_real_five_rung_ladder_selects_3_and_files_the_divergence():
    """The case the rule was written on: concave (c2 < 0), vertex 1.84
    inside [0.3, 9], nearest rung 3 (interior), CI [0.8045, 0.8241] above
    both edges' CIs -> selected 3. The planted controls' held-out losses
    disagree with the ED — R_mix_empty is minimal at rung 1 (34.45 vs 47.48
    at 3, plateau noise ~0.34), R_mixC_mix at rung 9 (149.67 vs 159.71) —
    so the divergence is FILED on both, and the selection stands."""
    per = _real_ladder()
    sel, table, rec = prd.select_lr_mult(per)
    assert sel == "3" and rec["verdict"] == prd.VERDICT_INTERIOR and rec["promoted"]
    assert rec["selected_lr_mult"] == 3.0 and table["3"]["selected"]
    fit = rec["fit"]
    assert fit["concave"] and fit["coefficients"]["c2"] < 0
    assert fit["vertex_lr_mult"] == pytest.approx(1.84, abs=0.05)
    assert fit["vertex_inside_ladder"] and fit["nearest_rung_lr_mult"] == 3.0
    assert rec["ci_separation"]["separated_from_both_edges"]
    assert rec["argmax_lr_mult"] == 3.0 and rec["argmax_on_edge"] is False
    hd = rec["heldout_loss"]
    assert rec["heldout_divergence_finding"] is True
    assert hd["R_mix_empty"]["ladder_min_lr_mult"] == 1.0 and hd["R_mix_empty"]["divergent"]
    assert hd["R_mix_empty"]["selected_loss"] == 47.48 and hd["R_mix_empty"]["min_loss"] == 34.45
    assert hd["R_mix_empty"]["noise"] == pytest.approx(0.3445, abs=1e-3)   # 1% floor > plateau range 0.1
    assert hd["R_mixC_mix"]["ladder_min_lr_mult"] == 9.0 and hd["R_mixC_mix"]["divergent"]
    sweep = {"selected_lr_mult": 3.0, "selection": rec, "per_mult": per, "complete": True}
    assert prd._selected_by(sweep, 3.0) == prd.SELECTION_RULE
    assert prd._selected_by(sweep, 1.0).startswith("user override: explicit --lr-mult 1 (the sweep selected 3)")
    sent = prd._selection_sentence(sweep)
    assert sent.startswith("selected lr_mult 3 (interior") and "HELD-OUT DIVERGENCE FINDING" in sent
    # within-noise agreement files nothing
    agree = _ladder({0.3: 0.70, 1: 0.74, 3: 0.75, 6: 0.74, 9: 0.70}, half_width=0.005,
                    heldout={0.3: 12, 1: 10.05, 3: 10.0, 6: 10.2, 9: 12},
                    curves={3: [12.0, 10.0, 10.3, 10.1]})
    _, _, rec = prd.select_lr_mult(agree)
    assert rec["heldout_divergence_finding"] is False
    assert rec["heldout_loss"]["R_mixC_mix"]["divergent"] is False


def test_selector_single_rung_is_the_users_choice_not_a_selection():
    """One multiplier is not a ladder: it passes through as single_rung,
    the stamp says so, and nothing claims the rule chose it."""
    per = _ladder({1: 0.7})
    sel, table, rec = prd.select_lr_mult(per)
    assert sel == "1" and rec["verdict"] == prd.VERDICT_SINGLE_RUNG and rec["promoted"]
    assert table["1"]["selected"] and rec["fit"] is None
    sweep = {"selected_lr_mult": 1.0, "selection": rec, "per_mult": per, "complete": True}
    assert prd._selected_by(sweep, 1.0).startswith("explicit single --lr-mult 1")
    assert prd._selected_by(None, 1.0) is None
    assert prd.SELECTION_RULE.startswith("a concave quadratic in log(lr_mult)")
    assert prd.MIN_FIT_RUNGS == 4
    with pytest.raises(prd.ResidualError, match="empty sweep"):
        prd.select_lr_mult({})


def test_wandb_steps_are_monotone_across_sections():
    """A telemetry defect: control (b) restarted W&B's step at 0/1 and
    every point below the run's high-water mark was dropped. Sections now
    log at a monotone offset; the local step rides along as a metric."""
    import types

    logged = []
    run = types.SimpleNamespace(log=lambda payload, step=None: logged.append((step, payload)))
    steps = prd.MonotoneSteps()
    a = prd._log_fn_factory("calib/lrmult1/R_mix_empty", run, steps=steps)
    steps.new_section()
    for st in (0, 1, 2, 3):
        a({"loss": 1.0 / (st + 1)}, st)
    b = prd._log_fn_factory("calib/lrmult1/R_mixC_mix", run, steps=steps)
    steps.new_section()
    for st in (0, 1, 2):
        b({"loss": 2.0 / (st + 1)}, st)
    c = prd._log_fn_factory("psr_residual/production", run, steps=steps)
    steps.new_section()
    c({"loss": 9.0}, 0)
    gsteps = [g for g, _ in logged]
    assert gsteps == [0, 1, 2, 3, 4, 5, 6, 7]         # strictly increasing
    assert all(x < y for x, y in zip(gsteps, gsteps[1:]))
    assert logged[4][1]["calib/lrmult1/R_mixC_mix/local_step"] == 0
    assert logged[7][1]["psr_residual/production/local_step"] == 0
    assert logged[7][1]["psr_residual/production/loss"] == 9.0
    # the module-level default is shared by every _log_fn_factory call, and
    # _distill opens a new section (the single seam every training passes)
    src = (REPO / "scripts/sdfbase/psr_residual_distill.py").read_text()
    body = src.split("def _distill(", 1)[1].split("\ndef ", 1)[0]
    assert ".new_section()" in body
    assert isinstance(prd.WANDB_STEPS, prd.MonotoneSteps)


def test_all_sweeps_lr_mults_refuses_a_short_ladder_and_stamps_an_override(
        bank, tiny_llama, capsys, monkeypatch):
    """``all`` with ``--lr-mult 0.5,1``: both multipliers' controls are run
    and recorded (``sweep.per_mult``, each with its own identity at its
    lr), every control carries a stop reason, the doc records carry
    lr_mult — and the SELECTOR REFUSES: two rungs cannot support the
    quadratic, so ``selected_lr_mult`` is null, the verdict is box_edge,
    the file's top level is a reference rung only, no production runs
    (rc 4) and no derived_meta.json exists. ``distill`` then refuses
    without a multiplier, and builds at an explicit one stamped as the
    user's override; an unswept one or an incomplete sweep still refuse."""
    model, tok = tiny_llama
    rc = prd.main(["all", *_common(bank), *_model_args(bank, lr_mult="0.5,1")])
    out = capsys.readouterr()
    assert rc == 4, out.err
    d = _resid_dir(bank)
    doc = json.loads((d / "calibration_controls.json").read_text())
    sw = doc["sweep"]
    assert sw["lr_mults"] == [0.5, 1.0] and sw["finished_lr_mults"] == [0.5, 1.0]
    assert sw["complete"] is True and set(sw["per_mult"]) == {"0.5", "1"}
    assert sw["budget"] == {"epochs_cap": 2, "patience": 2}
    assert sw["selection_rule"] == prd.SELECTION_RULE
    assert sw["selected_lr_mult"] is None
    sel = sw["selection"]
    assert sel["verdict"] == prd.VERDICT_BOX_EDGE and sel["promoted"] is False
    assert "too few scored rungs" in sel["reason"] and sel["n_scored"] == 2
    assert not any(row["selected"] for row in sw["table"].values())
    assert sw["top_level_is"].startswith("REFERENCE ONLY")
    ref_key = sel["reference_lr_mult_key"]
    assert ref_key in ("0.5", "1")
    for k in ("identity", "trainer", "attainable_ed_content", "controls"):
        assert doc[k] == sw["per_mult"][ref_key][k], k
    for key, pl in sw["per_mult"].items():
        assert pl["trainer"]["lr_mult"] == float(key)
        assert pl["identity"]["trainer"]["lr_mult"] == float(key)
        assert pl["identity"]["trainer"]["lr"] == pytest.approx(0.02 * float(key))
        assert pl["teacher_identity_check"]["passed"] is True
        assert pl["teacher_identity_check"]["teacher_hook_contract_passed"] is True
        for c in pl["controls"].values():
            assert c["stop_reason"] and ("epoch cap" in c["stop_reason"]
                                         or "early stop" in c["stop_reason"])
        assert sw["table"][key]["score_ci95"] is not None      # the CI the rule reads
    assert sw["per_mult"]["0.5"]["identity"]["fingerprint"] != \
        sw["per_mult"]["1"]["identity"]["fingerprint"]
    rep = json.loads((d / prd.IDENTITY_REPORT_NAME).read_text())
    assert rep["teacher_hook_contract"]["passed"] is True
    assert rep["teacher_hook_contract"]["object"] == str(_v_dir(bank) / "object.pt")
    recs = [json.loads(l) for l in (d / "calibration_doc_records.jsonl").read_text().splitlines()]
    assert {r["lr_mult"] for r in recs} == {0.5, 1.0}
    assert {r["control"] for r in recs} == {"R_mix_empty", "R_mixC_mix"}
    # nothing was promoted
    assert not (d / "derived_meta.json").exists()
    assert not (d / "object.pt").exists()
    assert "inexpressible" not in (d / "calibration_controls.json").read_text().lower()
    stdout = out.out
    assert "REFUSED (box_edge" in stdout and "nothing promoted" in stdout
    assert "no production distillation" in stdout
    assert "calibration controls at lr_mult=0.5" in stdout
    assert "calibration controls at lr_mult=1" in stdout
    # --no-production on a refused sweep is a successful calibration (rc 0)
    assert prd.main(["all", "--no-production", *_common(bank),
                     *_model_args(bank, lr_mult="0.5,1")]) == 0
    capsys.readouterr()

    # distill: no multiplier = refused; explicit swept = the user's override
    seen = []

    def fake_prod(model_, tok_, V, mix, ctl, train, hold, hp, calibration, **kw):
        sweep = kw.get("sweep") or {}
        seen.append((hp["lr_mult"], calibration["trainer"]["lr_mult"],
                     sweep.get("selected_lr_mult"),
                     prd._selected_by(sweep, hp["lr_mult"])))
        return {"object": "x", "verdict": "recovered", "ed": {}, "epochs_run": 0,
                "stop_reason": "s", "lr_mult": hp["lr_mult"], "wall_s": 0.0}

    monkeypatch.setattr(prd, "run_production", fake_prod)
    no_mult = _model_args(bank)
    i = no_mult.index("--lr-mult")
    no_mult = no_mult[:i] + no_mult[i + 2:]          # no --lr-mult at all
    rc = prd.main(["distill", *_common(bank), *no_mult])
    err = capsys.readouterr().err
    assert rc == 2 and "REFUSED" in err and "pass --lr-mult" in err
    assert prd.main(["distill", *_common(bank), *_model_args(bank, lr_mult="1")]) == 0
    assert len(seen) == 1
    assert seen[0][:3] == (1.0, 1.0, None)
    assert seen[0][3].startswith("user override: explicit --lr-mult 1 after the sweep's box_edge refusal")
    rc = prd.main(["distill", *_common(bank), *_model_args(bank, lr_mult="7")])
    err = capsys.readouterr().err
    assert rc == 2 and "not in the calibration sweep" in err and "the sweep refused" in err
    rc = prd.main(["distill", *_common(bank), *_model_args(bank, lr_mult="0.5,1")])
    err = capsys.readouterr().err
    assert rc == 2 and "ONE --lr-mult" in err
    doc["sweep"]["complete"] = False
    (d / "calibration_controls.json").write_text(json.dumps(doc))
    rc = prd.main(["distill", *_common(bank), *no_mult])
    err = capsys.readouterr().err
    assert rc == 2 and "INCOMPLETE" in err
    assert len(seen) == 1


# -- resume, checkpoints, the fan-out merge --


def _strip_volatile(payload: dict) -> dict:
    """A calibration payload without the fields that legitimately differ
    between two runs of the same identity: timestamps, wall clocks, and
    the identity report's path (a part writes its own)."""
    import copy

    p = copy.deepcopy(payload)
    p.pop("built_at", None)
    for c in (p.get("controls") or {}).values():
        c.pop("wall_s", None)
    (p.get("teacher_identity_check") or {}).pop("report", None)
    return p


def test_calibrate_reuses_finished_multipliers_and_reruns_other_identities(
        bank, tiny_llama, capsys, monkeypatch):
    """Finding 1(a): a sweep interrupted after its first multiplier is
    RESUMED — the retry reuses the finished payload (identity match, byte
    for byte) and runs only the unfinished multiplier; a payload of
    another identity (a trainer knob changed) is re-run and replaced,
    saying which field differed; ``--no-resume`` re-runs everything; a
    multiplier not in the requested sweep is dropped loudly."""
    real = prd._distill
    calls: list[str] = []
    state = {"kill_at": 3}

    def counting(*a, **k):
        calls.append(k["tag"])
        if state["kill_at"] is not None and len(calls) == state["kill_at"]:
            raise RuntimeError("container died")
        return real(*a, **k)

    monkeypatch.setattr(prd, "_distill", counting)
    argv = ["all", "--no-production", *_common(bank),
            *_model_args(bank, lr_mult="0.5,1")]
    with pytest.raises(RuntimeError, match="container died"):
        prd.main(argv)
    capsys.readouterr()
    d = _resid_dir(bank)
    partial = json.loads((d / "calibration_controls.json").read_text())
    assert partial["sweep"]["complete"] is False
    assert list(partial["sweep"]["per_mult"]) == ["0.5"]
    assert calls == ["calib/lrmult0.5/R_mix_empty", "calib/lrmult0.5/R_mixC_mix",
                     "calib/lrmult1/R_mix_empty"]
    # the finished multiplier's doc records are on disk too (rewritten per
    # multiplier, like the sweep file), so the retry can carry them over
    recs_partial = [json.loads(l) for l in
                    (d / "calibration_doc_records.jsonl").read_text().splitlines()]
    assert recs_partial and {r["lr_mult"] for r in recs_partial} == {0.5}
    # retry: the finished multiplier is reused, only lr_mult=1 runs
    calls.clear()
    state["kill_at"] = None
    assert prd.main(argv) == 0
    out = capsys.readouterr().out
    assert calls == ["calib/lrmult1/R_mix_empty", "calib/lrmult1/R_mixC_mix"]
    assert "lr_mult=0.5: REUSING the finished payload" in out
    assert "identity fingerprint" in out
    doc = json.loads((d / "calibration_controls.json").read_text())
    assert doc["sweep"]["complete"] is True
    assert list(doc["sweep"]["per_mult"]) == ["0.5", "1"]
    assert doc["sweep"]["per_mult"]["0.5"] == partial["sweep"]["per_mult"]["0.5"]
    assert doc["sweep"]["finished_lr_mults"] == [0.5, 1.0]
    recs = [json.loads(l) for l in
            (d / "calibration_doc_records.jsonl").read_text().splitlines()]
    assert {r["lr_mult"] for r in recs} == {0.5, 1.0}
    assert [r for r in recs if r["lr_mult"] == 0.5] == recs_partial   # carried over
    # a full reference sweep (no interruption) selects and scores the same
    ref = json.loads((d / "calibration_controls.json").read_text())
    # another identity at lr_mult=0.5 (a trainer knob): NOT reused, replaced
    calls.clear()
    argv2 = ["all", "--no-production", *_common(bank),
             *_model_args(bank, lr_mult="0.5,1", reg_coeff="0.2")]
    assert prd.main(argv2) == 0
    out = capsys.readouterr().out
    assert len(calls) == 4
    assert "lr_mult=0.5: the payload" in out and "NOT reusable" in out
    assert "trainer.reg_coeff" in out
    doc2 = json.loads((d / "calibration_controls.json").read_text())
    assert doc2["sweep"]["per_mult"]["0.5"]["identity"]["trainer"]["reg_coeff"] == 0.2
    assert doc2["sweep"]["per_mult"]["0.5"]["identity"]["fingerprint"] != \
        ref["sweep"]["per_mult"]["0.5"]["identity"]["fingerprint"]
    # --no-resume: everything runs again even under the same identity
    calls.clear()
    assert prd.main(argv2 + ["--no-resume"]) == 0
    assert len(calls) == 4 and "REUSING" not in capsys.readouterr().out
    # a narrower sweep drops the payload that is not requested, loudly
    calls.clear()
    assert prd.main(["all", "--no-production", *_common(bank),
                     *_model_args(bank, lr_mult="1", reg_coeff="0.2")]) == 0
    out = capsys.readouterr().out
    assert calls == [] and "REUSING" in out
    assert "dropping the finished lr_mult=0.5 payload" in out
    doc3 = json.loads((d / "calibration_controls.json").read_text())
    assert list(doc3["sweep"]["per_mult"]) == ["1"] and doc3["sweep"]["lr_mults"] == [1.0]
    # reusable_payload's own refusals
    pl = doc3["sweep"]["per_mult"]["1"]
    assert prd.reusable_payload(pl, pl["identity"]) == (
        True, f"identity fingerprint {pl['identity']['fingerprint'][:16]} matches")
    assert prd.reusable_payload(None, pl["identity"])[0] is False
    unfinished = {**pl, "controls": {"R_mix_empty": pl["controls"]["R_mix_empty"]}}
    ok, why = prd.reusable_payload(unfinished, pl["identity"])
    assert not ok and "R_mixC_mix has no ED table" in why
    ok, why = prd.reusable_payload({**pl, "identity": {}}, pl["identity"])
    assert not ok and "no identity fingerprint" in why


def test_calibration_controls_resume_from_trainer_checkpoints_bitwise(
        bank, tiny_llama, capsys, monkeypatch):
    """Finding 1(d): with ``--checkpoint-every`` every control fit writes
    resumable checkpoints keyed by the multiplier's identity fingerprint +
    the control's name; a calibrate killed mid-control and re-issued
    RESUMES that control from its checkpoint and reproduces the
    uninterrupted run's curves, best epochs and attainable EDs exactly
    (CPU fp32 — the fitckpt contract). The production checkpoint key is
    fingerprint-bound too."""
    import torch as _torch

    ref_out = bank / "ref" / "calibration_controls.json"
    argv = ["calibrate", *_common(bank), *_model_args(bank, lr_mult="1"),
            "--checkpoint-every", "1"]
    assert prd.main(argv + ["--out", str(ref_out)]) == 0
    capsys.readouterr()
    ref = json.loads(ref_out.read_text())
    assert (ref_out.parent / "ckpt" / "lrmult1" / "R_mix_empty" / "ckpt.pt").exists()
    assert (ref_out.parent / "ckpt" / "lrmult1" / "R_mixC_mix" / "ckpt.pt").exists()
    ck = _torch.load(ref_out.parent / "ckpt" / "lrmult1" / "R_mix_empty" / "ckpt.pt",
                     weights_only=False)
    fp = ref["identity"]["fingerprint"]
    assert ck["checkpoint_key"] == f"psr_residual_calib|{fp}|R_mix_empty"
    assert ck["trainer"] == "train_psr_residual"

    class _Killed(RuntimeError):
        pass

    orig = _torch.optim.AdamW.step
    n = {"steps": 0}

    def killing_step(self, *a, **k):
        r = orig(self, *a, **k)
        n["steps"] += 1
        if n["steps"] >= 4:
            raise _Killed()
        return r

    k_out = bank / "killed" / "calibration_controls.json"
    monkeypatch.setattr(_torch.optim.AdamW, "step", killing_step)
    with pytest.raises(_Killed):
        prd.main(argv + ["--out", str(k_out)])
    monkeypatch.setattr(_torch.optim.AdamW, "step", orig)
    capsys.readouterr()
    assert not k_out.exists()
    assert (k_out.parent / "ckpt" / "lrmult1" / "R_mix_empty" / "ckpt.pt").exists()
    assert prd.main(argv + ["--out", str(k_out)]) == 0
    out = capsys.readouterr().out
    assert "train_psr_residual RESUMING from checkpoint" in out
    res = json.loads(k_out.read_text())
    for name in ("R_mix_empty", "R_mixC_mix"):
        a, b = ref["controls"][name], res["controls"][name]
        for key in ("heldout_curve", "best_epoch", "epochs_run", "stopped_early",
                    "optimizer_steps_run", "best_heldout_loss", "ed",
                    "final_heldout_loss", "n_steps"):
            assert a[key] == b[key], (name, key)
    assert res["attainable_ed_content"] == ref["attainable_ed_content"]
    assert res["identity"]["fingerprint"] == fp
    # the production checkpoint spec is bound to the same identity
    spec = prd.checkpoint_spec(7, bank / "o", "production", f"psr_residual_prod|{fp}")
    assert spec == {"checkpoint_every": 7, "checkpoint_dir": str(bank / "o" / "ckpt" / "production"),
                    "checkpoint_key": f"psr_residual_prod|{fp}"}
    assert prd.checkpoint_spec(0, bank, "production", "x") is None
    src = (REPO / "scripts/sdfbase/psr_residual_distill.py").read_text()
    assert "psr_residual_prod|{identity['fingerprint']}" in src
    body = src.split("def cmd_distill", 1)[1].split("\ndef ", 1)[0]
    assert "checkpoint_spec(args.checkpoint_every, out_dir" in body


def test_merge_calibration_parts_equals_the_in_process_sweep(bank, tiny_llama,
                                                            capsys, monkeypatch):
    """Finding 1(c): the derived app runs ONE multiplier per container
    (``calibrate --lr-mult <m> --out calib_parts/lrmult<m>/...``) and
    ``merge-calibration`` assembles the sweep file — identical, field for
    field (timestamps / wall clocks / report paths aside), to the
    in-process sweep ``all`` writes, same selection, same doc records; a
    ``distill`` then gates on the merged file. Refusals: a missing
    multiplier, an unrequested one, an incomplete part, a part of another
    identity, a duplicated multiplier."""
    import copy

    d = _resid_dir(bank)
    assert prd.main(["all", "--no-production", *_common(bank),
                     *_model_args(bank, lr_mult="0.5,1")]) == 0
    capsys.readouterr()
    ref = json.loads((d / "calibration_controls.json").read_text())
    ref_recs = (d / "calibration_doc_records.jsonl").read_text().splitlines()
    (d / "calibration_controls.json").unlink()
    (d / "calibration_doc_records.jsonl").unlink()
    parts = {}
    for m in ("0.5", "1"):
        part = d / prd.CALIB_PARTS_DIRNAME / f"lrmult{m}" / "calibration_controls.json"
        assert prd.main(["calibrate", *_common(bank), *_model_args(bank, lr_mult=m),
                         "--out", str(part)]) == 0
        parts[m] = part
        pdoc = json.loads(part.read_text())
        assert pdoc["sweep"]["complete"] is True and list(pdoc["sweep"]["per_mult"]) == [m]
        assert (part.parent / "calibration_doc_records.jsonl").exists()
        assert (part.parent / prd.IDENTITY_REPORT_NAME).exists()
    capsys.readouterr()
    # the merge (default part paths under <out dir>/calib_parts/)
    assert prd.main(["merge-calibration", *_common(bank), "--lr-mult", "0.5,1"]) == 0
    out = capsys.readouterr().out
    merged = json.loads((d / "calibration_controls.json").read_text())
    assert merged["sweep"]["lr_mults"] == [0.5, 1.0]
    assert merged["sweep"]["complete"] is True
    assert merged["sweep"]["selected_lr_mult"] == ref["sweep"]["selected_lr_mult"] is None
    assert merged["sweep"]["selection"]["verdict"] == prd.VERDICT_BOX_EDGE      # 2 rungs: refused, like `all`
    assert merged["sweep"]["selection"]["reference_lr_mult_key"] == \
        ref["sweep"]["selection"]["reference_lr_mult_key"]
    assert merged["sweep"]["selection_rule"] == prd.SELECTION_RULE
    assert merged["sweep"]["budget"] == ref["sweep"]["budget"]
    for m in ("0.5", "1"):
        assert _strip_volatile(merged["sweep"]["per_mult"][m]) == \
            _strip_volatile(ref["sweep"]["per_mult"][m]), m
        assert merged["sweep"]["per_mult"][m]["teacher_identity_check"]["report"] == \
            str(parts[m].parent / prd.IDENTITY_REPORT_NAME)
    for k in ("identity", "attainable_ed_content", "floor", "trainer"):
        assert merged[k] == ref[k], k
    tbl_m = {k: {kk: v for kk, v in row.items() if kk != "controls"}
             for k, row in merged["sweep"]["table"].items()}
    tbl_r = {k: {kk: v for kk, v in row.items() if kk != "controls"}
             for k, row in ref["sweep"]["table"].items()}
    assert tbl_m == tbl_r
    recs = (d / "calibration_doc_records.jsonl").read_text().splitlines()
    assert len(recs) == len(ref_recs)
    assert [json.loads(r)["lr_mult"] for r in recs] == [json.loads(r)["lr_mult"] for r in ref_recs]
    assert f"{len(recs)} doc records" in out and "REFUSED (box_edge" in out
    # production gates on the merged file (through the identity check): the
    # refused sweep needs an explicit multiplier from the caller
    seen = []
    monkeypatch.setattr(prd, "run_production", lambda *a, **kw: seen.append(
        (a[7]["lr_mult"], a[8]["trainer"]["lr_mult"])) or {
        "object": "x", "verdict": "recovered", "ed": {}, "epochs_run": 0,
        "stop_reason": "s", "lr_mult": a[7]["lr_mult"], "wall_s": 0.0})
    no_mult = _model_args(bank)
    i = no_mult.index("--lr-mult")
    assert prd.main(["distill", *_common(bank), *(no_mult[:i] + no_mult[i + 2:])]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert prd.main(["distill", *_common(bank), *_model_args(bank, lr_mult="1")]) == 0
    assert seen == [(1.0, 1.0)]
    capsys.readouterr()

    # --- refusals -------------------------------------------------------
    def refused(argv, *needles):
        rc = prd.main(["merge-calibration", *_common(bank), *argv])
        err = capsys.readouterr().err
        assert rc == 2, err
        for n in needles:
            assert n in err, (n, err)

    refused(["--lr-mult", "0.5,1,3"], "no finished payload for lr_mult=['3']")
    refused(["--lr-mult", "1", "--parts", str(parts["0.5"])],
            "carries lr_mult=0.5, which is not in the requested sweep")
    refused(["--lr-mult", "0.5,1", "--parts", f"{parts['0.5']},{parts['0.5']},{parts['1']}"],
            "finished twice")
    other = copy.deepcopy(json.loads(parts["1"].read_text()))
    other["sweep"]["per_mult"]["1"]["identity"]["capture"]["seed"] = 7
    bad = bank / "bad_part.json"
    bad.write_text(json.dumps(other))
    refused(["--lr-mult", "0.5,1", "--parts", f"{parts['0.5']},{bad}"],
            "DIFFERENT identities", "capture.seed")
    inc = copy.deepcopy(json.loads(parts["1"].read_text()))
    inc["sweep"]["complete"] = False
    bad.write_text(json.dumps(inc))
    refused(["--lr-mult", "0.5,1", "--parts", f"{parts['0.5']},{bad}"], "INCOMPLETE")
    bad.write_text(json.dumps({"controls": {}}))
    refused(["--lr-mult", "0.5,1", "--parts", f"{parts['0.5']},{bad}"], "not a sweep file")
    refused(["--lr-mult", "0.5,1", "--parts", f"{parts['0.5']},{bank / 'nope.json'}"],
            "no finished payload for lr_mult=['1']", "missing part file(s)", "nope.json")
    # the sweep-variable fields are exactly lr and lr_mult
    assert prd._SWEEP_VARIABLE_FIELDS == frozenset({"trainer.lr", "trainer.lr_mult"})
    assert prd._identity_diff_paths(ref["sweep"]["per_mult"]["0.5"]["identity"],
                                    ref["sweep"]["per_mult"]["1"]["identity"]) == \
        ["trainer.lr", "trainer.lr_mult"]


# -- atomic V publish + verify-v; --no-resume is fresh --


def test_compose_v_publishes_atomically_and_verify_v_readmits(bank, tiny_llama, capsys,
                                                                monkeypatch):
    """Finding 2: object.pt and derived_meta.json are moved into place from
    temp names (sidecar last), so a compose that dies mid-way never leaves
    an object without its sidecar at the admitted path; ``verify-v`` re-
    admits the persisted V against the LIVE bank (exit 2 on a missing
    object, a torn publish, a parent whose object changed under the same
    cell id) — the compose stage's question before an existing V stands."""
    import torch as _torch

    d = _v_dir(bank)
    obj, meta = d / "object.pt", d / "derived_meta.json"
    # a compose that dies at the round-trip leaves NOTHING at the admitted path
    real_load = _torch.load

    def dying_load(path, *a, **k):
        if str(path).endswith("object.pt.tmp"):
            raise RuntimeError("container died mid-compose")
        return real_load(path, *a, **k)

    monkeypatch.setattr(_torch, "load", dying_load)
    with pytest.raises(RuntimeError, match="died mid-compose"):
        prd.main(["compose-v", *_common(bank)])
    monkeypatch.setattr(_torch, "load", real_load)
    assert not obj.exists() and not meta.exists()
    assert (d / "object.pt.tmp").exists()             # only the temp file
    rc = prd.main(["verify-v", *_common(bank)])
    err = capsys.readouterr().err
    assert rc == 2 and "missing" in err
    # a clean compose publishes both, no temp files left, verify-v passes
    assert prd.main(["compose-v", *_common(bank)]) == 0
    assert obj.exists() and meta.exists()
    assert not (d / "object.pt.tmp").exists() and not (d / "derived_meta.json.tmp").exists()
    assert prd.main(["verify-v", *_common(bank)]) == 0
    assert "verified against the live bank" in capsys.readouterr().out
    assert prd.verify_v(bank / "out", bank / "out/derived", KEY, MIX_SI) == []
    # a torn publish (object without sidecar) is a problem
    meta_bytes = meta.read_bytes()
    meta.unlink()
    probs = prd.verify_v(bank / "out", bank / "out/derived", KEY, MIX_SI)
    assert probs and "torn publish" in probs[0]
    meta.write_bytes(meta_bytes)
    # a parent refit under the same cell id: V no longer admits
    src_path = bank / f"out/{KEY}/cells_all/{MIX_SI}__psr_ce_all/object.pt"
    src_path.write_bytes(src_path.read_bytes() + b"\0")
    rc = prd.main(["verify-v", *_common(bank)])
    err = capsys.readouterr().err
    assert rc == 2 and "does not pass the derived-object verdict" in err
    # the compose order in the source: temp object -> temp sidecar -> replace, sidecar last
    src = (REPO / "scripts/sdfbase/psr_residual_distill.py").read_text()
    body = src.split("def compose_v(", 1)[1].split("\ndef ", 1)[0]
    assert body.index("torch.save(blob, tmp_obj)") < body.index("os.replace(tmp_obj, obj_path)")
    assert body.index("os.replace(tmp_obj, obj_path)") < body.index("os.replace(tmp_meta, meta_path)")


def test_no_resume_discards_checkpoints_and_runs_from_zero(bank, tiny_llama, capsys):
    """Finding 3: ``--no-resume`` used to skip the finished payload but still
    hand the trainer the same fingerprint-bound checkpoint dir, so the
    'fresh' run resumed the old trajectory. Now every fit under --no-resume
    discards its existing checkpoint dir first: no RESUMING line, the full
    optimizer-step count runs, and fresh checkpoints are written."""
    out = bank / "fresh" / "calibration_controls.json"
    argv = ["calibrate", *_common(bank), *_model_args(bank, lr_mult="1"),
            "--checkpoint-every", "1", "--out", str(out)]
    assert prd.main(argv) == 0
    capsys.readouterr()
    ref = json.loads(out.read_text())
    ck_a = out.parent / "ckpt" / "lrmult1" / "R_mix_empty" / "ckpt.pt"
    assert ck_a.exists()
    mtime = ck_a.stat().st_mtime_ns
    steps = ref["controls"]["R_mix_empty"]["optimizer_steps_run"]
    assert steps > 0
    # default: the finished payload is reused and no fit runs at all
    assert prd.main(argv) == 0
    o = capsys.readouterr().out
    assert "REUSING the finished payload" in o and "RESUMING" not in o
    assert ck_a.stat().st_mtime_ns == mtime
    # --no-resume: payload NOT reused, checkpoints discarded, fresh fit from zero
    assert prd.main(argv + ["--no-resume"]) == 0
    o = capsys.readouterr().out
    assert "REUSING" not in o and "RESUMING" not in o
    assert o.count("--no-resume: discarded the existing checkpoint dir") == 2
    res = json.loads(out.read_text())
    assert res["controls"]["R_mix_empty"]["optimizer_steps_run"] == steps
    assert res["controls"]["R_mix_empty"]["heldout_curve"] == \
        ref["controls"]["R_mix_empty"]["heldout_curve"]          # same fit, from zero
    assert ck_a.exists() and ck_a.stat().st_mtime_ns > mtime  # freshly written
    # checkpoint_spec's fresh flag is the mechanism; off means untouched
    spec = prd.checkpoint_spec(1, out.parent, "lrmult1/R_mix_empty", "k", fresh=False)
    assert Path(spec["checkpoint_dir"]).exists()
    prd.checkpoint_spec(1, out.parent, "lrmult1/R_mix_empty", "k", fresh=True)
    assert not Path(spec["checkpoint_dir"]).exists()
    assert "discarded" in capsys.readouterr().out
    src = (REPO / "scripts/sdfbase/psr_residual_distill.py").read_text()
    assert src.count("fresh=not resume") == 1 and src.count("fresh=args.no_resume") == 2


# -- capture budget (the full − full pair; user scope directive 2026-09-04) ---


def test_capture_budget_subsamples_each_pool_deterministically(bank, capsys):
    """``--capture-budget-tokens`` draws a seeded per-source subsample of that
    many official tokens (the doc crossing the budget is kept), records it
    in the manifest (and so in the calibration identity's capture sha), and
    leaves the whole-pool manifest byte-identical when absent; a budget the
    pool cannot meet, a non-positive one, and a doc without official_tokens
    refuse loudly."""
    from sisteer.extraction.residual import capture_manifest_text, load_capture_docs

    pools = bank / "pools"
    whole = bank / "cm_whole.json"
    assert prd.main(["capture-set", *_common(bank), "--out", str(whole)]) == 0
    m_whole = json.loads(whole.read_text())
    assert "capture_subsample" not in m_whole
    # the rows carry official_tokens 10..16 (mix, 7 docs = 91) / 10..15
    # (ctl, 6 docs = 75); a 30-token budget selects a strict subset of each
    out = bank / "cm_sub.json"
    assert prd.main(["capture-set", *_common(bank), "--out", str(out),
                     "--capture-budget-tokens", "30"]) == 0
    m = json.loads(out.read_text())
    sub = m["capture_subsample"]
    assert sub["budget_tokens"] == 30 and sub["rule"] == prd.CAPTURE_SUBSAMPLE_RULE
    assert sub["min_stratum_docs"] == prd.CAPTURE_MIN_STRATUM_DOCS == 2
    # ONE share table for both sources: the mix pool's realized token shares
    mix_docs = load_capture_docs(pools / f"pairs_{MIX_SI}.jsonl", "mix")
    shares = prd.family_token_shares(mix_docs)
    assert sub["family_shares"] == shares and set(shares) == {"advice", "stories"}
    assert abs(sum(shares.values()) - 1) < 1e-9
    for src, si, n_pool in (("mix", MIX_SI, len(_MIX_ROWS)), ("ctl", CTL_SI, len(_CTL_ROWS))):
        r = sub["per_source"][src]
        assert r["si"] == si and r["n_pool"] == n_pool
        assert 0 < r["n_selected"] < n_pool
        docs = load_capture_docs(pools / f"pairs_{si}.jsonl", src)
        tok = {d.doc_id: d.official_tokens for d in docs}
        fam = {d.doc_id: d.family for d in docs}
        chosen = [d["doc_id"] for d in m["docs"] if d["source"] == src]
        assert len(chosen) == r["n_selected"]
        assert sum(tok[i] for i in chosen) == r["official_tokens_selected"]
        for f, fr in r["per_family"].items():
            mine = [i for i in chosen if fam[i] == f]
            assert fr["quota_tokens"] == int(round(30 * shares[f]))
            assert len(mine) == fr["n_selected"] >= 2                 # a holdout exists
            assert sum(tok[i] for i in mine) == fr["official_tokens_selected"] >= fr["quota_tokens"]
            assert fr["seed_key"] == f"0|capture|{si}|{f}"
    assert m["n_train"] + m["n_holdout"] == sum(
        r["n_selected"] for r in sub["per_source"].values())
    assert all(v["n_holdout"] >= 1 for v in m["strata"].values())
    assert set(m["sources"]) == {"mix", "ctl"}
    out2 = bank / "cm_sub2.json"
    assert prd.main(["capture-set", *_common(bank), "--out", str(out2),
                     "--capture-budget-tokens", "30"]) == 0
    assert out2.read_bytes() == out.read_bytes()          # deterministic
    out3 = bank / "cm_sub3.json"
    assert prd.main(["capture-set", *_common(bank), "--out", str(out3),
                     "--capture-budget-tokens", "30", "--seed", "1"]) == 0
    assert json.loads(out3.read_text())["docs"] != m["docs"]   # seed moves the draw
    # the manifest text IS the identity's capture sha input: it differs from
    # the whole-pool one, and the whole-pool one is unchanged by the option
    docs_all, train, hold, manifest, sources = prd.build_capture_set(
        pools, MIX_SI, CTL_SI, frac=0.1, seed=0)
    assert capture_manifest_text(manifest, sources=sources) == whole.read_text()
    printed = capsys.readouterr().out
    assert "capture subsample: budget 30 official tokens per source" in printed
    # refusals
    for bad, msg in (("0", "must be > 0"), ("10000", "below the quota")):
        with pytest.raises(prd.ResidualError, match=msg):
            prd.build_capture_set(pools, MIX_SI, CTL_SI, frac=0.1, seed=0,
                                  budget_tokens=int(bad))
    rows = [json.loads(l) for l in (pools / f"pairs_{CTL_SI}.jsonl").read_text().splitlines()]
    rows[0].pop("official_tokens")
    bad_pool = bank / "pools_bad"
    bad_pool.mkdir()
    (bad_pool / f"pairs_{MIX_SI}.jsonl").write_bytes((pools / f"pairs_{MIX_SI}.jsonl").read_bytes())
    (bad_pool / f"pairs_{CTL_SI}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(prd.ResidualError, match="records no official_tokens"):
        prd.build_capture_set(bad_pool, MIX_SI, CTL_SI, frac=0.1, seed=0,
                              budget_tokens=30)


def test_calibration_identity_names_the_budget_only_when_set(bank, tiny_llama):
    """The identity's ``capture`` block gains ``budget_tokens`` only under a
    budget: every whole-pool identity (the 100k pair's) keeps its
    fingerprint byte for byte, and a budgeted run can never pass as an
    unbudgeted calibration (or vice versa)."""
    import types

    model, _tok = tiny_llama
    pools = bank / "pools"
    root = bank / "out"

    def ident(budget):
        args = types.SimpleNamespace(
            model=TINY, revision=None, model_key=KEY, holdout_frac=0.1, seed=0,
            n_boot=10, capture_budget_tokens=budget)
        _d, _tr, _ho, manifest, sources = prd.build_capture_set(
            pools, MIX_SI, CTL_SI, frac=0.1, seed=0, budget_tokens=budget)
        _p, pos = prd.load_parent_record(root, KEY, MIX_SI)
        _n, neg = prd.load_parent_record(root, KEY, CTL_SI)
        hp = {k: 1 for k in prd.TRAINER_HP_KEYS}
        return prd.calibration_identity(args, hp, model, pos_prov=pos, neg_prov=neg,
                                        manifest=manifest, sources=sources,
                                        layers=[0, 1])

    a, b, c = ident(None), ident(30), ident(None)
    assert "budget_tokens" not in a["capture"] and a["fingerprint"] == c["fingerprint"]
    assert "family_shares" not in a["capture"] and "min_stratum_docs" not in a["capture"]
    assert b["capture"]["budget_tokens"] == 30 and b["fingerprint"] != a["fingerprint"]
    assert set(b["capture"]["family_shares"]) == {"advice", "stories"}
    assert b["capture"]["min_stratum_docs"] == 2
    assert b["capture"]["manifest_sha256"] != a["capture"]["manifest_sha256"]
    # a namespace WITHOUT the attribute (older callers) reads as unbudgeted
    legacy = types.SimpleNamespace(model=TINY, revision=None, model_key=KEY,
                                   holdout_frac=0.1, seed=0, n_boot=10)
    _d, _tr, _ho, manifest, sources = prd.build_capture_set(pools, MIX_SI, CTL_SI,
                                                            frac=0.1, seed=0)
    _p, pos = prd.load_parent_record(root, KEY, MIX_SI)
    _n, neg = prd.load_parent_record(root, KEY, CTL_SI)
    d = prd.calibration_identity(legacy, {k: 1 for k in prd.TRAINER_HP_KEYS}, model,
                                 pos_prov=pos, neg_prov=neg, manifest=manifest,
                                 sources=sources, layers=[0, 1])
    assert d["fingerprint"] == a["fingerprint"]


def test_verify_v_refuses_a_same_id_v_of_another_control(bank, tiny_llama, capsys):
    """V's id names the corpus cell only; with the control named, verify-v
    refuses a V whose sidecar names another control as parent_neg (the
    compose stage then rebuilds instead of reusing a foreign teacher)."""
    from residual_fixtures import random_psr

    model, _tok = tiny_llama
    root, pools = bank / "out", bank / "pools"
    v_path = prd.compose_v(root, root / "derived", KEY, MIX_SI, CTL_SI, pools_dir=pools)
    assert prd.verify_v(root, root / "derived", KEY, MIX_SI) == []
    assert prd.verify_v(root, root / "derived", KEY, MIX_SI, CTL_SI) == []
    other = "constv2ctl2-full"
    _stage_cell(root, KEY, other, random_psr(model, 33))
    _write_pool(pools, other, _CTL_ROWS, "ctl2")
    probs = prd.verify_v(root, root / "derived", KEY, MIX_SI, other)
    assert len(probs) == 1 and "parent_neg.cell_id" in probs[0] and other in probs[0]
    # through the CLI: exit 2 (REFUSED) naming the mismatch; the pair's own
    # control re-admits
    assert prd.main(["verify-v", *_common(bank), "--ctl-si", other]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "parent_neg.cell_id" in err and other in err
    assert prd.main(["verify-v", *_common(bank), "--ctl-si", CTL_SI]) == 0
    assert v_path.is_file()
    # compose-v REFUSES to overwrite the other pair's V under the same id
    sha_before = prd.sha256_file(v_path)
    with pytest.raises(prd.ResidualError, match="another pair"):
        prd.compose_v(root, root / "derived", KEY, MIX_SI, other, pools_dir=pools)
    assert prd.sha256_file(v_path) == sha_before
    assert prd.main(["compose-v", *_common(bank), "--ctl-si", other]) == 2
    assert "another pair" in capsys.readouterr().err


def test_seed_replicate_keys_are_admitted_model_keys():
    """The three batch-64 seed-replicate keys are admitted --model-key
    values (the per-seed +mix -ctl residual distillation from SAME-seed
    parents); the existing registry order — pinned by the test on the
    [:3] prefix — is untouched, and the CLI's choices are derived from
    the one registry."""
    keys = ("qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
            "qwen35_4b_base_eb64_s3")
    assert prd.MODEL_KEYS[-3:] == keys
    assert prd.MODEL_KEYS[:3] == ("gemma4e4b_base", "qwen35_4b_base",
                                  "nemotron3nano4b_doc")
    assert len(prd.MODEL_KEYS) == len(set(prd.MODEL_KEYS))
    src = (REPO / "scripts/sdfbase/psr_residual_distill.py").read_text()
    assert 'choices=list(MODEL_KEYS) + ["tinybase"]' in src
