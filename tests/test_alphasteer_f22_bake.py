"""alphasteer-full22 Phase C — the lambda-ladder BAKE and its serving configs / manifests.

Pins:
  1. the lambda grid arithmetic (plan C.1) from a synthetic fit_meta — lambda_1 = floor / min_l
     rho_model_dtype.p05, doubling per rung; the argmin layer; the refusals (per_layer keys not the
     layers, a parent not fit at base_scale 1.0); and the three live targets' verified lambda_1
     values from their recorded minimum p05 (numbers only — never real paths or files).
  2. the bake over a SYNTHETIC parent: six rung dirs, each holding object.pt + fit_provenance.json (the
     ONE identity sidecar — train_budget.object_identity_kind says so, fit_provenance_problems is
     empty) + meta.json + dose_scaled_from.json + staged_from.json, a VIEW_MANIFEST.json per view dir,
     matrices bit-identical to the parent's, base_scale == lambda_i, the served coeff == lambda_i, a
     distinct intervention_hash per rung, and the REAL explicit-cell registry building over the rung's
     view dir; `--check` green; a re-run KEEPS; a missing object is a clean exit 3 (also mid multi-target).
  3. the verification catching a corrupted rung — at bake time (a tampered writer refuses, no record) and
     at check time (bytes changed after the bake refuse, naming the rung).
  4. the generated configs / manifests: byte-identity (`--check`), the view table against the roster, the
     config differing from the wave's in exactly the named keys, the manifest's serving pins byte-equal to
     the wave's (the --limit-mm-per-prompt text-only pin where the wave carries it), serving_impl_commit
     the UNFROZEN sentinel and never the production sha, and mt_serving_up.py's own two-way cell-set rule
     after registry_adapter.traintest_cell_id.
  5. the standup note, pinned: lens_gate_standup.gate_config over an explicit-cells config yields a config
     the explicit registry refuses, and the substitute configs/two_si_<key>.yaml pin the same model.
Synthetic data only (placeholder shas, tmp paths).
"""
from __future__ import annotations

import io
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "alphasteer_f22"))

import as22_targets as tg  # noqa: E402
import bake_rungs as br  # noqa: E402
import make_ladder_configs as gen  # noqa: E402
from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.eval import defense_cell as tt  # noqa: E402
from sisteer.eval.train_budget import fit_provenance_problems, object_identity_kind  # noqa: E402

ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
LIVE = ("gemma4e4b", "qwen35_4b", "glm46vflash")

# --------------------------------------------------------------------------- synthetic parent

D = 8
LAYERS = (3, 5)
P05 = {3: 0.012, 5: 0.03}          # -> lambda_1 = 3e-3 / 0.012 = 0.25 at layer 3


def _fit_meta(layers=LAYERS, p05=P05, *, base_scale=1.0, sha="0" * 64, d=D):
    per_layer = {str(l): {"hidden_dim": d, "model_dtype": "bfloat16",
                          "rho_fp32": {"p05": p05[l] * 1.001, "p50": p05[l] * 5, "p95": p05[l] * 8},
                          "rho_model_dtype": {"p05": p05[l], "p50": p05[l] * 5, "p95": p05[l] * 8}}
                 for l in layers}
    return {"schema": "alphasteer_fit_meta/1", "arm": "alphasteer", "cell_id": "alphasteer", "class": "B",
            "layers": list(layers), "chosen": {"layers": list(layers), "n_layers": len(layers)},
            "per_layer": per_layer, "base_scale": base_scale,
            "base_scale_source": "uncalibrated_fit (strength set by the caller, e.g. the served lambda ladder)",
            "site": "pre", "steer_positions": "prefill", "sha256": sha,
            "model_id": "org/synthetic-model", "model_revision": "a" * 40,
            "chat_template_hash": "0123456789abcdef", "code_sha": "b" * 40,
            "screen_f9_sha256": "c" * 64, "split_sha256": "d" * 64, "leakage_report_sha256": "e" * 64,
            "fit_mode": "content_si_agnostic_multilayer"}


def make_parent(root: Path, target: str = "gemma4e4b", *, layers=LAYERS, d=D, seed=0) -> Path:
    pdir = tg.parent_cell_dir(root, target)
    pdir.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    blob = {"kind": "alphasteer", "layer": None, "M": None, "refusal": None, "layers": list(layers),
            "Ms": {l: torch.randn(d, d, generator=g) for l in layers}, "refusals": None,
            "steer_positions": "prefill", "site": "pre", "base_scale": 1.0,
            "metadata": {"fit": "synthetic", "lambda_reg": 10.0}}
    torch.save(blob, pdir / "object.pt")
    # the fit's publication: fit_meta.sha256 is the RECORD's fingerprint (write_fingerprinted_json), never
    # the object's hash, and the .COMPLETE marker repeats it — the shape the real Phase B records have
    fm = _fit_meta(layers, d=d)
    fm.pop("sha256")
    fm["sha256"] = br.fit_meta_fingerprint(fm)
    (pdir / "fit_meta.json").write_text(json.dumps(fm, sort_keys=True, indent=1))
    (pdir / "fit_meta.json.COMPLETE").write_text(fm["sha256"] + "\n")
    (pdir / "fit_provenance.json").write_text(json.dumps({
        "resolved_model_id": "org/synthetic-model", "resolved_revision": "a" * 40,
        "chat_template_hash": "0123456789abcdef", "writer": "fit_class_b.py@synthetic",
        "positions_at_fit": "prefill"}, indent=1))
    (pdir / "meta.json").write_text(json.dumps({
        "arm": "alphasteer", "cell_id": "alphasteer", "class": "B", "si_id": None, "layer": 15,
        "layers": list(layers), "sidecars": ["fit_meta.json", "fit_provenance.json"]}, indent=1))
    return pdir


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "objects"
    r.mkdir()
    make_parent(r, "gemma4e4b")
    return r


def _run(*argv):
    return br.main(list(argv))


# --------------------------------------------------------------------------- 1: the grid

def test_lambda_grid_arithmetic():
    g = br.lambda_grid(_fit_meta())
    assert g["argmin_layer"] == 3 and g["min_rho_p05_model_dtype_at_unit"] == 0.012
    assert g["lambda_1"] == pytest.approx(0.25) and g["floor"] == 3e-3 and g["n_rungs"] == 6
    assert g["lambdas"] == pytest.approx([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    assert g["lambdas"][0] * 0.012 == pytest.approx(3e-3)               # rung 1 sits ON the floor
    assert br.lambda_grid(_fit_meta(), n_rungs=3)["lambdas"] == pytest.approx([0.25, 0.5, 1.0])
    f32, md = br.fit_rho_at(_fit_meta(), 2.0)
    assert md["3"] == {"p05": pytest.approx(0.024), "p50": pytest.approx(0.12), "p95": pytest.approx(0.192)}
    assert f32["5"]["p05"] == pytest.approx(2.0 * 0.03 * 1.001)


def test_lambda_grid_matches_the_verified_live_values():
    # the three live targets' recorded min_l rho_model_dtype.p05 (numbers only), and the lambda_1 the
    # coordinating session verified from them: 0.2357 / 0.2697 / 1.3937
    for p05, want in ((0.01272731418721378, 0.2357), (0.011121721006929874, 0.2697),
                      (0.002152471721637994, 1.3937)):
        g = br.lambda_grid(_fit_meta(p05={3: p05, 5: p05 * 3}))
        assert round(g["lambda_1"], 4) == want                      # 4 decimal places, as verified
        assert g["lambda_1_4sf"] == float(f"{want:.4g}")             # the record rounds to 4 sig figs
        assert g["lambdas"][-1] == pytest.approx(want * 32, rel=1e-3)


def test_lambda_grid_refusals():
    fm = _fit_meta()
    fm["per_layer"].pop("5")
    with pytest.raises(br.Refusal, match="per_layer keys"):
        br.lambda_grid(fm)
    with pytest.raises(br.Refusal, match="base_scale"):
        br.lambda_grid(_fit_meta(base_scale=0.5))
    fm = _fit_meta()
    fm["schema"] = "something_else/1"
    with pytest.raises(br.Refusal, match="schema"):
        br.lambda_grid(fm)
    fm = _fit_meta()
    fm["per_layer"]["3"]["rho_model_dtype"]["p05"] = 0.0
    with pytest.raises(br.Refusal, match="not > 0"):
        br.lambda_grid(fm)


# --------------------------------------------------------------------------- 2: the bake

def test_bake_lays_out_six_rungs_with_one_identity_sidecar(root, capsys):
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 0
    rec = json.loads((tg.ladder_dir(root, "gemma4e4b") / "ladder_bake.json").read_text())
    assert rec["schema"] == "alphasteer_f22_ladder_bake/1" and rec["view"] == "it"
    assert rec["mechanism"]["name"] == "scalar_base_scale" and rec["grid"]["lambdas"] == pytest.approx([0.25, 0.5, 1, 2, 4, 8])
    assert [r["i"] for r in rec["rungs"]] == [1, 2, 3, 4, 5, 6]
    parent_blob = torch.load(tg.parent_cell_dir(root, "gemma4e4b") / "object.pt", weights_only=True)
    parent_sha = br.sha256_file(tg.parent_cell_dir(root, "gemma4e4b") / "object.pt")
    hashes = set()
    for row, lam in zip(rec["rungs"], rec["grid"]["lambdas"]):
        cdir = tg.rung_cell_dir(root, "gemma4e4b", row["i"])
        assert Path(row["cell_dir"]) == cdir and Path(row["defenses_dir"]) == cdir.parent
        assert cdir.parent.name == "it" and cdir.parent.parent.name == "cells_view" and cdir.parent.parent.parent.name == f"lam{row['i']}"
        assert sorted(p.name for p in cdir.iterdir()) == ["dose_scaled_from.json", "fit_provenance.json", "meta.json",
                                                          "object.pt", "staged_from.json"]
        assert sorted(p.name for p in cdir.parent.iterdir()) == ["VIEW_MANIFEST.json", "alphasteer"]
        # exactly ONE identity sidecar, and the shared verdicts accept the dir as an alphasteer cell
        assert object_identity_kind(cdir / "object.pt") == "fit_provenance"
        assert fit_provenance_problems(cdir / "object.pt", "alphasteer") == []
        assert (cdir / "fit_provenance.json").read_bytes() == (tg.parent_cell_dir(root, "gemma4e4b") / "fit_provenance.json").read_bytes()
        blob = torch.load(cdir / "object.pt", weights_only=True)
        assert list(blob) == list(parent_blob)                       # same keys, same order
        assert blob["base_scale"] == lam and type(blob["base_scale"]) is float
        for l in LAYERS:
            assert torch.equal(blob["Ms"][l], parent_blob["Ms"][l]) and blob["Ms"][l].dtype == torch.float32
        assert row["object_sha256"] == br.sha256_file(cdir / "object.pt") != parent_sha
        assert row["parent_object_sha256"] == parent_sha
        assert row["files"] == {p.name: br.sha256_file(p) for p in cdir.iterdir()}
        # the served translation: coeff == lambda, hook layer == fit layer - 1, matrices identical
        ops = la.get_adapter("alphasteer")(br.resolved_spec_of(blob), (cdir / "object.pt").read_bytes(), b"")
        assert [op.layer for op in ops] == [l - 1 for l in LAYERS]
        assert all(op.coeff == lam and op.op == la.OP_PREFILL_LAST_MATMUL and op.phase == la.PHASE_PREFILL for op in ops)
        hashes.add(row["intervention_hash"])
        # the proposal column: lambda x the lambda=1 fit-side rho
        assert row["rho_fit_model_dtype_by_layer"]["3"]["p05"] == pytest.approx(lam * 0.012)
        assert row["rho_fit_min_p05_model_dtype"] == pytest.approx(lam * 0.012) and row["rho_fit_argmin_layer_model_dtype"] == 3
        assert {c["check"]: c["status"] for c in row["verification"]["checks"]} == {
            "tensors": "PASS", "scalar": "PASS", "served_ops": "PASS", "served_edit": "PASS"}
        dose = json.loads((cdir / "dose_scaled_from.json").read_text())
        assert dose["lambda"] == lam and dose["object_sha256"] == row["object_sha256"] and dose["tensors_scaled"] == []
        staged = json.loads((cdir / "staged_from.json").read_text())
        assert staged["object_sha256"] == row["object_sha256"] and staged["recorded_hash_check"]["status"] == "match"
        assert set(staged["staged_shas"]) == {"dose_scaled_from.json", "fit_provenance.json", "meta.json", "object.pt"}
        vm = json.loads((cdir.parent / "VIEW_MANIFEST.json").read_text())
        assert vm["entries"][0]["cell_id"] == "alphasteer" and vm["entries"][0]["object_sha256"] == row["object_sha256"]
        # the REAL explicit-cell registry builds over the rung's view dir (what the standup does)
        reg = tt.registry_from_explicit_cells({"explicit_cells": ["none", "alphasteer"], "defenses_dir": str(cdir.parent)},
                                              repo_root=REPO)
        assert [c.id for c in reg] == ["none", "alphasteer"] and reg[1].class_ == "B" and Path(reg[1].object_path) == cdir / "object.pt"
    assert len(hashes) == 6                                          # every rung is its own intervention
    assert rec["rungs"][2]["lambda"] == 1.0                          # the lambda=1 rung differs from the parent by nothing but...
    assert rec["rungs"][2]["object_sha256"] != parent_sha            # ...its bytes are still re-saved (a distinct file)
    # --check re-verifies without writing
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 0
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()} == before
    out = capsys.readouterr().out
    assert "lambda_1 = 0.003 / rho_p05[3]=0.012 = 0.25" in out and "outcomes: gemma4e4b=VERIFIED; exit 0" in out


def test_rebake_without_force_keeps_and_force_rebakes(root, capsys):
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 0
    rec1 = json.loads((tg.ladder_dir(root, "gemma4e4b") / "ladder_bake.json").read_text())
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 0
    assert "KEPT" in capsys.readouterr().out
    rec2 = json.loads((tg.ladder_dir(root, "gemma4e4b") / "ladder_bake.json").read_text())
    assert [r["object_sha256"] for r in rec1["rungs"]] == [r["object_sha256"] for r in rec2["rungs"]]
    assert _run("--target", "gemma4e4b", "--root", str(root), "--force") == 0
    assert "BAKED" in capsys.readouterr().out
    # an INTERRUPTED bake (the .BAKING marker still down, no dose_scaled_from.json yet) is ours: it is
    # not kept, --check refuses it by name, and the next bake rebuilds it without --force
    rdir2, cdir2 = tg.rung_dir(root, "gemma4e4b", 2), tg.rung_cell_dir(root, "gemma4e4b", 2)
    (cdir2 / "dose_scaled_from.json").unlink()
    (rdir2 / ".BAKING").write_text("interrupted\n")
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1
    assert "lam2" in capsys.readouterr().err and ".BAKING" in (rdir2 / ".BAKING").name
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 0
    assert not (rdir2 / ".BAKING").exists() and (cdir2 / "dose_scaled_from.json").is_file()
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 0
    # a stranger's object.pt in a rung dir (no dose_scaled_from.json, no .BAKING) is never removed
    (cdir2 / "dose_scaled_from.json").unlink()
    assert _run("--target", "gemma4e4b", "--root", str(root), "--force") == 1
    assert "not a previous bake of this producer" in capsys.readouterr().err
    assert (cdir2 / "object.pt").is_file()


def test_missing_object_is_a_clean_skip_not_a_traceback(root, capsys):
    assert _run("--target", "nemotron3nano4b", "--root", str(root)) == 3
    out = capsys.readouterr()
    assert "SKIP nemotron3nano4b: no fitted object" in out.out and "Traceback" not in out.err
    assert not tg.ladder_dir(root, "nemotron3nano4b").exists()
    # multi-target: the present target bakes, the missing one skips, exit 3 (never 1)
    assert _run("--target", "nemotron3nano4b,gemma4e4b", "--root", str(root)) == 3
    out = capsys.readouterr().out
    assert "outcomes: nemotron3nano4b=SKIPPED_NO_OBJECT, gemma4e4b=BAKED; exit 3" in out
    assert (tg.ladder_dir(root, "gemma4e4b") / "ladder_bake.json").is_file()
    # --check across targets: the missing one still skips cleanly
    assert _run("--target", "gemma4e4b", "--target", "nemotron3nano4b", "--root", str(root), "--check") == 3
    assert "gemma4e4b=VERIFIED, nemotron3nano4b=SKIPPED_NO_OBJECT" in capsys.readouterr().out
    # THE SEAM ladder.sbatch STANDS ON: exit 3 is fatal there (a single-target job with nothing to bake has nothing
    # to serve), so a SINGLE-target invocation must never see another target's absence. nemotron's object is absent
    # in this root and gemma's ladder still verifies clean — a missing nemotron object cannot fail a gemma ladder.
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 0
    assert "outcomes: gemma4e4b=VERIFIED; exit 0" in capsys.readouterr().out
    # an object WITHOUT its fit record is a torn publication: a refusal, not a skip
    pdir = tg.parent_cell_dir(root, "qwen35_4b")
    pdir.mkdir(parents=True)
    (pdir / "object.pt").write_bytes(b"x")
    assert _run("--target", "qwen35_4b", "--root", str(root)) == 1
    assert "torn fit publication" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="no view assignment"):
        _run("--target", "llama8b", "--root", str(root))


def test_parent_identity_refusals(root, capsys):
    pdir = tg.parent_cell_dir(root, "gemma4e4b")
    fm = json.loads((pdir / "fit_meta.json").read_text())
    fm["sha256"] = "f" * 64
    (pdir / "fit_meta.json").write_text(json.dumps(fm))
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 1
    assert "fit_meta.sha256" in capsys.readouterr().err
    fm["sha256"] = br.fit_meta_fingerprint(fm)                    # the writer's convention restores validity...
    (pdir / "fit_meta.json").write_text(json.dumps(fm))           # ...regardless of the file's own formatting
    (pdir / "fit_meta.json.COMPLETE").write_text(fm["sha256"] + "\n")
    (pdir / "train_budget_meta.json").write_text("{}")             # a second identity sidecar
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 1
    assert "exactly one identity" in capsys.readouterr().err


def test_fit_meta_sha256_is_the_records_fingerprint_never_the_object_hash(root, capsys):
    """Jobs 84542-84544 (2026-09-09) died in their first two minutes on every target because the bake read
    fit_meta.sha256 as the object's hash. It is write_fingerprinted_json's fingerprint of the record body
    (sha256 over json.dumps(body minus sha256, sort_keys=True, indent=1)), repeated in .COMPLETE; verified
    against the three real Phase B records. The exact stale reading is now a refusal that names the fingerprint,
    the marker is REQUIRED, and a marker carrying the object hash is torn."""
    pdir = tg.parent_cell_dir(root, "gemma4e4b")
    fm = json.loads((pdir / "fit_meta.json").read_text())
    body = {k: v for k, v in fm.items() if k != "sha256"}
    assert fm["sha256"] == br.fit_meta_fingerprint(fm) == hashlib.sha256(
        json.dumps(body, sort_keys=True, indent=1).encode()).hexdigest()
    assert fm["sha256"] != br.sha256_file(pdir / "object.pt")
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1     # nothing baked yet: not verified
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 0                # ...and the real shape bakes
    # the reading that killed the first submission: sha256 == object.pt's hash
    obj_sha = br.sha256_file(pdir / "object.pt")
    fm["sha256"] = obj_sha
    (pdir / "fit_meta.json").write_text(json.dumps(fm, sort_keys=True, indent=1))
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1
    err = capsys.readouterr().err
    assert "fit_meta.sha256" in err and "fingerprint" in err and "stale object/record pair" not in err
    fm["sha256"] = br.fit_meta_fingerprint(fm)
    (pdir / "fit_meta.json").write_text(json.dumps(fm, sort_keys=True, indent=1))
    (pdir / "fit_meta.json.COMPLETE").write_text(obj_sha + "\n")                 # marker = object hash: torn
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1
    assert "torn publication" in capsys.readouterr().err
    (pdir / "fit_meta.json.COMPLETE").unlink()                                     # no marker: incomplete
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1
    assert "incomplete publication" in capsys.readouterr().err
    (pdir / "fit_meta.json.COMPLETE").write_text(fm["sha256"] + "\n")
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 0


# --------------------------------------------------------------------------- 3: corruption

def test_bake_verification_catches_a_corrupted_rung_at_bake_time(root, monkeypatch, capsys):
    real = br.scaled_blob

    def tampered(parent_blob, lam):
        blob = real(parent_blob, lam)
        if lam == 0.5:                                    # rung 2: one matrix element nudged
            Ms = dict(blob["Ms"])
            M = Ms[5].clone()
            M[0, 0] = M[0, 0] * 1.0001
            Ms[5] = M
            blob["Ms"] = Ms
        return blob

    monkeypatch.setattr(br, "scaled_blob", tampered)
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 1
    err = capsys.readouterr().err
    assert "λ=0.5: Ms[5] is not bit-identical" in err
    assert not (tg.ladder_dir(root, "gemma4e4b") / "ladder_bake.json").exists()   # no record on a refusal

    def wrong_scalar(parent_blob, lam):
        return real(parent_blob, lam * 1.5 if lam == 2.0 else lam)

    monkeypatch.setattr(br, "scaled_blob", wrong_scalar)
    assert _run("--target", "gemma4e4b", "--root", str(root), "--force") == 1
    assert "rung base_scale 3.0 != λ 2.0" in capsys.readouterr().err


def test_check_catches_bytes_changed_after_the_bake(root, capsys):
    assert _run("--target", "gemma4e4b", "--root", str(root)) == 0
    cdir = tg.rung_cell_dir(root, "gemma4e4b", 4)
    blob = torch.load(cdir / "object.pt", weights_only=True)
    blob["Ms"][3] = blob["Ms"][3] * 2.0                    # a matrix quietly pre-multiplied
    torch.save(blob, cdir / "object.pt")
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1
    err = capsys.readouterr().err
    assert "gemma4e4b lam4" in err and "bytes changed after the bake" in err
    # and the tensor-level check itself, when the sidecar is made to agree with the new bytes
    dose = json.loads((cdir / "dose_scaled_from.json").read_text())
    dose["object_sha256"] = br.sha256_file(cdir / "object.pt")
    (cdir / "dose_scaled_from.json").write_text(json.dumps(dose))
    staged = json.loads((cdir / "staged_from.json").read_text())
    staged["object_sha256"] = dose["object_sha256"]
    staged["staged_shas"]["object.pt"] = dose["object_sha256"]
    staged["staged_shas"]["dose_scaled_from.json"] = br.sha256_file(cdir / "dose_scaled_from.json")
    (cdir / "staged_from.json").write_text(json.dumps(staged))
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1
    assert "Ms[3] is not bit-identical" in capsys.readouterr().err
    # a second identity sidecar planted in a rung dir is refused by --check
    cdir1 = tg.rung_cell_dir(root, "gemma4e4b", 1)
    (cdir1 / "derived_meta.json").write_text("{}")
    assert _run("--target", "gemma4e4b", "--root", str(root), "--check") == 1
    assert "exactly one" in capsys.readouterr().err


def test_verify_rung_refuses_when_base_scale_would_not_reach_the_served_op(root, monkeypatch):
    """The mechanism finding is ASSERTED, not assumed: if the adapter ever stopped carrying base_scale
    into the op coeff, verify_rung refuses instead of the bake silently shipping unscaled rungs."""
    parent = br.Parent(root, "gemma4e4b")
    real = br.adapter_ops

    def coeff_dropped(spec, artifact):
        from dataclasses import replace
        return tuple(replace(op, coeff=1.0) for op in real(spec, artifact))

    monkeypatch.setattr(br, "adapter_ops", coeff_dropped)
    blob = br.scaled_blob(parent.blob, 0.5)
    with pytest.raises(br.Refusal, match="did not reach the served op"):
        br.verify_rung(parent, blob, br.blob_bytes(blob), 0.5, probe_rows=4, seed=0)


def test_served_edit_is_exactly_lambda_times_parent_through_apply_hook_ops(root):
    parent = br.Parent(root, "gemma4e4b")
    for lam in (0.25, 3.0):
        blob = br.scaled_blob(parent.blob, lam)
        rec = br.verify_rung(parent, blob, br.blob_bytes(blob), lam, probe_rows=5, seed=1)
        edit = next(c for c in rec["checks"] if c["check"] == "served_edit")
        assert edit["float32"]["exact"] and edit["bfloat16"]["exact"]
        ops = next(c for c in rec["checks"] if c["check"] == "served_ops")
        assert ops["coeff_rung"] == lam and ops["coeff_parent"] == 1.0 and ops["hook_layers"] == [2, 4]


# --------------------------------------------------------------------------- 4: configs + manifests

def test_view_table_matches_the_roster():
    # every roster target is either BAKED here or declared NOT baked here with its reason (as22_targets: llama8b's object
    # comes from the off-roster wave source, arditi_targets.OFF_ROSTER_WAVE_SOURCES; the lambda ladder still serves it at
    # view `it`), never missing. Note: both sides still list glm46vflash (descoped from the paper, still in the roster) —
    # should GLM leave the roster this fails HERE, on the table, which is where the decision has to be made.
    assert set(tg.TARGETS) | set(tg.ROSTER_TARGETS_NOT_BAKED_HERE) == set(ROSTER["targets"])
    assert not set(tg.TARGETS) & set(tg.ROSTER_TARGETS_NOT_BAKED_HERE)
    # llama8b's object comes from the off-roster wave source; the front-door target qwen35_4b_demo (the worked example's roster block,
    # c7e73cd04) is declared here too — its steering objects come from the front-door object banks, never the AlphaSteer bake
    assert tg.ROSTER_TARGETS_NOT_BAKED_HERE == ("llama8b", "qwen35_4b_demo")
    assert "qwen35_4b_demo" in ROSTER["targets"] and "alphasteer" not in str(ROSTER["targets"]["qwen35_4b_demo"]["views"])
    for target in tg.TARGETS:
        view = tg.view_of(target)
        v = ROSTER["targets"][target]["views"][view]
        assert v["config"] == tg.wave_config_rel(target) and v["manifest"] == tg.wave_manifest_rel(target)
        assert (REPO / v["config"]).is_file() and (REPO / v["manifest"]).is_file()
    assert tg.view_of("gemma4e4b") == tg.view_of("qwen35_4b") == "it"
    assert tg.view_of("glm46vflash") == tg.view_of("nemotron3nano4b") == "main"
    assert tg.rung_view_dir(Path("/r"), "gemma4e4b", 3) == Path("/r/gemma4e4b/ladder/lam3/cells_view/it")
    assert tg.rung_cell_dir(Path("/r"), "glm46vflash", 6) == Path("/r/glm46vflash/ladder/lam6/cells_view/main/alphasteer")


def test_configs_and_manifests_regenerate_byte_identically(capsys):
    assert gen.main(["--check"]) == 0, capsys.readouterr().out
    assert len(gen.outputs()) == 8 and all(p.is_file() for p, _, _ in gen.outputs())


def test_generator_refuses_the_production_serving_sha(monkeypatch):
    monkeypatch.setattr(gen, "LADDER_SERVING_IMPL_COMMIT", tg.PRODUCTION_SERVING_IMPL_COMMIT)
    with pytest.raises(SystemExit, match="PRODUCTION serving sha"):
        gen.render_manifest("gemma4e4b")
    monkeypatch.setattr(gen, "LADDER_SERVING_IMPL_COMMIT", "not-a-sha")
    with pytest.raises(SystemExit, match="neither UNFROZEN nor"):
        gen.render_manifest("gemma4e4b")
    monkeypatch.setattr(gen, "LADDER_SERVING_IMPL_COMMIT", "1" * 40)      # all digits: YAML reads an INT
    with pytest.raises(SystemExit, match="integer"):
        gen.render_manifest("gemma4e4b")
    wl = "fedcba9876" * 4
    monkeypatch.setattr(gen, "LADDER_SERVING_IMPL_COMMIT", wl)            # a cut W_L sha renders + loads
    text = gen.render_manifest("gemma4e4b")
    assert f"serving_impl_commit: {wl}" in text
    spec = cm.load_petri_spec(_write_tmp(text))
    assert spec.serving_impl_commit == wl
    # and the harness hash no longer refuses on that field (the remaining inputs are the wave's frozen pins;
    # the declared prompt override needs the REAL prompt bytes, resolved as lens_gate_standup resolves them)
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
    prompt_files = resolve_prompt_files(spec, override_root=REPO / "configs/capeval/petri/prompts")
    h = cm.petri_harness_config_hash(spec, behavior_files={k: b"# b\n" for k in cm.petri_expected_behavior_keys(spec)},
                                     prompt_files=prompt_files, serving_identity="proxy_vllm+lens+interleaved")
    assert len(h) == 16


def test_the_generator_constant_stays_unfrozen_because_a_tree_cannot_name_its_own_sha():
    """The ladder manifests record W_L's serving_impl_commit — and W_L's single commit touches
    lens_worker.py, so by the field's definition (the last commit touching the serving paths) that value IS W_L.
    A tree cannot contain its own sha, so the checked-in value is the loader's UNFROZEN literal and the honest value
    is filed per rung by lambda_ladder.serving_impl_commit_of from the standup's verified HEAD. Setting this constant
    to "W_L's sha" and regenerating would move W_L (the manifests are in W_A, W_L = W_A + one commit) — a change here
    must come with a different mechanism, deliberately."""
    assert gen.LADDER_SERVING_IMPL_COMMIT == cm.PETRI_UNFROZEN == "UNFROZEN"
    for target in tg.TARGETS:
        assert cm.load_petri_spec(REPO / tg.manifest_rel(target)).serving_impl_commit == cm.PETRI_UNFROZEN


def _write_tmp(text: str) -> Path:
    import tempfile
    p = Path(tempfile.mkdtemp()) / "m.yaml"
    p.write_text(text)
    return p


@pytest.mark.parametrize("target", tg.TARGETS)
def test_config_differs_from_the_wave_config_in_exactly_the_named_keys(target):
    ours = yaml.safe_load((REPO / tg.serve_config_rel(target)).read_text())
    wave = yaml.safe_load((REPO / tg.wave_config_rel(target)).read_text())
    assert ours["explicit_cells"] == ["none", "alphasteer"]
    view = tg.view_of(target)
    assert ours["defenses_dir"] == f"out/alphasteer_f22/{target}/cells_view/{view}"
    assert ours["out_dir"] == f"out/alphasteer_f22_{target}" and ours["private_run"] == f"alphasteer_f22_{target}"
    changed = {k for k in set(ours) | set(wave) if ours.get(k) != wave.get(k)}
    assert changed == {"explicit_cells", "defenses_dir", "out_dir", "private_run"}
    assert (ours["model"], ours["model_revision"]) == (ROSTER["targets"][target]["model"], ROSTER["targets"][target]["model_revision"])
    text = (REPO / tg.serve_config_rel(target)).read_text()
    assert "GENERATED by scripts/alphasteer_f22/make_ladder_configs.py" in text and "_serve_" in text.splitlines()[0]
    assert "configs/sdfbase_full22_" not in "\n".join(l for l in text.splitlines() if not l.startswith("#"))


@pytest.mark.parametrize("target", tg.TARGETS)
def test_manifest_serving_pins_are_byte_identical_to_the_wave_manifest(target):
    ours_raw = yaml.safe_load((REPO / tg.manifest_rel(target)).read_text())["petri"]
    wave_raw = yaml.safe_load((REPO / tg.wave_manifest_rel(target)).read_text())["petri"]
    changed = {k for k in set(ours_raw) | set(wave_raw) if ours_raw.get(k) != wave_raw.get(k)}
    assert changed == {"audit", "cells", "serving_impl_commit"}
    assert ours_raw["audit"] == f"alphasteer_f22_{target}" and ours_raw["cells"] == ["none", "alphasteer"]
    spec = cm.load_petri_spec(REPO / tg.manifest_rel(target))
    assert spec.serving_impl_commit == cm.PETRI_UNFROZEN
    assert not spec.serving_impl_commit.startswith(tg.PRODUCTION_SERVING_IMPL_COMMIT[:8])
    assert tg.LADDER_SERVING_IMPL_COMMIT_MARKER in (REPO / tg.manifest_rel(target)).read_text()
    assert spec.target_model == ROSTER["targets"][target]["model"] and spec.target_revision == ROSTER["targets"][target]["model_revision"]
    wave_spec = cm.load_petri_spec(REPO / tg.wave_manifest_rel(target))
    for f in ("target_model", "target_revision", "chat_template_hash", "tool_parser", "serving_packages",
              "serving_launch_flags", "scenario_bundle_hash", "behaviors", "auditor_model",
              "auditor_fallback_models", "judge_model", "scenarios_model", "prompt_provenance",
              "serving_backend", "serving_mode", "prompted_prefix_hashes", "bloom_options"):
        assert getattr(spec, f) == getattr(wave_spec, f), f
    if target in ("qwen35_4b", "glm46vflash"):
        i = spec.serving_launch_flags.index("--limit-mm-per-prompt")
        assert spec.serving_launch_flags[i + 1] == '{"image":0,"video":0}'
    else:
        assert "--limit-mm-per-prompt" not in spec.serving_launch_flags
    # the harness hash refuses the placeholder, so a consumer that hashes this manifest AS-IS (mt_serving_up, a
    # production leg) cannot serve on it. The ladder's OWN standup never reads the field — lens_gate_standup replaces
    # it before enumeration — and the driver files the sha that ran from the standup's verified HEAD
    # (test_as22_ladder_generate.TestServingImplCommit). The literal is UNFROZEN by NECESSITY, not by omission: a
    # tree cannot record its own sha, and on W_L serving_impl_commit == W_L (test_the_generator_constant_stays_unfrozen).
    with pytest.raises(cm.ManifestError, match="serving_impl_commit is UNFROZEN"):
        cm.petri_harness_config_hash(spec, behavior_files={k: b"# b\n" for k in cm.petri_expected_behavior_keys(spec)},
                                     prompt_files={"auditor": b"# a\n", "judge": b"# j\n"},
                                     serving_identity="proxy_vllm+lens+interleaved")


@pytest.mark.parametrize("target", tg.TARGETS)
def test_config_and_manifest_cell_sets_round_trip_like_mt_serving_up(target):
    """scripts/sdfbase/mt_serving_up.py's own rule (its main, 'manifest/config cell mismatch'): every
    manifest cell translates to one of the config's explicit cells and every explicit cell is served."""
    cfg = yaml.safe_load((REPO / tg.serve_config_rel(target)).read_text())
    spec = cm.load_petri_spec(REPO / tg.manifest_rel(target))
    assert (spec.target_model, spec.target_revision) == (cfg["model"], cfg["model_revision"])
    tt_of = {c: ra.traintest_cell_id(c) for c in spec.cells}
    want = list(cfg["explicit_cells"])
    missing = [c for c, t in tt_of.items() if t not in want]
    unserved = [c for c in want if c not in tt_of.values()]
    assert not missing and not unserved
    assert [ra.traintest_cell_id(c) for c in spec.cells] == want == ["none", "alphasteer"]
    assert [ra.harness_cell_id(c) for c in want] == spec.cells          # and back
    assert ra.traintest_cell_id("alphasteer") == ra.harness_cell_id("alphasteer") == "alphasteer"


def test_explicit_config_is_not_consumable_by_lens_gate_standup_gate_config():
    """The ladder's standup is scripts/lens_gate_standup.py. Its gate_config() ADDS
    class_a_arms / class_b_arms to the config it is handed, and the explicit registry refuses a config
    carrying both those keys and explicit_cells — so the ladder job must hand it a NON-explicit config
    (configs/two_si_<key>.yaml, whose model pins equal ours) or mt_serving_up.py must take ours.
    Pinned here so the incompatibility is a test failure, never a standup crash."""
    import lens_gate_standup as lgs

    for target in tg.TARGETS:
        ours = yaml.safe_load((REPO / tg.serve_config_rel(target)).read_text())
        gcfg, added = lgs.gate_config(ours, ["alphasteer"])
        assert "class_b_arms" in gcfg and "explicit_cells" in gcfg
        with pytest.raises(ValueError, match="mutually exclusive"):
            tt.registry_from_explicit_cells(gcfg, repo_root=REPO)
        two_si = yaml.safe_load((REPO / f"configs/two_si_{target}.yaml").read_text())
        assert "explicit_cells" not in two_si
        assert (two_si["model"], two_si["model_revision"]) == (ours["model"], ours["model_revision"])
        gcfg2, added2 = lgs.gate_config(two_si, ["alphasteer"])
        assert "alphasteer" in gcfg2["class_b_arms"]


# ---------------------------------------------------------------- the rebound identity (2026-09-13)
def test_a_parent_that_binds_its_own_sha_gets_its_rung_identity_REBOUND_not_copied(tmp_path):
    """The byte-copy contract assumes the parent's identity record self-identifies by NOTHING. When
    the parent came from a producer that DOES bind an object_sha256 (llama8b's slice producer), the
    copy makes every rung assert the PARENT's identity — which is what a downstream mirror check
    reads, and it refused a pinned row on exactly this (2026-09-13). Such a rung is rebound instead:
    its own sha at the top level, the parent's record preserved WHOLE (never edited — a patched
    provenance file is indistinguishable from a correct one), the stage-3 keys carried up."""
    root = tmp_path / "obj"
    pdir = make_parent(root, "gemma4e4b")
    fp = json.loads((pdir / "fit_provenance.json").read_text())
    parent_sha = br.sha256_file(pdir / "object.pt")
    fp["object_sha256"] = parent_sha                      # the slice producer's shape
    fp["parent_object_sha256"] = "c" * 64                 # ... which already names ITS parent
    (pdir / "fit_provenance.json").write_text(json.dumps(fp, indent=1))

    rc = br.main(["--target", "gemma4e4b", "--root", str(root)])
    assert rc == br.EXIT_OK, rc

    for i in (1, 6):
        cdir = tg.rung_cell_dir(root, "gemma4e4b", i)
        got = json.loads((cdir / "fit_provenance.json").read_text())
        child_sha = br.sha256_file(cdir / "object.pt")
        # bound to the CHILD, and not the byte copy any more
        assert got["object_sha256"] == child_sha != parent_sha
        assert (cdir / "fit_provenance.json").read_bytes() != (pdir / "fit_provenance.json").read_bytes()
        # the parent's record survives WHOLE, including ITS OWN parent link, which a flat
        # parent_object_sha256 field at the top level would have clobbered
        assert got["derived_from"]["record"] == fp
        assert got["derived_from"]["record"]["parent_object_sha256"] == "c" * 64
        assert got["derived_from"]["object_sha256"] == parent_sha
        # the stage-3 keys the bank view requires are at the top level
        for k in br.STAGE3_FIT_REQUIRED_KEYS:
            assert got.get(k), k
        assert br.PRODUCER in got["writer"]
        # and the dir still reads as exactly one alphasteer identity
        assert object_identity_kind(cdir / "object.pt") == "fit_provenance"
        assert fit_provenance_problems(cdir / "object.pt", "alphasteer") == []


def test_a_parent_that_binds_nothing_still_gets_the_byte_copy(tmp_path):
    """The unchanged path: 18 of the 24 staged rungs are this case and must not move."""
    root = tmp_path / "obj"
    pdir = make_parent(root, "gemma4e4b")
    assert "object_sha256" not in json.loads((pdir / "fit_provenance.json").read_text())
    assert br.main(["--target", "gemma4e4b", "--root", str(root)]) == br.EXIT_OK
    for i in (1, 6):
        cdir = tg.rung_cell_dir(root, "gemma4e4b", i)
        assert (cdir / "fit_provenance.json").read_bytes() == (pdir / "fit_provenance.json").read_bytes()


def test_a_KEPT_rung_with_a_misbound_identity_is_REPAIRED_by_the_producer(tmp_path):
    """`kept` re-verifies but never rewrites, so a rung baked BEFORE the rebind existed keeps
    asserting its parent's identity through every re-run — which is how six staged rungs stayed
    wrong while the producer reported PASS (2026-09-13). A keep now repairs a misbound record and
    says REPAIRED; the object is never touched, and the staging record is re-stamped."""
    root = tmp_path / "obj"
    pdir = make_parent(root, "gemma4e4b")
    assert br.main(["--target", "gemma4e4b", "--root", str(root)]) == br.EXIT_OK   # bake: byte copy

    # make the parent self-identify AFTER the bake: the rungs on disk now assert the parent's sha,
    # which is exactly the state the six llama rungs were in
    fp = json.loads((pdir / "fit_provenance.json").read_text())
    parent_sha = br.sha256_file(pdir / "object.pt")
    fp["object_sha256"] = parent_sha
    (pdir / "fit_provenance.json").write_text(json.dumps(fp, indent=1))
    cdir = tg.rung_cell_dir(root, "gemma4e4b", 1)
    (cdir / "fit_provenance.json").write_text(json.dumps(fp, indent=1))
    obj_before = br.sha256_file(cdir / "object.pt")

    rc = br.main(["--target", "gemma4e4b", "--root", str(root)])
    assert rc == br.EXIT_OK, rc

    got = json.loads((cdir / "fit_provenance.json").read_text())
    assert got["object_sha256"] == obj_before != parent_sha      # bound to the rung
    assert got["derived_from"]["object_sha256"] == parent_sha
    assert br.sha256_file(cdir / "object.pt") == obj_before      # the OBJECT was not touched
    st = json.loads((cdir / "staged_from.json").read_text())
    assert st["staged_shas"]["fit_provenance.json"] == br.sha256_file(cdir / "fit_provenance.json")
    assert st.get("identity_repaired_at")
