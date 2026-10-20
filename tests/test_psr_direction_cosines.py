"""Tests for scripts/psr_direction_cosines_analysis.py.

All synthetic, CPU-only, no network, no HF model downloads: the stub seam
replaces any real model, fixtures are written by the CANONICAL PSRObject
writer (obj.save), and tokenizer tests either run offline against the
local cache (skipif absent) or assert the offline hard-error path.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import psr_direction_cosines_analysis as pdc  # noqa: E402
import sisteer  # noqa: E402
from sisteer import site as SITE  # noqa: E402
from sisteer.extraction.psr import PSRObject  # noqa: E402

REAL_LLAMA8B = [
    SITE.expand_path(e["path"]) for e in pdc.DEFAULT_OBJECTS
    if e["model_key"] == "llama8b"
]
HAVE_REAL_BANK = all(p.is_file() for p in REAL_LLAMA8B)

LLAMA8B_TOK_SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub"
    / "models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots"
    / "d10aef7999a2b5ba950ab3974312feeedbfe0b77")


# --------------------------------------------------------------------------
# Fixtures: synthetic objects written by the canonical writer
# --------------------------------------------------------------------------

def make_object(root: Path, *, model_key="llama8b", si="safety_refusal",
                arm="psr_kl_all", layers=(0, 1, 2, 3), d=8,
                directions=None, gate_w=None, gate_b=None,
                leg="response", header_len=4, flavour="A",
                complete=True, revision="deadbeefdeadbeef",
                sidecar_si=None, sidecar_model=None, sidecar_arm=None,
                target_layers="auto", dirname=None, write_sidecar=True
                ) -> dict:
    """Build a synthetic object directory. The on-disk blob format comes
    from the canonical writer (PSRObject.save on the stub seam), never
    hand-assembly. Returns the manifest entry."""
    layers = list(layers)
    rng = np.random.default_rng(7)
    if directions is None:
        directions = {l: torch.tensor(rng.normal(size=d),
                                      dtype=torch.float32) for l in layers}
    if gate_w is None:
        gate_w = {l: torch.tensor(rng.normal(size=d), dtype=torch.float32)
                  for l in layers}
    if gate_b is None:
        gate_b = {l: torch.ones(1) for l in layers}
    stub = pdc._StubModel(d, max(layers) + 1)
    obj = PSRObject(stub, layers)
    obj.set_directions(directions)
    for l in layers:
        obj.gate_w[l].data = gate_w[l].detach().float().clone()
        obj.gate_b[l].data = gate_b[l].detach().float().clone().reshape(1)
    if leg != "all":
        obj.set_fit_positions(leg, header_len)
    obj_dir = root / (dirname if dirname is not None else f"{si}__{arm}")
    obj_dir.mkdir(parents=True, exist_ok=True)
    obj.save(obj_dir / "object.pt")

    sc_si = sidecar_si if sidecar_si is not None else si
    sc_model = (sidecar_model if sidecar_model is not None
                else pdc.MODEL_KEY_TO_HF.get(model_key, "unknown/unknown"))
    sc_arm = sidecar_arm if sidecar_arm is not None else arm
    sc_hlen = header_len if leg != "all" else None
    if write_sidecar:
        if flavour == "A":
            side = {"si_id": sc_si, "model_id": sc_model,
                    "model_revision": revision, "arm": sc_arm,
                    "positions": leg, "positions_at_fit": leg,
                    "response_header_len": sc_hlen,
                    "cell_id": f"{sc_si}__{sc_arm}",
                    "target_layers": (layers if target_layers == "auto"
                                      else target_layers),
                    "fit_record": {"floor_ratio": 0.1,
                                   "best_heldout_loss": 0.01,
                                   "epochs_run": 2}}
            (obj_dir / "train_budget_meta.json").write_text(json.dumps(side))
        else:
            side = {"si": sc_si, "model": sc_model,
                    "model_revision": revision, "arm": sc_arm,
                    "positions": leg, "response_header_len": sc_hlen,
                    "fit_fingerprint": "f" * 64, "lr": 1e-3, "clip": 0.1,
                    "floor_ratio": 0.2, "effective_batch": 64, "epochs": 2,
                    "selection": "synthetic fixture",
                    "source": "tests"}
            (obj_dir / "provenance.json").write_text(json.dumps(side))
            # The real flavour-B bank always carries both alias sidecars;
            # write them consistent so the happy path EXERCISES the
            # identity-alias cross-check (mismatch tests corrupt them).
            (obj_dir / "fit_provenance.json").write_text(json.dumps(
                {"resolved_model_id": sc_model,
                 "resolved_revision": revision,
                 "positions_at_fit": leg,
                 "writer": "synthetic fixture"}))
            (obj_dir / "summary.json").write_text(json.dumps(
                {"position": leg, "model": sc_model,
                 "model_revision": revision, "si": sc_si, "arm": sc_arm,
                 "status": "ok", "floor_ratio": 0.2}))
    if complete:
        (obj_dir / "COMPLETE").write_text("")
    return {"model_key": model_key, "si": si, "arm": arm,
            "expect_leg": "response", "path": str(obj_dir / "object.pt")}


def run_pipeline(tmp_path: Path, entries: list[dict], extra_args=()) -> dict:
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(entries))
    out = tmp_path / "out"
    rc = pdc.run(["--manifest", str(mf), "--out", str(out), "--no-figs",
                  "--no-tokens", "--skip-regression", "--allow-partial",
                  *extra_args])
    assert rc == 0
    return json.loads((out / "results.json").read_text())


def pin(entries: list[dict]) -> list[dict]:
    """Copy entries with each one's real sha256 pinned (fix-3 tests)."""
    out = []
    for e in entries:
        e = dict(e)
        e["sha256"] = hashlib.sha256(
            Path(e["path"]).read_bytes()).hexdigest()
        out.append(e)
    return out


def unit(v, d=8):
    x = np.zeros(d)
    for i, val in (v.items() if isinstance(v, dict) else [(v, 1.0)]):
        x[i] = val
    return torch.tensor(x / np.linalg.norm(x), dtype=torch.float32)


# --------------------------------------------------------------------------
# 17. import-path guard (first: everything else assumes it)
# --------------------------------------------------------------------------

def test_import_path_guard():
    got = Path(sisteer.__file__).resolve()
    assert (REPO / "src") in got.parents, (
        f"sisteer imported from {got}, not this checkout -- PYTHONPATH "
        "shadowing drift")
    pdc.assert_import_provenance()


# --------------------------------------------------------------------------
# 1. known cosines through the full pipeline into the saved .npy
# --------------------------------------------------------------------------

def test_known_cosines_end_to_end(tmp_path):
    d = 8
    theta = math.radians(30)
    a_dirs = {0: unit(0, d), 1: unit(1, d),
              2: unit({0: math.cos(theta), 1: math.sin(theta)}, d),
              3: unit(2, d)}
    b_dirs = {0: unit(0, d),                      # identical -> +1
              1: unit({1: -1.0}, d),              # opposite  -> -1
              2: unit(0, d),                      # 30 degrees -> cos(30)
              3: unit(3, d)}                      # orthogonal -> 0
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", d=d,
                    directions=a_dirs),
        make_object(tmp_path / "a", si="si_constitution", d=d,
                    directions=b_dirs),
    ]
    res = run_pipeline(tmp_path, entries)
    M = np.load(tmp_path / "out" / "matrices"
                / "llama8b__safety_refusal__x__si_constitution.npy")
    diag = np.diag(M)
    expected = [1.0, -1.0, math.cos(theta), 0.0]
    assert np.allclose(diag, expected, atol=1e-9)
    # the -1 survives into the saved matrix: the no-abs/no-sign-fix guard
    assert diag[1] == pytest.approx(-1.0, abs=1e-9)
    block = next(b for b in res["blocks"]
                 if b["si_a"] == "safety_refusal"
                 and b["si_b"] == "si_constitution")
    assert block["same_layer_diagonal"]["1"] == pytest.approx(-1.0, 1e-9)


def test_unit_diagonal_within_si(tmp_path):
    entries = [make_object(tmp_path / "a", si="safety_refusal", d=8)]
    run_pipeline(tmp_path, entries)
    M = np.load(tmp_path / "out" / "matrices"
                / "llama8b__safety_refusal__x__safety_refusal.npy")
    assert np.allclose(np.diag(M), 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# 2. string-key indexing, non-contiguous layers
# --------------------------------------------------------------------------

def test_noncontiguous_string_key_layers(tmp_path):
    d = 8
    layers = (0, 2, 5)
    a_dirs = {0: unit(0, d), 2: unit(1, d), 5: unit(2, d)}
    b_dirs = {0: unit(0, d), 2: unit(1, d), 5: unit({2: -1.0}, d)}
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", d=d, layers=layers,
                    directions=a_dirs, header_len=4),
        make_object(tmp_path / "a", si="si_constitution", d=d, layers=layers,
                    directions=b_dirs, header_len=4),
    ]
    res = run_pipeline(tmp_path, entries)
    blob = torch.load(tmp_path / "a" / "safety_refusal__psr_kl_all"
                      / "object.pt", map_location="cpu", weights_only=True)
    assert set(blob["directions"]) == {"0", "2", "5"}  # string keys
    block = next(b for b in res["blocks"]
                 if b["si_a"] == "safety_refusal"
                 and b["si_b"] == "si_constitution")
    assert set(block["same_layer_diagonal"]) == {"0", "2", "5"}
    assert block["same_layer_diagonal"]["5"] == pytest.approx(-1.0, 1e-9)
    M = np.load(tmp_path / "out" / "matrices"
                / "llama8b__safety_refusal__x__si_constitution.npy")
    assert M.shape == (3, 3)
    assert M[2, 2] == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 3. byte-identity of the stub load seam
# --------------------------------------------------------------------------

def _assert_seam_byte_identical(blob):
    layers = [int(l) for l in blob["layers"]]
    d = int(blob["directions"][str(layers[0])].shape[0])
    obj = PSRObject.load_blob(blob, pdc._StubModel(d, max(layers) + 1), None)
    for l in layers:
        assert torch.equal(obj.directions[l].data, blob["directions"][str(l)])
        assert torch.equal(obj.gate_w[l].data, blob["gate_w"][str(l)])
        assert torch.equal(obj.gate_b[l].data, blob["gate_b"][str(l)])


def test_seam_byte_identity_synthetic(tmp_path):
    entry = make_object(tmp_path / "a")
    blob = torch.load(entry["path"], map_location="cpu", weights_only=True)
    _assert_seam_byte_identical(blob)


@pytest.mark.skipif(not HAVE_REAL_BANK, reason="real llama8b bank absent")
def test_seam_byte_identity_real_bank():
    blob = torch.load(REAL_LLAMA8B[0], map_location="cpu",
                      weights_only=True)
    _assert_seam_byte_identical(blob)


# --------------------------------------------------------------------------
# 4. Leg discipline
# --------------------------------------------------------------------------

def test_d13_leg_read_from_blob_not_the_trap(tmp_path):
    entry = make_object(tmp_path / "a", leg="response", header_len=4)
    lo = pdc.load_entry(entry, 1e-6, 1e-3)
    assert lo.record["leg"] == "response"
    assert lo.record["response_header_len"] == 4
    # The trap: the unrelated serve-time attribute (same name as the blob
    # key) reads 'all' on this very object -- proving the analysis did not
    # read it. getattr-by-string in the TEST only.
    blob = torch.load(entry["path"], map_location="cpu", weights_only=True)
    obj = PSRObject.load_blob(blob, pdc._StubModel(8, 4), None)
    assert getattr(obj, "positions") == "all"      # the trap reads WRONG
    assert obj.fit_positions == "response"          # the real leg


def test_d13_loader_hard_errors_on_leg_restore_drift(tmp_path, monkeypatch):
    entry = make_object(tmp_path / "a", leg="response", header_len=4)
    # Simulate load_blob's restore path silently breaking.
    monkeypatch.setattr(PSRObject, "set_fit_positions",
                        lambda self, *a, **k: None)
    with pytest.raises(SystemExit, match="leg restore drifted"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_d13_grep_ban_on_banned_attribute():
    src = Path(pdc.__file__).read_text()
    assert re.search(r"\.positions\b", src) is None, (
        "the banned leg attribute access appears in the analysis module")


# --------------------------------------------------------------------------
# 5. dead / near-dead masking
# --------------------------------------------------------------------------

def test_dead_and_near_dead_masking(tmp_path):
    d = 8
    layers = (0, 1, 2, 3, 4, 5)
    a_dirs = {l: unit(l % d, d) for l in layers}
    a_dirs[2] = torch.zeros(d)                       # dead (exact zero)
    a_dirs[3] = 1e-5 * unit(3, d)                    # near-dead (rel rule)
    b_dirs = {l: unit(l % d, d) for l in layers}
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", d=d, layers=layers,
                    directions=a_dirs),
        make_object(tmp_path / "a", si="si_constitution", d=d, layers=layers,
                    directions=b_dirs),
    ]
    res = run_pipeline(tmp_path, entries)
    rec = next(o for o in res["objects"] if o["si"] == "safety_refusal")
    assert rec["dead_layers"] == [2]
    assert rec["near_dead_layers"] == [3]
    M = np.load(tmp_path / "out" / "matrices"
                / "llama8b__safety_refusal__x__si_constitution.npy")
    assert np.isnan(M[2]).all() and np.isnan(M[3]).all()   # rows masked
    assert np.isfinite(M[0]).all()
    Mw = np.load(tmp_path / "out" / "matrices"
                 / "llama8b__safety_refusal__x__safety_refusal.npy")
    assert np.isnan(Mw[2]).all() and np.isnan(Mw[:, 2]).all()
    block = next(b for b in res["blocks"]
                 if b["si_a"] == "safety_refusal"
                 and b["si_b"] == "si_constitution")
    assert block["masked_layers"] == [2, 3]
    assert block["n_live_layers"] == 4
    assert block["same_layer_diagonal"]["2"] is None
    # masked layers excluded from every aggregate: all live diags are 1.0
    assert block["diag_mean"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 7. identity mismatches fail loudly (SystemExit), naming the field
# --------------------------------------------------------------------------

def test_mismatch_sidecar_si_vs_manifest(tmp_path):
    entry = make_object(tmp_path / "a", si="safety_refusal",
                        sidecar_si="si_constitution",
                        dirname="si_constitution__psr_kl_all")
    with pytest.raises(SystemExit, match="si"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_mismatch_dirname_vs_sidecar(tmp_path):
    entry = make_object(tmp_path / "a", si="safety_refusal",
                        dirname="si_irrelevant__psr_kl_all")
    with pytest.raises(SystemExit, match="dirname"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_mismatch_sidecar_model_id_vs_map(tmp_path):
    entry = make_object(tmp_path / "a", flavour="B",
                        sidecar_model="totally/other-model")
    with pytest.raises(SystemExit, match="MODEL_KEY_TO_HF"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_mismatch_model_key_absent_from_map(tmp_path):
    entry = make_object(tmp_path / "a")
    entry["model_key"] = "nonexistent_model"
    with pytest.raises(SystemExit, match="absent from MODEL_KEY_TO_HF"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_mismatch_target_layers_vs_blob(tmp_path):
    entry = make_object(tmp_path / "a", layers=(0, 1, 2, 3),
                        target_layers=[0, 1])
    with pytest.raises(SystemExit, match="target_layers"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_missing_both_sidecars(tmp_path):
    entry = make_object(tmp_path / "a", write_sidecar=False)
    with pytest.raises(SystemExit, match="neither provenance.json nor"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_both_sidecar_flavours_disagreeing(tmp_path):
    entry = make_object(tmp_path / "a", flavour="B")
    obj_dir = Path(entry["path"]).parent
    # add a flavour-A sidecar disagreeing on the si
    (obj_dir / "train_budget_meta.json").write_text(json.dumps(
        {"si_id": "si_irrelevant"}))
    with pytest.raises(SystemExit, match="flavours disagree"):
        pdc.load_entry(entry, 1e-6, 1e-3)


# --------------------------------------------------------------------------
# 8. COMPLETE never gates; 24. flavour field names
# --------------------------------------------------------------------------

def test_complete_marker_never_gates(tmp_path):
    entries = [make_object(tmp_path / "a", complete=False)]
    res = run_pipeline(tmp_path, entries)
    rec = res["objects"][0]
    assert rec["complete_marker_present"] is False
    assert len(res["blocks"]) == 1  # loaded and included


def test_sidecar_flavour_field_names(tmp_path):
    ea = make_object(tmp_path / "a", si="safety_refusal", flavour="A",
                     revision="aaaa1111")
    eb = make_object(tmp_path / "b", si="safety_refusal", flavour="B",
                     revision="bbbb2222", model_key="qwen35_9b",
                     dirname="9b")
    ra = pdc.load_entry(ea, 1e-6, 1e-3).record
    rb = pdc.load_entry(eb, 1e-6, 1e-3).record
    # model id read from model_id (flavour A) and model (flavour B)
    assert ra["sidecar_flavor"] == "train_budget_meta"
    assert ra["model_id"] == pdc.MODEL_KEY_TO_HF["llama8b"]
    assert ra["model_revision"] == "aaaa1111"
    assert rb["sidecar_flavor"] == "provenance"
    assert rb["model_id"] == pdc.MODEL_KEY_TO_HF["qwen35_9b"]
    assert rb["model_revision"] == "bbbb2222"
    # the identity-bearing aliases were actually cross-checked (fix 5)
    assert ra["alias_identity_checks"] == [
        "train_budget_meta.json:positions_at_fit"]
    assert set(rb["alias_identity_checks"]) == {
        "fit_provenance.json:positions_at_fit",
        "fit_provenance.json:resolved_model_id",
        "fit_provenance.json:resolved_revision",
        "summary.json:position", "summary.json:model",
        "summary.json:model_revision", "summary.json:si",
        "summary.json:arm"}


# --------------------------------------------------------------------------
# 9. leg quarantine
# --------------------------------------------------------------------------

def test_off_leg_quarantine(tmp_path):
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", leg="response"),
        make_object(tmp_path / "a", si="si_constitution", leg="all"),
    ]
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(entries))
    out = tmp_path / "out"
    with pytest.warns(RuntimeWarning, match="QUARANTINED"):
        rc = pdc.run(["--manifest", str(mf), "--out", str(out), "--no-figs",
                      "--no-tokens", "--skip-regression", "--allow-partial"])
    assert rc == 0  # under --allow-partial: quarantined, never a crash
    res = json.loads((out / "results.json").read_text())
    assert len(res["off_leg_objects"]) == 1
    off = res["off_leg_objects"][0]
    assert off["si"] == "si_constitution" and off["leg"] == "all"
    # excluded from the response-leg matrix set: no cross block exists
    assert all(b["si_a"] == b["si_b"] == "safety_refusal"
               for b in res["blocks"])
    # its within-SI matrix is separately labelled, never mixed in
    assert (out / "matrices"
            / "offleg__llama8b__si_constitution__x__si_constitution.npy"
            ).is_file()
    assert not (out / "matrices"
                / "llama8b__si_constitution__x__si_constitution.npy"
                ).is_file()


# --------------------------------------------------------------------------
# 10. SI metadata helper
# --------------------------------------------------------------------------

def test_si_meta_lengths_and_sources():
    m_saf = pdc.si_meta("safety_refusal")
    m_con = pdc.si_meta("si_constitution")
    m_irr = pdc.si_meta("si_irrelevant")
    assert (m_saf["words"], m_saf["chars"]) == (66, 419)
    assert (m_con["words"], m_con["chars"]) == (4896, 30952)
    assert (m_irr["words"], m_irr["chars"]) == (4930, 30407)
    # safety_refusal resolves via the module-attribute fallback -- it is
    # deliberately absent from BY_ID and resolve_si_id would raise
    assert m_saf["registry_source"] == "module_attribute"
    assert m_con["registry_source"] == "BY_ID"
    assert m_irr["registry_source"] == "BY_ID"


def test_si_meta_unknown_id_fails_loudly():
    with pytest.raises(SystemExit, match="unknown SI id"):
        pdc.si_meta("no_such_si_anywhere")


# --------------------------------------------------------------------------
# 11. gate health -- the real bank exercises none of these paths
# --------------------------------------------------------------------------

def test_frozen_gate_b_fingerprint_fires(tmp_path):
    layers = tuple(range(8))
    gate_b = {l: torch.tensor([0.924214]) for l in range(6)}
    gate_b[6] = torch.tensor([0.85])
    gate_b[7] = torch.tensor([0.99])
    entries = [make_object(tmp_path / "a", layers=layers, gate_b=gate_b)]
    res = run_pipeline(tmp_path, entries)
    rec = res["objects"][0]
    assert rec["frozen_fingerprint_fires"] is True
    assert rec["frozen_gate_b_groups"]["0.924214"] == [0, 1, 2, 3, 4, 5]


def test_provably_dead_gate_flag_never_masks_direction(tmp_path):
    d = 8
    layers = (0, 1, 2, 3)
    gate_w = {l: torch.tensor(np.ones(d), dtype=torch.float32)
              for l in layers}
    gate_w[2] = torch.zeros(d)
    gate_b = {l: torch.ones(1) for l in layers}
    gate_b[2] = torch.tensor([-0.5])   # ReLU(w.h+b) == 0 for every input
    entries = [make_object(tmp_path / "a", d=d, layers=layers,
                           gate_w=gate_w, gate_b=gate_b)]
    res = run_pipeline(tmp_path, entries)
    rec = res["objects"][0]
    assert rec["provably_dead_gate_layers"] == [2]   # EXACT zero w, b<=0
    assert rec["constant_gate_layers"] == []
    assert rec["tiny_nonzero_gate_w_layers"] == []   # zero is not "tiny"
    assert rec["dead_layers"] == []      # gate flags never mask directions
    M = np.load(tmp_path / "out" / "matrices"
                / "llama8b__safety_refusal__x__safety_refusal.npy")
    assert np.isfinite(M[2, 2])          # direction block stays live
    Mw = np.load(tmp_path / "out" / "matrices"
                 / "gatew__llama8b__safety_refusal__x__safety_refusal.npy")
    assert np.isnan(Mw[2]).all()         # gate_w block masks its own dead


def test_constant_gate_flag(tmp_path):
    d = 8
    layers = (0, 1, 2, 3)
    gate_w = {l: torch.tensor(np.ones(d), dtype=torch.float32)
              for l in layers}
    gate_w[1] = torch.zeros(d)
    gate_b = {l: torch.ones(1) for l in layers}
    gate_b[1] = torch.tensor([0.7])      # input-independent constant
    entries = [make_object(tmp_path / "a", d=d, layers=layers,
                           gate_w=gate_w, gate_b=gate_b)]
    res = run_pipeline(tmp_path, entries)
    rec = res["objects"][0]
    assert rec["constant_gate_layers"] == [1]        # EXACT zero w, b>0
    assert rec["provably_dead_gate_layers"] == []
    assert rec["tiny_nonzero_gate_w_layers"] == []
    assert rec["dead_layers"] == []


def test_tiny_nonzero_gate_w_is_heuristic_never_provable(tmp_path):
    # Epsilon-small but NONZERO gate_w proves nothing: some h still makes
    # the preactivation positive. It must NOT be claimed provably-dead or
    # constant; it gets the separate heuristic label (and stays a pure
    # observation -- no direction is masked).
    d = 8
    layers = (0, 1, 2, 3)
    gate_w = {l: torch.tensor(np.ones(d), dtype=torch.float32)
              for l in layers}
    gate_w[2] = torch.full((d,), 1e-9)   # ||w|| ~ 2.8e-9: < eps, nonzero
    gate_b = {l: torch.ones(1) for l in layers}
    gate_b[2] = torch.tensor([-0.5])     # would have counted as "provable"
    entries = [make_object(tmp_path / "a", d=d, layers=layers,
                           gate_w=gate_w, gate_b=gate_b)]
    res = run_pipeline(tmp_path, entries)
    rec = res["objects"][0]
    assert rec["provably_dead_gate_layers"] == []
    assert rec["constant_gate_layers"] == []
    assert rec["tiny_nonzero_gate_w_layers"] == [2]
    assert rec["dead_gate_w_layers"] == [2]  # numeric gate_w mask unchanged
    assert rec["dead_layers"] == []          # directions never masked


# --------------------------------------------------------------------------
# 12. chance baseline
# --------------------------------------------------------------------------

def test_chance_sigma_values():
    for d, expected in ((4096, 0.0156), (3072, 0.0180), (2560, 0.0198),
                        (1536, 0.0255)):
        assert round(1.0 / math.sqrt(d), 4) == pytest.approx(expected)


def test_sigma_units_and_se_on_synthetic_block(tmp_path):
    d = 16
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", d=d),
        make_object(tmp_path / "a", si="si_constitution", d=d),
    ]
    res = run_pipeline(tmp_path, entries)
    block = next(b for b in res["blocks"]
                 if b["si_a"] == "safety_refusal"
                 and b["si_b"] == "si_constitution")
    sigma = 1.0 / math.sqrt(d)
    assert block["sigma"] == pytest.approx(sigma)
    assert block["diag_mean_sigma_units"] == pytest.approx(
        block["diag_mean"] / sigma)
    assert block["diag_mean_se"] == pytest.approx(
        sigma / math.sqrt(block["n_live_layers"]))


# --------------------------------------------------------------------------
# 13/14. determinism + sha256 pinning
# --------------------------------------------------------------------------

def _make_pair_manifest(tmp_path):
    return [
        make_object(tmp_path / "a", si="safety_refusal", d=12,
                    layers=(0, 1, 2, 3, 4, 5, 6, 7)),
        make_object(tmp_path / "a", si="si_constitution", d=12,
                    layers=(0, 1, 2, 3, 4, 5, 6, 7)),
        make_object(tmp_path / "a", si="si_irrelevant", d=12,
                    layers=(0, 1, 2, 3, 4, 5, 6, 7)),
    ]


def test_determinism_byte_identical(tmp_path):
    entries = _make_pair_manifest(tmp_path)
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(entries))
    outs = []
    for name in ("out1", "out2"):
        out = tmp_path / name
        rc = pdc.run(["--manifest", str(mf), "--out", str(out), "--no-figs",
                      "--no-tokens", "--skip-regression", "--allow-partial"])
        assert rc == 0
        outs.append(out)
    for rel in sorted(p.relative_to(outs[0])
                      for p in outs[0].rglob("*") if p.is_file()):
        b1 = (outs[0] / rel).read_bytes()
        b2 = (outs[1] / rel).read_bytes()
        assert b1 == b2, f"{rel} differs between identical runs"
    files1 = {p.relative_to(outs[0]) for p in outs[0].rglob("*")}
    files2 = {p.relative_to(outs[1]) for p in outs[1].rglob("*")}
    assert files1 == files2


def test_sha256_pinning(tmp_path):
    entry = make_object(tmp_path / "a")
    lo = pdc.load_entry(entry, 1e-6, 1e-3)
    raw = Path(entry["path"]).read_bytes()
    assert lo.record["sha256"] == hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# 15/16. regression machinery + real-bank integration
# --------------------------------------------------------------------------

def test_regression_gate_fails_on_synthetic_bank_and_exits_nonzero(tmp_path):
    # A synthetic llama8b full grid cannot reproduce the real table: the
    # in-run gate must fire and the run must exit non-zero.
    entries = _make_pair_manifest(tmp_path)
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(entries))
    out = tmp_path / "out"
    rc = pdc.run(["--manifest", str(mf), "--out", str(out), "--no-figs",
                  "--no-tokens", "--allow-partial"])
    assert rc == 1
    rec = json.loads((out / "self_check.json").read_text())
    assert rec["passed"] is False
    # and --skip-regression is the only sanctioned way around it
    rc2 = pdc.run(["--manifest", str(mf), "--out", str(tmp_path / "out2"),
                   "--no-figs", "--no-tokens", "--skip-regression",
                   "--allow-partial"])
    assert rc2 == 0


def test_self_check_cannot_validate_exits_nonzero(tmp_path, monkeypatch):
    # Missing objects mean the check CANNOT validate -- that must be a
    # non-zero exit, never a clean 0 (the standing project rule).
    missing = [dict(e, path=str(tmp_path / "nope" / f"{i}.pt"))
               for i, e in enumerate(pdc.DEFAULT_OBJECTS)
               if e["model_key"] == "llama8b"]
    monkeypatch.setattr(pdc, "DEFAULT_OBJECTS", missing)
    with pytest.raises(SystemExit):
        pdc.run(["--self-check", "--out", str(tmp_path / "out")])


def test_regression_check_reports_mismatch():
    rec = pdc.run_regression_check([], [], [])   # nothing computed at all
    assert rec["passed"] is False


@pytest.mark.skipif(not HAVE_REAL_BANK, reason="real llama8b bank absent")
def test_regression_real_bank_self_check(tmp_path):
    rc = pdc.run(["--self-check", "--out", str(tmp_path / "out")])
    assert rc == 0
    rec = json.loads((tmp_path / "out" / "self_check.json").read_text())
    assert rec["passed"] is True
    assert all(c["passed"] for c in rec["checks"])
    assert len(rec["checks"]) == 22


# --------------------------------------------------------------------------
# 18/19. gauge invariance + factorisation identity
# --------------------------------------------------------------------------

def _random_bank(tmp_path, sub, scale=None, d=10, n_layers=6):
    rng = np.random.default_rng(3)
    layers = tuple(range(n_layers))
    entries = []
    for si in ("safety_refusal", "si_constitution"):
        dirs, gws, gbs = {}, {}, {}
        for l in layers:
            dv = rng.normal(size=d)
            wv = rng.normal(size=d)
            bv = rng.normal()
            c = 1.0 if scale is None else scale[l]
            dirs[l] = torch.tensor(dv / c, dtype=torch.float32)
            gws[l] = torch.tensor(wv * c, dtype=torch.float32)
            gbs[l] = torch.tensor([bv * c], dtype=torch.float32)
        entries.append(make_object(tmp_path / sub, si=si, d=d, layers=layers,
                                   directions=dirs, gate_w=gws, gate_b=gbs))
    return entries


def test_gauge_invariance_under_per_layer_rescale(tmp_path):
    # (w, b, d) -> (cw, cb, d/c) leaves the intervention exactly unchanged;
    # every reported cross-object quantity must not move.
    rng = np.random.default_rng(11)
    scale = {l: float(rng.uniform(0.2, 5.0)) for l in range(6)}
    res_base = run_pipeline(tmp_path / "base",
                            _random_bank(tmp_path, "base"))
    res_scaled = run_pipeline(tmp_path / "scaled",
                              _random_bank(tmp_path, "scaled", scale=scale))

    def mats(root):
        mdir = root / "out" / "matrices"
        return {p.name: np.load(p) for p in sorted(mdir.glob("*.npy"))}

    m1, m2 = mats(tmp_path / "base"), mats(tmp_path / "scaled")
    assert set(m1) == set(m2)
    for name in m1:   # direction, gate_w AND jacobian cosines all invariant
        assert np.allclose(m1[name], m2[name], atol=1e-6, equal_nan=True), (
            f"{name} moved under a pure gauge transformation")
    for o1, o2 in zip(res_base["objects"], res_scaled["objects"]):
        for field in ("jacobian_norms", "gate_thresholds"):
            for k in o1[field]:
                assert o1[field][k] == pytest.approx(o2[field][k],
                                                     rel=1e-5), (
                    f"{field}[{k}] is not gauge-invariant")
        # sanity: the transformation was NOT trivial -- raw norms moved
        assert any(abs(o1["per_layer_norms"][k]
                       - o2["per_layer_norms"][k]) > 1e-3
                   for k in o1["per_layer_norms"])


def test_jacobian_factorisation_identity():
    rng = np.random.default_rng(5)
    d = 16
    dA, wA = rng.normal(size=d), rng.normal(size=d)
    dB, wB = rng.normal(size=d), rng.normal(size=d)

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    lhs = cos(np.outer(dA, wA).ravel(), np.outer(dB, wB).ravel())
    rhs = cos(dA, dB) * cos(wA, wB)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_gauge_transformation_leaves_intervention_unchanged():
    rng = np.random.default_rng(9)
    d, n = 16, 7
    w, b, dvec = rng.normal(size=d), 0.3, rng.normal(size=d)
    h = rng.normal(size=(n, d))
    c = 3.7
    base = np.maximum(h @ w + b, 0.0)[:, None] * dvec
    scaled = np.maximum(h @ (c * w) + c * b, 0.0)[:, None] * (dvec / c)
    assert np.allclose(base, scaled, atol=1e-12)


# --------------------------------------------------------------------------
# 20. derived contrasts + two-sign-check verdict
# --------------------------------------------------------------------------

def _grid_bank(tmp_path, sub, model_key, dirs_by_si, d=8, layers=(0, 1, 2, 3)):
    return [make_object(tmp_path / sub / model_key, model_key=model_key,
                        si=si, d=d, layers=layers,
                        directions={l: dirs_by_si[si] for l in layers})
            for si in pdc.SI_ORDER]


def test_contrasts_and_two_sign_check_met(tmp_path):
    # safety == const (CS diag = 1), irrelevant orthogonal (CI = SI = 0):
    # dL = 0 - (1+0)/2 = -0.5, dS = 1 - 0 = +1 at every layer.
    dirs = {"safety_refusal": unit(0), "si_constitution": unit(0),
            "si_irrelevant": unit(1)}
    entries = (_grid_bank(tmp_path, "x", "llama8b", dirs)
               + _grid_bank(tmp_path, "x", "gemma4e4b", dirs))
    res = run_pipeline(tmp_path, entries)
    con = {c["model_key"]: c for c in res["contrasts"]}
    for mk in ("llama8b", "gemma4e4b"):
        assert con[mk]["dL"]["mean"] == pytest.approx(-0.5, abs=1e-9)
        assert con[mk]["dS"]["mean"] == pytest.approx(1.0, abs=1e-9)
        assert con[mk]["dL"]["l0"] == pytest.approx(-0.5, abs=1e-9)
        assert con[mk]["dL"]["sign_of_mean"] == "-"
        assert con[mk]["dS"]["sign_of_mean"] == "+"
    rep = res["replication"]
    assert rep["verdict"] == "two_sign_check_met"
    assert rep["exploratory_model"] == "llama8b"
    assert rep["confirmatory_model"] == "gemma4e4b"
    # the record must name the criterion and its limits, never call
    # the outcome a replication.
    assert "identifies no mechanism" in rep["protocol"]
    assert "confirmatory replication" not in rep["protocol"]


def test_two_sign_check_not_met_on_sign_flip(tmp_path):
    dirs_a = {"safety_refusal": unit(0), "si_constitution": unit(0),
              "si_irrelevant": unit(1)}       # dS = +1
    dirs_b = {"safety_refusal": unit(0), "si_constitution": unit(1),
              "si_irrelevant": unit(0)}       # CS=0, CI=0, SI=1 -> dS = -1
    entries = (_grid_bank(tmp_path, "x", "llama8b", dirs_a)
               + _grid_bank(tmp_path, "x", "gemma4e4b", dirs_b))
    res = run_pipeline(tmp_path, entries)
    assert res["replication"]["verdict"] == "two_sign_check_not_met"


# --------------------------------------------------------------------------
# 21. empirical null
# --------------------------------------------------------------------------

def test_empirical_null_machinery():
    k = 10
    rng = np.random.default_rng(2)
    M = rng.normal(scale=0.05, size=(k, k))
    np.fill_diagonal(M, 0.5)
    M[8, 6] = 0.9                       # late off-diagonal plant
    layers = list(range(k))
    live = np.ones(k, dtype=bool)
    null1 = pdc.empirical_null(M, live, layers)
    null2 = pdc.empirical_null(M, live, layers)
    assert null1 == null2               # seeded: deterministic
    assert null1["seed"] == pdc.NULL_SEED
    assert null1["n_perm"] == pdc.NULL_PERMS
    alln = null1["regions"]["all"]
    sh = alln["shuffled"]
    assert sh["n"] == pdc.NULL_PERMS
    assert sh["observed"] == pytest.approx(0.5)
    assert sh["observed_percentile"] == 1.0   # diag far above any shuffle
    assert alln["distance_matched"]["1"]["n"] == 2 * (k - 1)
    assert alln["circular_shift"]["n_shifts"] == k - 1
    lom = null1["late_offdiag_max"]
    assert (lom["layer_i"], lom["layer_j"]) == (8, 6)
    assert lom["value"] == pytest.approx(0.9)
    # every null family exists per region, each with observed percentiles
    for reg in ("all", "early", "late"):
        r = null1["regions"][reg]
        assert r["offdiag_entries"]["n"] > 0
        assert r["offdiag_entries"]["observed_percentile"] is not None
        assert r["shuffled"]["observed_percentile"] is not None
        assert r["circular_shift"]["observed_percentile"] is not None
        assert r["observed_diag_mean"] is not None


def test_empirical_null_in_block_record(tmp_path):
    entries = _make_pair_manifest(tmp_path)
    res = run_pipeline(tmp_path, entries)
    block = next(b for b in res["blocks"] if b["si_a"] != b["si_b"])
    assert block["null"] is not None
    assert block["null"]["seed"] == pdc.NULL_SEED
    assert block["null"]["regions"]["all"]["shuffled"] is not None


# --------------------------------------------------------------------------
# 22/23. SI token counts
# --------------------------------------------------------------------------

@pytest.mark.skipif(not LLAMA8B_TOK_SNAPSHOT.is_dir(),
                    reason="llama8b tokenizer not in local HF cache")
def test_token_counts_real_llama_tokenizer():
    texts = {s: pdc.si_meta(s)["text"] for s in pdc.SI_ORDER}
    meta = pdc.compute_token_meta(
        {"llama8b": (pdc.MODEL_KEY_TO_HF["llama8b"],
                     "d10aef7999a2b5ba950ab3974312feeedbfe0b77")}, texts)
    toks = meta["llama8b"]["si_tokens"]
    # recomputed under the recorded id+revision; the expected
    # values below are the earlier measurements, confirmed here.
    assert toks["safety_refusal"]["raw"] == 88
    assert toks["si_constitution"]["raw"] == 5872
    assert toks["si_irrelevant"]["raw"] == 6380
    for si in pdc.SI_ORDER:
        assert toks[si]["rendered"] >= toks[si]["raw"]
    r = meta["llama8b"]["ratios"]
    assert r["irrelevant_over_constitution_raw"] == pytest.approx(
        6380 / 5872)
    assert r["constitution_over_safety_raw"] == pytest.approx(5872 / 88)


def test_token_meta_hard_errors_offline():
    texts = {"safety_refusal": pdc.si_meta("safety_refusal")["text"]}
    with pytest.raises(SystemExit, match="no-such-model"):
        pdc.compute_token_meta(
            {"fakekey": ("no-such-org/no-such-model", "0" * 40)}, texts)
    with pytest.raises(SystemExit, match="no model_revision"):
        pdc.compute_token_meta(
            {"fakekey": ("no-such-org/no-such-model", None)}, texts)


# --------------------------------------------------------------------------
# 25. review fix 1: single read, hash of the parsed bytes, weights_only
# --------------------------------------------------------------------------

def test_load_hashes_the_exact_bytes_it_parses(tmp_path, monkeypatch):
    # The recorded sha256 must describe the SAME bytes torch.load parsed:
    # torch.load receives an in-memory buffer (never the path, so there is
    # no time-of-check/time-of-use seam) and runs weights_only=True.
    entry = make_object(tmp_path / "a")
    seen = []
    real_load = torch.load

    def spy(f, *a, **kw):
        seen.append((type(f), kw.get("weights_only")))
        return real_load(f, *a, **kw)

    monkeypatch.setattr(torch, "load", spy)
    lo = pdc.load_entry(entry, 1e-6, 1e-3)
    assert seen == [(io.BytesIO, True)]
    assert lo.record["sha256"] == hashlib.sha256(
        Path(entry["path"]).read_bytes()).hexdigest()


def test_weights_only_false_banned_in_analysis_module():
    src = Path(pdc.__file__).read_text()
    assert "weights_only=False" not in src, (
        "arbitrary pickle execution reintroduced in the analysis module")


# --------------------------------------------------------------------------
# 26. review fix 2: same-model-group identity invariant (positional
# cross-SI statistics must never pair different layers/coordinates)
# --------------------------------------------------------------------------

def _run_expect_group_mismatch(tmp_path, entries, match):
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(entries))
    with pytest.raises(SystemExit, match=match):
        pdc.run(["--manifest", str(mf), "--out", str(tmp_path / "out"),
                 "--no-figs", "--no-tokens", "--skip-regression",
                 "--allow-partial"])


def test_group_identity_shifted_layer_list_fails(tmp_path):
    # [0,2,5] vs [0,3,5]: positional pairing would silently report layer 2
    # against layer 3 as "same-layer".
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", layers=(0, 2, 5)),
        make_object(tmp_path / "a", si="si_constitution", layers=(0, 3, 5)),
    ]
    _run_expect_group_mismatch(
        tmp_path, entries, "identity mismatch on layers")


def test_group_identity_revision_mismatch_fails(tmp_path):
    entries = [
        make_object(tmp_path / "a", si="safety_refusal",
                    revision="aaaa1111"),
        make_object(tmp_path / "a", si="si_constitution",
                    revision="bbbb2222"),
    ]
    _run_expect_group_mismatch(
        tmp_path, entries, "identity mismatch on model_revision")


# --------------------------------------------------------------------------
# 27. review fix 3: the default run cannot exit 0 with the wrong object set
# --------------------------------------------------------------------------

def test_default_manifest_is_frozen_and_pinned():
    cells = [(e["model_key"], e["si"], e["arm"])
             for e in pdc.DEFAULT_OBJECTS]
    assert len(cells) == 10 and len(set(cells)) == 10
    for e in pdc.DEFAULT_OBJECTS:
        assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"])
    # the frozen manifest validates against itself
    pdc.validate_manifest_cells(pdc.DEFAULT_OBJECTS, pdc.DEFAULT_OBJECTS)


@pytest.mark.skipif(not HAVE_REAL_BANK, reason="real llama8b bank absent")
def test_frozen_pins_match_real_llama8b_bank():
    for e in pdc.DEFAULT_OBJECTS:
        if e["model_key"] != "llama8b":
            continue
        raw = SITE.expand_path(e["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"], e["si"]


def _frozen_synthetic(tmp_path, monkeypatch):
    """Monkeypatch the frozen default manifest to a pinned synthetic
    3-cell set, returning the pinned entries."""
    frozen = pin([make_object(tmp_path / "bank", si=si)
                  for si in pdc.SI_ORDER])
    monkeypatch.setattr(pdc, "DEFAULT_OBJECTS", frozen)
    return frozen


def _run_pinned(tmp_path, entries, match=None):
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(entries))
    argv = ["--manifest", str(mf), "--out", str(tmp_path / "out"),
            "--no-figs", "--no-tokens", "--skip-regression"]
    if match is None:
        return pdc.run(argv)
    with pytest.raises(SystemExit, match=match):
        pdc.run(argv)
    return None


def test_missing_cell_fails(tmp_path, monkeypatch):
    frozen = _frozen_synthetic(tmp_path, monkeypatch)
    _run_pinned(tmp_path, frozen[:2], match="missing cells")


def test_duplicate_cell_fails(tmp_path, monkeypatch):
    frozen = _frozen_synthetic(tmp_path, monkeypatch)
    _run_pinned(tmp_path, frozen + [frozen[0]], match="duplicate cells")


def test_extra_cell_fails(tmp_path, monkeypatch):
    frozen = _frozen_synthetic(tmp_path, monkeypatch)
    extra = pin([make_object(tmp_path / "extra", model_key="gemma4e4b",
                             si="safety_refusal")])
    _run_pinned(tmp_path, frozen + extra, match="extra cells")


def test_unpinned_manifest_fails_without_allow_partial(tmp_path,
                                                       monkeypatch):
    frozen = _frozen_synthetic(tmp_path, monkeypatch)
    unpinned = [{k: v for k, v in e.items() if k != "sha256"}
                for e in frozen]
    _run_pinned(tmp_path, unpinned, match="unpinned entries")


def test_wrong_hash_fails(tmp_path, monkeypatch):
    # The pin describes different bytes than the file holds (a refit under
    # the frozen manifest): fail-loud at load, naming both hashes.
    frozen = _frozen_synthetic(tmp_path, monkeypatch)
    frozen[0]["sha256"] = "0" * 64
    _run_pinned(tmp_path, frozen, match="sha256 mismatch")


def test_wrong_leg_required_object_exits_nonzero(tmp_path, monkeypatch):
    # A REQUIRED object on the wrong leg writes its quarantine diagnostics
    # and then exits non-zero (quarantine-and-continue is only for
    # unexpected extras, behind --allow-partial).
    entries = [
        make_object(tmp_path / "bank", si="safety_refusal", leg="response"),
        make_object(tmp_path / "bank", si="si_constitution", leg="all"),
    ]
    frozen = pin(entries)
    monkeypatch.setattr(pdc, "DEFAULT_OBJECTS", frozen)
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(frozen))
    out = tmp_path / "out"
    with pytest.warns(RuntimeWarning, match="QUARANTINED"):
        rc = pdc.run(["--manifest", str(mf), "--out", str(out),
                      "--no-figs", "--no-tokens", "--skip-regression"])
    assert rc == 1
    # diagnostics were written BEFORE the non-zero exit
    res = json.loads((out / "results.json").read_text())
    assert len(res["off_leg_objects"]) == 1
    assert res["off_leg_objects"][0]["si"] == "si_constitution"
    assert (out / "matrices"
            / "offleg__llama8b__si_constitution__x__si_constitution.npy"
            ).is_file()


# --------------------------------------------------------------------------
# 28. review fix 5: identity-bearing sidecar aliases are cross-checked
# --------------------------------------------------------------------------

def _corrupt_json(obj_dir: Path, name: str, field: str, value) -> None:
    p = obj_dir / name
    d = json.loads(p.read_text())
    d[field] = value
    p.write_text(json.dumps(d))


def test_alias_mismatch_flavour_a_positions_at_fit(tmp_path):
    entry = make_object(tmp_path / "a", flavour="A", leg="response")
    _corrupt_json(Path(entry["path"]).parent, "train_budget_meta.json",
                  "positions_at_fit", "all")
    with pytest.raises(SystemExit, match="positions_at_fit"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_alias_mismatch_flavour_b_fit_provenance_revision(tmp_path):
    entry = make_object(tmp_path / "a", flavour="B")
    _corrupt_json(Path(entry["path"]).parent, "fit_provenance.json",
                  "resolved_revision", "0" * 40)
    with pytest.raises(SystemExit, match="resolved_revision"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_alias_mismatch_flavour_b_summary_position(tmp_path):
    entry = make_object(tmp_path / "a", flavour="B", leg="response")
    _corrupt_json(Path(entry["path"]).parent, "summary.json",
                  "position", "all")
    with pytest.raises(SystemExit, match="summary.json alias position"):
        pdc.load_entry(entry, 1e-6, 1e-3)


def test_alias_metrics_fields_never_gate(tmp_path):
    # Writer-specific/metric fields in the alias sidecars are pure
    # observations: cosmetic differences must never gate the run.
    entry = make_object(tmp_path / "a", flavour="B")
    obj_dir = Path(entry["path"]).parent
    _corrupt_json(obj_dir, "summary.json", "floor_ratio", 0.999)
    _corrupt_json(obj_dir, "summary.json", "status", "weird-but-cosmetic")
    _corrupt_json(obj_dir, "fit_provenance.json", "writer", "someone else")
    lo = pdc.load_entry(entry, 1e-6, 1e-3)
    assert lo.record["si"] == "safety_refusal"


# --------------------------------------------------------------------------
# 29. F1: empirical null exact values, regions with observed percentiles,
# and the shared-mapping contrast nulls
# --------------------------------------------------------------------------

def test_null_offsets_extended_to_five():
    assert pdc.NULL_OFFSETS == (1, 2, 3, 4, 5)


def test_null_exact_small_matrix():
    # 4 layers, hand-computable: off the diagonal M[i, j] = 10*i + j,
    # diag = 100 (far above every null draw).
    k = 4
    M = np.array([[10.0 * i + j for j in range(k)] for i in range(k)])
    np.fill_diagonal(M, 100.0)
    null = pdc.empirical_null(M, np.ones(k, dtype=bool), [0, 1, 2, 3])
    alln = null["regions"]["all"]
    assert alln["observed_diag_mean"] == pytest.approx(100.0)
    # off-diagonal entries: all 12 ordered pairs
    offs = [10.0 * i + j for i in range(k) for j in range(k) if i != j]
    assert alln["offdiag_entries"]["n"] == 12
    assert alln["offdiag_entries"]["mean"] == pytest.approx(np.mean(offs))
    assert alln["offdiag_entries"]["observed_percentile"] == 1.0
    # distance-matched at TRUE layer distance, both orientations
    assert alln["distance_matched"]["1"]["n"] == 6
    assert alln["distance_matched"]["1"]["mean"] == pytest.approx(
        np.mean([1, 12, 23, 10, 21, 32]))
    assert alln["distance_matched"]["3"]["n"] == 2
    assert alln["distance_matched"]["3"]["mean"] == pytest.approx(
        np.mean([3, 30]))
    assert alln["distance_matched"]["5"]["n"] == 0    # no pairs that far
    assert alln["distance_matched"]["5"]["mean"] is None
    # circular shifts: s=1 (0,1),(1,2),(2,3),(3,0); s=2; s=3 -- each 16.5
    cs = alln["circular_shift"]
    assert cs["n_shifts"] == 3
    assert cs["mean"] == pytest.approx(16.5)
    assert cs["observed_percentile"] == 1.0
    assert alln["shuffled"]["observed_percentile"] == 1.0


def test_null_masked_layer_keeps_original_positions():
    # 8 layers, layer 3 masked. Distance-1 pairs must be TRUE layer
    # adjacency -- (0,1),(1,2),(4,5),(5,6),(6,7) in both orientations, 10
    # entries. A compressed axis would also pair layers 2 and 4 (12
    # entries) and would shift layer 4 out of the late region.
    k = 8
    M = np.arange(k * k, dtype=np.float64).reshape(k, k)
    np.fill_diagonal(M, 999.0)
    live = np.ones(k, dtype=bool)
    live[3] = False
    null = pdc.empirical_null(M, live, list(range(k)))
    alln = null["regions"]["all"]
    assert alln["n_live"] == 7
    pairs = [(0, 1), (1, 2), (4, 5), (5, 6), (6, 7)]
    expect = [M[i, j] for i, j in pairs] + [M[j, i] for i, j in pairs]
    dm1 = alln["distance_matched"]["1"]
    assert dm1["n"] == 10
    assert dm1["mean"] == pytest.approx(np.mean(expect))
    # regions from ORIGINAL fractional depth: late = l/7 >= 0.5 -> layers
    # 4..7 (the compressed live list would have started it at layer 5)
    assert null["regions"]["late"]["region_layers"] == [4, 5, 6, 7]
    assert null["regions"]["early"]["region_layers"] == [0, 1]
    # the masked layer is excluded from every region, not remapped
    for reg in ("all", "early", "late"):
        assert 3 not in null["regions"][reg]["region_layers"]


def _distinct_bank(tmp_path, layers=(0, 1, 2, 3, 4, 5, 6, 7), d=12):
    """Full 3-SI grid with DISTINCT random directions per SI (make_object's
    default reuses one seed, which degenerates cross-SI blocks to 1s)."""
    rng = np.random.default_rng(42)
    entries = []
    for si in pdc.SI_ORDER:
        dirs = {l: torch.tensor(rng.normal(size=d), dtype=torch.float32)
                for l in layers}
        entries.append(make_object(tmp_path / "a", si=si, d=d,
                                   layers=layers, directions=dirs))
    return entries


def test_contrast_null_shared_mapping(tmp_path):
    entries = _distinct_bank(tmp_path)
    res = run_pipeline(tmp_path, entries)
    con = res["contrasts"][0]
    assert con["null"] is not None
    for key, side in (("dL", con["dL"]), ("dS", con["dS"])):
        nrec = con["null"][key]
        alln = nrec["regions"]["all"]
        # the null's observed quantity IS the reported contrast aggregate
        assert alln["observed_diag_mean"] == pytest.approx(side["mean"])
        assert (nrec["regions"]["early"]["observed_diag_mean"]
                == pytest.approx(side["mean_early"]))
        assert (nrec["regions"]["late"]["observed_diag_mean"]
                == pytest.approx(side["mean_late"]))
        for reg in ("all", "early", "late"):
            r = nrec["regions"][reg]
            assert r["shuffled"]["observed_percentile"] is not None
            assert r["offdiag_entries"]["observed_percentile"] is not None
    # ONE shared mapping across the three pair blocks: the null is built
    # on the elementwise contrast matrix, so its off-diagonal pool must
    # equal the elementwise combination of the three SAVED pair matrices
    # (three independent per-block nulls could not reproduce this).
    mdir = tmp_path / "out" / "matrices"
    cs = np.load(mdir / "llama8b__safety_refusal__x__si_constitution.npy")
    si = np.load(mdir / "llama8b__safety_refusal__x__si_irrelevant.npy")
    ci = np.load(mdir / "llama8b__si_constitution__x__si_irrelevant.npy")
    for key, MdX in (("dL", ci - (cs + si) / 2.0), ("dS", cs - si)):
        off = ~np.eye(MdX.shape[0], dtype=bool)
        alln = con["null"][key]["regions"]["all"]
        assert alln["offdiag_entries"]["n"] == int(off.sum())
        assert alln["offdiag_entries"]["mean"] == pytest.approx(
            float(MdX[off].mean()))


def test_contrast_null_masked_layer(tmp_path):
    # A dead layer in one SI must drop out of the contrast null too, at
    # its ORIGINAL position.
    d = 12
    layers = (0, 1, 2, 3, 4, 5, 6, 7)
    entries = _distinct_bank(tmp_path)
    # rebuild the safety object with a dead layer 3
    rng = np.random.default_rng(42)
    dirs = {l: torch.tensor(rng.normal(size=d), dtype=torch.float32)
            for l in layers}
    dirs[3] = torch.zeros(d)
    entries[0] = make_object(tmp_path / "a", si="safety_refusal", d=d,
                             layers=layers, directions=dirs)
    res = run_pipeline(tmp_path, entries)
    con = res["contrasts"][0]
    assert con["n_live_layers"] == 7
    alln = con["null"]["dL"]["regions"]["all"]
    assert alln["n_live"] == 7
    assert 3 not in alln["region_layers"]
    # true-adjacency distance 1 excludes pairs through the dead layer:
    # (0,1),(1,2),(4,5),(5,6),(6,7) x2 orientations
    assert alln["distance_matched"]["1"]["n"] == 10


# --------------------------------------------------------------------------
# 30. F2: the per-entry pooled band actually pools (concatenates)
# --------------------------------------------------------------------------

def test_pooled_offdiag_band_concatenates():
    layers = [0, 1, 2, 3, 4]
    n = len(layers)
    M1 = np.full((n, n), 0.1)
    np.fill_diagonal(M1, 1.0)
    M2 = np.full((n, n), 0.5)
    np.fill_diagonal(M2, 1.0)
    live = np.ones(n, dtype=bool)
    band = pdc.pooled_offdiag_band([M1, M2], [live, live], layers)
    off = ~np.eye(n, dtype=bool)
    pool = np.concatenate([M1[off], M2[off]])
    lo, hi, cnt = band["all"]
    assert cnt == 40                       # 2 blocks x 20 entries: POOLED
    assert lo == pytest.approx(np.quantile(pool, 0.025))
    assert hi == pytest.approx(np.quantile(pool, 0.975))
    # a masked layer in ONE block drops only that block's entries
    live2 = live.copy()
    live2[4] = False
    band2 = pdc.pooled_offdiag_band([M1, M2], [live, live2], layers)
    assert band2["all"][2] == 20 + 12
    # early region has a single layer here (l/4 < 0.25 -> layer 0 only):
    # no within-region pair exists, so the band is None (figure falls back
    # to the all-layer pool rather than inventing one)
    assert band["early"] is None
    assert band["late"] is not None


# --------------------------------------------------------------------------
# 31. F3: fractional-depth regions on original layer indices
# --------------------------------------------------------------------------

def test_region_masks_fractional_depth_boundaries():
    e32, l32 = pdc.region_masks(list(range(32)))
    assert [l for l in range(32) if e32[l]] == list(range(8))
    assert [l for l in range(32) if l32[l]] == list(range(16, 32))
    # L=35: layer 8 has depth 8/34 = 0.235 < 0.25 -- IN the first quarter
    # (35 // 4 = 8 wrongly excluded it); layer 17 sits at depth 0.5
    # exactly -- IN the second half (ceil(35/2) = 18 wrongly excluded it).
    e35, l35 = pdc.region_masks(list(range(35)))
    assert [l for l in range(35) if e35[l]] == list(range(9))
    assert [l for l in range(35) if l35[l]] == list(range(17, 35))
    # L=42: layer 10 has depth 10/41 = 0.244 < 0.25 -- IN the first
    # quarter (42 // 4 = 10 wrongly excluded it).
    e42, l42 = pdc.region_masks(list(range(42)))
    assert [l for l in range(42) if e42[l]] == list(range(11))
    assert [l for l in range(42) if l42[l]] == list(range(21, 42))


def test_block_region_aggregates_use_fractional_depth(tmp_path):
    # L=35 boundary through the whole pipeline: the same-layer diagonal is
    # 1.0 everywhere except layer 8 (orthogonal), so the early mean is 8/9
    # under fractional depth. The positional n//4 split (layers 0..7)
    # would report exactly 1.0.
    d = 8
    layers = tuple(range(35))
    a_dirs = {l: unit(0, d) for l in layers}
    b_dirs = {l: unit(0, d) for l in layers}
    b_dirs[8] = unit(1, d)
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", d=d, layers=layers,
                    directions=a_dirs),
        make_object(tmp_path / "a", si="si_constitution", d=d, layers=layers,
                    directions=b_dirs),
    ]
    res = run_pipeline(tmp_path, entries)
    block = next(b for b in res["blocks"]
                 if b["si_a"] == "safety_refusal"
                 and b["si_b"] == "si_constitution")
    assert block["diag_mean_early"] == pytest.approx(8 / 9, abs=1e-9)
    assert block["diag_mean_late"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 32. F4: kind-specific chance scales -- 1/d for the Jacobian product
# --------------------------------------------------------------------------

def test_chance_sigma_kind_specific():
    for d in (1536, 2560, 3072, 4096):
        assert pdc.chance_sigma("direction", d) == pytest.approx(
            1.0 / math.sqrt(d))
        assert pdc.chance_sigma("gate_w", d) == pytest.approx(
            1.0 / math.sqrt(d))
        assert pdc.chance_sigma("jacobian", d) == pytest.approx(1.0 / d)
    assert pdc.chance_sigma("jacobian", 4096) == pytest.approx(
        0.000244140625)
    with pytest.raises(SystemExit, match="unknown block kind"):
        pdc.chance_sigma("no_such_kind", 4096)


def test_jacobian_chance_scale_is_one_over_d_empirically():
    # cos(d_A w_A^T, d_B w_B^T) = cos(d_A, d_B) * cos(w_A, w_B): the
    # product of two INDEPENDENT ~N(0, 1/d) cosines has sd 1/d -- the
    # rank-one matrices live in a d^2-dimensional ambient space.
    rng = np.random.default_rng(0)
    d, n = 256, 4000

    def cosines():
        a = rng.normal(size=(n, d))
        b = rng.normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        return np.einsum("ij,ij->i", a, b)

    sd = (cosines() * cosines()).std()
    assert sd == pytest.approx(1.0 / d, rel=0.1)
    assert abs(sd - 1.0 / math.sqrt(d)) > 10 * sd  # NOT the vector scale


def test_jacobian_block_sigma_units(tmp_path):
    d = 16
    entries = [
        make_object(tmp_path / "a", si="safety_refusal", d=d),
        make_object(tmp_path / "a", si="si_constitution", d=d),
    ]
    res = run_pipeline(tmp_path, entries)

    def cross(blist):
        return next(b for b in blist
                    if b["si_a"] == "safety_refusal"
                    and b["si_b"] == "si_constitution")

    db, gw, jb = (cross(res[k]) for k in
                  ("blocks", "blocks_gate_w", "blocks_jacobian"))
    assert db["sigma"] == pytest.approx(1.0 / math.sqrt(d))
    assert gw["sigma"] == pytest.approx(1.0 / math.sqrt(d))
    assert jb["sigma"] == pytest.approx(1.0 / d)
    assert jb["diag_mean_sigma_units"] == pytest.approx(
        jb["diag_mean"] * d)
    assert jb["diag_mean_se"] == pytest.approx(
        (1.0 / d) / math.sqrt(jb["n_live_layers"]))
    # the per-object record carries both theoretical scales
    rec = res["objects"][0]
    assert rec["chance_sigma"] == pytest.approx(1.0 / math.sqrt(d))
    assert rec["chance_sigma_jacobian"] == pytest.approx(1.0 / d)


# --------------------------------------------------------------------------
# 33. F5: non-finite tensors are a hard error naming object/tensor/layer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "pos-inf", "neg-inf"])
@pytest.mark.parametrize("tensor", ["directions", "gate_w", "gate_b"])
def test_nonfinite_tensor_hard_errors(tmp_path, tensor, bad):
    d = 8
    layers = (0, 1, 2, 3)
    kw = {}
    if tensor == "directions":
        dirs = {l: unit(l % d, d) for l in layers}
        v = dirs[2].clone()
        v[0] = bad
        dirs[2] = v
        kw["directions"] = dirs
    elif tensor == "gate_w":
        gws = {l: torch.tensor(np.ones(d), dtype=torch.float32)
               for l in layers}
        v = gws[2].clone()
        v[0] = bad
        gws[2] = v
        kw["gate_w"] = gws
    else:
        gbs = {l: torch.ones(1) for l in layers}
        gbs[2] = torch.tensor([bad])
        kw["gate_b"] = gbs
    entry = make_object(tmp_path / "a", d=d, layers=layers, **kw)
    with pytest.raises(SystemExit) as ei:
        pdc.load_entry(entry, 1e-6, 1e-3)
    msg = str(ei.value)
    assert "non-finite" in msg          # the named hard error, no fallback
    assert tensor in msg                # names the tensor
    assert "layer 2" in msg             # names the layer
    assert "llama8b/safety_refusal" in msg   # names the object


# --------------------------------------------------------------------------
# 34. F6: the withdrawn diffmean overlay stays off the figures
# --------------------------------------------------------------------------

def test_diffmean_overlay_removed():
    src = Path(pdc.__file__).read_text()
    assert "DIFFMEAN" not in src, (
        "the D14d-withdrawn diffmean comparison was reintroduced")
    assert "prior lane" not in src
    assert "0.8577" not in src and "0.86 (" not in src


# --------------------------------------------------------------------------
# 35. The overturned dS caption framing stays off the figures
# --------------------------------------------------------------------------

def test_d20_withdrawn_ds_framing_removed():
    """The framing "safety overlap at matched coarse length" is
    withdrawn -- the mixed pairs are NOT length-matched
    (constitution/safety 66.7x vs irrelevant/safety 72.5x on llama8b),
    so dS is length-confounded and its caption must say "mixed-pair
    difference" and state the ratios."""
    src = Path(pdc.__file__).read_text()
    assert "matched coarse length" not in src, (
        "the D20a-withdrawn dS caption framing was reintroduced")
    assert "safety overlap" not in src and "safety-overlap" not in src
    assert "mixed-pair difference" in src
    # the verdict enum never reaches a caption: the figure renders the
    # two-sign-check phrasing instead
    assert "preregistered two-sign check" in src


def test_d20d_withdrawn_replication_verdict_removed():
    """The adopted wording is "met the preregistered two-sign check",
    never unqualified "replicates" -- the criterion is the signs of
    two artifact-level means on ONE held-out model, with one fit per cell,
    no seed replication, no effect-size floor, and no control for the
    length or fit-quality confounds. results.json is the authoritative
    machine-readable verdict, so its enum must name the criterion, not the
    withdrawn claim."""
    src = Path(pdc.__file__).read_text()
    assert '"replicates"' not in src, (
        "the D20d-withdrawn 'replicates' verdict enum was reintroduced")
    assert '"does-not-replicate"' not in src
    assert "confirmatory replication" not in src, (
        "the outcome must not be called a replication (D20d)")
    assert "replicates in sign" not in src
    assert '"two_sign_check_met"' in src
    assert '"two_sign_check_not_met"' in src
    # the record states plainly that a sign criterion identifies no
    # mechanism (the protocol text carries it; the figure caption phrases
    # it as "does not identify a mechanism", split across source literals)
    assert "identifies no mechanism" in src


def test_mixed_pair_length_line_states_both_ratios():
    token_meta = {
        "llama8b": {"si_tokens": {
            "safety_refusal": {"raw": 88, "rendered": 103},
            "si_constitution": {"raw": 5872, "rendered": 5887},
            "si_irrelevant": {"raw": 6380, "rendered": 6395}}},
    }
    line = pdc._mixed_pair_length_line(["llama8b"], token_meta)
    assert "NOT length-matched" in line
    assert "constitution/safety x66.7" in line
    assert "irrelevant/safety x72.5" in line
    # no token counts (--no-tokens) -> no line, never a word-count guess
    assert pdc._mixed_pair_length_line(["llama8b"], {}) is None
    assert pdc._mixed_pair_length_line(
        ["llama8b"], {"llama8b": {"si_tokens": {}}}) is None


def test_si_token_footer_carries_mixed_pair_ratios():
    token_meta = {
        "llama8b": {
            "tokenizer_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
            "revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
            "si_tokens": {
                "safety_refusal": {"raw": 88, "rendered": 103},
                "si_constitution": {"raw": 5872, "rendered": 5887},
                "si_irrelevant": {"raw": 6380, "rendered": 6395}},
            "ratios": {
                "constitution_over_safety_raw": 5872 / 88,
                "irrelevant_over_constitution_raw": 6380 / 5872}},
    }
    si_meta_rec = {s: {"words": 1} for s in pdc.SI_ORDER}
    line = pdc._si_token_footer("llama8b", token_meta, si_meta_rec,
                                list(pdc.SI_ORDER))
    assert "constitution/safety x66.7" in line
    assert "irrelevant/safety x72.5" in line
    assert "MIXED pairs are not length-matched" in line


# --------------------------------------------------------------------------
# 36. Figure rendering fixes: shared colour endpoints from the data
# excluding the trivial unit diagonal, diagonal masked in the DRAWING
# only (the analysis-time == 1.0 assertion stays untouched), NONLINEAR
# asinh colour norm with the linear width at the block kind's 2 sigma
# chance scale (a single shared linear scale rendered the 0.02-0.15
# cross-SI bulk indistinguishable from zero), colorbar ticks at
# reader-relevant values drawn at their true nonlinear positions,
# spelled-out legends everywhere colour encodes information. Rendering
# only: no numeric output may move.
# --------------------------------------------------------------------------

def test_shared_scale_is_true_max_and_excludes_unit_diagonal():
    rng = np.random.default_rng(0)
    n = 40
    off = rng.uniform(-0.2, 0.2, size=(n, n))
    within = off.copy()
    np.fill_diagonal(within, 1.0)          # trivial unit diagonal
    mats = {("m", "a", "a", "direction"): within,
            ("m", "a", "b", "direction"): off,
            ("m", "a", "a", "gate_w"): np.full((n, n), 0.9)}
    scale = pdc._shared_scale(mats, "direction")
    eye = np.eye(n, dtype=bool)
    pool = np.concatenate([np.abs(within[~eye]), np.abs(off).ravel()])
    # V is the TRUE max over drawn cells: the asinh norm compresses the
    # top of the range, so no percentile clipping is needed and no drawn
    # cell is clipped
    assert scale["V"] == pytest.approx(float(pool.max()))
    # the 1.0 unit diagonal did NOT set the scale (it is masked from the
    # drawing), and the other-kind 0.9 block stayed out of the pool
    assert scale["V"] < 0.5
    # empty kind falls back without crashing
    assert pdc._shared_scale(mats, "jacobian") == {"V": 0.05}


def test_heat_norm_is_asinh_with_2sigma_linear_width():
    from matplotlib.colors import AsinhNorm
    sigma = 1.0 / math.sqrt(4096)
    norm = pdc._heat_norm(0.725, sigma)
    assert isinstance(norm, AsinhNorm)
    assert norm.linear_width == pytest.approx(2 * sigma)
    assert norm.vmin == -0.725 and norm.vmax == 0.725
    assert norm(0.0) == pytest.approx(0.5)   # zero at the neutral colour
    # symmetric: +cos and -cos sit equidistant from the neutral colour
    assert norm(0.2) - 0.5 == pytest.approx(0.5 - norm(-0.2))
    # compressive above the linear width: an equal step in value costs
    # ever less colour, which is what lets 0.05 and 0.6 coexist
    assert (norm(0.65) - norm(0.60)) < (norm(0.10) - norm(0.05)) / 3


def test_display_matrix_masks_only_the_within_si_diagonal():
    M = np.arange(9, dtype=np.float64).reshape(3, 3)
    same = pdc._display_matrix(M, True)
    assert np.isnan(np.diag(same)).all()   # trivial diagonal -> masked
    off = ~np.eye(3, dtype=bool)
    assert np.array_equal(same[off], M[off])
    cross = pdc._display_matrix(M, False)
    assert np.array_equal(cross, M)        # cross-SI diagonal is REAL data
    assert cross is not M                  # always a copy
    assert M[0, 0] == 0.0                  # the saved matrix never mutates


def test_asinh_ticks_reader_values_at_true_positions():
    V, lw = 0.7246, 2.0 / math.sqrt(4096)
    chance = lw                                  # default: lw = 2 sigma
    ticks = pdc._asinh_ticks(V, lw, chance)
    assert ticks == sorted(ticks)
    assert 0.0 in ticks
    assert ticks[0] == pytest.approx(-V)
    assert ticks[-1] == pytest.approx(V)         # true endpoints readable
    assert chance in ticks and -chance in ticks  # the hatched band edge
    # symmetric about zero
    assert all(any(u == pytest.approx(-t) for u in ticks) for t in ticks)
    # the inner ladder is round 1-2-5 decade values a reader cares about
    inner = [t for t in ticks
             if t > 0 and t != chance and abs(t - V) > 1e-12]
    for t in inner:
        m = t / 10.0 ** math.floor(math.log10(t) + 1e-9)
        assert min(abs(m - c) for c in (1.0, 2.0, 5.0)) < 1e-9
    # ... and no two positive ticks overprint: >= 8% of the half-bar
    # apart in DISPLAY (asinh) space, where matplotlib will place them
    pos = [math.asinh(t / lw) / math.asinh(V / lw)
           for t in ticks if t > 0]
    assert min(np.diff(sorted(pos))) >= 0.08 - 1e-9
    # the jacobian family (chance ~ 2/d, a 3-decade span to V) still
    # yields a readable multi-decade ladder
    t2 = pdc._asinh_ticks(0.439, 2.0 / 4096, 2.0 / 4096)
    assert len(t2) >= 11
    assert t2[0] == pytest.approx(-0.439) and 2.0 / 4096 in t2
    # degenerate guard: chance at/above V falls back to endpoints + zero
    assert pdc._asinh_ticks(0.05, 0.2, 0.2) == [-0.05, 0.0, 0.05]


def test_make_figures_writes_the_set_and_never_mutates_matrices(tmp_path):
    entries = _distinct_bank(tmp_path)
    loaded = [pdc.load_entry(e, 1e-6, 1e-3) for e in entries]
    results, mats = pdc.analyse(
        loaded, tmp_path / "out",
        {s: pdc.si_meta(s) for s in pdc.SI_ORDER})
    results["token_meta"] = {}
    snap = {k: M.copy() for k, M in mats.items()}
    figs = tmp_path / "figs"
    written = pdc.make_figures(figs, loaded, mats, results)
    # drawing masks the trivial diagonal on a COPY: the analysis matrices
    # (already byte-identically saved by write_outputs) never move
    for k in snap:
        assert np.array_equal(snap[k], mats[k], equal_nan=True), k
    # the within-SI unit diagonal is still asserted == 1.0 at analysis
    # time and still 1.0 in the matrices handed to the figures
    for (mk, a, b, kind), M in mats.items():
        if a == b and kind == "direction":
            assert np.allclose(np.diag(M), 1.0, atol=1e-9)
    assert set(written) == {
        "final_contrasts_llama8b_gemma4e4b.png",
        "final_diag_cross_si_llama8b_gemma4e4b.png",
        "final_heatmap_grid_llama8b.png",
        "final_gatew_grid_llama8b.png",
        "final_jac_grid_llama8b.png",
        "final_norms_llama8b.png",
        "final_gateb_departure_all.png"}
    for name in written:
        assert (figs / name).stat().st_size > 0


def test_figure_legend_and_scale_conventions_in_source():
    src = Path(pdc.__file__).read_text()
    # the two ambiguous colorbar ticks that both read "+/-2s" are gone;
    # the chance region is ONE labelled band on the bar itself
    assert "+/-2s" not in src
    assert "chance band" in src
    # colorbar carries an axis label naming the quantity + sign convention
    assert "direction cosine similarity" in src        # KIND_QTY values
    assert "rank-one Jacobian cosine similarity" in src
    assert "(red = aligned, blue = opposed)" in src    # on the bar label
    # inline end-of-line labels are gone in favour of real legends
    assert "draw_end_labels" not in src
    # legends spell the series out, expansions included
    assert "constitution x irrelevant (CI)" in src
    assert "constitution x safety (CS)" in src
    assert "safety x irrelevant (SI)" in src
    assert "(long-pair excess)" in src
    assert "(mixed-pair difference)" in src
    # the band and the dashed reference line have legend entries
    assert "empirical null band" in src
    assert "isotropic +/-2 sigma reference" in src
    # the heatmap colour scale is NONLINEAR (asinh), said so on every
    # heatmap caption, with the linear width tied to the chance scale,
    # endpoints at the true data max, and the no-clipping fact stated
    assert "NONLINEAR symmetric asinh" in src
    assert "AsinhNorm" in src
    assert "chance scale" in src
    assert "no drawn cell is clipped" in src
    assert "true max |value|" in src
    # colorbar ticks are drawn at their true (nonlinear) positions
    assert "true nonlinear positions" in src
