"""The condition-registry -> lens-worker-registry shim, and above all
the coefficient fold — CPU-only, synthetic artifacts.

The load-bearing test is the NON-UNIT BAKED SCALE one. A fold tested
only where the object's baked scale is 1.0 cannot tell a MULTIPLY
convention (derived edit = c * baked) from a REPLACE one (derived edit =
c regardless of baked), because the two coincide there. Every object
whose baked scale is not 1 is where the conventions diverge and where
this class of bug survives a test suite, so that case is exercised
explicitly and asserted against the multiply contract.
"""

import io
import json
from pathlib import Path

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import registry_shim as rs


D = 6


def _blob_bytes(payload) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _write(tmp, name, data: bytes):
    p = tmp / name
    p.write_bytes(data)
    return str(p)


def _static_r_blob(base_scale: float, layer: int = 16, seed: int = 1) -> bytes:
    g = torch.Generator().manual_seed(seed)
    return _blob_bytes({"kind": "static_r", "layer": layer,
                        "r": torch.randn(D, generator=g),
                        "base_scale": base_scale, "metadata": {}})


def _vector_all_blob(seed: int = 2) -> bytes:
    g = torch.Generator().manual_seed(seed)
    return _blob_bytes({"kind": "vector_all",
                        "vectors": {"3": torch.randn(D, generator=g)},
                        "positions": "all", "skip_bos": True})


def _condition(tmp, art_bytes, *, kind, coeff, cell_id="cell0",
               spec_extra=None, sidecar_bytes=None):
    art = _write(tmp, f"{cell_id}.pt", art_bytes)
    spec = {"kind": kind}
    if spec_extra:
        spec.update(spec_extra)
    cond = {"cell_id": cell_id, "artifact_path": art,
            "resolved_spec": spec, "coefficient": coeff,
            "object_sha256": cm.sha256_hex16(art_bytes)}
    if sidecar_bytes is not None:
        cond["sidecar_path"] = _write(tmp, f"{cell_id}.json", sidecar_bytes)
        cond["sidecar_sha256"] = cm.sha256_hex16(sidecar_bytes)
    return cond


# --- the coefficient fold -----------------------------------------------------


def test_non_unit_baked_scale_folds_as_multiply_not_replace(tmp_path):
    """THE discriminating test. base_scale=0.125, c=0.25. Multiply gives
    0.03125; replace would give 0.25. Only multiply is correct, and only
    a non-unit baked scale can tell them apart."""
    cond = _condition(tmp_path, _static_r_blob(0.125), kind="static_r",
                      coeff=0.25)
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    (c0,) = res.conditions
    # served identity: the fold happened and is recorded relative + absolute
    assert c0.effective_coefficient == 0.25
    assert c0.coefficient_was_folded is True
    assert c0.original_sha256 != c0.derived_sha256
    # and the served op actually carries 0.25*0.125, proving multiply
    from sisteer.capeval.serving import lens_adapters as la
    art = (Path(res.registry_dir) / "cell0" / "artifact.bin").read_bytes()
    (op,) = la.get_adapter("static_r")({"kind": "static_r"}, art, b"")
    assert op.coeff == pytest.approx(0.25 * 0.125)
    assert op.coeff != pytest.approx(0.25)  # replace would land here


def test_unit_baked_scale_cannot_distinguish_conventions_by_design(tmp_path):
    """Kept as a documented blind spot: at S=1 multiply and replace
    coincide, so this case proves nothing about the convention — it is
    here only to show the suite knows that, and to pin that S=1 still
    folds correctly."""
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=0.25)
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    from sisteer.capeval.serving import lens_adapters as la
    art = (Path(res.registry_dir) / "cell0" / "artifact.bin").read_bytes()
    (op,) = la.get_adapter("static_r")({"kind": "static_r"}, art, b"")
    assert op.coeff == pytest.approx(0.25)  # 0.25*1.0 — ambiguous by design


def test_coefficient_one_is_the_identity_fold(tmp_path):
    """c=1.0 changes nothing, so the served bytes ARE the fit bytes and
    the two shas coincide — a condition at trained strength."""
    cond = _condition(tmp_path, _static_r_blob(0.5), kind="static_r",
                      coeff=1.0)
    (c0,) = rs.build_lens_registry([cond], tmp_path / "reg").conditions
    assert c0.effective_coefficient == 1.0
    assert c0.coefficient_was_folded is False
    assert c0.original_sha256 == c0.derived_sha256


def test_effective_coefficient_is_c_never_the_post_bake_one(tmp_path):
    """The derived artifact serves at selector-scale 1.0; the stamp must
    record the folded c (0.25), not that 1.0 — else every point in a
    coefficient sweep reads identical."""
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=0.25)
    (c0,) = rs.build_lens_registry([cond], tmp_path / "reg").conditions
    assert c0.effective_coefficient == 0.25


def test_vector_locus_fold_scales_the_tensor_and_leaves_coeff(tmp_path):
    cond = _condition(tmp_path, _vector_all_blob(), kind="vector_all",
                      coeff=3.0)
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    from sisteer.capeval.serving import lens_adapters as la
    orig = la.get_adapter("vector_all")({"kind": "vector_all"},
                                        _vector_all_blob(), b"")
    art = (Path(res.registry_dir) / "cell0" / "artifact.bin").read_bytes()
    got = la.get_adapter("vector_all")({"kind": "vector_all"}, art, b"")
    assert got[0].coeff == orig[0].coeff  # coeff stays put
    assert torch.allclose(got[0].vector, 3.0 * orig[0].vector)


def test_loreft_refuses_a_non_unit_coefficient(tmp_path):
    """loreft's affine edit does not factor through one scaled tensor, so
    a byte fold cannot reproduce c*edit — refuse rather than emit a wrong
    artifact."""
    # a minimal loreft-shaped blob is enough: the refusal precedes adapt.
    blob = _blob_bytes({"kind": "loreft", "layers": [5], "rank": 1,
                        "rotate": {}, "learned": {}})
    cond = _condition(tmp_path, blob, kind="loreft", coeff=0.5)
    with pytest.raises(rs.RegistryShimError, match="no linear coefficient"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_loreft_at_trained_strength_needs_no_fold(tmp_path):
    """c=1.0 is the identity fold, so loreft at its trained strength
    serves without ever hitting the unfoldable path."""
    from sisteer.vectors import SteeringVector  # noqa: F401 (import guard)
    # build a real loreft artifact via its class so adapt succeeds
    from sisteer.extraction.reft import LoReFTObject
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(hidden_size=D, intermediate_size=2 * D,
                      num_hidden_layers=6, num_attention_heads=2,
                      num_key_value_heads=2, vocab_size=32)
    model = LlamaForCausalLM(cfg)
    obj = LoReFTObject(model, layers=[2], rank=1)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "o.pt"
        obj.save(p)
        art = p.read_bytes()
    cond = _condition(tmp_path, art, kind="loreft", coeff=1.0)
    (c0,) = rs.build_lens_registry([cond], tmp_path / "reg").conditions
    assert c0.coefficient_was_folded is False


# --- the two shas, and the selector ------------------------------------------


def test_selector_binds_the_derived_intervention_hash(tmp_path):
    cond = _condition(tmp_path, _static_r_blob(0.125), kind="static_r",
                      coeff=0.25)
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    (c0,) = res.conditions
    art = (Path(res.registry_dir) / "cell0" / "artifact.bin").read_bytes()
    ih = cm.petri_intervention_hash({"kind": "static_r"}, artifact_bytes=art,
                                    sidecar_bytes=b"")
    assert c0.selector == {"petri_object": "cell0", "expect_hash": ih}


def test_original_sha_must_match_the_bytes_on_disk(tmp_path):
    art = _static_r_blob(1.0)
    path = _write(tmp_path, "cell0.pt", art)
    cond = {"cell_id": "cell0", "artifact_path": path,
            "resolved_spec": {"kind": "static_r"}, "coefficient": 1.0,
            "object_sha256": "deadbeefdeadbeef"}  # wrong
    with pytest.raises(rs.RegistryShimError, match="not the object"):
        rs.build_lens_registry([cond], tmp_path / "reg")


# --- the cross-phase (response) condition ------------------------------------


def test_response_condition_needs_a_header(tmp_path):
    cond = _condition(tmp_path, _vector_all_blob(), kind="vector_all",
                      coeff=1.0, spec_extra={"positions": "response"})
    with pytest.raises(rs.RegistryShimError, match="assistant_header_ids"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_response_condition_serves_with_a_header(tmp_path):
    cond = _condition(tmp_path, _vector_all_blob(), kind="vector_all",
                      coeff=1.0, spec_extra={"positions": "response"})
    res = rs.build_lens_registry([cond], tmp_path / "reg",
                                 assistant_header_ids=[128006, 78191,
                                                       128007, 271])
    assert res.assistant_header_ids == (128006, 78191, 128007, 271)


# --- contract hygiene ---------------------------------------------------------


def test_missing_required_key_refuses(tmp_path):
    with pytest.raises(rs.RegistryShimError, match="missing required"):
        rs.build_lens_registry([{"cell_id": "x"}], tmp_path / "reg")


def test_unknown_key_refuses_rather_than_ignoring(tmp_path):
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=1.0)
    cond["typo_field"] = 3
    with pytest.raises(rs.RegistryShimError, match="unknown key"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_duplicate_cell_id_refuses(tmp_path):
    art_a = _static_r_blob(1.0, seed=1)
    art_b = _static_r_blob(1.0, seed=2)
    a = {"cell_id": "dup", "artifact_path": _write(tmp_path, "a.pt", art_a),
         "resolved_spec": {"kind": "static_r"}, "coefficient": 1.0,
         "object_sha256": cm.sha256_hex16(art_a)}
    b = {"cell_id": "dup", "artifact_path": _write(tmp_path, "b.pt", art_b),
         "resolved_spec": {"kind": "static_r"}, "coefficient": 1.0,
         "object_sha256": cm.sha256_hex16(art_b)}
    with pytest.raises(rs.RegistryShimError, match="duplicate cell_id"):
        rs.build_lens_registry([a, b], tmp_path / "reg")


def test_non_finite_coefficient_refuses(tmp_path):
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=float("inf"))
    with pytest.raises(rs.RegistryShimError, match="finite"):
        rs.build_lens_registry([cond], tmp_path / "reg")


# --- expect_served_coeff: the convention cross-check --------------------------


def test_expect_served_coeff_passes_when_the_fold_agrees(tmp_path):
    """base_scale 0.125, c=0.25 -> served coeff 0.03125. A driver that
    computed the same absolute independently passes."""
    cond = _condition(tmp_path, _static_r_blob(0.125), kind="static_r",
                      coeff=0.25)
    cond["expect_served_coeff"] = 0.25 * 0.125
    (c0,) = rs.build_lens_registry([cond], tmp_path / "reg").conditions
    assert c0.effective_coefficient == 0.25


def test_expect_served_coeff_catches_a_replace_vs_multiply_mismatch(tmp_path):
    """A driver on the REPLACE convention would declare 0.25 (the bare
    coefficient) as the served coeff; the multiply fold produces 0.03125.
    The cross-check must refuse — this is the whole point of the guard,
    the one check that can go red on a convention mismatch."""
    cond = _condition(tmp_path, _static_r_blob(0.125), kind="static_r",
                      coeff=0.25)
    cond["expect_served_coeff"] = 0.25  # the replace-convention value
    with pytest.raises(rs.RegistryShimError, match="convention mismatch"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_expect_served_coeff_refuses_on_a_distributed_magnitude_kind(tmp_path):
    """vector_all has no scalar coefficient for the driver to declare, so
    the field is meaningless there and refuses rather than pretending."""
    cond = _condition(tmp_path, _vector_all_blob(), kind="vector_all",
                      coeff=2.0)
    cond["expect_served_coeff"] = 2.0
    with pytest.raises(rs.RegistryShimError, match="single scalar"):
        rs.build_lens_registry([cond], tmp_path / "reg")


# --- non-lens kinds route elsewhere ------------------------------------------


def test_soft_token_kind_refuses_with_a_pointer_to_its_path(tmp_path):
    """token is served via prompt-embeds, not the lens hook. The shim
    refuses it upfront, naming the right path and the scale-0 embedding
    hazard, rather than failing obscurely at adapt time."""
    blob = _blob_bytes({"kind": "token", "param": torch.zeros(D), "layer": 0})
    cond = _condition(tmp_path, blob, kind="token", coeff=1.0)
    with pytest.raises(rs.RegistryShimError, match="soft_token"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_lora_kind_builds_and_folds_on_the_lens_path(tmp_path):
    """2026-08-28: the o_proj module site landed, so the lora kinds
    build through the shim (the old refusal-with-export-pointer test is
    retired with the refusal). The fold scales the WRITE basis alone:
    (x @ A) @ (cB)^T = c * the original edit."""
    g = torch.Generator().manual_seed(11)
    A, B = torch.randn(D, generator=g), torch.randn(D, generator=g)
    blob = _blob_bytes({"kind": "lora", "A": A, "B": B, "layer": 0})
    cond = _condition(tmp_path, blob, kind="lora", coeff=0.25)
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    assert res.conditions[0].coefficient_was_folded
    from sisteer.capeval.serving.lens_steer import WorkerObjectRegistry
    reg = WorkerObjectRegistry.load(Path(res.registry_dir))
    (op,) = reg.objects()[0].ops
    assert op.op == la.OP_LORA_OPROJ
    assert torch.allclose(op.lowrank_b, 0.25 * B.reshape(-1, 1))
    assert torch.equal(op.lowrank_a, A.reshape(1, -1))  # read basis fixed


# --- antidote: the coefficient lives in the sidecar --------------------------


def _antidote_artifact(base_alpha: float, seed: int = 3):
    """A minimal antidote object: a SteeringVector safetensors container
    plus its JSON sidecar carrying method + metadata.base_alpha (absence
    of which is a hard refusal in the adapter)."""
    from safetensors.torch import save as st_save
    g = torch.Generator().manual_seed(seed)
    art = st_save({"layer_5": torch.randn(D, generator=g)})
    side = json.dumps({"method": "antidote", "layers": [5],
                       "metadata": {"base_alpha": base_alpha}}).encode("utf-8")
    return art, side


def test_antidote_folds_coefficient_into_the_sidecar(tmp_path):
    """base_alpha 8.0, c=0.25 -> served coeff 2.0, via the SIDECAR; the
    artifact bytes are untouched, so the served identity must span both
    parts or it would read identical to the fit."""
    art, side = _antidote_artifact(8.0)
    art_p = _write(tmp_path, "cell0.bin", art)
    side_p = _write(tmp_path, "cell0.json", side)
    cond = {"cell_id": "cell0", "artifact_path": art_p,
            "resolved_spec": {"kind": "antidote"}, "coefficient": 0.25,
            "object_sha256": cm.sha256_hex16(art), "sidecar_path": side_p,
            "sidecar_sha256": cm.sha256_hex16(side)}
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    (c0,) = res.conditions
    # the artifact is unchanged, but the served identity still moved
    assert c0.derived_sha256 != c0.original_sha256
    assert c0.coefficient_was_folded is True
    # and the served op carries 0.25*8.0, with base_alpha PRESERVED (a
    # dropped base_alpha would make the object refuse to load at all)
    from sisteer.capeval.serving import lens_adapters as la
    d_art = (Path(res.registry_dir) / "cell0" / "artifact.bin").read_bytes()
    d_side = (Path(res.registry_dir) / "cell0" / "sidecar.bin").read_bytes()
    (op,) = la.get_adapter("antidote")({"kind": "antidote"}, d_art, d_side)
    assert op.coeff == pytest.approx(0.25 * 8.0)


def test_antidote_expect_served_coeff_cross_checks_the_sidecar_fold(tmp_path):
    art, side = _antidote_artifact(8.0)
    art_p = _write(tmp_path, "cell0.bin", art)
    side_p = _write(tmp_path, "cell0.json", side)
    cond = {"cell_id": "cell0", "artifact_path": art_p,
            "resolved_spec": {"kind": "antidote"}, "coefficient": 0.25,
            "object_sha256": cm.sha256_hex16(art), "sidecar_path": side_p,
            "sidecar_sha256": cm.sha256_hex16(side),
            "expect_served_coeff": 0.25 * 8.0}
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    assert res.conditions[0].effective_coefficient == 0.25
    # the replace-convention value (0.25) would refuse
    cond["expect_served_coeff"] = 0.25
    with pytest.raises(rs.RegistryShimError, match="convention mismatch"):
        rs.build_lens_registry([cond], tmp_path / "reg2")


# --- the sidecar is hash-bound, like the artifact -----------------------------


def test_sidecar_without_its_hash_refuses(tmp_path):
    """sidecar_path without sidecar_sha256: the sidecar carries served
    semantics (antidote's base_alpha), so unbound bytes are an
    unverified half of the object — refused up front."""
    art, side = _antidote_artifact(8.0)
    cond = {"cell_id": "cell0",
            "artifact_path": _write(tmp_path, "cell0.bin", art),
            "resolved_spec": {"kind": "antidote"}, "coefficient": 0.25,
            "object_sha256": cm.sha256_hex16(art),
            "sidecar_path": _write(tmp_path, "cell0.json", side)}
    with pytest.raises(rs.RegistryShimError, match="no sidecar_sha256"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_sidecar_hash_mismatch_refuses_before_adapting(tmp_path):
    """The sidecar on disk is NOT the bytes the condition names — and it
    is garbage that would crash the antidote fold if it were ever
    consumed, so the refusal message proves the hash check fired FIRST
    (verify before adapt, same as the artifact)."""
    art, side = _antidote_artifact(8.0)
    cond = {"cell_id": "cell0",
            "artifact_path": _write(tmp_path, "cell0.bin", art),
            "resolved_spec": {"kind": "antidote"}, "coefficient": 0.25,
            "object_sha256": cm.sha256_hex16(art),
            # hash of the REAL sidecar, but garbage bytes on disk:
            "sidecar_path": _write(tmp_path, "cell0.json",
                                   b"\x00not json at all"),
            "sidecar_sha256": cm.sha256_hex16(side)}
    with pytest.raises(rs.RegistryShimError,
                       match="not the ones the condition names"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_sidecar_hash_without_a_path_refuses(tmp_path):
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=1.0)
    cond["sidecar_sha256"] = "ab" * 8
    with pytest.raises(rs.RegistryShimError, match="sidecar_sha256 without"):
        rs.build_lens_registry([cond], tmp_path / "reg")


def test_missing_sidecar_file_refuses(tmp_path):
    art, side = _antidote_artifact(8.0)
    cond = {"cell_id": "cell0",
            "artifact_path": _write(tmp_path, "cell0.bin", art),
            "resolved_spec": {"kind": "antidote"}, "coefficient": 0.25,
            "object_sha256": cm.sha256_hex16(art),
            "sidecar_path": str(tmp_path / "nope.json"),
            "sidecar_sha256": cm.sha256_hex16(side)}
    with pytest.raises(rs.RegistryShimError, match="no sidecar at"):
        rs.build_lens_registry([cond], tmp_path / "reg")


# --- zero / negative coefficients refuse --------------------------------------


@pytest.mark.parametrize("bad", [0.0, 0, -1.0, -0.25])
def test_zero_and_negative_coefficients_refuse(tmp_path, bad):
    """c == 0 folds every op to an exact no-op (a passthrough served
    under an arm's identity); c < 0 serves the NEGATED edit. Neither is
    a sweep point; both refuse loudly."""
    cond = _condition(tmp_path, _static_r_blob(0.5), kind="static_r",
                      coeff=bad)
    with pytest.raises(rs.RegistryShimError, match="strictly positive"):
        rs.build_lens_registry([cond], tmp_path / "reg")


# --- the vector-locus fold verifier checks the WHOLE op -----------------------


def _psr_blob(scale_gate_by=None, seed=5) -> bytes:
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(D, generator=g)
    w = torch.randn(D, generator=g)
    b = torch.full((1,), 0.5)
    if scale_gate_by is not None:
        w = w * scale_gate_by
        b = b * scale_gate_by
    return _blob_bytes({"kind": "psr", "layers": [3],
                        "directions": {"3": d}, "gate_w": {"3": w},
                        "gate_b": {"3": b}})


def test_vector_locus_verifier_rejects_a_fold_that_scaled_the_gate(tmp_path):
    """A fold that scaled gate_w/gate_b along with the direction changes
    the op's MATH (which tokens fire), not just its magnitude — the old
    verifier looked only at op.coeff and op.vector and would have passed
    it. It must refuse, naming the moved tensor."""
    c = 2.0
    orig = _psr_blob()
    # the WRONG derived artifact: directions scaled (correct) AND the
    # gate scaled (the defect)
    blob = torch.load(io.BytesIO(orig), map_location="cpu",
                      weights_only=True)
    blob["directions"] = {k: v * c for k, v in blob["directions"].items()}
    blob["gate_w"] = {k: v * c for k, v in blob["gate_w"].items()}
    bad_deriv = _blob_bytes(blob)
    with pytest.raises(rs.RegistryShimError, match="only 'vector' may move"):
        rs._verify_fold("psr", {"kind": "psr"}, orig, b"", bad_deriv, b"",
                        c, None)


def test_vector_locus_verifier_rejects_an_op_structure_change(tmp_path):
    """A fold that also flipped a non-tensor parameter (skip_bos here)
    serves a DIFFERENT op under the folded identity — refused via the
    op-signature check, which the vector-locus branch previously never
    ran."""
    c = 2.0
    orig = _vector_all_blob()
    blob = torch.load(io.BytesIO(orig), map_location="cpu",
                      weights_only=True)
    blob["vectors"] = {k: v * c for k, v in blob["vectors"].items()}
    blob["skip_bos"] = False  # structure change riding along
    bad_deriv = _blob_bytes(blob)
    with pytest.raises(rs.RegistryShimError, match="op structure"):
        rs._verify_fold("vector_all", {"kind": "vector_all"}, orig, b"",
                        bad_deriv, b"", c, None)


def test_vector_locus_verifier_still_passes_a_correct_fold():
    c = 2.0
    orig = _psr_blob()
    blob = torch.load(io.BytesIO(orig), map_location="cpu",
                      weights_only=True)
    blob["directions"] = {k: v * c for k, v in blob["directions"].items()}
    rs._verify_fold("psr", {"kind": "psr"}, orig, b"", _blob_bytes(blob),
                    b"", c, None)  # no raise


def test_vector_kind_safetensors_fold_passes_the_hardened_verifier(tmp_path):
    """Kind 'vector' (the safetensors SteeringVector container, sidecar
    hash-bound): a straight c-fold of the layer tensors still passes the
    vector-locus verifier now that it also checks op signature and
    non-vector tensors."""
    from safetensors.torch import load as st_load, save as st_save
    g = torch.Generator().manual_seed(7)
    art = st_save({"layer_4": torch.randn(D, generator=g)})
    side = json.dumps({"method": "distill_soft", "model_name": "m",
                       "instruction_id": "i", "layers": [4],
                       "metadata": {}}).encode("utf-8")
    cond = {"cell_id": "cell0",
            "artifact_path": _write(tmp_path, "cell0.st", art),
            "resolved_spec": {"kind": "vector"}, "coefficient": 2.0,
            "object_sha256": cm.sha256_hex16(art),
            "sidecar_path": _write(tmp_path, "cell0.json", side),
            "sidecar_sha256": cm.sha256_hex16(side)}
    res = rs.build_lens_registry([cond], tmp_path / "reg")
    (c0,) = res.conditions
    assert c0.coefficient_was_folded is True
    d_art = (Path(res.registry_dir) / "cell0" / "artifact.bin").read_bytes()
    orig_t = st_load(art)["layer_4"]
    deriv_t = st_load(d_art)["layer_4"]
    assert torch.allclose(deriv_t, 2.0 * orig_t)


# --- rebuilds: immutable versioned builds behind an atomic pointer ------------


def _load_cell_ids(path):
    from sisteer.capeval.serving import lens_steer as ls
    return ls.WorkerObjectRegistry.load(rs.resolve_registry_dir(path)
                                        ).cell_ids()


def _no_staging_junk(root):
    """After a SUCCESSFUL build, no dot-staged build dirs or pointer
    temps survive (crashed ones may — they are unreachable junk by
    design, asserted separately)."""
    builds = root / rs.REGISTRY_BUILDS
    leftover = [p.name for p in builds.iterdir()
                if p.name.startswith(".building-")]
    leftover += [p.name for p in root.iterdir()
                 if p.name.startswith(f".{rs.REGISTRY_POINTER}.tmp-")]
    return leftover == []


def test_rebuild_publishes_a_new_build_and_keeps_the_old_intact(tmp_path):
    """The reader-concurrency contract: a rebuild publishes a NEW
    immutable build via the pointer; the old build a reader already
    resolved stays byte-identical and loadable forever — nothing is
    renamed away or deleted underneath anyone. (The pre-pointer swap
    moved the old dir aside and deleted it, so a concurrent reader's
    canonical path could vanish mid-read.)"""
    from sisteer.capeval.serving import lens_steer as ls

    root = tmp_path / "reg"
    a = _condition(tmp_path, _static_r_blob(0.5, seed=1), kind="static_r",
                   coeff=1.0, cell_id="cell_a")
    res_a = rs.build_lens_registry([a], root)
    dir_a = Path(res_a.registry_dir)
    assert res_a.root_dir == str(root)
    assert dir_a == rs.resolve_registry_dir(root)  # pointer names build A
    index_a = (dir_a / rs.REGISTRY_INDEX).read_bytes()

    b = _condition(tmp_path, _static_r_blob(0.5, seed=2), kind="static_r",
                   coeff=1.0, cell_id="cell_b")
    res_b = rs.build_lens_registry([b], root)
    dir_b = Path(res_b.registry_dir)
    assert dir_b != dir_a  # a NEW immutable build, not an edit
    # the root now resolves to B ...
    assert rs.resolve_registry_dir(root) == dir_b
    assert _load_cell_ids(root) == ("cell_b",)
    # ... while the reader that resolved A keeps a fully intact registry
    assert (dir_a / rs.REGISTRY_INDEX).read_bytes() == index_a
    assert ls.WorkerObjectRegistry.load(dir_a).cell_ids() == ("cell_a",)
    assert _no_staging_junk(root)


def test_rebuild_over_identical_conditions_succeeds(tmp_path):
    """A retry of the same build must not trip over the previous build's
    read-only files — it lands as a fresh versioned build."""
    root = tmp_path / "reg"
    cond = _condition(tmp_path, _static_r_blob(0.125), kind="static_r",
                      coeff=0.25)
    first = rs.build_lens_registry([cond], root)
    second = rs.build_lens_registry([cond], root)
    assert first.conditions == second.conditions
    assert _load_cell_ids(root) == ("cell0",)
    assert _no_staging_junk(root)


def test_failed_rebuild_leaves_the_published_registry_untouched(tmp_path):
    """A rebuild that refuses partway (wrong object_sha256 here) fails
    inside its dot-staged build dir: the pointer still names the old
    build, which is byte-for-byte intact — there is never an instant
    with no canonical registry."""
    root = tmp_path / "reg"
    good = _condition(tmp_path, _static_r_blob(0.5, seed=1), kind="static_r",
                      coeff=1.0, cell_id="cell_a")
    res = rs.build_lens_registry([good], root)
    dir_a = Path(res.registry_dir)
    index_before = (dir_a / rs.REGISTRY_INDEX).read_bytes()
    bad = _condition(tmp_path, _static_r_blob(0.5, seed=2), kind="static_r",
                     coeff=1.0, cell_id="cell_b")
    bad["object_sha256"] = "deadbeefdeadbeef"
    with pytest.raises(rs.RegistryShimError, match="not the object"):
        rs.build_lens_registry([bad], root)
    assert rs.resolve_registry_dir(root) == dir_a
    assert _load_cell_ids(root) == ("cell_a",)
    assert (dir_a / rs.REGISTRY_INDEX).read_bytes() == index_before
    assert _no_staging_junk(root)


def test_crash_during_pointer_publish_recovers(tmp_path, monkeypatch):
    """Forced interruption at the worst spot — the build is complete and
    verified but the pointer publish dies. The old pointer (and build)
    survive untouched, resolution never sees the unpublished build, and
    a plain retry publishes cleanly."""
    root = tmp_path / "reg"
    a = _condition(tmp_path, _static_r_blob(0.5, seed=1), kind="static_r",
                   coeff=1.0, cell_id="cell_a")
    dir_a = Path(rs.build_lens_registry([a], root).registry_dir)

    real_replace = rs.os.replace

    def dying_replace(src, dst):
        raise OSError("simulated crash mid-publish")

    b = _condition(tmp_path, _static_r_blob(0.5, seed=2), kind="static_r",
                   coeff=1.0, cell_id="cell_b")
    monkeypatch.setattr(rs.os, "replace", dying_replace)
    with pytest.raises(OSError, match="simulated crash"):
        rs.build_lens_registry([b], root)
    monkeypatch.setattr(rs.os, "replace", real_replace)
    # the canonical registry never wavered:
    assert rs.resolve_registry_dir(root) == dir_a
    assert _load_cell_ids(root) == ("cell_a",)
    # the retry publishes a fresh build; the orphaned one is inert junk
    res = rs.build_lens_registry([b], root)
    assert rs.resolve_registry_dir(root) == Path(res.registry_dir)
    assert _load_cell_ids(root) == ("cell_b",)


def test_concurrent_readers_never_see_a_missing_or_torn_registry(tmp_path):
    """Reader concurrency, live: one thread rebuilds the root repeatedly
    while readers resolve + fully load in a loop. Every single
    resolve+load must succeed — under the pre-pointer swap there was a
    window with NO registry at the canonical path, which this loop
    reliably fell into."""
    import threading

    root = tmp_path / "reg"
    conds = [
        _condition(tmp_path, _static_r_blob(0.5, seed=s), kind="static_r",
                   coeff=1.0, cell_id=f"cell_{s}")
        for s in (1, 2, 3)]
    rs.build_lens_registry([conds[0]], root)

    stop = threading.Event()
    build_errors: list = []

    def rebuild_loop():
        try:
            for _ in range(4):
                for c in conds:
                    rs.build_lens_registry([c], root)
        except BaseException as e:  # pragma: no cover — a test failure
            build_errors.append(e)
        finally:
            stop.set()

    from sisteer.capeval.serving import lens_steer as ls
    reader_errors: list = []
    n_reads = 0
    t = threading.Thread(target=rebuild_loop)
    t.start()
    try:
        while not stop.is_set():
            try:
                reg = ls.WorkerObjectRegistry.load(
                    rs.resolve_registry_dir(root))
                assert len(reg.cell_ids()) == 1
                n_reads += 1
            except BaseException as e:  # pragma: no cover — the defect
                reader_errors.append(e)
                break
    finally:
        t.join(timeout=120)
    assert not build_errors, build_errors
    assert not reader_errors, reader_errors
    assert n_reads > 0  # the loop genuinely read during the rebuilds


def test_durability_ordering_of_a_publish(tmp_path, monkeypatch):
    """The fsync/rename/replace ORDER is the crash-safety argument, so
    it is pinned mechanically: (1) every file and dir of the staged
    build (bottom-up, build dir included) syncs BEFORE the
    staging->final rename; (2) builds/ syncs AFTER that rename and
    BEFORE the pointer replace (the rename lives in builds/'s entries);
    (3) the pointer temp syncs before the replace and the root after
    it. Recorded by wrapping the module's own os calls — behavior
    unchanged, order observed."""
    events = []
    fd_paths = {}
    real = {name: getattr(rs.os, name)
            for name in ("open", "fsync", "rename", "replace")}

    def rec_open(path, flags, *a, **k):
        fd = real["open"](path, flags, *a, **k)
        fd_paths[fd] = str(path)
        return fd

    def rec_fsync(fd):
        events.append(("fsync", fd_paths.get(fd, f"fd:{fd}")))
        real["fsync"](fd)

    def rec_rename(src, dst):
        events.append(("rename", str(src), str(dst)))
        real["rename"](src, dst)

    def rec_replace(src, dst):
        events.append(("replace", str(src), str(dst)))
        real["replace"](src, dst)

    monkeypatch.setattr(rs.os, "open", rec_open)
    monkeypatch.setattr(rs.os, "fsync", rec_fsync)
    monkeypatch.setattr(rs.os, "rename", rec_rename)
    monkeypatch.setattr(rs.os, "replace", rec_replace)

    root = tmp_path / "reg"
    cond = _condition(tmp_path, _static_r_blob(0.5), kind="static_r",
                      coeff=1.0)
    res = rs.build_lens_registry([cond], root)

    i_rename = next(i for i, e in enumerate(events)
                    if e[0] == "rename" and ".building-" in e[1]
                    and e[2] == res.registry_dir)
    i_replace = next(i for i, e in enumerate(events)
                     if e[0] == "replace"
                     and e[2] == str(root / rs.REGISTRY_POINTER))
    staged_fsyncs = [i for i, e in enumerate(events)
                     if e[0] == "fsync" and ".building-" in e[1]]
    # (1) the whole staged tree synced, and synced BEFORE the rename:
    assert staged_fsyncs and all(i < i_rename for i in staged_fsyncs)
    staged_paths = {events[i][1] for i in staged_fsyncs}
    assert any(p.endswith("artifact.bin") for p in staged_paths)
    assert any(p.endswith(rs.REGISTRY_INDEX) for p in staged_paths)
    assert any(p.endswith("cell0") for p in staged_paths)      # nested dir
    assert any(".building-" in p and p.endswith(
        Path(p).name) and Path(p).name.startswith(".building-")
        for p in staged_paths)                                  # build dir
    # (2) builds/ synced after the rename, before the pointer replace:
    i_builds = next(i for i, e in enumerate(events)
                    if e[0] == "fsync"
                    and e[1] == str(root / rs.REGISTRY_BUILDS))
    assert i_rename < i_builds < i_replace
    # (3) pointer temp before the replace; root after it:
    i_tmp = next(i for i, e in enumerate(events)
                 if e[0] == "fsync"
                 and f".{rs.REGISTRY_POINTER}.tmp-" in e[1])
    assert i_tmp < i_replace
    i_root = next(i for i, e in enumerate(events)
                  if e[0] == "fsync" and e[1] == str(root))
    assert i_root > i_replace


def test_broken_pointer_refuses_loudly_and_a_rebuild_repairs_it(tmp_path):
    """The storage-crash fallback: a torn/orphaned CURRENT never
    resolves by guesswork (two orphaned builds would be ambiguous) —
    it refuses loudly, and the documented recovery is a rebuild into
    the same root, which publishes a fresh valid pointer over the
    broken one."""
    root = tmp_path / "reg"
    cond = _condition(tmp_path, _static_r_blob(0.5), kind="static_r",
                      coeff=1.0)
    rs.build_lens_registry([cond], root)
    pointer = root / rs.REGISTRY_POINTER
    # crash aftermath 1: pointer names a build the disk lost
    pointer.write_text("build-lost-to-a-crash\n")
    with pytest.raises(rs.RegistryShimError, match="disagree"):
        rs.resolve_registry_dir(root)
    # crash aftermath 2: torn/empty pointer content
    pointer.write_text("")
    with pytest.raises(rs.RegistryShimError, match="invalid build id"):
        rs.resolve_registry_dir(root)
    # recovery: a plain rebuild repairs the root
    res = rs.build_lens_registry([cond], root)
    assert rs.resolve_registry_dir(root) == Path(res.registry_dir)
    assert _load_cell_ids(root) == ("cell0",)


def test_non_registry_target_refuses_before_mutation(tmp_path):
    """A non-empty target that is not a registry root — replacing or
    writing into it could destroy files the shim does not own, so it
    refuses before touching anything."""
    target = tmp_path / "reg"
    target.mkdir()
    stray = target / "precious.txt"
    stray.write_text("not a registry")
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=1.0)
    with pytest.raises(rs.RegistryShimError, match="not its to touch"):
        rs.build_lens_registry([cond], target)
    assert stray.read_text() == "not a registry"  # untouched
    assert list(target.iterdir()) == [stray]      # nothing else appeared


def test_legacy_flat_registry_target_refuses_with_a_migration_note(tmp_path):
    """A FLAT registry (index at the root — the pre-pointer layout, or
    an export_worker_registry dir) cannot be mixed with the versioned
    layout; the refusal names the migration."""
    target = tmp_path / "reg"
    target.mkdir()
    (target / rs.REGISTRY_INDEX).write_text("{}")
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=1.0)
    with pytest.raises(rs.RegistryShimError, match="FLAT registry"):
        rs.build_lens_registry([cond], target)


def test_build_into_a_pre_created_empty_dir_still_works(tmp_path):
    reg = tmp_path / "reg"
    reg.mkdir()  # callers that mkdir first
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=1.0)
    rs.build_lens_registry([cond], reg)
    assert _load_cell_ids(reg) == ("cell0",)


def test_resolve_registry_dir_contract(tmp_path):
    """The consumer seam: a build dir resolves to itself (flat
    export_worker_registry dirs included), a root follows its pointer,
    and a broken pointer / non-registry path refuses loudly."""
    root = tmp_path / "reg"
    cond = _condition(tmp_path, _static_r_blob(1.0), kind="static_r",
                      coeff=1.0)
    res = rs.build_lens_registry([cond], root)
    build = Path(res.registry_dir)
    assert rs.resolve_registry_dir(root) == build
    assert rs.resolve_registry_dir(build) == build  # already resolved
    with pytest.raises(rs.RegistryShimError, match="nothing to serve"):
        rs.resolve_registry_dir(tmp_path / "nowhere")
    # a pointer naming a missing build refuses (pointer/builds disagree)
    (root / rs.REGISTRY_POINTER).write_text("build-nope\n")
    with pytest.raises(rs.RegistryShimError, match="disagree"):
        rs.resolve_registry_dir(root)
    # a pointer trying to escape builds/ refuses
    (root / rs.REGISTRY_POINTER).write_text("../evil\n")
    with pytest.raises(rs.RegistryShimError, match="invalid build id"):
        rs.resolve_registry_dir(root)


# --- the declared thinking-ON header in a
# --- shim-built registry, and the fold verifier covering the second header ---

from sisteer.capeval.serving import lens_steer as ls  # noqa: E402
from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402

OFF7 = (901, 902, 903, 904, 905, 906, 907)
ON5 = (901, 902, 903, 904, 908)


def _response_psr_blob(seed=8) -> bytes:
    g = torch.Generator().manual_seed(seed)
    return _blob_bytes({"kind": "psr", "layers": [0], "positions": "response",
                        "response_header_len": len(OFF7),
                        "directions": {"0": torch.randn(D, generator=g) * 0.05},
                        "gate_w": {"0": torch.randn(D, generator=g) * 0.3},
                        "gate_b": {"0": torch.randn(1, generator=g) * 0.5}})


def _thinking_condition(tmp, coeff=1.0, cell_id="cell_think"):
    return _condition(tmp, _response_psr_blob(), kind="psr", coeff=coeff, cell_id=cell_id,
                      spec_extra={**sd.apply_variant({"positions": "response"}, "thinking")})


def test_declaring_condition_records_and_attaches_the_on_header(tmp_path):
    res = rs.build_lens_registry([_thinking_condition(tmp_path)], tmp_path / "reg",
                                 assistant_header_ids=list(OFF7),
                                 assistant_header_ids_thinking_on=list(ON5))
    assert res.assistant_header_ids_thinking_on == ON5
    index = json.loads((Path(res.registry_dir) / rs.REGISTRY_INDEX).read_text())
    assert index[ls.REGISTRY_THINKING_HEADER_KEY] == list(ON5)
    loaded = ls.WorkerObjectRegistry.load(Path(res.registry_dir))
    assert loaded.serves_thinking_on()
    (obj,) = loaded.objects()
    assert all(op.thinking_header_ids == ON5 for op in obj.ops)
    assert loaded.serve_declarations_by_cell() == {"cell_think": {"serve_out_of_support": ["thinking"]}}


def test_declaring_condition_folds_with_the_second_header_in_the_signature(tmp_path):
    """c != 1 runs _verify_fold, which now adapts BOTH sides with the ON header
    attached — the op signature covers thinking_header_ids (and turn_end_ids)."""
    res = rs.build_lens_registry([_thinking_condition(tmp_path, coeff=0.5)], tmp_path / "reg",
                                 assistant_header_ids=list(OFF7),
                                 assistant_header_ids_thinking_on=list(ON5))
    (c0,) = res.conditions
    assert c0.coefficient_was_folded and c0.effective_coefficient == 0.5
    op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D), positions=la.POSITIONS_RESPONSE,
                   header_ids=OFF7, thinking_header_ids=ON5)
    assert rs._op_signature(op)[-3:-1] == (None, ON5)      # turn_end_ids, thinking_header_ids


def test_thinking_header_refusals(tmp_path):
    plain = _condition(tmp_path, _response_psr_blob(), kind="psr", coeff=1.0, cell_id="plain",
                       spec_extra={"positions": "response"})
    # given with no declaring condition
    with pytest.raises(rs.RegistryShimError, match="no condition declares"):
        rs.build_lens_registry([plain], tmp_path / "r1", assistant_header_ids=list(OFF7),
                               assistant_header_ids_thinking_on=list(ON5))
    # declaring, but neither ids nor a tokenizer
    with pytest.raises(rs.RegistryShimError, match="no assistant_header_ids_thinking_on"):
        rs.build_lens_registry([_thinking_condition(tmp_path)], tmp_path / "r2",
                               assistant_header_ids=list(OFF7))
    # equal to the OFF header: the header-INVARIANT template's case (gemma-4) — built,
    # the equal header recorded as the second (P9 withdrawn)
    res = rs.build_lens_registry([_thinking_condition(tmp_path)], tmp_path / "r3",
                                 assistant_header_ids=list(OFF7),
                                 assistant_header_ids_thinking_on=list(OFF7))
    assert tuple(res.assistant_header_ids_thinking_on) == tuple(OFF7)
    # a malformed declaration in the spec refuses before any bytes are read
    bad = _condition(tmp_path, _response_psr_blob(), kind="psr", coeff=1.0, cell_id="bad",
                     spec_extra={"positions": "response", sd.SERVE_OUT_OF_SUPPORT_KEY: "thinking"})
    with pytest.raises(rs.RegistryShimError, match="malformed serve declaration"):
        rs.build_lens_registry([bad], tmp_path / "r4", assistant_header_ids=list(OFF7))
    # a plain registry records no second header
    res = rs.build_lens_registry([plain], tmp_path / "r5", assistant_header_ids=list(OFF7))
    assert res.assistant_header_ids_thinking_on is None
    assert json.loads((Path(res.registry_dir) / rs.REGISTRY_INDEX).read_text())[
        ls.REGISTRY_THINKING_HEADER_KEY] is None


def test_thinking_header_derives_from_the_real_qwen_tokenizer(tmp_path):
    """Cache-permitting: the shim derives the ON header from the served
    tokenizer for a declaring condition (Qwen3.5-4B: 7 -> 5)."""
    import os
    transformers = pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pinned Qwen3.5-4B tokenizer not cached ({type(e).__name__})")
    from sisteer.models import pin_reasoning_template_off
    pin_reasoning_template_off(tok)
    res = rs.build_lens_registry([_thinking_condition(tmp_path)], tmp_path / "reg", tokenizer=tok)
    assert len(res.assistant_header_ids) == 7 and len(res.assistant_header_ids_thinking_on) == 5
    assert res.assistant_header_ids_thinking_on[:4] == res.assistant_header_ids[:4]


# --- one unwrapper ------------------------------------------------------------


def test_the_stacked_unwrapper_is_one_implementation():
    """``registry_shim._served_positions_of`` (the marker gates' read of a
    condition's served mode) delegates to ``defense_cells.served_positions_of``
    — the function the proxy's routing-table stamp reads the same fact
    through (``proxy.lens_served_positions``, the thinking-ON gate's
    header-locatedness) — so a stacked cell can never unwrap two ways, and the
    wrapper kind both read is the one ``stacked_resolved_spec`` writes."""
    from sisteer.capeval.petri import defense_cells as dc
    assert la.STACKED_KIND == cm.PETRI_STACKED_KIND
    wrapped = {"kind": la.STACKED_KIND, "si_id": "safety", "si_prefix_sha16": "0" * 16,
               "object": {"kind": "vector_all", "positions": "response_all_turns"}}
    for spec in (wrapped, {"positions": "response"}, {"kind": "vector_all"}, None,
                 {"kind": la.STACKED_KIND}, {**wrapped, "positions": "all"}):
        assert rs._served_positions_of(spec) == dc.served_positions_of(spec)
    assert rs._served_positions_of(wrapped) == la.POSITIONS_RESPONSE_ALL_TURNS
    assert rs._served_positions_of({**wrapped, "positions": "all"}) == la.POSITIONS_RESPONSE_ALL_TURNS
    assert rs._served_positions_of({"kind": la.STACKED_KIND}) is None
    assert rs._served_positions_of({"positions": "response"}) == la.POSITIONS_RESPONSE
    assert rs._served_positions_of(None) is None
    # the two deployment-shape vocabularies that state a served mask WITHOUT the `positions` key — the
    # Arditi `static_r` blob's `steer_positions` and the AlphaSteer blob's — read as the mode the thinking-ON
    # gate classifies; a stated `positions` still wins; an unknown value or kind reads None
    assert dc.SERVED_MODE_BY_STEER_POSITIONS == {"static_r": {"prefill_all": "all", "prompt_last": "last"}, "alphasteer": {"prefill": "all"}}
    assert dc.served_positions_of({"kind": "static_r", "steer_positions": "prefill_all", "site": "pre"}) == la.POSITIONS_ALL
    assert dc.served_positions_of({"kind": "static_r", "steer_positions": "prompt_last", "site": "post"}) == "last"
    assert dc.served_positions_of({"kind": "alphasteer", "steer_positions": "prefill", "site": "pre"}) == la.POSITIONS_ALL
    assert dc.served_positions_of({"kind": "alphasteer", "steer_positions": "decode"}) is None
    assert dc.served_positions_of({"kind": "static_r"}) is None and dc.served_positions_of({"kind": "lora_r4_ce", "steer_positions": "prefill"}) is None
    assert dc.served_positions_of({"kind": "static_r", "steer_positions": "prefill_all", "positions": "response"}) == la.POSITIONS_RESPONSE
    stacked_steer = {"kind": la.STACKED_KIND, "object": {"kind": "alphasteer", "steer_positions": "prefill"}}
    assert dc.served_positions_of(stacked_steer) == la.POSITIONS_ALL == rs._served_positions_of(stacked_steer)
