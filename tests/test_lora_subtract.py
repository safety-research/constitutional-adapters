"""LoRA task-vector subtraction objects (sdf-base-steering round-2,
user directive 2026-09-02).

Pins: the ``lora_subtraction`` artifact loads through the ordinary
``load_object`` path; both parents' factors ride bit-exact (so
"subtraction then re-adding control" — dropping the negative term —
recovers the corpus fit's deltas BITWISE); the hook algebra is
Δ_sub = Δ_mix − Δ_ctl and honors scale linearly; every same-shape
precondition refuses loudly; the builder script writes derived/ artifacts
with honest provenance (parent cell ids + shas + data scales, the
scale-matched/budget-mismatched labeling) and refuses torn or
wrong-identity parents.

The object tests ride the CPU tiny model; the builder tests are pure
tensor bookkeeping (no model at all — exactly how the script runs).
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import lora_subtract as ls  # noqa: E402
import parent_admission as pa  # noqa: E402
# The seeded rank-4 LoRA builder and the capture-inside-apply helper are
# test_scale_semantics' (the same objects, hooked the same way).
from test_scale_semantics import _captured_out, _lora_obj  # noqa: E402


def test_builders_share_one_parent_admission_gate():
    """The /simplify observation: lora_subtract and psr_residual_distill
    admit parents through ONE resolver (parent_admission.admit_parent —
    store-identity fallback for a null model_revision, revision source
    recorded, the shared cross-parent match), not two copies."""
    import importlib.util

    assert ls.admit_parent is pa.admit_parent
    assert ls.resolve_model_revision is pa.resolve_model_revision
    assert ls.STORE_IDENT == pa.STORE_IDENT == "_store_identity.json"
    assert ls.pool_budget is pa.pool_budget
    spec = importlib.util.spec_from_file_location(
        "_prd_for_admission", REPO / "scripts/sdfbase/psr_residual_distill.py")
    prd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prd)
    assert prd.admit_parent is pa.admit_parent
    assert prd.pool_budget is pa.pool_budget
    assert prd.PARENT_MATCH_FIELDS is pa.PARENT_MATCH_FIELDS
    for src in (REPO / "scripts/sdfbase/lora_subtract.py",
                REPO / "scripts/sdfbase/psr_residual_distill.py"):
        text = src.read_text()
        assert "def resolve_model_revision" not in text, src
        assert "staged_cell_problems(" not in text, src  # the gate's, not a copy


# --------------------------------------------------------------------------
# Object-level: compose, load, algebra, scale (tiny model)
# --------------------------------------------------------------------------


def _oproj_out(model, tok, obj=None, scale=1.0):
    """Layer-0 o_proj OUTPUT under an optional object (captured inside the
    apply ctx, after the object's own hooks, so it sees the edited tensor)."""
    from sisteer.models import get_decoder_layers

    return _captured_out(model, tok,
                         get_decoder_layers(model)[0].self_attn.o_proj,
                         obj, scale)


def _logits(model, tok, obj=None, scale=1.0):
    import torch

    enc = tok("the quick brown fox jumps", return_tensors="pt")
    ctx = obj.apply(model, scale=scale) if obj is not None else nullcontext()
    with torch.no_grad(), ctx:
        return model(**enc).logits.detach().clone()


@pytest.fixture(scope="module")
def sub_world(tiny_llama, tmp_path_factory):
    import torch

    from sisteer.extraction.multilayer import LoRASubtractionObject
    from sisteer.methods import load_object

    model, tok = tiny_llama
    root = tmp_path_factory.mktemp("lorasub")
    mix = _lora_obj(model, seed=11)
    ctl = _lora_obj(model, seed=22)
    mix.save(root / "mix.pt")
    ctl.save(root / "ctl.pt")
    blob = LoRASubtractionObject.compose_blob(
        torch.load(root / "mix.pt", map_location="cpu", weights_only=True),
        torch.load(root / "ctl.pt", map_location="cpu", weights_only=True),
        provenance={"parent_pos": {"cell_id": "constv2-mix-100k__lora_r4_ce"},
                    "parent_neg": {"cell_id": "constv2ctl-100k__lora_r4_ce"}})
    torch.save(blob, root / "sub.pt")
    sub = load_object(root / "sub.pt", model=model, tok=tok)
    import types

    return types.SimpleNamespace(model=model, tok=tok, mix=mix, ctl=ctl,
                                 sub=sub, root=root, blob=blob)


def test_load_object_round_trip_and_kind(sub_world):
    from sisteer.extraction.multilayer import LoRASubtractionObject

    sub = sub_world.sub
    assert isinstance(sub, LoRASubtractionObject)
    assert sub.kind == "lora_subtraction"
    assert sub.rank == 4
    assert sub.layers == sub_world.mix.layers
    assert sub.provenance["parent_pos"]["cell_id"] == \
        "constv2-mix-100k__lora_r4_ce"
    assert sub.trainable == []  # derived — nothing may train it further
    assert sub.n_params() == 0


def test_parent_factors_ride_bit_exact(sub_world):
    import torch

    for l in sub_world.mix.layers:
        assert torch.equal(sub_world.sub.pos.A[l], sub_world.mix.A[l])
        assert torch.equal(sub_world.sub.pos.B[l], sub_world.mix.B[l])
        assert torch.equal(sub_world.sub.neg.A[l], sub_world.ctl.A[l])
        assert torch.equal(sub_world.sub.neg.B[l], sub_world.ctl.B[l])


def test_readding_control_recovers_corpus_deltas_bitwise(sub_world):
    """The algebra pin, in the only form that is EXACT in floating point:
    re-adding the control cancels the negative term STRUCTURALLY, leaving
    the positive parent — which serves through MultiLayerLoRAObject's own
    op sequence on bit-identical tensors, so full-model logits under the
    reconstruction equal the original corpus fit's logits bitwise."""
    import torch

    from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                               MultiLayerLoRAObject)

    pos_only = MultiLayerLoRAObject.load_blob(
        LoRASubtractionObject._parent_blob(sub_world.blob, "pos"),
        sub_world.model, sub_world.tok)
    lg_mix = _logits(sub_world.model, sub_world.tok, sub_world.mix)
    lg_rec = _logits(sub_world.model, sub_world.tok, pos_only)
    assert torch.equal(lg_mix, lg_rec)


def test_hook_algebra_is_mix_minus_ctl(sub_world):
    import torch

    m, t = sub_world.model, sub_world.tok
    base = _oproj_out(m, t)
    d_mix = _oproj_out(m, t, sub_world.mix) - base
    d_ctl = _oproj_out(m, t, sub_world.ctl) - base
    d_sub = _oproj_out(m, t, sub_world.sub) - base
    assert torch.allclose(d_sub, d_mix - d_ctl, atol=1e-5)
    assert d_sub.abs().max() > 0


def test_scale_scales_the_task_vector_linearly(sub_world):
    import torch

    m, t = sub_world.model, sub_world.tok
    base = _oproj_out(m, t)
    d1 = _oproj_out(m, t, sub_world.sub, scale=1.0) - base
    d2 = _oproj_out(m, t, sub_world.sub, scale=2.0) - base
    assert d1.abs().max() > 0
    assert torch.allclose(d2, 2.0 * d1, atol=1e-5)
    # scale 0 is the exact no-op
    d0 = _oproj_out(m, t, sub_world.sub, scale=0.0) - base
    assert torch.equal(d0, torch.zeros_like(d0))


def test_compose_blob_refusals(sub_world):
    import torch

    from sisteer.extraction.multilayer import LoRASubtractionObject

    mix_blob = torch.load(sub_world.root / "mix.pt", map_location="cpu",
                          weights_only=True)
    ctl_blob = torch.load(sub_world.root / "ctl.pt", map_location="cpu",
                          weights_only=True)
    with pytest.raises(ValueError, match="kind"):
        LoRASubtractionObject.compose_blob({**mix_blob, "kind": "psr"},
                                           ctl_blob)
    bad_rank = dict(ctl_blob)
    bad_rank["A"] = {k: v[:, :2].clone() for k, v in ctl_blob["A"].items()}
    bad_rank["B"] = {k: v[:, :2].clone() for k, v in ctl_blob["B"].items()}
    bad_rank["rank"] = 2
    with pytest.raises(ValueError, match="rank"):
        LoRASubtractionObject.compose_blob(mix_blob, bad_rank)
    bad_layers = dict(ctl_blob)
    keep = sorted(ctl_blob["A"])[:1]
    bad_layers["A"] = {k: ctl_blob["A"][k] for k in keep}
    bad_layers["B"] = {k: ctl_blob["B"][k] for k in keep}
    bad_layers["layers"] = [int(k) for k in keep]
    with pytest.raises(ValueError, match="layer sets differ"):
        LoRASubtractionObject.compose_blob(mix_blob, bad_layers)
    with pytest.raises(ValueError, match="fit-position"):
        LoRASubtractionObject.compose_blob(
            mix_blob, {**ctl_blob, "positions": "response",
                       "response_header_len": 4})


# --------------------------------------------------------------------------
# Builder script: synthetic bank, provenance, labels, refusals (no model)
# --------------------------------------------------------------------------


def _synthetic_lora_blob(seed: int) -> dict:
    import torch

    g = torch.Generator().manual_seed(seed)
    layers = [0, 1]
    return {"kind": "lora_all", "layers": layers, "rank": 4,
            "A": {str(l): torch.randn(16, 4, generator=g) for l in layers},
            "B": {str(l): torch.randn(12, 4, generator=g) for l in layers},
            "positions": "all", "response_header_len": None}


PARENT_FP = "f" * 64


def _stage_cell(root: Path, key: str, si: str, seed: int, *,
                complete=True, cell_id_override=None,
                meta_over: dict | None = None, meta_drop: tuple = (),
                fp: str = PARENT_FP) -> Path:
    """A FULLY staged synthetic parent: object + a document-schema budget
    sidecar carrying every identity/geometry field the builder requires +
    a coherent COMPLETE manifest — exactly what the shared staged-cell
    gate validates on the real bank."""
    import torch

    cell_id = f"{si}__{ls.ARM}"
    d = root / key / "cells_all" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    ident = d.parent / ls.STORE_IDENT
    if not ident.exists():  # the bank-level identity every writer matched
        ident.write_text(json.dumps({"model": "fake/model",
                                     "model_revision": "rev0",
                                     "positions": "all"}))
    torch.save(_synthetic_lora_blob(seed), d / "object.pt")
    meta = {"cell_id": cell_id_override or cell_id,
            "model_id": "fake/model", "model_revision": "rev0",
            "object_sha256": ls.sha256_file(d / "object.pt"),
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": "ab" * 32, "heldout_pool_sha": "cd" * 32,
            "positions": "all", "response_header_len": None,
            "target_layers": [0, 1], "template_hash": "e3b0c44298fc1c14"}
    meta.update(meta_over or {})
    for k in meta_drop:
        meta.pop(k, None)
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    if complete:
        files = {n: ls.sha256_file(d / n)
                 for n in ("object.pt", "train_budget_meta.json")}
        (d / "COMPLETE").write_text(json.dumps(
            {"cell_id": cell_id, "fit_fingerprint": fp, "files": files,
             "object_file": "object.pt",
             "object_sha256": files["object.pt"]}))
    return d


def _stage_pool_manifest(pools: Path, si: str, budget, official) -> None:
    pools.mkdir(parents=True, exist_ok=True)
    (pools / f"pairs_{si}.manifest.json").write_text(json.dumps(
        {"cell": si, "budget": budget, "official_tokens": official}))


@pytest.fixture()
def bank(tmp_path):
    root = tmp_path / "out"
    _stage_cell(root, "tinybase", "constv2-mix-100k", seed=1)
    _stage_cell(root, "tinybase", "constv2-mix-1m", seed=2)
    _stage_cell(root, "tinybase", ls.CTL_SI, seed=3)
    pools = tmp_path / "pools"
    _stage_pool_manifest(pools, "constv2-mix-100k", 100_000, 100_000)
    _stage_pool_manifest(pools, "constv2-mix-1m", 1_000_000, 1_000_000)
    _stage_pool_manifest(pools, ls.CTL_SI, 100_000, 100_000)
    return tmp_path


def test_builder_end_to_end_labels_and_provenance(bank, capsys):
    import torch

    rc = ls.main(["--root", str(bank / "out"), "--pools",
                  str(bank / "pools"), "--models", "tinybase"])
    assert rc == 0
    derived = bank / "out/derived/lora_subtraction/tinybase"
    for si, matched in (("constv2-mix-100k", True), ("constv2-mix-1m", False)):
        d = derived / f"{si}__{ls.ARM}_minus_ctl"
        blob = torch.load(d / "object.pt", map_location="cpu",
                          weights_only=True)
        assert blob["kind"] == "lora_subtraction"
        prov = blob["provenance"]
        assert prov["scale_matched"] is matched
        if matched:
            assert prov["comparison"].startswith("scale_matched (primary")
        else:
            assert prov["comparison"].startswith("budget_mismatched")
        assert prov["parent_pos"]["cell_id"] == f"{si}__{ls.ARM}"
        assert prov["parent_neg"]["cell_id"] == f"{ls.CTL_SI}__{ls.ARM}"
        # parent shas recorded and correct
        for side in ("parent_pos", "parent_neg"):
            p = Path(prov[side]["object_path"])
            assert prov[side]["object_sha256"] == ls.sha256_file(p)
        # data scales recorded from the pool manifests
        assert prov["parent_neg"]["data_scale"]["official_tokens"] == 100_000
        meta = json.loads((d / "derived_meta.json").read_text())
        assert meta["scale_matched"] is matched
        assert meta["object_sha256"] == ls.sha256_file(d / "object.pt")
        # the sidecar is NOT named train_budget_meta.json (identity gates
        # must never mistake a derived object for a fit)
        assert not (d / "train_budget_meta.json").exists()
        # parent factors ride bit-exact into the artifact
        src = torch.load(
            bank / f"out/tinybase/cells_all/{si}__{ls.ARM}/object.pt",
            map_location="cpu", weights_only=True)
        for l in ("0", "1"):
            assert torch.equal(blob["A_pos"][l], src["A"][l])
            assert torch.equal(blob["B_pos"][l], src["B"][l])


def test_builder_refuses_missing_complete_and_wrong_identity(bank, capsys):
    # torn publish: no COMPLETE on the mix parent
    root2 = bank / "out2"
    _stage_cell(root2, "k", "constv2-mix-100k", seed=1, complete=False)
    _stage_cell(root2, "k", ls.CTL_SI, seed=3)
    rc = ls.main(["--root", str(root2), "--pools", str(bank / "pools"),
                  "--models", "k", "--cells", "constv2-mix-100k"])
    assert rc == 2
    assert "COMPLETE" in capsys.readouterr().err
    # wrong-identity sidecar
    root3 = bank / "out3"
    _stage_cell(root3, "k", "constv2-mix-100k", seed=1,
                cell_id_override="constv2-mix-1m__lora_r4_ce")
    _stage_cell(root3, "k", ls.CTL_SI, seed=3)
    rc = ls.main(["--root", str(root3), "--pools", str(bank / "pools"),
                  "--models", "k", "--cells", "constv2-mix-100k"])
    assert rc == 2
    assert "wrong-identity" in capsys.readouterr().err


def test_builder_refuses_a_parent_failing_the_shared_gate(bank, capsys):
    """Parents pass through train_budget's staged-cell gate, object-sha
    binding included — swapped bytes under a valid-looking sidecar + marker
    refuse before any tensor is read."""
    import torch

    root = bank / "out"
    obj = root / "tinybase" / "cells_all" / f"{ls.CTL_SI}__{ls.ARM}" / "object.pt"
    torch.save(_synthetic_lora_blob(99), obj)
    rc = ls.main(["--root", str(root), "--pools", str(bank / "pools"),
                  "--models", "tinybase", "--cells", "constv2-mix-100k"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "staged-cell identity gate" in err
    assert "object_sha256 mismatch" in err
    assert not (root / "derived").exists()


def test_builder_refuses_parents_that_disagree_or_lack_identity(bank, capsys):
    root2 = bank / "out2"
    _stage_cell(root2, "k", "constv2-mix-100k", seed=1)
    _stage_cell(root2, "k", ls.CTL_SI, seed=3,
                meta_over={"model_id": "other/model"})
    rc = ls.main(["--root", str(root2), "--pools", str(bank / "pools"),
                  "--models", "k", "--cells", "constv2-mix-100k"])
    assert rc == 2
    # a parent naming another model than its own bank's store identity is
    # refused as a foreign cell BEFORE the pair is even compared
    assert "a cell from another bank" in capsys.readouterr().err
    # the cross-parent match itself, on a field the store does not carry
    root2b = bank / "out2b"
    _stage_cell(root2b, "k", "constv2-mix-100k", seed=1)
    _stage_cell(root2b, "k", ls.CTL_SI, seed=3,
                meta_over={"template_hash": "0000000000000000"})
    rc = ls.main(["--root", str(root2b), "--pools", str(bank / "pools"),
                  "--models", "k", "--cells", "constv2-mix-100k"])
    assert rc == 2
    assert "parents disagree on template_hash" in capsys.readouterr().err
    # a parent whose sidecar records no model identity (the R2 finding:
    # meta["model"] read None and nothing noticed) is refused by name
    root3 = bank / "out3"
    _stage_cell(root3, "k", "constv2-mix-100k", seed=1,
                meta_over={"model_id": None})
    _stage_cell(root3, "k", ls.CTL_SI, seed=3)
    rc = ls.main(["--root", str(root3), "--pools", str(bank / "pools"),
                  "--models", "k", "--cells", "constv2-mix-100k"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "records no" in err and "model_id" in err
    # geometry mismatches refuse too (a task vector across layer sets is
    # not a defined object) — the builder names the field
    root4 = bank / "out4"
    _stage_cell(root4, "k", "constv2-mix-100k", seed=1)
    _stage_cell(root4, "k", ls.CTL_SI, seed=3,
                meta_over={"target_layers": [0, 1, 2]})
    rc = ls.main(["--root", str(root4), "--pools", str(bank / "pools"),
                  "--models", "k", "--cells", "constv2-mix-100k"])
    assert rc == 2
    assert "parents disagree on target_layers" in capsys.readouterr().err


def test_model_revision_resolves_from_the_store_identity(bank, capsys):
    """The qwen35_4b_base bank's budget sidecars record model_revision
    null although its fits config pins one and the bank's
    _store_identity.json (the store-level identity every writer had to
    match) records it. The builder reads the AUTHORITATIVE record: a null
    sidecar revision resolves from the store identity (source recorded);
    a sidecar revision that contradicts the store identity refuses; a
    sidecar without the key, a missing store identity, or a store identity
    naming another model all refuse by name."""
    from sisteer.eval.train_budget import derived_object_problems

    def _build(root):
        return ls.main(["--root", str(root), "--pools", str(bank / "pools"),
                        "--models", "k", "--cells", "constv2-mix-100k"])

    def _meta(root):
        return json.loads((root / "derived/lora_subtraction/k"
                           / f"constv2-mix-100k__{ls.ARM}_minus_ctl"
                           / "derived_meta.json").read_text())

    # null on both parents (the qwen shape): resolved, non-null, sourced
    root5 = bank / "out5"
    _stage_cell(root5, "k", "constv2-mix-100k", seed=1,
                meta_over={"model_revision": None})
    _stage_cell(root5, "k", ls.CTL_SI, seed=3,
                meta_over={"model_revision": None})
    assert _build(root5) == 0
    m = _meta(root5)
    assert m["model_revision"] == "rev0"
    for side in ("parent_pos", "parent_neg"):
        assert m[side]["model_revision"] == "rev0"
        assert m[side]["model_revision_source"] == ls.STORE_IDENT
    assert derived_object_problems(
        root5 / "derived/lora_subtraction/k"
        / f"constv2-mix-100k__{ls.ARM}_minus_ctl" / "object.pt",
        parents_root=root5) == []
    # a recorded sidecar revision that AGREES is sourced from the sidecar
    root_ok = bank / "out"  # the fixture bank: model key "tinybase"
    assert ls.main(["--root", str(root_ok), "--pools", str(bank / "pools"),
                    "--models", "tinybase", "--cells", "constv2-mix-100k"]) == 0
    meta_ok = json.loads((root_ok / "derived/lora_subtraction/tinybase"
                          / f"constv2-mix-100k__{ls.ARM}_minus_ctl"
                          / "derived_meta.json").read_text())
    assert meta_ok["parent_pos"]["model_revision_source"] == \
        "train_budget_meta.json"
    # a sidecar revision contradicting the store identity refuses
    root6 = bank / "out6"
    _stage_cell(root6, "k", "constv2-mix-100k", seed=1)
    _stage_cell(root6, "k", ls.CTL_SI, seed=3,
                meta_over={"model_revision": "revZ"})
    assert _build(root6) == 2
    assert "store identity" in capsys.readouterr().err
    # a sidecar without the key at all refuses (pre-schema record)
    root7 = bank / "out7"
    _stage_cell(root7, "k", "constv2-mix-100k", seed=1,
                meta_drop=("model_revision",))
    _stage_cell(root7, "k", ls.CTL_SI, seed=3)
    assert _build(root7) == 2
    assert "records no ['model_revision']" in capsys.readouterr().err
    # no store identity: not a bank
    root8 = bank / "out8"
    _stage_cell(root8, "k", "constv2-mix-100k", seed=1)
    _stage_cell(root8, "k", ls.CTL_SI, seed=3)
    (root8 / "k" / "cells_all" / ls.STORE_IDENT).unlink()
    assert _build(root8) == 2
    assert ls.STORE_IDENT in capsys.readouterr().err
    # a store identity naming another model than the sidecar refuses
    root9 = bank / "out9"
    _stage_cell(root9, "k", "constv2-mix-100k", seed=1)
    _stage_cell(root9, "k", ls.CTL_SI, seed=3)
    (root9 / "k" / "cells_all" / ls.STORE_IDENT).write_text(json.dumps(
        {"model": "other/model", "model_revision": "rev0",
         "positions": "all"}))
    assert _build(root9) == 2
    assert "store identity" in capsys.readouterr().err


def test_derived_meta_binds_parents_and_passes_the_shared_verdict(bank):
    import torch

    from sisteer.eval.train_budget import (assert_object_identity,
                                           derived_object_problems)

    root = bank / "out"
    assert ls.main(["--root", str(root), "--pools", str(bank / "pools"),
                    "--models", "tinybase", "--cells", "constv2-mix-100k"]) == 0
    derived_id = f"constv2-mix-100k__{ls.ARM}_minus_ctl"
    d = root / "derived/lora_subtraction/tinybase" / derived_id
    obj = d / "object.pt"
    meta = json.loads((d / "derived_meta.json").read_text())
    assert meta["derived_id"] == derived_id
    assert meta["model_id"] == "fake/model"          # was None (the finding)
    assert meta["model_revision"] == "rev0"
    for side in ("parent_pos", "parent_neg"):
        assert meta[side]["model_id"] == "fake/model"
        assert meta[side]["fit_fingerprint"] == PARENT_FP
        assert meta[side]["target_layers"] == [0, 1]
        assert meta[side]["budget_meta_sha256"] == ls.sha256_file(
            Path(meta[side]["object_path"]).parent / "train_budget_meta.json")
        assert "model" not in meta[side]           # the mis-keyed field is gone
    # the shared verdict: bytes <-> sidecar <-> embedded provenance, and
    # the parents re-verified on the bank
    assert derived_object_problems(obj, derived_id=derived_id,
                                   parents_root=root) == []
    assert assert_object_identity(obj, derived_id, consumer="t") == "derived"
    with pytest.raises(SystemExit, match="wrong-identity"):
        assert_object_identity(obj, f"constv2-mix-1m__{ls.ARM}_minus_ctl",
                               consumer="t")
    # a parent REFIT under the same cell id (coherent new object + sidecar
    # + marker) is caught only through the bank: the derived object names
    # the object it consumed by sha
    _stage_cell(root, "tinybase", ls.CTL_SI, seed=77, fp="e" * 64)
    assert derived_object_problems(obj, derived_id=derived_id) == []
    probs = derived_object_problems(obj, derived_id=derived_id,
                                    parents_root=root)
    assert any("not the object staged under that id now" in p for p in probs)
    # tampered derived bytes: refused with or without the bank
    torch.save({**torch.load(obj, weights_only=True), "rank": 4}, obj)
    assert any("object_sha256 mismatch" in p
               for p in derived_object_problems(obj, derived_id=derived_id))


def test_consumer_load_paths_validate_derived_objects(bank):
    """Every load path runs the derived verdict — probe_score's --object
    gate (with and without --cell) and base_mmlu's resolution (bank cell or
    explicit --object path) — with the parents RE-VERIFIED against the live
    bank: a derived object without --parents-root is refused, a REPLACED
    parent under the same cell id is refused before any model load, a
    tampered artifact refuses naming its consumer, and a dir claiming both
    identities is refused rather than guessed."""
    import shutil

    import torch

    import base_mmlu
    import probe_score as ps

    root = bank / "out"
    assert ls.main(["--root", str(root), "--pools", str(bank / "pools"),
                    "--models", "tinybase", "--cells", "constv2-mix-100k"]) == 0
    derived_id = f"constv2-mix-100k__{ls.ARM}_minus_ctl"
    d = root / "derived/lora_subtraction/tinybase" / derived_id
    obj = d / "object.pt"
    # a derived object REQUIRES the bank root — both consumers, both paths
    with pytest.raises(SystemExit, match="parents-root"):
        ps._validate_object_identity(obj, None)
    with pytest.raises(SystemExit, match="parents-root"):
        ps._validate_object_identity(obj, derived_id)
    with pytest.raises(SystemExit, match="parents-root"):
        base_mmlu.resolve_object_path(None, derived_id, explicit_path=obj)
    assert ps._validate_object_identity(obj, None, parents_root=root) == "derived"
    assert ps._validate_object_identity(obj, derived_id, parents_root=root) \
        == "derived"
    with pytest.raises(SystemExit, match="wrong-identity"):
        ps._validate_object_identity(obj, f"constv2-mix-1m__{ls.ARM}_minus_ctl",
                                     parents_root=root)
    # base_mmlu: the derived app's explicit --object path (derived ids live
    # under derived/, not in a bank's cells_all/) ...
    assert base_mmlu.resolve_object_path(None, derived_id, parents_root=root,
                                         explicit_path=obj) == obj
    # ... and a derived cell staged in a bank's cells_all/
    bank2 = bank / "bank2"
    dst = bank2 / "cells_all" / derived_id
    shutil.copytree(d, dst)
    with pytest.raises(SystemExit, match="parents-root"):
        base_mmlu.resolve_object_path(bank2, derived_id)
    assert base_mmlu.resolve_object_path(bank2, derived_id, parents_root=root) \
        == dst / "object.pt"
    # a REPLACED parent (a refit under the same cell id: coherent new
    # object + sidecar + marker) is caught only through the bank — every
    # consumer refuses it before any model load, naming the mismatch
    _stage_cell(root, "tinybase", ls.CTL_SI, seed=77, fp="e" * 64)
    for call in (lambda: ps._validate_object_identity(obj, None, parents_root=root),
                 lambda: ps._validate_object_identity(obj, derived_id,
                                                      parents_root=root),
                 lambda: base_mmlu.resolve_object_path(None, derived_id,
                                                       parents_root=root,
                                                       explicit_path=obj),
                 lambda: base_mmlu.resolve_object_path(bank2, derived_id,
                                                       parents_root=root)):
        with pytest.raises(SystemExit, match="not the object staged under"):
            call()
    _stage_cell(root, "tinybase", ls.CTL_SI, seed=3)  # the original parent back
    assert ps._validate_object_identity(obj, derived_id, parents_root=root) \
        == "derived"
    # tamper: both consumers refuse, each naming itself
    torch.save({**torch.load(dst / "object.pt", weights_only=True),
                "rank": 4}, dst / "object.pt")
    with pytest.raises(SystemExit, match="base-mmlu"):
        base_mmlu.resolve_object_path(bank2, derived_id, parents_root=root)
    with pytest.raises(SystemExit, match="probe-score"):
        ps._validate_object_identity(dst / "object.pt", None, parents_root=root)
    # a staged (fitted) cell needs no parents root: it has no parents
    fitted = root / "tinybase" / "cells_all" / f"constv2-mix-100k__{ls.ARM}"
    assert ps._validate_object_identity(fitted / "object.pt", None) == "staged"
    assert base_mmlu.resolve_object_path(
        root / "tinybase", f"constv2-mix-100k__{ls.ARM}") == fitted / "object.pt"
    # ambiguous dir (a budget sidecar next to the derived one) refuses
    (d / "train_budget_meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="both|more than one identity sidecar"):
        ps._validate_object_identity(obj, derived_id, parents_root=root)


def test_builder_records_unavailable_pool_manifests_honestly(bank):
    import torch

    rc = ls.main(["--root", str(bank / "out"), "--pools",
                  str(bank / "nonexistent-pools"), "--models", "tinybase",
                  "--cells", "constv2-mix-100k"])
    assert rc == 0
    d = (bank / "out/derived/lora_subtraction/tinybase"
         / f"constv2-mix-100k__{ls.ARM}_minus_ctl")
    prov = torch.load(d / "object.pt", map_location="cpu",
                      weights_only=True)["provenance"]
    assert prov["parent_pos"]["data_scale"]["source"].startswith("unavailable")
    assert prov["scale_matched"] is False  # unknown scale is never "matched"


# -- the full − full pair: --ctl-si, --keep-valid, atomic publish --------------


def test_ctl_si_selects_the_control_and_rides_in_provenance(bank, capsys):
    """``--ctl-si constv2ctl2-full`` subtracts the r2 control (the derived
    id keeps the corpus-cell template; the control is ``parent_neg``), the
    pair reads scale-matched off the pool manifests, and the round-1
    control appears nowhere in the artifact."""
    import torch

    root, pools = bank / "out", bank / "pools"
    _stage_cell(root, "tinybase", "constv2-mix-full", seed=7)
    _stage_cell(root, "tinybase", "constv2ctl2-full", seed=8)
    _stage_pool_manifest(pools, "constv2-mix-full", None, 42_785_435)
    _stage_pool_manifest(pools, "constv2ctl2-full", 42_785_435, 42_785_435)
    rc = ls.main(["--root", str(root), "--pools", str(pools), "--models", "tinybase",
                  "--cells", "constv2-mix-full", "--ctl-si", "constv2ctl2-full"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "constv2-mix-full - constv2ctl2-full ->" in out
    d = root / "derived/lora_subtraction/tinybase" / f"constv2-mix-full__{ls.ARM}_minus_ctl"
    meta = json.loads((d / "derived_meta.json").read_text())
    assert meta["parent_pos"]["cell_id"] == f"constv2-mix-full__{ls.ARM}"
    assert meta["parent_neg"]["cell_id"] == f"constv2ctl2-full__{ls.ARM}"
    assert meta["scale_matched"] is True and meta["comparison"].startswith("scale_matched")
    assert meta["parent_neg"]["data_scale"]["official_tokens"] == 42_785_435
    assert "constv2ctl-100k" not in (d / "derived_meta.json").read_text()
    blob = torch.load(d / "object.pt", map_location="cpu", weights_only=True)
    neg = _synthetic_lora_blob(8)
    for l in ("0", "1"):
        assert torch.equal(blob["A_neg"][l], neg["A"][l])
    assert not list(d.glob("*.tmp"))
    assert ls.derived_id_for("constv2-mix-full") == d.name


def test_keep_valid_is_idempotent_and_rebuilds_a_bad_or_foreign_object(bank, capsys):
    """``--keep-valid``: a publication that passes the shared verdict against
    THIS bank and names THESE parents is kept (same bytes, no rewrite); a
    tampered object, a torn publish, or a same-id object built against
    ANOTHER control is rebuilt — and the rebuild replaces a READ-ONLY
    publication atomically (rename, never open-for-write: the EACCES trap)."""
    import os
    import stat

    root, pools = bank / "out", bank / "pools"
    _stage_cell(root, "tinybase", "constv2-mix-full", seed=7)
    _stage_cell(root, "tinybase", "constv2ctl2-full", seed=8)
    _stage_pool_manifest(pools, "constv2-mix-full", None, 42_785_435)
    _stage_pool_manifest(pools, "constv2ctl2-full", 42_785_435, 42_785_435)
    argv = ["--root", str(root), "--pools", str(pools), "--models", "tinybase",
            "--cells", "constv2-mix-full", "--ctl-si", "constv2ctl2-full"]
    assert ls.main(argv) == 0
    d = root / "derived/lora_subtraction/tinybase" / f"constv2-mix-full__{ls.ARM}_minus_ctl"
    obj, meta = d / "object.pt", d / "derived_meta.json"
    sha1 = ls.sha256_file(obj)
    mtime1 = obj.stat().st_mtime_ns
    capsys.readouterr()
    # kept: same bytes, not rewritten
    assert ls.main(argv + ["--keep-valid"]) == 0
    assert "already published and valid" in capsys.readouterr().out
    assert ls.sha256_file(obj) == sha1 and obj.stat().st_mtime_ns == mtime1
    # a same-id build against ANOTHER control is REFUSED (never rebuilt in
    # place), with or without --keep-valid: the id names the corpus cell only
    for extra in ([], ["--keep-valid"]):
        capsys.readouterr()
        assert ls.main(["--root", str(root), "--pools", str(pools), "--models", "tinybase",
                        "--cells", "constv2-mix-full", "--ctl-si", ls.CTL_SI] + extra) == 2
        err = capsys.readouterr().err
        assert "REFUSED" in err and "another pair" in err and "parent_neg.cell_id" in err
    assert json.loads(meta.read_text())["parent_neg"]["cell_id"] == f"constv2ctl2-full__{ls.ARM}"
    assert ls.sha256_file(obj) == sha1

    def same_factors():
        # a REBUILT blob differs in bytes (its provenance carries built_at)
        # but carries the same factors; its sidecar names its own bytes
        import torch
        b = torch.load(obj, map_location="cpu", weights_only=True)
        for side, seed in (("A_pos", 7), ("A_neg", 8)):
            ref = _synthetic_lora_blob(seed)["A"]
            assert all(torch.equal(b[side][l], ref[l]) for l in ("0", "1"))
        assert json.loads(meta.read_text())["object_sha256"] == ls.sha256_file(obj)
        assert not list(d.glob("*.tmp"))

    same_factors()
    # a tampered sidecar is rebuilt; a READ-ONLY publication is replaced by rename
    sha_now = ls.sha256_file(obj)
    meta.write_text(meta.read_text().replace(sha_now, "0" * 64))
    os.chmod(obj, stat.S_IRUSR | stat.S_IRGRP)
    os.chmod(meta, stat.S_IRUSR | stat.S_IRGRP)
    capsys.readouterr()
    assert ls.main(argv + ["--keep-valid"]) == 0
    assert "rebuilding" in capsys.readouterr().err
    same_factors()
    # a torn publish (object without sidecar) is rebuilt
    meta.unlink()
    assert ls.main(argv + ["--keep-valid"]) == 0
    assert meta.is_file()
    same_factors()
    # without --keep-valid the builder still rebuilds unconditionally
    m2 = obj.stat().st_mtime_ns
    assert ls.main(argv) == 0
    assert obj.stat().st_mtime_ns >= m2
    same_factors()
    _obj, problems = ls.existing_valid(root, root / "derived/lora_subtraction",
                                       "tinybase", "constv2-mix-full", "constv2ctl2-full")
    assert problems == []
    _obj, problems = ls.existing_valid(root, root / "derived/lora_subtraction",
                                       "tinybase", "constv2-mix-1m", "constv2ctl2-full")
    assert problems and "missing" in problems[0]


# --------------------------------------------------------------------------
# Object-level pins from main's version of this file (bcb-bfcl-200, 2026-09-10)
# --------------------------------------------------------------------------


def test_save_recomposes_through_the_one_schema_owner(sub_world):
    """(main's version of this file, bcb-bfcl-200 2026-09-10) save() re-composes
    through the ONE schema owner; the blob round-trips key-for-key and factor-for-factor."""
    import torch

    sub = sub_world.sub
    sub.save(sub_world.root / "sub2.pt")
    again = torch.load(sub_world.root / "sub2.pt", map_location="cpu",
                       weights_only=True)
    assert again["kind"] == "lora_subtraction"
    assert sorted(again) == sorted(sub_world.blob)
    for side in ("pos", "neg"):
        for l in sub.layers:
            assert torch.equal(again[f"A_{side}"][str(l)],
                               sub_world.blob[f"A_{side}"][str(l)])


def test_the_blob_schema_is_the_one_the_adapter_reads(sub_world):
    """The keys the serving adapter (lens_adapters._adapt_lora_subtraction)
    and the derived verdict (train_budget.derived_object_problems) read:
    kind, layers, rank, the four factor maps, provenance, the fit leg."""
    assert set(sub_world.blob) == {
        "kind", "layers", "rank", "A_pos", "B_pos", "A_neg", "B_neg",
        "provenance", "positions", "response_header_len"}
    assert sub_world.blob["positions"] == "all"
    assert sub_world.blob["response_header_len"] is None


def test_hydration_refuses_parents_on_different_models(sub_world):
    """``__init__`` refuses two parents hydrated against different o_proj
    modules — the difference is defined on ONE model's modules only."""
    from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                               MultiLayerLoRAObject)

    pos = MultiLayerLoRAObject.load_blob(
        LoRASubtractionObject._parent_blob(sub_world.blob, "pos"),
        sub_world.model, sub_world.tok)
    neg = MultiLayerLoRAObject.load_blob(
        LoRASubtractionObject._parent_blob(sub_world.blob, "neg"),
        sub_world.model, sub_world.tok)
    # swap ONE layer's module for a fresh Linear of the same shape
    import torch
    l0 = neg.layers[0]
    neg._o_projs[l0] = torch.nn.Linear(*reversed(neg._o_projs[l0].weight.shape),
                                       bias=False)
    with pytest.raises(RuntimeError, match="SAME model"):
        LoRASubtractionObject(pos, neg)


# -- the default sweep is the pre-Llama set; Llama needs --models --


def test_default_models_are_the_old_set_and_llama_needs_the_flag(
        monkeypatch, tmp_path):
    """MODEL_KEYS doubled as the --models default, so registering the two
    Llama keys made an existing invocation that omits the flag request
    Llama parents that exist in neither Llama config (exit 2 where it used
    to exit 0). The supported-key registry and the default selection are
    separate: the default is the old pair, the Llama keys are supported
    and reachable only by naming them."""
    seen: list[tuple[str, str]] = []

    def fake_build_one(root, out_root, key, si, **kw):
        seen.append((key, si))
        return tmp_path / key / si

    monkeypatch.setattr(ls, "build_one", fake_build_one)
    assert ls.main(["--root", str(tmp_path), "--pools", str(tmp_path),
                    "--cells", "constv2-mix-100k"]) == 0
    assert [k for k, _si in seen] == ["gemma4e4b_base", "qwen35_4b_base"]
    assert ls.DEFAULT_MODEL_KEYS == ("gemma4e4b_base", "qwen35_4b_base")
    assert {"llama8b_base", "llama8b_it_doc"} <= set(ls.MODEL_KEYS)
    assert not ({"llama8b_base", "llama8b_it_doc"} & set(ls.DEFAULT_MODEL_KEYS))
    seen.clear()
    assert ls.main(["--root", str(tmp_path), "--pools", str(tmp_path),
                    "--cells", "constv2-mix-100k", "--models",
                    "llama8b_base"]) == 0
    assert [k for k, _si in seen] == ["llama8b_base"]


def test_seed_replicate_keys_are_registered_and_never_default_swept():
    """Since 2026-09-15 the three batch-64 seed-replicate keys are
    SUPPORTED — their constv2-mix-full minus constv2ctl2-full pair is built
    from SAME-seed parents by naming the key — and, like the Llama keys,
    never part of the default sweep; the existing admissions and the default
    pair are untouched."""
    keys = ("qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
            "qwen35_4b_base_eb64_s3")
    assert ls.MODEL_KEYS[-3:] == keys
    assert ls.MODEL_KEYS[:2] == ("gemma4e4b_base", "qwen35_4b_base")
    assert len(ls.MODEL_KEYS) == len(set(ls.MODEL_KEYS))
    assert ls.DEFAULT_MODEL_KEYS == ("gemma4e4b_base", "qwen35_4b_base")
    assert not (set(keys) & set(ls.DEFAULT_MODEL_KEYS))
