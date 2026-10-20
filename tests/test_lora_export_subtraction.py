"""``_export_lora_subtraction`` — the ``lora_subtraction`` kind
(extraction/multilayer.LoRASubtractionObject) -> ONE rank-2r PEFT adapter
by RANK STACKING (native-multilora-serving; serving/lora_export.py).

Load-bearing here:

- STRUCTURAL EXACTNESS, bitwise: ``lora_A[:r] == A_pos.T``, ``lora_A[r:]
  == A_neg.T``, ``lora_B[:, :r] == B_pos``, ``lora_B[:, r:] == -B_neg``
  (``torch.equal``), so both parents are recoverable from the adapter;
  ``r == 2·rank``, ``lora_alpha == float(2·rank)`` (scaling exactly 1), and
  the sign convention is RECORDED in the config;
- HOOK EQUIVALENCE on the tiny model: ``LoRASubtractionObject.apply`` vs
  the PEFT arithmetic ``(x @ lora_A.T) @ lora_B.T · alpha/r`` to fp32
  rounding (``atol=1e-5``, the lora_all rank-r pin) — the accumulation
  order differs BY CONSTRUCTION (two r-term sums then a subtraction vs
  one 2r-term sum), so bitwise is not claimed;
- determinism: same artifact bytes -> byte-identical adapter files;
- every ``_refuse_mismatch`` precondition is a loud export refusal on the
  raw blob (layer sets, ranks, per-layer shapes, one shared positions
  leg), plus the module-wide ones (position-masked fits, non-fp32,
  sidecar keys, spec cross-checks, layers outside the tree's o_proj set);
- the real bank shapes (gemma 42 layers A[2048,4]/B[2560,4]; nemotron 4
  layers A[5120,4]/B[3136,4]) export under their own layouts;
- ``verify_zero_adapter`` never certifies it.

All fixtures are synthetic: a from-config tiny Llama (no hub access),
generator-seeded tensors, invented hashes and cell ids.
"""

import io
import json

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx
from lora_snapshot_fixtures import write_safetensors_snapshot

HIDDEN = 32
N_LAYERS = 2
BASE = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_BASE = dict(base_model=BASE, decoder_prefix=lx.LLAMA_DECODER_PREFIX)
WRAPPED_PREFIX = "model.language_model.layers"
IH = "ab" * 8
PROVENANCE = {
    "kind": "lora_subtraction",
    "parent_pos": {"cell_id": "mix__lora_r4_ce", "object_sha256": "a" * 64,
                   "model_id": BASE},
    "parent_neg": {"cell_id": "ctl__lora_r4_ce", "object_sha256": "b" * 64,
                   "model_id": BASE},
}


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=97, max_position_embeddings=64,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_base(tiny_model, tmp_path_factory):
    """The tiny llama SAVED as a local snapshot (config + model.safetensors):
    the export derives the layout off the config and spells the module names
    off the weights' own namespace — hermetic, no hub, no cache."""
    d = tmp_path_factory.mktemp("tiny-llama-snapshot")
    tiny_model.save_pretrained(d)
    return str(d)


def _blob_bytes(d) -> bytes:
    buf = io.BytesIO()
    torch.save(d, buf)
    return buf.getvalue()


def _multilayer_object(tiny_model, rank=4, seed=7):
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    obj = MultiLayerLoRAObject(tiny_model, rank=rank)
    g = torch.Generator().manual_seed(seed)
    for l in obj.layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g)
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.1
    if not hasattr(obj, "fit_positions"):
        obj.fit_positions = "all"
        obj.fit_response_header_len = None
    return obj


def _subtraction_object(tiny_model, rank=4, seeds=(11, 12)):
    from sisteer.extraction.multilayer import LoRASubtractionObject

    pos = _multilayer_object(tiny_model, rank=rank, seed=seeds[0])
    neg = _multilayer_object(tiny_model, rank=rank, seed=seeds[1])
    return LoRASubtractionObject(pos, neg, provenance=dict(PROVENANCE))


def _subtraction_blob(tiny_model, rank=4, positions="all", **overrides):
    """The artifact dict the class's ``save`` writes (``compose_blob``,
    the ONE owner of the schema), with the fit leg as requested."""
    from sisteer.extraction.multilayer import LoRASubtractionObject

    pos = _multilayer_object(tiny_model, rank=rank, seed=11)
    neg = _multilayer_object(tiny_model, rank=rank, seed=12)
    pb, nb = pos.to_blob(), neg.to_blob()
    pb["positions"] = nb["positions"] = positions
    blob = LoRASubtractionObject.compose_blob(pb, nb,
                                              provenance=dict(PROVENANCE))
    blob.update(overrides)
    return blob


def _f32(t):
    return t.detach().float().cpu()


def _load(adapter):
    from safetensors.torch import load_file

    return (json.loads((adapter / lx.ADAPTER_CONFIG).read_text()),
            load_file(str(adapter / lx.ADAPTER_WEIGHTS)))


# --------------------------------------------------------------------------
# rank stacking: structural exactness and the recorded sign
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 4])
def test_subtraction_export_is_rank_stacking_bitwise(tiny_model, tmp_path,
                                                     rank):
    """Bitwise: rows [0, r) of lora_A are A_pos.T, rows [r, 2r) A_neg.T;
    columns [0, r) of lora_B are B_pos, columns [r, 2r) are -B_neg. Rank 1
    keeps the legacy 1-D parents (unsqueezed exactly as lora_all does)."""
    sub = _subtraction_object(tiny_model, rank=rank)
    art = tmp_path / "object.pt"
    sub.save(art)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction", "rank": rank, "positions": "all",
         "layers": list(sub.layers), "intervention_hash": IH},
        art.read_bytes(), tmp_path / "adapter", **LLAMA_BASE)
    cfg, tensors = _load(adapter)

    assert cfg["r"] == 2 * rank and cfg["lora_alpha"] == float(2 * rank)
    assert cfg["target_modules"] == ["o_proj"]
    assert cfg["layers_to_transform"] == list(range(N_LAYERS))
    assert cfg["petri_genuine_adapter"] is True
    assert "petri_zero_adapter" not in cfg
    assert cfg["petri_source_kind"] == "lora_subtraction"
    assert cfg["petri_source_layers"] == list(range(N_LAYERS))
    assert cfg["petri_fit_positions"] == "all"
    assert cfg["petri_decoder_prefix"] == lx.LLAMA_DECODER_PREFIX
    assert cfg["petri_intervention_hash"] == IH
    assert "petri_oproj_site" not in cfg
    # the sign convention, recorded
    assert cfg["petri_lora_subtraction_parent_rank"] == rank
    assert cfg["petri_lora_subtraction_negated_block"] == "lora_B[:, rank:]"
    assert (cfg["petri_lora_subtraction_negated_block"]
            == lx.SUBTRACTION_NEGATED_BLOCK)
    assert cfg["petri_lora_subtraction_provenance"] == {
        "parent_pos": {"cell_id": "mix__lora_r4_ce",
                       "object_sha256": "a" * 64},
        "parent_neg": {"cell_id": "ctl__lora_r4_ce",
                       "object_sha256": "b" * 64}}

    expected = sorted(
        f"base_model.model.model.layers.{l}.self_attn.o_proj.{w}.weight"
        for l in range(N_LAYERS) for w in ("lora_A", "lora_B"))
    assert sorted(tensors) == expected
    for l in sub.layers:
        prefix = f"base_model.model.model.layers.{l}.self_attn.o_proj"
        lora_A = tensors[f"{prefix}.lora_A.weight"]
        lora_B = tensors[f"{prefix}.lora_B.weight"]
        A_pos, B_pos = _f32(sub.pos.A[l]), _f32(sub.pos.B[l])
        A_neg, B_neg = _f32(sub.neg.A[l]), _f32(sub.neg.B[l])
        d_in, d_out = A_pos.shape[0], B_pos.shape[0]
        assert tuple(lora_A.shape) == (2 * rank, d_in)
        assert tuple(lora_B.shape) == (d_out, 2 * rank)
        assert lora_A.dtype == lora_B.dtype == torch.float32
        if rank == 1:
            A_pos, A_neg = A_pos[:, None], A_neg[:, None]      # [d_in, 1]
            B_pos, B_neg = B_pos[:, None], B_neg[:, None]      # [d_out, 1]
        assert torch.equal(lora_A[:rank], A_pos.t())
        assert torch.equal(lora_A[rank:], A_neg.t())
        assert torch.equal(lora_B[:, :rank], B_pos)
        assert torch.equal(lora_B[:, rank:], -B_neg)
        # the negated block genuinely differs from the positive one
        assert not torch.equal(lora_B[:, rank:], -B_pos)
        # dense form: ΔW = B_pos A_posᵀ − B_neg A_negᵀ (to fp32 rounding —
        # one 2r-term reduction vs two r-term ones)
        dW = lora_B @ lora_A
        assert torch.allclose(dW, B_pos @ A_pos.t() - B_neg @ A_neg.t(),
                              rtol=0.0, atol=1e-5)


def test_subtraction_export_matches_the_hook_on_the_tiny_model(tiny_model,
                                                               tmp_path):
    """The served PEFT arithmetic on the exported tensors reproduces
    ``LoRASubtractionObject.apply`` (pos hooks at +1, neg hooks at -1 on
    the same o_proj) to fp32 rounding on every layer, non-vacuously — and
    differs from the positive parent alone, so the negated block is
    doing the subtracting."""
    from sisteer.models import get_decoder_layers

    sub = _subtraction_object(tiny_model, rank=4)
    art = tmp_path / "object.pt"
    sub.save(art)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction"}, art.read_bytes(), tmp_path / "adapter",
        **LLAMA_BASE)
    cfg, tensors = _load(adapter)
    scaling = cfg["lora_alpha"] / cfg["r"]
    assert scaling == 1.0

    decoder = get_decoder_layers(tiny_model)
    g = torch.Generator().manual_seed(41)
    for l in sub.layers:
        prefix = f"base_model.model.model.layers.{l}.self_attn.o_proj"
        lora_A = tensors[f"{prefix}.lora_A.weight"]
        lora_B = tensors[f"{prefix}.lora_B.weight"]
        mod = decoder[l].self_attn.o_proj
        x = torch.randn(2, 5, mod.in_features, generator=g)
        with torch.no_grad():
            base_out = mod(x)
            with sub.apply(tiny_model, scale=1.0):
                hooked_out = mod(x)
            with sub.pos.apply(tiny_model, scale=1.0):
                pos_only_out = mod(x)
            peft_delta = (x @ lora_A.t()) @ lora_B.t() * scaling
        assert not torch.equal(hooked_out, base_out)        # non-vacuous
        assert not torch.equal(hooked_out, pos_only_out)    # neg matters
        assert torch.allclose(hooked_out, base_out + peft_delta,
                              rtol=0.0, atol=1e-5), (
            (hooked_out - (base_out + peft_delta)).abs().max().item())


def test_subtraction_export_is_deterministic(tiny_model, tmp_path):
    """Same artifact bytes -> byte-identical adapter files (no RNG, sorted
    keys, no timestamps in the two PEFT files)."""
    art = _blob_bytes(_subtraction_blob(tiny_model, rank=4))
    spec = {"kind": "lora_subtraction", "intervention_hash": IH}
    a = lx.export_genuine_adapter(spec, art, tmp_path / "a", **LLAMA_BASE)
    b = lx.export_genuine_adapter(spec, art, tmp_path / "b", **LLAMA_BASE)
    for name in (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_subtraction_is_a_genuine_kind_and_never_the_zero_adapter(
        tiny_model, tmp_path):
    assert "lora_subtraction" in lx.GENUINE_EXPORT_KINDS
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction"},
        _blob_bytes(_subtraction_blob(tiny_model)), tmp_path / "adapter",
        **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError,
                       match="not marked petri_zero_adapter"):
        lx.verify_zero_adapter(adapter)


def test_subtraction_config_key_set_is_pinned(tiny_model, tmp_path):
    """The adapter config's EXACT key set — the adapter block, the
    attestation and the fingerprint preflight all read this file, so a
    key appearing or vanishing is a contract change, not a detail."""
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction", "intervention_hash": IH},
        _blob_bytes(_subtraction_blob(tiny_model)), tmp_path / "adapter",
        **LLAMA_BASE)
    cfg, _ = _load(adapter)
    assert set(cfg) == {
        "peft_type", "base_model_name_or_path", "lora_dropout", "bias",
        "fan_in_fan_out", "task_type", "petri_genuine_adapter",
        "petri_decoder_prefix", "r", "lora_alpha", "target_modules",
        "layers_to_transform", "petri_source_kind", "petri_source_layers",
        "petri_fit_positions", "petri_lora_subtraction_parent_rank",
        "petri_lora_subtraction_negated_block",
        "petri_lora_subtraction_provenance", "petri_intervention_hash"}
    # the config is JSON the bank hashes — round-trips byte-identically
    text = (adapter / lx.ADAPTER_CONFIG).read_text()
    assert json.dumps(json.loads(text), indent=2, sort_keys=True) == text


def test_subtraction_export_derives_the_layout_from_the_base_config(
        tiny_model, tiny_base, tmp_path):
    """Called WITHOUT decoder_prefix= (the production discipline) the
    export derives prefix, site and the o_proj layer set from the base's
    own (local) config through a meta skeleton, and proves the fit
    layers against it."""
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction"},
        _blob_bytes(_subtraction_blob(tiny_model)), tmp_path / "adapter",
        base_model=tiny_base)
    cfg, tensors = _load(adapter)
    assert cfg["petri_decoder_prefix"] == "model.layers"
    assert cfg["base_model_name_or_path"] == tiny_base
    assert all(k.startswith("base_model.model.model.layers.")
               and ".self_attn.o_proj." in k for k in tensors)


# --------------------------------------------------------------------------
# refusals: every _refuse_mismatch precondition, on the raw blob
# --------------------------------------------------------------------------


def test_subtraction_refuses_parents_of_different_rank(tiny_model, tmp_path):
    good = _subtraction_blob(tiny_model, rank=4)
    neg2 = _multilayer_object(tiny_model, rank=2, seed=12)
    mixed = dict(good,
                 A_neg={str(l): _f32(neg2.A[l]) for l in neg2.layers},
                 B_neg={str(l): _f32(neg2.B[l]) for l in neg2.layers})
    # with the declared rank present the negative parent contradicts it
    with pytest.raises(lx.LoraExportError, match="declares rank 4"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(mixed), tmp_path / "a",
                                  **LLAMA_BASE)
    # without it the parents contradict EACH OTHER
    undeclared = {k: v for k, v in mixed.items() if k != "rank"}
    with pytest.raises(lx.LoraExportError, match="parent ranks differ"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(undeclared), tmp_path / "b",
                                  **LLAMA_BASE)
    # a declared rank both parents' tensors contradict
    with pytest.raises(lx.LoraExportError, match="declares rank 2"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(dict(good, rank=2)),
                                  tmp_path / "c", **LLAMA_BASE)


def test_subtraction_refuses_mismatched_layer_sets(tiny_model, tmp_path):
    good = _subtraction_blob(tiny_model, rank=4)
    for name in ("B_pos", "A_neg", "B_neg"):
        partial = dict(good, **{name: {"0": good[name]["0"]}})
        with pytest.raises(lx.LoraExportError,
                           match="cover different layer sets") as err:
            lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                      _blob_bytes(partial), tmp_path / name,
                                      **LLAMA_BASE)
        assert name in str(err.value)
    # A_pos short: every other map disagrees with it
    with pytest.raises(lx.LoraExportError, match="cover different layer sets"):
        lx.export_genuine_adapter(
            {"kind": "lora_subtraction"},
            _blob_bytes(dict(good, A_pos={"0": good["A_pos"]["0"]})),
            tmp_path / "apos", **LLAMA_BASE)
    # the blob's own `layers` metadata disagreeing with the maps
    with pytest.raises(lx.LoraExportError,
                       match="disagrees with the A_pos/B_pos/A_neg/B_neg"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(dict(good, layers=[0, 5])),
                                  tmp_path / "meta", **LLAMA_BASE)
    # an empty map / non-mapping refuses naming the map
    with pytest.raises(lx.LoraExportError,
                       match="A_neg must be a non-empty mapping"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(dict(good, A_neg={})),
                                  tmp_path / "empty", **LLAMA_BASE)


def test_subtraction_refuses_per_layer_shape_mismatches(tiny_model,
                                                        tmp_path):
    good = _subtraction_blob(tiny_model, rank=4)
    d_in = good["A_pos"]["0"].shape[0]
    d_out = good["B_pos"]["0"].shape[0]
    # neg parent's d_out differs from pos's on one layer
    bad_b = dict(good, B_neg=dict(good["B_neg"],
                                  **{"1": torch.randn(d_out + 1, 4)}))
    with pytest.raises(lx.LoraExportError, match="parent shapes differ"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(bad_b), tmp_path / "b",
                                  **LLAMA_BASE)
    # neg parent's d_in differs
    bad_a = dict(good, A_neg=dict(good["A_neg"],
                                  **{"1": torch.randn(d_in + 3, 4)}))
    with pytest.raises(lx.LoraExportError, match="parent shapes differ"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(bad_a), tmp_path / "a",
                                  **LLAMA_BASE)
    # a rank column-count that contradicts the object's rank, named by map
    bad_rank = dict(good, B_neg={k: v[:, :2] for k, v in good["B_neg"].items()})
    with pytest.raises(lx.LoraExportError,
                       match=r"B_neg .* rank the config would mis-declare"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(bad_rank), tmp_path / "r",
                                  **LLAMA_BASE)
    # BOTH parents changing shape together on one layer is the lora_all
    # cross-layer refusal (o_proj shares its shape across layers)
    both = dict(good,
                A_pos=dict(good["A_pos"], **{"1": torch.randn(d_in + 1, 4)}),
                A_neg=dict(good["A_neg"], **{"1": torch.randn(d_in + 1, 4)}))
    with pytest.raises(lx.LoraExportError, match="corrupt blob"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(both), tmp_path / "both",
                                  **LLAMA_BASE)


def test_subtraction_refuses_position_masked_fits(tiny_model, tmp_path):
    """The parents' shared fit leg gates the export exactly as for every
    kind here: a response-only fit serves on the lens o_proj site (which
    carries the mask); native multi-LoRA cannot express it."""
    blob = _subtraction_blob(tiny_model, rank=4, positions="response")
    with pytest.raises(lx.LoraExportError,
                       match="DIFFERENT intervention") as err:
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(blob), tmp_path / "resp",
                                  **LLAMA_BASE)
    assert "lens o_proj" in str(err.value)
    for positions, match in (("last", "position-masked"),
                             ("prompt", "position-masked"),
                             ("sideways", "not a known position mode")):
        blob = _subtraction_blob(tiny_model, rank=4, positions=positions)
        with pytest.raises(lx.LoraExportError, match=match):
            lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                      _blob_bytes(blob),
                                      tmp_path / positions, **LLAMA_BASE)
    good = _subtraction_blob(tiny_model, rank=4)
    nopos = {k: v for k, v in good.items() if k != "positions"}
    with pytest.raises(lx.LoraExportError, match=r"lacks \['positions'\]"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(nopos), tmp_path / "nopos",
                                  **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="positions must be a string"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(dict(good, positions=3)),
                                  tmp_path / "int", **LLAMA_BASE)


def test_subtraction_refuses_malformed_blobs(tiny_model, tmp_path):
    good = _subtraction_blob(tiny_model, rank=4)
    cases = [
        ({k: v for k, v in good.items() if k != "A_neg"}, r"lacks \['A_neg'\]"),
        (dict(good, extra=1), "unexpected keys"),
        (dict(good, A_pos={k: v.double() for k, v in good["A_pos"].items()}),
         "float32"),
        (dict(good, B_neg={k: v.double() for k, v in good["B_neg"].items()}),
         "float32"),
        (dict(good, A_neg={k: torch.full_like(v, float("nan"))
                           for k, v in good["A_neg"].items()}), "non-finite"),
        (dict(good, A_pos={k: "not-a-tensor" for k in good["A_pos"]}),
         "torch.Tensor"),
        (dict(good, response_header_len=-1),
         "response_header_len must be an int"),
        (dict(good, provenance="mix-minus-ctl"), "provenance must be a mapping"),
        (dict(good, provenance={"parent_pos": ["x"]}),
         r"provenance\['parent_pos'\] must be a mapping"),
        (dict(good, A_pos={"x": good["A_pos"]["0"], "1": good["A_pos"]["1"]}),
         "keys must be layer indices"),
    ]
    for i, (blob, match) in enumerate(cases):
        with pytest.raises(lx.LoraExportError, match=match):
            lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                      _blob_bytes(blob), tmp_path / str(i),
                                      **LLAMA_BASE)
    # the bytes must store THIS kind
    with pytest.raises(lx.LoraExportError, match="stores kind"):
        lx.export_genuine_adapter(
            {"kind": "lora_subtraction"},
            _blob_bytes({"kind": "lora_all", "A": {}, "B": {},
                         "positions": "all"}),
            tmp_path / "wrongkind", **LLAMA_BASE)


def test_subtraction_sidecar_and_spec_cross_checks(tiny_model, tmp_path):
    art = _blob_bytes(_subtraction_blob(tiny_model, rank=4))
    spec = {"kind": "lora_subtraction"}
    with pytest.raises(lx.LoraExportError, match="unrecognized keys"):
        lx.export_genuine_adapter(spec, art, tmp_path / "a", **LLAMA_BASE,
                                  sidecar_bytes=b'{"mystery": 1}')
    with pytest.raises(lx.LoraExportError, match="disagree"):
        lx.export_genuine_adapter(
            spec, art, tmp_path / "b", **LLAMA_BASE,
            sidecar_bytes=b'{"positions_at_fit": "response"}')
    with pytest.raises(lx.LoraExportError, match="unparseable"):
        lx.export_genuine_adapter(spec, art, tmp_path / "c", **LLAMA_BASE,
                                  sidecar_bytes=b"not json")
    out = lx.export_genuine_adapter(
        spec, art, tmp_path / "ok", **LLAMA_BASE,
        sidecar_bytes=b'{"positions_at_fit": "all"}')
    assert (out / lx.ADAPTER_WEIGHTS).is_file()
    # F9-style: the resolved spec must agree with the bytes
    for bad_spec in ({"kind": "lora_subtraction", "rank": 2},
                     {"kind": "lora_subtraction", "layers": [0]},
                     {"kind": "lora_subtraction", "positions": "response"}):
        with pytest.raises(lx.LoraExportError, match="cross-check"):
            lx.export_genuine_adapter(bad_spec, art, tmp_path / "x",
                                      **LLAMA_BASE)
    # and an intervention_hash that is present must be a non-empty string
    with pytest.raises(lx.LoraExportError, match="intervention_hash"):
        lx.export_genuine_adapter(
            {"kind": "lora_subtraction", "intervention_hash": ""}, art,
            tmp_path / "h", **LLAMA_BASE)


def test_subtraction_refuses_layers_without_an_attention_oproj(tiny_model):
    """With the layout resolved off the tree (NemotronH: attention at
    [12, 17, 24, 32]) the fit layers 0 and 1 name modules the base does
    not have — refuse, never emit a name that points at nothing."""
    blob = _subtraction_blob(tiny_model, rank=4)   # layers 0 and 1
    with pytest.raises(lx.LoraExportError, match="carry no attention o_proj"):
        lx._export_lora_subtraction(
            {"kind": "lora_subtraction"}, blob, decoder_prefix="model.layers",
            oproj_site="mixer.o_proj", oproj_layers=(12, 17, 24, 32))


def test_subtraction_provenance_copies_strings_only(tiny_model, tmp_path):
    """Only ``cell_id`` / ``object_sha256`` STRINGS ride into the config;
    anything else (a missing block, an int, a tensor) is recorded as
    None — the config is JSON and part of the adapter's hashed identity."""
    good = _subtraction_blob(tiny_model, rank=4)
    weird = dict(good, provenance={
        "parent_pos": {"cell_id": 5, "object_sha256": torch.zeros(2),
                       "unrelated": "ignored"},
        "parent_neg": {"cell_id": "", "object_sha256": "c" * 64}})
    out = lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                    _blob_bytes(weird), tmp_path / "weird",
                                    **LLAMA_BASE)
    cfg, _ = _load(out)
    assert cfg["petri_lora_subtraction_provenance"] == {
        "parent_pos": {"cell_id": None, "object_sha256": None},
        "parent_neg": {"cell_id": None, "object_sha256": "c" * 64}}
    # an ad-hoc object (empty provenance) and a legacy blob with none
    for i, blob in enumerate((dict(good, provenance={}),
                              {k: v for k, v in good.items()
                               if k != "provenance"})):
        out = lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                        _blob_bytes(blob), tmp_path / str(i),
                                        **LLAMA_BASE)
        cfg, _ = _load(out)
        assert cfg["petri_lora_subtraction_provenance"] == {
            "parent_pos": {"cell_id": None, "object_sha256": None},
            "parent_neg": {"cell_id": None, "object_sha256": None}}


# --------------------------------------------------------------------------
# the real bank shapes, under their own layouts
# --------------------------------------------------------------------------


def _bank_shaped_blob(layers, d_in, d_out, rank=4, seed=3):
    """A subtraction blob with the REAL bank's per-layer shapes (synthetic
    values): both parents' factor maps, the shared rank and positions."""
    g = torch.Generator().manual_seed(seed)
    maps = {}
    for name, d in (("A_pos", d_in), ("B_pos", d_out),
                    ("A_neg", d_in), ("B_neg", d_out)):
        maps[name] = {str(l): torch.randn(d, rank, generator=g)
                      for l in layers}
    return {"kind": "lora_subtraction", "layers": list(layers), "rank": rank,
            **maps, "provenance": dict(PROVENANCE), "positions": "all",
            "response_header_len": None}


def test_subtraction_exports_the_gemma_bank_shapes(tmp_path):
    """gemma4e4b under a CALLER-STATED layout (no tree): 42 attention
    layers at the LOCAL-attention shape A [2048, 4] / B [2560, 4] (d_in !=
    d_out, so a transpose could not hide), under the VLM-wrapper prefix
    ``model.language_model.layers`` + ``self_attn.o_proj``. The real E4B is
    NOT uniform — its 7 global-attention layers have d_in 4096 — and that
    mix exports only through the derived layout, where each layer is held
    to its own module (``test_subtraction_admits_a_mixed_width_base_and_
    refuses_a_layer_that_does_not_fit_its_module`` below); with no tree the
    export keeps the historical one-shape heuristic, which this uniform
    blob satisfies."""
    layers = list(range(42))
    blob = _bank_shaped_blob(layers, d_in=2048, d_out=2560)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction", "rank": 4, "layers": layers,
         "intervention_hash": IH},
        _blob_bytes(blob), tmp_path / "gemma",
        base_model="google/gemma-4-E4B-it", decoder_prefix=WRAPPED_PREFIX)
    cfg, tensors = _load(adapter)
    assert cfg["r"] == 8 and cfg["lora_alpha"] == 8.0
    assert cfg["layers_to_transform"] == layers
    assert cfg["petri_decoder_prefix"] == WRAPPED_PREFIX
    assert cfg["petri_lora_subtraction_parent_rank"] == 4
    assert "petri_oproj_site" not in cfg
    assert len(tensors) == 2 * 42
    for l in (0, 21, 41):
        prefix = f"base_model.model.{WRAPPED_PREFIX}.{l}.self_attn.o_proj"
        lora_A, lora_B = tensors[f"{prefix}.lora_A.weight"], \
            tensors[f"{prefix}.lora_B.weight"]
        assert tuple(lora_A.shape) == (8, 2048)
        assert tuple(lora_B.shape) == (2560, 8)
        assert torch.equal(lora_A[:4], blob["A_pos"][str(l)].t())
        assert torch.equal(lora_A[4:], blob["A_neg"][str(l)].t())
        assert torch.equal(lora_B[:, :4], blob["B_pos"][str(l)])
        assert torch.equal(lora_B[:, 4:], -blob["B_neg"][str(l)])


def test_subtraction_exports_the_nemotron_bank_shapes(tmp_path):
    """nemotron3nano4b: the 4 attention blocks [12, 17, 24, 32] among 42,
    A [5120, 4] / B [3136, 4], site ``mixer.o_proj`` (recorded, since it is
    not the default) — both with the caller-stated layout and with the
    tree-DERIVED o_proj set, which the fit layers must lie inside."""
    layers = [12, 17, 24, 32]
    blob = _bank_shaped_blob(layers, d_in=5120, d_out=3136)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction", "rank": 4, "layers": layers},
        _blob_bytes(blob), tmp_path / "nemo",
        base_model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        decoder_prefix="model.layers", oproj_site="mixer.o_proj")
    cfg, tensors = _load(adapter)
    assert cfg["r"] == 8 and cfg["lora_alpha"] == 8.0
    assert cfg["layers_to_transform"] == layers
    assert cfg["petri_source_layers"] == layers
    assert cfg["petri_oproj_site"] == "mixer.o_proj"
    assert cfg["target_modules"] == ["o_proj"]
    assert sorted(tensors) == sorted(
        f"base_model.model.model.layers.{l}.mixer.o_proj.{w}.weight"
        for l in layers for w in ("lora_A", "lora_B"))
    for l in layers:
        prefix = f"base_model.model.model.layers.{l}.mixer.o_proj"
        assert tuple(tensors[f"{prefix}.lora_A.weight"].shape) == (8, 5120)
        assert tuple(tensors[f"{prefix}.lora_B.weight"].shape) == (3136, 8)
        assert torch.equal(tensors[f"{prefix}.lora_B.weight"][:, 4:],
                           -blob["B_neg"][str(l)])
    # the derived-layout path: the fit layers lie inside the tree's set
    t2, c2 = lx._export_lora_subtraction(
        {"kind": "lora_subtraction"}, blob, decoder_prefix="model.layers",
        oproj_site="mixer.o_proj", oproj_layers=(12, 17, 24, 32))
    assert sorted(t2) == sorted(tensors)
    assert c2["layers_to_transform"] == layers
    # and a fit layer outside it refuses (a Mamba2/MLP block)
    outside = _bank_shaped_blob([12, 17, 24, 33], d_in=5120, d_out=3136)
    with pytest.raises(lx.LoraExportError, match=r"fit layers \[33\] carry"):
        lx._export_lora_subtraction(
            {"kind": "lora_subtraction"}, outside,
            decoder_prefix="model.layers", oproj_site="mixer.o_proj",
            oproj_layers=(12, 17, 24, 32))


# --------------------------------------------------------------------------
# a MIXED-WIDTH base (native-multilora): per-layer shapes are held to
# each layer's OWN o_proj module, never to a neighbour's
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hetero_base(tmp_path_factory):
    """A tiny gemma-4 TEXT config with two attention widths — ``head_dim`` 8
    on two sliding layers, ``global_head_dim`` 16 on one full-attention layer
    — so o_proj.in_features is 32 on layers 0-1 and 64 on layer 2 with
    out_features 32 everywhere: the real E4B's 2048/4096 -> 2560 pattern in
    miniature, through the REAL derivation path (a local config dir)."""
    from transformers import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=16,
        layer_types=["sliding_attention", "sliding_attention",
                     "full_attention"],
        vocab_size=97, max_position_embeddings=64, sliding_window=16,
        num_kv_shared_layers=0, hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=97)
    d = tmp_path_factory.mktemp("tiny-gemma4-text-snapshot")
    cfg.save_pretrained(d)
    # the export spells module names off the checkpoint's weights. A
    # text-only Gemma4 checkpoint spells its decoder as the skeleton does
    # (model.layers), so a header-only safetensors with those names stands in
    write_safetensors_snapshot(
        d, [f"model.layers.{l}.self_attn.o_proj.weight" for l in range(3)]
        + ["model.embed_tokens.weight"])
    return str(d)


def _hetero_subtraction_blob(shapes, rank=4, seed=5):
    g = torch.Generator().manual_seed(seed)
    layers = sorted(shapes)
    maps = {}
    for name, idx in (("A_pos", 0), ("B_pos", 1), ("A_neg", 0), ("B_neg", 1)):
        maps[name] = {str(l): torch.randn(shapes[l][idx], rank, generator=g)
                      for l in layers}
    return {"kind": "lora_subtraction", "layers": layers, "rank": rank,
            **maps, "provenance": dict(PROVENANCE), "positions": "all",
            "response_header_len": None}


def test_subtraction_admits_a_mixed_width_base_and_refuses_a_layer_that_does_not_fit_its_module(
        hetero_base, tmp_path):
    """Layer 2's parents are twice as wide as layers 0-1's — the base's own
    shapes — and the export admits them under the derived layout (the
    one-shape-across-layers rule used to call this a corrupt blob); each
    stacked pair has ITS module's shape and C's fingerprint accepts the dir.
    The wide layer shrunk to the narrow width refuses NAMING layer 2 and its
    module's (64, 32) — it agrees with its neighbours, which is exactly what
    the old rule admitted; under a caller-stated prefix (no tree) the
    historical heuristic stays and says why."""
    from sisteer.capeval.serving import worker_reports as wr

    _, shapes = lx._derived_layout(hetero_base)
    assert shapes == {0: (32, 32), 1: (32, 32), 2: (64, 32)}
    good = _hetero_subtraction_blob(shapes)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_subtraction", "rank": 4, "layers": [0, 1, 2],
         "intervention_hash": IH},
        _blob_bytes(good), tmp_path / "hetero", base_model=hetero_base)
    cfg, tensors = _load(adapter)
    assert cfg["r"] == 8 and cfg["layers_to_transform"] == [0, 1, 2]
    for l, (d_in, d_out) in shapes.items():
        prefix = f"base_model.model.model.layers.{l}.self_attn.o_proj"
        lora_A, lora_B = tensors[f"{prefix}.lora_A.weight"], \
            tensors[f"{prefix}.lora_B.weight"]
        assert tuple(lora_A.shape) == (8, d_in)
        assert tuple(lora_B.shape) == (d_out, 8)
        assert torch.equal(lora_A[:4], good["A_pos"][str(l)].t())
        assert torch.equal(lora_B[:, 4:], -good["B_neg"][str(l)])
    fingerprint, keys = wr.expected_adapter_fingerprint_bf16(adapter)
    assert len(fingerprint) == 64 and len(keys) == 3
    # both parents of layer 2 shrunk to the narrow width: uniform across
    # layers, wrong for the module -> refused by name
    narrow = dict(good,
                  A_pos=dict(good["A_pos"], **{"2": torch.randn(32, 4)}),
                  A_neg=dict(good["A_neg"], **{"2": torch.randn(32, 4)}))
    with pytest.raises(lx.LoraExportError,
                       match=r"layer 2 has \(d_in, d_out\)=\(32, 32\) but the "
                             r"base's own o_proj at that layer is .*\(64, 32\)"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(narrow), tmp_path / "narrow",
                                  base_model=hetero_base)
    # the same GOOD blob under a caller-stated prefix has no tree to hold
    # each layer to; the no-tree heuristic refuses and names the way out
    with pytest.raises(lx.LoraExportError, match="mixed-width base"):
        lx.export_genuine_adapter({"kind": "lora_subtraction"},
                                  _blob_bytes(good), tmp_path / "stated",
                                  base_model=hetero_base,
                                  decoder_prefix=lx.LLAMA_DECODER_PREFIX)
