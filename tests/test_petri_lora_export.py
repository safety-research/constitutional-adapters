"""export_genuine_adapter (two-SI): the landed weight-level kinds
(lora / circuit_breaker) -> real PEFT adapter dirs for vLLM's native
multi-LoRA (serving/lora_export.py).

Load-bearing here:

- the rank-1 conversion reproduces RankOneLoRAObject's hook delta
  BIT-EXACTLY through the exported tensors: ``dW = lora_B @ lora_A ==
  B ⊗ A``, and the PEFT arithmetic ``(x @ lora_A.T) @ lora_B.T ·
  (alpha/r)`` matches the hook's own o_proj output on a tiny synthetic
  model (torch.equal, not allclose — the §10 gate proves EXACT equality);
- determinism: same artifact bytes -> byte-identical adapter files;
- THE FIT LEG: a response-only (or any position-masked) fit REFUSES —
  native multi-LoRA has no position mask, so serving it all-position
  would be a DIFFERENT intervention under the fit's hash — and a sidecar
  ``positions_at_fit`` that disagrees with the blob refuses;
- rank>1 under the rank-1 kind refuses rather than mis-declaring r=1;
- circuit_breaker copies its PEFT-native A/B across the seven-projection
  grid with scaling alpha/r, and refuses partial grids and non-1.0
  base_scale;
- ``verify_zero_adapter`` never certifies a genuine adapter.

All fixtures are synthetic: a from-config tiny Llama (no hub access),
generator-seeded tensors, invented hashes.
"""

import io
import pathlib
import json

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx
from lora_snapshot_fixtures import write_safetensors_snapshot

HIDDEN = 32
N_LAYERS = 2
BASE = "meta-llama/Llama-3.1-8B-Instruct"
#: every existing export here pins the LLAMA layout (the
#: ``base_model.model.model.layers...`` strings asserted below), stated
#: explicitly so the tests stay hermetic — left unstated, the export
#: derives the layout from the base's locally cached config
#: (resolve_decoder_prefix), which a cache-less machine cannot do.
LLAMA_BASE = dict(base_model=BASE, decoder_prefix=lx.LLAMA_DECODER_PREFIX)
#: the VLM-wrapper layout (gemma-4, GLM-4.6V): the decoder sits under
#: ``language_model`` — one of models._LAYER_PATHS.
WRAPPED_PREFIX = "model.language_model.layers"


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


def _blob_bytes(d) -> bytes:
    buf = io.BytesIO()
    torch.save(d, buf)
    return buf.getvalue()


def _rank1_object(tiny_model, layer=1, seed=5):
    from sisteer.extraction.objects import RankOneLoRAObject

    obj = RankOneLoRAObject(tiny_model, layer=layer)
    g = torch.Generator().manual_seed(seed)
    obj.A.data = torch.randn(obj.A.shape[0], generator=g)
    obj.B.data = torch.randn(obj.B.shape[0], generator=g) * 0.1
    return obj


def _save_with_fit_leg(obj, path, positions="all", header_len=0):
    """Save through the real class, then set the fit-leg keys explicitly.

    ``RankOneLoRAObject.save`` writes both (``positions`` and
    ``response_header_len``) today; the exporter REQUIRES ``positions``: a
    blob that does not say which positions it was fit at cannot say which
    intervention it IS, and exporting it as all-position would be a guess
    wearing the fit's hash. Setting the keys here lets a test pin the fit
    lane's schema (and the requested leg) independently of the class's
    defaults."""
    obj.save(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    blob["positions"] = positions
    blob["response_header_len"] = header_len
    torch.save(blob, path)
    return path


def _rank1_blob(**overrides) -> dict:
    g = torch.Generator().manual_seed(23)
    blob = {"kind": "lora", "A": torch.randn(8, generator=g),
            "B": torch.randn(8, generator=g), "layer": 0,
            "positions": "all"}
    blob.update(overrides)
    return blob


# --------------------------------------------------------------------------
# rank-1 lora: the exact round-trip
# --------------------------------------------------------------------------


def test_rank1_export_reproduces_the_hook_delta_exactly(tiny_model, tmp_path):
    """The exported tensors reproduce the hook body's o_proj delta
    bit-exactly: dW = lora_B @ lora_A == B ⊗ A, and the served PEFT
    arithmetic equals the object's own apply() output (torch.equal)."""
    from safetensors.torch import load_file

    from sisteer.models import get_decoder_layers

    obj = _rank1_object(tiny_model, layer=1)
    art = tmp_path / "object.pt"
    _save_with_fit_leg(obj, art)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora", "layer": 1, "positions": "all",
         "intervention_hash": "ab" * 8},
        art.read_bytes(), tmp_path / "adapter", **LLAMA_BASE)

    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["r"] == 1 and cfg["lora_alpha"] == 1.0
    assert cfg["target_modules"] == ["o_proj"]
    assert cfg["layers_to_transform"] == [1]
    assert cfg["base_model_name_or_path"] == BASE
    # honest provenance — a GENUINE adapter, never the zero marker
    assert "petri_zero_adapter" not in cfg
    assert cfg["petri_genuine_adapter"] is True
    assert cfg["petri_source_kind"] == "lora"
    assert cfg["petri_source_layers"] == [1]
    assert cfg["petri_fit_positions"] == "all"
    assert cfg["petri_intervention_hash"] == "ab" * 8

    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    prefix = "base_model.model.model.layers.1.self_attn.o_proj"
    assert sorted(tensors) == [f"{prefix}.lora_A.weight",
                               f"{prefix}.lora_B.weight"]
    lora_A = tensors[f"{prefix}.lora_A.weight"]
    lora_B = tensors[f"{prefix}.lora_B.weight"]
    A = obj.A.detach().float().cpu()
    B = obj.B.detach().float().cpu()
    assert tuple(lora_A.shape) == (1, A.shape[0])   # [r, in_features]
    assert tuple(lora_B.shape) == (B.shape[0], 1)   # [out_features, r]
    assert torch.equal(lora_A.squeeze(0), A)
    assert torch.equal(lora_B.squeeze(1), B)
    # dW = lora_B @ lora_A is EXACTLY the hook's outer product B ⊗ A
    assert torch.equal(lora_B @ lora_A, torch.outer(B, A))

    # numeric ground truth: the hook's own o_proj output vs base + the
    # PEFT arithmetic on the exported tensors — bit-exact, not allclose
    mod = get_decoder_layers(tiny_model)[1].self_attn.o_proj
    g = torch.Generator().manual_seed(31)
    x = torch.randn(2, 5, mod.in_features, generator=g)
    with torch.no_grad():
        base_out = mod(x)
        with obj.apply(tiny_model, scale=1.0):
            hooked_out = mod(x)
        scaling = cfg["lora_alpha"] / cfg["r"]
        peft_delta = (x @ lora_A.t()) @ lora_B.t() * scaling
    assert not torch.equal(hooked_out, base_out)  # non-vacuous
    assert torch.equal(hooked_out, base_out + peft_delta)


def test_rank1_orientation_on_nonsquare_dims(tmp_path):
    """A carries in_features and B out_features (the hook body's
    orientation) — pinned on d_in != d_out, where a silent transpose
    could not hide behind square shapes."""
    from safetensors.torch import load_file

    g = torch.Generator().manual_seed(7)
    blob = _rank1_blob(A=torch.randn(10, generator=g),
                       B=torch.randn(6, generator=g), layer=3)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora"}, _blob_bytes(blob), tmp_path / "adapter",
        **LLAMA_BASE)
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    prefix = "base_model.model.model.layers.3.self_attn.o_proj"
    assert tuple(tensors[f"{prefix}.lora_A.weight"].shape) == (1, 10)
    assert tuple(tensors[f"{prefix}.lora_B.weight"].shape) == (6, 1)


def test_rank1_export_is_deterministic(tmp_path):
    """Same artifact bytes -> byte-identical adapter files."""
    art = _blob_bytes(_rank1_blob())
    spec = {"kind": "lora", "layer": 0}
    a = lx.export_genuine_adapter(spec, art, tmp_path / "a", **LLAMA_BASE)
    b = lx.export_genuine_adapter(spec, art, tmp_path / "b", **LLAMA_BASE)
    assert (a / lx.ADAPTER_WEIGHTS).read_bytes() \
        == (b / lx.ADAPTER_WEIGHTS).read_bytes()
    assert (a / lx.ADAPTER_CONFIG).read_bytes() \
        == (b / lx.ADAPTER_CONFIG).read_bytes()


# --------------------------------------------------------------------------
# rank-1 lora: refusals
# --------------------------------------------------------------------------


def test_rank1_refuses_a_higher_rank_representation(tmp_path):
    """[r, d_in]/[d_out, r] matrices under the rank-1 kind refuse —
    never a silently mis-declared r=1 config."""
    g = torch.Generator().manual_seed(13)
    blob = _rank1_blob(A=torch.randn(2, 8, generator=g),
                       B=torch.randn(8, 2, generator=g))
    with pytest.raises(lx.LoraExportError, match="rank-2"):
        lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(blob),
                                  tmp_path / "x", **LLAMA_BASE)
    # mismatched 2-D shapes are a plain shape refusal, not a rank claim
    blob = _rank1_blob(A=torch.randn(2, 8, generator=g),
                       B=torch.randn(8, 3, generator=g))
    with pytest.raises(lx.LoraExportError, match=r"A must be \[d_in\]"):
        lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(blob),
                                  tmp_path / "y", **LLAMA_BASE)


def test_rank1_refuses_a_response_only_fit(tmp_path):
    """The fit leg: a response-only fit served all-position is a
    DIFFERENT intervention — hard refusal naming the honest serving
    path (the lens o_proj module site, which carries the response
    mask). The refusal message was reworded when that site landed
    2026-08-28; this assertion tracked the older masked-LoRA-extension
    wording and went stale with it."""
    blob = _rank1_blob(positions="response")
    with pytest.raises(lx.LoraExportError,
                       match="DIFFERENT intervention") as err:
        lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(blob),
                                  tmp_path / "x", **LLAMA_BASE)
    msg = str(err.value)
    assert "lens o_proj" in msg and "response" in msg
    # every other position-masked mode refuses too; unknown values refuse
    for positions, match in (("last", "position-masked"),
                             ("prompt", "position-masked"),
                             ("front", "position-masked"),
                             ("sideways", "not a known position mode")):
        blob = _rank1_blob(positions=positions)
        with pytest.raises(lx.LoraExportError, match=match):
            lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(blob),
                                      tmp_path / positions, **LLAMA_BASE)


def test_rank1_sidecar_positions_must_agree(tmp_path):
    """A sidecar positions_at_fit that contradicts the blob refuses; an
    agreeing one (and an absent sidecar) exports; unrecognized sidecar
    content refuses (it is hashed into the intervention identity)."""
    art = _blob_bytes(_rank1_blob(positions="all"))
    spec = {"kind": "lora"}
    with pytest.raises(lx.LoraExportError, match="disagree"):
        lx.export_genuine_adapter(
            spec, art, tmp_path / "x", **LLAMA_BASE,
            sidecar_bytes=json.dumps({"positions_at_fit": "response"})
            .encode("utf-8"))
    with pytest.raises(lx.LoraExportError, match="unrecognized keys"):
        lx.export_genuine_adapter(
            spec, art, tmp_path / "y", **LLAMA_BASE,
            sidecar_bytes=b'{"mystery": 1}')
    with pytest.raises(lx.LoraExportError, match="unparseable"):
        lx.export_genuine_adapter(spec, art, tmp_path / "z",
                                  **LLAMA_BASE, sidecar_bytes=b"not json")
    out = lx.export_genuine_adapter(
        spec, art, tmp_path / "ok", **LLAMA_BASE,
        sidecar_bytes=b'{"positions_at_fit": "all"}')
    assert (out / lx.ADAPTER_WEIGHTS).is_file()


def test_export_refuses_wrong_and_unknown_kinds(tmp_path):
    # blob bytes storing another kind under a lora cell
    psr_blob = _blob_bytes({"kind": "psr", "layers": [0]})
    with pytest.raises(lx.LoraExportError, match="stores kind"):
        lx.export_genuine_adapter({"kind": "lora"}, psr_blob,
                                  tmp_path / "x", **LLAMA_BASE)
    # kinds with no committed genuine export refuse loudly
    for kind in ("loreft", "lora_r1", "token", None):
        with pytest.raises(lx.LoraExportError,
                           match="no genuine PEFT export"):
            lx.export_genuine_adapter({"kind": kind}, b"bytes",
                                      tmp_path / "y", **LLAMA_BASE)


def test_rank1_refuses_malformed_blobs(tmp_path):
    cases = [
        ({"kind": "lora", "A": torch.randn(8), "layer": 0},
         "lacks"),                                       # missing B
        (_rank1_blob(extra=1), "unexpected keys"),
        (_rank1_blob(layer=-1), "layer must be an int"),
        (_rank1_blob(layer=True), "layer must be an int"),
        (_rank1_blob(A=torch.full((8,), float("nan"))), "non-finite"),
        (_rank1_blob(A=torch.randn(8, dtype=torch.float64)), "float32"),
        (_rank1_blob(A="not-a-tensor"), "torch.Tensor"),
        (_rank1_blob(response_header_len=-3),
         "response_header_len must be an int"),
        (_rank1_blob(positions=7), "positions must be a string"),
    ]
    for i, (blob, match) in enumerate(cases):
        with pytest.raises(lx.LoraExportError, match=match):
            lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(blob),
                                      tmp_path / str(i), **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="empty artifact"):
        lx.export_genuine_adapter({"kind": "lora"}, b"", tmp_path / "e",
                                  **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="torch payload"):
        lx.export_genuine_adapter({"kind": "lora"}, b"garbage",
                                  tmp_path / "g", **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="base_model"):
        lx.export_genuine_adapter({"kind": "lora"},
                                  _blob_bytes(_rank1_blob()),
                                  tmp_path / "b", base_model="")
    with pytest.raises(lx.LoraExportError, match="intervention_hash"):
        lx.export_genuine_adapter({"kind": "lora", "intervention_hash": ""},
                                  _blob_bytes(_rank1_blob()),
                                  tmp_path / "h", **LLAMA_BASE)


def test_rank1_export_accepts_the_classes_own_none_header_len(tiny_model, tmp_path):
    """THE FINDING (code review 2026-09-11, P2): ``RankOneLoRAObject.save``
    writes ``response_header_len: None`` on every all-position fit — its own
    default — and the rank-one exporter refused it ("must be an int >= 0"),
    so a valid admitted ``lora_r1`` cell aborted the native standup. The
    UNTOUCHED save bytes now export, exactly as the multilayer and
    subtraction exports treat their ``None``; a real length still validates
    and a negative one still refuses."""
    from safetensors.torch import load_file

    obj = _rank1_object(tiny_model, layer=1)
    art = tmp_path / "object.pt"
    obj.save(art)                                     # the class's own bytes, untouched
    blob = torch.load(art, map_location="cpu", weights_only=True)
    assert blob["positions"] == "all" and blob["response_header_len"] is None
    adapter = lx.export_genuine_adapter(
        {"kind": "lora", "layer": 1, "positions": "all", "intervention_hash": "ab" * 8},
        art.read_bytes(), tmp_path / "adapter", **LLAMA_BASE)
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    a_key = "base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight"
    assert torch.equal(tensors[a_key], obj.A.detach().float().unsqueeze(0))
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["r"] == 1 and cfg["petri_intervention_hash"] == "ab" * 8
    # the hand-built shape agrees, and the int cases keep their meaning
    lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(_rank1_blob(response_header_len=None)),
                              tmp_path / "none", **LLAMA_BASE)
    lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(_rank1_blob(response_header_len=0)),
                              tmp_path / "zero", **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="response_header_len must be an int"):
        lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(_rank1_blob(response_header_len=-3)),
                                  tmp_path / "neg", **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="response_header_len must be an int"):
        lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(_rank1_blob(response_header_len=True)),
                                  tmp_path / "bool", **LLAMA_BASE)


def test_rank1_spec_cross_checks(tmp_path):
    art = _blob_bytes(_rank1_blob(layer=2))
    with pytest.raises(lx.LoraExportError, match="cross-check"):
        lx.export_genuine_adapter({"kind": "lora", "layer": 5}, art,
                                  tmp_path / "x", **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="cross-check"):
        lx.export_genuine_adapter({"kind": "lora", "positions": "response"},
                                  art, tmp_path / "y", **LLAMA_BASE)


def test_verify_zero_adapter_refuses_a_genuine_adapter(tmp_path):
    """The no-op certificate must never certify a genuine adapter."""
    adapter = lx.export_genuine_adapter(
        {"kind": "lora"}, _blob_bytes(_rank1_blob()), tmp_path / "adapter",
        **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError,
                       match="not marked petri_zero_adapter"):
        lx.verify_zero_adapter(adapter)


# --------------------------------------------------------------------------
# circuit_breaker: the multi-module rank-r conversion
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cb_blob(tiny_model):
    """A real CircuitBreakerObject (deterministic init — kaiming A, tiny
    Gaussian B, both non-zero) through its own save path."""
    from sisteer.extraction.circuit_breaker import CircuitBreakerObject

    import tempfile
    from pathlib import Path

    obj = CircuitBreakerObject(tiny_model, transform_layers=[0, 1], rank=2,
                               lora_alpha=4.0, target_layers=[1], seed=3)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "object.pt"
        obj.save(path)
        art = path.read_bytes()
    return obj, art


def test_circuit_breaker_export_reproduces_the_hook_delta_exactly(
        tiny_model, cb_blob, tmp_path):
    from safetensors.torch import load_file

    from sisteer.extraction.circuit_breaker import PROJECTIONS
    from sisteer.models import get_decoder_layers

    obj, art = cb_blob
    adapter = lx.export_genuine_adapter(
        {"kind": "circuit_breaker", "rank": 2}, art, tmp_path / "adapter",
        **LLAMA_BASE)
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["r"] == 2 and cfg["lora_alpha"] == 4.0
    assert cfg["target_modules"] == sorted(PROJECTIONS)
    assert cfg["layers_to_transform"] == [0, 1]
    assert "petri_zero_adapter" not in cfg
    assert cfg["petri_source_kind"] == "circuit_breaker"
    assert cfg["petri_source_target_layers"] == [1]
    assert cfg["petri_fit_positions"] == "all"

    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    assert len(tensors) == 2 * len(PROJECTIONS) * 2  # layers x proj x A/B
    # attention projections live under self_attn, MLP ones under mlp,
    # and the tensors ARE the object's own (PEFT-native layout)
    for (l, name), parent, in_dim in (
            ((0, "o_proj"), "self_attn", HIDDEN),
            ((1, "gate_proj"), "mlp", HIDDEN),
            ((1, "down_proj"), "mlp", 64)):
        prefix = f"base_model.model.model.layers.{l}.{parent}.{name}"
        lora_A = tensors[f"{prefix}.lora_A.weight"]
        lora_B = tensors[f"{prefix}.lora_B.weight"]
        assert tuple(lora_A.shape) == (2, in_dim)
        assert torch.equal(lora_A,
                           obj.A[(l, name)].detach().float().cpu())
        assert torch.equal(lora_B,
                           obj.B[(l, name)].detach().float().cpu())
        # numeric ground truth per adapted projection: the hook's own
        # output vs base + PEFT arithmetic on the exported tensors
        block = get_decoder_layers(tiny_model)[l]
        mod = getattr(getattr(block, parent), name)
        g = torch.Generator().manual_seed(41 + l)
        x = torch.randn(2, 5, mod.in_features, generator=g)
        with torch.no_grad():
            base_out = mod(x)
            with obj.apply(tiny_model, scale=1.0):
                hooked_out = mod(x)
            scaling = cfg["lora_alpha"] / cfg["r"]
            peft_delta = (x @ lora_A.t()) @ lora_B.t() * scaling
        assert not torch.equal(hooked_out, base_out)  # non-vacuous
        assert torch.equal(hooked_out, base_out + peft_delta)


def test_circuit_breaker_export_is_deterministic(cb_blob, tmp_path):
    _, art = cb_blob
    spec = {"kind": "circuit_breaker"}
    a = lx.export_genuine_adapter(spec, art, tmp_path / "a", **LLAMA_BASE)
    b = lx.export_genuine_adapter(spec, art, tmp_path / "b", **LLAMA_BASE)
    assert (a / lx.ADAPTER_WEIGHTS).read_bytes() \
        == (b / lx.ADAPTER_WEIGHTS).read_bytes()
    assert (a / lx.ADAPTER_CONFIG).read_bytes() \
        == (b / lx.ADAPTER_CONFIG).read_bytes()


def test_circuit_breaker_refusals(cb_blob, tmp_path):
    _, art = cb_blob

    def mutated(**changes):
        blob = torch.load(io.BytesIO(art), map_location="cpu",
                          weights_only=True)
        blob.update(changes)
        return _blob_bytes(blob)

    spec = {"kind": "circuit_breaker"}
    # base_scale is pinned at 1.0 — folding would reassociate the scaling
    with pytest.raises(lx.LoraExportError, match="base_scale"):
        lx.export_genuine_adapter(spec, mutated(base_scale=2.0),
                                  tmp_path / "a", **LLAMA_BASE)
    # a partial grid (a missing projection) refuses
    blob = torch.load(io.BytesIO(art), map_location="cpu",
                      weights_only=True)
    blob["A"] = dict(blob["A"])
    blob["A"].pop("1.gate_proj")
    with pytest.raises(lx.LoraExportError, match="seven-projection grid"):
        lx.export_genuine_adapter(spec, _blob_bytes(blob), tmp_path / "b",
                                  **LLAMA_BASE)
    # a declared rank the tensors contradict refuses
    with pytest.raises(lx.LoraExportError, match="declares rank"):
        lx.export_genuine_adapter(spec, mutated(rank=3), tmp_path / "c",
                                  **LLAMA_BASE)
    # representation drift refuses
    with pytest.raises(lx.LoraExportError, match="unexpected keys"):
        lx.export_genuine_adapter(spec, mutated(extra=1), tmp_path / "d",
                                  **LLAMA_BASE)
    # F9-style cross-check against the resolved spec
    with pytest.raises(lx.LoraExportError, match="cross-check"):
        lx.export_genuine_adapter({"kind": "circuit_breaker", "rank": 16},
                                  art, tmp_path / "e", **LLAMA_BASE)
    # the fit-leg sidecar check runs for this kind too
    with pytest.raises(lx.LoraExportError, match="disagree"):
        lx.export_genuine_adapter(
            spec, art, tmp_path / "f", **LLAMA_BASE,
            sidecar_bytes=b'{"positions_at_fit": "response"}')


def test_rank1_refuses_a_blob_with_no_fit_leg_recorded():
    """Absence is a REFUSAL, not "all":
    a keyless blob predates the fit-leg schema, and reading it as
    all-position would serve a possibly-decode-only object at every
    position under the fit's own hash."""
    blob = _rank1_blob()
    del blob["positions"]
    with pytest.raises(lx.LoraExportError, match="no 'positions' field"):
        lx.export_genuine_adapter({"kind": "lora"}, _blob_bytes(blob),
                                  pathlib.Path("/nonexistent"), **LLAMA_BASE)


# --------------------------------------------------------------------------
# the decoder layout: derived from the base's module tree, never hardcoded
# (GLM-4.6V / gemma-4 wrap the decoder under language_model)
# --------------------------------------------------------------------------


def _tiny_glm4v_skeleton():
    """A weight-free ``Glm4vForConditionalGeneration`` on the meta device:
    the real transformers class (5.14 registers glm4v ONLY under the
    image-text-to-text auto-map), a 2-layer text stack, a 1-block ViT."""
    from transformers import Glm4vConfig, Glm4vForConditionalGeneration

    cfg = Glm4vConfig(
        text_config=dict(
            hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=97,
            max_position_embeddings=64, attention_bias=True,
            rope_parameters=dict(mrope_section=[2, 3, 3],
                                 partial_rotary_factor=0.5,
                                 rope_theta=10000.0, rope_type="default")),
        vision_config=dict(
            hidden_size=32, depth=1, num_heads=2, out_hidden_size=HIDDEN,
            intermediate_size=64, patch_size=14, image_size=28,
            spatial_merge_size=2, temporal_patch_size=1, in_channels=3),
        image_token_id=90, video_token_id=91, image_start_token_id=92,
        image_end_token_id=93, video_start_token_id=94,
        video_end_token_id=95)
    with torch.device("meta"):
        return Glm4vForConditionalGeneration(cfg)


def test_decoder_layers_path_follows_the_module_tree(tiny_model):
    """models.decoder_layers_path is the attribute path get_decoder_layers
    resolves — llama's ``model.layers``; the VLM wrapper's
    ``model.language_model.layers`` (the glm4v HF tree names its decoder
    o_proj ``model.language_model.layers.N.self_attn.o_proj``); a module
    with no known layout refuses with the same message."""
    from sisteer.models import (_LAYER_PATHS, decoder_layers_path,
                                get_decoder_layers, hidden_size, num_layers)

    assert decoder_layers_path(tiny_model) == lx.LLAMA_DECODER_PREFIX
    assert get_decoder_layers(tiny_model) is tiny_model.model.layers

    glm = _tiny_glm4v_skeleton()
    assert decoder_layers_path(glm) == WRAPPED_PREFIX
    assert WRAPPED_PREFIX in _LAYER_PATHS
    layers = get_decoder_layers(glm)
    assert len(layers) == 2 and num_layers(glm) == 2
    assert hidden_size(glm) == HIDDEN          # falls back to text_config
    names = {n for n, _ in glm.named_modules()}
    assert f"{WRAPPED_PREFIX}.0.self_attn.o_proj" in names
    assert "model.layers.0.self_attn.o_proj" not in names

    with pytest.raises(ValueError, match="Could not locate decoder layers"):
        decoder_layers_path(torch.nn.Linear(2, 2))


def _vllm_0_25_1_glm4v_module(peft_key: str) -> str:
    """MIRROR of the pinned vLLM 0.25.1 LoRA name resolution for glm4_1v,
    read at the tag — NOT an import (the wheel is GPU-only):
    ``lora/utils.parse_fine_tuned_lora_name`` strips ``base_model.model.``,
    applies the model's ``hf_to_vllm_mapper`` (``worker_manager.py`` hands
    ``get_unstacked_mapper()`` in as ``weights_mapper``; glm4_1v's
    ``orig_to_new_prefix`` carries ``"model.language_model." ->
    "language_model.model."``), then drops the ``lora_A/lora_B.weight``
    tail. The live load is GPU-VERIFY; this pins what the export must
    spell for that resolution to land on the served module."""
    assert peft_key.startswith("base_model.model.")
    name = peft_key[len("base_model.model."):]
    for orig, new in {"lm_head.": "language_model.lm_head.",
                      "model.language_model.": "language_model.model.",
                      "model.visual.": "visual."}.items():
        if name.startswith(orig):
            name = new + name[len(orig):]
            break
    parts = name.split(".")
    assert parts[-1] == "weight" and parts[-2] in ("lora_A", "lora_B")
    return ".".join(parts[:-2])


def test_rank1_export_on_a_wrapped_decoder_spells_the_hf_path(tmp_path):
    """Under the VLM-wrapper layout the export names the HF attribute
    path (what PEFT itself would write), and the pinned vLLM's own mapper
    turns exactly that into the served module
    ``language_model.model.layers.N.self_attn.o_proj`` — emitting vLLM's
    internal path here would be double-mapped and load nothing."""
    from safetensors.torch import load_file

    blob = _rank1_blob(layer=3)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora", "layer": 3}, _blob_bytes(blob), tmp_path / "glm",
        base_model="zai-org/GLM-4.6V-Flash", decoder_prefix=WRAPPED_PREFIX)
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    prefix = f"base_model.model.{WRAPPED_PREFIX}.3.self_attn.o_proj"
    assert sorted(tensors) == [f"{prefix}.lora_A.weight",
                               f"{prefix}.lora_B.weight"]
    for key in tensors:
        assert (_vllm_0_25_1_glm4v_module(key)
                == "language_model.model.layers.3.self_attn.o_proj")
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_decoder_prefix"] == WRAPPED_PREFIX
    assert cfg["target_modules"] == ["o_proj"]     # short names, unchanged
    assert cfg["layers_to_transform"] == [3]
    # the llama export is byte-for-byte what it always was
    llama = lx.export_genuine_adapter(
        {"kind": "lora", "layer": 3}, _blob_bytes(blob), tmp_path / "llama",
        **LLAMA_BASE)
    lt = load_file(str(llama / lx.ADAPTER_WEIGHTS))
    assert sorted(lt) == [
        "base_model.model.model.layers.3.self_attn.o_proj.lora_A.weight",
        "base_model.model.model.layers.3.self_attn.o_proj.lora_B.weight"]
    assert json.loads((llama / lx.ADAPTER_CONFIG).read_text())[
        "petri_decoder_prefix"] == lx.LLAMA_DECODER_PREFIX


def test_circuit_breaker_export_honours_the_decoder_prefix(cb_blob, tmp_path):
    from safetensors.torch import load_file

    obj, art = cb_blob
    adapter = lx.export_genuine_adapter(
        {"kind": "circuit_breaker"}, art, tmp_path / "cb",
        base_model="zai-org/GLM-4.6V-Flash", decoder_prefix=WRAPPED_PREFIX)
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    assert tensors
    assert all(k.startswith(f"base_model.model.{WRAPPED_PREFIX}.")
               for k in tensors)
    assert all(_vllm_0_25_1_glm4v_module(k).startswith(
        "language_model.model.layers.") for k in tensors)


def test_export_refuses_a_free_form_decoder_prefix(tmp_path):
    """Only a KNOWN layout (models._LAYER_PATHS) may be stated — a free
    string would emit module names tied to no module tree."""
    art = _blob_bytes(_rank1_blob())
    for bogus in ("layers", "model.layers.bogus", "", 7,
                  "language_model.model.layers.self_attn"):
        with pytest.raises(lx.LoraExportError,
                           match="not a known decoder layout"):
            lx.export_genuine_adapter({"kind": "lora"}, art, tmp_path / "x",
                                      base_model=BASE, decoder_prefix=bogus)


def test_export_derives_the_prefix_from_the_local_config_or_refuses(
        tmp_path):
    """No stated prefix -> derived from the base's LOCALLY cached config;
    a base that is not cached refuses loudly (naming the fix) rather
    than assuming llama's ``model.layers``. local_files_only: this never
    reaches the network."""
    art = _blob_bytes(_rank1_blob())
    with pytest.raises(lx.LoraExportError,
                       match="not in the local HF cache") as err:
        lx.export_genuine_adapter({"kind": "lora"}, art, tmp_path / "x",
                                  base_model="TESTORG/not-cached-anywhere")
    assert "decoder_prefix=" in str(err.value)
    # the refusal on kind still comes first (the default-deny edge the
    # serving gate pins), before any layout resolution is attempted
    with pytest.raises(lx.LoraExportError, match="no genuine PEFT export"):
        lx.export_genuine_adapter({"kind": "not_a_kind"}, b"bytes",
                                  tmp_path / "k",
                                  base_model="TESTORG/not-cached-anywhere")
    # the derivation itself, on a cached llama config when present
    try:
        prefix = lx.resolve_decoder_prefix(
            BASE, "0e9e39f249a16976918f6564b8830bc894c89659")
    except lx.LoraExportError as e:
        if "not in the local HF cache" not in str(e):
            raise
        pytest.skip(f"pinned {BASE} config not in the local HF cache")
    assert prefix == lx.LLAMA_DECODER_PREFIX


# --------------------------------------------------------------------------
# lora_all (MultiLayerLoRAObject): the all-layer rank-r o_proj LoRA
# --------------------------------------------------------------------------


def _multilayer_object(tiny_model, rank=1, seed=7, layers=None):
    """A MultiLayerLoRAObject with non-zero, generator-seeded factors on
    every targeted layer (B is zero-init by construction, so it is filled
    here to make the delta non-vacuous)."""
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    obj = MultiLayerLoRAObject(tiny_model, layers=layers, rank=rank)
    g = torch.Generator().manual_seed(seed)
    for l in obj.layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g)
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.1
    if not hasattr(obj, "fit_positions"):
        obj.fit_positions = "all"
        obj.fit_response_header_len = None
    return obj


def _multilayer_blob(tiny_model, rank=1, positions="all", **overrides):
    obj = _multilayer_object(tiny_model, rank=rank)
    blob = {"kind": "lora_all", "layers": list(obj.layers), "rank": rank,
            "A": {str(l): obj.A[l].detach().float().cpu() for l in obj.layers},
            "B": {str(l): obj.B[l].detach().float().cpu() for l in obj.layers},
            "positions": positions, "response_header_len": None}
    blob.update(overrides)
    return obj, blob


@pytest.mark.parametrize("rank", [1, 4])
def test_lora_all_export_reproduces_the_hook_delta_exactly(
        tiny_model, tmp_path, rank):
    """Every targeted layer's exported pair reproduces the hook body's
    o_proj delta bit-exactly: rank 1 keeps the legacy outer-product path,
    rank r the (x @ A) @ B.T path; lora_alpha == r makes PEFT's scaling
    exactly 1, so the served arithmetic IS the hook's arithmetic."""
    from safetensors.torch import load_file

    from sisteer.models import get_decoder_layers

    obj = _multilayer_object(tiny_model, rank=rank)
    art = tmp_path / "object.pt"
    obj.save(art)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_all", "rank": rank, "positions": "all",
         "layers": list(obj.layers), "intervention_hash": "cd" * 8},
        art.read_bytes(), tmp_path / "adapter", **LLAMA_BASE)

    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["r"] == rank and cfg["lora_alpha"] == float(rank)
    assert cfg["target_modules"] == ["o_proj"]
    assert cfg["layers_to_transform"] == list(range(N_LAYERS))
    assert cfg["petri_genuine_adapter"] is True
    assert "petri_zero_adapter" not in cfg
    assert cfg["petri_source_kind"] == "lora_all"
    assert cfg["petri_source_layers"] == list(range(N_LAYERS))
    assert cfg["petri_fit_positions"] == "all"
    assert cfg["petri_decoder_prefix"] == lx.LLAMA_DECODER_PREFIX
    assert "petri_oproj_site" not in cfg   # default site adds nothing

    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    expected = sorted(
        f"base_model.model.model.layers.{l}.self_attn.o_proj.{w}.weight"
        for l in range(N_LAYERS) for w in ("lora_A", "lora_B"))
    assert sorted(tensors) == expected

    decoder = get_decoder_layers(tiny_model)
    g = torch.Generator().manual_seed(41)
    for l in obj.layers:
        prefix = f"base_model.model.model.layers.{l}.self_attn.o_proj"
        lora_A, lora_B = tensors[f"{prefix}.lora_A.weight"], \
            tensors[f"{prefix}.lora_B.weight"]
        A = obj.A[l].detach().float().cpu()
        B = obj.B[l].detach().float().cpu()
        mod = decoder[l].self_attn.o_proj
        assert tuple(lora_A.shape) == (rank, mod.in_features)
        assert tuple(lora_B.shape) == (mod.out_features, rank)
        if rank == 1:
            assert torch.equal(lora_A.squeeze(0), A)
            assert torch.equal(lora_B.squeeze(1), B)
            assert torch.equal(lora_B @ lora_A, torch.outer(B, A))
        else:
            assert torch.equal(lora_A, A.t())
            assert torch.equal(lora_B, B)
            assert torch.equal(lora_B @ lora_A, B @ A.t())
        x = torch.randn(2, 5, mod.in_features, generator=g)
        with torch.no_grad():
            base_out = mod(x)
            with obj.apply(tiny_model, scale=1.0):
                hooked_out = mod(x)
            peft_delta = (x @ lora_A.t()) @ lora_B.t() * (cfg["lora_alpha"]
                                                          / cfg["r"])
        assert not torch.equal(hooked_out, base_out)   # non-vacuous
        if rank == 1:
            # the outer-product path has one op structure on both sides
            assert torch.equal(hooked_out, base_out + peft_delta)
        else:
            # rank r: PEFT multiplies by lora_A.t() — a TRANSPOSED VIEW of the
            # exported factor — while the hook multiplies by the contiguous A.
            # Same math, different BLAS layout, so fp32 accumulation order is
            # not guaranteed identical across torch builds (bit-equal on one
            # grading host venv, one-ulp off on another, 2026-09-03); the pin is
            # fp32 rounding, not bitwise. The served-side proof of the same
            # conversion is the HF-vs-native arbitration on real hardware.
            assert torch.allclose(hooked_out, base_out + peft_delta,
                                  rtol=0.0, atol=1e-5), (
                (hooked_out - (base_out + peft_delta)).abs().max().item())


def test_lora_all_export_is_deterministic(tiny_model, tmp_path):
    _, blob = _multilayer_blob(tiny_model, rank=4)
    art = _blob_bytes(blob)
    spec = {"kind": "lora_all"}
    a = lx.export_genuine_adapter(spec, art, tmp_path / "a", **LLAMA_BASE)
    b = lx.export_genuine_adapter(spec, art, tmp_path / "b", **LLAMA_BASE)
    for name in (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_lora_all_names_follow_the_base_layout(tiny_model, tmp_path):
    """The VLM wrappers (GLM-4.6V) put the decoder under language_model;
    NemotronH's attention lives at mixer.o_proj — both are the base's own
    layout, never llama's assumed. A non-default site is recorded."""
    from safetensors.torch import load_file

    _, blob = _multilayer_blob(tiny_model, rank=4)
    art = _blob_bytes(blob)
    glm = lx.export_genuine_adapter(
        {"kind": "lora_all"}, art, tmp_path / "glm",
        base_model="zai-org/GLM-4.6V-Flash", decoder_prefix=WRAPPED_PREFIX)
    names = sorted(load_file(str(glm / lx.ADAPTER_WEIGHTS)))
    assert names[0].startswith(
        "base_model.model.model.language_model.layers.0.self_attn.o_proj.")
    cfg = json.loads((glm / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_decoder_prefix"] == WRAPPED_PREFIX
    assert "petri_oproj_site" not in cfg

    nemo = lx.export_genuine_adapter(
        {"kind": "lora_all"}, art, tmp_path / "nemo",
        base_model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        decoder_prefix="model.layers", oproj_site="mixer.o_proj")
    names = sorted(load_file(str(nemo / lx.ADAPTER_WEIGHTS)))
    assert names[0].startswith(
        "base_model.model.model.layers.0.mixer.o_proj.")
    cfg = json.loads((nemo / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_oproj_site"] == "mixer.o_proj"
    assert cfg["target_modules"] == ["o_proj"]


def test_lora_all_refuses_layers_without_an_attention_oproj(tiny_model):
    """With the layout resolved off the tree (NemotronH: attention at
    [12, 17, 24, 32]), a fit layer outside it names a module the base does
    not have — refuse, never emit a name that points at nothing."""
    _, blob = _multilayer_blob(tiny_model, rank=1)   # layers 0 and 1
    with pytest.raises(lx.LoraExportError, match="carry no attention o_proj"):
        lx._export_multilayer_lora(
            {"kind": "lora_all"}, blob, decoder_prefix="model.layers",
            oproj_site="mixer.o_proj", oproj_layers=(12, 17, 24, 32))


def test_lora_all_refuses_a_response_only_fit(tiny_model, tmp_path):
    """The response-masked lora_all arm serves on the lens o_proj site
    (which carries the mask); native multi-LoRA cannot express it."""
    _, blob = _multilayer_blob(tiny_model, rank=4, positions="response")
    with pytest.raises(lx.LoraExportError, match="lens o_proj"):
        lx.export_genuine_adapter({"kind": "lora_all"}, _blob_bytes(blob),
                                  tmp_path / "x", **LLAMA_BASE)


def test_lora_all_refuses_malformed_blobs(tiny_model, tmp_path):
    _, good = _multilayer_blob(tiny_model, rank=4)
    d_in = good["A"]["0"].shape[0]
    d_out = good["B"]["0"].shape[0]
    cases = [
        (dict(good, rank=2), "declares rank 2"),                 # rank vs tensors
        (dict(good, B={k: v[:, :2] for k, v in good["B"].items()}),
         "rank the config would mis-declare"),                  # B rank mismatch
        (dict(good, layers=[0, 5]), "disagrees with the A/B key set"),
        (dict(good, extra=1), "unexpected keys"),
        (dict(good, A={"0": good["A"]["0"]}), "different layer sets"),
        (dict(good, A={k: v.double() for k, v in good["A"].items()}),
         "float32"),
        (dict(good, A={k: torch.full_like(v, float("nan"))
                       for k, v in good["A"].items()}), "non-finite"),
        (dict(good, B={"0": good["B"]["0"],
                       "1": torch.randn(d_out + 1, 4)}), "corrupt blob"),
        (dict(good, positions=3), "positions must be a string"),
    ]
    for blob, msg in cases:
        with pytest.raises(lx.LoraExportError, match=msg):
            lx.export_genuine_adapter({"kind": "lora_all"}, _blob_bytes(blob),
                                      tmp_path / "bad", **LLAMA_BASE)
    # a missing positions field is a refusal, never read as all-position
    nopos = {k: v for k, v in good.items() if k != "positions"}
    with pytest.raises(lx.LoraExportError, match="lacks \\['positions'\\]"):
        lx.export_genuine_adapter({"kind": "lora_all"}, _blob_bytes(nopos),
                                  tmp_path / "nopos", **LLAMA_BASE)
    # spec cross-checks
    with pytest.raises(lx.LoraExportError, match="F9-style"):
        lx.export_genuine_adapter({"kind": "lora_all", "rank": 1},
                                  _blob_bytes(good), tmp_path / "s1",
                                  **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError, match="F9-style"):
        lx.export_genuine_adapter({"kind": "lora_all", "layers": [0]},
                                  _blob_bytes(good), tmp_path / "s2",
                                  **LLAMA_BASE)
    # a legacy rank-1 blob without `rank`/`layers` keys infers rank 1
    legacy = {"kind": "lora_all", "positions": "all",
              "A": {"0": torch.randn(d_in), "1": torch.randn(d_in)},
              "B": {"0": torch.randn(d_out), "1": torch.randn(d_out)}}
    out = lx.export_genuine_adapter({"kind": "lora_all"}, _blob_bytes(legacy),
                                    tmp_path / "legacy", **LLAMA_BASE)
    assert json.loads((out / lx.ADAPTER_CONFIG).read_text())["r"] == 1


def test_verify_zero_adapter_refuses_a_genuine_lora_all_adapter(
        tiny_model, tmp_path):
    _, blob = _multilayer_blob(tiny_model, rank=4)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_all"}, _blob_bytes(blob), tmp_path / "adapter",
        **LLAMA_BASE)
    with pytest.raises(lx.LoraExportError):
        lx.verify_zero_adapter(adapter)


# --------------------------------------------------------------------------
# oproj_target_modules: the zero adapter's module set, off the meta skeleton
# --------------------------------------------------------------------------


def test_genuine_export_kinds_include_the_subtraction():
    """The exact tuple: the enumeration predicate's servable set is
    pinned equal to it (tests/test_native_lora_constants_agree.py)."""
    assert lx.GENUINE_EXPORT_KINDS == ("circuit_breaker", "lora", "lora_all",
                                       "lora_subtraction",
                                       # native-psr-serving: the residual-slot kinds
                                       "psr", "vector", "vector_all", "diffmean", "static_r",
                                       # relu-differenced-native-qwen: the paired write
                                       "psr_paired_write")


def test_oproj_target_modules_on_the_tiny_llama_fixture(tiny_model, tmp_path):
    """Every attention o_proj of the base, named off its own layout and
    shaped off the same meta skeleton — what export_zero_adapter takes for
    the multi-layer zero adapter. Hermetic: the tiny llama's config is
    saved to a local dir and resolved with local_files_only."""
    from safetensors.torch import load_file

    from sisteer.models import get_decoder_layers

    base = tmp_path / "tiny-llama-config"
    tiny_model.config.save_pretrained(base)
    modules = lx.oproj_target_modules(str(base))
    decoder = get_decoder_layers(tiny_model)
    assert modules == {
        f"model.layers.{l}.self_attn.o_proj":
            (decoder[l].self_attn.o_proj.in_features,
             decoder[l].self_attn.o_proj.out_features)
        for l in range(N_LAYERS)}
    assert modules == {f"model.layers.{l}.self_attn.o_proj": (HIDDEN, HIDDEN)
                       for l in range(N_LAYERS)}
    # the same skeleton build backs resolve_decoder_layout
    layout = lx.resolve_decoder_layout(str(base))
    assert layout == lx.DecoderLayout(prefix=lx.LLAMA_DECODER_PREFIX,
                                      oproj_site=lx.DEFAULT_OPROJ_SITE,
                                      oproj_layers=tuple(range(N_LAYERS)))
    assert set(modules) == {f"{layout.prefix}.{l}.{layout.oproj_site}"
                            for l in layout.oproj_layers}
    cfg, skeleton = lx._decoder_skeleton(str(base))
    assert cfg.model_type == "llama"
    assert next(skeleton.parameters()).device.type == "meta"   # weight-free

    # fed to export_zero_adapter: one zero (A, B) pair per o_proj layer,
    # short names one per module (un-deduplicated, all "o_proj")
    zero = lx.export_zero_adapter(tmp_path / "zero", base_model=str(base),
                                  rank=1, target_modules=modules)
    lx.verify_zero_adapter(zero)
    tensors = load_file(str(zero / lx.ADAPTER_WEIGHTS))
    assert sorted(tensors) == sorted(
        f"base_model.model.model.layers.{l}.self_attn.o_proj.{w}.weight"
        for l in range(N_LAYERS) for w in ("lora_A", "lora_B"))
    assert tuple(tensors[
        "base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight"
    ].shape) == (1, HIDDEN)
    zcfg = json.loads((zero / lx.ADAPTER_CONFIG).read_text())
    assert zcfg["target_modules"] == ["o_proj"] * N_LAYERS
    assert set(zcfg["target_modules"]) == {"o_proj"}

    # an uncached base refuses naming the fix, never a guessed layout
    with pytest.raises(lx.LoraExportError,
                       match="not in the local HF cache") as err:
        lx.oproj_target_modules("TESTORG/not-cached-anywhere")
    assert "decoder_prefix=" in str(err.value)
    with pytest.raises(lx.LoraExportError, match="base_model must be"):
        lx.oproj_target_modules("")


# --------------------------------------------------------------------------
# a MIXED-WIDTH base: each layer's factors are held to ITS module's shape
# (native multi-LoRA — gemma-4-E4B's global-attention o_proj is twice as
# wide on the input side as its local layers'; the export used to demand one
# shape across a blob's layers and refused the real certification object)
# --------------------------------------------------------------------------

#: a tiny gemma-4 TEXT config with two attention widths: ``head_dim`` 8 on the
#: sliding layers and ``global_head_dim`` 16 on the full-attention layer, so
#: o_proj.in_features is 4*8 = 32 on layers 0, 1 and 4*16 = 64 on layer 2 while
#: out_features (hidden_size) is 32 everywhere — the real E4B's 2048/4096 ->
#: 2560 pattern in miniature, instantiated through the REAL derivation path
#: (AutoConfig -> auto_model_class -> meta skeleton), no monkeypatch.
HETERO_LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention"]


@pytest.fixture(scope="module")
def hetero_base(tmp_path_factory):
    from transformers import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=16, layer_types=list(HETERO_LAYER_TYPES),
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


def _hetero_lora_all_blob(shapes, rank=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    layers = sorted(shapes)
    A = {str(l): torch.randn(shapes[l][0], rank, generator=g) for l in layers}
    B = {str(l): torch.randn(shapes[l][1], rank, generator=g) * 0.1
         for l in layers}
    return {"kind": "lora_all", "layers": layers, "rank": rank, "A": A, "B": B,
            "positions": "all", "response_header_len": None}


def test_oproj_target_modules_reads_each_module_s_own_shape_on_a_mixed_width_base(
        hetero_base):
    """Two widths on one base: the per-module map carries both, and the zero
    adapter written from it has a differently shaped lora_A per layer."""
    from safetensors.torch import load_file

    modules = lx.oproj_target_modules(hetero_base)
    assert modules == {"model.layers.0.self_attn.o_proj": (32, 32),
                       "model.layers.1.self_attn.o_proj": (32, 32),
                       "model.layers.2.self_attn.o_proj": (64, 32)}
    layout, shapes = lx._derived_layout(hetero_base)
    assert layout.oproj_layers == (0, 1, 2)
    assert shapes == {0: (32, 32), 1: (32, 32), 2: (64, 32)}
    zero = lx.export_zero_adapter(pathlib.Path(hetero_base) / "zero",
                                  base_model=hetero_base, rank=1,
                                  target_modules=modules)
    lx.verify_zero_adapter(zero)
    t = load_file(str(zero / lx.ADAPTER_WEIGHTS))
    assert tuple(t["base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight"].shape) == (1, 32)
    assert tuple(t["base_model.model.model.layers.2.self_attn.o_proj.lora_A.weight"].shape) == (1, 64)


def test_lora_all_export_admits_a_mixed_width_blob_that_fits_its_modules(
        hetero_base, tmp_path):
    """A blob whose layer 2 factors are wider than layers 0-1 — exactly the
    base's own shapes — exports under the derived layout (the very case the
    one-shape-across-layers rule refused as 'corrupt'); every tensor's shape
    is the module's, and C's launcher-side fingerprint accepts the dir."""
    from safetensors.torch import load_file

    from sisteer.capeval.serving import worker_reports as wr

    _, shapes = lx._derived_layout(hetero_base)
    blob = _hetero_lora_all_blob(shapes)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_all", "rank": 4, "positions": "all", "layers": [0, 1, 2],
         "intervention_hash": "ef" * 8},
        _blob_bytes(blob), tmp_path / "adapter", base_model=hetero_base)
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert cfg["layers_to_transform"] == [0, 1, 2] and cfg["r"] == 4
    t = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    for l, (d_in, d_out) in shapes.items():
        a = t[f"base_model.model.model.layers.{l}.self_attn.o_proj.lora_A.weight"]
        b = t[f"base_model.model.model.layers.{l}.self_attn.o_proj.lora_B.weight"]
        assert tuple(a.shape) == (4, d_in) and tuple(b.shape) == (d_out, 4)
        assert torch.equal(a, blob["A"][str(l)].t())
    fingerprint, keys = wr.expected_adapter_fingerprint_bf16(adapter)
    assert len(fingerprint) == 64 and len(keys) == 3
    # the same blob under a rank-1 legacy fit on the wide layer alone
    wide = {"kind": "lora", "layer": 2, "A": torch.randn(64), "B": torch.randn(32),
            "positions": "all"}
    r1 = lx.export_genuine_adapter({"kind": "lora", "layer": 2, "positions": "all"},
                                   _blob_bytes(wide), tmp_path / "r1",
                                   base_model=hetero_base)
    t1 = load_file(str(r1 / lx.ADAPTER_WEIGHTS))
    assert tuple(t1["base_model.model.model.layers.2.self_attn.o_proj.lora_A.weight"].shape) == (1, 64)


def test_lora_all_export_refuses_a_layer_that_does_not_fit_its_own_module(
        hetero_base, tmp_path):
    """The refusal is PER MODULE and names the layer and the module's shape:
    the wide layer's factors shrunk to the narrow width refuse (they agree
    with their neighbours, which is exactly what the old rule would have
    admitted), and a narrow layer widened to the wide width refuses too —
    never a 'corrupt blob' verdict for merely differing from another layer."""
    _, shapes = lx._derived_layout(hetero_base)
    good = _hetero_lora_all_blob(shapes)
    # layer 2's A shrunk to layer 0/1's width: uniform across layers, wrong for its module
    narrow2 = dict(good, A=dict(good["A"], **{"2": torch.randn(32, 4)}))
    with pytest.raises(lx.LoraExportError,
                       match=r"layer 2 has \(d_in, d_out\)=\(32, 32\) but the base's own "
                             r"o_proj at that layer is \(in_features, out_features\)=\(64, 32\)"
                             r".*\(model\.layers\.2\.self_attn\.o_proj\)"):
        lx.export_genuine_adapter({"kind": "lora_all", "positions": "all"},
                                  _blob_bytes(narrow2), tmp_path / "n2",
                                  base_model=hetero_base)
    # layer 0's A widened to layer 2's width
    wide0 = dict(good, A=dict(good["A"], **{"0": torch.randn(64, 4)}))
    with pytest.raises(lx.LoraExportError, match=r"layer 0 has \(d_in, d_out\)=\(64, 32\)"):
        lx.export_genuine_adapter({"kind": "lora_all", "positions": "all"},
                                  _blob_bytes(wide0), tmp_path / "w0",
                                  base_model=hetero_base)
    # a rank-1 fit on the wide layer with the narrow width refuses the same
    # way, naming the module and both sizes (the per-module check applies to
    # every kind on the derived path; see the dedicated test below for why)
    bad_r1 = {"kind": "lora", "layer": 2, "A": torch.randn(32), "B": torch.randn(32),
              "positions": "all"}
    with pytest.raises(lx.LoraExportError,
                       match=r"kind 'lora': layer 2 has \(d_in, d_out\)=\(32, 32\) but the "
                             r"base's own o_proj at that layer is \(in_features, "
                             r"out_features\)=\(64, 32\).*\(model\.layers\.2\.self_attn\.o_proj\)"):
        lx.export_genuine_adapter({"kind": "lora", "positions": "all"}, _blob_bytes(bad_r1),
                                  tmp_path / "bad_r1", base_model=hetero_base)
    # the no-tree path keeps the historical heuristic (there is no module to
    # hold a layer to), and says so
    with pytest.raises(lx.LoraExportError, match="mixed-width base"):
        lx.export_genuine_adapter({"kind": "lora_all", "positions": "all"},
                                  _blob_bytes(good), tmp_path / "stated",
                                  base_model=hetero_base,
                                  decoder_prefix=lx.LLAMA_DECODER_PREFIX)


def test_rank1_export_refuses_factors_that_do_not_fit_the_module_they_name(
        hetero_base, tmp_path):
    """A rank-1 artifact whose factors do not fit the named projection is
    refused BY THE EXPORT, naming the module and both sizes — the check the
    branch dropped once on the claim that vLLM refuses such a fit at
    load. It does not: vLLM 0.25.1's BaseLinearLayerWithLoRA.set_lora zeroes
    its full-size lora_a/lora_b stacks and copies each factor into a slice
    sized from the INCOMING tensor, so an A of length 32 on a module that
    takes 64 inputs is zero-padded without an error and serves a padded
    intervention (the reference hook's ``x @ A`` would have failed its
    matmul), while the attestation fingerprints the CPU-side factors as
    exported and validates module names, not dimensions — so matching
    hashes would certify it. The wide layer of the mixed-width base is the
    real E4B pattern in miniature: o_proj takes 64 inputs there, 32 on the
    narrow layers, all projecting to 32."""
    from safetensors.torch import load_file

    _, shapes = lx._derived_layout(hetero_base)
    assert shapes[2] == (64, 32) and shapes[0] == (32, 32)
    module = r"\(model\.layers\.2\.self_attn\.o_proj\)"
    spec = {"kind": "lora", "layer": 2, "positions": "all"}

    def blob(d_in, d_out):
        g = torch.Generator().manual_seed(11)
        return _blob_bytes({"kind": "lora", "layer": 2, "positions": "all",
                            "A": torch.randn(d_in, generator=g),
                            "B": torch.randn(d_out, generator=g)})

    # A undersized for the module (the narrow layers' width): refused by name
    with pytest.raises(lx.LoraExportError,
                       match=r"kind 'lora': layer 2 has \(d_in, d_out\)=\(32, 32\) but the "
                             r"base's own o_proj at that layer is \(in_features, "
                             r"out_features\)=\(64, 32\) — the factors do not fit the "
                             r"module they name " + module) as ei:
        lx.export_genuine_adapter(spec, blob(32, 32), tmp_path / "short_a",
                                  base_model=hetero_base)
    # the refusal says WHY the export is the place that catches it
    assert "set_lora zero-pads an undersized factor silently" in str(ei.value)
    assert not (tmp_path / "short_a").exists()
    # B off by one on the output side refuses the same way
    with pytest.raises(lx.LoraExportError,
                       match=r"layer 2 has \(d_in, d_out\)=\(64, 31\).*=\(64, 32\).*" + module):
        lx.export_genuine_adapter(spec, blob(64, 31), tmp_path / "short_b",
                                  base_model=hetero_base)
    # an A LONGER than the module takes refuses too (the check is equality,
    # not a bound — an oversized factor is just as much another module's fit)
    with pytest.raises(lx.LoraExportError,
                       match=r"layer 2 has \(d_in, d_out\)=\(128, 32\).*=\(64, 32\).*" + module):
        lx.export_genuine_adapter(spec, blob(128, 32), tmp_path / "long_a",
                                  base_model=hetero_base)
    # the same layer with the module's OWN sizes exports, at those sizes
    adapter = lx.export_genuine_adapter(spec, blob(64, 32), tmp_path / "fits",
                                        base_model=hetero_base)
    t = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    assert tuple(t["base_model.model.model.layers.2.self_attn.o_proj.lora_A.weight"].shape) == (1, 64)
    assert tuple(t["base_model.model.model.layers.2.self_attn.o_proj.lora_B.weight"].shape) == (32, 1)
    # and the narrow width is right for a narrow layer — the yardstick is the
    # named module, never a neighbour or a base-wide constant
    g = torch.Generator().manual_seed(12)
    narrow = _blob_bytes({"kind": "lora", "layer": 0, "positions": "all",
                          "A": torch.randn(32, generator=g), "B": torch.randn(32, generator=g)})
    lx.export_genuine_adapter({"kind": "lora", "layer": 0, "positions": "all"}, narrow,
                              tmp_path / "narrow", base_model=hetero_base)
    # WITHOUT the tree (a caller-stated decoder_prefix) there is no module to
    # hold one layer to — the boundary of the check, which is why every
    # production caller (export_or_reuse) derives the layout from the tree
    lx.export_genuine_adapter(spec, blob(32, 32), tmp_path / "stated_r1",
                              base_model=hetero_base,
                              decoder_prefix=lx.LLAMA_DECODER_PREFIX)
