"""native-psr-serving step 2 — the RESIDUAL-SLOT EXPORTER (``lora_export.
_export_residual_slot`` behind ``export_or_reuse`` / ``export_genuine_adapter``).

Pinned here, on the tiny llama saved as a local snapshot (hermetic, as
``test_adapter_bank`` does it):

- ROUND TRIPS: a ``psr`` blob (``PSRObject.save``'s shape), a ``vector_all``
  blob, a ``diffmean`` blob (int-keyed vectors) and a ``vector`` SteeringVector
  container + sidecar each bank as a rank-1 ZERO o_proj pair on the ANCHOR
  (``resolve_decoder_layout().oproj_layers[0]``) plus the payload under
  ``sisteer_residual.layers.<l>.*`` and a ``petri_residual_slot`` block; the
  worker's reader (``native_residual.read_residual_payload``) gives back the
  blob's tensors BYTE-exact; ``export_or_reuse`` reuses the entry byte-
  identically; the LoRA fingerprint / kernel sizing readers accept the entry
  (the payload keys skipped), the digests mark it genuine with its kind;
- the ANCHOR on a hybrid layout: Qwen3.5-4B's first o_proj block is 3, so a
  payload on the linear-attention block 0 rides a carrier on block 3;
- REFUSALS with the lens's rules: a stage-2 mode (response-fit psr, static_r's
  prefill_all / prompt_last) refuses at export while stage 2 is unarmed;
  static_r ``site='pre'`` at block 0 refuses FIRST; a masked vector_all
  fit deployed at every position; a PSR gate_b of the wrong numel, a non-fp32
  vector, a layer missing from ``directions``, a spec/blob layer disagreement,
  a sidecar on a blob kind, a blob of the wrong kind; the stated-prefix path;
  the vector kind's antidote / projection_targets / norm_relative / missing
  sidecar; a mis-keyed static_r norm; a non-discriminating norm record;
- with stage 2 ARMED (monkeypatched) the static_r ``pre`` shift lands one
  boundary earlier with ``coeff = base_scale``.

Every known-bad input is exercised. CPU-only."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import worker_reports as wr

HIDDEN = 32
N_LAYERS = 2
IH = "ab" * 8


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=97,
        max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


#: the chat template the tiny base is saved with — the GEMMA-4 SHAPE (every turn trails
#: scaffold after its terminator, so the terminator-anchored located header is one token
#: WIDER than the generation-prompt header the mask must use). Stage 2's response exports
#: derive their header ids from the served tokenizer, so the fixture carries one.
_GEMMA_SHAPE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}"
    "{{ message['content'] }}{{ '<|eot_id|>' }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>' }}{% endif %}")


@pytest.fixture(scope="module")
def tiny_base(tiny_model, tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    d = tmp_path_factory.mktemp("tiny-llama-snapshot")
    tiny_model.save_pretrained(d)
    # the SERVED tokenizer beside the weights: a response-mode residual payload's boundary
    # is located by the ids `add_generation_prompt` appends, which the exporter reads off
    # THIS base's own tokenizer (chat_markers.generation_header_ids)
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM", local_files_only=True)
    except Exception as e:  # noqa: BLE001 — not cached here: skip, never fetch
        pytest.skip(f"tiny llama tokenizer not in the local HF cache ({type(e).__name__}: {e})")
    tok.add_special_tokens({"additional_special_tokens": [
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]})
    tok.chat_template = _GEMMA_SHAPE_TEMPLATE
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def served_header_ids(tiny_base):
    """The ids the exporter will derive for ``tiny_base`` — computed HERE through the same
    one function (chat_markers), so the test states the expectation rather than the width."""
    from sisteer.chat_markers import generation_header_ids
    from sisteer.models import pin_reasoning_template_off
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_base, local_files_only=True)
    pin_reasoning_template_off(tok)
    return tuple(int(t) for t in generation_header_ids(tok))


def _bytes(d) -> bytes:
    buf = io.BytesIO()
    torch.save(d, buf)
    return buf.getvalue()


def _psr_blob(layers=(0, 1), seed=5, positions="all", header=None, d=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    return {"kind": "psr", "layers": list(layers), "train_direction": False,
            "positions": positions, "response_header_len": header,
            "directions": {str(l): torch.randn(d, generator=g) for l in layers},
            "gate_w": {str(l): torch.randn(d, generator=g) for l in layers},
            "gate_b": {str(l): torch.randn(1, generator=g) for l in layers}}


def _vector_all_blob(layers=(0, 1), seed=6, positions="all", fit="all", skip_bos=True, d=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    return {"kind": "vector_all",
            "vectors": {str(l): torch.randn(d, generator=g) for l in layers},
            "positions": positions, "skip_bos": skip_bos,
            "fit_positions": fit, "fit_response_header_len": 3 if fit == "response" else None}


def _diffmean_blob(layers=(0, 1), seed=7, positions="all", d=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    return {"kind": "diffmean", "vectors": {l: torch.randn(d, generator=g) for l in layers},
            "skip_bos": True, "positions": positions,
            "response_header_len": 3 if positions == "response" else None,
            "metadata": {"fit": "closed-form"}}


def _static_r_blob(layers=(1,), seed=8, steer="prefill_all", site="pre", base_scale=0.05,
                   norms=True, d=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    rs = {l: torch.randn(d, generator=g) * (1.0 + 0.5 * i) for i, l in enumerate(layers)}
    blob = {"kind": "static_r", "rs": rs, "base_scale": base_scale, "steer_positions": steer,
            "site": site, "layers": list(layers), "metadata": {}}
    if norms:
        blob["metadata"]["per_layer_r_norm"] = {str(l): float(v.norm()) for l, v in rs.items()}
    return blob


def _vector_container(tmp_path: Path, layers=(0, 1), seed=9, method="caa", positions=None,
                      skip_bos=None, norm_relative=None, projection_targets=None, d=HIDDEN):
    from sisteer.vectors import SteeringVector

    g = torch.Generator().manual_seed(seed)
    meta = {}
    if positions is not None or skip_bos is not None or norm_relative is not None:
        meta["vector_object"] = {"positions": positions or "all",
                                 "skip_bos": True if skip_bos is None else skip_bos,
                                 "norm_relative": bool(norm_relative)}
    if projection_targets:
        meta["projection_targets"] = projection_targets
    sv = SteeringVector(vectors={l: torch.randn(d, generator=g) for l in layers}, method=method,
                        model_name="tiny", instruction_id="t", metadata=meta)
    p = tmp_path / "vec.safetensors"
    sv.save(p)
    return p.read_bytes(), p.with_suffix(".json").read_bytes()


def _spec(kind, ih=IH, **over):
    return {"kind": kind, "intervention_hash": ih, **over}


def _keys(adir: Path):
    from safetensors.torch import load_file
    return load_file(str(adir / lx.ADAPTER_WEIGHTS))


# ---------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------


def test_psr_round_trip_banks_a_zero_carrier_on_the_anchor_plus_the_payload(tiny_base, tmp_path):
    blob = _psr_blob()
    root = tmp_path / "bank"
    adir, dig, reused = lx.export_or_reuse(_spec("psr", layers=[0, 1], positions="all"), _bytes(blob),
                                            root, base_model=tiny_base, name_hint="si_x__psr_ce_all")
    assert not reused and adir == lx.adapter_dir_for(root, IH)
    cfg = json.loads((adir / lx.ADAPTER_CONFIG).read_text())
    assert cfg["r"] == 1 and cfg["lora_alpha"] == 1.0 and cfg["target_modules"] == ["o_proj"]
    assert cfg["layers_to_transform"] == [0] and cfg["petri_source_kind"] == "psr"
    assert cfg["petri_source_layers"] == [0, 1] and cfg["petri_fit_positions"] == "all"
    assert cfg["petri_genuine_adapter"] is True and cfg["petri_intervention_hash"] == IH
    block = cfg["petri_residual_slot"]
    assert block == {"schema": 1, "kind": "psr", "op": "relu_gated_vector", "layers": [0, 1],
                     "boundaries": [0, 1], "position_mode": "all", "response_header_len": None,
                     "response_header_ids": [], "skip_bos": False, "coeff": 1.0,
                     "hidden_size": HIDDEN, "n_layers_model": N_LAYERS, "anchor_layer": 0,
                     "dtype_policy": nr.DTYPE_POLICY_GATED}
    tensors = _keys(adir)
    carrier = {k for k in tensors if ".lora_" in k}
    assert carrier == {"base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
                       "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"}
    a = tensors["base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"]
    assert tuple(a.shape) == (1, HIDDEN) and tuple(b.shape) == (HIDDEN, 1)
    assert not a.any() and not b.any() and a.dtype == b.dtype == torch.float32
    assert set(tensors) - carrier == {f"sisteer_residual.layers.{l}.{f}" for l in (0, 1)
                                      for f in ("gate_w", "gate_b", "direction")}
    # the worker's reader gives the blob's tensors back BYTE-exact
    payload = nr.read_residual_payload(adir)
    assert payload.kind == "psr" and payload.layers == (0, 1) and payload.gated
    for l in (0, 1):
        for key, group in (("gate_w", payload.gate_w), ("directions", payload.direction)):
            assert torch.equal(group[l], blob[key][str(l)]) and group[l].dtype == torch.float32
        assert torch.equal(payload.gate_b[l], blob["gate_b"][str(l)].reshape(1))
    # the digests: a GENUINE entry of kind psr; the LoRA readers accept the entry
    assert dig.petri_zero_adapter is False and dig.petri_source_kind == "psr" and dig.r == 1
    fp, keys = wr.expected_adapter_fingerprint_bf16(adir)
    assert keys == ("L0.self_attn.o_proj",) and len(fp) == 64     # the canonical module key
    assert sa.adapter_module_dims(adir) == {"base_model.model.model.layers.0.self_attn.o_proj":
                                            (HIDDEN, HIDDEN)}
    # reuse is byte-identical
    before = {p.name: p.read_bytes() for p in adir.iterdir()}
    adir2, dig2, reused2 = lx.export_or_reuse(_spec("psr", layers=[0, 1], positions="all"),
                                               _bytes(blob), root, base_model=tiny_base,
                                               name_hint="si_x__psr_ce_all")
    assert reused2 and adir2 == adir and dig2 == dig
    assert {p.name: p.read_bytes() for p in adir.iterdir()} == before
    # a tampered payload byte refuses the reuse (weights_sha256 binds the payload)
    other = _psr_blob(seed=99)
    with pytest.raises(lx.LoraExportError, match="NOT byte-identical"):
        lx.export_or_reuse(_spec("psr", layers=[0, 1], positions="all"), _bytes(other), root,
                           base_model=tiny_base)
    # the residual entry is never the zero adapter (its config marks it genuine)
    with pytest.raises(lx.LoraExportError, match="not marked petri_zero_adapter"):
        lx.verify_zero_adapter(adir)


@pytest.mark.parametrize("kind", ["vector_all", "diffmean"])
def test_constant_kind_round_trips(tiny_base, tmp_path, kind):
    blob = _vector_all_blob() if kind == "vector_all" else _diffmean_blob()
    spec = _spec(kind, skip_bos=True, positions="all")
    adir, dig, _ = lx.export_or_reuse(spec, _bytes(blob), tmp_path / "bank", base_model=tiny_base)
    cfg = json.loads((adir / lx.ADAPTER_CONFIG).read_text())
    block = cfg["petri_residual_slot"]
    assert block["kind"] == kind and block["op"] == "add_vector" and block["position_mode"] == "all"
    assert block["skip_bos"] is True and block["coeff"] == 1.0 and block["layers"] == [0, 1]
    assert block["dtype_policy"] == nr.DTYPE_POLICY_CONSTANT and cfg["petri_source_kind"] == kind
    payload = nr.read_residual_payload(adir)
    assert not payload.gated
    for l in (0, 1):
        want = blob["vectors"][str(l) if kind == "vector_all" else l]
        assert torch.equal(payload.vector[l], want) and payload.vector[l].dtype == torch.float32
    assert dig.petri_source_kind == kind
    wr.expected_adapter_fingerprint_bf16(adir)


def test_vector_container_round_trip_and_its_refusals(tiny_base, tmp_path):
    art, side = _vector_container(tmp_path / "c1")
    adir, dig, _ = lx.export_or_reuse(_spec("vector"), art, tmp_path / "bank", base_model=tiny_base,
                                       sidecar_bytes=side)
    cfg = json.loads((adir / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_source_kind"] == "vector" and cfg["petri_residual_slot"]["skip_bos"] is True
    assert cfg["petri_residual_slot"]["position_mode"] == "all"
    payload = nr.read_residual_payload(adir)
    from safetensors.torch import load as st_load
    raw = st_load(art)
    for l in (0, 1):
        assert torch.equal(payload.vector[l], raw[f"layer_{l}"])
    # the sidecar is REQUIRED
    with pytest.raises(lx.LoraExportError, match="sidecar is REQUIRED"):
        lx.export_or_reuse(_spec("vector"), art, tmp_path / "b2", base_model=tiny_base)
    # antidote hydrates as another object
    art2, side2 = _vector_container(tmp_path / "c2", method="antidote")
    with pytest.raises(lx.LoraExportError, match="antidote"):
        lx.export_or_reuse(_spec("vector"), art2, tmp_path / "b3", base_model=tiny_base,
                           sidecar_bytes=side2)
    # the projection edit is not an additive op
    art3, side3 = _vector_container(tmp_path / "c3", projection_targets={"0": 1.0})
    with pytest.raises(lx.LoraExportError, match="projection_targets"):
        lx.export_or_reuse(_spec("vector"), art3, tmp_path / "b4", base_model=tiny_base,
                           sidecar_bytes=side3)
    # norm_relative, stated or persisted
    with pytest.raises(lx.LoraExportError, match="norm_relative"):
        lx.export_or_reuse(_spec("vector", norm_relative=True), art, tmp_path / "b5",
                           base_model=tiny_base, sidecar_bytes=side)
    art4, side4 = _vector_container(tmp_path / "c4", norm_relative=True)
    with pytest.raises(lx.LoraExportError, match="norm_relative"):
        lx.export_or_reuse(_spec("vector"), art4, tmp_path / "b6", base_model=tiny_base,
                           sidecar_bytes=side4)
    # the two position records must agree; a response deployment carries a header width (the
    # SteeringVector container records none, so the response deployment refuses for that —
    # stage 2 serves the mode, it does not invent the fit's boundary)
    art5, side5 = _vector_container(tmp_path / "c5", positions="response")
    with pytest.raises(lx.LoraExportError, match="refusing to pick one"):
        lx.export_or_reuse(_spec("vector", positions="all"), art5, tmp_path / "b7",
                           base_model=tiny_base, sidecar_bytes=side5)
    with pytest.raises(lx.LoraExportError, match="carries no response_header_len"):
        lx.export_or_reuse(_spec("vector"), art5, tmp_path / "b8", base_model=tiny_base,
                           sidecar_bytes=side5)


@pytest.mark.skipif(not (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B").exists(),
                    reason="the Qwen3.5-4B config is not in the local HF cache")
def test_the_anchor_on_a_hybrid_layout_is_the_first_oproj_block(tmp_path):
    """Qwen3.5-4B: full attention every 4 blocks from 3; a payload on the
    linear-attention block 0 rides a carrier on block 3 (the plan's 'vector on
    a linear-attention block' case is a POSITIVE case for the residual slot —
    the write is at the block output, not at an o_proj)."""
    from lora_snapshot_fixtures import snapshot_like

    base = "Qwen/Qwen3.5-4B"
    try:
        layout = lx.resolve_decoder_layout(base)
    except lx.LoraExportError as e:
        pytest.skip(f"the Qwen3.5 skeleton cannot be built here: {e}")
    assert layout.oproj_layers[0] == 3
    hidden, n_layers = lx._derived_dims(base)
    snap = snapshot_like(base, tmp_path / "snap", layout=layout)
    blob = _vector_all_blob(layers=(0, 1, 5), d=hidden)
    out = lx.export_genuine_adapter(_spec("vector_all", skip_bos=True), _bytes(blob), tmp_path / "out",
                                    base_model=base, snapshot_dir=snap)
    cfg = json.loads((out / lx.ADAPTER_CONFIG).read_text())
    assert cfg["layers_to_transform"] == [3] and cfg["petri_residual_slot"]["anchor_layer"] == 3
    assert cfg["petri_residual_slot"]["layers"] == [0, 1, 5]
    assert cfg["petri_residual_slot"]["n_layers_model"] == n_layers == 32
    keys = _keys(out)
    carrier = sorted(k for k in keys if ".lora_" in k)
    assert carrier == ["base_model.model.model.language_model.layers.3.self_attn.o_proj.lora_A.weight",
                       "base_model.model.model.language_model.layers.3.self_attn.o_proj.lora_B.weight"]
    assert f"sisteer_residual.layers.0.vector" in keys


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def _export(tiny_base, tmp_path, spec, blob_bytes, sidecar=b"", tag="x"):
    return lx.export_or_reuse(spec, blob_bytes, tmp_path / f"bank-{tag}", base_model=tiny_base,
                              sidecar_bytes=sidecar)


def test_stage_2_modes_refuse_at_export_while_stage_2_is_unarmed(tiny_base, tmp_path, monkeypatch):
    """The BACKSTOP, pinned in the stage-1 world: the disposition skips such a cell
    first, and the exporter refuses it anyway, so a stage-1 bank entry cannot carry a
    restricted mode. The flag is monkeypatched False because the tree now arms it (step 7);
    the armed behaviour is the two tests below."""
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", False)
    with pytest.raises(lx.LoraExportError, match="'response' is stage-2"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(_psr_blob(positions="response", header=3)), tag="a")
    with pytest.raises(lx.LoraExportError, match="'prefill_all' is stage-2"):
        _export(tiny_base, tmp_path, _spec("static_r"), _bytes(_static_r_blob()), tag="b")
    with pytest.raises(lx.LoraExportError, match="'prompt_last' is stage-2"):
        _export(tiny_base, tmp_path, _spec("static_r"),
                _bytes(_static_r_blob(steer="prompt_last", site="post")), tag="c")
    with pytest.raises(lx.LoraExportError, match="'response' is stage-2"):
        _export(tiny_base, tmp_path, _spec("vector_all"),
                _bytes(_vector_all_blob(positions="response", fit="response")), tag="d")
    with pytest.raises(lx.LoraExportError, match="'response' is stage-2"):
        _export(tiny_base, tmp_path, _spec("diffmean"), _bytes(_diffmean_blob(positions="response")), tag="e")
    # a mode outside the native vocabulary refuses as such
    with pytest.raises(lx.LoraExportError, match="not a native residual-slot mode"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(_psr_blob(positions="last")), tag="f")


def test_static_r_block_0_pre_refuses_first_and_the_armed_shift_lands_one_boundary_earlier(
        tiny_base, tmp_path, monkeypatch, served_header_ids):
    with pytest.raises(lx.LoraExportError, match="resid_pre of layer 0"):        # before the stage check
        _export(tiny_base, tmp_path, _spec("static_r"), _bytes(_static_r_blob(layers=(0, 1))), tag="a")
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", True)
    adir, _, _ = _export(tiny_base, tmp_path, _spec("static_r"), _bytes(_static_r_blob(layers=(1,))), tag="b")
    block = json.loads((adir / lx.ADAPTER_CONFIG).read_text())["petri_residual_slot"]
    assert block["layers"] == [1] and block["boundaries"] == [0]                   # one block earlier
    assert block["position_mode"] == "prefill_all" and block["coeff"] == 0.05
    assert block["skip_bos"] is False and block["response_header_len"] is None
    assert block["response_header_ids"] == []      # no header outside the response mode
    payload = nr.read_residual_payload(adir)
    assert payload.boundaries == (0,) and payload.coeff == 0.05
    # post site: no shift
    adir2, _, _ = _export(tiny_base, tmp_path, _spec("static_r"),
                          _bytes(_static_r_blob(layers=(1,), site="post", steer="prompt_last")), tag="c")
    assert json.loads((adir2 / lx.ADAPTER_CONFIG).read_text())["petri_residual_slot"]["boundaries"] == [1]
    # THE CROSS-PHASE CELL (spec positions == response): an all-position fit re-masked by the
    # CELL, so there is no fit width to honour — stage 2 DERIVES both the ids and the width
    # from the served template (the one exception to the "a response record needs its width"
    # rule below)
    adir3, _, _ = _export(tiny_base, tmp_path, _spec("static_r", positions="response"),
                          _bytes(_static_r_blob(layers=(1,))), tag="d0")
    blk0 = json.loads((adir3 / lx.ADAPTER_CONFIG).read_text())["petri_residual_slot"]
    assert blk0["position_mode"] == "response"
    assert tuple(blk0["response_header_ids"]) == served_header_ids
    assert blk0["response_header_len"] == len(served_header_ids)
    assert nr.read_residual_payload(adir3).response_header_ids == served_header_ids
    # a STATED width must agree with the served header exactly (like with like)
    adir4, _, _ = _export(tiny_base, tmp_path,
                          _spec("static_r", positions="response",
                                response_header_len=len(served_header_ids)),
                          _bytes(_static_r_blob(layers=(1,))), tag="d")
    blk3 = json.loads((adir4 / lx.ADAPTER_CONFIG).read_text())["petri_residual_slot"]
    assert blk3["position_mode"] == "response" and blk3["response_header_len"] == len(served_header_ids)
    with pytest.raises(lx.LoraExportError, match="the fit recorded a 99-token assistant header"):
        _export(tiny_base, tmp_path, _spec("static_r", positions="response", response_header_len=99),
                _bytes(_static_r_blob(layers=(1,))), tag="d2")
    # a stated positions other than response is ambiguous
    with pytest.raises(lx.LoraExportError, match="ambiguous"):
        _export(tiny_base, tmp_path, _spec("static_r", positions="all"), _bytes(_static_r_blob(layers=(1,))), tag="e")
    # the recorded-norm cross-check: a mis-keyed norm refuses; a vacuous record refuses
    blob = _static_r_blob(layers=(1,))
    blob["metadata"]["per_layer_r_norm"] = {"1": 123.456}
    with pytest.raises(lx.LoraExportError, match="records 123.456"):
        _export(tiny_base, tmp_path, _spec("static_r"), _bytes(blob), tag="f")
    blob = _static_r_blob(layers=(0, 1), site="post", steer="prompt_last")
    blob["metadata"]["per_layer_r_norm"] = {"0": 1.0, "1": 1.0}
    with pytest.raises(lx.LoraExportError, match="do not discriminate"):
        _export(tiny_base, tmp_path, _spec("static_r"), _bytes(blob), tag="g")
    blob = _static_r_blob(layers=(1,))
    blob["layers"] = [1, 2]
    with pytest.raises(lx.LoraExportError, match="inconsistent artifact"):
        _export(tiny_base, tmp_path, _spec("static_r"), _bytes(blob), tag="h")
    with pytest.raises(lx.LoraExportError, match="unknown steer_positions"):
        _export(tiny_base, tmp_path, _spec("static_r"), _bytes(_static_r_blob(layers=(1,), steer="prefill")), tag="i")
    with pytest.raises(lx.LoraExportError, match="base_scale"):
        blob = _static_r_blob(layers=(1,))
        del blob["base_scale"]
        _export(tiny_base, tmp_path, _spec("static_r"), _bytes(blob), tag="j")


def test_armed_response_exports_carry_the_served_header_ids(tiny_base, tmp_path, monkeypatch,
                                                            served_header_ids):
    """Stage 2's response payloads: the block carries the SERVED generation-prompt
    header's exact token ids beside the fit's recorded width, because the serve-time boundary
    is LOCATED by matching those ids — never counted off a width. The fit's width and the
    served header must agree exactly (the lens's own ``_fit_leg_positions`` check)."""
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", True)
    h = len(served_header_ids)
    for kind, blob, tag in (
            ("psr", _psr_blob(positions="response", header=h), "r-psr"),
            ("diffmean", {**_diffmean_blob(positions="response"), "response_header_len": h}, "r-dm"),
            ("vector_all", {**_vector_all_blob(positions="response", fit="response"),
                            "fit_response_header_len": h}, "r-va")):
        adir, _, _ = _export(tiny_base, tmp_path, _spec(kind), _bytes(blob), tag=tag)
        block = json.loads((adir / lx.ADAPTER_CONFIG).read_text())["petri_residual_slot"]
        assert block["position_mode"] == "response", kind
        assert tuple(block["response_header_ids"]) == served_header_ids, kind
        assert block["response_header_len"] == h, kind
        # the worker's own reader gives the ids back
        assert nr.read_residual_payload(adir).response_header_ids == served_header_ids, kind
    # KNOWN-BAD: a response fit with NO recorded width refuses (only the cross-phase static_r
    # cell, which has no fit width at all, takes the served header's own)
    with pytest.raises(lx.LoraExportError, match="carries no response_header_len"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(_psr_blob(positions="response")), tag="r-nh")
    # KNOWN-BAD: a fit width that DISAGREES with the served header — template drift; the two
    # masks would cover different regions
    with pytest.raises(lx.LoraExportError, match=f"but the served generation-prompt header is {h}"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(_psr_blob(positions="response", header=h + 1)),
                tag="r-drift")
    # KNOWN-BAD: a base whose template exposes NO locatable assistant header — no ids can be
    # derived, so the payload cannot be located at all and the export refuses by name
    from sisteer.chat_markers import ChatMarkerError
    monkeypatch.setattr(lx, "_served_generation_header_ids",
                        lambda *a, **k: (_ for _ in ()).throw(
                            lx.LoraExportError("the served chat template exposes no derivable "
                                               "generation-prompt assistant header")))
    with pytest.raises(lx.LoraExportError, match="no derivable"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(_psr_blob(positions="response", header=h)),
                tag="r-noheader")
    assert ChatMarkerError is not None   # the real derivation's own error type, named above


def test_the_served_header_ids_come_from_the_bases_own_tokenizer(tiny_base, served_header_ids):
    """The exporter's derivation IS ``chat_markers.generation_header_ids`` over the base's
    pinned tokenizer — not the terminator-anchored located header, which on this
    gemma-shaped template absorbs a prompt-side token and would steer outside the fit
    support. A base whose tokenizer is not cached refuses by name."""
    from sisteer.chat_markers import derive_assistant_markers
    from transformers import AutoTokenizer

    assert lx._served_generation_header_ids(tiny_base, None) == served_header_ids
    tok = AutoTokenizer.from_pretrained(tiny_base, local_files_only=True)
    located = tuple(int(t) for t in derive_assistant_markers(tok).header)
    assert located != served_header_ids and located[-len(served_header_ids):] == served_header_ids
    with pytest.raises(lx.LoraExportError, match="cannot load"):
        lx._served_generation_header_ids("no-such-org/no-such-model", None)


def test_psr_blob_refusals(tiny_base, tmp_path):
    good = _psr_blob()
    # the four keys
    for key in ("layers", "directions", "gate_w", "gate_b"):
        blob = dict(good)
        del blob[key]
        with pytest.raises(lx.LoraExportError, match=f"lacks '{key}'"):
            _export(tiny_base, tmp_path, _spec("psr"), _bytes(blob), tag=f"k-{key}")
    # a layer missing from directions
    blob = _psr_blob()
    del blob["directions"]["1"]
    with pytest.raises(lx.LoraExportError, match="lacks layer 1"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(blob), tag="a")
    # gate_b of the wrong numel
    blob = _psr_blob()
    blob["gate_b"]["0"] = torch.zeros(2)
    with pytest.raises(lx.LoraExportError, match="single-element"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(blob), tag="b")
    # a non-fp32 direction (the save path writes fp32)
    blob = _psr_blob()
    blob["directions"]["0"] = blob["directions"]["0"].to(torch.bfloat16)
    with pytest.raises(lx.LoraExportError, match="float32"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(blob), tag="c")
    # a direction of the wrong width
    blob = _psr_blob()
    blob["directions"]["0"] = torch.zeros(HIDDEN + 1)
    with pytest.raises(lx.LoraExportError, match="dim"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(blob), tag="d")
    # the spec's layers disagree with the blob's
    with pytest.raises(lx.LoraExportError, match="F9-style"):
        _export(tiny_base, tmp_path, _spec("psr", layers=[0, 2]), _bytes(good), tag="e")
    # a sidecar on a blob kind
    with pytest.raises(lx.LoraExportError, match="takes no sidecar"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(good), sidecar=b"{}", tag="f")
    # the wrong kind's bytes
    with pytest.raises(lx.LoraExportError, match="stores kind 'vector_all'"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(_vector_all_blob()), tag="g")
    # a layer outside the base
    blob = _psr_blob(layers=(0, 5))
    with pytest.raises(lx.LoraExportError, match="outside the base's 2 decoder blocks"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(blob), tag="h")
    # an H <= 0 header on a response fit
    with pytest.raises(lx.LoraExportError, match="positive int"):
        _export(tiny_base, tmp_path, _spec("psr"), _bytes(_psr_blob(positions="response", header=0)), tag="i")
    # extra blob keys are tolerated (train_direction is one; another unknown one too)
    blob = _psr_blob()
    blob["fit_note"] = "extra provenance"
    adir, _, _ = _export(tiny_base, tmp_path, _spec("psr"), _bytes(blob), tag="j")
    assert nr.read_residual_payload(adir).layers == (0, 1)


def test_vector_all_and_diffmean_refusals(tiny_base, tmp_path):
    with pytest.raises(lx.LoraExportError, match="fit=response / serve=all"):
        _export(tiny_base, tmp_path, _spec("vector_all"),
                _bytes(_vector_all_blob(positions="all", fit="response")), tag="a")
    blob = _vector_all_blob()
    blob["skip_bos"] = "yes"
    with pytest.raises(lx.LoraExportError, match="skip_bos must be a bool"):
        _export(tiny_base, tmp_path, _spec("vector_all"), _bytes(blob), tag="b")
    with pytest.raises(lx.LoraExportError, match="F9-style"):
        _export(tiny_base, tmp_path, _spec("vector_all", skip_bos=False), _bytes(_vector_all_blob()), tag="c")
    blob = _vector_all_blob()
    blob["vectors"] = {}
    with pytest.raises(lx.LoraExportError, match="non-empty 'vectors'"):
        _export(tiny_base, tmp_path, _spec("vector_all"), _bytes(blob), tag="d")
    blob = _diffmean_blob()
    blob["vectors"][0] = blob["vectors"][0].double()
    with pytest.raises(lx.LoraExportError, match="float32"):
        _export(tiny_base, tmp_path, _spec("diffmean"), _bytes(blob), tag="e")
    with pytest.raises(lx.LoraExportError, match="F9-style"):
        _export(tiny_base, tmp_path, _spec("diffmean", positions="response"), _bytes(_diffmean_blob()), tag="f")


def test_residual_kinds_export_on_the_derived_path_only(tiny_base, tmp_path):
    with pytest.raises(lx.LoraExportError, match="DERIVED path only"):
        lx.export_genuine_adapter(_spec("psr"), _bytes(_psr_blob()), tmp_path / "out",
                                  base_model=tiny_base, decoder_prefix="model.layers")
    # the exporter's kind table and the worker's are one set
    assert set(lx.RESIDUAL_EXPORT_KINDS) == set(nr.RESIDUAL_SLOT_KINDS)
    assert set(lx.RESIDUAL_EXPORT_KINDS) < set(lx.GENUINE_EXPORT_KINDS)
    assert lx.GENUINE_EXPORT_KINDS == ("circuit_breaker", "lora", "lora_all", "lora_subtraction",
                                       "psr", "vector", "vector_all", "diffmean", "static_r",
                                       "psr_paired_write")   # the paired write (relu-differenced-native-qwen)


def test_derived_dims_reads_the_text_config_and_the_tree(tiny_base):
    assert lx._derived_dims(tiny_base) == (HIDDEN, N_LAYERS)
