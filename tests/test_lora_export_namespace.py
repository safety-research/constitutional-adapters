"""The export spells module names in the SERVED CHECKPOINT's namespace
(``lora_export.resolve_module_namespace`` and friends).

The defect (qwen certification canary, job 489694 at ceba8874): both qwen
adapters named their modules ``model.layers.N.self_attn.o_proj`` — the harness
skeleton's spelling (``AutoModelForCausalLM`` builds the text-only class) —
while the served ``Qwen3_5ForConditionalGeneration`` checkpoint spells them
``model.language_model.layers.N.self_attn.o_proj`` and vLLM's LoRA loader maps
PEFT names with the served model's ``hf_to_vllm_mapper`` (``"model.language_
model." -> "language_model.model."``, no rule for a bare ``model.layers.``):
the adapters loaded but bound to no execution module. Pinned here:

- the three checkpoint shapes: a qwen-shaped INDEX (suffix path, with the real
  ``mtp.layers.0.self_attn.o_proj`` decoy), a gemma-shaped single-file HEADER
  (exact match wins, bytes unchanged), a nemotron-shaped header
  (``backbone.layers.{12,17}.mixer.o_proj``);
- the refusals: two suffix matches, a missing layer, two prefixes across
  layers, a prefix outside ``models._LAYER_PATHS``, a snapshot with neither
  index nor safetensors, a malformed index, a truncated file;
- the record: ``adapter_digests.json`` carries ``module_namespace``; a
  pre-rule record reads as "not recorded" and is still reused when
  byte-identical; a skeleton-spelled entry at the address is refused naming
  both spellings and the remedy;
- the zero adapter and the subtraction export spell the same names as the
  cell; the stated-prefix path takes no snapshot;
- a base whose two spellings agree exports byte-identically to before.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pytest
import torch
from lora_snapshot_fixtures import (oproj_names, write_index_snapshot,
                                    write_safetensors_snapshot)

from sisteer.capeval.serving import lora_export as lx
from sisteer.models import _LAYER_PATHS

IH = "ab" * 8
D_IN, D_OUT, RANK = 16, 8, 2
QWEN_LAYERS = (3, 7, 11)
QWEN_SKELETON = lx.DecoderLayout(prefix="model.layers", oproj_site="self_attn.o_proj",
                                 oproj_layers=QWEN_LAYERS)
QWEN_SHAPES = {l: (D_IN, D_OUT) for l in QWEN_LAYERS}
QWEN_CKPT = "model.language_model.layers"
QWEN_DECOYS = ("mtp.layers.0.self_attn.o_proj.weight",
               "model.visual.blocks.0.mlp.linear_fc1.weight",
               "model.visual.merger.linear_fc1.weight",
               "model.language_model.embed_tokens.weight")
NEMO_SKELETON = lx.DecoderLayout(prefix="model.layers", oproj_site="mixer.o_proj",
                                 oproj_layers=(12, 17))
GEMMA_SKELETON = lx.DecoderLayout(prefix="model.language_model.layers",
                                  oproj_site="self_attn.o_proj", oproj_layers=(0, 1, 2))


def _bytes(d) -> bytes:
    buf = io.BytesIO()
    torch.save(d, buf)
    return buf.getvalue()


def _lora_all(layers=QWEN_LAYERS, rank=RANK, seed=3, d_in=D_IN, d_out=D_OUT) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"kind": "lora_all", "layers": list(layers), "rank": rank,
            "A": {str(l): torch.randn(d_in, rank, generator=g) for l in layers},
            "B": {str(l): torch.randn(d_out, rank, generator=g) * 0.1 for l in layers},
            "positions": "all", "response_header_len": None}


def _rank1(layer=3, seed=5) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"kind": "lora", "A": torch.randn(D_IN, generator=g),
            "B": torch.randn(D_OUT, generator=g), "layer": layer,
            "positions": "all", "response_header_len": None}


def _subtraction(layers=QWEN_LAYERS, rank=RANK) -> dict:
    pos, neg = _lora_all(layers, rank, seed=11), _lora_all(layers, rank, seed=12)
    return {"kind": "lora_subtraction", "layers": list(layers), "rank": rank,
            "A_pos": pos["A"], "B_pos": pos["B"], "A_neg": neg["A"], "B_neg": neg["B"],
            "positions": "all", "response_header_len": None,
            "provenance": {"parent_pos": {"cell_id": "mix", "object_sha256": "a" * 64},
                           "parent_neg": {"cell_id": "ctl", "object_sha256": "b" * 64}}}


def _qwen_snapshot(d: Path) -> Path:
    return write_index_snapshot(d, oproj_names(QWEN_CKPT, QWEN_LAYERS) + list(QWEN_DECOYS))


@pytest.fixture
def qwen_world(tmp_path, monkeypatch):
    """A qwen-shaped world: the SKELETON says ``model.layers`` for layers 3/7/11
    (``_derived_layout`` patched — the config-only cache cannot build the real
    class here), the local snapshot dir is the base model AND carries a
    qwen-shaped index. Returns the snapshot path (== base_model)."""
    snap = _qwen_snapshot(tmp_path / "qwen-snapshot")
    monkeypatch.setattr(lx, "_derived_layout",
                        lambda base, rev=None: (QWEN_SKELETON, dict(QWEN_SHAPES)))
    return snap


def _tensor_names(adapter: Path):
    from safetensors.torch import load_file
    return sorted(load_file(str(adapter / lx.ADAPTER_WEIGHTS)))


# --------------------------------------------------------------------------
# checkpoint_module_names: the three shapes and the refusals
# --------------------------------------------------------------------------


def test_qwen_shaped_index_maps_the_skeleton_spelling_to_the_checkpoints(tmp_path):
    snap = _qwen_snapshot(tmp_path)
    names, source = lx.checkpoint_weight_names(snap)
    assert source == "index" and "mtp.layers.0.self_attn.o_proj.weight" in names
    modules, ns = lx.checkpoint_module_names(QWEN_SKELETON, names, base_model="Qwen/x",
                                             snapshot=snap, source=source)
    assert modules == {l: f"{QWEN_CKPT}.{l}.self_attn.o_proj" for l in QWEN_LAYERS}
    assert ns == lx.ModuleNamespace(skeleton="model.layers", checkpoint=QWEN_CKPT,
                                    source="index", n_layers=3, snapshot=str(snap))
    assert ns.renamed is True and ns.checkpoint in _LAYER_PATHS


def test_gemma_shaped_single_file_header_keeps_the_exact_names(tmp_path):
    snap = write_safetensors_snapshot(
        tmp_path, oproj_names(GEMMA_SKELETON.prefix, GEMMA_SKELETON.oproj_layers)
        + ["model.language_model.embed_tokens.weight",
           "model.vision_tower.encoder.layers.0.self_attn.out_proj.weight"])
    names, source = lx.checkpoint_weight_names(snap)
    assert source == "safetensors-header" and len(names) == 5
    modules, ns = lx.checkpoint_module_names(GEMMA_SKELETON, names, base_model="google/x",
                                             snapshot=snap, source=source)
    assert modules == {l: f"{GEMMA_SKELETON.prefix}.{l}.self_attn.o_proj" for l in (0, 1, 2)}
    assert ns.renamed is False and ns.checkpoint == ns.skeleton == GEMMA_SKELETON.prefix
    assert (ns.source, ns.n_layers) == ("safetensors-header", 3)


def test_nemotron_shaped_header_maps_model_layers_to_backbone(tmp_path):
    snap = write_safetensors_snapshot(
        tmp_path, oproj_names("backbone.layers", (12, 17), "mixer.o_proj")
        + ["backbone.layers.12.mixer.q_proj.weight", "backbone.layers.12.norm.weight",
           "backbone.layers.0.mixer.in_proj.weight", "backbone.embeddings.weight"])
    names, source = lx.checkpoint_weight_names(snap)
    modules, ns = lx.checkpoint_module_names(NEMO_SKELETON, names, base_model="nvidia/x",
                                             snapshot=snap, source=source)
    assert modules == {12: "backbone.layers.12.mixer.o_proj",
                       17: "backbone.layers.17.mixer.o_proj"}
    assert (ns.skeleton, ns.checkpoint, ns.renamed) == ("model.layers", "backbone.layers", True)


def test_two_suffix_matches_refuse_naming_both_candidates(tmp_path):
    layout = QWEN_SKELETON._replace(oproj_layers=(0, 3))
    names = frozenset(oproj_names(QWEN_CKPT, (0, 3)) + ["mtp.layers.0.self_attn.o_proj.weight"])
    with pytest.raises(lx.LoraExportError, match=r"2 modules \[.*mtp\.layers\.0.*\].*layer 0") as err:
        lx.checkpoint_module_names(layout, names, base_model="Qwen/x", snapshot=tmp_path,
                                   source="index")
    msg = str(err.value)
    assert "model.language_model.layers.0.self_attn.o_proj" in msg and "refusing to guess" in msg


def test_a_layer_the_checkpoint_lacks_refuses_naming_it(tmp_path):
    names = frozenset(oproj_names(QWEN_CKPT, (3, 11)) + list(QWEN_DECOYS))   # no layer 7
    with pytest.raises(lx.LoraExportError, match=r"no module named like layer 7's attention "
                                                 r"self_attn\.o_proj.*'model\.layers\.7\.self_attn\.o_proj'"):
        lx.checkpoint_module_names(QWEN_SKELETON, names, base_model="Qwen/x",
                                   snapshot=tmp_path, source="index")


def test_layers_under_two_prefixes_refuse(tmp_path):
    names = frozenset([f"{QWEN_CKPT}.3.self_attn.o_proj.weight",
                       "backbone.layers.7.self_attn.o_proj.weight",
                       f"{QWEN_CKPT}.11.self_attn.o_proj.weight"])
    with pytest.raises(lx.LoraExportError, match="under 2 prefixes"):
        lx.checkpoint_module_names(QWEN_SKELETON, names, base_model="Qwen/x",
                                   snapshot=tmp_path, source="index")


def test_a_prefix_outside_the_known_layouts_refuses(tmp_path):
    names = frozenset(oproj_names("mtp.layers", QWEN_LAYERS))
    with pytest.raises(lx.LoraExportError, match="'mtp.layers'.*not a known decoder layout"):
        lx.checkpoint_module_names(QWEN_SKELETON, names, base_model="Qwen/x",
                                   snapshot=tmp_path, source="index")
    with pytest.raises(lx.LoraExportError, match="not one of"):
        lx.checkpoint_module_names(QWEN_SKELETON, names, base_model="Qwen/x",
                                   snapshot=tmp_path, source="guess")


def test_several_tensors_of_one_module_count_once(tmp_path):
    """An fp8 checkpoint carries ``o_proj.weight`` AND ``o_proj.weight_scale``
    under one module: one candidate, not two."""
    names = frozenset(oproj_names(QWEN_CKPT, QWEN_LAYERS)
                      + [f"{QWEN_CKPT}.{l}.self_attn.o_proj.weight_scale" for l in QWEN_LAYERS])
    modules, ns = lx.checkpoint_module_names(QWEN_SKELETON, names, base_model="Qwen/x",
                                             snapshot=tmp_path, source="index")
    assert set(modules.values()) == {f"{QWEN_CKPT}.{l}.self_attn.o_proj" for l in QWEN_LAYERS}
    assert ns.n_layers == 3


# --------------------------------------------------------------------------
# checkpoint_weight_names / the header reader / the snapshot resolution
# --------------------------------------------------------------------------


def test_snapshot_with_neither_index_nor_safetensors_refuses(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(lx.LoraExportError, match=rf"neither {lx.WEIGHT_INDEX_FILE} nor any "
                                                 r"\*\.safetensors.*config\.json.*tokenizer\.json"):
        lx.checkpoint_weight_names(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"\x80\x02")
    with pytest.raises(lx.LoraExportError, match="pytorch_model.bin"):
        lx.checkpoint_weight_names(tmp_path)
    with pytest.raises(lx.LoraExportError, match="not a directory"):
        lx.checkpoint_weight_names(tmp_path / "absent")


def test_the_index_wins_over_shard_headers_and_headers_union(tmp_path):
    both = tmp_path / "both"
    write_safetensors_snapshot(both, ["from.header.weight"])
    write_index_snapshot(both, ["from.index.weight"])
    assert lx.checkpoint_weight_names(both) == (frozenset({"from.index.weight"}), "index")
    shards = tmp_path / "shards"
    write_safetensors_snapshot(shards, ["a.0.weight"], filename="model-00001-of-00002.safetensors")
    write_safetensors_snapshot(shards, ["b.1.weight"], filename="model-00002-of-00002.safetensors")
    assert lx.checkpoint_weight_names(shards) == (frozenset({"a.0.weight", "b.1.weight"}),
                                                  "safetensors-header")


def test_malformed_index_and_truncated_files_refuse(tmp_path):
    bad = tmp_path / "bad-index"
    bad.mkdir()
    (bad / lx.WEIGHT_INDEX_FILE).write_text("{not json")
    with pytest.raises(lx.LoraExportError, match="unreadable weight index"):
        lx.checkpoint_weight_names(bad)
    (bad / lx.WEIGHT_INDEX_FILE).write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(lx.LoraExportError, match="no non-empty weight_map"):
        lx.checkpoint_weight_names(bad)
    (bad / lx.WEIGHT_INDEX_FILE).write_text(json.dumps({"metadata": {}}))
    with pytest.raises(lx.LoraExportError, match="no non-empty weight_map"):
        lx.checkpoint_weight_names(bad)
    short = tmp_path / "short"
    short.mkdir()
    (short / "model.safetensors").write_bytes(b"\x01\x02")
    with pytest.raises(lx.LoraExportError, match="no 8-byte header length"):
        lx.checkpoint_weight_names(short)
    (short / "model.safetensors").write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(lx.LoraExportError, match="not plausible"):
        lx.checkpoint_weight_names(short)
    (short / "model.safetensors").write_bytes(struct.pack("<Q", 4) + b"nope")
    with pytest.raises(lx.LoraExportError, match="not JSON"):
        lx.checkpoint_weight_names(short)
    (short / "model.safetensors").write_bytes(struct.pack("<Q", 2) + b"[]")
    with pytest.raises(lx.LoraExportError, match="not a JSON object"):
        lx.checkpoint_weight_names(short)
    (short / "model.safetensors").write_bytes(struct.pack("<Q", 19) + b'{"__metadata__":{}}')
    with pytest.raises(lx.LoraExportError, match="names no tensor"):
        lx.checkpoint_weight_names(short)


def test_header_reader_reads_the_genuine_format_and_skips_metadata(tmp_path):
    snap = write_safetensors_snapshot(tmp_path, ["x.weight", "y.weight"])
    f = snap / "model.safetensors"
    assert sorted(lx._safetensors_header_keys(f)) == ["x.weight", "y.weight"]
    with open(f, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        assert "__metadata__" in json.loads(fh.read(n))          # present, skipped


def test_checkpoint_snapshot_dir_resolution(tmp_path):
    local = tmp_path / "local-base"
    local.mkdir()
    assert lx.checkpoint_snapshot_dir(str(local)) == local.resolve()
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    assert lx.checkpoint_snapshot_dir("org/model", "deadbeef", explicit) == explicit.resolve()
    with pytest.raises(lx.LoraExportError, match="not a directory"):
        lx.checkpoint_snapshot_dir("org/model", None, tmp_path / "absent")
    with pytest.raises(lx.LoraExportError, match="not in the local HF cache.*snapshot_dir="):
        lx.checkpoint_snapshot_dir("TESTORG/not-cached-anywhere-22222222",
                                   "0" * 40)
    for bad in ("", None, 7):
        with pytest.raises(lx.LoraExportError, match="non-empty model id"):
            lx.checkpoint_snapshot_dir(bad)
    with pytest.raises(lx.LoraExportError, match="revision must be"):
        lx.checkpoint_snapshot_dir("org/model", "")


# --------------------------------------------------------------------------
# the derived export path: every kind, the zero adapter, the record
# --------------------------------------------------------------------------


def test_derived_export_spells_qwen_names_in_the_checkpoint_namespace(qwen_world, tmp_path):
    snap = qwen_world
    for kind, blob, layers in (("lora_all", _lora_all(), QWEN_LAYERS),
                               ("lora", _rank1(3), (3,)),
                               ("lora_subtraction", _subtraction(), QWEN_LAYERS)):
        adapter = lx.export_genuine_adapter({"kind": kind}, _bytes(blob),
                                            tmp_path / kind, base_model=str(snap))
        want = sorted(f"base_model.model.{QWEN_CKPT}.{l}.self_attn.o_proj.{ab}.weight"
                      for l in layers for ab in ("lora_A", "lora_B"))
        assert _tensor_names(adapter) == want, kind
        cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
        assert cfg["petri_decoder_prefix"] == QWEN_CKPT, kind
        assert cfg["target_modules"] == ["o_proj"] and cfg["layers_to_transform"] == list(layers)
        assert "petri_oproj_site" not in cfg                     # the default site adds nothing
        # nothing in the config says model.layers any more
        assert "model.layers" not in json.dumps(cfg).replace(QWEN_CKPT, "")


def test_snapshot_dir_overrides_the_base_and_the_stated_path_takes_none(qwen_world, tmp_path):
    snap = qwen_world
    # a base that is NOT a directory and not cached, with the snapshot stated
    adapter = lx.export_genuine_adapter({"kind": "lora_all"}, _bytes(_lora_all()), tmp_path / "a",
                                        base_model="Qwen/Qwen3.5-4B", base_revision="0" * 40,
                                        snapshot_dir=snap)
    assert all(n.startswith(f"base_model.model.{QWEN_CKPT}.") for n in _tensor_names(adapter))
    # ... and without it the uncached base refuses at the snapshot, not silently
    with pytest.raises(lx.LoraExportError, match="not in the local HF cache"):
        lx.export_genuine_adapter({"kind": "lora_all"}, _bytes(_lora_all()), tmp_path / "b",
                                  base_model="Qwen/Qwen3.5-4B", base_revision="0" * 40)
    # the stated-prefix path names the modules itself: no snapshot is read, and
    # stating both is a contradiction
    stated = lx.export_genuine_adapter({"kind": "lora_all"}, _bytes(_lora_all()), tmp_path / "c",
                                       base_model="Qwen/Qwen3.5-4B",
                                       decoder_prefix="model.layers")
    assert all(n.startswith("base_model.model.model.layers.") for n in _tensor_names(stated))
    with pytest.raises(lx.LoraExportError, match="not both"):
        lx.export_genuine_adapter({"kind": "lora_all"}, _bytes(_lora_all()), tmp_path / "d",
                                  base_model="Qwen/Qwen3.5-4B", decoder_prefix="model.layers",
                                  snapshot_dir=snap)
    assert not (tmp_path / "b").exists() and not (tmp_path / "d").exists()


def test_bank_records_the_namespace_and_the_zero_adapter_shares_the_spelling(qwen_world, tmp_path):
    snap = qwen_world
    root = tmp_path / "bank"
    d, dig, reused = lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH},
                                        _bytes(_lora_all()), root, base_model=str(snap),
                                        name_hint="si_mix__lora_r4_ce")
    assert reused is False and dig.petri_intervention_hash == IH
    rec = json.loads((d / lx.ADAPTER_DIGESTS).read_text())
    assert rec["module_namespace"] == {"skeleton": "model.layers", "checkpoint": QWEN_CKPT,
                                       "source": "index", "n_layers": 3,
                                       "snapshot": str(snap.resolve())}
    assert lx.read_module_namespace(d) == lx.ModuleNamespace(**rec["module_namespace"])
    assert lx.read_module_namespace(d).renamed is True
    # reuse: byte-identical, nothing rewritten, the record stands
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    d2, dig2, reused2 = lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH},
                                           _bytes(_lora_all()), root, base_model=str(snap),
                                           name_hint="si_mix__lora_r4_ce")
    assert (d2, reused2) == (d, True) and dig2 == dig
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before
    # the zero adapter: checkpoint_target_modules names, the same spelling
    modules, ns = lx.checkpoint_target_modules(str(snap))
    assert modules == {f"{QWEN_CKPT}.{l}.self_attn.o_proj": (D_IN, D_OUT) for l in QWEN_LAYERS}
    assert ns.checkpoint == QWEN_CKPT
    assert lx.oproj_target_modules(str(snap)) == {          # the skeleton oracle is unchanged
        f"model.layers.{l}.self_attn.o_proj": (D_IN, D_OUT) for l in QWEN_LAYERS}
    zd, zdig, z_reused = lx.export_or_reuse_zero(root, base_model=str(snap))
    assert z_reused is False and zdig.petri_zero_adapter is True
    assert _tensor_names(zd) == sorted(
        f"base_model.model.{QWEN_CKPT}.{l}.self_attn.o_proj.{ab}.weight"
        for l in QWEN_LAYERS for ab in ("lora_A", "lora_B"))
    assert lx.read_module_namespace(zd) == ns
    # its bank key moves with the spelling: the skeleton-named key is another dir
    skeleton_key = lx.zero_adapter_key(base_model=str(snap), base_revision=None,
                                       target_modules=lx.oproj_target_modules(str(snap)), rank=1)
    assert zd.name != f"{lx.ZERO_ADAPTER_NAME}__{skeleton_key}"
    assert zd.name == f"{lx.ZERO_ADAPTER_NAME}__" + lx.zero_adapter_key(
        base_model=str(snap), base_revision=None, target_modules=modules, rank=1)
    lx.verify_zero_adapter(zd)


def test_a_pre_rule_record_reads_as_not_recorded_and_still_reuses_identical_bytes(
        qwen_world, tmp_path):
    snap = qwen_world
    root = tmp_path / "bank"
    d, dig, _ = lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH},
                                   _bytes(_lora_all()), root, base_model=str(snap))
    rec = json.loads((d / lx.ADAPTER_DIGESTS).read_text())
    del rec["module_namespace"]                            # an entry predating the field
    (d / lx.ADAPTER_DIGESTS).write_text(json.dumps(rec))
    assert lx.read_module_namespace(d) is None
    d2, dig2, reused = lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH},
                                          _bytes(_lora_all()), root, base_model=str(snap))
    assert reused is True and d2 == d and dig2 == dig._replace(name_hint=None)
    assert "module_namespace" not in json.loads((d / lx.ADAPTER_DIGESTS).read_text())  # untouched
    # a malformed record is tampering, not "not recorded"
    for bad in ({"skeleton": "model.layers"}, "model.layers",
                {"skeleton": "model.layers", "checkpoint": QWEN_CKPT, "source": "guess",
                 "n_layers": 3, "snapshot": str(snap)},
                {"skeleton": "model.layers", "checkpoint": QWEN_CKPT, "source": "index",
                 "n_layers": 0, "snapshot": str(snap)},
                {"skeleton": "", "checkpoint": QWEN_CKPT, "source": "index",
                 "n_layers": 3, "snapshot": str(snap)}):
        rec["module_namespace"] = bad
        (d / lx.ADAPTER_DIGESTS).write_text(json.dumps(rec))
        with pytest.raises(lx.LoraExportError, match="stale or tampered.*module_namespace"):
            lx.read_module_namespace(d)
        with pytest.raises(lx.LoraExportError, match="stale or tampered.*module_namespace"):
            lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH},
                               _bytes(_lora_all()), root, base_model=str(snap))


def test_a_skeleton_spelled_entry_at_the_address_is_refused_with_the_remedy(qwen_world, tmp_path):
    """The qwen canary's defect on disk: an entry exported before the rule
    sits at the cell's intervention_hash address spelling ``model.layers``;
    the bank refuses it (not byte-identical), naming both spellings and the
    remedy — never writes over it, never trusts it by address."""
    snap = qwen_world
    root = tmp_path / "bank"
    root.mkdir()
    target = lx.adapter_dir_for(root, IH)
    lx.export_genuine_adapter({"kind": "lora_all", "intervention_hash": IH}, _bytes(_lora_all()),
                              target, base_model=str(snap), decoder_prefix="model.layers")
    lx._write_digests_file(target, lx.adapter_digests(target)._replace(dir=str(target)),
                           module_namespace=None)
    rec = json.loads((target / lx.ADAPTER_DIGESTS).read_text())
    del rec["module_namespace"]
    (target / lx.ADAPTER_DIGESTS).write_text(json.dumps(rec))
    stale = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(lx.LoraExportError, match="stale or tampered bank entry; not reusing") as err:
        lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH}, _bytes(_lora_all()),
                           root, base_model=str(snap), name_hint="cell")
    msg = str(err.value)
    assert "exported before the checkpoint-namespace rule (D97)" in msg
    assert f"'{QWEN_CKPT}'" in msg and "skeleton 'model.layers'" in msg
    assert "remove the entry's dir" in msg and "job 489694" in msg
    assert {p.name: p.read_bytes() for p in target.iterdir()} == stale        # untouched
    assert [p.name for p in root.iterdir() if p.name.startswith(".tmp-")] == []
    # the same, when the entry RECORDS another spelling
    rec["module_namespace"] = {"skeleton": "model.layers", "checkpoint": "model.layers",
                               "source": "index", "n_layers": 3, "snapshot": "/elsewhere"}
    (target / lx.ADAPTER_DIGESTS).write_text(json.dumps(rec))
    with pytest.raises(lx.LoraExportError, match="its modules spelled under 'model.layers'"):
        lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH}, _bytes(_lora_all()),
                           root, base_model=str(snap))


def test_a_config_only_snapshot_refuses_the_derived_export(tmp_path):
    """The fixture shape every derived-path test used before this rule — a
    saved config and nothing else — is not a checkpoint: refused, never
    exported under the skeleton's spelling."""
    from transformers import LlamaConfig

    base = tmp_path / "config-only"
    LlamaConfig(hidden_size=D_IN, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=97,
                max_position_embeddings=64).save_pretrained(base)
    blob = _lora_all(layers=(0, 1))
    with pytest.raises(lx.LoraExportError, match=rf"neither {lx.WEIGHT_INDEX_FILE} nor any"):
        lx.export_genuine_adapter({"kind": "lora_all"}, _bytes(blob), tmp_path / "a",
                                  base_model=str(base))
    with pytest.raises(lx.LoraExportError, match=rf"neither {lx.WEIGHT_INDEX_FILE} nor any"):
        lx.export_or_reuse_zero(tmp_path / "bank", base_model=str(base))
    assert not (tmp_path / "a").exists()
    # the layout oracles still answer off the config alone
    assert lx.resolve_decoder_layout(str(base)).oproj_layers == (0, 1)
    assert set(lx.oproj_target_modules(str(base))) == {
        "model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"}


def test_names_that_already_match_export_byte_identically_to_before(tmp_path):
    """A base whose checkpoint spells the decoder as the skeleton does (llama,
    gemma-4, GLM): the derived export equals the stated-prefix export byte
    for byte — the rule changed nothing there — and the record says so."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from sisteer.capeval.serving import serving_attestation as sa

    cfg = LlamaConfig(hidden_size=D_IN, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=97,
                      max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    base = tmp_path / "tiny-llama"
    model.save_pretrained(base)                                   # config + model.safetensors
    names, source = lx.checkpoint_weight_names(base)
    assert source == "safetensors-header"
    assert {f"model.layers.{l}.self_attn.o_proj.weight" for l in (0, 1)} <= names
    blob = _lora_all(layers=(0, 1), d_in=D_IN, d_out=D_IN)          # llama's o_proj is square
    derived = lx.export_genuine_adapter({"kind": "lora_all"}, _bytes(blob), tmp_path / "derived",
                                        base_model=str(base))
    stated = lx.export_genuine_adapter({"kind": "lora_all"}, _bytes(blob), tmp_path / "stated",
                                       base_model=str(base), decoder_prefix=lx.LLAMA_DECODER_PREFIX)
    for f in (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS):
        assert sa.sha256_file(derived / f) == sa.sha256_file(stated / f), f
    d, dig, _ = lx.export_or_reuse({"kind": "lora_all", "intervention_hash": IH}, _bytes(blob),
                                   tmp_path / "bank", base_model=str(base))
    ns = lx.read_module_namespace(d)
    assert ns.renamed is False and ns.skeleton == ns.checkpoint == "model.layers"
    assert (ns.source, ns.n_layers) == ("safetensors-header", 2)
    zd, _, _ = lx.export_or_reuse_zero(tmp_path / "bank", base_model=str(base))
    assert zd.name == f"{lx.ZERO_ADAPTER_NAME}__" + lx.zero_adapter_key(
        base_model=str(base), base_revision=None,
        target_modules=lx.oproj_target_modules(str(base)), rank=1)     # the pre-rule key
