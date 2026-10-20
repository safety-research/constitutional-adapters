"""The content-addressed adapter bank (serving/lora_export.py:
``adapter_dir_for``, ``export_or_reuse``,
``export_or_reuse_zero``, ``adapter_digests``, ``verify_adapter_dir``,
``zero_adapter_key`` / ``zero_adapter_dir_for``, ``oproj_target_modules``).

Load-bearing here:

- CONTENT ADDRESSING: a genuine adapter lives at ``root/<intervention_
  hash>``; the zero adapter at ``root/none__lora0__<key>`` with the key a
  hash over (base, revision, every o_proj module's shape, rank);
- ATOMIC WRITE: the export lands through a private tmp dir and ONE
  rename; no ``.tmp-*`` dir is ever left behind — not after success, not
  after a refusal, not after a failed export;
- REUSE is verified, never trusted: an entry is reused only when it
  carries its own ``adapter_digests.json``, re-hashes to it, is the kind
  of adapter expected at that address, AND a fresh export of the same
  input bytes is byte-identical to it — and reuse REWRITES NOTHING (the
  files and the record are byte-identical before and after);
- a tampered weights or config file, a missing record, a same-hash entry
  whose fresh export differs, or a zero adapter sitting at a genuine
  address all REFUSE naming the dir, and the bank never writes over the
  disagreeing entry;
- the rename race (a peer installs the entry between our look and our
  rename) falls into the verified-reuse branch;
- the zero adapter is exported over EVERY o_proj layer of the base
  (``oproj_target_modules`` off a meta skeleton), passes
  ``verify_zero_adapter``, keeps ``lora_alpha == r``, and its config's
  ``target_modules`` is the un-deduplicated ``["o_proj", "o_proj", …]``
  whose unique contents are exactly {"o_proj"} (what the lora-native
  compose-time preflight admits).

Hermetic: the tiny llama's config is saved to a LOCAL directory and used
as ``base_model``, so the export's layout derivation (``local_files_
only``) never touches the hub or the HF cache.
"""

import hashlib
import io
import json
from pathlib import Path
import re
import shutil

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx

HIDDEN = 32
N_LAYERS = 2
IH = "ab" * 8
IH2 = "cd" * 8
HEX16 = re.compile(r"[0-9a-f]{16}")
HEX64 = re.compile(r"[0-9a-f]{64}")
PEFT_FILES = (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS)


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


def _lora_all_art(tiny_model, seed=7, rank=4) -> bytes:
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    obj = MultiLayerLoRAObject(tiny_model, rank=rank)
    g = torch.Generator().manual_seed(seed)
    for l in obj.layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g)
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.1
    if not hasattr(obj, "fit_positions"):
        obj.fit_positions = "all"
        obj.fit_response_header_len = None
    return _blob_bytes(obj.to_blob())


def _subtraction_art(tiny_model, seeds=(11, 12), rank=4) -> bytes:
    from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                               MultiLayerLoRAObject)

    def parent(seed):
        obj = MultiLayerLoRAObject(tiny_model, rank=rank)
        g = torch.Generator().manual_seed(seed)
        for l in obj.layers:
            obj.A[l].data = torch.randn(obj.A[l].shape, generator=g)
            obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.1
        if not hasattr(obj, "fit_positions"):
            obj.fit_positions = "all"
            obj.fit_response_header_len = None
        return obj.to_blob()

    return _blob_bytes(LoRASubtractionObject.compose_blob(
        parent(seeds[0]), parent(seeds[1]),
        provenance={"parent_pos": {"cell_id": "mix", "object_sha256": "a" * 64},
                    "parent_neg": {"cell_id": "ctl", "object_sha256": "b" * 64}}))


def _spec(ih=IH, kind="lora_all"):
    return {"kind": kind, "intervention_hash": ih}


def _snapshot(d):
    """Every file's bytes under ``d`` (non-recursive) — to prove the bank
    rewrote nothing."""
    return {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}


def _entries(root):
    return sorted(p.name for p in root.iterdir())


def _tmp_dirs(root):
    return [p.name for p in root.iterdir() if p.name.startswith(".tmp-")]


# --------------------------------------------------------------------------
# addresses
# --------------------------------------------------------------------------


def test_bank_addresses_are_content_addresses(tmp_path):
    assert lx.adapter_dir_for(tmp_path, IH) == tmp_path / IH
    # a real key, minted at run time (never a key-shaped literal: the
    # gitleaks hook reads this file too)
    key = lx.zero_adapter_key(
        base_model="org/base", base_revision=None,
        target_modules={"model.layers.0.self_attn.o_proj": (8, 8)}, rank=1)
    assert HEX16.fullmatch(key)
    assert lx.zero_adapter_dir_for(tmp_path, key) == \
        tmp_path / f"{lx.ZERO_ADAPTER_NAME}__{key}"
    assert lx.ZERO_ADAPTER_NAME == "none__lora0"
    for bad in ("AB" * 8, "ab" * 7, "ab" * 9, "", None, "../" + "a" * 13,
                "zz" * 8, 7):
        with pytest.raises(lx.LoraExportError, match="16-hex"):
            lx.adapter_dir_for(tmp_path, bad)
        with pytest.raises(lx.LoraExportError, match="16-hex"):
            lx.zero_adapter_dir_for(tmp_path, bad)
    # nothing was created by asking for an address
    assert _entries(tmp_path) == []


# --------------------------------------------------------------------------
# export_or_reuse: export, then verified reuse; nothing rewritten
# --------------------------------------------------------------------------


def test_export_or_reuse_exports_then_reuses_byte_identically(
        tiny_model, tiny_base, tmp_path):
    root = tmp_path / "bank"
    art = _lora_all_art(tiny_model)

    d1, g1, reused1 = lx.export_or_reuse(_spec(), art, root,
                                         base_model=tiny_base,
                                         name_hint="si_mix__lora_r4_ce")
    assert reused1 is False
    assert d1 == lx.adapter_dir_for(root.resolve(), IH)
    assert _entries(root) == [IH] and _tmp_dirs(root) == []
    assert sorted(p.name for p in d1.iterdir()) == sorted(
        (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS, lx.ADAPTER_DIGESTS))
    # the digests describe the files on disk
    assert isinstance(g1, lx.AdapterDigests)
    assert g1.dir == str(d1) and g1.name_hint == "si_mix__lora_r4_ce"
    assert g1.r == 4 and g1.petri_intervention_hash == IH
    assert g1.petri_zero_adapter is False
    assert g1.petri_source_kind == "lora_all"
    for field, name in (("config_sha256", lx.ADAPTER_CONFIG),
                        ("weights_sha256", lx.ADAPTER_WEIGHTS)):
        assert HEX64.fullmatch(getattr(g1, field))
        assert getattr(g1, field) == hashlib.sha256(
            (d1 / name).read_bytes()).hexdigest()
    # the record beside the files: the fields + provenance, nothing else
    rec = json.loads((d1 / lx.ADAPTER_DIGESTS).read_text())
    assert set(rec) == set(lx.AdapterDigests._fields) | {
        "exported_at", "exporter_tree_sha", "module_namespace"}
    # the spelling the names carry, where it was read from — the tiny
    # llama's checkpoint spells the decoder as the skeleton does
    assert rec["module_namespace"] == {
        "skeleton": "model.layers", "checkpoint": "model.layers",
        "source": "safetensors-header", "n_layers": N_LAYERS,
        "snapshot": str(Path(tiny_base).resolve())}
    assert lx.read_module_namespace(d1) == lx.ModuleNamespace(**rec["module_namespace"])
    assert {k: rec[k] for k in lx.AdapterDigests._fields} == g1._asdict()
    assert isinstance(rec["exported_at"], float)
    assert rec["exporter_tree_sha"] is None or re.fullmatch(
        r"[0-9a-f]{40}", rec["exporter_tree_sha"])
    # a re-verify reads the same content identity (no cell hint)
    assert lx.adapter_digests(d1) == g1._replace(name_hint=None)
    lx.verify_adapter_dir(d1, g1)
    # the adapter itself is what export_genuine_adapter writes
    cfg = json.loads((d1 / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_genuine_adapter"] is True
    assert cfg["petri_intervention_hash"] == IH
    assert cfg["petri_decoder_prefix"] == "model.layers"

    before = _snapshot(d1)
    d2, g2, reused2 = lx.export_or_reuse(_spec(), art, root,
                                         base_model=tiny_base,
                                         name_hint="another_cell_same_object")
    assert reused2 is True and d2 == d1
    assert g2 == g1._replace(name_hint="another_cell_same_object")
    # REUSE REWROTE NOTHING: files and record byte-identical, no tmp dir
    assert _snapshot(d1) == before
    assert _entries(root) == [IH] and _tmp_dirs(root) == []
    # a second object under its own hash sits beside it
    d3, g3, reused3 = lx.export_or_reuse(
        _spec(IH2), _lora_all_art(tiny_model, seed=8), root,
        base_model=tiny_base)
    assert reused3 is False and d3 == root.resolve() / IH2
    assert g3.name_hint is None and g3.weights_sha256 != g1.weights_sha256
    assert _entries(root) == [IH, IH2]


def test_export_or_reuse_round_trips_untouched_rank_one_save_bytes(
        tiny_model, tiny_base, tmp_path):
    """THE FINDING (code review 2026-09-11, P2): a policy-to-bank round trip
    on the bytes ``RankOneLoRAObject.save`` writes UNTOUCHED — which carry
    ``response_header_len: None`` for an all-position fit. The bank exported
    them as a refusal ("must be an int >= 0") before; now the entry is
    written, verified, and reused byte-identically like every other kind."""
    from sisteer.extraction.objects import RankOneLoRAObject

    obj = RankOneLoRAObject(tiny_model, layer=1)
    g = torch.Generator().manual_seed(31)
    obj.A.data = torch.randn(obj.A.shape[0], generator=g)
    obj.B.data = torch.randn(obj.B.shape[0], generator=g) * 0.1
    art_path = tmp_path / "object.pt"
    obj.save(art_path)
    art = art_path.read_bytes()
    blob = torch.load(art_path, map_location="cpu", weights_only=True)
    assert blob["response_header_len"] is None and blob["positions"] == "all"
    spec = {"kind": "lora", "layer": 1, "positions": "all", "intervention_hash": IH2}
    root = tmp_path / "bank"
    d1, g1, reused1 = lx.export_or_reuse(spec, art, root, base_model=tiny_base,
                                         name_hint="si_a__lora_r1")
    assert reused1 is False and d1 == lx.adapter_dir_for(root.resolve(), IH2)
    assert g1.r == 1 and g1.petri_intervention_hash == IH2
    assert g1.petri_zero_adapter is False
    lx.verify_adapter_dir(d1, g1)
    before = _snapshot(d1)
    d2, g2, reused2 = lx.export_or_reuse(spec, art, root, base_model=tiny_base,
                                         name_hint="si_a__lora_r1")
    assert reused2 is True and d2 == d1 and g2 == g1 and _snapshot(d1) == before


def test_export_or_reuse_requires_an_intervention_hash(tiny_model, tiny_base,
                                                       tmp_path):
    root = tmp_path / "bank"
    art = _lora_all_art(tiny_model)
    with pytest.raises(lx.LoraExportError, match="no intervention_hash"):
        lx.export_or_reuse({"kind": "lora_all"}, art, root,
                           base_model=tiny_base)
    for bad in ("", "AB" * 8, "ab" * 5, 12):
        with pytest.raises(lx.LoraExportError, match="16-hex"):
            lx.export_or_reuse(_spec(bad), art, root, base_model=tiny_base)
    for bad_hint in ("", 7):
        with pytest.raises(lx.LoraExportError, match="name_hint"):
            lx.export_or_reuse(_spec(), art, root, base_model=tiny_base,
                               name_hint=bad_hint)
    with pytest.raises(lx.LoraExportError, match="resolved_spec must be"):
        lx.export_or_reuse(["kind"], art, root, base_model=tiny_base)
    # every refusal above came before anything touched the root
    assert not root.exists()


def test_export_or_reuse_leaves_nothing_behind_on_a_failed_export(
        tiny_model, tiny_base, tmp_path):
    """A refused export (a response-only fit) creates neither the entry
    nor a tmp dir; a bad kind refuses the same way."""
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    root = tmp_path / "bank"
    obj = MultiLayerLoRAObject(tiny_model, rank=4)
    blob = obj.to_blob()
    blob["positions"] = "response"
    with pytest.raises(lx.LoraExportError, match="lens o_proj"):
        lx.export_or_reuse(_spec(), _blob_bytes(blob), root,
                           base_model=tiny_base)
    assert _entries(root) == []
    with pytest.raises(lx.LoraExportError, match="no genuine PEFT export"):
        lx.export_or_reuse(_spec(kind="loreft"), b"bytes", root,
                           base_model=tiny_base)
    assert _entries(root) == []
    # an uncached base refuses (layout derivation) — still nothing behind
    with pytest.raises(lx.LoraExportError, match="not in the local HF cache"):
        lx.export_or_reuse(_spec(), _lora_all_art(tiny_model), root,
                           base_model="TESTORG/not-cached-anywhere")
    assert _entries(root) == []


def test_export_or_reuse_refuses_a_tampered_entry_and_never_writes_over_it(
        tiny_model, tiny_base, tmp_path):
    root = tmp_path / "bank"
    art = _lora_all_art(tiny_model)
    d, digests, _ = lx.export_or_reuse(_spec(), art, root,
                                       base_model=tiny_base)

    # (a) weights tampered: flip one byte of tensor data
    w = d / lx.ADAPTER_WEIGHTS
    raw = bytearray(w.read_bytes())
    raw[-1] ^= 0x01
    w.write_bytes(bytes(raw))
    tampered = _snapshot(d)
    with pytest.raises(lx.LoraExportError, match="weights_sha256") as err:
        lx.verify_adapter_dir(d, digests)
    assert "disagree" in str(err.value)
    with pytest.raises(lx.LoraExportError,
                       match="stale or tampered bank entry; not reusing") as err:
        lx.export_or_reuse(_spec(), art, root, base_model=tiny_base)
    assert "weights_sha256" in str(err.value)
    assert _snapshot(d) == tampered          # untouched, not "repaired"
    assert _tmp_dirs(root) == [] and _entries(root) == [IH]

    # (b) config tampered (an extra key), weights restored
    shutil.rmtree(d)
    d, digests, _ = lx.export_or_reuse(_spec(), art, root,
                                       base_model=tiny_base)
    cfg_path = d / lx.ADAPTER_CONFIG
    cfg = json.loads(cfg_path.read_text())
    cfg_path.write_text(json.dumps({**cfg, "tampered": 1}, indent=2,
                                   sort_keys=True))
    tampered = _snapshot(d)
    with pytest.raises(lx.LoraExportError,
                       match="stale or tampered bank entry; not reusing") as err:
        lx.export_or_reuse(_spec(), art, root, base_model=tiny_base)
    assert "config_sha256" in str(err.value)
    assert _snapshot(d) == tampered and _tmp_dirs(root) == []

    # (c) the record is missing / malformed: not written by this bank
    shutil.rmtree(d)
    d, digests, _ = lx.export_or_reuse(_spec(), art, root,
                                       base_model=tiny_base)
    (d / lx.ADAPTER_DIGESTS).unlink()
    with pytest.raises(lx.LoraExportError,
                       match=f"no {lx.ADAPTER_DIGESTS}"):
        lx.export_or_reuse(_spec(), art, root, base_model=tiny_base)
    assert not (d / lx.ADAPTER_DIGESTS).exists() and _tmp_dirs(root) == []
    rec = json.loads(json.dumps(digests._asdict()))
    rec.update(exported_at=1.0, exporter_tree_sha=None, extra="x")
    (d / lx.ADAPTER_DIGESTS).write_text(json.dumps(rec))
    with pytest.raises(lx.LoraExportError, match="keys"):
        lx.export_or_reuse(_spec(), art, root, base_model=tiny_base)
    rec.pop("extra")
    rec["config_sha256"] = "not-a-sha"
    (d / lx.ADAPTER_DIGESTS).write_text(json.dumps(rec))
    with pytest.raises(lx.LoraExportError, match="64-hex"):
        lx.export_or_reuse(_spec(), art, root, base_model=tiny_base)
    # a record that lies about the hash (well-formed, wrong value)
    rec["config_sha256"] = "0" * 64
    (d / lx.ADAPTER_DIGESTS).write_text(json.dumps(rec))
    with pytest.raises(lx.LoraExportError,
                       match="stale or tampered bank entry; not reusing"):
        lx.export_or_reuse(_spec(), art, root, base_model=tiny_base)
    assert _tmp_dirs(root) == []


def test_export_or_reuse_refuses_a_same_hash_entry_whose_fresh_export_differs(
        tiny_model, tiny_base, tmp_path):
    """Two DIFFERENT objects claiming one intervention hash: the second
    finds a well-formed, self-consistent entry whose bytes are not what
    its own export produces — refuse, and leave the first entry alone."""
    root = tmp_path / "bank"
    art_x = _lora_all_art(tiny_model, seed=7)
    art_y = _lora_all_art(tiny_model, seed=99)
    assert art_x != art_y
    d, _, _ = lx.export_or_reuse(_spec(), art_x, root, base_model=tiny_base)
    before = _snapshot(d)
    with pytest.raises(lx.LoraExportError,
                       match="stale or tampered bank entry; not reusing") as err:
        lx.export_or_reuse(_spec(), art_y, root, base_model=tiny_base)
    assert "NOT byte-identical" in str(err.value)
    assert _snapshot(d) == before
    assert _entries(root) == [IH] and _tmp_dirs(root) == []
    # the same bytes under a DIFFERENT name_hint are still the same entry
    _, g, reused = lx.export_or_reuse(_spec(), art_x, root,
                                      base_model=tiny_base, name_hint="alias")
    assert reused is True and g.name_hint == "alias"


def test_export_or_reuse_refuses_the_wrong_kind_of_entry_at_an_address(
        tiny_model, tiny_base, tmp_path):
    """A zero adapter parked at a genuine address, or a genuine adapter at
    the zero address, refuses by kind even when self-consistent."""
    root = tmp_path / "bank"
    zero_dir, _, _ = lx.export_or_reuse_zero(root, base_model=tiny_base)
    genuine_addr = lx.adapter_dir_for(root, IH)
    shutil.copytree(zero_dir, genuine_addr)
    with pytest.raises(lx.LoraExportError,
                       match="expected a genuine adapter under") as err:
        lx.export_or_reuse(_spec(), _lora_all_art(tiny_model), root,
                           base_model=tiny_base)
    assert "stale or tampered bank entry; not reusing" in str(err.value)
    # a genuine adapter whose recorded hash is another cell's
    shutil.rmtree(genuine_addr)
    other, _, _ = lx.export_or_reuse(_spec(IH2), _lora_all_art(tiny_model),
                                     root, base_model=tiny_base)
    shutil.copytree(other, genuine_addr)
    with pytest.raises(lx.LoraExportError,
                       match="expected a genuine adapter under") as err:
        lx.export_or_reuse(_spec(), _lora_all_art(tiny_model), root,
                           base_model=tiny_base)
    assert IH2 in str(err.value)
    # and a genuine adapter parked at the zero address
    shutil.rmtree(zero_dir)
    shutil.copytree(other, zero_dir)
    with pytest.raises(lx.LoraExportError,
                       match="recorded as a genuine adapter"):
        lx.export_or_reuse_zero(root, base_model=tiny_base)
    assert _tmp_dirs(root) == []


def test_export_or_reuse_race_falls_into_the_verified_reuse_branch(
        tiny_model, tiny_base, tmp_path, monkeypatch):
    """A peer installs the entry between our existence check and our
    rename: the kernel refuses the rename (EEXIST/ENOTEMPTY), and we
    verify the peer's entry against our fresh export instead of failing
    or writing over it."""
    root = tmp_path / "bank"
    art = _lora_all_art(tiny_model)
    real_rename = lx._rename_dir
    raced = []

    def racing_rename(src, dst):
        if not raced:
            raced.append(dst)
            monkeypatch.setattr(lx, "_rename_dir", real_rename)
            peer_dir, _, peer_reused = lx.export_or_reuse(
                _spec(), art, root, base_model=tiny_base, name_hint="peer")
            assert peer_reused is False and peer_dir == dst
        real_rename(src, dst)

    monkeypatch.setattr(lx, "_rename_dir", racing_rename)
    d, digests, reused = lx.export_or_reuse(_spec(), art, root,
                                            base_model=tiny_base,
                                            name_hint="me")
    assert raced and reused is True
    assert d == root.resolve() / IH and digests.name_hint == "me"
    assert _entries(root) == [IH] and _tmp_dirs(root) == []
    assert json.loads((d / lx.ADAPTER_DIGESTS).read_text())["name_hint"] \
        == "peer"                      # the peer's record stands, unrewritten
    # a racing peer with DIFFERENT bytes would be a refusal, not a reuse
    monkeypatch.setattr(lx, "_rename_dir", real_rename)
    shutil.rmtree(d)
    raced.clear()
    art_other = _lora_all_art(tiny_model, seed=5)

    def racing_other(src, dst):
        if not raced:
            raced.append(dst)
            monkeypatch.setattr(lx, "_rename_dir", real_rename)
            lx.export_or_reuse(_spec(), art_other, root, base_model=tiny_base)
        real_rename(src, dst)

    monkeypatch.setattr(lx, "_rename_dir", racing_other)
    with pytest.raises(lx.LoraExportError, match="NOT byte-identical"):
        lx.export_or_reuse(_spec(), art, root, base_model=tiny_base)
    assert _entries(root) == [IH] and _tmp_dirs(root) == []


def test_export_or_reuse_banks_the_subtraction_kind(tiny_model, tiny_base,
                                                    tmp_path):
    """The new kind rides the same bank: dispatched by
    export_genuine_adapter with the DERIVED layout, r == 2·rank recorded."""
    root = tmp_path / "bank"
    art = _subtraction_art(tiny_model)
    d, g, reused = lx.export_or_reuse(_spec(kind="lora_subtraction"), art,
                                      root, base_model=tiny_base,
                                      name_hint="si_mix__lora_r4_ce_minus_ctl")
    assert reused is False and g.r == 8
    assert g.petri_source_kind == "lora_subtraction"
    assert g.petri_intervention_hash == IH and g.petri_zero_adapter is False
    cfg = json.loads((d / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_lora_subtraction_parent_rank"] == 4
    assert cfg["petri_decoder_prefix"] == "model.layers"
    _, g2, reused2 = lx.export_or_reuse(_spec(kind="lora_subtraction"), art,
                                        root, base_model=tiny_base)
    assert reused2 is True and g2 == g._replace(name_hint=None)
    # the lora_all bytes under the subtraction's hash: a different export
    with pytest.raises(lx.LoraExportError, match="stores kind"):
        lx.export_or_reuse(_spec(kind="lora_subtraction"),
                           _lora_all_art(tiny_model), root,
                           base_model=tiny_base)
    assert _tmp_dirs(root) == []


# --------------------------------------------------------------------------
# adapter_digests / verify_adapter_dir on their own
# --------------------------------------------------------------------------


def test_adapter_digests_reads_this_modules_exports_and_refuses_others(
        tiny_model, tmp_path):
    from safetensors.torch import save_file

    adapter = lx.export_genuine_adapter(
        {"kind": "lora_all", "intervention_hash": IH},
        _lora_all_art(tiny_model), tmp_path / "adapter",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        decoder_prefix=lx.LLAMA_DECODER_PREFIX)
    g = lx.adapter_digests(adapter)
    assert g.dir == str(adapter.resolve()) and g.name_hint is None
    assert g.r == 4 and g.petri_intervention_hash == IH
    assert g.petri_zero_adapter is False and g.petri_source_kind == "lora_all"
    lx.verify_adapter_dir(adapter, g)
    with pytest.raises(lx.LoraExportError, match="AdapterDigests"):
        lx.verify_adapter_dir(adapter, g._asdict())
    # a record that disagrees on several fields names them all
    wrong = g._replace(r=7, petri_source_kind="lora_subtraction")
    with pytest.raises(lx.LoraExportError, match="disagree") as err:
        lx.verify_adapter_dir(adapter, wrong)
    assert "'r'" in str(err.value) and "petri_source_kind" in str(err.value)
    # dir / name_hint are place and use, not content
    lx.verify_adapter_dir(adapter, g._replace(dir="/elsewhere",
                                              name_hint="whatever"))

    # a zero adapter reads as such
    zero = lx.export_zero_adapter(
        tmp_path / "zero", base_model="meta-llama/Llama-3.1-8B-Instruct",
        rank=1, target_modules={"model.layers.0.self_attn.o_proj": (8, 8)})
    gz = lx.adapter_digests(zero)
    assert gz.petri_zero_adapter is True and gz.petri_intervention_hash is None
    assert gz.petri_source_kind is None and gz.r == 1

    # refusals: a foreign PEFT adapter (neither marker), both markers,
    # a bad r, a missing file, an unparseable config
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    save_file({"w": torch.zeros(2)}, str(foreign / lx.ADAPTER_WEIGHTS))
    (foreign / lx.ADAPTER_CONFIG).write_text(json.dumps(
        {"peft_type": "LORA", "r": 4, "lora_alpha": 4.0}))
    with pytest.raises(lx.LoraExportError, match="neither"):
        lx.adapter_digests(foreign)
    (foreign / lx.ADAPTER_CONFIG).write_text(json.dumps(
        {"r": 4, "petri_zero_adapter": True, "petri_genuine_adapter": True}))
    with pytest.raises(lx.LoraExportError, match="both"):
        lx.adapter_digests(foreign)
    (foreign / lx.ADAPTER_CONFIG).write_text(json.dumps(
        {"r": 0, "petri_genuine_adapter": True}))
    with pytest.raises(lx.LoraExportError, match="positive int"):
        lx.adapter_digests(foreign)
    (foreign / lx.ADAPTER_CONFIG).write_text("{not json")
    with pytest.raises(lx.LoraExportError, match="unparseable"):
        lx.adapter_digests(foreign)
    (foreign / lx.ADAPTER_WEIGHTS).unlink()
    with pytest.raises(lx.LoraExportError, match="not a PEFT adapter dir"):
        lx.adapter_digests(foreign)


# --------------------------------------------------------------------------
# the zero adapter: key, multi-layer export, verified reuse
# --------------------------------------------------------------------------


def test_zero_adapter_key_is_stable_and_shape_sensitive():
    tm = {"model.layers.0.self_attn.o_proj": (32, 32),
          "model.layers.1.self_attn.o_proj": (32, 32)}
    key = lx.zero_adapter_key(base_model="org/base", base_revision=None,
                              target_modules=tm, rank=1)
    assert HEX16.fullmatch(key)
    # stable across calls and across dict insertion order
    assert key == lx.zero_adapter_key(
        base_model="org/base", base_revision=None,
        target_modules=dict(reversed(list(tm.items()))), rank=1)
    # a list-shaped entry is the same key as the tuple-shaped one
    assert key == lx.zero_adapter_key(
        base_model="org/base", base_revision=None,
        target_modules={m: list(s) for m, s in tm.items()}, rank=1)
    variants = [
        dict(base_model="org/other", base_revision=None, target_modules=tm,
             rank=1),
        dict(base_model="org/base", base_revision="abc", target_modules=tm,
             rank=1),
        dict(base_model="org/base", base_revision=None, target_modules=tm,
             rank=2),
        dict(base_model="org/base", base_revision=None,
             target_modules={**tm, "model.layers.2.self_attn.o_proj": (32, 32)},
             rank=1),
        dict(base_model="org/base", base_revision=None,
             target_modules={**tm, "model.layers.1.self_attn.o_proj": (32, 40)},
             rank=1),
        dict(base_model="org/base", base_revision=None,
             target_modules={"model.language_model.layers.0.self_attn.o_proj":
                             (32, 32),
                             "model.language_model.layers.1.self_attn.o_proj":
                             (32, 32)}, rank=1),
    ]
    keys = {lx.zero_adapter_key(**v) for v in variants}
    assert len(keys) == len(variants) and key not in keys
    for bad in (dict(base_model="", base_revision=None, target_modules=tm,
                     rank=1),
                dict(base_model="org/base", base_revision="", target_modules=tm,
                     rank=1),
                dict(base_model="org/base", base_revision=None,
                     target_modules=tm, rank=0),
                dict(base_model="org/base", base_revision=None,
                     target_modules=tm, rank=True),
                dict(base_model="org/base", base_revision=None,
                     target_modules={}, rank=1),
                dict(base_model="org/base", base_revision=None,
                     target_modules={"m": (32,)}, rank=1),
                dict(base_model="org/base", base_revision=None,
                     target_modules={"m": (32, 0)}, rank=1)):
        with pytest.raises(lx.LoraExportError):
            lx.zero_adapter_key(**bad)


def test_export_or_reuse_zero_exports_every_oproj_layer_and_reuses(
        tiny_model, tiny_base, tmp_path):
    from safetensors.torch import load_file, save_file

    root = tmp_path / "bank"
    tm = lx.oproj_target_modules(tiny_base)
    assert tm == {"model.layers.0.self_attn.o_proj": (HIDDEN, HIDDEN),
                  "model.layers.1.self_attn.o_proj": (HIDDEN, HIDDEN)}
    key = lx.zero_adapter_key(base_model=tiny_base, base_revision=None,
                              target_modules=tm, rank=1)

    d, g, reused = lx.export_or_reuse_zero(root, base_model=tiny_base)
    assert reused is False
    assert d == lx.zero_adapter_dir_for(root.resolve(), key)
    assert d.name == f"none__lora0__{key}"
    assert g.dir == str(d) and g.name_hint == lx.ZERO_ADAPTER_NAME
    assert g.r == 1 and g.petri_zero_adapter is True
    assert g.petri_intervention_hash is None and g.petri_source_kind is None
    lx.verify_zero_adapter(d)
    lx.verify_adapter_dir(d, g)
    # the MULTI-LAYER zero export: A/B for every o_proj layer, all zero
    tensors = load_file(str(d / lx.ADAPTER_WEIGHTS))
    assert sorted(tensors) == sorted(
        f"base_model.model.model.layers.{l}.self_attn.o_proj.{w}.weight"
        for l in range(N_LAYERS) for w in ("lora_A", "lora_B"))
    for name, t in tensors.items():
        assert torch.all(t == 0) and t.dtype == torch.float32
        assert tuple(t.shape) == ((1, HIDDEN) if "lora_A" in name
                                  else (HIDDEN, 1))
    cfg = json.loads((d / lx.ADAPTER_CONFIG).read_text())
    assert cfg["petri_zero_adapter"] is True
    assert "petri_genuine_adapter" not in cfg
    assert cfg["r"] == 1 and cfg["lora_alpha"] == 1.0 == float(cfg["r"])
    # the un-deduplicated short-name list export_zero_adapter writes,
    # whose UNIQUE contents are exactly {"o_proj"}
    assert cfg["target_modules"] == ["o_proj"] * N_LAYERS
    assert set(cfg["target_modules"]) == {"o_proj"}
    # byte-identical to a direct export_zero_adapter at the default alpha
    direct = lx.export_zero_adapter(tmp_path / "direct", base_model=tiny_base,
                                    rank=1, target_modules=tm)
    for name in PEFT_FILES:
        assert (d / name).read_bytes() == (direct / name).read_bytes()

    before = _snapshot(d)
    d2, g2, reused2 = lx.export_or_reuse_zero(root, base_model=tiny_base)
    assert reused2 is True and d2 == d and g2 == g
    assert _snapshot(d) == before
    assert _entries(root) == [d.name] and _tmp_dirs(root) == []

    # another rank is another entry, still alpha == r
    d4, g4, _ = lx.export_or_reuse_zero(root, base_model=tiny_base, rank=4)
    assert d4 != d and g4.r == 4
    cfg4 = json.loads((d4 / lx.ADAPTER_CONFIG).read_text())
    assert cfg4["lora_alpha"] == 4.0
    lx.verify_zero_adapter(d4)
    assert len(_entries(root)) == 2
    for bad in (0, True, "1"):
        with pytest.raises(lx.LoraExportError, match="rank"):
            lx.export_or_reuse_zero(root, base_model=tiny_base, rank=bad)

    # a tampered zero entry (a non-zero tensor) refuses on reuse
    tensors = load_file(str(d / lx.ADAPTER_WEIGHTS))
    first = sorted(tensors)[0]
    tensors[first] = torch.ones_like(tensors[first])
    save_file(tensors, str(d / lx.ADAPTER_WEIGHTS))
    tampered = _snapshot(d)
    with pytest.raises(lx.LoraExportError,
                       match="stale or tampered bank entry; not reusing"):
        lx.export_or_reuse_zero(root, base_model=tiny_base)
    assert _snapshot(d) == tampered and _tmp_dirs(root) == []
    # genuine and zero entries share one root
    dg, _, _ = lx.export_or_reuse(_spec(), _lora_all_art(tiny_model), root,
                                  base_model=tiny_base)
    assert set(_entries(root)) == {d.name, d4.name, IH}


def test_export_or_reuse_zero_refuses_an_uncached_base(tmp_path):
    with pytest.raises(lx.LoraExportError, match="not in the local HF cache"):
        lx.export_or_reuse_zero(tmp_path / "bank",
                                base_model="TESTORG/not-cached-anywhere")
    assert not (tmp_path / "bank").exists()
