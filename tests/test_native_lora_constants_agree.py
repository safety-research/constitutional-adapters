"""Cross-workstream pins for native multi-LoRA serving.

The export + bank, enumeration, composer + attestation and table +
proxy pieces were written in parallel with NO import edges between
several of them, so every constant two of them must agree on is a
literal on each side. This module holds those equalities:

- B <-> D <-> A: ``defense_cells.NOOP_LORA_CELL_ID`` ==
  ``contracts.ROUTE_NONE_CELL + proxy.LORA_ZERO_CELL_SUFFIX`` ==
  ``lora_export.ZERO_ADAPTER_NAME``; ``NATIVE_LORA_SERVABLE_KINDS`` ==
  ``GENUINE_EXPORT_KINDS`` (strict, now that A landed ``lora_subtraction``);
  the two PEFT file names spelled in A, C and D.
- C <-> D: ``serving_attestation.engine_identity`` emits EXACTLY
  ``proxy.ENGINE_ATTESTATION_KEYS`` with per-adapter EXACTLY
  ``proxy.ENGINE_ADAPTER_KEYS`` — proven on a REAL attestation written by the
  real readiness path against the scripted server, not on a hand-typed dict —
  and ``proxy.main --serving-attestation`` binds through C's REAL reader (D's
  own test stubs it).
- A <-> C: every admitted export (``lora``, ``lora_all``,
  ``lora_subtraction`` and the banked zero adapter at rank 1 AND at another
  rank) writes ``lora_alpha == r`` and o_proj-only ``target_modules``, which
  C's compose-time fingerprint preflight requires; and A's
  ``oproj_target_modules`` equals the module-tree map C's zero-adapter test
  built by hand.
- A <-> C <-> D: one digest definition (sha256 over the raw file bytes) in
  ``lora_export.adapter_digests``, ``serving_attestation.sha256_file`` and
  the adapter block the proxy validates and re-hashes.

Everything here runs on CPU in seconds; the scripted server is the same
stand-in C's readiness tests use (GPU-verify items stay GPU-verify).
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import proxy_engine as pe
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.serving import worker_reports as wr
from sisteer.capeval.tinycap import contracts as C

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_petri_serving_gate import (  # noqa: E402
    _fake_lora_native_launch, _native_sched_report)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HIDDEN = 32
N_LAYERS = 2
BASE = "meta-llama/Llama-3.1-8B-Instruct"
PASS = "proxy_vllm+passthrough+interleaved"
LORA = "proxy_vllm+lora+interleaved"
NONE_HASH = "aa" * 8
GENUINE_CELL = "si_x__lora_all_r4"


# ---------------------------------------------------------------------------
# 1. the literals every side spells
# ---------------------------------------------------------------------------


def test_zero_adapter_cell_id_is_one_name_across_a_b_and_d():
    """B's constant, A's adapter name and D's ``none`` + suffix spelling are
    the same string."""
    assert dc.NOOP_LORA_CELL_ID == lx.ZERO_ADAPTER_NAME
    assert dc.NOOP_LORA_CELL_ID == f"{C.ROUTE_NONE_CELL}{px.LORA_ZERO_CELL_SUFFIX}"
    assert dc.NOOP_LORA_CELL_ID == f"{C.ROUTE_NONE_CELL}{pe.SCALE0_LORA_TWIN_SUFFIX}"
    assert pe.SCALE0_LORA_TWIN_SUFFIX == px.LORA_ZERO_CELL_SUFFIX == "__lora0"
    assert dc.NOOP_LORA_CELL_ID.endswith(px.LORA_ZERO_CELL_SUFFIX)


def test_native_servable_kinds_equal_the_exporters_kinds_strictly():
    """B's enumeration predicate and A's export dispatch name the SAME kinds
    (strict ``==`` now that ``lora_subtraction`` is an export kind)."""
    assert set(dc.NATIVE_LORA_SERVABLE_KINDS) == set(lx.GENUINE_EXPORT_KINDS)
    assert len(set(dc.NATIVE_LORA_SERVABLE_KINDS)) == len(dc.NATIVE_LORA_SERVABLE_KINDS)
    assert len(set(lx.GENUINE_EXPORT_KINDS)) == len(lx.GENUINE_EXPORT_KINDS)
    assert "lora_subtraction" in lx.GENUINE_EXPORT_KINDS


def test_residual_slot_constants_agree_across_the_torch_free_gate_the_exporter_and_the_worker():
    """The residual kinds, the native position
    modes, the stage-2 flag, the payload prefix / fields and the two op names are
    spelled in a torch-free module (``defense_cells``), the exporter, the worker's
    module and the attestation classifier — pinned equal here."""
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import native_residual as nr
    from sisteer.capeval.serving import serving_attestation as sa

    assert dc.RESIDUAL_SLOT_KINDS == nr.RESIDUAL_SLOT_KINDS == lx.RESIDUAL_EXPORT_KINDS
    assert set(dc.RESIDUAL_SLOT_KINDS) < set(dc.NATIVE_LORA_SERVABLE_KINDS)
    assert dc.NATIVE_POSITION_MODES == nr.POSITION_MODES
    assert dc.NATIVE_STAGE2_ARMED is nr.STAGE2_POSITION_CLASSES
    assert nr.PAYLOAD_PREFIX == sa.RESIDUAL_KEY_PREFIX == "sisteer_residual."
    assert nr.RESIDUAL_FIELDS == tuple(sa.RESIDUAL_KEY_FIELDS)
    assert nr.OP_RELU_GATED_VECTOR == la.OP_RELU_GATED_VECTOR
    assert nr.OP_ADD_VECTOR == la.OP_ADD_VECTOR
    assert nr.CONFIG_BLOCK == "petri_residual_slot"


def test_the_two_peft_file_names_agree_in_a_c_and_d():
    assert px.ADAPTER_CONFIG_FILE == lx.ADAPTER_CONFIG == wr.ADAPTER_CONFIG_FILE \
        == "adapter_config.json"
    assert px.ADAPTER_WEIGHTS_FILE == lx.ADAPTER_WEIGHTS == wr.ADAPTER_WEIGHTS_FILE \
        == "adapter_model.safetensors"


def test_worker_extension_class_path_agrees_and_resolves():
    assert vs.NATIVE_WORKER_EXTENSION_CLS == nw.WORKER_EXTENSION_CLS
    mod_name, _, cls_name = nw.WORKER_EXTENSION_CLS.rpartition(".")
    assert getattr(importlib.import_module(mod_name), cls_name) is nw.NativeWorkerExtension


def test_serving_attestation_file_name_agrees():
    assert vs.SERVING_ATTESTATION_FILE == sa.SERVING_ATTESTATION_FILE == "serving_attestation.json"
    assert px.ENGINE_KIND_LORA_NATIVE == sa.SERVING_ATTESTATION_KIND == "lora_native"


def test_the_zero_twin_resolves_through_bs_constant():
    """D's twin resolution names the zero leg by suffix; the route it lands
    on carries B's cell id."""
    routes = [
        px.RouteEntry(virtual_model="petri__base", cell_id=C.ROUTE_NONE_CELL,
                      cell_class="base", serving_identity=PASS,
                      intervention_hash=NONE_HASH),
        px.RouteEntry(virtual_model=f"petri__{GENUINE_CELL}", cell_id=GENUINE_CELL,
                      cell_class="class_a", serving_identity=LORA,
                      intervention_hash="cc" * 8),
        px.RouteEntry(virtual_model=f"petri__{dc.NOOP_LORA_CELL_ID}",
                      cell_id=dc.NOOP_LORA_CELL_ID, cell_class="control",
                      serving_identity=LORA, intervention_hash=NONE_HASH,
                      control_of=C.ROUTE_NONE_CELL),
    ]
    assert pe.scale0_twin_suffix(routes[1]) == px.LORA_ZERO_CELL_SUFFIX
    assert pe.scale0_twin(routes[1], routes).cell_id == dc.NOOP_LORA_CELL_ID
    assert pe.is_steered_route(routes[1]) and not pe.is_lens_route(routes[1])
    assert pe.scale0_twin(routes[0], routes) is None


# ---------------------------------------------------------------------------
# 2. A <-> C: a tiny llama, A's real exports, C's real preflight
# ---------------------------------------------------------------------------


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


def _multilayer_object(tiny_model, rank, seed):
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


def _ih(name: str) -> str:
    return cm.sha256_hex16(name.encode())


@pytest.fixture(scope="module")
def bank(tiny_model, tiny_base, tmp_path_factory):
    """A's REAL exports into a content-addressed bank: one adapter per admitted
    kind (``lora`` r=1, ``lora_all`` r=4, ``lora_subtraction`` at parent rank
    1 -> r=2 and parent rank 4 -> r=8) plus the zero adapter at rank 1 (the
    default) AND at rank 4 (where ``lora_alpha == r`` is A's deliberate
    choice). ``{name: (dir, AdapterDigests)}``."""
    from sisteer.extraction.multilayer import LoRASubtractionObject

    root = tmp_path_factory.mktemp("bank")
    arts = tmp_path_factory.mktemp("artifacts")
    out = {}

    g = torch.Generator().manual_seed(23)
    blob = {"kind": "lora", "A": torch.randn(HIDDEN, generator=g),
            "B": torch.randn(HIDDEN, generator=g), "layer": 0, "positions": "all"}
    d, dig, _ = lx.export_or_reuse(
        {"kind": "lora", "layer": 0, "positions": "all", "intervention_hash": _ih("lora")},
        _blob_bytes(blob), root, base_model=tiny_base, name_hint="si_x__lora_r1")
    out["si_x__lora_r1"] = (d, dig)

    obj = _multilayer_object(tiny_model, rank=4, seed=7)
    blob = {"kind": "lora_all", "layers": list(obj.layers), "rank": 4,
            "A": {str(l): obj.A[l].detach().float().cpu() for l in obj.layers},
            "B": {str(l): obj.B[l].detach().float().cpu() for l in obj.layers},
            "positions": "all", "response_header_len": None}
    d, dig, _ = lx.export_or_reuse(
        {"kind": "lora_all", "rank": 4, "positions": "all", "layers": list(obj.layers),
         "intervention_hash": _ih("lora_all")},
        _blob_bytes(blob), root, base_model=tiny_base, name_hint=GENUINE_CELL)
    out[GENUINE_CELL] = (d, dig)

    for rank in (1, 4):
        pos = _multilayer_object(tiny_model, rank=rank, seed=11)
        neg = _multilayer_object(tiny_model, rank=rank, seed=12)
        sub = LoRASubtractionObject(pos, neg, provenance={
            "kind": "lora_subtraction",
            "parent_pos": {"cell_id": "mix", "object_sha256": "a" * 64, "model_id": BASE},
            "parent_neg": {"cell_id": "ctl", "object_sha256": "b" * 64, "model_id": BASE}})
        art = arts / f"sub_r{rank}.pt"
        sub.save(art)
        name = f"si_x__lora_sub_r{rank}"
        d, dig, _ = lx.export_or_reuse(
            {"kind": "lora_subtraction", "rank": rank, "positions": "all",
             "layers": list(sub.layers), "intervention_hash": _ih(name)},
            art.read_bytes(), root, base_model=tiny_base, name_hint=name)
        out[name] = (d, dig)

    for rank in (1, 4):
        d, dig, _ = lx.export_or_reuse_zero(root / f"zero_r{rank}", base_model=tiny_base,
                                            rank=rank)
        out[f"zero_r{rank}"] = (d, dig)
    return out


def test_every_admitted_export_satisfies_cs_fingerprint_preflight(bank):
    """The A <-> C contract: ``lora_alpha == r``, o_proj-only
    ``target_modules``, fp32 factors — every dir A writes passes
    ``expected_adapter_fingerprint_bf16`` and its module keys are the
    fixture's layers."""
    for name, (d, dig) in bank.items():
        cfg = json.loads((d / lx.ADAPTER_CONFIG).read_text())
        assert cfg["lora_alpha"] == cfg["r"] == dig.r, name
        assert set(cfg["target_modules"]) == {"o_proj"}, name
        fp, keys = wr.expected_adapter_fingerprint_bf16(d)
        assert HEX64.match(fp), name
        assert all(k.endswith(".self_attn.o_proj") for k in keys), name
        assert keys == tuple(sorted(keys)), name
    # r per kind, as the plan states them
    assert bank["si_x__lora_r1"][1].r == 1
    assert bank[GENUINE_CELL][1].r == 4
    assert bank["si_x__lora_sub_r1"][1].r == 2 and bank["si_x__lora_sub_r4"][1].r == 8
    assert bank["zero_r1"][1].r == 1 and bank["zero_r4"][1].r == 4
    # the zero adapter's config is A's; the non-default rank keeps alpha == r
    z4 = json.loads((bank["zero_r4"][0] / lx.ADAPTER_CONFIG).read_text())
    assert z4["lora_alpha"] == 4.0 and z4["r"] == 4 and z4["petri_zero_adapter"] is True


def test_cs_launcher_side_expectation_covers_the_whole_bank(bank):
    """``expected_native_lora_state_for`` (C's per-name expectation) and
    ``fitted_max_lora_rank`` (C's rank-cap fit) run over A's real dirs."""
    dirs = {name: str(d) for name, (d, _) in bank.items()}
    names = tuple(sorted(dirs))
    exp = sa.expected_native_lora_state_for(names, dirs)
    assert exp["n_adapters"] == len(bank)
    for i, name in enumerate(names):
        info = exp["adapters"][name]
        assert info["lora_int_id"] == i + 1
        assert info["r"] == bank[name][1].r
        assert info["fingerprint_bf16"] == wr.expected_adapter_fingerprint_bf16(bank[name][0])[0]
    # distinct adapters, distinct fingerprints (the zero adapters differ by rank)
    fps = [exp["adapters"][n]["fingerprint_bf16"] for n in names]
    assert len(set(fps)) == len(fps)
    assert vs.fitted_max_lora_rank(Path(d) for d in dirs.values()) == 8   # max r=8 -> cap 8


def test_one_digest_definition_in_a_c_and_d(bank):
    """``adapter_digests`` (A), ``sha256_file`` (C) and the block the proxy
    validates and re-hashes (D) all read sha256 over the raw file bytes."""
    for name, (d, dig) in bank.items():
        cfg_sha = hashlib.sha256((d / lx.ADAPTER_CONFIG).read_bytes()).hexdigest()
        w_sha = hashlib.sha256((d / lx.ADAPTER_WEIGHTS).read_bytes()).hexdigest()
        assert dig.config_sha256 == sa.sha256_file(d / px.ADAPTER_CONFIG_FILE) == cfg_sha
        assert dig.weights_sha256 == sa.sha256_file(d / px.ADAPTER_WEIGHTS_FILE) == w_sha
        redig = lx.adapter_digests(d)
        assert (redig.config_sha256, redig.weights_sha256, redig.r) == (cfg_sha, w_sha, dig.r)
        zero = name.startswith("zero_")
        cell = dc.NOOP_LORA_CELL_ID if zero else name
        ih = NONE_HASH if zero else dig.petri_intervention_hash
        block = px.validated_adapter_block(
            {"name": cell, "dir": str(d), "config_sha256": dig.config_sha256,
             "weights_sha256": dig.weights_sha256, "r": dig.r,
             "petri_intervention_hash": ih, "petri_zero_adapter": dig.petri_zero_adapter,
             "petri_source_kind": dig.petri_source_kind},
            cell_id=cell, intervention_hash=ih)
        px.verify_adapter_block_on_disk(block)           # D re-hashes what A wrote
        assert block["petri_zero_adapter"] is zero


def test_oproj_target_modules_equals_the_module_tree_map_c_built_by_hand(tiny_model, tiny_base,
                                                                        tmp_path):
    """C's zero-adapter test derived ``{module: (in_features, out_features)}``
    off the tiny llama's module tree; A's ``oproj_target_modules`` must return
    exactly that, and a zero export over either map is byte-identical."""
    by_hand = {f"model.layers.{i}.self_attn.o_proj":
               (layer.self_attn.o_proj.in_features, layer.self_attn.o_proj.out_features)
               for i, layer in enumerate(tiny_model.model.layers)}
    assert len(by_hand) == N_LAYERS
    assert lx.oproj_target_modules(tiny_base) == by_hand
    a = lx.export_zero_adapter(tmp_path / "by_a", base_model=tiny_base, rank=1,
                               target_modules=lx.oproj_target_modules(tiny_base))
    c = lx.export_zero_adapter(tmp_path / "by_c", base_model=tiny_base, rank=1,
                               target_modules=by_hand)
    for f in (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS):
        assert sa.sha256_file(a / f) == sa.sha256_file(c / f), f
    assert wr.expected_adapter_fingerprint_bf16(a) == wr.expected_adapter_fingerprint_bf16(c)
    assert wr.expected_adapter_fingerprint_bf16(a)[1] == tuple(
        f"L{i}.self_attn.o_proj" for i in range(N_LAYERS))


# ---------------------------------------------------------------------------
# 3. C <-> D on a REAL attestation: A's bank -> C's readiness -> D's proxy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_attestation(bank, tmp_path_factory):
    """``serving_attestation.json`` written by the REAL lora-native readiness
    path (``vllm_server.launch_server``) against the scripted server, for A's
    real genuine ``lora_all`` adapter plus A's banked zero adapter. Yields
    ``(att_path, adapters)``; the child is terminated after the module."""
    tmp = tmp_path_factory.mktemp("att")
    adapters = {GENUINE_CELL: bank[GENUINE_CELL][0],
                dc.NOOP_LORA_CELL_ID: bank["zero_r1"][0]}
    launch = _fake_lora_native_launch(tmp, adapters=adapters, model=BASE)
    proc = vs.launch_server(launch, log_path=tmp / "srv.log",
                            ready_timeout_s=30, install_timeout_s=10)
    try:
        yield tmp / vs.SERVING_ATTESTATION_FILE, adapters
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_engine_identity_emits_exactly_the_proxys_expected_shape(real_attestation):
    """The C <-> D contract: exact key sets at both levels, lowercase 64-hex
    digests, distinct positive ids, and the proxy's validator round-trips the
    identity unchanged."""
    att_path, adapters = real_attestation
    att = sa.read_serving_attestation(att_path)           # C's strict reader
    ident = sa.engine_identity(att)
    assert set(ident) == set(px.ENGINE_ATTESTATION_KEYS)
    assert tuple(ident) == px.ENGINE_ATTESTATION_KEYS      # same order too
    assert ident["kind"] == px.ENGINE_KIND_LORA_NATIVE
    assert ident["enforce_eager"] is False
    assert isinstance(ident["cudagraph_mode"], str) and ident["cudagraph_mode"]
    assert set(ident["lora_config"]) == set(sa.LORA_CONFIG_KEYS)
    assert set(ident["adapters"]) == set(adapters)
    ids = []
    for name, entry in ident["adapters"].items():
        assert set(entry) == set(px.ENGINE_ADAPTER_KEYS), name
        assert tuple(entry) == px.ENGINE_ADAPTER_KEYS, name
        for k in ("config_sha256", "weights_sha256", "fingerprint_bf16"):
            assert HEX64.match(entry[k]), (name, k)
        assert isinstance(entry["lora_int_id"], int) and entry["lora_int_id"] > 0
        ids.append(entry["lora_int_id"])
        d = Path(adapters[name])
        assert entry["config_sha256"] == sa.sha256_file(d / lx.ADAPTER_CONFIG)
        assert entry["weights_sha256"] == sa.sha256_file(d / lx.ADAPTER_WEIGHTS)
        assert entry["fingerprint_bf16"] == wr.expected_adapter_fingerprint_bf16(d)[0]
    assert sorted(ids) == [1, 2]
    # the asserted id order is the sorted-name order
    assert ident["adapters"][dc.NOOP_LORA_CELL_ID]["lora_int_id"] == 1
    assert ident["adapters"][GENUINE_CELL]["lora_int_id"] == 2
    # the attested zero flag rides in the identity: the proxy holds it to the
    # table's block before exempting a route from the intervention-hash check
    assert ident["adapters"][dc.NOOP_LORA_CELL_ID]["petri_zero_adapter"] is True
    assert ident["adapters"][dc.NOOP_LORA_CELL_ID]["petri_intervention_hash"] is None
    assert ident["adapters"][GENUINE_CELL]["petri_zero_adapter"] is False
    assert px.validated_engine_attestation(ident) == ident


def _real_blocks(adapters, routes):
    by_cell = {r.cell_id: r for r in routes}
    blocks = {}
    for name, d in adapters.items():
        dig = lx.adapter_digests(Path(d))
        blocks[name] = {"name": name, "dir": str(Path(d).resolve()),
                        "config_sha256": dig.config_sha256,
                        "weights_sha256": dig.weights_sha256, "r": dig.r,
                        "petri_intervention_hash": by_cell[name].intervention_hash,
                        "petri_zero_adapter": dig.petri_zero_adapter,
                        "petri_source_kind": dig.petri_source_kind}
    return blocks


def _real_routes(bank):
    return [
        px.RouteEntry(virtual_model="petri__base", cell_id=C.ROUTE_NONE_CELL, cell_class="base",
                      serving_identity=PASS, intervention_hash=NONE_HASH),
        px.RouteEntry(virtual_model=f"petri__{GENUINE_CELL}", cell_id=GENUINE_CELL,
                      cell_class="class_a", serving_identity=LORA,
                      intervention_hash=bank[GENUINE_CELL][1].petri_intervention_hash),
        px.RouteEntry(virtual_model=f"petri__{dc.NOOP_LORA_CELL_ID}",
                      cell_id=dc.NOOP_LORA_CELL_ID, cell_class="control",
                      serving_identity=LORA, intervention_hash=NONE_HASH,
                      control_of=C.ROUTE_NONE_CELL),
    ]


def _write_v2_table(path, routes, blocks):
    doc = {"version": 2, "base_model": BASE, "serving_mode": "interleaved",
           "exact_replay_strategy": None, "base_revision": "r",
           "routes": [asdict(r) for r in routes], "prompted_prefixes": [],
           px.LENS_SERVED_POSITIONS_KEY: {}, px.ADAPTERS_KEY: blocks}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


def _cli(table_path, *extra):
    return ["--routing-table", str(table_path), "--upstream", "http://127.0.0.1:1",
            "--port", "1", *extra]


def test_proxy_main_binds_a_real_attestation_through_cs_real_reader(real_attestation, bank,
                                                                    tmp_path, monkeypatch):
    """D's positive ``--serving-attestation`` path with C's REAL reader (D's
    own test stubs the module): the table's blocks come from A's digests, the
    attestation from C's readiness, and the proxy starts bound to both —
    the identity it holds IS ``engine_identity(read_serving_attestation(path))``."""
    pytest.importorskip("aiohttp")
    monkeypatch.delitem(sys.modules, "sisteer.capeval.serving.serving_attestation",
                        raising=False)          # no stub may be left over
    att_path, adapters = real_attestation
    routes = _real_routes(bank)
    blocks = _real_blocks(adapters, routes)
    for name, block in blocks.items():
        px.verify_adapter_block_on_disk(px.validated_adapter_block(
            block, cell_id=name, intervention_hash=block["petri_intervention_hash"]))
    table_path = _write_v2_table(tmp_path / "routing_table.json", routes, blocks)
    table = px.load_routing_table(table_path)
    assert table["version"] == 2 and set(table[px.ADAPTERS_KEY]) == set(adapters)

    seen = {}
    monkeypatch.setattr(px, "make_app", lambda proxy: seen.setdefault("proxy", proxy) and "app")
    import aiohttp.web
    monkeypatch.setattr(aiohttp.web, "run_app",
                        lambda app, host, port: seen.setdefault("ran", (app, host, port)))
    px.main(_cli(table_path, "--serving-attestation", str(att_path)))
    proxy = seen["proxy"]
    expected = sa.engine_identity(sa.read_serving_attestation(att_path))
    assert proxy.engine_attestation == expected == px.validated_engine_attestation(expected)
    assert proxy.adapter_blocks == table[px.ADAPTERS_KEY]
    engine = proxy.models_payload()["petri_proxy"][px.ENGINE_KEY]
    assert engine == {"kind": "lora_native", "enforce_eager": False,
                      "cudagraph_mode": expected["cudagraph_mode"],
                      "residual_slot": expected["residual_slot"],       # native-psr-serving
                      "adapters": expected["adapters"]}
    assert seen["ran"] == ("app", "127.0.0.1", 1)


def test_proxy_main_refuses_a_real_attestation_the_reader_rejects(real_attestation, bank,
                                                                  tmp_path, monkeypatch):
    """Through the real reader, a record that says the engine ran eager is
    refused BEFORE the proxy exists (C's refusal, not D's)."""
    pytest.importorskip("aiohttp")
    att_path, adapters = real_attestation
    routes = _real_routes(bank)
    table_path = _write_v2_table(tmp_path / "routing_table.json", routes,
                                 _real_blocks(adapters, routes))
    eager = json.loads(att_path.read_text())
    eager["engine"]["enforce_eager"] = True
    bad = tmp_path / "eager.json"
    bad.write_text(json.dumps(eager))
    monkeypatch.setattr(px, "make_app", lambda proxy: pytest.fail("proxy was built"))
    with pytest.raises(sa.ServingAttestationError, match="enforce_eager"):
        px.main(_cli(table_path, "--serving-attestation", str(bad)))


def test_proxy_main_refuses_a_table_digest_the_real_attestation_contradicts(real_attestation,
                                                                             bank, tmp_path,
                                                                             monkeypatch):
    """A well-formed block whose weights digest differs from what the real
    attestation attests refuses to start (D's binding, fed by C's record)."""
    pytest.importorskip("aiohttp")
    att_path, adapters = real_attestation
    routes = _real_routes(bank)
    blocks = _real_blocks(adapters, routes)
    blocks[GENUINE_CELL]["weights_sha256"] = hashlib.sha256(b"not those bytes").hexdigest()
    table_path = _write_v2_table(tmp_path / "routing_table.json", routes, blocks)
    monkeypatch.setattr(px, "make_app", lambda proxy: pytest.fail("proxy was built"))
    with pytest.raises(px.ProxyError, match="weights_sha256"):
        px.main(_cli(table_path, "--serving-attestation", str(att_path)))


def test_proxy_main_refuses_a_table_intervention_hash_the_real_attestation_contradicts(
        real_attestation, bank, tmp_path, monkeypatch):
    """THE FINDING (code review 2026-09-11, P1): the standalone
    ``--serving-attestation`` path with BOTH file digests matching the real
    attestation but a regenerated route/block claiming another intervention
    hash. Before, ``engine_identity`` dropped the attested
    ``petri_intervention_hash`` and the proxy started — stamping rows with
    the declared hash over the attested object's weights. Now the real
    reader's identity carries the hash and the proxy refuses to start,
    naming both. The zero adapter (attested None, block `none`'s hash) stays
    exempt and the agreeing table still starts."""
    pytest.importorskip("aiohttp")
    att_path, adapters = real_attestation
    ident = sa.engine_identity(sa.read_serving_attestation(att_path))
    assert ident["adapters"][dc.NOOP_LORA_CELL_ID]["petri_intervention_hash"] is None
    real_ih = ident["adapters"][GENUINE_CELL]["petri_intervention_hash"]
    assert real_ih == bank[GENUINE_CELL][1].petri_intervention_hash
    other_ih = hashlib.sha256(b"another fit of the same shape").hexdigest()[:16]
    assert other_ih != real_ih
    routes = [r if r.cell_id != GENUINE_CELL
              else px.RouteEntry(**{**asdict(r), "intervention_hash": other_ih})
              for r in _real_routes(bank)]
    blocks = _real_blocks(adapters, routes)          # digests from the REAL dirs; hash from the route
    assert blocks[GENUINE_CELL]["petri_intervention_hash"] == other_ih
    assert blocks[GENUINE_CELL]["weights_sha256"] == ident["adapters"][GENUINE_CELL]["weights_sha256"]
    table_path = _write_v2_table(tmp_path / "routing_table.json", routes, blocks)
    assert px.load_routing_table(table_path)["version"] == 2       # shape-valid: the CLI loads it
    monkeypatch.setattr(px, "make_app", lambda proxy: pytest.fail("proxy was built"))
    with pytest.raises(px.ProxyError, match=f"the table binds intervention '{other_ih}' but the "
                                            f"engine attestation records .* '{real_ih}'"):
        px.main(_cli(table_path, "--serving-attestation", str(att_path)))


# ---------------------------------------------------------------------------
# 4. the pins beyond the seed
# ---------------------------------------------------------------------------


def test_the_composers_extension_string_resolves_plugin_free():
    """The STRING the composer writes into ``--worker-extension-cls``
    (``vs.NATIVE_WORKER_EXTENSION_CLS``) is resolved the way vLLM resolves
    it (module path + attribute) in a FRESH interpreter where ``vllm`` and
    ``vllm_lens`` are blocked: it yields C's class, and resolving it imports
    no ``vllm_lens*`` module and never ``lens_worker`` (whose extension
    subclasses vllm-lens's). The plugin-free contract, from the composer's
    side of the seam."""
    code = r"""
import importlib, json, sys
qualname = sys.argv[1]
for m in ("vllm", "vllm_lens", "vllm_lens._worker_ext", "vllm_lens._activations_plugin"):
    sys.modules[m] = None                       # an import of either RAISES
mod_name, _, cls_name = qualname.rpartition(".")
cls = getattr(importlib.import_module(mod_name), cls_name)
from sisteer.capeval.serving import native_worker as nw
assert cls is nw.NativeWorkerExtension, cls
leaked = sorted(m for m in sys.modules
                if sys.modules[m] is not None and (
                    m == "vllm_lens" or m.startswith("vllm_lens.")
                    or m == "vllm" or m.startswith("vllm.")
                    or m == "sisteer.capeval.serving.lens_worker"))
print(json.dumps({"leaked": leaked,
                  "attrs": [a for a in dir(cls) if not a.startswith("__")]}))
"""
    src = str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.run([sys.executable, "-c", code, vs.NATIVE_WORKER_EXTENSION_CLS],
                         capture_output=True, text=True,
                         env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    # the readiness RPC, the three LoRA routing recorder RPCs, the Nemotron backend
    # report PORTED from the lens worker and the residual
    # slot's two (the slot report and the per-request failure
    # claim, the lens worker's contract) — seven methods, deliberately; every attribute
    # sisteer_-prefixed, nothing else on the class
    assert got == {"leaked": [], "attrs": ["sisteer_arm_lora_routing_recorder",
                                           "sisteer_disarm_lora_routing_recorder",
                                           "sisteer_lora_routing_observations",
                                           "sisteer_mamba_backend_report",
                                           "sisteer_residual_slot_report",
                                           "sisteer_scheduler_config",
                                           "sisteer_take_request_failure"]}
    from sisteer.capeval.serving import mamba_backend as mb
    assert mb.RPC_METHOD == "sisteer_mamba_backend_report" and hasattr(nw.NativeWorkerExtension, mb.RPC_METHOD)


def test_lens_worker_reexports_every_moved_report_object():
    """``lens_worker.X is worker_reports.X`` for every object the plan moved
    (the class itself, so every ``pytest.raises(lw.LensWorkerError)`` in the
    tree still holds) and every ``BATCH_INVARIANT_*`` constant
    ``worker_reports`` defines."""
    from sisteer.capeval.serving import lens_worker as lw

    moved = ["LensWorkerError", "SCHEDULER_CONFIG_FIELDS", "scheduler_config_report",
             "kv_group_report", "batch_invariant_active", "batch_invariant_report"]
    batch_invariant = sorted(n for n in dir(wr) if n.startswith("BATCH_INVARIANT_"))
    assert batch_invariant, "worker_reports defines no BATCH_INVARIANT_* constant"
    for name in moved + batch_invariant:
        assert getattr(lw, name) is getattr(wr, name), name
    assert issubclass(wr.FingerprintError, lw.LensWorkerError)


def test_the_two_attestation_files_are_distinct_and_the_strict_reader_keeps_them_apart(
        real_attestation, tmp_path):
    """The lens path's ``steering_attestation.json`` and the lora-native
    ``serving_attestation.json`` are two different basenames (three lens
    readers parse the former with lens expectations), and C's strict reader
    refuses a lens-shaped record — no ``kind`` — as well as a lora-native
    record relabelled with the lens file's spelling."""
    assert vs.SERVING_ATTESTATION_FILE != vs.STEERING_ATTESTATION_FILE
    for name in (vs.SERVING_ATTESTATION_FILE, vs.STEERING_ATTESTATION_FILE):
        assert Path(name).name == name and name.endswith(".json")
    att_path, _ = real_attestation
    lens_like = tmp_path / vs.STEERING_ATTESTATION_FILE
    lens_like.write_text(json.dumps({"registry_dir": "/x", "rpc_method": "sisteer_install_steering",
                                     "n_workers": 1, "worker_reports": [{}]}))
    with pytest.raises(sa.ServingAttestationError, match="lacks required keys"):
        sa.read_serving_attestation(lens_like)
    relabelled = json.loads(att_path.read_text())
    relabelled["kind"] = "lens"
    bad = tmp_path / "relabelled.json"
    bad.write_text(json.dumps(relabelled))
    with pytest.raises(sa.ServingAttestationError, match="kind='lens'"):
        sa.read_serving_attestation(bad)


def test_the_composer_wrappers_equal_the_pure_attestation_bodies(bank, tmp_path):
    """``vs.expected_native_lora_state`` / ``vs.validate_native_lora_reports``
    are thin wrappers over
    ``serving_attestation.expected_native_lora_state_for`` /
    ``native_lora_report_problems`` — same expectation from the same launch,
    the wrapper raising exactly when the body reports a problem, carrying
    every problem line."""
    adapters = {GENUINE_CELL: bank[GENUINE_CELL][0], dc.NOOP_LORA_CELL_ID: bank["zero_r1"][0],
                "si_x__lora_sub_r4": bank["si_x__lora_sub_r4"][0]}
    launch = _fake_lora_native_launch(tmp_path, adapters=adapters, model=BASE)
    exp = vs.expected_native_lora_state(launch)
    assert exp == sa.expected_native_lora_state_for(launch.lora_modules, launch.lora_adapter_dirs)
    assert launch.lora_modules == tuple(sorted(adapters))
    rep = _native_sched_report(exp)
    assert sa.native_lora_report_problems([rep], exp) == []
    assert vs.validate_native_lora_reports([rep], exp) == [rep]
    bad = _native_sched_report(exp)
    bad["lora_adapters"][0]["rank"] = 99
    bad["lora_loaded_count"] = 2
    problems = sa.native_lora_report_problems([bad], exp)
    assert len(problems) == 2
    with pytest.raises(vs.VllmServerError) as err:
        vs.validate_native_lora_reports([bad], exp)
    assert all(p in str(err.value) for p in problems)
