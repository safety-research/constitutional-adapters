"""The adapter-swap test at every binding point.

ONE world, built by A's REAL bank exporter on a tiny llama: two GENUINE
``lora_all`` adapters of IDENTICAL shape and rank (r=4, the same two
o_proj layers, different bytes) plus the banked zero adapter — three
serving names, three ``lora_int_id``s, one version-2 routing table whose
blocks carry the digests A wrote. Every test below then performs ONE
swap that leaves every COUNT and every MULTISET unchanged (the same
three adapters, the same fingerprints, the same roots, the same module
keys and bytes — merely under the other name, id, layer or path) and
asserts the refusal, so a pass proves the BINDING of bytes to a serving
name, never a count. Three binding points, in the order the standup
meets them:

1. the standup's pre-launch re-hash — D's ``verify_adapter_block_on_disk``
   and A's ``verify_adapter_dir`` (and A's bank refuses to REUSE the
   entry): the two adapters' weight files exchanged on disk;
2. the routing table and the proxy — the loader refuses a table-level
   swap of the blocks between cells; a byte-level swap passes the
   CPU-light loader by design and is caught when the ATTESTED proxy
   holds the table's block to what the server actually loaded under that
   name (a real ``serving_attestation.json`` from C's readiness path over
   the swapped directories);
3. the worker attestation — C's per-name checks against the scripted
   server: the two genuine fingerprints permuted between their ids; one
   adapter's two same-shaped layers exchanged (the fingerprint itself
   carries layer identity); the two ``/v1/models`` roots exchanged; and
   the strict reader refusing a record whose ``id_name_map`` is permuted.

What this file cannot prove (GPU-verify; the gemma canary's job): the real
frontend's id order and the real worker's LoRA surfaces. The plan's point
(2), ``bring_up_lora`` refusing before launch on the same tampering, is the
last test here: E's REAL standup (``test_lora_serving_up``'s helpers — the
gemma campaign config over a synthetic bank view, the vllm child replaced by
the scripted lora-native upstream) banks two genuine adapters, their weight
files are exchanged, and the second bring-up refuses before any child is
spawned.

Every probe key is minted at run time (gitleaks flags hex literals).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.serving import worker_reports as wr
from sisteer.capeval.tinycap import contracts as C

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_petri_proxy import StubBackend, _TailRender  # noqa: E402
from test_petri_serving_gate import (  # noqa: E402
    _fake_lora_native_launch, _native_sched_report)

HIDDEN = 32
N_LAYERS = 2
RANK = 4
BASE = "meta-llama/Llama-3.1-8B-Instruct"
PASS = "proxy_vllm+passthrough+interleaved"
LORA = "proxy_vllm+lora+interleaved"
NONE_HASH = "aa" * 8
CELL_A = "si_a__lora_all_r4"
CELL_B = "si_b__lora_all_r4"
ZERO = dc.NOOP_LORA_CELL_ID
GENUINE = (CELL_A, CELL_B)
#: the composed --lora-modules order: sorted names -> lora_int_id = i + 1
NAMES = tuple(sorted((CELL_A, CELL_B, ZERO)))
ID_OF = {name: i + 1 for i, name in enumerate(NAMES)}


# ---------------------------------------------------------------------------
# the world: A's real exports, D's table, one honest baseline
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


def _lora_all_artifact(tiny_model, *, seed):
    """A ``lora_all`` object blob at :data:`RANK` over every layer of the
    tiny llama, with generator-seeded factors — two seeds give two
    adapters of identical shape and rank whose bytes differ."""
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    obj = MultiLayerLoRAObject(tiny_model, rank=RANK)
    g = torch.Generator().manual_seed(seed)
    for l in obj.layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g)
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.1
    blob = {"kind": "lora_all", "layers": list(obj.layers), "rank": RANK,
            "A": {str(l): obj.A[l].detach().float().cpu() for l in obj.layers},
            "B": {str(l): obj.B[l].detach().float().cpu() for l in obj.layers},
            "positions": "all", "response_header_len": None}
    return list(obj.layers), _blob_bytes(blob)


@pytest.fixture(scope="module")
def pristine(tiny_model, tiny_base, tmp_path_factory):
    """A's bank with the two genuine adapters and the zero adapter, never
    mutated: ``dirs``/``digests``/``specs``/``artifacts`` per cell. Tests
    that tamper work on a per-test COPY (:func:`world`)."""
    root = tmp_path_factory.mktemp("bank")
    dirs, digests, specs, artifacts = {}, {}, {}, {}
    for cell, seed in ((CELL_A, 7), (CELL_B, 8)):
        layers, art = _lora_all_artifact(tiny_model, seed=seed)
        spec = {"kind": "lora_all", "rank": RANK, "positions": "all",
                "layers": layers, "intervention_hash": cm.sha256_hex16(cell.encode())}
        d, dig, reused = lx.export_or_reuse(spec, art, root, base_model=tiny_base,
                                            name_hint=cell)
        assert not reused
        dirs[cell], digests[cell], specs[cell], artifacts[cell] = d, dig, spec, art
    d, dig, reused = lx.export_or_reuse_zero(root / "zero", base_model=tiny_base, rank=1)
    assert not reused
    dirs[ZERO], digests[ZERO] = d, dig
    return SimpleNamespace(root=root, dirs=dirs, digests=digests, specs=specs,
                           artifacts=artifacts, base=tiny_base)


def _routes(digests):
    return [
        px.RouteEntry(virtual_model="petri__base", cell_id=C.ROUTE_NONE_CELL,
                      cell_class="base", serving_identity=PASS, intervention_hash=NONE_HASH),
        px.RouteEntry(virtual_model=f"petri__{CELL_A}", cell_id=CELL_A, cell_class="class_a",
                      serving_identity=LORA,
                      intervention_hash=digests[CELL_A].petri_intervention_hash),
        px.RouteEntry(virtual_model=f"petri__{CELL_B}", cell_id=CELL_B, cell_class="class_b",
                      serving_identity=LORA,
                      intervention_hash=digests[CELL_B].petri_intervention_hash),
        px.RouteEntry(virtual_model=f"petri__{ZERO}", cell_id=ZERO, cell_class="control",
                      serving_identity=LORA, intervention_hash=NONE_HASH,
                      control_of=C.ROUTE_NONE_CELL),
    ]


def _blocks(dirs, digests, routes):
    by_cell = {r.cell_id: r for r in routes}
    return {cell: {"name": cell, "dir": str(Path(dirs[cell]).resolve()),
                   "config_sha256": digests[cell].config_sha256,
                   "weights_sha256": digests[cell].weights_sha256,
                   "r": digests[cell].r,
                   "petri_intervention_hash": by_cell[cell].intervention_hash,
                   "petri_zero_adapter": digests[cell].petri_zero_adapter,
                   "petri_source_kind": digests[cell].petri_source_kind}
            for cell in NAMES}


def _write_v2_table(path, routes, blocks):
    doc = {"version": 2, "base_model": BASE, "serving_mode": "interleaved",
           "exact_replay_strategy": None, "base_revision": "r",
           "routes": [asdict(r) for r in routes], "prompted_prefixes": [],
           px.LENS_SERVED_POSITIONS_KEY: {}, px.ADAPTERS_KEY: blocks}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


@pytest.fixture
def world(pristine, tmp_path):
    """A per-test COPY of the pristine bank (so a swap never leaks into
    the next test) with its routes, blocks and version-2 table. The
    digests are the pristine ones — content identical, ``dir`` moved —
    exactly the record a standup holds from its own export."""
    root = tmp_path / "bank"
    shutil.copytree(pristine.root, root)
    dirs = {cell: root / Path(d).relative_to(pristine.root) for cell, d in pristine.dirs.items()}
    for cell, d in dirs.items():
        assert (d / lx.ADAPTER_CONFIG).is_file() and (d / lx.ADAPTER_WEIGHTS).is_file(), cell
    routes = _routes(pristine.digests)
    blocks = _blocks(dirs, pristine.digests, routes)
    table = _write_v2_table(tmp_path / "routing_table.json", routes, blocks)
    return SimpleNamespace(root=root, dirs=dirs, digests=pristine.digests,
                           specs=pristine.specs, artifacts=pristine.artifacts,
                           base=pristine.base, routes=routes, blocks=blocks, table=table,
                           launch_dirs={cell: str(d.resolve()) for cell, d in dirs.items()})


def _fingerprint(d):
    return wr.expected_adapter_fingerprint_bf16(d)[0]


def _snapshot(d):
    return {f.name: f.read_bytes() for f in Path(d).iterdir() if f.is_file()}


def _swap_weight_files(dir_a, dir_b):
    """Exchange the two adapters' ``adapter_model.safetensors`` — the
    on-disk multiset of weight bytes is unchanged; each dir now holds the
    OTHER adapter's factors under its own config."""
    a, b = Path(dir_a) / lx.ADAPTER_WEIGHTS, Path(dir_b) / lx.ADAPTER_WEIGHTS
    a_bytes, b_bytes = a.read_bytes(), b.read_bytes()
    a.write_bytes(b_bytes)
    b.write_bytes(a_bytes)


def _port_free(port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                time.sleep(0.2)
        except OSError:
            return True
    return False


def _launch(tmp_path, launch):
    return vs.launch_server(launch, log_path=tmp_path / "srv.log",
                            ready_timeout_s=30, install_timeout_s=10)


def _refused_launch(tmp_path, launch, match):
    """``launch_server`` against the scripted server must refuse with
    ``match``, leave NO attestation of either kind, and shut the child."""
    with pytest.raises(vs.VllmServerError, match=match) as err:
        _launch(tmp_path, launch)
    assert not (tmp_path / vs.SERVING_ATTESTATION_FILE).exists()
    assert not (tmp_path / vs.STEERING_ATTESTATION_FILE).exists()
    assert _port_free(launch.port), "child not shut down"
    return str(err.value)


def _proxy(routes, blocks, attestation=None):
    return px.RoutingProxy(routes, StubBackend(), base_model=BASE, serving_mode="interleaved",
                           adapter_blocks=blocks, engine_attestation=attestation,
                           agent_loop_render=_TailRender())


@pytest.fixture(scope="module")
def honest_attestation(pristine, tmp_path_factory):
    """A real ``serving_attestation.json`` from C's readiness path over the
    PRISTINE bank (the scripted server agrees with the launcher by
    construction). Module-scoped: one launch serves every test that needs
    an honest record; the child is terminated after the module."""
    tmp = tmp_path_factory.mktemp("honest")
    adapters = {cell: pristine.dirs[cell] for cell in NAMES}
    launch = _fake_lora_native_launch(tmp, adapters=adapters, model=BASE)
    proc = _launch(tmp, launch)
    try:
        yield tmp / vs.SERVING_ATTESTATION_FILE
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_the_world_is_honest_before_any_swap(world, honest_attestation):
    """The baseline every swap below is one step from: two genuine
    adapters of the SAME rank, the SAME module keys and the SAME tensor
    shapes but DIFFERENT bytes and fingerprints, a zero adapter distinct
    from both; every binding point PASSES on the untouched world."""
    from safetensors.torch import load_file

    dig_a, dig_b, dig_z = (world.digests[c] for c in (CELL_A, CELL_B, ZERO))
    assert dig_a.r == dig_b.r == RANK and dig_z.r == 1
    assert dig_a.petri_source_kind == dig_b.petri_source_kind == "lora_all"
    assert dig_z.petri_zero_adapter is True and dig_z.petri_source_kind is None
    assert not dig_a.petri_zero_adapter and not dig_b.petri_zero_adapter
    shapes = {c: {k: tuple(t.shape) for k, t in
                  load_file(str(world.dirs[c] / lx.ADAPTER_WEIGHTS)).items()}
              for c in GENUINE}
    assert shapes[CELL_A] == shapes[CELL_B]            # identical shape, same keys
    assert len(shapes[CELL_A]) == 2 * N_LAYERS
    assert dig_a.weights_sha256 != dig_b.weights_sha256  # different bytes
    fps = {c: wr.expected_adapter_fingerprint_bf16(world.dirs[c]) for c in NAMES}
    assert fps[CELL_A][1] == fps[CELL_B][1] == tuple(
        f"L{i}.self_attn.o_proj" for i in range(N_LAYERS))
    assert len({fp for fp, _ in fps.values()}) == 3
    # binding point 1: the re-hash, both spellings
    for cell in NAMES:
        block = px.validated_adapter_block(
            world.blocks[cell], cell_id=cell,
            intervention_hash=world.blocks[cell]["petri_intervention_hash"])
        px.verify_adapter_block_on_disk(block)
        lx.verify_adapter_dir(world.dirs[cell], world.digests[cell])
    # binding point 2: the table loads, the proxy binds the real record
    table = px.load_routing_table(world.table)
    assert table["version"] == 2 and set(table[px.ADAPTERS_KEY]) == set(NAMES)
    att = sa.read_serving_attestation(honest_attestation)
    ident = sa.engine_identity(att)
    assert {n: e["lora_int_id"] for n, e in ident["adapters"].items()} == ID_OF
    proxy = _proxy(world.routes, table[px.ADAPTERS_KEY], ident)
    assert proxy.engine_attestation == ident
    # binding point 3: the launcher's expectation agrees with an honest report
    exp = sa.expected_native_lora_state_for(NAMES, world.launch_dirs)
    assert {n: i["lora_int_id"] for n, i in exp["adapters"].items()} == ID_OF
    assert sa.native_lora_report_problems([_native_sched_report(exp)], exp) == []
    body = {"data": [{"id": BASE}] + [{"id": n, "root": world.launch_dirs[n]} for n in NAMES]}
    assert sa.native_models_problems(body, model=BASE, adapter_dirs=world.launch_dirs) == []


# ---------------------------------------------------------------------------
# 1. the standup's re-hash: weight files exchanged on disk
# ---------------------------------------------------------------------------


def test_exchanged_weight_files_refuse_at_the_rehash_and_the_bank(world):
    """A's and B's ``adapter_model.safetensors`` exchanged: the on-disk
    fingerprint MULTISET is unchanged (each dir now yields the other's),
    every config still agrees with its block (``r``, the zero flag, the
    hash, the kind) — and BOTH re-hashes refuse naming ``weights_sha256``
    and nothing else, the bank refuses to reuse either entry, and the zero
    adapter beside them is untouched. Nothing is 'repaired'."""
    fp_before = {c: _fingerprint(world.dirs[c]) for c in GENUINE}
    _swap_weight_files(world.dirs[CELL_A], world.dirs[CELL_B])
    fp_after = {c: _fingerprint(world.dirs[c]) for c in GENUINE}
    assert sorted(fp_after.values()) == sorted(fp_before.values())    # multiset unchanged
    assert fp_after[CELL_A] == fp_before[CELL_B] and fp_after[CELL_B] == fp_before[CELL_A]
    snapshots = {c: _snapshot(world.dirs[c]) for c in NAMES}

    for cell in GENUINE:
        other = CELL_B if cell == CELL_A else CELL_A
        # D's re-hash: the config still hashes, only the weights disagree
        block = px.validated_adapter_block(
            world.blocks[cell], cell_id=cell,
            intervention_hash=world.blocks[cell]["petri_intervention_hash"])
        with pytest.raises(px.ProxyError, match="weights_sha256") as err:
            px.verify_adapter_block_on_disk(block)
        msg = str(err.value)
        assert "config_sha256" not in msg and world.digests[other].weights_sha256 in msg
        assert "not the one the table binds" in msg
        # A's re-hash: exactly the weights field, no other content field
        with pytest.raises(lx.LoraExportError, match=r"\['weights_sha256'\]") as err:
            lx.verify_adapter_dir(world.dirs[cell], world.digests[cell])
        assert "tampered" in str(err.value)
        # A's bank: the entry at this address is not reused, and not rewritten
        with pytest.raises(lx.LoraExportError,
                           match="stale or tampered bank entry; not reusing") as err:
            lx.export_or_reuse(world.specs[cell], world.artifacts[cell], world.root,
                               base_model=world.base, name_hint=cell)
        assert "weights_sha256" in str(err.value)
    # the zero adapter is untouched and still verifies; nothing was written back
    zero_block = px.validated_adapter_block(world.blocks[ZERO], cell_id=ZERO,
                                            intervention_hash=NONE_HASH)
    px.verify_adapter_block_on_disk(zero_block)
    lx.verify_adapter_dir(world.dirs[ZERO], world.digests[ZERO])
    assert {c: _snapshot(world.dirs[c]) for c in NAMES} == snapshots
    assert not [p for p in world.root.iterdir() if p.name.startswith(".tmp-")]


def test_a_block_pointing_at_the_other_adapters_dir_is_caught_by_as_verify(world):
    """The table-level cousin of the byte swap: block A points at B's
    directory WITH B's digests (so the bytes at ``dir`` agree with the
    block). A's ``verify_adapter_dir``, holding the export's OWN record,
    refuses on ``petri_intervention_hash`` — the content field that names
    which object the bytes came from."""
    with pytest.raises(lx.LoraExportError, match=r"petri_intervention_hash") as err:
        lx.verify_adapter_dir(world.dirs[CELL_B], world.digests[CELL_A])
    assert "weights_sha256" in str(err.value) and "config_sha256" in str(err.value)


def test_a_block_pointing_at_the_other_adapters_dir_is_refused_by_ds_rehash(world):
    """D's re-hash reads the config it already opens and holds its
    ``petri_intervention_hash`` to the block's (a genuine block; the zero
    adapter's config carries none) and ``petri_source_kind`` to every
    block's, refusing the pointer swap by name — closing the gap where a
    block points cell A at B's dir with B's digests re-hashed clean and
    agreed on r and the zero flag. Was a strict xfail until then."""
    pointed = dict(world.blocks[CELL_A], dir=world.blocks[CELL_B]["dir"],
                   config_sha256=world.blocks[CELL_B]["config_sha256"],
                   weights_sha256=world.blocks[CELL_B]["weights_sha256"])
    block = px.validated_adapter_block(
        pointed, cell_id=CELL_A, intervention_hash=world.blocks[CELL_A]["petri_intervention_hash"])
    with pytest.raises(px.ProxyError, match="petri_intervention_hash"):
        px.verify_adapter_block_on_disk(block)


# ---------------------------------------------------------------------------
# 2. the table and the proxy
# ---------------------------------------------------------------------------


def test_blocks_exchanged_between_cells_refuse_at_load(world, tmp_path):
    """A table whose two genuine blocks sit under each other's cell refuses
    to LOAD (``name`` is the route's cell id); swapping only their
    ``petri_intervention_hash`` refuses too (the route's hash). Both keep
    the block count and every digest — the loader binds a block to its
    route by name and hash, and that is what the swap breaks."""
    doc = json.loads(world.table.read_text())
    swapped = dict(doc[px.ADAPTERS_KEY])
    swapped[CELL_A], swapped[CELL_B] = doc[px.ADAPTERS_KEY][CELL_B], doc[px.ADAPTERS_KEY][CELL_A]
    assert sorted(b["weights_sha256"] for b in swapped.values()) == sorted(
        b["weights_sha256"] for b in doc[px.ADAPTERS_KEY].values())
    path = tmp_path / "swapped_blocks.json"
    path.write_text(json.dumps(dict(doc, **{px.ADAPTERS_KEY: swapped}), sort_keys=True))
    with pytest.raises(px.ProxyError, match=rf"\[{CELL_A!r}\]\.name is {CELL_B!r}"):
        px.load_routing_table(path)
    # the proxy given the same map refuses identically (one validator)
    with pytest.raises(px.ProxyError, match=r"\.name is"):
        _proxy(world.routes, swapped)
    hashes = {c: dict(doc[px.ADAPTERS_KEY][c]) for c in GENUINE}
    hashes[CELL_A]["petri_intervention_hash"], hashes[CELL_B]["petri_intervention_hash"] = (
        doc[px.ADAPTERS_KEY][CELL_B]["petri_intervention_hash"],
        doc[px.ADAPTERS_KEY][CELL_A]["petri_intervention_hash"])
    path.write_text(json.dumps(dict(doc, **{px.ADAPTERS_KEY: dict(doc[px.ADAPTERS_KEY], **hashes)}),
                               sort_keys=True))
    with pytest.raises(px.ProxyError, match="exported from a different object"):
        px.load_routing_table(path)


def test_exchanged_weight_files_pass_the_loader_and_refuse_at_the_attested_proxy(world,
                                                                                tmp_path):
    """The byte-level swap the loader CANNOT see (it never touches disk —
    the proxy stays CPU-light) is caught one step later: C's readiness
    attests the bytes ACTUALLY loaded under each name, and the proxy holds
    the table's block to that record. Over the exchanged files, the real
    attestation records B's weights under A's name and A's under B's (count
    and digest multiset unchanged), and
    ``RoutingProxy(adapter_blocks=table, engine_attestation=...)`` refuses
    to start naming the route and the field."""
    _swap_weight_files(world.dirs[CELL_A], world.dirs[CELL_B])
    table = px.load_routing_table(world.table)              # shape only: loads
    assert set(table[px.ADAPTERS_KEY]) == set(NAMES)
    unattested = _proxy(world.routes, table[px.ADAPTERS_KEY])   # no disk, no attestation: builds
    assert unattested.engine_attestation is None
    # the launcher attests what is on disk under each name
    launch = _fake_lora_native_launch(tmp_path, adapters=dict(world.dirs), model=BASE)
    proc = _launch(tmp_path, launch)
    try:
        att = sa.read_serving_attestation(tmp_path / vs.SERVING_ATTESTATION_FILE)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    ident = sa.engine_identity(att)
    assert ident["adapters"][CELL_A]["weights_sha256"] == world.digests[CELL_B].weights_sha256
    assert ident["adapters"][CELL_B]["weights_sha256"] == world.digests[CELL_A].weights_sha256
    assert ident["adapters"][CELL_A]["config_sha256"] == world.digests[CELL_A].config_sha256
    assert sorted(e["weights_sha256"] for e in ident["adapters"].values()) == sorted(
        b["weights_sha256"] for b in table[px.ADAPTERS_KEY].values())      # multiset unchanged
    assert {n: e["lora_int_id"] for n, e in ident["adapters"].items()} == ID_OF
    with pytest.raises(px.ProxyError,
                       match=rf"lora route {CELL_A!r}: adapter block weights_sha256 "
                             rf"{world.digests[CELL_A].weights_sha256} != the engine "
                             rf"attestation's {world.digests[CELL_B].weights_sha256}") as err:
        _proxy(world.routes, table[px.ADAPTERS_KEY], ident)
    assert "not the one the table binds; refusing to start" in str(err.value)
    # ... and through the CLI, with C's real reader
    pytest.importorskip("aiohttp")
    with pytest.raises(px.ProxyError, match="adapter block weights_sha256"):
        px.main(["--routing-table", str(world.table), "--upstream", "http://127.0.0.1:1",
                 "--port", "1", "--serving-attestation",
                 str(tmp_path / vs.SERVING_ATTESTATION_FILE)])


# ---------------------------------------------------------------------------
# 3. the worker attestation, against the scripted server
# ---------------------------------------------------------------------------


def _report(world, mutate=None, **over):
    exp = sa.expected_native_lora_state_for(NAMES, world.launch_dirs)
    rep = _native_sched_report(exp, **over)
    if mutate is not None:
        mutate(rep["lora_adapters"])
    return exp, rep


def _by_id(adapters):
    return {a["id"]: a for a in adapters}


def _assert_counts_unchanged(rep, exp):
    """The report's counts and id set are the expectation's, and the
    fingerprint multiset is the expectation's too."""
    assert rep["lora_loaded_count"] == exp["n_adapters"] == len(NAMES)
    assert sorted(a["id"] for a in rep["lora_adapters"]) == sorted(ID_OF.values())
    assert sorted(a["fingerprint_bf16"] for a in rep["lora_adapters"]) == sorted(
        i["fingerprint_bf16"] for i in exp["adapters"].values())
    assert sorted(a["rank"] for a in rep["lora_adapters"]) == sorted(
        i["r"] for i in exp["adapters"].values())
    for a in rep["lora_adapters"]:
        assert a["tensor_dtype"] == sa.EXPECTED_TENSOR_DTYPE
        assert a["n_modules"] == N_LAYERS


def test_genuine_fingerprints_permuted_between_ids_refuse_naming_both(world, tmp_path):
    """The two genuine adapters' fingerprints exchanged between their
    ``lora_int_id``s (the zero adapter's untouched): count, id set,
    fingerprint multiset, ranks and module keys all unchanged. The
    per-name check names BOTH names and BOTH ids and says whose
    fingerprint landed where; the wrapper raises; the launch refuses
    readiness with no attestation left behind."""
    def permute(ads):
        a, b = _by_id(ads)[ID_OF[CELL_A]], _by_id(ads)[ID_OF[CELL_B]]
        a["fingerprint_bf16"], b["fingerprint_bf16"] = b["fingerprint_bf16"], a["fingerprint_bf16"]

    exp, rep = _report(world, permute)
    _assert_counts_unchanged(rep, exp)
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 2, problems
    for cell, other in ((CELL_A, CELL_B), (CELL_B, CELL_A)):
        mine = [p for p in problems if p.startswith(
            f"worker 0: adapter {cell!r} (lora_int_id {ID_OF[cell]})")]
        assert len(mine) == 1, problems
        # adapter identity is the PAIR (carrier fingerprint, residual fingerprint) since
        # native-psr-serving step 3 (attestation schema 2): the permutation names both
        assert (f"that (fingerprint_bf16, residual_fingerprint) pair is the one expected for "
                f"adapter {other!r} (lora_int_id {ID_OF[other]})") in mine[0]
        assert "the worker's ids do not bind to the composed names" in mine[0]
    assert not any(ZERO in p for p in problems)
    with pytest.raises(vs.VllmServerError, match="do not bind to the composed names") as err:
        vs.validate_native_lora_reports([rep], exp)
    assert all(p in str(err.value) for p in problems)
    launch = _fake_lora_native_launch(tmp_path, adapters=dict(world.dirs), model=BASE,
                                      scheduler_results=[rep])
    msg = _refused_launch(tmp_path, launch,
                          rf"expected for adapter {CELL_B!r} \(lora_int_id {ID_OF[CELL_B]}\)")
    assert f"adapter {CELL_B!r} (lora_int_id {ID_OF[CELL_B]}) reports" in msg


def _module_weights(adapter_dir):
    """A's exported factors as C's worker-side ``ModuleWeights`` (one fp32 ->
    bf16 cast, the launcher's contract), keyed by PEFT module name."""
    from safetensors.torch import load_file

    cfg = wr.read_adapter_config(adapter_dir)
    grouped = {}
    for key, t in load_file(str(Path(adapter_dir) / lx.ADAPTER_WEIGHTS)).items():
        module, which = key.rsplit(".lora_", 1)
        grouped.setdefault(module, {})[which[0]] = t.to(torch.bfloat16)
    return {m: wr.ModuleWeights(lora_a=ab["A"], lora_b=ab["B"], rank=int(cfg["r"]),
                                lora_alpha=float(cfg["lora_alpha"]), is_packed=False)
            for m, ab in grouped.items()}


def test_two_layers_exchanged_inside_one_adapter_refuse_readiness(world, tmp_path):
    """Adapter A with its two same-shaped layers' tensors EXCHANGED: the
    same bytes, the same module keys, the same rank and count — and a
    DIFFERENT fingerprint, because the canonical layer key is inside the
    hash. A worker reporting that fingerprint under A's id is refused by
    name; no other adapter is named; the launch refuses readiness."""
    modules = _module_weights(world.dirs[CELL_A])
    fp, keys = wr.adapter_fingerprint_bf16(modules)
    assert (fp, keys) == wr.expected_adapter_fingerprint_bf16(world.dirs[CELL_A])
    (m0, w0), (m1, w1) = sorted(modules.items())
    assert wr.canonical_module_key(m0) == "L0.self_attn.o_proj"
    assert wr.canonical_module_key(m1) == "L1.self_attn.o_proj"
    assert w0.lora_a.shape == w1.lora_a.shape and w0.lora_b.shape == w1.lora_b.shape
    swapped_fp, swapped_keys = wr.adapter_fingerprint_bf16({m0: w1, m1: w0})
    assert swapped_keys == keys                         # module keys unchanged
    assert swapped_fp != fp                             # layer identity is in the hash
    bytes_multiset = sorted(wr.bf16_bytes(t) for w in (w0, w1) for t in (w.lora_a, w.lora_b))
    assert sorted(wr.bf16_bytes(t) for w in (w1, w0) for t in (w.lora_a, w.lora_b)) == bytes_multiset

    def swap_layers(ads):
        _by_id(ads)[ID_OF[CELL_A]]["fingerprint_bf16"] = swapped_fp

    exp, rep = _report(world, swap_layers)
    assert rep["lora_loaded_count"] == exp["n_adapters"]
    assert sorted(a["id"] for a in rep["lora_adapters"]) == sorted(ID_OF.values())
    assert [a["module_keys"] for a in rep["lora_adapters"]] == [
        list(exp["adapters"][n]["module_keys"]) for n in NAMES]
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 1, problems
    assert problems[0].startswith(f"worker 0: adapter {CELL_A!r} (lora_int_id {ID_OF[CELL_A]})")
    assert swapped_fp in problems[0] and fp in problems[0]
    assert "expected for adapter" not in problems[0]     # nobody else's fingerprint
    assert CELL_B not in problems[0] and ZERO not in problems[0]
    launch = _fake_lora_native_launch(tmp_path, adapters=dict(world.dirs), model=BASE,
                                      scheduler_results=[rep])
    _refused_launch(tmp_path, launch,
                    rf"adapter {CELL_A!r} \(lora_int_id {ID_OF[CELL_A]}\) reports "
                    rf"fingerprint_bf16 {swapped_fp!r}")


def test_models_roots_exchanged_between_names_refuse_readiness(world, tmp_path):
    """``/v1/models`` with A's and B's ``root`` paths exchanged: the same
    cards, the same root multiset, every name listed exactly once. The
    frontend's own name -> path binding is checked PER NAME, so both are
    refused and the launch never reaches the worker RPC."""
    healthy = [{"id": BASE}] + [{"id": n, "root": world.launch_dirs[n]} for n in NAMES]
    swapped = [dict(c) for c in healthy]
    cards = {c.get("id"): c for c in swapped}
    cards[CELL_A]["root"], cards[CELL_B]["root"] = cards[CELL_B]["root"], cards[CELL_A]["root"]
    assert len(swapped) == len(healthy) == len(NAMES) + 1
    assert sorted(c["root"] for c in swapped if "root" in c) == sorted(
        c["root"] for c in healthy if "root" in c)
    problems = sa.native_models_problems({"data": swapped}, model=BASE,
                                         adapter_dirs=world.launch_dirs)
    assert len(problems) == 2, problems
    for cell, other in ((CELL_A, CELL_B), (CELL_B, CELL_A)):
        mine = [p for p in problems if p.startswith(f"adapter {cell!r}: /v1/models root")]
        assert len(mine) == 1 and world.launch_dirs[other] in mine[0], problems
        assert "binding does not match the composed one" in mine[0]
    launch = _fake_lora_native_launch(tmp_path, adapters=dict(world.dirs), model=BASE,
                                      models_data=swapped)
    msg = _refused_launch(tmp_path, launch, "binding does not match")
    assert CELL_A in msg and CELL_B in msg and ZERO not in msg


def test_a_permuted_id_name_map_in_the_record_is_refused_by_the_reader(honest_attestation,
                                                                       tmp_path):
    """The written record's ``id_name_map`` permuted between the two genuine
    names (the same ids, the same names): the strict reader refuses it
    against the adapters' asserted ``lora_int_id``s — a record cannot
    re-bind names to ids after the fact."""
    att = json.loads(honest_attestation.read_text())
    a, b = str(ID_OF[CELL_A]), str(ID_OF[CELL_B])
    assert att["id_name_map"][a] == CELL_A and att["id_name_map"][b] == CELL_B
    att["id_name_map"][a], att["id_name_map"][b] = CELL_B, CELL_A
    assert sorted(att["id_name_map"].values()) == sorted(NAMES)
    bad = tmp_path / "permuted_ids.json"
    bad.write_text(json.dumps(att))
    with pytest.raises(sa.ServingAttestationError,
                       match="id_name_map .* disagrees with the adapters' lora_int_id"):
        sa.read_serving_attestation(bad)


# ---------------------------------------------------------------------------
# 4. the standup: E's REAL bring_up_lora refuses BEFORE launch on the swap
# ---------------------------------------------------------------------------


def test_bring_up_lora_refuses_before_launch_when_two_banked_adapters_weight_files_are_exchanged(
        tmp_path, monkeypatch):
    """On the REAL ``bring_up_lora`` (the helpers of
    ``test_lora_serving_up``: the gemma campaign config over a synthetic bank
    VIEW, the vllm child replaced by the scripted lora-native upstream; skips
    when the pinned gemma config is not cached). A first standup banks TWO
    genuine gemma ``lora_all`` adapters — the certification's own cell and its
    ``constv2-mix`` sibling, identical shape and rank, different bytes — plus
    the zero adapter, and passes its five controls. Their weight files are
    then exchanged (:func:`_swap_weight_files`: count and on-disk multiset
    unchanged). The second bring-up over the SAME bank refuses at the bank
    (each entry is re-hashed against its own record and a fresh export of
    the same object bytes — A's ``_reuse_verified``) BEFORE any launch is
    composed, names ``weights_sha256`` and the cell, and leaves no
    attestation, no endpoint and no proxy behind."""
    import test_lora_serving_up as E   # E's fixtures as plain helpers

    lsu = E._load_lsu()
    cfg_path, man_path, cfg = E._campaign("gemma4e4b")
    E.cached_or_skip(cfg)
    sibling = "constv2-mix-full__lora_r4_ce"
    assert sibling in cfg["explicit_cells"] and E.GEMMA_CELL in cfg["explicit_cells"]
    sibling_h = E.ra.harness_cell_id(sibling)
    chat = E.default_chat(cfg["model"])
    chat[sibling_h] = {"ids": [41, 42, 43, 44, 45], "text": "another steered reply"}
    cells = (E.GEMMA_CELL, sibling, E.PROMPTED)
    real = (E.GEMMA_CELL, sibling)
    # the bank VIEW: E's staging seeds every real cell alike, so the sibling is
    # re-staged under its own seed — same shape and rank, DIFFERENT bytes
    view, shas = E.fake_bank(tmp_path, cfg, real_cells=real)
    shas[sibling] = E.stage_real_lora_all(view, sibling, cfg["model"], cfg["model_revision"], seed=11)
    bank_dir = tmp_path / "adapter_bank"

    # 1. the honest world: two genuine adapters + the zero adapter banked and served
    su, cap = E.bring_up_fake_standup(lsu, tmp_path, monkeypatch, cells=cells, real_cells=real,
                                      bank=(view, shas), extra={"adapter_bank": bank_dir},
                                      fake_kwargs={"chat": chat, "genuine_name": E.GEMMA_HCELL})
    try:
        assert set(su.lora_cells) == {E.GEMMA_HCELL, sibling_h}
        assert su.positive_controls["passed"] is True
        dirs = {c: Path(su.adapters[c]["dir"]) for c in su.lora_cells}
        digests_before = {c: lx.adapter_digests(d) for c, d in dirs.items()}
        assert digests_before[E.GEMMA_HCELL].weights_sha256 != digests_before[sibling_h].weights_sha256
        assert (digests_before[E.GEMMA_HCELL].r, digests_before[sibling_h].r) == (4, 4)
    finally:
        su.tear_down()

    # 2. the swap: the two weight files exchanged in place
    _swap_weight_files(dirs[E.GEMMA_HCELL], dirs[sibling_h])
    after = {c: lx.adapter_digests(d) for c, d in dirs.items()}
    assert after[E.GEMMA_HCELL].weights_sha256 == digests_before[sibling_h].weights_sha256
    assert after[sibling_h].weights_sha256 == digests_before[E.GEMMA_HCELL].weights_sha256
    assert {d.config_sha256 for d in after.values()} == {d.config_sha256 for d in digests_before.values()}

    # 3. the second bring-up over the SAME bank: refused before any launch
    composed = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: composed.append(1))
    with pytest.raises(lsu.StandupRefused, match="weights_sha256") as ei:
        E.bring_up_fake_standup(lsu, tmp_path / "again", monkeypatch, cells=cells, real_cells=real,
                                bank=(view, shas), extra={"adapter_bank": bank_dir},
                                fake_kwargs={"chat": chat, "genuine_name": E.GEMMA_HCELL})
    msg = str(ei.value)
    assert ei.value.rc == 2 and composed == []
    assert "stale or tampered bank entry" in msg and "adapter export refused for cell" in msg
    assert any(c in msg for c in (E.GEMMA_HCELL, sibling_h))         # the cell is named
    out2 = tmp_path / "again" / "out"
    assert not (out2 / vs.SERVING_ATTESTATION_FILE).exists()
    assert not (out2 / "endpoint.json").exists()
    assert not (out2 / "routing_table.json").exists()                   # refused before the table
    # the bank was not written over: the swapped bytes are still there, unrepaired
    assert lx.adapter_digests(dirs[E.GEMMA_HCELL]).weights_sha256 == digests_before[sibling_h].weights_sha256
