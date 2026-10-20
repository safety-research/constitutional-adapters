"""Version-2 routing tables and the per-lora-route ADAPTER BLOCK.


What must hold, CPU-only:

1. GOLDEN BYTE-IDENTITY IN BOTH DIRECTIONS. ``tests/fixtures/
   routing_table_v1_golden.json`` is the table the PRE-CHANGE exporter at
   9486b825 wrote for ``test_petri_proxy``'s fixture cell set (lora routes
   included; the artifact bytes, not their paths, enter the hashes, so the
   bytes are independent of the temp dir — verified at capture). (i) The new
   exporter's ``adapter_blocks=None`` output equals it byte for byte; (ii)
   every route dict carries EXACTLY the seven pre-change ``RouteEntry`` field
   names, so the OLD route schema constructs from it; (iii) a VERBATIM copy of
   the pre-change loader reads the new version-1 output and yields the same
   routes; (iv) the new loader reads the golden with ``version == 1`` and
   ``adapters is None``; (v) the old loader REFUSES a version-2 table.
2. VERSION 2 round-trips with blocks; a lora route without a block, a block
   for a non-lora / unrouted cell, a wrong name, a wrong hash, an unknown key,
   a wrong zero flag, version 3 all refuse; a version-1 document smuggling an
   adapters map refuses.
3. ``verify_adapter_block_on_disk`` refuses a tampered file.
4. ``RoutingProxy(adapter_blocks=, engine_attestation=)``: refuses a lens
   route beside an attestation, a lora route with no block beside one, a
   digest that disagrees, an intervention hash that disagrees, and an
   attested ``petri_zero_adapter`` flag that disagrees with the block's in
   EITHER direction (a genuine adapter attested under ``none__lora0``; the
   zero adapter attested under a genuine name) — the exemption from the hash
   check is earned by the ATTESTED flag, never the table's alone; echoes the
   engine block in ``/v1/models`` and stamps ``petri.engine`` /
   ``petri.engine_state`` on every routed response; a lens standup's payload
   and stamps are unchanged.
5. ``proxy.main --serving-attestation``: required with a version-2 table,
   refused with a version-1 one; reads through C's reader.

Everything synthetic (placeholder hashes, ``/bank/...`` dirs, the demo cell
set); no server, no model.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import proxy as px

from test_petri_proxy import (StubBackend, _TailRender, _ask, _cells, _chat,  # noqa: E402
                              _proxy, _source, _spec)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIXTURES / "routing_table_v1_golden.json"

#: the SEVEN pre-change RouteEntry field names, pinned as a literal — the
#: schema every existing table loader builds rows from
OLD_ROUTE_FIELDS = ("virtual_model", "cell_id", "cell_class", "serving_identity",
                    "intervention_hash", "control_of", "si_prefix_hash")
#: ... plus the ONE stage-2 field (native-psr-serving §1.8: the residual slot's
#: native position mode, default None) — a ROW carries it only when set
#: (``proxy.route_row``), and the exporter refuses to set it on a version-1
#: table, so every version-1 row still has EXACTLY the seven old keys
ROUTE_FIELDS = OLD_ROUTE_FIELDS + ("petri_position_mode",)

PASS = "proxy_vllm+passthrough+interleaved"
LENS = "proxy_vllm+lens+interleaved"
LORA = "proxy_vllm+lora+interleaved"
CUDAGRAPH_MODE = "FULL_AND_PIECEWISE"


@dataclass(frozen=True)
class _OldRouteEntry:
    """``RouteEntry`` exactly as it stood at 9486b825 (seven fields)."""

    virtual_model: str
    cell_id: str
    cell_class: str
    serving_identity: str
    intervention_hash: str
    control_of: Optional[str] = None
    si_prefix_hash: Optional[str] = None


def _old_loader(path):
    """A VERBATIM copy of ``load_routing_table``'s body at 9486b825 (the
    pre-change loader), with ``RouteEntry`` spelled as :class:`_OldRouteEntry`
    and the module-level helpers it called bound to the current module (those
    helpers are unchanged by the landing). The old-loader-against-new-table
    direction: an existing reader must keep reading version-1 tables the new
    exporter writes, and must refuse version 2 by its number."""
    doc = json.loads(Path(path).read_text())
    if doc.get("version") != 1:
        raise px.ProxyError(f"routing table version {doc.get('version')!r} != 1")
    strategy = doc.get("exact_replay_strategy")
    if doc["serving_mode"] == "exact_replay":
        if strategy not in px.EXACT_REPLAY_STRATEGIES:
            raise px.ProxyError(
                f"exact_replay routing table at {path} records no valid "
                f"exact_replay_strategy (got {strategy!r}) — a stale export; "
                "re-run export_routing_table with the stated mechanism "
                "(D0 constraint 3)")
    elif strategy is not None:
        raise px.ProxyError(
            f"routing table at {path} carries exact_replay_strategy="
            f"{strategy!r} on serving_mode {doc['serving_mode']!r} — "
            "inconsistent export")
    header = doc.get(px.THINKING_HEADER_KEY)
    header = px._validated_thinking_header(header) if header is not None else None
    return {
        "base_model": doc["base_model"],
        "serving_mode": doc["serving_mode"],
        "exact_replay_strategy": strategy,
        "base_revision": doc.get("base_revision"),
        "routes": tuple(_OldRouteEntry(**r) for r in doc["routes"]),
        "prompted_prefixes": tuple(px.PromptedPrefixSpec(**s)
                                   for s in doc["prompted_prefixes"]),
        px.THINKING_HEADER_KEY: header,
        px.THINKING_HEADER_ERROR_KEY: px._validated_thinking_header_error(
            header, doc.get(px.THINKING_HEADER_ERROR_KEY)),
        px.LENS_SERVED_POSITIONS_KEY: px._validated_lens_served_positions(
            doc.get(px.LENS_SERVED_POSITIONS_KEY)),
    }


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _world(tmp_path):
    spec = _spec()
    source = _source(tmp_path)
    return spec, source, _cells(spec, source)


def _export(tmp_path, name="table.json", **kw):
    spec, source, cells = _world(tmp_path)
    path = tmp_path / name
    px.export_routing_table(cells, source, path, base_model=spec.target_model,
                            serving_mode=spec.serving_mode,
                            exact_replay_strategy="serialize", **kw)
    return path, cells


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_of(route) -> str:
    return cm.parse_serving_identity(route.serving_identity).path


def _lora_routes(routes):
    return [r for r in routes if _path_of(r) == "lora"]


def _block(route, **over):
    cid = route.cell_id
    zero = cid.endswith(px.LORA_ZERO_CELL_SUFFIX)
    block = {"name": cid, "dir": f"/bank/{cid}",
             "config_sha256": _sha(f"cfg:{cid}"), "weights_sha256": _sha(f"w:{cid}"),
             "r": 1 if zero else 4, "petri_intervention_hash": route.intervention_hash,
             "petri_zero_adapter": zero,
             "petri_source_kind": None if zero else "lora_all"}
    block.update(over)
    return block


def _blocks(routes):
    return {r.cell_id: _block(r) for r in _lora_routes(routes)}


def _native_routes():
    """A LORA-NATIVE table's route set (no lens route): the base, one genuine
    lora cell, the zero-adapter control."""
    return [
        px.RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="base",
                      serving_identity=PASS, intervention_hash="aa" * 8),
        px.RouteEntry(virtual_model="petri__si_safety__lora_r1", cell_id="si_safety__lora_r1",
                      cell_class="class_a", serving_identity=LORA, intervention_hash="cc" * 8),
        px.RouteEntry(virtual_model="petri__none__lora0", cell_id="none__lora0",
                      cell_class="control", serving_identity=LORA, intervention_hash="aa" * 8,
                      control_of="none"),
    ]


def _attestation(blocks, **over):
    att = {"kind": "lora_native", "enforce_eager": False, "cudagraph_mode": CUDAGRAPH_MODE,
           "lora_config": {"max_loras": len(blocks), "max_lora_rank": 8,
                           "max_cpu_loras": len(blocks), "lora_dtype": "torch.bfloat16",
                           "fully_sharded_loras": False},
           # native-psr-serving (attestation schema 2): no residual slot on a LoRA-only engine
           "residual_slot": {"armed": False},
           "adapters": {cid: {"config_sha256": b["config_sha256"],
                              "weights_sha256": b["weights_sha256"],
                              "fingerprint_bf16": _sha(f"fp:{cid}"), "lora_int_id": i + 1,
                              # the loaded adapter_config's hash: the zero adapter's is None
                              "petri_intervention_hash": (None if b["petri_zero_adapter"]
                                                          else b["petri_intervention_hash"]),
                              # ... and its zero flag, as the config on disk carries it
                              "petri_zero_adapter": b["petri_zero_adapter"],
                              # ... the kind and the residual identity (None for a plain LoRA)
                              "petri_source_kind": b.get("petri_source_kind"),
                              "residual_fingerprint": None, "position_mode": None}
                        for i, (cid, b) in enumerate(sorted(blocks.items()))}}
    att.update(over)
    return att


def _native_proxy(routes=None, blocks=None, attestation=None, **kw):
    routes = routes if routes is not None else _native_routes()
    blocks = blocks if blocks is not None else _blocks(routes)
    att = attestation if attestation is not None else _attestation(blocks)
    backend = StubBackend()
    proxy = px.RoutingProxy(routes, backend, base_model="org/model", serving_mode="interleaved",
                            adapter_blocks=blocks, engine_attestation=att,
                            agent_loop_render=_TailRender(), **kw)
    return proxy, backend


# --------------------------------------------------------------------------- #
# 1. the golden version-1 table, both directions                               #
# --------------------------------------------------------------------------- #


def test_golden_v1_bytes_are_unchanged_by_the_new_exporter(tmp_path):
    """(i) The new exporter with ``adapter_blocks=None`` writes the pre-change
    bytes exactly — no new key anywhere, lora routes included."""
    golden = GOLDEN.read_bytes()
    doc = json.loads(golden)
    assert doc["version"] == 1 and px.ADAPTERS_KEY not in doc
    lora_cells = [r["cell_id"] for r in doc["routes"] if "+lora+" in r["serving_identity"]]
    assert set(lora_cells) == {"si_safety__lora_r1", "si_constitution__lora_r1", "none__lora0"}
    path, _ = _export(tmp_path)
    assert path.read_bytes() == golden


def test_new_v1_routes_carry_exactly_the_seven_old_fields(tmp_path):
    """(ii) The per-route ROW schema is untouched: every version-1 route dict
    has EXACTLY the pre-change field names, so the old dataclass constructs
    from each. ``RouteEntry`` itself carries those seven plus the stage-2
    ``petri_position_mode`` (default None, never written when None)."""
    assert tuple(f.name for f in dataclasses.fields(px.RouteEntry)) == ROUTE_FIELDS
    assert dataclasses.fields(px.RouteEntry)[-1].default is None
    path, _ = _export(tmp_path)
    for r in json.loads(path.read_text())["routes"]:
        assert tuple(sorted(r)) == tuple(sorted(OLD_ROUTE_FIELDS)), r
        _OldRouteEntry(**r)


def test_a_position_mode_never_reaches_a_version_1_table(tmp_path):
    """Known-bad (native-psr-serving §1.8): a NON-None mode on an export with
    no adapter blocks refuses before a byte is written — a version-1 table
    never carries the stage-2 key, so the golden and every old loader stay
    exact; all-None modes are a no-op (the same bytes as no modes at all)."""
    path, cells = _export(tmp_path, "plain.json")
    lora = _lora_routes(px.routes_from_cells(cells))[0]
    with pytest.raises(px.ProxyError, match="version-1 table"):
        _export(tmp_path, "bad.json", position_modes={lora.cell_id: "response"})
    assert not (tmp_path / "bad.json").exists()
    none_path, _ = _export(tmp_path, "none.json", position_modes={lora.cell_id: None})
    assert none_path.read_bytes() == path.read_bytes()


def test_old_loader_reads_the_new_v1_table_and_yields_the_same_routes(tmp_path):
    """(iii) The verbatim pre-change loader, on the new exporter's version-1
    output: the routes it builds equal ``routes_from_cells(cells)`` field by
    field, the prefix specs and every top-level fact too."""
    path, cells = _export(tmp_path)
    old = _old_loader(path)
    expected = px.routes_from_cells(cells)
    # the old dataclass has the seven fields; the new route's ROW (route_row) is exactly those
    # seven on a version-1 table (the stage-2 field is never written when None)
    assert [asdict(r) for r in old["routes"]] == [px.route_row(r) for r in expected]
    assert all(r.petri_position_mode is None for r in expected)
    new = px.load_routing_table(path)
    assert old["prompted_prefixes"] == new["prompted_prefixes"]
    for key in ("base_model", "serving_mode", "exact_replay_strategy", "base_revision",
                px.THINKING_HEADER_KEY, px.THINKING_HEADER_ERROR_KEY,
                px.LENS_SERVED_POSITIONS_KEY):
        assert old[key] == new[key], key


def test_new_loader_reads_the_golden_as_version_1_with_no_adapters(tmp_path):
    """(iv) The committed golden file loads through the new loader with
    ``version == 1`` and ``adapters is None``; its routes are the fixture
    cell set's."""
    _, _, cells = _world(tmp_path)
    table = px.load_routing_table(GOLDEN)
    assert table["version"] == 1 and table[px.ADAPTERS_KEY] is None
    assert set(table["routes"]) == set(px.routes_from_cells(cells))
    # the loaded dict gained exactly the new keys: the adapters map, the thinking header pair, the served-positions
    # stamp, the organism overlay list (em-misalign: empty on a table that declares none) and the engine-neutral
    # served-positions map (native-positions-resolve: None on this table, which states no mode)
    assert set(table) == {"version", px.ADAPTERS_KEY, "base_model", "serving_mode",
                          "exact_replay_strategy", "base_revision", "routes",
                          "prompted_prefixes", px.THINKING_HEADER_KEY,
                          px.THINKING_HEADER_ERROR_KEY, px.LENS_SERVED_POSITIONS_KEY, px.ORGANISMS_KEY,
                          px.SERVED_POSITIONS_KEY}
    assert table[px.SERVED_POSITIONS_KEY] is None
    assert table[px.ORGANISMS_KEY] == ()


def test_old_loader_refuses_a_version_2_table(tmp_path):
    """(v) No old reader can silently misread a lora-native table: it refuses
    by version number before touching a route."""
    _, _, cells = _world(tmp_path)
    path, _ = _export(tmp_path, adapter_blocks=_blocks(px.routes_from_cells(cells)))
    assert json.loads(path.read_text())["version"] == 2
    with pytest.raises(px.ProxyError, match=r"version 2 != 1"):
        _old_loader(path)


# --------------------------------------------------------------------------- #
# 2. version 2: round trip + refusals                                          #
# --------------------------------------------------------------------------- #


def test_version_2_round_trip_with_blocks(tmp_path):
    v1_path, cells = _export(tmp_path, "v1.json")
    routes = px.routes_from_cells(cells)
    blocks = _blocks(routes)
    v2_path, _ = _export(tmp_path, "v2.json", adapter_blocks=blocks)
    v1, v2 = json.loads(v1_path.read_text()), json.loads(v2_path.read_text())
    assert v2["version"] == 2
    assert set(v2[px.ADAPTERS_KEY]) == {r.cell_id for r in _lora_routes(routes)}
    # everything but the version and the adapter map is the version-1 document
    assert {k: v for k, v in v2.items() if k not in ("version", px.ADAPTERS_KEY)} == \
        {k: v for k, v in v1.items() if k != "version"}
    table = px.load_routing_table(v2_path)
    assert table["version"] == 2
    assert table[px.ADAPTERS_KEY] == blocks
    old = px.load_routing_table(v1_path)
    assert {k: v for k, v in table.items() if k not in ("version", px.ADAPTERS_KEY)} == \
        {k: v for k, v in old.items() if k not in ("version", px.ADAPTERS_KEY)}
    # validated blocks are fresh dicts, never the caller's objects
    for cid, block in table[px.ADAPTERS_KEY].items():
        assert block == blocks[cid] and block is not blocks[cid]


def test_export_refuses_a_lora_route_without_its_block(tmp_path):
    _, _, cells = _world(tmp_path)
    blocks = _blocks(px.routes_from_cells(cells))
    blocks.pop("none__lora0")
    with pytest.raises(px.ProxyError, match=r"lora route\(s\) \['none__lora0'\] have no adapters block"):
        _export(tmp_path, adapter_blocks=blocks)
    with pytest.raises(px.ProxyError, match="have no adapters block"):
        _export(tmp_path, adapter_blocks={})


def test_export_refuses_a_block_for_a_non_lora_or_unrouted_cell(tmp_path):
    _, _, cells = _world(tmp_path)
    routes = px.routes_from_cells(cells)
    by_cell = {r.cell_id: r for r in routes}
    base = _blocks(routes)
    for cell in ("none", "prompted__safety", "si_safety__psr_kl_all", "none__lens0"):
        assert _path_of(by_cell[cell]) != "lora"
        bad = dict(base, **{cell: _block(by_cell[cell], petri_zero_adapter=cell.endswith("__lora0"))})
        with pytest.raises(px.ProxyError, match=f"non-lora route\\(s\\) \\['{cell}'\\]"):
            _export(tmp_path, adapter_blocks=bad)
    ghost = px.RouteEntry(virtual_model="petri__ghost", cell_id="ghost", cell_class="class_a",
                          serving_identity=by_cell["none__lora0"].serving_identity,
                          intervention_hash="ee" * 8)
    with pytest.raises(px.ProxyError, match=r"names cell\(s\) \['ghost'\] that no route serves"):
        _export(tmp_path, adapter_blocks=dict(base, ghost=_block(ghost)))


@pytest.mark.parametrize("cell,over,match", [
    ("si_safety__lora_r1", {"name": "si_constitution__lora_r1"}, r"\.name is 'si_constitution__lora_r1'"),
    ("si_safety__lora_r1", {"config_sha256": "abc"}, r"config_sha256 must be a 64-hex"),
    ("si_safety__lora_r1", {"weights_sha256": "A" * 64}, r"weights_sha256 must be a 64-hex"),
    ("si_safety__lora_r1", {"petri_intervention_hash": "ff" * 8}, r"petri_intervention_hash 'ff+' != the route's"),
    ("si_safety__lora_r1", {"petri_zero_adapter": True}, r"petri_zero_adapter is True but the cell is not a zero-adapter"),
    ("none__lora0", {"petri_zero_adapter": False}, r"petri_zero_adapter is False but the cell is a zero-adapter"),
    ("si_safety__lora_r1", {"petri_zero_adapter": "no"}, r"petri_zero_adapter must be a bool"),
    ("si_safety__lora_r1", {"r": 0}, r"\.r must be a positive int"),
    ("si_safety__lora_r1", {"r": True}, r"\.r must be a positive int"),
    ("si_safety__lora_r1", {"dir": "relative/dir"}, r"\.dir must be an absolute path"),
    ("si_safety__lora_r1", {"petri_source_kind": ""}, r"petri_source_kind must be a non-empty string or None"),
])
def test_export_refuses_a_malformed_block(tmp_path, cell, over, match):
    _, _, cells = _world(tmp_path)
    routes = px.routes_from_cells(cells)
    by_cell = {r.cell_id: r for r in routes}
    blocks = _blocks(routes)
    blocks[cell] = _block(by_cell[cell], **over)
    with pytest.raises(px.ProxyError, match=match):
        _export(tmp_path, adapter_blocks=blocks)


def test_export_refuses_unknown_or_missing_block_keys(tmp_path):
    _, _, cells = _world(tmp_path)
    routes = px.routes_from_cells(cells)
    blocks = _blocks(routes)
    blocks["si_safety__lora_r1"]["object_sha256"] = "x"       # the block carries NO object sha
    with pytest.raises(px.ProxyError, match=r"unknown \['object_sha256'\]"):
        _export(tmp_path, adapter_blocks=blocks)
    blocks = _blocks(routes)
    del blocks["si_safety__lora_r1"]["dir"]
    with pytest.raises(px.ProxyError, match=r"missing \['dir'\]"):
        _export(tmp_path, adapter_blocks=blocks)
    blocks = _blocks(routes)
    blocks["si_safety__lora_r1"] = ["not", "a", "mapping"]
    with pytest.raises(px.ProxyError, match="must be a mapping"):
        _export(tmp_path, adapter_blocks=blocks)
    with pytest.raises(px.ProxyError, match="must be a cell_id -> adapter block mapping"):
        _export(tmp_path, adapter_blocks=["si_safety__lora_r1"])


def test_load_refuses_bad_versions_and_inconsistent_adapter_maps(tmp_path):
    v1_path, cells = _export(tmp_path, "v1.json")
    doc = json.loads(v1_path.read_text())
    # version 3: refused by number
    bad = tmp_path / "v3.json"
    bad.write_text(json.dumps(dict(doc, version=3)))
    with pytest.raises(px.ProxyError, match=r"version 3 not in \(1, 2\)"):
        px.load_routing_table(bad)
    # a version-1 document smuggling an adapters map
    blocks = _blocks(px.routes_from_cells(cells))
    bad.write_text(json.dumps(dict(doc, **{px.ADAPTERS_KEY: blocks})))
    with pytest.raises(px.ProxyError, match="version 1 .* carries an adapters map"):
        px.load_routing_table(bad)
    # version 2 with no adapters map
    bad.write_text(json.dumps(dict(doc, version=2)))
    with pytest.raises(px.ProxyError, match="version 2 .* carries no adapters map"):
        px.load_routing_table(bad)
    # version 2 with an uncovered lora route / a tampered block / a non-lora block
    blocks_missing = dict(blocks)
    blocks_missing.pop("si_constitution__lora_r1")
    bad.write_text(json.dumps(dict(doc, version=2, **{px.ADAPTERS_KEY: blocks_missing})))
    with pytest.raises(px.ProxyError, match=r"\['si_constitution__lora_r1'\] have no adapters block"):
        px.load_routing_table(bad)
    tampered = json.loads(json.dumps(blocks))
    tampered["none__lora0"]["petri_intervention_hash"] = "ee" * 8
    bad.write_text(json.dumps(dict(doc, version=2, **{px.ADAPTERS_KEY: tampered})))
    with pytest.raises(px.ProxyError, match="petri_intervention_hash .* != the route's"):
        px.load_routing_table(bad)
    extra = dict(blocks, none=_block(next(r for r in px.routes_from_cells(cells) if r.cell_id == "none")))
    bad.write_text(json.dumps(dict(doc, version=2, **{px.ADAPTERS_KEY: extra})))
    with pytest.raises(px.ProxyError, match=r"non-lora route\(s\) \['none'\]"):
        px.load_routing_table(bad)


# --------------------------------------------------------------------------- #
# 3. the on-disk binding                                                       #
# --------------------------------------------------------------------------- #


def _adapter_dir(tmp_path, *, r=4, zero=False, name="si_safety__lora_r1", intervention_hash="cc" * 8,
                 source_kind="lora_all"):
    """A synthetic adapter dir shaped like the exporter's: the zero adapter's
    config carries the zero marker and NO intervention hash; a genuine one
    carries the genuine marker, the object's ``petri_intervention_hash`` and
    its ``petri_source_kind`` (what the re-hash holds the block to)."""
    d = tmp_path / "bank" / name
    d.mkdir(parents=True)
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": float(r), "target_modules": ["o_proj"]}
    if zero:
        cfg["petri_zero_adapter"] = True
    else:
        cfg["petri_genuine_adapter"] = True
        cfg["petri_intervention_hash"] = intervention_hash
        cfg["petri_source_kind"] = source_kind
    (d / px.ADAPTER_CONFIG_FILE).write_text(json.dumps(cfg, indent=2, sort_keys=True))
    (d / px.ADAPTER_WEIGHTS_FILE).write_bytes(b"synthetic-safetensors-bytes:" + name.encode())
    route = px.RouteEntry(virtual_model=f"petri__{name}", cell_id=name, cell_class="class_a",
                          serving_identity=LORA, intervention_hash=intervention_hash)
    block = _block(route, dir=str(d), r=r,
                   config_sha256=hashlib.sha256((d / px.ADAPTER_CONFIG_FILE).read_bytes()).hexdigest(),
                   weights_sha256=hashlib.sha256((d / px.ADAPTER_WEIGHTS_FILE).read_bytes()).hexdigest())
    return d, px.validated_adapter_block(block, cell_id=name, intervention_hash=intervention_hash)


def test_verify_adapter_block_on_disk_passes_then_refuses_every_tamper(tmp_path):
    d, block = _adapter_dir(tmp_path)
    px.verify_adapter_block_on_disk(block)                       # the honest case
    # tampered weights
    (d / px.ADAPTER_WEIGHTS_FILE).write_bytes(b"tampered")
    with pytest.raises(px.ProxyError, match="weights_sha256"):
        px.verify_adapter_block_on_disk(block)
    # tampered config (the weights restored)
    (d / px.ADAPTER_WEIGHTS_FILE).write_bytes(b"synthetic-safetensors-bytes:si_safety__lora_r1")
    px.verify_adapter_block_on_disk(block)
    cfg_path = d / px.ADAPTER_CONFIG_FILE
    original = cfg_path.read_text()
    cfg_path.write_text(original.replace('"r": 4', '"r": 8'))
    with pytest.raises(px.ProxyError, match="config_sha256"):
        px.verify_adapter_block_on_disk(block)
    # a block whose hashes match the file but whose r does not
    wrong_r = dict(block, r=8, config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest())
    cfg_path.write_text(original)
    with pytest.raises(px.ProxyError, match=r"r=4 != the block's r=8"):
        px.verify_adapter_block_on_disk(dict(wrong_r, config_sha256=block["config_sha256"]))
    # a missing file
    (d / px.ADAPTER_WEIGHTS_FILE).unlink()
    with pytest.raises(px.ProxyError, match="is missing"):
        px.verify_adapter_block_on_disk(block)
    # an unvalidated block is refused before any hashing
    with pytest.raises(px.ProxyError, match="validate it first"):
        px.verify_adapter_block_on_disk({k: v for k, v in block.items() if k != "r"})


def test_verify_adapter_block_on_disk_holds_the_zero_flag_both_ways(tmp_path):
    d, block = _adapter_dir(tmp_path, r=1, zero=True, name="none__lora0")
    px.verify_adapter_block_on_disk(block)
    # the file says zero, the block says genuine (a validated genuine block
    # for the same bytes would need a genuine cell id; build it by hand)
    genuine = dict(block, name="si_x__lora_r1", petri_zero_adapter=False)
    with pytest.raises(px.ProxyError, match="petri_zero_adapter=True disagrees with the block's False"):
        px.verify_adapter_block_on_disk(genuine)
    # the file says genuine, the block says zero
    d2, block2 = _adapter_dir(tmp_path / "two", r=1, zero=False, name="si_y__lora_r1")
    zero_claim = dict(block2, name="none__lora0", petri_zero_adapter=True)
    with pytest.raises(px.ProxyError, match="petri_zero_adapter=None disagrees with the block's True"):
        px.verify_adapter_block_on_disk(zero_claim)


def test_verify_adapter_block_on_disk_holds_the_configs_provenance_to_the_block(tmp_path):
    """A block that points cell A at adapter B's directory WITH B's digests
    re-hashes clean and agrees on r and the zero flag, so the config's own
    ``petri_intervention_hash`` (a genuine block) and ``petri_source_kind``
    (every block) are held to the block's too — the pointer swap refuses
    naming ``petri_intervention_hash``; a kind mismatch refuses naming
    ``petri_source_kind``; the zero adapter, whose config carries no hash
    while its block carries ``none``'s, is exempt from the hash comparison
    and still checked on the kind."""
    d_a, block_a = _adapter_dir(tmp_path / "a", name="si_a__lora_r1", intervention_hash="aa" * 8)
    d_b, block_b = _adapter_dir(tmp_path / "b", name="si_b__lora_r1", intervention_hash="bb" * 8)
    px.verify_adapter_block_on_disk(block_a)
    px.verify_adapter_block_on_disk(block_b)
    # cell A's block pointed at B's dir with B's digests: every earlier check agrees
    pointed = dict(block_a, dir=block_b["dir"], config_sha256=block_b["config_sha256"],
                   weights_sha256=block_b["weights_sha256"])
    with pytest.raises(px.ProxyError, match=r"petri_intervention_hash='bb+' != the block's 'aa+'"):
        px.verify_adapter_block_on_disk(pointed)
    # a block claiming another export kind for the same bytes
    with pytest.raises(px.ProxyError, match=r"petri_source_kind='lora_all' != the block's 'lora_subtraction'"):
        px.verify_adapter_block_on_disk(dict(block_a, petri_source_kind="lora_subtraction"))
    # the zero adapter: no hash in its config, none's hash in its block -> the
    # hash is not compared; the kind (None both sides) is
    d_z, block_z = _adapter_dir(tmp_path / "z", r=1, zero=True, name="none__lora0")
    assert block_z["petri_source_kind"] is None
    px.verify_adapter_block_on_disk(block_z)
    with pytest.raises(px.ProxyError, match="petri_source_kind=None != the block's 'lora_all'"):
        px.verify_adapter_block_on_disk(dict(block_z, petri_source_kind="lora_all"))


# --------------------------------------------------------------------------- #
# 4. the proxy: startup bindings, /v1/models, the per-row stamp                #
# --------------------------------------------------------------------------- #


def test_attested_proxy_echoes_the_engine_and_stamps_every_row():
    routes = _native_routes()
    blocks = _blocks(routes)
    att = _attestation(blocks)
    proxy, backend = _native_proxy(routes, blocks, att)
    assert proxy.adapter_blocks == blocks
    assert proxy.engine_attestation == att
    identity = proxy.models_payload()["petri_proxy"]
    assert identity[px.ENGINE_KEY] == {
        "kind": "lora_native", "enforce_eager": False, "cudagraph_mode": CUDAGRAPH_MODE,
        "residual_slot": {"armed": False}, "adapters": att["adapters"]}
    assert set(identity[px.ENGINE_KEY]["adapters"]) == {"si_safety__lora_r1", "none__lora0"}
    for vmodel, upstream_model in (("petri__si_safety__lora_r1", "si_safety__lora_r1"),
                                   ("petri__none__lora0", "none__lora0"),
                                   ("petri__base", "org/model")):
        out = _ask(proxy, _chat(vmodel))
        assert out["petri"][px.ENGINE_KEY] == "lora_native"
        assert out["petri"][px.ENGINE_STATE_KEY] == {"enforce_eager": False,
                                                     "cudagraph_mode": CUDAGRAPH_MODE}
        # the four contract stamps are untouched by the additive keys
        assert out["model"] == vmodel and out["petri"]["virtual_model"] == vmodel
        assert backend.payloads[-1]["model"] == upstream_model   # _rewrite's lora branch unchanged
        assert px.UPSTREAM_EXTRA_ARGS_KEY not in backend.payloads[-1]


def test_lens_standup_payload_and_stamps_are_unchanged(tmp_path):
    proxy, _, _ = _proxy(tmp_path)
    assert proxy.adapter_blocks is None and proxy.engine_attestation is None
    assert px.ENGINE_KEY not in proxy.models_payload()["petri_proxy"]
    out = _ask(proxy, _chat("petri__si_safety__lora_r1"))
    assert px.ENGINE_KEY not in out["petri"] and px.ENGINE_STATE_KEY not in out["petri"]
    with pytest.raises(px.ProxyError, match="no engine attestation"):
        proxy.engine_payload()


def test_blocks_without_an_attestation_are_validated_and_echo_nothing():
    routes = _native_routes()
    blocks = _blocks(routes)
    proxy = px.RoutingProxy(routes, StubBackend(), base_model="org/model", serving_mode="interleaved",
                            adapter_blocks=blocks, agent_loop_render=_TailRender())
    assert proxy.adapter_blocks == blocks and proxy.engine_attestation is None
    assert px.ENGINE_KEY not in proxy.models_payload()["petri_proxy"]
    # the same coverage rule as the exporter and the loader
    partial = {k: v for k, v in blocks.items() if k != "none__lora0"}
    with pytest.raises(px.ProxyError, match=r"\['none__lora0'\] have no adapters block"):
        px.RoutingProxy(routes, StubBackend(), base_model="org/model", serving_mode="interleaved",
                        adapter_blocks=partial, agent_loop_render=_TailRender())


def test_attested_proxy_refuses_a_lens_route():
    routes = _native_routes() + [
        px.RouteEntry(virtual_model="petri__none__lens0", cell_id="none__lens0",
                      cell_class="control", serving_identity=LENS, intervention_hash="aa" * 8,
                      control_of="none")]
    with pytest.raises(px.ProxyError, match=r"lens route\(s\) \['none__lens0'\] on a proxy started with a lora_native"):
        _native_proxy(routes, _blocks(routes))


def test_attested_proxy_refuses_a_lora_route_without_its_block():
    routes = _native_routes()
    blocks = _blocks(routes)
    att = _attestation(blocks)
    with pytest.raises(px.ProxyError, match="engine attestation given but no adapters blocks"):
        px.RoutingProxy(routes, StubBackend(), base_model="org/model", serving_mode="interleaved",
                        adapter_blocks=None, engine_attestation=att, agent_loop_render=_TailRender())
    partial = {k: v for k, v in blocks.items() if k != "si_safety__lora_r1"}
    with pytest.raises(px.ProxyError, match=r"\['si_safety__lora_r1'\] have no adapters block"):
        _native_proxy(routes, partial, att)


def test_attested_proxy_refuses_a_digest_that_disagrees():
    routes = _native_routes()
    blocks = _blocks(routes)
    for key in ("config_sha256", "weights_sha256"):
        att = _attestation(blocks)
        att["adapters"]["si_safety__lora_r1"][key] = _sha("some other adapter")
        with pytest.raises(px.ProxyError, match=f"lora route 'si_safety__lora_r1': adapter block {key}"):
            _native_proxy(routes, blocks, att)


def test_attested_proxy_refuses_an_intervention_hash_that_disagrees():
    """THE FINDING (code review 2026-09-11, P1): both file digests agree, but
    the attestation records the adapter loaded under this name was exported
    from intervention A while the route/block claim B — rows would carry B's
    hash over A's weights. Refused naming both hashes. The zero adapter is
    exempt (its config carries no hash; its block carries `none`'s)."""
    routes = _native_routes()
    blocks = _blocks(routes)
    att = _attestation(blocks)
    assert att["adapters"]["si_safety__lora_r1"]["petri_intervention_hash"] == "cc" * 8
    assert att["adapters"]["none__lora0"]["petri_intervention_hash"] is None
    _native_proxy(routes, blocks, att)                       # agreeing: starts
    att["adapters"]["si_safety__lora_r1"]["petri_intervention_hash"] = "dd" * 8
    with pytest.raises(px.ProxyError, match=r"lora route 'si_safety__lora_r1': the table binds "
                                            r"intervention 'cccccccccccccccc' but the engine "
                                            r"attestation records .* 'dddddddddddddddd'"):
        _native_proxy(routes, blocks, att)
    # an attestation with NO hash for a genuine adapter (a zero-shaped config
    # loaded under a genuine name) refuses too — None != the route's hash
    att = _attestation(blocks)
    att["adapters"]["si_safety__lora_r1"]["petri_intervention_hash"] = None
    with pytest.raises(px.ProxyError, match="was exported from None"):
        _native_proxy(routes, blocks, att)
    # the zero adapter: attested None beside the block's `none` hash is the
    # documented exemption, not a mismatch
    att = _attestation(blocks)
    assert blocks["none__lora0"]["petri_intervention_hash"] == "aa" * 8
    proxy, _ = _native_proxy(routes, blocks, att)
    assert proxy.engine_attestation["adapters"]["none__lora0"]["petri_intervention_hash"] is None


def test_attested_proxy_refuses_a_genuine_adapter_attested_under_the_zero_name():
    """The REVERSE direction of the one above. The table's ``none__lora0``
    block is flagged zero (``validated_adapter_block`` derives that from the
    cell id) and carries `none`'s hash; the server attests that the adapter
    it LOADED under that name is a genuine one, exported from intervention
    `bb..`. With that adapter's digests copied into the block, the old
    binding read the table's flag alone, took the zero exemption and
    started — a ``none__lora0`` request then returned `bb..`'s steering
    stamped `aa..`, the no-op control. Refused at start, naming the route,
    the table's flag and the attested state."""
    routes = _native_routes()
    blocks = _blocks(routes)
    assert blocks["none__lora0"]["petri_zero_adapter"] is True
    assert blocks["none__lora0"]["petri_intervention_hash"] == "aa" * 8
    att = _attestation(blocks)
    att["adapters"]["none__lora0"].update(petri_zero_adapter=False,
                                          petri_intervention_hash="bb" * 8)
    with pytest.raises(px.ProxyError) as err:
        _native_proxy(routes, blocks, att)
    msg = str(err.value)
    assert msg.startswith("lora route 'none__lora0': the table's adapter block flags "
                          "petri_zero_adapter=True but the engine attestation records a "
                          "GENUINE adapter loaded under this name")
    assert "(petri_zero_adapter=False, petri_intervention_hash='bbbbbbbbbbbbbbbb')" in msg
    assert msg.endswith("genuine steering would be returned and stamped as the no-op "
                        "control; refusing to start")
    # the FLAG is what discriminates: a genuine adapter whose attested hash
    # happens to equal the route's `none` hash refuses too, and so does one
    # attested genuine with no hash at all
    for ih in ("aa" * 8, None):
        att = _attestation(blocks)
        att["adapters"]["none__lora0"].update(petri_zero_adapter=False,
                                              petri_intervention_hash=ih)
        with pytest.raises(px.ProxyError,
                           match=r"lora route 'none__lora0': the table's adapter block flags "
                                 r"petri_zero_adapter=True but the engine attestation records "
                                 r"a GENUINE adapter"):
            _native_proxy(routes, blocks, att)


def test_attested_proxy_refuses_the_zero_adapter_attested_under_a_genuine_name():
    """The other direction on the flag: the server attests the adapter loaded
    under the genuine cell's name IS the zero adapter (flag True, no hash)
    while the table's block is genuine. Refused on the flag, before the hash
    comparison, naming the consequence (base served under a steered
    identity)."""
    routes = _native_routes()
    blocks = _blocks(routes)
    att = _attestation(blocks)
    att["adapters"]["si_safety__lora_r1"].update(petri_zero_adapter=True,
                                                 petri_intervention_hash=None)
    with pytest.raises(px.ProxyError) as err:
        _native_proxy(routes, blocks, att)
    msg = str(err.value)
    assert msg.startswith("lora route 'si_safety__lora_r1': the table's adapter block flags "
                          "petri_zero_adapter=False but the engine attestation records the "
                          "ZERO adapter loaded under this name (petri_zero_adapter=True, "
                          "petri_intervention_hash=None)")
    assert msg.endswith("the base model would be served under a steered identity; "
                        "refusing to start")


def test_attested_proxy_loads_the_zero_adapter_when_both_sides_flag_it_zero():
    """The positive case: attested zero + table zero + a null attested hash
    loads; the exemption from the hash check is earned by the ATTESTED flag;
    the identity and ``/v1/models`` echo the flag per adapter; a
    ``none__lora0`` request routes to the zero adapter and stamps `none`'s
    hash."""
    routes = _native_routes()
    blocks = _blocks(routes)
    att = _attestation(blocks)
    assert att["adapters"]["none__lora0"]["petri_zero_adapter"] is True
    assert att["adapters"]["none__lora0"]["petri_intervention_hash"] is None
    assert att["adapters"]["si_safety__lora_r1"]["petri_zero_adapter"] is False
    proxy, backend = _native_proxy(routes, blocks, att)
    ident = proxy.engine_attestation["adapters"]
    assert ident["none__lora0"]["petri_zero_adapter"] is True
    assert ident["none__lora0"]["petri_intervention_hash"] is None
    assert ident["si_safety__lora_r1"]["petri_zero_adapter"] is False
    assert ident["si_safety__lora_r1"]["petri_intervention_hash"] == "cc" * 8
    assert proxy.models_payload()["petri_proxy"][px.ENGINE_KEY]["adapters"] == att["adapters"]
    out = _ask(proxy, _chat("petri__none__lora0"))
    assert out["petri_intervention_hash"] == "aa" * 8
    assert backend.payloads[-1]["model"] == "none__lora0"


def test_attested_proxy_refuses_an_adapter_set_that_differs_from_the_table():
    routes = _native_routes()
    blocks = _blocks(routes)
    # the server attests an adapter no lora route serves
    att = _attestation(blocks)
    att["adapters"]["si_extra__lora_r1"] = dict(att["adapters"]["si_safety__lora_r1"], lora_int_id=9)
    with pytest.raises(px.ProxyError, match=r"names adapter\(s\) \['si_extra__lora_r1'\] that no lora route"):
        _native_proxy(routes, blocks, att)
    # the table serves an adapter the server never loaded
    att = _attestation(blocks)
    del att["adapters"]["none__lora0"]
    with pytest.raises(px.ProxyError, match="engine attestation names no adapter 'none__lora0'"):
        _native_proxy(routes, blocks, att)


@pytest.mark.parametrize("over,match", [
    ({"kind": "lens"}, r"kind 'lens' != 'lora_native'"),
    ({"enforce_eager": True}, r"enforce_eager is True, not False"),
    ({"enforce_eager": None}, r"enforce_eager is None, not False"),
    ({"cudagraph_mode": ""}, r"cudagraph_mode must be a non-empty string"),
    ({"lora_config": None}, r"lora_config must be a mapping"),
    ({"adapters": {}}, r"adapters must be a non-empty"),
    ({"extra": 1}, r"unknown \['extra'\]"),
])
def test_engine_attestation_shape_is_exact(over, match):
    blocks = _blocks(_native_routes())
    att = _attestation(blocks)
    att.update(over)
    with pytest.raises(px.ProxyError, match=match):
        px.validated_engine_attestation(att)
    missing = _attestation(blocks)
    del missing["lora_config"]
    with pytest.raises(px.ProxyError, match=r"missing \['lora_config'\]"):
        px.validated_engine_attestation(missing)
    with pytest.raises(px.ProxyError, match="must be a mapping"):
        px.validated_engine_attestation(["lora_native"])


def test_engine_attestation_adapter_entries_are_exact():
    blocks = _blocks(_native_routes())
    att = _attestation(blocks)
    entry = att["adapters"]["si_safety__lora_r1"]
    for over, match in (
            ({"fingerprint_bf16": "zz"}, r"fingerprint_bf16 must be a 64-hex"),
            ({"lora_int_id": 0}, r"lora_int_id must be a positive int"),
            ({"lora_int_id": True}, r"lora_int_id must be a positive int"),
            ({"lora_int_id": att["adapters"]["none__lora0"]["lora_int_id"]}, r"lora_int_id 1 is claimed by both"),
            ({"petri_intervention_hash": "zz"}, r"petri_intervention_hash must be a 16-hex"),
            ({"petri_intervention_hash": "CC" * 8}, r"petri_intervention_hash must be a 16-hex"),
            ({"petri_zero_adapter": "no"}, r"petri_zero_adapter must be a bool, got 'no'"),
            ({"petri_zero_adapter": 1}, r"petri_zero_adapter must be a bool, got 1"),
            ({"petri_zero_adapter": None}, r"petri_zero_adapter must be a bool, got None"),
            # a zero-flagged entry carrying an intervention hash is not the
            # zero adapter, whatever the flag says
            ({"petri_zero_adapter": True},
             r"is flagged petri_zero_adapter=True but carries petri_intervention_hash "
             r"'cccccccccccccccc'"),
            ({"dir": "/x"}, r"must carry exactly the keys")):
        bad = json.loads(json.dumps(att))
        bad["adapters"]["si_safety__lora_r1"] = dict(entry, **over)
        with pytest.raises(px.ProxyError, match=match):
            px.validated_engine_attestation(bad)
    for key in ("lora_int_id", "petri_zero_adapter"):
        bad = json.loads(json.dumps(att))
        del bad["adapters"]["si_safety__lora_r1"][key]
        with pytest.raises(px.ProxyError, match="must carry exactly the keys"):
            px.validated_engine_attestation(bad)
    clean = px.validated_engine_attestation(att)
    assert clean == att and clean is not att


# --------------------------------------------------------------------------- #
# 5. the CLI: --serving-attestation                                            #
# --------------------------------------------------------------------------- #


def _cli(table_path, *extra):
    return ["--routing-table", str(table_path), "--upstream", "http://127.0.0.1:1",
            "--port", "1", *extra]


def test_main_requires_the_attestation_for_a_version_2_table_and_refuses_it_for_v1(tmp_path):
    v1_path, cells = _export(tmp_path, "v1.json")
    v2_path, _ = _export(tmp_path, "v2.json", adapter_blocks=_blocks(px.routes_from_cells(cells)))
    with pytest.raises(px.ProxyError, match="needs --serving-attestation"):
        px.main(_cli(v2_path))
    with pytest.raises(px.ProxyError, match="version-1 routing table"):
        px.main(_cli(v1_path, "--serving-attestation", str(tmp_path / "att.json")))


def test_main_reads_the_attestation_through_the_reader_and_binds_it(tmp_path, monkeypatch):
    """The positive path, with C's reader STUBBED (its module lands in its
    own workstream): ``read_serving_attestation`` is handed the path,
    ``engine_identity`` its result, and the proxy that ``make_app`` receives
    carries the attestation and the table's blocks."""
    pytest.importorskip("aiohttp")
    routes = _native_routes()
    blocks = _blocks(routes)
    att = _attestation(blocks)
    doc = {"version": 2, "base_model": "org/model", "serving_mode": "interleaved",
           "exact_replay_strategy": None, "base_revision": "r",
           "routes": [asdict(r) for r in routes], "prompted_prefixes": [],
           px.LENS_SERVED_POSITIONS_KEY: {}, px.ADAPTERS_KEY: blocks}
    table_path = tmp_path / "routing_table.json"
    table_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    att_path = tmp_path / "serving_attestation.json"
    att_path.write_text("{}")
    seen = {}
    stub = types.ModuleType("sisteer.capeval.serving.serving_attestation")

    def read_serving_attestation(path):
        seen["read"] = Path(path)
        return {"raw": "attestation"}

    def engine_identity(raw):
        seen["reduced"] = raw
        return att
    stub.read_serving_attestation = read_serving_attestation
    stub.engine_identity = engine_identity
    monkeypatch.setitem(sys.modules, "sisteer.capeval.serving.serving_attestation", stub)

    def fake_make_app(proxy):
        seen["proxy"] = proxy
        return "app"
    monkeypatch.setattr(px, "make_app", fake_make_app)
    import aiohttp.web
    monkeypatch.setattr(aiohttp.web, "run_app", lambda app, host, port: seen.setdefault("ran", (app, host, port)))
    px.main(_cli(table_path, "--serving-attestation", str(att_path)))
    assert seen["read"] == att_path and seen["reduced"] == {"raw": "attestation"}
    proxy = seen["proxy"]
    assert proxy.engine_attestation == att and proxy.adapter_blocks == blocks
    assert seen["ran"] == ("app", "127.0.0.1", 1)


def test_main_refuses_a_genuine_adapter_attested_under_the_zero_name(tmp_path, monkeypatch):
    """The finding's trigger is THIS path — the standalone
    ``--serving-attestation`` proxy, which runs no on-disk verification (the
    managed standup's ``verify_adapter_block_on_disk`` never happens here).
    With the reader stubbed to hand back an identity recording a GENUINE
    adapter (intervention `bb..`) loaded under ``none__lora0`` beside a
    regenerated table whose zero block copied that adapter's digests, main
    refuses before ``make_app``; nothing is served."""
    pytest.importorskip("aiohttp")
    routes = _native_routes()
    blocks = _blocks(routes)
    att = _attestation(blocks)
    att["adapters"]["none__lora0"].update(petri_zero_adapter=False,
                                          petri_intervention_hash="bb" * 8)
    doc = {"version": 2, "base_model": "org/model", "serving_mode": "interleaved",
           "exact_replay_strategy": None, "base_revision": "r",
           "routes": [asdict(r) for r in routes], "prompted_prefixes": [],
           px.LENS_SERVED_POSITIONS_KEY: {}, px.ADAPTERS_KEY: blocks}
    table_path = tmp_path / "routing_table.json"
    table_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    att_path = tmp_path / "serving_attestation.json"
    att_path.write_text("{}")
    stub = types.ModuleType("sisteer.capeval.serving.serving_attestation")
    stub.read_serving_attestation = lambda path: {"raw": "attestation"}
    stub.engine_identity = lambda raw: att
    monkeypatch.setitem(sys.modules, "sisteer.capeval.serving.serving_attestation", stub)
    made = []
    monkeypatch.setattr(px, "make_app", lambda proxy: made.append(proxy))
    with pytest.raises(px.ProxyError,
                       match=r"lora route 'none__lora0': the table's adapter block flags "
                             r"petri_zero_adapter=True but the engine attestation records a "
                             r"GENUINE adapter loaded under this name \(petri_zero_adapter=False, "
                             r"petri_intervention_hash='bbbbbbbbbbbbbbbb'\)"):
        px.main(_cli(table_path, "--serving-attestation", str(att_path)))
    assert made == []
