"""native-positions-resolve, the PRODUCER half: the ENGINE-NEUTRAL ``served_positions`` map.

Every natively served fitted card recorded ``transfer_provenance.served_positions: null`` — and a block whose
``header_located`` / ``header_check_class`` were derived from that null, fail closed — because the served driver
read the standup's LENS map only (``lens_served_positions``), which a native standup writes EMPTY; the fact lived
only in the routes' ``petri_position_mode`` attestation. Now ``proxy.export_routing_table`` stamps ONE map for both
engines beside the lens map (``proxy.SERVED_POSITIONS_KEY``): the lens legs' stated modes (the same entries as the
lens map, which stays for old readers) and, per lora ROUTE, the ATTESTED position mode — derived from the route's
own attestation, never from the engine's name; a route attesting none is absent and every reader refuses it by
name. The loader validates it (routed cells, the lens + native vocabularies, agreement with the lens map), both
standups copy it into endpoint.json iff the table carries it, and the driver reads it FIRST with the lens map as
the fallback for an older record (``tests/test_served_capability_thinking_on.py``). The native standup derives a
PLAIN LoRA adapter's mode as ``all`` from its KIND and a residual-slot adapter's from its own block
(``native_standup.attested_position_mode``). Synthetic data only (the proxy tests' demo cell set)."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402

from test_lora_route_adapter_block import GOLDEN, _blocks, _export, _lora_routes, _world  # noqa: E402
from test_petri_proxy import CANONICAL_22, LENS, _cells, _positioned_source, _spec  # noqa: E402

KEY = px.SERVED_POSITIONS_KEY


def _v2(tmp_path, name="v2.json", **modes):
    """A lora-native (version-2) export of the demo cell set with ``position_modes`` = the named lora cells' modes."""
    _spec_, source, cells = _world(tmp_path)
    routes = px.routes_from_cells(cells)
    lora = {r.cell_id: r for r in _lora_routes(routes)}
    position_modes = {lora[c].cell_id: m for c, m in modes.items()}
    path, cells = _export(tmp_path, name, adapter_blocks=_blocks(routes), position_modes=position_modes or None)
    return path, json.loads(path.read_text()), lora


def test_the_constants_are_one_literal_across_the_proxy_the_contracts_the_native_standup_and_the_reader():
    assert KEY == "served_positions" == tc.SERVED_POSITIONS_KEY == ns.SERVED_POSITIONS_KEY
    assert KEY != px.LENS_SERVED_POSITIONS_KEY == tc.LENS_SERVED_POSITIONS_KEY
    assert KEY in ns._CORE_RECORD_KEYS and px.LENS_SERVED_POSITIONS_KEY not in ns._CORE_RECORD_KEYS
    assert ns.POSITION_MODE_ALL == "all" == dc.NATIVE_POSITION_MODES[0] and "all" in px.POSITION_MODES
    # the GSM8K reader resolves a native card's positions off this very key (the reader half of the change)
    src = (REPO / "scripts" / "tinycap_gsm8k_validators.py").read_text()
    assert "SERVED_POSITIONS_KEY = C.SERVED_POSITIONS_KEY" in src


def test_a_native_table_stamps_each_lora_routes_attested_mode_into_the_neutral_map_and_leaves_unattested_routes_out(tmp_path):
    """The mixed-mode fixture: two lora routes attesting `all` and `response` are stamped, each as attested (a
    list-of-modes value, the lens map's shape); the zero adapter's route (no attestation) is absent from the map and
    carries no `petri_position_mode`; the lens map stays empty on a native table; the loader round-trips the map."""
    path, doc, lora = _v2(tmp_path, si_safety__lora_r1="all", si_constitution__lora_r1="response")
    assert doc["version"] == 2 and doc[px.LENS_SERVED_POSITIONS_KEY] == {}
    assert doc[KEY] == {"si_safety__lora_r1": ["all"], "si_constitution__lora_r1": ["response"]}
    by_cell = {r["cell_id"]: r for r in doc["routes"]}
    assert by_cell["si_safety__lora_r1"]["petri_position_mode"] == "all" and by_cell["si_constitution__lora_r1"]["petri_position_mode"] == "response"
    assert "petri_position_mode" not in by_cell[dc.NOOP_LORA_CELL_ID] and dc.NOOP_LORA_CELL_ID not in doc[KEY]
    assert dc.NOOP_LORA_CELL_ID in lora                                    # the zero adapter IS a lora route — unattested, so absent
    table = px.load_routing_table(path)
    assert table[KEY] == doc[KEY] and table[px.LENS_SERVED_POSITIONS_KEY] == {}
    # the map is DERIVED from the stamped routes, the same fact twice
    assert px.served_positions_stamp(_cells(*_world(tmp_path)[:2]), _world(tmp_path)[1], table["routes"]) == doc[KEY]
    # a native table attesting NO mode on any route carries no map at all — the readers refuse such a cell by name
    plain, plain_doc, _ = _v2(tmp_path, "plain.json")
    assert KEY not in plain_doc and px.load_routing_table(plain)[KEY] is None
    # the v2-vs-v1 invariant the adapter-block tests pin still holds: nothing but the version and the adapters differ
    v1_path, _cells_ = _export(tmp_path, "v1.json")
    v1 = json.loads(v1_path.read_text())
    assert {k: v for k, v in plain_doc.items() if k not in ("version", px.ADAPTERS_KEY)} == {k: v for k, v in v1.items() if k != "version"}


def test_the_version_1_golden_is_byte_identical_and_a_lens_table_with_a_stated_mode_writes_the_same_entry_under_both_keys(tmp_path):
    """The pre-change exporter's bytes (`routing_table_v1_golden.json`) are unchanged: a table with no stated mode
    gains no key. A LENS standup whose psr cell's resolved spec STATES `all` writes the neutral map beside the lens map
    with the SAME entry (one shape for both engines), and the loader returns both."""
    path, _cells_ = _export(tmp_path, "golden_check.json")
    assert path.read_bytes() == GOLDEN.read_bytes() and KEY not in json.loads(path.read_text())
    spec = _spec(serving_mode="interleaved", cells=list(CANONICAL_22))
    source = _positioned_source(tmp_path, "all")
    cells = _cells(spec, source)
    lens_path = tmp_path / "lens.json"
    px.export_routing_table(cells, source, lens_path, base_model=spec.target_model, serving_mode="interleaved")
    doc = json.loads(lens_path.read_text())
    assert doc["version"] == 1 and doc[px.LENS_SERVED_POSITIONS_KEY] == {LENS: ["all"]} == doc[KEY]
    table = px.load_routing_table(lens_path)
    assert table[KEY] == table[px.LENS_SERVED_POSITIONS_KEY] == {LENS: ["all"]}
    # a lens leg that also "attests" a route mode is not one standup's shape: the stamp refuses before a byte is written
    routes = px.routes_from_cells(cells)
    torn = tuple(dataclasses.replace(r, petri_position_mode="all") if r.cell_id == LENS else r for r in routes)
    with pytest.raises(px.ProxyError, match=r"named by the lens map .* AND attests"):
        px.served_positions_stamp(cells, source, torn)


def test_the_loader_refuses_a_neutral_map_it_cannot_bind_to_the_routes_or_classify(tmp_path):
    """Known-bad, each by name: a cell no route serves, a mode outside the lens + native vocabularies, a scalar or
    empty value, a non-mapping, and a value that disagrees with the lens map's entry for the same cell."""
    path, doc, _lora = _v2(tmp_path, si_safety__lora_r1="all")

    def refuses(edit, needle, name):
        bad = json.loads(json.dumps(doc))
        edit(bad)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(px.ProxyError, match=needle):
            px.load_routing_table(p)
    refuses(lambda d: d.__setitem__(KEY, {"nobody": ["all"]}), "which no route serves", "unrouted")
    refuses(lambda d: d.__setitem__(KEY, {"si_safety__lora_r1": ["sideways"]}), "outside", "vocabulary")
    refuses(lambda d: d.__setitem__(KEY, {"si_safety__lora_r1": "all"}), "non-empty list", "scalar")
    refuses(lambda d: d.__setitem__(KEY, {"si_safety__lora_r1": []}), "non-empty list", "empty")
    refuses(lambda d: d.__setitem__(KEY, ["all"]), "mapping or None", "shape")
    refuses(lambda d: d.__setitem__(px.LENS_SERVED_POSITIONS_KEY, {"si_safety__lora_r1": ["response"]}), "disagree", "disagree")
    # the native vocabulary IS admitted (a stage-2 restricted mode rides the map as attested)
    ok = json.loads(json.dumps(doc))
    ok[KEY] = {"si_safety__lora_r1": ["prefill_all"]}
    ok["routes"] = [{**r, "petri_position_mode": "prefill_all"} if r["cell_id"] == "si_safety__lora_r1" else r for r in ok["routes"]]
    (tmp_path / "native_mode.json").write_text(json.dumps(ok))
    assert px.load_routing_table(tmp_path / "native_mode.json")[KEY] == {"si_safety__lora_r1": ["prefill_all"]}


def test_the_native_standup_attests_a_plain_lora_adapters_mode_from_its_kind_and_a_residual_adapters_from_its_block():
    """`attested_position_mode`: a residual-slot adapter's mode is its `petri_residual_slot` block's (read off the digest —
    `all` / `response` / `prompt_last` / `prefill_all`, stamped as attested); a PLAIN LoRA adapter (a kind outside
    RESIDUAL_SLOT_KINDS, no residual block by construction) attests `all` from its kind; a residual-kind adapter whose
    block states no mode stays None — the route carries no attestation, the map no entry, and the readers refuse by
    name — never `all` from the engine's name."""
    Dig = dataclasses.make_dataclass("Dig", [("petri_position_mode", object, None)])
    for kind in ("lora_all", "lora_subtraction", "circuit_breaker", "lora", "lora_r4"):
        assert kind not in dc.RESIDUAL_SLOT_KINDS
        assert ns.attested_position_mode(Dig(None), kind=kind) == "all"
    for kind in dc.RESIDUAL_SLOT_KINDS:
        assert ns.attested_position_mode(Dig(None), kind=kind) is None
        for mode in dc.NATIVE_POSITION_MODES:
            assert ns.attested_position_mode(Dig(mode), kind=kind) == mode
    # an attested mode on a plain LoRA (a residual block on a lora config would be the export's defect, not this rule's)
    assert ns.attested_position_mode(Dig("response"), kind="lora_all") == "response"
    # the standup's read site derives per cell from the digest AND the kind, and the endpoint copies the table's map
    src = (REPO / "src" / "sisteer" / "capeval" / "serving" / "native_standup.py").read_text()
    assert "position_modes[cell.cell_id] = attested_position_mode(dig, kind=str(entry.kind))" in src
    assert 'getattr(dig, "petri_position_mode", None)\n        reused' not in src
    assert "self.table.get(SERVED_POSITIONS_KEY)" in src
    # ... and so does the lens standup's, iff the table carries one
    cap = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    assert 'served_positions=table.get("served_positions")' in cap and '"served_positions": self.served_positions' in cap


def test_the_contracts_reader_reads_the_neutral_map_and_none_on_an_older_record():
    ep = {KEY: {"si_x__lora_r4": ["all"], "si_y__psr": "response"}, px.LENS_SERVED_POSITIONS_KEY: {}}
    assert tc.recorded_served_positions_of_cell(ep, "si_x__lora_r4") == ["all"]
    assert tc.recorded_served_positions_of_cell(ep, "si_y__psr") == ["response"]
    assert tc.recorded_served_positions_of_cell(ep, "si_z__none") is None
    assert tc.recorded_served_positions_of_cell({px.LENS_SERVED_POSITIONS_KEY: {"si_x__lora_r4": ["all"]}}, "si_x__lora_r4") is None
    assert tc.recorded_served_positions_of_cell(None, "si_x__lora_r4") is None and tc.recorded_served_positions_of_cell({KEY: None}, "x") is None
    assert tc.served_positions_of_cell(ep, "si_x__lora_r4") is None          # the lens reader is untouched: it reads the lens map only


def test_the_unclassified_native_mode_refusal_is_one_name_at_both_read_sites():
    """A native-vocabulary mode outside the lens vocabulary (`prompt_last` / `prefill_all`) is what `contracts.header_located`
    reads as header-located, fail closed — a WRONG POSITIVE if stamped. `unclassified_native_modes` names exactly those; `all`
    and `response` (shared by both vocabularies) are classifiable. Both read sites refuse by the contract's ONE name."""
    assert tc.NATIVE_POSITION_MODE_UNCLASSIFIED == "NATIVE_POSITION_MODE_UNCLASSIFIED"
    shared = [m for m in dc.NATIVE_POSITION_MODES if m in px.POSITION_MODES]
    native_only = [m for m in dc.NATIVE_POSITION_MODES if m not in px.POSITION_MODES]
    assert shared == ["all", "response"] and native_only == ["prompt_last", "prefill_all"]
    assert tc.unclassified_native_modes(native_only) == native_only and tc.unclassified_native_modes(shared) == []
    assert tc.unclassified_native_modes("prefill_all") == ["prefill_all"] and tc.unclassified_native_modes(["all", "prompt_last"]) == ["prompt_last"]
    assert tc.unclassified_native_modes(None) == tc.unclassified_native_modes(7) == tc.unclassified_native_modes(["sideways"]) == []
    for m in native_only:
        assert tc.header_located([m]) is True                                  # the fail-closed reading the refusal pre-empts
    driver = (REPO / "scripts" / "served_capability_run.py").read_text()
    reader = (REPO / "scripts" / "tinycap_gsm8k_validators.py").read_text()
    assert "tc.unclassified_native_modes(modes)" in driver and "tc.NATIVE_POSITION_MODE_UNCLASSIFIED" in driver
    assert reader.count("C.unclassified_native_modes(resolved)") == 2 and "NATIVE_POSITION_MODE_UNCLASSIFIED = C.NATIVE_POSITION_MODE_UNCLASSIFIED" in reader
