"""native-psr-serving step 7 — the cross-component seams of stage 2, each
pinned with a known-bad: the residual entry's NATIVE position mode rides the
ROUTE (``RouteEntry.petri_position_mode`` — set by ``export_routing_table(position_modes=)`` from
``AdapterDigests.petri_position_mode``, read off the adapter's ``petri_residual_slot`` block, a
derived field outside the content identity), the salt-isolation report treats a restricted-mode
native route as request-unique BY DESIGN, the controls driver's stamp-regime expectation
takes the leg's mode, ``f22_roster`` admits the pin-referencing mapping form, and the smoke's
controls override (a fallback while the controls driver is incomplete) is positional, recorded and
off by default. Synthetic data only."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.serving import lora_export as lx  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402

SMOKE = REPO / "scripts/sdfbase/full22/smoke_full22.sbatch"


def _route(cell, *, mode=None, lens=False):
    ident = "proxy_vllm+lens+interleaved" if lens else "proxy_vllm+lora+interleaved"
    return px.RouteEntry(virtual_model=f"petri__{cell}", cell_id=cell, cell_class="class_a",
                         serving_identity=ident, intervention_hash="cc" * 8, petri_position_mode=mode)


def test_route_entry_carries_the_mode_and_the_loader_round_trips_with_and_without_it():
    r = _route("si_x__psr_kl_all", mode="response")
    row = dataclasses.asdict(r)
    assert row["petri_position_mode"] == "response" and px.RouteEntry(**row) == r
    # a stage-1 / lens table row has no such key: the loader's default is None (byte-compatible)
    old = {k: v for k, v in row.items() if k != "petri_position_mode"}
    assert px.RouteEntry(**old).petri_position_mode is None


def test_adapter_digests_read_the_residual_blocks_mode_as_a_derived_field(tmp_path):
    """``adapter_digests`` reads ``petri_residual_slot.position_mode`` (None on a LoRA / zero
    adapter); the digest FILE reader tolerates a stage-1 file without the key; the field is NOT
    in the content identity, so a stage-1 digest file still verifies against the same bytes."""
    d = tmp_path / "adapter"
    d.mkdir()
    cfg = {"r": 1, "petri_genuine_adapter": True, "petri_intervention_hash": "ab" * 8, "petri_source_kind": "psr",
           "petri_residual_slot": {"schema": 1, "kind": "psr", "position_mode": "prefill_all"}}
    (d / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg))
    (d / lx.ADAPTER_WEIGHTS).write_bytes(b"\x00" * 16)
    dig = lx.adapter_digests(d)
    assert dig.petri_position_mode == "prefill_all" and dig.petri_source_kind == "psr"
    assert "petri_position_mode" not in lx._DIGEST_CONTENT_FIELDS
    # a plain LoRA config: None
    cfg_l = {"r": 4, "petri_genuine_adapter": True, "petri_intervention_hash": "ab" * 8, "petri_source_kind": "lora_all"}
    (d / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg_l))
    assert lx.adapter_digests(d).petri_position_mode is None
    # known-bad: a malformed mode refuses by name
    cfg_bad = dict(cfg); cfg_bad["petri_residual_slot"] = {"position_mode": ""}
    (d / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg_bad))
    with pytest.raises(lx.LoraExportError, match="position_mode"):
        lx.adapter_digests(d)


def test_export_routing_table_stamps_the_mode_on_the_residual_routes_only(tmp_path):
    """``export_routing_table(position_modes=)`` on a lora-native (version-2) table puts the mode
    on exactly the lora routes named with a non-None mode — the ROW carries the key only there
    (every other row keeps the seven pre-change keys) — and the loader reads it back; the
    pre-change loader refuses the table by its VERSION, never by a key it does not know.
    Known-bads: a cell no route serves, a non-lora route, a mode outside the vocabulary."""
    from test_lora_route_adapter_block import _blocks, _export, _lora_routes, _old_loader
    _, cells = _export(tmp_path, "probe.json")
    routes = px.routes_from_cells(cells)
    blocks = _blocks(routes)
    lora = [r for r in _lora_routes(routes) if not r.cell_id.endswith(px.LORA_ZERO_CELL_SUFFIX)]
    named, other = lora[0], lora[1]
    path, _ = _export(tmp_path, "v2.json", adapter_blocks=blocks,
                      position_modes={named.cell_id: "response", other.cell_id: None})
    rows = {r["cell_id"]: r for r in json.loads(path.read_text())["routes"]}
    assert rows[named.cell_id]["petri_position_mode"] == "response"
    assert all("petri_position_mode" not in row for cid, row in rows.items() if cid != named.cell_id)
    table = px.load_routing_table(path)
    by = {r.cell_id: r.petri_position_mode for r in table["routes"]}
    assert by[named.cell_id] == "response" and all(m is None for cid, m in by.items() if cid != named.cell_id)
    with pytest.raises(px.ProxyError, match="version 2"):
        _old_loader(path)
    # known-bads, each refused by name before a byte is written
    base = [r for r in routes if r.cell_id == "none"][0]
    for modes, match in (({"nobody__psr_kl_all": "response"}, "no route serves"),
                         ({base.cell_id: "response"}, "not lora-served"),
                         ({named.cell_id: "sideways"}, "is not one of")):
        with pytest.raises(px.ProxyError, match=match):
            _export(tmp_path, "bad.json", adapter_blocks=blocks, position_modes=modes)
        assert not (tmp_path / "bad.json").exists()
    # the vocabulary the proxy holds the mode to is the residual slot's, verbatim
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import native_residual as nr
    assert px.NATIVE_POSITION_MODES == dc.NATIVE_POSITION_MODES == nr.POSITION_MODES


def test_a_stage1_bank_entry_without_the_derived_mode_key_is_reused_and_a_foreign_key_set_is_still_stale(tmp_path):
    """The digest FILE of every bank entry written before stage 2 lacks
    ``petri_position_mode`` — a DERIVED field outside the content identity — and
    ``_read_digests_doc`` held the key set to the current schema, so every stage-1
    entry read as "stale or tampered" and the standup REFUSED. The reader now takes the pre-stage-2
    record (and the one written before ``module_namespace``, and both absences together) as the
    record it was, the absent field None; the known-bads keep refusing: a record missing a CONTENT
    key, and one with a key this bank never wrote."""
    d = tmp_path / "entry"
    d.mkdir()
    cfg = {"r": 1, "petri_genuine_adapter": True, "petri_intervention_hash": "ab" * 8, "petri_source_kind": "psr",
           "petri_residual_slot": {"schema": 1, "kind": "psr", "position_mode": "all"}}
    (d / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg))
    (d / lx.ADAPTER_WEIGHTS).write_bytes(b"\x00" * 16)
    dig = lx.adapter_digests(d)
    full = {**{f: getattr(dig, f) for f in lx.AdapterDigests._fields}, "exported_at": 1.0, "exporter_tree_sha": "0" * 40,
            "module_namespace": None}
    full["dir"] = str(d)
    assert set(full) == set(lx.AdapterDigests._fields) | set(lx._DIGEST_FILE_EXTRA)

    def write(doc):
        (d / lx.ADAPTER_DIGESTS).write_text(json.dumps(doc))

    # the current record, the pre-stage-2 record, the record written before
    # module_namespace, and both absences: all read
    for drop in ((), ("petri_position_mode",), ("module_namespace",), ("petri_position_mode", "module_namespace")):
        write({k: v for k, v in full.items() if k not in drop})
        doc = lx._read_digests_doc(d)
        assert doc["module_namespace"] is None and doc["petri_position_mode"] == (None if "petri_position_mode" in drop else "all")
        rec = lx._read_digests_file(d)
        assert rec.petri_position_mode == (None if "petri_position_mode" in drop else "all")
        lx.verify_adapter_dir(d, rec)                       # the content identity holds without the derived field
    # known-bads: a CONTENT key missing, a key this bank never wrote
    write({k: v for k, v in full.items() if k != "weights_sha256"})
    with pytest.raises(lx.LoraExportError, match="stale or tampered bank entry.*keys"):
        lx._read_digests_doc(d)
    write({**full, "petri_extra": 1})
    with pytest.raises(lx.LoraExportError, match="stale or tampered bank entry.*keys"):
        lx._read_digests_doc(d)


def test_view_manifest_digests_take_a_pin_form_entrys_sha_from_the_authority_it_records(tmp_path):
    """The mt standup derives --expect-object-sha256 from the view
    manifest + the WAVE pins by staged id, so a CROSS-BANK pin-form entry (the Arditi rung, pinned in
    configs/misalign_baseline_pins.json) read as "not pinned". The pin-form entry now takes its sha from
    the pins file it records — the EXACT recorded path resolved against the checkout root (the first cut
    reduced a relative path to its basename beside the wave pins, so an entry naming an absent
    archive/misalign_baseline_pins.json would have read configs/misalign_baseline_pins.json) — held to the same
    three-way agreement; a string entry is unchanged; the known-bads — a decoy sibling, a missing pins file, a missing
    pin, a drifted object — still refuse by name."""
    import hashlib
    import mt_serving_up as mt
    view, cfg = tmp_path / "view", tmp_path / "configs"
    cfg.mkdir()
    (view / "lora").mkdir(parents=True); (view / "arditi").mkdir()
    lora_b, arditi_b = b"lora-bytes", b"arditi-bytes"
    (view / "lora" / "object.pt").write_bytes(lora_b); (view / "arditi" / "object.pt").write_bytes(arditi_b)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    (cfg / "sdfbase_full22_pins.json").write_text(json.dumps({"gemma4e4b": {"base__lora": {"object_sha256": sha(lora_b)}}}))
    (cfg / "misalign_baseline_pins.json").write_text(json.dumps({"gemma4e4b": {"arditi": {"object_sha256": sha(arditi_b), "staged_id": "ar_r0p05__arditi"}}}))
    manifest = {"entries": [
        {"cell_id": "lora", "staged_id": "base__lora", "object_sha256": sha(lora_b)},
        {"cell_id": "arditi", "form": "pin", "pins": "configs/misalign_baseline_pins.json", "pin": "gemma4e4b.arditi",
         "staged_id": "ar_r0p05__arditi", "object_sha256": sha(arditi_b)}]}
    (view / mt.VIEW_MANIFEST_FILE).write_text(json.dumps(manifest))
    pins = cfg / "sdfbase_full22_pins.json"
    kw = dict(pins_path=pins, target="gemma4e4b", repo_root=tmp_path)
    out = mt.view_manifest_digests(view, cells=["none", "lora", "arditi"], harness_ids=["none", "si_x__lora", "arditi"], **kw)
    assert out == [f"lora={sha(lora_b)}", f"arditi={sha(arditi_b)}"]
    # the checkout root defaults to the wave pins file's grandparent (<root>/configs/<pins>.json)
    assert mt.view_manifest_digests(view, pins_path=pins, target="gemma4e4b", cells=["arditi"], harness_ids=["arditi"]) == [f"arditi={sha(arditi_b)}"]
    # KNOWN-BAD: the EXACT recorded path is resolved — an entry naming an absent
    # archive/misalign_baseline_pins.json must not read the decoy sibling configs/misalign_baseline_pins.json
    decoy = dict(manifest["entries"][1]); decoy["pins"] = "archive/misalign_baseline_pins.json"
    (view / mt.VIEW_MANIFEST_FILE).write_text(json.dumps({"entries": [manifest["entries"][0], decoy]}))
    with pytest.raises(SystemExit, match=r"archive/misalign_baseline_pins.json.*no such file; refusing an unpinned object"):
        mt.view_manifest_digests(view, cells=["arditi"], harness_ids=["arditi"], **kw)
    (view / mt.VIEW_MANIFEST_FILE).write_text(json.dumps(manifest))
    # known-bads, by name
    (cfg / "misalign_baseline_pins.json").write_text(json.dumps({"gemma4e4b": {}}))
    with pytest.raises(SystemExit, match="has no gemma4e4b.arditi object_sha256"):
        mt.view_manifest_digests(view, cells=["arditi"], harness_ids=["arditi"], **kw)
    (cfg / "misalign_baseline_pins.json").unlink()
    with pytest.raises(SystemExit, match="no such file; refusing an unpinned object"):
        mt.view_manifest_digests(view, cells=["arditi"], harness_ids=["arditi"], **kw)
    (cfg / "misalign_baseline_pins.json").write_text(json.dumps({"gemma4e4b": {"arditi": {"object_sha256": "00" * 32}}}))
    with pytest.raises(SystemExit, match="disagree — a drifted object"):
        mt.view_manifest_digests(view, cells=["arditi"], harness_ids=["arditi"], **kw)
    # a string entry whose staged id is not in the wave pins still refuses as before
    (cfg / "sdfbase_full22_pins.json").write_text(json.dumps({"gemma4e4b": {}}))
    with pytest.raises(SystemExit, match="is not pinned in"):
        mt.view_manifest_digests(view, pins_path=pins, target="gemma4e4b", cells=["lora"], harness_ids=["si_x__lora"])


def test_salt_isolation_report_treats_a_restricted_native_route_as_request_unique_by_design():
    """At the pre-flight salt check, a restricted-mode native route mints request-unique salts
    (like a lens leg) and is ACCEPTED; a stable salt on it is the known-bad; an all-positions /
    LoRA route stays cell-stable; the report names the mode."""
    lora, psr_all, psr_resp = _route("si_a__lora_r4_ce"), _route("si_b__psr_ce_all", mode="all"), _route("si_c__psr_kl_all", mode="response")
    rep = bs.cell_salt_isolation_report([lora, psr_all, psr_resp])
    per = rep["per_cell"]
    assert per[lora.virtual_model]["stable"] is True and per[lora.virtual_model]["restricted_native"] is False
    assert per[psr_all.virtual_model]["stable"] is True and per[psr_all.virtual_model]["position_mode"] == "all"
    assert per[psr_resp.virtual_model]["stable"] is False and per[psr_resp.virtual_model]["restricted_native"] is True
    bs.assert_cell_salt_isolation([lora, psr_all, psr_resp])          # accepted
    # known-bad: a restricted route whose salt is STABLE
    stable_report = bs.cell_salt_isolation_report([lora, psr_resp])
    stable_report["per_cell"][psr_resp.virtual_model]["stable"] = True

    def fake_report(routes):
        return stable_report
    orig = bs.cell_salt_isolation_report
    bs.cell_salt_isolation_report = fake_report
    try:
        with pytest.raises(bs.BonServedError, match="restricted-mode native cell .* STABLE"):
            bs.assert_cell_salt_isolation([lora, psr_resp])
    finally:
        bs.cell_salt_isolation_report = orig


def test_stamp_regime_expectation_takes_the_legs_mode():
    from sisteer.capeval.tinycap import contracts as tc
    import served_capability_run as scr
    import inspect
    sig = inspect.signature(scr._probe_stamp_problem)
    assert "mode_by_leg" in sig.parameters
    assert "position_mode" in inspect.signature(tc.expected_stamp_regimes).parameters
    src = inspect.getsource(scr._probe_stamp_problem)
    assert 'position_mode=(mode_by_leg or {}).get(leg)' in src
    call = Path(scr.__file__).read_text()
    assert 'mode_by_leg={usr.PROBE_LEG_STEERED: getattr(route, "petri_position_mode", None)' in call


def test_f22_roster_admits_the_pin_form_and_excludes_it_from_the_staged_ids():
    import f22_roster as fr
    fr._check_cell("v", "arditi", {"pins": "configs/misalign_baseline_pins.json", "pin": "gemma4e4b.arditi"})
    with pytest.raises(ValueError, match="mapping form takes exactly"):
        fr._check_cell("v", "arditi", {"pins": "p", "pin": "x", "staged_id": "ar_r0p1__arditi"})
    with pytest.raises(ValueError, match="mapping form takes exactly"):
        fr._check_cell("v", "arditi", {"pins": "p"})
    with pytest.raises(ValueError, match="prompted cell"):
        fr._check_cell("v", "prompted__si_x", {"pins": "p", "pin": "x"})
    roster = {"targets": {"g": {"views": {"psr2cert": {"cells": {"a__psr_kl_all": "si12__a__psr_kl_all",
                                                                 "arditi": {"pins": "p", "pin": "g.arditi"},
                                                                 "prompted__x": None}}},
                               "lanes": {}}}}
    assert fr.every_staged_id(roster, "g") == ["si12__a__psr_kl_all"]
    assert fr.is_pin_form({"pins": "p", "pin": "x"}) and not fr.is_pin_form("sid") and not fr.is_pin_form({"pins": "p"})


def test_smoke_controls_override_is_positional_recorded_and_off_by_default():
    subprocess.run(["bash", "-n", str(SMOKE)], check=True)
    src = SMOKE.read_text()
    assert 'CTL_CONTROLS_OVERRIDE="${6:-}"' in src
    assert 'VIEW_CONTROLS="$CTL_CONTROLS_OVERRIDE"' in src
    assert 'controls=$VIEW_CONTROLS${CTL_CONTROLS_OVERRIDE:+ (OVERRIDE)}; ' in src
    # the override changes the VALUE of --controls only: no driver flag is added to the smoke's argv (the wave test
    # pins the smoke's driver argv equal to the wave's), so a driver policy the wave would refuse under is not
    # smoked over — no `--sampling` escape hatch here
    assert "--sampling" not in src and "CTL_SAMPLING" not in src
    # the wave's controls read is unchanged when the positionals are empty (the override is additive)
    assert 'VIEW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" controls --roster "$F22_ROSTER")' in src
    # the hard rules: no sbatch export flag, no --mem=0 (the usage COMMENT may name the flag it forbids)
    import re
    assert not re.search(r"^#SBATCH --export|^\s*sbatch .*--export=", src, re.M) and "--mem=0" not in src
