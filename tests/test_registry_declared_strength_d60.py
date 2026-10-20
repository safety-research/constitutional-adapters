"""The strength-ladder adapter's declared records, on SYNTHETIC fixtures only:

* DISCOVERY — the default record set is enumerated from ``<root>/*/battery_curves.json`` (+ sibling MMLU tables), so a record
  directory that appears later is read with no code edit; the two paper sources keep their pinned spellings, any other
  directory is ``strength_lane_<dir>``; arm roster, lane and transfer tags are resolved from the record itself.
* LINEAGE — the same (model, arm, dose) from two records with different serving lineage shas is TWO runs (two run_ids, two
  code_shas); ``code_sha`` is the store's wave sha, the battery tool's sha rides ``extra.record_tool_git_sha``; a cell whose
  record says its stores match the expected lineage while naming a stray wave sha is refused.
* THE DOSE UNIT — filed from the record on steered rows (cell -> view -> provenance), the column default on unsteered rows,
  refused outside ``vocab.STRENGTH_UNITS``; the MMLU dose table takes the sibling battery record's unit.

Fixtures are invented (tests/registry_declared_fixtures.py, post-edited here for the fields the builder lacks); placeholder
shas, hostname TESTHOST; nothing here reads a real store.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V
from sisteer.registry.adapters import strength_ladder as SL
from sisteer.registry.adapters.base import AdapterContext
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-17T06:00:00.000Z"
PRODUCER = {"script": "tests.declared_strength_d60", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
LINEAGE_ONE = hashlib.sha1(b"synthetic serving lineage one").hexdigest()
LINEAGE_TWO = hashlib.sha1(b"synthetic serving lineage two").hexdigest()
STRAY = hashlib.sha1(b"synthetic stray wave sha").hexdigest()
CONST = "constv2-mix-full__lora_r4_ce"
DIFF = "constv2-mix-full__lora_r4_ce_minus_ctl"


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER))


def _read_all(adapter):
    """discover -> comparison groups -> read; every record validated (as tests/test_registry_adapter_declared.py::_read_all does)."""
    inputs = adapter.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(adapter.comparison_group(i), []).append(i)
    recs = []
    ctx = AF.primed_ctx(adapter, now=NOW, producer=PRODUCER, groups=groups)     # the frame of the pass AFTER the fold: every own anchor current
    for g, ins in sorted(groups.items()):
        for batch in adapter.read(ctx, g, ins):
            recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    return recs


def _by_type(recs):
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save(path, d):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh)


def _d60ify(path: str, lineage: str, *, unit=None, arms: bool = True, matches: bool = True, cell_same_instrument: bool = True) -> None:
    """Post-edit a ``build_strength`` record with the fields the builder lacks: the record's expected serving lineage, each
    target's ``arms`` roster, per-cell ``provenance.wave_sha_by_store`` + ``provenance.same_instrument``, the view-level wave
    shas, and (optionally) a dose unit on cell, view and provenance."""
    d = _load(path)
    d["expected_serving_lineage_sha"] = lineage
    d["expected_serving_lineage_shas"] = [lineage]
    d["lineage_equivalence_rule"] = "synthetic: a wave sha is the same instrument as the head iff it equals the head"
    for _model, t in d["targets"].items():
        if arms:
            t["arms"] = sorted({c["cell_id"] for c in t["cells"].values()})
        for _v, si in t["same_instrument"].items():
            si["wave_shas"] = {lineage: 11}
            si["matches_expected_lineage"] = matches
            si["flag"] = None if matches else "synthetic lineage flag"
        for v in t["views"]:
            if unit is not None:
                v["dose_unit"] = unit
        for _key, c in t["cells"].items():
            prov = c["provenance"]
            if unit is not None:
                c["dose_unit"] = unit
                prov["dose_unit"] = unit
            prov["wave_sha_by_store"] = {k: lineage for k in prov["serving_by_store"]}
            if cell_same_instrument:
                prov["same_instrument"] = {"all_rungs_same_wave_sha": True, "wave_shas": [lineage], "matches_expected_lineage": matches,
                                           "expected_lineage_shas": [lineage], "flag": None if matches else "synthetic lineage flag"}
    _save(path, d)


def _build_b2i(root: str, lineage: str, views=(("d0p5", 0.5), ("d1", 1.0)), unit=None, **kw) -> dict:
    fx = F.build_strength(root, source="synthetic", views=views, engine="proxy_vllm+lora+interleaved", **kw)
    _d60ify(fx["path"], lineage, unit=unit)
    return fx


# ============================================================================================
# discovery
# ============================================================================================

def test_discovery_enumerates_record_directories_without_a_code_edit(tmp_path):
    root = tmp_path / "strength"
    fx_lora = F.build_strength(str(root / "lora"))                                  # it_* views: the paper's first source
    _d60ify(fx_lora["path"], LINEAGE_ONE)
    _build_b2i(str(root / "lora_base2it_native"), LINEAGE_ONE)
    F.build_strength_mmlu_table(str(root / "lora_base2it_native"))
    ad = SL.StrengthLadderAdapter(root=str(root))
    # the paper's two spellings kept; the MMLU table discovered beside its battery record; roots are the record directories
    assert ad.covers_sources == ("strength_lane", "strength_lane_base2it_native", "strength_lane_base2it_native_mmlu")
    assert ad.roots == (str(root / "lora"), str(root / "lora_base2it_native"))
    assert len(ad.discover()) == 3
    assert all(r.from_record for r in ad.records)
    # a NEW record directory appears (the gemma hi-dose ladder): the SAME adapter instance sees it on the next pass, no edit
    fx_hi = _build_b2i(str(root / "lora_base2it_native_gemma_hidose"), LINEAGE_TWO, views=(("d2", 2.0), ("d4", 4.0)))
    assert "strength_lane_lora_base2it_native_gemma_hidose" in ad.covers_sources
    assert str(root / "lora_base2it_native_gemma_hidose") in ad.roots
    assert len(ad.discover()) == 4
    recs = _read_all(ad)
    runs = _by_type(recs)["run"]
    hi = [r for r in runs if r["lane_source_legacy"] == "strength_lane_lora_base2it_native_gemma_hidose"]
    assert hi, "the discovered record emitted nothing"
    # lane / arms / transfer tag resolved FROM the record: d* views of a transfer family -> base->instruct, tagged with the model key
    res = ad.resolved(fx_hi["path"])
    assert res.lane_legacy == "base→instruct" and res.arms == ("const_lora", "diff_lora") and res.transfer_tag == {"gemma4e4b": "gemma4e4b"}
    assert {r["extra"]["paper_lane_kind"] for r in hi} == {"base→instruct"} and {r["view_legacy"] for r in hi} == {"gemma4e4b"}
    assert {r["strength_nominal"] for r in hi if r["object_id"] != "none"} == {2.0, 4.0}
    assert {r["code_sha"] for r in hi} == {LINEAGE_TWO}
    # the first record's it_* views read as instruct-direct throughout; the pinned source spelling rides every row
    assert ad.resolved(fx_lora["path"]).lane_legacy == "instruct-direct"
    lo = [r for r in runs if r["lane_source_legacy"] == "strength_lane"]
    assert lo and {r["code_sha"] for r in lo} == {LINEAGE_ONE} and {r["transfer_kind"] for r in lo if r["object_id"] != "none"} == {"instruct_direct"}
    # the MMLU table's declaration is resolved from the record too (its cells are its roster; d* views -> base->instruct)
    mm = ad.resolved(str(root / "lora_base2it_native" / "mmlu_dose_table.json"))
    assert mm.lane_legacy == "base→instruct" and mm.arms == ("const_lora", "ctl_lora")


def test_default_root_without_records_discovers_nothing(tmp_path):
    ad = SL.StrengthLadderAdapter(root=str(tmp_path / "absent"))
    assert ad.records == () and ad.covers_sources == () and ad.roots == () and ad.discover() == []


def test_source_spelling_is_the_pinned_one_for_the_paper_dirs_and_derived_elsewhere():
    assert SL.source_for_dir("lora") == "strength_lane"
    assert SL.source_for_dir("lora_base2it_native") == "strength_lane_base2it_native"
    assert SL.source_for_dir("relu_qwen") == "strength_lane_relu_qwen"


# ============================================================================================
# lane / arms from the record
# ============================================================================================

def test_lane_and_arms_come_from_the_record_or_the_record_is_refused(tmp_path):
    # llama with an it_ view AND a d view: two lane kinds in one record -> lane per view (None), as the paper's mixed record
    fx = F.build_strength(str(tmp_path / "mix"), model="llama8b", views=(("it_d1", 1.0), ("d1", 1.0)))
    _d60ify(fx["path"], LINEAGE_ONE)
    decl = SL.discover_records(str(tmp_path))[0]
    ad = SL.StrengthLadderAdapter(records=(decl,))
    rows, facts = ad.long_rows(decl)
    assert ad.resolved(fx["path"]).lane_legacy is None and facts["lane"] is None
    assert {r["lane_kind"] for r in rows} == {"instruct-direct", "base→instruct"}
    assert {r["view"] for r in rows} == {"llama8b_it_doc", "llama8b"}          # the transfer tag is the model key, as at the pin
    assert facts["arms"] == ["const_lora", "diff_lora"]
    # an EXPLICIT declaration still checks its lane against every view (the pinned reader's rule is kept)
    strict = SL.StrengthRecordDecl(fx["path"], "strength_lane_x", ("const_lora", "diff_lora"), "instruct-direct", {"llama8b": "llama8b"})
    with pytest.raises(SL.RecordIncoherent):
        SL.StrengthLadderAdapter(records=(strict,)).long_rows(strict)
    # a view outside the writer's grammar: the lane cannot be determined -> refused
    d = _load(fx["path"])
    d["targets"]["llama8b"]["views"].append({"view": "weird_view", "dose": 3.0, "dose_unit": "lora_write_basis_multiplier", "cells": {}})
    _save(fx["path"], d)
    with pytest.raises(SL.RecordIncoherent, match="grammar"):
        SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    d["targets"]["llama8b"]["views"].pop()
    # a target without its `arms` roster: refused, not guessed from the cells
    del d["targets"]["llama8b"]["arms"]
    _save(fx["path"], d)
    with pytest.raises(SL.RecordIncoherent, match="arms"):
        SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    # an arm cell id this paper does not know: refused
    d["targets"]["llama8b"]["arms"] = [CONST, "somebody_elses__arm"]
    _save(fx["path"], d)
    with pytest.raises(SL.RecordIncoherent, match="not an arm"):
        SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    # no target this paper knows: no view to read a lane from -> refused
    d["targets"] = {"unknown_family": d["targets"]["llama8b"]}
    _save(fx["path"], d)
    with pytest.raises(SL.RecordIncoherent, match="lane cannot be determined"):
        SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)


# ============================================================================================
# lineage is a first-class run field
# ============================================================================================

def test_same_point_from_two_lineages_is_two_runs_with_two_code_shas(tmp_path):
    """The same (model, arm, dose) under the SAME source label from two records at different serving lineages: two run_ids, two
    code_shas — the dose alone never identifies a point."""
    a = _build_b2i(str(tmp_path / "a"), LINEAGE_ONE, views=(("d1", 1.0),))
    b = _build_b2i(str(tmp_path / "b"), LINEAGE_TWO, views=(("d1", 1.0),))
    mk = lambda p: SL.StrengthRecordDecl(p, "strength_lane_synth", ("const_lora", "diff_lora"), "base→instruct", {"gemma4e4b": "gemma4e4b"})  # noqa: E731
    ad = SL.StrengthLadderAdapter(records=(mk(a["path"]), mk(b["path"])))
    recs = _read_all(ad)
    bt = _by_type(recs)
    runs = bt["run"]
    pt = [r for r in runs if r["campaign_cell_id_legacy"] == CONST and r["eval_id"] == "pasttense" and r["strength_nominal"] == 1.0]
    assert len(pt) == 2, "one point per record expected"
    assert len({r["run_id"] for r in pt}) == 2 and {r["code_sha"] for r in pt} == {LINEAGE_ONE, LINEAGE_TWO}
    assert len({r["producer_run_key_legacy"] for r in pt}) == 1              # identical producer key: only the lineage separates them
    assert len({r["instrument_id"] for r in pt}) == 2                        # the wave sha is the instrument's harness_version
    for r in runs:
        assert r["code_sha"] in (LINEAGE_ONE, LINEAGE_TWO) and r["harness_version"] == r["code_sha"]
        assert r["extra"]["serving_lineage_sha"] == r["code_sha"] and r["extra"]["store_wave_sha"] == r["code_sha"]
        assert r["extra"]["lineage_matches_expected"] is True
        assert r["extra"]["record_tool_git_sha"] == F.CODE_SHA and r["code_sha"] != F.CODE_SHA     # the tool's sha is NOT the run's
        assert "code_sha" not in r["field_confidence_json"]                    # a filed sha carries no unverified entry
    # the contrast names both stores' wave shas (the arm's and its same-standup anchor's) and the record's lineage
    for c in bt["run_contrast"]:
        run = next(r for r in runs if r["run_id"] == c["run_id"])
        assert c["extra"]["arm_wave_sha"] == c["extra"]["anchor_wave_sha"] == run["code_sha"] == c["extra"]["serving_lineage_sha"]
    # a record naming SEVERAL expected lineage shas files the list
    d = _load(a["path"])
    d["expected_serving_lineage_shas"] = [LINEAGE_ONE, LINEAGE_TWO]
    d["expected_serving_lineage_sha"] = ",".join([LINEAGE_ONE, LINEAGE_TWO])
    _save(a["path"], d)
    rows, facts = ad.long_rows(mk(a["path"]))
    assert facts["expected_lineage"] == sorted([LINEAGE_ONE, LINEAGE_TWO])
    assert all(r["_extra"]["serving_lineage_sha"] == sorted([LINEAGE_ONE, LINEAGE_TWO]) for r in rows if r.get("_extra"))


def test_store_wave_sha_falls_back_to_the_cell_then_the_view_and_files_unverified_when_absent(tmp_path):
    fx = _build_b2i(str(tmp_path / "a"), LINEAGE_ONE, views=(("d1", 1.0),))
    decl = SL.discover_records(str(tmp_path))[0]
    d = _load(fx["path"])
    cells = d["targets"]["gemma4e4b"]["cells"]
    # no per-store map, one sha over the cell -> the cell's
    del cells[f"d1:{CONST}"]["provenance"]["wave_sha_by_store"]
    # no per-store map, no cell summary -> the view's
    del cells[f"d1:{DIFF}"]["provenance"]["wave_sha_by_store"]
    del cells[f"d1:{DIFF}"]["provenance"]["same_instrument"]
    _save(fx["path"], d)
    rows, _ = SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    measured = [r for r in rows if r.get("status") == "graded"]
    assert measured and all(r["_code_sha"] == LINEAGE_ONE for r in measured)
    srcs = {r["cell_id"]: r["_extra"]["store_wave_sha_source"] for r in measured if r["cell_id"] != "none"}
    assert "same_instrument.wave_shas" in srcs[CONST] and "<view>" in srcs[DIFF]
    # nothing names a sha anywhere -> unverified with the reason, never a guess
    for c in cells.values():
        c["provenance"].pop("wave_sha_by_store", None)
        c["provenance"].pop("same_instrument", None)
    d["targets"]["gemma4e4b"]["same_instrument"] = {}
    _save(fx["path"], d)
    rows, _ = SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    measured = [r for r in rows if r.get("status") == "graded"]
    assert all(r["_code_sha"] == "unverified" and "names no wave sha" in r["_code_sha_reason"] for r in measured)


def test_refuses_a_lineage_cross_check_the_record_contradicts(tmp_path):
    fx = _build_b2i(str(tmp_path / "a"), LINEAGE_ONE, views=(("d1", 1.0),))
    decl = SL.discover_records(str(tmp_path))[0]
    d = _load(fx["path"])
    prov = d["targets"]["gemma4e4b"]["cells"][f"d1:{CONST}"]["provenance"]
    # the record SAYS the stores match the expected lineage, yet one store's wave sha is outside it: the record contradicts itself
    prov["wave_sha_by_store"][f"cap4096/{CONST}"] = STRAY
    _save(fx["path"], d)
    with pytest.raises(SL.RecordIncoherent, match=STRAY):
        SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    # the record that says they do NOT match is READ, and the flag rides the run (no rule of our own is applied)
    prov["same_instrument"]["matches_expected_lineage"] = False
    prov["same_instrument"]["flag"] = "synthetic: this cell was re-served at a later sha"
    d["targets"]["gemma4e4b"]["same_instrument"]["d1"]["matches_expected_lineage"] = False
    _save(fx["path"], d)
    ad = SL.StrengthLadderAdapter(records=(decl,))
    recs = _read_all(ad)
    runs = _by_type(recs)["run"]
    const = [r for r in runs if r["campaign_cell_id_legacy"] == CONST]
    assert const and all(r["extra"]["lineage_matches_expected"] is False for r in const)
    stray = [r for r in const if r["eval_id"] != "xstest"]                      # the cap-4096 static stores carry the stray sha
    assert stray and all(r["code_sha"] == STRAY for r in stray)
    assert ad.last_facts["strength_lane_a"]["lineage_flags"]
    # a record that names NO expected lineage is not second-guessed: the view's word is mirrored, nothing refused
    d2 = _load(fx["path"])
    for k in ("expected_serving_lineage_sha", "expected_serving_lineage_shas"):
        d2.pop(k, None)
    for c in d2["targets"]["gemma4e4b"]["cells"].values():
        c["provenance"]["same_instrument"].pop("expected_lineage_shas", None)
        c["provenance"]["same_instrument"]["matches_expected_lineage"] = True
    _save(fx["path"], d2)
    rows, facts = SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    assert facts["expected_lineage"] == [] and all(r["_extra"]["serving_lineage_sha"] is None for r in rows if r.get("_extra"))


# ============================================================================================
# the dose unit
# ============================================================================================

def test_strength_unit_is_filed_from_the_record_on_steered_rows_and_defaults_on_none(tmp_path):
    fx = _build_b2i(str(tmp_path / "relu"), LINEAGE_ONE, views=(("d0p5", 0.5), ("d1", 1.0)), unit="psr_direction_norm_multiplier")
    ad = SL.StrengthLadderAdapter(root=str(tmp_path))
    recs = _read_all(ad)
    runs = _by_type(recs)["run"]
    steered = [r for r in runs if r["object_id"] != "none"]
    none = [r for r in runs if r["object_id"] == "none"]
    assert steered and none
    assert {r["strength_unit"] for r in steered} == {"psr_direction_norm_multiplier"}
    assert all("strength_unit" not in r["field_confidence_json"] for r in steered)               # filed from the record: nothing unverified
    assert {r["strength_unit"] for r in none} == {"unverified"}                                  # no strength on an unsteered row: the column default
    assert all(r["field_confidence_json"]["strength_unit"]["confidence"] == "unverified" for r in none)
    assert all(r["extra"]["dose_unit"] == "psr_direction_norm_multiplier" for r in steered)
    # the fallback chain: cell -> view -> provenance
    d = _load(fx["path"])
    t = d["targets"]["gemma4e4b"]
    for c in t["cells"].values():
        del c["dose_unit"]                                                       # the view still declares it
    _save(fx["path"], d)
    decl = ad.records[0]
    rows, _ = ad.long_rows(decl)
    assert {r["_strength_unit"] for r in rows if r.get("status") == "graded" and r["cell_id"] != "none"} == {"psr_direction_norm_multiplier"}
    for v in t["views"]:
        del v["dose_unit"]                                                       # only the provenance declares it
    _save(fx["path"], d)
    rows, _ = ad.long_rows(decl)
    assert {r["_strength_unit"] for r in rows if r.get("status") == "graded" and r["cell_id"] != "none"} == {"psr_direction_norm_multiplier"}
    assert all(r["_strength_unit"] is None for r in rows if r.get("status") == "graded" and r["cell_id"] == "none")


def test_refuses_a_unit_outside_the_vocabulary_or_absent(tmp_path):
    fx = _build_b2i(str(tmp_path / "a"), LINEAGE_ONE, views=(("d1", 1.0),), unit="furlongs_per_fortnight")
    decl = SL.discover_records(str(tmp_path))[0]
    assert "furlongs_per_fortnight" not in V.STRENGTH_UNITS
    with pytest.raises(SL.RecordIncoherent, match="STRENGTH_UNITS"):
        SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)
    d = _load(fx["path"])
    t = d["targets"]["gemma4e4b"]
    for c in t["cells"].values():
        c.pop("dose_unit", None)
        c["provenance"].pop("dose_unit", None)
    for v in t["views"]:
        v.pop("dose_unit", None)
    _save(fx["path"], d)
    with pytest.raises(SL.RecordIncoherent, match="no dose_unit"):
        SL.StrengthLadderAdapter(records=(decl,)).long_rows(decl)


def test_mmlu_rows_take_the_sibling_battery_units(tmp_path):
    root = tmp_path / "strength"
    _build_b2i(str(root / "lora_base2it_native"), LINEAGE_ONE, unit="psr_direction_norm_multiplier")
    F.build_strength_mmlu_table(str(root / "lora_base2it_native"))
    ad = SL.StrengthLadderAdapter(root=str(root))
    mm = next(r for r in ad.records if r.kind == "mmlu_table")
    rows, _ = ad.long_rows(mm)
    measured = [r for r in rows if r.get("status") == "graded"]
    assert measured
    assert {r["_strength_unit"] for r in measured if r["cell_id"] != "none"} == {"psr_direction_norm_multiplier"}
    assert all(r["_strength_unit"] is None for r in measured if r["cell_id"] == "none")
    assert all("sibling battery" in r["_extra"]["dose_unit_source"] for r in measured if r["cell_id"] != "none")
    recs = _read_all(ad)
    runs = [r for r in _by_type(recs)["run"] if r["lane_source_legacy"] == "strength_lane_base2it_native_mmlu"]
    assert {r["strength_unit"] for r in runs if r["object_id"] != "none"} == {"psr_direction_norm_multiplier"}
    assert {r["strength_unit"] for r in runs if r["object_id"] == "none"} == {"unverified"}
    # the table's own declaration wins over the sibling's when it has one
    tab = _load(str(root / "lora_base2it_native" / "mmlu_dose_table.json"))
    tab["dose_unit"] = "lora_write_basis_multiplier"
    _save(str(root / "lora_base2it_native" / "mmlu_dose_table.json"), tab)
    rows, _ = ad.long_rows(mm)
    assert {r["_strength_unit"] for r in rows if r.get("status") == "graded" and r["cell_id"] != "none"} == {"lora_write_basis_multiplier"}
    assert all("table's own" in r["_extra"]["dose_unit_source"] for r in rows if r.get("status") == "graded" and r["cell_id"] != "none")
    # a table unit outside the vocabulary refuses
    tab["dose_unit"] = "not_a_unit"
    _save(str(root / "lora_base2it_native" / "mmlu_dose_table.json"), tab)
    with pytest.raises(SL.RecordIncoherent, match="STRENGTH_UNITS"):
        ad.long_rows(mm)
