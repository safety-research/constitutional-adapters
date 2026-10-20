"""GSM8K paired-200 adapter, the directory inventory, on SYNTHETIC roots.

Each test builds a root of invented directories with ``F.build_gsm8k_report`` (placeholder shas, hostname TESTHOST, invented
counts) and passes the classification constants explicitly, so nothing here reads a real store. Covered: a canonical directory
plus its ``X-superseded`` twin share one comparison group and the twin's runs name the live runs; a declared (suffix-less)
supersession; a ``*_paper_rows_v2`` re-export is an input / artifact and never a run; a not-production pass files dispositions
only; an undeclared directory is listed but not read; failed gates and a wrong pin on a CANONICAL directory still refuse;
``paper_arm`` disagreement refuses by name; a canonical sibling's identical repeat is verified once and recorded on the owner.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from sisteer.registry import validate as VAL
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import gsm8k_p200 as G8
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal
from sisteer.registry import synth

sys.path.insert(0, os.path.dirname(__file__))
import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-17T12:00:00.000Z"
PRODUCER = {"script": "tests.declared.gsm8k_d60", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
PIN = "8f5f149"
CELL = "constv2-mix-full__lora_r4_ce"


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER))


def _read_all(adapter):
    """discover -> comparison groups -> read; every record validated (as tests/test_registry_adapter_declared.py::_read_all does).
    The returned report SUMS the per-group emit reports (this adapter reads several groups; ``last_report`` is the last group's)."""
    inputs = adapter.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(adapter.comparison_group(i), []).append(i)
    recs = []
    total = DC.EmitReport()
    ctx = AF.primed_ctx(adapter, now=NOW, producer=PRODUCER, groups=groups)     # the frame of the pass AFTER the fold: every own anchor current
    for g, ins in sorted(groups.items()):
        for batch in adapter.read(ctx, g, ins):
            recs.extend(batch.records)
        rep = adapter.last_report
        for k in ("runs", "runs_reused", "superseded_unmeasured_skipped", "metrics", "contrasts", "items", "degeneration", "dispositions", "superseded",
                  "unverified_fields", "instruments", "judges", "objects"):
            setattr(total, k, getattr(total, k) + getattr(rep, k))
        total.refusals.extend(rep.refusals)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    return recs, total


def _by_type(recs):
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out


def _adapter(root, canonical=(("canon", PIN),), superseded=None, not_production=None):
    return G8.Gsm8kP200Adapter(root=str(root), canonical=tuple(canonical), superseded=dict(superseded or {}), not_production=dict(not_production or {}))


def _paper_rows(dirname: str, fx: dict, paper_arm_cell: str, extra_fields=None) -> str:
    """A synthetic ``paper_rows/2`` file beside a fixture report: the r8192 and merged rows of its two cells, at the FIXTURE's counts."""
    rows = []
    k_none, k_cell = int(round(fx["acc_none"] * fx["n"])), int(round(fx["acc_cell"] * fx["n"]))
    for cell_id, k, arm in (("none", k_none, "none"), (fx["cell"], k_cell, paper_arm_cell)):
        for leg in ("r8192", "merged"):
            rows.append({"instrument": "gsm8k_paired200", "model_key": fx["model"], "view": fx["view"], "cell": cell_id, "arm": "bare" if cell_id == "none" else "lora_mix",
                         "paper_arm": arm, "cap": 8192, "leg": leg, "n_paired": fx["n"], "n_scored": fx["n"], "n_correct": k,
                         "point_conditional": k / fx["n"], "rung_state": "not_applicable", "object_generation": "synthetic",
                         "selection_outcome": None if cell_id == "none" else "ruled_2026-09-14", "source": "synthetic", **dict(extra_fields or {})})
    doc = {"schema": "paper_rows/2", "instrument": "gsm8k_paired200", "report": fx["path"], "n_rows": len(rows), "rows": rows,
           "fields": ["instrument", "model_key", "view", "cell", "arm", "paper_arm", "cap", "leg", "n_paired", "n_scored", "n_correct", "point_conditional",
                      "rung_state", "object_generation", "selection_outcome", "source"]}
    path = os.path.join(os.path.dirname(fx["path"]), "paper_rows.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return path


def _set_gates(path: str, failed):
    d = json.load(open(path))
    d["gates"]["failed"] = list(failed)
    json.dump(d, open(path, "w"))


# ============================================================================================
# canonical + X-superseded
# ============================================================================================

def test_gsm8k_d60_a_directory_renamed_superseded_declares_the_un_marked_paths_retired(tmp_path):
    """`X` renamed `X-superseded` beside its successor: the files the un-marked name spelled left the group's input set on purpose, so the
    reader declares them retired (the inputs-dropped guard) — and nothing when `X` still stands beside its copy."""
    root = tmp_path / "gsm"
    F.build_gsm8k_report(str(root / "canon"), k_none=20, k_cell=17)
    F.build_gsm8k_report(str(root / "old-superseded"), k_none=16, k_cell=15)
    ad = _adapter(root, canonical=(("canon", PIN),))
    groups = {ad.comparison_group(i) for i in ad.discover() if not AB.is_listing_input(i)}        # the root pseudo-group aside (ROOT_LISTING_RULE)
    (group,) = groups
    assert ad.inputs_retired(ad.root_group()) == []
    retired = ad.inputs_retired(group)
    assert retired == sorted([str(root / "old" / "report.json"), str(root / "old" / "paper_rows.json")])
    F.build_gsm8k_report(str(root / "old"), k_none=16, k_cell=15)                 # the un-marked pass is back beside its copy
    ad2 = _adapter(root, canonical=(("canon", PIN),))
    assert ad2.inputs_retired(group) == []


def test_gsm8k_d60_superseded_twin_shares_group_and_names_the_live_run(tmp_path):
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "canon"), k_none=17, k_cell=18)
    old = F.build_gsm8k_report(str(root / "canon-superseded"), k_none=16, k_cell=15)     # an older pass: different readings, failed gates
    _set_gates(old["path"], ["A0_completeness", "A6_ladder"])
    ad = _adapter(root)
    inputs = [i for i in ad.discover() if not AB.is_listing_input(i)]                     # the files; the root listing is the root pseudo-group's
    assert sorted(os.path.basename(os.path.dirname(i.uri)) for i in inputs) == ["canon", "canon-superseded"]
    assert {ad.comparison_group(i) for i in inputs} == {"canon"}                          # one emit pass: the twin's rows name the live ones
    assert ad.inventory["canon-superseded"]["kind"] == "superseded" and ad.inventory["canon-superseded"]["successor"] == "canon"
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.refusals == [] and rep.runs == 4 and rep.superseded == 2 and rep.contrasts == 2 and rep.dispositions == 0
    sup = {s["run_id"] for s in bt["run_state"] if s["state"] == "superseded"}
    live = {r["run_id"]: r for r in bt["run"] if r["run_id"] not in sup}
    old_runs = [r for r in bt["run"] if r["run_id"] in sup]
    assert len(live) == 2 and len(old_runs) == 2
    for r in old_runs:
        assert r["superseded_by_run_id"] in live and live[r["superseded_by_run_id"]]["campaign_cell_id_legacy"] == r["campaign_cell_id_legacy"]
        assert r["producer_run_key_legacy"].endswith("@canon-superseded") and r["status_ruling_id"].startswith("placeholder:")
        assert r["extra"]["classification"] == "superseded" and r["extra"]["successor"] == "canon" and r["extra"]["failed_gates"] == ["A0_completeness", "A6_ladder"]
    assert all(r["extra"]["classification"] == "canonical" and r["extra"]["owner"] == "canon" for r in live.values())
    # the superseded readings keep THEIR values (never the live ones), and the ruling placeholder names the successor and the failed gates
    old_metrics = {m["run_id"]: m for m in bt["run_metric"] if m["run_id"] in sup}
    assert sorted(m["k"] for m in old_metrics.values()) == [15, 16]
    rulings = [r for r in bt["ruling"] if r["action"] == "supersede"]
    assert len(rulings) == 2 and all("superseded by canon" in r["reason"] and "A6_ladder" in r["reason"] for r in rulings)
    assert all(r["target_json"]["superseded_by_run_id"] in live for r in rulings)
    # the superseded twin's contrast pairs with ITS OWN anchor, not the live one
    anchors = {c["run_id"]: c["anchor_run_id"] for c in bt["run_contrast"]}
    for rid, anchor in anchors.items():
        assert (rid in sup) == (anchor in sup)
    facts = ad.last_facts["canon"]["directories"]
    assert facts["canon"]["n_live"] == 2 and facts["canon-superseded"]["n_superseded"] == 2 and facts["canon-superseded"]["failed_gates"] == ["A0_completeness", "A6_ladder"]


def test_gsm8k_d60_declared_supersession_without_suffix(tmp_path):
    root = tmp_path / "gsm8k"
    F.build_gsm8k_report(str(root / "canon"))
    F.build_gsm8k_report(str(root / "older_pass"), k_cell=16)
    ad = _adapter(root, superseded={"older_pass": ("canon", "the base report the paper read before the extension")})
    assert {ad.comparison_group(i) for i in ad.discover()} == {"canon", ad.root_group()}
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.runs == 4 and rep.superseded == 2
    reasons = {r["reason"] for r in bt["ruling"] if r["action"] == "supersede"}
    assert len(reasons) == 1 and next(iter(reasons)).startswith("superseded by canon: the base report the paper read before the extension")
    sup = {s["run_id"] for s in bt["run_state"] if s["state"] == "superseded"}
    assert all(r["superseded_by_run_id"] and r["superseded_by_run_id"] not in sup for r in bt["run"] if r["run_id"] in sup)


def test_gsm8k_d60_superseded_directory_with_no_read_successor_is_still_filed_superseded(tmp_path):
    root = tmp_path / "gsm8k"
    F.build_gsm8k_report(str(root / "canon"))
    F.build_gsm8k_report(str(root / "other_model-superseded"), model="qwen35_4b", view="main")   # a family no canonical directory measures
    ad = _adapter(root)
    inputs = ad.discover()
    assert ad.inventory["other_model-superseded"]["successor"] is None
    assert {ad.comparison_group(i) for i in inputs} == {"canon", "other_model-superseded", ad.root_group()}
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    sup = [r for r in bt["run"] if r["extra"].get("classification") == "superseded"]
    assert len(sup) == 2 and all(r["superseded_by_run_id"] is None for r in sup)
    assert all("successor unknown" in r["reason"] for r in bt["ruling"] if r["action"] == "supersede")


# ============================================================================================
# re-exports
# ============================================================================================

def test_gsm8k_d60_re_export_is_an_input_not_a_run(tmp_path):
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "canon"))
    os.makedirs(root / "canon_paper_rows_v2")
    shutil.copyfile(fx["path"], root / "canon_paper_rows_v2" / "report.json")                     # byte-identical re-export
    _paper_rows("canon_paper_rows_v2", dict(fx, path=str(root / "canon_paper_rows_v2" / "report.json")), paper_arm_cell="const_lora")
    ad = _adapter(root)
    inputs = ad.discover()
    roles = {os.path.relpath(i.uri, root): i.role for i in inputs if not AB.is_listing_input(i)}
    assert roles == {"canon/report.json": "report", "canon_paper_rows_v2/report.json": "snapshot", "canon_paper_rows_v2/paper_rows.json": "rows"}
    assert {ad.comparison_group(i) for i in inputs} == {"canon", ad.root_group()}
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.runs == 2 and rep.superseded == 0 and rep.dispositions == 0 and rep.refusals == []
    assert all(r["extra"]["re_exports"] == ["canon_paper_rows_v2"] for r in bt["run"])
    assert all(r["extra"]["paper_rows_re_export"]["paper_arm"] in ("none", "const_lora") for r in bt["run"])
    f = ad.last_facts["canon"]["directories"]["canon_paper_rows_v2"]
    assert f["kind"] == "re_export" and f["target"] == "canon" and f["byte_identical"] is True and f["rows_agree"] is True


def test_gsm8k_d60_re_export_that_disagrees_refuses_by_name(tmp_path):
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "canon"))
    other = F.build_gsm8k_report(str(root / "canon_paper_rows_v2"), k_cell=16)                    # different bytes AND different counts
    _paper_rows("canon_paper_rows_v2", other, paper_arm_cell="const_lora")
    ad = _adapter(root)
    with pytest.raises(AdapterRefusal, match="canon_paper_rows_v2"):
        _read_all(ad)
    # different bytes but paper_rows agreeing by cell with the re-exported report is accepted (the fallback check)
    root2 = tmp_path / "gsm8k2"
    fx2 = F.build_gsm8k_report(str(root2 / "canon"))
    os.makedirs(root2 / "canon_paper_rows_v2")
    with open(root2 / "canon_paper_rows_v2" / "report.json", "w") as fh:
        json.dump(json.load(open(fx2["path"])), fh, indent=2)                                          # same content, other bytes
    _paper_rows("canon_paper_rows_v2", dict(fx2, path=str(root2 / "canon_paper_rows_v2" / "report.json")), paper_arm_cell="const_lora")
    ad2 = _adapter(root2)
    recs, rep = _read_all(ad2)
    f = ad2.last_facts["canon"]["directories"]["canon_paper_rows_v2"]
    assert f["byte_identical"] is False and f["rows_agree"] is True and rep.runs == 2


# ============================================================================================
# not production / undeclared
# ============================================================================================

def test_gsm8k_d60_default_check_files_dispositions_only(tmp_path):
    root = tmp_path / "gsm8k"
    F.build_gsm8k_report(str(root / "canon"))
    F.build_gsm8k_report(str(root / "canon_default_check"), k_cell=16)
    ad = _adapter(root, not_production={"canon_default_check": "the default-decode check pass"})
    inputs = ad.discover()
    assert {ad.comparison_group(i) for i in inputs} == {"canon", "canon_default_check", ad.root_group()}          # its own refresh boundary
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.runs == 2 and rep.superseded == 0 and rep.dispositions == 2
    assert all(r["extra"].get("classification") == "canonical" for r in bt["run"])
    for dsp in bt["disposition"]:
        assert dsp["state"] == "not_applicable"
        assert dsp["reason"].startswith("not applicable: not a production run — the default-decode check pass; the paper reads canon")
    assert {d["extra"]["campaign_cell_id_legacy"] for d in bt["disposition"]} == {"none", CELL}
    assert ad.last_facts["canon_default_check"]["directories"]["canon_default_check"]["n_dispositions"] == 2


def test_gsm8k_d60_undeclared_directory_is_listed_not_read(tmp_path):
    root = tmp_path / "gsm8k"
    F.build_gsm8k_report(str(root / "canon"))
    F.build_gsm8k_report(str(root / "final_other_pass"), k_cell=16)
    os.makedirs(root / "pull_stores")                                                                   # no report at all
    ad = _adapter(root)
    inputs = ad.discover()
    assert [os.path.relpath(i.uri, root) for i in inputs if not AB.is_listing_input(i)] == ["canon/report.json"]
    assert ad.inventory["final_other_pass"]["kind"] == "undeclared" and ad.inventory["final_other_pass"]["group"] is None
    assert ad.inventory["pull_stores"]["kind"] == "undeclared" and ad.inventory["pull_stores"]["has_report"] is False
    recs, rep = _read_all(ad)
    assert rep.runs == 2
    assert ad.last_facts["canon"]["inventory"]["final_other_pass"]["kind"] == "undeclared"
    states = ad.source_states(_ctx(), inputs)
    # the root inventory is the ROOT pseudo-group's row on the root listing INPUT (gsm8k_p200.ROOT_LISTING_RULE): keyed on the listing, its fingerprint the sha
    (listing,) = [i for i in inputs if AB.is_listing_input(i)]
    assert ad.comparison_group(listing) == ad.root_group() == G8.ROOT_GROUP and listing.entries == ("canon", "final_other_pass", "pull_stores")
    dirs = [s for s in states if s.json_path == "$.dirs" and s.found == 1]
    assert len(dirs) == 1 and dirs[0].input_uri == listing.uri == AB.listing_uri(f"gsm8k_p200:{ad.root_group()}", G8.ROOT_LISTING_LABEL)
    assert dirs[0].source_id == f"gsm8k_p200:{ad.root_group()}" and dirs[0].current_members == ["canon", "final_other_pass", "pull_stores"] and dirs[0].input_sha256 == listing.sha256
    # the pre-rule root-path key is stated absent beside it (ROOT_LISTING_RULE)
    (legacy,) = [s for s in states if s.json_path == "$.dirs" and s.found == 0]
    assert (legacy.source_id, legacy.input_uri, legacy.current_members, legacy.input_sha256) == (f"gsm8k_p200:{os.path.basename(str(root))}", str(root), [], None)
    cells = [s for s in states if s.json_path == "$.cells"]
    assert len(cells) == 1 and cells[0].current_members == [f"gemma4e4b/it/{CELL}", "gemma4e4b/it/none"]


# ============================================================================================
# canonical strictness
# ============================================================================================

def test_gsm8k_d60_canonical_failed_gates_and_wrong_pin_still_refuse(tmp_path):
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "canon"))
    _set_gates(fx["path"], ["A0_completeness"])
    with pytest.raises(AdapterRefusal, match="failed gates"):
        _read_all(_adapter(root))
    root2 = tmp_path / "gsm8k2"
    F.build_gsm8k_report(str(root2 / "canon"), pin="deadbee")
    with pytest.raises(AdapterRefusal, match="store pin"):
        _read_all(_adapter(root2))
    # the same directory RELAXED (as a superseded pass) reads, recording the failure
    ad = _adapter(root2, canonical=(("canon2", PIN),), superseded={"canon": ("canon2", "declared")})
    F.build_gsm8k_report(str(root2 / "canon2"))
    recs, rep = _read_all(ad)
    assert rep.superseded == 2 and ad.last_facts["canon2"]["directories"]["canon"]["pin_ok"] is False


def test_gsm8k_d60_paper_arm_disagreement_refuses_and_agreement_is_carried(tmp_path):
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "canon"))
    _paper_rows("canon", fx, paper_arm_cell="ctl_lora")                                                # the cell is const_lora
    with pytest.raises(AdapterRefusal, match="paper_arm 'ctl_lora'"):
        _read_all(_adapter(root))
    _paper_rows("canon", fx, paper_arm_cell="const_lora")
    ad = _adapter(root)
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    arm = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == CELL][0]
    assert arm["extra"]["paper_rows"] == {"file": "canon/paper_rows.json", "schema": "paper_rows/2", "leg": "merged", "paper_arm": "const_lora",
                                          "selection_outcome": "ruled_2026-09-14", "rung_state": "not_applicable"}
    none = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == "none"][0]
    assert none["extra"]["paper_rows"]["paper_arm"] == "none" and none["extra"]["paper_rows"]["selection_outcome"] is None
    # a paper_arm of None (an arm the paper does not draw) is not a disagreement
    _paper_rows("canon", fx, paper_arm_cell=None)
    recs, rep = _read_all(_adapter(root))
    assert rep.runs == 2


def test_gsm8k_d60_paper_rows_counts_disagreeing_with_the_report_refuse(tmp_path):
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "canon"), k_cell=18)
    path = _paper_rows("canon", fx, paper_arm_cell="const_lora")
    d = json.load(open(path))
    for r in d["rows"]:
        if r["cell"] == CELL:
            r["n_correct"] = 11
    json.dump(d, open(path, "w"))
    with pytest.raises(AdapterRefusal, match="paper_rows reads 11/20"):
        _read_all(_adapter(root))


# ============================================================================================
# ownership across canonical siblings
# ============================================================================================

def test_gsm8k_d60_canonical_sibling_repeat_is_verified_once_and_recorded_on_the_owner(tmp_path):
    root = tmp_path / "gsm8k"
    F.build_gsm8k_report(str(root / "a"))
    F.build_gsm8k_report(str(root / "b"))                                   # the same (model, view, cell), identical readings
    F.build_gsm8k_report(str(root / "c"), view="main")                      # another view of the family: its own component
    ad = _adapter(root, canonical=(("a", PIN), ("b", PIN), ("c", PIN)))
    inputs = ad.discover()
    groups = {os.path.basename(os.path.dirname(i.uri)): ad.comparison_group(i) for i in inputs if not AB.is_listing_input(i)}
    assert groups == {"a": "a", "b": "a", "c": "c"}
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.runs == 4 and rep.superseded == 0 and rep.refusals == []
    a_runs = [r for r in bt["run"] if r["extra"].get("owner") == "a"]
    assert len(a_runs) == 2 and all(r["extra"]["repeated_by"] == ["b"] for r in a_runs)
    assert ad.last_facts["a"]["directories"]["b"]["n_live"] == 0 and len(ad.last_facts["a"]["directories"]["b"]["repeats_verified"]) == 2
    # a repeat that DISAGREES refuses the group, naming the repeating directory
    root2 = tmp_path / "gsm8k2"
    F.build_gsm8k_report(str(root2 / "a"))
    F.build_gsm8k_report(str(root2 / "b"), k_cell=16)
    with pytest.raises(AdapterRefusal, match="DIFFERENT reading") as ei:
        _read_all(_adapter(root2, canonical=(("a", PIN), ("b", PIN))))
    assert "/b/report.json" in str(ei.value)


def test_gsm8k_d60_measured_beats_held_across_canonical_siblings(tmp_path):
    """A cell HELD in the first canonical directory and MEASURED in the second is owned by the second: one live run, no held disposition
    (the it-view constitution cells: held in ext r2 for the batch-64 refit, measured by itdoc r3)."""
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "a"))
    d = json.load(open(fx["path"]))
    d["held"] = [{"model_key": fx["model"], "view": fx["view"], "cell": "constv2ctl2-full__lora_r4_ce", "reason": "held for the refit"}]
    json.dump(d, open(fx["path"], "w"))
    F.build_gsm8k_report(str(root / "b"), cell="constv2ctl2-full__lora_r4_ce")
    ad = _adapter(root, canonical=(("a", PIN), ("b", PIN)))
    recs, rep = _read_all(ad)
    bt = _by_type(recs)
    assert rep.runs == 3 and rep.dispositions == 0 and rep.contrasts == 2
    owners = {r["campaign_cell_id_legacy"]: r["extra"]["owner"] for r in bt["run"]}
    assert owners == {"none": "a", CELL: "a", "constv2ctl2-full__lora_r4_ce": "b"}
    assert ad.last_facts["a"]["directories"]["a"]["dispositions_not_filed"] == [f"gemma4e4b/it/constv2ctl2-full__lora_r4_ce: held: held for the refit (owned by b)"]


# ============================================================================================
# the explicit-list constructor still works (the legacy signature the sibling module uses)
# ============================================================================================

def test_gsm8k_d60_explicit_list_constructor_reads_only_the_named_reports(tmp_path):
    root = tmp_path / "gsm8k"
    fx = F.build_gsm8k_report(str(root / "canon"))
    F.build_gsm8k_report(str(root / "canon-superseded"), k_cell=16)                                    # present but NOT named: not read
    ad = G8.Gsm8kP200Adapter(report=fx["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    assert ad.explicit and [os.path.relpath(i.uri, root) for i in ad.discover() if not AB.is_listing_input(i)] == ["canon/report.json"]
    assert [i.entries for i in ad.discover() if AB.is_listing_input(i)] == [("canon", "canon-superseded")]      # the root inventory's publisher, explicit mode too
    rows, facts = ad.long_rows(fx["path"])
    assert [r["status"] for r in rows] == ["measured", "measured"] and facts["store_pin"] == G8.STORE_PIN
    recs, rep = _read_all(ad)
    assert rep.runs == 2 and rep.superseded == 0
