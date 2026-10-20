"""qwen-seedrep-native-battery F-AN (fixer fp2): the request-evidence audit's arm -> vmodel binding resolves a store's arm in
EITHER spelling through the STANDUP'S OWN RECORDS — never a string transform.

The S4 crescendo root (and, the evidence shows, the S5 pasttense FULL and S3 bon rows) refused every store verdict:
`REFUSED: …/exp/serve/endpoint.json: arm 'constv2-mix-full__lora_r4_ce' is not a served cell (vmodels: ['none', 'none__lora0',
'prompted__constitution', 'prompted__safety', 'si_constv2_mix_full__lora_r4_ce', …])`. The static / BoN / multi-turn drivers take
`--defense <TrainTest cell id>` and name their store dirs by it, so `discover_stores` hands the TrainTest spelling; the endpoint's
`vmodels` map is keyed by the served HARNESS id (`si_constv2_mix_full__lora_r4_ce`). The standup records the identity itself:
`endpoint.json` `cells` <-> `harness_cells` (index-aligned), `cells_served.json` `cells_requested` <-> `harness_cells` + the
`canonical` / `controls` rows' `cell_id` -> `virtual_model_name`, `routing_table.json` routes' `cell_id` -> `virtual_model`.
`audit_common.standup_vmodel_of` / `Standup.binding_of` resolve through them, record which spelling matched, and refuse by name
when neither spelling resolves, when the two spellings resolve to DIFFERENT vmodels, or when the records disagree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402

TT = "constv2-mix-full__lora_r4_ce"                 # the TrainTest CELL id (the plan child's `cell`; the drivers' --defense; the store dir)
HARNESS = "si_constv2_mix_full__lora_r4_ce"         # the served HARNESS id (`endpoint.json.vmodels` key; cells_served canonical cell_id)
TT2, HARNESS2 = "constv2ctl2-full__psr_ce_all", "si_constv2ctl2_full__psr_ce_all"
HARNESS_OF = {TT: HARNESS, TT2: HARNESS2}


def _audit(root: fx.Root, legs, out: Path):
    ar = ac.read_attempt_root(root.path)
    return rea.audit_root(ar, out_dir=out / "audits", legs=list(legs), policy_file=None, thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT),
                          petri_behaviours=list(fx.PETRI_BEHAVIOURS), model_key=ac.MODEL_KEY, client_status="absent", siblings=[])


def _w22_shaped_root(tmp_path: Path, *, leg: str = "crescendo", arms=("none", TT)) -> fx.Root:
    """A root in the real crescendo shape: the store dir named by the `TT` arm id, `vmodels` keyed by the harness id,
    the two spellings bound by the standup's own records."""
    root = fx.make_root(tmp_path / "stores", leg=leg, view="s1", arms=arms, harness_of=HARNESS_OF)
    fx.populate_root(root, s=1, arms=arms, n_items=2)
    ep = ac.read_json(root.serve / ac.ENDPOINT_FILE)
    assert TT not in ep["vmodels"] and HARNESS in ep["vmodels"], "the fixture must reproduce the record's shape (harness-keyed vmodels)"
    assert ep["cells"].index(TT) == ep["harness_cells"].index(HARNESS)
    stores = ac.discover_stores(ac.read_attempt_root(root.path), leg)
    assert {s.arm for s in stores} == set(arms), "the store readers hand the DIR name = the TrainTest spelling"
    return root


def test_translation_table_is_the_lanes_own():
    """The synthetic seam is the lane's real one (`seedrep_launch_plan.HARNESS_ID_OF`), so the fixture's records are the real shape."""
    import seedrep_launch_plan as lp
    assert lp.harness_id(TT) == HARNESS and lp.harness_id(TT2) == HARNESS2 and lp.traintest_id(HARNESS) == TT


# --------------------------------------------------------------------------- #
# known-bad -> green: the crescendo refusal                                                                    #
# --------------------------------------------------------------------------- #

def test_known_bad_w22_crescendo_root_reaches_the_stores_verdict(tmp_path: Path):
    """At the tip: `REFUSED: …/endpoint.json: arm 'constv2-mix-full__lora_r4_ce' is not a served cell (vmodels: [...])` before any
    verdict. Fixed: the audit reaches every store's verdict, and the fitted arm's records WHICH spelling matched and the harness id."""
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    verdicts = _audit(root, ["crescendo"], tmp_path)
    by_arm = {v["arm"]: v for v in verdicts}
    assert set(by_arm) == {"none", TT}
    assert by_arm[TT]["verdict"] == "AGREES", by_arm[TT].get("problem")
    assert by_arm["none"]["verdict"] == "AGREES", by_arm["none"].get("problem")
    b = by_arm[TT]["arm_binding"]
    assert b["arm"] == TT and b["harness_id"] == HARNESS and b["vmodel"] == root.vmodels[TT] == f"petri__{HARNESS}"
    assert b["spelling"] == ac.ARM_SPELLING_TRAINTEST and b["traintest_id"] == TT
    assert by_arm["none"]["arm_binding"]["spelling"] == ac.ARM_SPELLING_BOTH
    # the records the binding was read from are named
    assert b["records"][ac.ENDPOINT_FILE] == "present" and b["records"][ac.CELLS_SERVED_FILE] == "present" \
        and b["records"][ac.ROUTING_TABLE_FILE] == "present"
    # the verdict file is the plan's `<arm>__s<s>.json` for the store's spelling
    assert (tmp_path / "audits" / "request_log" / "crescendo" / f"{TT}__s1.json").is_file()


@pytest.mark.parametrize("leg", ["pasttense", "bon", "pair", "actorattack"])
def test_every_leg_whose_driver_names_the_store_by_the_traintest_id(tmp_path: Path, leg: str):
    """Static (`jailbreak_run --defense`), BoN (`bon_served_run --defense`) and the multi-turn attacks all name their store dir by the
    `TT` arm id — the pasttense FULL and bon rows carried the same refusal (`audits/request_log/*` absent, tool rc 2)."""
    root = _w22_shaped_root(tmp_path, leg=leg)
    verdicts = _audit(root, [leg], tmp_path)
    assert {(v["arm"], v["verdict"]) for v in verdicts} == {("none", "AGREES"), (TT, "AGREES")}
    assert {v["arm"]: v["arm_binding"]["spelling"] for v in verdicts} == {"none": ac.ARM_SPELLING_BOTH, TT: ac.ARM_SPELLING_TRAINTEST}


def test_both_spellings_resolve_to_the_one_vmodel_and_the_harness_spelling_matches_directly(tmp_path: Path):
    """A reader that hands the HARNESS id (the misalign runner names its cell dir by `cells.json`'s `cell_id`; the controls / tinycap
    CellStore manifests) resolves directly; the TrainTest spelling resolves THROUGH the records to the same vmodel."""
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    via_tt, via_h = ac.standup_vmodel_of(root.path, TT), ac.standup_vmodel_of(root.path, HARNESS)
    assert via_tt.vmodel == via_h.vmodel == f"petri__{HARNESS}" and via_tt.harness_id == via_h.harness_id == HARNESS
    assert via_tt.spelling == ac.ARM_SPELLING_TRAINTEST and via_h.spelling == ac.ARM_SPELLING_HARNESS
    assert via_h.traintest_id == TT   # the harness spelling still knows its TrainTest id from the records
    assert ac.vmodel_of_arm(root.path, TT) == ac.vmodel_of_arm(root.path, HARNESS)
    st = ac.read_standup(ac.read_attempt_root(root.path))
    assert st.vmodel_of(TT) == st.vmodel_of(HARNESS) == f"petri__{HARNESS}"
    assert st.binding_of("none").spelling == ac.ARM_SPELLING_BOTH and st.binding_of("none").harness_id == "none"


# --------------------------------------------------------------------------- #
# refusals stay refusals, by name                                                                                   #
# --------------------------------------------------------------------------- #

def test_an_arm_served_by_neither_spelling_still_refuses(tmp_path: Path):
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    with pytest.raises(ac.AuditRefusal, match=r"arm 'constv2ctl2-full__lora_r4_ce' is not a served cell .*vmodels: \['none', 'si_constv2_mix_full__lora_r4_ce'\]"):
        ac.standup_vmodel_of(root.path, "constv2ctl2-full__lora_r4_ce")
    with pytest.raises(ac.AuditRefusal, match=r"arm 'si_constv2ctl2_full__lora_r4_ce' is not a served cell"):
        ac.read_standup(ac.read_attempt_root(root.path)).vmodel_of("si_constv2ctl2_full__lora_r4_ce")
    # a TrainTest spelling the records DO bind, to a harness cell the standup lists no vmodel for (a skipped cell): refused naming both
    ep = dict(root.endpoint)
    ep["cells"], ep["harness_cells"], ep["cells_requested"] = [*ep["cells"], TT2], [*ep["harness_cells"], HARNESS2], [*ep["cells_requested"], TT2]
    root.rewrite_endpoint(**ep)
    with pytest.raises(ac.AuditRefusal, match=rf"arm '{TT2}' is the TrainTest spelling of harness cell '{HARNESS2}', which the standup lists no vmodel for"):
        ac.standup_vmodel_of(root.path, TT2)


def test_two_spellings_bound_to_different_vmodels_refuse_by_name(tmp_path: Path):
    """The endpoint lists BOTH spellings as served cells with different virtual models: never resolved by picking one."""
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    vm = dict(root.endpoint["vmodels"])
    vm[TT] = "petri__other"
    root.rewrite_endpoint(vmodels=vm)
    with pytest.raises(ac.AuditRefusal, match=rf"arm '{TT}' resolves to vmodel 'petri__other' as a served cell id and to 'petri__{HARNESS}' through its TrainTest spelling \(harness cell '{HARNESS}'\)"):
        ac.standup_vmodel_of(root.path, TT)
    with pytest.raises(ac.AuditRefusal, match="resolves to vmodel 'petri__other' as a served cell id"):
        _audit(root, ["crescendo"], tmp_path)


def test_routing_table_disagreeing_with_the_endpoint_refuses_by_name(tmp_path: Path):
    """The routing table is the standup's third record of the same binding: a route naming another virtual model for the harness
    cell than `endpoint.json.vmodels` refuses (two records of one standup disagree — neither is picked)."""
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    rt_p = root.serve / ac.ROUTING_TABLE_FILE
    rt = json.loads(rt_p.read_text())
    for r in rt["routes"]:
        if r["cell_id"] == HARNESS:
            r["virtual_model"] = "petri__elsewhere"
    rt_p.write_text(json.dumps(rt))
    with pytest.raises(ac.AuditRefusal, match=rf"{ac.ROUTING_TABLE_FILE}: route for cell '{HARNESS}' names virtual model 'petri__elsewhere' but {ac.ENDPOINT_FILE}\.vmodels names 'petri__{HARNESS}'"):
        ac.standup_vmodel_of(root.path, TT)
    # `none` is bound consistently, yet the standup's records disagree — every resolution on this root refuses (the identity is one)
    with pytest.raises(ac.AuditRefusal, match=f"{ac.ROUTING_TABLE_FILE}: route for cell"):
        ac.standup_vmodel_of(root.path, "none")


def test_cells_served_disagreeing_with_the_endpoint_refuses_by_name(tmp_path: Path):
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    cs_p = root.serve / ac.CELLS_SERVED_FILE
    cs = json.loads(cs_p.read_text())
    cs["harness_cells"] = [cs["harness_cells"][0], "si_some_other_cell"]
    cs_p.write_text(json.dumps(cs))
    with pytest.raises(ac.AuditRefusal, match=rf"{ac.CELLS_SERVED_FILE}: cells_requested <-> harness_cells .* disagree with {ac.ENDPOINT_FILE}'s cells <-> harness_cells"):
        ac.standup_vmodel_of(root.path, TT)
    cs["harness_cells"] = json.loads((root.serve / ac.ENDPOINT_FILE).read_text())["harness_cells"]
    cs["canonical"][1]["virtual_model_name"] = "petri__elsewhere"
    cs_p.write_text(json.dumps(cs))
    with pytest.raises(ac.AuditRefusal, match=rf"{ac.CELLS_SERVED_FILE}: canonical/controls row for cell '.*' names virtual_model_name 'petri__elsewhere' but {ac.ENDPOINT_FILE}\.vmodels names"):
        ac.standup_vmodel_of(root.path, TT)


def test_misaligned_spelling_lists_refuse(tmp_path: Path):
    """`cells` and `harness_cells` are index-aligned by the writer; different lengths (or a repeated id) are not a map."""
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    root.rewrite_endpoint(harness_cells=["none"])
    with pytest.raises(ac.AuditRefusal, match=rf"{ac.ENDPOINT_FILE}: cells \(2\) and harness_cells \(1\) are not index-aligned"):
        ac.standup_vmodel_of(root.path, TT)


def test_a_record_spelling_no_traintest_ids_refuses_a_traintest_arm_by_name(tmp_path: Path):
    """An endpoint without `cells` / `harness_cells` (and a cells_served without `cells_requested`) binds nothing but the harness
    spelling: the harness arm resolves (spelling recorded), the TrainTest arm refuses saying the record spells no TrainTest ids."""
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    ep = {k: v for k, v in root.endpoint.items() if k not in ("cells", "harness_cells", "cells_requested")}
    root.endpoint = ep
    root._write_serve()
    cs_p = root.serve / ac.CELLS_SERVED_FILE
    cs = json.loads(cs_p.read_text())
    cs.pop("cells_requested"), cs.pop("harness_cells")
    cs_p.write_text(json.dumps(cs))
    b = ac.standup_vmodel_of(root.path, HARNESS)
    assert b.spelling == ac.ARM_SPELLING_HARNESS and b.traintest_id is None and b.records["traintest_spellings"] == "absent"
    with pytest.raises(ac.AuditRefusal, match=rf"arm '{TT}' is not a served cell .* and the standup's records spell no TrainTest ids"):
        ac.standup_vmodel_of(root.path, TT)


def test_records_absent_are_recorded_not_invented(tmp_path: Path):
    """`cells_served.json` / `routing_table.json` absent from a pulled root: the binding still resolves off the endpoint record (the
    contract `read_standup` already requires) and RECORDS the absence by name — never a silent pass-through of a cross-check."""
    root = _w22_shaped_root(tmp_path, leg="crescendo")
    (root.serve / ac.CELLS_SERVED_FILE).unlink()
    (root.serve / ac.ROUTING_TABLE_FILE).unlink()
    b = ac.standup_vmodel_of(root.path, TT)
    assert b.vmodel == f"petri__{HARNESS}" and b.spelling == ac.ARM_SPELLING_TRAINTEST
    assert b.records[ac.CELLS_SERVED_FILE] == "absent" and b.records[ac.ROUTING_TABLE_FILE] == "absent" and b.records[ac.ENDPOINT_FILE] == "present"
