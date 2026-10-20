"""W15d B1 (readiness audit): S1b's `adapter_hashes_banked` derivation looks the row's cells up in `cells.json` by HARNESS id.

`cells.json` (the container's own enumeration, `seedrep_served.py`: "HARNESS ids, as served") lists `si_constv2_mix_full__psr_ce_all`;
the row's `route_set` names the same cell with the training-side id `constv2-mix-full__psr_ce_all`. Before: `hashes.get(c)` with that
id found nothing, every fitted cell read "banks no intervention hash", `derive_smoke_pass(S1b)` refused every row, no S1b PASS, the
certificate refused. Same lines: the plan-disagreement guard tested cell ids against a STAGED-id map (`base64s1__...`) -- a silent no-op.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    sys.path.insert(0, str(p))
import seedrep_launch_plan as lp  # noqa: E402

W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"
SMOKE_WAVE = "11111111-2222-4333-8444-555555555555"
AID = "01SYNTHS2TASK0000000000001-222222222222"
S1B_ROW = "pasttense__s2__base__shard00"
#: the id form of the S1 evidence's cells_served.json canonical rows (f_l_evidence/S1/): harness ids
EVIDENCE_HARNESS_ID = "si_constv2_mix_full__psr_ce_all"
EVIDENCE_TRAINTEST_ID = "constv2-mix-full__psr_ce_all"


@pytest.fixture(scope="module")
def inputs():
    return lp.load_inputs(REPO, config_roots=(REPO.parent / "seedrep-ws1dp",))


def _plan(inputs, **kw):
    base = dict(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"], arditi_pins=inputs["arditi_pins"],
                arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z", config_private_runs=inputs.get("config_private_runs"))
    base.update(kw)
    return lp.build_plan(**base)


@pytest.fixture(scope="module")
def draft(inputs):
    return _plan(inputs)


def _row(plan, standup_id):
    return [r for r in plan["standups"] + plan["rungs"] if r["standup_id"] == standup_id][0]


def _cells_json(row, *, ids, hash_of):
    """The container's cells.json: `cells` rows per served model (canonical + the derived `none__lora0` control), `serving` bound to the attempt."""
    rows = [{"cell_id": ids(c), "virtual_model": f"petri__{ids(c)}", "serving_identity": "vllm_lora_native", "intervention_hash": hash_of(c), "si_id": None,
             "kind": ("none" if c == "none" else "lora_all")} for c in row["route_set"]]
    rows.append({"cell_id": lp.NONE_LORA0, "virtual_model": f"petri__{lp.NONE_LORA0}", "serving_identity": "vllm_lora_native", "intervention_hash": "0" * 16,
                 "si_id": None, "kind": "none"})
    return json.dumps({"cells": rows, "serving": {"substrate": "modal", "attempt_id": AID, "standup_id": row["standup_id"], "attested_max_num_seqs": 45,
                                                  "attested_max_model_len": 11520, "engine": "lora_native", "app": "seedrep-served-2222222"}})


def _check(plan, row, text):
    ident = lp.smoke_identity(plan, "S1b", SMOKE_WAVE)
    root = lp.attempt_root_for(ident["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=AID, kind=row.get("kind"))
    # modal_evidence.json["producer_evidence"], keyed by rel: the producer's binding of the file (sha, writer, tree sha, rc)
    producer = {lp.CELLS_JSON_REL: {"sha256": lp._sha_text(text), "writer": lp.PRODUCER_WRITERS["adapter_hashes_banked"], "tree_sha": plan["W"], "rc": 0}}
    return lp._smoke_check("adapter_hashes_banked", text, path=f"{root}/{lp.CELLS_JSON_REL}", plan=plan, accepted_attempts=[AID], row=row, aid=AID, root=root,
                           producer=producer)


def test_the_evidence_id_form_is_the_harness_id_of_the_rows_traintest_cell():
    assert lp.harness_id(EVIDENCE_TRAINTEST_ID) == EVIDENCE_HARNESS_ID and EVIDENCE_TRAINTEST_ID in lp.FITTED_CELLS


def test_the_real_cells_json_id_form_banks_all_five_fitted_cells(draft):
    row = _row(draft, S1B_ROW)
    fitted = [c for c in row["route_set"] if c != "none" and not c.startswith("prompted__")]
    assert len(fitted) == 5 and EVIDENCE_TRAINTEST_ID in fitted
    served = {c: f"{i:016x}" for i, c in enumerate(row["route_set"], start=0x1000)}
    facts = _check(draft, row, _cells_json(row, ids=lp.harness_id, hash_of=served.get))
    assert facts["n_banked"] == facts["n_planned"] == 5
    assert facts["cells"] == {c: served[c] for c in fitted}                       # keyed by the row's own cell, valued by the SERVED hash
    assert facts["producer"]["writer"] == "scripts/modal/seedrep_served.py"


def test_the_old_traintest_keyed_cells_json_is_refused_by_name_for_every_fitted_cell(draft):
    """The double the earlier test used is NOT what the app writes; against the real derivation it refuses --
    and it is the mirror image of the bug: the old derivation refused the REAL file the same way."""
    row = _row(draft, S1B_ROW)
    fitted = sorted(c for c in row["route_set"] if c != "none" and not c.startswith("prompted__"))
    with pytest.raises(ValueError, match="banks no intervention hash") as ei:
        _check(draft, row, _cells_json(row, ids=lambda c: c, hash_of=lambda c: "a" * 16))
    msg = str(ei.value)
    assert all(c in msg for c in fitted) and EVIDENCE_HARNESS_ID in msg          # names the missing cells AND the harness ids it looked for


def test_the_plan_disagreement_guard_is_live_keyed_through_the_rows_staged_ids(inputs):
    ids = sorted(a["staged_id"] for a in _plan(inputs)["adapter_hash_set"])
    hashes = {sid: f"ih{sid[:6]}".ljust(16, "0") for sid in ids}
    final = _plan(inputs, materialisation="final", s4=lp.S4_DEFAULTS, adapter_hashes=hashes)
    row = _row(final, S1B_ROW)
    staged = {c: row["objects"][c]["staged_id"] for c in row["objects"]}
    planned_of = lambda c: hashes.get(staged.get(c)) or "f" * 16  # noqa: E731     # prompted cells carry no staged object: any hash
    # served == planned: passes, and the facts carry the planned hashes
    facts = _check(final, row, _cells_json(row, ids=lp.harness_id, hash_of=planned_of))
    assert facts["cells"] == {c: hashes[staged[c]] for c in facts["cells"]} and facts["n_banked"] == 5
    # ONE cell served under another hash: refused, naming the cell (before: the guard compared cell ids against staged ids and never fired)
    drift = lambda c: ("d" * 16) if c == EVIDENCE_TRAINTEST_ID else planned_of(c)  # noqa: E731
    with pytest.raises(ValueError, match="disagree with the plan's adapter_hash_set") as ei:
        _check(final, row, _cells_json(row, ids=lp.harness_id, hash_of=drift))
    assert f"['{EVIDENCE_TRAINTEST_ID}']" in str(ei.value) and "d" * 16 in str(ei.value)
    # the guard reads the row's objects, not a cell-id lookup: a row whose objects lack the cell has nothing planned for it -> no refusal
    bare = copy.deepcopy(row)
    bare["objects"] = {c: o for c, o in row["objects"].items() if c != EVIDENCE_TRAINTEST_ID}
    assert _check(final, bare, _cells_json(row, ids=lp.harness_id, hash_of=drift))["n_banked"] == 5


def test_a_cells_json_served_for_another_attempt_or_row_still_refuses(draft):
    row = _row(draft, S1B_ROW)
    text = _cells_json(row, ids=lp.harness_id, hash_of=lambda c: "b" * 16)
    doc = json.loads(text)
    doc["serving"]["standup_id"] = "pasttense__s3__base__shard00"
    with pytest.raises(ValueError, match="was served for attempt"):
        _check(draft, row, json.dumps(doc))
