"""A smoke row's `adapters[*].expected_reused` follows the smoke's DECLARED role, never production's.

Before the fix, the smoke rows (pasttense s2 / s3) failed readiness with `adapter for <cell> (si_...) was EXPORTED (reused=False), not
re-used -- a race or a drift (gate 10)` x5, because `smoke_row` kept the production expectation (`expected_reused=True`) although
S1 / S1b are the FIRST exporters of the adapters they serve (§8 S1b: reads `adapters_reused == false` for its own five cells, `true`
for `none__lora0`). The consumer (`seedrep_served.readiness_problems`, gate 10) skips a cell whose row expects False and refuses
`reused=False` where it expects True. The fix derives the expectation from a role table (`SMOKE_ADAPTER_ROLES`, a reason per role):
first exporters (S1: everything; S1b / S2: their fitted cells) -> False; consumers of a prior smoke's exports (S3, S4; `none__lora0`
under S1b / S2) -> True; production rows unchanged (True); `validate_plan` refuses a first-export smoke row that expects True.
"""
from __future__ import annotations

import copy
import importlib.util
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
#: the launch's smoke rows (launch_report 05:27Z HOLD): S1 = the s1 view engine, S1b = the s2 / s3 view engines, S2 = the Arditi engine
W14_S1_ROW = "pasttense__s1__base__shard00"
W14_S1B_ROWS = ("pasttense__s2__base__shard00", "pasttense__s3__base__shard00")
W14_S2_ROWS = ("pasttense__arditi__base__shard00", "controls__arditi__base__shard00")
W14_S3_ROWS = ("controls__s1__base__shard00", "bon__s1__base__shard00", "misalign__s1__base__shard00", "petri__s1__base__shard00", "gsm8k__s1__base__shard00")


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inputs():
    probe_roster, probe_pins = REPO / lp.PROBE_ROSTER_REL, REPO / lp.PROBE_PINS_REL
    have_probe = probe_roster.is_file() and probe_pins.is_file()
    return lp.load_inputs(REPO, peer_roster=(probe_roster if have_probe else None), peer_pins=(probe_pins if have_probe else None),
                          config_roots=(REPO.parent / "seedrep-ws1dp",))


@pytest.fixture(scope="module")
def draft(inputs):
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                         arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z",
                         config_private_runs=inputs.get("config_private_runs"), peer_roster=inputs["peer_roster"], peer_pins=inputs["peer_pins"])


def _rows(plan):
    return list(plan["standups"]) + list(plan["rungs"]) + list(plan.get("s5_probe_rows") or [])


def _row(plan, standup_id):
    hits = [r for r in _rows(plan) if r["standup_id"] == standup_id]
    assert len(hits) == 1, standup_id
    return hits[0]


def _smoke(plan, standup_id, smoke_id):
    return lp.smoke_row(_row(plan, standup_id), lp.smoke_identity(plan, smoke_id, SMOKE_WAVE), max_children=5, max_timeout_s=4 * 3600)


def _expectations(row):
    return {cell: a["expected_reused"] for cell, a in row["adapters"].items()}


# ------------------------------------------------------------------ the role table: one reason per §8 role, every §8 smoke declared

def test_the_role_table_covers_every_s8_smoke_with_a_reason_and_a_known_first_export_kind():
    assert set(lp.SMOKE_ADAPTER_ROLES) == {"S1", "S1B", "S2", "S3", "S4", "S5"}
    kinds = {lp.SMOKE_FIRST_EXPORTS_ALL, lp.SMOKE_FIRST_EXPORTS_FITTED, lp.SMOKE_FIRST_EXPORTS_NONE}
    for sid, role in lp.SMOKE_ADAPTER_ROLES.items():
        assert role["first_exports"] in kinds and "§8" in role["reason"] and len(role["reason"]) > 40, sid
    # the exact table (order S1 -> {S1b, S2, S3, S5} -> S4)
    assert {k: v["first_exports"] for k, v in lp.SMOKE_ADAPTER_ROLES.items()} == {
        "S1": "all", "S1B": "fitted", "S2": "fitted", "S3": "none", "S4": "none", "S5": "all"}
    # the role lookup is case-insensitive (`S1b` is how §8 spells it; the id rides the operator's --smoke-id) and names the reason
    assert lp.smoke_adapter_role("S1b") == lp.smoke_adapter_role("S1B") == {"role": "S1B", **lp.SMOKE_ADAPTER_ROLES["S1B"]}
    assert lp.smoke_adapter_role("s5")["role"] == "S5"
    # every pass-record smoke id (S1..S4) has a role; a smoke id outside the table declares none and is refused by name
    for sid in lp.SMOKE_PASS_IDS:
        lp.smoke_adapter_role(sid)
    with pytest.raises(ValueError, match="declares no §8 adapter role"):
        lp.smoke_adapter_role("S9")
    with pytest.raises(ValueError, match="declares no §8 adapter role"):
        lp.smoke_adapter_role("")


# ------------------------------------------------------------------ the failing rows: first exporters no longer expect a re-use

def test_w14_s1_row_the_waves_first_engine_asserts_no_reuse_on_any_adapter(draft):
    prod = _row(draft, W14_S1_ROW)
    assert all(v is True for v in _expectations(prod).values())                        # before: the production expectation on every cell
    srow = _smoke(draft, W14_S1_ROW, "S1")
    exp = _expectations(srow)
    assert set(exp) == set(prod["adapters"]) and lp.NONE_LORA0 in exp and len(exp) == len(lp.FITTED_CELLS) + 1
    assert all(v is False for v in exp.values()), exp                                    # after: S1 first-exports everything, the zero adapter included
    assert srow["smoke"]["adapter_role"] == lp.smoke_adapter_role("S1") and srow["smoke"]["adapter_role"]["first_exports"] == "all"


@pytest.mark.parametrize("standup_id", W14_S1B_ROWS)
def test_w14_s1b_rows_first_export_their_own_five_cells_and_reuse_the_zero_adapter(draft, standup_id):
    prod = _row(draft, standup_id)
    assert all(v is True for v in _expectations(prod).values())                        # the production shape: True everywhere -> EXPORTED x5 at gate 10
    srow = _smoke(draft, standup_id, "S1b")
    exp = _expectations(srow)
    fitted = {c for c in exp if c != lp.NONE_LORA0}
    assert len(fitted) == 5 and all(exp[c] is False for c in fitted), exp                # §8 S1b: adapters_reused false for its own five cells
    assert exp[lp.NONE_LORA0] is True                                                    # ... and true for none__lora0 (S1 banked it)
    assert srow["smoke"]["smoke_id"] == "S1b" and srow["smoke"]["adapter_role"]["role"] == "S1B"


def test_a_smoke_row_changes_only_expected_reused_in_its_adapters_block(draft):
    prod, srow = _row(draft, W14_S1B_ROWS[0]), _smoke(draft, W14_S1B_ROWS[0], "S1b")
    for cell, a in prod["adapters"].items():
        assert {k: v for k, v in srow["adapters"][cell].items() if k != "expected_reused"} == {k: v for k, v in a.items() if k != "expected_reused"}
    # nothing else scientific moved: the production row's objects, route set, readiness, geometry, argv are the smoke's
    for k in ("objects", "route_set", "readiness", "geometry", "argv", "cells_served_expected"):
        assert srow[k] == prod[k], k
    assert prod["adapters"][lp.FITTED_CELLS[0]]["expected_reused"] is True               # the production row itself is untouched (deep copy)


# ------------------------------------------------------------------ S2 (Arditi), S3 / S4 (consumers), S5 (measurement)

@pytest.mark.parametrize("standup_id", W14_S2_ROWS)
def test_s2_the_arditi_engine_first_exports_arditi_allpos_and_reuses_the_zero_adapter(draft, standup_id):
    """S1b's pass rule: the bank holds 15 fitted + zero + Arditi only AFTER S1 + S1b + S2 -- S2 is the FIRST exporter of the
    Arditi hash (the generator's own `adapter_expectations` says the same: S1 / S1b / S2 banked every hash first). So S2 asserts no
    re-use of `arditi_allpos` and DOES assert `none__lora0` re-used (S1 banked the zero adapter; every lora_native engine serves it)."""
    exp = _expectations(_smoke(draft, standup_id, "S2"))
    assert exp == {lp.ARDITI_CELL: False, lp.NONE_LORA0: True}, exp


@pytest.mark.parametrize("standup_id", W14_S3_ROWS)
def test_s3_rows_against_the_s1_engine_expect_every_adapter_reused(draft, standup_id):
    exp = _expectations(_smoke(draft, standup_id, "S3"))
    assert exp and all(v is True for v in exp.values()), exp                              # S1 banked the s1 view's hashes before S3


def test_s4_the_multiturn_window_decision_expects_every_adapter_reused(draft):
    exp = _expectations(_smoke(draft, "crescendo__s1__base__shard00", "S4"))
    assert exp and all(v is True for v in exp.values()), exp


def test_s5_asserts_no_reuse_on_production_class_rows_and_on_the_probe_rows(draft):
    for sid in (W14_S1_ROW, W14_S1B_ROWS[0], W14_S2_ROWS[0], "crescendo__s1__base__shard00"):
        exp = _expectations(_smoke(draft, sid, "S5"))
        assert exp and all(v is False for v in exp.values()), (sid, exp)                  # in parallel with S1b / S2: nothing asserted
    probe = list(draft.get("s5_probe_rows") or [])
    if not probe:
        pytest.skip("no S5 probe rows in this tree (the probe roster / pins are absent)")
    for r in probe:
        # B3 (re-pinned from the W15d (d) commit, which read True here): the plan's OWN probe row is a first export by construction -> False at
        # the plan level too, none__lora0 included (before B3 `adapter_expectations` minted production's True on it)
        assert r.get("smoke_only") is True and all(v is False for v in _expectations(r).values()), (r["standup_id"], _expectations(r))
        exp = _expectations(lp.smoke_row(r, lp.smoke_identity(draft, "s5", SMOKE_WAVE), max_children=2, max_timeout_s=600))
        assert all(v is False for v in exp.values()), (r["standup_id"], exp)               # S5 is the probe cells' first exporter


# ------------------------------------------------------------------ production rows: unchanged, and the gate pins both sides

def test_production_rows_still_expect_every_adapter_reused_and_pass_the_gate(draft):
    rows = list(draft["standups"]) + list(draft["rungs"])
    assert rows
    for r in rows:
        assert "smoke" not in r and not r.get("smoke_only")
        assert r["adapters"] and all(a["expected_reused"] is True for a in r["adapters"].values()), r["standup_id"]
        assert lp.smoke_adapter_problems(r) == []
    for r in draft.get("s5_probe_rows") or []:                                                  # B3: smoke-only probe rows: no re-use asserted
        assert r["adapters"] and all(a["expected_reused"] is False for a in r["adapters"].values()), r["standup_id"]
        assert lp.smoke_adapter_problems(r) == []
    assert [p for p in lp.validate_plan(draft) if "adapter" in p and ("gate 10" in p or "FIRST-EXPORTS" in p or "RE-USES" in p or "(B3)" in p)] == []


def test_b3_adapter_expectations_mints_no_reuse_for_a_first_export_row_and_the_gate_refuses_a_probe_row_minted_true(draft):
    objects = {lp.FITTED_CELLS[0]: {"staged_id": "base64s1__x", "object_sha256": "a" * 64}}
    assert {c: a["expected_reused"] for c, a in lp.adapter_expectations(objects, {}).items()} == {lp.FITTED_CELLS[0]: True, lp.NONE_LORA0: True}
    assert {c: a["expected_reused"] for c, a in lp.adapter_expectations(objects, {}, expected_reused=False).items()} == {lp.FITTED_CELLS[0]: False, lp.NONE_LORA0: False}
    probe = list(draft.get("s5_probe_rows") or [])
    if not probe:
        pytest.skip("no S5 probe rows in this tree (the probe roster / pins are absent)")
    # the shape B3 removed: a probe row carrying production's expectation on the cell it exports first
    bad = copy.deepcopy(probe[0])
    cell = [c for c in bad["adapters"] if c != lp.NONE_LORA0][0]
    bad["adapters"][cell]["expected_reused"] = True
    probs = lp.smoke_adapter_problems(bad)
    assert len(probs) == 1 and "FIRST EXPORT of adapter" in probs[0] and cell in probs[0] and "(B3)" in probs[0], probs
    named = [p for p in lp.validate_plan({**draft, "s5_probe_rows": [bad] + probe[1:]}) if p.startswith(bad["standup_id"]) and "(B3)" in p]
    assert len(named) == 1, named
    # and the S5 smoke row derived from the mis-minted probe row is repaired by the role (S5 first-exports everything)
    srow = lp.smoke_row(bad, lp.smoke_identity(draft, "S5", SMOKE_WAVE), max_children=2, max_timeout_s=600)
    assert all(v is False for v in _expectations(srow).values()) and lp.smoke_adapter_problems(srow) == []


def test_validate_plan_refuses_a_first_export_smoke_row_that_expects_true_and_a_production_row_that_expects_false(draft):
    ident = lp.smoke_identity(draft, "S1b", SMOKE_WAVE)
    good = lp.smoke_row(_row(draft, W14_S1B_ROWS[0]), ident, max_children=5, max_timeout_s=3600)
    assert lp.smoke_adapter_problems(good) == []
    # the known-bad shape: the smoke row re-keyed but carrying the PRODUCTION expectation on its own five cells
    bad = copy.deepcopy(good)
    for cell in bad["adapters"]:
        bad["adapters"][cell]["expected_reused"] = True
    probs = lp.smoke_adapter_problems(bad)
    assert len(probs) == 5 and all("smoke S1B FIRST-EXPORTS adapter" in p and "W14 F-L(d)" in p and "§8" in p for p in probs), probs
    assert not any(f"adapter {lp.NONE_LORA0} (" in p for p in probs)                     # none__lora0 IS a re-use under S1b: True is right there
    # and the reverse drift: a consumer smoke row that stopped asserting the re-use it depends on
    lax = copy.deepcopy(lp.smoke_row(_row(draft, W14_S3_ROWS[0]), lp.smoke_identity(draft, "S3", SMOKE_WAVE), max_children=5, max_timeout_s=3600))
    lax["adapters"][lp.NONE_LORA0]["expected_reused"] = False
    assert [p for p in lp.smoke_adapter_problems(lax) if "smoke S3 RE-USES adapter none__lora0" in p]
    # validate_plan names the bad smoke row (S0's PLAN gate refuses the plan) and leaves the derived one alone
    plan_bad = {**draft, "standups": list(draft["standups"]) + [bad]}
    named = [p for p in lp.validate_plan(plan_bad) if p.startswith(bad["standup_id"]) and "FIRST-EXPORTS" in p]
    assert len(named) == 5, named
    plan_good = {**draft, "standups": list(draft["standups"]) + [good]}
    assert [p for p in lp.validate_plan(plan_good) if p.startswith(good["standup_id"]) and ("FIRST-EXPORTS" in p or "RE-USES" in p)] == []
    # a production row that expects no re-use is the other side's drift
    prod_bad = copy.deepcopy(_row(draft, W14_S1_ROW))
    prod_bad["adapters"][lp.FITTED_CELLS[0]]["expected_reused"] = False
    assert [p for p in lp.validate_plan({**draft, "standups": [prod_bad] + list(draft["standups"])[1:]})
            if p.startswith(W14_S1_ROW) and "production row expects adapter" in p and lp.FITTED_CELLS[0] in p]
    # a smoke row under an id with no §8 role is refused by name, at derivation and at the gate
    with pytest.raises(ValueError, match="declares no §8 adapter role"):
        lp.smoke_row(_row(draft, W14_S1_ROW), lp.smoke_identity(draft, "S9", SMOKE_WAVE), max_children=5, max_timeout_s=3600)
    stray = copy.deepcopy(good)
    stray["smoke"]["smoke_id"] = "S9"
    assert [p for p in lp.smoke_adapter_problems(stray) if "declares no §8 adapter role" in p]


# ------------------------------------------------------------------ the consumer: the exact refusal message goes away on the derived row

def test_readiness_problems_no_longer_refuses_a_first_exporters_own_export(draft):
    """`seedrep_served.readiness_problems` (gate 10, read-only here) over an attestation that reports the S1b engine's own five cells
    EXPORTED and `none__lora0` re-used: the PRODUCTION row refuses x5 with the EXPORTED message; the derived smoke row does not."""
    fx = _load_by_path("seedrep_review_fixes_helpers_w15d", "tests/test_seedrep_review_fixes.py")
    spec = importlib.util.spec_from_file_location("seedrep_served_w15d", REPO / "scripts" / "modal" / "seedrep_served.py")
    srv = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal")
    sys.modules["modal"] = fx._stub_modal()
    try:
        spec.loader.exec_module(srv)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    prod = _row(draft, W14_S1B_ROWS[0])
    h = lambda c: c if c == lp.NONE_LORA0 else lp.harness_id(c)  # noqa: E731
    reused = {h(c): (c == lp.NONE_LORA0) for c in prod["adapters"]}                       # what S1b's engine attests (§8 S1b)
    ep = fx._native_endpoint(prod, adapters_reused=reused)
    cells = fx._native_cells_served(prod)
    exported = lambda probs: [p for p in probs if "was EXPORTED" in p and "(gate 10)" in p]  # noqa: E731
    assert len(exported(srv.readiness_problems(ep, cells, prod))) == 5                    # reproduced: the production expectation refuses
    srow = lp.smoke_row(prod, lp.smoke_identity(draft, "S1b", SMOKE_WAVE), max_children=5, max_timeout_s=3600)
    assert exported(srv.readiness_problems(ep, cells, srow)) == []                        # W15d: the role-derived row accepts its own export
    assert [p for p in srv.readiness_problems(ep, cells, srow) if "(gate 10)" in p] == []
    # ... while a zero adapter the S1b engine EXPORTED (S1 never ran) is still refused: the dependency §8 declares
    assert len(exported(srv.readiness_problems(fx._native_endpoint(prod, adapters_reused={**reused, lp.NONE_LORA0: False}), cells, srow))) == 1
