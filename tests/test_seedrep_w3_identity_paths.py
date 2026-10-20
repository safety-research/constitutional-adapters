"""Every child's registry identity path is a file the producer actually writes.

Before this fix the plan spelled ``registry/plans/<eval_leg>/<arm>__s<s>.json`` with ITS OWN labels while the producer
(``registry_plans.plan_relpath`` / ``store_relpath``) files the Arditi children under ``ARDITI_ARM`` (``arditi_allpos``, not the staged
id ``ar_c0p3__arditi_allpos``) and the controls children ONLY as the per-task STORE map ``_stores/controls/<arm>__s<s>.json`` -> 450 + 96
= 546 of 2,274 children named files that never exist (S0 IDENTITY: FAIL, 546 missing / 0 content mismatches). The plan now CALLS the
producer's path functions with the producer's arm spelling (``lp.identity_plan_rel`` -> ``registry_plans_module()``), so this test
renders the plan and generates the producer's plan set on the REAL tree and holds every child to a file that exists and loads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import seedrep_launch_plan as lp  # noqa: E402

# the tree sha the plan is built at: a DERIVED 40-hex sha, never forty identical digits -- the registry refuses a planned run whose
# code_sha is a placeholder (validate.run_code_sha_refusal), the way the synthetic corpus states derived shas since that rule
W = hashlib.sha1(b"w3-identity-paths").hexdigest()
WAVE = "22222222-3333-4444-8444-555555555555"
CAMPAIGN = "seedrep_battery_w3_test"


@pytest.fixture(scope="module")
def RP():
    return lp.registry_plans_module()


@pytest.fixture(scope="module")
def plan():
    inp = lp.load_inputs(REPO)
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inp["reqs_sha256"], roster=inp["roster"], pins=inp["pins"], arditi_pins=inp["arditi_pins"],
                         arms=inp["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z", config_private_runs=inp["config_private_runs"])


@pytest.fixture(scope="module")
def producer(RP, tmp_path_factory):
    """The producer's plan set for THIS tree at W, written the way `registry_plans.py --out` writes it -> (PlanSet, out dir)."""
    out = tmp_path_factory.mktemp("rs") / "registry" / "plans"
    inp = RP.load_inputs(campaign=CAMPAIGN, w_sha=W)
    ps = RP.build(inp)
    RP.write_plan_set(ps, out)
    assert RP.check_plan_set(ps, out) == []
    return ps, out


def _children(plan):
    return [(r, k) for r in plan["standups"] + plan["rungs"] for k in r["children"]]


def test_every_child_identity_path_is_a_file_the_producer_writes_and_loads(plan, RP, producer):
    """The S0 IDENTITY gate's own walk (standups + rungs; the smoke-only s5_probe_rows produce no registry run): 0 missing, 0 unloadable."""
    _, out = producer
    kids = _children(plan)
    assert len(kids) == 2274
    missing, mismatched = [], []
    for row, k in kids:
        rel = k["registry_identity_rel"]
        assert rel.startswith(lp.PLANS_REL + "/") and rel.endswith(".json")
        p = out / rel[len(lp.PLANS_REL) + 1:]
        if not p.is_file():
            missing.append(rel)
            continue
        pf = RP.load_plan_file(p)
        want_arm = RP.ARDITI_ARM if k["arm"] == lp.ARDITI_STAGED_ID else k["arm"]
        if pf.arm != want_arm or pf.s != int(k["s"]) or pf.container != RP.container_of_leg(str(k["eval_leg"])):
            mismatched.append((rel, pf.arm, pf.s, pf.container, k["arm"], k["s"], k["eval_leg"]))
    assert missing == [], f"{len(missing)} child identity file(s) the producer never writes, e.g. {missing[:5]}"
    assert mismatched == []


def test_the_arditi_and_controls_children_name_the_producers_files(plan, RP):
    """The two mis-spellings by name: Arditi children -> `<leg>/arditi_allpos__s<s>.json`; controls children -> the STORE map."""
    rels = {k["registry_identity_rel"] for _, k in _children(plan)}
    assert not any("/ar_c0p3__arditi_allpos__s" in r for r in rels)
    assert not any(r.startswith(f"{lp.PLANS_REL}/controls/") for r in rels)
    arditi = {r for _, k in _children(plan) if k["arm"] == lp.ARDITI_STAGED_ID for r in [k["registry_identity_rel"]]}
    assert arditi and all(r.rsplit("/", 1)[1].startswith(f"{RP.ARDITI_ARM}__s") for r in arditi)
    controls = {k["registry_identity_rel"] for _, k in _children(plan) if k["eval_leg"] == "controls"}
    assert controls and all(r.startswith(f"{lp.PLANS_REL}/{RP.STORES_DIR}/controls/") for r in controls)
    assert f"{lp.PLANS_REL}/{RP.STORES_DIR}/controls/{RP.ARDITI_ARM}__s2.json" in controls
    assert f"{lp.PLANS_REL}/pasttense/{RP.ARDITI_ARM}__s1.json" in arditi
    # the lane children keep the earlier spelling byte-for-byte (they matched the producer already)
    assert f"{lp.PLANS_REL}/bon/none__s2.json" in rels and f"{lp.PLANS_REL}/misalign_leaking/none__s3.json" in rels


def test_identity_plan_rel_is_the_producers_relpath_for_every_planned_run(RP, producer):
    """Drift lock: over the producer's INDEX (every leg x arm x seed + every store map), the plan's path == the producer's path,
    computed by the producer's own functions -- no second spelling exists to drift."""
    ps, _ = producer
    idx = ps.index
    n = 0
    for leg, block in idx["legs"].items():
        for arm, per_seed in block["arms"].items():
            plan_arm = lp.ARDITI_STAGED_ID if arm == RP.ARDITI_ARM else arm
            for s, ent in per_seed.items():
                if leg in RP.STORE_LEGS.get("controls", ()):
                    continue                     # a controls leg's child reads the STORE map (below), never the per-leg file
                assert lp.identity_plan_rel(leg, plan_arm, int(s)) == f"{lp.PLANS_REL}/{ent['path']}" == f"{lp.PLANS_REL}/{RP.plan_relpath(leg, arm, int(s))}"
                n += 1
    for store, block in idx["stores"].items():
        for arm, per_seed in block["arms"].items():
            plan_arm = lp.ARDITI_STAGED_ID if arm == RP.ARDITI_ARM else arm
            for s, ent in per_seed.items():
                assert lp.identity_plan_rel(store, plan_arm, int(s)) == f"{lp.PLANS_REL}/{ent['path']}" == f"{lp.PLANS_REL}/{RP.store_relpath(store, arm, int(s))}"
                n += 1
    assert n == 12 * 27 + 27          # 12 single-identity legs x 27 arm-runs + the 27 controls store maps
    assert lp.ARDITI_STAGED_ID == RP.ARDITI_STAGED_ID and lp.registry_plans_module() is RP


def test_child_registry_env_exports_the_producers_file(plan, RP):
    """The container env (`child_registry_env`, the same derivation): the Arditi controls child names the STORE map under the producer's arm."""
    ctl = [k for r in plan["standups"] if r["leg"] == "controls" and r["engine"] == lp.ARDITI for k in r["children"] if k["arm"] == lp.ARDITI_STAGED_ID]
    assert ctl
    k = ctl[0]
    env = lp.child_registry_env(plans_root="/p", registry_identity_rel=k["registry_identity_rel"], attempt_root="/a")
    assert env[lp.RUN_IDENTITY_ENV] == f"/p/{RP.STORES_DIR}/controls/{RP.ARDITI_ARM}__s{int(k['s'])}.json" and env[lp.SPOOL_ENV] == f"/a/{lp.SPOOL_REL}"
    st = [k for r in plan["standups"] if r["leg"] == "pasttense" and r["engine"] == lp.ARDITI for k in r["children"]][0]
    assert lp.child_registry_env(plans_root="/p", registry_identity_rel=st["registry_identity_rel"], attempt_root="/a")[lp.RUN_IDENTITY_ENV] \
        == f"/p/pasttense/{RP.ARDITI_ARM}__s{int(st['s'])}.json"
    lane = [k for r in plan["standups"] if r["leg"] == "pasttense" and r["engine"] == "s1" for k in r["children"] if k["arm"] == "none"][0]
    assert lp.child_registry_env(plans_root="/p", registry_identity_rel=lane["registry_identity_rel"], attempt_root="/a")[lp.RUN_IDENTITY_ENV] == "/p/pasttense/none__s1.json"


def test_the_s0_identity_gate_passes_on_the_real_tree(plan, RP, producer, tmp_path):
    """The gate itself (`gate_identity_contents`), against a run-state root holding the producer's set for this tree."""
    _, out = producer
    rs = out.parents[1]
    spec = importlib.util.spec_from_file_location("seedrep_s0_gates_w3_identity", REPO / "scripts/modal/seedrep_s0_gates.py")
    S0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S0)
    res = S0.gate_identity_contents(plan=plan, tree=REPO, rs=rs, campaign=CAMPAIGN, registry_plans=RP)
    assert res["ok"] is True and res["n_missing"] == 0 and res["n_unloadable"] == 0 and res["n_problems"] == 0 and res["n_children"] == 2274, res
    # the old spelling, re-created beside it, is what the gate refused: 546 files missing, 0 content mismatches
    w2 = [f"{lp.PLANS_REL}/{k['eval_leg']}/{k['arm']}__s{int(k['s'])}.json" for _, k in _children(plan)]
    gone = [r for r in w2 if not (out / r[len(lp.PLANS_REL) + 1:]).is_file()]
    assert len(gone) == 546
