"""The Qwen seed-replicate native battery — the coverage table on a synthetic wave run through the
whole chain (audit → identity → coverage): every expected arm-run `complete` or `missing(<named reason>)` — no store,
attempt incomplete, store incomplete, audit REFUSED (with the audit's reason), identity FAIL, unaudited; the
`expected` subcommand derives the arm-runs from the s-view configs and refuses a view without both prompted routes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import coverage_table as ct  # noqa: E402
import engine_identity_table as eit  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402

PIN = "d" * 64
ARMS = ("none", "constv2-mix-full__psr_ce_all")


def _chain(tmp_path: Path, *, legs=("pasttense",), arditi=True):
    stores = tmp_path / "stores"
    roots = fx.build_wave(stores, legs=legs, views=("s1", "s2", "s3"), arms=ARMS, n_items=2, arditi=arditi, arditi_pin=PIN)
    audits = tmp_path / "audits"
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(audits), "--legs", ",".join(legs)]) == 0
    pins = fx.pins_file(tmp_path / "pins.json", pin=PIN)
    ident = tmp_path / "tables" / "engine_identity.csv"
    rc = ac.run_main(eit.main, ["--stores", str(stores), "--out", str(ident), "--arditi-pins", str(pins)])
    assert rc == (0 if arditi else ac.EXIT_FAILED)
    expected = fx.expected_file(tmp_path / "expected.json", arms=ARMS, views=("s1", "s2", "s3"), arditi_seeds=(1, 2, 3) if arditi else (), legs=legs)
    fx.launch_plan_file(tmp_path / "launch_plan.json", list(roots.values()))
    return stores, roots, audits / rea.SUMMARY_NAME, tmp_path / "tables" / "engine_identity_legs.json", expected


def _rewrite_summary(path: Path, fn) -> None:
    """Map `fn` over the audit summary's CSV rows in place (None drops the row) — how the tests plant a synthetic verdict defect."""
    with open(path, newline="") as f:
        rows = [r for r in (fn(r) for r in csv.DictReader(f)) if r is not None]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _rows(stores, expected, summary, legs_json, launch_plan=None):
    plan = launch_plan or (Path(expected).parent / "launch_plan.json")
    return ct.build(stores, expected=expected, launch_plan=plan, audit_summary=summary, identity_legs=legs_json,
                    petri_behaviours=list(fx.PETRI_BEHAVIOURS))


def test_full_chain_every_arm_run_complete(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    rows = _rows(stores, expected, summary, legs_json)
    assert len(rows) == 9  # 2 arms × 3 views + Arditi × 3 seeds
    assert all(r["status"] == ct.COMPLETE and r["reason"] is None and r["audit_verdict"] == "AGREES" for r in rows), [r for r in rows if r["status"] != "complete"]
    assert {(r["arm"], r["view"], r["s"]) for r in rows} == {(a, v, ac.seed_of_view(v)) for a in ARMS for v in ("s1", "s2", "s3")} | {(ac.ARDITI_CELL, "arditi", s) for s in (1, 2, 3)}
    ard = [r for r in rows if r["view"] == "arditi"]
    assert all(r["attempt_id"] == roots[("pasttense", "arditi")].attempt_id and r["n_rows"] == 4 for r in ard)
    assert all(r["store_dir"] and f"_seedrep_s{r['s']}_" in r["store_dir"] for r in ard)


def test_named_misses(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    # attempt incomplete: the s2 root loses its COMPLETE.json → both s2 arm-runs missing with the reason
    (roots[("pasttense", "s2")].path / ac.COMPLETE_FILE).unlink()
    # audit REFUSED for (none, s1): rewrite the summary row
    _rewrite_summary(summary, lambda r: {**r, "verdict": "REFUSED", "problem": "sampling / seed evidence: synthetic defect"}
                     if (r["arm"], r["s"]) == ("none", "1") else r)
    # an expected arm-run the plan says the s1 standup serves, but its accepted attempt never ran it (no store)
    exp = json.loads(expected.read_text())
    exp["arm_runs"].append({"arm": "prompted__safety_refusal", "view": "s1", "s": 1})
    expected.write_text(json.dumps(exp))
    fx.launch_plan_file(tmp_path / "launch_plan.json", list(roots.values()),
                        extra_arms={("pasttense", "s1", "base", 0): ["prompted__safety_refusal"]})
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    assert rows[("none", 2)]["status"] == "missing([shard00] attempt incomplete: no COMPLETE.json on the accepted root)"
    assert rows[("constv2-mix-full__psr_ce_all", 2)]["status"].startswith("missing([shard00] attempt incomplete")
    assert rows[("none", 1)]["status"] == "missing(audit REFUSED: [shard00] sampling / seed evidence: synthetic defect)"
    assert rows[("none", 1)]["audit_verdict"] == "REFUSED"
    aid = roots[("pasttense", "s1")].attempt_id
    assert rows[("prompted__safety_refusal", 1)]["status"] == f"missing([shard00] accepted attempt {aid} of standup pasttense__s1__base__shard00 holds no store for this arm-run)"
    assert rows[("prompted__safety_refusal", 1)]["attempt_id"] is None and rows[("prompted__safety_refusal", 1)]["audit_verdict"] == "unaudited"
    assert rows[("none", 3)]["status"] == ct.COMPLETE and rows[(ac.ARDITI_CELL, 3)]["status"] == ct.COMPLETE


def test_identity_fail_marks_every_arm_run_of_the_leg(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path, arditi=False)
    rows = _rows(stores, expected, summary, legs_json)
    assert len(rows) == 6 and all(r["status"] == "missing(identity: Arditi absent)" and r["audit_verdict"] == "AGREES" for r in rows)


def test_unaudited_arm_run_is_a_miss_and_store_incomplete_is_named(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    _rewrite_summary(summary, lambda r: None if (r["arm"], r["s"]) == ("none", "3") else r)
    # a static store whose progress lost a goal's terminal record
    shard = roots[("pasttense", "s1")].path / "exp" / f"{ac.MODEL_KEY}_s1_sampled_t0" / "pasttense" / "none" / "shard00"
    prog = ac.read_jsonl(shard / "progress.jsonl")
    (shard / "progress.jsonl").write_text(json.dumps(prog[0]) + "\n")
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    assert rows[("none", 3)]["status"] == "missing(unaudited: no request-evidence verdict for this arm-run)"
    st = rows[("none", 1)]["status"]
    assert st.startswith("missing([shard00] store incomplete: run_static's completeness rule:") and "4 response rows != 2 expected" in st
    assert "2 stored (goal_id, sample_idx)(s) the writer declares nowhere" in st


def test_misalign_result_rc_is_required(tmp_path: Path):
    stores = tmp_path / "stores"
    root = fx.make_root(stores, leg="misalign", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=1)
    (root.path / ac.RESULT_FILE).write_text("2026-09-18T00:00:00+00:00 job=manual target=x rc=1 agentic_rc=1\n")
    ar = ac.read_attempt_root(root.path)
    (st,) = ac.discover_stores(ar, "misalign_leaking")
    assert ct.store_status(ar, st, s=1) == ("missing", "misalign: RESULT rc=1")
    (root.path / ac.RESULT_FILE).unlink()
    assert ct.store_status(ar, st, s=1)[1].startswith("misalign: no RESULT line")


def test_refusals_on_missing_inputs(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    with pytest.raises(ac.AuditRefusal, match="run request_evidence_audit.py first"):
        _rows(stores, expected, tmp_path / "nope.csv", legs_json)
    with pytest.raises(ac.AuditRefusal, match="run engine_identity_table.py first"):
        _rows(stores, expected, summary, tmp_path / "nope.json")
    bad = tmp_path / "bad_expected.json"
    bad.write_text(json.dumps({"legs": ["pasttense"], "arm_runs": [{"arm": "none", "view": "s1", "s": 1}, {"arm": "none", "view": "s1", "s": 1}]}))
    with pytest.raises(ac.AuditRefusal, match="duplicate arm_runs"):
        _rows(stores, bad, summary, legs_json)
    bad.write_text(json.dumps({"legs": ["nope"], "arm_runs": [{"arm": "none", "view": "s1", "s": 1}]}))
    with pytest.raises(ac.AuditRefusal, match="unknown leg"):
        _rows(stores, bad, summary, legs_json)
    with pytest.raises(ac.AuditRefusal, match="missing"):
        _rows(stores, expected, summary, legs_json, launch_plan=tmp_path / "nope_plan.json")
    badplan = tmp_path / "bad_plan.json"
    badplan.write_text(json.dumps({"schema": "other/1", "W": fx.W_FIXTURE, "wave_uuid": fx.WAVE_UUID, "standups": [], "rungs": []}))
    with pytest.raises(ac.AuditRefusal, match="schema 'other/1' != 'seedrep_launch_plan/1'"):
        _rows(stores, expected, summary, legs_json, launch_plan=badplan)


def test_cli_build_exit_codes(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    out = tmp_path / "tables" / "coverage.csv"
    argv = ["build", "--stores", str(stores), "--expected", str(expected), "--launch-plan", str(tmp_path / "launch_plan.json"),
            "--audit-summary", str(summary), "--identity-legs", str(legs_json), "--out", str(out), "--petri-behaviours", ",".join(fx.PETRI_BEHAVIOURS)]
    assert ac.run_main(ct.main, argv) == 0
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 9 and {r["status"] for r in rows} == {"complete"} and set(rows[0]) == {c.rstrip("?") for c in ct.COLUMNS}
    assert all(r["reason"] == "" for r in rows)
    (roots[("pasttense", "s1")].path / ac.COMPLETE_FILE).unlink()
    assert ac.run_main(ct.main, argv) == ac.EXIT_FAILED
    assert ac.run_main(ct.main, argv[:-6] + ["--identity-legs", str(tmp_path / "nope.json"), "--out", str(out)]) == ac.EXIT_REFUSED


def test_expected_subcommand_from_view_configs(tmp_path: Path):
    cells = ["none", "c1__psr", "c2__lora", "c3__psr", "c4__lora", "c5__diff", *ac.PROMPTED_ARMS]
    for v in ("s1", "s2"):
        (tmp_path / f"sdfbase_full22_qwen35_4b_{v}.yaml").write_text("target: x\nexplicit_cells:\n" + "".join(f"- {c}\n" for c in cells) + "\ndefenses_dir: y\n")
    out = tmp_path / "expected.json"
    rc = ac.run_main(ct.main, ["expected", "--view-config", str(tmp_path / "sdfbase_full22_qwen35_4b_s1.yaml"),
                               "--view-config", str(tmp_path / "sdfbase_full22_qwen35_4b_s2.yaml"), "--out", str(out)])
    assert rc == 0
    exp = json.loads(out.read_text())
    assert exp["legs"] == list(ac.LEGS) and len(exp["arm_runs"]) == 8 * 2 + 3
    assert {"arm": "prompted__safety_refusal", "view": "s2", "s": 2} in exp["arm_runs"]
    assert [r for r in exp["arm_runs"] if r["view"] == "arditi"] == [{"arm": ac.ARDITI_CELL, "view": "arditi", "s": s} for s in (1, 2, 3)]
    # a view without both prompted routes refuses (they have not landed on it)
    (tmp_path / "sdfbase_full22_qwen35_4b_s3.yaml").write_text("explicit_cells:\n- none\n- c1__psr\n")
    assert ac.run_main(ct.main, ["expected", "--view-config", str(tmp_path / "sdfbase_full22_qwen35_4b_s3.yaml"), "--out", str(out)]) == ac.EXIT_REFUSED
    # the REAL s1 config at this base has no prompted routes yet → refused
    real = REPO / "configs" / "sdfbase_full22_qwen35_4b_s1.yaml"
    cells_now = ct._explicit_cells(real)
    if not ct.PROMPTED_MISSING(cells_now):
        pytest.skip("WS0-1 has landed the prompted routes on s1")
    assert ac.run_main(ct.main, ["expected", "--view-config", str(real), "--out", str(out)]) == ac.EXIT_REFUSED


# --------------------------------------------------------------------------- #
# P1-3: coverage resolves the FINAL accepted materialisation and verifies every stage of its ancestry                   #
# --------------------------------------------------------------------------- #

def _ladder_chain(tmp_path: Path, *, rung: bool, rung_complete: bool = True):
    """A static base (g0 at cap — 1/2 of the rows, above the 1 % trigger) with, optionally, its 8192 rung root; the audit
    and identity CLIs run over the WHOLE stores root (rung roots included) exactly as the orchestrator runs them."""
    stores = tmp_path / "stores"
    roots = {}
    for view in ("s1", "s2", "s3"):
        r = fx.make_root(stores, leg="pasttense", view=view, arms=("none",))
        base_shard = fx.write_static_store(r, family="pasttense", arm="none", s=ac.seed_of_view(view), goals=["g0", "g1"], at_cap_goals=("g0",))
        r.flush_pipeline()
        roots[view] = (r, base_shard)
    ard = fx.make_root(stores, leg="pasttense", view="arditi", arms=(ac.ARDITI_CELL,), arditi_pin=PIN)
    for s in (1, 2, 3):
        fx.write_static_store(ard, family="pasttense", arm=ac.ARDITI_CELL, s=s, goals=["g0", "g1"])
    ard.flush_pipeline()
    rung_root = None
    if rung:  # the s1 rung: g0 re-drawn at 8192, terminated → the ladder exits at 8192
        rung_root = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
        fx.write_static_rung(rung_root, base_shard=roots["s1"][1], family="pasttense", arm="none", s=1, cap_from=4096, cap_to=8192, still_at_cap=())
        rung_root.flush_pipeline(complete=rung_complete)
    audits = tmp_path / "audits"
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(audits), "--legs", "pasttense"]) == 0
    pins = fx.pins_file(tmp_path / "pins.json", pin=PIN)
    ident = tmp_path / "tables" / "engine_identity.csv"
    assert ac.run_main(eit.main, ["--stores", str(stores), "--out", str(ident), "--arditi-pins", str(pins)]) == 0
    expected = fx.expected_file(tmp_path / "expected.json", arms=("none",), views=("s1", "s2", "s3"), arditi_seeds=(1, 2, 3), legs=("pasttense",))
    fx.launch_plan_file(tmp_path / "launch_plan.json", [r for r, _ in roots.values()] + [ard] + ([rung_root] if rung_root else []))
    return stores, roots, rung_root, audits / rea.SUMMARY_NAME, tmp_path / "tables" / "engine_identity_legs.json", expected


def test_base_agrees_does_not_suffice_when_the_ladder_requires_a_rung(tmp_path: Path):
    """Every base store AGREES, yet the three s-view arm-runs are above the 1 % trigger with no rung root: the old table
    called them complete; now the ladder state is a named miss. The Arditi arm-runs (nothing at cap) are complete."""
    stores, roots, _, summary, legs_json, expected = _ladder_chain(tmp_path, rung=False)
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    for s in (1, 2, 3):
        r = rows[("none", s)]
        assert r["status"] == "missing(ladder: escalate_pending(8192) — the final rung is not materialised (rungs present: [4096], final at-cap share 0.5))"
        assert r["audit_verdict"] == "AGREES" and r["ladder_state"] == "escalate_pending(8192)" and r["final_stage"] is None and r["n_stages_verified"] == 1
        assert rows[(ac.ARDITI_CELL, s)]["status"] == ct.COMPLETE and rows[(ac.ARDITI_CELL, s)]["ladder_state"] == "exit_le_1pct"
        assert rows[(ac.ARDITI_CELL, s)]["final_stage"] == "base" and rows[(ac.ARDITI_CELL, s)]["rungs_present"] == [4096]


def test_rung_materialised_and_agreeing_completes_with_its_ancestry(tmp_path: Path):
    stores, roots, rung_root, summary, legs_json, expected = _ladder_chain(tmp_path, rung=True)
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    r = rows[("none", 1)]
    assert r["status"] == ct.COMPLETE and r["final_stage"] == "8192" and r["ladder_state"] == "exit_le_1pct" and r["rungs_present"] == [4096, 8192]
    assert r["n_stages_verified"] == 2 and r["attempt_id"] == f"{roots['s1'][0].attempt_id},{rung_root.attempt_id}"
    assert r["n_rows"] == 4 and "_cap8192" in r["store_dir"] and r["audit_verdict"] == "AGREES"
    assert rows[("none", 2)]["status"].startswith("missing(ladder: escalate_pending(8192)")  # s2 / s3 still have no rung


def test_failed_pending_or_missing_final_rungs_are_named_misses(tmp_path: Path):
    stores, roots, rung_root, summary, legs_json, expected = _ladder_chain(tmp_path, rung=True)
    # (a) the rung's audit verdict REFUSED → the arm-run is a miss naming the rung, even though the base AGREES
    _rewrite_summary(summary, lambda r: {**r, "verdict": "REFUSED", "problem": "row binding: synthetic defect on the rung"}
                     if (r["arm"], r["s"], r["stage"]) == ("none", "1", "8192") else r)
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    assert rows[("none", 1)]["status"] == "missing(rung 8192: audit REFUSED: [shard00] row binding: synthetic defect on the rung)"
    assert rows[("none", 1)]["n_stages_verified"] == 1 and rows[("none", 1)]["final_stage"] is None
    # (b) the rung's verdict absent from the summary → unaudited at that stage
    _rewrite_summary(summary, lambda r: None if (r["arm"], r["s"], r["stage"]) == ("none", "1", "8192") else r)
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    assert rows[("none", 1)]["status"] == "missing(rung 8192: unaudited: no request-evidence verdict for this arm-run at stage 8192)"


def test_rung_root_without_complete_json_is_a_named_miss(tmp_path: Path):
    stores, roots, rung_root, summary, legs_json, expected = _ladder_chain(tmp_path, rung=True, rung_complete=False)
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    assert rows[("none", 1)]["status"] == "missing(rung 8192: [shard00] attempt incomplete: no COMPLETE.json on the accepted root)"
    assert rows[("none", 1)]["audit_verdict"] == "AGREES"  # the rung's verdict itself agreed; the attempt is what is incomplete


# --------------------------------------------------------------------------- #
# P1-4: the launch plan is the declared shard / producer inventory; every verdict is bound to the accepted attempt       #
# --------------------------------------------------------------------------- #

def test_verdict_of_another_attempt_is_stale_evidence(tmp_path: Path):
    """The (leg, arm, s) join let an OLD attempt's AGREES certify the current store; a verdict is now bound to the accepted
    attempt id the plan row resolves to."""
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    _rewrite_summary(summary, lambda r: {**r, "attempt_id": "ta-pasttense-s1-shard00-base-0ld0ld0l"} if (r["arm"], r["s"]) == ("none", "1") else r)
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    cur = roots[("pasttense", "s1")].attempt_id
    assert rows[("none", 1)]["status"] == f"missing([shard00] the audit verdict is for attempt 'ta-pasttense-s1-shard00-base-0ld0ld0l', not the accepted {cur!r} — stale evidence)"
    assert rows[("none", 2)]["status"] == ct.COMPLETE


def test_absent_planned_shard_is_enumerated(tmp_path: Path):
    """The plan declares TWO base shards for (pasttense, s1); only shard00 was pulled → every s1 arm-run names the missing
    planned standup (the old table saw one complete shard and called the arm-run complete)."""
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    fx.launch_plan_file(tmp_path / "launch_plan.json", list(roots.values()), extra=[("pasttense", "s1", "base", 1, list(ARMS))])
    rows = {(r["arm"], r["s"]): r for r in _rows(stores, expected, summary, legs_json)}
    for arm in ARMS:
        assert rows[(arm, 1)]["status"] == "missing(planned standup pasttense__s1__base__shard01 has no accepted attempt)"
        assert rows[(arm, 2)]["status"] == ct.COMPLETE
    plan = json.loads((tmp_path / "launch_plan.json").read_text())
    assert {r["nshards"] for r in plan["standups"] if r["leg"] == "pasttense" and r["view"] == "s1"} == {2}


def test_two_planned_shards_both_accepted_complete_and_list_both_attempts(tmp_path: Path):
    stores = tmp_path / "stores"
    shards = {}
    for shard, goals in (("shard00", ["g0", "g1"]), ("shard01", ["g2"])):
        r = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), shard=shard)
        fx.write_static_store(r, family="pasttense", arm="none", s=1, goals=goals)
        r.flush_pipeline()
        shards[shard] = r
    audits = tmp_path / "audits"
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(audits), "--legs", "pasttense"]) == 0
    legs_json = tmp_path / "engine_identity_legs.json"
    legs_json.write_text(json.dumps({"pasttense": {"verdict": "PASS", "problems": []}}))
    expected = fx.expected_file(tmp_path / "expected.json", arms=("none",), views=("s1",), legs=("pasttense",))
    plan = fx.launch_plan_file(tmp_path / "launch_plan.json", list(shards.values()))
    (row,) = _rows(stores, expected, audits / rea.SUMMARY_NAME, legs_json, launch_plan=plan)
    assert row["status"] == ct.COMPLETE and row["n_rows"] == 6 and row["n_stages_verified"] == 1
    assert row["attempt_id"] == f"{shards['shard00'].attempt_id},{shards['shard01'].attempt_id}"
    # the same wave with shard01's root NOT pulled: the planned shard is enumerated as the miss
    import shutil
    shutil.rmtree(shards["shard01"].path)
    (row,) = _rows(stores, expected, audits / rea.SUMMARY_NAME, legs_json, launch_plan=plan)
    assert row["status"] == "missing(planned standup pasttense__s1__base__shard01 has no accepted attempt)"


def test_undeclared_attempt_or_another_tree_refuses(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    # a root the plan does not declare (the plan written WITHOUT the s3 root)
    fx.launch_plan_file(tmp_path / "launch_plan.json", [r for k, r in roots.items() if k[1] != "s3"])
    with pytest.raises(ac.AuditRefusal, match="executed standup 'pasttense__s3__base__shard00', which the launch plan does not declare"):
        _rows(stores, expected, summary, legs_json)
    # a plan from another tree
    fx.launch_plan_file(tmp_path / "launch_plan.json", list(roots.values()), W="0" * 40)
    with pytest.raises(ac.AuditRefusal, match="ran from W f{40} but the launch plan is W 0{40}"):
        _rows(stores, expected, summary, legs_json)
    # an expected arm-run no planned standup serves
    fx.launch_plan_file(tmp_path / "launch_plan.json", list(roots.values()))
    exp = json.loads(expected.read_text())
    exp["arm_runs"].append({"arm": "prompted__safety_refusal", "view": "s1", "s": 1})
    expected.write_text(json.dumps(exp))
    with pytest.raises(ac.AuditRefusal, match="no planned base standup of \\(pasttense, s1\\) serves arm-run \\(pasttense, prompted__safety_refusal, s1\\)"):
        _rows(stores, expected, summary, legs_json)


# --------------------------------------------------------------------------- #
# the launch-plan fixture is hand-shaped to the generator — its spellings are PINNED to the generator's                         #
# --------------------------------------------------------------------------- #

def test_launch_plan_fixture_spells_ws2d_generators_ids():
    """The fixture's `standup_id` / `control_key` / Arditi arm label / schema are the generator's
    (`scripts/modal/seedrep_launch_plan.py` @ 0e51107b: `standup_id`, `control_key`, `ARDITI_ARM_ID`, `PLAN_SCHEMA`), pinned
    here as literals read at that tip; when the generator itself is on the tree (the W merge) the same values are held to
    ITS functions, so a spelling change on either side fails this test rather than the coverage table at wave time."""
    W7, wave = "abcdef0", "11111111-2222-3333-4444-555555555555"
    assert fx.standup_id("bon", "s1", "base", 0) == "bon__s1__base__shard00"
    assert fx.standup_id("pasttense", "arditi", "8192", 1) == "pasttense__arditi__8192__shard01"
    assert fx.control_key(W7, wave, "bon", "s1", "base", 0) == f"{W7}/{wave}/bon/s1/base/0"
    assert fx.control_key(W7, wave, "pasttense", "arditi", "8192", 1) == f"{W7}/{wave}/pasttense/arditi/8192/1"
    assert fx.ARDITI_PLAN_ARM == "ar_c0p3__arditi_allpos" and fx.LAUNCH_PLAN_SCHEMA == ac.LAUNCH_PLAN_SCHEMA == "seedrep_launch_plan/1"
    # the audit's own reader agrees with the fixture's spelling (standup_id_of / parse_control_key round-trip)
    assert ac.standup_id_of("bon", "s1", "base", 0) == fx.standup_id("bon", "s1", "base", 0)
    assert ac.parse_control_key(fx.control_key(W7, wave, "bon", "s1", "base", 0)) == {
        "W7": W7, "wave_uuid": wave, "leg": "bon", "view": "s1", "stage": "base", "shard": "shard00"}
    gen_path = REPO / "scripts" / "modal" / "seedrep_launch_plan.py"
    if not gen_path.is_file():
        return  # the launch-plan generator is not on this tree; the literal pins above hold its spelling until the W merge
    gen = ac.import_by_path("_ws2d_launch_plan", "scripts/modal/seedrep_launch_plan.py", what="WS2-D's launch-plan generator")
    for leg, view, stage, shard in (("bon", "s1", "base", 0), ("pasttense", "arditi", "8192", 1), ("controls", "s3", "2048", 3)):
        assert fx.standup_id(leg, view, stage, shard) == gen.standup_id(leg, view, stage, shard)
        assert fx.control_key(W7, wave, leg, view, stage, shard) == gen.control_key(W7, wave, leg, view, stage, shard)
    assert fx.ARDITI_PLAN_ARM == gen.ARDITI_ARM_ID and fx.LAUNCH_PLAN_SCHEMA == gen.PLAN_SCHEMA and gen.STAGE_BASE == ac.STAGE_BASE
    # the generator's Arditi arm-runs: label ARDITI_ARM_ID, cell arditi_allpos, s in {1,2,3} — the shape `LaunchPlan.serves` keys on
    ar = gen.arm_runs(gen.ARDITI)
    assert [(a["arm"], a["cell"], a["s"]) for a in ar] == [(gen.ARDITI_ARM_ID, ac.ARDITI_CELL, s) for s in (1, 2, 3)]
    assert [(a["arm"], a["cell"], a["s"]) for a in gen.arm_runs("s2")][0] == ("none", "none", 2)


# --------------------------------------------------------------------------- #
# nshards + a contiguous shard inventory; the attempt's wave is REQUIRED and bound to the plan                         #
# --------------------------------------------------------------------------- #

def test_launch_plan_nshards_contiguity_and_attempt_wave_are_required(tmp_path: Path):
    stores, roots, summary, legs_json, expected = _chain(tmp_path)
    plan_p = tmp_path / "launch_plan.json"
    # nshards 2 on the one declared shard row → refused (the inventory the plan claims is not the one it declares)
    plan = json.loads(plan_p.read_text())
    for r in plan["standups"]:
        if r["leg"] == "pasttense" and r["view"] == "s1":
            r["nshards"] = 2
    plan_p.write_text(json.dumps(plan))
    with pytest.raises(ac.AuditRefusal, match="\\(pasttense, s1, base\\): nshards 2 but 1 row\\(s\\) declared"):
        _rows(stores, expected, summary, legs_json)
    # shards 00 and 02 (nshards 2 each) → not the contiguous inventory
    fx.launch_plan_file(plan_p, list(roots.values()), extra=[("pasttense", "s1", "base", 2, list(ARMS))])
    with pytest.raises(ac.AuditRefusal, match="shards \\['shard00', 'shard02'\\] are not the contiguous inventory \\['shard00', 'shard01'\\]"):
        _rows(stores, expected, summary, legs_json)
    fx.launch_plan_file(plan_p, list(roots.values()))
    # an attempt record naming NO wave (neither wave_uuid nor a control_key) → refused at the reader
    r1 = roots[("pasttense", "s1")]
    att = json.loads((r1.path / ac.ATTEMPT_FILE).read_text())
    (r1.path / ac.ATTEMPT_FILE).write_text(json.dumps({k: v for k, v in att.items() if k not in ("wave_uuid", "control_key")}))
    with pytest.raises(ac.AuditRefusal, match="names no wave"):
        ac.read_attempt_root(r1.path)
    # the control key alone names the wave → derived from it, and bound
    (r1.path / ac.ATTEMPT_FILE).write_text(json.dumps({k: v for k, v in att.items() if k != "wave_uuid"}))
    assert ac.read_attempt_root(r1.path).wave_uuid == fx.WAVE_UUID
    assert all(r["status"] == ct.COMPLETE for r in _rows(stores, expected, summary, legs_json) if r["view"] == "s1")
    # an attempt of ANOTHER wave (record and control key agreeing on it) → the plan binding refuses
    other = "22222222-3333-4444-5555-666666666666"
    (r1.path / ac.ATTEMPT_FILE).write_text(json.dumps({**att, "wave_uuid": other, "control_key": att["control_key"].replace(fx.WAVE_UUID, other)}))
    with pytest.raises(ac.AuditRefusal, match=f"belongs to wave {other} but the launch plan is wave {fx.WAVE_UUID}"):
        _rows(stores, expected, summary, legs_json)


# --------------------------------------------------------------------------- #
# a ladder gap is a NAMED miss in coverage, never complete                                                             #
# --------------------------------------------------------------------------- #

def test_ladder_gap_is_a_named_miss(tmp_path: Path):
    stores = tmp_path / "stores"
    base = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",))
    base_shard = fx.write_static_store(base, family="pasttense", arm="none", s=1, goals=["g0", "g1"], at_cap_goals=("g0",))
    base.flush_pipeline()
    ceil = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="16384")
    fx.write_static_rung(ceil, base_shard=base_shard, family="pasttense", arm="none", s=1, cap_from=4096, cap_to=16384, still_at_cap=())
    ceil.flush_pipeline()
    audits = tmp_path / "audits"
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(audits), "--legs", "pasttense"]) == 0  # both stores AGREE
    legs_json = tmp_path / "engine_identity_legs.json"
    legs_json.write_text(json.dumps({"pasttense": {"verdict": "PASS", "problems": []}}))
    expected = fx.expected_file(tmp_path / "expected.json", arms=("none",), views=("s1",), legs=("pasttense",))
    plan = fx.launch_plan_file(tmp_path / "launch_plan.json", [base, ceil])
    (row,) = _rows(stores, expected, audits / rea.SUMMARY_NAME, legs_json, launch_plan=plan)
    assert row["status"] == ("missing(ladder: ladder_gap(present [4096, 16384], ladder prefix [4096, 8192], missing [8192]) — a rung the "
                             "ladder requires below the final one was never materialised)")
    assert row["audit_verdict"] == "AGREES" and row["final_stage"] is None and row["rungs_present"] == [4096, 16384]
