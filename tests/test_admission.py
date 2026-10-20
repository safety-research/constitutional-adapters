"""reproduce/lib/admission.py — the GLOBAL unsteered-first gate over the schema-2 inventory lib/stage.py freezes from the ONE
unit plan 06 submits.

The baseline wave needs no gate (06 skips it for the baseline view: u0 when the table declares it, the single-phase recipe's one wave battery on
the demo, whose steered legs it still prices); a steered launch is refused BY NAME for: a never-submitted unit
(`unbound`), a submitted-but-unregistered unit, a missing padding level, a shard whose UNIQUE COMPLETED goal ids miss any frozen
goal (however many JSONL lines it holds), a BoN climb shard whose manifest draws are not all written, a tinycap store short of
its item set (or attesting other item ids), a baseline bound to another code sha, an expected eval INSIDE a unit with no bound
run, a coverage record collected for other run ids, a NULL or non-exact decode identity where one is frozen, a foreign decode
policy, a Petri cell store short of a frozen `<behavior>/<scenario>` (the runner's failure ledger or a missing dimension), a missing
family verdict, a torn binding record, a schema-1 inventory; the
inventory, bindings and verdicts survive a "restart" (they are files); `--force-gate` is the recorded override. The frozen
instrument ids are the 64-hex ids the registry's own planner mints (never a hand-written seed name); the frozen work identities
are the producers' own goal / scenario ids and the seeds' item-set sizes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, PETRI_BEHAVIORS, add_grading_tables, make_registry, petri_bundle, petri_scenario_ids  # noqa: E402
from seedrep_history_fixtures import pinned_commit_or_skip  # noqa: E402

sys.path.insert(0, str(LIB))
import admission as A  # noqa: E402
import stage as ST  # noqa: E402

SHA = "a" * 40
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_RE = re.compile(r"^sdfbase_full22_(?P<key>[a-z0-9_]+?)_sampled_t(?P<trial>[01])(?P<esc>_cap(?:8192|16384))?$")   # grade_static_sampled.py
MODEL = "qwen35_4b_demo"


def goal_fixture(tmp: Path) -> Path:
    """A synthetic goal pool in the offline fixture's shape (grade_static_sampled.py --goals-jsonl): 259 goals, the first 100 JBB."""
    p = tmp / "goals.jsonl"
    p.write_text("".join(json.dumps({"goal_id": f"goal-{i:03d}", "dataset": "jbb" if i < 100 else "harmbench", "text": "t", "category": "c", "revision": "r"}) + "\n" for i in range(259)))
    return p


def wave_config(tmp: Path) -> Path:
    p = tmp / "wave.yaml"
    p.write_text(f"private_run: sdfbase_full22_{MODEL}_battery\nbon:\n  n_max: 75\n  goal_subset: jbb\n")
    return p


def freeze(state: Path, families=None) -> dict:
    (state / "gate").mkdir(parents=True, exist_ok=True)
    units = ST.expected_units(MODEL, SHA, str(REPRO / "models.yaml"), families=families, goals_jsonl=str(goal_fixture(state)), wave_config=str(wave_config(state)),
                              petri_bundle=str(petri_bundle(state)))
    doc = {"schema": 2, "model": MODEL, "view": "battery", "code_sha": SHA, "units": units}
    ST.write_atomic(str(state / "gate" / "expected_units.json"), doc)
    return doc


def probe_ok(ev: dict, *, array_task=None, store="/stores/x", items_sha=None, drop_goals=(), climb_expected=None):
    """A coverage_probe.py document that SATISFIES the frozen eval (minus `drop_goals`) — the shape the real probe prints per layout."""
    cov = ev["coverage"]; lay = cov["layout"]
    blk = lambda **kw: {"present": True, "completed": [], "errored": [], "refused": [], "expected": None, "notes": [], **kw}  # noqa: E731
    if lay == "jailbreak_cellstore":
        return {"layout": lay, "path": store, "exists": True, "notes": [],
                "shards": {s: blk(completed=[g for g in ids if g not in drop_goals], errored=list(drop_goals)) for s, ids in cov["expected"].items()}}
    if lay == "bon_store512":
        return {"layout": lay, "path": store, "exists": True, "notes": [], "shards": {"shard00": blk(completed=[g for g in cov["expected"]["shard00"] if g not in drop_goals])}}
    if lay == "bon_climb_flat":
        exp = list(climb_expected if climb_expected is not None else ["gA", "gB", "gC"])
        return {"layout": lay, "path": store, "exists": True, "notes": [], "shards": {f"shard{array_task or 0:02d}": blk(expected=exp, completed=[g for g in exp if g not in drop_goals])}}
    if lay == "capability_cellstore":
        n = int(cov["n_items"]) - len(drop_goals)
        return {"layout": lay, "path": store, "exists": True, "notes": [], "shards": {},
                "items": {"present": True, "task": cov["task"], "completed": [f"item-{i}" for i in range(n)], "n_rows": n, "tasks_seen": {cov["task"]: n},
                          "metrics_seen": ["acc"], "items_sha256": items_sha if items_sha is not None else cov.get("items_sha256"), "n_attested_items": n}}
    if lay == "misalign_responses":
        return {"layout": lay, "path": store, "exists": True, "notes": [], "shards": {"shard00": blk(completed=[g for g in cov["expected"]["shard00"] if g not in drop_goals])}}
    if lay == "petri_cellstore":
        return {"layout": lay, "path": store, "exists": True, "notes": [], "results_json_present": not drop_goals, "unresolved_n": len(drop_goals), "behavior": cov.get("behavior"),
                "shards": {"shard00": blk(completed=[g for g in cov["expected"]["shard00"] if g not in drop_goals], errored=list(drop_goals))}}
    return {"layout": lay, "path": store, "exists": False, "shards": {}, "notes": ["unverified"]}


def satisfy(state: Path, unit: dict, runs: list, policies: list, *, code_sha=SHA, state_name="registered", drop_eval=None, cov_run_ids=None,
            policy_ref=None, decode_policy_id="set", decode_match=None, probe=None, drop_goals=(), items_sha=None, contract_sha=None,
            rid_suffix="", rule=True):
    """Bind + register + cover one unit the way 06 --gate and the harvester would: one run per (eval, task) with a NON-NULL decode id
    resolving to the frozen ref (decode_policy_match exact) unless the test says otherwise; the coverage record carries a satisfying
    probe document per run (or the test's `probe`). The family verdict is (re)written the way `06 --gate <family> --headline … --range …`
    records it — `run_ids` = the family's registered bindings NOW (stage.family_bound_run_ids; a verdict is bound to the runs it ruled) —
    unless `rule=False` (a bind-only re-gate, which writes no verdict). `rid_suffix` mints OTHER run ids for the unit (a re-run's registration
    after a resubmission: new ids bound and covered, the registry rows added beside the dead attempt's)."""
    lw = unit["logical_work_id"]; ledger = str(state / "submissions.jsonl")
    ST.write_binding(ledger, ST.submitted_record(lw, "1", "cluster_a", f"{lw}@cluster_a@1", array=unit.get("array")))
    tasks = ST.array_task_indices(unit["array"]) if unit.get("array") else [None]
    rids = {(e, t): f"run-{ST.safe_id(lw)[:40]}-{e}{'' if t is None else f'-t{t}'}{rid_suffix}" for e in unit["evals"] if e != drop_eval for t in tasks}
    if state_name == "registered":
        ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": list(rids.values()), "job_id": "1", "cluster": "cluster_a",
                                  "runs": [{"run_id": r, "eval_id": e, "object": "none", "array_task": t} for (e, t), r in rids.items()]})
        for (e, t), rid in rids.items():
            ev = unit["evals"][e]
            pid = f"pid-{ST.safe_id(lw)[:40]}-{e}" if decode_policy_id == "set" else decode_policy_id     # one policy row per (unit, eval)
            if ev["decode_policy_ref"] and pid:
                policies.append((pid, policy_ref or ev["decode_policy_ref"]))
            runs.append({"run_id": rid, "lifecycle_state": "ingested", "model_id": unit["model_id"], "code_sha": code_sha,
                         "instrument_id": ev["instrument_id"] or f"inst-{e}", "decode_policy_id": pid if ev["decode_policy_ref"] else None,
                         "decode_policy_match": decode_match, "eval_id": e, "object_id": f"obj:{ST.safe_id(lw)[:40]}", "store_uri": f"/stores/{rid}"})   # DISTINCT per unit
    # the record is TIED to the inventory's coverage contract as it is to the run ids — 06 --gate writes the contract block it
    # read the wave under; this helper writes the inventory's sha the same way (A.inventory_contract_sha; `contract_sha` overrides it for the tests)
    cov = {"logical_work_id": lw, "run_ids": cov_run_ids if cov_run_ids is not None else list(rids.values()), "runs": {},
           "coverage_contract": {"sha256": contract_sha if contract_sha is not None else A.inventory_contract_sha(str(state)), "version": ST.COVERAGE_CONTRACT_VERSION}}
    for (e, t), rid in rids.items():
        ev = unit["evals"][e]
        cov["runs"][rid] = {"eval_id": e, "object": "none", "array_task": t, "store_uri": f"/stores/{rid}",
                            "probe": probe if probe is not None else probe_ok(ev, array_task=t, store=f"/stores/{rid}", drop_goals=drop_goals, items_sha=items_sha), "probe_error": None}
    ST.write_atomic(ST.coverage_path(str(state), lw), cov)
    if rule:
        rule_family(state, unit["family"])


def rule_family(state: Path, family: str, **fields) -> dict:
    """Write gate/<family>.json as `06 --gate <family> --headline … --range …` does: a `pass` (unless `fields` says otherwise) whose `run_ids` are
    the family's registered bindings NOW (stage.family_bound_run_ids over the frozen inventory's units of the family — the ONE rule 06 and
    admission share). A standing verdict's other fields are kept."""
    vp = state / "gate" / f"{family}.json"
    doc = json.load(open(vp)) if vp.exists() else {"family": family, "headline": 0.5, "plausible_range": [0, 1], "verdict": "pass", "ruled_utc": ST.utc_now()}
    doc["run_ids"] = family_bindings(state, family)
    doc.update(fields)
    ST.write_atomic(str(vp), doc)
    return doc


def family_bindings(state: Path, family: str) -> list:
    """The run ids a family's units are bound to NOW: the ordered-unique union of each inventory unit's registered binding — what 06 --gate's
    verdict step records (computed here from binding_state alone, so the fixture stands on a tree without the production helper and the
    tests fail there for the defect, never for an import)."""
    ledger = str(state / "submissions.jsonl"); out: list = []
    for u in json.load(open(state / "gate" / "expected_units.json"))["units"]:
        if u["family"] == family:
            out.extend(r for r in (ST.binding_state(ledger, u["logical_work_id"]).get("run_ids") or []) if r not in out)
    return out


def by_unit(doc):
    return {u["logical_work_id"].split("/", 2)[2]: u for u in doc["units"]}


def test_inventory_freezes_the_unit_plan_never_run_ids(tmp_path):
    doc = freeze(tmp_path)
    fams = {u["family"] for u in doc["units"]}
    assert fams == {"static", "padding", "bon", "multiturn", "petri", "agentic", "controls", "mmlu", "gsm8k"}
    assert all("run_id" not in u for u in doc["units"])
    u = by_unit(doc)
    # padding: FIVE levels, one unit each — the single-phase baseline wave pads through the obj launcher's PER-VIEW family (grok420 alone,
    # target_static_padded_obj.sbatch); the four-statics padding path (target_static_padded.sbatch) is a declared u0's
    assert sorted(x["padding_tokens"] for x in doc["units"] if x["family"] == "padding") == [0, 8192, 16384, 32768, 65536]
    assert set(u["padding/p0"]["evals"]) == {"grok420"} and u["padding/p0"]["script"].endswith("static_padding/target_static_padded_obj.sbatch")
    pad_rows = [r for r in ST.unit_plan(MODEL, "battery", table=str(REPRO / "models.yaml"), families=["padding"], sha=SHA) if r["registers"]]
    assert [r["args"][-4:] for r in pad_rows] == [["grok420", "3", "all", "battery"]] * 5 and all(r["name"] == f"padobj-{MODEL}_battery-p{r['padding_tokens']}" for r in pad_rows)
    # the statics in the SAMPLED lane's form: two trials, four attacks each, a private_run the static grader admits
    assert set(u["static/t0"]["evals"]) == {"pasttense", "codeattack", "renellm", "grok420"} and "static/t1" in u
    assert RUN_RE.match(u["static/t0"]["private_run"]) and u["static/t0"]["private_run"] == f"sdfbase_full22_{MODEL}_battery_sampled_t0"
    assert u["static/t0"]["script"].endswith("sampled/target_sampled.sbatch")
    # the BoN store (submit_full22.sh's third job) and the climb are two units of the bon family
    assert set(x["unit"] for x in doc["units"] if x["family"] == "bon") == {"store512", "climb"}
    assert u["bon/climb"]["evals"]["strongreject_bon"]["expected_shards"] == ["shard00", "shard01", "shard02", "shard03"] and u["bon/climb"]["array"] == "0-3"
    # multi-turn: 3 attacks x the 4 disjoint shard groups, each unit covering ITS group only (grouped jobs bind exactly)
    mt = {k: v for k, v in u.items() if v["family"] == "multiturn"}
    assert set(mt) == {f"multiturn/{a}/{g}" for a in ("crescendo", "pair", "actorattack") for g in ("g012", "g34", "g56", "g78")}
    assert mt["multiturn/pair/g34"]["evals"]["pair"]["expected_shards"] == ["shard03", "shard04"]
    # controls: two trials, both over-refusal evals each
    assert set(u["controls/t1"]["evals"]) == {"xstest", "or_bench_hard"}
    # one gsm8k unit per cell of the view — the single-phase recipe's one wave: `none`, the two prompted instructions and every arm
    import models as MT
    s1_cells = MT.model(MODEL, str(REPRO / "models.yaml"))["views"]["battery"]["cells"]
    assert len(s1_cells) == 10 and {"none", "prompted__safety_refusal", "prompted__si_constitution", "difference_relu"} <= set(s1_cells)
    assert {x["unit"] for x in doc["units"] if x["family"] == "gsm8k"} == set(s1_cells)


def test_frozen_ids_are_the_registry_planners_not_seed_names(tmp_path):
    doc = freeze(tmp_path, families=["static", "bon", "controls", "mmlu"])
    u = by_unit(doc)
    for name, ev, ref, seed in (("static/t0", "pasttense", f"{MODEL}|static|off|vendor_preset", "static_sampled_vendor_preset_4096_8192"),
                                ("bon/climb", "strongreject_bon", f"{MODEL}|bon|off|grandfathered_T1", "bon_512store_4096climb_T1"),
                                ("controls/t0", "xstest", f"{MODEL}|over_refusal|off|vendor_preset", "overrefusal_xstest_sampled_vendor_preset_ladder"),
                                ("mmlu/mmlu", "mmlu", f"{MODEL}|mmlu|off|greedy", "mmlu_letter_prefill_full_1531")):
        e = u[name]["evals"][ev]
        assert HEX64.match(e["instrument_id"]), (name, ev, e)                  # a sha256 over the canonical fields, never a name
        assert e["decode_policy_ref"] == ref and seed in e["instrument_id_source"], e


def test_frozen_coverage_is_work_identity_from_the_producers_recipes(tmp_path):
    """The freeze carries the goal ids per shard the drivers' own slicing yields, the BoN pool under `bon.goal_subset` with `bon.n_max`,
    the agentic scenario ids, the item set's declared size (and ids sha256 where the seeds record one) — never an items-per-shard count;
    the climb's shards are the array's tasks, its expected set the run-time manifest; the mmlu unit registers NATIVELY under the
    1,531-item instrument like every other unit (F22_REGISTRY_EVALS=mmlu; the sbatch refuses any narrower identity) and binds from its
    own pending file."""
    doc = freeze(tmp_path)
    u = by_unit(doc)
    fixture = [json.loads(l) for l in open(tmp_path / "goals.jsonl")]
    ids = [r["goal_id"] for r in fixture]
    pt = u["static/t0"]["evals"]["pasttense"]["coverage"]
    assert pt["layout"] == "jailbreak_cellstore" and pt["host"] == "cluster"
    assert pt["expected"]["shard00"] == ids[0::9] and pt["expected"]["shard08"] == ids[8::9] and len(pt["expected"]) == 9        # goals[s::nshards]
    assert pt["path"]["root_rel"] == f"exp/sdfbase_full22_native/transcripts_results/sdfbase_full22_{MODEL}_battery_sampled_t0/pasttense/<cell>"
    assert "expected_items_per_shard" not in u["static/t0"]["evals"]["pasttense"] and not hasattr(ST, "ITEMS_PER_SHARD")
    mt = u["multiturn/pair/g34"]["evals"]["pair"]["coverage"]
    assert set(mt["expected"]) == {"shard03", "shard04"} and mt["expected"]["shard03"] == ids[3::9] and mt["private_run"] == f"sdfbase_full22_{MODEL}_battery"
    pad = u["padding/p8192"]["evals"]["grok420"]["coverage"]
    assert list(pad["expected"]) == ["shard00", "shard01", "shard02"] and pad["expected"]["shard01"] == ids[1::3] and f"pad_static_{MODEL}_battery_p8192" in pad["path"]["root_rel"]
    b512 = u["bon/store512"]["evals"]["strongreject_bon"]["coverage"]
    assert b512["layout"] == "bon_store512" and b512["expected"]["shard00"] == ids[:100] and b512["n_max"] == 75             # the JBB subset, n_max draws
    climb = u["bon/climb"]["evals"]["strongreject_bon"]["coverage"]
    assert climb["layout"] == "bon_climb_flat" and climb["expected"] is None and "climb_manifest.jsonl" in climb["expected_source"] and climb["nshards"] == 4
    assert climb["path"]["root_rel"] == f"exp/sdfbase_full22_native/bon_climb/{MODEL}_battery"
    mm = u["mmlu/mmlu"]
    assert mm["registration"] == "native"                                                    # The front door's mmlu job registers itself
    assert mm["evals"]["mmlu"]["coverage"]["n_items"] == 1531 and mm["evals"]["mmlu"]["coverage"]["item_set_id"] == "mmlu_1531"
    assert HEX64.match(mm["evals"]["mmlu"]["coverage"]["items_sha256"]) and mm["evals"]["mmlu"]["coverage"]["host"] == "cluster"
    assert mm["evals"]["mmlu"]["coverage"]["task"] == "s2parity/mmlu_letter" and mm["evals"]["mmlu"]["expected_shards"] == []
    assert mm["evals"]["mmlu"]["coverage"]["path"] == {"from": "store_uri"}                 # the CellStore the run's row records ($WAVE_ROOT/mmlu_full/<TAG>_t0/cells/<key>)
    assert u["controls/t0"]["evals"]["xstest"]["coverage"]["n_items"] == 250 and u["controls/t0"]["evals"]["or_bench_hard"]["coverage"]["n_items"] == 1319
    assert u["gsm8k/none"]["evals"]["gsm8k"]["coverage"] == {**u["gsm8k/none"]["evals"]["gsm8k"]["coverage"], "task": "inspect/gsm8k_paired200", "n_items": 200}
    ag = u["agentic/proxy"]["evals"]["agentic_leaking"]["coverage"]
    assert ag["layout"] == "misalign_responses" and ag["behavior"] == "leaking" and ag["n_samples"] == 10 and len(ag["expected"]["shard00"]) == 51
    assert len(u["agentic/proxy"]["evals"]["agentic_murder"]["coverage"]["expected"]["shard00"]) == 49
    # Petri: the per-cell store on the grading host, the frozen bundle's `<behavior>/<scenario>` ids — the pooled
    # eval expects every behavior's, each `petri_<behavior>` its own; the layout is a REAL reader now, never `unverified`
    pe = u["petri/proxy"]["evals"]
    assert pe["petri_pooled"]["coverage"]["layout"] == "petri_cellstore" and pe["petri_pooled"]["coverage"]["host"] == "grading_host" and pe["petri_pooled"]["coverage"]["behavior"] is None
    ids = petri_scenario_ids()
    assert pe["petri_pooled"]["coverage"]["expected"]["shard00"] == [f"{b}/{sid}" for b in ST.PETRI_BEHAVIORS for sid in ids[b]] and len(pe["petri_pooled"]["coverage"]["expected"]["shard00"]) == 60
    assert pe["petri_self_preservation"]["coverage"]["behavior"] == "self_preservation" and pe["petri_self_preservation"]["coverage"]["expected"]["shard00"] == [f"self_preservation/{sid}" for sid in ids["self_preservation"]]
    assert "Petri scenario bundle" in pe["petri_pooled"]["coverage"]["expected_source"] and pe["petri_pooled"]["coverage"]["path"] == {"from": "store_uri"}
    assert set(ST.PETRI_BEHAVIORS) == set(PETRI_BEHAVIORS) and all(ST.PETRI_BEHAVIOR_OF_EVAL[f"petri_{b}"] == b for b in ST.PETRI_BEHAVIORS)
    from sisteer.capeval.manifest import PETRI_BEHAVIORS as MANIFEST_BEHAVIORS
    assert tuple(ST.PETRI_BEHAVIORS) == tuple(MANIFEST_BEHAVIORS)                          # the two tables cannot drift
    assert "unverified" not in {ev["coverage"]["layout"] for x in doc["units"] for ev in x["evals"].values()}
    # EVERY registering plan row passes its evals to the job explicitly and registers natively — the mmlu unit included (the
    # registry's planner mints the 1,531-item instrument for the front-door models under eval `mmlu`)
    for r in ST.unit_plan(MODEL, "battery", table=str(REPRO / "models.yaml"), sha=SHA):
        if r["registers"]:
            assert r["registration"] == "native" and r["env"]["F22_REGISTRY_EVALS"] == ",".join(r["evals"]), r["logical_work_id"]
    assert ST.REGISTRATION_KINDS == ("native",)


def test_freeze_refuses_when_an_identity_set_cannot_be_minted(tmp_path):
    """A frozen null identity set would silently weaken the gate: no goal pool (no config, no fixture) or no `bon.n_max` refuses."""
    with pytest.raises(RuntimeError, match="goal identities cannot be frozen"):
        ST.expected_units(MODEL, SHA, str(REPRO / "models.yaml"), families=["static"], wave_config=str(tmp_path / "absent.yaml"))
    cfg = tmp_path / "no_nmax.yaml"; cfg.write_text("private_run: x\n")
    with pytest.raises(RuntimeError, match="bon.n_max"):
        ST.expected_units(MODEL, SHA, str(REPRO / "models.yaml"), families=["static"], goals_jsonl=str(goal_fixture(tmp_path)), wave_config=str(cfg))


def test_unit_plan_places_multiturn_disjointly_and_matches_the_freeze():
    rows = ST.unit_plan(MODEL, "battery", table=str(REPRO / "models.yaml"), clusters=["cluster_a", "cluster_b"], sha=SHA)
    targets = [r for r in rows if r["family"] == "multiturn" and r["kind"] == "target"]
    assert len(targets) == 12 and len({r["logical_work_id"] for r in targets}) == 12          # every logical unit ONCE
    for lab in ("cluster_a", "cluster_b"):
        assert sum(1 for r in rows if r["kind"] == "attacker" and r["cluster"] == lab) == ST.attacker_pool_size(6) == 2    # commit 31: one member per 3 targets
        mine = [r for r in targets if r["cluster"] == lab]
        assert len(mine) == 6 and all(r["group"] == f"mt-battery-{lab}" for r in mine)
    assert rows == ST.unit_plan(MODEL, "battery", table=str(REPRO / "models.yaml"), clusters=["cluster_a", "cluster_b"], sha=SHA)   # deterministic
    frozen = {r["logical_work_id"] for r in ST.unit_plan(MODEL, "battery", table=str(REPRO / "models.yaml"), sha=SHA) if r["registers"]}
    assert frozen == {r["logical_work_id"] for r in rows if r["registers"]}                   # the freeze and the submissions agree
    with pytest.raises(ValueError):
        ST.shard_groups(9, "0,1,2 3,4 5,6 7")                                                 # not a cover
    with pytest.raises(ValueError):
        ST.shard_groups(9, "0,1,2 2,3,4 5,6 7,8")                                             # overlap


def test_all_satisfied_admits_the_petri_unit_included_without_any_override(tmp_path):
    state = tmp_path / "state"; doc = freeze(state)
    runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(doc["units"])                                   # no --force-gate anywhere
    assert not (state / "gate" / "forced.jsonl").exists()


def test_a_petri_scenario_short_or_unresolved_refuses_the_unit_by_name(tmp_path):
    """The real reader over a synthetic store: a scenario the runner's failure ledger holds unresolved (or one short of a required dimension)
    is named per eval — the pooled eval and the behavior's own — and admission refuses the Petri unit by name; a full store admits."""
    from front_door_testlib import lay_down_petri_store
    sys.path.insert(0, str(LIB)); import coverage_probe as CP
    state = tmp_path / "state"; doc = freeze(state, families=["petri"])
    u = by_unit(doc)["petri/proxy"]; ids = petri_scenario_ids()
    missing = f"self_preservation/{ids['self_preservation'][3]}"
    cell = lay_down_petri_store(tmp_path / "petri_full22_m_s1" / "cells" / "petricell-b2", ids, drop=(missing,))
    probes = {e: CP.probe_petri_cellstore(str(cell), ev["coverage"]["behavior"]) for e, ev in u["evals"].items()}
    runs, pols = [], []
    satisfy(state, u, runs, pols, probe=probes["petri_pooled"])
    # per-eval probes (satisfy() takes one document for every run: rewrite the coverage record with each eval's own)
    cov = json.load(open(ST.coverage_path(str(state), u["logical_work_id"])))
    for rid, c in cov["runs"].items():
        c["probe"] = probes[c["eval_id"]]
    ST.write_atomic(ST.coverage_path(str(state), u["logical_work_id"]), cov)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert ok == [] and len(problems) == 2, problems                                          # the pooled eval AND petri_self_preservation; the other five are complete
    assert all("petri/proxy" in p and "1 of " in p and missing in p and "not complete" in p for p in problems), problems
    assert any("of behavior 'self_preservation'" in p and "1 of 10" in p for p in problems) and any("1 of 60" in p for p in problems), problems
    # the store completed: admitted
    lay_down_petri_store(cell, ids)
    for rid, c in cov["runs"].items():
        c["probe"] = CP.probe_petri_cellstore(str(cell), u["evals"][c["eval_id"]]["coverage"]["behavior"])
    ST.write_atomic(ST.coverage_path(str(state), u["logical_work_id"]), cov)
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and ok == [u["logical_work_id"]]
    # the freeze REFUSES without the bundle (a frozen null identity set would silently weaken the gate)
    with pytest.raises(RuntimeError, match="Petri scenario bundle dir .*absent \\(--petri-bundle\\) is not a directory"):
        ST.expected_units(MODEL, SHA, str(REPRO / "models.yaml"), families=["petri"], petri_bundle=str(tmp_path / "absent"))
    # a bundle of ANOTHER freeze (its manifest's content hash is not the lock's) refuses by name too
    other = tmp_path / "other_bundle"; other.mkdir(); (other / "bundle_manifest.json").write_text(json.dumps({"content_hash": "0" * 16}))
    with pytest.raises(RuntimeError, match="is bundle '0000000000000000', the lock names"):
        ST.expected_units(MODEL, SHA, str(REPRO / "models.yaml"), families=["petri"], petri_bundle=str(other))


def test_missing_padding_level_and_unbound_unit_refuse_by_name(tmp_path):
    state = tmp_path / "state"; doc = freeze(state); runs, pols = [], []
    for u in doc["units"]:
        if u["unit"] in ("p32768", "mmlu"):
            continue
        satisfy(state, u, runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, _ = A.check(str(state), str(db), {})
    assert any("padding/p32768: never submitted (unbound" in p for p in problems), problems
    assert any("mmlu/mmlu: never submitted" in p for p in problems), problems


def test_submitted_but_unregistered_and_stale_code_sha_refuse(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["static", "bon"]); runs, pols = [], []
    u = by_unit(doc)
    satisfy(state, u["static/t0"], runs, pols, state_name="submitted")               # submitted, never registered
    satisfy(state, u["static/t1"], runs, pols, code_sha="b" * 40)                   # another code sha
    satisfy(state, u["bon/store512"], runs, pols)
    satisfy(state, u["bon/climb"], runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert any("static/t0: submitted (job 1 on cluster_a) but not registered" in p for p in problems), problems
    assert any("static/t1" in p and "code_sha" in p and "the unit's recorded sha" in p for p in problems), problems     # Per-unit sha
    assert f"{MODEL}/battery/bon/climb" in ok and f"{MODEL}/battery/bon/store512" in ok


def test_half_the_goals_missing_refuses_by_identity_however_many_lines_the_store_holds(tmp_path):
    """A shard with 14 of 29 goals completed (its progress + responses lines summing past 28) is NOT covered — the gate names the
    missing goal ids; errored goals are excluded from the completed set."""
    state = tmp_path / "state"; doc = freeze(state, families=["static"]); runs, pols = [], []
    u = by_unit(doc)
    exp = u["static/t0"]["evals"]["pasttense"]["coverage"]["expected"]["shard04"]
    assert len(exp) == 29
    satisfy(state, u["static/t0"], runs, pols, drop_goals=tuple(exp[14:]))       # 15 goals missing / errored in EVERY shard of every eval
    satisfy(state, u["static/t1"], runs, pols)
    satisfy(state, u["bon/store512"], runs, pols, drop_goals=("goal-000", "goal-001"))
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    hit = [p for p in problems if "static/t0" in p and "(pasttense) shard shard04" in p]
    assert hit and "15 of 29 expected goal(s) not completed" in hit[0] and exp[14] in hit[0] and "errored: 15" in hit[0], problems
    assert any("bon/store512" in p and "2 of 100 expected goal(s) lack 75 draws" in p and "goal-000" in p for p in problems), problems
    assert f"{MODEL}/battery/static/t1" in ok and f"{MODEL}/battery/static/t0" not in ok


def test_climb_coverage_is_the_manifests_expected_set_per_array_task(tmp_path):
    """The climb unit binds EVERY array task's runs and each shard's expected goals are the run-time manifest's; a shard with a manifest
    goal not fully climbed, or without a manifest, refuses by name."""
    state = tmp_path / "state"; doc = freeze(state, families=["bon"]); runs, pols = [], []       # the `bon` stage family = the climb array
    u = by_unit(doc)
    assert list(u) == ["bon/climb"]
    satisfy(state, u["bon/climb"], runs, pols, drop_goals=("gB",))
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    b = ST.binding_state(str(state / "submissions.jsonl"), f"{MODEL}/battery/bon/climb")
    assert b["array"] == "0-3" and len(b["run_ids"]) == 4 and sorted(r["array_task"] for r in b["runs"]) == [0, 1, 2, 3]
    climb = [p for p in problems if "bon/climb" in p]
    assert len(climb) == 4 and all("1 of 3 manifest goal(s) not fully climbed" in p and "'gB'" in p for p in climb), problems
    assert {re.search(r"shard (shard0\d)", p).group(1) for p in climb} == {"shard00", "shard01", "shard02", "shard03"}
    # no manifest yet: the expected set is unknown — refused, never assumed empty
    ev = u["bon/climb"]["evals"]["strongreject_bon"]
    nomanifest = {"layout": "bon_climb_flat", "path": "/x", "exists": True, "notes": [],
                  "shards": {"shard00": {"present": False, "completed": [], "errored": [], "refused": [], "expected": None, "notes": ["no climb manifest at /x/plan/climb_manifest.jsonl"]}}}
    assert any("climb manifest is missing" in p for p in A.check_run_coverage(ev, {"probe": nomanifest}, None, "strongreject_bon"))


def test_capability_store_must_hold_the_item_set_and_attest_its_ids(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu", "controls", "gsm8k"]); runs, pols = [], []
    u = by_unit(doc)
    satisfy(state, u["mmlu/mmlu"], runs, pols, drop_goals=("one",))                                    # 1530 of 1531 items
    satisfy(state, u["controls/t0"], runs, pols)
    satisfy(state, u["controls/t1"], runs, pols, items_sha="")                                          # the controls' item sets record no sha: fine
    for g in ("none", "prompted__safety_refusal", "prompted__si_constitution"):
        satisfy(state, u[f"gsm8k/{g}"], runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert any("mmlu/mmlu" in p and "1530 unique item(s) of task 's2parity/mmlu_letter' < the item set's 1531 (mmlu_1531)" in p for p in problems), problems
    assert f"{MODEL}/battery/controls/t0" in ok and f"{MODEL}/battery/controls/t1" in ok and f"{MODEL}/battery/gsm8k/none" in ok
    # the right count but ANOTHER item list (the 200-item head, say): the ids sha256 the seeds record for mmlu_1531 must match
    state2 = tmp_path / "state2"; doc2 = freeze(state2, families=["mmlu"]); runs2, pols2 = [], []
    satisfy(state2, by_unit(doc2)["mmlu/mmlu"], runs2, pols2, items_sha=hashlib.sha256(b"other ids").hexdigest())
    problems2, _ = A.check(str(state2), str(make_registry(tmp_path / "reg2", runs2, pols2)), {})
    assert any("item ids sha256" in p and "!= the item set mmlu_1531's" in p for p in problems2), problems2
    # a store that attests no id list at all while the seeds record one
    state3 = tmp_path / "state3"; doc3 = freeze(state3, families=["mmlu"]); runs3, pols3 = [], []
    satisfy(state3, by_unit(doc3)["mmlu/mmlu"], runs3, pols3, items_sha="")
    problems3, _ = A.check(str(state3), str(make_registry(tmp_path / "reg3", runs3, pols3)), {})
    assert any("attests no item-id list" in p for p in problems3), problems3


def test_every_expected_eval_inside_a_unit_must_be_bound(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["static"]); runs, pols = [], []
    u = by_unit(doc)
    satisfy(state, u["static/t0"], runs, pols, drop_eval="codeattack")               # three of the four attacks registered
    satisfy(state, u["static/t1"], runs, pols)
    satisfy(state, u["bon/store512"], runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert any("static/t0: expected eval 'codeattack' has no bound run" in p for p in problems), problems
    assert f"{MODEL}/battery/static/t1" in ok


def test_coverage_collected_for_other_runs_and_a_foreign_decode_policy_refuse(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu", "controls"]); runs, pols = [], []
    u = by_unit(doc)
    satisfy(state, u["mmlu/mmlu"], runs, pols, cov_run_ids=["run-from-another-wave"])
    satisfy(state, u["controls/t0"], runs, pols, policy_ref=f"{MODEL}|over_refusal|off|greedy")
    satisfy(state, u["controls/t1"], runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, _ = A.check(str(state), str(db), {})
    assert any("mmlu/mmlu: the coverage record is for runs ['run-from-another-wave']" in p for p in problems), problems
    assert any("controls/t0" in p and f"decode policy '{MODEL}|over_refusal|off|greedy' != frozen '{MODEL}|over_refusal|off|vendor_preset'" in p for p in problems), problems


def test_null_or_non_exact_decode_identity_is_refused_where_one_is_frozen(tmp_path):
    """The schema permits decode_policy_id null and decode_match stamps it only on an `exact` match — a complete run with an unrecorded
    or ambiguous decode is NOT this baseline."""
    state = tmp_path / "state"; doc = freeze(state, families=["controls", "mmlu"]); runs, pols = [], []
    u = by_unit(doc)
    satisfy(state, u["controls/t0"], runs, pols, decode_policy_id=None)                          # null id, match `unrecorded`
    satisfy(state, u["controls/t1"], runs, pols, decode_match="ambiguous")                       # an id, but not an exact derivation
    satisfy(state, u["mmlu/mmlu"], runs, pols)                                                   # exact: fine
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert any("controls/t0" in p and "(xstest) records NO decode_policy_id (decode_policy_match='unrecorded')" in p and "off|vendor_preset" in p for p in problems), problems
    assert any("controls/t1" in p and "decode_policy_match='ambiguous', not 'exact'" in p for p in problems), problems
    assert f"{MODEL}/battery/mmlu/mmlu" in ok and not any("mmlu/mmlu" in p for p in problems)


def test_incomplete_run_and_missing_verdict_refuse(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu", "gsm8k"]); runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    runs[0]["lifecycle_state"] = "generated"
    os.remove(state / "gate" / "gsm8k.json")
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, _ = A.check(str(state), str(db), {})
    assert any("is generated, not complete" in p for p in problems), problems
    assert any("no sanity verdict for family gsm8k" in p for p in problems), problems


# ------------------------------------------------------------------------- a pass verdict is bound to the RUNS it ruled

D3008_REMEDY = "re-gate the family: 06_eval_waves.sh --gate mmlu --verdict (or --headline <value> --range <lo,hi> by hand)"


def _resubmit_mmlu(tmp_path):
    """Two families ADMITTED, then the mmlu unit's job re-runs (a resubmission): the re-run registers OTHER run ids, 06 --gate's bind-only pass
    binds and covers them (the registry rows added beside the dead attempt's) and writes NO verdict — gate/mmlu.json stands as ruled over the
    OLD ids. Returns (state, db, unit, old ids, new ids, runs, pols, the standing verdict's bytes)."""
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu", "gsm8k"]); runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(doc["units"]), problems
    u = by_unit(doc)["mmlu/mmlu"]; lw = u["logical_work_id"]; ledger = str(state / "submissions.jsonl")
    old = list(ST.binding_state(ledger, lw)["run_ids"]); v_bytes = (state / "gate" / "mmlu.json").read_bytes()
    assert old and json.loads(v_bytes)["run_ids"] == old and json.loads(v_bytes)["verdict"] == "pass"
    satisfy(state, u, runs, pols, rid_suffix="-rerun", rule=False)                      # the re-run's registration: bound + covered, NO verdict
    db = make_registry(tmp_path / "reg", runs, pols)
    new = list(ST.binding_state(ledger, lw)["run_ids"])
    assert new and set(new).isdisjoint(old) and (state / "gate" / "mmlu.json").read_bytes() == v_bytes
    cov = json.load(open(ST.coverage_path(str(state), lw)))
    assert set(cov["run_ids"]) == set(new) and all(isinstance(c["probe"], dict) for c in cov["runs"].values())   # the new runs are covered: nothing else refuses
    return state, db, u, old, new, runs, pols, v_bytes


def test_d3008_a_bind_only_resubmission_that_changes_the_run_ids_refuses_the_family_naming_them(tmp_path):
    """A `pass` is bound to the RUNS it ruled, not to a coverage generation. After a resubmission re-binds the unit to other run ids and the
    bind-only re-gate writes no verdict, the standing pass (recorded over the old ids) admits nothing: the family is refused by name with the
    ids ADDED (bound now, never ruled) and REMOVED (ruled, no longer bound) and the remedy; the other family, whose bindings did not move,
    admits; the `verdict == pass` check stands beside it. The remedy — the re-gate's verdict over the runs bound now — admits again. Before
    the fix admission read `verdict == pass` alone and ADMITTED the re-bound runs on a ruling nobody made over them."""
    state, db, u, old, new, *_ = _resubmit_mmlu(tmp_path)
    lw = u["logical_work_id"]
    problems, ok = A.check(str(state), str(db), {})
    assert lw not in ok and len(problems) == 1, problems
    p = problems[0]
    assert p.startswith(f"{lw}: family mmlu verdict 'pass' (ruled ") and "ruled over runs that differ from the family's current registered bindings" in p, p
    assert f"{len(new)} bound now that it never ruled {sorted(new)}; {len(old)} it ruled that are no longer bound {sorted(old)}" in p and "D3008" in p and D3008_REMEDY in p, p
    assert any(o.startswith(f"{MODEL}/battery/gsm8k/") for o in ok) and not any("gsm8k" in x for x in problems)
    r = subprocess.run([sys.executable, str(LIB / "admission.py"), "check", "--state-root", str(state), "--registry", str(db)], capture_output=True, text=True)
    assert r.returncode == 3 and f"REFUSED: {lw}: family mmlu verdict 'pass' (ruled " in r.stderr and "ADMITTED" not in r.stdout, r.stdout + r.stderr
    # the `pass` check stays, independent: a `fail` over the old ids says both
    rule_family(state, "mmlu", verdict="fail"); v = json.load(open(state / "gate" / "mmlu.json")); v["run_ids"] = old; ST.write_atomic(str(state / "gate" / "mmlu.json"), v)
    problems, ok = A.check(str(state), str(db), {})
    assert [x for x in problems if "verdict is 'fail', not pass" in x] and [x for x in problems if "ruled over runs that differ" in x] and len(problems) == 2, problems
    # the remedy: the verdict recorded over the runs bound NOW
    rule_family(state, "mmlu", verdict="pass")
    assert json.load(open(state / "gate" / "mmlu.json"))["run_ids"] == new
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and lw in ok, problems


def test_d3008_a_coverage_refresh_over_the_same_runs_under_another_probe_sha_still_admits(tmp_path):
    """The other half of the rule (why the verdict is not tied to the coverage generation): a coverage record re-collected over the
    SAME run ids under another probe commit, file digest and collected time, with a `coverage_generation` the verdict never saw, is a RE-READ
    of the runs the verdict ruled — still admitted; so is a verdict whose own probe sha differs from the record's, and either list in another
    order (a set comparison)."""
    state = tmp_path / "state"; doc = freeze(state, families=["controls"]); runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    vp = state / "gate" / "controls.json"
    v0 = rule_family(state, "controls", probe_code_sha="a" * 40, probe_sha256="1" * 64, inventory_code_sha=SHA)
    assert len(v0["run_ids"]) >= 4                                                       # two units x two evals: an order change below is a real one
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(doc["units"]), problems
    for u in doc["units"]:
        cov_p = Path(ST.coverage_path(str(state), u["logical_work_id"])); cov = json.load(open(cov_p))
        cov.update({"probe_code_sha": "b" * 40, "probe_sha256": "2" * 64, "collected_utc": "2026-12-31T23:59:59Z", "coverage_generation": 7,
                    "run_ids": list(reversed(cov["run_ids"]))})
        ST.write_atomic(str(cov_p), cov)
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(doc["units"]), problems
    v = json.load(open(vp)); v["run_ids"] = list(reversed(v["run_ids"])); ST.write_atomic(str(vp), v)
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(doc["units"]), problems


def test_d3008_a_verdict_without_run_ids_refuses_by_name(tmp_path):
    """A verdict with no or empty `run_ids` names no run it ruled over and vouches for none: refused by name — re-rule this family."""
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu"]); runs, pols = [], []
    u = doc["units"][0]; lw = u["logical_work_id"]; satisfy(state, u, runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    vp = state / "gate" / "mmlu.json"; good = json.load(open(vp))
    assert good["run_ids"]
    problems, ok = A.check(str(state), str(db), {}); assert problems == [] and lw in ok, problems
    for bad in ([], None, "missing", [""], "a,b"):
        v = dict(good)
        if bad == "missing":
            v.pop("run_ids")
        else:
            v["run_ids"] = bad
        ST.write_atomic(str(vp), v)
        problems, ok = A.check(str(state), str(db), {})
        assert lw not in ok and len(problems) == 1, (bad, problems)
        assert f"{lw}: family mmlu verdict 'pass' (ruled {good['ruled_utc']}) records no run_ids (" in problems[0] and "re-rule this family" in problems[0] and D3008_REMEDY in problems[0], (bad, problems[0])
    ST.write_atomic(str(vp), good)
    problems, ok = A.check(str(state), str(db), {}); assert problems == [] and lw in ok, problems


def test_d3008_a_failed_verdict_write_after_a_resubmission_leaves_the_old_pass_standing_and_refused(tmp_path):
    """The re-gate after a resubmission reaches the verdict step and the WRITE fails (gate/ unwritable) — `stage.py verdict` exits
    non-zero (neither a recorded verdict's 0 nor a refusal's 2), nothing is written, the old pass stands over the old ids, and admission refuses
    the family by the ids until a verdict is recorded over the runs bound now (06 --gate propagates that status as exit 1 by name)."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory modes")
    state, db, u, old, new, runs, pols, v_bytes = _resubmit_mmlu(tmp_path)
    lw = u["logical_work_id"]; contract = contract_arg(state)                     # the contract block is composed BEFORE gate/ is locked (it writes the sidecar there)
    verdict = [sys.executable, str(LIB / "stage.py"), "verdict", "--state-root", str(state), "--family", "mmlu", "--headline", "0.5", "--range", "0,1",
               "--run-ids", ",".join(new), "--inventory-sha", SHA, "--probe-sha", "p" * 40, "--probe-sha256", "f" * 64, "--contract", contract]
    gate = state / "gate"; mode = gate.stat().st_mode & 0o777; gate.chmod(0o500)
    try:
        r = subprocess.run(verdict, capture_output=True, text=True)
    finally:
        gate.chmod(mode)
    assert r.returncode not in (0, 2) and "gate verdict mmlu" not in r.stdout and (gate / "mmlu.json").read_bytes() == v_bytes, r.stdout + r.stderr
    problems, ok = A.check(str(state), str(db), {})
    assert lw not in ok and len(problems) == 1 and f"{len(new)} bound now that it never ruled {sorted(new)}; {len(old)} it ruled that are no longer bound {sorted(old)}" in problems[0], problems
    r = subprocess.run(verdict, capture_output=True, text=True)                   # the write lands: the verdict over the runs bound now
    assert r.returncode == 0 and "gate verdict mmlu: pass" in r.stdout, r.stdout + r.stderr
    assert json.load(open(gate / "mmlu.json"))["run_ids"] == new
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and lw in ok, problems


# ------------------------------------------------------------------------- a verdict read from the registry vouches for the grades it read

GRADER = "judge-" + "d" * 58
CTL_HEADLINE = {"xstest": "xst_refused", "or_bench_hard": "orb_refused"}


def _graded_controls(tmp_path):
    """The controls family bound and covered on UNSTEERED runs (object `none`; the battery's inventory pins no object) with no verdict yet, and the
    registry carrying each run's one current complete judgment by its judge and the eval's headline row under it, as stage 08 files them.
    Returns (state, db, the inventory's units, the run ids, {run id: value}, {run id: its headline metric})."""
    state = tmp_path / "state"; doc = freeze(state, families=["controls"]); runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols, rule=False)
    for r in runs:
        r["object_id"] = "none"
    db = make_registry(tmp_path / "reg", runs, pols)
    vals = {r["run_id"]: (0.04, 0.62, 0.03, 0.66)[i % 4] for i, r in enumerate(runs)}
    add_grading_tables(db, evals=[{"eval_id": e, "headline_metric_id": m} for e, m in CTL_HEADLINE.items()],
                       judgments=[{"judgment_id": f"j-{rid}", "run_id": rid, "judge_id": GRADER} for rid in vals],
                       metrics=[{"run_id": r["run_id"], "judgment_id": f"j-{r['run_id']}", "metric_id": CTL_HEADLINE[r["eval_id"]], "value": vals[r["run_id"]]} for r in runs],
                       run_judges={rid: GRADER for rid in vals})
    return state, db, doc["units"], [r["run_id"] for r in runs], vals, {r["run_id"]: CTL_HEADLINE[r["eval_id"]] for r in runs}


def _rule_from_registry(tmp_path, state, db, family="controls"):
    """`06 --gate <family> --verdict`'s verdict step as 06 runs it: lib/gate_headline.py read over the family's bindings (exit 3 = skipped, nothing
    written), then `stage.py verdict --headline-record` over the same runs, with the provenance 06 hands it."""
    rec = tmp_path / f"headline.{family}.json"
    r = subprocess.run([sys.executable, str(LIB / "gate_headline.py"), "read", "--registry", str(db), "--state-root", str(state), "--family", family, "--out", str(rec)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return r
    return subprocess.run([sys.executable, str(LIB / "stage.py"), "verdict", "--state-root", str(state), "--family", family, "--headline-record", f"@{rec}",
                           "--run-ids", ",".join(family_bindings(state, family)), "--inventory-sha", SHA, "--probe-sha", "p" * 40, "--probe-sha256", "f" * 64,
                           "--contract", contract_arg(state)], capture_output=True, text=True)


def _set(db, sql, *args):
    conn = sqlite3.connect(db); conn.execute(sql, args); conn.commit(); conn.close()


def test_a_verdict_read_from_the_registry_admits_only_while_every_leg_it_read_still_reads_the_same(tmp_path):
    """A pass ruled on the registry's grades (`06 --gate --verdict`, which records every leg it read) vouches for those grades, not only for its
    runs. A re-grade that begins after the ruling (the leg's judgment abandoned, its replacement not complete) refuses the family by name, the
    leg and its reason given, and so does the re-gate's skip, which leaves the pass standing. Before, admission checked the verdict and its
    run ids alone and admitted the dose wave with no current graded baseline. A replacement grade, or a value refiled under the same judgment,
    is a grade the verdict never ruled on: refused until the family is ruled again. A verdict recorded by hand carries no legs and stays bound to
    its runs alone."""
    state, db, units, rids, vals, metric = _graded_controls(tmp_path)
    vp = state / "gate" / "controls.json"
    r = _rule_from_registry(tmp_path, state, db)
    assert r.returncode == 0 and "gate verdict controls: pass" in r.stdout, r.stdout + r.stderr
    assert {g["run_id"] for g in json.load(open(vp))["headline_legs"]} == set(rids)
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(units), problems
    victim = rids[1]
    lws = [u["logical_work_id"] for u in units]

    def refused(*words):
        problems, ok = A.check(str(state), str(db), {})
        assert not ok and len(problems) == len(units), problems
        for lw, p in zip(lws, problems):
            assert p.startswith(f"{lw}: family controls verdict 'pass' (ruled ") and "re-run 06_eval_waves.sh --gate controls --verdict" in p, p
            assert all(w in p for w in words), (words, p)
        return problems
    # a re-grade begun: the leg's judgment abandoned, the replacement not complete
    _set(db, "UPDATE judgment SET status = 'abandoned' WHERE judgment_id = ?", f"j-{victim}")
    refused("was ruled on grades the registry no longer reads whole", f"run {victim} (", "no current complete judgment by its judge")
    v_bytes = vp.read_bytes()
    r = _rule_from_registry(tmp_path, state, db)                                 # the re-gate skips the family and writes nothing
    assert r.returncode == 3 and vp.read_bytes() == v_bytes, r.stdout + r.stderr
    refused(f"run {victim} (")
    # the replacement grade lands: another judgment, another value — the pass ruled on the old one admits nothing until the family is ruled again
    add_grading_tables(db, judgments=[{"judgment_id": f"j2-{victim}", "run_id": victim, "judge_id": GRADER}],
                       metrics=[{"run_id": victim, "judgment_id": f"j2-{victim}", "metric_id": metric[victim], "value": 0.5}])
    refused("the registry's grades changed since the ruling", f"run {victim}: ruled judgment j-{victim} value {vals[victim]:g}, reads judgment j2-{victim} value 0.5 now")
    r = _rule_from_registry(tmp_path, state, db)
    assert r.returncode == 0 and "gate verdict controls: pass" in r.stdout, r.stdout + r.stderr
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(units), problems
    # a value refiled under the same judgment is another grade too
    _set(db, "UPDATE run_metric SET value = 0.51 WHERE judgment_id = ?", f"j2-{victim}")
    refused(f"run {victim}: ruled judgment j2-{victim} value 0.5, reads judgment j2-{victim} value 0.51 now")
    # a verdict recorded by hand (`--headline/--range`) carries no legs: the operator's ruling, bound to its runs alone, whatever the grades
    _set(db, "UPDATE judgment SET status = 'abandoned' WHERE judgment_id = ?", f"j2-{victim}")
    h = subprocess.run([sys.executable, str(LIB / "stage.py"), "verdict", "--state-root", str(state), "--family", "controls", "--headline", "0.6", "--range", "0,0.95",
                        "--run-ids", ",".join(family_bindings(state, "controls")), "--inventory-sha", SHA, "--probe-sha", "p" * 40, "--probe-sha256", "f" * 64,
                        "--contract", contract_arg(state)], capture_output=True, text=True)
    assert h.returncode == 0 and "headline_legs" not in json.load(open(vp)), h.stdout + h.stderr
    problems, ok = A.check(str(state), str(db), {})
    assert problems == [] and len(ok) == len(units), problems


def test_torn_binding_reads_as_submitted(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu"]); runs, pols = [], []
    u = doc["units"][0]; satisfy(state, u, runs, pols)
    (state / "bindings" / f"{ST.safe_id(u['logical_work_id'])}.registered.json").write_text('{"logical_work_id": "x", "state": "registe')   # torn: no rename happened
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, _ = A.check(str(state), str(db), {})
    assert any("but not registered" in p for p in problems), problems


def test_force_gate_is_recorded_and_registry_none_fails_closed(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu"]); runs, pols = [], []
    u = doc["units"][0]; satisfy(state, u, runs, pols, state_name="submitted")
    lw = u["logical_work_id"]
    problems, _ = A.check(str(state), "none", {})
    assert problems
    r = subprocess.run([sys.executable, str(LIB / "admission.py"), "check", "--state-root", str(state), "--registry", "none", "--force-gate", f"{lw}=smoke only, recorded"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "FORCED" in r.stdout, r.stdout + r.stderr
    forced = [json.loads(l) for l in open(state / "gate" / "forced.jsonl")]
    assert forced[0]["logical_work_id"] == lw and forced[0]["reason"] == "smoke only, recorded"


def test_force_gate_refuses_an_empty_reason_and_an_unknown_unit_by_name_and_writes_under_the_state_root_only(tmp_path):
    """demo-import M7 (Q10's three tightenings, which apply to 06 and 09 alike). (1) `--force-gate X` (no `=`) and `--force-gate X=` (an EMPTY
    reason) refuse with exit 2 — a forced admission is recorded with why, so a reason-less override is never written; (2) an LWID that names
    no unit of the FROZEN inventory refuses with exit 3 NAMING it (before this a typo was ignored silently and the operator read an unrelated
    'never submitted'); (3) a forced check still writes `gate/forced.jsonl` under the STATE root only — nothing under the registry root —
    and the no-force path is untouched (test_all_satisfied_admits_the_petri_unit_included_without_any_override pins it)."""
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu"]); runs, pols = [], []
    u = doc["units"][0]; satisfy(state, u, runs, pols, state_name="submitted")
    lw = u["logical_work_id"]
    reg_root = tmp_path / "reg"; db = make_registry(reg_root, runs, pols)
    before_reg = {str(p) for p in reg_root.rglob("*")}
    cli = [sys.executable, str(LIB / "admission.py"), "check", "--state-root", str(state), "--registry", str(db)]
    for bad in ("X", f"{lw}=", f"{lw}=   ", "=smoke"):
        r = subprocess.run(cli + ["--force-gate", bad], capture_output=True, text=True)
        assert r.returncode == 2, (bad, r.stdout, r.stderr)
        assert "REFUSED: --force-gate wants LWID=REASON" in r.stderr and repr(bad) in r.stderr, (bad, r.stderr)
        assert "ADMITTED" not in r.stdout and "FORCED" not in r.stdout
    assert not (state / "gate" / "forced.jsonl").exists()                                    # a refused override is never recorded
    # an unknown LWID: exit 3, named, the inventory's units listed; the typo is not silently ignored into a 'never submitted'
    typo = lw + "_typo"
    r = subprocess.run(cli + ["--force-gate", f"{typo}=smoke"], capture_output=True, text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    assert f"REFUSED: --force-gate names no unit of the frozen inventory: {typo!r}" in r.stderr and lw in r.stderr, r.stderr
    assert "ADMISSION REFUSED" in r.stderr and not (state / "gate" / "forced.jsonl").exists()
    problems, ok = A.check(str(state), str(db), {typo: "smoke"})
    assert len(problems) == 1 and typo in problems[0] and ok == []
    # the frozen unit forced with a reason: admitted, recorded under the state root ONLY — the registry root is untouched
    r = subprocess.run(cli + ["--force-gate", f"{lw}=smoke only, recorded"], capture_output=True, text=True)
    assert r.returncode == 0 and f"ok   {lw}: FORCED (smoke only, recorded)" in r.stdout and "ADMITTED" in r.stdout, r.stdout + r.stderr
    forced = [json.loads(l) for l in open(state / "gate" / "forced.jsonl")]
    assert forced == [{"logical_work_id": lw, "reason": "smoke only, recorded", "utc": forced[0]["utc"]}]
    assert {str(p) for p in reg_root.rglob("*")} == before_reg
    assert not list(reg_root.rglob("forced*")) and (state / "gate" / "forced.jsonl").is_file()
    # --dry-run: admitted, printed, NOT recorded (record_forced stays as-is: append-only, on ADMITTED, never under --dry-run)
    n = len(forced)
    r = subprocess.run(cli + ["--force-gate", f"{lw}=again", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "FORCED (again)" in r.stdout and len(open(state / "gate" / "forced.jsonl").read().splitlines()) == n


def test_cli_refuses_with_exit_3_and_names_the_unit(tmp_path):
    state = tmp_path / "state"; freeze(state, families=["mmlu"])
    r = subprocess.run([sys.executable, str(LIB / "admission.py"), "check", "--state-root", str(state), "--registry", "none"], capture_output=True, text=True)
    assert r.returncode == 3 and f"REFUSED: {MODEL}/battery/mmlu/mmlu: never submitted" in r.stderr


def test_schema1_inventory_is_refused_by_name(tmp_path):
    state = tmp_path / "state"; (state / "gate").mkdir(parents=True)
    ST.write_atomic(str(state / "gate" / "expected_units.json"), {"schema": 1, "units": [{"logical_work_id": "m/s1/mmlu/mmlu", "family": "mmlu"}]})
    problems, _ = A.check(str(state), "none", {})
    assert len(problems) == 1 and "schema 1" in problems[0] and "re-freeze" in problems[0]


def test_probe_plan_and_array_tasks_cli(tmp_path):
    """stage.py probe-plan forms the probe argv per bound run from the run's RECORDED store (the registry row's store_uri, the climb's
    extra.climb_root — never the frozen root_rel with the run's cell), the host per layout, the seventh column naming the record; a run with no
    store on record is a `none` line saying so; array-tasks parses sacct's JobIDRaw rows into raw ids + task indices (a pending summary row is
    skipped) — from a literal or from a file (`@<path>`, the gate's hand-off: MAX_ARG_STRLEN)."""
    state = tmp_path / "state"; doc = freeze(state, families=["bon", "mmlu", "agentic"])
    u = by_unit(doc)
    runs = json.dumps([{"run_id": "r-climb", "eval_id": "strongreject_bon", "object": "none", "array_task": 2, "store_uri": "/stores/512/none", "extra": {"climb_root": "/recorded/bon_climb/x"}},
                       {"run_id": "r-climb-norec", "eval_id": "strongreject_bon", "object": "none", "array_task": 1, "store_uri": "/stores/512/none"},
                       {"run_id": "r-mmlu", "eval_id": "mmlu", "object": "none", "array_task": None, "store_uri": "/mirror/cells/none__mmlu"},
                       {"run_id": "r-x", "eval_id": "not_an_eval", "object": None, "array_task": None, "store_uri": None}])
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", u["bon/climb"]["logical_work_id"],
                        "--runs-json", runs, "--shared", "/shared", "--probe", "/eval/sha/reproduce/lib/coverage_probe.py"], capture_output=True, text=True)
    lines = {ln.split("\t")[0]: ln.split("\t") for ln in r.stdout.splitlines()}
    assert lines["r-climb"][1] == "cluster" and lines["r-climb"][2] == "/eval/sha/reproduce/lib/coverage_probe.py probe --layout bon_climb_flat --path /recorded/bon_climb/x --cell none --shard-index 2 --nshards 4"
    assert lines["r-climb"][6] == "registry.run.extra_json.climb_root"
    assert lines["r-climb-norec"][1] == "none" and "not covered, no store on record for the climb's flat root" in lines["r-climb-norec"][3] and "/stores/512/none" in lines["r-climb-norec"][3] \
        and "D3003 (1)" in lines["r-climb-norec"][3] and lines["r-climb-norec"][6] == "none"
    assert "/shared/exp/sdfbase_full22_native" not in r.stdout                                                      # the frozen root_rel is never recomputed
    assert lines["r-mmlu"][1] == "none" and "not one of the unit's" in lines["r-mmlu"][3]
    runs_f = tmp_path / "runs.json"; runs_f.write_text(runs)
    r2 = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", u["mmlu/mmlu"]["logical_work_id"],
                         "--runs-json", f"@{runs_f}", "--shared", "/shared", "--probe", "/p.py"], capture_output=True, text=True)
    lines2 = {ln.split("\t")[0]: ln.split("\t") for ln in r2.stdout.splitlines()}
    assert lines2["r-mmlu"][1] == "cluster" and lines2["r-mmlu"][2] == "/p.py probe --layout capability_cellstore --path /mirror/cells/none__mmlu --task s2parity/mmlu_letter"
    assert lines2["r-mmlu"][6] == "registry.run.store_uri"
    r3 = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", u["agentic/proxy"]["logical_work_id"],
                         "--runs-json", json.dumps([{"run_id": "r-ag", "eval_id": "agentic_murder", "object": "prompted__si_constitution", "store_uri": f"/shared/exp/misalign/{MODEL}_battery/agentic/prompted__constitution"},
                                                    {"run_id": "r-ag-none", "eval_id": "agentic_murder", "object": "prompted__si_constitution", "store_uri": None}]),
                         "--shared", "/shared", "--probe", "/p.py"], capture_output=True, text=True)
    rows3 = {ln.split("\t")[0]: ln.split("\t") for ln in r3.stdout.splitlines()}
    assert rows3["r-ag"][2] == f"/p.py probe --layout misalign_responses --path /shared/exp/misalign/{MODEL}_battery/agentic/prompted__constitution --cell prompted__si_constitution --behavior murder --n-samples 10"
    assert rows3["r-ag-none"][1] == "none" and "not covered, no store on record" in rows3["r-ag-none"][3] and "store_uri null" in rows3["r-ag-none"][3]
    sac = "3001|3000_0|COMPLETED\n3002|3000_1|RUNNING\n3000|3000_[2-3]|PENDING\n"
    at = subprocess.run([sys.executable, str(LIB / "stage.py"), "array-tasks", "--sacct", sac], capture_output=True, text=True)
    assert json.loads(at.stdout) == [{"raw": "3001", "task": 0, "state": "COMPLETED"}, {"raw": "3002", "task": 1, "state": "RUNNING"}]
    sac_f = tmp_path / "sacct.txt"; sac_f.write_text(sac)
    at2 = subprocess.run([sys.executable, str(LIB / "stage.py"), "array-tasks", "--sacct", f"@{sac_f}"], capture_output=True, text=True)
    assert at2.stdout == at.stdout


def contract_arg(state: Path) -> str:
    """`--contract @<file>` as `06 --gate` hands it to `stage.py coverage`: the block `gate-contract --write` produces for this state (the header's
    frozen record, or the admit-once sidecar on an inventory frozen before the field existed — this module's `freeze` builds the header by hand).
    A test that drives `coverage` itself passes it, so the record it writes is tied to the inventory's contract."""
    blk = state.parent / f"{state.name}.contract_block.json"
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--admit-once", "--write", str(blk)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return f"@{blk}"


def retie(state: Path, lw: str | None = None) -> None:
    """Re-tie EVERY coverage record under the state to the inventory's CURRENT coverage contract — what `06 --gate all`'s next pass records after
    the inventory's identities moved (the record is tied to the contract as it is to the run ids; the contract is one sha
    over the WHOLE unit set, so one unit's object pin moving stales every unit's record, and a fixture that stamps a pin by hand after `satisfy`
    plays the gate's re-read with this call). The runs and probes stay. `lw` names the unit the caller moved (documentation; every record is re-tied)."""
    want = A.inventory_contract_sha(str(state))
    for p in sorted(Path(ST.coverage_path(str(state), lw or "x")).parent.glob("*.json")):
        cov = json.load(open(p))
        cov["coverage_contract"] = {**(cov.get("coverage_contract") or {}), "sha256": want, "version": ST.COVERAGE_CONTRACT_VERSION}
        ST.write_atomic(str(p), cov)


def _prov(state: Path, lw: str, entry: dict) -> None:
    """Record one sha-provenance entry the way the funnel does: the append-only ledger event (the truth) AND the sidecar receipt."""
    import grant_ledger as GL
    ST.record_sha_provenance(str(state / "submissions.jsonl"), lw, entry)
    GL.Ledger(str(state / "attempts.jsonl")).append({"event": "sha_provenance", "logical_work_id": lw, **{k: v for k, v in entry.items()}})


# ------------------------------------------------------------------------------------------- per-unit sha provenance

def test_admission_compares_each_unit_to_its_own_recorded_sha_and_the_frozen_contract(tmp_path):
    """Completed work is canonical. The inventory is frozen once; a tree sha is provenance, the unit's EVAL CONTRACT is its identity.
    A unit re-run at a newer sha through the funnel's re-run path records (unit, new sha, the death it answers, the fix commit, the
    contract hash) and admits when its runs carry THAT sha and the contract is identical; a recorded advance whose contract differs
    refuses by name; the units complete at the freeze sha still admit; a re-freeze over a wave with recorded attempts refuses by name."""
    state = tmp_path / "state"; doc = freeze(state, families=["static"]); u = by_unit(doc); runs, pols = [], []
    ledger = str(state / "submissions.jsonl")
    new_sha = "b" * 40
    # static/t0 complete at the freeze sha; static/t1 re-run at the new sha with an identical contract
    satisfy(state, u["static/t0"], runs, pols)
    satisfy(state, u["static/t1"], runs, pols, code_sha=new_sha)
    _prov(state, u["static/t1"]["logical_work_id"], {"sha": SHA, "kind": "first", "attempt_id": "a1", "contract_sha256": ST.contract_sha256(u["static/t1"])})
    _prov(state, u["static/t1"]["logical_work_id"], {"sha": new_sha, "kind": "advance", "attempt_id": "a2",
                                                    "answers": {"job_id": "1", "state": "FAILED", "why": "died young"},
                                                    "fix_commit": f"{new_sha} fix the thing", "contract_sha256": ST.contract_sha256(u["static/t1"])})
    satisfy(state, u["bon/store512"], runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {}, families=["static", "bon"])
    assert not problems, problems
    assert set(ok) >= {u["static/t0"]["logical_work_id"], u["static/t1"]["logical_work_id"]}
    table = ST.sha_table(ledger)
    assert table[u["static/t1"]["logical_work_id"]] == {"sha": new_sha, "frozen_sha": SHA, "n_recorded": 2, "advanced": True, "amended": 0}
    assert table[u["static/t0"]["logical_work_id"]] == {"sha": SHA, "frozen_sha": SHA, "n_recorded": 0, "advanced": False, "amended": 0}
    # the runs at a sha that is NOT the unit's recorded one refuse (static/t0 has no advance: its sha is the freeze sha)
    state2 = tmp_path / "state2"; doc2 = freeze(state2, families=["static"]); u2 = by_unit(doc2); runs2, pols2 = [], []
    satisfy(state2, u2["static/t0"], runs2, pols2, code_sha=new_sha)
    problems2, _ = A.check(str(state2), str(make_registry(tmp_path / "reg2", runs2, pols2)), {}, families=["static"])
    assert any("static/t0" in p and "the unit's recorded sha" in p for p in problems2), problems2
    # a recorded advance whose CONTRACT hash differs from the frozen contract refuses by name (a contract change is a new wave)
    state3 = tmp_path / "state3"; doc3 = freeze(state3, families=["static"]); u3 = by_unit(doc3); runs3, pols3 = [], []
    satisfy(state3, u3["static/t1"], runs3, pols3, code_sha=new_sha)
    _prov(state3, u3["static/t1"]["logical_work_id"], {"sha": new_sha, "kind": "advance", "contract_sha256": "0" * 64})
    problems3, _ = A.check(str(state3), str(make_registry(tmp_path / "reg3", runs3, pols3)), {}, families=["static"])
    assert any("static/t1" in p and "eval contract changed across trees" in p for p in problems3), problems3
    # the contract hash ignores provenance-only fields and names any eval-contract difference
    a, b = dict(u["static/t0"]), json.loads(json.dumps(u["static/t0"]))
    b["code_sha"] = new_sha; b["sbatch_name"] = "other"
    assert ST.contract_sha256(a) == ST.contract_sha256(b)
    b["evals"]["grok420"]["decode_policy_ref"] = "other|static|off|vendor_preset"
    assert ST.contract_sha256(a) != ST.contract_sha256(b) and ST.contract_diff(ST.unit_contract(a), ST.unit_contract(b)) == [
        "evals.grok420.decode_policy_ref: 'qwen35_4b_demo|static|off|vendor_preset' -> 'other|static|off|vendor_preset'"]
    # a re-freeze over a wave with recorded attempts refuses by name; identical at the same sha is a no-op; no attempts -> allowed
    out = str(state / "gate" / "expected_units.json")
    assert ST.refreeze_refusal(out, doc["units"], SHA) is None                                  # identical inventory: no-op
    assert ST.refreeze_refusal(out, doc["units"], new_sha) is None                              # no attempts recorded yet: the freeze may still change
    (state / "attempts.jsonl").write_text(json.dumps({"event": "attempt", "logical_work_id": u["static/t0"]["logical_work_id"], "attempt_id": "x", "job_id": "1"}) + "\n")
    why = ST.refreeze_refusal(out, doc["units"], new_sha)
    assert why and "frozen ONCE (D200)" in why and "the tree would move" in why, why
    changed = json.loads(json.dumps(doc["units"])); changed[0]["evals"][next(iter(changed[0]["evals"]))]["instrument_id"] = "f" * 64
    why2 = ST.refreeze_refusal(out, changed, SHA)
    assert why2 and "unit contract(s) would change" in why2, why2
    assert ST.refreeze_refusal(out, doc["units"], SHA) is None


# --------------------------------------------------------------------------------------- the receipted contract amendment
# amend-contract's tests live in tests/test_amend_contract_enforceable.py (the derived old export, the sacct-proven old runs, the exact
# neutral transitions and --recheck) — they build a synthetic two-commit repo and inject the funnel's sacct reader.


def test_admission_takes_the_units_sha_from_the_append_only_ledger_and_refuses_a_sidecar_that_disagrees(tmp_path):
    """Admission read the sidecar's last sha as truth and checked only the contract hash inside the same file —
    a repaired or copied sidecar with a foreign last sha and a matching hash admitted a foreign tree. The attempts ledger's `sha_provenance`
    events are the truth: the sidecar must agree entry for entry, and the unit's sha is the ledger's."""
    state = tmp_path / "state"; doc = freeze(state, families=["static"]); u = by_unit(doc); runs, pols = [], []
    ledger = str(state / "submissions.jsonl"); lw = u["static/t0"]["logical_work_id"]; new_sha = "b" * 40
    satisfy(state, u["static/t0"], runs, pols, code_sha=new_sha)
    # the ledger says the unit advanced to new_sha (the funnel's sha_provenance event) and the sidecar agrees: admitted
    import grant_ledger as GL
    L = GL.Ledger(str(state / "attempts.jsonl"))
    for e in ({"sha": SHA, "kind": "first", "attempt_id": "a1", "job_id": "1"}, {"sha": new_sha, "kind": "advance", "attempt_id": "a2", "job_id": "2"}):
        ST.record_sha_provenance(ledger, lw, {**e, "contract_sha256": ST.contract_sha256(u["static/t0"])})
        L.append({"event": "sha_provenance", "logical_work_id": lw, **e, "cluster": "cluster_a", "contract_sha256": ST.contract_sha256(u["static/t0"])})
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {}, families=["static"])
    assert not [p for p in problems if lw in p], [p for p in problems if lw in p]
    assert ST.reconcile_provenance(ledger) == []
    # a SIDECAR-ONLY change of the last sha (a foreign tree, the same hash): the ledger disagrees -> refused by name, the unit not admitted
    side = ST.read_provenance_doc(ledger); side["units"][lw][-1]["sha"] = "c" * 40
    ST.write_atomic(ST.provenance_path(ledger), side)
    probs = ST.reconcile_provenance(ledger)
    assert probs and lw in probs[0] and "disagree with the attempts ledger" in probs[0], probs
    problems2, ok2 = A.check(str(state), str(db), {}, families=["static"])
    assert any(lw in p and "disagree with the attempts ledger" in p for p in problems2) and lw not in ok2, problems2
    # the ledger alone decides the recorded sha: with the sidecar removed the unit still admits from the ledger's entries? no — the sidecar
    # must EXIST and agree (an absent receipt is a disagreement when the ledger has entries)
    (state / "gate" / ST.PROVENANCE_FILE).unlink()
    probs3 = ST.reconcile_provenance(ledger)
    assert probs3 and lw in probs3[0], probs3


EV = "pasttense"      # the eval the retired run and its successor share (static/t0 -> static/t1: distinct lwids under satisfy()'s 40-char run-id
                      # prefix — the multi-turn shard-group lwids all truncate to the same prefix and would mint ONE run id for every group)


def _supersede(state: Path, doc: dict, runs: list, *, victim_suffix="/static/t0", donor_suffix="/static/t1", ev=EV):
    """Retire the victim unit's run of `ev` the way a `supersede` ruling does — the row `superseded` with no store — and name
    the donor unit's run of the same eval (complete, bound to the donor) as its successor; returns (victim unit, victim run id, donor unit, donor run id)."""
    ledger = str(state / "submissions.jsonl")
    victim = next(u for u in doc["units"] if u["logical_work_id"].endswith(victim_suffix))
    donor = next(u for u in doc["units"] if u["logical_work_id"].endswith(donor_suffix))
    v_rid = next(r["run_id"] for r in ST.binding_state(ledger, victim["logical_work_id"])["runs"] if r["eval_id"] == ev)
    d_rid = next(r["run_id"] for r in ST.binding_state(ledger, donor["logical_work_id"])["runs"] if r["eval_id"] == ev)
    assert v_rid != d_rid
    v_obj = next(r["object_id"] for r in runs if r["run_id"] == v_rid)
    for r in runs:
        if r["run_id"] == v_rid:
            r["lifecycle_state"] = "superseded"; r["store_uri"] = None
        if r["run_id"] == d_rid:
            r["object_id"] = v_obj          # the real shape: the same cell, the same object — a successor of ANOTHER object is refused (the test below)
    return victim, v_rid, donor, d_rid


def _ruling(v_rid: str, d_rid: str, ruling_id="r2v4-supersede-test-g78-by-g012") -> dict:
    return {"ruling_id": ruling_id, "action": "supersede", "scope": {"run_id": v_rid, "match": {"run_ids": [v_rid]}},
            "target": {"superseded_by_run_id": d_rid, "decision": "D572", "shards_produced": [7, 8]}}


def _cover_through(state: Path, victim: dict, v_rid: str, d_rid: str, *, ruling_id="r2v4-supersede-test-g78-by-g012", cluster="cluster_a", owner=None, drop_goals=()) -> None:
    """The coverage record as 06 --gate writes it for a superseded run: probed at the SUCCESSOR's recorded store for the victim's shards, `successor` on the run."""
    lw = victim["logical_work_id"]; cov_p = ST.coverage_path(str(state), lw); cov = json.load(open(cov_p))
    c = cov["runs"][v_rid]; ev = c["eval_id"]
    c["successor"] = {"run_id": d_rid, "ruling_id": ruling_id, "store_uri": f"/stores/{d_rid}", "cluster": cluster, "owner": owner}
    c["store_uri"] = f"/stores/{d_rid}"
    c["probe"] = probe_ok(victim["evals"][ev], store=f"/stores/{d_rid}", drop_goals=drop_goals); c["probe_error"] = None
    ST.write_atomic(cov_p, cov)


def test_a_superseded_run_is_admitted_through_the_successor_its_ruling_names(tmp_path):
    """A bound run a `supersede` RULING retired is admitted through its SUCCESSOR when the ruling names one
    (target_json.superseded_by_run_id), the successor is complete, of the same eval, bound to a unit of the inventory, and the coverage record was
    probed at the successor's recorded store for THIS unit's shards. Not a force; the inventory is not re-frozen. Synthetic ids throughout."""
    state = tmp_path / "state"; doc = freeze(state, families=["static"])
    runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    victim, v_rid, donor, d_rid = _supersede(state, doc, runs)
    lw = victim["logical_work_id"]
    # the ruled path: the ruling names the successor; the record was probed at its store
    _cover_through(state, victim, v_rid, d_rid, owner=donor["logical_work_id"])
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg", runs, pols, [_ruling(v_rid, d_rid)])), {}, families=["static"])
    assert problems == [] and lw in ok, problems
    # the schema column path: the run row itself names the successor (no ruling row needed)
    runs2 = [dict(r) for r in runs]
    for r in runs2:
        if r["run_id"] == v_rid:
            r["superseded_by_run_id"] = d_rid; r["status_ruling_id"] = "placeholder:col"
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_col", runs2, pols, [_ruling(v_rid, d_rid, ruling_id="placeholder:col")])), {}, families=["static"])
    assert problems == [] and lw in ok, problems          # admission-fix-2: the column is a POINTER — its linked ruling (status_ruling_id) grants it
    # the successor's identity is checked in the run's place: a successor under another instrument refuses, naming both ids
    runs3 = [dict(r) for r in runs]
    for r in runs3:
        if r["run_id"] == d_rid:
            r["instrument_id"] = "inst-foreign"
    problems, _ = A.check(str(state), str(make_registry(tmp_path / "reg_inst", runs3, pols, [_ruling(v_rid, d_rid)])), {}, families=["static"])
    assert any(f"{lw}: run {v_rid} via successor {d_rid} ({EV}) instrument_id='inst-foreign' != frozen" in p for p in problems), problems


def test_a_superseded_run_refuses_by_name_without_a_complete_covering_ruled_successor(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["static"])
    runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    victim, v_rid, donor, d_rid = _supersede(state, doc, runs)
    lw = victim["logical_work_id"]; ev = EV
    def check(name, runs_, rulings, want):
        problems, ok = A.check(str(state), str(make_registry(tmp_path / f"reg_{name}", runs_, pols, rulings)), {}, families=["static"])
        assert any(want in p for p in problems) and lw not in ok, (name, want, problems)
    # (1) superseded, no ruling names a successor — the record untouched (its own null store)
    check("noruling", runs, [], f"{lw}: run {v_rid} ({ev}) is superseded and no `supersede` ruling names its successor")
    # (2) an EXPIRED ruling names none either
    check("expired", runs, [dict(_ruling(v_rid, d_rid), expired_utc="2026-01-02T00:00:00.000Z")], "no `supersede` ruling names its successor")
    # (3) two rulings naming different successors: ambiguous
    check("ambiguous", runs, [_ruling(v_rid, d_rid), _ruling(v_rid, "run-elsewhere", ruling_id="r2v4-other")], f"is superseded by rulings naming different successors")
    # (4) the successor is not in the registry
    check("absent", runs, [_ruling(v_rid, "run-nobody-knows")], f"{lw}: run {v_rid} ({ev}) is superseded by run-nobody-knows (r2v4-supersede-test-g78-by-g012), which is not in the registry")
    # (5) the successor is not complete
    runs5 = [dict(r) for r in runs]
    for r in runs5:
        if r["run_id"] == d_rid:
            r["lifecycle_state"] = "generated"
    _cover_through(state, victim, v_rid, d_rid, owner=donor["logical_work_id"])
    check("incomplete", runs5, [_ruling(v_rid, d_rid)], f"which is generated, not complete (judged|ingested)")
    # (6) the successor is of another eval (the donor's codeattack run)
    p_rid = next(r["run_id"] for r in ST.binding_state(str(state / "submissions.jsonl"), donor["logical_work_id"])["runs"] if r["eval_id"] == "codeattack")
    check("foreign", runs, [_ruling(v_rid, p_rid)], f"is superseded by {p_rid}, which is eval 'codeattack' — a successor of another eval stands in for nothing")
    # (7) the successor is complete but bound to NO unit of the inventory
    runs7 = [dict(r) for r in runs] + [{**next(r for r in runs if r["run_id"] == d_rid), "run_id": "run-unbound-twin"}]
    check("unbound", runs7, [_ruling(v_rid, "run-unbound-twin")], "which no unit of this inventory binds — a successor admitted nowhere stands in for nothing")
    # (8) the record was NOT probed at the successor's store (the retired run's own record stands): re-run 06 --gate
    cov_p = ST.coverage_path(str(state), lw); cov = json.load(open(cov_p)); c = cov["runs"][v_rid]
    c.pop("successor", None); c["store_uri"] = None; c["probe"] = probe_ok(victim["evals"][ev], store="/stores/retired"); ST.write_atomic(cov_p, cov)
    check("notprobed", runs, [_ruling(v_rid, d_rid)], f"{lw}: run {v_rid} ({ev}) is superseded by {d_rid} but the coverage record was not probed at the successor's recorded store")
    # (9) probed at the successor's store, a goal short: the shard's missing goals are named, as for any run
    _cover_through(state, victim, v_rid, d_rid, owner=donor["logical_work_id"], drop_goals=[victim["evals"][ev]["coverage"]["expected"][victim["evals"][ev]["expected_shards"][0]][0]])
    check("short", runs, [_ruling(v_rid, d_rid)], "expected goal(s) not completed")
    # (10) the successor row's record store moved since the gate ran (the record names another store): re-run 06 --gate
    runs10 = [dict(r) for r in runs]
    for r in runs10:
        if r["run_id"] == d_rid:
            r["store_uri"] = "/stores/moved"
    _cover_through(state, victim, v_rid, d_rid, owner=donor["logical_work_id"])
    check("moved", runs10, [_ruling(v_rid, d_rid)], "was not probed at the successor's recorded store")


def test_probe_plan_sends_a_superseded_run_to_its_successors_store_and_cluster(tmp_path):
    """stage.py probe_argv probes a run carrying `successor` at its (the successor's) store_uri — never the frozen root_rel
    (the retired job's own path) — for the unit's shards; probe-plan's fifth column names the successor's cluster and the probe comes from --probe-map."""
    state = tmp_path / "state"; doc = freeze(state, families=["multiturn"])
    u = next(x for x in doc["units"] if x["logical_work_id"].endswith("/multiturn/crescendo/g78"))
    succ = {"run_id": "r-new", "ruling_id": "r-x", "store_uri": "/stores/consolidated/none", "cluster": "cluster_b", "owner": "o"}
    old = {"run_id": "r-old", "eval_id": "crescendo", "object": "none", "array_task": None, "store_uri": "/stores/consolidated/none", "successor": succ}
    host, argv, why = ST.probe_argv(u, old, "/shared_a")
    assert host == "cluster" and why is None and argv[:5] == ["probe", "--layout", "jailbreak_cellstore", "--path", "/stores/consolidated/none"]
    assert argv[argv.index("--shards") + 1] == ",".join(u["evals"]["crescendo"]["expected_shards"])
    own = {**old, "run_id": "r-own", "store_uri": "/stores/own/none", "successor": None}
    assert ST.probe_argv(u, own, "/shared_a")[1][4] == "/stores/own/none"                                      # a run of its own: ITS recorded store, never the frozen root_rel
    own_none = {**own, "store_uri": None}
    assert ST.probe_argv(u, own_none, "/shared_a")[1] is None and "not covered, no store on record" in ST.probe_argv(u, own_none, "/shared_a")[2]
    assert not any("/shared_a/" in str(x) for x in ST.probe_argv(u, own_none, "/shared_a")[1:])              # never guessed under the shared root
    none_store = {**old, "store_uri": None}
    assert ST.probe_argv(u, none_store, "/shared_a")[1] is None and "records no store (store_uri null)" in ST.probe_argv(u, none_store, "/shared_a")[2]
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", u["logical_work_id"],
                        "--runs-json", json.dumps([old, own]), "--shared", "/shared_a", "--probe", "/a/coverage_probe.py",
                        "--probe-map", "cluster_a=/a/coverage_probe.py;cluster_b=/b/coverage_probe.py"], capture_output=True, text=True)
    rows = {ln.split("\t")[0]: ln.split("\t") for ln in r.stdout.splitlines()}
    assert rows["r-old"][1] == "cluster" and rows["r-old"][2].startswith("/b/coverage_probe.py probe --layout jailbreak_cellstore --path /stores/consolidated/none") and rows["r-old"][4] == "cluster_b"
    assert rows["r-own"][1] == "cluster" and rows["r-own"][2].startswith("/a/coverage_probe.py probe") and rows["r-own"][4] == ""
    # without a map entry for the successor's cluster the unit's probe is used (fail open to the plain path is NOT the design: the gate builds the map from every label)
    r2 = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", u["logical_work_id"],
                         "--runs-json", json.dumps([old]), "--shared", "/shared_a", "--probe", "/a/coverage_probe.py"], capture_output=True, text=True)
    assert r2.stdout.split("\t")[2].startswith("/a/coverage_probe.py probe") and r2.stdout.rstrip("\n").split("\t")[4] == "cluster_b"


def test_a_successor_of_another_object_or_with_an_unrecorded_object_never_certifies_the_retired_unit(tmp_path):
    """admission-fix-2: a `supersede` ruling pointing at a complete, inventory-bound run of ANOTHER steered object under the
    same eval must not certify the retired object's unit — the run's and the successor's object_id must both be recorded and equal, and equal to
    the unit's frozen per-cell pin when the inventory carries one. The fixture's objects are distinct per unit by default, so the donor stands in
    only once it carries the victim's object."""
    state = tmp_path / "state"; doc = freeze(state, families=["static"])
    runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    victim, v_rid, donor, d_rid = _supersede(state, doc, runs)
    lw = victim["logical_work_id"]; ev = EV
    v_obj = next(r["object_id"] for r in runs if r["run_id"] == v_rid)
    assert v_obj.startswith("obj:") and any(r["object_id"] != v_obj for r in runs)                                # distinct per unit by default (the donor unit's other runs keep its own)
    _cover_through(state, victim, v_rid, d_rid, owner=donor["logical_work_id"])
    def check(name, runs_, want=None, inv_objects=None):
        if inv_objects is not None:
            inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
            for u in inv["units"]:
                if u["logical_work_id"] == lw:
                    u["objects"] = inv_objects
            ST.write_atomic(str(inv_p), inv); retie(state, lw)                       # the pin is identity: the record is re-tied as the gate's next pass would
        problems, ok = A.check(str(state), str(make_registry(tmp_path / f"reg_{name}", runs_, pols, [_ruling(v_rid, d_rid)])), {}, families=["static"])
        if want is None:
            assert problems == [] and lw in ok, (name, problems)
        else:
            assert any(want in p for p in problems) and lw not in ok, (name, want, problems)
    # the positive case: the same (real, non-`none`) object on both sides
    check("same", runs)
    # a DIFFERENT object with the same eval / instrument / decode: refused by name
    runs2 = [dict(r) for r in runs]
    for r in runs2:
        if r["run_id"] == d_rid:
            r["object_id"] = "object-for-another-fit"
    check("other", runs2, f"{lw}: run {v_rid} ({ev}) is of object {v_obj!r} but its successor {d_rid} is of object 'object-for-another-fit' — a successor of ANOTHER object stands in for nothing")
    # null on the successor's side, then on the run's side: a refusal, never a match
    runs3 = [dict(r) for r in runs]
    for r in runs3:
        if r["run_id"] == d_rid:
            r["object_id"] = None
    check("null_succ", runs3, f"is superseded by {d_rid}, whose object identity is unrecorded (object_id null)")
    runs4 = [dict(r) for r in runs]
    for r in runs4:
        if r["run_id"] == v_rid:
            r["object_id"] = None
    check("null_run", runs4, f"{lw}: run {v_rid} ({ev}) records NO object_id — the object identity is unrecorded")
    # the unit's frozen pin for the cell: equal admits; another pin refuses the run (direct path) and the successor alike
    cell = next(r["object"] for r in ST.binding_state(str(state / "submissions.jsonl"), lw)["runs"] if r["run_id"] == v_rid)
    check("pin_ok", runs, inv_objects={cell: v_obj})
    check("pin_other", runs, f"{lw}: run {v_rid} ({ev}) is of object {v_obj!r} but the unit's frozen pin for cell {cell!r} is 'pinned-elsewhere' — not this unit's object", inv_objects={cell: "pinned-elsewhere"})
    runs5 = [dict(r) for r in runs]
    for r in runs5:
        if r["run_id"] == d_rid:
            r["object_id"] = "obj:drifted"
    check("pin_succ", runs5, f"is of object {v_obj!r} but its successor {d_rid} is of object 'obj:drifted'", inv_objects={cell: v_obj})


def test_the_direct_path_requires_a_recorded_object_and_honours_a_frozen_pin(tmp_path):
    """admission-fix-2: the non-superseded path was blind to the object (RUN_COLUMNS never selected object_id). A bound run with object_id null
    refuses by name; a unit whose freeze pins the cell's object refuses a row of another object and admits the pinned one."""
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu"]); unit = by_unit(doc)["mmlu/mmlu"]; lw = unit["logical_work_id"]
    runs, pols = [], []
    satisfy(state, unit, runs, pols)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg0", runs, pols)), {}, families=["mmlu"])
    assert problems == [] and lw in ok, problems
    rid = runs[0]["run_id"]; obj = runs[0]["object_id"]
    runs_null = [dict(r, object_id=None) if r["run_id"] == rid else dict(r) for r in runs]
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg1", runs_null, pols)), {}, families=["mmlu"])
    assert any(f"{lw}: run {rid} (mmlu) records NO object_id — the object identity is unrecorded" in p for p in problems) and lw not in ok, problems
    inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    cell = ST.binding_state(str(state / "submissions.jsonl"), lw)["runs"][0]["object"]
    for u in inv["units"]:
        if u["logical_work_id"] == lw:
            u["objects"] = {cell: obj}
    ST.write_atomic(str(inv_p), inv); retie(state, lw)                               # the pin is identity: the record re-tied as the gate's next pass would
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg2", runs, pols)), {}, families=["mmlu"])
    assert problems == [] and lw in ok, problems
    for u in inv["units"]:
        if u["logical_work_id"] == lw:
            u["objects"] = {cell: "another-object"}
    ST.write_atomic(str(inv_p), inv); retie(state, lw)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg3", runs, pols)), {}, families=["mmlu"])
    assert any(f"is of object {obj!r} but the unit's frozen pin for cell {cell!r} is 'another-object'" in p for p in problems) and lw not in ok, problems


def test_the_column_pointer_is_validated_against_its_ruling_before_it_names_a_successor(tmp_path):
    """admission-fix-2: `run.superseded_by_run_id` is a pointer, not an authority — the ruling it links to must be
    active, a `supersede`, name this run in its scope and the column's run in its target; an expired ruling with the column still set admitted
    the successor before. Refusals by name: expired, absent, not a supersede, a ruling naming another successor or another run."""
    state = tmp_path / "state"; doc = freeze(state, families=["static"])
    runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    victim, v_rid, donor, d_rid = _supersede(state, doc, runs)
    lw = victim["logical_work_id"]; ev = EV
    _cover_through(state, victim, v_rid, d_rid, owner=donor["logical_work_id"])
    def with_col(rulings_id):
        out = [dict(r) for r in runs]
        for r in out:
            if r["run_id"] == v_rid:
                r["superseded_by_run_id"] = d_rid; r["status_ruling_id"] = rulings_id
        return out
    def check(name, runs_, rulings, want=None):
        problems, ok = A.check(str(state), str(make_registry(tmp_path / f"reg_{name}", runs_, pols, rulings)), {}, families=["static"])
        if want is None:
            assert problems == [] and lw in ok, (name, problems)
        else:
            assert any(want in p for p in problems) and lw not in ok, (name, want, problems)
    R = "r2v4-supersede-test-g78-by-g012"
    # the column + its ACTIVE linked ruling: admitted (the ruling is the authority; the column the pointer)
    check("active", with_col(R), [_ruling(v_rid, d_rid, ruling_id=R)])
    # the column with the linked ruling EXPIRED: refused by name
    check("expired", with_col(R), [dict(_ruling(v_rid, d_rid, ruling_id=R), expired_utc="2026-01-02T00:00:00.000Z")],
          f"{lw}: run {v_rid} ({ev}) is superseded but superseded_by_run_id names {d_rid} but the linked ruling {R} is EXPIRED (2026-01-02T00:00:00.000Z)")
    # the column pointing at a run the linked ruling does not name as the successor: conflicting authority
    check("conflict", with_col(R), [_ruling(v_rid, "run-somebody-else", ruling_id=R)], f"the linked ruling {R} names 'run-somebody-else' as the successor (conflicting authority)")
    # the linked ruling's scope names ANOTHER run: conflicting authority
    other = _ruling("run-not-this-one", d_rid, ruling_id=R)
    check("scope", with_col(R), [other], f"the linked ruling {R}'s scope does not name this run (conflicting authority)")
    # the column names a status_ruling_id that does not exist, or no ruling table at all: absent authority
    check("absent_id", with_col("placeholder:gone"), [_ruling(v_rid, d_rid, ruling_id=R)], "status_ruling_id placeholder:gone, but no such ruling exists (absent authority)")
    check("no_table", with_col(R), [], f"superseded_by_run_id names {d_rid} but the registry carries no ruling table to grant it (absent authority)")
    # the linked ruling is not a supersede
    check("kind", with_col(R), [dict(_ruling(v_rid, d_rid, ruling_id=R), action="hold_out")], f"the linked ruling {R} is a 'hold_out' ruling, not a supersede")
    # the column with NO status_ruling_id: an ACTIVE supersede ruling naming this run with that successor grants it; none does -> absent authority
    check("by_scope", with_col(None), [_ruling(v_rid, d_rid, ruling_id=R)])
    check("by_scope_expired", with_col(None), [dict(_ruling(v_rid, d_rid, ruling_id=R), expired_utc="2026-01-02T00:00:00.000Z")], "no ACTIVE `supersede` ruling names this run with that successor (absent authority)")


def test_a_pinned_unit_refuses_a_run_without_a_resolvable_pinned_cell(tmp_path):
    """admission-fix-3: with object pins on the unit, a binding that records no cell for a run (`stage.py bind --state
    registered --run-ids` without --runs-json) or a cell the pins do not name used to bypass the digest comparison silently — a complete run of
    ANOTHER object admitted. Both refuse by name now; a unit without pins keeps the lenient form (the baseline inventories carry no pins)."""
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu"]); unit = by_unit(doc)["mmlu/mmlu"]; lw = unit["logical_work_id"]
    runs, pols = [], []
    satisfy(state, unit, runs, pols)
    ledger = str(state / "submissions.jsonl"); rid = runs[0]["run_id"]; obj = runs[0]["object_id"]
    b = ST.binding_state(ledger, lw); cell = b["runs"][0]["object"]
    def pin(objects):
        inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
        for u in inv["units"]:
            if u["logical_work_id"] == lw:
                if objects is None:
                    u.pop("objects", None)
                else:
                    u["objects"] = objects
        ST.write_atomic(str(inv_p), inv); retie(state, lw)                           # the pin is identity: the record re-tied as the gate's next pass would
    def check(name, runs_, want=None):
        problems, ok = A.check(str(state), str(make_registry(tmp_path / f"reg_{name}", runs_, pols)), {}, families=["mmlu"])
        if want is None:
            assert problems == [] and lw in ok, (name, problems)
        else:
            assert any(want in p_ for p_ in problems) and lw not in ok, (name, want, problems)
    # the pinned unit with the full binding (cell present, the pinned object): admitted
    pin({cell: obj}); check("full", runs)
    # the INCOMPLETE binding form (run ids only, no cells) on a pinned unit: refused by name — even with a different object in the registry
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": b["run_ids"], "job_id": b.get("job_id"), "cluster": b.get("cluster")})
    assert ST.binding_state(ledger, lw).get("runs") in (None, [])
    check("nocell", runs, f"{lw}: run {rid} (mmlu) has no cell recorded in its binding (route.object missing) while the unit pins objects for {[cell]!r}")
    runs_other = [dict(r, object_id="object-for-another-fit") if r["run_id"] == rid else dict(r) for r in runs]
    check("nocell_other", runs_other, "has no cell recorded in its binding (route.object missing)")            # the reproduction: refused, not admitted
    # a cell the pins do not name: a foreign cell
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": b["run_ids"], "job_id": b.get("job_id"), "cluster": b.get("cluster"),
                              "runs": [dict(r_, object="mystery_cell") for r_ in b["runs"]]})
    check("foreign", runs, f"{lw}: run {rid} (mmlu) is bound to cell 'mystery_cell', which the unit's object pins do not name ({[cell]!r}) — a foreign cell")
    # the cell restored: the wrong digest still refuses, the pinned one admits
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": b["run_ids"], "job_id": b.get("job_id"), "cluster": b.get("cluster"), "runs": b["runs"]})
    check("wrong_digest", runs_other, f"is of object 'object-for-another-fit' but the unit's frozen pin for cell {cell!r} is {obj!r}")
    check("full_again", runs)
    # no pins on the unit: the run-ids-only form is still admitted (the baseline inventories carry no pins; the registry's object identity is the row's)
    pin(None)
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": b["run_ids"], "job_id": b.get("job_id"), "cluster": b.get("cluster")})
    check("unpinned_nocell", runs)


def test_a_superseded_run_on_a_pinned_unit_also_needs_its_retired_cell(tmp_path):
    """admission-fix-3: the successor path probes the successor's store without the retired cell, so the cell rule must hold before the
    successor logic — a superseded run with no recorded cell on a pinned unit refuses by name."""
    state = tmp_path / "state"; doc = freeze(state, families=["static"])
    runs, pols = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, pols)
    victim, v_rid, donor, d_rid = _supersede(state, doc, runs)
    lw = victim["logical_work_id"]; ledger = str(state / "submissions.jsonl")
    _cover_through(state, victim, v_rid, d_rid, owner=donor["logical_work_id"])
    b = ST.binding_state(ledger, lw); cell = next(r["object"] for r in b["runs"] if r["run_id"] == v_rid); v_obj = next(r["object_id"] for r in runs if r["run_id"] == v_rid)
    inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    for u in inv["units"]:
        if u["logical_work_id"] == lw:
            u["objects"] = {cell: v_obj}
    ST.write_atomic(str(inv_p), inv); retie(state, lw)                               # the pin is identity: the record re-tied as the gate's next pass would
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_ok", runs, pols, [_ruling(v_rid, d_rid)])), {}, families=["static"])
    assert problems == [] and lw in ok, problems
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": b["run_ids"], "job_id": b.get("job_id"), "cluster": b.get("cluster")})
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_nocell", runs, pols, [_ruling(v_rid, d_rid)])), {}, families=["static"])
    assert any(f"{lw}: run {v_rid} ({EV}) has no cell recorded in its binding (route.object missing)" in p_ for p_ in problems) and lw not in ok, problems


# --------------------------------------------------------------- the producer-retired BoN climb registration

FACTS_SHA, FACTS_N = "f" * 64, 9     # the store facts a close files on BOTH registrations of one store: identical by default, as required


def _store_facts(db: Path, rows: dict) -> None:
    """Stamp the run rows' `store_completion_sha256` / `n_items_generated` on a FIXTURE registry — front_door_testlib.make_registry carries neither
    column (the real `run` table does: schema_def), so they are added here — rows = {run_id: (sha, n_items)}; a None fact stays NULL (unrecorded)."""
    conn = sqlite3.connect(db)
    have = {r[1] for r in conn.execute("PRAGMA table_info(run)")}
    for col, typ in (("store_completion_sha256", "TEXT"), ("n_items_generated", "INTEGER")):
        if col not in have:
            conn.execute(f"ALTER TABLE run ADD COLUMN {col} {typ}")
    for rid, (sha, n) in rows.items():
        conn.execute("UPDATE run SET store_completion_sha256 = ?, n_items_generated = ? WHERE run_id = ?", (sha, n, rid))
    conn.commit(); conn.close()


def _producer_superseded(db: Path, rid: str, by: str, *, reason="duplicate_registration", decision="D2599", utc="2026-10-11T19:02:00.000Z", facts=True) -> None:
    """The producer's record as the full22 closer files it (registry_close_stores.file_superseded) and the harvester folds it: a run_state_event row,
    state `superseded`, extra_json {superseded_by_run_id, reason, decision, …}; NO ruling row, and run.superseded_by_run_id null (the column is
    identity, never patched — the census effect is this record). `facts`: the store facts on the two rows — True stamps the identical
    bytes and count on both (the closer's run_patch), a {run_id: (sha, n_items)} dict stamps those, False leaves the fixture's run table
    without the columns (a registry that cannot show the facts)."""
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS run_state_event (run_id TEXT NOT NULL, state TEXT NOT NULL, event_utc TEXT NOT NULL, evidence_kind TEXT NOT NULL, "
                 "evidence_uri TEXT NOT NULL DEFAULT '', scheduler_state TEXT, note TEXT, extra_json TEXT NOT NULL DEFAULT '{}')")
    conn.execute("INSERT INTO run_state_event VALUES (?,?,?,?,?,?,?,?)",
                 (rid, "superseded", utc, "completion_artifact", "/stores/x/bon_served_manifest.json", None,
                  "duplicate registration of one store by two jobs; the store's measurement is carried by the store512 run",
                  json.dumps({"superseded_by_run_id": by, "reason": reason, "decision": decision, "sibling_job_id": "2", "superseded_role": "climb", "survivor_role": "store512"})))
    conn.commit(); conn.close()
    if facts is True:
        _store_facts(db, {rid: (FACTS_SHA, FACTS_N), by: (FACTS_SHA, FACTS_N)})
    elif isinstance(facts, dict):
        _store_facts(db, facts)


def _climb_shared(state: Path, climb: dict, s512: dict, runs: list, *, lifecycle="superseded", tasks=(0, 1, 2, 3), shards=(0, 1, 2, 3), shard_indices=True) -> str:
    """The shared-run climb as 06 --gate binds and covers it under the companion key: ONE run id bound once per array task (the binding's `runs`
    rows keep the tasks, `run_ids` names it once), the run row over the SAME store and object as the store512's (one store, two registrations), the
    coverage record with `array_tasks`, the successor the registry names (no ruling id) and ONE probe document over `shards` (`shard_indices` named
    unless the test says a probe before it read them). Returns the climb run id."""
    lw = climb["logical_work_id"]; ledger = str(state / "submissions.jsonl")
    s_rid = ST.binding_state(ledger, s512["logical_work_id"])["run_ids"][0]
    s_row = next(r for r in runs if r["run_id"] == s_rid)
    rid = f"run-{ST.safe_id(lw)[:40]}-strongreject_bon"           # one id for the cell: the job-independent key
    ST.write_binding(ledger, ST.submitted_record(lw, "9", "cluster_a", f"{lw}@cluster_a@1", array=climb.get("array")))
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": [rid], "job_id": "9", "cluster": "cluster_a",
                              "runs": [{"run_id": rid, "eval_id": "strongreject_bon", "object": "none", "array_task": t} for t in tasks]})
    ev = climb["evals"]["strongreject_bon"]
    runs.append({**s_row, "run_id": rid, "lifecycle_state": lifecycle, "instrument_id": ev["instrument_id"] or "inst-strongreject_bon"})
    exp = ["gA", "gB", "gC"]
    probe = {"layout": "bon_climb_flat", "path": "/shared/bon_climb/tag", "exists": True, "notes": [],
             "shards": {f"shard{k:02d}": {"present": True, "expected": exp, "completed": list(exp), "errored": [], "refused": [], "notes": []} for k in shards}}
    if shard_indices:
        probe["shard_indices"] = list(shards)
    succ = {"run_id": s_rid, "ruling_id": None, "store_uri": s_row["store_uri"], "cluster": "cluster_a", "owner": s512["logical_work_id"]} if lifecycle == "superseded" else None
    # the record is tied to the inventory's coverage contract as 06 --gate writes it (satisfy's rule)
    cov = {"logical_work_id": lw, "run_ids": [rid], "coverage_contract": {"sha256": A.inventory_contract_sha(str(state)), "version": ST.COVERAGE_CONTRACT_VERSION},
           "runs": {rid: {"eval_id": "strongreject_bon", "object": "none", "array_task": list(tasks)[-1], "array_tasks": list(tasks),
                          "store_uri": s_row["store_uri"], "probe": probe, "probe_error": None, "successor": succ}}}
    ST.write_atomic(ST.coverage_path(str(state), lw), cov)
    rule_family(state, climb["family"])        # the gate's verdict step after the bind: ruled over the family's bindings as they stand now
    return rid


def test_a_climb_run_the_producer_retired_as_a_duplicate_registration_is_admitted_through_the_store512_run(tmp_path):
    """The closer files a BoN climb registration `superseded` by the store512's with a PRODUCER run_state record (reason
    duplicate_registration, extra.superseded_by_run_id; no ruling — the census effect), and the gate refused every such run ("no `supersede`
    ruling names its successor"). Now `supersession` reads the record, CHECKS its claim against the rows (one store: store_uri, eval, object,
    model equal on both) and admits the climb unit through the store512 run — complete, bound by the store512 unit, the climb's own shards
    probed; a claim the rows do not bear out, an unrecorded store, an incomplete survivor or two records naming different successors refuse by
    name; another reason names nothing; a ruling that names a successor wins over the record. Synthetic ids throughout."""
    state = tmp_path / "state"; doc = freeze(state, families=["static", "bon"]); runs, pols = [], []
    u = by_unit(doc)
    assert {"bon/store512", "bon/climb"} <= set(u) and u["bon/store512"]["family"] == u["bon/climb"]["family"] == "bon"
    for name in ("static/t0", "static/t1", "bon/store512"):
        satisfy(state, u[name], runs, pols)
    rid = _climb_shared(state, u["bon/climb"], u["bon/store512"], runs)
    lw = u["bon/climb"]["logical_work_id"]; s_lw = u["bon/store512"]["logical_work_id"]
    s_rid = ST.binding_state(str(state / "submissions.jsonl"), s_lw)["run_ids"][0]

    def reg(name, runs_, rulings=(), record=True, by=None, **kw):
        db = make_registry(tmp_path / f"reg_{name}", runs_, pols, list(rulings))
        if record:
            _producer_superseded(db, rid, by or s_rid, **kw)
        return db
    # before the record: refused — nothing names a successor (the refusal now names the third source too)
    problems, ok = A.check(str(state), str(reg("none", runs, record=False)), {}, families=["bon"])
    assert any(f"{lw}: run {rid} (strongreject_bon) is superseded and no `supersede` ruling names its successor" in p and "duplicate_registration run_state record" in p
               for p in problems) and lw not in ok, problems
    # the record: admitted through the store512 run, no ruling anywhere; the resolution says what granted it
    db = reg("ok", runs)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sup = A.supersession(conn, rid); conn.close()
    assert sup["successor"] == s_rid and sup["ruling_id"] is None and sup["duplicate_registration"] and sup["decision"] == "D2599" and "duplicate_registration" in sup["source"], sup
    problems, ok = A.check(str(state), str(db), {}, families=["bon"])
    assert problems == [] and lw in ok and s_lw in ok, problems
    # the claim is CHECKED against the rows: a survivor over another store, of another object, or of another model is not one store — refused by name
    for col, val in (("store_uri", "/stores/another"), ("object_id", "obj:another"), ("model_id", "another/model")):
        runs_x = [dict(r) for r in runs]
        for r in runs_x:
            if r["run_id"] == s_rid:
                r[col] = val
        problems, ok = A.check(str(state), str(reg(f"diff_{col}", runs_x)), {}, families=["bon"])
        assert any(f"{lw}: run {rid} (strongreject_bon) is superseded but the producer's duplicate_registration record (D2599) names {s_rid} as the other registration "
                   f"of this run's store, but the rows differ in {col}" in p for p in problems) and lw not in ok, (col, problems)
    # a trailing slash is not a different store
    runs_s = [dict(r) for r in runs]
    for r in runs_s:
        if r["run_id"] == s_rid:
            r["store_uri"] = r["store_uri"] + "/"
    cov_p = ST.coverage_path(str(state), lw); cov = json.load(open(cov_p)); cov["runs"][rid]["store_uri"] = cov["runs"][rid]["store_uri"] + "/"
    cov["runs"][rid]["successor"]["store_uri"] = cov["runs"][rid]["store_uri"]; ST.write_atomic(cov_p, cov)
    problems, ok = A.check(str(state), str(reg("slash", runs_s)), {}, families=["bon"])
    assert problems == [] and lw in ok, problems
    cov["runs"][rid]["store_uri"] = cov["runs"][rid]["store_uri"][:-1]; cov["runs"][rid]["successor"]["store_uri"] = cov["runs"][rid]["store_uri"]; ST.write_atomic(cov_p, cov)
    # the store unrecorded on the survivor: refused — verified by the rows, never by the record
    runs_n = [dict(r) for r in runs]
    for r in runs_n:
        if r["run_id"] == s_rid:
            r["store_uri"] = None
    problems, ok = A.check(str(state), str(reg("nostore", runs_n)), {}, families=["bon"])
    assert any(f"but store_uri is unrecorded on {s_rid} — a duplicate registration is verified by the rows, never by the record" in p for p in problems) and lw not in ok, problems
    # another reason names nothing; a survivor not in the registry cannot be checked; the survivor still generating is not complete
    problems, _ = A.check(str(state), str(reg("reason", runs, reason="consolidation")), {}, families=["bon"])
    assert any("no `supersede` ruling names its successor" in p for p in problems), problems
    problems, _ = A.check(str(state), str(reg("absent", runs, by="run-nobody-knows")), {}, families=["bon"])
    assert any("names run-nobody-knows as the other registration of this run's store, but run-nobody-knows is not in the registry — the claim cannot be checked" in p for p in problems), problems
    runs_r = [dict(r) for r in runs]
    for r in runs_r:
        if r["run_id"] == s_rid:
            r["lifecycle_state"] = "generated"
    problems, ok = A.check(str(state), str(reg("running", runs_r)), {}, families=["bon"])
    assert any(f"is superseded by {s_rid} (run_state_event superseded, reason duplicate_registration, decision D2599" in p and "which is generated, not complete (judged|ingested)" in p
               for p in problems) and lw not in ok, problems
    # two records naming different successors: ambiguous, nothing admitted
    db2 = reg("two", runs); _producer_superseded(db2, rid, "run-other-survivor", utc="2026-10-11T19:03:00.000Z")
    problems, _ = A.check(str(state), str(db2), {}, families=["bon"])
    assert any(f"is superseded by rulings naming different successors {sorted(['run-other-survivor', s_rid])}" in p for p in problems), problems
    # a ruling that names a successor WINS over the record: it names a run nobody knows, and that is what refuses
    problems, _ = A.check(str(state), str(reg("ruled", runs, rulings=[_ruling(rid, "run-nobody-knows")])), {}, families=["bon"])
    assert any("is superseded by run-nobody-knows (r2v4-supersede-test-g78-by-g012), which is not in the registry" in p for p in problems), problems
    # the coverage record must still name the successor the registry names (06 --gate writes it): without it, re-run 06 --gate
    cov["runs"][rid]["successor"] = None; ST.write_atomic(cov_p, cov)
    problems, ok = A.check(str(state), str(reg("nosucc", runs)), {}, families=["bon"])
    assert any(f"{lw}: run {rid} (strongreject_bon) is superseded by {s_rid} but the coverage record does not name the successor the registry names" in p for p in problems) and lw not in ok, problems


def test_a_duplicate_registration_claim_is_verified_by_the_recorded_bytes_and_item_count_never_by_the_uri_alone(tmp_path):
    """producer_supersession checked the record's claim against store_uri, eval, object and model alone, so a climb and its purported store512
    successor with ONE store_uri but different store_completion_sha256 or n_items_generated (a later patch or correction) still admitted the
    superseded climb through that successor — against the explicit bytes-and-count condition. Now both facts must be recorded on both rows and
    identical: each mismatch and each missing fact refuses by name (naming the row); the count as text on one side is the same count; a no-fit
    store's recorded 0 is a value, not a gap; the store_uri check still comes first (the earlier refusal byte for byte); a registry whose run
    table has no such column cannot verify the claim and refuses by name — never admitted on the uri alone. Synthetic ids throughout."""
    state = tmp_path / "state"; doc = freeze(state, families=["static", "bon"]); runs, pols = [], []
    u = by_unit(doc)
    for name in ("static/t0", "static/t1", "bon/store512"):
        satisfy(state, u[name], runs, pols)
    rid = _climb_shared(state, u["bon/climb"], u["bon/store512"], runs)
    lw = u["bon/climb"]["logical_work_id"]; s_lw = u["bon/store512"]["logical_work_id"]
    s_rid = ST.binding_state(str(state / "submissions.jsonl"), s_lw)["run_ids"][0]
    pre = (f"{lw}: run {rid} (strongreject_bon) is superseded but the producer's duplicate_registration record (D2599) names {s_rid} as the other registration "
           f"of this run's store, but ")

    def reg(name, facts, runs_=None):
        db = make_registry(tmp_path / f"reg_{name}", runs_ or runs, pols, [])
        _producer_superseded(db, rid, s_rid, facts=facts)
        return db

    def sup_of(db):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return A.supersession(conn, rid)
        finally:
            conn.close()
    # identical bytes and count on both rows: admitted, and the resolution says what it verified
    db = reg("ok", {rid: (FACTS_SHA, FACTS_N), s_rid: (FACTS_SHA, FACTS_N)})
    sup = sup_of(db)
    assert sup["successor"] == s_rid and sup["duplicate_registration"] and "store_completion_sha256 and n_items_generated identical" in sup["source"], sup
    problems, ok = A.check(str(state), str(db), {}, families=["bon"])
    assert problems == [] and lw in ok and s_lw in ok, problems
    # the count spelled as text on one side is the same count (the schema column is INTEGER; a filing may spell it)
    problems, ok = A.check(str(state), str(reg("text_n", {rid: (FACTS_SHA, "9"), s_rid: (FACTS_SHA, 9)})), {}, families=["bon"])
    assert problems == [] and lw in ok, problems
    # a no-fit store: 0 items recorded on both rows is a value, not a gap
    problems, ok = A.check(str(state), str(reg("nofit", {rid: (FACTS_SHA, 0), s_rid: (FACTS_SHA, 0)})), {}, families=["bon"])
    assert problems == [] and lw in ok, problems
    # one store_uri with two digests is not a duplicate: refused by name; two counts likewise
    other = "e" * 64
    problems, ok = A.check(str(state), str(reg("sha", {rid: (FACTS_SHA, FACTS_N), s_rid: (other, FACTS_N)})), {}, families=["bon"])
    assert any(p.startswith(pre + f"the rows differ in store_completion_sha256 ({FACTS_SHA!r} vs {other!r}) — one store_uri with different bytes or item count is not a "
                            f"duplicate registration (D456); the record's claim is not borne out") and p.endswith("— nothing stands in for its work") for p in problems) and lw not in ok, problems
    problems, ok = A.check(str(state), str(reg("n", {rid: (FACTS_SHA, FACTS_N), s_rid: (FACTS_SHA, FACTS_N + 1)})), {}, families=["bon"])
    assert any(p.startswith(pre + f"the rows differ in n_items_generated ({FACTS_N!r} vs {FACTS_N + 1!r}) — one store_uri with different bytes or item count") for p in problems) \
        and lw not in ok, problems
    # a fact missing on either row: refused by name, naming the row that lacks it
    for who, facts in (("this run", {rid: (None, FACTS_N), s_rid: (FACTS_SHA, FACTS_N)}), (s_rid, {rid: (FACTS_SHA, FACTS_N), s_rid: (None, FACTS_N)})):
        problems, ok = A.check(str(state), str(reg(f"nosha_{who[:6]}", facts)), {}, families=["bon"])
        assert any(p.startswith(pre + f"store_completion_sha256 is unrecorded on {who} — a duplicate registration is verified by the rows, never by the record "
                                f"(D456: identical store_completion_sha256 and n_items_generated)") for p in problems) and lw not in ok, (who, problems)
    for who, facts in (("this run", {rid: (FACTS_SHA, None), s_rid: (FACTS_SHA, FACTS_N)}), (s_rid, {rid: (FACTS_SHA, FACTS_N), s_rid: (FACTS_SHA, None)})):
        problems, ok = A.check(str(state), str(reg(f"non_{who[:6]}", facts)), {}, families=["bon"])
        assert any(p.startswith(pre + f"n_items_generated is unrecorded on {who} — a duplicate registration is verified by the rows") for p in problems) and lw not in ok, (who, problems)
    # the store_uri check still comes first: an unrecorded store_uri is the earlier refusal byte for byte, whatever the facts say
    runs_n = [dict(r) for r in runs]
    for r in runs_n:
        if r["run_id"] == rid:
            r["store_uri"] = None
    assert sup_of(reg("nouri", True, runs_n)) == {"refused": f"the producer's duplicate_registration record (D2599) names {s_rid} as the other registration of this run's store, "
                                                             f"but store_uri is unrecorded on this run — a duplicate registration is verified by the rows, never by the record"}
    # a registry whose run table carries neither fact column cannot verify the claim: refused by name, never admitted on the uri alone
    db = reg("nocols", False)
    assert sup_of(db) == {"refused": f"the producer's duplicate_registration record (D2599) names {s_rid} as the other registration of this run's store, but the registry's "
                                     f"run table has no `store_completion_sha256` column — the store's bytes and item count (D456: identical store_completion_sha256 and "
                                     f"n_items_generated) cannot be verified; the claim is not granted"}
    problems, ok = A.check(str(state), str(db), {}, families=["bon"])
    assert any(p.startswith(pre + "the registry's run table has no `store_completion_sha256` column") for p in problems) and lw not in ok, problems


def test_a_climb_run_shared_by_the_arrays_tasks_is_read_over_every_shard_and_a_one_shard_record_refuses_by_name(tmp_path):
    """The climb array's four tasks register ONE run per cell; 06 --gate binds that id once per task and its coverage pass keeps one row per
    run id (the LAST task's index), so the probe read the last task's shard alone and the gate judged the cell on it. Now `stage.py bind`
    names the run once and keeps the per-task rows; `probe-plan` reads the task identities from the binding and plans ONE probe over every
    shard (`--shard-index` per task); `coverage` records `array_tasks`; admission requires every expected shard read across the eval's bound
    runs — the document a coverage_probe.py before the shared-run probe prints (one shard, no `shard_indices`) refuses by name, saying why;
    a shard short of a goal refuses by name; the per-task shape is unchanged."""
    state = tmp_path / "state"; doc = freeze(state, families=["static", "bon"]); runs, pols = [], []
    u = by_unit(doc); climb = u["bon/climb"]; lw = climb["logical_work_id"]; ledger = str(state / "submissions.jsonl")
    for name in ("static/t0", "static/t1", "bon/store512"):
        satisfy(state, u[name], runs, pols)
    # (1) the CLIs 06 drives, on the shared shape: bind with the id once per task (06's per-task enumeration), probe-plan on the collapsed row, coverage
    rid = "run-climb-shared"
    ST.write_binding(ledger, ST.submitted_record(lw, "9", "cluster_a", f"{lw}@cluster_a@1", array="0-3"))
    rows = [{"run_id": rid, "eval_id": "strongreject_bon", "object": "none", "array_task": t, "store_uri": None} for t in (0, 1, 2, 3)]
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "bind", "--ledger", ledger, "--logical-work-id", lw, "--state", "registered",
                        "--run-ids", ",".join([rid] * 4), "--pending-file", "/p/<raw task id>.json", "--runs-json", json.dumps(rows)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    b = ST.binding_state(ledger, lw)
    assert b["run_ids"] == [rid] and sorted(x["array_task"] for x in b["runs"]) == [0, 1, 2, 3]       # the SET of runs, once; the four task identities kept
    assert ST.bound_array_tasks(str(state), lw) == {rid: [0, 1, 2, 3]}
    rule_family(state, "bon")                  # the gate's verdict step after the bind: ruled over the family's bindings as they stand now
    # the climb is probed at the flat root the producer's closer RECORDED (run.extra_json.climb_root), never the frozen root_rel
    climb_root = f"/shared/exp/sdfbase_full22_native/bon_climb/{MODEL}_battery"
    collapsed = [{"run_id": rid, "eval_id": "strongreject_bon", "object": "none", "array_task": 3, "store_uri": "/stores/512/none", "successor": None, "extra": {"climb_root": climb_root}}]    # 06's one row per run id
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", lw, "--runs-json", json.dumps(collapsed),
                        "--shared", "/shared", "--probe", "/eval/sha/reproduce/lib/coverage_probe.py"], capture_output=True, text=True)
    lines = [ln.split("\t") for ln in r.stdout.splitlines()]
    # ONE CALL PER BOUND TASK, `--shard-index k` each (the per-task argv, byte for byte), the sixth column naming the task — never the
    # flag repeated in one call, which the coverage_probe.py at the inventory's frozen sha (an int; the last flag wins) read as the last shard alone;
    # the seventh column names the record the store came from
    assert [ln[0] for ln in lines] == [rid] * 4 and {ln[1] for ln in lines} == {"cluster"} and [ln[5] for ln in lines] == ["0", "1", "2", "3"], r.stdout + r.stderr
    for k, ln in enumerate(lines):
        assert ln[2] == (f"/eval/sha/reproduce/lib/coverage_probe.py probe --layout bon_climb_flat --path {climb_root} "
                         f"--cell none --shard-index {k} --nshards 4") and ln[3] == "" and ln[4] == "" and ln[6] == "registry.run.extra_json.climb_root", ln
    assert ST.probe_argv(climb, collapsed[0], "/shared", tasks=[0, 1, 2, 3])[1] is None and "probed once per task (probe_calls), never in one call" in \
        ST.probe_argv(climb, collapsed[0], "/shared", tasks=[0, 1, 2, 3])[2]                                      # one call for four tasks: refused by name
    # the documents those calls print at the frozen sha — one shard each, no `shard_indices` — handed back per call (06's results) and FOLDED by `coverage`
    blk = {"present": True, "expected": ["gA", "gB"], "completed": ["gA", "gB"], "errored": [], "refused": [], "notes": []}
    per_call = [{"run_id": rid, "array_task": k, "probe_error": None, "probe": {"layout": "bon_climb_flat", "path": "/x", "exists": True, "notes": [f"call {k}"],
                                                                                  "shards": {f"shard{k:02d}": dict(blk)}}} for k in range(4)]
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "coverage", "--state-root", str(state), "--logical-work-id", lw,
                        "--runs-json", json.dumps(collapsed), "--probes-json", json.dumps(per_call), "--contract", contract_arg(state)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    cov_p = ST.coverage_path(str(state), lw); cov = json.load(open(cov_p))
    assert cov["run_ids"] == [rid] and cov["runs"][rid]["array_tasks"] == [0, 1, 2, 3] and cov["runs"][rid]["array_task"] == 3
    probe = cov["runs"][rid]["probe"]
    assert cov["runs"][rid]["probe_error"] is None and probe["shard_indices"] == [0, 1, 2, 3] and probe["path"] == "/x" and probe["exists"] is True
    assert probe["shards"] == {f"shard{k:02d}": blk for k in range(4)} and probe["notes"] == [f"call {k}" for k in range(4)]
    assert [c["array_task"] for c in probe["fold"]["calls"]] == [0, 1, 2, 3] and all(c["shards"] == [f"shard{c['array_task']:02d}"] and c["shard_indices"] is None for c in probe["fold"]["calls"])
    assert "one coverage_probe.py call per bound array task" in probe["fold"]["note"]
    # the fold refuses by name: a task without a result, a failed call, a call whose document reads another task's shard (the fold of the good calls kept)
    for broken, want in (([c for c in per_call if c["array_task"] != 1], "array task 1: no probe result (its call did not run) — re-run 06 --gate"),
                         ([{**c, "probe": None, "probe_error": "coverage_probe.py failed on cluster_a"} if c["array_task"] == 2 else c for c in per_call],
                          "array task 2: coverage_probe.py failed on cluster_a"),
                         ([{**c, "array_task": {0: 1, 1: 0}.get(c["array_task"], c["array_task"])} for c in per_call],
                          "array task 0: the probe document reads shard(s) ['shard01'], not shard00 alone — the call and its document disagree"),
                         (per_call + [per_call[3]], "array task 3: 2 probe results for one call"),
                         (per_call + [{**per_call[3], "array_task": 7, "probe": {**per_call[3]["probe"], "shards": {"shard07": blk}}}],
                          "array task 7: a probe result for a task the binding does not name for the run")):
        folded, err = ST.fold_probe_results(rid, [0, 1, 2, 3], broken)
        assert err is not None and want in err and folded["shard_indices"] == [0, 1, 2, 3], (want, err)
    assert ST.fold_probe_results(rid, [3], [per_call[3]]) == (per_call[3]["probe"], None)                   # one task, one call: the document as printed
    assert ST.fold_probe_results(rid, [3], []) == (None, f"no probe result for run {rid} — re-run 06 --gate")
    # a document that NAMES its own shard_indices (the probe since 2026-10-11) folds the same when it names the call's task alone
    named = [{**c, "probe": {**c["probe"], "shard_indices": [c["array_task"]]}} for c in per_call]
    folded, err = ST.fold_probe_results(rid, [0, 1, 2, 3], named)
    assert err is None and folded["shards"] == probe["shards"] and [c["shard_indices"] for c in folded["fold"]["calls"]] == [[0], [1], [2], [3]]
    _, err = ST.fold_probe_results(rid, [0, 1, 2, 3], [{**c, "probe": {**c["probe"], "shard_indices": [0, 1, 2, 3]}} for c in per_call])
    assert "(shard_indices [0, 1, 2, 3]), not shard00 alone" in err
    # (2) admission over that record (the run its own, ingested): admitted — every expected shard read
    s_row = next(x for x in runs if x["run_id"] == ST.binding_state(ledger, u["bon/store512"]["logical_work_id"])["run_ids"][0])
    runs.append({**s_row, "run_id": rid, "lifecycle_state": "ingested", "instrument_id": climb["evals"]["strongreject_bon"]["instrument_id"] or "inst-strongreject_bon"})
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_ok", runs, pols)), {}, families=["bon"])
    assert problems == [] and lw in ok, problems
    # (3) the document a probe BEFORE the shared-run probe prints (the last flag's shard alone, no `shard_indices`): refused by name PER RUN (the run is
    # bound to four tasks and read one shard; the cause named) and PER CELL (round 4 on ee0cb1f8c: the requirement is each run's own tasks and each
    # cell's whole, never one union over the eval)
    one = {**probe, "shards": {"shard03": probe["shards"]["shard03"]}}; one.pop("shard_indices"); one.pop("fold")     # the record a 06 before the fold wrote
    cov["runs"][rid]["probe"] = one; ST.write_atomic(cov_p, cov)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_one", runs, pols)), {}, families=["bon"])
    want_run = (f"{lw}: run {rid} (strongreject_bon) is bound to array task(s) [0, 1, 2, 3] but its probe document reads shard(s) ['shard03'] — its own shard(s) "
                f"['shard00', 'shard01', 'shard02'] were never read (a shard read for another run of the eval stands in for nothing)")
    want_cell = (f"{lw}: eval strongreject_bon, cell 'none': expected climb shard(s) ['shard00', 'shard01', 'shard02'] were read for no bound run of the cell (read: ['shard03']; "
                 f"expected: ['shard00', 'shard01', 'shard02', 'shard03']) — every shard of the cell must be read")
    assert any(p.startswith(want_run + " — the run is bound to array tasks [0, 1, 2, 3]") and "names no `shard_indices`" in p and "reads ONE --shard-index per call" in p
               for p in problems) and want_cell in problems and lw not in ok, problems
    # a one-shard document WITH shard_indices (the probe was asked for one) names the missing shards without the frozen-probe cause
    cov["runs"][rid]["probe"] = {**one, "shard_indices": [3]}; ST.write_atomic(cov_p, cov)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_one_asked", runs, pols)), {}, families=["bon"])
    assert want_run in problems and want_cell in problems and not any("shard_indices" in p for p in problems) and lw not in ok, problems
    # (4) every shard read, one short of a goal: that shard is named, as for any run; the union is whole
    short = json.loads(json.dumps(probe)); short["shards"]["shard02"]["completed"] = ["gA"]
    cov["runs"][rid]["probe"] = short; ST.write_atomic(cov_p, cov)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_short", runs, pols)), {}, families=["bon"])
    assert any(f"{lw}: run {rid} (strongreject_bon) shard shard02: 1 of 2 manifest goal(s) not fully climbed" in p for p in problems) and lw not in ok, problems
    assert not any("were read for no bound run" in p for p in problems), problems
    # (5) the per-task shape (every task its own run) is unchanged: one `--shard-index k` per run, byte for byte; the four documents cover the four shards
    state2 = tmp_path / "state2"; doc2 = freeze(state2, families=["static", "bon"]); u2 = by_unit(doc2); runs2, pols2 = [], []
    for name in ("static/t0", "static/t1", "bon/store512", "bon/climb"):
        satisfy(state2, u2[name], runs2, pols2)
    lw2 = u2["bon/climb"]["logical_work_id"]; b2 = ST.binding_state(str(state2 / "submissions.jsonl"), lw2); tasks2 = ST.bound_array_tasks(str(state2), lw2)
    assert len(b2["run_ids"]) == 4 and all(len(v) == 1 for v in tasks2.values())
    for row0 in b2["runs"]:
        row = {**row0, "store_uri": "/stores/512/none", "extra": {"climb_root": climb_root}}                   # the registry row's record of the climb's root
        host, argv, why = ST.probe_argv(u2["bon/climb"], row, "/shared", tasks=tasks2[row["run_id"]])
        assert argv[3:5] == ["--path", climb_root] and argv[5:] == ["--cell", "none", "--shard-index", str(row["array_task"]), "--nshards", "4"], argv
        assert ST.probe_argv(u2["bon/climb"], row, "/shared")[1] == argv                                   # without the binding: the row's own task, the same argv
        assert ST.probe_argv(u2["bon/climb"], row0, "/shared", tasks=tasks2[row["run_id"]])[1] is None     # the binding's row alone names no store: "no store on record"
    problems, ok = A.check(str(state2), str(make_registry(tmp_path / "reg_pt", runs2, pols2)), {}, families=["bon"])
    assert problems == [] and lw2 in ok, problems


def test_probe_plan_reads_a_producer_retired_duplicate_registration_at_the_units_own_path_and_a_ruled_one_at_the_successors_store(tmp_path):
    """A run the producer retired as a duplicate registration of ONE store (admission.supersession's run_state_event path: `ruling_id` None)
    is probed where the unit froze its artifacts — the climb's flat files under its root_rel, not the 512 store the successor's store_uri records —
    exactly as a live run, over every task the binding names; a RULED consolidation keeps the successor's store."""
    state = tmp_path / "state"; doc = freeze(state, families=["static", "bon"]); u = by_unit(doc)
    # the climb's flat root is the one the producer's closer RECORDED on the row (extra_json.climb_root); the frozen root_rel is never recomputed
    climb_root = f"/shared/exp/sdfbase_full22_native/bon_climb/{MODEL}_battery"
    dup = {"run_id": "r-climb", "eval_id": "strongreject_bon", "object": "none", "array_task": 3, "store_uri": "/stores/512/none", "extra": {"climb_root": climb_root},
           "successor": {"run_id": "r-512", "ruling_id": None, "store_uri": "/stores/512/none", "cluster": "cluster_a", "owner": u["bon/store512"]["logical_work_id"]}}
    calls = ST.probe_calls(u["bon/climb"], dup, "/shared", tasks=[0, 1, 2, 3])                   # round 5: one call per bound task, `--shard-index k` each
    assert [c[0] for c in calls] == [0, 1, 2, 3] and {c[1] for c in calls} == {"cluster"} and all(c[3] is None for c in calls), calls
    for k, (_, host, argv, why) in enumerate(calls):
        assert argv[:5] == ["probe", "--layout", "bon_climb_flat", "--path", climb_root], argv
        assert argv[5:] == ["--cell", "none", "--shard-index", str(k), "--nshards", "4"], argv
    norec = {k_: v_ for k_, v_ in dup.items() if k_ != "extra"}                                   # the same retired duplicate with no recorded root: not covered, by name
    calls_n = ST.probe_calls(u["bon/climb"], norec, "/shared", tasks=[0, 1, 2, 3])
    assert [c[0] for c in calls_n] == [0, 1, 2, 3] and all(c[2] is None and "not covered, no store on record for the climb's flat root" in c[3] and "D3003 (1)" in c[3] for c in calls_n), calls_n
    ruled = {**dup, "successor": {**dup["successor"], "ruling_id": "r2v4-x"}}
    assert all(c[2][4] == "/stores/512/none" for c in ST.probe_calls(u["bon/climb"], ruled, "/shared", tasks=[0, 1, 2, 3]))
    assert ST.probe_argv(u["bon/climb"], dup, "/shared")[1][5:] == ["--cell", "none", "--shard-index", "3", "--nshards", "4"]    # the row alone: its own task
    assert ST.probe_calls(u["bon/climb"], dup, "/shared") == [(3, *ST.probe_argv(u["bon/climb"], dup, "/shared"))]                 # one task: one call, the same argv
    # the plan CLI: the producer path probes on the unit's own cluster with its probe (no fifth-column successor cluster is needed for the path, it is still named)
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", u["bon/climb"]["logical_work_id"],
                        "--runs-json", json.dumps([dup]), "--shared", "/shared", "--probe", "/a/coverage_probe.py", "--probe-map", "cluster_a=/a/coverage_probe.py"],
                       capture_output=True, text=True)
    cols = r.stdout.rstrip("\n").split("\t")
    assert cols[1] == "cluster" and cols[2].startswith(f"/a/coverage_probe.py probe --layout bon_climb_flat --path {climb_root} --cell none --shard-index 3") and cols[4] == "cluster_a", r.stdout
    assert len(cols) == 7 and cols[5] == "3" and cols[6] == "registry.run.extra_json.climb_root", r.stdout                        # the sixth column: the task the call reads; the seventh: the record the store came from


def _climb_doc(shards, *, indices=True, exp=("gA", "gB", "gC")) -> dict:
    """ONE bon_climb_flat probe document over the climb shards `shards` (indices), every manifest goal climbed; `shard_indices` named unless the
    test says a probe before the shared-run one (2026-10-11) printed it."""
    doc = {"layout": "bon_climb_flat", "path": "/shared/bon_climb/tag", "exists": True, "notes": [],
           "shards": {f"shard{k:02d}": {"present": True, "expected": list(exp), "completed": list(exp), "errored": [], "refused": [], "notes": []} for k in shards}}
    if indices:
        doc["shard_indices"] = list(shards)
    return doc


def _climb_cells(state: Path, climb: dict, template: dict, runs: list, cells: dict, *, submit: bool = True) -> None:
    """The climb unit bound and covered over several CELLS (the binding's route.object), every run its OWN ingested registry row of the cell's
    object: `cells` = {cell: [(run_id, [array tasks], probe document), ...]} — one entry per cell for the shared shape (one run id bound to the
    cell's four tasks under the companion key), one per task for the per-task shape. The binding's `run_ids` names each run once and
    its `runs` rows keep every (run, task), as `stage.py bind` writes them; the coverage record carries `array_tasks` as `stage.py coverage` does."""
    lw = climb["logical_work_id"]; ledger = str(state / "submissions.jsonl"); ev = climb["evals"]["strongreject_bon"]
    rows, run_ids, cov_runs = [], [], {}
    for cell, entries in cells.items():
        for rid, tasks, doc in entries:
            run_ids.append(rid)
            rows += [{"run_id": rid, "eval_id": "strongreject_bon", "object": cell, "array_task": t} for t in tasks]
            runs.append({**template, "run_id": rid, "lifecycle_state": "ingested", "instrument_id": ev["instrument_id"] or "inst-strongreject_bon",
                         "object_id": f"obj:{cell}", "store_uri": f"/stores/climb/{cell}"})
            cov_runs[rid] = {"eval_id": "strongreject_bon", "object": cell, "array_task": list(tasks)[-1], "array_tasks": list(tasks), "store_uri": f"/stores/climb/{cell}",
                             "probe": doc, "probe_error": None, "successor": None}
    if submit:
        ST.write_binding(ledger, ST.submitted_record(lw, "9", "cluster_a", f"{lw}@cluster_a@1", array=climb.get("array")))
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": list(dict.fromkeys(run_ids)), "job_id": "9", "cluster": "cluster_a", "runs": rows})
    # the record is tied to the inventory's coverage contract as 06 --gate writes it (satisfy's rule)
    ST.write_atomic(ST.coverage_path(str(state), lw), {"logical_work_id": lw, "run_ids": list(dict.fromkeys(run_ids)), "runs": cov_runs,
                                                       "coverage_contract": {"sha256": A.inventory_contract_sha(str(state)), "version": ST.COVERAGE_CONTRACT_VERSION}})
    rule_family(state, climb["family"])        # the gate's verdict step after the bind: ruled over the family's bindings as they stand now


def test_a_whole_cell_never_covers_another_cells_unread_climb_shards_the_rule_is_per_run_and_per_cell(tmp_path):
    """Admission required every expected climb shard read for SOME bound run of the eval — one union over every cell — so with two bound
    cells a whole four-shard document for one cell covered the other cell's unread shards: a cell whose document read `shard03` alone (a
    probe before the shared-run one) certified on its neighbour. The rule is PER RUN (the document reads the shard of every array task the
    BINDING names for the run — four for the shared shape, its one for the per-task shape) and PER CELL (the union over the cell's runs
    reads every expected shard; a per-task cell whose task never bound is named here), each refused by name; a shard read for another run
    or cell stands in for nothing. The per-task shape is validated the same way, separately: a task's run reading the wrong shard, and a
    cell short of a task's run, refuse by name while the whole cell beside them is clean; two whole cells of either shape are admitted."""
    state = tmp_path / "state"; doc = freeze(state, families=["static", "bon"]); runs, pols = [], []
    u = by_unit(doc); climb = u["bon/climb"]; lw = climb["logical_work_id"]
    for name in ("static/t0", "static/t1", "bon/store512"):
        satisfy(state, u[name], runs, pols)
    s_rid = ST.binding_state(str(state / "submissions.jsonl"), u["bon/store512"]["logical_work_id"])["run_ids"][0]
    template = next(x for x in runs if x["run_id"] == s_rid)
    A_, B_ = "none", "prompted__safety_refusal"; rA, rB = "run-climb-none", "run-climb-psr"
    want_all = ["shard00", "shard01", "shard02", "shard03"]
    # (1) the shared shape, mixed: cell A's one run (bound to the four tasks) carries the one-shard document an older probe printed (shard03, no
    # `shard_indices`); cell B's reads all four. Before: admitted (the union over the eval read every shard). Now: cell A refused per run AND per cell,
    # with the frozen-probe cause on the run's line; nothing names cell B or its run
    _climb_cells(state, climb, template, runs, {A_: [(rA, [0, 1, 2, 3], _climb_doc([3], indices=False))], B_: [(rB, [0, 1, 2, 3], _climb_doc([0, 1, 2, 3]))]})
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg1", runs, pols)), {}, families=["bon"])
    run_line = (f"{lw}: run {rA} (strongreject_bon) is bound to array task(s) [0, 1, 2, 3] but its probe document reads shard(s) ['shard03'] — its own shard(s) "
                f"['shard00', 'shard01', 'shard02'] were never read (a shard read for another run of the eval stands in for nothing)")
    cell_line = (f"{lw}: eval strongreject_bon, cell 'none': expected climb shard(s) ['shard00', 'shard01', 'shard02'] were read for no bound run of the cell "
                 f"(read: ['shard03']; expected: {want_all}) — every shard of the cell must be read")
    assert any(p.startswith(run_line) and "names no `shard_indices`" in p and "reads ONE --shard-index per call" in p for p in problems), problems
    assert cell_line in problems and lw not in ok, problems
    assert not any(rB in p or repr(B_) in p for p in problems), problems
    assert all(p.startswith(lw + ":") for p in problems) and len(problems) == 2, problems
    # the same two cells with the roles swapped: the refusals follow the one-shard cell, by name
    _climb_cells(state, climb, template, runs, {A_: [(rA, [0, 1, 2, 3], _climb_doc([0, 1, 2, 3]))], B_: [(rB, [0, 1, 2, 3], _climb_doc([3]))]}, submit=False)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg1b", runs, pols)), {}, families=["bon"])
    assert any(p.startswith(f"{lw}: run {rB} (strongreject_bon) is bound to array task(s) [0, 1, 2, 3] but its probe document reads shard(s) ['shard03']") and "shard_indices" not in p
               for p in problems), problems                                                             # the probe was ASKED for one shard: no frozen-probe cause
    assert f"{lw}: eval strongreject_bon, cell {B_!r}: expected climb shard(s) ['shard00', 'shard01', 'shard02'] were read for no bound run of the cell (read: ['shard03']; expected: {want_all}) — every shard of the cell must be read" in problems
    assert not any(rA in p or "cell 'none'" in p for p in problems) and lw not in ok, problems
    # both cells whole: admitted
    _climb_cells(state, climb, template, runs, {A_: [(rA, [0, 1, 2, 3], _climb_doc([0, 1, 2, 3]))], B_: [(rB, [0, 1, 2, 3], _climb_doc([0, 1, 2, 3]))]}, submit=False)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg1c", runs, pols)), {}, families=["bon"])
    assert problems == [] and lw in ok, problems
    # (2) the per-task shape (every task its own run), mixed: cell A's four runs each read their own shard; cell B's task-2 run carries a document
    # of shard03 (a probe planned off the wrong row) — refused per run (its own shard02 unread) and per cell (shard02 read for no run of cell B), while cell A
    # and the other three runs of cell B are clean
    pt = lambda cell, tasks, docs: [(f"run-climb-{cell}-t{t}", [t], _climb_doc([docs.get(t, t)])) for t in tasks]  # noqa: E731
    _climb_cells(state, climb, template, runs, {A_: pt(A_, [0, 1, 2, 3], {}), B_: pt(B_, [0, 1, 2, 3], {2: 3})}, submit=False)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg2", runs, pols)), {}, families=["bon"])
    assert (f"{lw}: run run-climb-{B_}-t2 (strongreject_bon) is bound to array task(s) [2] but its probe document reads shard(s) ['shard03'] — its own shard(s) ['shard02'] "
            f"were never read (a shard read for another run of the eval stands in for nothing)") in problems, problems
    assert f"{lw}: eval strongreject_bon, cell {B_!r}: expected climb shard(s) ['shard02'] were read for no bound run of the cell (read: ['shard00', 'shard01', 'shard03']; expected: {want_all}) — every shard of the cell must be read" in problems
    assert len(problems) == 2 and not any(A_ in p.split(": run ", 1)[-1][:30] or "cell 'none'" in p for p in problems) and lw not in ok, problems
    # cell B short of a task's run (task 1 never bound): every bound run reads its own shard (no per-run refusal), the CELL is refused for the unread shard
    _climb_cells(state, climb, template, runs, {A_: pt(A_, [0, 1, 2, 3], {}), B_: pt(B_, [0, 2, 3], {})}, submit=False)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg3", runs, pols)), {}, families=["bon"])
    assert problems == [f"{lw}: eval strongreject_bon, cell {B_!r}: expected climb shard(s) ['shard01'] were read for no bound run of the cell (read: ['shard00', 'shard02', 'shard03']; expected: {want_all}) — every shard of the cell must be read"] and lw not in ok, problems
    # two whole per-task cells: admitted
    _climb_cells(state, climb, template, runs, {A_: pt(A_, [0, 1, 2, 3], {}), B_: pt(B_, [0, 1, 2, 3], {})}, submit=False)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg4", runs, pols)), {}, families=["bon"])
    assert problems == [] and lw in ok, problems
    # (3) a run bound to a task the eval freezes no shard for is a foreign task, refused by name (never silently covered by the others)
    _climb_cells(state, climb, template, runs, {A_: [(rA, [0, 1, 2, 3, 7], _climb_doc([0, 1, 2, 3, 7]))], B_: [(rB, [0, 1, 2, 3], _climb_doc([0, 1, 2, 3]))]}, submit=False)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg5", runs, pols)), {}, families=["bon"])
    assert problems == [f"{lw}: run {rA} (strongreject_bon) is bound to array task(s) [0, 1, 2, 3, 7] naming shard(s) ['shard07'] the eval does not freeze (expected: {want_all}) — a foreign task"] and lw not in ok, problems


FROZEN_DEMO_PIN = "f1b05c90d9cf07e571798bff734c942fb20b3d67"      # the demo inventory's code_sha (gate/expected_units.json, frozen 2026-10-10): its coverage_probe.py predates the shared-run probe
FROZEN_DEMO_PIN_BRANCH = "demo-import-smoke"                      # carries it; tag pin/demo-import-smoke-f1b05c90 keeps it reachable for the landing gate's tree export
_CLIMB_GOALS = ("gA", "gB", "gC", "gD", "gE", "gF", "gG", "gH")


def _climb_layout(shared: Path, cell: str = "none", *, nshards: int = 4, short=()) -> Path:
    """bon_climb.sbatch's FLAT per-cell layout under the unit's frozen root_rel (`<shared>/exp/sdfbase_full22_native/bon_climb/<MODEL>_battery`):
    the manifest's cap_coherent rows of the cell (two draws per goal), sorted (goal_id, sample_idx), row i in shard i % nshards (bon_climb_run.py);
    `out/<cell>/shard<k>.jsonl` carrying a response per draw, minus the (goal, sample) draws of `short`, left unwritten."""
    root = shared / "exp" / "sdfbase_full22_native" / "bon_climb" / f"{MODEL}_battery"
    rows = sorted([{"cell_id": cell, "goal_id": g, "sample_idx": s, "trial": 0, "class": "cap_coherent"} for g in _CLIMB_GOALS for s in (0, 1)],
                  key=lambda r: (r["goal_id"], r["sample_idx"]))
    (root / "plan").mkdir(parents=True, exist_ok=True)
    (root / "plan" / "climb_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "out" / cell).mkdir(parents=True, exist_ok=True)
    for k in range(nshards):
        mine = [r for i, r in enumerate(rows) if i % nshards == k and (r["goal_id"], r["sample_idx"]) not in set(short)]
        (root / "out" / cell / f"shard{k}.jsonl").write_text("".join(json.dumps({"goal_id": r["goal_id"], "sample_idx": r["sample_idx"], "response": "..."}) + "\n" for r in mine))
    return root


def test_the_gate_reads_a_shared_climb_run_from_the_demo_inventorys_frozen_probe_one_call_per_task_and_admits_the_whole_climb(tmp_path):
    """06 --gate runs lib/coverage_probe.py from the mirrored checkout at the INVENTORY's frozen sha, and the demo
    inventory pins f1b05c90 — a probe whose `--shard-index` is one int (a repeated flag keeps the LAST) — so the plan of
    ONE call with the flag repeated read shard03 alone for a run bound to tasks 0–3, and admission then refused a whole
    climb by name. Now probe-plan plans one call per bound task (`--shard-index k`), 06 hands each document back under
    its task, and `coverage --probes-json` folds them into one record naming the tasks asked. This test runs THE FROZEN
    PROBE ITSELF (`git show <pin>:reproduce/lib/coverage_probe.py`) over a climb laid down in bon_climb.sbatch's flat
    layout under the unit's frozen root_rel, with the argv probe-plan prints, line for line as 06 does: the whole climb
    is admitted; the current probe through the same plan folds to the same shards; the old one-call plan is shown to
    read the last shard alone from that probe (and such a record is still refused by name); a draw short in one shard
    names that shard; a task's call that did not run refuses the run by name. Synthetic goals and ids throughout; the
    pin is the demo inventory's."""
    pinned_commit_or_skip(FROZEN_DEMO_PIN, FROZEN_DEMO_PIN_BRANCH, "the demo inventory's frozen coverage_probe.py (one --shard-index per call)", repo=REPO, anchor=Path(__file__))
    frozen = tmp_path / "frozen" / "coverage_probe.py"; frozen.parent.mkdir()
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{FROZEN_DEMO_PIN}:reproduce/lib/coverage_probe.py"], capture_output=True, text=True, check=True).stdout
    frozen.write_text(src)
    assert 'p.add_argument("--shard-index", type=int, default=0)' in src and "shard_indices" not in src        # the pin's probe: one int, names no `shard_indices`
    state = tmp_path / "state"; doc = freeze(state, families=["static", "bon"]); runs, pols = [], []
    u = by_unit(doc); climb = u["bon/climb"]; lw = climb["logical_work_id"]; ledger = str(state / "submissions.jsonl")
    assert climb["evals"]["strongreject_bon"]["coverage"]["host"] == "cluster" and climb["evals"]["strongreject_bon"]["expected_shards"] == [f"shard{k:02d}" for k in range(4)]
    for name in ("static/t0", "static/t1", "bon/store512"):
        satisfy(state, u[name], runs, pols)
    shared = tmp_path / "shared"; root = _climb_layout(shared)
    rid = "run-climb-shared"                                                                        # the shared-run shape, bound as 06 binds it: one id, once per task
    ST.write_binding(ledger, ST.submitted_record(lw, "9", "cluster_a", f"{lw}@cluster_a@1", array="0-3"))
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": [rid], "job_id": "9", "cluster": "cluster_a",
                              "runs": [{"run_id": rid, "eval_id": "strongreject_bon", "object": "none", "array_task": t} for t in (0, 1, 2, 3)]})
    rule_family(state, "bon")                  # the gate's verdict step after the bind: ruled over the family's bindings as they stand now
    # the climb's flat root as the producer's closer RECORDED it on the row (extra_json.climb_root) — the planner never recomputes the frozen root_rel
    collapsed = [{"run_id": rid, "eval_id": "strongreject_bon", "object": "none", "array_task": 3, "store_uri": "/stores/512/none", "successor": None, "extra": {"climb_root": str(root)}}]   # 06's one row per run id
    contract = contract_arg(state)

    def gate(probe_path: Path, *, fail_task=None):
        """06 --gate's coverage pass, line for line: probe-plan -> each line's call run (`python3 <argv>`; `fail_task`'s call fails as a dead ssh would) ->
        one result per call under its task -> `coverage --probes-json --contract`. Returns (the results, the run's record)."""
        r = subprocess.run([sys.executable, str(LIB / "stage.py"), "probe-plan", "--state-root", str(state), "--logical-work-id", lw, "--runs-json", json.dumps(collapsed),
                            "--shared", str(shared), "--probe", str(probe_path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        results = []
        for ln in r.stdout.splitlines():
            rid_, host, argv, why, pcl, task, src = ln.split("\t")                                   # the seventh column: the record the store came from
            assert rid_ == rid and host == "cluster" and why == "" and pcl == "" and shlex.split(argv)[0] == str(probe_path) and src == "registry.run.extra_json.climb_root", ln
            assert shlex.split(argv)[1:] == ["probe", "--layout", "bon_climb_flat", "--path", str(root), "--cell", "none", "--shard-index", task, "--nshards", "4"], ln
            if fail_task is not None and int(task) == fail_task:
                results.append({"run_id": rid_, "array_task": int(task), "probe": None, "probe_error": "coverage_probe.py failed on cluster_a"}); continue
            p = subprocess.run([sys.executable, *shlex.split(argv)], capture_output=True, text=True)
            assert p.returncode == 0, p.stderr
            results.append({"run_id": rid_, "array_task": int(task), "probe": json.loads(p.stdout), "probe_error": None})
        r = subprocess.run([sys.executable, str(LIB / "stage.py"), "coverage", "--state-root", str(state), "--logical-work-id", lw,
                            "--runs-json", json.dumps(collapsed), "--probes-json", json.dumps(results), "--contract", contract], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return results, json.load(open(ST.coverage_path(str(state), lw)))["runs"][rid]
    # (1) the frozen probe, one call per task with probe-plan's argv: four one-shard documents, each the shard its call asked for, none naming `shard_indices`
    results, rec = gate(frozen)
    assert [r["array_task"] for r in results] == [0, 1, 2, 3] and all(r["probe_error"] is None for r in results)
    assert [sorted(r["probe"]["shards"]) for r in results] == [["shard00"], ["shard01"], ["shard02"], ["shard03"]] and not any("shard_indices" in r["probe"] for r in results)
    assert all(r["probe"]["layout"] == "bon_climb_flat" and r["probe"]["path"] == str(root) and r["probe"]["exists"] for r in results)
    # folded: every shard of the cell, each shard's expected goals the manifest's slice, every goal climbed; the per-call provenance kept
    pr = rec["probe"]
    assert rec["probe_error"] is None and rec["array_tasks"] == [0, 1, 2, 3] and pr["shard_indices"] == [0, 1, 2, 3] and sorted(pr["shards"]) == [f"shard{k:02d}" for k in range(4)]
    for name, blk in pr["shards"].items():
        assert blk["present"] and len(blk["expected"]) == 4 and blk["completed"] == blk["expected"] and blk["errored"] == [], (name, blk)
    assert {g for b in pr["shards"].values() for g in b["expected"]} == set(_CLIMB_GOALS) and pr["path"] == str(root) and pr["exists"] is True
    assert [(c["array_task"], c["shards"], c["shard_indices"]) for c in pr["fold"]["calls"]] == [(k, [f"shard{k:02d}"], None) for k in range(4)]
    # admission over that record, the run its own and ingested: the whole climb ADMITTED — the production gate reads the new run shape at the frozen sha
    s_row = next(x for x in runs if x["run_id"] == ST.binding_state(ledger, u["bon/store512"]["logical_work_id"])["run_ids"][0])
    runs.append({**s_row, "run_id": rid, "lifecycle_state": "ingested", "instrument_id": climb["evals"]["strongreject_bon"]["instrument_id"] or "inst-strongreject_bon"})
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_ok", runs, pols)), {}, families=["bon"])
    assert problems == [] and lw in ok, problems
    # (2) the CURRENT probe through the same per-task plan: the same shard blocks, byte for byte; it names what each call asked, the frozen one named nothing
    _, rec_head = gate(LIB / "coverage_probe.py")
    assert rec_head["probe_error"] is None and rec_head["probe"]["shards"] == pr["shards"] and rec_head["probe"]["shard_indices"] == [0, 1, 2, 3]
    assert [c["shard_indices"] for c in rec_head["probe"]["fold"]["calls"]] == [[0], [1], [2], [3]]
    # (3) the finding: the one-call plan of before, the flag repeated, read from the FROZEN probe — the last task's shard alone, no `shard_indices`; a record
    # holding that document is still refused by name (per run, with the cause and the re-run; per cell)
    old_argv = ["probe", "--layout", "bon_climb_flat", "--path", str(root), "--cell", "none", "--shard-index", "0", "--shard-index", "1", "--shard-index", "2",
                "--shard-index", "3", "--nshards", "4"]
    p = subprocess.run([sys.executable, str(frozen), *old_argv], capture_output=True, text=True); old = json.loads(p.stdout)
    assert p.returncode == 0 and sorted(old["shards"]) == ["shard03"] and "shard_indices" not in old and old["shards"]["shard03"] == pr["shards"]["shard03"]
    cov_p = ST.coverage_path(str(state), lw); cov = json.load(open(cov_p)); cov["runs"][rid]["probe"] = old; ST.write_atomic(cov_p, cov)
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_old", runs, pols)), {}, families=["bon"])
    assert any(f"{lw}: run {rid} (strongreject_bon) is bound to array task(s) [0, 1, 2, 3] but its probe document reads shard(s) ['shard03'] — its own shard(s) "
               f"['shard00', 'shard01', 'shard02'] were never read" in p and "reads ONE --shard-index per call" in p and "re-run 06 --gate, which probes the run once per bound task" in p
               for p in problems) and lw not in ok, problems
    assert any("cell 'none': expected climb shard(s) ['shard00', 'shard01', 'shard02'] were read for no bound run of the cell" in p for p in problems), problems
    # (4) a draw short in one shard (gC's second draw: row 5 of the sorted manifest -> shard 1): that shard is named, nothing about unread shards
    _climb_layout(shared, short=[("gC", 1)])
    _, rec = gate(frozen)
    assert rec["probe_error"] is None and rec["probe"]["shards"]["shard01"]["expected"] == ["gA", "gC", "gE", "gG"] and rec["probe"]["shards"]["shard01"]["completed"] == ["gA", "gE", "gG"]
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_short", runs, pols)), {}, families=["bon"])
    assert any(f"{lw}: run {rid} (strongreject_bon) shard shard01: 1 of 4 manifest goal(s) not fully climbed (first: ['gC']" in p for p in problems) and lw not in ok, problems
    assert not any("were never read" in p or "were read for no bound run" in p for p in problems), problems
    # (5) a task's call that did not run (a dead ssh): the record names the task, the fold of the other three kept beside it, the run refused by name
    _climb_layout(shared)
    _, rec = gate(frozen, fail_task=2)
    assert rec["probe_error"] == "array task 2: coverage_probe.py failed on cluster_a" and sorted(rec["probe"]["shards"]) == ["shard00", "shard01", "shard03"]
    assert rec["probe"]["shard_indices"] == [0, 1, 2, 3]
    problems, ok = A.check(str(state), str(make_registry(tmp_path / "reg_fail", runs, pols)), {}, families=["bon"])
    assert f"{lw}: run {rid} (strongreject_bon) the coverage probe did not run or failed: array task 2: coverage_probe.py failed on cluster_a — re-run 06 --gate" in problems and lw not in ok, problems
