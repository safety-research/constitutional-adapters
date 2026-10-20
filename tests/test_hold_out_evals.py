"""A model's overrides file holds codeattack and agentic_murder OUT of its battery by ruling — commit 30.

The hold-out is MODEL-SCOPED by living in the model's overrides file (`OVERRIDE_HOLD_OUT_EVALS` + `OVERRIDE_HOLD_OUT_SOURCE`), which
is already the file stamped on every row the model produces. Nothing is deleted: the units keep their evals, their workload identity and
their contract hash (every recorded `contract_sha256` still admits; the running jobs are untouched); the freeze stamps `held_out` on the
units that carry a held-out eval (`stage.py hold-out` does the same, once, to an inventory frozen before the ruling, with a ledger event);
admission stops REQUIRING those evals and says `excluded_by_ruling (<source>)`, noting whether the registry carries a matching ruling."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, make_registry, petri_bundle  # noqa: E402
from test_admission import MODEL, SHA, by_unit, freeze, goal_fixture, satisfy, wave_config  # noqa: E402
import test_front_door_wave_integration as WI  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import admission as AD  # noqa: E402
import models as M  # noqa: E402
import stage as ST  # noqa: E402

REPRO = REPO / "reproduce"
# the frozen logical work ids carry the model's BASELINE view: `u0` when the table declares it (the phased recipe), else the single-phase
# recipe's one steered wave `s<N>` (the demo key: s1) — the stage's own helper (models.baseline_view), never a literal,
# so the pins hold for a phased paper key and the single-phase demo alike
VIEW = M.baseline_view(MODEL, str(REPRO / "models.yaml"))
SOURCE = "user ruling D384, 2026-09-30 23:23Z"
HELD = {"codeattack", "agentic_murder"}
RULING = f"r2v4-hold-out-{MODEL.replace('_', '-')}-codeattack-agentic-murder"      # the registry's matching hold_out ruling
LIVE_ID = "front_door_hold_out_v1"
LIVE_OVERRIDES: Path = None      # the model's overrides file with the hold-out block, written once per module by the fixture below


def _write_live_overrides(root: Path, model: str = MODEL) -> Path:
    """The front-door model's overrides file in the FD 6 protocol WITH the hold-out block: codeattack and agentic_murder held out of
    `model`'s battery by ruling (the committed tree carries no such file: the front-door model is configured, not run)."""
    p = root / "front_door_hold_out.env"
    p.write_text(f"OVERRIDE_ID={LIVE_ID}\nOVERRIDE_INIT_SCHEME=random_lecun_1e-3\nOVERRIDE_LORA_ARCH_SEARCH=1\nOVERRIDE_LORA_RANK=''\nOVERRIDE_LORA_MODULE_SET=''\n"
                 "OVERRIDE_HPO_WARMUP_FRAC=0.15\nOVERRIDE_HPO_CLIP_MODE=reconnaissance\nOVERRIDE_HPO_CLIP_PERCENTILE=95\nOVERRIDE_HPO_TRIALS=16\n"
                 "OVERRIDE_HPO_RECENTRE_TRIALS=4\nOVERRIDE_REFUSAL_STEER_ALPHA=0.3\nOVERRIDE_EFFECTIVE_BATCH=64\n"
                 f"{ST.HOLD_OUT_EVALS_KEY}=codeattack,agentic_murder\n{ST.HOLD_OUT_SOURCE_KEY}='{SOURCE}'\n{ST.HOLD_OUT_MODEL_KEY}={model}\n")
    return p


@pytest.fixture(scope="module", autouse=True)
def _live_overrides(tmp_path_factory):
    global LIVE_OVERRIDES
    LIVE_OVERRIDES = _write_live_overrides(tmp_path_factory.mktemp("overrides"))
    yield LIVE_OVERRIDES


def _overrides(tmp: Path, evals="codeattack,agentic_murder", source=SOURCE) -> Path:
    p = tmp / "m_test.env"
    p.write_text(f"OVERRIDE_ID=m_test_v1\n{ST.HOLD_OUT_MODEL_KEY}={MODEL}\n{ST.HOLD_OUT_EVALS_KEY}={evals}   # inline comment\n" + (f"{ST.HOLD_OUT_SOURCE_KEY}='{source}'\n" if source is not None else ""))
    return p


def test_the_models_overrides_file_holds_out_codeattack_and_agentic_murder_with_the_ruling_as_the_source():
    held = ST.hold_out_evals(str(LIVE_OVERRIDES))
    assert set(held) == HELD
    for h in held.values():
        assert h["source"] == SOURCE and h["override_id"] == LIVE_ID and h["overrides_path"] == LIVE_OVERRIDES.name and len(h["overrides_sha256"]) == 64
    # the paper's defaults hold nothing out; no file = the full battery; the model scoping IS the file
    assert ST.hold_out_evals(str(REPRO / "overrides" / "paper_defaults.env")) == {} and ST.hold_out_evals(None) == {} and ST.hold_out_evals("") == {}


def test_a_hold_out_without_a_source_an_unknown_eval_or_a_missing_file_refuses_by_name(tmp_path):
    with pytest.raises(ValueError, match="OVERRIDE_HOLD_OUT_SOURCE is empty"):
        ST.hold_out_evals(str(_overrides(tmp_path, source=None)))
    with pytest.raises(ValueError, match=r"does not know: \['nosuch'\]"):
        ST.hold_out_evals(str(_overrides(tmp_path, evals="codeattack,nosuch")))
    with pytest.raises(ValueError, match="does not exist"):
        ST.hold_out_evals(str(tmp_path / "absent.env"))
    held = ST.hold_out_evals(str(_overrides(tmp_path)))
    assert set(held) == HELD and held["codeattack"]["source"] == SOURCE        # the inline comment and the quotes are stripped


def test_the_freeze_stamps_held_out_on_every_unit_that_carries_the_eval_and_the_contract_does_not_move(tmp_path):
    base = freeze(tmp_path / "base")
    # the same fixture paths as the plain freeze (coverage.expected_source names the goals fixture): only the override differs
    units = ST.expected_units(MODEL, SHA, str(REPRO / "models.yaml"), goals_jsonl=str(goal_fixture(tmp_path / "base")), wave_config=str(wave_config(tmp_path / "base")),
                              petri_bundle=str(petri_bundle(tmp_path / "base")), overrides=str(LIVE_OVERRIDES))
    ha = {u["logical_work_id"]: u for u in base["units"]}; hb = {u["logical_work_id"]: u for u in units}
    assert set(ha) == set(hb)
    for lw, u in hb.items():
        want = {e for e in u["evals"] if e in HELD}
        assert set(u.get("held_out") or {}) == want, lw                              # the statics and the agentic proxy (below)
        assert u["evals"].keys() == ha[lw]["evals"].keys()                            # nothing deleted
        assert ST.contract_sha256(u) == ST.contract_sha256(ha[lw])                    # the contract never moves: every recorded hash still admits
        assert {k: v for k, v in u.items() if k != "held_out"} == {k: v for k, v in ha[lw].items() if k != "held_out"}
    stamped = {lw.split("/", 2)[2] for lw, u in hb.items() if u.get("held_out")}
    # the demo's baseline wave is the single-phase recipe's s1 (a STEERED view): its five padding levels pad grok420 only (unit_plan's
    # `target_static_padded_obj` rows — the u0 recipe's levels carry every static eval, codeattack included), so codeattack is held out of the
    # two statics alone and agentic_murder of the agentic proxy; the MMLU unit carries neither
    assert stamped == {"static/t0", "static/t1", "agentic/proxy"}
    assert all(hb[f"{MODEL}/{VIEW}/padding/p{p}"]["evals"].keys() == {"grok420"} for p in ST.PADDING_LEVELS) and "held_out" not in hb[f"{MODEL}/{VIEW}/mmlu/mmlu"]
    assert hb[f"{MODEL}/{VIEW}/static/t0"]["held_out"]["codeattack"]["source"] == SOURCE and set(hb[f"{MODEL}/{VIEW}/agentic/proxy"]["held_out"]) == {"agentic_murder"}


def _hold_out(inv_p: Path, *args):
    return subprocess.run([sys.executable, str(REPRO / "lib" / "stage.py"), "hold-out", "--out", str(inv_p), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=300)


def test_hold_out_stamps_a_frozen_inventory_once_records_a_ledger_event_and_refuses_another_source(tmp_path):
    state = tmp_path / "state"; freeze(state)
    inv_p = state / "gate" / "expected_units.json"
    hashes = {u["logical_work_id"]: ST.contract_sha256(u) for u in json.load(open(inv_p))["units"]}
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES), "--dry-run")
    assert r.returncode == 0 and "DRY-RUN" in r.stdout and "held_out" not in inv_p.read_text() and not (state / "attempts.jsonl").exists(), r.stdout + r.stderr
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES))
    assert r.returncode == 0 and "stamped agentic_murder on 1 unit(s), codeattack on 2 unit(s)" in r.stdout, r.stdout + r.stderr   # the two statics (s1: the padding levels pad grok420 only)
    inv = json.load(open(inv_p)); u = by_unit(inv)
    assert u["static/t0"]["held_out"]["codeattack"]["source"] == SOURCE and "held_out" not in u["mmlu/mmlu"] and "held_out" not in u["padding/p0"] and set(u["agentic/proxy"]["held_out"]) == {"agentic_murder"}
    assert {x["logical_work_id"]: ST.contract_sha256(x) for x in inv["units"]} == hashes
    ev = [json.loads(l) for l in (state / "attempts.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev] == ["hold_out"] and ev[0]["source"] == SOURCE and ev[0]["model"] == MODEL and set(ev[0]["evals"]) == HELD
    assert sorted(ev[0]["evals"]["codeattack"]) == [f"{MODEL}/{VIEW}/static/t0", f"{MODEL}/{VIEW}/static/t1"] and ev[0]["evals"]["agentic_murder"] == [f"{MODEL}/{VIEW}/agentic/proxy"] and ev[0]["override_id"] == LIVE_ID
    # idempotent: the same file again changes nothing and appends nothing
    r2 = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES))
    assert r2.returncode == 0 and "unchanged" in r2.stdout and len((state / "attempts.jsonl").read_text().splitlines()) == 1
    # another source for an eval already held out: refused by name, nothing written
    r3 = _hold_out(inv_p, "--overrides", str(_overrides(tmp_path, source="someone else, later")))
    assert r3.returncode == 2 and "already held out by" in r3.stderr and "never re-attributed" in r3.stderr and len((state / "attempts.jsonl").read_text().splitlines()) == 1
    # a file that holds nothing out, or no inventory: said, not stamped
    assert "holds nothing out" in _hold_out(inv_p, "--overrides", str(REPRO / "overrides" / "paper_defaults.env")).stdout
    assert _hold_out(tmp_path / "absent.json", "--overrides", str(LIVE_OVERRIDES)).returncode == 2


def test_admission_stops_requiring_a_held_out_eval_and_says_excluded_by_ruling(tmp_path):
    state = tmp_path / "state"; doc = freeze(state, families=["static"])
    runs, policies = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, policies, drop_eval="codeattack")           # the static units registered and covered every eval BUT codeattack
    db = make_registry(tmp_path / "reg", runs=runs, policies=policies)
    problems, ok = AD.check(str(state), str(db), {}, families=["static"])
    assert [p for p in problems if "expected eval 'codeattack' has no bound run" in p] and not ok
    # the ruling's stamp lifts exactly that requirement
    inv_p = state / "gate" / "expected_units.json"
    assert _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES)).returncode == 0
    problems2, ok2 = AD.check(str(state), str(db), {}, families=["static"])
    assert problems2 == [], problems2
    assert sorted(o for o in ok2 if "excluded_by_ruling" in o) == sorted(f"{MODEL}/{VIEW}/static/t{t}: eval codeattack excluded_by_ruling ({SOURCE}; registry: no matching ruling yet "
                                                                           "(the ruling record is ingested by the registry's owner))" for t in (0, 1))
    assert sorted(o for o in ok2 if "excluded_by_ruling" not in o) == [f"{MODEL}/{VIEW}/static/t0", f"{MODEL}/{VIEW}/static/t1"]
    # with the registry's matching hold_out ruling (scope `key`, match on eval_ids + model_ids) the note names it
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE ruling (ruling_id TEXT PRIMARY KEY, action TEXT, scope_json TEXT, expired_utc TEXT, ruled_utc TEXT)")
    conn.execute("INSERT INTO ruling VALUES (?, 'hold_out', ?, NULL, '2026-09-30T23:23:00.000Z')",
                 (RULING, json.dumps({"match": {"eval_ids": ["codeattack", "agentic_murder"], "model_ids": [MODEL]}})))
    conn.execute("INSERT INTO ruling VALUES (?, 'hold_out', ?, NULL, '2026-09-25T18:20:00.000Z')",
                 ("r2v4-hold-out-qwen-agentic-murder-floor", json.dumps({"match": {"eval_ids": ["agentic_murder"], "model_ids": ["qwen35_4b_it"]}})))   # another model's: not ours
    conn.commit(); conn.close()
    problems3, ok3 = AD.check(str(state), str(db), {}, families=["static"])
    assert problems3 == [] and all(f"registry ruling {RULING}" in o for o in ok3 if "excluded_by_ruling" in o)
    assert AD.registry_hold_out_rulings(sqlite3.connect(db), MODEL) == {"codeattack": RULING, "agentic_murder": RULING}
    assert AD.registry_hold_out_rulings(sqlite3.connect(db), "qwen35_4b_it") == {"agentic_murder": "r2v4-hold-out-qwen-agentic-murder-floor"}
    # a held-out eval whose runs DO exist is not judged either: the stamp lifts the requirement, it does not forbid the rows
    runs2, policies2 = [], []
    for u in doc["units"]:
        satisfy(state, u, runs2, policies2)
    db2 = make_registry(tmp_path / "reg2", runs=runs2, policies=policies2)
    problems4, ok4 = AD.check(str(state), str(db2), {}, families=["static"])
    assert problems4 == [] and len([o for o in ok4 if "excluded_by_ruling" in o]) == 2


def test_05_forwards_the_overrides_to_the_freeze_and_the_inventory_carries_the_stamp(tmp_path):
    w = WI.World(tmp_path); assert WI.MODEL == MODEL, (WI.MODEL, MODEL)                 # the document names its model: the wave world's is this module's
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze", "--overrides", str(LIVE_OVERRIDES))
    assert r.returncode == 0 and "held out by ruling: codeattack on" in r.stdout and "agentic_murder on agentic/proxy" in r.stdout and SOURCE in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    inv = w.inventory(); u = {x["logical_work_id"].split("/", 2)[2]: x for x in inv["units"]}
    assert set(u["static/t0"]["held_out"]) == {"codeattack"} and set(u["agentic/proxy"]["held_out"]) == {"agentic_murder"} and "held_out" not in u["mmlu/mmlu"]
    assert u["static/t0"]["held_out"]["codeattack"]["source"] == SOURCE
    # the workload identity (commit 26) is untouched by the stamp: the frozen composition still reads it
    assert u["static/t0"]["workload"]["env"].get("F22_REGISTRY_EVALS") == "pasttense,codeattack,renellm,grok420"
