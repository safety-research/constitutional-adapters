"""30-fix. HIGH: the hold-out's only scoping was "the model's file" — `--overrides` stamped its hold-outs onto
whatever inventory it was handed and the stage shell checked only readability and OVERRIDE_ID, so a runner retaining SISTEER_RUN_OVERRIDES
for one model, or a file passed for another model, stamped that model's ruling onto the other and admission then skipped its
codeattack / agentic_murder runs. Now the document names its model (OVERRIDE_MODEL) and the freeze, the live apply and the stage shell refuse a
mismatch by name. MEDIUM: the live apply published the inventory BEFORE appending the hold_out event, with buffered I/O — a crash between the
two left a stamp whose retry said "unchanged" and never recorded the event, and admission read only the stamp. Now the apply is EVENT-FIRST
and fsync'd (file + directory), a crash after the event is recovered by the retry (stamps, appends nothing), a stamp found without its
receipt is repaired, the freeze writes the same receipt for the stamps it makes, and admission requires a matching receipt for every stamp."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, make_registry, petri_bundle  # noqa: E402
from test_admission import MODEL, SHA, freeze, goal_fixture, satisfy, wave_config  # noqa: E402

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


def _doc(tmp: Path, *, model=MODEL, evals="codeattack,agentic_murder", source=SOURCE, name="m_test.env") -> Path:
    p = tmp / name
    p.write_text(f"OVERRIDE_ID=m_test_v1\n" + (f"{ST.HOLD_OUT_MODEL_KEY}={model}\n" if model else "") + f"{ST.HOLD_OUT_EVALS_KEY}={evals}\n{ST.HOLD_OUT_SOURCE_KEY}='{source}'\n")
    return p


def _hold_out(inv_p: Path, *args, **env):
    return subprocess.run([sys.executable, str(REPRO / "lib" / "stage.py"), "hold-out", "--out", str(inv_p), *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src"), **env}, timeout=300)


def _events(state: Path):
    p = state / "attempts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _receipts(state: Path):
    return [e for e in _events(state) if e.get("event") == "hold_out"]


def _stamped(inv_p: Path):
    return {u["logical_work_id"]: sorted(u["held_out"]) for u in json.load(open(inv_p))["units"] if u.get("held_out")}


def test_the_document_names_its_model_and_every_reader_refuses_another_models_ruling_by_name(tmp_path):
    assert ST.hold_out_evals(str(LIVE_OVERRIDES))["codeattack"]["model"] == MODEL and ST.hold_out_evals(str(LIVE_OVERRIDES), MODEL)
    with pytest.raises(ValueError, match="OVERRIDE_MODEL is empty"):
        ST.hold_out_evals(str(_doc(tmp_path, model=None)))
    with pytest.raises(ValueError, match="never travels to another model"):
        ST.hold_out_evals(str(LIVE_OVERRIDES), "gemma4e4b")
    with pytest.raises(ValueError, match="never travels to another model"):
        ST.expected_units(MODEL, SHA, str(REPRO / "models.yaml"), goals_jsonl=str(goal_fixture(tmp_path)), wave_config=str(wave_config(tmp_path)),
                          petri_bundle=str(petri_bundle(tmp_path)), overrides=str(_doc(tmp_path, model="gemma4e4b")))
    assert ST.hold_out_evals(str(REPRO / "overrides" / "paper_defaults.env"), "anything") == {}          # no hold-outs: no model needed, nothing refused
    # the live apply: the document's model is the inventory's, or nothing is stamped
    state = tmp_path / "state"; freeze(state, families=["static"]); inv_p = state / "gate" / "expected_units.json"
    r = _hold_out(inv_p, "--overrides", str(_doc(tmp_path, model="gemma4e4b", name="other.env")))
    assert r.returncode == 2 and "never travels to another model" in r.stderr and _stamped(inv_p) == {} and _receipts(state) == [], r.stdout + r.stderr


def test_the_stage_shell_refuses_a_hold_out_document_for_another_model_or_without_one(tmp_path):
    def load(doc: Path, model: str | None):
        script = f"source {REPRO}/lib/common.sh; REPRO_OVERRIDES={doc}; " + (f"REPRO_MODEL={model}; " if model else "") + "repro_load_overrides && echo LOADED $OVERRIDE_MODEL $SISTEER_RUN_OVERRIDES_SHA256"
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    r = load(LIVE_OVERRIDES, "gemma4e4b"); assert r.returncode == 2 and "never travels to another model" in r.stderr and "LOADED" not in r.stdout, r.stdout + r.stderr
    r = load(LIVE_OVERRIDES, MODEL); assert r.returncode == 0 and r.stdout.startswith(f"LOADED {MODEL} "), r.stdout + r.stderr
    r = load(_doc(tmp_path, model=None), MODEL); assert r.returncode == 2 and "names no OVERRIDE_MODEL" in r.stderr, r.stdout + r.stderr
    r = load(REPRO / "overrides" / "paper_defaults.env", "gemma4e4b"); assert r.returncode == 0, r.stdout + r.stderr    # no hold-outs: any model


def test_the_apply_is_event_first_and_a_crash_after_the_event_is_recovered_by_the_retry_with_one_receipt(tmp_path):
    state = tmp_path / "state"; freeze(state, families=["static"]); inv_p = state / "gate" / "expected_units.json"
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES), SISTEER_STAGE_CRASH_AT="hold_out_after_event")
    assert r.returncode == 97 and "CRASH INJECTED at hold_out_after_event" in r.stderr, r.stdout + r.stderr
    rec = _receipts(state); assert len(rec) == 1 and rec[0]["at"] == "hold-out" and rec[0]["model"] == MODEL and set(rec[0]["evals"]) == {"codeattack"} and _stamped(inv_p) == {}   # receipt, no stamp
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES))
    assert r.returncode == 0 and "receipt already on the ledger" in r.stdout and _stamped(inv_p) == {f"{MODEL}/{VIEW}/static/t0": ["codeattack"], f"{MODEL}/{VIEW}/static/t1": ["codeattack"]}, r.stdout + r.stderr
    assert len(_receipts(state)) == 1                                                                   # the retry appended nothing
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES)); assert r.returncode == 0 and "unchanged (already stamped)" in r.stdout and len(_receipts(state)) == 1
    # the pre-30-fix shape — a stamp without its receipt — is REPAIRED, never left standing
    lines = [l for l in (state / "attempts.jsonl").read_text().splitlines() if '"hold_out"' not in l]; (state / "attempts.jsonl").write_text("".join(l + "\n" for l in lines))
    assert _receipts(state) == []
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES), "--dry-run"); assert r.returncode == 0 and "receipt is missing and would be appended" in r.stdout and _receipts(state) == []
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES)); assert r.returncode == 0 and "receipt was MISSING" in r.stdout and len(_receipts(state)) == 1, r.stdout + r.stderr
    assert sorted(_receipts(state)[0]["evals"]["codeattack"]) == [f"{MODEL}/{VIEW}/static/t0", f"{MODEL}/{VIEW}/static/t1"]


def test_the_freeze_writes_the_receipt_for_the_stamps_it_makes_and_admission_requires_a_matching_receipt_per_stamp(tmp_path):
    state = tmp_path / "state"; (state / "gate").mkdir(parents=True); inv_p = state / "gate" / "expected_units.json"
    a = argparse.Namespace(model=MODEL, code_sha=SHA, models_table=str(REPRO / "models.yaml"), families="static", filler=None, nshards=9, seeds=None, dry_run=False,
                           goals_jsonl=str(goal_fixture(state)), wave_config=str(wave_config(state)), petri_bundle=str(petri_bundle(state)), clusters=None,
                           overrides=str(LIVE_OVERRIDES), roster=None, out=str(inv_p))
    assert ST.cmd_freeze(a) == 0
    rec = _receipts(state); assert len(rec) == 1 and rec[0]["at"] == "freeze" and rec[0]["override_id"] == LIVE_ID and set(rec[0]["evals"]) == {"codeattack"}
    assert sorted(rec[0]["evals"]["codeattack"]) == sorted(_stamped(inv_p)) and rec[0]["overrides_sha256"] == json.load(open(inv_p))["units"][0]["held_out"]["codeattack"]["overrides_sha256"]
    assert ST.cmd_freeze(a) == 0 and len(_receipts(state)) == 1                                        # unchanged: no second receipt
    # admission: the stamp lifts codeattack's requirement WITH its receipt …
    doc = json.load(open(inv_p)); runs, policies = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, policies, drop_eval="codeattack")
    db = make_registry(tmp_path / "reg", runs=runs, policies=policies)
    problems, ok = AD.check(str(state), str(db), {}, families=["static"]); assert problems == [] and len([o for o in ok if "excluded_by_ruling" in o]) == 2, problems
    # … and admits nothing without it
    lines = [l for l in (state / "attempts.jsonl").read_text().splitlines() if '"hold_out"' not in l]; (state / "attempts.jsonl").write_text("".join(l + "\n" for l in lines))
    problems, ok = AD.check(str(state), str(db), {}, families=["static"])
    assert len(problems) == 2 and all("no matching hold_out receipt" in p and "admits nothing" in p for p in problems) and ok == [], problems
    # a receipt for ANOTHER document (a different sha) does not cover the stamp
    stale = dict(rec[0]); stale["overrides_sha256"] = "0" * 64; (state / "attempts.jsonl").open("a").write(json.dumps(stale) + "\n")
    problems, _ = AD.check(str(state), str(db), {}, families=["static"]); assert len(problems) == 2 and all("no matching hold_out receipt" in p for p in problems)
    # the live apply repairs it and admission is whole again
    r = _hold_out(inv_p, "--overrides", str(LIVE_OVERRIDES)); assert r.returncode == 0 and "receipt was MISSING" in r.stdout, r.stdout + r.stderr
    problems, ok = AD.check(str(state), str(db), {}, families=["static"]); assert problems == [] and len(ok) == 4, problems


def _freeze_cli(state: Path, inv_p: Path, *, families: str = "static", **env):
    """The REAL freeze-inventory command as a subprocess (the crash hook exits the process); `families` "" = the whole wave."""
    args = ["--model", MODEL, "--code-sha", SHA, "--models-table", str(REPRO / "models.yaml"), "--families", families, "--nshards", "9",
            "--goals-jsonl", str(goal_fixture(state)), "--wave-config", str(wave_config(state)), "--petri-bundle", str(petri_bundle(state)),
            "--overrides", str(LIVE_OVERRIDES), "--out", str(inv_p)]
    return subprocess.run([sys.executable, str(REPRO / "lib" / "stage.py"), "freeze-inventory", *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src"), **env}, timeout=600)


def test_the_freeze_writes_the_receipt_before_publishing_and_a_death_between_them_is_recovered_by_the_re_freeze(tmp_path):
    """30-fix-2: 30-fix published the stamped inventory and THEN appended the receipt; a death between the two
    left an inventory admission refused (no receipt) that an unchanged re-freeze returned early on, never repairing. Now the receipt is the
    proof before the publish, and an unchanged re-freeze verifies and repairs a missing one."""
    state = tmp_path / "state"; (state / "gate").mkdir(parents=True); inv_p = state / "gate" / "expected_units.json"
    # the freeze dies AFTER the receipt and BEFORE the publish: a receipt, no inventory
    r = _freeze_cli(state, inv_p, SISTEER_STAGE_CRASH_AT="freeze_after_receipt")
    assert r.returncode == 97 and "CRASH INJECTED at freeze_after_receipt" in r.stderr, r.stdout + r.stderr          # (os._exit drops the unflushed stdout; the disk is the proof)
    assert not inv_p.exists() and len(_receipts(state)) == 1 and _receipts(state)[0]["at"] == "freeze"
    # the re-freeze publishes; the receipt already stands, so none is appended twice
    r = _freeze_cli(state, inv_p); assert r.returncode == 0 and "frozen" in r.stdout and "hold-out receipt" not in r.stdout, r.stdout + r.stderr
    rec = _receipts(state); assert inv_p.exists() and len(rec) == 1 and sorted(rec[0]["evals"]["codeattack"]) == sorted(_stamped(inv_p))
    doc = json.load(open(inv_p)); runs, policies = [], []
    for u in doc["units"]:
        satisfy(state, u, runs, policies, drop_eval="codeattack")
    db = make_registry(tmp_path / "reg", runs=runs, policies=policies)
    problems, ok = AD.check(str(state), str(db), {}, families=["static"]); assert problems == [] and len([o for o in ok if "excluded_by_ruling" in o]) == 2, problems
    # the pre-fix shape — an inventory published without its receipt — is repaired by the next (unchanged) re-freeze alone
    lines = [l for l in (state / "attempts.jsonl").read_text().splitlines() if '"hold_out"' not in l]; (state / "attempts.jsonl").write_text("".join(l + "\n" for l in lines))
    problems, _ = AD.check(str(state), str(db), {}, families=["static"]); assert len(problems) == 2 and all("no matching hold_out receipt" in p for p in problems)
    r = _freeze_cli(state, inv_p); assert r.returncode == 0 and "inventory unchanged" in r.stdout and "hold-out receipt REPAIRED" in r.stdout, r.stdout + r.stderr
    assert len(_receipts(state)) == 1 and json.load(open(inv_p)) == doc
    problems, ok = AD.check(str(state), str(db), {}, families=["static"]); assert problems == [] and len(ok) == 4, problems
    # a dry-run re-freeze reports and writes nothing
    lines = [l for l in (state / "attempts.jsonl").read_text().splitlines() if '"hold_out"' not in l]; (state / "attempts.jsonl").write_text("".join(l + "\n" for l in lines))
    r = subprocess.run([sys.executable, str(REPRO / "lib" / "stage.py"), "freeze-inventory", "--dry-run", "--model", MODEL, "--code-sha", SHA, "--models-table", str(REPRO / "models.yaml"),
                        "--families", "static", "--nshards", "9", "--goals-jsonl", str(goal_fixture(state)), "--wave-config", str(wave_config(state)), "--petri-bundle", str(petri_bundle(state)),
                        "--overrides", str(LIVE_OVERRIDES), "--out", str(inv_p)], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=600)
    assert r.returncode == 0 and "inventory unchanged" in r.stdout and _receipts(state) == [], r.stdout + r.stderr


def test_write_atomic_and_the_receipt_append_fsync_the_directory(tmp_path, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(os.fstat(fd).st_mode), real(fd)))
    import stat
    ST.write_atomic(str(tmp_path / "d" / "x.json"), {"a": 1})
    assert any(stat.S_ISDIR(m) for m in synced) and any(stat.S_ISREG(m) for m in synced), synced               # the file AND its directory
    synced.clear()
    ST.append_fsync(str(tmp_path / "d" / "ledger.jsonl"), json.dumps({"event": "hold_out"}))
    assert any(stat.S_ISDIR(m) for m in synced) and any(stat.S_ISREG(m) for m in synced), synced
    assert (tmp_path / "d" / "ledger.jsonl").read_text() == '{"event": "hold_out"}\n'


def test_a_wider_freeze_under_an_already_receipted_document_receipts_exactly_the_pairs_it_adds(tmp_path):
    """30-fix-3: the freeze skipped a whole override document as soon as ANY stamp had a receipt, but admission
    requires one per (eval, unit). Live shape: the static family frozen first (the receipt covers static/codeattack), then the full wave
    re-frozen, adding agentic/agentic_murder under the SAME override id and sha — the new stamps published unreceipted, admission refused them,
    and an unchanged re-freeze could not repair. Now coverage is per pair: exactly the missing set is receipted, on the wider freeze and on the
    unchanged re-freeze repair."""
    state = tmp_path / "state"; (state / "gate").mkdir(parents=True); inv_p = state / "gate" / "expected_units.json"
    r = _freeze_cli(state, inv_p); assert r.returncode == 0, r.stdout + r.stderr                                           # static first
    rec = _receipts(state); assert len(rec) == 1 and set(rec[0]["evals"]) == {"codeattack"}
    r = _freeze_cli(state, inv_p, families=""); assert r.returncode == 0 and "inventory CHANGED" in r.stdout, r.stdout + r.stderr   # the full wave
    doc = json.load(open(inv_p)); fam = lambda u: u["logical_work_id"].split("/")[2]; by_lw = {u["logical_work_id"]: u for u in doc["units"]}
    stamps = [(u["logical_work_id"], e, h) for u in doc["units"] for e, h in (u.get("held_out") or {}).items()]
    assert {e for _, e, _ in stamps} == {"codeattack", "agentic_murder"} and {fam(by_lw[lw]) for lw, _, _ in stamps} >= {"static", "agentic"}
    rec = _receipts(state); assert len(rec) == 2
    covered = [(lw, e) for x in rec for e, lws in x["evals"].items() for lw in lws]
    assert sorted(covered) == sorted((lw, e) for lw, e, _ in stamps) and len(covered) == len(set(covered))                 # every stamped pair receipted exactly once
    assert rec[1]["evals"]["agentic_murder"] == [lw for lw, e, _ in stamps if e == "agentic_murder"]         and set(rec[0]["evals"]["codeattack"]).isdisjoint(rec[1]["evals"].get("codeattack", []))                             # the second receipt = exactly the pairs the wider freeze added (agentic, and the new units that carry codeattack)
    assert all(any(ST.hold_out_receipt_matches(x, h, e, lw) for x in rec) for lw, e, h in stamps)                           # admission's own predicate, every pair
    # admission over both families: every stamp lifts its requirement WITH its receipt
    runs, policies = [], []
    for u in doc["units"]:
        if fam(u) in ("static", "agentic"):
            satisfy(state, u, runs, policies, drop_eval="codeattack" if fam(u) == "static" else "agentic_murder")
    db = make_registry(tmp_path / "reg", runs=runs, policies=policies)
    problems, ok = AD.check(str(state), str(db), {}, families=["static", "agentic"]); assert problems == [] and len([o for o in ok if "excluded_by_ruling" in o]) == 3, problems
    # the pre-fix shape — the static receipt alone, the agentic stamps bare — is refused by admission and repaired by the next unchanged re-freeze
    lines = [l for l in (state / "attempts.jsonl").read_text().splitlines() if not ('"hold_out"' in l and "agentic_murder" in l)]; (state / "attempts.jsonl").write_text("".join(l + "\n" for l in lines))
    problems, _ = AD.check(str(state), str(db), {}, families=["static", "agentic"]); assert len(problems) == 1 and "agentic_murder" in problems[0] and "no matching hold_out receipt" in problems[0], problems
    r = _freeze_cli(state, inv_p, families=""); assert r.returncode == 0 and "inventory unchanged" in r.stdout and "hold-out receipt REPAIRED" in r.stdout and "agentic_murder on 1 unit(s)" in r.stdout, r.stdout + r.stderr
    rec = _receipts(state); assert len(rec) == 2 and rec[1]["evals"]["agentic_murder"] == [lw for lw, e, _ in stamps if e == "agentic_murder"]
    covered = [(lw, e) for x in rec for e, lws in x["evals"].items() for lw in lws]; assert sorted(covered) == sorted((lw, e) for lw, e, _ in stamps)   # whole again, nothing twice
    problems, ok = AD.check(str(state), str(db), {}, families=["static", "agentic"]); assert problems == [] and len([o for o in ok if "excluded_by_ruling" in o]) == 3, problems
