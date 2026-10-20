"""hpo_probe_status.py reads ACROSS records: a main record's verdict + menu stamp the PARITY-PASS record
they read adapter parity from (`parity_record`: dir, result_generation, n_results, plan_launches, windows, scale table),
but the main record's OWN generation stamps do not move when the parity record's results do. Before this, a parity-reference redo
(`run --resume --reference-only` on the parity record) that changed its per-adapter verdicts left the main menu READY, so
`03_hpo.sh --phase probe` re-fetched / accepted it and design / trials could pick a pair whose CURRENT parity is a mismatch.
Now `--record <main>` resolves `parity_record.dir`, demands a complete checkpoint there, compares its LIVE result generation and
plan identity against the stamp, and reports PARITY_STALE (never READY) on any disagreement — with the reason named. 03's
fetch / accept / revalidate paths all read this state (READY is the only fetch state; `probe_local_is_current` needs READY), so
they refuse without a change of their own. CPU only; fake engines; synthetic ids; no cluster path, no host name."""
from __future__ import annotations

import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as g

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import hpo_probe_status as hps  # noqa: E402
import serving_probe as sp  # noqa: E402
from sisteer.capeval.serving import standup_manifest as sm  # noqa: E402
from test_serving_probe_parity_pass import MODEL, ScaledFake, _main_record, _parity_plan  # noqa: E402
from test_serving_probe_reference_missing import _hf, _ref_deps  # noqa: E402
from test_serving_probe_revocation import _mint_receipt  # noqa: E402
from test_serving_probe_smoke import _passed  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE", sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)


def _judge_pass(pending):
    """Every adapter agrees with the engine's first token: a passing HF+PEFT judgement."""
    return {it["dir"]: {**sp.first_token_parity(_hf(it["vllm_first_token"]["id"], -4.5), it["vllm_first_token"]), "hf": _hf(it["vllm_first_token"]["id"], -4.5)}
            for it in pending}


def _parity_record_at_rest(pmain: Path, *, judge=_judge_pass):
    """A parity-pass record as the carrier leaves it: the 3 natural launches passed with the driver's `reference_pending` items, the
    HF+PEFT phase run through the fake deps (reference_check.json + each launch's reference_check), its OWN verdict written (the
    parity summary), a complete checkpoint with no attempt. Returns (plan, results, deps)."""
    plan = _parity_plan()
    sp.write_json_atomic(pmain / sp.PLAN_FILE, plan)
    sp.build_adapter_dirs(plan, pmain, ScaledFake())
    results = {}
    for rec in plan["launches"]:
        m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
        res = _passed(rec, m, reference=None)
        res["reference_pending"] = [
            {"dir": str(sp.slot_dir(pmain, m.name, a.name)), "name": a.name, "kind": a.kind, "role": a.role, "rank": a.rank,
             "lora_modules": list(a.lora_modules),
             "vllm_first_token": {"id": 1000 + len(a.name), "logprob": -4.5, "token": f"token_id:{1000 + len(a.name)}"}}
            for a in m.adapters if a.kind == sm.KIND_GENUINE]
        res["probes"] = {"base": {"token_ids": [29108], "rows": [{"logprob": -0.1404, "token": "token_id:29108", "top": []}]}, "adapters": {}}
        results[rec["id"]] = res
        sp.write_json_atomic(sp.launch_result_path(pmain, rec["id"]), res)
    deps, _calls = _ref_deps(plan, judge)
    assert sp.reference_phase(plan, pmain, deps)["status"] == "done"
    results = sp.load_results(pmain)
    sp.write_verdict(plan, results, probe_job_id="p1", out=pmain, verdict_uri=str(pmain / sp.VERDICT_FILE))
    sp.write_progress(plan, results, pmain)
    assert (pmain / sp.VERDICT_FILE).is_file() and (pmain / sp.REFERENCE_CHECK_FILE).is_file() and sp.load_progress(pmain)["complete"] is True
    return plan, results, deps


def _ready_pair(tmp_path: Path):
    """A main record whose menu was written by `reverdict --parity-record` against a complete parity record AT REST: READY, parity current."""
    main, pmain, site = tmp_path / "main", tmp_path / "parity", tmp_path / "site"
    main.mkdir(); pmain.mkdir(); site.mkdir()
    plan, results = _main_record(main)
    sp.write_verdict(plan, results, probe_job_id="800", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site)
    sp.write_progress(plan, results, main)
    pplan, presults, _deps = _parity_record_at_rest(pmain)
    nonce = _mint_receipt(main, sp.result_generation(main)["generation"])
    assert sp.main(["reverdict", "--record", str(main), "--parity-record", str(pmain), "--reopen-token", nonce, "--geometry-dir", str(site)]) == 0
    st, detail, info = hps.probe_state_record(main, job_active=False)
    assert st == hps.READY and info["generation_check"] == "stamped_match" and info["parity_check"] == "current", (st, detail)
    assert info["parity_record"]["dir"] == str(pmain) and info["parity_record"]["live_generation"] == info["parity_record"]["stamped_generation"]
    assert info["parity_record"]["complete"] is True and "parity record current" in detail
    return main, pmain, pplan, presults


def _redo_parity_reference(pmain: Path, pplan, presults, *, manifest="wo_r4_S1cc"):
    """What `run --resume --reference-only` on the PARITY record leaves behind: one launch's reference_check re-judged (here: a
    fail on the control adapter, which rides the S1cc server), the result file rewritten, the checkpoint re-stamped — the parity
    record's result generation moves, the main's does not."""
    lid = next(r["id"] for r in pplan["launches"] if r["manifest"] == manifest)
    res = dict(presults[lid])
    res["reference_check"] = {"status": "fail", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"control__lora": {"status": "fail", "top1_equal": False}}}
    presults[lid] = res
    sp.write_json_atomic(sp.launch_result_path(pmain, lid), res)
    sp.write_progress(pplan, presults, pmain)
    return lid


def test_a_parity_reference_redo_with_the_main_generation_unchanged_is_parity_stale_never_ready(tmp_path):
    main, pmain, pplan, presults = _ready_pair(tmp_path)
    main_gen = sp.result_generation(main)["generation"]
    stamped = json.loads((main / sp.VERDICT_FILE).read_text())[sp.PARITY_RECORD_KEY][sp.RESULT_GENERATION_KEY]
    _redo_parity_reference(pmain, pplan, presults)
    assert sp.result_generation(main)["generation"] == main_gen                                 # the main record did not move …
    live = sp.result_generation(pmain)["generation"]
    assert live != stamped                                                                       # … the parity record did
    st, detail, info = hps.probe_state_record(main, job_active=False)
    assert st == hps.PARITY_STALE and st != hps.READY
    assert info["complete"] is True and info["generation_check"] == "stamped_match"              # the same-record half is still green
    assert info["parity_check"] == "mismatch" and info["parity_record"]["live_generation"] == live and info["parity_record"]["stamped_generation"] == stamped
    assert "NEVER fetch it" in detail and str(pmain) in detail and live[:12] in detail and stamped[:12] in detail
    assert "reference redo" in detail and "reverdict --parity-record" in detail and "reopen" in detail
    assert "queued/running" in hps.probe_state_record(main, job_active=True)[1]
    # the CLI prints the state, the reason and the json the front door reads
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert hps.main(["--record", str(main), "--job-active", "0"]) == 0
    state, det, js = buf.getvalue().rstrip("\n").split("\t", 2)
    assert state == "PARITY_STALE" and json.loads(js)["parity_check"] == "mismatch" and json.loads(js)["parity_record"]["dir"] == str(pmain)
    # a fresh reverdict against the redone parity record (a reopen: the receipt) reads the fail — but the parity record's OWN verdict is
    # still the one written before its results moved: not at rest (in_progress) until the carrier's verdict rewrites it
    nonce = _mint_receipt(main, main_gen)
    assert sp.main(["reverdict", "--record", str(main), "--parity-record", str(pmain), "--reopen-token", nonce, "--geometry-dir", str(tmp_path / "site")]) == 0
    st, detail, info = hps.probe_state_record(main, job_active=False)
    assert (st, info["parity_check"]) == (hps.PARITY_STALE, "in_progress") and "own verdict is stamped for result generation" in detail and "carrier" in detail
    sp.write_verdict(pplan, sp.load_results(pmain), probe_job_id="p2", out=pmain, verdict_uri=str(pmain / sp.VERDICT_FILE))
    st, detail, info = hps.probe_state_record(main, job_active=False)
    assert st == hps.READY and info["parity_check"] == "current" and info["parity_record"]["stamped_generation"] == live
    v = json.loads((main / sp.VERDICT_FILE).read_text())
    assert any("reference_mismatch:control__lora" in e["reasons"] for e in v["excluded"] if e["set"] == "wo")


def test_every_other_disagreement_with_the_stamp_is_parity_stale_with_its_reason(tmp_path):
    main, pmain, pplan, presults = _ready_pair(tmp_path)

    def state():
        st, detail, info = hps.probe_state_record(main, job_active=False)
        return st, info["parity_check"], detail

    # (a) the parity record's checkpoint is not complete (a re-run / redo under way or stalled)
    prog = json.loads((pmain / sp.PROGRESS_FILE).read_text())
    sp.write_json_atomic(pmain / sp.PROGRESS_FILE, {**prog, "complete": False})
    st, chk, detail = state()
    assert (st, chk) == (hps.PARITY_STALE, "incomplete") and "complete=false" in detail and "in motion" in detail
    (pmain / sp.PROGRESS_FILE).unlink()
    assert state()[:2] == (hps.PARITY_STALE, "incomplete")
    sp.write_json_atomic(pmain / sp.PROGRESS_FILE, prog)
    assert state()[0] == hps.READY
    # (a2) a driver's attempt on the checkpoint (a reference-only re-run marks it so BEFORE its HF phase; a dead worker leaves it) —
    # refused as in_progress even with complete=true and the generation unmoved
    sp.write_json_atomic(pmain / sp.PROGRESS_FILE, {**prog, "attempt": {"started_utc": "2026-09-30T18:00:00Z", "job_id": "999", "host": "TESTHOST", "reference_only": True}})
    st, chk, detail = state()
    assert (st, chk) == (hps.PARITY_STALE, "in_progress") and "a driver holds it" in detail and "reference-only re-run" in detail and "job 999" in detail
    sp.write_json_atomic(pmain / sp.PROGRESS_FILE, prog)
    # (a3) the parity record's own derived artifacts archived (the re-run's first act) — in_progress whatever the generation says
    for name in (sp.VERDICT_FILE, sp.REFERENCE_CHECK_FILE):
        shutil.move(str(pmain / name), str(tmp_path / f"aside.{name}"))
        st, chk, detail = state()
        assert (st, chk) == (hps.PARITY_STALE, "in_progress") and "lacks its own derived artifact" in detail and name in detail
        shutil.move(str(tmp_path / f"aside.{name}"), str(pmain / name))
    assert state()[0] == hps.READY
    # (b) the checkpoint stamped for another generation than the result files (mid-rewrite)
    sp.write_json_atomic(pmain / sp.PROGRESS_FILE, {**prog, sp.RESULT_GENERATION_KEY: "f" * 64})
    st, chk, detail = state()
    assert (st, chk) == (hps.PARITY_STALE, "mismatch") and "mid-rewrite" in detail
    sp.write_json_atomic(pmain / sp.PROGRESS_FILE, prog)
    # (c) the plan is another one: the scale table moved (re-planned with other --parity-scales)
    plan_doc = json.loads((pmain / sp.PLAN_FILE).read_text())
    sp.write_json_atomic(pmain / sp.PLAN_FILE, {**plan_doc, "synthetic_scale_by_set": {**plan_doc["synthetic_scale_by_set"], "wo": 0.5}})
    st, chk, detail = state()
    assert (st, chk) == (hps.PARITY_STALE, "plan_mismatch") and "scale table" in detail and "re-planned" in detail
    sp.write_json_atomic(pmain / sp.PLAN_FILE, {**plan_doc, "revision": "other-rev"})
    assert state()[:2] == (hps.PARITY_STALE, "plan_mismatch") and "revision" in state()[2]
    sp.write_json_atomic(pmain / sp.PLAN_FILE, {**plan_doc, "parity_pass": False})
    assert state()[:2] == (hps.PARITY_STALE, "plan_mismatch") and "not a parity pass" in state()[2]
    sp.write_json_atomic(pmain / sp.PLAN_FILE, plan_doc)
    assert state()[0] == hps.READY
    # (d) the parity record is gone (moved / deleted)
    shutil.move(str(pmain), str(tmp_path / "parity.moved"))
    st, chk, detail = state()
    assert (st, chk) == (hps.PARITY_STALE, "absent") and "absent or unreadable" in detail
    shutil.move(str(tmp_path / "parity.moved"), str(pmain))
    assert state()[0] == hps.READY
    # (e) the verdict and the menu name different parity records (one rewritten without the other)
    menu = json.loads((main / sp.ALLOWED_PAIRS_FILE).read_text())
    sp.write_json_atomic(main / sp.ALLOWED_PAIRS_FILE, {**menu, sp.PARITY_RECORD_KEY: {**menu[sp.PARITY_RECORD_KEY], "dir": str(tmp_path / "elsewhere")}})
    st, chk, detail = state()
    assert (st, chk) == (hps.PARITY_STALE, "mismatch") and "do not name the same parity record" in detail
    sp.write_json_atomic(main / sp.ALLOWED_PAIRS_FILE, {**menu, sp.PARITY_RECORD_KEY: None})
    assert state()[:2] == (hps.PARITY_STALE, "mismatch")
    sp.write_json_atomic(main / sp.ALLOWED_PAIRS_FILE, menu)
    assert state()[0] == hps.READY
    # the same-record half still outranks: a moved MAIN result (its generation no longer the stamped one) is MENU_STALE, whatever
    # the parity record says
    mlid = json.loads((main / sp.PLAN_FILE).read_text())["launches"][0]["id"]
    mres = json.loads(sp.launch_result_path(main, mlid).read_text())
    sp.write_json_atomic(sp.launch_result_path(main, mlid), {**mres, "touched": True})
    assert hps.probe_state_record(main, job_active=False)[0] == hps.MENU_STALE


def test_parity_stale_is_a_state_of_its_own_and_a_record_without_a_parity_record_is_unaffected(tmp_path):
    assert hps.PARITY_STALE in hps.STATES and hps.PARITY_STALE not in hps.FETCH_STATES and hps.PARITY_STALE not in hps.SUBMIT_STATES
    assert hps.FETCH_STATES == (hps.READY,) and hps.PARITY_CHECK_CURRENT == ("none", "current")
    # no parity record read (parity_record None on both artifacts): nothing to hold
    assert hps.verify_parity_record(tmp_path, {sp.PARITY_RECORD_KEY: None}, {sp.PARITY_RECORD_KEY: None}) == {"parity_check": "none", "parity_why": None, "parity_record": None}
    assert hps.verify_parity_record(tmp_path, {}, {})["parity_check"] == "none"                     # a verdict written before the key existed
    # a stamp whose dir is empty / unreadable is absent, never current
    assert hps.verify_parity_record(tmp_path, {sp.PARITY_RECORD_KEY: {"dir": "", sp.RESULT_GENERATION_KEY: "a" * 64}},
                                    {sp.PARITY_RECORD_KEY: {"dir": "", sp.RESULT_GENERATION_KEY: "a" * 64}})["parity_check"] == "absent"
    # the front door's contract: 03 fetches in READY alone (FETCH_STATES) — the ONLY state this module lets a menu travel in — so
    # PARITY_STALE refuses every consuming phase (03's own revalidation reads this module's state)
    src = (REPO / "reproduce" / "03_hpo.sh").read_text()
    assert "hpo_probe_status.py" in src and "probe_fetch_ready" in src[src.index("READY)", src.index('case "$PSTATE" in')):]
    doc = hps.__doc__
    assert "PARITY_STALE" in doc and "parity_check" in doc and "D191" in doc


def _reference_only(pmain: Path, *, token=None):
    argv = ["run", "--model-key", MODEL, "--out", str(pmain), "--resume", "--reference-only", "--plan-args", "--sets wo,mlp --ranks 4 --tp 1"]
    if token:
        argv += ["--reopen-token", token]
    return sp.main(argv)


def test_the_main_menu_is_never_ready_while_the_parity_records_reference_phase_is_being_redone(tmp_path, monkeypatch):
    """`run --resume --reference-only` on the complete parity record used to re-stamp its checkpoint complete=true
    with the OLD generation before the HF phase, so between that write and the first per-launch rewrite the main record read READY.
    Now the re-run marks the checkpoint in progress (complete=false, attempt, hold) before the phase, and the cross-record check refuses
    on the attempt (and on the archived own-artifacts) whatever the generation says; the completing write and the carrier's verdict on
    the parity record restore READY."""
    main, pmain, pplan, presults = _ready_pair(tmp_path)
    seen = []

    def judge(pending):
        # INSIDE the HF phase: the checkpoint is in progress and the record's own artifacts are archived; the main must not be READY
        st, detail, info = hps.probe_state_record(main, job_active=False)
        seen.append((st, info["parity_check"], detail))
        prog = sp.load_progress(pmain)
        seen.append((prog["complete"], prog["hold"], bool(prog["attempt"]), prog["attempt"].get(sp.REFERENCE_RERUN_STARTED_KEY) is not None))
        return _judge_pass(pending)
    deps, calls = _ref_deps(pplan, judge)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    gen_p = sp.result_generation(pmain)["generation"]
    nonce = _mint_receipt(pmain, gen_p)
    assert _reference_only(pmain, token=nonce) == 0 and calls["reference"]
    assert seen[0][0] == hps.PARITY_STALE and seen[0][1] == "in_progress" and "a driver holds it" in seen[0][2] and "reference-only re-run" in seen[0][2]
    assert seen[1] == (False, sp.HOLD_REFERENCE_ONLY, True, True)                              # the checkpoint said so before the phase
    # after the completing write: complete again, no attempt, no hold — but the record's own verdict is still archived -> in_progress
    prog = sp.load_progress(pmain)
    assert prog["complete"] is True and prog["attempt"] is None and prog["hold"] is None and prog["stopped_reason"] is None
    st, detail, info = hps.probe_state_record(main, job_active=False)
    assert (st, info["parity_check"]) == (hps.PARITY_STALE, "in_progress") and sp.VERDICT_FILE in detail
    # the carrier's verdict on the parity record: at rest; the same judgements -> the same generation -> the main is READY again
    sp.write_verdict(pplan, sp.load_results(pmain), probe_job_id="p2", out=pmain, verdict_uri=str(pmain / sp.VERDICT_FILE))
    assert sp.result_generation(pmain)["generation"] == gen_p
    st, detail, info = hps.probe_state_record(main, job_active=False)
    assert st == hps.READY and info["parity_check"] == "current"


def test_a_reference_worker_that_dies_leaves_the_parity_record_in_progress_and_the_main_never_ready_until_it_is_resumed(tmp_path, monkeypatch):
    main, pmain, pplan, presults = _ready_pair(tmp_path)

    def dies(pending):
        raise SystemExit("worker killed mid-phase")                       # not an Exception: the phase does not record it, the process dies
    deps, _ = _ref_deps(pplan, dies)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    gen_p = sp.result_generation(pmain)["generation"]
    nonce = _mint_receipt(pmain, gen_p)
    with pytest.raises(SystemExit, match="worker killed"):
        _reference_only(pmain, token=nonce)
    prog = sp.load_progress(pmain)
    assert prog["complete"] is False and prog["hold"] == sp.HOLD_REFERENCE_ONLY and prog["stopped_reason"] == sp.HOLD_REFERENCE_ONLY
    assert prog["attempt"]["reference_only"] is True and prog["attempt"]["interrupted_utc"] and "SystemExit" in prog["attempt"]["interrupted_by"]
    assert sp.result_generation(pmain)["generation"] == gen_p                                    # nothing rewritten: the generation did not move …
    for _ in range(2):                                                                            # … and the main is NOT READY, and stays so
        st, detail, info = hps.probe_state_record(main, job_active=False)
        assert (st, info["parity_check"]) == (hps.PARITY_STALE, "in_progress") and "interrupted" in detail and "died before its completing write" in detail
    # the parity record read on its own says what holds it and how to resume
    st, detail, info = hps.probe_state_record(pmain, job_active=False)
    assert st == hps.INCOMPLETE_STALLED and sp.HOLD_REFERENCE_ONLY in detail and "run --resume --reference-only" in detail
    # the resume needs NO new receipt (the record is not complete; the first attempt consumed it), completes, and the verdict restores READY
    deps_ok, calls = _ref_deps(pplan, _judge_pass)
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps_ok))
    assert _reference_only(pmain) == 0 and calls["reference"]
    prog = sp.load_progress(pmain)
    assert prog["complete"] is True and prog["attempt"] is None and prog["hold"] is None
    sp.write_verdict(pplan, sp.load_results(pmain), probe_job_id="p3", out=pmain, verdict_uri=str(pmain / sp.VERDICT_FILE))
    st, detail, info = hps.probe_state_record(main, job_active=False)
    assert st == hps.READY and info["parity_check"] == "current"
