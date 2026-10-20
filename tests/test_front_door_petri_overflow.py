"""06_eval_waves.sh --pass overflow: the Petri CONTEXT-OVERFLOW RE-RUN PASS through the front door.

A first-pass drive that leaves legs INCOMPLETE on context-overflow ledgers settles `incomplete` in the daemon's record with a result
`overflow_pending` and a per-leg `legs.detail`; finishing those legs needs a second proxy at the doubled window (`petri_proxy.sbatch …
overflow-rerun <cells>`) driven under F22_OVERFLOW_RERUN=1 — until now a hand step. `--pass overflow` reads THAT record (lib/petri_overflow.py),
re-reads the ledgers through the drive's own done-ness predicate (`petri_run.py legs-status`), derives the cells the rerun proxy must serve
and declare (the still-ledgered legs' cells; a ledgered CONTROL leg brings its `control_of` parent `none`, the ONE way the drive keeps the
controls twin), plans the one `petri.overflow` row and submits it through the funnel — refusing by name until the daemon has recorded
that verdict, when nothing is ledgered any more, and with --units / --cells / --twin or another family. A BOUND overflow unit (`--resubmit`
after a partial pass whose proxy died) carries its declaration VERBATIM from the attempt record. Synthetic state in the daemon's and the
drive's real shapes (the frozen DEMO record's leg identities with synthetic hashes); a stub legs-status; the real stage's dry run on a
copy of reproduce/site.example; neutral labels, placeholder ids."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPRO, VENV_PY, run_stage  # noqa: E402

sys.path.insert(0, str(LIB))
import petri_overflow as PO  # noqa: E402
import stage as ST  # noqa: E402

MODEL, VIEW, TAG = "qwen35_4b_demo", "battery", "qwen35_4b_demo_battery"
FIRST_LWID = f"{MODEL}/{VIEW}/petri/proxy"
OVER_LWID = f"{MODEL}/{VIEW}/petri/overflow"
#: the frozen demo record's legs (cells.json: canonical + controls, in record order), synthetic run keys
LEGS = [("none", "none", None), ("prompted__safety", "prompted", None), ("prompted__constitution", "prompted", None), ("arditi_allpos", "vector_all", None),
        ("si_constv2ctl2_full__lora_r4_ce", "lora_all", None), ("si_constv2_mix_full__lora_r4_ce", "lora_all", None),
        ("si_constv2_mix_full__lora_r4_ce_minus_ctl", "lora_subtraction", None), ("si_constv2ctl2_full__psr_ce_all", "psr", None),
        ("si_constv2_mix_full__psr_ce_all", "psr", None), ("si_constv2_mix_full__psr_ce_all_minus_ctl", "psr_paired_write", None), ("none__lora0", "none", "none")]
KEYS = [f"{i:016x}" for i in range(len(LEGS))]
CELLS_2_4_10 = ["none", "prompted__si_constitution", "constv2ctl2-full__lora_r4_ce"]   # leg 2's cell, leg 4's cell, and `none` = leg 10's parent


def _leg(i, *, done=False, ledgered=0, refused=None):
    cid, kind, parent = LEGS[i]
    return {"idx": i, "cell_id": cid, "control_of": parent, "kind": kind, "petri_run_key": KEYS[i], "cell_dir": f"/synthetic/out/cells/{KEYS[i]}",
            "done": done, "refused": refused is not None, "reason": refused, "results_present": done, "unresolved": ledgered, "overflow_unresolved": ledgered}


def _legs_status_doc(out_root, *, ledgered=(2, 4, 10), done=(), refused=(), parent_of=None):
    legs = [_leg(i, done=i in done, ledgered=(57 if i == 2 else 3) if i in ledgered else 0, refused=("synthetic refusal" if i in refused else None)) for i in range(len(LEGS))]
    if parent_of:
        for i, p in parent_of.items():
            legs[i]["control_of"] = p
    return {"out_root": str(out_root), "cells_json": f"{out_root}/cells.json", "audit": f"petri_full22_{TAG}", "serving_path": "lora_native", "record_refused": None, "legs": legs}


def _stub(tmp: Path, doc: dict, rc: int = 0) -> Path:
    """A stand-in `petri_run.py` for legs-status (SISTEER_PETRI_RUN): prints `doc`, exits `rc`; records its argv."""
    d = tmp / "stub"; d.mkdir(parents=True, exist_ok=True)
    (d / "doc.json").write_text(json.dumps(doc))
    p = d / "petri_run_stub.py"
    p.write_text("import json, sys\nopen(sys.argv[0] + '.argv', 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                 f"sys.exit({rc}) if {rc} else print(open({str(d / 'doc.json')!r}).read())\n")
    return p


def _state(tmp: Path, *, first_job="11111", first_bound=True, attempt_state="incomplete", verdict="overflow_pending", named=(2, 4, 10), with_record=True,
           with_detail=True, overflow_job=None, overflow_args=None, overflow_transition=False, cluster="cluster_a", overflow_cluster=None) -> Path:
    """The model's state dir in the daemon's and the funnel's real shapes: the first-pass unit's `submitted` binding + ledger row (on `cluster`), the
    daemon's record with one attempt for `first_job` and its result file (verdict, legs.detail, paths.out_root); optionally a bound overflow unit
    (on `overflow_cluster`, default the first pass's) whose attempt record carries `spec.args` (positional 5 = the declaration)."""
    st = tmp / "state" / MODEL; st.mkdir(parents=True, exist_ok=True)
    subs = st / "submissions.jsonl"; subs.touch()
    aid1 = f"{FIRST_LWID}@{cluster}@1"
    if first_bound:
        with open(subs, "a") as fh:
            fh.write(json.dumps({"event": "submitted", "cluster": cluster, "job_id": first_job, "unit": "petri.proxy", "model": MODEL, "logical_work_id": FIRST_LWID,
                                 "attempt_id": aid1, "job_name": f"f22-petri-{TAG}", "extra": {"kind": "target", "group": None, "sha": "0" * 40}}) + "\n")
        ST.write_binding(str(subs), ST.submitted_record(FIRST_LWID, first_job, cluster, aid1, f"f22-petri-{TAG}"))
    out_root = tmp / "petri_out"; out_root.mkdir(exist_ok=True)
    if with_record:
        grade = st / "grade"; grade.mkdir(exist_ok=True)
        result = grade / f"petri_drive.{TAG}.1.log.result.json"
        detail = [{"idx": i, "cell_id": LEGS[i][0], "control_of": LEGS[i][2], "kind": LEGS[i][1], "petri_run_key": KEYS[i], "registered": LEGS[i][2] is None,
                   "in_scope": True, "status": "incomplete_overflow" if i in named else "complete", "unresolved": 3 if i in named else 0, "overflow_unresolved": 3 if i in named else 0}
                  for i in range(len(LEGS))]
        doc = {"verdict": verdict, "rc": 3 if verdict == "overflow_pending" else 0, "legs": {"scope": "all", "n_scope": 11, "complete": 11 - len(named), "incomplete_overflow": len(named),
                                                                                                "failed": 0, "skipped": 0, **({"detail": detail} if with_detail else {})},
               "handoff": {"mode": "on", "verify_scope": "all"}, **({"paths": {"out_root": str(out_root), "cells_json": str(out_root / "cells.json")}} if with_detail else {})}
        result.write_text(json.dumps(doc))
        rec = {"tag": TAG, "model": MODEL, "view": VIEW, "cap": 3, "attempts": [{"n": 1, "state": attempt_state, "identity": {"job_id": first_job, "endpoint_mtime": "1", "wave_sha": "0" * 40},
                                                                                "result_file": str(result), "reason": "synthetic"}]}
        (grade / f"petri_drive.{TAG}.json").write_text(json.dumps(rec))
    if overflow_job:
        ocl = overflow_cluster or cluster
        aid2 = f"{OVER_LWID}@{ocl}@1"
        with open(subs, "a") as fh:
            fh.write(json.dumps({"event": "submitted", "cluster": ocl, "job_id": overflow_job, "unit": "petri.overflow", "model": MODEL, "logical_work_id": OVER_LWID,
                                 "attempt_id": aid2, "job_name": f"f22-petri-{TAG}-rerun", "extra": {"kind": "target", "group": None, "sha": "0" * 40}}) + "\n")
        ST.write_binding(str(subs), ST.submitted_record(OVER_LWID, overflow_job, ocl, aid2, f"f22-petri-{TAG}-rerun"))
        with open(st / "attempts.jsonl", "a") as fh:
            fh.write(json.dumps({"event": "attempt", "logical_work_id": OVER_LWID, "attempt_id": aid2, "cluster": ocl, "job_id": overflow_job,
                                 "spec": {"script": ST.PETRI_PROXY_SCRIPT, "args": [MODEL, VIEW, "0" * 40, "overflow-rerun", *([overflow_args] if overflow_args is not None else [])]}}) + "\n")
        if overflow_transition:
            ST.write_transition(str(subs), {"logical_work_id": OVER_LWID, "intended_attempt": f"{OVER_LWID}@{ocl}@2", "utc": "2026-01-01T00:00:00Z"})
    return st


def _cells(tmp, monkeypatch, st, doc):
    monkeypatch.setenv(PO.PETRI_RUN_ENV, str(_stub(tmp, doc)))
    return PO.overflow_cells(str(st), MODEL, VIEW, python=sys.executable)


def _refused(tmp, monkeypatch, st, doc, name):
    with pytest.raises(PO.Refused) as ei:
        _cells(tmp, monkeypatch, st, doc)
    assert ei.value.name == name, (ei.value.name, ei.value.text)
    return ei.value.text


# ------------------------------------------------------------------------------------------------------- the cells
def test_the_still_ledgered_legs_name_the_cells_and_a_control_leg_brings_its_parent_none(tmp_path, monkeypatch):
    st = _state(tmp_path)
    d = _cells(tmp_path, monkeypatch, st, _legs_status_doc(tmp_path / "petri_out"))
    assert d["cells"] == CELLS_2_4_10 and d["declaration_source"] == "computed" and d["overflow_bound"] is False
    assert [l["idx"] for l in d["still_ledgered"]] == [2, 4, 10] and d["done_legs"] == [] and d["named_legs"] == [2, 4, 10]
    assert d["first_job_id"] == "11111" and d["out_root"] == str(tmp_path / "petri_out") and d["overflow_lwid"] == OVER_LWID
    # the stub was asked exactly the drive's question: legs-status over the result's out root under the audit's manifest
    argv = json.loads((tmp_path / "stub" / "petri_run_stub.py.argv").read_text().splitlines()[-1])
    assert argv[argv.index("--out-root") + 1] == str(tmp_path / "petri_out") and argv[argv.index("legs-status") - 1].endswith(f"petri_full22_{TAG}.yaml") and "legs-status" in argv
    # (b2) only the LoRA leg: its cell alone (nothing needs the twin); (b3) only the control leg: `none` alone
    st2 = _state(tmp_path / "b2", named=(4,))
    assert _cells(tmp_path / "b2", monkeypatch, st2, _legs_status_doc(tmp_path / "b2" / "petri_out", ledgered=(4,)))["cells"] == ["constv2ctl2-full__lora_r4_ce"]
    st3 = _state(tmp_path / "b3", named=(10,))
    assert _cells(tmp_path / "b3", monkeypatch, st3, _legs_status_doc(tmp_path / "b3" / "petri_out", ledgered=(10,)))["cells"] == ["none"]
    # the pure derivation: a prompted leg translates to its config spelling; order is the config's
    cfg = ST.view_cells_of(MODEL, VIEW)
    assert PO.derive_cells([_leg(10, ledgered=3), _leg(2, ledgered=57)], [l[0] for l in LEGS[:-1]], cfg) == ["none", "prompted__si_constitution"]


# ------------------------------------------------------------------------------------------------------- the refusals, by name
def test_every_refusal_is_by_name(tmp_path, monkeypatch):
    doc = _legs_status_doc(tmp_path / "petri_out")
    _refused(tmp_path, monkeypatch, _state(tmp_path / "u", first_bound=False), doc, "petri_overflow_first_pass_unbound")
    _refused(tmp_path, monkeypatch, _state(tmp_path / "nr", with_record=False), doc, "petri_overflow_no_record")
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "st", attempt_state="started"), doc, "petri_overflow_not_pending"); assert "is running" in t
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "co", attempt_state="completed", verdict="complete"), doc, "petri_overflow_not_pending"); assert "nothing to re-run" in t
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "fa", attempt_state="failed", verdict="failed"), doc, "petri_overflow_not_pending"); assert "re-drive the first pass" in t
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "nv", attempt_state="incomplete", verdict="complete"), doc, "petri_overflow_not_pending"); assert "overflow_pending" in t
    _refused(tmp_path, monkeypatch, _state(tmp_path / "nd", with_detail=False), doc, "petri_overflow_no_detail")
    # the ledgers moved: a named leg the drive would refuse; a named leg neither done nor ledgered; an unnamed leg on a ledger
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "m1"), _legs_status_doc(tmp_path / "petri_out", refused=(4,)), "petri_overflow_ledgers_moved"); assert "would refuse it" in t
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "m2"), _legs_status_doc(tmp_path / "petri_out", ledgered=(2, 4)), "petri_overflow_ledgers_moved"); assert "leg 10" in t and "neither done" in t
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "m3"), _legs_status_doc(tmp_path / "petri_out", ledgered=(2, 4, 10, 7)), "petri_overflow_ledgers_moved"); assert "leg 7" in t and "did not name it" in t
    # nothing ledgered any more (every named leg done) with no overflow unit: said by name, nothing to run
    t = _refused(tmp_path, monkeypatch, _state(tmp_path / "nl"), _legs_status_doc(tmp_path / "petri_out", ledgered=(), done=(2, 4, 10)), "petri_overflow_nothing_ledgered"); assert "9f" not in t
    # a control leg whose parent is not a frozen canonical leg
    _refused(tmp_path, monkeypatch, _state(tmp_path / "cp"), _legs_status_doc(tmp_path / "petri_out", parent_of={10: "ghost"}), "petri_overflow_control_parent")
    # the legs-status stub failing: the reading is unavailable, nothing is provably ledgered
    monkeypatch.setenv(PO.PETRI_RUN_ENV, str(_stub(tmp_path / "rc", doc, rc=2)))
    with pytest.raises(PO.Refused) as ei:
        PO.overflow_cells(str(_state(tmp_path / "rc")), MODEL, VIEW, python=sys.executable)
    assert ei.value.name == "petri_overflow_ledgers_moved" and "legs-status refused" in ei.value.text
    # a record_refused reading
    monkeypatch.setenv(PO.PETRI_RUN_ENV, str(_stub(tmp_path / "rr", {**doc, "record_refused": "foreign record"})))
    with pytest.raises(PO.Refused) as ei:
        PO.overflow_cells(str(_state(tmp_path / "rr")), MODEL, VIEW, python=sys.executable)
    assert ei.value.name == "petri_overflow_ledgers_moved" and "would refuse every leg" in ei.value.text


def test_the_cli_prints_the_decision_or_refuses_by_name(tmp_path, monkeypatch):
    st = _state(tmp_path)
    env = dict(os.environ, **{PO.PETRI_RUN_ENV: str(_stub(tmp_path, _legs_status_doc(tmp_path / "petri_out")))})
    r = subprocess.run([sys.executable, str(LIB / "petri_overflow.py"), "--state-root", str(st), "--model", MODEL, "--view", VIEW, "--python", sys.executable],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads(r.stdout)["cells"] == CELLS_2_4_10
    r = subprocess.run([sys.executable, str(LIB / "petri_overflow.py"), "--state-root", str(tmp_path / "none" / "state" / MODEL), "--model", MODEL, "--view", VIEW],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 3 and r.stderr.startswith("REFUSED: petri_overflow_first_pass_unbound:") and r.stdout == ""


# ------------------------------------------------------------------------------------------- a BOUND overflow unit: the declaration verbatim
def test_a_bound_overflow_unit_carries_its_declaration_verbatim_and_the_remainders_are_credited(tmp_path, monkeypatch):
    csv = ",".join(CELLS_2_4_10)
    # (i) the rerun proxy J2 drove leg 2 to completion and died: legs-status reads leg 2 done, legs 4 and 10 still ledgered -> the SAME declaration
    st = _state(tmp_path / "i", overflow_job="22222", overflow_args=csv)
    d = _cells(tmp_path / "i", monkeypatch, st, _legs_status_doc(tmp_path / "i" / "petri_out", ledgered=(4, 10), done=(2,)))
    assert d["cells"] == CELLS_2_4_10 and d["declaration_source"] == "bound-attempt" and d["overflow_bound"] is True and d["overflow_job_id"] == "22222"
    assert [l["idx"] for l in d["still_ledgered"]] == [4, 10] and [l["idx"] for l in d["done_legs"]] == [2]
    # (i2) legs 2 AND 4 done: a control-only remainder, the same declaration
    st2 = _state(tmp_path / "i2", overflow_job="22222", overflow_args=csv)
    d2 = _cells(tmp_path / "i2", monkeypatch, st2, _legs_status_doc(tmp_path / "i2" / "petri_out", ledgered=(10,), done=(2, 4)))
    assert d2["cells"] == CELLS_2_4_10 and [l["idx"] for l in d2["still_ledgered"]] == [10]
    # (j) every named leg done under a bound unit: nothing left, nothing to re-mint
    t = _refused(tmp_path / "j", monkeypatch, _state(tmp_path / "j", overflow_job="22222", overflow_args=csv), _legs_status_doc(tmp_path / "j" / "petri_out", ledgered=(), done=(2, 4, 10)),
                 "petri_overflow_complete")
    assert "22222" in t
    # (k) a bound declaration that does not cover a still-ledgered leg (a hand-submitted proxy)
    t = _refused(tmp_path / "k", monkeypatch, _state(tmp_path / "k", overflow_job="22222", overflow_args="prompted__si_constitution"), _legs_status_doc(tmp_path / "k" / "petri_out"),
                 "petri_overflow_declaration_short")
    assert "constv2ctl2-full__lora_r4_ce" in t and "none" in t
    # a bound attempt with no cells positional at all (a whole-baseline rerun by hand) refuses the same way; a transition token is reconciled first
    _refused(tmp_path / "k2", monkeypatch, _state(tmp_path / "k2", overflow_job="22222", overflow_args=None), _legs_status_doc(tmp_path / "k2" / "petri_out"), "petri_overflow_declaration_short")
    _refused(tmp_path / "tr", monkeypatch, _state(tmp_path / "tr", overflow_job="22222", overflow_args=csv, overflow_transition=True), _legs_status_doc(tmp_path / "tr" / "petri_out"),
             "petri_overflow_transition")


# ------------------------------------------------------------------------------------------------------- the plan row
def test_the_plan_emits_exactly_the_overflow_row_and_first_pass_rows_carry_no_pass(tmp_path):
    sha = "0" * 40
    rows = ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, clusters=["cluster_a", "cluster_b"], pass_="overflow", overflow_cells=CELLS_2_4_10, overflow_cluster="cluster_a")
    assert len(rows) == 1
    (r,) = rows
    assert r["unit"] == "overflow" and r["logical_work_id"] == OVER_LWID and r["name"] == f"f22-petri-{TAG}-rerun" and r["script"] == ST.PETRI_PROXY_SCRIPT
    assert r["args"] == [MODEL, VIEW, sha, "overflow-rerun", ",".join(CELLS_2_4_10)] and r["env"]["F22_REGISTRY_CELLS"] == ",".join(CELLS_2_4_10)
    assert r["pass"] == "overflow" and r["pass_of"] == FIRST_LWID and r["time_key"] == "petri" and r["cells"] == CELLS_2_4_10 and r["registers"]
    assert r["env"]["F22_REGISTRY_EVALS"].split(",") == list(ST.PETRI_EVALS)
    assert r["cluster"] == "cluster_a" and r["twin_ok"] is False        # pinned to the first pass's cluster, never `*`, never twinned
    first = ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, clusters=["cluster_a", "cluster_b"])
    assert len(first) == 1 and "pass" not in first[0] and first[0]["args"] == [MODEL, VIEW, sha] and first[0]["cluster"] == "*"      # the inventoried row is untouched
    with pytest.raises(ValueError, match="--family petri, no other family"):
        ST.unit_plan(MODEL, VIEW, families=["petri", "static"], sha=sha, pass_="overflow", overflow_cells=CELLS_2_4_10, overflow_cluster="cluster_a")
    with pytest.raises(ValueError, match="takes no --cells"):
        ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, pass_="overflow", overflow_cells=CELLS_2_4_10, cells=["none"], overflow_cluster="cluster_a")
    with pytest.raises(ValueError, match="no cells"):
        ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, pass_="overflow", overflow_cells=[], overflow_cluster="cluster_a")
    with pytest.raises(ValueError, match="are not in"):
        ST.overflow_rows(MODEL, VIEW, ["nowhere"], cluster="cluster_a", sha=sha)
    # the plan's TSV columns are untouched (06 and 09 read them positionally): the pass rides the JSON row only
    assert "pass" not in ST.PLAN_COLUMNS and len(ST.plan_tsv(rows).rstrip("\n").split("\t")) == len(ST.PLAN_COLUMNS)


# ------------------------------------------------------------------------------------------------------- the stage, dry runs
def _site(root: Path, state: Path | None = None) -> Path:
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={state or (root / 'state')}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _six(site, *args, env=None):
    r = run_stage("06_eval_waves.sh", "--dry-run", "--model", MODEL, "--view", VIEW, *args, site=site, env=env, timeout=600)
    return r, r.stdout + r.stderr


def test_a_bare_sites_dry_run_names_the_record_and_composes_nothing(tmp_path):
    """The README-sequence's shape: with no daemon record the dry run says what a real call needs and exits 0, writing no state."""
    site = _site(tmp_path)
    r, out = _six(site, "--family", "petri", "--pass", "overflow")
    assert r.returncode == 0, out[-2500:]
    assert "DRY-RUN: --pass overflow reads the daemon's record" in out and f"petri_drive.{TAG}.json" in out and "petri_overflow_no_record" in out
    assert "submitted " not in r.stdout and "0 unit(s) placed (overflow pass" in out
    assert not (tmp_path / "state").exists() or not list((tmp_path / "state").rglob("*"))


def test_the_dry_run_with_a_record_composes_the_rerun_proxy_line(tmp_path, monkeypatch):
    """With the daemon's record settled incomplete and the ledgers read, the dry run composes the ONE rerun proxy submission: the sbatch's positional 5
    = the ledgered legs' cells + `none`, F22_REGISTRY_CELLS the same, the petri walltime, the rerun name, the serving_rerun follow-up, no WAVE file."""
    st = _state(tmp_path)
    site = _site(tmp_path, state=tmp_path / "state")
    stub = _stub(tmp_path, _legs_status_doc(tmp_path / "petri_out"))
    r, out = _six(site, "--family", "petri", "--pass", "overflow", env={PO.PETRI_RUN_ENV: str(stub)})
    assert r.returncode == 0, out[-3000:]
    csv = ",".join(CELLS_2_4_10)
    subs = [ln for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    assert len(subs) == 1 and subs[0].split()[3] == "petri.overflow" and subs[0].split()[1] == "cluster_a", subs     # on the first pass's cluster
    assert f"petri_proxy.sbatch {MODEL} {VIEW} " in out and f" overflow-rerun {csv}" in out, out[-3000:]
    assert f"-J f22-petri-{TAG}-rerun" in out and f"F22_REGISTRY_CELLS={csv}" in out and "--time=1-00:00:00" in out
    assert "overflow pass: first pass job 11111 attempt 1 overflow_pending" in out and "still ledgered: 2:prompted__constitution, 4:si_constv2ctl2_full__lora_r4_ce, 10:none__lora0" in out
    assert "followup:" in out and f"petri/{TAG}/serving_rerun/endpoint.json" in out and "legs=ledgered" in out and "petri_drive_full22.sh" in out
    assert "the wave file is NOT written" in out and "frozen but not planned" not in out and "REFUSED" not in out
    assert not (st / f"WAVE.{VIEW}.json").exists()
    # the refusals through the stage, by name, before any plan
    for extra, needle in ((["--family", "static"], "--family petri, no other family"), (["--family", "petri", "--units", "petri.proxy"], "takes no --units"),
                          (["--family", "petri", "--cells", "none"], "takes no --cells"), (["--family", "petri", "--twin"], "takes no --twin"),
                          (["--family", "petri", "--pass", "bogus"], "must be first or overflow")):
        r2, out2 = _six(site, *([*extra, "--pass", "overflow"] if "--pass" not in extra else extra), env={PO.PETRI_RUN_ENV: str(stub)})
        assert r2.returncode == 2 and needle in out2, (extra, out2[-800:])
    # a record whose attempt is still running: refused by name through the stage
    (st / "grade" / f"petri_drive.{TAG}.json").write_text(json.dumps({**json.loads((st / "grade" / f"petri_drive.{TAG}.json").read_text()),
                                                                     "attempts": [{"n": 1, "state": "started", "identity": {"job_id": "11111"}}]}))
    r3, out3 = _six(site, "--family", "petri", "--pass", "overflow", env={PO.PETRI_RUN_ENV: str(stub)})
    assert r3.returncode == 2 and "petri_overflow_not_pending" in out3 and "submitted " not in r3.stdout


# ------------------------------------------------------------------------------------- the cluster: the first pass's
def test_the_rerun_proxy_is_pinned_to_the_first_pass_cluster_in_the_lib_and_the_plan(tmp_path, monkeypatch):
    """The first pass bound on the site's NON-primary cluster: the lib names it (`first_cluster`), the plan's row carries it (never `*`, never
    twinned), a universe or a --only-clusters selection that excludes it refuses by name, and a plan without it refuses by name. The proxy's
    object bank and bank view live on the first pass's cluster (05 --clusters), so a `*` row — expanded to every label and taken by the
    funnel's first label — ran where nothing was staged."""
    st = _state(tmp_path, cluster="cluster_b")
    d = _cells(tmp_path, monkeypatch, st, _legs_status_doc(tmp_path / "petri_out"))
    assert d["first_cluster"] == "cluster_b" and d["overflow_cluster"] is None and d["cells"] == CELLS_2_4_10
    # a bound overflow unit on ANOTHER cluster (a placement from before this rule): the pin is still the first pass's; the bound cluster is reported
    st2 = _state(tmp_path / "b", cluster="cluster_b", overflow_job="22222", overflow_args=",".join(CELLS_2_4_10), overflow_cluster="cluster_a")
    d2 = _cells(tmp_path / "b", monkeypatch, st2, _legs_status_doc(tmp_path / "b" / "petri_out", ledgered=(4, 10), done=(2,)))
    assert d2["first_cluster"] == "cluster_b" and d2["overflow_cluster"] == "cluster_a" and d2["overflow_bound"] is True and d2["cells"] == CELLS_2_4_10
    # a first-pass binding that names no cluster (a torn / hand-written binding): refused by name, nothing derived
    st3 = _state(tmp_path / "c", cluster="cluster_b")
    bp = Path(ST.bindings_dir(str(st3 / "submissions.jsonl"))) / f"{ST.safe_id(FIRST_LWID)}.submitted.json"
    rec = json.loads(bp.read_text()); rec.pop("cluster"); bp.write_text(json.dumps(rec))
    t = _refused(tmp_path / "c", monkeypatch, st3, _legs_status_doc(tmp_path / "c" / "petri_out"), "petri_overflow_first_pass_cluster_unknown")
    assert "names no cluster" in t and "11111" in t
    # the plan: the row is PINNED to the named cluster; the TSV's cluster column carries it (06's loop expands `*` alone) and twin_ok is 0
    sha = "0" * 40
    (r,) = ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, clusters=["cluster_a", "cluster_b"], pass_="overflow", overflow_cells=CELLS_2_4_10, overflow_cluster="cluster_b")
    assert r["cluster"] == "cluster_b" and r["twin_ok"] is False
    cols = ST.plan_tsv([r]).rstrip("\n").split("\t")
    assert cols[ST.PLAN_COLUMNS.index("cluster")] == "cluster_b" and cols[ST.PLAN_COLUMNS.index("twin_ok")] == "0"
    (r2,) = ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, clusters=["cluster_a", "cluster_b"], only_clusters=["cluster_b"], pass_="overflow",
                         overflow_cells=CELLS_2_4_10, overflow_cluster="cluster_b")
    assert r2["cluster"] == "cluster_b"
    with pytest.raises(ValueError, match="follows the first pass's cluster 'cluster_b', but --only-clusters selects"):
        ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, clusters=["cluster_a", "cluster_b"], only_clusters=["cluster_a"], pass_="overflow",
                     overflow_cells=CELLS_2_4_10, overflow_cluster="cluster_b")
    with pytest.raises(ValueError, match="not in the placement universe"):
        ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, clusters=["cluster_a", "cluster_b"], pass_="overflow", overflow_cells=CELLS_2_4_10, overflow_cluster="cluster_c")
    for missing in (None, "", "*"):
        with pytest.raises(ValueError, match="needs --overflow-cluster"):
            ST.unit_plan(MODEL, VIEW, families=["petri"], sha=sha, clusters=["cluster_a", "cluster_b"], pass_="overflow", overflow_cells=CELLS_2_4_10, overflow_cluster=missing)
    # the CLI: the same refusal by name with nothing planned; with the label, the pinned row
    base = [sys.executable, str(LIB / "stage.py"), "plan", "--model", MODEL, "--view", VIEW, "--families", "petri", "--sha", sha, "--clusters", "cluster_a,cluster_b",
            "--pass", "overflow", "--overflow-cells", ",".join(CELLS_2_4_10), "--json"]
    r3 = subprocess.run(base, capture_output=True, text=True, timeout=600)
    assert r3.returncode == 2 and "needs --overflow-cluster" in r3.stderr and r3.stdout == "", (r3.stdout, r3.stderr[-800:])
    r4 = subprocess.run(base + ["--overflow-cluster", "cluster_b"], capture_output=True, text=True, timeout=600)
    assert r4.returncode == 0, r4.stderr[-800:]
    assert [row["cluster"] for row in json.loads(r4.stdout)] == ["cluster_b"]
    r5 = subprocess.run(base + ["--overflow-cluster", "cluster_b", "--only-clusters", "cluster_a"], capture_output=True, text=True, timeout=600)
    assert r5.returncode == 2 and "follows the first pass's cluster" in r5.stderr and r5.stdout == ""


def test_the_stage_places_the_rerun_proxy_on_the_first_pass_cluster(tmp_path, monkeypatch):
    """Through 06: the first pass bound on cluster_b (the site's non-primary label). Without --clusters the dry run submits the rerun proxy on
    cluster_b and mirrors the sha there alone; --clusters naming cluster_b (alone or with cluster_a, in either order) does the same; --clusters
    cluster_a is refused by name after the record is read, nothing submitted."""
    st = _state(tmp_path, cluster="cluster_b")
    site = _site(tmp_path, state=tmp_path / "state")
    stub = _stub(tmp_path, _legs_status_doc(tmp_path / "petri_out"))
    env = {PO.PETRI_RUN_ENV: str(stub)}
    for extra in ((), ("--clusters", "cluster_b"), ("--clusters", "cluster_a,cluster_b"), ("--clusters", "cluster_b,cluster_a")):
        r, out = _six(site, "--family", "petri", "--pass", "overflow", *extra, env=env)
        assert r.returncode == 0, (extra, out[-3000:])
        subs = [ln for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
        assert len(subs) == 1 and subs[0].split()[1] == "cluster_b" and subs[0].split()[3] == "petri.overflow", (extra, subs)
        assert "overflow pass: first pass job 11111 attempt 1 overflow_pending on cluster_b;" in out, (extra, out[-3000:])
        assert "clusters=cluster_b (placement over cluster_a,cluster_b)" in out, (extra, out[-3000:])
        assert "DRY-RUN: ssh cluster-b-login" in out and "DRY-RUN: ssh cluster-a-login" not in out, (extra, out[-3000:])   # the mirror goes to the first pass's cluster alone
        assert "REFUSED" not in out and not (st / f"WAVE.{VIEW}.json").exists()
    r2, out2 = _six(site, "--family", "petri", "--pass", "overflow", "--clusters", "cluster_a", env=env)
    assert r2.returncode == 2 and "petri_overflow_cluster_excluded" in out2 and "'cluster_b'" in out2 and "job 11111" in out2, out2[-1500:]
    assert "submitted " not in r2.stdout and "DRY-RUN: ssh" not in out2
    # the bare-site dry run says where the proxy would go
    r3, out3 = _six(_site(tmp_path / "bare"), "--family", "petri", "--pass", "overflow")
    assert r3.returncode == 0 and "on the cluster the first pass is bound on" in out3, out3[-1500:]
    # a BOUND overflow unit placed on the other cluster before this rule (job 22222 on cluster_a; the first pass on cluster_b): the pass still
    # pins cluster_b — the stage says the bound proxy moves back, mirrors to cluster_b alone, and the funnel's idempotence skips the bound unit
    # (a --resubmit after its death is the fenced cross-cluster replacement, 24d); nothing goes to cluster_a
    st4 = _state(tmp_path / "moved", cluster="cluster_b", overflow_job="22222", overflow_args=",".join(CELLS_2_4_10), overflow_cluster="cluster_a")
    site4 = _site(tmp_path / "moved", state=tmp_path / "moved" / "state")
    stub4 = _stub(tmp_path / "moved", _legs_status_doc(tmp_path / "moved" / "petri_out", ledgered=(4, 10), done=(2,)))
    r4, out4 = _six(site4, "--family", "petri", "--pass", "overflow", env={PO.PETRI_RUN_ENV: str(stub4)})
    assert r4.returncode == 0, out4[-3000:]
    assert "overflow pass: first pass job 11111 attempt 1 overflow_pending on cluster_b;" in out4 and "declaration (bound-attempt)" in out4, out4[-3000:]
    assert "the bound overflow proxy (job 22222) sits on cluster_a and moves back to cluster_b" in out4, out4[-3000:]
    assert "clusters=cluster_b (placement over cluster_a,cluster_b)" in out4 and "DRY-RUN: ssh cluster-b-login" in out4 and "DRY-RUN: ssh cluster-a-login" not in out4
    assert "skipped cluster_a 22222 petri.overflow (already submitted" in out4 and "submitted " not in r4.stdout, out4[-3000:]
    assert "REFUSED" not in out4 and not (st4 / f"WAVE.{VIEW}.json").exists()


# ------------------------------------------ the token: an interrupted submission recovers through the front door
def _real_world(tmp_path):
    """The wave-integration World (the REAL stages against fake Slurm; a committed copy of reproduce/ is the repo the stages run from), with the
    demo view's Petri manifest in the checkout — the preflight's done-ness reading names it — committed, so the tree is clean at the wave sha."""
    from test_front_door_wave_integration import World as WaveWorld
    w = WaveWorld(tmp_path)
    man = w.repo / "configs" / "capeval" / "manifests"; man.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPRO.parent / "configs" / "capeval" / "manifests" / f"petri_full22_{TAG}.yaml", man)
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]
    subprocess.run([*g, "add", "-A"], cwd=w.repo, check=True); subprocess.run([*g, "commit", "-q", "-m", "the demo view's Petri manifest"], cwd=w.repo, check=True)
    w.sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True, check=True).stdout.strip()
    return w


def test_an_interrupted_overflow_submit_recovers_through_the_front_door(tmp_path, monkeypatch):
    """A `06 --pass overflow --resubmit` interrupted after its sbatch left the overflow unit's transition token, and the
    preflight (lib/petri_overflow.py) refused `petri_overflow_transition` BEFORE 06 reached the funnel — the one path that reconciles a token —
    so the same command refused forever and Petri could not be completed without a hand reconciliation. Through the REAL stage against fake
    Slurm: the overflow proxy J2 is submitted and bound; it dies unregistered; `--resubmit` is interrupted right after sbatch accepted J3 (the
    token stands, J3 exists, nothing records it); a dry run says so and writes nothing; the SAME real command reconciles the token through the
    funnel first (J3 adopted by its nonce read back, bound), reads the bound declaration verbatim, and the funnel's walk finds J3 live — one J3,
    no J4."""
    from front_door_testlib import job_ids, set_job_state
    w = _real_world(tmp_path)
    st = _state(tmp_path, named=(2, 10))                     # the first pass bound (job 11111 on cluster_a); its drive settled incomplete on legs 2 and 10
    stub = _stub(tmp_path, _legs_status_doc(tmp_path / "petri_out", ledgered=(2, 10)))
    env = {"PYTHONPATH": str(REPRO.parent / "src"), "F22_ROSTER": str(REPRO.parent / "configs" / "sdfbase_full22_roster.yaml"), PO.PETRI_RUN_ENV: str(stub)}
    subs = str(st / "submissions.jsonl")

    def six(*args, **extra):
        return w.stage("06_eval_waves.sh", "--model", MODEL, "--view", VIEW, "--family", "petri", "--pass", "overflow", *args, **env, **extra)
    csv = "none,prompted__si_constitution"                     # leg 2's cell + leg 10's parent `none` (the World's config: none, prompted__safety_refusal, prompted__si_constitution)
    # ---- J2: the first mint of the overflow proxy, through the funnel, on the first pass's cluster
    r1 = six(); out1 = r1.stdout + r1.stderr
    assert r1.returncode == 0, out1[-4000:]
    assert "submitted cluster_a 1001 petri.overflow" in r1.stdout and f"declaration (computed): {csv}" in out1, out1[-4000:]
    assert ST.binding_state(subs, OVER_LWID)["job_id"] == "1001"
    argv = (w.fake["cluster_a"] / "jobs" / "1001.argv").read_text(); assert f" overflow-rerun {csv}" in argv and f"-J f22-petri-{TAG}-rerun" in argv, argv
    # ---- J2 dies before registering anything: provably dead (sacct terminal; cluster_a's producer spool traversable; the ingest watermark past its end)
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    (w.shared["cluster_a"] / "registry" / "records").mkdir(parents=True, exist_ok=True)
    reg_state = w.tmp / "registry" / "state"; reg_state.mkdir(exist_ok=True)
    (reg_state / "mirror_status.json").write_text(json.dumps({"cluster_a": {"ingested_through_utc": "2026-01-01T00:20:00"}}))
    # ---- the --resubmit interrupted right after sbatch accepted J3: the token stands, J3 exists, nothing records it; the pass reports the submission failed
    r2 = six("--resubmit", SISTEER_TWIN_CRASH_AT="submit_after_sbatch"); out2 = r2.stdout + r2.stderr
    assert r2.returncode != 0 and "CRASH INJECTED at submit_after_sbatch" in out2 and "submitted cluster_a 1002 petri.overflow" in r2.stdout, out2[-4000:]
    assert f"declaration (bound-attempt): {csv}" in out2 and "RESUBMIT " in out2, out2[-4000:]
    b = ST.binding_state(subs, OVER_LWID); assert b["state"] == "transition" and b["transition"]["intended_attempt"] == f"{OVER_LWID}@cluster_a@2" and b["transition"]["cluster"] == "cluster_a", b
    assert job_ids(w.fake["cluster_a"]) == [1001, 1002]
    # ---- a dry run with the token standing: says so, decides nothing, writes nothing (the token stands)
    r3 = six("--resubmit", "--dry-run"); out3 = r3.stdout + r3.stderr
    assert r3.returncode == 0 and f"DRY-RUN: the overflow unit {OVER_LWID} is mid re-run (a transition token" in out3 and f"reconcile-token --lwid {OVER_LWID}" in out3, out3[-3000:]
    assert "0 unit(s) placed (overflow pass, dry run with the overflow unit's transition token standing)" in out3 and "petri_overflow_transition" not in out3, out3[-3000:]
    assert not [ln for ln in r3.stdout.splitlines() if ln.startswith("submitted ")], out3[-3000:]
    assert ST.binding_state(subs, OVER_LWID)["state"] == "transition"
    # ---- the SAME real command: the token is reconciled THROUGH the funnel first (J3 adopted by its nonce, bound), the declaration is read from the
    #      bound attempt verbatim, and the walk finds J3 live — nothing minted twice. Before this fix: `petri_overflow_transition`, on every call.
    r4 = six("--resubmit"); out4 = r4.stdout + r4.stderr
    assert r4.returncode == 0, out4[-5000:]
    assert f"overflow pass: the overflow unit {OVER_LWID} is mid re-run (a transition token) — reconciled through the funnel" in out4, out4[-5000:]
    assert f"RECONCILE {OVER_LWID}: the transition token's job exists — f22-petri-{TAG}-rerun 1002 on cluster_a" in out4 and f"ADOPTED as {OVER_LWID}@cluster_a@2" in out4, out4[-5000:]
    assert "token\tadopted\tnonce" in r4.stdout and "petri_overflow_transition" not in out4, out4[-5000:]
    assert f"declaration (bound-attempt): {csv}" in out4 and "skipped cluster_a 1002 petri.overflow (still RUNNING" in r4.stdout, out4[-5000:]
    assert not [ln for ln in r4.stdout.splitlines() if ln.startswith("submitted ")], out4[-5000:]      # the `submitted <cluster> <job> <unit>` line: nothing minted this call
    assert job_ids(w.fake["cluster_a"]) == [1001, 1002], "a duplicate proxy was launched"
    b = ST.binding_state(subs, OVER_LWID); assert b["state"] == "submitted" and b["job_id"] == "1002" and b["cluster"] == "cluster_a", b
    atts = [e for e in w.attempts() if e["event"] == "attempt" and e["logical_work_id"] == OVER_LWID]
    assert [a["attempt_id"] for a in atts] == [f"{OVER_LWID}@cluster_a@1", f"{OVER_LWID}@cluster_a@2"] and atts[-1]["identified_by"] == "nonce" and atts[-1]["spec"]["args"][4] == csv, atts
    assert not (st / f"WAVE.{VIEW}.json").exists()
    # the preflight alone (a direct call, the same legs-status stub) still refuses a standing token by name, pointing at the funnel's
    # reconciliation — 06 is what runs it first
    ST.write_transition(subs, {"logical_work_id": OVER_LWID, "intended_attempt": f"{OVER_LWID}@cluster_a@3", "utc": "2026-01-01T00:00:00Z"})
    monkeypatch.setenv(PO.PETRI_RUN_ENV, str(stub))
    with pytest.raises(PO.Refused) as ei:
        PO.overflow_cells(str(st), MODEL, VIEW, python=sys.executable, repo=str(w.repo))
    assert ei.value.name == "petri_overflow_transition" and f"reconcile-token --lwid {OVER_LWID}" in ei.value.text, ei.value.text


def test_the_readme_names_the_pass_and_the_proxy_header_names_the_subset_rule():
    readme = " ".join((REPRO / "README.md").read_text().split())          # the page's words, however its lines wrap
    assert f"reproduce/06_eval_waves.sh --model {MODEL} --view {VIEW} --family petri --pass overflow" in readme
    assert "SISTEER_TIME_PETRI" in readme and "--resubmit` re-mints a Petri proxy whose dead drive already filed some legs" in readme
    assert "An overflow-pass submit interrupted after its reservation is reconciled by the next `--pass overflow` call itself" in readme
    assert "reconciles it through\n#              the funnel (submit_twin.py reconcile-token) BEFORE the preflight reads the declaration" in (REPRO / "06_eval_waves.sh").read_text()
    proxy = (REPRO.parent / "scripts/sdfbase/full22/petri_proxy.sbatch").read_text()
    assert "the overflow pass declares the ledgered cells plus `none`" in proxy
    six = (REPRO / "06_eval_waves.sh").read_text()
    assert "--pass) PASS=\"$2\"; shift 2;;" in six and '[ "$VIEW" = "$BASE_VIEW" ] && [ "$PASS" = first ]' in six
