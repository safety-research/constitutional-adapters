"""06 --gate's hand-offs are FILES, never argv strings: Linux caps ONE argv string at
MAX_ARG_STRLEN = 131,072 bytes, and the smoke's static pending files are 149,440 B each, the climb's four task files ~146 KB together, mmlu's ten
1,531-id probe documents far more — `Argument list too long` left the substitution empty and the shell read it as "the pending file names no
runs" / `probe: null`. Pins: `stage.py arg_doc` (`@<path>` / `@-`), `pending-runs` (stdout = the runs list and nothing else; the MISSING paths in
their own file; a file that does not parse REFUSES by name, never an empty list), `probe-append` + `coverage --probes-jsonl` (the per-call results
in a file; an exec failure is recorded by its rc and stderr, never as "printed no JSON"), `gate-contract` (structural; the admit-once
sidecar; a predicate / identity / unit-set change refuses by name, a receipted rerun stamp amends), `verdict` (both shas + the contract,
required for a 06 family, not for stage 07's bcb_bfcl/<view>). Synthetic ids and paths throughout."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPRO, make_registry  # noqa: E402
from test_admission import MODEL, SHA, by_unit, freeze, satisfy  # noqa: E402

sys.path.insert(0, str(LIB))
import admission as A  # noqa: E402
import coverage_probe as CP  # noqa: E402
import stage as ST  # noqa: E402

MAX_ARG_STRLEN = 131072
STAGE = [sys.executable, str(LIB / "stage.py")]


def _cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([*STAGE, *args], capture_output=True, text=True, input=stdin, timeout=120)


def _pending(n_runs: int, eval_id: str = "pasttense", pad: int = 0, run_prefix: str = "r") -> dict:
    """A producer's pending file (the registration identity): runs[].{run_id, plan, route}; `pad` bytes of plan.extra per run."""
    return {"run_id": f"{run_prefix}0", "runs": [{"run_id": f"{run_prefix}{i}", "plan": {"eval_id": eval_id, "extra": {"pad": "x" * pad}}, "route": {"object": "none", "eval_id": eval_id}}
                                                for i in range(n_runs)]}


def _capture(files: dict) -> str:
    """06 --gate's framed capture of several pending files: `PENDING <f>` + the text (+ a newline), or `MISSING <f>`."""
    out = []
    for f, doc in files.items():
        out.append(f"MISSING {f}\n" if doc is None else f"PENDING {f}\n{json.dumps(doc, indent=1)}\n")
    return "".join(out)


# ------------------------------------------------------------------------------------------------------- arg_doc + pending-runs
def test_arg_doc_reads_a_file_or_stdin_and_keeps_literals(tmp_path):
    big = json.dumps({"k": "v" * 200_000})
    assert len(big) > MAX_ARG_STRLEN
    p = tmp_path / "doc.json"; p.write_text(big)
    assert ST.arg_doc(f"@{p}") == big and ST.arg_doc("literal") == "literal" and ST.arg_doc(None) is None
    r = _cli("array-tasks", "--sacct", "@-", stdin="1|1_0|COMPLETED\n")
    assert r.returncode == 0 and json.loads(r.stdout) == [{"raw": "1", "task": 0, "state": "COMPLETED"}]


def test_pending_runs_reads_a_plain_job_file_larger_than_one_argv_string(tmp_path):
    doc = _pending(45, pad=4000)
    cap = tmp_path / "cap.txt"; cap.write_text(_capture({"/shared/registry/pending/984288.json": doc}))
    assert cap.stat().st_size > MAX_ARG_STRLEN, cap.stat().st_size                                  # the S4 shape: one file over the cap
    miss = tmp_path / "miss.json"
    r = _cli("pending-runs", "--capture", str(cap), "--missing-out", str(miss))
    assert r.returncode == 0, r.stderr
    runs = json.loads(r.stdout)                                                                         # stdout is ONE JSON document
    assert [x["run_id"] for x in runs] == [f"r{i}" for i in range(45)] and all(x["eval_id"] == "pasttense" and x["object"] == "none" and x["array_task"] is None for x in runs)
    assert json.loads(miss.read_text()) == []


def test_pending_runs_array_capture_over_the_cap_with_one_missing_task_file(tmp_path):
    tasks = [{"raw": "3001", "task": 0, "state": "COMPLETED"}, {"raw": "3002", "task": 1, "state": "COMPLETED"}, {"raw": "3003", "task": 2, "state": "RUNNING"}, {"raw": "3004", "task": 3, "state": "RUNNING"}]
    files = {f"/shared/registry/pending/{t['raw']}.json": (None if t["task"] == 3 else _pending(1, "strongreject_bon", pad=48_000, run_prefix=f"c{t['task']}-")) for t in tasks}   # 3 x 48 KB > the cap
    cap = tmp_path / "cap.txt"; cap.write_text(_capture(files))
    assert cap.stat().st_size > MAX_ARG_STRLEN
    tj = tmp_path / "tasks.json"; tj.write_text(json.dumps(tasks)); miss = tmp_path / "miss.json"
    r = _cli("pending-runs", "--capture", str(cap), "--missing-out", str(miss), "--tasks-json", f"@{tj}")
    assert r.returncode == 0, r.stderr
    runs = json.loads(r.stdout)
    assert [(x["run_id"], x["array_task"]) for x in runs] == [("c0-0", 0), ("c1-0", 1), ("c2-0", 2)]
    assert json.loads(miss.read_text()) == ["/shared/registry/pending/3004.json"]                     # the MISSING list in ITS file, never on stdout
    assert "MISSING" not in r.stdout


def test_pending_runs_missing_only_is_not_yet_and_bad_json_or_an_empty_capture_refuse_by_name(tmp_path):
    miss = tmp_path / "miss.json"
    cap = tmp_path / "cap.txt"; cap.write_text("MISSING /shared/registry/pending/77.json\n")
    r = _cli("pending-runs", "--capture", str(cap), "--missing-out", str(miss))
    assert r.returncode == 0 and json.loads(r.stdout) == [] and json.loads(miss.read_text()) == ["/shared/registry/pending/77.json"]
    # a file that does not parse: REFUSED naming the file and the JSON error — never `[]` (the old shell's "names no runs")
    cap.write_text("PENDING /shared/registry/pending/78.json\n{\"runs\": [{\"run_id\": \"r1\"\n")
    r = _cli("pending-runs", "--capture", str(cap), "--missing-out", str(miss))
    assert r.returncode == 2 and "REFUSED: pending-runs: pending file /shared/registry/pending/78.json:" in r.stderr and "nothing bound" in r.stderr and r.stdout == ""
    # a parsed file whose runs list is EMPTY (every run skipped): `[]`, rc 0, said so on stderr — the shell then refuses the empty registration by name
    cap.write_text(_capture({"/shared/registry/pending/79.json": {"job_id": "79", "runs": [], "skipped": [{"why": "x"}, {"why": "y"}]}}))
    r = _cli("pending-runs", "--capture", str(cap), "--missing-out", str(miss))
    assert r.returncode == 0 and json.loads(r.stdout) == [] and "names no run" in r.stderr and "skipped: 2" in r.stderr
    # an EMPTY capture (the remote command printed nothing): a refusal, never "no pending file yet"
    cap.write_text("")
    r = _cli("pending-runs", "--capture", str(cap), "--missing-out", str(miss))
    assert r.returncode == 2 and "is empty" in r.stderr


# ------------------------------------------------------------------------------------------------- probe-append + coverage --probes-jsonl
def _world(tmp_path: Path, families=("mmlu",)):
    state = tmp_path / "state"; doc = freeze(state, families=list(families))
    return state, by_unit(doc)


def _register(state: Path, unit: dict, rids: list) -> None:
    lw = unit["logical_work_id"]; ledger = str(state / "submissions.jsonl")
    ST.write_binding(ledger, ST.submitted_record(lw, "1", "cluster_a", f"{lw}@cluster_a@1", array=unit.get("array")))
    ST.write_binding(ledger, {"logical_work_id": lw, "state": "registered", "run_ids": rids, "job_id": "1", "cluster": "cluster_a",
                              "runs": [{"run_id": r, "eval_id": next(iter(unit["evals"])), "object": "none", "array_task": None} for r in rids]})


def test_probe_append_and_coverage_fold_ten_full_width_documents_through_a_file(tmp_path):
    state, u = _world(tmp_path)
    unit = u["mmlu/mmlu"]; lw = unit["logical_work_id"]
    rids = [f"run-mmlu-{i}" for i in range(10)]; _register(state, unit, rids)
    ids = [f"mmlu-{i:04d}" for i in range(1531)]
    results = tmp_path / "results.jsonl"
    for i, rid in enumerate(rids):
        doc = {"layout": "capability_cellstore", "path": f"/stores/{rid}", "exists": True, "shards": {}, "notes": [],
               "items": {"present": True, "task": "s2parity/mmlu_letter", "completed": ids, "n_rows": 1532, "tasks_seen": {"s2parity/mmlu_letter": 1532}, "metrics_seen": ["acc"],
                         "items_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "n_attested_items": 1531, "mmlu_item_set_spec": {"n": 1531}}}
        out = tmp_path / f"p{i}.out"; err = tmp_path / f"p{i}.err"; out.write_text(json.dumps(doc) + "\n"); err.write_text("")
        r = _cli("probe-append", "--results", str(results), "--run-id", rid, "--array-task", "", "--out-file", str(out), "--err-file", str(err), "--rc", "0", "--store-source", "registry.run.store_uri")
        assert r.returncode == 0, r.stderr
    assert results.stat().st_size > MAX_ARG_STRLEN                                                   # the S4 overflow: ten documents, one file
    runs = [{"run_id": r, "eval_id": "mmlu", "object": "none", "array_task": None, "store_uri": f"/stores/{r}", "successor": None} for r in rids]
    runs_f = tmp_path / "runs.json"; runs_f.write_text(json.dumps(runs))
    contract = {"version": "cc1", "sha256": "c" * 64, "record_sha256": "c" * 64, "source": "frozen", "admitted_utc": None, "amendments": []}
    cf = tmp_path / "contract.json"; cf.write_text(json.dumps(contract))
    r = _cli("coverage", "--state-root", str(state), "--logical-work-id", lw, "--runs-json", f"@{runs_f}", "--probes-jsonl", str(results),
             "--probe-sha", "p" * 40, "--probe-sha256", "f" * 64, "--inventory-sha", SHA, "--contract", f"@{cf}")
    assert r.returncode == 0, r.stderr
    cov = json.load(open(ST.coverage_path(str(state), lw)))
    assert cov["probe_code_sha"] == "p" * 40 and cov["probe_sha256"] == "f" * 64 and cov["inventory_code_sha"] == SHA and cov["coverage_contract"] == contract
    assert "ignored (D3003 (1))" in cov["frozen_path_spec"]
    for rid in rids:
        c = cov["runs"][rid]
        assert isinstance(c["probe"], dict) and len(c["probe"]["items"]["completed"]) == 1531 and c["probe_error"] is None
        assert c["probe_rc"] == 0 and c["store_source"] == "registry.run.store_uri" and c["probe_stderr_head"] is None


def test_probe_append_records_an_exec_failure_by_its_rc_and_stderr_never_as_no_json_and_its_status_says_whether_a_document_was_read(tmp_path):
    """Every call is RECORDED (one line each); the status then says whether a document was READ: rc 3 = none was — the call exited non-zero, or
    exited 0 and printed no JSON object (a login banner before the document, a traceback on stdout, a JSON array); rc 0 = a document, or `--why`
    (a `none` host: the probe never ran, a coverage FACT). The rc-0 no-JSON case used to return 0, so 06 --gate's per-family
    watermark never moved and the pass wrote a fresh `pass` verdict over a record that names the failure per run."""
    results = tmp_path / "results.jsonl"
    out = tmp_path / "p.out"; err = tmp_path / "p.err"
    out.write_text(""); err.write_text("bash: /shared/scripts/registry/coverage_probe.deadbeef.py: No such file or directory\n")
    assert _cli("probe-append", "--results", str(results), "--run-id", "r1", "--out-file", str(out), "--err-file", str(err), "--rc", "127").returncode == 3
    out.write_text(""); err.write_text("")
    assert _cli("probe-append", "--results", str(results), "--run-id", "r2", "--out-file", str(out), "--err-file", str(err), "--rc", "255").returncode == 3
    out.write_text("Traceback (most recent call last):\n"); err.write_text("")
    assert _cli("probe-append", "--results", str(results), "--run-id", "r3", "--out-file", str(out), "--err-file", str(err), "--rc", "0").returncode == 3
    out.write_text("Welcome to TESTHOST — scheduled maintenance tonight\n" + json.dumps({"layout": "capability_cellstore", "path": "/s", "exists": True}) + "\n"); err.write_text("")
    assert _cli("probe-append", "--results", str(results), "--run-id", "r3b", "--out-file", str(out), "--err-file", str(err), "--rc", "0").returncode == 3    # a banner before the document
    out.write_text("[1, 2]\n"); err.write_text("")
    assert _cli("probe-append", "--results", str(results), "--run-id", "r3c", "--out-file", str(out), "--err-file", str(err), "--rc", "0").returncode == 3    # JSON, not a document
    out.write_text(""); err.write_text("")
    assert _cli("probe-append", "--results", str(results), "--run-id", "r4", "--out-file", str(out), "--err-file", str(err), "--rc", "0", "--why", "run r4: not covered, no store on record", "--store-source", "none").returncode == 0
    out.write_text(json.dumps({"layout": "bon_store512", "path": "/s", "exists": True, "shards": {}, "notes": []})); err.write_text("")
    assert _cli("probe-append", "--results", str(results), "--run-id", "r5", "--array-task", "3", "--out-file", str(out), "--err-file", str(err), "--rc", "0").returncode == 0
    lines = [json.loads(l) for l in results.read_text().splitlines()]
    assert [l["run_id"] for l in lines] == ["r1", "r2", "r3", "r3b", "r3c", "r4", "r5"]                                      # every call recorded, rc 3 or not
    by = {l["run_id"]: l for l in lines}
    assert by["r1"]["probe"] is None and by["r1"]["probe_error"].startswith("the probe exited 127: bash: /shared/scripts/registry/coverage_probe.deadbeef.py") and by["r1"]["probe_rc"] == 127
    assert "No such file" in by["r1"]["probe_stderr_head"] and "printed no JSON" not in by["r1"]["probe_error"]
    assert by["r2"]["probe_error"].startswith("the probe exited 255 (ssh could not reach the host") and by["r2"]["probe_rc"] == 255
    assert by["r3"]["probe"] is None and by["r3"]["probe_error"].startswith("the probe printed no JSON document (rc 0;")
    assert by["r3b"]["probe"] is None and by["r3b"]["probe_rc"] == 0 and by["r3b"]["probe_error"].startswith("the probe printed no JSON document (rc 0;") and "Welcome to TESTHOST" in by["r3b"]["probe_error"]
    assert by["r3c"]["probe"] is None and by["r3c"]["probe_error"].startswith("the probe printed no JSON document (rc 0;")
    assert by["r4"]["probe"] is None and by["r4"]["probe_error"] == "run r4: not covered, no store on record" and by["r4"]["store_source"] == "none" and by["r4"]["probe_rc"] == 0
    assert by["r5"]["probe"]["layout"] == "bon_store512" and by["r5"]["probe_error"] is None and by["r5"]["array_task"] == 3


def test_coverage_names_a_run_the_results_file_never_reached(tmp_path):
    state, u = _world(tmp_path)
    unit = u["mmlu/mmlu"]; lw = unit["logical_work_id"]
    _register(state, unit, ["ra", "rb"])
    results = tmp_path / "results.jsonl"; results.write_text(json.dumps({"run_id": "ra", "array_task": None, "probe": {"layout": "capability_cellstore"}, "probe_error": None, "probe_rc": 0}) + "\n")
    runs = [{"run_id": r, "eval_id": "mmlu", "object": "none", "array_task": None, "store_uri": f"/s/{r}"} for r in ("ra", "rb")]
    r = _cli("coverage", "--state-root", str(state), "--logical-work-id", lw, "--runs-json", json.dumps(runs), "--probes-jsonl", str(results))
    assert r.returncode == 0, r.stderr
    cov = json.load(open(ST.coverage_path(str(state), lw)))
    assert cov["runs"]["ra"]["probe"] == {"layout": "capability_cellstore"}
    assert cov["runs"]["rb"]["probe"] is None and cov["runs"]["rb"]["probe_error"] == "no probe result recorded for run rb (the gate loop never reached it) — re-run 06 --gate"


# ---------------------------------------------------------------------------------------------------------------- the coverage contract
def test_coverage_contract_carries_every_identity_admission_checks_and_ignores_the_readers_locations(tmp_path):
    state, u = _world(tmp_path, families=("controls", "mmlu", "bon", "agentic", "static"))
    units = list({**u}.values())
    base = ST.coverage_contract(units, CP.DONE_PREDICATES)
    assert base["version"] == "cc1" and set(base) == {"version", "sha256", "unit_set", "identities", "done_predicates"} and base["unit_set"] == sorted(x["logical_work_id"] for x in units)
    ident = base["identities"][u["controls/t0"]["logical_work_id"]]
    for k in ("instrument_id", "decode_policy_ref", "thinking_mode", "expected_shards", "layout", "item_set_id", "items_sha256", "n_items", "task", "n_max", "nshards", "n_samples", "behavior", "expected_ids_sha256", "n_expected"):
        assert k in ident["evals"]["xstest"], k                                                        # unit_contract's keys + the coverage identity keys
    pt = base["identities"][u["static/t0"]["logical_work_id"]]["evals"]["pasttense"]
    assert set(pt["expected_ids_sha256"]) == set(u["static/t0"]["evals"]["pasttense"]["expected_shards"]) and pt["n_expected"]["shard00"] == 29
    assert "path" not in ident["evals"]["xstest"] and "host" not in ident["evals"]["xstest"] and "expected_source" not in ident["evals"]["xstest"]

    def sha_after(mutate):
        import copy
        us = copy.deepcopy(units); mutate({x["logical_work_id"].split("/", 2)[2]: x for x in us}); return ST.coverage_contract(us, CP.DONE_PREDICATES)["sha256"]
    # a READER / layout fix changes nothing: the frozen path spec, the host, the source note, a hold-out stamp, the sbatch name, the tree
    assert sha_after(lambda d: d["controls/t0"]["evals"]["xstest"]["coverage"].update(path={"root_rel": "/elsewhere/<cell>"}, host="grading_host", expected_source="other")) == base["sha256"]
    assert sha_after(lambda d: d["static/t0"].update(held_out={"pasttense": {"source": "ruling"}}, sbatch_name="x", code_sha="b" * 40, workload={})) == base["sha256"]
    # every identity admission checks moves the sha: instrument_id, decode_policy_ref, thinking_mode, n_items, items_sha256, an expected id, a unit dropped, a predicate
    for mut in (lambda d: d["controls/t0"]["evals"]["xstest"].update(instrument_id="0" * 64),
                lambda d: d["controls/t0"]["evals"]["xstest"].update(decode_policy_ref="qwen35_4b_demo|over_refusal|off|greedy"),
                lambda d: d["mmlu/mmlu"]["evals"]["mmlu"].update(thinking_mode="on"),
                lambda d: d["controls/t0"]["evals"]["xstest"]["coverage"].update(n_items=251),
                lambda d: d["controls/t0"]["evals"]["xstest"]["coverage"].update(items_sha256="e" * 64),
                lambda d: d["static/t0"]["evals"]["pasttense"]["coverage"]["expected"]["shard00"].pop(),
                lambda d: d["bon/store512"]["evals"]["strongreject_bon"]["coverage"].update(n_max=76),
                lambda d: d["agentic/proxy"]["evals"]["agentic_murder"]["coverage"].update(n_samples=9)):
        assert sha_after(mut) != base["sha256"]
    assert ST.coverage_contract(units[:-1], CP.DONE_PREDICATES)["sha256"] != base["sha256"]
    assert ST.coverage_contract(units, {**CP.DONE_PREDICATES, "capability_cellstore": "capability_cellstore:v2 — x"})["sha256"] != base["sha256"]
    diff = ST.coverage_contract_diff(base, ST.coverage_contract(units, {**CP.DONE_PREDICATES, "capability_cellstore": "capability_cellstore:v2 — x"}))
    # the diff line quotes each predicate with !r (double quotes when the sentence carries an apostrophe, as the capability one does)
    assert len(diff) == 1 and diff[0].startswith("done predicate capability_cellstore: ") and "capability_cellstore:v1 — " in diff[0] and " -> 'capability_cellstore:v2 — x'" in diff[0]


def _probe_copy(tmp_path: Path, predicates: dict) -> Path:
    """A committed-looking alternative coverage_probe.py: this tree's file with DONE_PREDICATES replaced (what a predicate change looks like)."""
    src = (LIB / "coverage_probe.py").read_text()
    i = src.index("DONE_PREDICATES = {"); j = src.index("\n}\n", i) + 3
    alt = src[:i] + "DONE_PREDICATES = " + json.dumps(predicates, indent=1) + "\n" + src[j:]
    p = tmp_path / "alt_probe.py"; p.write_text(alt)
    return p


def test_gate_contract_frozen_header_equal_ok_and_a_predicate_or_identity_change_refuses_naming_the_ruling(tmp_path):
    state, u = _world(tmp_path, families=("controls",))
    inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    inv["coverage_contract"] = ST.coverage_contract_record(inv["units"], CP.DONE_PREDICATES, source="frozen", frozen_utc="2026-10-11T04:45:59Z", inventory_code_sha=SHA)
    ST.write_atomic(str(inv_p), inv)
    blk = tmp_path / "block.json"
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--probe-sha", "p" * 40, "--probe-sha256", "f" * 64, "--write", str(blk))
    assert r.returncode == 0 and "frozen with the inventory at 2026-10-11T04:45:59Z" in r.stdout and "+ 0 receipted amendment(s) — ok" in r.stdout, r.stdout + r.stderr
    b = json.load(open(blk))
    assert b["source"] == "frozen" and b["sha256"] == b["record_sha256"] == inv["coverage_contract"]["sha256"] and b["amendments"] == [] and b["probe_code_sha"] == "p" * 40 and b["inventory_code_sha"] == SHA
    assert not (state / "gate" / "coverage_contract.json").exists()                                   # a header record never gets a sidecar
    # the predicate bumped (a committed change to the probe): refused BEFORE any bind, naming the diff line and the ruling
    alt = _probe_copy(tmp_path, {**CP.DONE_PREDICATES, "capability_cellstore": "capability_cellstore:v2 — an item is done at n_unique == n_items exactly"})
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(alt))
    assert r.returncode == 2 and "REFUSED: 06 --gate: the current probe's done predicates differ" in r.stderr and "done predicate capability_cellstore: " in r.stderr and "capability_cellstore:v1 — " in r.stderr \
        and "-> 'capability_cellstore:v2" in r.stderr and "the operator must rule (D3003 condition 2)" in r.stderr and "nothing bound, no coverage written" in r.stderr, r.stderr
    # an identity edited by hand (instrument_id, decode_policy_ref, n_items): refused naming the key
    for field, mut in (("instrument_id", lambda ev: ev.update(instrument_id="0" * 64)), ("decode_policy_ref", lambda ev: ev.update(decode_policy_ref="x|y|z")),
                       ("n_items", lambda ev: ev["coverage"].update(n_items=251))):
        inv2 = json.load(open(inv_p)); mut(next(x for x in inv2["units"] if x["logical_work_id"].endswith("/controls/t0"))["evals"]["xstest"]); ST.write_atomic(str(inv_p), inv2)
        r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"))
        assert r.returncode == 2 and "unit identities differ from the coverage contract" in r.stderr and f"/controls/t0/evals.xstest.{field}:" in r.stderr and "D3003 condition 2" in r.stderr, (field, r.stderr)
        ST.write_atomic(str(inv_p), inv)
    # a unit appended by hand (no rerun receipt) / a unit removed: refused naming it
    inv3 = json.load(open(inv_p)); extra = json.loads(json.dumps(inv3["units"][0])); extra["logical_work_id"] = f"{MODEL}/battery/controls/t9"; inv3["units"].append(extra); ST.write_atomic(str(inv_p), inv3)
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"))
    assert r.returncode == 2 and f"unit(s) ['{MODEL}/battery/controls/t9'] are not in the frozen coverage contract's set and no rerun receipt adds them" in r.stderr, r.stderr
    inv4 = json.loads(json.dumps(inv)); inv4["units"] = inv4["units"][1:]; ST.write_atomic(str(inv_p), inv4)     # from the recorded inventory (not the one with t9 still appended)
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"))
    assert r.returncode == 2 and "of the frozen coverage contract are missing from the inventory" in r.stderr, r.stderr


def test_gate_contract_admits_a_pre_field_inventory_once_and_the_sidecar_is_the_record_after(tmp_path):
    state, u = _world(tmp_path, families=("mmlu",))
    side = state / "gate" / "coverage_contract.json"; blk = tmp_path / "block.json"
    # no record, no --admit-once: refused by name (the gate passes the flag; a hand run says so)
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"))
    assert r.returncode == 2 and "carries no coverage contract" in r.stderr and "--admit-once" in r.stderr and not side.exists()
    # --dry-run admits nothing on disk
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--admit-once", "--dry-run", "--write", str(blk))
    assert r.returncode == 0 and "would be ADMITTED ONCE" in r.stdout and not side.exists() and not blk.exists()
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--admit-once", "--probe-sha", "p" * 40, "--probe-sha256", "f" * 64, "--write", str(blk))
    assert r.returncode == 0 and "ADMITTED ONCE at " in r.stdout and "frozen before the field existed (D3003 (2))" in r.stdout, r.stdout + r.stderr
    rec = json.load(open(side)); first = side.read_bytes()
    cur = ST.coverage_contract(json.load(open(state / "gate" / "expected_units.json"))["units"], CP.DONE_PREDICATES)
    assert rec["source"] == "admitted_once" and rec["sha256"] == cur["sha256"] and rec["contract"]["unit_set"] == cur["unit_set"] and rec["probe_code_sha"] == "p" * 40 and rec["inventory_code_sha"] == SHA
    b = json.load(open(blk)); assert b["source"] == "admitted_once" and b["record_sha256"] == cur["sha256"] and b["admitted_utc"] == rec["admitted_utc"]
    # the second pass reads the sidecar and rewrites nothing (bytes equal)
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--admit-once", "--write", str(blk))
    assert r.returncode == 0 and f"admitted once on {rec['admitted_utc']}" in r.stdout and "— ok" in r.stdout and side.read_bytes() == first
    # a mutated sidecar (a predicate string edited by hand, the digest left as written) refuses as a HAND EDIT — the digest no longer covers the body,
    # which is read before any predicate is compared; both a header AND the sidecar refuse by name
    bad = json.loads(first); bad["contract"]["done_predicates"]["capability_cellstore"] = "capability_cellstore:v0 — edited"; ST.write_atomic(str(side), bad)
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--admit-once")
    assert r.returncode == 2 and f"the admitted_once coverage contract at {side} records sha256 {rec['sha256'][:12]}… but its body digests to" in r.stderr \
        and "was edited by hand (its body or its digest)" in r.stderr and "done predicates differ" not in r.stderr, r.stderr
    # the same edit with the digest RECOMPUTED over the edited body (a forger who knows the rule): now the structural comparison refuses, naming the predicate
    bad["sha256"] = ST.coverage_contract_sha256(bad["version"], bad["contract"]); ST.write_atomic(str(side), bad)
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--admit-once")
    assert r.returncode == 2 and "done predicates differ from the ones admitted once on" in r.stderr and "capability_cellstore:v0 — edited" in r.stderr, r.stderr
    side.write_bytes(first)
    inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p)); inv["coverage_contract"] = rec; ST.write_atomic(str(inv_p), inv)
    r = _cli("gate-contract", "--state-root", str(state), "--probe", str(LIB / "coverage_probe.py"), "--admit-once")
    assert r.returncode == 2 and "two authorities" in r.stderr and "the tools never create both" in r.stderr


def test_the_contract_digest_covers_its_version_so_one_sha_names_one_rule_and_content(tmp_path):
    """The digest excluded the version, so a cc2 record over the same body carried the sha a cc1 reading computes and
    admission's sha tie could not tell them apart. Now sha256 = contract_dict_sha256({version, unit_set, identities, done_predicates})."""
    state, u = _world(tmp_path, families=("mmlu",))
    c = ST.coverage_contract(list(u.values()), CP.DONE_PREDICATES)
    body = {k: c[k] for k in ST.COVERAGE_CONTRACT_BODY_KEYS}
    assert c["sha256"] == ST.coverage_contract_sha256("cc1", body) == ST.contract_dict_sha256({"version": "cc1", **body})
    assert ST.coverage_contract_sha256("cc2", body) != c["sha256"] and ST.contract_dict_sha256(body) != c["sha256"]
    rec = ST.coverage_contract_record(list(u.values()), CP.DONE_PREDICATES, source="frozen")
    assert rec["sha256"] == c["sha256"] and ST.coverage_contract_record_check(rec, "frozen", "/inv.json") == (body, None)
    # the record check, each refusal by name: another version (the same body), no version, a digest that does not cover the body, no body
    other = {**rec, "version": "cc2"}
    body2, why = ST.coverage_contract_record_check(other, "frozen", "/inv.json")
    assert body2 is None and why.startswith("the frozen coverage contract at /inv.json is version 'cc2', this tree's coverage contract is version 'cc1'") and "the operator must rule (D3003 (2))" in why
    none_v = dict(rec); none_v.pop("version")
    assert ST.coverage_contract_record_check(none_v, "admitted_once", None)[1].startswith("the admitted_once coverage contract is version None, this tree's")
    stale = {**rec, "sha256": "0" * 64}
    _b, why = ST.coverage_contract_record_check(stale, "frozen", "/inv.json")
    assert _b is None and f"records sha256 000000000000… but its body digests to {rec['sha256'][:12]}… under version cc1" in why and "edited by hand" in why
    assert ST.coverage_contract_record_check({**rec, "contract": {"unit_set": []}}, "frozen", "/inv.json")[1].endswith("is malformed (no contract.unit_set / identities / done_predicates) — a hand edit")


def test_gate_contract_refuses_a_record_of_another_version_or_a_stale_digest_before_any_structural_comparison(tmp_path):
    """A frozen inventory bearing version cc2 with the SAME unit set, identities and predicates
    passed `06 --gate` and a cc1 verdict was written over it. A differing / absent version refuses by name, naming the rule; a digest that does
    not cover the body refuses as a hand edit; nothing is written (no sidecar, no --write block); the same with an admit-once sidecar; and the
    block a passing record yields names the record's version beside its sha."""
    state, u = _world(tmp_path, families=("controls",))
    inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    good = ST.coverage_contract_record(inv["units"], CP.DONE_PREDICATES, source="frozen", frozen_utc="2026-10-11T04:45:59Z", inventory_code_sha=SHA)
    blk = tmp_path / "block.json"; side = state / "gate" / "coverage_contract.json"
    probe = str(LIB / "coverage_probe.py")

    def gate(*extra: str) -> subprocess.CompletedProcess:
        return _cli("gate-contract", "--state-root", str(state), "--probe", probe, "--admit-once", "--probe-sha", "p" * 40, "--probe-sha256", "f" * 64, "--write", str(blk), *extra)
    # the finding: version cc2, the body and the digest exactly as the tool wrote them for cc1
    inv["coverage_contract"] = {**good, "version": "cc2"}; ST.write_atomic(str(inv_p), inv)
    r = gate()
    assert r.returncode == 2, r.stdout + r.stderr
    assert f"REFUSED: 06 --gate: the frozen coverage contract at {inv_p} is version 'cc2', this tree's coverage contract is version 'cc1'" in r.stderr, r.stderr
    assert "cannot be compared structurally" in r.stderr and "the operator must rule (D3003 (2))" in r.stderr and "nothing bound, no coverage written" in r.stderr, r.stderr
    assert not blk.exists() and not side.exists() and r.stdout == "" and "— ok" not in r.stdout
    # no version at all (a record written by hand): refused the same way, never admitted once (the header IS a record)
    none_v = dict(good); none_v.pop("version"); inv["coverage_contract"] = none_v; ST.write_atomic(str(inv_p), inv)
    r = gate()
    assert r.returncode == 2 and "is version None, this tree's coverage contract is version 'cc1'" in r.stderr and not blk.exists() and not side.exists() and "ADMITTED ONCE" not in r.stdout, r.stderr
    # the right version, a digest that does not cover the body (the body edited after the freeze, or the digest): a hand edit, named with both shas
    inv["coverage_contract"] = {**good, "sha256": "e" * 64}; ST.write_atomic(str(inv_p), inv)
    r = gate()
    assert r.returncode == 2 and f"REFUSED: 06 --gate: the frozen coverage contract at {inv_p} records sha256 eeeeeeeeeeee… but its body digests to {good['sha256'][:12]}… under version cc1" in r.stderr \
        and "was edited by hand (its body or its digest)" in r.stderr and not blk.exists(), r.stderr
    # the record as written: ok, and the block names the record's version beside its sha
    inv["coverage_contract"] = good; ST.write_atomic(str(inv_p), inv)
    r = gate()
    assert r.returncode == 0 and f"record cc1 {good['sha256'][:12]} + 0 receipted amendment(s) — ok" in r.stdout, r.stdout + r.stderr
    b = json.load(open(blk))
    assert b["record_version"] == "cc1" == b["version"] and b["record_sha256"] == good["sha256"] == b["sha256"] and b["source"] == "frozen"
    # the admit-once SIDECAR of another version refuses the same way (the smoke's shape after a future tree bumps the version)
    inv.pop("coverage_contract"); ST.write_atomic(str(inv_p), inv); blk.unlink()
    r = gate(); assert r.returncode == 0 and "ADMITTED ONCE" in r.stdout and side.exists(), r.stdout + r.stderr
    first = side.read_bytes(); rec = json.loads(first); assert json.load(open(blk))["record_version"] == "cc1"
    ST.write_atomic(str(side), {**rec, "version": "cc2"}); blk.unlink()
    r = gate()
    assert r.returncode == 2 and f"REFUSED: 06 --gate: the admitted_once coverage contract at {side} is version 'cc2', this tree's coverage contract is version 'cc1'" in r.stderr \
        and not blk.exists() and json.loads(side.read_text())["version"] == "cc2", r.stderr                    # never rewritten, never re-admitted
    side.write_bytes(first)
    r = gate(); assert r.returncode == 0 and f"admitted once on {rec['admitted_utc']}, record cc1 {rec['sha256'][:12]}" in r.stdout, r.stdout + r.stderr
    # admission's sha tie: a coverage record whose contract sha is the cc1 body's digest WITHOUT the version (the pre-fix digest) is refused as another contract
    cur = ST.coverage_contract(inv["units"], CP.DONE_PREDICATES)
    assert A.inventory_contract_sha(str(state)) == cur["sha256"] != ST.contract_dict_sha256({k: cur[k] for k in ST.COVERAGE_CONTRACT_BODY_KEYS})


def test_admission_ties_the_coverage_record_to_the_inventorys_contract(tmp_path):
    """A record collected under another contract (a rerun stamped since, an edited inventory, a predicate change) or before the contract existed
    proves nothing until 06 --gate reads the wave again — the same tie as the binding's run ids."""
    state = tmp_path / "state"; doc = freeze(state, families=["mmlu"]); runs, pols = [], []
    u = by_unit(doc)["mmlu/mmlu"]; lw = u["logical_work_id"]
    satisfy(state, u, runs, pols)
    db = make_registry(tmp_path / "reg", runs, pols)
    problems, ok = A.check(str(state), str(db), {})
    assert lw in ok and not problems, problems
    satisfy(state, u, [], [], contract_sha="0" * 64)
    problems, ok = A.check(str(state), str(db), {})
    assert any(f"{lw}: the coverage record was collected under coverage contract 000000000000…" in p and "re-run 06 --gate" in p for p in problems), problems
    cov_p = Path(ST.coverage_path(str(state), lw)); cov = json.load(open(cov_p)); cov.pop("coverage_contract"); ST.write_atomic(str(cov_p), cov)
    problems, ok = A.check(str(state), str(db), {})
    assert any("collected under coverage contract none (before the contract existed)" in p for p in problems), problems


# ------------------------------------------------------------------------------------------------------------------------ the verdict
def test_verdict_requires_both_shas_and_the_contract_for_a_06_family_and_not_for_stage_07(tmp_path):
    state, u = _world(tmp_path, families=("mmlu",))
    r = _cli("verdict", "--state-root", str(state), "--family", "mmlu", "--headline", "0.7", "--range", "0.5,0.9", "--run-ids", "a,b")
    assert r.returncode == 2 and "REFUSED: verdict: family 'mmlu' is a family of the frozen 06 inventory" in r.stderr and "['--inventory-sha', '--probe-sha', '--contract']" in r.stderr \
        and "06_eval_waves.sh --gate mmlu" in r.stderr and not (state / "gate" / "mmlu.json").exists(), r.stderr
    contract = {"version": "cc1", "sha256": "c" * 64, "record_sha256": "d" * 64, "source": "admitted_once", "admitted_utc": "2026-10-13T00:00:00Z", "amendments": []}
    cf = tmp_path / "c.json"; cf.write_text(json.dumps(contract))
    r = _cli("verdict", "--state-root", str(state), "--family", "mmlu", "--headline", "0.7", "--range", "0.5,0.9", "--run-ids", "a,b", "--inventory-sha", SHA, "--probe-sha", "p" * 40,
             "--probe-sha256", "f" * 64, "--contract", f"@{cf}")
    assert r.returncode == 0, r.stderr
    v = json.load(open(state / "gate" / "mmlu.json"))
    assert v["verdict"] == "pass" and v["inventory_code_sha"] == SHA and v["probe_code_sha"] == "p" * 40 and v["probe_sha256"] == "f" * 64 and v["coverage_contract"] == contract and v["run_ids"] == ["a", "b"]
    # stage 07's own two-unit inventory (gate/bcb_bfcl/<view>.json; reproduce/tools/stage07_receipt.py verdict_command) has no 06 inventory: its shape is untouched
    r = _cli("verdict", "--state-root", str(state), "--family", "bcb_bfcl/battery", "--headline", "0.455", "--range", "0.3,0.6", "--run-ids", "k1,k2", "--note", "headline: none acc; receipt abc")
    assert r.returncode == 0, r.stderr
    v7 = json.load(open(state / "gate" / "bcb_bfcl" / "battery.json"))
    assert v7 == {"family": "bcb_bfcl/battery", "headline": 0.455, "plausible_range": [0.3, 0.6], "verdict": "pass", "ruled_utc": v7["ruled_utc"], "run_ids": ["k1", "k2"], "note": "headline: none acc; receipt abc"}
    # the generated stage-07 command, byte for byte as tools/stage07_receipt.py composes it, still runs against a state root that carries a 06 inventory
    sys.path.insert(0, str(REPRO / "tools"))
    import stage07_receipt as R7
    cmd = R7.verdict_command(state, "battery", {"store_cell_keys": ["k1", "k2"], "headline": {"cell": "none", "task": "inspect/bigcodebench_200", "metric": "acc", "value": 0.455}}, "a" * 64, python=sys.executable)
    import shlex
    r = subprocess.run(shlex.split(cmd.replace("<lo,hi>", "0.3,0.6")), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_verdict_refuses_a_malformed_headline_or_range_by_name_leaves_a_standing_verdict_untouched_and_a_recorded_fail_is_rc_0(tmp_path):
    """`stage.py verdict` raised on a malformed --range AFTER 06 --gate had bound and covered the family, and its
    traceback's rc 1 was the status a persisted `fail` returned too, so 06's `|| true` swallowed both and the pass exited 0 with the OLD verdict
    standing over refreshed coverage. Now a RECORDED verdict, pass or fail, is rc 0 (the ruling is on the line and in the file); a malformed
    --headline / --range is a `REFUSED: verdict:` line and rc 2 with nothing written, under --dry-run too; and `parse_verdict_args` is the one
    rule, which 06 --gate applies before anything is bound."""
    state = tmp_path / "state"; (state / "gate").mkdir(parents=True)
    ok = ("--state-root", str(state), "--family", "bcb_bfcl/battery", "--run-ids", "k1")
    r = _cli("verdict", *ok, "--headline", "0.5", "--range", "0.3,0.6")
    assert r.returncode == 0 and "gate verdict bcb_bfcl/battery: pass (headline 0.5 in [0.3, 0.6])" in r.stdout, r.stdout + r.stderr
    vp = state / "gate" / "bcb_bfcl" / "battery.json"; standing = vp.read_bytes()
    for headline, rng, words in (("0.5", "0;1", "two numbers separated by one comma"), ("0.5", "a,b", "two numbers"), ("0.5", "1,0", "lo <= hi"),
                                 ("0.5", "0,1,2", "one comma"), ("0.5", ",1", "one comma"), ("abc", "0,1", "--headline must be a number"),
                                 ("nan", "0,1", "finite numbers"), ("0.5", "0,inf", "finite numbers")):
        r = _cli("verdict", *ok, "--headline", headline, "--range", rng)
        assert r.returncode == 2 and r.stderr.startswith("REFUSED: verdict: ") and words in r.stderr and "nothing written" in r.stderr and "Traceback" not in r.stderr, (headline, rng, r.stderr)
        assert vp.read_bytes() == standing, (headline, rng)
        r = _cli("verdict", *ok, "--headline", headline, "--range", rng, "--dry-run")
        assert r.returncode == 2 and "DRY-RUN" not in r.stdout and vp.read_bytes() == standing, (headline, rng, r.stdout + r.stderr)
    with pytest.raises(ValueError, match="one comma"):
        ST.parse_verdict_args("0.5", "0;1")
    assert ST.parse_verdict_args(" 0.5 ", "0.3, 0.6") == (0.5, 0.3, 0.6)
    # a persisted `fail` is a RECORDED verdict: rc 0, the line and the file both say fail (it returned 1 before — a traceback's status)
    r = _cli("verdict", *ok, "--headline", "5", "--range", "0.3,0.6")
    assert r.returncode == 0 and "gate verdict bcb_bfcl/battery: fail (headline 5 in [0.3, 0.6])" in r.stdout, r.stdout + r.stderr
    assert json.loads(vp.read_text())["verdict"] == "fail" and vp.read_bytes() != standing


def test_the_probe_is_stdlib_only(tmp_path):
    """The shipped probe runs under a login host's bare python3 (06 --gate ships ONE file by content): every import is stdlib."""
    import ast
    tree = ast.parse((LIB / "coverage_probe.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)
    assert set(CP.DONE_PREDICATES) == set(CP.LAYOUTS) and all(v.startswith(f"{k}:v") for k, v in CP.DONE_PREDICATES.items())


def test_probe_done_predicates_reads_the_file_as_it_is_never_a_same_second_same_size_bytecode_cache(tmp_path):
    """integration round 3a (2026-10-14): the gate's end-to-end test bumps the probe `capability_cellstore:v1` -> `:v2`, runs the gate, reverts to
    `:v1` and runs it again -- a SAME-LENGTH edit, and on a fast node all within one second. `probe_done_predicates` imported the file through
    importlib's source loader, which trusts a `__pycache__` entry whose recorded source mtime (whole seconds) and size equal the file's, so the
    reverted probe was read as the bumped one and the gate refused a contract that matched. The reader now compiles the file's bytes
    directly. The race, made deterministic: a stale cache entry for the SAME mtime second and size -- a pyc compiled from the v2 text while the
    file holds v1 with an identical (seconds) mtime -- is never trusted: the predicates read are the file's, and no cache entry is written."""
    import importlib.util, py_compile
    probe = tmp_path / "coverage_probe.py"
    v1 = 'DONE_PREDICATES = {"capability_cellstore": "capability_cellstore:v1 — the artifact holds every item"}\n'
    v2 = v1.replace(":v1 — ", ":v2 — "); assert len(v1) == len(v2)
    probe.write_text(v2)
    pyc = importlib.util.cache_from_source(str(probe))
    py_compile.compile(str(probe), cfile=pyc, doraise=True)            # the cache entry the bumped run would leave behind
    probe.write_text(v1)                                               # the revert: same size ...
    st = probe.stat(); os.utime(probe, ns=(st.st_atime_ns, int(st.st_mtime) * 10 ** 9))   # ... and a mtime of the SAME whole second the pyc records
    spec = importlib.util.spec_from_file_location("stale_probe_through_the_loader", str(probe))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)             # type: ignore[union-attr]
    assert mod.DONE_PREDICATES["capability_cellstore"].startswith("capability_cellstore:v2"), "the loader trusts the stale cache: the premise of this test"
    assert ST.probe_done_predicates(str(probe)) == {"capability_cellstore": "capability_cellstore:v1 — the artifact holds every item"}
    os.remove(pyc); probe.write_text(v2 + "\n")
    assert ST.probe_done_predicates(str(probe))["capability_cellstore"].startswith("capability_cellstore:v2")
    assert not os.path.exists(pyc), "the gate's reader writes no cache entry"
    with pytest.raises(ValueError, match="declares no DONE_PREDICATES"):
        probe.write_text("X = 1\n"); ST.probe_done_predicates(str(probe))
    with pytest.raises(ValueError, match="cannot be read"):
        ST.probe_done_predicates(str(tmp_path / "absent.py"))
    # every reader of this tree's predicates in stage.py goes through the bytes reader: the freeze (expected_units' coverage_contract record),
    # rerun-units' amendment check and the gate; a `coverage_probe` import would put the bytecode cache back between an edit and the next read
    src = (LIB / "stage.py").read_text()
    assert "import coverage_probe" not in src and "CP.DONE_PREDICATES" not in src and src.count("this_tree_done_predicates()") >= 3
    assert ST.this_tree_done_predicates() == CP.DONE_PREDICATES          # the same declaration this test module imported
