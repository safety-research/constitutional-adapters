"""The live-drill producer: ``scripts/registry/smoke_producer.py`` mints a
DETERMINISTIC synthetic identity (the same campaign + run key → the same run id, object, instrument; every id
prefixed ``drill``), refuses a non-drill campaign, and drives one run's whole lifecycle through the harness's own
hooks — the records validate, fold through the real harvester to ``ingested`` with exactly the counts the
script's summary promises, and a rebuild from the same files verifies equal; it runs under ``python3 -S`` as a
bare file (the compute-node invocation) and writes the pushlog; the push script is called with ``--final`` on
both directories; an unwritable spool is a loud exit 3 for the DRILL (the harness's hooks are guarded — the same
failure costs a real job one stderr line). The Modal registration path and the Modal fan-out app left with the
Modal transport.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import emit, hooks, identity
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PRODUCER = REPO / "scripts" / "registry" / "smoke_producer.py"
SHA = synth.code_sha("a")


def _load_producer():
    spec = importlib.util.spec_from_file_location("smoke_producer_under_test", PRODUCER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SP = _load_producer()


def _records(paths):
    return [(p, rec, sha) for p in paths for _, rec, sha in R.read_record_file(p)]


def _summary(out: str) -> dict:
    line = [l for l in out.splitlines() if l.startswith(SP.SUMMARY_SENTINEL)][-1]
    return json.loads(line[len(SP.SUMMARY_SENTINEL):])


@pytest.fixture
def quiet_env(monkeypatch, tmp_path):
    """No scheduler, no platform, no registry env — and NO host: every probe ``emit`` walks is looked up under the
    empty ``<tmp>/host`` (``emit.PROBE_ROOT_ENV``), so neither the in-process producer nor a subprocess handed
    :func:`_bare_env` meets the real host's ``~/registry`` (a grading host) or a provisioned shared root (a cluster node)."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.PUSH_SCRIPT_ENV, raising=False)
    monkeypatch.delenv(emit.SPOOL_ENV, raising=False)
    monkeypatch.delenv(emit.SUBSTRATE_ENV, raising=False)
    monkeypatch.delenv("REGISTRY_EMIT", raising=False)
    for var in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURMD_NODENAME", *emit.SESSION_ENV_VARS,
                "SLURM_CLUSTER_NAME", "RUNPOD_POD_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USER", "nobody-synthetic")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "host").mkdir()
    monkeypatch.setenv(emit.PROBE_ROOT_ENV, str(tmp_path / "host"))
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    return tmp_path


def _bare_env(quiet_env, **extra) -> dict:
    """The environment of a producer run as a bare ``python3 -S`` subprocess: the checkout's ``src`` on PYTHONPATH, a
    synthetic $HOME / $USER, and the empty ``<tmp>/host`` as the probe root — nothing of the real host (its ``/vol/registry``
    on the Modal CPU suite decided the substrate of the first -S test there, 2026-09-17)."""
    env = {"PYTHONPATH": str(SRC), "PATH": os.environ.get("PATH", ""), "HOME": str(quiet_env / "home"), "USER": "nobody-synthetic",
           emit.PROBE_ROOT_ENV: str(quiet_env / "host")}
    env.update(extra)
    return env


def _run_bare(quiet_env, copy, spool, extra_env=None, *args):
    env = _bare_env(quiet_env, **(extra_env or {}))
    argv = [sys.executable, "-S", str(copy), "--campaign", "drill_s", "--spool", str(spool), "--no-push", "--code-sha", SHA,
            "--n-items", "2", "--json"] + list(args)
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd="/", timeout=120)


# --------------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------------


def test_identity_is_deterministic_and_every_id_says_drill():
    a = SP.planned_run("drill_20260101T000000Z", "drill_run_007", 5, SHA)
    b = SP.planned_run("drill_20260101T000000Z", "drill_run_007", 5, SHA)
    assert a == b
    rid = SP.run_id_for("drill_20260101T000000Z", "drill_run_007")
    assert rid == identity.run_id("drill_20260101T000000Z", "drill_run_007", a["producer_object_key_legacy"], SP.EVAL_ID, SP.INSTRUMENT_ID)
    assert rid.startswith("drill_20260101T000000Z:drill_eval:")
    # the store would mint the SAME run id from its own spec (no run_id handed over): its object key is the plan's
    spec = SP.resolved_spec("drill_20260101T000000Z", "drill_run_007", "4242")
    assert hooks.StoreRecorder.object_key_of(spec) == a["producer_object_key_legacy"]
    assert SP.INSTRUMENT_ID == identity.instrument_id(SP.INSTRUMENT) and SP.INSTRUMENT["judge_id"] == SP.JUDGE_ID
    # a different campaign → a different run id AND a different object (its campaign column is the drill's)
    assert SP.run_id_for("drill_other", "drill_run_007") != rid
    assert SP.object_id_for("drill_other") != SP.object_id_for("drill_20260101T000000Z") == a["object_id"]
    for k in ("model_id", "eval_id", "item_set_id", "campaign", "producer_run_key_legacy"):
        assert str(a[k]).startswith("drill"), (k, a[k])
    assert a["extra"]["synthetic"] == "drill" and a["code_sha"] == SHA and a["n_items_planned"] == 5


def test_dimension_payloads_are_pure_validate_and_never_name_the_shared_none_object():
    dims = SP.dimension_payloads("drill_x")
    assert dims == SP.dimension_payloads("drill_x")
    assert [rt for rt, _ in dims] == ["model", "model", "eval", "item_set", "judge", "instrument", "object"]
    prod = {"script": "t", "code_sha": SHA, "session": None, "host": "TESTHOST", "substrate": "grading_host"}
    for rt, payload in dims:
        rec = R.make_record(rt, payload, prod, identity.producer_record_id(), "2026-01-01T00:00:00.000Z")
        v = VAL.validate_record(rec, "<mem>")
        assert isinstance(v, VAL.Validated), (rt, getattr(v, "field", None), getattr(v, "reason", None))
        assert payload["extra"]["synthetic"] == "drill"
    obj = dict(dims)["object"]
    assert obj["object_id"] == SP.object_id_for("drill_x") != "none" and obj["campaign"] == "drill_x" and obj["status"] == "reference"
    assert dict(dims)["eval"]["paper_status"] == "dropped"
    assert all(p["paper_status"] == "reference" for rt, p in dims if rt == "model")
    # the plan's run cites exactly these rows
    plan = SP.planned_run("drill_x", "k", 3, SHA)
    assert plan["object_id"] == obj["object_id"] and plan["model_id"] == SP.MODEL_IT and plan["instrument_id"] == dict(dims)["instrument"]["instrument_id"]


def test_refusals(quiet_env, capsys):
    spool = str(quiet_env / "spool")
    assert SP.main(["--campaign", "synthcampaign", "--spool", spool, "--no-push", "--code-sha", SHA]) == 2
    assert "refusing campaign" in capsys.readouterr().err
    assert SP.main(["--campaign", "drill_x", "--spool", spool, "--no-push", "--code-sha", "nothex"]) == 2
    assert SP.main(["--campaign", "drill_x", "--spool", spool, "--no-push", "--code-sha", SHA, "--n-items", "0"]) == 2
    os.environ[emit.DISABLED_ENV] = "1"
    try:
        assert SP.main(["--campaign", "drill_x", "--spool", spool, "--no-push", "--code-sha", SHA]) == 2
    finally:
        del os.environ[emit.DISABLED_ENV]
    assert not os.path.exists(spool)


# --------------------------------------------------------------------------------------------
# the lifecycle, folded by the real harvester
# --------------------------------------------------------------------------------------------


def test_lifecycle_folds_to_ingested_with_exactly_the_promised_counts(quiet_env, capsys):
    root = str(quiet_env / "reg")
    H.ensure_layout(root)
    spool = os.path.join(root, "records")
    rc = SP.main(["--campaign", "drill_fold", "--run-key", "drill_run_003", "--spool", spool, "--no-push", "--code-sha", SHA,
                  "--n-items", "4", "--job-id", "4242", "--job-name", "D_synthetic", "--node", "TESTHOST", "--json"])
    assert rc == 0
    s = _summary(capsys.readouterr().out)
    assert s["ok"] and s["run_id"] == SP.run_id_for("drill_fold", "drill_run_003") and s["push"] is False
    # the record sequence a registered job leaves behind, in emission order
    run_paths = list(R.iter_record_files(s["run_dir"]))
    types_ = [os.path.basename(p).split("-", 1)[1].split(".")[0] for p in run_paths]
    assert types_ == ["run", "run_state", "run_patch",                       # register: queued + the job binding
                      "run_state", "run_patch",                              # s2_init: running + node/job/gpu
                      "run", "run_state",                                    # the store's manifest: its run + running
                      "item_result", "item_flags", "run_patch",              # rows (flushed on the marker) + generation complete
                      "run_state", "run_patch", "artifact", "run_degeneration",   # results: generated, completion, ONE artifact, degeneration
                      "judgment", "item_result", "run_metric", "judgment", "run_degeneration", "run_state", "run_patch"]  # judging
    recs = _records(run_paths) + _records(R.iter_record_files(s["dims_dir"]))
    for p, rec, sha in recs:
        assert isinstance(VAL.validate_record(rec, p, sha), VAL.Validated), (p, rec.get("record_type"))
        assert rec["producer"]["code_sha"] == SHA and rec["producer"]["host"] == "TESTHOST"
    assert s["records"] == {"artifact": 1, "item_flags": 4, "item_result": 8, "judgment": 2, "run": 2, "run_degeneration": 8,
                            "run_metric": 1, "run_patch": 5, "run_state": 5}
    assert s["dimension_records"] == {"eval": 1, "instrument": 1, "item_set": 1, "judge": 1, "model": 2, "object": 1}
    assert s["n_files"] == len(run_paths) + 7 and len(s["files"]) == s["n_files"]
    # the drill's scratch (pending file, plan, store) lives OUTSIDE the ingestible namespace
    assert os.path.exists(s["pending_file"]) and json.load(open(s["pending_file"]))["run_id"] == s["run_id"]
    assert not s["pending_file"].startswith(spool) and not s["store_dir"].startswith(spool)
    assert sorted(os.listdir(s["store_dir"])) == ["generation_complete.json", "manifest.json", "results.json"]   # no rows.jsonl → one artifact
    # the fold: the real harvester, one cycle
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    rid = s["run_id"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = dict(conn.execute("SELECT * FROM run WHERE run_id = ?", (rid,)).fetchone())
    assert row["lifecycle_state"] == "ingested" and row["judged_utc"] and row["generated_utc"] and row["started_utc"] and row["queued_utc"]
    assert row["queued_utc"] <= row["started_utc"] <= row["generated_utc"] <= row["judged_utc"]
    assert row["cluster_job_id"] == "4242" and row["job_name"] == "D_synthetic" and row["node_hostname"] == "TESTHOST" and row["gpu_count"] == 0
    assert row["n_items_generated"] == 4 and row["n_items_judged"] == 4 and row["judge_id"] == SP.JUDGE_ID and row["code_sha"] == SHA
    assert row["object_id"] == SP.object_id_for("drill_fold") and row["model_id"] == SP.MODEL_IT and row["campaign"] == "drill_fold"
    assert json.loads(row["extra_json"])["synthetic"] == "drill"
    events = [tuple(r) for r in conn.execute("SELECT state, evidence_kind FROM run_state_event WHERE run_id = ? ORDER BY event_utc", (rid,))]
    assert [e[0] for e in events] == ["queued", "running", "running", "generated", "judged"]
    assert {e[1] for e in events if e[0] == "running"} == {"job_log", "completion_artifact"}
    exp = s["expected_rows"]
    for table in ("item_result", "item_flags", "run_degeneration", "run_metric", "judgment"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (rid,)).fetchone()[0] == exp[table], table
    assert exp == {"artifact": 1, "item_flags": 4, "item_result": 8, "judgment": 1, "n_items": 4, "run": 1, "run_degeneration": 8,
                   "run_metric": 1, "run_state_event": 5}
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND judgment_id = ''", (rid,)).fetchone()[0] == 4      # the store's
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND judgment_id != ''", (rid,)).fetchone()[0] == 4     # the judge's
    assert conn.execute("SELECT COUNT(*) FROM run_state_event WHERE run_id = ?", (rid,)).fetchone()[0] == exp["run_state_event"]
    assert conn.execute("SELECT status, n_items_judged, api_pool FROM judgment WHERE run_id = ?", (rid,)).fetchone()[:] == ("complete", 4, None)
    assert conn.execute("SELECT kind, uri_primary FROM artifact").fetchone()[:] == ("completion", os.path.join(s["store_dir"], "results.json"))
    assert conn.execute("SELECT value, k, n FROM run_metric WHERE run_id = ?", (rid,)).fetchone()[:] == (0.5, 2, 4)
    assert conn.execute("SELECT COUNT(*) FROM object WHERE object_id = 'none'").fetchone()[0] == 0     # the drill never seeds the shared row
    assert [tuple(r) for r in conn.execute("SELECT model_id, paper_status FROM model ORDER BY model_id")] == \
        [(SP.MODEL_BASE, "reference"), (SP.MODEL_IT, "reference")]
    conn.close()
    # on the drill alone: a rebuild from the same files equals the folded database
    into = str(quiet_env / "rebuilt.sqlite")
    assert cli.main(["--root", root, "rebuild", "--from", root, "--into", into, "--verify", "--workers", "0"]) == 0
    # a second identical campaign re-states the same dimension rows (idempotent) and adds nothing to the run
    assert SP.main(["--campaign", "drill_fold", "--run-key", "drill_run_003", "--spool", spool, "--no-push", "--code-sha", SHA,
                    "--n-items", "4", "--job-id", "4242", "--node", "TESTHOST", "--json"]) == 0
    capsys.readouterr()
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 1 and conn.execute("SELECT COUNT(*) FROM model").fetchone()[0] == 2
    conn.close()


def test_runs_under_python_dash_S_as_a_bare_file_and_writes_the_pushlog(quiet_env):
    """The compute-node and Modal invocation: no venv, no site-packages, the script copied out of the checkout."""
    copy = quiet_env / "drills" / "smoke_producer.py"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(PRODUCER.read_bytes())
    spool = quiet_env / "spool"
    pushlog = quiet_env / "pushlog" / "call-synthetic.log"
    out = _run_bare(quiet_env, copy, spool, None, "--pushlog", str(pushlog))
    assert out.returncode == 0, out.stderr
    s = _summary(out.stdout)
    # grading_host is what the ladder DERIVES on a host where nothing resolves (the probe root is an empty directory) — on the
    # Modal CPU suite the same child, probing the real host, said `modal` from the container's /vol/registry (2026-09-17)
    assert s["ok"] and s["substrate"] == "grading_host" and s["pushlog_lines"] == s["n_files"] == 7 + 21
    lines = pushlog.read_text().splitlines()
    assert lines == sorted(s["files"]) and all(not l.startswith("/") for l in lines)
    assert all((spool / l).exists() for l in lines)
    # every path is <campaign>/<dir>/<seq>-<type>.jsonl[.gz]: what stage_bundle takes relative to /registry/records
    assert all(l.split("/")[0] == "drill_s" and R.RECORD_FILE_RE.match(l.split("/")[-1]) for l in lines)
    # the run's own record files all validate under the -S interpreter too (read back here with the venv)
    for p, rec, sha in _records(R.iter_record_files(str(spool))):
        assert isinstance(VAL.validate_record(rec, p, sha), VAL.Validated), p


def test_substrate_override_reaches_a_bare_file_producer_and_an_unknown_name_falls_back(quiet_env):
    """Known-bad, the shape of the Modal gate's failure turned around: on THIS host (whatever it is) the -S child is told
    ``SISTEER_REGISTRY_SUBSTRATE=modal`` and every producer block and derived ``run.substrate`` it writes says ``modal`` —
    the override rides the subprocess environment, where no monkeypatch reaches. A name outside the vocabulary is one
    ``[registry]`` stderr line for the whole process and the ladder decides (``grading_host`` on the empty probe root); the
    producer still exits 0 with every file it promised."""
    copy = quiet_env / "drills" / "smoke_producer.py"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(PRODUCER.read_bytes())
    out = _run_bare(quiet_env, copy, quiet_env / "spool_m", {emit.SUBSTRATE_ENV: "modal"}, "--run-key", "drill_run_m")
    assert out.returncode == 0, out.stderr
    s = _summary(out.stdout)
    assert s["ok"] and s["substrate"] == "modal" and s["n_files"] == 7 + 21
    recs = _records(R.iter_record_files(str(quiet_env / "spool_m")))
    assert recs and all(rec["producer"]["substrate"] == "modal" for _, rec, _ in recs)
    assert {rec["payload"]["substrate"] for _, rec, _ in recs if rec["record_type"] == "run"} == {"modal"}
    assert "is not a substrate" not in out.stderr
    # an unknown name: the notice names the variable, the value and the vocabulary, once; the ladder's answer stands
    out = _run_bare(quiet_env, copy, quiet_env / "spool_u", {emit.SUBSTRATE_ENV: "laptop"}, "--run-key", "drill_run_u")
    assert out.returncode == 0, out.stderr
    s = _summary(out.stdout)
    assert s["ok"] and s["substrate"] == "grading_host" and s["n_files"] == 7 + 21
    notices = [l for l in out.stderr.splitlines() if "is not a substrate" in l]
    assert len(notices) == 1 and "SISTEER_REGISTRY_SUBSTRATE='laptop'" in notices[0] and "cluster_a|cluster_b|modal|grading_host|runpod" in notices[0]
    recs = _records(R.iter_record_files(str(quiet_env / "spool_u")))
    assert recs and all(rec["producer"]["substrate"] == "grading_host" for _, rec, _ in recs)


def test_no_dims_skips_the_dimension_rows_on_a_second_run_and_the_removed_registration_flags_are_refused(quiet_env, capsys):
    """Two drills of one campaign: the first seeds the dimension rows, the second (`--no-dims`) skips them and writes exactly the
    run's 21 files; both fold to `ingested` with the job binding. The Modal registration path (`--register`, `--modal-app-id`,
    `--modal-call-id`) left with the Modal transport: argparse refuses the flags (exit 2)."""
    root = str(quiet_env / "reg")
    H.ensure_layout(root)
    spool = os.path.join(root, "records")
    assert SP.main(["--campaign", "drill_m", "--run-key", "drill_run_000", "--spool", spool, "--no-push", "--code-sha", SHA, "--n-items", "2",
                    "--job-id", "job-synthetic-0", "--node", "TESTHOST", "--json"]) == 0
    s0 = _summary(capsys.readouterr().out)
    assert SP.main(["--campaign", "drill_m", "--run-key", "drill_run_001", "--spool", spool, "--no-push", "--code-sha", SHA, "--n-items", "2",
                    "--job-id", "job-synthetic-1", "--node", "TESTHOST", "--no-dims", "--json"]) == 0
    s1 = _summary(capsys.readouterr().out)
    assert "dims_dir" in s0 and "dims_dir" not in s1 and s1["n_files"] == 21 and s1["register"] == "slurm"
    recs = _records(R.iter_record_files(s1["run_dir"]))
    run, state = (r["payload"] for _, r, _ in recs[:2])
    assert run["lifecycle_state"] == "queued" and run["substrate"] == "grading_host" and state["state"] == "queued"
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = [tuple(r) for r in conn.execute("SELECT run_id, lifecycle_state, substrate, cluster_job_id FROM run ORDER BY run_id")]
    assert rows == [(s0["run_id"], "ingested", "grading_host", "job-synthetic-0"), (s1["run_id"], "ingested", "grading_host", "job-synthetic-1")]
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 8
    conn.close()
    for flags in (["--register", "slurm"], ["--modal-app-id", "x"], ["--modal-call-id", "y"]):
        with pytest.raises(SystemExit) as ex:
            SP.main(["--campaign", "drill_m", "--run-key", "drill_run_002", "--spool", spool, "--no-push", "--code-sha", SHA] + flags)
        assert ex.value.code == 2
    capsys.readouterr()


def test_push_script_is_called_with_final_on_both_directories(quiet_env, capsys):
    calls = quiet_env / "calls.log"
    script = quiet_env / "push.sh"
    script.write_text(f"#!/bin/sh\necho \"$@\" >> {calls}\nexit 0\n")
    script.chmod(0o755)
    spool = str(quiet_env / "spool")
    assert SP.main(["--campaign", "drill_p", "--spool", spool, "--push-script", str(script), "--code-sha", SHA, "--n-items", "1", "--json"]) == 0
    s = _summary(capsys.readouterr().out)
    assert s["push"] is True and s["push_script"] == str(script)
    lines = calls.read_text().strip().splitlines()
    dirs = {l.split()[0] for l in lines}
    assert dirs == {s["dims_dir"], s["run_dir"]}
    finals = [l for l in lines if l.endswith("--final")]
    assert any(l.startswith(s["dims_dir"]) for l in finals) and sum(1 for l in finals if l.startswith(s["run_dir"])) >= 2   # results + judged
    assert emit.PUSH_SCRIPT_ENV not in os.environ         # main restored the environment it borrowed
    assert emit.git_sha.__name__ == "git_sha"             # and the module function


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes (the Modal CPU suite runs as root)")
def test_unwritable_spool_is_a_loud_exit_3_for_the_drill(quiet_env, capsys):
    """The DRILL surfaces a spool it cannot write (exit 3, the exception named). A REAL job does not: ``CellStore._record``
    and the ``@guarded`` hooks turn the same PermissionError into one ``[registry]`` stderr line, ``emit_cli state`` exits
    0 with a notice, and the submit wrapper reports 'submitted but NOT registered'."""
    spool = quiet_env / "spool"
    spool.mkdir()
    spool.chmod(0o500)
    try:
        rc = SP.main(["--campaign", "drill_ro", "--spool", str(spool), "--no-push", "--code-sha", SHA, "--json"])
    finally:
        spool.chmod(0o700)
    assert rc == 3
    out = capsys.readouterr()
    s = _summary(out.out)
    assert s["ok"] is False and "PermissionError" in s["error"] and "FAILED" in out.err
    assert not any(spool.iterdir())
    # the harness-side contract on the same spool: the job shell's entry (`python3 -S -m sisteer.registry.emit_cli state`)
    # exits 0 with a notice — the guard lives in emit_cli.main, NOT in the library function emit_cli.state, which raises
    from sisteer.registry import emit_cli
    spool.chmod(0o500)
    try:
        assert emit_cli.main(["state", "--state", "running", "--run-id", SP.run_id_for("drill_ro", "k"), "--node", "TESTHOST",
                              "--spool", str(spool)]) == 0
        err = capsys.readouterr().err
        assert "PermissionError" in err and "job continues" in err
        with pytest.raises(PermissionError):
            emit_cli.state("running", run_id=SP.run_id_for("drill_ro", "k"), node="TESTHOST", spool=str(spool))
        # the store's recorder raises too; CellStore._record is the guard around it (one stderr line, the run continues)
        rec = hooks.StoreRecorder("k", SP.resolved_spec("drill_ro", "k", "1"), {"*": dict(SP.planned_run("drill_ro", "k", 1, SHA),
                                  run_id=SP.run_id_for("drill_ro", "k"))}, spool=str(spool), code_sha=SHA)
        with pytest.raises(PermissionError):
            rec.on_manifest(SP.resolved_spec("drill_ro", "k", "1"), str(quiet_env / "manifest.json"))
    finally:
        spool.chmod(0o700)
