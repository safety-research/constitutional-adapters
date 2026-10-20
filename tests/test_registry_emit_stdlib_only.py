"""emit.py and identity.py are STDLIB ONLY: importable on a compute node before `uv sync` under
`python3 -S` (no site-packages), with no torch / pandas / numpy / yaml / pyarrow / jinja2 loaded. Also the
emitter's disabled mode, spool layout, debounce and the planned-run registration.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from sisteer.registry import emit, identity, records as R, synth, validate as VAL
from sisteer.registry import vocab as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

_PROBE = """
import sys
import sisteer.registry.emit, sisteer.registry.identity, sisteer.registry.records, sisteer.registry.vocab
import sisteer.registry.hooks, sisteer.registry.emit_cli        # the harness-facing hooks and the job-shell CLI (PR-4)
heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch','pandas','numpy','yaml','pyarrow','jinja2','sqlite3','scipy'))
print(__import__('json').dumps({'heavy': heavy, 'site': 'site' in sys.modules}))
"""


def test_emit_and_identity_import_under_python_dash_S():
    env = {"PYTHONPATH": SRC, "PATH": os.environ.get("PATH", "")}
    out = subprocess.run(["python3", "-S", "-c", _PROBE], capture_output=True, text=True, env=env, cwd="/", timeout=60)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["heavy"] == [], data


def test_emit_writes_run_lifecycle_files(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = str(tmp_path / "spool")
    w = emit.RecordWriter(spool, "camp", "camp:ev:abcdefabcdef:0123456789abcdef", synth.producer(), disabled=False)
    p1 = w.emit_run(synth.run_payload("k", i=0))
    p2 = w.emit_state("running", "job_log", scheduler_state="RUNNING")
    p3 = w.emit_patch(fields={"node_hostname": "TESTHOST"}, verify={"serving_path": "lora_native"})
    p4 = w.emit_items([{"item_id": "q1", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": "",
                        "value": 1.0, "label": None, "api_pool": None, "created_utc": synth.t(0), "judged_utc": None,
                        "source_revision_id": "", "withdrawn_by_revision_id": None}])
    assert [os.path.basename(p) for p in (p1, p2, p3, p4)] == ["000001-run.jsonl", "000002-run_state.jsonl", "000003-run_patch.jsonl",
                                                               "000004-item_result.jsonl.gz"]
    assert os.path.dirname(p1) == os.path.join(spool, "camp", identity.run_dir_name(w.run_id))
    for p in (p1, p2, p3, p4):
        for _, rec, sha in R.read_record_file(p):
            v = VAL.validate_record(rec, p, sha)
            assert isinstance(v, VAL.Validated), v
            assert rec["producer"]["host"] == "TESTHOST"
    # a second writer on the same dir continues the sequence
    w2 = emit.RecordWriter(spool, "camp", w.run_id, synth.producer(), disabled=False)
    assert os.path.basename(w2.emit_state("generated", "completion_artifact")) == "000005-run_state.jsonl"


def test_disabled_env_makes_writer_noop(tmp_path, monkeypatch):
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    w = emit.RecordWriter(str(tmp_path), "camp", "camp:e:i:x", synth.producer())
    assert w.disabled and w.emit_run(synth.run_payload("k")) is None and w.push() == 0
    assert not os.listdir(tmp_path)
    monkeypatch.delenv(emit.DISABLED_ENV)
    w3 = emit.RecordWriter(None, "camp", "camp:e:i:x", synth.producer())
    assert w3.disabled and "spool" in w3.disabled_reason


def test_push_debounce_and_always_returns(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    script = tmp_path / "push.sh"
    log = tmp_path / "calls.log"
    script.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n")
    script.chmod(0o755)
    w = emit.RecordWriter(str(tmp_path / "spool"), "camp", "camp:e:i:x", synth.producer(), push_script=str(script), disabled=False)
    w.emit_state("queued", "submit_wrapper")
    assert w.push() == 0
    (tmp_path / "spool" / "camp" / identity.run_dir_name(w.run_id) / ".last_push").touch()
    assert w.push() == 0  # debounced: no second call
    assert w.push(final=True) == 0
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 2 and calls[1].endswith("--final")
    # a missing script never raises
    w2 = emit.RecordWriter(str(tmp_path / "spool"), "camp", "camp:e:i:x", synth.producer(), push_script="/nonexistent/push.sh", disabled=False)
    assert w2.push(final=True) == 0


def test_write_planned_run_mints_registry_id_and_queued_state(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test writes to its own tmp spool, so it opts in
    plan = {k: v for k, v in synth.run_payload("plan_key", i=3).items() if k != "run_id"}
    paths = emit.write_planned_run(str(tmp_path), plan, script="sbatch_registered.sh")
    assert len(paths) == 2
    recs = [rec for p in paths for _, rec, _ in R.read_record_file(p)]
    assert recs[0]["record_type"] == "run" and recs[0]["payload"]["lifecycle_state"] == "queued"
    assert recs[1]["record_type"] == "run_state" and recs[1]["payload"]["state"] == "queued"
    assert recs[0]["payload"]["run_id"] == identity.run_id(plan["campaign"], plan["producer_run_key_legacy"],
                                                           plan["producer_object_key_legacy"], plan["eval_id"], plan["instrument_id"])


def test_substrate_and_spool_from_env(unprovisioned_host, monkeypatch, tmp_path):
    """The substrate ladder over probes the fixture points under tmp_path: nothing on the real host (a cluster node's
    shared root, a provisioned ``~/registry``) decides it. No cluster name or shared root is fixed in the emitter: the site map
    (``SISTEER_SLURM_SUBSTRATES``) names both, and a Slurm job with no site information keeps the fallbacks
    (tests/test_registry_emit_site_map.py pins the whole order). The Modal rungs (a ``/vol`` mount, ``MODAL_TASK_ID``) name no
    substrate."""
    h = unprovisioned_host
    for var in (emit.SITE_MAP_ENV, emit.CLUSTER_LABEL_ENV, emit.PREVIOUS_HELPER_ENV):
        monkeypatch.delenv(var, raising=False)
    assert h.shared == {} and emit.substrate_from_env() == "grading_host"
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-one")
    assert emit.substrate_from_env() == "grading_host"                 # no site information: no name in code decides it
    monkeypatch.setenv(emit.SITE_MAP_ENV, f"cluster-one=cluster_a cluster-two=cluster_b@{h.root}/shared-b/{{user}}")
    assert emit.substrate_from_env() == "cluster_a"
    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    # a mapped shared root by its bare existence (spool_root wants registry/ provisioned too)
    (h.root / "shared-b" / h.user).mkdir(parents=True)
    assert emit.substrate_from_env() == "cluster_b" and emit.spool_root() is None
    (h.root / "shared-b" / h.user).rmdir()
    monkeypatch.delenv(emit.SITE_MAP_ENV)
    # the removed Modal rungs: a container task id or a volume mount under the probe root decide nothing
    monkeypatch.setenv("MODAL_TASK_ID", "ta-synthetic")
    (h.root / "vol" / "registry").mkdir(parents=True)
    assert emit.substrate_from_env() == "grading_host" and emit.spool_root() is None
    monkeypatch.delenv("MODAL_TASK_ID")
    (h.root / "vol" / "registry").rmdir()
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-synthetic")
    assert emit.substrate_from_env() == "runpod"
    monkeypatch.delenv("RUNPOD_POD_ID")
    assert emit.substrate_from_env() == "grading_host"
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path))
    assert emit.spool_root() == str(tmp_path)
    assert emit.git_sha(cwd=str(tmp_path)) == "unverified"


def test_probe_root_relocates_every_host_probe_and_the_substrate_override_is_validated(tmp_path, monkeypatch, capsys):
    """The subprocess-carrying form of the ``unprovisioned_host`` redirection (no fixture here: the two are alternatives, a
    root on top of the redirected constants would prefix them twice). Each rung under the root (a provisioned shared root the
    site map names, ``~/registry`` at ``<root>/$HOME/registry``), an empty root (nothing resolves, the grading host), a
    ``vol/registry`` under the root resolving NOTHING, the explicit spool beating the root, and the name override: valid → it
    wins over every probe, ``SLURM_CLUSTER_NAME`` included, and moves no spool; unknown → one stderr notice per process and the
    probes decide."""
    monkeypatch.setattr(emit, "_NOTICED", set())
    for var in (emit.SPOOL_ENV, emit.SUBSTRATE_ENV, "SLURM_CLUSTER_NAME", "RUNPOD_POD_ID", emit.CLUSTER_LABEL_ENV, emit.PREVIOUS_HELPER_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USER", "nobody-synthetic")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "host"
    root.mkdir()
    monkeypatch.setenv(emit.PROBE_ROOT_ENV, str(root))
    monkeypatch.setenv(emit.SITE_MAP_ENV, "cluster-one=cluster_a@/srv/shared/{user}")
    # the mapped root as the probes see it: relocated under the probe root
    shared = root / "srv" / "shared" / "nobody-synthetic"
    assert emit.probe_path("~/registry") == str(root / home.relative_to("/") / "registry")
    # a provisioned shared root under the probe root names its substrate and its spool
    (shared / "registry").mkdir(parents=True)
    assert emit.substrate_from_env() == "cluster_a" and emit.spool_root() == str(shared / "registry" / "records")
    (shared / "registry").rmdir()
    shared.rmdir()
    # the removed probe: a provisioned volume path under the root is NOT a registry host
    gone = root / "vol" / "registry"
    gone.mkdir(parents=True)
    assert emit.substrate_from_env() == "grading_host" and emit.spool_root() is None
    gone.rmdir()
    gone.parent.rmdir()
    # ~/registry is looked up under the root too: <root>/<$HOME>/registry
    home_reg = root / home.relative_to("/") / "registry"
    home_reg.mkdir(parents=True)
    assert emit.substrate_from_env() == "grading_host" and emit.spool_root() == str(home_reg / "records")
    home_reg.rmdir()
    # an empty root is an unprovisioned host, whatever the real one carries
    assert emit.spool_root() is None and emit.substrate_from_env() == "grading_host"
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "explicit"))
    assert emit.spool_root() == str(tmp_path / "explicit")             # an explicit spool still beats every probe
    monkeypatch.delenv(emit.SPOOL_ENV)
    # the name override: validated, case-insensitive, wins over the scheduler variable and the probes, moves no spool
    monkeypatch.setenv(emit.SUBSTRATE_ENV, "modal")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-one")
    assert emit.substrate_override() == "modal" and emit.substrate_from_env() == "modal" and emit.spool_root() is None
    assert emit.producer_block("t", code_sha=synth.code_sha("a"))["substrate"] == "modal"
    monkeypatch.setenv(emit.SUBSTRATE_ENV, " RunPod ")
    assert emit.substrate_from_env() == "runpod"
    monkeypatch.setenv(emit.SUBSTRATE_ENV, "")
    assert emit.substrate_from_env() == "cluster_a"                    # empty is unset: the site map names the job's cluster
    capsys.readouterr()
    monkeypatch.setenv(emit.SUBSTRATE_ENV, "laptop")
    assert emit.substrate_override() is None and emit.substrate_from_env() == "cluster_a"
    err = capsys.readouterr().err
    assert err.count("is not a substrate") == 1 and "SISTEER_REGISTRY_SUBSTRATE='laptop'" in err and "|".join(V.SUBSTRATES) in err
    assert emit.substrate_from_env() == "cluster_a" and capsys.readouterr().err == ""   # said once per process
    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    assert emit.substrate_from_env() == "grading_host"


def test_emit_cli_runs_under_python_dash_S_as_a_module(tmp_path):
    """The job shell's exact invocation: ``python3 -S -m sisteer.registry.emit_cli state …`` before any venv."""
    env = {"PYTHONPATH": SRC, "PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "USER": "nobody-synthetic"}
    plan = {k: v for k, v in synth.run_payload("cli_key").items() if k != "run_id"}
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(plan))
    out = subprocess.run(["python3", "-S", "-m", "sisteer.registry.emit_cli", "register", "--plan", str(pf), "--spool", str(tmp_path / "spool"),
                          "--job-id", "4242", "--pending-dir", str(tmp_path / "pending"), "--no-push"],
                         capture_output=True, text=True, env=env, cwd="/", timeout=60)
    assert out.returncode == 0, out.stderr
    assert len(out.stdout.strip().splitlines()) == 3 and (tmp_path / "pending" / "4242.json").exists()
    out = subprocess.run(["python3", "-S", "-m", "sisteer.registry.emit_cli", "state", "--state", "running", "--pending-dir", str(tmp_path / "pending"),
                          "--job", "4242", "--node", "TESTHOST", "--gpu-count", "1", "--spool", str(tmp_path / "spool")],
                         capture_output=True, text=True, env=env, cwd="/", timeout=60)
    assert out.returncode == 0, out.stderr
    recs = [rec for p in R.iter_record_files(str(tmp_path / "spool")) for _, rec, _ in R.read_record_file(p)]
    assert [r["record_type"] for r in recs] == ["run", "run_state", "run_patch", "run_state", "run_patch"]
    assert recs[3]["payload"]["state"] == "running" and recs[4]["payload"]["fields"]["node_hostname"] == "TESTHOST"
