"""Every script that reaches a cluster (an ssh or rsync target, a `host:path`, a printed command an operator copies) takes the
cluster's login host from the cluster's block in the site config, never from the cluster's registry label, and refuses by name
when the site config does not describe the cluster.

The site written here gives each cluster three different names: a site label (the clusters/<label>.env file: site_one,
site_two), a registry label (CLUSTER_SUBSTRATE, the name data carries: REG["a"], REG["b"]) and an ssh alias (login-a,
login-b). A script that still used the registry label as a host would hand `REG["b"]` to ssh; every case below asserts the
alias instead. Fake `ssh`, `rsync` and `sbatch` on PATH record what they were given; nothing leaves this machine."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
LIB = REPO / "reproduce" / "lib"
SHA = "ab" * 20
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_site import REGISTRY as REG, SITE_LABEL as SITE, expect, write_site  # noqa: E402


# --------------------------------------------------------------------------------------------------------------- helpers

def fakes(tmp: Path, ssh_out: str = "", ssh_rc: int = 1) -> tuple[Path, Path]:
    """`ssh`, `rsync` and `sbatch` stand-ins on a bin dir: each appends its name and argv (JSON) to the record. ssh prints
    `ssh_out` and exits `ssh_rc`; rsync fails; sbatch prints a job id."""
    bindir, rec = tmp / "bin", tmp / "calls.jsonl"
    bindir.mkdir(exist_ok=True)
    for name, tail in (("ssh", f"printf '%b' {json.dumps(ssh_out)}; exit {ssh_rc}"), ("rsync", "exit 23"),
                       ("sbatch", "echo 4242")):
        p = bindir / name
        p.write_text("#!/bin/bash\n"
                     f"python3 -c 'import json,sys; print(json.dumps([sys.argv[1]] + sys.argv[2:]))' {name} \"$@\" >> {rec}\n"
                     f"{tail}\n")
        p.chmod(0o755)
    return bindir, rec


def calls(rec: Path, name: str) -> list:
    if not rec.exists():
        return []
    return [c[1:] for c in (json.loads(ln) for ln in rec.read_text().splitlines()) if c[0] == name]


def env_for(cfg: Path | None, bindir: Path | None = None, **more) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("SISTEER_CONFIG", "SISTEER_SITE_DIR")}
    if cfg is not None:
        env["SISTEER_CONFIG"] = str(cfg)
    else:
        env["SISTEER_SITE_DIR"] = "/nonexistent-site-dir"
    if bindir is not None:
        env["PATH"] = f"{bindir}:{env['PATH']}"
    env.update({k: str(v) for k, v in more.items()})
    return env


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # registered first: a dataclass looks its module up while the module executes
    spec.loader.exec_module(mod)
    return mod


def run(argv, env, cwd=REPO, timeout=180):
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)


@pytest.fixture
def site(tmp_path, monkeypatch):
    cfg = write_site(tmp_path)
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    return cfg


@pytest.fixture
def site_without_b(tmp_path, monkeypatch):
    cfg = write_site(tmp_path, keys=("a",))
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    return cfg


# --------------------------------------------------------------------------------------------- the one reader of the blocks

def test_clusters_resolves_a_site_label_or_a_registry_label_and_refuses_the_rest(tmp_path):
    CL = load("clusters_under_test", LIB / "clusters.py")
    cfg = write_site(tmp_path)
    blocks = CL.load_all(str(cfg.parent))
    assert CL.resolve_label(blocks, SITE["b"]) == SITE["b"] and CL.resolve_label(blocks, REG["b"]) == SITE["b"]
    with pytest.raises(SystemExit, match="cluster_zz"):
        CL.resolve_label(blocks, "cluster_zz")
    assert CL.block_for(REG["a"], str(cfg))["CLUSTER_SSH"] == "login-a"
    same = write_site(tmp_path / "same", extra={})        # two blocks carrying one registry label is ambiguous, refused
    (same.parent / "clusters" / f"{SITE['b']}.env").write_text(
        (same.parent / "clusters" / f"{SITE['b']}.env").read_text().replace(f"CLUSTER_SUBSTRATE={REG['b']}", f"CLUSTER_SUBSTRATE={REG['a']}"))
    with pytest.raises(SystemExit, match="more than one"):
        CL.resolve_label(CL.load_all(str(same.parent)), REG["a"])
    r = run([sys.executable, str(LIB / "clusters.py"), "--config", str(cfg), "--get", REG["b"], "CLUSTER_SSH"], env_for(cfg))
    assert r.returncode == 0 and r.stdout.strip() == "login-b", r.stderr
    r = run([sys.executable, str(LIB / "clusters.py"), "--config", str(cfg), "--get", "cluster_zz", "CLUSTER_SSH"], env_for(cfg))
    assert r.returncode == 3 and "cluster_zz" in r.stderr and r.stdout == ""


# ------------------------------------------------------------------------------------------------------ shell scripts

def test_mirror_sampled_reads_the_waves_cluster_from_the_site(tmp_path):
    cfg = write_site(tmp_path)
    bindir, rec = fakes(tmp_path)
    script = F22 / "sampled" / "mirror_sampled.sh"
    r = run(["bash", str(script), "--dry-run", "--dest", str(tmp_path / "dest")], env_for(cfg, bindir))
    (argv,) = calls(rec, "ssh")
    assert "login-b" in argv and REG["b"] not in argv, argv
    assert f"{tmp_path}/shared_b/exp/sdfbase_full22" in argv, argv
    bad = run(["bash", str(script), "--dry-run", "--dest", str(tmp_path / "dest2")], env_for(write_site(tmp_path / "no_b", keys=("a",)), bindir))
    assert bad.returncode == 2 and REG["b"] in bad.stderr and "CLUSTER_SSH" in bad.stderr, bad.stderr
    assert len(calls(rec, "ssh")) == 1
    only = run(["bash", str(script), "--host", "login-z", "--dest", str(tmp_path / "dest3")], env_for(cfg, bindir))
    assert only.returncode == 2 and "--remote-root" in only.stderr


def test_padded_family_mirror_reads_the_cluster_from_the_site(tmp_path):
    cfg = write_site(tmp_path)
    script = F22 / "static_padding" / "grade_padded_family.sh"
    base = ["bash", str(script), "16384", "pasttense", "--check-only", "--print", "--dest", str(tmp_path / "dest"),
            "--python", sys.executable]
    r = run(base, env_for(cfg))
    assert r.returncode == 0, r.stdout + r.stderr
    mirror = r.stdout.strip().splitlines()[0]
    assert f"login-b:{tmp_path}/shared_b/exp/sdfbase_full22_native/static_padding/transcripts_results/" in mirror, mirror
    assert f"{REG['b']}:" not in mirror
    r = run(base + ["--cluster", REG["a"]], env_for(cfg))
    assert f"login-a:{tmp_path}/shared_a/" in r.stdout.splitlines()[0], r.stdout
    bad = run(base, env_for(write_site(tmp_path / "no_b", keys=("a",))))
    assert bad.returncode == 2 and REG["b"] in bad.stderr and "CLUSTER_SSH" in bad.stderr
    assert "needs --remote-root" in run(base + ["--host", "login-z"], env_for(cfg)).stderr


def test_checkout_stager_reads_cluster_b_from_the_site(tmp_path):
    # a minimal clean checkout carrying the stager and the block reader, so the stager reaches its first ssh
    co = tmp_path / "co"
    for rel in ("scripts/stage_twosi_cluster_b_checkout.sh", "reproduce/lib/clusters.py", "src/sisteer/site.py"):
        (co / rel).parent.mkdir(parents=True, exist_ok=True)
        (co / rel).write_bytes((REPO / rel).read_bytes())
    git = ["git", "-C", str(co), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "c", "--no-verify"], check=True)
    cfg = write_site(tmp_path, extra={"b": {"CLUSTER_PROJECT_ROOT": "$SHARED/project"}})
    bindir, rec = fakes(tmp_path)
    script = co / "scripts" / "stage_twosi_cluster_b_checkout.sh"
    r = run(["bash", str(script)], env_for(cfg, bindir), cwd=co)
    first = calls(rec, "ssh")[0]
    assert "login-b" in first and REG["b"] not in first, (first, r.stderr)
    assert any(f"{tmp_path}/shared_b/project/checkouts/" in a for a in first), first     # the checkout under cluster B's project root
    seen = len(calls(rec, "ssh"))
    bad = run(["bash", str(script)], env_for(write_site(tmp_path / "no_b", keys=("a",)), bindir), cwd=co)
    assert bad.returncode == 2 and REG["b"] in bad.stderr and len(calls(rec, "ssh")) == seen
    assert run(["bash", str(script), "login-z"], env_for(cfg, bindir), cwd=co).returncode == 2


def test_full22_printer_names_each_clusters_login_host_qos_and_directories(tmp_path):
    cfg = write_site(tmp_path)
    r = run(["bash", str(F22 / "submit_full22.sh"), SHA], env_for(cfg))
    assert r.returncode == 0, r.stdout + r.stderr
    ssh_lines = [ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")]
    assert ssh_lines and all(ln.startswith(("ssh login-a '", "ssh login-b '")) for ln in ssh_lines), ssh_lines[:3]
    for k, login in (("a", "login-a"), ("b", "login-b")):
        e = expect(tmp_path, k)
        jobs = [ln for ln in ssh_lines if ln.startswith(f"ssh {login} '") and "sbatch" in ln]
        assert jobs and all(f"cd {e['eval']}/{SHA}" in ln and f"sbatch {e['sbatch']} " in ln for ln in jobs), (k, jobs[:2])
    assert "watch-job.sh --slurm <jobid> --host login-b" in r.stdout and "--host login-a" in r.stdout
    assert f"ssh {REG['b']}" not in r.stdout and f"--host {REG['b']}" not in r.stdout
    bad = run(["bash", str(F22 / "submit_full22.sh"), SHA], env_for(write_site(tmp_path / "no_b", keys=("a",))))
    assert bad.returncode != 0 and REG["b"] in bad.stderr and "ssh " not in bad.stdout


def test_colocate_reads_a_source_clusters_login_host_from_the_site(tmp_path):
    al = load("artifact_locations_under_test", REPO / "scripts" / "sdfbase" / "seedrep" / "artifact_locations.py")
    key, cell = "qwen35_4b_base_eb64_s9", "constv2ctl2-full__lora_r4_ce"
    rs = tmp_path / "rs"
    obj = b"o" * 64
    d = tmp_path / "remote_b" / "cells" / cell
    al.write_record(key=key, cell=cell, cluster=REG["b"], bank_dir=str(d), object_sha256=hashlib.sha256(obj).hexdigest(),
                    fit_fingerprint="c" * 64, kind="fit", rs=rs)
    target = tmp_path / "target_a"
    target.mkdir()
    cfg = write_site(tmp_path)
    bindir, rec = fakes(tmp_path)
    env = {k: v for k, v in env_for(cfg, bindir).items() if not k.startswith("SEEDREP_")}
    env.update(SEEDREP_CLUSTER_ROOT_cluster_a=str(target), SEEDREP_STAGING=str(tmp_path / "staging"))
    script = REPO / "scripts" / "sdfbase" / "seedrep" / "colocate_cells.sh"
    r = run(["bash", str(script), key, REG["a"], "--cells", cell, "--rs", str(rs)], env)
    first = calls(rec, "ssh")[0]
    assert "login-b" in first and REG["b"] not in first, (first, r.stdout, r.stderr)
    env["SISTEER_CONFIG"] = str(write_site(tmp_path / "no_b", keys=("a",)))
    bad = run(["bash", str(script), key, REG["a"], "--cells", cell, "--rs", str(rs)], env)
    assert bad.returncode == 2 and REG["b"] in bad.stderr and len(calls(rec, "ssh")) == 1


def test_petri_drive_reads_its_cluster_from_the_site(tmp_path):
    cfg = write_site(tmp_path)
    bindir, _rec = fakes(tmp_path)
    r = run(["bash", str(F22 / "petri_drive_full22.sh"), "qwen35_4b", "main", REG["b"], SHA], env_for(cfg, bindir))
    assert f"[cluster] {REG['b']} from the site config: ssh=login-b shared={tmp_path}/shared_b" in r.stdout, r.stdout + r.stderr
    bad = run(["bash", str(F22 / "petri_drive_full22.sh"), "qwen35_4b", "main", REG["b"], SHA],
              env_for(write_site(tmp_path / "no_b", keys=("a",)), bindir))
    assert bad.returncode == 2 and f"bad cluster '{REG['b']}'" in bad.stdout


def test_alphasteer_stagers_reach_cluster_b_by_its_alias(tmp_path):
    cfg = write_site(tmp_path)
    bindir, rec = fakes(tmp_path)
    state = {"AS22_RECORDS_ROOT": tmp_path / "rec", "AS22_STAGE_STATE": tmp_path / "stage",
             "AS22_STRENGTH_STAGE_STATE": tmp_path / "strength"}
    for script in ("stage_alphasteer.sh", "stage_strength.sh"):
        before = len(calls(rec, "ssh"))
        r = run(["bash", str(REPO / "scripts" / "alphasteer_f22" / script), SHA, "--dry-run"], env_for(cfg, bindir, **state))
        reached = calls(rec, "ssh")[before:]
        assert reached and "login-b" in reached[0] and REG["b"] not in reached[0], (script, reached, r.stdout, r.stderr)
        bad = run(["bash", str(REPO / "scripts" / "alphasteer_f22" / script), SHA, "--dry-run"],
                  env_for(write_site(tmp_path / f"no_b_{script}", keys=("a",)), bindir, **state))
        assert bad.returncode == 2 and REG["b"] in bad.stderr, (script, bad.stderr)
        assert len(calls(rec, "ssh")) == before + len(reached)


def test_strength_stager_cpu_steps_take_the_sites_partition_and_qos():
    """Its sruns (bake --check, the bank views) run on the partition and QoS of the cluster's site block, read by the same
    lookup that gives its login host and refuses a cluster the site does not describe (the case above)."""
    src = (REPO / "scripts" / "alphasteer_f22" / "stage_strength.sh").read_text()
    sruns = [ln for ln in src.splitlines() if "srun " in ln and not ln.lstrip().startswith("#")]
    assert sruns and all("srun -p ${PART[$1]} --qos=${HIQOS[$1]} " in ln for ln in sruns), sruns
    assert '--get "$1" CLUSTER_PARTITION' in src and "PART[$1]=$p" in src


def test_derived_chain_submits_with_the_sites_host_qos_and_paths(tmp_path):
    cfg = write_site(tmp_path)
    bindir, rec = fakes(tmp_path)
    script = REPO / "scripts" / "sdfbase" / "seedrep" / "submit_derived_chain.sh"
    rs = tmp_path / "rs"
    r = run(["bash", str(script), "qwen35_4b_base_eb64_s9", "reseed", "--sha", SHA, "--local", "--rs", str(rs), "--cluster", REG["b"]],
            env_for(cfg, bindir))
    assert r.returncode == 0, r.stdout + r.stderr
    subs = calls(rec, "sbatch")
    assert subs and all("--qos=qos-b" in a and f"--output={tmp_path}/shared_b/logs/%x_%j.out" in a
                        and f"{tmp_path}/shared_b/evalrepo/{SHA}/scripts/sbatch/sdfbase_derived_build.sbatch" in a for a in subs), subs[:1]
    led = [json.loads(ln) for ln in (rs / "reseed" / "submissions.jsonl").read_text().splitlines()]
    assert all(x["cluster"] == REG["b"] and x["host"] == "login-b" and x["qos"] == "qos-b" for x in led)
    bad = run(["bash", str(script), "qwen35_4b_base_eb64_s9", "reseed", "--sha", SHA, "--local", "--rs", str(tmp_path / "rs2"),
               "--cluster", REG["b"]], env_for(write_site(tmp_path / "no_b", keys=("a",)), bindir))
    assert bad.returncode == 2 and REG["b"] in bad.stderr and len(calls(rec, "sbatch")) == len(subs)


def test_cluster_a_jobs_carry_the_sites_partition_log_dir_account_and_extra_flags(tmp_path, monkeypatch):
    """The job scripts' own headers name one site's partition and log directory; every printed or submitted job on either
    cluster overrides them with its site block's values (here an alternate shared root), plus the account and extra flags."""
    extra = {"a": {"CLUSTER_ACCOUNT": "acct-a", "CLUSTER_SBATCH_EXTRA": "--exclude=hostq-a"}}
    cfg = write_site(tmp_path, extra=extra)
    a = expect(tmp_path, "a")
    want = f"--qos=qos-a --partition=part-a --account=acct-a --output={a['logs']}/%x_%j.out"
    r = run(["bash", str(F22 / "submit_full22.sh"), SHA], env_for(cfg))
    assert r.returncode == 0, r.stdout + r.stderr
    a_jobs = [ln for ln in r.stdout.splitlines() if ln.startswith("ssh login-a '") and "sbatch" in ln]
    assert a_jobs and all(f"sbatch {want} --exclude=hostq-a " in ln for ln in a_jobs), a_jobs[:2]
    bindir, rec = fakes(tmp_path)
    script = REPO / "scripts" / "sdfbase" / "seedrep" / "submit_derived_chain.sh"
    r = run(["bash", str(script), "qwen35_4b_base_eb64_s9", "reseed", "--sha", SHA, "--local", "--rs", str(tmp_path / "rs"),
             "--cluster", REG["a"]], env_for(cfg, bindir))
    assert r.returncode == 0, r.stdout + r.stderr
    subs = calls(rec, "sbatch")
    assert subs and all(f"--output={a['logs']}/%x_%j.out" in argv and "--partition=part-a" in argv and "--qos=qos-a" in argv
                        and "--account=acct-a" in argv and "--exclude=hostq-a" in argv for argv in subs), subs[:1]
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    ss = load("submit_sampled_under_test_a", F22 / "sampled" / "submit_sampled.py")
    job = ss.Job(name="j1", key="k", phase="p", cluster=REG["a"], time="1:00:00", block="base", est_h="1", args="a W b")
    assert ss.sbatch_line(job, SHA, {REG["a"]: "hostz"}) == (
        f"ssh login-a 'cd {a['eval']}/{SHA} && sbatch {want} --time=1:00:00 --exclude=hostq-a --exclude=hostz -J j1 "
        f"scripts/sdfbase/full22/sampled/target_sampled.sbatch a {SHA} b'")
    nm = load("submit_newmodels_under_test_a", F22 / "submit_newmodels.py")
    assert nm.where(REG["a"])["sbatch"] == f"{want} --exclude=hostq-a"


# ------------------------------------------------------------------------------------------------------ python modules

def test_strength_stager_routes_a_registry_label_through_the_site(tmp_path):
    cfg = write_site(tmp_path)
    bindir, rec = fakes(tmp_path, ssh_out="MISSING\n0\n", ssh_rc=0)
    bank = tmp_path / "bank"
    cell = bank / "m" / "cells" / "dose__c"
    cell.mkdir(parents=True)
    (cell / "object.pt").write_bytes(b"x" * 100)
    (cell / "staged_from.json").write_text(json.dumps({"recorded_hash_check": {}}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cells": [{"out": str(cell), "target": "m", "staged_id": "dose__c", "cell_id": "c",
                                             "object_sha256": hashlib.sha256(b"x" * 100).hexdigest(), "scale": 0.75,
                                             "parent_staged_id": "p", "parent_object_sha256": "0" * 64, "producer": "t",
                                             "object_type": "lora", "scaling_method": "s"}]}))
    def argv(site_dir):
        return [sys.executable, str(F22 / "strength" / "stage_strength.py"), "--ledger", str(ledger), "--clusters", REG["b"],
                "--site-dir", str(site_dir), "--bank", str(bank), "--dry-run", "--no-manifest"]
    r = run(argv(cfg.parent), env_for(cfg, bindir, PYTHONPATH=REPO / "src"))
    assert r.returncode in (0, 2) and f"[{REG['b']}] m/dose__c: would_rsync" in r.stdout, r.stdout + r.stderr
    hosts = {c[c.index("ConnectTimeout=30") + 1] for c in calls(rec, "ssh")}
    assert hosts == {"login-b"}, hosts
    ss = load("stage_strength_under_test", F22 / "strength" / "stage_strength.py")
    ss.load_site(str(cfg.parent))
    assert ss.record_key(SITE["b"]) == REG["b"] and ss.record_key(REG["b"]) == REG["b"]
    assert ss.shared_of(REG["a"]) == f"{tmp_path}/shared_a"
    bad = run(argv(write_site(tmp_path / "no_b", keys=("a",)).parent), env_for(cfg, bindir, PYTHONPATH=REPO / "src"))
    assert bad.returncode != 0 and REG["b"] in bad.stdout + bad.stderr


def test_stage_objects_finds_a_pins_source_cluster_by_its_registry_label(tmp_path, monkeypatch):
    so = load("stage_objects_under_test", F22 / "stage_objects.py")
    cfg = write_site(tmp_path)
    blocks = so.CL.load_all(str(cfg.parent))
    hosts = so.hosts_by_label(blocks, False)
    assert hosts[REG["b"]].ssh == "login-b" and hosts[SITE["b"]] is hosts[REG["b"]]
    assert so.record_key(blocks, SITE["a"]) == REG["a"]
    bindir, rec = fakes(tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    with pytest.raises(subprocess.CalledProcessError):
        so.pull_source(f"{REG['b']}:/remote/cell", tmp_path / "dest", hosts, False)
    (argv,) = calls(rec, "rsync")
    assert "login-b:/remote/cell/" in argv and not any(a.startswith(f"{REG['b']}:") for a in argv), argv
    with pytest.raises(SystemExit, match=REG["b"]):
        so.pull_source(f"{REG['b']}:/remote/cell", tmp_path / "dest", so.hosts_by_label({SITE["a"]: blocks[SITE["a"]]}, False), False)


def _fan_out_to_a_only(cfg: Path) -> Path:
    """The site write_site wrote, with SISTEER_CLUSTERS naming cluster a alone while cluster b's block stays under clusters/: a site
    that submits to one cluster by default and reaches the second only when a caller or the data names it."""
    text = cfg.read_text()
    listed = text.replace(f'SISTEER_CLUSTERS="{SITE["a"]} {SITE["b"]}"', f'SISTEER_CLUSTERS="{SITE["a"]}"')
    assert listed != text
    cfg.write_text(listed)
    return cfg


def test_strength_stager_reaches_a_cluster_outside_the_default_fan_out(tmp_path):
    cfg = _fan_out_to_a_only(write_site(tmp_path))
    bindir, rec = fakes(tmp_path, ssh_out="MISSING\n0\n", ssh_rc=0)
    bank = tmp_path / "bank"
    cell = bank / "m" / "cells" / "dose__c"
    cell.mkdir(parents=True)
    (cell / "object.pt").write_bytes(b"x" * 100)
    (cell / "staged_from.json").write_text(json.dumps({"recorded_hash_check": {}}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cells": [{"out": str(cell), "target": "m", "staged_id": "dose__c", "cell_id": "c",
                                             "object_sha256": hashlib.sha256(b"x" * 100).hexdigest(), "scale": 0.75,
                                             "parent_staged_id": "p", "parent_object_sha256": "0" * 64, "producer": "t",
                                             "object_type": "lora", "scaling_method": "s"}]}))
    argv = [sys.executable, str(F22 / "strength" / "stage_strength.py"), "--ledger", str(ledger), "--clusters", REG["b"],
            "--site-dir", str(cfg.parent), "--bank", str(bank), "--dry-run", "--no-manifest"]
    r = run(argv, env_for(cfg, bindir, PYTHONPATH=REPO / "src"))
    assert r.returncode in (0, 2) and f"[{REG['b']}] m/dose__c: would_rsync" in r.stdout, r.stdout + r.stderr
    assert {c[c.index("ConnectTimeout=30") + 1] for c in calls(rec, "ssh")} == {"login-b"}
    ss = load("stage_strength_unlisted_under_test", F22 / "strength" / "stage_strength.py")
    ss.load_site(str(cfg.parent))
    assert ss.host_of(REG["b"]) == "login-b" and ss.record_key(SITE["b"]) == REG["b"] and ss.shared_of(REG["a"]) == f"{tmp_path}/shared_a"


def test_stage_objects_reaches_a_cluster_outside_the_default_fan_out(tmp_path, monkeypatch, capsys):
    """A pins source on cluster b (named by its registry label) and a destination named on --clusters both resolve through cluster
    b's block when SISTEER_CLUSTERS lists cluster a alone; with no --clusters the destinations stay the listed cluster."""
    so = load("stage_objects_unlisted_under_test", F22 / "stage_objects.py")
    cfg = _fan_out_to_a_only(write_site(tmp_path))
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps([{"cell_id": "c1", "lane": "base", "source": f"{REG['b']}:/remote/cell", "kind": "fit"}]))
    base = ["--site-dir", str(cfg.parent), "--bank", str(tmp_path / "bank"), "--target", "m", "--cells", str(cells), "--no-manifest"]
    assert so.main(base + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"-> {SITE['a']}:{tmp_path}/shared_a/" in out and f"-> {SITE['b']}:" not in out, out
    assert so.main(base + ["--dry-run", "--clusters", REG["b"]]) == 0
    out = capsys.readouterr().out
    assert f"-> {SITE['b']}:{tmp_path}/shared_b/" in out and f"-> {SITE['a']}:" not in out, out
    seen: dict = {}

    def pull(src, dest, hosts, dry):
        seen.update(hosts)
        raise RuntimeError("stopped once the source cluster resolved")
    monkeypatch.setattr(so, "preflight_remote_manifests", lambda *a, **k: {})
    monkeypatch.setattr(so, "pull_source", pull)
    assert so.main(base) == 2
    assert seen[REG["b"]].ssh == "login-b" and seen[SITE["b"]] is seen[REG["b"]] and seen[SITE["a"]].ssh == "login-a"
    blocks = so.CL.load_addressable(str(cfg.parent))
    assert so.record_key(blocks, SITE["b"]) == REG["b"]


def test_screen_verdict_reads_its_cluster_from_the_site(tmp_path):
    cfg = write_site(tmp_path)
    sv = [sys.executable, str(F22 / "strength" / "screen_verdict.py"), "--cluster", REG["b"], "--dry-run",
          "--roster", str(REPO / "configs" / "sdfbase_full22_roster.yaml")]
    r = run(sv, env_for(cfg, PYTHONPATH=REPO / "src"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would run: ssh login-b python3 - --gather" in r.stdout and f"ssh {REG['b']}" not in r.stdout
    assert f"{tmp_path}/shared_b/evalrepo/*/scripts/sdfbase/full22/sampled/at_cap_controls.py" in r.stdout
    rr = run(sv + ["--run-reclassify"], env_for(cfg, PYTHONPATH=REPO / "src"))
    assert rr.returncode == 0 and "reclassify runner: srun --partition=devpart-b --qos=devqos-b on login-b" in rr.stdout, rr.stdout + rr.stderr
    bad = run(sv, env_for(write_site(tmp_path / "no_b", keys=("a",)), PYTHONPATH=REPO / "src"))
    assert bad.returncode == 2 and REG["b"] in bad.stdout + bad.stderr


def test_sampled_and_newmodel_and_leg_printers_read_the_site(site, tmp_path, monkeypatch):
    ss = load("submit_sampled_under_test", F22 / "sampled" / "submit_sampled.py")
    job = ss.Job(name="j1", key="k", phase="p", cluster=REG["b"], time="1:00:00", block="base", est_h="1", args="a W b")
    line = ss.sbatch_line(job, SHA, {})
    assert line.startswith(f"ssh login-b 'cd {tmp_path}/shared_b/evalrepo/{SHA} && sbatch {expect(tmp_path, 'b')['sbatch']} --time="), line
    job_a = ss.Job(name="j1", key="k", phase="p", cluster=REG["a"], time="1:00:00", block="base", est_h="1", args="a W b")
    assert ss.sbatch_line(job_a, SHA, {}).startswith(f"ssh login-a 'cd {tmp_path}/shared_a/evalrepo/{SHA} && sbatch "
                                                     f"{expect(tmp_path, 'a')['sbatch']} --time="), ss.sbatch_line(job_a, SHA, {})
    nm = load("submit_newmodels_under_test", F22 / "submit_newmodels.py")
    assert nm.where(REG["a"])["host"] == "login-a"
    remote = nm.remote_script(REG["b"], SHA, [], [])
    assert f"S={tmp_path}/shared_b;" in remote and f'REPO="{tmp_path}/shared_b/evalrepo/$SHA"' in remote
    bindir, rec = fakes(tmp_path)
    monkeypatch.setenv("F22_SSH", str(bindir / "ssh"))
    nm.probe(REG["b"], SHA, [], [])
    assert "login-b" in calls(rec, "ssh")[0]
    sl = load("submit_legs_under_test", F22 / "strength" / "submit_legs.py")
    assert sl.pool_line(REG["b"], "pool1", "1:00:00", SHA).startswith(f"ssh login-b 'cd {tmp_path}/shared_b/evalrepo/{SHA} && sbatch --qos=qos-b")
    assert sl.site_block(REG["b"])["CLUSTER_SSH"] == "login-b"              # found by registry label, though its file is site_two.env
    assert all(ln.startswith("ssh login-b") for ln in sl.release_commands(REG["b"], ["m1"], [])[1:])


def test_printers_refuse_a_cluster_the_site_does_not_describe(site_without_b):
    ss = load("submit_sampled_under_test2", F22 / "sampled" / "submit_sampled.py")
    with pytest.raises(SystemExit, match=REG["b"]):
        ss.where(REG["b"])
    nm = load("submit_newmodels_under_test2", F22 / "submit_newmodels.py")
    with pytest.raises(SystemExit, match=REG["b"]):
        nm.where(REG["b"])
    sl = load("submit_legs_under_test2", F22 / "strength" / "submit_legs.py")
    with pytest.raises(sl.Refusal, match=REG["b"]):
        sl.site_block(REG["b"])


def test_pull_and_fetch_locations_come_from_the_site(site, tmp_path, monkeypatch):
    ls = load("ladder_select_under_test", REPO / "scripts" / "alphasteer_f22" / "ladder_select.py")
    assert ls.pull_location(REG["b"], None, None) == ("login-b", f"{tmp_path}/shared_b/exp/objects/alphasteer_f22")
    assert ls.pull_location(REG["b"], "login-z", "/r") == ("login-z", "/r")
    with pytest.raises(ls.LadderRefusal, match="--remote-root"):
        ls.pull_location(REG["b"], "login-z", None)
    with pytest.raises(ls.LadderRefusal, match="cluster_zz"):
        ls.pull_location("cluster_zz", None, None)
    br = load("battery_record_under_test", F22 / "strength" / "battery_record.py")
    assert br.remote_for(REG["b"], None) == ("login-b", f"{tmp_path}/shared_b/exp")
    with pytest.raises(SystemExit, match="cluster_zz"):
        br.remote_for("cluster_zz", None)
    # the OR-Bench grader imports two harness modules by name; stand-ins, as its own judge test loads it
    import types

    class Refuse(Exception):
        pass
    gas = types.ModuleType("grade_arditi_static")
    gas.Refuse, gas.DETECTORS, gas.ts = Refuse, (), (lambda: "t")
    gas._detector_view, gas.degeneration_block = (lambda *a, **k: None), (lambda *a, **k: {})
    bc = types.ModuleType("battery_curves")
    bc.GRADED_BY_LABEL, bc.GRADED_AT_CAP, bc.GATE_AT_CAP = {}, 0.5, 1.0
    monkeypatch.setitem(sys.modules, "grade_arditi_static", gas)
    monkeypatch.setitem(sys.modules, "battery_curves", bc)
    ga = load("grade_arditi_orbench_under_test", REPO / "scripts" / "arditi" / "grade_arditi_orbench.py")
    assert ga.cluster_location(REG["a"]) == ("login-a", f"{tmp_path}/shared_a")
    with pytest.raises(Refuse, match="cluster_zz"):
        ga.cluster_location("cluster_zz")


def test_sacct_reads_the_login_host_from_the_site(site):
    dg = load("degeneracy_gap_fill_under_test", F22 / "degeneracy_gap_fill.py")
    asked = []
    got = dg.sacct_states(REG["a"], ["7"], cache={}, query=lambda host, ids: asked.append(host) or {"7": "COMPLETED"})
    assert got == {"7": "COMPLETED"} and asked == ["login-a"]
    with pytest.raises(SystemExit):
        dg.sacct_states("cluster_zz", ["7"], cache={}, query=lambda host, ids: {})


def test_no_script_names_a_registry_label_as_a_host():
    """A static guard over the files this module covers: no ssh / rsync target, host default or host table spells a
    registry label."""
    files = [F22 / "sampled" / "mirror_sampled.sh", F22 / "static_padding" / "grade_padded_family.sh", F22 / "submit_full22.sh",
             F22 / "petri_drive_full22.sh", F22 / "strength" / "stage_strength.py", F22 / "strength" / "screen_verdict.py",
             F22 / "sampled" / "submit_sampled.py", F22 / "submit_newmodels.py", F22 / "strength" / "submit_legs.py",
             F22 / "strength" / "battery_record.py", F22 / "degeneracy_gap_fill.py", F22 / "stage_objects.py",
             REPO / "scripts" / "stage_twosi_cluster_b_checkout.sh", REPO / "scripts" / "sdfbase" / "seedrep" / "colocate_cells.sh",
             REPO / "scripts" / "sdfbase" / "seedrep" / "submit_derived_chain.sh", REPO / "scripts" / "alphasteer_f22" / "stage_alphasteer.sh",
             REPO / "scripts" / "alphasteer_f22" / "stage_strength.sh", REPO / "scripts" / "alphasteer_f22" / "ladder_select.py",
             REPO / "scripts" / "arditi" / "grade_arditi_orbench.py"]
    import re
    pats = []
    for lab in REG.values():
        pats += [rf"\bssh\s+(?:-\S+\s+\S+\s+)*{lab}\b", rf"HOST=\$?\{{?\d*:?-?{lab}\b", rf"\[{lab}\]={lab}\b",
                 rf"[\"']host[\"']\s*:\s*[\"']{lab}[\"']", rf"\b{lab}\)\s*(?:HOST=|echo\s+{lab}\b)", rf"=\s*[\"']{lab}[\"']\s*#.*ssh alias",
                 rf"==\s*[\"']{lab}[\"']\s*else\s*[\"']{{2}}", rf"\[\s*\"\$CLUSTER\"\s*=\s*{lab}\s*\]\s*&&\s*\w+_OPT=",
                 rf"[\"']{lab}[\"']\s*:\s*[\"'](?:dev|high)(?:-eur)?[\"']"]   # no log path, flag or QoS keyed by a label
    hits = [(f.name, ln) for f in files for ln in f.read_text().splitlines()
            if not ln.lstrip().startswith("#") and any(re.search(p, ln) for p in pats)]
    assert not hits, hits
