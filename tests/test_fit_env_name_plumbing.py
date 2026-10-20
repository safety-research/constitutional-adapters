"""Stage-E prep pass 2: the fit family runs on the TARGET cluster's prebuilt fit venv. Before this, every fit-family
sbatch (02 memprobe, 03 recon / trials, 04 fit cells / derived builds / the refusal direction) defaulted to `sdfbase-fit` on the
job side whatever the site config said — absent on the overflow cluster, so a twin there died at $VENV/bin/python — and the
attempt record carried no venv identity. Now: (1) `cluster_fit_env_name` / `Site.fit_env_name` resolve CLUSTER_FIT_ENV_NAME >
SISTEER_FIT_ENV_NAME > sdfbase-fit; (2) EVERY job env (lib/slurm.sh slurm_job_env, submit_twin.job_env, the stages' no-lib
fallbacks) carries SDFBASE_FIT_ENV_NAME=<that cluster's name> as a VAR=value prefix — never --export=; (3) the grading-host-side fit
venv paths (02 cl_python, 04 FIT_VENV_PY) derive from the same name; (4) a `--kind fit` submission reads the venv's
BUILD-COMPLETE (freeze sha16) + requirements sha256 on the target cluster BEFORE minting an identity, records them on the
attempt, refuses a venv without BUILD-COMPLETE by name, and refuses a cross-cluster pair whose identities differ by name.
Synthetic site (cluster_a / cluster_b, fake Slurm under $FAKE_SLURM_DIR); nothing here names a real host, user or path."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, job_ids, make_registry, write_fake_slurm, write_site  # noqa: E402

SHA = "d" * 40
FIT_SCRIPTS = ("scripts/sbatch/memprobe_fit.sbatch", "scripts/sbatch/hpo_trial.sbatch", "scripts/sbatch/sdfbase_fit_cell.sbatch",
               "scripts/sbatch/sdfbase_derived_build.sbatch", "scripts/sbatch/fit_refusal_direction.sbatch")
MODEL = "qwen35_4b_demo"


def _site(tmp_path: Path, *, a_name: str | None = "venv-a", b_name: str | None = "venv-b", global_name: str | None = "venv-global", ssh=None) -> Path:
    """cluster_a's block names `a_name` (None = no CLUSTER_FIT_ENV_NAME line), cluster_b's `b_name`; the config's SISTEER_FIT_ENV_NAME = `global_name`."""
    extra_cfg = f"SISTEER_FIT_ENV_NAME={global_name}\n" if global_name else ""
    site = write_site(tmp_path, ssh=ssh, extra_config=extra_cfg)
    for lab, name in (("cluster_a", a_name), ("cluster_b", b_name)):
        blk = site / "clusters" / f"{lab}.env"
        text = "\n".join(ln for ln in blk.read_text().splitlines() if not ln.startswith("CLUSTER_FIT_ENV_NAME=")) + "\n"
        if name is not None:
            text += f"CLUSTER_FIT_ENV_NAME={name}\n"
        blk.write_text(text)
    return site


def _sh(site: Path, body: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ, SISTEER_STATE_DIR=str(site.parent / "state")); e.pop("SISTEER_FIT_ENV_NAME", None)
    e.update(env or {})
    return subprocess.run(["bash", "-c", f"source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={site}/config.env; repro_load_config; {body}"],
                          capture_output=True, text=True, env=e)


def test_shell_resolution_cluster_block_then_global_then_the_sbatch_default(tmp_path):
    site = _site(tmp_path, a_name=None, b_name="venv-b", global_name="venv-global")
    r = _sh(site, 'echo "A=$(cluster_fit_env_name cluster_a) B=$(cluster_fit_env_name cluster_b)"; slurm_job_env cluster_a; echo; slurm_job_env cluster_b')
    assert r.returncode == 0, r.stdout + r.stderr
    assert "A=venv-global B=venv-b" in r.stdout                                      # the block wins for its cluster, the global fills the other
    envs = [ln for ln in r.stdout.splitlines() if ln.startswith("SISTEER_SHARED_ROOT=")]
    assert len(envs) == 2 and "SDFBASE_FIT_ENV_NAME=venv-global" in envs[0] and "SDFBASE_FIT_ENV_NAME=venv-b" in envs[1], envs
    assert "SISTEER_CLUSTER_QOS=qos_cluster_a" in envs[0]                               # the existing table is intact
    # no block, no global: the sbatches' own default, said by name
    site2 = _site(tmp_path / "two", a_name=None, b_name=None, global_name=None)
    r = _sh(site2, 'echo "A=$(cluster_fit_env_name cluster_a)"')
    assert r.returncode == 0 and "A=sdfbase-fit" in r.stdout, r.stdout + r.stderr
    # clusters.py exposes the key with an empty default (an unset block passes through as '')
    out = subprocess.run([sys.executable, str(LIB / "clusters.py"), "--site-dir", str(site), "--json"], capture_output=True, text=True, check=True).stdout
    blocks = json.loads(out)
    assert blocks["cluster_a"]["CLUSTER_FIT_ENV_NAME"] == "" and blocks["cluster_b"]["CLUSTER_FIT_ENV_NAME"] == "venv-b"


def _dry(site: Path, stage: str, *args: str, timeout: int = 600) -> str:
    e = dict(os.environ); e.pop("SISTEER_SITE_DIR", None); e.pop("SISTEER_CONFIG", None); e.pop("SISTEER_FIT_ENV_NAME", None)
    e["SISTEER_STATE_DIR"] = str(site.parent / "state")
    r = subprocess.run(["bash", str(REPRO / stage), "--dry-run", *args, "--config", str(site / "config.env")], capture_output=True, text=True, env=e, cwd=str(REPO), timeout=timeout)
    assert r.returncode == 0, (stage, args, (r.stdout + r.stderr)[-2500:])
    return r.stdout + r.stderr


def _fit_sbatch_lines(out: str) -> list:
    return [ln for ln in out.splitlines() if "sbatch " in ln and any(s in ln for s in FIT_SCRIPTS)]


@pytest.mark.parametrize("stage,args,cluster,expect_name", [
    ("02_prepare_model.sh", ("--step", "memprobe", "--model", MODEL), "cluster_a", "venv-a"),
    ("03_hpo.sh", ("--object-class", "relu", "--phase", "all", "--model", MODEL), "cluster_a", "venv-a"),
    ("03_hpo.sh", ("--object-class", "lora", "--phase", "all", "--model", MODEL), "cluster_a", "venv-a"),
    ("04_fit_objects.sh", ("--object", "all", "--model", MODEL), "cluster_a", "venv-a"),
    ("04_fit_objects.sh", ("--object", "all", "--cluster", "cluster_b", "--model", MODEL), "cluster_b", "venv-b"),
])
def test_every_fit_family_sbatch_line_carries_the_target_clusters_venv_name(tmp_path, stage, args, cluster, expect_name):
    site = _site(tmp_path, a_name="venv-a", b_name="venv-b", global_name="venv-global")
    out = _dry(site, stage, *args)
    lines = _fit_sbatch_lines(out)
    assert lines, (stage, args, out[-2000:])                                             # the dry run planned at least one fit-family submission
    for ln in lines:
        assert f"SDFBASE_FIT_ENV_NAME={expect_name}" in ln and "--export=" not in ln, (stage, ln[:400])
        assert f"ssh {cluster.replace('_', '-')}-login" in ln, (stage, ln[:200])          # the line goes to the cluster whose name it carries
        assert "sdfbase-fit/" not in ln and "SDFBASE_FIT_ENV_NAME=venv-global" not in ln, ln[:400]   # never the global when the block names one
    # the grading-host-side venv paths derive from the SAME name
    if stage == "04_fit_objects.sh":
        assert f"/envs/{expect_name}/bin/python" in out and "/envs/sdfbase-fit/bin/python" not in out and "/envs/venv-global/" not in out


def test_02_grading_host_side_python_uses_the_primary_clusters_name(tmp_path):
    site = _site(tmp_path, a_name="venv-a", b_name="venv-b", global_name="venv-global")
    out = _dry(site, "02_prepare_model.sh", "--step", "all", "--model", MODEL)
    assert "/shared_cluster_a/envs/venv-a/bin/python" in out and "/envs/venv-global/" not in out and "/envs/sdfbase-fit/" not in out
    # the memprobe submission carries the primary's name in its env prefix (through lib/slurm.sh -> submit_twin.job_env)
    mem = [ln for ln in out.splitlines() if "memprobe_fit.sbatch" in ln and "sbatch " in ln]
    assert mem and all("SDFBASE_FIT_ENV_NAME=venv-a" in ln for ln in mem), mem[:2]


# ----------------------------------------------------------------------------------------- the certification record (submit_twin)
class FitWorld:
    """Two fake clusters; `envs/<name>/BUILD-COMPLETE` + `envs/<name>.build/requirements.txt` laid down per the sbatch's own format."""

    def __init__(self, tmp: Path, a_name="venv-a", b_name="venv-a"):
        self.tmp = tmp
        self.fake = {lab: tmp / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
        for d in self.fake.values():
            d.mkdir(); (d / "policy").write_text("start_on_submit\n")
        self.bin = write_fake_slurm(tmp / "bin")
        self.site = _site(tmp, a_name=a_name, b_name=b_name, global_name=None, ssh={lab: f"local:{d}" for lab, d in self.fake.items()})
        self.shared = {lab: tmp / f"shared_{lab}" for lab in self.fake}
        for lab, sh in self.shared.items():
            co = sh / "sisteer-eval-repo" / SHA / "scripts/sbatch"; co.mkdir(parents=True)
            (co / "hpo_trial.sbatch").write_text("#!/bin/bash\nf22_attempt_wait   # sisteer-attempt-wait-hook\n")
        make_registry(tmp / "registry")
        self.state = tmp / "state" / "m"
        self.names = {"cluster_a": a_name, "cluster_b": b_name}

    def certify(self, lab: str, freeze_sha: str, req_text: str = "torch==2.11.0\ntransformers==5.15.0\n", name: str | None = None):
        name = name or self.names[lab]
        venv = self.shared[lab] / "envs" / name; venv.mkdir(parents=True, exist_ok=True)
        (venv / "BUILD-COMPLETE").write_text(f"2026-09-16T05:09:17+00:00 {freeze_sha}\n")     # sdfbase_fit_env.sbatch: `printf '%s %s\n' "$(date -Is)" "$sha"`
        bd = self.shared[lab] / "envs" / f"{name}.build"; bd.mkdir(parents=True, exist_ok=True)
        (bd / "requirements.txt").write_text(req_text)

    def env(self, **extra):
        e = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", SISTEER_STATE_DIR=str(self.tmp / "state"), SISTEER_NO_WATCHER="1",
                 SISTEER_REGISTRY_ROOT=str(self.tmp / "registry"))
        e.update(extra); return e

    def submit(self, *extra, twin=False, clusters="cluster_a", dry=False):
        args = [*(["--dry-run"] if dry else []), "submit", "--unit", "recon", "--lwid", "m/hpo/relu/recon", "--script", "scripts/sbatch/hpo_trial.sbatch", "--name", "hpo-m-recon",
                "--args", "spec.json root relu cfg pairs", "--clusters", clusters, "--kind", "fit", "--wave-sha", SHA, *extra]
        if twin:
            args += ["--twin", "--foreground", "--poll-s", "0.2", "--max-wait-s", "20"]
        return subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(self.site / "config.env"), "--model", "m", *args],
                              capture_output=True, text=True, env=self.env(), timeout=120)

    def attempts(self):
        p = self.state / "attempts.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def test_fit_attempt_records_the_venvs_build_complete_sha_and_requirements_sha(tmp_path):
    w = FitWorld(tmp_path); w.certify("cluster_a", "40b49ecf00359e44")
    r = w.submit()
    assert r.returncode == 0 and "submitted cluster_a " in r.stdout, r.stdout + r.stderr
    att = [e for e in w.attempts() if e["event"] == "attempt"][0]
    fe = att["fit_env"]
    assert fe["name"] == "venv-a" and fe["venv"] == f"{w.shared['cluster_a']}/envs/venv-a" and fe["build_complete_sha"] == "40b49ecf00359e44"
    assert fe["build_complete_utc"] == "2026-09-16T05:09:17+00:00" and re.fullmatch(r"[0-9a-f]{64}", fe["requirements_sha256"])
    # the job env carried the cluster's venv name as a VAR=value prefix (the fake sbatch records its environment), never --export=
    jid = job_ids(w.fake["cluster_a"])[0]
    assert "SDFBASE_FIT_ENV_NAME=venv-a" in (w.fake["cluster_a"] / "jobs" / f"{jid}.env").read_text()
    assert "--export" not in (w.fake["cluster_a"] / "jobs" / f"{jid}.argv").read_text()


def test_fit_submission_refuses_a_venv_without_build_complete_by_name(tmp_path):
    w = FitWorld(tmp_path)                                    # nothing certified on cluster_a
    r = w.submit()
    assert r.returncode == 2 and "REFUSED: fit venv venv-a on cluster_a is not certified" in r.stderr and "BUILD-COMPLETE" in r.stderr, r.stderr
    assert job_ids(w.fake["cluster_a"]) == [] and not (w.state / "attempts.jsonl").exists()          # nothing minted, nothing submitted
    # a --twin pair is checked on BOTH clusters before the primary is minted
    w.certify("cluster_a", "40b49ecf00359e44")
    r = w.submit(twin=True, clusters="cluster_a,cluster_b")
    assert r.returncode == 2 and "venv-a on cluster_b is not certified" in r.stderr, r.stderr
    assert job_ids(w.fake["cluster_a"]) == [] and job_ids(w.fake["cluster_b"]) == []


def test_twin_across_clusters_refuses_differing_stacks_by_name_and_admits_identical_ones(tmp_path):
    w = FitWorld(tmp_path, a_name="venv-a", b_name="venv-b")
    w.certify("cluster_a", "40b49ecf00359e44"); w.certify("cluster_b", "b44d4b420846121c")          # two builds, two freezes
    r = w.submit(twin=True, clusters="cluster_a,cluster_b")
    assert r.returncode == 2 and "--twin refused" in r.stderr and "BUILD-COMPLETE freeze sha 40b49ecf00359e44 != b44d4b420846121c" in r.stderr, r.stderr
    assert "venv-a" in r.stderr and "venv-b" in r.stderr and job_ids(w.fake["cluster_a"]) == []
    # the same freeze but another recipe: refused too (the requirements sha is part of the identity when both sides record one)
    w.certify("cluster_b", "40b49ecf00359e44", req_text="torch==2.11.0\ntransformers==5.14.1\n")
    r = w.submit(twin=True, clusters="cluster_a,cluster_b")
    assert r.returncode == 2 and "requirements.txt sha256" in r.stderr, r.stderr
    # identical identities on both: the twin runs, each attempt records ITS cluster's venv name and the shared identity
    w.certify("cluster_b", "40b49ecf00359e44")
    (w.fake["cluster_a"] / "policy").write_text("")            # a stays PENDING so the twin goes out to b (threshold 0 in the site)
    r = w.submit(twin=True, clusters="cluster_a,cluster_b")
    assert r.returncode == 0, r.stdout + r.stderr
    atts = [e for e in w.attempts() if e["event"] == "attempt"]
    assert [a["cluster"] for a in atts] == ["cluster_a", "cluster_b"]
    assert atts[0]["fit_env"]["name"] == "venv-a" and atts[1]["fit_env"]["name"] == "venv-b"
    assert atts[0]["fit_env"]["build_complete_sha"] == atts[1]["fit_env"]["build_complete_sha"] == "40b49ecf00359e44"
    for lab in ("cluster_a", "cluster_b"):
        jid = job_ids(w.fake[lab])[0]
        assert f"SDFBASE_FIT_ENV_NAME={w.names[lab]}" in (w.fake[lab] / "jobs" / f"{jid}.env").read_text()


def test_dry_run_reads_nothing_and_a_non_fit_kind_reads_no_venv(tmp_path):
    w = FitWorld(tmp_path)                                    # nothing certified anywhere
    r = w.submit(dry=True)                                    # --dry-run is the CLI's top-level flag (before the subcommand)
    assert r.returncode == 0 and "DRY-RUN: ssh" in r.stdout and "SDFBASE_FIT_ENV_NAME=venv-a" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m", "submit", "--unit", "u",
                        "--lwid", "m/s1/static/u", "--script", "scripts/sbatch/hpo_trial.sbatch", "--name", "n", "--args", "x", "--clusters", "cluster_a",
                        "--kind", "target", "--wave-sha", SHA], capture_output=True, text=True, env=w.env(), timeout=120)
    assert r.returncode == 0 and "submitted cluster_a" in r.stdout, r.stdout + r.stderr       # a target unit needs no fit venv
    assert [e for e in w.attempts() if e["event"] == "attempt"][0]["fit_env"] is None


# ---- the records' campaign is STATED to every fit-family job (2026-10-02) ----------------------------------------------------------

def test_every_fit_family_job_env_states_the_registry_campaign_default_sdfbase_full22(tmp_path):
    """The native emitters refuse without REGISTRY_CAMPAIGN since 2026-10-02 (their silent `two_si` default filed a front-door
    model's fit objects outside the paper frame's sdfbase_full22). So the front door STATES it on every fit-family job env — the Slurm branch
    (submit_twin.job_env / lib/slurm.sh slurm_job_env) and 04's no-lib fallback alike — as SISTEER_REGISTRY_CAMPAIGN, default
    sdfbase_full22 (the key and default slurm_base_env / 08_grade already use); `two_si` only when a caller names it."""
    site = _site(tmp_path, a_name="venv-a", b_name="venv-b", global_name="venv-global")
    out = _dry(site, "04_fit_objects.sh", "--object", "all", "--model", MODEL)
    lines = _fit_sbatch_lines(out)
    assert lines
    for ln in lines:
        assert "REGISTRY_CAMPAIGN=sdfbase_full22" in ln and "REGISTRY_CAMPAIGN=two_si" not in ln and "--export=" not in ln, ln[:400]
    out3 = _dry(site, "03_hpo.sh", "--object-class", "relu", "--phase", "all", "--model", MODEL)
    assert all("REGISTRY_CAMPAIGN=sdfbase_full22" in ln for ln in _fit_sbatch_lines(out3)), out3[-1500:]
    # the site names another campaign: every job carries it instead
    site2 = _site(tmp_path / "two", a_name="venv-a", b_name="venv-b", global_name="venv-global")
    with open(site2 / "config.env", "a") as fh:
        fh.write("SISTEER_REGISTRY_CAMPAIGN=paper_frame\n")
    out2 = _dry(site2, "04_fit_objects.sh", "--object", "all", "--model", MODEL)
    assert all("REGISTRY_CAMPAIGN=paper_frame" in ln for ln in _fit_sbatch_lines(out2)), out2[-1500:]
    # the shell job env (lib/slurm.sh) carries the same resolution
    r = _sh(site, "slurm_job_env cluster_a")
    assert r.returncode == 0 and "REGISTRY_CAMPAIGN=sdfbase_full22" in r.stdout, r.stdout + r.stderr
    r = _sh(site2, "slurm_job_env cluster_a")
    assert r.returncode == 0 and "REGISTRY_CAMPAIGN=paper_frame" in r.stdout, r.stdout + r.stderr
    # 04's no-lib fallback resolves the same way and passes the value explicitly (never a bare two_si default)
    src = (REPRO / "04_fit_objects.sh").read_text()
    assert 'CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"' in src and 'REGISTRY_CAMPAIGN:-two_si' not in src
    assert "REGISTRY_CAMPAIGN=%q" in src
