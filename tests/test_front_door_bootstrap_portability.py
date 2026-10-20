"""BOOTSTRAP PORTABILITY: the job bootstraps honour a configured shared root
(`SISTEER_SHARED_ROOT` + `SISTEER_CLUSTER_LABEL`, `SISTEER_EVAL_REPO_ROOT`, `SISTEER_HF_HOME`) BEFORE their two-root probe,
so a site whose storage is not /workspace* can launch; unset, every job is byte-for-byte what it was. Pytest-collected (the
gate collects only tests/): the REAL edited `common.sh f22_init` runs inside a synthetic checkout (a git repo at the wave
sha) under a NEUTRAL label and a synthetic root; the sbatch bootstraps' own source lines resolve through the override; the
in-body grant hook denies / grants; the REAL `petri_drive_full22.sh` routes a configured label to its ssh host and root;
`stage_strength.py --dry-run` and `submit_twin.py --dry-run` place a configured label; and with the variables UNSET the
probe text is unchanged."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, assert_no_tokens, git, write_fake_ssh, write_site  # noqa: E402

F22 = REPO / "scripts/sdfbase/full22"
SBATCHES = ["target_attacks.sbatch", "target_static.sbatch", "petri_proxy.sbatch", "bon_climb.sbatch", "mmlu_full.sbatch",
            "misalign/misalign_proxy.sbatch", "static_padding/target_static_padded.sbatch", "static_padding/target_static_padded_obj.sbatch"]
OVERRIDE_SRC = '{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \\'


@pytest.fixture
def checkout(tmp_path):
    """A synthetic site: shared root <tmp>/scratch/shared (no /workspace anywhere), the wave checkout at <root>/<eval-repo>/<sha>
    holding the REAL common.sh + attempt_wait.sh + the roster + a src/sisteer stub, committed so f22_init's HEAD check holds."""
    root = tmp_path / "scratch" / "shared"; root.mkdir(parents=True)
    co = root / "sisteer-eval-repo" / "work"; (co / "scripts/sdfbase/full22").mkdir(parents=True)
    (co / "reproduce/lib").mkdir(parents=True); (co / "src/sisteer").mkdir(parents=True); (co / "configs").mkdir()
    shutil.copy(F22 / "common.sh", co / "scripts/sdfbase/full22/common.sh")
    shutil.copy(LIB / "attempt_wait.sh", co / "reproduce/lib/attempt_wait.sh")
    shutil.copy(REPO / "configs/sdfbase_full22_roster.yaml", co / "configs/sdfbase_full22_roster.yaml")
    (co / "src/sisteer/__init__.py").write_text("")
    git(co, "init", "-q"); git(co, "config", "user.email", "t@example.invalid"); git(co, "config", "user.name", "t")
    git(co, "add", "-A"); git(co, "commit", "-qm", "wave")
    sha = git(co, "rev-parse", "HEAD")
    final = co.parent / sha; co.rename(final)
    (root / "hf_cache").mkdir()
    return {"root": root, "sha": sha, "co": final}


def harness(checkout, body: str, env: dict | None = None):
    root, sha = checkout["root"], checkout["sha"]
    script = f'''set -uo pipefail
WAVE_SHA={sha}
{body}
'''
    e = dict(os.environ, SISTEER_SHARED_ROOT=str(root), SISTEER_CLUSTER_LABEL="cluster_a", SISTEER_EVAL_REPO_ROOT=str(root / "sisteer-eval-repo"),
             SISTEER_HF_HOME=str(root / "hf_cache"), F22_SERVING_PATH="lora_native", USER="tester", SLURM_JOB_NAME="f22-test", SLURM_JOB_ID="4242")
    e.pop("SISTEER_REGISTRY_SPOOL", None); e.pop("SISTEER_ATTEMPT_ID", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=120)


def test_f22_init_honours_the_configured_root_before_the_two_root_probe(checkout):
    root, sha = checkout["root"], checkout["sha"]
    r = harness(checkout, OVERRIDE_SRC + '\n  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"\n'
                          'f22_attempt_wait\nF22_LAUNCH_CELLS=()\nf22_init "$WAVE_SHA"\n'
                          'echo "PROBE SHARED=$SHARED CLUSTER=$CLUSTER REPO_DIR=$REPO_DIR HF_HOME=$HF_HOME WAVE_ROOT=$WAVE_ROOT FIGS_DIR=$FIGS_DIR"')
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    probe = [ln for ln in r.stdout.splitlines() if ln.startswith("PROBE ")][0]
    assert f"SHARED={root} CLUSTER=cluster_a REPO_DIR={root}/sisteer-eval-repo/{sha} HF_HOME={root}/hf_cache WAVE_ROOT={root}/exp/sdfbase_full22_native" in probe
    assert (root / "exp/sdfbase_full22_native").is_dir() and (root / "exp/figs/f22-test").is_dir()
    assert "cluster=cluster_a shared=" in r.stdout            # f22_init's own log line names the configured label
    import socket
    scan = r.stdout.replace(str(REPO), "<repo>").replace(str(root), "<root>").replace(socket.gethostname(), "<host>").replace(os.path.expanduser("~"), "<home>")
    assert_no_tokens(scan, scan[-600:])


@pytest.mark.parametrize("name", SBATCHES)
def test_every_sbatch_source_line_carries_the_override_and_keeps_the_probe(name):
    src = (F22 / name).read_text()
    assert OVERRIDE_SRC in src, name
    assert '|| source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"' in src, name
    assert 'source "/workspace' not in src, name
    i_src = src.index(OVERRIDE_SRC); i_hook = src.index("f22_attempt_wait   # sisteer-attempt-wait-hook"); i_init = src.index('f22_init "$WAVE_SHA"')
    assert i_src < i_hook < i_init, name                        # the hook is the FIRST executable line after the source, before f22_init


def test_sampled_arditi_gsm8k_fit_bootstraps_carry_the_override():
    s = (F22 / "sampled/target_sampled.sbatch").read_text()
    assert 'source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/sampled/common_sampled.sh"' in s and "sisteer-attempt-wait-hook" in s
    a = (REPO / "scripts/arditi/sbatch/ar_cell.sbatch").read_text()
    assert 'source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$SHA/scripts/arditi/sbatch/ar_common.sh"' in a and "sisteer-attempt-wait-hook" in a
    for g in (REPO / "scripts/sdfbase/gsm8k").glob("*.sbatch"):
        t = g.read_text()
        assert 'elif [ -n "${SISTEER_SHARED_ROOT:-}" ]; then REPO_DIR="${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA"' in t, g.name
        assert t.index("trap cleanup EXIT SIGTERM SIGINT SIGQUIT") < t.index("sisteer-attempt-wait-hook") < t.index('g8k_init "$WAVE_SHA"'), g.name
    f = (REPO / "scripts/sbatch/sdfbase_fit_cell.sbatch").read_text()
    assert 'if [ -n "${SISTEER_SHARED_ROOT:-}" ]; then SHARED="$SISTEER_SHARED_ROOT"; CLUSTER_QOS="${SISTEER_CLUSTER_QOS:-high}"' in f
    assert '[ -z "${SISTEER_ATTEMPT_ID:-}" ] || source "$REPO/reproduce/lib/attempt_wait.sh"' in f
    assert f.index("attempt_wait.sh") < f.index("slurm_fit_cell.py\" \\")


def test_hook_denies_before_init_and_grants_through(checkout, tmp_path):
    attempts = checkout["root"] / "exp/reproduce/m/attempts"; aid = "m/u0/static/x@cluster_a@1"
    body = OVERRIDE_SRC + '\n  || exit 9\nf22_attempt_wait\necho REACHED_INIT\nF22_LAUNCH_CELLS=()\nf22_init "$WAVE_SHA"\necho INIT_DONE'
    env = {"SISTEER_ATTEMPT_ID": aid, "SISTEER_ATTEMPTS_ROOT": str(attempts), "SISTEER_GRANT_TIMEOUT_MIN": "1", "SISTEER_GRANT_POLL_S": "0.2"}
    (attempts / aid).mkdir(parents=True); (attempts / aid / "DENY").write_text("{}")
    r = harness(checkout, body, env)
    assert r.returncode == 0 and "DENY" in r.stdout and "REACHED_INIT" not in r.stdout, r.stdout + r.stderr
    assert list((attempts / aid).glob("WAITING.*"))                    # the admission side's RUNNING observation marker
    os.remove(attempts / aid / "DENY"); (attempts / aid / "GRANT").write_text("{}")
    r = harness(checkout, body, env)
    assert r.returncode == 0 and "GRANT" in r.stdout and "INIT_DONE" in r.stdout, r.stdout + r.stderr
    # unset: the hook is a no-op line
    r = harness(checkout, body)
    assert "REACHED_INIT" in r.stdout and "[attempt]" not in r.stdout


def test_fit_resubmit_keeps_the_sites_qos(tmp_path):
    """Finding 10: a configured site's fit job resubmits its escalation at ITS QoS (SISTEER_CLUSTER_QOS from the cluster block, passed
    in every site submission's job env — slurm_job_env and submit_twin.job_env) and hands it to the successor on the sbatch line."""
    f = (REPO / "scripts/sbatch/sdfbase_fit_cell.sbatch").read_text()
    assert ('SISTEER_SHARED_ROOT="${SISTEER_SHARED_ROOT:-}" SISTEER_CLUSTER_QOS="$CLUSTER_QOS" \\\n'
            '  SDFBASE_FIT_ENV_NAME="$FIT_ENV_NAME" SDFBASE_FIT_WIDTH="${FIT_WIDTH:-1}" \\\n'
            '  sbatch --qos="$ESCALATE_QOS"') in f
    # the prologue, run for real: the site's QoS wins; unset, the two-root probe's literal stays
    m = re.search(r'^if \[ -n "\$\{SISTEER_SHARED_ROOT:-\}" \]; then SHARED=.*?\nESCALATE_QOS=.*?$', f, re.S | re.M)
    assert m, "the shared-root / QoS prologue moved"
    out = subprocess.run(["bash", "-c", m.group(0) + '\necho "$CLUSTER_QOS $ESCALATE_QOS $SHARED"'], capture_output=True, text=True,
                         env={"PATH": os.environ["PATH"], "USER": "u", "SISTEER_SHARED_ROOT": str(tmp_path), "SISTEER_CLUSTER_QOS": "normal"})
    assert out.stdout.split() == ["normal", "normal", str(tmp_path)], out.stdout + out.stderr
    # the shell env table of every site submission carries it (lib/slurm.sh slurm_job_env), from the cluster block
    site = write_site(tmp_path)
    env = dict(os.environ, SISTEER_STATE_DIR=str(tmp_path / "state"))
    sh = subprocess.run(["bash", "-c", f'source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={site}/config.env; repro_load_config; slurm_job_env cluster_b'],
                        capture_output=True, text=True, env=env)
    assert "SISTEER_CLUSTER_QOS=qos_cluster_b" in sh.stdout, sh.stdout + sh.stderr


def test_unset_shared_root_stops_the_job_by_name(tmp_path):
    """f22_init takes the shared root and the label from the job's site values only: set, they are the job's; unset,
    the job stops by name, and no storage path is ever probed for."""
    common = (F22 / "common.sh").read_text()
    init = common[common.index("f22_init(){"):common.index("f22_stop_requested(){")]
    assert '[ -d "/' not in init
    m = re.search(r'^  if \[ -n "\$\{SISTEER_SHARED_ROOT:-\}" \]; then SHARED=.*?\n  else f22_fatal .*?; fi$', init, re.M | re.S)
    assert m, "f22_init lost its root selection"
    script = 'f22_fatal(){ echo "FATAL $*"; exit 1; }\n' + m.group(0) + '\necho "SHARED=$SHARED CLUSTER=$CLUSTER"'
    unset = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": os.environ["PATH"], "USER": "u"})
    assert unset.returncode == 1 and "FATAL SISTEER_SHARED_ROOT is unset" in unset.stdout, unset.stdout + unset.stderr
    env = {"PATH": os.environ["PATH"], "USER": "u", "SISTEER_SHARED_ROOT": str(tmp_path / "shared"), "SISTEER_CLUSTER_LABEL": "cluster_a"}
    given = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert given.returncode == 0 and f"SHARED={tmp_path / 'shared'} CLUSTER=cluster_a" in given.stdout, given.stdout + given.stderr


def test_petri_drive_routes_a_configured_label(tmp_path):
    """The REAL petri_drive_full22.sh under label cluster_a: HOST/SHARED come from the site's cluster block (a fake ssh records the
    first remote call; the stub venv python satisfies the venv checks); the script reaches its endpoint read on THAT host."""
    site = write_site(tmp_path)
    wt = tmp_path / "wt"; (wt / "scripts/sdfbase/full22").mkdir(parents=True); (wt / ".venv/bin").mkdir(parents=True); (wt / "reproduce/lib").mkdir(parents=True)
    shutil.copy(F22 / "petri_drive_full22.sh", wt / "scripts/sdfbase/full22/petri_drive_full22.sh")
    shutil.copy(LIB / "clusters.py", wt / "reproduce/lib/clusters.py")
    (wt / "src/sisteer").mkdir(parents=True); shutil.copy(REPO / "src/sisteer/site.py", wt / "src/sisteer/site.py")   # clusters.py parses with it
    (wt / ".venv/bin/python").write_text("#!/bin/bash\nexit 0\n"); (wt / ".venv/bin/python").chmod(0o755)
    git(wt, "init", "-q"); git(wt, "config", "user.email", "t@example.invalid"); git(wt, "config", "user.name", "t"); git(wt, "add", "-A"); git(wt, "commit", "-qm", "x")
    sha = git(wt, "rev-parse", "HEAD")
    home = tmp_path / "home"; (home / "stage/sdfbase/full22_objects/gemma4e4b/cells_view/main").mkdir(parents=True)   # the bank under $STAGE (~/stage)
    (home / ".local/state/petri/scenarios/8812ba083bb5473c").mkdir(parents=True)
    rec = tmp_path / "ssh.log"; fake = write_fake_ssh(tmp_path / "bin", rec, stdout="", rc=1)
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}", HOME=str(home), USER="tester", SISTEER_SITE_DIR=str(site),
               F22_SERVING_PATH="lora_native", SISTEER_SECRETS_MODE="env", ANTHROPIC_API_KEY="synthetic", OPENROUTER_API_KEY="synthetic")
    r = subprocess.run(["bash", "scripts/sdfbase/full22/petri_drive_full22.sh", "gemma4e4b", "main", "cluster_a", sha], cwd=wt, capture_output=True, text=True, env=env, timeout=120)
    out = r.stdout + r.stderr
    assert "[cluster] cluster_a from the site config: ssh=cluster-a-login shared=" in out, out[-1500:]
    assert "FATAL: no endpoint.json at cluster-a-login:" in out and str(tmp_path / "shared_cluster_a") in out, out[-1500:]
    calls = [json.loads(l) for l in rec.read_text().splitlines()]
    assert calls and calls[0][-2] == "cluster-a-login", calls
    assert "[secrets] env mode" in out and "synthetic" not in out          # the key is used, never printed
    # an unconfigured, unknown label is refused by name (the built-ins stay for the two known labels)
    r2 = subprocess.run(["bash", "scripts/sdfbase/full22/petri_drive_full22.sh", "gemma4e4b", "main", "cluster_zz", sha], cwd=wt, capture_output=True, text=True, env=env, timeout=60)
    assert r2.returncode == 2 and "bad cluster 'cluster_zz'" in r2.stdout + r2.stderr
    # the bank root follows F22_GRADING_HOST_BANK_ROOT (the grading daemon hands its --petri-bank-root, the front door's
    # SISTEER_OBJECTS_BANK); the legacy $STAGE root above is only the default — with it gone and the override's view present, the
    # script reaches the same endpoint read on the configured host and DRIVE_START names the root it read
    shutil.rmtree(home / "stage")
    bank = tmp_path / "site_bank"; (bank / "gemma4e4b/cells_view/main").mkdir(parents=True)
    r3 = subprocess.run(["bash", "scripts/sdfbase/full22/petri_drive_full22.sh", "gemma4e4b", "main", "cluster_a", sha], cwd=wt, capture_output=True, text=True,
                        env=dict(env, F22_GRADING_HOST_BANK_ROOT=str(bank)), timeout=120)
    out3 = r3.stdout + r3.stderr
    assert "FATAL: no endpoint.json at cluster-a-login:" in out3 and f" bank_root={bank} " in out3 and str(home / "stage") not in out3, out3[-1500:]


def test_stage_strength_and_submit_twin_dry_runs_place_a_configured_label(tmp_path):
    site = write_site(tmp_path)
    rec = tmp_path / "ssh.log"; write_fake_ssh(tmp_path / "bin", rec, stdout="MISSING\n0\n", rc=0)
    bank = tmp_path / "bank"; cell = bank / "m/cells/dose__c"; cell.mkdir(parents=True)
    (cell / "object.pt").write_bytes(b"x" * 100); (cell / "staged_from.json").write_text(json.dumps({"recorded_hash_check": {}}))
    import hashlib
    sha = hashlib.sha256(b"x" * 100).hexdigest()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cells": [{"out": str(cell), "target": "m", "staged_id": "dose__c", "cell_id": "c", "object_sha256": sha, "scale": 0.75,
                                             "parent_staged_id": "p", "parent_object_sha256": "0" * 64, "producer": "t", "object_type": "lora", "scaling_method": "s"}]}))
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}", PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, str(F22 / "strength/stage_strength.py"), "--ledger", str(ledger), "--clusters", "cluster_a,cluster_b", "--site-dir", str(site),
                        "--bank", str(bank), "--dry-run", "--no-manifest"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode in (0, 2), r.stdout + r.stderr            # 2 = "not verified" in a dry run (nothing was copied), the script's own rule
    hosts = [json.loads(l)[-3] for l in rec.read_text().splitlines()]
    assert set(hosts) == {"cluster-a-login", "cluster-b-login"}, hosts
    assert "[cluster_a] m/dose__c: would_rsync" in r.stdout
    r3 = subprocess.run([sys.executable, str(F22 / "strength/stage_strength.py"), "--ledger", str(ledger), "--clusters", "cluster_zz", "--site-dir", str(site), "--dry-run", "--no-manifest"],
                        capture_output=True, text=True, env=env, timeout=60)
    assert r3.returncode != 0 and "cluster_zz" in r3.stdout + r3.stderr
    r2 = subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(site / "config.env"), "--model", "m", "--dry-run", "submit", "--unit", "u", "--lwid", "m/u0/static/x",
                         "--script", "scripts/sdfbase/full22/target_static.sbatch", "--name", "n", "--args", "m u0 pasttense " + "d" * 40 + " 9", "--wave-sha", "d" * 40, "--clusters", "cluster_b"],
                        capture_output=True, text=True, env=dict(os.environ, SISTEER_STATE_DIR=str(tmp_path / "state")), timeout=60)
    assert r2.returncode == 0 and "ssh cluster-b-login" in r2.stdout and "--qos=qos_cluster_b --partition=gpu_cluster_b --account=acct_cluster_b" in r2.stdout, r2.stdout + r2.stderr
    assert f"SISTEER_SHARED_ROOT={tmp_path}/shared_cluster_b" in r2.stdout and "SISTEER_REGISTRY_CLUSTER_LABEL=cluster_b" in r2.stdout
