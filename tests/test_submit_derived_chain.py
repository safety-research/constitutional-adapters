"""submit_derived_chain.sh: prep -> calibrate (array) ->
assess, then [calibrate <m> -> assess2], then finish — every link
--dependency=afterok on its predecessor, --comment=sha:<F> on every job,
reseed-derived-<key>-<stage> names, the site's --qos, --partition and
--output on every cluster, one sbatch per statement, every id recorded to
RS/reseed/submissions.jsonl — against a fake `sbatch` on PATH."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sdfbase" / "seedrep" / "submit_derived_chain.sh"
KEY, BANK = "qwen35_4b_base_eb64_s2", "reseed"
SHA = "0123456789abcdef" * 2 + "01234567"
assert len(SHA) == 40


@pytest.fixture
def world(tmp_path):
    binp = tmp_path / "bin"
    binp.mkdir()
    (binp / "sbatch").write_text(
        "#!/bin/bash\n"
        f"c={tmp_path}/counter; n=$(( $(cat \"$c\" 2>/dev/null || echo 1000) + 1 )); echo $n > \"$c\"\n"
        f"printf '%s\\t%s\\t%s\\n' \"$n\" \"${{SDFBASE_FIT_ENV_NAME:-}}\" \"$*\" >> {tmp_path}/sbatch.log\n"
        "echo \"$n;cluster\"\n")   # --parsable on an array prints id;cluster
    (binp / "sbatch").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "RS"}
    env["PATH"] = f"{binp}:{env['PATH']}"
    # each cluster's login host, eval-repo root, QoS and log dir come from its block in the site config (site labels of their own)
    site = tmp_path / "site"
    (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="site_one site_two"\nSISTEER_PRIMARY_CLUSTER=site_one\n')
    for k, reg, lab in (("a", "cluster_a", "site_one"), ("b", "cluster_b", "site_two")):
        (site / "clusters" / f"{lab}.env").write_text(
            f"CLUSTER_SSH=login-{k}\nCLUSTER_SHARED_ROOT=/shared-{k}\nCLUSTER_PARTITION=p-{k}\nCLUSTER_QOS=qos-{k}\n"
            f"CLUSTER_SUBSTRATE={reg}\nCLUSTER_LABEL={lab}\nCLUSTER_EVAL_REPO_ROOT='$SHARED/evalrepo'\nCLUSTER_LOG_DIR='$SHARED/logs'\n")
    env["SISTEER_CONFIG"] = str(site / "config.env")
    return tmp_path, env


def _run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), KEY, BANK, "--sha", SHA, "--local",
                           "--rs", str(env["_RS"]), *args],
                          capture_output=True, text=True, env={k: v for k, v in env.items()
                                                               if k != "_RS"}, timeout=60)


def _records(tmp_path):
    p = tmp_path / "rs" / "reseed" / "submissions.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _log(tmp_path):
    return [l.split("\t", 2) for l in (tmp_path / "sbatch.log").read_text().splitlines()]


def test_start_phase_submits_prep_calib_assess_with_afterok(world):
    tmp_path, env = world
    env["_RS"] = tmp_path / "rs"
    r = _run(env, "--repo", "/ckpt/F")
    assert r.returncode == 0, r.stdout + r.stderr
    recs = _records(tmp_path)
    assert [x["stage"] for x in recs] == ["prep", "calibrate", "assess"]
    assert [x["name"] for x in recs] == [f"reseed-derived-{KEY}-prep", f"reseed-derived-{KEY}-calib",
                                         f"reseed-derived-{KEY}-assess"]
    p, c, a = (x["job"] for x in recs)
    assert (p, c, a) == (1001, 1002, 1003)
    assert recs[0]["depends_on"] is None and recs[1]["depends_on"] == p and recs[2]["depends_on"] == c
    assert r.stdout.strip() == str(a)                      # the last id alone on stdout
    log = _log(tmp_path)
    assert len(log) == 3
    for _, _, argv in log:
        assert f"--comment=sha:{SHA}" in argv and "--qos=qos-a" in argv and "--parsable" in argv
        assert "--partition=p-a" in argv and "--output=/shared-a/logs/%x_%j.out" in argv     # the site's, not a header's
        assert "/ckpt/F/scripts/sbatch/sdfbase_derived_build.sbatch" in argv
        assert f" {KEY} constv2-mix-full constv2ctl2-full {BANK}" in argv
        assert "--export" not in argv
    assert "--array=0-2" in log[1][2] and log[1][2].endswith(f"{BANK} -")   # the array's lr arg is `-`
    assert f"--dependency=afterok:{p}" in log[1][2] and f"--dependency=afterok:{c}" in log[2][2]
    assert "--dependency" not in log[0][2]
    assert log[0][2].split("sdfbase_derived_build.sbatch ")[1].startswith("prep ")
    assert log[2][2].split("sdfbase_derived_build.sbatch ")[1].startswith("assess ")
    assert all(x["sha"] == SHA and x["cluster"] == "cluster_a" and x["qos"] == "qos-a" and x["host"] == "login-a"
               and x["key"] == KEY and x["bank"] == BANK and x["kind"] == "derived" for x in recs)
    assert recs[1]["lr_mult"] is None


def test_extend_and_finish_phases(world):
    tmp_path, env = world
    env["_RS"] = tmp_path / "rs"
    r = _run(env, "--phase", "extend", "--after", "77", "--extension", "0.1")
    assert r.returncode == 0, r.stdout + r.stderr
    recs = _records(tmp_path)
    assert [x["name"] for x in recs] == [f"reseed-derived-{KEY}-calib-x0.1", f"reseed-derived-{KEY}-assess2"]
    assert recs[0]["depends_on"] == 77 and recs[0]["lr_mult"] == "0.1" and recs[1]["depends_on"] == recs[0]["job"]
    log = _log(tmp_path)
    assert log[0][2].endswith(f"{BANK} 0.1") and "--array" not in log[0][2]
    assert r.stdout.strip() == str(recs[1]["job"])
    r = _run(env, "--phase", "finish", "--after", r.stdout.strip())
    assert r.returncode == 0, r.stdout + r.stderr
    recs = _records(tmp_path)
    assert recs[-1]["stage"] == "finish" and recs[-1]["depends_on"] == recs[1]["job"]
    assert recs[-1]["name"] == f"reseed-derived-{KEY}-finish"
    # argument guards
    assert _run(env, "--phase", "extend", "--after", "77").returncode == 2
    assert _run(env, "--phase", "finish").returncode == 2
    assert _run(env, "--phase", "extend", "--after", "77", "--extension", "x").returncode == 2


def test_cluster_b_takes_its_qos_and_log_root_from_the_site_and_env_name_rides_as_prefix(world):
    tmp_path, env = world
    env["_RS"] = tmp_path / "rs"
    r = _run(env, "--cluster", "cluster_b", "--env-name", "sdfbase-fit-tf515")
    assert r.returncode == 0, r.stdout + r.stderr
    for _, envname, argv in _log(tmp_path):
        assert "--qos=qos-b" in argv and "--partition=p-b" in argv and "--output=/shared-b/logs/%x_%j.out" in argv
        assert envname == "sdfbase-fit-tf515"           # seen by sbatch through the environment
        assert "--export" not in argv
        assert "/shared-b/evalrepo/" + SHA in argv
        # 2026-09-17: the remote path used to read `/workspace/\$USER/...`; printf %q shipped it over ssh as a
        # literal `$USER` and sbatch could not open the script. The path must carry no unexpanded variable.
        assert "$USER" not in argv and "\\$" not in argv
    recs = _records(tmp_path)
    assert all(x["cluster"] == "cluster_b" and x["qos"] == "qos-b" and x["host"] == "login-b"
               for x in recs)


def test_dry_run_records_nothing_and_bad_sha_refuses(world):
    tmp_path, env = world
    env["_RS"] = tmp_path / "rs"
    r = _run(env, "--dry-run")
    assert r.returncode == 0 and r.stdout.strip() == "DRY-assess"
    assert _records(tmp_path) == [] and not (tmp_path / "sbatch.log").exists()
    assert r.stderr.count("DRY:") == 3
    r = subprocess.run(["bash", str(SCRIPT), KEY, BANK, "--sha", "abc", "--local"],
                       capture_output=True, text=True, env={k: v for k, v in env.items() if k != "_RS"})
    assert r.returncode == 2 and "40-hex" in r.stderr
    r = subprocess.run(["bash", str(SCRIPT), KEY], capture_output=True, text=True,
                       env={k: v for k, v in env.items() if k != "_RS"})
    assert r.returncode == 2


def test_a_failed_sbatch_stops_the_chain(world):
    tmp_path, env = world
    env["_RS"] = tmp_path / "rs"
    (tmp_path / "bin" / "sbatch").write_text("#!/bin/bash\necho 'sbatch: error: invalid qos' >&2; exit 1\n")
    r = _run(env)
    assert r.returncode == 1 and "sbatch FAILED" in r.stderr
    assert _records(tmp_path) == []


def test_script_hygiene():
    txt = SCRIPT.read_text()
    assert "--export" not in txt.replace("never --export", "")
    assert "--parsable" in txt and "--dependency=\"afterok:$dep\"" in txt
    assert txt.count("submit prep") == 1 and txt.count("submit calibrate") == 2 \
        and txt.count("submit assess") == 2 and txt.count("submit finish") == 1
    assert "for " not in "".join(l for l in txt.splitlines() if "sbatch" in l and "submit" in l)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_remote_path_is_resolved_before_rendering_from_the_sites_block(world):
    """The rendered sbatch line travels over ssh verbatim (printf %q), so its paths must already be literal: the cluster's
    eval-repo root from its site config block, its `$SHARED` substituted here, never a variable left for the remote shell."""
    tmp_path, env = world
    env["_RS"] = tmp_path / "rs"
    r = _run(env, "--dry-run", "--cluster", "cluster_a")
    assert r.returncode == 0, r.stdout + r.stderr
    rendered = [ln for ln in (r.stdout + r.stderr).splitlines() if "DRY:" in ln and " sbatch " in ln]
    assert rendered, r.stdout + r.stderr
    for ln in rendered:
        assert f"/shared-a/evalrepo/{SHA}/" in ln
        assert "SISTEER_SHARED_ROOT=/shared-a " in ln     # the job env names the block's shared root, literally
        assert "$USER" not in ln and "$SHARED" not in ln and "\\$" not in ln
    env["SISTEER_CONFIG"] = str(tmp_path / "no-site" / "config.env")      # no site config: refused by name, nothing rendered
    r = _run(env, "--dry-run", "--cluster", "cluster_a")
    assert r.returncode == 2 and "cluster_a" in r.stderr and "DRY:" not in r.stderr
