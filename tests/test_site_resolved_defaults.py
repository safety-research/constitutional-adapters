"""Defaults that name a site location resolve through the site config, and refuse by name where it says nothing.

The scripts and modules below used to carry a literal storage root, login host or home-relative staging and state path as a
default. Each now reads the site's own setting through the existing mechanisms: ``sisteer.site`` placeholders (``$STAGE``,
``$STATE``, ``$WORK``, ``$SHARED``), ``scripts/site_paths.py`` for scripts that must not import the ``sisteer`` package, the
cluster's shared root from ``SISTEER_SHARED_ROOT`` / ``CLUSTER_SHARED_ROOT`` (or the root a positional argument sits under), and
the login host from the site's cluster block. Every value here is synthetic: a temp site directory, ``login-a``, ``/shared``.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # a module defining dataclasses must be importable by name while it executes
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def bare_site(tmp_path, monkeypatch):
    """A host with no site config and none of the placeholder variables: every lookup must refuse or come back empty."""
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    for var in ("SISTEER_STATE_ROOT", "SISTEER_STAGE_DIR", "SISTEER_WORK_DIR", "SISTEER_GRADE_DIR", "SISTEER_SHARED_ROOT",
                "CLUSTER_SHARED_ROOT", "SISTEER_CLUSTER_LABEL", "HF_HOME", "SISTEER_HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture()
def one_cluster_site(tmp_path, monkeypatch):
    """A site config naming one cluster, ``alpha``, whose block gives a login host and a shared root."""
    site = tmp_path / "site"
    (site / "clusters").mkdir(parents=True)
    shared = tmp_path / "shared" / "u"
    shared.mkdir(parents=True)
    (site / "config.env").write_text(f"SISTEER_CLUSTERS=alpha\nSISTEER_PRIMARY_CLUSTER=alpha\nSISTEER_STAGE_DIR='{tmp_path / 'stage'}'\n")
    (site / "clusters" / "alpha.env").write_text(f"CLUSTER_SSH=login-a\nCLUSTER_SHARED_ROOT={shared}\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    for var in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT", "SISTEER_CLUSTER_LABEL", "SISTEER_STAGE_DIR"):
        monkeypatch.delenv(var, raising=False)
    return {"site": site, "shared": shared, "stage": tmp_path / "stage"}


# --- scripts/site_paths.py: the site module by file path ------------------------------------------------------------------

def test_site_paths_expands_a_placeholder_from_the_environment_and_refuses_an_unset_one_by_name(bare_site, monkeypatch):
    sp = _load("_site_paths_under_test", "scripts/site_paths.py")
    monkeypatch.setenv("SISTEER_STAGE_DIR", str(bare_site / "stage"))
    assert sp.site_path("$STAGE/sdfbase/pools", "--pools") == bare_site / "stage" / "sdfbase" / "pools"
    assert sp.site_path("/plain/path", "--x") == Path("/plain/path")              # no placeholder: the path as given
    monkeypatch.delenv("SISTEER_STAGE_DIR")
    with pytest.raises(sp.SitePathError) as ei:
        sp.site_path("$STAGE/x", "--pools")
    assert "--pools" in str(ei.value) and "SISTEER_STAGE_DIR" in str(ei.value)
    assert isinstance(ei.value, SystemExit)                                        # uncaught, it ends a script with the message


def test_site_paths_main_checkout_is_the_parent_of_gits_common_dir():
    sp = _load("_site_paths_checkout", "scripts/site_paths.py")
    common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert sp.main_checkout() == Path(common).parent


# --- Python defaults ------------------------------------------------------------------------------------------------------

def test_the_login_host_default_is_the_site_cluster_blocks(one_cluster_site, monkeypatch):
    wave = _load("_capeval_exec_wave_under_test", "scripts/capeval_exec_wave.py")
    assert wave.site_login_host() == "login-a"


def test_the_login_host_default_refuses_by_name_without_a_site(bare_site):
    wave = _load("_capeval_exec_wave_bare", "scripts/capeval_exec_wave.py")
    with pytest.raises(SystemExit) as ei:
        wave.site_login_host()
    assert "CLUSTER_SSH" in str(ei.value) and "--cluster-host" in str(ei.value)


def test_native_standup_shared_root_reads_the_site_and_counts_only_a_directory_here(bare_site, monkeypatch, tmp_path):
    from sisteer.capeval.serving import native_standup as ns
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path))
    assert ns.shared_root() == tmp_path
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "absent"))
    assert ns.shared_root() is None                                               # a root that is not a directory here: off-cluster
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    assert ns.shared_root() is None                                               # no site, no variable


def test_make_bank_view_default_shared_comes_from_the_site_and_refuses_by_name(one_cluster_site, monkeypatch):
    sys.path.insert(0, str(SCRIPTS / "sdfbase" / "full22"))
    mbv = _load("_make_bank_view_under_test", "scripts/sdfbase/full22/make_bank_view.py")
    assert mbv.default_shared() == one_cluster_site["shared"]                      # the cluster block's CLUSTER_SHARED_ROOT
    monkeypatch.setenv("SISTEER_CONFIG", str(one_cluster_site["site"] / "absent.env"))
    with pytest.raises(SystemExit) as ei:
        mbv.default_shared()
    assert "--bank" in str(ei.value)


def test_staging_defaults_expand_under_the_site_stage_root(one_cluster_site):
    stage = one_cluster_site["stage"]
    sys.path.insert(0, str(SCRIPTS / "sdfbase"))
    bp = _load("_build_pools_under_test", "scripts/sdfbase/build_pools.py")
    assert bp.DEFAULT_OUT == Path("$STAGE/sdfbase/pools") and bp.site_path(bp.DEFAULT_OUT, "--out") == stage / "sdfbase" / "pools"
    drill = _load("_continuation_drill_under_test", "scripts/sdfbase/continuation_drill.py")
    ns = drill.build_parser().parse_args(["--config-sha", "x"])
    drill.expand_site_paths(ns)
    assert ns.out == str(stage / "sdfbase" / "drill_verdict.json") and ns.compare_root == str(stage / "sdfbase" / "drill")
    dv = _load("_derived_verdict_under_test", "scripts/sdfbase/derived_verdict.py")
    with pytest.raises(dv.InputError) as ei:                                       # the default pools dir is the site's, read there
        dv.load_pool_manifest(dv.DEFAULT_POOLS_DIR)
    assert str(stage / "sdfbase" / "pools") in str(ei.value)


def test_parity_legacy_roots_follow_the_site_and_refuse_a_root_they_cannot_place(bare_site, monkeypatch, tmp_path):
    from sisteer.registry import parity as PA
    monkeypatch.setenv("SISTEER_STAGE_DIR", str(tmp_path / "stage"))
    monkeypatch.setenv("SISTEER_GRADE_DIR", str(tmp_path / "grade"))
    state_root = next(r for r in PA.LEGACY_ROOTS if r.startswith("$STATE/"))   # the run-state directory, as parity.py declares it
    with pytest.raises(PA.ParityRefused) as ei:                                    # $STATE unset: its root cannot be guarded
        PA.legacy_roots(home=str(tmp_path / "home"))
    assert state_root in str(ei.value) and "SISTEER_STATE_ROOT" in str(ei.value)
    monkeypatch.setenv("SISTEER_STATE_ROOT", str(tmp_path / "state"))
    with pytest.raises(PA.ParityRefused) as ei:                                    # $WORK unset: the same
        PA.legacy_roots(home=str(tmp_path / "home"))
    assert "$WORK/bcb-bfcl-200" in str(ei.value) and "SISTEER_WORK_DIR" in str(ei.value)
    monkeypatch.setenv("SISTEER_WORK_DIR", str(tmp_path / "work"))
    roots = PA.legacy_roots(home=str(tmp_path / "home"))
    assert {str(tmp_path / "stage"), str(tmp_path / "grade"), str(tmp_path / "state" / state_root.split("/", 1)[1]),
            str(tmp_path / "work" / "bcb-bfcl-200")} <= set(roots)                    # every site root, at the site's value
    assert str(tmp_path / "home" / "figs" / "degeneracy") in roots                  # a home-relative root stays under the home
    assert not any("$" in r for r in roots)


def test_the_parity_gate_refuses_before_either_leg_when_a_guard_root_is_unset(bare_site, monkeypatch, tmp_path):
    from sisteer.registry import parity as PA

    def _never(*a, **k):
        raise AssertionError("the gate went on with a guard root it could not place")
    for name in ("checkout_sha", "run_legacy_side", "run_registry_side"):
        monkeypatch.setattr(PA, name, _never)
    with pytest.raises(PA.ParityRefused) as ei:
        PA.check_paper_parity(str(tmp_path / "checkout"), "0" * 40, str(tmp_path / "registry"), write_records=False)
    assert "legacy-read guard cannot place" in str(ei.value)


def test_the_registry_leg_itself_refuses_when_a_guard_root_is_unset(bare_site, tmp_path):
    """The leg's own process installs the guard first, so a host whose site leaves a root unset ends the leg before it opens
    the registry (a fresh subprocess with this test's environment: no site config, no placeholder variables)."""
    from sisteer.registry import parity as PA
    with pytest.raises(PA.ParityRefused) as ei:
        PA.run_registry_side(str(tmp_path / "registry"), str(tmp_path / "work"), ["value"], None, timeout_s=600)
    assert "registry leg failed" in str(ei.value) and "legacy-read guard cannot place" in str(ei.value)
    assert not (tmp_path / "work" / "registry_side.json").exists()


def test_a_job_shared_root_resolves_from_the_environment_or_refuses_by_name(bare_site, monkeypatch, tmp_path):
    pytest.importorskip("torch")                                                    # dose_capture.py imports it at module level
    sys.path.insert(0, str(SCRIPTS / "sdfbase" / "full22" / "strength"))
    dc = _load("_dose_capture_under_test", "scripts/sdfbase/full22/strength/dose_capture.py")
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path))
    assert dc.detect_shared() == tmp_path
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    with pytest.raises(SystemExit) as ei:
        dc.detect_shared()
    assert "--shared" in str(ei.value)


# --- shell defaults -------------------------------------------------------------------------------------------------------

def _shared_line(rel: str) -> str:
    return next(ln for ln in (REPO / rel).read_text().splitlines() if ln.startswith("SHARED=${SISTEER_SHARED_ROOT"))


@pytest.mark.parametrize("rel, var, value, want", [
    ("scripts/registry/landing/read_array.sbatch", "VENV", "/shared/u/envs/repo-gate-0123456789ab", "/shared/u"),
    ("scripts/registry/landing/attest.sbatch", "TREE", "/shared/u/gates/trees/" + "a" * 40, "/shared/u"),
])
def test_the_landing_jobs_take_the_shared_root_their_argument_sits_under(rel, var, value, want):
    line = _shared_line(rel)
    env = {k: v for k, v in os.environ.items() if k not in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT")}
    got = subprocess.run(["bash", "-c", f'{var}="$1"; {line}; echo "$SHARED"', "x", value], capture_output=True, text=True, env=env)
    assert got.stdout.strip() == want, got.stderr
    env["CLUSTER_SHARED_ROOT"] = "/site/root"                                       # the environment's value wins over the argument's
    got = subprocess.run(["bash", "-c", f'{var}="$1"; {line}; echo "$SHARED"', "x", value], capture_output=True, text=True, env=env)
    assert got.stdout.strip() == "/site/root"


def test_mirror_figs_takes_the_root_it_is_installed_under(tmp_path):
    shared = tmp_path / "shared"
    (shared / "scripts").mkdir(parents=True)
    helper = shared / "scripts" / "mirror-figs.sh"
    shutil.copy(REPO / "reproduce" / "lib" / "mirror_figs.sh", helper)
    figs = tmp_path / "figs"
    figs.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "u"}
    r = subprocess.run(["bash", str(helper), str(figs)], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and f"no push key at {shared}/.ssh/figs-push" in r.stderr, r.stderr   # the installed root, nothing probed


def test_push_records_reads_the_site_values_installed_beside_it(tmp_path):
    inst = tmp_path / "shared" / "scripts" / "registry"
    inst.mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "registry" / "push_records.sh", inst / "push_records.sh")
    site_root = tmp_path / "siteroot"
    (inst / "registry_site.env").write_text(f"REGISTRY_SITE_SHARED_ROOT={site_root}\nREGISTRY_SITE_SUBSTRATE=cluster_x\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "u"}
    r = subprocess.run(["bash", str(inst / "push_records.sh"), str(run_dir)], capture_output=True, text=True, env=env)
    assert r.returncode == 0
    log = (site_root / "registry" / "push.log").read_text()
    assert " cluster_x " in log and "no push key at" in log, log                  # the label and the root came from the file


def test_sbatch_registered_refuses_a_spool_nothing_names(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "USER": "u"}
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    r = subprocess.run(["bash", str(REPO / "scripts" / "registry" / "sbatch_registered.sh"), "--planned", str(plan), "--", "true"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "no spool" in r.stderr and "SISTEER_REGISTRY_SPOOL" in r.stderr, r.stderr


@pytest.mark.parametrize("rel", ["scripts/sdfbase/full22/judge/snapshot_rows_sha.py", "scripts/sdfbase/full22/judge/verify_remote_stores.py"])
def test_the_remote_store_readers_refuse_a_root_nothing_names(rel, tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "u"}
    r = subprocess.run([sys.executable, "-"], input=(REPO / rel).read_text(), capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "F22_CONTROLS_REMOTE" in r.stderr and "SISTEER_SHARED_ROOT" in r.stderr, r.stderr
