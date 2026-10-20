"""The Petri scenario bundles' durable root is the cluster's shared root from the site config ($SHARED through sisteer.site),
never a literal storage mount.

  * default_bundle_root() is <shared root>/petri/scenarios: the job env's SISTEER_SHARED_ROOT, else the named (or primary)
    cluster block's CLUSTER_SHARED_ROOT
  * with no shared root anywhere it refuses by name
  * freeze_lock's default durable root is the same shared root: a bundle outside it is refused
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sisteer.capeval.petri import scenarios as psc


@pytest.fixture
def no_site(tmp_path, monkeypatch):
    for k in ("SISTEER_CONFIG", "SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT", "SISTEER_CLUSTER_LABEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))
    return tmp_path


def test_the_job_env_names_the_root(no_site, monkeypatch):
    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/shared/testuser")
    root = psc.default_bundle_root()
    assert root == Path("/shared/testuser/petri/scenarios") and psc.validate_bundle_root(root) == root


def test_a_named_cluster_block_names_the_root(no_site, monkeypatch):
    site = no_site / "site"
    (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="one two"\nSISTEER_PRIMARY_CLUSTER=one\n')
    (site / "clusters" / "one.env").write_text("CLUSTER_SHARED_ROOT=/shared/one\n")
    (site / "clusters" / "two.env").write_text("CLUSTER_SHARED_ROOT=/shared/two\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    assert psc.default_bundle_root() == Path("/shared/one/petri/scenarios")
    assert psc.default_bundle_root("two") == Path("/shared/two/petri/scenarios")


def test_no_shared_root_anywhere_refuses_by_name(no_site):
    with pytest.raises(psc.ScenarioError, match="no durable bundle root"):
        psc.default_bundle_root()


def test_no_literal_mount_in_the_module():
    assert "/workspace" + "-vast" not in Path(psc.__file__).read_text()     # assembled, so this line names no site path itself
