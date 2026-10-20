"""A cluster-read shard runs on a node with no site config, so the plan carries the grading host's values of the site placeholders the
staged configuration and the adapters' store roots are written with, and the shard sets each one its environment lacks before it loads
the configuration or imports an adapter module (sisteer.registry.cluster_reads.shard_site_env / apply_shard_site_env)."""
from __future__ import annotations

import os

import pytest

from sisteer import site as S
from sisteer.registry import cluster_reads as CR

_VARS = sorted(set(CR.SHARD_SITE_VARS.values()) | {"SISTEER_CONFIG", "SISTEER_SITE_DIR"})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site-here"))
    S._SITE_CACHE.clear()
    yield
    S._SITE_CACHE.clear()


def test_the_plan_carries_the_hosts_values_as_the_site_writes_them(tmp_path, monkeypatch):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "config.env").write_text("SISTEER_GRADE_DIR='~/grade'\nSISTEER_STAGE_DIR=/srv/stage\n")
    monkeypatch.setenv("SISTEER_SITE_DIR", str(site_dir))
    monkeypatch.setenv("SISTEER_WORK_DIR", "~/work")
    got = CR.shard_site_env()
    assert got == {"SISTEER_GRADE_DIR": "~/grade", "SISTEER_STAGE_DIR": "/srv/stage", "SISTEER_WORK_DIR": "~/work"}   # $STATE unset: left out


def test_the_shard_sets_only_what_its_environment_lacks(monkeypatch):
    monkeypatch.setenv("SISTEER_STAGE_DIR", "/node/stage")
    plan = {"site_env": {"SISTEER_GRADE_DIR": "~/grade", "SISTEER_STAGE_DIR": "/host/stage", "PATH": "/evil", "SISTEER_WORK_DIR": ""}}
    applied = CR.apply_shard_site_env(plan)
    assert applied == {"SISTEER_GRADE_DIR": "~/grade"}
    assert os.environ["SISTEER_GRADE_DIR"] == "~/grade" and os.environ["SISTEER_STAGE_DIR"] == "/node/stage"
    assert os.environ.get("PATH") != "/evil" and not os.environ.get("SISTEER_WORK_DIR")
    assert S.expand_path("$GRADE/objects") == S.expand_path("~/grade/objects")


def test_a_plan_without_site_values_changes_nothing():
    assert CR.apply_shard_site_env({}) == {} and CR.apply_shard_site_env({"site_env": None}) == {}
