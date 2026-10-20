"""A cluster read shard reads persisted state under the site's alias table, as the grading host does: ``plan_window`` names the grading
host root's ``site_aliases.toml``, ``stage_window`` copies it beside the shard's registry root (``<window root>/root/site_aliases.toml``)
and ``run_shard`` installs it, so an adapter that reads a legacy spelling through ``site_aliases.canonical`` / ``get_compat`` reads it
on the cluster route too. A site with no alias file stages none, and the shard runs under the empty table, never a table an earlier
pass of the same process left installed. Synthetic world only (the cluster read tests' own: a local destination and a published
generation of synthetic stores); the legacy spelling is invented."""
from __future__ import annotations

from pathlib import Path

import pytest

from sisteer.registry import cluster_reads as CR
from sisteer.registry import site_aliases as SA
from sisteer.registry import vocab as V

import test_registry_cluster_read as TCR

#: an invented legacy spelling (never a label of the code's vocabulary) and the label it maps to
LEGACY = "oldsite_a"
TARGET = V.SUBSTRATES[0]


@pytest.fixture(autouse=True)
def _empty_installed_table():
    """Every test starts and ends with the empty table installed (a shard installs its root's table process-wide)."""
    SA.reset()
    yield
    SA.reset()


def test_the_shard_installs_the_grading_hosts_alias_table_staged_beside_its_root(tmp_path, monkeypatch):
    w = TCR._world_with_generation(tmp_path, monkeypatch)
    src = Path(w["root"]) / SA.FILE_NAME
    src.write_text(f'# the site\'s legacy spellings\n[site_aliases]\n{LEGACY} = "{TARGET}"\n')
    plan, _plan_file, _info = TCR._plan_and_stage(w, tmp_path)
    assert plan["site_aliases"] == str(src) and plan["tasks"]
    staged = Path(plan["root"]) / SA.FILE_NAME
    assert staged.read_bytes() == src.read_bytes(), "the stage copies the grading host's file beside the shard's registry root"
    TCR._node(w)
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
    table, digest, source = SA.active()
    assert table == {LEGACY: TARGET} and digest == SA.digest({LEGACY: TARGET}) and Path(source).resolve() == staged.resolve()
    # what the adapters read through it on the cluster route
    assert SA.canonical(LEGACY) == TARGET and SA.get_compat({f"{LEGACY}_path": "/p"}, f"{TARGET}_path") == "/p"


def test_a_site_without_an_alias_file_stages_none_and_the_shard_runs_under_the_empty_table(tmp_path, monkeypatch):
    w = TCR._world_with_generation(tmp_path, monkeypatch)
    plan, _plan_file, _info = TCR._plan_and_stage(w, tmp_path)
    assert plan["site_aliases"] is None and plan["tasks"] and not (Path(plan["root"]) / SA.FILE_NAME).exists()
    TCR._node(w)
    SA.install_table({LEGACY: TARGET}, "an earlier pass")            # a table left installed in this process
    assert CR.run_shard(f"{plan['window_root']}/plan.json", 0, home=w["home"]) == 0
    assert SA.installed() == {} and SA.active()[1] == SA.EMPTY_DIGEST and SA.canonical(LEGACY) == LEGACY
