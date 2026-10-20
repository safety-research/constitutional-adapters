"""`06_eval_waves.sh --clusters` SELECTS the placed units this call submits; it never re-partitions the multi-turn groups.

The twelve multi-turn target units are partitioned round-robin over the site's configured clusters (one attacker pool + six
targets per cluster). The planner used to partition over whatever labels the CALLER handed in, so a per-cluster call
(`--clusters cluster_a`, the canary order: one cluster's group first) re-partitioned all twelve onto that cluster and moved
crescendo/g34's group from the cluster B pool to the cluster A one; the funnel's workload-spec proof refused the submission
(`group: 'mt-s1-cluster_b' -> 'mt-s1-cluster_a'`, 2026-09-30 18:17Z). Since the freeze plans placement-free, the
inventory check cannot see placement — so the front door must compose it right: the PLACEMENT UNIVERSE is every configured
label (`unit_plan(clusters=<all>)`, the same every call) and the caller's subset is a SELECTION (`only_clusters`, the CLI's
`--only-clusters`, 06's `--clusters`). Synthetic site (`cluster_a`/`cluster_b`), no cluster access.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
BOTH = ["cluster_a", "cluster_b"]


def _plan(**kw):
    return ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, **kw)


def test_only_clusters_selects_the_placed_units_and_keeps_the_partition():
    both = _plan(clusters=BOTH)
    a = _plan(clusters=BOTH, only_clusters=["cluster_a"])
    b = _plan(clusters=BOTH, only_clusters=["cluster_b"])
    # a selection is exactly the universe's rows on that label — the same dicts (group, cluster, args, env), nothing re-placed
    assert a == [r for r in both if r["cluster"] == "cluster_a"] and b == [r for r in both if r["cluster"] == "cluster_b"]
    assert len(a) == 2 + 6 and len(b) == 2 + 6          # a pool of two members per three... per six targets (commit 31) + six targets
    assert {r["logical_work_id"] for r in a}.isdisjoint({r["logical_work_id"] for r in b})
    assert sorted(r["logical_work_id"] for r in a + b) == sorted(r["logical_work_id"] for r in both)
    # crescendo/g34 is placed on cluster_b, group mt-battery-cluster_b, whichever label the caller selects: absent from a, present in b
    assert {r["cluster"] for r in both if r["unit"] == "crescendo/g34"} == {"cluster_b"}
    assert not [r for r in a if r["unit"] == "crescendo/g34"]
    (g34,) = [r for r in b if r["unit"] == "crescendo/g34"]
    assert g34["group"] == "mt-battery-cluster_b"
    # the defect, pinned as the OTHER thing: the subset handed in as the universe re-partitions and moves g34's group
    old = _plan(clusters=["cluster_a"])
    (moved,) = [r for r in old if r["unit"] == "crescendo/g34"]
    assert moved["group"] == "mt-battery-cluster_a" and old != a


def test_a_selection_outside_the_universe_is_refused_by_name():
    with pytest.raises(ValueError, match=r"'cluster_z'.*not in the placement universe"):
        _plan(clusters=BOTH, only_clusters=["cluster_z"])


def test_single_gpu_units_are_placement_free_and_survive_every_selection():
    universe = ST.unit_plan(KEY, "battery", families=["static", "mmlu"], sha=SHA, clusters=BOTH)
    assert universe and all(r["cluster"] == "*" for r in universe)
    assert ST.unit_plan(KEY, "battery", families=["static", "mmlu"], sha=SHA, clusters=BOTH, only_clusters=["cluster_a"]) == universe


def _cli(*extra):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([VENV_PY, str(REPRO / "lib" / "stage.py"), "plan", "--model", KEY, "--view", "battery", "--families", "multiturn",
                           "--sha", SHA, "--clusters", ",".join(BOTH), *extra], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=300)


def test_the_plan_cli_only_clusters_is_the_universe_filtered_by_the_cluster_column():
    full = _cli(); sel = _cli("--only-clusters", "cluster_a")
    assert full.returncode == 0 and sel.returncode == 0, full.stderr[-800:] + sel.stderr[-800:]
    col = ST.PLAN_COLUMNS.index("cluster")
    assert sel.stdout.splitlines() == [ln for ln in full.stdout.splitlines() if ln.split("\t")[col] == "cluster_a"]
    bad = _cli("--only-clusters", "cluster_z")
    assert bad.returncode == 2 and "not in the placement universe" in bad.stderr, bad.stderr[-600:]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("fd")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _dry(site, *extra):
    r = run_stage("06_eval_waves.sh", "--dry-run", "--view", "battery", "--family", "multiturn", "--model", KEY, *extra, site=site, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    subs = [ln for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    return r.stdout, sorted((ln.split()[1], ln.split()[3]) for ln in subs)       # (cluster, unit) per `submitted <cluster> <job> <unit>`


def test_06_clusters_submits_the_named_clusters_group_from_the_site_wide_partition(site):
    out_all, both = _dry(site)
    out_a, only_a = _dry(site, "--clusters", "cluster_a")
    out_b, only_b = _dry(site, "--clusters", "cluster_b")
    assert len(both) == 16 and len(only_a) == 8 and len(only_b) == 8
    # each per-cluster call submits EXACTLY that cluster's half of the two-cluster wave: same units, same cluster
    assert only_a == [cu for cu in both if cu[0] == "cluster_a"] and only_b == [cu for cu in both if cu[0] == "cluster_b"]
    assert ("cluster_b", "multiturn.crescendo.g34") in only_b and not any(u == "multiturn.crescendo.g34" for _, u in only_a)
    # the per-cluster call stood up ONE pool (its own) and named the placement universe it selected from
    assert "mt-battery-cluster_a" in out_a and "mt-battery-cluster_b" not in out_a
    assert "mt-battery-cluster_b" in out_b and "mt-battery-cluster_a" not in out_b
    assert "clusters=cluster_a (placement over cluster_a,cluster_b)" in out_a


def test_06_units_submits_only_the_named_units_and_writes_no_wave_file(site):
    """`--units gsm8k.none`: the canary order's "one cell first" through the front door — the plan, the inventory check and the
    placement are unchanged; only the named units go out; the wave file is a deliberate non-write (re-run without --units completes it)."""
    r = run_stage("06_eval_waves.sh", "--dry-run", "--view", "battery", "--family", "gsm8k", "--model", KEY, "--units", "gsm8k.none", site=site, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    subs = [ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    assert subs == ["gsm8k.none"], subs
    assert "units=gsm8k.none selected: the wave file is NOT written" in r.stdout
    full = run_stage("06_eval_waves.sh", "--dry-run", "--view", "battery", "--family", "gsm8k", "--model", KEY, site=site, timeout=600)
    import models as MT
    s1_cells = MT.model(KEY, str(REPRO / "models.yaml"))["views"]["battery"]["cells"]                    # the single-phase wave: none, the prompted legs, every arm (10 rows)
    assert len(s1_cells) == 10 and sorted(ln.split()[3] for ln in full.stdout.splitlines() if ln.startswith("submitted ")) == sorted(f"gsm8k.{c}" for c in s1_cells)
    assert "selected: the wave file is NOT written" not in full.stdout
