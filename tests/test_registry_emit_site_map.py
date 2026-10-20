"""The emitter learns its substrate and its shared roots from the site, never from cluster names in code.

Order (``emit.substrate_from_env``): ``SISTEER_REGISTRY_SUBSTRATE``; inside a Slurm job at a site that declared the map
``SISTEER_SLURM_SUBSTRATES``, the entry named by ``SLURM_CLUSTER_NAME``, else the generic ``slurm`` before any shared root is
probed; a mapped shared root that exists; inside a Slurm job at a site that declared a cluster label or the older helper's
variable, the generic ``slurm``; RunPod; the grading host. The generic ``slurm`` is refused by name without its label. A Slurm job
with no site information at all falls through to the last two, so the suite's own jobs keep the substrate they always had.
"""
from __future__ import annotations

import os

import pytest

from sisteer.registry import emit

SITE_VARS = (emit.SITE_MAP_ENV, emit.CLUSTER_LABEL_ENV, emit.PREVIOUS_HELPER_ENV)


@pytest.fixture
def host(unprovisioned_host, monkeypatch):
    for var in SITE_VARS:
        monkeypatch.delenv(var, raising=False)
    emit._NOTICED.clear()
    return unprovisioned_host


def test_no_cluster_name_or_shared_root_is_spelled_in_the_emitter(host):
    import ast
    assert emit.SHARED_BASES == () and emit.shared_bases() == ()
    with open(emit.__file__, encoding="utf-8") as fh:
        src = fh.read()
    assert "/workspace" not in src
    literals = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not literals & {"cluster_a", "cluster_b"}, "no cluster label is a literal of the emitter: the site map names them"


def test_the_site_map_parses_entries_and_skips_a_malformed_one_with_one_notice(host, capsys):
    raw = "Cluster-One=cluster_a  cluster-two=cluster_b@/srv/shared/{user},bad-entry  three=@/x  four=cluster_a@relative"
    assert emit.site_map(raw) == [("cluster-one", "cluster_a", None), ("cluster-two", "cluster_b", "/srv/shared/{user}")]
    err = capsys.readouterr().err
    for entry in ("bad-entry", "three=@/x", "four=cluster_a@relative"):
        assert f"entry '{entry}' is not" in err
    emit.site_map(raw)
    assert capsys.readouterr().err == "", "each notice once per process"
    assert emit.site_map("") == [] and emit.site_map("   ") == []


def test_the_declared_substrate_wins_over_the_map_and_every_probe(host, monkeypatch):
    shared = host.root / "shared-a"
    (shared / host.user / "registry").mkdir(parents=True)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-one")
    monkeypatch.setenv(emit.SITE_MAP_ENV, f"cluster-one=cluster_b cluster-x=cluster_a@{shared}/{{user}}")
    monkeypatch.setenv(emit.SUBSTRATE_ENV, "runpod")
    assert emit.substrate_from_env() == "runpod"


def test_the_map_names_the_label_of_the_job_s_cluster_whole_and_case_insensitively(host, monkeypatch):
    monkeypatch.setenv(emit.SITE_MAP_ENV, "cluster-one=cluster_a cluster-two=cluster_b")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "CLUSTER-TWO")
    assert emit.substrate_from_env() == "cluster_b"
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-one")
    assert emit.substrate_from_env() == "cluster_a"
    # a name that only CONTAINS a mapped name is not it: a site map entry names one cluster exactly
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-one-eu")
    assert emit.substrate_from_env() == emit.GENERIC_SLURM_SUBSTRATE


def test_a_mapped_shared_root_is_probed_for_the_substrate_and_the_spool(host, monkeypatch):
    base_b = host.root / "shared-b"
    monkeypatch.setenv(emit.SITE_MAP_ENV, f"cluster-one=cluster_a@{host.root}/shared-a/{{user}} cluster-two=cluster_b@{base_b}/{{user}}")
    assert emit.shared_bases() == (("cluster_a", f"{host.root}/shared-a/{{user}}"), ("cluster_b", f"{base_b}/{{user}}"))
    assert emit.substrate_from_env() == "grading_host" and emit.spool_root() is None
    (base_b / host.user).mkdir(parents=True)                           # a user dir alone names the substrate, not a spool
    assert emit.substrate_from_env() == "cluster_b" and emit.spool_root() is None
    (base_b / host.user / "registry").mkdir()
    assert emit.spool_root() == str(base_b / host.user / "registry" / "records")
    monkeypatch.setenv(emit.SPOOL_ENV, str(host.root / "explicit"))
    assert emit.spool_root() == str(host.root / "explicit"), "the explicit spool still wins"


def test_a_cluster_the_map_does_not_name_is_the_generic_slurm_and_is_refused_without_its_label(host, monkeypatch):
    monkeypatch.setenv(emit.SITE_MAP_ENV, "cluster-one=cluster_a")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-nine")
    assert emit.substrate_from_env() == emit.GENERIC_SLURM_SUBSTRATE
    with pytest.raises(emit.SubstrateRefused) as exc:
        emit.check_substrate(emit.substrate_from_env())
    assert emit.CLUSTER_LABEL_ENV in str(exc.value) and emit.SITE_MAP_ENV in str(exc.value)
    monkeypatch.setenv(emit.CLUSTER_LABEL_ENV, "site-nine")
    assert emit.check_substrate(emit.substrate_from_env()) == "site-nine"


def test_a_cluster_the_map_does_not_name_is_refused_where_another_entry_s_shared_root_exists(host, monkeypatch, tmp_path):
    """A Slurm job whose cluster the map does not name never takes the label of another entry's shared root that is mounted on its
    machine (a cluster renamed while its storage stays mounted): it is the generic ``slurm``, and a writer refuses it by name until
    a label is declared. Off Slurm, the mapped root still names its substrate."""
    base_a = host.root / "shared-a"
    (base_a / host.user / "registry").mkdir(parents=True)
    monkeypatch.setenv(emit.SITE_MAP_ENV, f"cluster-one=cluster_a@{base_a}/{{user}}")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-one")
    assert emit.substrate_from_env() == "cluster_a"
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-renamed")
    assert emit.substrate_from_env() == emit.GENERIC_SLURM_SUBSTRATE
    producer = emit.producer_block("tests.site_map", code_sha="0" * 40)
    with pytest.raises(emit.SubstrateRefused) as exc:
        emit.RecordWriter(str(tmp_path / "spool"), "campaign", "run-1", producer, disabled=False)
    assert emit.CLUSTER_LABEL_ENV in str(exc.value) and emit.SITE_MAP_ENV in str(exc.value)
    monkeypatch.setenv(emit.CLUSTER_LABEL_ENV, "site-renamed")
    writer = emit.RecordWriter(str(tmp_path / "spool"), "campaign", "run-1", emit.producer_block("tests.site_map", code_sha="0" * 40),
                               disabled=False)
    assert writer.producer["substrate"] == emit.GENERIC_SLURM_SUBSTRATE and writer.payload_extra[emit.CLUSTER_LABEL_KEY] == "site-renamed"
    monkeypatch.delenv(emit.CLUSTER_LABEL_ENV)
    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    assert emit.substrate_from_env() == "cluster_a"


def test_a_cluster_label_alone_makes_a_slurm_job_the_generic_slurm(host, monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-nine")
    monkeypatch.setenv(emit.CLUSTER_LABEL_ENV, "site-nine")
    assert emit.substrate_from_env() == emit.GENERIC_SLURM_SUBSTRATE
    assert emit.check_substrate(emit.GENERIC_SLURM_SUBSTRATE) == "site-nine"


def test_a_job_that_sourced_the_older_helper_is_refused_until_the_helpers_are_installed_again(host, monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-nine")
    monkeypatch.setenv(emit.PREVIOUS_HELPER_ENV, "an-older-name")
    assert emit.substrate_from_env() == emit.GENERIC_SLURM_SUBSTRATE
    with pytest.raises(emit.SubstrateRefused) as exc:
        emit.check_substrate(emit.substrate_from_env())
    assert "--what helpers" in str(exc.value)


def test_a_slurm_job_with_no_site_information_keeps_the_fallbacks(host, monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-nine")
    assert emit.substrate_from_env() == "grading_host"
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    assert emit.substrate_from_env() == "runpod"
    monkeypatch.delenv("SLURM_CLUSTER_NAME")
    monkeypatch.delenv("RUNPOD_POD_ID")
    assert emit.substrate_from_env() == "grading_host"


def test_a_mapped_label_outside_the_vocabulary_is_refused_by_name_at_the_writer(host, monkeypatch):
    monkeypatch.setenv(emit.SITE_MAP_ENV, "cluster-one=not_a_label")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-one")
    assert emit.substrate_from_env() == "not_a_label"
    with pytest.raises(emit.SubstrateRefused) as exc:
        emit.check_substrate(emit.substrate_from_env())
    assert "'not_a_label'" in str(exc.value)


def test_a_subprocess_producer_reads_the_map_from_its_environment(host, tmp_path):
    """The stdlib-only path (``python3 -S``): the map reaches a child through its environment, as registry_env.sh exports it."""
    import subprocess
    import sys
    src = os.path.dirname(os.path.dirname(os.path.dirname(emit.__file__)))
    env = {k: v for k, v in os.environ.items() if k not in SITE_VARS + (emit.SUBSTRATE_ENV, "RUNPOD_POD_ID")}
    env.update({emit.SITE_MAP_ENV: "cluster-two=cluster_b", "SLURM_CLUSTER_NAME": "cluster-two", "PYTHONPATH": src,
                emit.PROBE_ROOT_ENV: str(tmp_path / "empty-host")})
    out = subprocess.run([sys.executable, "-S", "-c", "from sisteer.registry import emit; print(emit.substrate_from_env())"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "cluster_b"
