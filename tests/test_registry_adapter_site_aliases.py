"""Adapters read persisted state through the root's site aliases: a staging record, a lane file or a provenance file a site wrote with
its own spelling of a cluster reads as the label, and the adapter pass installs the alias table of the root it runs under. Every legacy
spelling here is invented (``oldsite_a``)."""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import site_aliases as SA
from sisteer.registry.adapters import objects_bank as OB
from sisteer.registry.adapters import runner as RUN


@pytest.fixture(autouse=True)
def _empty_active_table():
    SA.reset()
    yield
    SA.reset()


def _staging_record(root, sha, spelling):
    os.makedirs(os.path.join(root, "_records"), exist_ok=True)
    with open(os.path.join(root, "_records", "obj.json"), "w", encoding="utf-8") as fh:
        json.dump({"object_sha256": sha, f"{spelling}_path": "/stage/obj", f"{spelling}_verified": True,
                   f"{spelling}_remote_sha256": sha, "pulled_at": "2026-01-01T00:00:00Z", "bytes": 7}, fh)


def test_a_staging_record_keyed_by_a_legacy_spelling_reads_as_the_label(tmp_path):
    sha = "a" * 64
    _staging_record(str(tmp_path), sha, "oldsite_a")
    assert OB.staging_locations(str(tmp_path), None).get(sha) is None          # no alias table: the keys name no cluster
    SA.install_table({"oldsite_a": "cluster_a"})
    (loc,) = OB.staging_locations(str(tmp_path), None)[sha]
    assert loc["uri"] == "cluster_a:/stage/obj" and loc["remote_sha256"] == sha and loc["verified_utc"] == "2026-01-01T00:00:00Z"


def test_the_adapter_pass_installs_the_alias_table_of_its_root(tmp_path):
    root = tmp_path / "reg"
    root.mkdir()
    (root / SA.FILE_NAME).write_text('[site_aliases]\noldsite_a = "cluster_a"\n')
    with pytest.raises(ValueError, match="force applies to a re-read only"):
        RUN.run_pass(str(root), [], force=True)                                # refused right after the install
    assert SA.installed() == {"oldsite_a": "cluster_a"}
