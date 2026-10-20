"""wandb_runs: only run ids a sidecar / trial log names are resolved; the offline mirror is consulted first; a
disabled API leaves the URL null with the reason; `apply_to_objects` fills the URL and the confidence entry.
"""
from __future__ import annotations

import json
import os

from sisteer.registry.adapters import wandb_runs as WB
from registry_fit_fixtures import _write


def _mirror(tmp_path):
    m = str(tmp_path / "mirror")
    d = os.path.join(m, "offline-run-20260811_005649-abcd1234", "files")
    _write(d, "wandb-metadata.json", {"project": "synthetic-project", "program": "tune_arm.py", "host": "TESTHOST"})
    _write(d, "config.yaml", "lr:\n  value: 0.001\n_wandb:\n  value: {}\n")
    os.makedirs(os.path.join(m, "offline-run-20260811_005650-nometa", "files"))
    return m


def test_mirror_resolves_named_ids_only(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    m = _mirror(tmp_path)
    res = WB.resolve(["abcd1234", "nometa", "zzzz9999"], entity="synthetic-entity", mirror=m, use_api=True)
    assert set(res) == {"abcd1234", "nometa", "zzzz9999"}
    r = res["abcd1234"]
    assert r.url == "https://wandb.ai/synthetic-entity/synthetic-project/runs/abcd1234" and r.project == "synthetic-project"
    assert r.source.startswith("mirror:") and r.config == {"lr": 0.001}
    assert res["nometa"].url is None and "WANDB_MODE=disabled" in res["nometa"].reason
    assert res["zzzz9999"].url is None and res["zzzz9999"].confidence()["confidence"] == "unverified"


def test_api_not_consulted_when_disabled(tmp_path):
    m = _mirror(tmp_path)
    res = WB.resolve(["zzzz9999"], mirror=m, use_api=False)
    assert res["zzzz9999"].url is None and "use_api=False" in res["zzzz9999"].reason


def test_apply_to_objects_fills_url_and_confidence(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    m = _mirror(tmp_path)
    objs = [{"object_id": "a", "fit_wandb_run_id": "abcd1234", "field_confidence_json": {}, "extra": {}},
            {"object_id": "b", "fit_wandb_run_id": "zzzz9999", "field_confidence_json": {}},
            {"object_id": "c", "field_confidence_json": {}}]
    res = WB.resolve([o.get("fit_wandb_run_id") for o in objs if o.get("fit_wandb_run_id")], entity="e", mirror=m)
    n = WB.apply_to_objects(objs, res)
    assert n == 1
    assert objs[0]["fit_wandb_url"] == "https://wandb.ai/e/synthetic-project/runs/abcd1234"
    assert objs[0]["field_confidence_json"]["fit_wandb_url"]["confidence"] == "recorded" and objs[0]["extra"]["wandb"]["project"] == "synthetic-project"
    assert "fit_wandb_url" not in objs[1] and objs[1]["field_confidence_json"]["fit_wandb_url"]["confidence"] == "unverified"
    assert "fit_wandb_url" not in objs[2]["field_confidence_json"]


def test_adapter_registers_only_named_mirror_metadata(tmp_path):
    m = _mirror(tmp_path)
    ad = WB.WandbRunsAdapter(mirror=m, run_ids=["abcd1234", "nometa", "zzzz9999"])
    inputs = ad.discover()
    assert [os.path.basename(os.path.dirname(os.path.dirname(i.uri))) for i in inputs] == ["offline-run-20260811_005649-abcd1234"]
    assert ad.comparison_group(inputs[0]).endswith("offline-run-20260811_005649-abcd1234")
    assert list(ad.read(None, "g", inputs)) == []
