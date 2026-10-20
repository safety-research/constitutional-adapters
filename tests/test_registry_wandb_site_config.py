"""The W&B resolver's entity, candidate projects and offline mirror are the site's (the registry configuration's ``[wandb]`` table),
never defaults in code: with none configured the API is not consulted and every id says why, and the bare adapter declares no root.
"""
from __future__ import annotations

import ast
import json
import os

from sisteer.registry.adapters import wandb_runs as WB


def _root(tmp_path, body: str = "") -> str:
    root = tmp_path / "reg"
    root.mkdir()
    (root / "config.toml").write_text(body, encoding="utf-8")
    return str(root)


def _mirror(tmp_path) -> str:
    m = tmp_path / "mirror"
    d = m / "offline-run-20260811_005649-abcd1234" / "files"
    d.mkdir(parents=True)
    (d / "wandb-metadata.json").write_text(json.dumps({"project": "example-project"}), encoding="utf-8")
    return str(m)


def test_no_entity_project_or_mirror_is_a_literal_of_the_module():
    with open(WB.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert not names & {"DEFAULT_ENTITY", "DEFAULT_PROJECTS", "DEFAULT_MIRROR"}


def test_settings_read_the_wandb_table(tmp_path):
    assert WB.settings(cfg={}) == WB.Settings(None, (), None)
    s = WB.settings(cfg={"wandb": {"entity": " example-entity ", "projects": ["p1", "", "p2"], "mirror": "~/w/wandb"}})
    assert s == WB.Settings("example-entity", ("p1", "p2"), os.path.expanduser("~/w/wandb"))
    assert WB.settings(cfg={"wandb": {"projects": "one"}}).projects == ("one",)
    root = _root(tmp_path, '[wandb]\nentity = "example-entity"\nprojects = ["p1"]\nmirror = "/srv/w/wandb"\n')
    assert WB.settings(root) == WB.Settings("example-entity", ("p1",), "/srv/w/wandb")


def test_with_no_entity_configured_the_api_is_not_consulted_and_each_id_says_why(tmp_path):
    root = _root(tmp_path)
    res = WB.resolve(["zzzz9999"], root=root, use_api=True)
    assert res["zzzz9999"].url is None and "[wandb] entity" in res["zzzz9999"].reason
    assert res["zzzz9999"].confidence()["confidence"] == "unverified"


def test_the_configured_mirror_and_entity_resolve_a_named_run(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    m = _mirror(tmp_path)
    root = _root(tmp_path, f'[wandb]\nentity = "example-entity"\nprojects = ["p1"]\nmirror = "{m}"\n')
    res = WB.resolve(["abcd1234", "zzzz9999"], root=root)
    assert res["abcd1234"].url == "https://wandb.ai/example-entity/example-project/runs/abcd1234"
    assert "WANDB_MODE=disabled" in res["zzzz9999"].reason


def test_the_bare_adapter_takes_its_root_from_the_configuration_or_declares_none(tmp_path, monkeypatch):
    m = _mirror(tmp_path)
    monkeypatch.setenv("REGISTRY_ROOT", _root(tmp_path, f'[wandb]\nmirror = "{m}"\n'))
    ad = WB.WandbRunsAdapter(run_ids=["abcd1234"])
    assert ad.roots == (m,) and len(ad.discover()) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("REGISTRY_ROOT", str(empty))
    bare = WB.WandbRunsAdapter(run_ids=["abcd1234"])
    assert bare.mirror is None and bare.roots == () and bare.discover() == []
