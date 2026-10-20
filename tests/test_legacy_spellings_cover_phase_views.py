"""Every PHASE VIEW of every front-door model in reproduce/models.yaml has a resolvable `<key>_<view>` row in
configs/registry/legacy_spellings.yaml model_keys (the model_keys seed is a HARD prerequisite of job-start
self-registration — `common.sh f22_registry_register` -> `jobplan.build_plans` -> `jobplan.model_for_key`).

Why: the view `s2r` was added to models.yaml (b6fbb8287) without its model_keys row, so every job targeting it logged
`register: self-registration refused (JobPlanRefused: model key
'<key>_s2r' is not a legacy model spelling in legacy_spellings.yaml model_keys)` and ran UNREGISTERED. A view declared in
models.yaml and absent from the spellings is exactly that failure, caught here before the first job instead of in its log."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO  # noqa: E402

sys.path.insert(0, str(LIB))
import models as M  # noqa: E402

from sisteer.registry import jobplan as JP  # noqa: E402

SPELLINGS = REPO / "configs" / "registry" / "legacy_spellings.yaml"


def _rows():
    doc = yaml.safe_load(SPELLINGS.read_text())
    rows = doc.get("model_keys")
    assert isinstance(rows, list) and rows, "legacy_spellings.yaml: model_keys is not a non-empty list"
    return rows


def _phase_views(key):
    return M.phase_views(key)      # the reader's ONE classification: the grammar's views + the row's declared wave (demo-import M3; the demo's `battery`)


def test_every_phase_view_of_every_front_door_model_has_a_model_keys_spelling():
    rows = _rows()
    keys = M.front_door_keys()
    assert keys, "models.yaml names no front-door model"
    missing = []
    for key in keys:
        served_id, _ = JP.model_for_key(rows, key)       # the bare key resolves (the served checkpoint's id)
        for view in _phase_views(key):
            spelling = f"{key}_{view}"
            try:
                mid, tk = JP.model_for_key(rows, spelling)
            except JP.JobPlanRefused as exc:
                missing.append((spelling, str(exc)))
                continue
            assert mid == served_id, (spelling, mid, served_id)
            assert tk, spelling
    assert not missing, ("phase views declared in reproduce/models.yaml with no legacy_spellings.yaml model_keys row — job-start "
                         "self-registration refuses them by name and the wave runs unregistered: " + repr(missing))


def test_an_undeclared_view_spelling_still_refuses_by_name():
    rows = _rows()
    with pytest.raises(JP.JobPlanRefused, match="is not a legacy model spelling"):
        JP.model_for_key(rows, "qwen35_4b_demo_s9r")
    with pytest.raises(JP.JobPlanRefused, match="is not a legacy model spelling"):
        JP.model_for_key(rows, "qwen35_4b_demo_s2r")      # gemma declares no s2r view, so it has no spelling either
    with pytest.raises(JP.JobPlanRefused, match="is not a legacy model spelling"):
        JP.model_for_key(rows, "qwen35_4b_demo_s1")       # the wave's former spelling: REWRITTEN to qwen35_4b_demo_battery with no alias (Q6, demo-import M3)
    assert JP.model_for_key(rows, "qwen35_4b_demo_battery")[0] == JP.model_for_key(rows, "qwen35_4b_demo")[0]
    assert "battery" in M.phase_views("qwen35_4b_demo") and not M.PHASE_VIEW_RE.match("battery")


def test_the_demo_rows_label_the_fit_your_own_seed_s3_and_cite_the_users_ruling():
    """The three demo rows (`qwen35_4b_demo_battery`, `_d0p75`, `_d0p75_relu`) carry `fit_replicate_label: s3` — the fit-your-own default
    training seed is 3 (the rule of 2026-10-12, matching the one-seed objects/DEMO.json), moved from `s1` by the registry owner's
    ruling of 2026-10-13 05:56 (the label only describes the lane; no reader stamps a legacy row's label onto a demo record) — and each row's
    `source` cites that ruling where "fit seed 1" stood, the rest of the citation kept. RED before the edit: `s1` and "fit seed 1" on all three."""
    rows = _rows()
    cite = "fit seed 3 (the authors' rule: the fit-your-own default training seed is 3, matching the one-seed objects/DEMO.json)"
    kept = {"qwen35_4b_demo_battery": ("SINGLE run", "a standup, not the paper's qwen35_4b_s1 fit-seed replicate"),
            "qwen35_4b_demo_d0p75": ("phase d0p75 on the worked example", "unit lora_write_basis_multiplier"),
            "qwen35_4b_demo_d0p75_relu": ("phase d0p75_relu on the worked example", "unit psr_direction_norm_multiplier, object_family relu")}
    for spelling, phrases in kept.items():
        hits = [r for r in rows if spelling in [x.strip() for x in str(r.get("legacy", "")).split("|")]]
        assert len(hits) == 1, (spelling, hits)
        row = hits[0]
        assert row.get("fit_replicate_label") == "s3", (spelling, row.get("fit_replicate_label"))
        assert cite in row["source"] and "fit seed 1" not in row["source"], (spelling, row["source"])
        for phrase in phrases:
            assert phrase in row["source"], (spelling, phrase)
    # the demo's bare key names no replicate (it serves the checkpoint, not a fitted object), and the paper's own s1 replicate row is untouched
    bare = [r for r in rows if str(r.get("legacy", "")).strip() == "qwen35_4b_demo"]
    assert len(bare) == 1 and "fit_replicate_label" not in bare[0]
    paper_s1 = [r for r in rows if "qwen35_4b_s1" in [x.strip() for x in str(r.get("legacy", "")).split("|")]]
    assert len(paper_s1) == 1 and paper_s1[0].get("fit_replicate_label") == "s1", paper_s1
