"""The GLM descope adapter: one `descope` ruling under the deterministic id every GLM run references, the lane's
files registered as artifacts, the model seed's reason carried."""
from __future__ import annotations

import json
import os

import _b1_fixtures as F
from sisteer.registry.adapters import legacy_glm as LG
from sisteer.registry.adapters import lineage_common as LC


def test_descope_ruling_and_artifacts(tmp_path):
    rs = tmp_path / "rs"
    os.makedirs(rs / "glm_errored117", exist_ok=True)
    os.makedirs(rs / "controls_glm", exist_ok=True)
    json.dump({"n_triples": 3, "pairs": {}}, open(rs / "glm_errored117" / "comparison.json", "w"))
    json.dump({"target": "glm46vflash", "cells": {}}, open(rs / "controls_glm" / "controls_table_glm.json", "w"))
    seeds = F.synthetic_seeds()
    ad = LG.LegacyGlmAdapter(rs=str(rs), seeds=seeds)
    root = F.registry_root(tmp_path, seeds)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals)
    rul = F.q(root, "SELECT ruling_id, action, scope_kind, scope_json, reason, ruled_utc_precision FROM ruling")
    assert len(rul) == 1 and rul[0][0] == LC.DESCOPE_GLM_RULING_ID and rul[0][1] == "descope" and rul[0][2] == "model"
    assert json.loads(rul[0][3]) == {"model_id": "glm46vflash"} and "descoped" in rul[0][4] and rul[0][5] == "date"
    arts = F.q(root, "SELECT kind, uri_primary FROM artifact ORDER BY 2")
    assert len(arts) == 2 and all(a[0] == "report" for a in arts)
    # a second pass writes nothing new
    rep2, _, _ = F.run_and_ingest(root, ad)
    assert rep2.groups_refreshed == 0
