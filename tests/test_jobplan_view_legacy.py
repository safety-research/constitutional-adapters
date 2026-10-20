"""`run.view_legacy` is STAMPED by job-start registration (paper-companion-repo review C11): every plan `jobplan.build_plans` builds
carries the job's model-key spelling (`<key>` or `<key>_<selection>`) on the COLUMN the paper views read -- before 2026-09-29 it was parked
in `extra.view_legacy` only and all 320 job-start-registered runs read NULL there. Folded through the harvester: the column is set, the
run is `not_in_roster` on `v_paper_current` (a spelling no roster names), and the paper's current selection is untouched by it."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from sisteer.registry import db as DB
from sisteer.registry import emit, emit_cli
from sisteer.registry import harvest as H
from sisteer.registry import jobplan as JP
from sisteer.registry.adapters import dimensions as D
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "configs" / "registry"


def _checkout(tmp_path: Path, key: str = "qwen35_4b") -> dict:
    root = tmp_path / "checkout"
    (root / "configs").mkdir(parents=True)
    os.symlink(SEEDS, root / "configs" / "registry")
    cfg = root / "configs" / f"synth_full22_{key}.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": ["none", "prompted__si_constitution"], "model": "Qwen/Qwen3.5-4B"}))
    (root / "configs" / "sdfbase_full22_roster.yaml").write_text(yaml.safe_dump({"targets": {key: {"views": {
        "main": {"cells": {"prompted__si_constitution": None}}, "s1": {"cells": {"prompted__si_constitution": None}}}}}}))
    (root / "configs" / "sdfbase_full22_pins.json").write_text("{}")
    return {"root": root, "cfg": cfg}


def test_every_plan_stamps_view_legacy_with_the_model_key_spelling(tmp_path):
    co = _checkout(tmp_path)
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["agentic_murder"], serving_path="lora_native", campaign="sdfbase_full22",
                       job_id="1", job_name="j", substrate="cluster_a", code_sha=synth.code_sha("a"))
    assert b.runs and all(r.plan["view_legacy"] == "qwen35_4b" for r in b.runs)
    b2 = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b_s1", roster_selection="s1", evals=["agentic_murder"], serving_path="lora_native",
                        campaign="sdfbase_full22", job_id="2", job_name="j", substrate="cluster_a", code_sha=synth.code_sha("a"))
    assert b2.runs and all(r.plan["view_legacy"] == "qwen35_4b_s1" and r.plan["extra"]["view_legacy"] == "s1" for r in b2.runs)
    # the plan validates as the record it becomes (view_legacy is a real run column)
    producer = emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="cluster_a")
    for r in b.runs + b2.runs:
        assert emit.check_planned_run(r.plan, producer) == r.run_id


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("reg") / "root")
    H.ensure_layout(root)
    D.write_seed_records(root, str(SEEDS), producer=emit.producer_block("tests.view_legacy", code_sha=synth.code_sha("a"), substrate="grading_host"),
                         now="2026-09-16T12:00:00.000Z")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def _selection_hash(conn) -> str:
    rows = conn.execute("SELECT run_id, in_current_paper, selection_state FROM v_paper_current ORDER BY run_id").fetchall()
    return hashlib.sha256(json.dumps([list(r) for r in rows]).encode()).hexdigest()


def test_folded_run_carries_the_column_reads_not_in_roster_and_moves_no_paper_selection(registry, tmp_path, monkeypatch):
    root = registry
    co = _checkout(tmp_path)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = os.path.join(root, "records", "producer")
    with DB.reading(DB.db_path(root)) as conn:
        before = _selection_hash(conn)
    rc = emit_cli.register_from_config(str(co["cfg"]), model_key="qwen35_4b_s1", roster_selection="s1", evals=["agentic_murder"], serving_path="lora_native",
                                       campaign="sdfbase_full22", job_id="8801", pending_dir=str(tmp_path / "pending"), job_name="f22-misalign-qwen35_4b",
                                       substrate="cluster_a", code_sha=synth.code_sha("a"), spool=spool, push=False)
    assert rc == 0
    pending = json.loads((tmp_path / "pending" / "8801.json").read_text())
    rids = [r["run_id"] for r in pending["runs"]]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    with DB.reading(DB.db_path(root)) as conn:
        for rid in rids:
            row = conn.execute("SELECT view_legacy, json_extract(extra_json, '$.view_legacy'), model_id FROM run WHERE run_id = ?", (rid,)).fetchone()
            assert tuple(row) == ("qwen35_4b_s1", "s1", "qwen35_4b_it"), tuple(row)
            pc = conn.execute("SELECT selection_state, in_current_paper FROM v_paper_current WHERE run_id = ?", (rid,)).fetchone()
            assert pc is None or (pc[0] == "not_in_roster" and pc[1] == 0), pc
        assert _selection_hash(conn) == before or conn.execute("SELECT COUNT(*) FROM v_paper_current WHERE in_current_paper = 1").fetchone()[0] == 0
