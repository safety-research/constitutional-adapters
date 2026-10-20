"""`--emit-expected-goal-ids` binds PENDING_DIR from the pending file before resolving the registered config,
as close_job does — the narrowed registration (`<job>.config.yaml` beside the pending file, stamped `pending/<job>.config.yaml` on the
pre-fix pending files) was refused by the ids emitter while the closer itself would have found it. The emitter's resolver is stubbed at the
function boundary (the real resolver needs the harness's datasets); what is pinned is the binding and the path the stub is handed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import registry_close_stores as RC  # noqa: E402


def test_emit_expected_goal_ids_resolves_the_registered_config_beside_the_pending_file(tmp_path, monkeypatch):
    pend = tmp_path / "shared" / "registry" / "pending"; pend.mkdir(parents=True)
    cfg = pend / "4242.config.yaml"; cfg.write_text("explicit_cells: [none]\nprivate_run: synthetic\n")
    pending = pend / "4242.json"
    pending.write_text(json.dumps({"job_name": "f22-att-m-g01234-crescendo", "runs": [{"run_id": "r1", "plan": {"extra": {"config": "pending/4242.config.yaml"}}}]}))
    run = json.loads(pending.read_text())["runs"][0]
    # before the binding (PENDING_DIR unset): the legacy relative stamp resolves nowhere — the pre-fix emitter's refusal
    monkeypatch.setattr(RC, "PENDING_DIR", None)
    why, path, rel = RC._registered_config_path(run, repo_root=str(tmp_path / "checkout"))
    assert why and "is not on disk (tried" in why and rel == "pending/4242.config.yaml" and str(pend) not in why, why
    seen = {}

    def stub(run_, out_path, *, repo_root):
        seen["pending_dir"] = RC.PENDING_DIR
        seen["resolved"] = RC._registered_config_path(run_, repo_root=repo_root)
        Path(out_path).write_text(json.dumps({"goal_ids": ["g"]})); return None

    monkeypatch.setattr(RC, "write_expected_goal_ids", stub)
    out = tmp_path / "ids.json"
    rc = RC.main(["--pending", str(pending), "--emit-expected-goal-ids", str(out), "--spool", str(tmp_path / "spool")])
    assert rc == 0 and out.is_file()
    assert seen["pending_dir"] == str(pend) and seen["resolved"] == (None, str(cfg), "pending/4242.config.yaml"), seen
    # the binder is one function used by both entries, and main binds it before the emit branch
    src = Path(RC.__file__).read_text()
    assert src.count("_bind_pending_dir(") == 3 and src.index("_bind_pending_dir(a.pending)") < src.index("if a.emit_expected_goal_ids:")
    assert "    _bind_pending_dir(pending_path)\n    with open(pending_path, encoding=\"utf-8\") as fh:" in src
