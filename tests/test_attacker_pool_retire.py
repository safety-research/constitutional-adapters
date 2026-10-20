"""31d: the attacker pool member RETIRE path — attackers/<job>.RETIRE makes the member withdraw its record, drain and self-exit as `retired`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from test_attacker_pool_lifecycle import _call, _world  # noqa: E402

def test_a_retire_sentinel_is_read_and_the_pool_sbatch_withdraws_drains_and_self_exits_as_retired(tmp_path):
    w = _world(tmp_path)
    r = _call(w, "f22_pool_retire_requested", str(w["root"] / "attackers"), "4242"); assert r.returncode == 1
    (w["root"] / "attackers" / "4242.RETIRE").write_text(json.dumps({"group": "g", "why": "ratio", "utc": "2026-01-01T00:00:00Z"}))
    r = _call(w, "f22_pool_retire_requested", str(w["root"] / "attackers"), "4242"); assert r.returncode == 0
    sb = (REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool.sbatch").read_text()
    assert "f22_pool_retire_requested" in sb and 'rm -f "$REC" "$PENDING"' in sb and "F22_POOL_RETIRE_GRACE_S" in sb and '"reason":"retired"' in sb
    assert sb.index("f22_pool_retire_requested") < sb.index("f22_pool_member_verdict")              # retire is read before the lifecycle verdict
