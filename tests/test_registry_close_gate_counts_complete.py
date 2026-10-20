"""closer-fix-4: a gate record whose counts block LACKS a GATE_COUNT_KEYS key is refused by name — the
re-derivation loop used to skip an absent key, so a counts block of another closer generation (a classifier's count missing) could read as
"reproduced" on the keys it happened to carry. The producer (ladder_decide.decide_controls, through the shared fixture) writes every key."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_registry_close_gate_rederivation import _gate_path  # noqa: E402
from test_registry_close_stores_gate_decision import KEY, OBJ, RC, TAG, _world  # noqa: E402


def test_a_count_key_the_record_lacks_is_a_refusal_by_name(tmp_path):
    root = _world(tmp_path / "w", rungs={512: True, 2048: False}, gate={512: "escalate", 2048: "exit_le_1pct"})
    p = _gate_path(root, 2048); base = json.loads(p.read_text())
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["final_rung"] == 2048 and facts["gate_rederived"] == "reproduced", why
    assert set(RC.GATE_COUNT_KEYS) <= set(base["counts"]), sorted(base["counts"])        # the producer writes every key the closer compares
    for k in ("n_sibling_unreadable", "n_boundary_any", "n_responses"):
        d = json.loads(json.dumps(base)); d["counts"].pop(k); p.write_text(json.dumps(d))
        why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
        if k == "n_responses":
            assert why and "final_rung" not in facts, (k, why)                           # the closer-fix-2 count check refuses first, by name
        else:
            assert why and f"counts block lacks {k} (GATE_COUNT_KEYS)" in why and "re-run the ladder gate at this rung" in why and "final_rung" not in facts, (k, why)
            assert "gate_rederived" not in facts and "rederived" not in facts
    # several absent keys: all named, in GATE_COUNT_KEYS order
    d = json.loads(json.dumps(base)); d["counts"].pop("n_loop"); d["counts"].pop("n_repetitive"); p.write_text(json.dumps(d))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why and "counts block lacks n_loop, n_repetitive (GATE_COUNT_KEYS)" in why, why
    # intact again: reproduced
    p.write_text(json.dumps(base))
    why, facts, _s, _a = RC.inspect_controls(str(root), TAG, 0, OBJ, KEY, planned=250)
    assert why is None and facts["gate_rederived"] == "reproduced", why
