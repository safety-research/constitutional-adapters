"""An errored goal must NOT count as covered, on every attack path.

Job 206660 errored all 209 goals of a shard, wrote 209 progress rows and zero
responses, and exited 0. The all-errored guard in ``run_static`` closes that
for the FIRST run. It cannot close the second: every later run of the same
shard loaded those 209 rows as "covered", found nothing to do, and reported a
complete pass. A failure that is permanent AND looks like success is strictly
worse than the original bug, so the resume set is now success-only.
"""
import json

from sisteer.eval.attacks.static_common import load_covered_goals
from sisteer.eval.attacks.gcg import _load_covered_goals as gcg_covered
from sisteer.eval.attacks.pyrit_adapters import (
    _load_covered_goals as pyrit_covered)

ALL = (load_covered_goals, gcg_covered, pyrit_covered)


def _write(tmp_path, rows):
    p = tmp_path / "progress.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_errored_goal_is_not_covered(tmp_path):
    p = _write(tmp_path, [
        {"goal_id": "g0", "outcome": "done"},
        {"goal_id": "g1", "outcome": "error:ValueError"},
    ])
    for fn in ALL:
        assert fn(p) == {"g0"}, fn


def test_all_errored_leaves_todo_nonempty(tmp_path):
    """The exact 206660 shape: a resume must re-attempt every goal, so the
    all-errored guard has a non-empty ``todo`` to rule on."""
    p = _write(tmp_path, [
        {"goal_id": f"g{i}", "outcome": "error:ValueError"} for i in range(209)
    ])
    for fn in ALL:
        assert fn(p) == set(), fn


def test_later_error_supersedes_earlier_pass(tmp_path):
    p = _write(tmp_path, [
        {"goal_id": "g0", "outcome": "done"},
        {"goal_id": "g0", "outcome": "error:RuntimeError"},
    ])
    for fn in ALL:
        assert fn(p) == set(), fn


def test_missing_outcome_still_counts_as_covered(tmp_path):
    """Older rows predate the outcome field; absent is not errored."""
    p = _write(tmp_path, [{"goal_id": "g0"}])
    for fn in ALL:
        assert fn(p) == {"g0"}, fn


def test_torn_final_line_is_skipped(tmp_path):
    p = tmp_path / "progress.jsonl"
    p.write_text(json.dumps({"goal_id": "g0", "outcome": "done"}) + "\n"
                 + '{"goal_id": "g1", "outc')
    for fn in ALL:
        assert fn(p) == {"g0"}, fn
