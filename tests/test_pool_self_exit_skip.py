"""pool-self-exit skip: the `--resubmit` walk re-placed attacker pool 2 of the s1 wave (880002) — its bound job had COMPLETED
with attackers/<job>.SELF_EXIT (exit_unbound: the group was placed and every target bound elsewhere), the walk fenced it as a teardown and then
re-minted it as a dead, never-registered unit, beside the live member every remaining target was bound to; the new pool stood up to find no
consumer and was cancelled by ruling. Now: a self-exited member is a FINISHED pool slot while another member of its group is live — skipped by
name, nothing submitted, the teardown recorded; with no live member left it is fenced like a torn-down one and re-minted, as before."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP, POOL, _members  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402


def _walk(w, i):
    return w.run("submit", "--unit", f"multiturn.attacker{i}", "--lwid", f"m/u0/multiturn/{GROUP}/attackers/{i}", "--script", POOL, "--name", f"f22-atk-m-{i}",
                 "--args", SHA, "--clusters", "cluster_a", "--kind", "attacker", "--group", GROUP, "--wave-sha", SHA, "--resubmit", "--walk")


def test_a_self_exited_member_is_skipped_while_another_member_of_the_group_is_live_and_re_minted_once_the_pool_is_gone(tmp_path):
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"; pool.mkdir(parents=True, exist_ok=True)
    # member 1 closed itself (the group placed, nothing bound it) and COMPLETED; member 2 serves on
    (pool / f"{j1}.SELF_EXIT").write_text(json.dumps({"job_id": j1, "reason": "exit_unbound", "consumers": "",
                                                       "why": "placement complete (2 target(s)), every target bound elsewhere or ended, none bound me", "utc": "2026-01-01T02:00:00Z"}))
    set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00"); set_job_state(w.fake["cluster_a"], j2, "RUNNING")
    w.spool("cluster_a")
    before = job_ids(w.fake["cluster_a"])
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "SELF-EXITED" in out and f"served by live member(s) {j2}" in out and "never re-placed (D1713)" in out, out
    assert "RESUBMIT" not in r.stdout and job_ids(w.fake["cluster_a"]) == before                                    # nothing submitted
    assert len([e for e in w.ledger() if e["event"] == "torn_down" and e.get("self_exit") and e["job_id"] == j1]) == 1   # the teardown is recorded ONCE
    assert not [e for e in w.ledger() if e["event"] in ("attempt", "submitted") and e.get("logical_work_id", "").endswith("/attackers/1") and str(e.get("job_id")) != j1]
    # the walk is idempotent on it: a second pass skips again, no second torn_down flood is required of the caller
    r = _walk(w, 1); assert r.returncode == 0 and "never re-placed (D1713)" in (r.stdout + r.stderr) and job_ids(w.fake["cluster_a"]) == before
    assert len([e for e in w.ledger() if e["event"] == "torn_down" and e["job_id"] == j1]) == 1                      # no torn_down per pass
    # the pool is gone (member 2 COMPLETED, no sentinel): the self-exited slot is fenced like a torn-down one and re-minted, as before
    set_job_state(w.fake["cluster_a"], j2, "COMPLETED", end="2026-01-01T03:00:00"); w.spool("cluster_a")
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT" in r.stdout and "SELF-EXITED" in out and "never re-placed" not in out, out
    assert len(job_ids(w.fake["cluster_a"])) == len(before) + 1


def test_the_decision_names_the_rule_and_reads_live_members_in_the_self_exit_branch():
    from front_door_testlib import LIB
    src = (LIB / "submit_twin.py").read_text()
    body = src[src.index("def resubmit_decision("):]; body = body[:body.index("def arr(ts)")]
    i_se = body.index("self.member_self_exited(label, spec.get(\"group\"), job_id)"); i_live = body.index("self.live_members(str(spec.get(\"group\")))"); i_dead = body.index("dead, completed = list(tasks), []", i_se)
    assert i_se < i_live < i_dead                                                                                      # skip decided before the fence
    assert "a finished pool slot is never re-placed (D1713)" in body and "SELF-EXITED (attackers/<job>.SELF_EXIT) while another member of its group is live -> `skip`" in src
