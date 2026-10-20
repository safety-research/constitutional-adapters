"""The funnel's `reclose` submits scripts/sdfbase/full22/reclose_stores.sbatch FROM
THE TREE THE CLOSER RUNS FROM — `--tree`'s checkout when given, else the job's own — and refuses by name a run tree that lacks
`gate_binding` in registry_close_stores.py (26a27aac3 is the first tree carrying it: a closer that does not bind the gate record may misread
it) or whose reclose script predates reclose-at-tree (no `CLOSER_SHA` positional 3). Before: the funnel ran the JOB tree's script, which on every
pre-26a27aac3 job tree dropped positional 3, so `--tree` was inert and nine recloses had to be submitted by hand. The fake sbatch records
its working directory; the ledger event and the reservation token record `script_tree`. Synthetic world (tests/test_submit_twin.World), synthetic
shas; the job's sha stays the recipe's identity and the runs' code sha."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_closer_refusal_outcome import _argv, _need_closer_history, _tree_with_closer, _world  # noqa: E402
from test_reclose_at_tree import SHA2, _closer_tree  # noqa: E402
from test_submit_resubmit_proof import MMLU  # noqa: E402
from test_submit_twin import SHA  # noqa: E402


def _cwd(w, job: str) -> Path:
    return Path((w.fake["cluster_a"] / "jobs" / f"{job}.cwd").read_text().strip()).resolve()


def _job_tree(w) -> Path:
    return w.shared["cluster_a"] / "sisteer-eval-repo" / SHA


def test_with_tree_the_script_is_submitted_from_the_closers_tree_and_every_record_says_so(tmp_path):
    _need_closer_history()
    w = _world(tmp_path); co2 = _closer_tree(w, SHA2)
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and f"(the script from tree {SHA2[:12]})" in r.stdout, out
    assert _cwd(w, "1002") == co2.resolve()                                                            # sbatch ran in the CLOSER tree's checkout
    argv = _argv(w, "1002")
    assert f"scripts/sdfbase/full22/reclose_stores.sbatch 1001 {SHA} {SHA2}" in argv and "-J f22-reclose-1001" in argv, argv   # the job's sha stays positional 2
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    assert len(ev) == 1 and ev[0]["code_sha"] == SHA and ev[0]["closer_sha"] == SHA2 and ev[0]["script_tree"] == SHA2, ev


def test_a_plain_reclose_runs_from_the_jobs_tree_and_records_it(tmp_path):
    _need_closer_history()
    w = _world(tmp_path)
    r = w.run("reclose", "--job", "1001"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, r.stdout + r.stderr
    assert _cwd(w, "1002") == _job_tree(w).resolve() and f"(the script from tree {SHA[:12]})" in r.stdout
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    assert len(ev) == 1 and ev[0]["script_tree"] == SHA and "closer_sha" not in ev[0]
    # the 06 walk reaches the same line (exit 75 on a registered unit): the script from the job's tree
    w2 = _world(tmp_path / "w2")
    r = w2.run(*MMLU, "--resubmit")
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, r.stdout + r.stderr
    assert _cwd(w2, "1002") == _job_tree(w2).resolve()


def test_the_d1437_shape_a_job_tree_whose_script_predates_reclose_at_tree(tmp_path):
    _need_closer_history()                           # part 1 submits; part 2's script refusal is the guard's LAST check, past the ancestry check
    # the job's tree: the closer carries gate_binding, but its reclose script drops positional 3 (every job tree before reclose-at-tree landed)
    w = _world(tmp_path); _tree_with_closer(_job_tree(w), plumbed=False); co2 = _closer_tree(w, SHA2)
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, out                     # --tree: the script comes from the closer's tree, positional 3 intact
    assert _cwd(w, "1002") == co2.resolve() and f"reclose_stores.sbatch 1001 {SHA} {SHA2}" in _argv(w, "1002")
    # WITHOUT --tree the same job tree is refused by name — a script that would drop the positional; nothing submitted, nothing on the ledger
    w2 = _world(tmp_path / "w2"); _tree_with_closer(_job_tree(w2), plumbed=False)
    r = w2.run("reclose", "--job", "1001"); out = r.stdout + r.stderr
    assert r.returncode == 3 and "predates reclose-at-tree (no CLOSER_SHA positional 3)" in r.stderr and "pass --tree <a landed tree at or past b36e48a2f>" in r.stderr \
        and "D1437" in r.stderr and "nothing submitted" in r.stderr, out
    assert not (w2.fake["cluster_a"] / "jobs" / "1002.argv").exists() and [e for e in w2.ledger() if e["event"] == "reclose"] == []
    assert not any(p.name.endswith(".reclose.json") for p in (w2.state / "bindings").glob("*")) if (w2.state / "bindings").exists() else True   # no reservation left behind


def test_a_tree_lacking_closer_fix_2_is_refused_by_name_plain_and_explicit(tmp_path):
    _need_closer_history()                           # the fix-2 and script refusals sit past the ancestry check; w4's wording is pinned everywhere by test_reclose_at_tree
    w = _world(tmp_path); _tree_with_closer(_job_tree(w), fix2=False)
    r = w.run("reclose", "--job", "1001"); out = r.stdout + r.stderr
    assert r.returncode == 3 and f"the job's tree {SHA[:12]} lacks closer-fix-2 (no `gate_binding` in" in r.stderr and "b36e48a2f is the minimum tree" in r.stderr \
        and "pass --tree <a landed tree at or past b36e48a2f>" in r.stderr and "nothing submitted" in r.stderr, out
    assert not (w.fake["cluster_a"] / "jobs" / "1002.argv").exists() and [e for e in w.ledger() if e["event"] == "reclose"] == []
    w2 = _world(tmp_path / "w2"); _tree_with_closer(w2.shared["cluster_a"] / "sisteer-eval-repo" / SHA2, fix2=False)
    r = w2.run("reclose", "--job", "1001", "--tree", SHA2); out = r.stdout + r.stderr
    assert r.returncode == 3 and f"--tree {SHA2[:12]} lacks closer-fix-2" in r.stderr and "land and mirror a tree that carries it" in r.stderr, out
    assert not (w2.fake["cluster_a"] / "jobs" / "1002.argv").exists()
    # an explicit tree whose reclose script lacks the plumbing: refused by name too (the funnel would submit THAT script)
    w3 = _world(tmp_path / "w3"); _tree_with_closer(w3.shared["cluster_a"] / "sisteer-eval-repo" / SHA2, plumbed=False)
    r = w3.run("reclose", "--job", "1001", "--tree", SHA2); out = r.stdout + r.stderr
    assert r.returncode == 3 and "predates reclose-at-tree (no CLOSER_SHA positional 3)" in r.stderr and SHA2 in r.stderr and "land and mirror a tree that carries it" in r.stderr, out
    # the missing-tree refusals keep their wording (test_reclose_at_tree pins them): no checkout at all, and src/sisteer without the closer script
    w4 = _world(tmp_path / "w4")
    r = w4.run("reclose", "--job", "1001", "--tree", SHA2); assert r.returncode == 3 and "no checkout with the closer at" in r.stderr and SHA2[:12] in r.stderr, r.stderr
    _closer_tree(w4, SHA2, with_closer=False)
    r = w4.run("reclose", "--job", "1001", "--tree", SHA2); assert r.returncode == 3 and "no checkout with the closer at" in r.stderr, r.stderr


def test_the_real_reclose_script_and_the_real_closer_carry_the_markers_the_funnel_checks(tmp_path):
    """The markers are read from the REAL files, so a rename there fails here before it fails on the cluster."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce" / "lib"))
    import submit_twin as ST
    repo = Path(__file__).resolve().parents[1]
    closer = (repo / "scripts/sdfbase/full22/registry_close_stores.py").read_text()
    assert ST.Coordinator.CLOSER_FIX2_MARKER in closer
    assert ST.Coordinator.RECLOSE_TREE_MARKER in (repo / ST.RECLOSE_SCRIPT).read_text()
    assert ST.Coordinator.CLOSER_FIX2_SHA == "26a27aac30fc50398604aa450970f91d8373e85d"
    # the publication marker the refusal names is the one name the publish step, the snapshot precondition and the test fixtures share
    assert ST.Coordinator.PUBLISHED_MARKER == ".history-squashed"
