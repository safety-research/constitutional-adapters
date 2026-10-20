"""reclose-at-tree: the recovery for a closer defect fixed AFTER the eval job is a reclose that runs the CLOSER
from the fixed tree — never a hand close. `submit_twin.py reclose --job J --tree <sha>` submits reclose_stores.sbatch with a THIRD positional,
the closer's tree; the job's own tree stays the recipe's identity (the sbatch's recipe refusal is untouched) and the runs' --code-sha stays the
job's; the reservation token, the ledger `reclose` event and the sbatch's receipt record BOTH shas. Refusals by name: a --tree that is not a
40-hex sha, the job's own tree, a tree with no checkout carrying the closer (scripts/sdfbase/full22/registry_close_stores.py + src/sisteer).
Synthetic world (tests/test_submit_twin.World), synthetic shas."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from test_closer_refusal_outcome import _argv, _need_closer_history, _tree_with_closer, _world  # noqa: E402
from test_submit_twin import SHA  # noqa: E402

SHA2 = "b" * 40
RECLOSE = REPO / "scripts/sdfbase/full22/reclose_stores.sbatch"


def _closer_tree(w, sha: str, *, with_closer: bool = True):
    co = w.shared["cluster_a"] / "sisteer-eval-repo" / sha
    (co / "scripts/sdfbase/full22").mkdir(parents=True, exist_ok=True)
    (co / "src/sisteer").mkdir(parents=True, exist_ok=True)
    if with_closer:
        _tree_with_closer(co)            # closer-fix-3: the closer (with gate_binding) and the reclose script the funnel submits from this tree
    return co


def test_reclose_at_tree_submits_with_the_closers_tree_as_positional_3_and_records_both_shas(tmp_path):
    _need_closer_history()
    w = _world(tmp_path)
    _closer_tree(w, SHA2)
    r = w.run("reclose", "--job", "1001", "--tree", SHA2); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, out
    assert f"with the closer of tree {SHA2[:12]} (reclose-at-tree: the job's sha stays {SHA[:12]})" in r.stdout, r.stdout
    argv = _argv(w, "1002")
    assert f"scripts/sdfbase/full22/reclose_stores.sbatch 1001 {SHA} {SHA2}" in argv and "-J f22-reclose-1001" in argv, argv      # positional 3 = the closer's tree
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    assert len(ev) == 1 and ev[0]["code_sha"] == SHA and ev[0]["closer_sha"] == SHA2 and ev[0]["job_id"] == "1001" and ev[0]["reclose_job_id"] == "1002"


def test_a_plain_reclose_is_byte_identical(tmp_path):
    _need_closer_history()
    w = _world(tmp_path)
    r = w.run("reclose", "--job", "1001"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, r.stdout + r.stderr
    argv = _argv(w, "1002")
    assert argv.rstrip().endswith(f"scripts/sdfbase/full22/reclose_stores.sbatch 1001 {SHA}") or f"reclose_stores.sbatch 1001 {SHA}\n" in argv or f"reclose_stores.sbatch 1001 {SHA} " not in argv
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    assert len(ev) == 1 and "closer_sha" not in ev[0] and "reclose-at-tree" not in r.stdout


def test_the_tree_refusals_are_by_name(tmp_path):
    """Refusals raised before the closer-tree guard reads the tree's commit: they run everywhere, the published repository included (a fresh
    world: a reclose already in flight would otherwise be the reason)."""
    w2 = _world(tmp_path / "w2")
    r = w2.run("reclose", "--job", "1001", "--tree", "not-a-sha"); assert r.returncode != 0 and "--tree names no 40-hex sha" in r.stderr, r.stdout + r.stderr
    r = w2.run("reclose", "--job", "1001", "--tree", SHA); assert r.returncode != 0 and "is the job's own tree" in r.stderr, r.stdout + r.stderr
    r = w2.run("reclose", "--job", "1001", "--tree", SHA2); assert r.returncode != 0 and "no checkout with the closer at" in r.stderr and SHA2[:12] in r.stderr, r.stdout + r.stderr
    _closer_tree(w2, SHA2, with_closer=False)                                                             # src/sisteer present, the closer script absent: still refused
    r = w2.run("reclose", "--job", "1001", "--tree", SHA2); assert r.returncode != 0 and "no checkout with the closer at" in r.stderr, r.stdout + r.stderr
    assert [e for e in w2.ledger() if e["event"] == "reclose"] == []                                     # nothing submitted by a refusal


def test_the_sbatch_runs_the_closer_from_the_closers_tree_and_keeps_the_jobs_identity():
    src = RECLOSE.read_text()
    assert subprocess.run(["bash", "-n", str(RECLOSE)]).returncode == 0
    i_pos = src.index('ORIG="${1:?original job id}"; WAVE_SHA="${2:?wave sha}"; CLOSER_SHA="${3:-}"')
    i_ref = src.index('[ "$R_CODE_SHA" = "$WAVE_SHA" ] || reclose_refuse')                                 # the recipe's JOB-sha refusal stands
    i_dir = src.index('CLOSER_DIR="${SISTEER_EVAL_REPO_ROOT:-$_shared/sisteer-eval-repo}/$CLOSER_SHA"')
    i_chk = src.index('|| reclose_refuse "--tree $CLOSER_SHA: no checkout with the closer at $CLOSER_DIR')
    i_run = src.index('"$F22_PY" "$CLOSER_DIR/scripts/sdfbase/full22/registry_close_stores.py" --pending "$R_PENDING"')
    assert i_pos < i_ref < i_dir < i_chk < i_run
    assert '--code-sha "$WAVE_SHA"' in src[i_run:i_run + 400]                                              # the runs' code sha stays the job's
    assert 'PYTHONPATH="$CLOSER_DIR/src' in src[i_run - 200:i_run]                                          # the closer tree's sisteer
    assert '"job_sha": os.environ.get("R_JOB_SHA") or None, "closer_sha": os.environ.get("R_CLOSER_SHA") or os.environ.get("R_JOB_SHA") or None' in src
    assert 'R_JOB_SHA="$WAVE_SHA" R_CLOSER_SHA="${CLOSER_SHA:-}" R_CLOSER_EXTRA="${CLOSER_EXTRA:-}" python3 -S -' in src   # the receipt writer sees both trees and the closer args
    assert 'CLOSER_DIR="$REPO_DIR"' in src and src.index('CLOSER_DIR="$REPO_DIR"') < i_dir                 # a plain reclose: the job's tree, as before


def test_a_closer_arg_with_shell_metacharacters_is_refused_by_name_and_the_sbatch_takes_the_args_on_positional_4(tmp_path):
    # the refusal (before the closer-tree guard) and the script's source: everywhere, the published repository included
    w3 = _world(tmp_path / "w3")
    r = w3.run("reclose", "--job", "1001", "--closer-arg=--x; rm -rf /"); assert r.returncode != 0 and "shell metacharacters" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert not (w3.fake["cluster_a"] / "jobs" / "1002.argv").exists() and [e for e in w3.ledger() if e["event"] == "reclose"] == []
    src = RECLOSE.read_text()
    assert 'CLOSER_EXTRA="${4:-}"' in src and 'eval "CLOSER_EXTRA_ARGS=($CLOSER_EXTRA)"' in src and '${CLOSER_EXTRA_ARGS[@]+"${CLOSER_EXTRA_ARGS[@]}"} 2>&1)' in src
    assert '"closer_extra_args": os.environ.get("R_CLOSER_EXTRA") or None' in src and 'R_CLOSER_EXTRA="${CLOSER_EXTRA:-}" python3 -S -' in src


def test_closer_args_ride_positional_4_and_are_recorded(tmp_path):
    _need_closer_history()
    w = _world(tmp_path)
    _closer_tree(w, SHA2)
    r = w.run("reclose", "--job", "1001", "--tree", SHA2, "--closer-arg=--ruled-out-shards", "--closer-arg=5,6,7,8", "--closer-arg=--ruled-out-objects", "--closer-arg", "none",
              "--closer-arg=--ruling-evidence-uri", "--closer-arg", "rulings/added/x.jsonl"); out = r.stdout + r.stderr          # a value that starts with -- rides the = form
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "with closer args --ruled-out-shards 5,6,7,8 --ruled-out-objects none --ruling-evidence-uri rulings/added/x.jsonl" in r.stdout, out
    argv = _argv(w, "1002")
    assert f"reclose_stores.sbatch 1001 {SHA} {SHA2} " in argv and "--ruled-out-shards 5,6,7,8 --ruled-out-objects none --ruling-evidence-uri rulings/added/x.jsonl" in argv, argv
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    assert len(ev) == 1 and ev[0]["closer_sha"] == SHA2 and ev[0]["closer_args"] == ["--ruled-out-shards", "5,6,7,8", "--ruled-out-objects", "none", "--ruling-evidence-uri", "rulings/added/x.jsonl"]
    # closer args without --tree: positional 3 is the empty string (the job's own tree), positional 4 the args
    w2 = _world(tmp_path / "w2")
    r = w2.run("reclose", "--job", "1001", "--closer-arg=--dry-run"); assert r.returncode == 0, r.stdout + r.stderr
    assert f"reclose_stores.sbatch 1001 {SHA} '' --dry-run" in _argv(w2, "1002") or f"reclose_stores.sbatch 1001 {SHA}  --dry-run" in _argv(w2, "1002"), _argv(w2, "1002")
