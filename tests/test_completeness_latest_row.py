"""completeness-latest (2026-10-01): the COMPLETENESS lines of target_attacks.sbatch (and the smoke's count_progress) counted every
historical error row of the append-only progress files — a resumed cell whose errored goals were all retried still read error_outcomes=18 — so
the summary could not tell "stale rows" from "goals still failing". The lines now also report `latest_error_or_refused`: the goals whose LATEST
row is error:* / refused:* (the grader's reading: a goal's latest row is its outcome). Executed: the real heredocs, extracted from the sbatch
text, over a synthetic store."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

F22 = REPO / "scripts" / "sdfbase" / "full22"


def _heredoc(text: str, after: str) -> str:
    """The python heredoc body that follows `after` in the sbatch text: the first `<<'PY'` after the marker opens it (on the marker's own line in
    target_attacks.sbatch, where `| tee` follows the opener; on the next line in the smoke's count_progress), `PY` on its own line closes it."""
    i = text.rfind("\n", 0, text.index(after)) + 1          # the marker's own line may carry the opener before the marker (`<<'PY' | tee …`)
    j = text.index("<<'PY'", i); j = text.index("\n", j) + 1; k = text.index("\nPY\n", j)
    return text[j:k]


def _store(tmp: Path) -> Path:
    root = tmp / "transcripts_results" / "run"
    rows = {"shard00": [{"goal_id": "g1", "outcome": "error:timeout", "n_target_forwards": 0}, {"goal_id": "g2", "outcome": "failure", "n_target_forwards": 7},
                        {"goal_id": "g1", "outcome": "success", "n_target_forwards": 5}],                                   # g1 errored, then its retry succeeded
            "shard01": [{"goal_id": "g3", "outcome": "error:connection", "n_target_forwards": 0}, {"goal_id": "g4", "outcome": "refused:policy", "n_target_forwards": 1},
                        {"goal_id": "g5", "outcome": "failure", "n_target_forwards": 9}]}                                   # g3 still errored, g4 refused
    for sh, rs in rows.items():
        d = root / "actorattack" / "none" / sh; d.mkdir(parents=True)
        (d / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    return root


def test_the_target_jobs_completeness_lines_report_historical_error_rows_and_the_goals_whose_latest_row_still_fails(tmp_path):
    code = _heredoc((F22 / "target_attacks.sbatch").read_text(), 'tee "$SERVE_DIR/completeness.txt"')
    root = _store(tmp_path)
    r = subprocess.run([sys.executable, "-", str(root), "actorattack", "none"], input=code, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "COMPLETENESS actorattack none unique_goals=5 error_outcomes=2 latest_error_or_refused=2", r.stdout     # g1's old row counts historically, not latest; g3 + g4 do
    assert re.search(r"error_outcomes=\d+ latest_error_or_refused=\d+$", r.stdout.strip())


def test_the_smokes_counter_reports_the_same_latest_row_figure(tmp_path):
    code = _heredoc((F22 / "smoke_full22.sbatch").read_text(), "count_progress(){")
    root = _store(tmp_path)
    r = subprocess.run([sys.executable, "-", str(root), "actorattack", "none"], input=code, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "unique_goals=5 error_outcomes=2 latest_error_or_refused=2 forwards=22", r.stdout
