"""The front door's GSM8K row makes its own prompt-budget artifact in-job when none was made before launch (paper-companion-repo
stage E, 2026-09-30). The paper's rows had their (key, class, rung) budgets made by hand before launch; a front-door model has no such
step, so its three u0 rows died at `no prompt-budget artifact`. `g8k_budget_make_if_absent` runs the same producer the smoke's in-job
G-FIT runs (tinycap_oos_prompt_budget.py: task / arm / cap / geometry-eval / config / model-key / tree-sha / offline / out), at THIS
tree's sha so the sidecar's pin holds, only when the artifact is absent, and the sbatch calls it immediately before the artifact check
that holds the row to it."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "scripts/sdfbase/gsm8k/g8k_common.sh"
SBATCH = REPO / "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch"


def _body(src: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}\(\)\{{.*?^\}}", src, re.M | re.S)
    assert m, name
    return m.group(0)


def _flags(body: str) -> set:
    return set(re.findall(r"(--[a-z-]+)", body.split("tinycap_oos_prompt_budget.py", 1)[1].split("\\n\\n")[0]))


def test_the_row_makes_its_budget_in_job_before_the_artifact_check_with_the_smokes_producer_flags():
    for f in (COMMON, SBATCH):
        subprocess.run(["bash", "-n", str(f)], check=True)
    common, sbatch = COMMON.read_text(), SBATCH.read_text()
    make, smoke = _body(common, "_g8k_budget_make_locked"), _body(common, "g8k_smoke_budget")     # the producer call lives in the locked section
    # the same producer, the same flags, the wave sha as the tree pin, offline, on CPU
    # the smoke's flags plus the PROMPT CLASS: a prompted row folds its SI into the budget; the smoke is cell `none` by construction
    # the producer's flags, the smoke's plus the prompt class (which rides `${sp[@]}` — the `sp=(--system-prompt-cell …)` line is pinned below)
    assert "tinycap_oos_prompt_budget.py" in make and _flags(make) | {"--system-prompt-cell"} == _flags(smoke) | {"--system-prompt-cell"}, (_flags(make), _flags(smoke))
    assert 'case "$CELL" in prompted__*) sp=(--system-prompt-cell "$CELL");; esac' in make
    # … and the artifact check holds the sidecar to the row's prompt class (system_prompt_cell + the SI's sha16 for a prompted row)
    art = _body(common, "g8k_budget_artifact")
    assert '"$G8K_THINKING" "$CELL" "$KEY" <<' in art and "system_prompt_cell" in art and "system_prompt_sha16" in art
    assert '--tree-sha "$WAVE_SHA"' in make and "--offline" in make and 'CUDA_VISIBLE_DEVICES=""' in make
    assert '--cap "$RUNG"' in make and '--out "$BUDGET_SRC"' in make        # the production row's cap and artifact path (g8k_core)
    # only when absent: an existing artifact + sidecar is never rewritten
    # an existing pair is honoured only when the currency predicate says CURRENT; a stale pair is superseded and re-made
    assert 'if [ -f "$BUDGET_SRC" ] && [ -f "$BUDGET_SRC.fit.json" ]; then' in make and "CURRENT*) return 0;;" in make and "STALE*)" in make
    assert '_g8k_budget_make_locked 9>"$BUDGET_SRC.lock"' in _body(common, "g8k_budget_make_if_absent")
    # the sbatch: make-if-absent immediately before the check that holds the row to the artifact
    lines = [ln.strip().split("#")[0].strip() for ln in sbatch.splitlines()]
    i = lines.index("g8k_budget_artifact")
    assert lines[i - 1] == "g8k_budget_make_if_absent", lines[i - 2:i + 1]
