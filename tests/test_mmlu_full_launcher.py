"""mmlu-full-1531: the two cluster launchers of the full-pool MMLU read.

``scripts/sdfbase/full22/mmlu_full.sbatch`` (every roster view: one standup per cell class, the
driver's ``--mmlu-full``, its OWN root ``mmlu_full/``) and the ``legs=mmlu_full`` leg of
``scripts/arditi/sbatch/ar_cell.sbatch`` (the Arditi lane's one cell). Pinned here: the cluster hard
rules (no ``--export=``, no ``--mem=0``, never ``dev`` in sbatch), the wave's hygiene header, the
flag read from the contracts BY IMPORT (never a retyped literal), the distinct out-roots (never the
200-item stores'), and a job name outside ``f22_registry_evals_for_job``'s table so nothing
self-registers a 1,531-item run under the 200-item ``mmlu`` identity."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

SBATCH = REPO / "scripts" / "sdfbase" / "full22" / "mmlu_full.sbatch"
AR_CELL = REPO / "scripts" / "arditi" / "sbatch" / "ar_cell.sbatch"
COMMON = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"


@pytest.mark.parametrize("path", [SBATCH, AR_CELL])
def test_launchers_parse_and_keep_the_cluster_hard_rules(path):
    txt = path.read_text()
    assert not re.search(r"^#SBATCH --export", txt, re.M)          # hard rule 1
    assert "--export=" not in "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"^#SBATCH --mem=0", txt, re.M)           # hard rule 2
    assert not re.search(r"^#SBATCH --qos=dev", txt, re.M)         # dev is srun-only
    assert "CUDA_VISIBLE_DEVICES=" not in "\n".join(                # hard rule 3: never WIDENED (the empty strict subset is the one allowed spelling)
        l for l in txt.splitlines() if "CUDA_VISIBLE_DEVICES=" in l and 'CUDA_VISIBLE_DEVICES=""' not in l and not l.lstrip().startswith("#"))
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_mmlu_full_sbatch_is_the_controls_phase_at_the_full_width_into_its_own_root():
    txt = SBATCH.read_text()
    for header in ("#SBATCH --qos=high", "#SBATCH --gres=gpu:1", "#SBATCH --open-mode=append",
                   "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120", "#SBATCH --mem=200G"):
        assert header in txt, header
    # the flag by CONSTANT, read under the driver venv — never the literal
    assert "print(tc.FLAG_MMLU_FULL)" in txt
    assert C.FLAG_MMLU_FULL not in txt.replace("--mmlu-full:", "").replace("--mmlu-full —", "").replace("driver's --mmlu-full", ""), \
        "the flag's spelling belongs to the contracts; the sbatch reads it by import"
    assert '--controls mmlu "$MMLU_FULL_FLAG"' in txt
    # the same instrument otherwise: the server's --max-logprobs IS the driver's --top-logprobs, read by import
    assert "print(int(m.MMLU_TOP_LOGPROBS))" in txt
    assert 'f22_standup attacks "$SERVE_DIR/$cls" "" capability:short "$L" "" "$CTL_TOP_LOGPROBS"' in txt
    assert '--top-logprobs "$CTL_TOP_LOGPROBS"' in txt
    # its OWN root, never the 200-item stores'
    assert 'OUT_ROOT="$WAVE_ROOT/mmlu_full/$TAGT"' in txt
    assert "/sampled/mmlu/" not in txt.replace("(sampled/mmlu/,", "")
    # the positive control and the served-cells filter, as the controls phase
    assert "f22_positive_control" in txt and "f22_served_cells_filter cells" in txt
    # long-SI cells: a second standup at a wider window, the roster's controls window winning when declared
    assert "LONG_SI_CELLS=(prompted__si_constitution prompted__si_irrelevant)" in txt
    assert 'run_class longsi "${VIEW_WINDOW_CONTROLS:-$MMLU_FULL_LONG_SI_WINDOW}"' in txt
    # the job-name prefix registers nothing: no kind in the registry table matches it
    assert "#SBATCH --job-name=mmluF" in txt
    table = COMMON.read_text()
    m = re.search(r"f22_registry_evals_for_job\(\)\{.*?\n\}", table, re.S)
    assert m and "mmluF" not in m.group(0)


def test_ar_cell_mmlu_full_leg_is_alone_in_its_own_root():
    txt = AR_CELL.read_text()
    assert "all|controls|static|xstest|orbench|mmlu_full)" in txt
    assert "  mmlu_full) RUN_MMLU_FULL=1;;" in txt
    # `all` does NOT imply it (the published per-rung set is unchanged)
    assert "  all)      RUN_CONTROLS=1; RUN_MMLU=1; RUN_STATIC=1;;" in txt
    assert 'if [ "$RUN_MMLU_FULL" = 1 ]; then' in txt
    assert '--out-root "$CTL_ROOT/mmlu_full"' in txt
    assert '--controls mmlu "$MMLU_FULL_FLAG"' in txt
    assert "print(tc.FLAG_MMLU_FULL)" in txt
    # the standard 200-item leg is byte-for-byte what it was: its root, its argv
    assert '--out-root "$CTL_ROOT/mmlu" \\' in txt
    assert txt.count('--controls mmlu --trial "$TRIAL" --concurrency "$TARGET_S"') == 1
    assert '"mmlu_full": ["mmlu_full"]}[legs]' in txt
