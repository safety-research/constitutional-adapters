"""``REPRO_DRY_RUN`` is fail-closed across ``source reproduce/lib/common.sh``: a caller that
exported ``REPRO_DRY_RUN=1`` BEFORE sourcing the libs stays dry — the libs only ever RAISE the flag (``--dry-run``), never
reset it to 0. Before: line 35 re-initialised it to 0 at source time, so a driver that set the flag in its environment and
then sourced the libs ran ``slurm_revoke_geometry`` for real and archived the qwen35 geometry overlays on the site and
both clusters while believing it was dry."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "reproduce" / "lib" / "common.sh"


def _dry_after_source(env_value: str | None, args: str = "") -> str:
    pre = f"REPRO_DRY_RUN={env_value}; export REPRO_DRY_RUN; " if env_value is not None else "unset REPRO_DRY_RUN; "
    script = (pre + f"source '{COMMON}'; repro_parse_common_args {args}; printf '%s' \"$REPRO_DRY_RUN\"")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True, cwd=str(REPO))
    return out.stdout.strip()


def test_an_exported_dry_run_flag_survives_sourcing_the_libs():
    assert _dry_after_source("1") == "1"
    assert _dry_after_source("1", "--model x") == "1"


def test_unset_is_not_dry_and_dash_dash_dry_run_raises_it():
    assert _dry_after_source(None) == "0"
    assert _dry_after_source(None, "--dry-run") == "1"
    assert _dry_after_source("0", "-n") == "1"        # the flag is only ever raised, never lowered, by the libs


def test_the_libs_never_lower_an_explicit_value_to_zero_by_assignment():
    text = COMMON.read_text(encoding="utf-8")
    assert 'REPRO_DRY_RUN="${REPRO_DRY_RUN:-0}"' in text and "REPRO_DRY_RUN=0;" not in text
