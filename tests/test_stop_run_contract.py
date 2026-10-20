"""The STOP-sentinel contract between the stop helper and the self-resubmitting serving probe.

Before: `scripts/sbatch/serving_probe.sbatch` consulted `$OUT/STOP` only, while `reproduce/lib/stop_run.sh` (and the
gpu-avail `stop-run.sh` helper) wrote `$SHARED/exp/training/<job-name>/STOP` before `scancel` — a user who stopped the
probe through the documented helper left no sentinel where the SIGTERM handler looked, and the handler queued a
successor. Now the probe honours BOTH locations and `stop_run.sh` writes both (the contract location always, every
`--stop-dir` in addition), so the whole cancellation path is exercised here: the REAL `stop_run.sh` through the shell
library's `local:` seam against fake `squeue` / `scancel`, then the carrier's OWN `escalate()` under bash.

Synthetic site (neutral labels, placeholder paths), fake Slurm binaries, hostname TESTHOST; nothing reaches a cluster.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from front_door_testlib import assert_no_tokens, write_site

REPO = Path(__file__).resolve().parents[1]
STOP_RUN = REPO / "reproduce/lib/stop_run.sh"
SBATCH = REPO / "scripts/sbatch/serving_probe.sbatch"
NAME = "serving-probe-qwen35_4b_demo"


def _fake_bin(root: Path) -> Path:
    """squeue prints one job id until scancel has run; scancel records its argv; sleep is a no-op (the helper's two
    5 s propagation waits are not the contract under test)."""
    b = root / "bin"; b.mkdir(parents=True, exist_ok=True)
    (b / "squeue").write_text(f'#!/bin/bash\n[ -e "{root}/cancelled" ] || echo 424242\n')
    (b / "scancel").write_text(f'#!/bin/bash\necho "$*" >> "{root}/scancel_calls"; touch "{root}/cancelled"\n')
    (b / "sleep").write_text("#!/bin/bash\nexit 0\n")
    for f in ("squeue", "scancel", "sleep"):
        (b / f).chmod(0o755)
    return b


def _run_stop(site: Path, fake: Path, *args: str) -> subprocess.CompletedProcess:
    env = {"PATH": f"{fake / 'bin'}:{os.environ['PATH']}", "HOME": str(site.parent), "USER": "nobody-testhost"}
    return subprocess.run(["bash", str(STOP_RUN), "--cluster", "cluster_a", "--name", NAME, "--config", str(site / "config.env"), *args],
                          capture_output=True, text=True, timeout=120, env=env)


def _run_escalate(tmp_path: Path, *, out: Path, shared: Path) -> tuple[str, list[str]]:
    """The carrier's OWN sentinel definitions + escalate(), run under bash with `sbatch` stubbed."""
    text = SBATCH.read_text()
    frag = (text[text.index('STOP_FILE="$OUT/STOP"'):text.index("log(){ echo")] + "\n"
            + text[text.index("escalate(){"):text.index('WORK=""\nhandled=0')])
    (tmp_path / "frag.sh").write_text(frag)
    b = tmp_path / "sbin"; b.mkdir(exist_ok=True)
    calls = tmp_path / "sbatch_calls"; calls.unlink(missing_ok=True)
    (b / "sbatch").write_text(f'#!/bin/bash\necho "$*" >> "{calls}"; echo "999;cluster"\n'); (b / "sbatch").chmod(0o755)
    script = (f'set -u; OUT="{out}"; SHARED="{shared}"; SLURM_JOB_NAME={NAME}; ATTEMPT=2; MAX_ATTEMPTS=4; '
              f'SCRIPT_PATH="{tmp_path / "frag.sh"}"; ESCALATE_QOS=high; TIMELIMIT=04:00:00; TP=1; NGPU=8; PARALLEL=auto; STDOUT_PATH=; HEAD_SHA=abc; '
              'COMMENT=sha:abc,nonce:0123456789ab; '      # the job's comment (pin + the funnel's nonce): the successor carries it whole (pin-fix)
              'MODEL_KEY=qwen35_4b_demo; SERVED_ID=org/toy; REVISION=r; SETS=wo; RANKS=4; MODE=tp1; SUCCESSOR=; '
              f'log(){{ echo "[log] $*"; }}; source "{tmp_path / "frag.sh"}"; escalate "SIGTERM"; echo "SUCCESSOR=$SUCCESSOR"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env={"PATH": f"{b}:{os.environ['PATH']}", "USER": "nobody-testhost", "HOME": str(tmp_path)})
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout, (calls.read_text().splitlines() if calls.exists() else [])


def test_stop_run_writes_the_contract_sentinel_and_every_stop_dir_then_cancels_and_the_probe_stands_down(tmp_path):
    fake = tmp_path / "fake"; _fake_bin(fake)
    shared = tmp_path / "shared_a"
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{fake}"}, shared={"cluster_a": str(shared)})
    out = shared / "exp/multilora_cert/qwen35_4b_demo_probe_424242"; out.mkdir(parents=True)
    # BEFORE the helper runs: the carrier resubmits (the record has no sentinel anywhere)
    log, calls = _run_escalate(tmp_path, out=out, shared=shared)
    # positional 9 (parallel) rides the successor after the attempt counter and cap
    assert len(calls) == 1 and "successor job 999 queued" in log and calls[0].endswith("qwen35_4b_demo org/toy r wo 4 tp1 3 4 auto")
    # the REAL helper, through the library's shell path, with the probe's record dir as --stop-dir
    r = _run_stop(site, fake, "--stop-dir", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    contract = shared / "exp/training" / NAME / "STOP"
    assert contract.is_file(), r.stdout                                          # the helper's contract location, ALWAYS
    assert (out / "STOP").is_file(), r.stdout                                    # the job's own location, from --stop-dir
    assert f"STOP sentinel written: {contract}" in r.stdout and f"STOP sentinel written: {out / 'STOP'}" in r.stdout
    # both sentinels were written BEFORE the cancel, and the cancel was by NAME
    assert r.stdout.index(f"STOP sentinel written: {out / 'STOP'}") < r.stdout.index("cancelling: 424242")
    assert (fake / "scancel_calls").read_text().splitlines() == [f"-u nobody-testhost --name={NAME}"]
    assert f"confirmed: no {NAME} jobs remain (no successor was submitted)" in r.stdout
    # AFTER: the carrier's escalate() stands down on the record's sentinel …
    log, calls = _run_escalate(tmp_path, out=out, shared=shared)
    assert calls == [] and f"STOP sentinel {out / 'STOP'} present" in log and "not resubmitting (SIGTERM)" in log
    # … and, with only the CONTRACT sentinel left (the helper run WITHOUT --stop-dir — the reviewer's exact scenario), still
    # stands down: this is the line that failed before the fix
    (out / "STOP").unlink()
    log, calls = _run_escalate(tmp_path, out=out, shared=shared)
    assert calls == [] and f"STOP sentinel {contract} present" in log and "not resubmitting" in log and "SUCCESSOR=\n" in log
    # the runner's OWN paths (the pytest tmp root, HOME) are not hard-coded identifiers — normalised before the scan, the
    # stage-02 dry-run test's convention
    scan = r.stdout.replace(str(tmp_path), "<tmp>").replace(os.path.expanduser("~"), "<home>")
    for name, text in (("the escalation's output", scan), (str(STOP_RUN), STOP_RUN.read_text()), (str(SBATCH), SBATCH.read_text())):
        assert_no_tokens(text, name)


def test_stop_run_without_stop_dir_writes_the_contract_location_only_and_dry_run_prints_both(tmp_path):
    fake = tmp_path / "fake"; _fake_bin(fake)
    shared = tmp_path / "shared_a"
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{fake}"}, shared={"cluster_a": str(shared)})
    r = _run_stop(site, fake)
    assert r.returncode == 0 and (shared / "exp/training" / NAME / "STOP").is_file(), r.stdout + r.stderr
    assert r.stdout.count("STOP sentinel written:") == 1
    # --stop-dir is repeatable and deduplicated against the contract location; --dry-run prints the remote string untouched
    r = _run_stop(site, fake, "--dry-run", "--stop-dir", "/x/probe_out", "--stop-dir", "/x/probe_out",
                  "--stop-dir", str(shared / "exp/training" / NAME))
    assert r.returncode == 0, r.stdout + r.stderr
    line = next(l for l in r.stdout.splitlines() if l.startswith("DRY-RUN: ssh"))
    assert line.count("touch ") == 2 and f"touch '{shared}/exp/training/{NAME}/STOP'" in line and "touch '/x/probe_out/STOP'" in line
    assert line.index("touch '/x/probe_out/STOP'") < line.index("scancel")
    r = _run_stop(site, fake, "--bogus")
    assert r.returncode == 2 and "unknown argument --bogus" in r.stderr


def test_helper_header_documents_both_locations():
    txt = STOP_RUN.read_text()
    assert "--stop-dir DIR]..." in txt and "ALWAYS written" in txt and "SERVING_PROBE_OUT" in txt and "never $WAVE_ROOT/STOP" in txt
    sb = SBATCH.read_text()
    assert 'HELPER_STOP_FILE="$SHARED/exp/training/${SLURM_JOB_NAME:-serving-probe}/STOP"' in sb and "stop_run.sh" in sb
