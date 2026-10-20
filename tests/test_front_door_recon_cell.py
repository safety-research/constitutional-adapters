"""The reconnaissance cell: the front door's DEFAULT recon cell is the paper's constv2ctl2-full (150 steps fit in one pass at
effective batch 64), SISTEER_HPO_RECON_CELL names another, and a cell too short for the recipe's 150 steps in one pass REFUSES
by name (rows, steps per pass, the budget) unless SISTEER_HPO_RECON_ALLOW_CYCLING=1 — then the passes are RECORDED
(cycled_passes) rather than taken silently. The refusal is 03_hpo.sh's own `recon_passes_for_cell`, executed under bash."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

#: the front-door model: the lane whose fits config realises effective batch 64 (the EFF64 sentinel; demo-import M4), which the FD 6
#: file's OVERRIDE_EFFECTIVE_BATCH=64 must equal — 03 cross-checks the two and refuses the paper's qwen35_4b (its config realises 8)
#: under this file by name (tests/test_hpo_effective_batch_cross_check.py)
MODEL = "qwen35_4b_demo"
#: a SYNTHETIC run-overrides file in the FD 6 protocol (03_hpo.sh: reconnaissance clip + one warm-up level + the rank x module-set
#: search) — the reconnaissance phase exists in that protocol only; under the paper's defaults it "does not apply and says so"
FD6_OVERRIDES = """OVERRIDE_ID=fd6_protocol_under_test
OVERRIDE_INIT_SCHEME=random_lecun_1e-3
OVERRIDE_LORA_ARCH_SEARCH=1
OVERRIDE_LORA_RANK=''
OVERRIDE_LORA_MODULE_SET=''
OVERRIDE_HPO_WARMUP_FRAC=0.15
OVERRIDE_HPO_CLIP_MODE=reconnaissance
OVERRIDE_HPO_CLIP_PERCENTILE=95
OVERRIDE_HPO_TRIALS=16
OVERRIDE_HPO_RECENTRE_TRIALS=4
OVERRIDE_REFUSAL_STEER_ALPHA=0.3
OVERRIDE_EFFECTIVE_BATCH=64
"""


def _site(root: Path, extra: str = "") -> Path:
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    overrides = site / "overrides" / "fd6_protocol_under_test.env"
    overrides.write_text(FD6_OVERRIDES)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n"
                 f"SISTEER_RUN_OVERRIDES={overrides}\n{extra}")
    return site


def _recon_line(out: str) -> str:
    lines = [ln for ln in out.splitlines() if "sbatch" in ln and "HPO_JOB_MODE=recon" in ln]
    assert len(lines) == 1, out[-2500:]
    return lines[0]


def test_default_recon_cell_is_the_papers_and_the_ladder_is_an_array(tmp_path):
    assert re.search(r"^SISTEER_HPO_RECON_CELL=constv2ctl2-full\b", (REPRO / "site.example" / "config.env").read_text(), re.M)
    site = _site(tmp_path)
    # the example config names the paper's cell explicitly; the SCRIPT's own default is the same cell when the key is absent
    cfg = site / "config.env"; cfg.write_text(re.sub(r"^SISTEER_HPO_RECON_CELL=.*$", "", cfg.read_text(), flags=re.M))
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "recon", "--model", MODEL, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    line = _recon_line(out)
    assert "/pairs_constv2ctl2-full.jsonl" in line and "--array=0-4" in line, line
    assert re.search(r'recon_spec\.json: \{.*"si": "constv2ctl2-full"', out), out
    assert "recon_passes_for_cell ROWS 64 150" in out, out                      # the live check is stated by the dry run


def test_site_names_another_recon_cell(tmp_path):
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "recon", "--model", MODEL,
                  site=_site(tmp_path, "SISTEER_HPO_RECON_CELL=constv2-mix-full\n"), timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "/pairs_constv2-mix-full.jsonl" in _recon_line(out) and '"si": "constv2-mix-full"' in out


def _passes(rows: int, eff: int = 64, steps: int = 150, env: dict | None = None) -> subprocess.CompletedProcess:
    src = (REPO / "reproduce/03_hpo.sh").read_text()
    frag = src[src.index("recon_passes_for_cell(){"):src.index("widest_pair(){")]
    script = (f"set -u; source {REPO}/reproduce/lib/common.sh; RECON_SI=cell_under_test\n{frag}\n"
              f"recon_passes_for_cell {rows} {eff} {steps}")
    e = {"PATH": os.environ["PATH"], "HOME": str(Path.home())}
    e.update(env or {})
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=60)


def test_a_cell_that_covers_the_budget_in_one_pass_is_one_pass():
    r = _passes(17073)                                            # the paper's constv2ctl2-full: 267 steps per pass >= 150
    assert r.returncode == 0 and r.stdout.strip() == "1", r.stdout + r.stderr
    r = _passes(9600)                                             # exactly 150 steps per pass
    assert r.returncode == 0 and r.stdout.strip() == "1", r.stdout + r.stderr


def test_a_short_cell_refuses_by_name_unless_cycling_is_opted_into():
    r = _passes(4102)                                             # the trial pool: 65 steps per pass
    assert r.returncode != 0 and r.stdout.strip() == "", r.stdout
    msg = r.stderr
    assert "REFUSED" in msg and "4102 rows" in msg and "65 optimizer steps per pass" in msg and "150" in msg, msg
    assert "SISTEER_HPO_RECON_CELL" in msg and "SISTEER_HPO_RECON_ALLOW_CYCLING=1" in msg and "cycling the cell 3 times" in msg, msg
    r = _passes(4102, env={"SISTEER_HPO_RECON_ALLOW_CYCLING": "1"})
    assert r.returncode == 0 and r.stdout.strip() == "3", r.stdout + r.stderr
    assert "CYCLING 3 passes" in r.stderr and "cycled_passes=3" in r.stderr, r.stderr
    r = _passes(0)
    assert r.returncode != 0 and "cannot count the rows" in r.stderr, r.stderr


def _archive(tmp_path: Path, *, dry: bool, stale: bool) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """03_hpo.sh's own recon_archive_record under bash with a RECORDING `ssh` (never a network) and a local mirror dir."""
    src = (REPO / "reproduce/03_hpo.sh").read_text()
    frag = src[src.index("recon_archive_record(){"):src.index("widest_pair(){")]
    import tempfile
    tmp_path = Path(tempfile.mkdtemp(dir=tmp_path))                                              # one world per call
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    rec = tmp_path / "ssh.log"
    (bin_dir / "ssh").write_text("#!/bin/bash\n" f'printf "%s\\n" "$*" >> {rec}\necho none\n'); (bin_dir / "ssh").chmod(0o755)
    local = tmp_path / "mirror"; local.mkdir()
    if stale:
        for n in ("recon_0.json", "recon_2.stamp.json", "recon.json"):
            (local / n).write_text("{}")
    (local / "recon_spec.json").write_text("{}")
    script = (f"set -u; source {REPO}/reproduce/lib/common.sh; REPRO_DRY_RUN={1 if dry else 0}; CL_HOST=cluster-a-login\n"
              'disp(){ echo "$1"; }\n' 'fetch_invalidate(){ echo "fetch-cache invalidated for $1 -> $2"; }\n' f"{frag}\n" f'recon_archive_record /shared/exp/reproduce/m/03_hpo/relu "{local}" first-reason\n'
              + ('' if not stale else f'for n in recon_1.json recon.json; do echo "{{}}" > "{local}/$n"; done\n'
                                       f'recon_archive_record /shared/exp/reproduce/m/03_hpo/relu "{local}" second-reason\n'))
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path)}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60), rec, local


def test_archive_moves_stale_rung_files_to_superseded_on_both_sides_and_never_deletes(tmp_path):
    r, rec, local = _archive(tmp_path, dry=False, stale=True)
    assert r.returncode == 0, r.stdout + r.stderr
    import shlex
    toks = shlex.split(rec.read_text())                              # the remote body rides %q-quoted: read it as bash would
    body = toks[toks.index("-c") + 1]
    assert "cluster-a-login" in toks and toks[-2:] == ["/shared/exp/reproduce/m/03_hpo/relu", toks[-1]] and re.fullmatch(r"\d{8}T\d{6}Z\.\d+\.[A-Za-z0-9_-]+", toks[-1]), toks
    assert "superseded/" in body and "mkdir -p" in body and " mv " in body and "recon_[0-9]*.json" in body and "recon.json" in body, body
    assert "rm " not in body and "rm\t" not in body
    # TWO archivals in the same second (a foreign marker and a foreign verdict in one invocation) land in two DISTINCT directories —
    # utc + nanoseconds + the transition's reason — and both sets survive
    sup = sorted((local / "superseded").iterdir()); assert len(sup) == 2, sup
    assert sup[0].name.endswith(".first-reason") and sup[1].name.endswith(".second-reason") and sup[0].name.split(".")[0] and sup[0] != sup[1]
    assert sorted(p.name for p in sup[0].iterdir()) == ["recon.json", "recon_0.json", "recon_2.stamp.json"]
    assert sorted(p.name for p in sup[1].iterdir()) == ["recon.json", "recon_1.json"]
    assert not (local / "recon_0.json").exists() and (local / "recon_spec.json").exists()      # the spec stays; nothing deleted
    assert r.stdout.count("the grading host mirror archived ->") == 2 and " mv -n " in body
    assert r.stdout.count("fetch-cache invalidated for /shared/exp/reproduce/m/03_hpo/relu ->") == 2   # every archival forgets the pair


def test_archive_with_nothing_stale_is_quiet_and_a_dry_run_touches_nothing(tmp_path):
    r, rec, local = _archive(tmp_path, dry=False, stale=False)
    assert r.returncode == 0 and not (local / "superseded").exists() and "archived" not in r.stdout, r.stdout + r.stderr
    r, rec, local = _archive(tmp_path, dry=True, stale=True)
    assert r.returncode == 0 and "DRY-RUN: archive any recon_*.json" in r.stdout and not rec.exists(), r.stdout + r.stderr
    assert (local / "recon_0.json").exists() and not (local / "superseded").exists()


def test_a_colliding_archival_directory_is_refused_never_overwritten(tmp_path):
    """With `date` frozen (a fake on PATH), the second archival's directory name collides with the first: the cluster body exits 3 with
    COLLISION and the front door refuses; the first archival's files are untouched."""
    src = (REPO / "reproduce/03_hpo.sh").read_text()
    frag = src[src.index("recon_archive_record(){"):src.index("widest_pair(){")]
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    (bin_dir / "date").write_text("#!/bin/bash\ncase \"$*\" in *%N*) echo 123456789;; *) echo 20260930T120000Z;; esac\n"); (bin_dir / "date").chmod(0o755)
    (bin_dir / "ssh").write_text('#!/bin/bash\nwhile [ "${1:-}" = -o ]; do shift 2; done; shift\nexec bash -c "$*"\n'); (bin_dir / "ssh").chmod(0o755)
    rec = tmp_path / "record"; rec.mkdir(); (rec / "recon_0.json").write_text("{}")
    local = tmp_path / "mirror"; local.mkdir()
    script = (f"set -u; source {REPO}/reproduce/lib/common.sh; REPRO_DRY_RUN=0; CL_HOST=fake\n" 'disp(){ echo "$1"; }\n' 'fetch_invalidate(){ :; }\n' f"{frag}\n"
              f'recon_archive_record "{rec}" "{local}" r; echo "first rc=$?"; echo "{{}}" > "{rec}/recon_0.json"; recon_archive_record "{rec}" "{local}" r; echo "second rc=$?"')
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path)}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    out = r.stdout + r.stderr
    assert "first rc=0" in out and "second rc=" not in out and r.returncode != 0, out          # the second refuses (repro_fatal) …
    assert "COLLISION" in out and "nothing overwritten" in out, out
    sup = list((rec / "superseded").iterdir()); assert len(sup) == 1 and sup[0].name == "20260930T120000Z.123456789.r"
    assert (sup[0] / "recon_0.json").read_text() == "{}" and (rec / "recon_0.json").exists()    # … and the first archival is intact
