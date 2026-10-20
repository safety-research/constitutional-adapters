"""The data-parallel fit width through the front door.

The fits config's top-level `fit_width: N` is THE width: 04_fit_objects.sh sizes the sbatch allocation from it
(`--gres=gpu:N --cpus-per-task=16N --mem=128N G`, lib/common.sh repro_fit_width_flags) and puts SDFBASE_FIT_WIDTH=N in the
job env; 03_hpo.sh states that its trials stay width 1; the sbatch's own fit_width_flags spells the same flags. No committed fits
config declares the key now (every lane, the demo's included, fits at width 1: absent = 1), so the width-4 case runs on a SYNTHETIC
front-door row whose fits config declares `fit_width: 4`, in a tmp checkout of the repo (the stages locate the repo from their own
path, so the checkout copies `reproduce/`, links the rest and carries the synthetic config and table row).
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

FIT = REPO / "scripts" / "sbatch" / "sdfbase_fit_cell.sbatch"
COMMON = REPRO / "lib" / "common.sh"
DEMO = "qwen35_4b_demo"                       # the tree's front-door model: no fit_width line, width 1
W4 = "synth_w4"                               # a SYNTHETIC front-door row whose fits config declares fit_width: 4 (tmp checkout only)
W4_FLAGS = "--gres=gpu:4 --cpus-per-task=64 --mem=512G"


def _width4_checkout(root: Path) -> Path:
    """A tmp checkout the stages run from (lib/common.sh derives REPRO_REPO from the script's path): `reproduce/` copied, `scripts/`,
    `src/` and every config linked, plus the SYNTHETIC row `synth_w4` (the demo's pins under its own ids) in the checkout's models table and its
    fits config — the demo's with a top-level `fit_width: 4` line — committed, so the stage's sha resolves."""
    co = root / "checkout"
    shutil.copytree(REPO / "reproduce", co / "reproduce", ignore=shutil.ignore_patterns("__pycache__"))
    for d in ("scripts", "src"):
        (co / d).symlink_to(REPO / d)
    (co / "configs").mkdir()
    for p in (REPO / "configs").iterdir():
        (co / "configs" / p.name).symlink_to(p)
    demo_cfg = (REPO / "configs" / f"sdfbase_{DEMO}_base_fits.yaml").read_text()
    assert re.search(r"^fit_width:", demo_cfg, re.M) is None                    # the demo fits at width 1
    (co / "configs" / f"sdfbase_{W4}_base_fits.yaml").write_text(demo_cfg.replace("\nmodel:", "\nfit_width: 4\nmodel:", 1))
    t = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text())
    row = copy.deepcopy(t["models"][DEMO]); row["view_legacy_prefix"] = W4
    row["fit"]["checkpoint"]["model_id"] = f"{W4}_base"; row["serve"]["checkpoint"]["model_id"] = W4
    t["models"][W4] = row
    (co / "reproduce" / "models.yaml").write_text(yaml.safe_dump(t, sort_keys=False))
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
           "HOME": str(root), "PATH": os.environ["PATH"]}
    subprocess.run(["git", "-C", str(co), "init", "-q"], check=True, env=env)
    subprocess.run(["git", "-C", str(co), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(co), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "synthetic width-4 row"], check=True, env=env)
    return co


def _run_checkout_stage(co: Path, stage: str, *args: str, site: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    e = dict(os.environ); e.pop("SISTEER_SITE_DIR", None); e.pop("SISTEER_CONFIG", None)
    e["SISTEER_STATE_DIR"] = str(site.parent / "state_w4")
    return subprocess.run(["bash", str(co / "reproduce" / stage), *args, "--config", str(site / "config.env")], capture_output=True, text=True, env=e,
                          cwd=str(co), timeout=timeout)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("fw")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


@pytest.fixture(scope="module")
def width4(tmp_path_factory) -> Path:
    return _width4_checkout(tmp_path_factory.mktemp("w4"))


def test_no_committed_config_declares_a_fit_width_and_the_synthetic_row_declares_4(width4):
    declared = {}
    for p in sorted((REPO / "configs").glob("sdfbase_*_fits.yaml")):
        m = re.search(r"^fit_width: (\d+)$", p.read_text(), re.M)
        declared[p.name] = int(m.group(1)) if m else None
    assert declared[f"sdfbase_{DEMO}_base_fits.yaml"] is None                       # the demo lane: no key, width 1 (like the paper's)
    assert declared["sdfbase_qwen35_4b_base_fits.yaml"] is None                    # the paper lane: no key, width 1
    assert declared and all(v is None for v in declared.values()), {k: v for k, v in declared.items() if v is not None}
    r = subprocess.run(["bash", "-c", f'source {COMMON} 2>/dev/null; repro_fit_width_of_config {REPO}/configs/sdfbase_{DEMO}_base_fits.yaml; '
                        f'repro_fit_width_of_config {width4}/configs/sdfbase_{W4}_base_fits.yaml'], capture_output=True, text=True)
    assert r.stdout.split() == ["1", "4"], r.stdout + r.stderr


@pytest.mark.parametrize("n,expect", [("1", ""), ("2", "--gres=gpu:2 --cpus-per-task=32 --mem=256G"),
                                      ("4", "--gres=gpu:4 --cpus-per-task=64 --mem=512G"),
                                      ("8", "--gres=gpu:8 --cpus-per-task=128 --mem=1024G")])
def test_the_submitters_flags_equal_the_sbatchs_own(n, expect):
    """ONE derivation, two shells: lib/common.sh repro_fit_width_flags (04's submission) and the sbatch's fit_width_flags
    (its self-resubmit) must spell the identical allocation — never --mem=0, never --export."""
    lib = subprocess.run(["bash", "-c", f'source {COMMON} 2>/dev/null; repro_fit_width_flags {n}'], capture_output=True, text=True)
    m = re.search(r"^fit_width_flags\(\) \{.*?^\}\n", FIT.read_text(), re.M | re.S)
    assert m, "the sbatch defines fit_width_flags"
    sb = subprocess.run(["bash", "-c", f"{m.group(0)}\nfit_width_flags {n}"], capture_output=True, text=True)
    assert lib.returncode == 0 and sb.returncode == 0, (lib.stderr, sb.stderr)
    assert lib.stdout.strip() == sb.stdout.strip() == expect
    assert "--export" not in expect and "--mem=0" not in expect


def test_fit_width_of_config_reads_the_top_level_line_only(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: a/b\nfit_width: 4\ntrain_kwargs:\n  default:\n    fit_width: 9\n")
    r = subprocess.run(["bash", "-c", f'source {COMMON} 2>/dev/null; repro_fit_width_of_config {cfg}; repro_fit_width_of_config {tmp_path}/absent'],
                       capture_output=True, text=True)
    assert r.stdout.split() == ["4", "1"], r.stdout + r.stderr


@pytest.mark.parametrize("model,fit_id,width,flags", [(W4, f"{W4}_base", 4, W4_FLAGS),                       # the synthetic row's config: fit_width 4
                                                     (DEMO, f"{DEMO}_base", 1, ""),                            # absent key = width 1: no width flags at all
                                                     ("qwen35_4b", "qwen35_4b_base", 1, "")])
def test_04_dry_run_sizes_the_allocation_from_the_config(site, width4, model, fit_id, width, flags):
    if model == W4:
        r = _run_checkout_stage(width4, "04_fit_objects.sh", "--dry-run", "--object", "constitution_relu", "--model", model, site=site)
    else:
        r = run_stage("04_fit_objects.sh", "--dry-run", "--object", "constitution_relu", "--model", model, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    lines = [ln for ln in out.splitlines() if "sdfbase_fit_cell.sbatch" in ln and "sbatch" in ln and "DRY-RUN" in ln]
    assert lines, out[-2500:]
    ln = lines[0]
    if flags:
        assert flags in ln, ln
        assert "data-parallel fit width 4" in out
    else:
        assert "--gres=" not in ln and "--cpus-per-task=" not in ln and "--mem=" not in ln, ln
    assert f"SDFBASE_FIT_WIDTH={width}" in ln and "--export" not in ln, ln
    assert ln.rstrip("'").endswith(f"{fit_id} constv2-mix-full psr_ce_all full22"), ln
    assert f"fit_width={width}" in out


def test_03_dry_run_states_that_trials_stay_width_1(site, width4):
    r = _run_checkout_stage(width4, "03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "design", "--model", W4, site=site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "stay width 1" in out and f"fit_width 4 in sdfbase_{W4}_base_fits.yaml" in out, out[-2500:]   # the note names the config that declares the width
    assert "--gres=gpu:4" not in out
    # the demo (width 1): nothing to note — its trials and its fits are both width 1
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "design", "--model", DEMO, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "stay width 1" not in out and "--gres=gpu:4" not in out, out[-2500:]


def test_models_table_carries_no_second_copy_of_the_fit_width():
    """`reproduce/models.yaml` is the front-door table; the fit width lives ONLY in the fits config's `fit_width:` line,
    so no model's `fit.resources` may carry a `gpus:` row (a second copy that nothing read and that could disagree with the
    config). The header says where the width lives; the one reader agrees."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M  # noqa: E402
    table = M.load_models(str(REPRO / "models.yaml"))
    assert table, "an empty models table"
    for key, m in table.items():
        res = (m.get("fit") or {}).get("resources") or {}
        assert "gpus" not in res, f"{key}: fit.resources carries a gpus row — the width is the fits config's fit_width line"
        assert {"microbatch", "accum_to_effective_batch", "seq_len"} <= set(res), (key, sorted(res))   # the shape rows stay
    head = (REPRO / "models.yaml").read_text().split("schema:")[0]
    assert "fit_width" in head and "repro_fit_width_of_config" in head
    r = subprocess.run([VENV_PY, str(REPRO / "lib" / "models.py"), "--table", str(REPRO / "models.yaml"), "get", "qwen35_4b_demo", "fit.resources"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and '"gpus"' not in r.stdout, r.stdout + r.stderr
