"""03_hpo.sh's effective-batch cross-check (demo-import M4). OVERRIDE_EFFECTIVE_BATCH is the stage's ARITHMETIC — `recon_passes_for_cell`
and the recon spec's `effective_batch` — while every rung and trial TRAINS at the resolved fits config's own effective batch (tune_arm
`config_effective_batch`: batch_size_by_arm x accum_steps once the memprobe is folded, else the PENDING_MEMPROBE_EFF<n>_ACCUM sentinel's
n). A disagreement is a recorded batch that lies about what ran, so the stage refuses it by name on every invocation, before any phase.

The REAL stage, --dry-run, against the committed tree: the demo config (the EFF64 sentinel) agrees with the paper defaults' 64 and
runs; the paper's qwen35_4b_base config realises 8 (2 x 4), so the paper's model REFUSES under the defaults, naming the override, the
config and both batches — and runs under an overrides file that says 8; the demo refuses an override of 8 the same way; a non-integer
override refuses. The check sits between the config's resolution and the first phase, so a `--phase probe` (which trains nothing) is
held to it too: a misconfigured overrides file is refused the way a mixed protocol is."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

DEMO, PAPER = "qwen35_4b_demo", "qwen35_4b"
DEMO_CFG, PAPER_CFG = "configs/sdfbase_qwen35_4b_demo_base_fits.yaml", "configs/sdfbase_qwen35_4b_base_fits.yaml"


def _site(root: Path, effective_batch: str | None = None) -> Path:
    """A site over the example config; with `effective_batch`, a run-overrides file that is the paper defaults with that
    OVERRIDE_EFFECTIVE_BATCH (and its own OVERRIDE_ID), named by SISTEER_RUN_OVERRIDES."""
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    extra = ""
    if effective_batch is not None:
        text = (REPRO / "overrides" / "paper_defaults.env").read_text()
        text = re.sub(r"^OVERRIDE_EFFECTIVE_BATCH=.*$", f"OVERRIDE_EFFECTIVE_BATCH={effective_batch}", text, flags=re.M)
        text = text.replace("OVERRIDE_ID=paper_defaults", f"OVERRIDE_ID=eb_{effective_batch}_under_test")
        assert f"OVERRIDE_EFFECTIVE_BATCH={effective_batch}\n" in text or text.rstrip().endswith(f"OVERRIDE_EFFECTIVE_BATCH={effective_batch}")
        ovr = site / "overrides" / f"eb_{effective_batch}_under_test.env"
        ovr.write_text(text)
        extra = f"SISTEER_RUN_OVERRIDES={ovr}\n"
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n{extra}")
    return site


def _03(site: Path, model: str, *phase: str):
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", *(phase or ("--phase", "all")), "--model", model, site=site, timeout=600)
    return r.returncode, r.stdout + r.stderr


def test_the_committed_configs_realise_the_batches_the_check_compares():
    """The fixture facts, read from the tree: the paper defaults say 64; the demo config's sentinel names 64; the paper's config is folded
    at 2 x 4 = 8 (the frozen batch-8 reference lane)."""
    import yaml
    assert re.search(r"^OVERRIDE_EFFECTIVE_BATCH=64\s*$", (REPRO / "overrides" / "paper_defaults.env").read_text(), re.M)
    demo = yaml.safe_load((REPO / DEMO_CFG).read_text()); paper = yaml.safe_load((REPO / PAPER_CFG).read_text())
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert demo["train_kwargs"][arm]["accum_steps"] == "PENDING_MEMPROBE_EFF64_ACCUM" and demo["batch_size_by_arm"][arm] == "MEMPROBE"
        assert paper["batch_size_by_arm"][arm] * paper["train_kwargs"][arm]["accum_steps"] == 8
    src = (REPO / "reproduce" / "03_hpo.sh").read_text()
    # the check reads the RESOLVED config and sits before the first phase
    assert src.index('CFG_EFFECTIVE_BATCH="$(') > src.index('FIT_CFG="$(repro_fits_config_for')
    assert src.index('[ "$CFG_EFFECTIVE_BATCH" = "$EFFECTIVE_BATCH" ]') < src.index("if want probe; then")


def test_the_demo_lane_agrees_with_the_paper_defaults_and_runs(tmp_path):
    rc, out = _03(_site(tmp_path), DEMO)
    assert rc == 0, out[-2500:]
    assert "REFUSED" not in out


def test_the_papers_model_refuses_the_64_default_naming_the_config_and_both_batches(tmp_path):
    rc, out = _03(_site(tmp_path), PAPER)
    assert rc == 2, out[-2500:]
    assert "REFUSED" in out and "OVERRIDE_EFFECTIVE_BATCH=64" in out and PAPER_CFG in out and "realises effective batch 8" in out, out[-2500:]
    assert "overrides paper_defaults" in out and "set OVERRIDE_EFFECTIVE_BATCH=8" in out, out[-2500:]
    assert "sbatch" not in out.split("REFUSED", 1)[1]                        # nothing composed past the refusal


def test_the_papers_model_runs_under_an_overrides_file_that_says_8(tmp_path):
    rc, out = _03(_site(tmp_path, "8"), PAPER)
    assert rc == 0, out[-2500:]


def test_the_demo_lane_refuses_an_override_of_8(tmp_path):
    rc, out = _03(_site(tmp_path, "8"), DEMO)
    assert rc == 2 and "OVERRIDE_EFFECTIVE_BATCH=8" in out and DEMO_CFG in out and "realises effective batch 64" in out, out[-2500:]
    assert "overrides eb_8_under_test" in out


@pytest.mark.parametrize("bad", ["sixty-four", "0", ""])
def test_a_non_integer_or_zero_override_refuses_by_name(tmp_path, bad):
    rc, out = _03(_site(tmp_path, bad), DEMO)
    if bad == "":
        # an EMPTY value falls to the stage's default (64), which the demo agrees with: ${OVERRIDE_EFFECTIVE_BATCH:-64}
        assert rc == 0, out[-2500:]
    else:
        assert rc == 2 and f"OVERRIDE_EFFECTIVE_BATCH='{bad}' is not a positive integer" in out, out[-2500:]


def test_the_probe_phase_is_held_to_the_check_too(tmp_path):
    """A phase that trains nothing still runs under the overrides file as a whole: the check is about the file's consistency with the
    lane, not about which phase consumes the number."""
    rc, out = _03(_site(tmp_path, "8"), DEMO, "--phase", "probe")
    assert rc == 2 and "realises effective batch 64" in out, out[-2500:]
